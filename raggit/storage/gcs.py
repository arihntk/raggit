"""Google Cloud Storage backend."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, cast

from raggit.api.models import SourceType, StorageConfig
from raggit.core.logging import get_logger
from raggit.storage.base import (
    FileAddedEvent,
    FileDeletedEvent,
    FileEvent,
    FileEventCallback,
    FileModifiedEvent,
    Storage,
    StorageFile,
)

logger = get_logger("raggit.storage.gcs")

# Supported document extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt"}


def _is_supported(name: str) -> bool:
    """Return True if the blob name extension is supported."""
    return any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _to_uri(bucket: str, name: str) -> str:
    """Build a gs:// URI for a bucket and blob name."""
    return f"gs://{bucket}/{name}"


def _to_storage_file(bucket: str, name: str, size: int, modified_at: datetime) -> StorageFile:
    """Convert GCS blob metadata to StorageFile."""
    return StorageFile(
        path=_to_uri(bucket, name),
        relative_path=name,
        size=size,
        modified_at=modified_at,
        content_hash=None,
    )


class GCSStorage(Storage):
    """Storage backend for Google Cloud Storage buckets."""

    source_type = SourceType.GCS.value

    def __init__(self, config: StorageConfig) -> None:
        if config.bucket is None:
            msg = "GCS storage requires a bucket name"
            raise ValueError(msg)

        self.config = config
        self.bucket: str = config.bucket
        self.prefix = (config.prefix or "").rstrip("/")
        if self.prefix:
            self.prefix += "/"
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-load and cache the GCS client."""
        if self._client is not None:
            return self._client

        try:
            from google.cloud.storage import Client
        except ImportError as exc:
            msg = (
                "GCS support requires google-cloud-storage. "
                "Install it with: uv pip install 'raggit[gcs]'"
            )
            raise ImportError(msg) from exc

        if self.config.gcs_service_account_path:
            from google.oauth2.service_account import Credentials

            credentials = cast(Any, Credentials).from_service_account_file(
                self.config.gcs_service_account_path
            )
            self._client = Client(credentials=credentials, project=credentials.project_id)
        else:
            self._client = Client()
        return self._client

    def _get_bucket(self) -> Any:
        """Return the configured bucket."""
        return self._get_client().bucket(self.bucket)

    async def list_files(self) -> list[StorageFile]:
        """List all supported blobs under the configured prefix."""
        loop = asyncio.get_running_loop()
        bucket = self._get_bucket()

        blobs = await loop.run_in_executor(
            None, lambda: list(bucket.list_blobs(prefix=self.prefix or None))
        )

        files: list[StorageFile] = []
        for blob in blobs:
            if not _is_supported(blob.name):
                continue
            updated = blob.updated
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=UTC)
            files.append(
                _to_storage_file(
                    bucket=self.bucket,
                    name=blob.name,
                    size=blob.size or 0,
                    modified_at=updated,
                )
            )
        return files

    async def read_file(self, path: str) -> bytes:
        """Read blob bytes from GCS."""
        loop = asyncio.get_running_loop()
        bucket = self._get_bucket()
        name = self._name_from_path(path)
        blob = bucket.blob(name)
        return await loop.run_in_executor(None, blob.download_as_bytes)

    async def file_exists(self, path: str) -> bool:
        """Check if a blob exists in GCS."""
        loop = asyncio.get_running_loop()
        bucket = self._get_bucket()
        name = self._name_from_path(path)
        blob = bucket.blob(name)
        return await loop.run_in_executor(None, blob.exists)

    async def compute_hash(self, path: str) -> str:
        """Return the blob MD5 hash or CRC32C as a stable hash.

        ``blob.reload()`` mutates the blob in place and returns None, so we
        read hash fields from the blob after reload completes.
        """
        loop = asyncio.get_running_loop()
        bucket = self._get_bucket()
        name = self._name_from_path(path)
        blob = bucket.blob(name)
        await loop.run_in_executor(None, blob.reload)
        if blob.md5_hash:
            return cast(str, blob.md5_hash)
        if blob.crc32c:
            return cast(str, blob.crc32c)
        return await self._compute_sha256(path)

    async def _compute_sha256(self, path: str) -> str:
        """Fallback SHA-256 over blob bytes."""
        import hashlib

        data = await self.read_file(path)
        return hashlib.sha256(data).hexdigest()

    async def watch(
        self,
        on_event: FileEventCallback,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """Poll GCS and emit events when blobs change."""
        previous: dict[str, StorageFile] = {}
        for file in await self.list_files():
            previous[file.path] = file

        logger.info("Started GCS storage watcher", bucket=self.bucket, prefix=self.prefix)

        try:
            while True:
                await asyncio.sleep(poll_interval_seconds)
                current_files = await self.list_files()
                current = {f.path: f for f in current_files}

                for path, file in current.items():
                    prev = previous.get(path)
                    if prev is None:
                        await self._emit(on_event, FileAddedEvent(file))
                    elif prev.modified_at != file.modified_at or prev.size != file.size:
                        await self._emit(on_event, FileModifiedEvent(file))

                for path, file in previous.items():
                    if path not in current:
                        await self._emit(on_event, FileDeletedEvent(file))

                previous = current
        finally:
            await self.close()

    async def _emit(self, on_event: FileEventCallback, event: FileEvent) -> None:
        try:
            await on_event(event)
        except Exception:
            logger.exception("Error handling GCS storage event", path=event.file.path)

    async def close(self) -> None:
        """Close the GCS client transport."""
        if self._client is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._client.close)
            self._client = None

    def _name_from_path(self, path: str) -> str:
        """Extract the blob name from a URI or name string."""
        if path.startswith(f"gs://{self.bucket}/"):
            return path[len(f"gs://{self.bucket}/") :]
        return path
