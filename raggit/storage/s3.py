"""AWS S3 storage backend."""

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

logger = get_logger("raggit.storage.s3")

# Supported document extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt"}


def _is_supported(key: str) -> bool:
    """Return True if the key extension is supported."""
    return any(key.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _to_uri(bucket: str, key: str) -> str:
    """Build an s3:// URI for a bucket and key."""
    return f"s3://{bucket}/{key}"


def _to_storage_file(bucket: str, key: str, size: int, modified_at: datetime) -> StorageFile:
    """Convert S3 object metadata to StorageFile."""
    return StorageFile(
        path=_to_uri(bucket, key),
        relative_path=key,
        size=size,
        modified_at=modified_at,
        content_hash=None,
    )


class S3Storage(Storage):
    """Storage backend for AWS S3 buckets."""

    source_type = SourceType.S3.value

    def __init__(self, config: StorageConfig) -> None:
        if config.bucket is None:
            msg = "S3 storage requires a bucket name"
            raise ValueError(msg)

        self.config = config
        self.bucket: str = config.bucket
        self.prefix = (config.prefix or "").rstrip("/")
        if self.prefix:
            self.prefix += "/"
        self._client: Any | None = None
        self._client_ctx: Any | None = None

    async def _get_client(self) -> Any:
        """Lazy-load and cache the aiobotocore S3 client.

        aiobotocore's ``create_client`` returns an async context manager; we
        enter it once and keep the live client for the lifetime of this backend.
        """
        if self._client is not None:
            return self._client

        try:
            from aiobotocore.session import get_session
        except ImportError as exc:
            msg = (
                "S3 support requires aiobotocore. "
                "Install it with: uv pip install 'raggit[s3]'"
            )
            raise ImportError(msg) from exc

        session = get_session()
        kwargs: dict[str, Any] = {}
        if self.config.region:
            kwargs["region_name"] = self.config.region
        if self.config.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key

        self._client_ctx = session.create_client("s3", **kwargs)
        self._client = await self._client_ctx.__aenter__()
        return self._client

    async def list_files(self) -> list[StorageFile]:
        """List all supported objects under the configured prefix."""
        client = await self._get_client()
        files: list[StorageFile] = []
        paginator = client.get_paginator("list_objects_v2")

        params: dict[str, Any] = {"Bucket": self.bucket}
        if self.prefix:
            params["Prefix"] = self.prefix

        async for page in paginator.paginate(**params):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not _is_supported(key):
                    continue
                last_modified = obj["LastModified"]
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=UTC)
                files.append(
                    _to_storage_file(
                        bucket=self.bucket,
                        key=key,
                        size=obj["Size"],
                        modified_at=last_modified,
                    )
                )
        return files

    async def read_file(self, path: str) -> bytes:
        """Read object bytes from S3."""
        client = await self._get_client()
        key = self._key_from_path(path)
        response = await client.get_object(Bucket=self.bucket, Key=key)
        async with response["Body"] as stream:
            data = await stream.read()
            return cast(bytes, data)

    async def file_exists(self, path: str) -> bool:
        """Check if an object exists in S3."""
        client = await self._get_client()
        key = self._key_from_path(path)
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", None) or {}
            error_code = str(response.get("Error", {}).get("Code", ""))
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise
        return True

    async def compute_hash(self, path: str) -> str:
        """Return the ETag for the S3 object as a stable hash."""
        client = await self._get_client()
        key = self._key_from_path(path)
        response = await client.head_object(Bucket=self.bucket, Key=key)
        etag = response.get("ETag", "").strip('"')
        return etag or await self._compute_sha256(path)

    async def _compute_sha256(self, path: str) -> str:
        """Fallback SHA-256 over object bytes."""
        import hashlib

        data = await self.read_file(path)
        return hashlib.sha256(data).hexdigest()

    async def watch(
        self,
        on_event: FileEventCallback,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """Poll S3 and emit events when objects change."""
        previous: dict[str, StorageFile] = {}
        for file in await self.list_files():
            previous[file.path] = file

        logger.info("Started S3 storage watcher", bucket=self.bucket, prefix=self.prefix)

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
            logger.exception("Error handling S3 storage event", path=event.file.path)

    async def close(self) -> None:
        """Close the aiobotocore client context."""
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client_ctx = None
            self._client = None
        elif self._client is not None:
            # Fallback for tests that inject a bare mock client
            close = getattr(self._client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            self._client = None

    def _key_from_path(self, path: str) -> str:
        """Extract the S3 key from a URI or key string."""
        if path.startswith(f"s3://{self.bucket}/"):
            return path[len(f"s3://{self.bucket}/") :]
        return path
