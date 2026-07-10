"""Azure Blob Storage backend."""

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

logger = get_logger("raggit.storage.azure")

# Supported document extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt"}


def _is_supported(name: str) -> bool:
    """Return True if the blob name extension is supported."""
    return any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _to_uri(container: str, name: str) -> str:
    """Build an azure:// URI for a container and blob name."""
    return f"azure://{container}/{name}"


def _to_storage_file(
    container: str, name: str, size: int, modified_at: datetime
) -> StorageFile:
    """Convert Azure blob metadata to StorageFile."""
    return StorageFile(
        path=_to_uri(container, name),
        relative_path=name,
        size=size,
        modified_at=modified_at,
        content_hash=None,
    )


class AzureBlobStorage(Storage):
    """Storage backend for Azure Blob Storage containers."""

    source_type = SourceType.AZURE_BLOB.value

    def __init__(self, config: StorageConfig) -> None:
        if config.container is None:
            msg = "Azure Blob storage requires a container name"
            raise ValueError(msg)
        if config.azure_connection_string is None:
            msg = "Azure Blob storage requires a connection string"
            raise ValueError(msg)

        self.config = config
        self.container: str = config.container
        self.connection_string: str = config.azure_connection_string
        self.prefix = (config.prefix or "").rstrip("/")
        if self.prefix:
            self.prefix += "/"
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazy-load and cache the Azure BlobServiceClient."""
        if self._client is not None:
            return self._client

        try:
            from azure.storage.blob.aio import BlobServiceClient
        except ImportError as exc:
            msg = (
                "Azure Blob support requires azure-storage-blob. "
                "Install it with: uv pip install 'raggit[azure]'"
            )
            raise ImportError(msg) from exc

        self._client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        return self._client

    def _get_container_client(self) -> Any:
        """Return the configured container client."""
        return self._get_client().get_container_client(self.container)

    async def list_files(self) -> list[StorageFile]:
        """List all supported blobs under the configured prefix."""
        container_client = self._get_container_client()
        files: list[StorageFile] = []

        async for blob in container_client.list_blobs(name_starts_with=self.prefix or None):
            if not _is_supported(blob.name):
                continue
            modified = blob.last_modified
            if modified.tzinfo is None:
                modified = modified.replace(tzinfo=UTC)
            files.append(
                _to_storage_file(
                    container=self.container,
                    name=blob.name,
                    size=blob.size,
                    modified_at=modified,
                )
            )
        return files

    async def read_file(self, path: str) -> bytes:
        """Read blob bytes from Azure."""
        container_client = self._get_container_client()
        name = self._name_from_path(path)
        blob_client = container_client.get_blob_client(name)
        downloader = await blob_client.download_blob()
        data = await downloader.readall()
        return cast(bytes, data)

    async def file_exists(self, path: str) -> bool:
        """Check if a blob exists in Azure."""
        container_client = self._get_container_client()
        name = self._name_from_path(path)
        blob_client = container_client.get_blob_client(name)
        return cast(bool, await blob_client.exists())

    async def compute_hash(self, path: str) -> str:
        """Return the blob Content-MD5 or ETag as a stable hash."""
        container_client = self._get_container_client()
        name = self._name_from_path(path)
        blob_client = container_client.get_blob_client(name)
        properties = await blob_client.get_blob_properties()
        if properties.content_settings.content_md5:
            return cast(str, properties.content_settings.content_md5.hex())
        etag = properties.etag.strip('"')
        return cast(str, etag) or await self._compute_sha256(path)

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
        """Poll Azure and emit events when blobs change."""
        previous: dict[str, StorageFile] = {}
        for file in await self.list_files():
            previous[file.path] = file

        logger.info(
            "Started Azure Blob storage watcher",
            container=self.container,
            prefix=self.prefix,
        )

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
            logger.exception("Error handling Azure storage event", path=event.file.path)

    async def close(self) -> None:
        """Close the Azure BlobServiceClient."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _name_from_path(self, path: str) -> str:
        """Extract the blob name from a URI or name string."""
        if path.startswith(f"azure://{self.container}/"):
            return path[len(f"azure://{self.container}/") :]
        return path
