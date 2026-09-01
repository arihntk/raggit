"""Factory for creating storage backends."""

from __future__ import annotations

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.base import Storage
from raggit.storage.local import LocalStorage


class UnsupportedStorageError(Exception):
    """Raised when a storage type is not supported."""


def _is_cloud_uri(path: str) -> bool:
    """Return True if path looks like a cloud storage URI."""
    return path.startswith(("s3://", "gs://", "azure://"))


def _apply_cloud_uri_to_config(uri: str, config: StorageConfig) -> None:
    """Parse a cloud URI and update the storage config's bucket/container/prefix.

    Examples:
        s3://my-bucket/prefix -> bucket=my-bucket, prefix=prefix, uri=full
        gs://my-bucket/docs   -> bucket=my-bucket, prefix=docs
        azure://my-container/prefix -> container=my-container, prefix=prefix
    """
    # Normalize and update uri
    config.uri = uri
    if uri.startswith("s3://"):
        # s3://bucket/prefix
        rest = uri[len("s3://") :]
        parts = rest.split("/", 1)
        config.bucket = parts[0] or config.bucket
        config.prefix = parts[1] if len(parts) > 1 else None
        config.source_type = SourceType.S3
    elif uri.startswith("gs://"):
        rest = uri[len("gs://") :]
        parts = rest.split("/", 1)
        config.bucket = parts[0] or config.bucket
        config.prefix = parts[1] if len(parts) > 1 else None
        config.source_type = SourceType.GCS
    elif uri.startswith("azure://"):
        rest = uri[len("azure://") :]
        parts = rest.split("/", 1)
        config.container = parts[0] or config.container
        config.prefix = parts[1] if len(parts) > 1 else None
        config.source_type = SourceType.AZURE_BLOB


def create_storage(config: StorageConfig) -> Storage:
    """Create a storage backend from configuration."""
    if config.source_type == SourceType.LOCAL:
        return LocalStorage(config.uri)
    if config.source_type == SourceType.S3:
        from raggit.storage.s3 import S3Storage

        return S3Storage(config)
    if config.source_type == SourceType.GCS:
        from raggit.storage.gcs import GCSStorage

        return GCSStorage(config)
    if config.source_type == SourceType.AZURE_BLOB:
        from raggit.storage.azure import AzureBlobStorage

        return AzureBlobStorage(config)
    msg = f"Storage source type '{config.source_type}' is not yet supported"
    raise UnsupportedStorageError(msg)
