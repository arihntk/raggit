"""Factory for creating storage backends."""

from __future__ import annotations

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.base import Storage
from raggit.storage.local import LocalStorage


class UnsupportedStorageError(Exception):
    """Raised when a storage type is not supported."""


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
