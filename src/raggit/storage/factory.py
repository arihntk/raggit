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
    msg = f"Storage source type '{config.source_type}' is not yet supported"
    raise UnsupportedStorageError(msg)
