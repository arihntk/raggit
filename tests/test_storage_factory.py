"""Tests for the storage factory."""

from __future__ import annotations

import pytest

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.factory import UnsupportedStorageError, create_storage
from raggit.storage.local import LocalStorage


def test_create_local_storage() -> None:
    config = StorageConfig(source_type=SourceType.LOCAL, uri="./data")
    storage = create_storage(config)
    assert isinstance(storage, LocalStorage)


def test_unsupported_storage() -> None:
    # Use a fake source type by bypassing enum validation
    config = StorageConfig(source_type=SourceType.LOCAL, uri="./data")  # type: ignore[arg-type]
    config.source_type = "unknown"  # type: ignore[assignment]
    with pytest.raises(UnsupportedStorageError):
        create_storage(config)
