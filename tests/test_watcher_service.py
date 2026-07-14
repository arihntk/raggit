"""Tests for the lightweight WatcherService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from raggit.api.models import RAGConfig, SourceType, StorageConfig
from raggit.core.watcher import WatcherService
from raggit.storage.base import (
    FileAddedEvent,
    FileDeletedEvent,
    FileEventCallback,
    FileModifiedEvent,
    Storage,
    StorageFile,
)


class _FakeSession:
    """Minimal async session stand-in for watcher tests."""

    def begin(self) -> _FakeSession:
        return self

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class _FakeSessionLocal:
    def __call__(self) -> _FakeSession:
        return _FakeSession()


class _MockStorage(Storage):
    """In-memory storage backend for watcher tests."""

    source_type = SourceType.LOCAL.value

    def __init__(self, files: list[StorageFile] | None = None) -> None:
        self.files = list(files or [])
        self._callback: FileEventCallback | None = None
        self._watch_started = False
        self._stop_event = asyncio.Event()

    async def list_files(self) -> list[StorageFile]:
        return list(self.files)

    async def read_file(self, path: str) -> bytes:
        return b"mock content"

    async def file_exists(self, path: str) -> bool:
        return any(f.path == path for f in self.files)

    async def compute_hash(self, path: str) -> str:
        return "mockhash"

    async def watch(
        self,
        on_event: FileEventCallback,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self._callback = on_event
        self._watch_started = True
        await self._stop_event.wait()

    async def close(self) -> None:
        self._stop_event.set()

    def emit(self, event: Any) -> None:
        if self._callback is not None:
            asyncio.get_running_loop().call_soon(
                lambda e=event: asyncio.create_task(self._callback(e))
            )


def _make_config() -> RAGConfig:
    return RAGConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test",
        storage=StorageConfig(
            source_type=SourceType.LOCAL,
            uri="/tmp/test",
        ),
    )


@pytest.fixture
def sample_file() -> StorageFile:
    return StorageFile(
        path="/tmp/test/doc.md",
        relative_path="doc.md",
        size=12,
        modified_at=datetime.now(UTC),
    )


@pytest.fixture
def patched_watcher_deps(monkeypatch: Any) -> None:
    """Patch storage creation, session factory, and log_event for unit tests."""
    import raggit.core.watcher as watcher_module

    monkeypatch.setattr(watcher_module, "AsyncSessionLocal", _FakeSessionLocal())
    monkeypatch.setattr(watcher_module, "log_event", AsyncMock())


@pytest.fixture
def mock_storage_factory(monkeypatch: Any) -> _MockStorage:
    """Return a mock storage and patch create_storage to use it."""
    import raggit.core.watcher as watcher_module

    storage = _MockStorage()
    monkeypatch.setattr(watcher_module, "create_storage", lambda cfg: storage)
    return storage


async def test_watcher_service_starts_storage_watch(
    patched_watcher_deps: None,
    mock_storage_factory: _MockStorage,
) -> None:
    config = _make_config()
    service = WatcherService(config)
    service.indexer.sync_all = AsyncMock()  # type: ignore[method-assign]
    service.indexer.close = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(service.start())
    await asyncio.sleep(0.05)

    assert mock_storage_factory._watch_started

    await service.stop()
    await asyncio.wait_for(task, timeout=1.0)


async def test_watcher_service_runs_initial_sync(
    patched_watcher_deps: None,
    mock_storage_factory: _MockStorage,
) -> None:
    config = _make_config()
    service = WatcherService(config)
    service.indexer.sync_all = AsyncMock()  # type: ignore[method-assign]
    service.indexer.close = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(service.start())
    await asyncio.sleep(0.05)

    service.indexer.sync_all.assert_awaited_once()  # type: ignore[attr-defined]

    await service.stop()
    await asyncio.wait_for(task, timeout=1.0)


async def test_watcher_service_indexes_added_file(
    patched_watcher_deps: None,
    mock_storage_factory: _MockStorage,
    sample_file: StorageFile,
) -> None:
    config = _make_config()
    service = WatcherService(config, debounce_seconds=0.05)
    service.indexer.sync_all = AsyncMock()  # type: ignore[method-assign]
    service.indexer.index_file = AsyncMock()  # type: ignore[method-assign]
    service.indexer.close = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(service.start())
    await asyncio.sleep(0.05)

    mock_storage_factory.emit(FileAddedEvent(sample_file))
    await asyncio.sleep(0.15)

    service.indexer.index_file.assert_awaited_once()  # type: ignore[attr-defined]

    await service.stop()
    await asyncio.wait_for(task, timeout=1.0)


async def test_watcher_service_removes_deleted_file(
    patched_watcher_deps: None,
    mock_storage_factory: _MockStorage,
    sample_file: StorageFile,
) -> None:
    config = _make_config()
    service = WatcherService(config, debounce_seconds=0.05)
    service.indexer.sync_all = AsyncMock()  # type: ignore[method-assign]
    service.indexer.remove_file = AsyncMock()  # type: ignore[method-assign]
    service.indexer.close = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(service.start())
    await asyncio.sleep(0.05)

    mock_storage_factory.emit(FileDeletedEvent(sample_file))
    await asyncio.sleep(0.15)

    service.indexer.remove_file.assert_awaited_once()  # type: ignore[attr-defined]

    await service.stop()
    await asyncio.wait_for(task, timeout=1.0)


async def test_watcher_service_debounces_rapid_events(
    patched_watcher_deps: None,
    mock_storage_factory: _MockStorage,
    sample_file: StorageFile,
) -> None:
    config = _make_config()
    service = WatcherService(config, debounce_seconds=0.1)
    service.indexer.sync_all = AsyncMock()  # type: ignore[method-assign]
    service.indexer.index_file = AsyncMock()  # type: ignore[method-assign]
    service.indexer.close = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(service.start())
    await asyncio.sleep(0.05)

    mock_storage_factory.emit(FileAddedEvent(sample_file))
    await asyncio.sleep(0.05)
    mock_storage_factory.emit(FileModifiedEvent(sample_file))
    await asyncio.sleep(0.05)
    mock_storage_factory.emit(FileModifiedEvent(sample_file))
    await asyncio.sleep(0.25)

    # Should be called at least once but not three times.
    assert service.indexer.index_file.await_count <= 2  # type: ignore[attr-defined]
    assert service.indexer.index_file.await_count >= 1  # type: ignore[attr-defined]

    await service.stop()
    await asyncio.wait_for(task, timeout=1.0)
