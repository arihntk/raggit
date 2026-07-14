"""Tests for local filesystem storage safety and deletion handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from raggit.storage.base import FileAddedEvent
from raggit.storage.local import LocalStorage, _to_deleted_storage_file


async def test_read_file_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "ok.md").write_text("hello", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")

    storage = LocalStorage(str(root))

    data = await storage.read_file(str(root / "ok.md"))
    assert data == b"hello"

    with pytest.raises(PermissionError):
        await storage.read_file(str(secret))


async def test_deleted_storage_file_does_not_require_stat(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    missing = root / "gone.md"
    # File never created — simulates post-delete watcher event
    file = _to_deleted_storage_file(missing, root.resolve())
    assert file.path.endswith("gone.md")
    assert file.relative_path == "gone.md"
    assert file.size == 0


async def test_compute_hash(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "note.md"
    path.write_text("raggit", encoding="utf-8")

    storage = LocalStorage(str(root))
    digest = await storage.compute_hash(str(path.resolve()))
    assert len(digest) == 64


async def test_local_watch_emits_events_without_polling(tmp_path: Path) -> None:
    """Local storage uses OS-native events and does not rely on a poll loop."""
    root = tmp_path / "watch"
    root.mkdir()
    storage = LocalStorage(str(root))

    received: list[Any] = []

    async def on_event(event: Any) -> None:
        received.append(event)

    watch_task = asyncio.create_task(
        storage.watch(on_event, poll_interval_seconds=3600.0)
    )
    # Give watchdog a moment to start observing.
    await asyncio.sleep(0.2)

    (root / "new.md").write_text("hello", encoding="utf-8")
    # Wait for the OS to deliver the event; should be well under the poll interval.
    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.05)

    await storage.close()
    await asyncio.wait_for(watch_task, timeout=2.0)

    assert len(received) >= 1
    assert isinstance(received[0], FileAddedEvent)
    assert received[0].file.relative_path == "new.md"
