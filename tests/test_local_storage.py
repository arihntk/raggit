"""Tests for local filesystem storage safety and deletion handling."""

from __future__ import annotations

from pathlib import Path

import pytest

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
