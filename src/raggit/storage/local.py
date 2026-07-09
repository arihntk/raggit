"""Local filesystem storage backend with watchdog-based monitoring."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from watchdog.events import (
    DirCreatedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from raggit.api.models import SourceType
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

logger = get_logger("raggit.storage.local")

# Supported document extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt"}


def _is_supported(path: Path) -> bool:
    """Return True if the path extension is supported."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _to_storage_file(path: Path, root: Path) -> StorageFile:
    """Convert a Path to a StorageFile."""
    stat = path.stat()
    return StorageFile(
        path=str(path.resolve()),
        relative_path=str(path.relative_to(root)),
        size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        content_hash=None,
    )


class _LocalEventHandler(FileSystemEventHandler):
    """Watchdog event handler that forwards to async callback."""

    def __init__(
        self,
        root: Path,
        on_event: FileEventCallback,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.root = root
        self.on_event = on_event
        self.loop = loop

    def on_created(self, event: FileSystemEvent) -> None:
        if isinstance(event, DirCreatedEvent):
            return
        path = Path(event.src_path)
        if not _is_supported(path):
            return
        file = _to_storage_file(path, self.root)
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._emit(FileAddedEvent(file)))
        )

    def on_modified(self, event: FileSystemEvent) -> None:
        if isinstance(event, DirCreatedEvent):
            return
        path = Path(event.src_path)
        if not _is_supported(path):
            return
        file = _to_storage_file(path, self.root)
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._emit(FileModifiedEvent(file)))
        )

    def on_deleted(self, event: FileSystemEvent) -> None:
        if isinstance(event, DirCreatedEvent):
            return
        path = Path(event.src_path)
        if not _is_supported(path):
            return
        file = _to_storage_file(path, self.root)
        self.loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._emit(FileDeletedEvent(file)))
        )

    async def _emit(self, event: FileEvent) -> None:
        try:
            await self.on_event(event)
        except Exception:
            logger.exception("Error handling storage event", path=event.file.path)


class LocalStorage(Storage):
    """Storage backend for local filesystem directories."""

    source_type = SourceType.LOCAL.value

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._observer: Observer | None = None

    async def list_files(self) -> list[StorageFile]:
        """Recursively list all supported files under root."""
        files: list[StorageFile] = []
        for path in self.root.rglob("*"):
            if path.is_file() and _is_supported(path):
                files.append(_to_storage_file(path, self.root))
        return files

    async def read_file(self, path: str) -> bytes:
        """Read file bytes."""
        return Path(path).read_bytes()

    async def file_exists(self, path: str) -> bool:
        """Check file existence."""
        return Path(path).is_file()

    async def compute_hash(self, path: str) -> str:
        """Compute SHA-256 hash of file contents."""
        hasher = hashlib.sha256()
        hasher.update(await self.read_file(path))
        return hasher.hexdigest()

    async def watch(
        self,
        on_event: FileEventCallback,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """Watch the directory tree for changes using watchdog."""
        loop = asyncio.get_running_loop()
        handler = _LocalEventHandler(self.root, on_event, loop)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.root), recursive=True)
        self._observer.start()
        logger.info("Started local storage watcher", root=str(self.root))

        try:
            while True:
                await asyncio.sleep(poll_interval_seconds)
        finally:
            self._observer.stop()
            self._observer.join()

    async def close(self) -> None:
        """Stop the watcher if running."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
