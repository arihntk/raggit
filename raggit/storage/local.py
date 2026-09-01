"""Local filesystem storage backend with watchdog-based monitoring.

Rewritten from scratch for robust, independent operation:

- Ignores directory events via ``is_directory`` (not fragile isinstance checks).
- Handles ``moved`` events (rename) as delete+add.
- Thread-safe emission via ``call_soon_threadsafe`` with loop-closed guards.
- Cancellable watch loop – reacts to ``CancelledError`` without leaking the
  observer thread, and ``close()`` is idempotent.
- ``watch`` runs indefinitely until ``close()`` or task cancellation; it is
  automatic and does not require manual polling.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
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


def _event_path(event: FileSystemEvent) -> Path:
    """Normalize a watchdog event path to Path."""
    src = event.src_path
    if isinstance(src, bytes):
        src = src.decode("utf-8", errors="replace")
    return Path(src)


def _event_dest_path(event: FileSystemEvent) -> Path | None:
    """Return dest_path for moved events, else None."""
    dest = getattr(event, "dest_path", None)
    if not dest:
        return None
    if isinstance(dest, bytes):
        dest = dest.decode("utf-8", errors="replace")
    return Path(dest)


def _to_storage_file(path: Path, root: Path) -> StorageFile:
    """Convert a Path to a StorageFile (file must exist)."""
    stat = path.stat()
    return StorageFile(
        path=str(path.resolve()),
        relative_path=str(path.relative_to(root)),
        size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        content_hash=None,
    )


def _to_deleted_storage_file(path: Path, root: Path) -> StorageFile:
    """Build a StorageFile for a path that no longer exists on disk."""
    try:
        resolved = path if path.is_absolute() else (root / path).resolve()
        try:
            relative = str(resolved.relative_to(root))
        except ValueError:
            relative = path.name
        path_str = str(resolved)
    except OSError:
        path_str = str(path)
        relative = path.name

    return StorageFile(
        path=path_str,
        relative_path=relative,
        size=0,
        modified_at=datetime.now(UTC),
        content_hash=None,
    )


class _LocalEventHandler(FileSystemEventHandler):
    """Watchdog event handler that forwards to async callback safely."""

    def __init__(
        self,
        root: Path,
        on_event: FileEventCallback,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self.root = root
        self.on_event = on_event
        self.loop = loop

    # -- helpers -------------------------------------------------------

    def _schedule(self, event: FileEvent) -> None:
        if self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._emit(event))
            )
        except RuntimeError:
            # Loop may be closing.
            return

    async def _emit(self, event: FileEvent) -> None:
        try:
            await self.on_event(event)
        except Exception:
            logger.exception("Error handling storage event", path=event.file.path)

    # -- watchdog callbacks --------------------------------------------

    def on_created(self, event: FileSystemEvent) -> None:
        if getattr(event, "is_directory", False):
            return
        path = _event_path(event)
        if not _is_supported(path):
            return
        try:
            file = _to_storage_file(path, self.root)
        except OSError:
            # File may have been deleted before we could stat it, or is
            # still being written. Best effort – watcher will catch the next
            # modification event once the write completes.
            logger.debug("Could not stat created file", path=str(path))
            return
        self._schedule(FileAddedEvent(file))

    def on_modified(self, event: FileSystemEvent) -> None:
        if getattr(event, "is_directory", False):
            return
        path = _event_path(event)
        if not _is_supported(path):
            return
        try:
            file = _to_storage_file(path, self.root)
        except OSError:
            logger.debug("Could not stat modified file", path=str(path))
            return
        self._schedule(FileModifiedEvent(file))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if getattr(event, "is_directory", False):
            return
        path = _event_path(event)
        if not _is_supported(path):
            return
        file = _to_deleted_storage_file(path, self.root)
        self._schedule(FileDeletedEvent(file))

    def on_moved(self, event: FileSystemEvent) -> None:
        if getattr(event, "is_directory", False):
            return
        src = _event_path(event)
        dest = _event_dest_path(event)
        # Treat move as delete of src + add of dest (if supported).
        if _is_supported(src):
            src_file = _to_deleted_storage_file(src, self.root)
            self._schedule(FileDeletedEvent(src_file))
        if dest is not None and _is_supported(dest):
            try:
                dest_file = _to_storage_file(dest, self.root)
            except OSError:
                logger.debug("Could not stat moved destination", path=str(dest))
                return
            self._schedule(FileAddedEvent(dest_file))


class LocalStorage(Storage):
    """Storage backend for local filesystem directories.

    The watcher is fully automatic: it uses OS-native events (FSEvents /
    inotify / kqueue) and runs independently in a background task. Callers
    never need to poll manually – just ``await storage.watch(callback)`` and
    the method blocks until ``close()`` or cancellation.
    """

    source_type = SourceType.LOCAL.value

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._observer: Any | None = None
        self._stop_event: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    def _resolve_safe(self, path: str) -> Path:
        """Resolve a path and ensure it stays within the storage root."""
        resolved = Path(path).expanduser().resolve()
        # On Python 3.9+ is_relative_to is available; mypy handles it.
        if not resolved.is_relative_to(self.root):  # type: ignore[attr-defined]
            msg = f"Path is outside storage root: {path}"
            raise PermissionError(msg)
        return resolved

    async def list_files(self) -> list[StorageFile]:
        """Recursively list all supported files under root."""
        files: list[StorageFile] = []
        for path in self.root.rglob("*"):
            if path.is_file() and _is_supported(path):
                try:
                    files.append(_to_storage_file(path, self.root))
                except OSError:
                    continue
        return files

    async def read_file(self, path: str) -> bytes:
        """Read file bytes, rejecting paths outside the storage root."""
        return self._resolve_safe(path).read_bytes()

    async def file_exists(self, path: str) -> bool:
        """Check file existence within the storage root."""
        try:
            return self._resolve_safe(path).is_file()
        except PermissionError:
            return False

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
        """Watch the directory tree for changes using watchdog.

        ``poll_interval_seconds`` is accepted for interface compatibility but is
        ignored by the local backend – filesystem events are delivered instantly
        by the OS. The coroutine blocks until :meth:`close` is called or the
        task is cancelled.
        """
        # Guard against double-watch on the same instance.
        async with self._lock:
            if self._observer is not None:
                msg = "LocalStorage is already watching"
                raise RuntimeError(msg)

        loop = asyncio.get_running_loop()
        handler = _LocalEventHandler(self.root, on_event, loop)
        observer = Observer()
        observer.schedule(handler, str(self.root), recursive=True)
        observer.start()

        stop_event = asyncio.Event()
        # Publish under lock.
        async with self._lock:
            self._observer = observer
            self._stop_event = stop_event

        logger.info("Started local storage watcher", root=str(self.root))

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            logger.info("Local watcher cancelled", root=str(self.root))
            raise
        finally:
            # Ensure observer is stopped exactly once.
            async with self._lock:
                obs = self._observer
                evt = self._stop_event
                self._observer = None
                self._stop_event = None

            if obs is not None:
                try:
                    obs.stop()
                except Exception:
                    pass
                # join() blocks; run in thread pool to avoid blocking the event loop.
                try:
                    await asyncio.to_thread(obs.join, 2.0)
                except Exception:
                    pass
                if obs.is_alive():
                    logger.warning("Local watcher thread did not exit promptly")

            # Unblock any concurrent close() waiter.
            if evt is not None and not evt.is_set():
                evt.set()

            logger.info("Stopped local storage watcher", root=str(self.root))

    async def close(self) -> None:
        """Stop the watcher if running. Idempotent."""
        # Snapshot under lock.
        async with self._lock:
            stop_event = self._stop_event
            observer = self._observer

        if stop_event is not None and not stop_event.is_set():
            stop_event.set()

        # If watch() is currently in its ``finally`` block it will handle
        # observer shutdown. If close() is called outside watch() (e.g. tests
        # that never started watch), we still need to stop the observer.
        if observer is not None:
            # Only try to join here if watch() is not running (watch will
            # clear _observer before we get here in the normal shutdown path).
            # We check again under lock to avoid double-join.
            async with self._lock:
                # Re-read – watch's finally may have cleared it.
                current_obs = self._observer
                if current_obs is not observer:
                    # watch() already took ownership; nothing to do.
                    return
            try:
                observer.stop()
            except Exception:
                pass
            try:
                await asyncio.to_thread(observer.join, 2.0)
            except Exception:
                pass
