"""Robust document watcher service – single canonical implementation.

The service runs an initial full sync, then listens for storage events and
indexes changed files with per-path debouncing and file-stability checks.
It is designed to run independently as a background task (inside
``raggit serve`` or the FastAPI lifespan) without requiring manual
``raggit watch`` invocations.

All storage backends are supported:

- Local filesystem: OS-native events via watchdog (instant, no polling).
- Cloud (S3/GCS/Azure): polling with snapshot diffing.

The service is safe to start/stop multiple times and handles cancellation,
supervision, and graceful shutdown without leaking tasks or threads.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from raggit.api.models import RAGConfig
from raggit.core.audit import log_event
from raggit.core.logging import get_logger
from raggit.db.session import AsyncSessionLocal
from raggit.ingestion.indexer import Indexer
from raggit.storage.base import (
    FileAddedEvent,
    FileDeletedEvent,
    FileEvent,
    FileModifiedEvent,
)
from raggit.storage.factory import create_storage

logger = get_logger("raggit.core.watcher")

EventCallback = Callable[[FileEvent], Any] | None
WatcherState = Literal["idle", "starting", "running", "stopping", "stopped", "failed"]


class WatcherService:
    """Single-process watcher that keeps storage and index in sync.

    Lifecycle
    ---------
    1. ``await watcher.start()`` – runs an initial ``sync_all`` inside a DB
       transaction, then spawns a background ``storage.watch`` task.
       The call returns *immediately* after the sync; the watch loop runs
       independently in the background.
    2. File events are debounced per path (``debounce_seconds``) and, for
       local storage, waited until the file size is stable
       (``file_stability_delay``) before indexing. This avoids indexing
       half-written files.
    3. ``await watcher.stop()`` – cancels debounces, cancels the watch loop,
       closes storage and the indexer, and emits an audit log.

    The service is idempotent: ``start`` while already running is a no-op,
    and ``stop`` while idle is a no-op. It can be used as an async context
    manager or supervised by ``raggit serve`` / FastAPI lifespan.
    """

    def __init__(
        self,
        config: RAGConfig,
        *,
        debounce_seconds: float = 0.5,
        file_stability_delay: float = 0.4,
        on_event: EventCallback = None,
    ) -> None:
        if config.storage is None:
            msg = "WatcherService requires a storage configuration"
            raise ValueError(msg)
        self.config = config
        self.debounce_seconds = debounce_seconds
        self.file_stability_delay = file_stability_delay
        self._on_event_callback = on_event
        self.storage = create_storage(config.storage)
        self.indexer = Indexer(self.storage, config)
        self._pending: dict[str, asyncio.Task[None]] = {}
        self._watch_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._state: WatcherState = "idle"

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def state(self) -> WatcherState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == "running" and self._watch_task is not None and not self._watch_task.done()

    @property
    def is_stopped(self) -> bool:
        return self._state in ("idle", "stopped", "failed")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run initial sync and start listening for events."""
        async with self._lock:
            if self._state in ("starting", "running"):
                logger.warning("Watcher already running", state=self._state)
                return
            if self._state == "stopping":
                logger.warning("Watcher is stopping, cannot start now")
                return
            self._state = "starting"

        logger.info(
            "Starting watcher service",
            storage_type=self.storage.source_type,
            uri=self.config.storage.uri if self.config.storage else None,
        )

        # Initial sync – failures should not leave service in running state.
        try:
            async with AsyncSessionLocal() as session, session.begin():
                await log_event(
                    session,
                    level="INFO",
                    component="raggit.core.watcher",
                    message="Watcher service started",
                    extra={
                        "storage_type": self.storage.source_type,
                        "uri": self.config.storage.uri if self.config.storage else None,
                    },
                )
                await self.indexer.sync_all(session)
        except Exception:
            async with self._lock:
                self._state = "failed"
            logger.exception("Watcher initial sync failed")
            raise

        poll_interval = (
            self.config.storage.poll_interval_seconds
            if self.config.storage is not None
            else 30.0
        )

        # Spawn background watch loop. It runs until cancelled via stop().
        self._watch_task = asyncio.create_task(
            self._run_watch_loop(poll_interval),
            name="raggit-watcher",
        )
        self._watch_task.add_done_callback(self._on_watch_done)

        async with self._lock:
            self._state = "running"
        logger.info("Watcher service is running", poll_interval=poll_interval)

    async def _run_watch_loop(self, poll_interval: float) -> None:
        """Await the storage backend's watch indefinitely until cancelled."""
        try:
            await self.storage.watch(self._on_event, poll_interval_seconds=poll_interval)
        except asyncio.CancelledError:
            logger.info("Watcher watch loop cancelled")
            raise
        except Exception:
            logger.exception("Watcher watch loop crashed")
            raise

    def _on_watch_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Watcher background task failed", error=str(exc))
            # Mark failed if we are still supposed to be running.
            if self._state == "running":
                self._state = "failed"

    async def stop(self) -> None:
        """Stop listening and release resources. Idempotent."""
        async with self._lock:
            if self._state in ("idle", "stopped"):
                return
            if self._state == "stopping":
                return
            # Allow stopping from starting/running/failed
            prev_state = self._state
            self._state = "stopping"
            logger.info("Stopping watcher service", prev_state=prev_state)

        # Cancel pending debounce tasks.
        for task in list(self._pending.values()):
            task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending.values(), return_exceptions=True)
        self._pending.clear()

        # Cancel background watch task.
        watch_task = self._watch_task
        self._watch_task = None
        if watch_task is not None:
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                try:
                    await asyncio.wait_for(watch_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Watcher task did not stop within timeout")

        # Close storage and indexer – must be idempotent.
        try:
            await self.storage.close()
        except Exception:
            logger.exception("Error closing storage during watcher stop")

        try:
            await self.indexer.close()
        except Exception:
            logger.exception("Error closing indexer during watcher stop")

        # Audit log – best effort.
        try:
            async with AsyncSessionLocal() as session, session.begin():
                await log_event(
                    session,
                    level="INFO",
                    component="raggit.core.watcher",
                    message="Watcher service stopped",
                    extra={
                        "storage_type": self.storage.source_type,
                        "uri": self.config.storage.uri if self.config.storage else None,
                    },
                )
        except Exception:
            logger.exception("Failed to write watcher stop audit log")

        async with self._lock:
            self._state = "stopped"
        logger.info("Watcher service stopped")

    # ------------------------------------------------------------------
    # Async context manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> WatcherService:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_event(self, event: FileEvent) -> None:
        """Handle a storage event with lightweight per-path debouncing."""
        path = event.file.path
        logger.debug("Received storage event", path=path, event_type=type(event).__name__)

        if self._on_event_callback is not None:
            try:
                result = self._on_event_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in user event callback", path=path)

        existing = self._pending.pop(path, None)
        if existing is not None:
            existing.cancel()
            # Do not await – let the cancelled task exit via CancelledError.

        self._pending[path] = asyncio.create_task(
            self._process_event_after_debounce(path, event),
            name=f"raggit-debounce-{Path(path).name}",
        )

    async def _process_event_after_debounce(
        self, path: str, event: FileEvent
    ) -> None:
        """Wait briefly, then process the event unless superseded."""
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return

        # If a newer event for the same path replaced this task, the pending
        # dict will point to a different task object. In that case this older
        # debounce should not process the event (it was superseded). We detect
        # that by checking if the current pending entry is still *this* task.
        # However we already popped the old task in _on_event before creating
        # the new one, so the dict now points to this task unless a newer event
        # arrived and overwrote it. In the overwrite case this task was cancelled
        # and would have returned above. So reaching here means we are the latest.
        # Pop ourselves before processing to avoid stale entries.
        current = self._pending.get(path)
        # ``current`` should be this task; if it was already removed/cancelled,
        # it is None. If a newer task replaced it, ``current`` is the newer task,
        # and we should not pop it nor process the stale event.
        this_task = asyncio.current_task()
        if current is not None and current is not this_task:
            # Stale – a newer event superseded us but we were not cancelled
            # in time (race). Do not process or pop the newer task.
            return
        self._pending.pop(path, None)

        # File-stability guard for local backends – avoid indexing partial writes.
        # Only run for real LocalStorage (not mock storages used in tests).
        if not isinstance(event, FileDeletedEvent) and self.storage.source_type == "local":
            # Late import to avoid circular dependency.
            try:
                from raggit.storage.local import LocalStorage

                is_real_local = isinstance(self.storage, LocalStorage)
            except Exception:
                is_real_local = False
            if is_real_local:
                try:
                    stable = await self._wait_for_file_stable(path)
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("File stability check failed", path=path)
                    stable = True
                if not stable:
                    logger.warning("File did not become stable, skipping", path=path)
                    return

        try:
            async with AsyncSessionLocal() as session, session.begin():
                if isinstance(event, (FileAddedEvent, FileModifiedEvent)):
                    await self.indexer.index_file(session, event.file)
                elif isinstance(event, FileDeletedEvent):
                    await self.indexer.remove_file(session, event.file)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error processing storage event", path=path)

    async def _wait_for_file_stable(self, path: str, timeout: float = 8.0) -> bool:
        """Wait until a local file's size is stable for ``file_stability_delay``.

        Returns True if the file appears stable, False if the file vanished or
        did not stabilize within ``timeout``.
        """
        if self.file_stability_delay <= 0:
            return True

        # Resolve the absolute path; storage may give us an absolute resolved path.
        p = Path(path)
        loop = asyncio.get_running_loop()
        start = loop.time()
        last_size: int | None = None
        stable_since: float | None = None

        while True:
            if loop.time() - start > timeout:
                return False
            try:
                # Use file_exists via storage if available, falling back to Path.
                exists = await self.storage.file_exists(path)
                if not exists:
                    # File was deleted before we could index it – not stable.
                    return False
                # For local storage we can stat directly for speed.
                try:
                    size = p.stat().st_size
                except OSError:
                    # File may be in a transient state.
                    size = -1
            except Exception:
                size = -1

            now = loop.time()
            if last_size is not None and size == last_size and size != -1:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= self.file_stability_delay:
                    return True
            else:
                stable_since = None
            last_size = size
            await asyncio.sleep(0.15)
