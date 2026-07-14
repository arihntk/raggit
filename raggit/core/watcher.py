"""Lightweight document watcher service."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

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


class WatcherService:
    """Lightweight single-process watcher that keeps storage and index in sync.

    The service runs an initial full sync, then listens for storage events and
    indexes changed files. Local filesystem backends receive instant OS-native
    events; cloud backends fall back to polling.
    """

    def __init__(
        self,
        config: RAGConfig,
        *,
        debounce_seconds: float = 0.5,
        on_event: EventCallback = None,
    ) -> None:
        if config.storage is None:
            msg = "WatcherService requires a storage configuration"
            raise ValueError(msg)
        self.config = config
        self.debounce_seconds = debounce_seconds
        self._on_event_callback = on_event
        self.storage = create_storage(config.storage)
        self.indexer = Indexer(self.storage, config)
        self._pending: dict[str, asyncio.Task[None]] = {}
        self._watch_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Run initial sync and start listening for events."""
        logger.info(
            "Starting watcher service",
            storage_type=self.storage.source_type,
            uri=self.config.storage.uri if self.config.storage else None,
        )

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

        self._watch_task = asyncio.create_task(self.storage.watch(self._on_event))
        logger.info("Watcher service is running")

    async def stop(self) -> None:
        """Stop listening and release resources."""
        logger.info("Stopping watcher service")

        # Cancel pending debounce tasks.
        for task in list(self._pending.values()):
            task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending.values(), return_exceptions=True)
        self._pending.clear()

        await self.storage.close()
        if self._watch_task is not None:
            await asyncio.wait_for(self._watch_task, timeout=5.0)
            self._watch_task = None
        await self.indexer.close()

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

        logger.info("Watcher service stopped")

    async def _on_event(self, event: FileEvent) -> None:
        """Handle a storage event with lightweight per-path debouncing."""
        path = event.file.path
        logger.debug("Received storage event", path=path, event_type=type(event).__name__)

        if self._on_event_callback is not None:
            try:
                self._on_event_callback(event)
            except Exception:
                logger.exception("Error in user event callback", path=path)

        existing = self._pending.pop(path, None)
        if existing is not None:
            existing.cancel()

        self._pending[path] = asyncio.create_task(
            self._process_event_after_debounce(path, event)
        )

    async def _process_event_after_debounce(
        self, path: str, event: FileEvent
    ) -> None:
        """Wait briefly, then process the event unless superseded."""
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return

        self._pending.pop(path, None)

        try:
            async with AsyncSessionLocal() as session, session.begin():
                if isinstance(event, (FileAddedEvent, FileModifiedEvent)):
                    await self.indexer.index_file(session, event.file)
                elif isinstance(event, FileDeletedEvent):
                    await self.indexer.remove_file(session, event.file)
        except Exception:
            logger.exception("Error processing storage event", path=path)
