"""Abstract storage backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StorageFile:
    """Represents a file known to a storage backend."""

    path: str
    relative_path: str
    size: int
    modified_at: datetime
    content_hash: str | None = None


class FileEvent:
    """Base class for file watcher events."""

    def __init__(self, file: StorageFile) -> None:
        self.file = file


class FileAddedEvent(FileEvent):
    """Emitted when a new file is detected."""


class FileModifiedEvent(FileEvent):
    """Emitted when an existing file changes."""


class FileDeletedEvent(FileEvent):
    """Emitted when a file is removed."""


FileEventCallback = Callable[[FileEvent], Awaitable[None]]


class Storage(ABC):
    """Abstract document storage backend."""

    source_type: str

    @abstractmethod
    async def list_files(self) -> list[StorageFile]:
        """Return all files currently available in storage."""

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read raw bytes for a file."""

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Return True if the file exists in storage."""

    @abstractmethod
    async def compute_hash(self, path: str) -> str:
        """Compute a stable hash for the file contents."""

    @abstractmethod
    async def watch(
        self,
        on_event: FileEventCallback,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """Start watching for changes and call on_event for each change.

        This method is expected to run indefinitely.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the backend."""
