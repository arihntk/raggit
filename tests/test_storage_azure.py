"""Tests for the Azure Blob storage backend."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.azure import AzureBlobStorage
from raggit.storage.base import FileAddedEvent


def _make_azure_storage() -> AzureBlobStorage:
    config = StorageConfig(
        source_type=SourceType.AZURE_BLOB,
        uri="azure://my-container/docs",
        container="my-container",
        prefix="docs",
        azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net",
    )
    return AzureBlobStorage(config)


class _AsyncIterator:
    def __init__(self, items: list) -> None:
        self.iterator = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.fixture
def mock_azure_client():
    client = MagicMock()
    with patch("azure.storage.blob.aio.BlobServiceClient") as mock_client_cls:
        mock_client_cls.from_connection_string.return_value = client
        yield client


async def test_list_files(mock_azure_client) -> None:
    blob1 = MagicMock()
    blob1.name = "docs/report.pdf"
    blob1.size = 1024
    blob1.last_modified = datetime(2024, 1, 1, tzinfo=UTC)

    blob2 = MagicMock()
    blob2.name = "docs/notes.txt"
    blob2.size = 64
    blob2.last_modified = datetime(2024, 1, 2, tzinfo=UTC)

    blob3 = MagicMock()
    blob3.name = "docs/ignore.exe"
    blob3.size = 32
    blob3.last_modified = datetime(2024, 1, 3, tzinfo=UTC)

    container_client = MagicMock()
    container_client.list_blobs.return_value = _AsyncIterator([blob1, blob2, blob3])
    mock_azure_client.get_container_client.return_value = container_client

    storage = _make_azure_storage()
    files = await storage.list_files()

    assert len(files) == 2
    assert files[0].path == "azure://my-container/docs/report.pdf"
    assert files[1].path == "azure://my-container/docs/notes.txt"


async def test_read_file(mock_azure_client) -> None:
    downloader = AsyncMock()
    downloader.readall = AsyncMock(return_value=b"pdf bytes")
    blob_client = MagicMock()
    blob_client.download_blob = AsyncMock(return_value=downloader)
    container_client = MagicMock()
    container_client.get_blob_client.return_value = blob_client
    mock_azure_client.get_container_client.return_value = container_client

    storage = _make_azure_storage()
    data = await storage.read_file("azure://my-container/docs/report.pdf")
    assert data == b"pdf bytes"
    container_client.get_blob_client.assert_called_once_with("docs/report.pdf")


async def test_file_exists(mock_azure_client) -> None:
    blob_client = MagicMock()
    blob_client.exists = AsyncMock(return_value=True)
    container_client = MagicMock()
    container_client.get_blob_client.return_value = blob_client
    mock_azure_client.get_container_client.return_value = container_client

    storage = _make_azure_storage()
    assert await storage.file_exists("azure://my-container/docs/report.pdf") is True


async def test_compute_hash(mock_azure_client) -> None:
    properties = MagicMock()
    properties.content_settings.content_md5 = bytes.fromhex("abc123")
    properties.etag = '"etag123"'
    blob_client = MagicMock()
    blob_client.get_blob_properties = AsyncMock(return_value=properties)
    container_client = MagicMock()
    container_client.get_blob_client.return_value = blob_client
    mock_azure_client.get_container_client.return_value = container_client

    storage = _make_azure_storage()
    assert await storage.compute_hash("azure://my-container/docs/report.pdf") == "abc123"


async def test_watch_emits_events() -> None:
    file = MagicMock(
        path="azure://my-container/docs/report.pdf",
        relative_path="docs/report.pdf",
        size=1024,
        modified_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    calls: list[int] = []

    async def mock_list_files() -> list:
        if not calls:
            calls.append(1)
            return []
        return [file]

    storage = _make_azure_storage()
    storage.list_files = mock_list_files  # type: ignore[method-assign]

    events = []
    event_received = asyncio.Event()

    async def on_event(event) -> None:
        events.append(event)
        event_received.set()

    task = asyncio.create_task(storage.watch(on_event, poll_interval_seconds=0.01))
    await asyncio.wait_for(event_received.wait(), timeout=1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(events) == 1
    assert isinstance(events[0], FileAddedEvent)
