"""Tests for the GCS storage backend."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.base import FileAddedEvent
from raggit.storage.gcs import GCSStorage


def _make_gcs_storage() -> GCSStorage:
    config = StorageConfig(
        source_type=SourceType.GCS,
        uri="gs://my-bucket/docs",
        bucket="my-bucket",
        prefix="docs",
        gcs_service_account_path="/path/to/sa.json",
    )
    return GCSStorage(config)


@pytest.fixture
def mock_gcs_client():
    client = MagicMock()
    credentials = MagicMock()
    credentials.project_id = "test-project"
    with (
        patch("google.cloud.storage.Client") as mock_client_cls,
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_file"
        ) as mock_creds,
    ):
        mock_creds.return_value = credentials
        mock_client_cls.return_value = client
        yield client


async def test_list_files(mock_gcs_client) -> None:
    blob1 = MagicMock()
    blob1.name = "docs/report.pdf"
    blob1.size = 1024
    blob1.updated = datetime(2024, 1, 1, tzinfo=UTC)

    blob2 = MagicMock()
    blob2.name = "docs/notes.txt"
    blob2.size = 64
    blob2.updated = datetime(2024, 1, 2, tzinfo=UTC)

    blob3 = MagicMock()
    blob3.name = "docs/ignore.exe"
    blob3.size = 32
    blob3.updated = datetime(2024, 1, 3, tzinfo=UTC)

    bucket = MagicMock()
    bucket.list_blobs.return_value = [blob1, blob2, blob3]
    mock_gcs_client.bucket.return_value = bucket

    storage = _make_gcs_storage()
    files = await storage.list_files()

    assert len(files) == 2
    assert files[0].path == "gs://my-bucket/docs/report.pdf"
    assert files[1].path == "gs://my-bucket/docs/notes.txt"


async def test_read_file(mock_gcs_client) -> None:
    blob = MagicMock()
    blob.download_as_bytes.return_value = b"pdf bytes"
    bucket = MagicMock()
    bucket.blob.return_value = blob
    mock_gcs_client.bucket.return_value = bucket

    storage = _make_gcs_storage()
    data = await storage.read_file("gs://my-bucket/docs/report.pdf")
    assert data == b"pdf bytes"
    bucket.blob.assert_called_once_with("docs/report.pdf")


async def test_file_exists(mock_gcs_client) -> None:
    blob = MagicMock()
    blob.exists = True
    bucket = MagicMock()
    bucket.blob.return_value = blob
    mock_gcs_client.bucket.return_value = bucket

    storage = _make_gcs_storage()
    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
        assert await storage.file_exists("gs://my-bucket/docs/report.pdf") is True


async def test_compute_hash(mock_gcs_client) -> None:
    blob = MagicMock()
    blob.md5_hash = "abc123"
    blob.crc32c = None
    blob.reload = MagicMock(return_value=None)
    bucket = MagicMock()
    bucket.blob.return_value = blob
    mock_gcs_client.bucket.return_value = bucket

    storage = _make_gcs_storage()
    assert await storage.compute_hash("gs://my-bucket/docs/report.pdf") == "abc123"
    blob.reload.assert_called_once()


async def test_watch_emits_events() -> None:
    file = MagicMock(
        path="gs://my-bucket/docs/report.pdf",
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

    storage = _make_gcs_storage()
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
