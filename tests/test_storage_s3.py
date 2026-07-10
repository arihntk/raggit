"""Tests for the S3 storage backend."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raggit.api.models import SourceType, StorageConfig
from raggit.storage.base import FileAddedEvent
from raggit.storage.s3 import S3Storage


def _make_s3_storage() -> S3Storage:
    config = StorageConfig(
        source_type=SourceType.S3,
        uri="s3://my-bucket/docs",
        bucket="my-bucket",
        prefix="docs",
        region="us-east-1",
        aws_access_key_id="AKIA",
        aws_secret_access_key="secret",
    )
    return S3Storage(config)


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
def mock_s3_client():
    client = MagicMock()
    client.exceptions.ClientError = Exception
    with patch("aiobotocore.session.get_session") as mock_session:
        mock_session.return_value.create_client.return_value = client
        yield client


async def test_list_files(mock_s3_client) -> None:
    paginator = MagicMock()
    paginator.paginate.return_value = _AsyncIterator(
        [
            {
                "Contents": [
                    {
                        "Key": "docs/report.pdf",
                        "Size": 1024,
                        "LastModified": datetime(2024, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "docs/notes.txt",
                        "Size": 64,
                        "LastModified": datetime(2024, 1, 2, tzinfo=UTC),
                    },
                    {
                        "Key": "docs/ignore.exe",
                        "Size": 32,
                        "LastModified": datetime(2024, 1, 3, tzinfo=UTC),
                    },
                ]
            }
        ]
    )
    mock_s3_client.get_paginator.return_value = paginator

    storage = _make_s3_storage()
    files = await storage.list_files()

    assert len(files) == 2
    assert files[0].path == "s3://my-bucket/docs/report.pdf"
    assert files[0].relative_path == "docs/report.pdf"
    assert files[0].size == 1024
    assert files[1].path == "s3://my-bucket/docs/notes.txt"


async def test_read_file(mock_s3_client) -> None:
    body = AsyncMock()
    body.__aenter__ = AsyncMock(return_value=body)
    body.__aexit__ = AsyncMock(return_value=None)
    body.read = AsyncMock(return_value=b"pdf bytes")
    mock_s3_client.get_object = AsyncMock(return_value={"Body": body})

    storage = _make_s3_storage()
    data = await storage.read_file("s3://my-bucket/docs/report.pdf")
    assert data == b"pdf bytes"
    mock_s3_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="docs/report.pdf")


async def test_file_exists(mock_s3_client) -> None:
    mock_s3_client.head_object = AsyncMock(return_value={})
    storage = _make_s3_storage()
    assert await storage.file_exists("s3://my-bucket/docs/report.pdf") is True


async def test_file_missing(mock_s3_client) -> None:
    error = Exception("not found")
    error.response = {"Error": {"Code": "404"}}
    mock_s3_client.head_object = AsyncMock(side_effect=error)
    storage = _make_s3_storage()
    assert await storage.file_exists("s3://my-bucket/docs/missing.pdf") is False


async def test_compute_hash(mock_s3_client) -> None:
    mock_s3_client.head_object = AsyncMock(return_value={"ETag": '"abc123"'})
    storage = _make_s3_storage()
    assert await storage.compute_hash("s3://my-bucket/docs/report.pdf") == "abc123"


async def test_watch_emits_events(mock_s3_client) -> None:
    file = MagicMock(
        path="s3://my-bucket/docs/report.pdf",
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

    storage = _make_s3_storage()
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
    assert events[0].file.path == "s3://my-bucket/docs/report.pdf"
