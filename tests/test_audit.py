"""Tests for structured audit logging."""

from unittest.mock import AsyncMock, MagicMock

from raggit.core.audit import log_event


async def test_log_event_persists_json_extra() -> None:
    mock_session = MagicMock()
    mock_repo = AsyncMock()
    mock_session.attach_mock(mock_repo, "add")
    # Simulate LogRepository by returning the added model
    created_logs: list[object] = []

    class FakeRepo:
        def __init__(self, session: object) -> None:
            self.session = session

        async def create(
            self,
            level: str,
            component: str,
            message: str,
            extra: str | None = None,
        ) -> object:
            created_logs.append(
                {"level": level, "component": component, "message": message, "extra": extra}
            )
            return MagicMock()

    # Patch LogRepository on the module under test
    from raggit.core import audit
    original_repo = audit.LogRepository
    audit.LogRepository = FakeRepo

    try:
        await log_event(
            mock_session,
            level="INFO",
            component="raggit.test",
            message="hello audit",
            extra={"question": "what is raggit", "answer": "a RAG system"},
        )
    finally:
        audit.LogRepository = original_repo

    assert len(created_logs) == 1
    log = created_logs[0]
    assert log["level"] == "INFO"
    assert log["component"] == "raggit.test"
    assert log["message"] == "hello audit"
    assert "raggit" in log["extra"]
    assert "a RAG system" in log["extra"]


async def test_log_event_skips_empty_extra() -> None:
    mock_session = MagicMock()
    created_logs: list[object] = []

    class FakeRepo:
        def __init__(self, session: object) -> None:
            self.session = session

        async def create(
            self,
            level: str,
            component: str,
            message: str,
            extra: str | None = None,
        ) -> object:
            created_logs.append(
                {"level": level, "component": component, "message": message, "extra": extra}
            )
            return MagicMock()

    from raggit.core import audit
    original_repo = audit.LogRepository
    audit.LogRepository = FakeRepo

    try:
        await log_event(
            mock_session,
            level="ERROR",
            component="raggit.test",
            message="empty extra",
            extra={},
        )
    finally:
        audit.LogRepository = original_repo

    assert len(created_logs) == 1
    assert created_logs[0]["extra"] is None
