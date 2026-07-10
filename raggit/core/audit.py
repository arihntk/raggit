"""Audit logging helper for persisting inputs/outputs to Postgres.

The audit log captures the principal inputs and outputs of the system so
operators can inspect what was requested, what context was retrieved, and
what answers were produced. It is intentionally separate from operational
console logging and writes synchronously through the provided async session.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from raggit.db.repository import LogRepository


def _serialize_extra(extra: dict[str, Any] | None) -> str | None:
    """Serialize extra dict to a compact JSON string, skipping None values."""
    if not extra:
        return None
    cleaned = {k: v for k, v in extra.items() if v is not None}
    if not cleaned:
        return None
    return json.dumps(cleaned, default=str, separators=(",", ":"))


async def log_event(
    session: AsyncSession,
    level: str,
    component: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a structured audit event to the Postgres `logs` table.

    Args:
        session: Active async SQLAlchemy session.
        level: Event severity (e.g. INFO, ERROR).
        component: Logical component emitting the event (e.g. raggit.cli.query).
        message: Human-readable summary.
        extra: JSON-serializable key/value context for the event.
    """
    repo = LogRepository(session)
    await repo.create(
        level=level.upper(),
        component=component,
        message=message,
        extra=_serialize_extra(extra),
    )
