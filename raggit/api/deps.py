"""FastAPI dependencies for the raggit API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from raggit.db.session import get_session


async def _get_db_session() -> AsyncGenerator[AsyncSession]:
    """Yield an async database session for a request."""
    async for session in get_session():
        yield session


SessionDep = Annotated[AsyncSession, Depends(_get_db_session)]
