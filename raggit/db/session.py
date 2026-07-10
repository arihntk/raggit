"""Async database session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from raggit.core.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return (and lazily create) the shared async engine."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.log_level.upper() == "DEBUG",
            future=True,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared session factory, creating the engine if needed."""
    get_engine()
    assert _session_factory is not None
    return _session_factory


class _SessionLocalProxy:
    """Callable/context-manager proxy matching async_sessionmaker usage."""

    def __call__(self) -> AsyncSession:
        return get_session_factory()()

    def __getattr__(self, name: str) -> object:
        return getattr(get_session_factory(), name)


# Backwards-compatible name used throughout the codebase.
AsyncSessionLocal = _SessionLocalProxy()


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Yield an async database session."""
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine (useful in tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
