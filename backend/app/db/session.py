"""Async engine and session handling.

The engine is built lazily on first use so importing this module never opens a
socket — Celery workers, Alembic and the test suite all import models without
wanting a connection pool.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

_DEFAULT_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/orchestrator"

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _coerce_async_url(url: str) -> str:
    """Force the asyncpg driver regardless of how the DSN was written."""
    if url.startswith("postgresql+asyncpg://") or url.startswith("postgres+asyncpg://"):
        return url.replace("postgres+asyncpg://", "postgresql+asyncpg://", 1)
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    return url


def database_url() -> str:
    """The DSN, from app.config if it is importable, else the environment."""
    try:  # app.config is owned by another module; do not hard-fail on import order
        from app.config import settings  # type: ignore[import-not-found]

        for attr in ("database_url", "DATABASE_URL", "postgres_dsn", "db_url"):
            value = getattr(settings, attr, None)
            if value:
                return _coerce_async_url(str(value))
    except Exception:  # pragma: no cover - config not present yet
        pass
    return _coerce_async_url(os.environ.get("DATABASE_URL", _DEFAULT_URL))


def _engine_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": os.environ.get("SQL_ECHO", "").lower() in {"1", "true", "yes"},
        "future": True,
        # Pre-ping is OFF for asyncpg on purpose.
        #
        # It issues a real round trip during checkout. On the async engine that
        # checkout can happen from a context SQLAlchemy did not enter through
        # its greenlet bridge — the resume path opens a fresh session inside a
        # gather of step tasks — and the ping then raises MissingGreenlet from
        # deep inside the pool, long after the work it was guarding has already
        # committed. The endpoint 500s on a request that succeeded.
        #
        # pool_recycle below covers what pre-ping was there for: connections are
        # retired before Postgres or anything in between drops them.
        "pool_pre_ping": False,
        # asyncpg caches prepared statements per connection; PgBouncer in
        # transaction mode hates that, so allow turning it off by env.
        "connect_args": {},
    }
    if os.environ.get("DB_DISABLE_STATEMENT_CACHE", "").lower() in {"1", "true", "yes"}:
        kwargs["connect_args"]["statement_cache_size"] = 0
        kwargs["connect_args"]["prepared_statement_cache_size"] = 0
    if os.environ.get("DB_NULL_POOL", "").lower() in {"1", "true", "yes"}:
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_size"] = int(os.environ.get("DB_POOL_SIZE", "10"))
        kwargs["max_overflow"] = int(os.environ.get("DB_MAX_OVERFLOW", "20"))
        kwargs["pool_recycle"] = int(os.environ.get("DB_POOL_RECYCLE", "1800"))
        kwargs["pool_timeout"] = float(os.environ.get("DB_POOL_TIMEOUT", "10"))
    return kwargs


def get_engine() -> AsyncEngine:
    """The process-wide async engine, built on first call."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(database_url(), **_engine_kwargs())
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """The process-wide session factory."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """A session wrapped in one transaction: commit on success, rollback on error.

    This is the unit the implementation rules ask for — an assistant message and
    every prompt and action its content references are written together.
    """
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
        await session.commit()
    except BaseException:
        # `BaseException`, not `Exception`, and that is the whole point.
        #
        # `asyncio.CancelledError` is a BaseException, and a cancelled request
        # is the *normal* case here: somebody navigates away mid-stream, or a
        # client drops an SSE connection. Catching only `Exception` let that
        # skip the rollback, and `close()` then handed a connection back to the
        # pool still inside a transaction. The next request to borrow it died
        # on its very first read with "cannot use Connection.transaction() in a
        # manually started transaction" — a failure with no visible connection
        # to the request that actually caused it.
        try:
            await session.rollback()
        except Exception:  # noqa: BLE001 - the connection may already be gone
            pass
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Commits when the request handler returns cleanly."""
    async with session_scope() as session:
        yield session


async def shutdown_engine() -> None:
    """Dispose the pool. Called from the app lifespan and worker shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
