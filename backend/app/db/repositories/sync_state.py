"""sync_state — where each sync got to.

``cursor`` holds whatever the service uses: Gmail's historyId, Calendar's
syncToken, Drive's pageToken. It is advanced only after the upsert commits, so
a crash reprocesses rather than skips.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import new_id
from app.db.models import SyncService, SyncState
from app.db.repositories import ensure_utc, utcnow

#: Backoff ladder for the circuit breaker, indexed by consecutive failures.
_CIRCUIT_BACKOFF = (60, 300, 900, 1800, 3600)


async def get_state(
    session: AsyncSession, user_id: str, service: SyncService | str
) -> SyncState | None:
    """One service's state for one user."""
    result = await session.execute(
        select(SyncState).where(
            SyncState.user_id == user_id, SyncState.service == SyncService(service)
        )
    )
    return result.scalar_one_or_none()


async def get_all_states(session: AsyncSession, user_id: str) -> list[SyncState]:
    """All three services — what /sync/status reads."""
    result = await session.execute(
        select(SyncState)
        .where(SyncState.user_id == user_id)
        .order_by(SyncState.service.asc())
    )
    return list(result.scalars().all())


async def ensure_state(
    session: AsyncSession, user_id: str, service: SyncService | str
) -> SyncState:
    """Find-or-create. Safe to call from every sync task."""
    stmt = (
        pg_insert(SyncState)
        .values(
            id=new_id(),
            user_id=user_id,
            service=SyncService(service),
            updated_at=utcnow(),
        )
        .on_conflict_do_update(
            index_elements=[SyncState.user_id, SyncState.service],
            set_={"updated_at": utcnow()},
        )
        .returning(SyncState)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def set_cursor(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    cursor: dict[str, Any] | None,
) -> SyncState | None:
    """Advance the incremental cursor. Call this after the upsert has committed,
    never before."""
    result = await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(cursor=cursor, updated_at=utcnow())
        .returning(SyncState)
    )
    return result.scalar_one_or_none()


async def set_backfill(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    *,
    backfill_cursor: dict[str, Any] | None = None,
    complete: bool | None = None,
) -> SyncState | None:
    """Move the historical backfill along, or mark it finished."""
    values: dict[str, Any] = {"updated_at": utcnow()}
    if backfill_cursor is not None or complete:
        values["backfill_cursor"] = None if complete else backfill_cursor
    if complete is not None:
        values["backfill_complete"] = complete
    result = await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(**values)
        .returning(SyncState)
    )
    return result.scalar_one_or_none()


async def mark_attempt(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    *,
    at: dt.datetime | None = None,
) -> None:
    """A sync run started. ``last_synced_at`` is the attempt; ``last_success_at``
    is what the lag figure reads."""
    now = ensure_utc(at) or utcnow()
    await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(last_synced_at=now, updated_at=now)
    )


async def mark_success(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    *,
    items_indexed: int = 0,
    cursor: dict[str, Any] | None = None,
    at: dt.datetime | None = None,
) -> SyncState | None:
    """A clean pass: clear the error, close the circuit, count the items.

    Passing ``cursor`` here is the normal path — the cursor and the success
    stamp move together, in the transaction that follows the committed upsert.
    """
    now = ensure_utc(at) or utcnow()
    values: dict[str, Any] = {
        "last_synced_at": now,
        "last_success_at": now,
        "last_error": None,
        "consecutive_failures": 0,
        "circuit_open_until": None,
        "items_indexed": SyncState.items_indexed + int(items_indexed),
        "updated_at": now,
    }
    if cursor is not None:
        values["cursor"] = cursor
    result = await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(**values)
        .returning(SyncState)
    )
    return result.scalar_one_or_none()


async def mark_failure(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    error: dict[str, Any],
    *,
    open_circuit_for: float | None = None,
    at: dt.datetime | None = None,
) -> SyncState | None:
    """Count a failure and, past the first couple, hold the circuit open.

    With no explicit duration the backoff ladder is used, so a service that is
    down stops being hammered without anyone deciding that by hand.
    """
    now = ensure_utc(at) or utcnow()
    state = await get_state(session, user_id, service)
    failures = (state.consecutive_failures if state else 0) + 1

    if open_circuit_for is None:
        index = min(failures - 1, len(_CIRCUIT_BACKOFF) - 1)
        seconds = _CIRCUIT_BACKOFF[index] if failures >= 3 else 0
    else:
        seconds = float(open_circuit_for)

    result = await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(
            last_synced_at=now,
            last_error=error,
            consecutive_failures=failures,
            circuit_open_until=(now + dt.timedelta(seconds=seconds)) if seconds else None,
            updated_at=now,
        )
        .returning(SyncState)
    )
    return result.scalar_one_or_none()


async def reset_circuit(
    session: AsyncSession, user_id: str, service: SyncService | str
) -> None:
    """Force the circuit shut — a manual /sync/trigger says try again now."""
    await session.execute(
        update(SyncState)
        .where(SyncState.user_id == user_id, SyncState.service == SyncService(service))
        .values(circuit_open_until=None, consecutive_failures=0, updated_at=utcnow())
    )


async def is_open(
    session: AsyncSession,
    user_id: str,
    service: SyncService | str,
    *,
    now: dt.datetime | None = None,
) -> bool:
    """True when the circuit is holding this service back right now."""
    state = await get_state(session, user_id, service)
    if state is None or state.circuit_open_until is None:
        return False
    return state.circuit_open_until > (ensure_utc(now) or utcnow())


async def list_due(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    service: SyncService | str | None = None,
    stale_after: dt.timedelta = dt.timedelta(minutes=15),
    now: dt.datetime | None = None,
    limit: int = 500,
) -> list[SyncState]:
    """Services whose mirror has gone stale and whose circuit is shut.

    ``user_id=None`` scans every user — for ``sync.dispatch_all_users`` only.
    """
    at = ensure_utc(now) or utcnow()
    cutoff = at - stale_after
    stmt = select(SyncState).where(
        or_(SyncState.circuit_open_until.is_(None), SyncState.circuit_open_until <= at),
        or_(SyncState.last_synced_at.is_(None), SyncState.last_synced_at <= cutoff),
    )
    if user_id is not None:
        stmt = stmt.where(SyncState.user_id == user_id)
    if service is not None:
        stmt = stmt.where(SyncState.service == SyncService(service))
    stmt = stmt.order_by(SyncState.last_synced_at.asc().nullsfirst()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def freshness(
    session: AsyncSession, user_id: str, *, now: dt.datetime | None = None
) -> dict[str, dict[str, Any]]:
    """Per-service lag in seconds plus the headline fields for /sync/status."""
    at = ensure_utc(now) or utcnow()
    out: dict[str, dict[str, Any]] = {}
    for state in await get_all_states(session, user_id):
        lag = (
            (at - state.last_success_at).total_seconds()
            if state.last_success_at is not None
            else None
        )
        out[state.service.value] = {
            "lag_seconds": lag,
            "last_success_at": state.last_success_at,
            "last_synced_at": state.last_synced_at,
            "backfill_complete": state.backfill_complete,
            "items_indexed": state.items_indexed,
            "consecutive_failures": state.consecutive_failures,
            "circuit_open": (
                state.circuit_open_until is not None and state.circuit_open_until > at
            ),
            "last_error": state.last_error,
        }
    return out


__all__ = [
    "get_state",
    "get_all_states",
    "ensure_state",
    "set_cursor",
    "set_backfill",
    "mark_attempt",
    "mark_success",
    "mark_failure",
    "reset_circuit",
    "is_open",
    "list_due",
    "freshness",
]
