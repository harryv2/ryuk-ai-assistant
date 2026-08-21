"""Background jobs.

Five queues, one Celery application, and the work that keeps the mirror fresh:

``celery_app``   the application, its policies and the beat schedule
``sync_gmail`` ``sync_gcal`` ``sync_gdrive`` (modules) — incremental sync and
  backfill, writing into ``sync_messages`` / ``sync_events`` / ``sync_files``
``embed``        vectors for anything whose content changed
``actions``      the one place an approved write reaches Google
``maintenance``  tokens, prompt expiry, the DLQ sweeper, pruning, gauges

**No task is defined in this module.** ``celery_app.py`` reads ``TASK_MODULES``
from here, so anything this module imported from the task modules would be a
cycle. What lives here is the handful of helpers all of them share.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")

#: Every module holding tasks. ``celery_app`` imports these at worker start,
#: and beat needs them registered to schedule them.
TASK_MODULES: tuple[str, ...] = (
    "app.tasks.sync_gmail",
    "app.tasks.sync_gcal",
    "app.tasks.sync_gdrive",
    "app.tasks.embed",
    "app.tasks.actions",
    "app.tasks.maintenance",
)

#: The beat interval, in seconds. A user's work is smeared across it.
SMEAR_WINDOW_S = 900

#: Users read per page by ``dispatch_all_users``.
USER_PAGE = 1000

#: Rows handed to one ``embed.embed_batch``.
EMBED_CHUNK = 128


def utcnow() -> dt.datetime:
    """Tz-aware now, in UTC. The only clock the task layer reads."""
    return dt.datetime.now(dt.UTC)


def smear_countdown(user_id: str, salt: str = "", window_s: int = SMEAR_WINDOW_S) -> int:
    """Which second of the window this user's job runs in.

    blake2b, never Python's ``hash()``: ``hash()`` on a str is salted per
    process, so a user would land in a different slot on every beat tick, which
    turns a 15-minute interval into a uniform draw over [0, 30] minutes. The
    whole point is that each user occupies the *same* slot every cycle.

    ``salt`` separates the three services so one user's Gmail, Calendar and
    Drive pulls do not all fire in the same second.
    """
    digest = hashlib.blake2b(f"{salt}\x1f{user_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % max(1, int(window_s))


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Walk a sequence in fixed-size pieces."""
    if size < 1:
        raise ValueError("size must be at least 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def http_status(exc: BaseException) -> int | None:
    """The HTTP status behind an exception, if it carries one.

    googleapiclient puts it on ``resp.status``, httpx on
    ``response.status_code``, and our own AppError on ``http``. A 410 on a sync
    cursor has to be recognised whichever library raised it.
    """
    for attr in ("status_code", "http", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "resp", None) or getattr(exc, "response", None)
    for attr in ("status", "status_code"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def classify_error(exc: BaseException) -> str:
    """The error class of a failure, as a plain string.

    ``app.google.retry`` owns the taxonomy. It is imported lazily and its
    absence falls back to ``UNKNOWN`` rather than raising, because this is
    called from failure handlers, and a failure handler that raises loses the
    original failure.
    """
    try:
        from app.google.retry import classify

        return str(classify(exc))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - never mask the failure being classified
        status = http_status(exc)
        if status is None:
            return "UNKNOWN"
        if status == 401:
            return "AUTH_EXPIRED"
        if status == 403:
            return "RATE_LIMITED"
        if status == 404:
            return "NOT_FOUND"
        if status in (409, 412):
            return "PRECONDITION"
        if status == 429:
            return "RATE_LIMITED"
        if status >= 500:
            return "TRANSIENT"
        return "INVALID"


#: Failure classes a retry can plausibly fix.
RETRYABLE_CLASSES = frozenset(
    {"TRANSIENT", "RATE_LIMITED", "QUOTA_EXHAUSTED", "AUTH_EXPIRED", "UNKNOWN"}
)


def is_retryable(error_class: str) -> bool:
    """True when re-running the job could give a different answer."""
    try:
        from app.google.retry import ErrorClass, retryable

        return bool(retryable(ErrorClass(error_class)))
    except Exception:  # noqa: BLE001 - the table above is the fallback
        return error_class in RETRYABLE_CLASSES


def error_payload(exc: BaseException) -> dict[str, Any]:
    """The JSON blob written to ``sync_state.last_error`` and to the DLQ."""
    payload: dict[str, Any] = {
        "class": classify_error(exc),
        "type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    status = http_status(exc)
    if status is not None:
        payload["status"] = status
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        payload["code"] = code
    return payload


async def open_circuit_user_ids(
    session: AsyncSession, service: str, *, now: dt.datetime | None = None
) -> set[str]:
    """Users whose breaker is holding this service back right now.

    One query for the whole fleet, so the beat fan-out can skip them without a
    round trip per user. ``sync_state`` has no repository call for the inverse
    of ``list_due``, and the dispatcher is the only caller that wants it.
    """
    from app.db.models import SyncService, SyncState

    at = now or utcnow()
    result = await session.execute(
        select(SyncState.user_id).where(
            SyncState.service == SyncService(service),
            SyncState.circuit_open_until.is_not(None),
            SyncState.circuit_open_until > at,
        )
    )
    return set(result.scalars().all())


__all__ = [
    "TASK_MODULES",
    "SMEAR_WINDOW_S",
    "USER_PAGE",
    "EMBED_CHUNK",
    "RETRYABLE_CLASSES",
    "utcnow",
    "smear_countdown",
    "chunked",
    "http_status",
    "classify_error",
    "is_retryable",
    "error_payload",
    "open_circuit_user_ids",
]
