"""Calendar sync: `events.list` with a `syncToken`, or a bounded window walk.

Same shape as Gmail — circuit breaker, background quota share, upsert then
advance the cursor — with three differences that are Calendar's own:

* one row per event, never chunked, so ``chunk_index`` never enters the key;
* ``etag`` is stored, because every later update or delete sends it as
  ``If-Match`` and that is what stops us overwriting somebody else's edit;
* a cancellation arrives through the sync token as a stub — an id and
  ``status: cancelled`` and nothing else. Upserting that would blank the title
  and violate ``starts_at NOT NULL``, so a stub is a delete from the mirror.
  A full read that returns a complete cancelled event still mirrors it, which
  is what the ``not_cancelled`` search filter is for.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.config import settings
from app.tasks import cursors
from app.core.ids import fingerprint_parts
from app.core.logging import get_logger
from app.core.ratelimit import acquire_google
from app.db.repositories import mirror
from app.db.repositories import sync_state as sync_state_repo
from app.db.repositories import users as users_repo
from app.db.session import session_scope
from app.tasks import (
    SMEAR_WINDOW_S,
    USER_PAGE,
    chunked,
    classify_error,
    error_payload,
    http_status,
    is_retryable,
    open_circuit_user_ids,
    smear_countdown,
    utcnow,
)
from app.tasks.celery_app import AppTask, NonRetryable, celery_app, run_async
from app.tasks.embed import fan_to_embed

log = get_logger(__name__)

SERVICE = "gcal"
TABLE = "gcal"
CALENDAR_ID = "primary"

PAGE = min(int(settings.SYNC_PAGE_SIZE), 250)
MAX_PAGES = 10

#: The bounds on every non-incremental walk: 90 days either side of today.
BACKFILL_DAYS = 90
BACKFILL_EVENTS = 500

ROW_FIELDS = (
    "event_id",
    "calendar_id",
    "recurring_event_id",
    "title",
    "description",
    "location",
    "organizer_email",
    "attendees",
    "starts_at",
    "ends_at",
    "all_day",
    "event_timezone",
    "status",
    "etag",
)


# --------------------------------------------------------------------------- #
# Beat fan-out
# --------------------------------------------------------------------------- #


async def _dispatch_all_users() -> dict[str, Any]:
    async with session_scope() as session:
        user_ids = await users_repo.list_connected_user_ids(session, None)
        blocked = await open_circuit_user_ids(session, SERVICE)

    enqueued = 0
    for page in chunked(user_ids, USER_PAGE):
        for user_id in page:
            if user_id in blocked:
                continue
            incremental.apply_async(
                args=[user_id],
                countdown=smear_countdown(user_id, SERVICE),
                queue="sync",
                expires=SMEAR_WINDOW_S + 300,
            )
            enqueued += 1

    log.info(
        "sync.dispatch",
        service=SERVICE,
        users=len(user_ids),
        enqueued=enqueued,
        circuit_open=len(blocked),
    )
    return {"service": SERVICE, "users": len(user_ids), "enqueued": enqueued,
            "circuit_open": len(blocked)}


@celery_app.task(
    base=AppTask,
    bind=True,
    name="sync.gcal.dispatch_all_users",
    queue="sync",
    user_arg=None,
    max_retries=2,
)
def dispatch_all_users(self: AppTask) -> dict[str, Any]:
    """Enqueue every connected user, smeared across the 15-minute window."""
    return run_async(_dispatch_all_users())


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #


async def _clients(user_id: str) -> Any:
    from app.google.client import clients_for

    async with session_scope() as session:
        return await clients_for(session, user_id)


async def _note_failure(user_id: str, exc: BaseException) -> str:
    error_class = classify_error(exc)
    async with session_scope() as session:
        await sync_state_repo.mark_failure(
            session, user_id, SERVICE, error_payload(exc)
        )
    log.warning(
        "sync.failed",
        service=SERVICE,
        user_id=user_id,
        error_class=error_class,
        error=str(exc)[:300],
    )
    return error_class


def _terminal(user_id: str, exc: BaseException, error_class: str) -> NonRetryable:
    return NonRetryable(
        f"{SERVICE} sync stopped for {user_id}: {exc}",
        error_class=error_class,
        cause=exc,
        details={"service": SERVICE, "user_id": user_id},
    )


def _content_hash(row: dict[str, Any]) -> Any:
    """uuid5 over what a search reads. ``etag`` is not in it — Google bumps the
    etag for changes that do not touch a single searchable field."""
    starts_at = row.get("starts_at")
    return fingerprint_parts(
        "sync.gcal",
        row["event_id"],
        row.get("title") or "",
        row.get("location") or "",
        row.get("description") or "",
        starts_at.isoformat() if isinstance(starts_at, dt.datetime) else "",
        sorted(
            (a.get("email") or "").lower()
            for a in (row.get("attendees") or [])
            if isinstance(a, dict)
        ),
    )


async def _event_rows(
    user_id: str, raw_events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Normalise a page of events into ``(changed, unchanged, deleted_refs)``."""
    from app.services import gcal as gcal_api

    if not raw_events:
        return [], [], []

    prepared: list[dict[str, Any]] = []
    deleted: list[str] = []
    for raw in raw_events:
        parsed = gcal_api.normalise_event(raw, calendar_id=CALENDAR_ID)
        row = {field: parsed.get(field) for field in ROW_FIELDS}
        row["event_id"] = row["event_id"] or raw.get("id")
        row["calendar_id"] = row["calendar_id"] or CALENDAR_ID
        if not row["event_id"]:
            continue
        if row.get("starts_at") is None:
            # A cancellation stub: an id, a status, and nothing to store.
            deleted.append(row["event_id"])
            continue
        row["all_day"] = bool(row.get("all_day"))
        row["attendees"] = list(row.get("attendees") or [])
        row["content_hash"] = _content_hash(row)
        prepared.append(row)

    refs = [row["event_id"] for row in prepared]
    async with session_scope() as session:
        known = await mirror.existing_hashes(session, user_id, TABLE, refs)

    changed = [r for r in prepared if known.get((r["event_id"], 0)) != r["content_hash"]]
    unchanged = [r for r in prepared if known.get((r["event_id"], 0)) == r["content_hash"]]
    return changed, unchanged, deleted


async def _commit(
    user_id: str,
    changed: list[dict[str, Any]],
    unchanged: list[dict[str, Any]],
    deleted: list[str],
) -> list[str]:
    """One transaction. The cursor moves only after this has committed."""
    async with session_scope() as session:
        changed_ids = await mirror.upsert(session, user_id, TABLE, changed)
        if unchanged:
            await mirror.upsert(session, user_id, TABLE, unchanged)
        if deleted:
            await mirror.delete_by_refs(session, user_id, TABLE, deleted)
    return changed_ids


async def _walk(
    user_id: str,
    clients: Any,
    *,
    sync_token: str | None = None,
    page_token: str | None = None,
    time_min: dt.datetime | None = None,
    time_max: dt.datetime | None = None,
    limit: int,
    max_pages: int = MAX_PAGES,
) -> dict[str, Any]:
    """Page `events.list`, upserting each page before asking for the next.

    Returns ``{indexed, deleted, page_token, sync_token, pages}``. A
    ``page_token`` coming back means a bound stopped the walk; a ``sync_token``
    means Google gave us the cursor for next time.
    """
    from app.services import gcal as gcal_api

    indexed = 0
    removed = 0
    pages = 0
    token = page_token
    next_sync: str | None = None

    while pages < max_pages and indexed < limit:
        await acquire_google(user_id, "gcal.events.list", share="background")
        response = await gcal_api.events_list(
            clients,
            calendar_id=CALENDAR_ID,
            sync_token=sync_token if token is None else None,
            page_token=token,
            time_min=time_min,
            time_max=time_max,
            max_results=min(PAGE, limit - indexed),
            show_deleted=True,
        )
        items = list(response.get("items") or [])
        changed, unchanged, deleted = await _event_rows(user_id, items)
        row_ids = await _commit(user_id, changed, unchanged, deleted)
        fan_to_embed(user_id, TABLE, row_ids)

        indexed += len(changed) + len(unchanged)
        removed += len(deleted)
        pages += 1
        token = response.get("nextPageToken")
        next_sync = response.get("nextSyncToken") or next_sync
        if not token:
            break

    return {
        "indexed": indexed,
        "deleted": removed,
        "pages": pages,
        "page_token": token,
        "sync_token": next_sync,
    }


# --------------------------------------------------------------------------- #
# Incremental sync
# --------------------------------------------------------------------------- #


async def _full_resync(
    user_id: str, clients: Any, *, reason: str, page_token: str | None = None
) -> dict[str, Any]:
    """A bounded window walk when there is no usable sync token.

    90 days either side of today: far enough back for "what did we agree in
    that meeting", far enough forward for anything anyone has scheduled. A page
    token left over from a walk the bound cut short is picked up rather than
    starting the window again.
    """
    now = utcnow()
    outcome = await _walk(
        user_id,
        clients,
        page_token=page_token,
        time_min=now - dt.timedelta(days=BACKFILL_DAYS),
        time_max=now + dt.timedelta(days=BACKFILL_DAYS),
        limit=BACKFILL_EVENTS,
    )
    cursor: dict[str, Any] = {}
    if outcome["sync_token"]:
        cursor[cursors.SYNC_TOKEN] = outcome["sync_token"]
    if outcome["page_token"]:
        cursor[cursors.PAGE_TOKEN] = outcome["page_token"]

    async with session_scope() as session:
        await sync_state_repo.mark_success(
            session,
            user_id,
            SERVICE,
            items_indexed=outcome["indexed"],
            cursor=cursor or None,
        )
    log.info(
        "sync.full_resync",
        service=SERVICE,
        user_id=user_id,
        reason=reason,
        indexed=outcome["indexed"],
    )
    return {"service": SERVICE, "mode": "full", "reason": reason, **outcome}


async def _incremental(
    user_id: str, clients: Any, cursor: dict[str, Any]
) -> dict[str, Any]:
    sync_token = cursors.get(cursor, cursors.SYNC_TOKEN)
    try:
        outcome = await _walk(
            user_id,
            clients,
            sync_token=sync_token,
            page_token=cursors.get(cursor, cursors.PAGE_TOKEN),
            limit=BACKFILL_EVENTS,
        )
    except Exception as exc:  # noqa: BLE001 - 410 is a resync, not a failure
        if http_status(exc) == 410:
            return await _full_resync(user_id, clients, reason="sync_token_gone")
        raise

    if outcome["page_token"]:
        next_cursor = {cursors.SYNC_TOKEN: sync_token, cursors.PAGE_TOKEN: outcome["page_token"]}
    else:
        next_cursor = {cursors.SYNC_TOKEN: outcome["sync_token"] or sync_token}

    async with session_scope() as session:
        await sync_state_repo.mark_success(
            session,
            user_id,
            SERVICE,
            items_indexed=outcome["indexed"],
            cursor=next_cursor,
        )
    if outcome["page_token"]:
        incremental.apply_async(args=[user_id], countdown=5, queue="sync")

    log.info(
        "sync.incremental",
        service=SERVICE,
        user_id=user_id,
        indexed=outcome["indexed"],
        deleted=outcome["deleted"],
        pages=outcome["pages"],
        more=bool(outcome["page_token"]),
    )
    return {"service": SERVICE, "mode": "incremental", **outcome}


async def _run(user_id: str, mode: str) -> dict[str, Any]:
    async with session_scope() as session:
        state = await sync_state_repo.ensure_state(session, user_id, SERVICE)
        cursor = dict(state.cursor or {})
        open_until = state.circuit_open_until
        backfill_complete = bool(state.backfill_complete)

    if open_until is not None and open_until > utcnow():
        log.info("sync.circuit_open", service=SERVICE, user_id=user_id,
                 until=open_until.isoformat())
        return {
            "service": SERVICE,
            "skipped": "circuit_open",
            "until": open_until.isoformat(),
        }

    async with session_scope() as session:
        await sync_state_repo.mark_attempt(session, user_id, SERVICE)

    try:
        clients = await _clients(user_id)
        if mode == "full" or not cursors.get(cursor, cursors.SYNC_TOKEN):
            outcome = await _full_resync(
                user_id,
                clients,
                reason="no_cursor" if mode != "full" else "requested",
                page_token=None if mode == "full" else cursors.get(cursor, cursors.PAGE_TOKEN),
            )
        else:
            outcome = await _incremental(user_id, clients, cursor)
    except Exception as exc:
        error_class = await _note_failure(user_id, exc)
        if is_retryable(error_class):
            raise
        raise _terminal(user_id, exc, error_class) from exc

    if not backfill_complete:
        backfill.apply_async(args=[user_id], countdown=30, queue="sync")
    return outcome


async def sync_incremental(user_id: str, mode: str = "incremental") -> dict[str, Any]:
    """One sync pass for one user, in the caller's event loop.

    The public async half of the ``sync.gcal`` task. The Celery task is a sync
    wrapper around this; anything already inside a loop calls this directly
    rather than nesting an ``asyncio.run`` inside a running one.
    """
    return await _run(user_id, mode)


@celery_app.task(base=AppTask, bind=True, name="sync.gcal", queue="sync")
def incremental(
    self: AppTask, user_id: str, mode: str = "incremental"
) -> dict[str, Any]:
    """One user's Calendar pass, advancing from the stored ``syncToken``."""
    return run_async(_run(user_id, mode))


# --------------------------------------------------------------------------- #
# Backfill
# --------------------------------------------------------------------------- #


async def _backfill(user_id: str) -> dict[str, Any]:
    async with session_scope() as session:
        state = await sync_state_repo.ensure_state(session, user_id, SERVICE)
        if state.backfill_complete:
            return {"service": SERVICE, "skipped": "backfill_complete"}
        if state.circuit_open_until is not None and state.circuit_open_until > utcnow():
            return {"service": SERVICE, "skipped": "circuit_open"}
        cursor = dict(state.backfill_cursor or {})

    now = utcnow()
    try:
        clients = await _clients(user_id)
        outcome = await _walk(
            user_id,
            clients,
            page_token=cursors.get(cursor, cursors.PAGE_TOKEN),
            time_min=now - dt.timedelta(days=BACKFILL_DAYS),
            time_max=now + dt.timedelta(days=BACKFILL_DAYS),
            limit=BACKFILL_EVENTS,
        )
    except Exception as exc:
        error_class = await _note_failure(user_id, exc)
        if is_retryable(error_class):
            raise
        raise _terminal(user_id, exc, error_class) from exc

    token = outcome["page_token"]
    async with session_scope() as session:
        if token:
            await sync_state_repo.set_backfill(
                session, user_id, SERVICE, backfill_cursor={cursors.PAGE_TOKEN: token}
            )
        else:
            await sync_state_repo.set_backfill(
                session, user_id, SERVICE, complete=True
            )
    if token:
        backfill.apply_async(args=[user_id], countdown=5, queue="sync")

    log.info(
        "sync.backfill",
        service=SERVICE,
        user_id=user_id,
        indexed=outcome["indexed"],
        complete=not token,
    )
    return {"service": SERVICE, "mode": "backfill", "complete": not token, **outcome}


@celery_app.task(base=AppTask, bind=True, name="sync.gcal.backfill", queue="sync")
def backfill(self: AppTask, user_id: str) -> dict[str, Any]:
    """One bounded page of the window: 90 days either side, 500 events.

    Chains itself on the page token until the window is exhausted.
    """
    return run_async(_backfill(user_id))


__all__ = [
    "sync_incremental",
    "SERVICE",
    "TABLE",
    "CALENDAR_ID",
    "PAGE",
    "BACKFILL_DAYS",
    "BACKFILL_EVENTS",
    "dispatch_all_users",
    "incremental",
    "backfill",
]
