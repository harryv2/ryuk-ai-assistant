"""Gmail sync: `history.list` from the stored cursor, or a bounded backfill.

The shape every sync task in this package shares:

1. read ``sync_state.cursor``;
2. return early if the circuit breaker is open — no task, no quota unit, no
   worker slot spent on a service that is already failing;
3. acquire Google quota on the background share, so a backfill can never starve
   someone typing;
4. page the change feed;
5. upsert on the natural key, computing ``content_hash`` per chunk;
6. hand the changed row ids to ``embed.embed_batch`` in 128s;
7. **advance the cursor only after the upsert transaction has committed**, so a
   crash reprocesses a page rather than skipping it.

A ``410 Gone`` means the mailbox's history no longer reaches back to our cursor.
That is not an error to retry: it is a bounded full resync, then a fresh
``historyId``.
"""

from __future__ import annotations

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

SERVICE = "gmail"
TABLE = "gmail"

#: Messages per list page. Capped so one task cannot approach the broker's
#: 900 second visibility timeout.
PAGE = min(int(settings.SYNC_PAGE_SIZE), 100)

#: History pages per incremental run. Anything left over re-enqueues itself
#: rather than holding a worker.
MAX_PAGES = 10

#: The bounds on every non-incremental walk.
BACKFILL_DAYS = 90
BACKFILL_MESSAGES = 500

#: Columns of ``sync_gmail`` this task fills. Anything else a normaliser hands
#: back is ignored rather than passed into the insert.
ROW_FIELDS = (
    "message_id",
    "thread_id",
    "subject",
    "from_email",
    "from_name",
    "to_emails",
    "labels",
    "has_attachments",
    "received_at",
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
    name="sync.gmail.dispatch_all_users",
    queue="sync",
    user_arg=None,
    max_retries=2,
)
def dispatch_all_users(self: AppTask) -> dict[str, Any]:
    """Enqueue every connected user, smeared across the 15-minute window.

    ``countdown = blake2b(user_id) % 900`` puts each user in the same second of
    every cycle. Firing all of them at :00 would ask Google for the whole
    cycle's quota in five seconds and get 403s for most of it.
    """
    return run_async(_dispatch_all_users())


# --------------------------------------------------------------------------- #
# Incremental sync
# --------------------------------------------------------------------------- #


async def _clients(user_id: str) -> Any:
    """Authorised Google clients for this user."""
    from app.google.client import clients_for

    async with session_scope() as session:
        return await clients_for(session, user_id)


async def _note_failure(user_id: str, exc: BaseException) -> str:
    """Record the failure against the service and return its class."""
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


def _history_refs(response: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The message ids a history page touched, split into changed and deleted.

    Label changes count as changed: the body is the same but the labels column
    is not, and a `starred` that never lands in the mirror is a search that
    quietly misses.
    """
    changed: dict[str, None] = {}
    deleted: dict[str, None] = {}
    for record in response.get("history") or []:
        for added in record.get("messagesAdded") or []:
            ref = (added.get("message") or {}).get("id")
            if ref:
                changed[ref] = None
        for label_change in (record.get("labelsAdded") or []) + (
            record.get("labelsRemoved") or []
        ):
            ref = (label_change.get("message") or {}).get("id")
            if ref:
                changed[ref] = None
        for removed in record.get("messagesDeleted") or []:
            ref = (removed.get("message") or {}).get("id")
            if ref:
                deleted[ref] = None
    for ref in deleted:
        changed.pop(ref, None)
    return list(changed), list(deleted)


def _content_hash(base: dict[str, Any], index: int, text: str) -> Any:
    """uuid5 over exactly what will be embedded, plus what identifies it.

    Labels are deliberately outside it: marking a message read must not cost an
    embedding.
    """
    return fingerprint_parts(
        "sync.gmail",
        base["message_id"],
        str(index),
        base.get("subject") or "",
        base.get("from_email") or "",
        text,
    )


async def _message_rows(
    user_id: str, clients: Any, message_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Fetch, normalise and chunk messages into mirror rows.

    Returns ``(changed, unchanged, chunk_counts)``. Unchanged rows are still
    upserted — labels move without the body moving — but only changed rows are
    worth an embedding.
    """
    from app.search.chunking import chunk
    from app.services import gmail as gmail_api

    if not message_ids:
        return [], [], {}

    async with session_scope() as session:
        known = await mirror.existing_hashes(session, user_id, TABLE, message_ids)

    changed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for ref in message_ids:
        await acquire_google(user_id, "gmail.messages.get", share="background")
        try:
            raw = await gmail_api.messages_get(clients, ref)
        except Exception as exc:  # noqa: BLE001 - one gone message is not a failure
            if classify_error(exc) == "NOT_FOUND" or http_status(exc) == 404:
                log.info("sync.message_gone", user_id=user_id, message_id=ref)
                continue
            raise
        if not raw:
            continue

        parsed = gmail_api.normalise_message(raw)
        base = {field: parsed.get(field) for field in ROW_FIELDS}
        base["message_id"] = base["message_id"] or ref
        base["to_emails"] = list(base.get("to_emails") or [])
        base["labels"] = list(base.get("labels") or [])
        base["has_attachments"] = bool(base.get("has_attachments"))
        base["received_at"] = base.get("received_at") or utcnow()

        pieces = chunk(parsed.get("body_clean") or "") or [""]
        counts[base["message_id"]] = len(pieces)
        for index, piece in enumerate(pieces):
            row = {
                **base,
                "chunk_index": index,
                "body_clean": piece,
                "content_hash": _content_hash(base, index, piece),
            }
            if known.get((base["message_id"], index)) == row["content_hash"]:
                unchanged.append(row)
            else:
                changed.append(row)

    return changed, unchanged, counts


async def _commit(
    user_id: str,
    changed: list[dict[str, Any]],
    unchanged: list[dict[str, Any]],
    counts: dict[str, int],
    deleted: list[str],
) -> list[str]:
    """One transaction: upsert, trim stale chunks, drop deleted messages.

    Returns the ids of the rows whose content changed, which are the only ones
    the embed queue needs to hear about. The caller advances the cursor after
    this returns — that is the whole crash-safety story.
    """
    async with session_scope() as session:
        changed_ids = await mirror.upsert(session, user_id, TABLE, changed)
        if unchanged:
            await mirror.upsert(session, user_id, TABLE, unchanged)
        for ref, keep in counts.items():
            await mirror.delete_extra_chunks(session, user_id, TABLE, ref, keep)
        if deleted:
            await mirror.delete_by_refs(session, user_id, TABLE, deleted)
    return changed_ids


async def _walk_query(
    user_id: str,
    clients: Any,
    *,
    query: str,
    limit: int,
    page_token: str | None = None,
) -> tuple[int, str | None]:
    """Walk `messages.list` for a query, bounded by ``limit`` messages.

    Returns ``(indexed, next_page_token)``. A token coming back means the walk
    was cut short by the bound and the caller should chain another task.
    """
    from app.services import gmail as gmail_api

    indexed = 0
    token = page_token
    while indexed < limit:
        await acquire_google(user_id, "gmail.messages.list", share="background")
        response = await gmail_api.messages_list(
            clients,
            query=query,
            page_token=token,
            max_results=min(PAGE, limit - indexed),
        )
        refs = [m.get("id") for m in (response.get("messages") or []) if m.get("id")]
        if refs:
            changed, unchanged, counts = await _message_rows(user_id, clients, refs)
            row_ids = await _commit(user_id, changed, unchanged, counts, [])
            fan_to_embed(user_id, TABLE, row_ids)
            indexed += len(refs)
        token = response.get("nextPageToken")
        if not token:
            return indexed, None
    return indexed, token


async def _full_resync(user_id: str, clients: Any, *, reason: str) -> dict[str, Any]:
    """A bounded re-walk when the cursor is no longer usable.

    The mailbox's current ``historyId`` is taken **first**, so anything that
    arrives during the walk is picked up by the next incremental pass rather
    than falling in the gap.
    """
    from app.services import gmail as gmail_api

    await acquire_google(user_id, "gmail.users.getProfile", share="background")
    profile = await gmail_api.get_profile(clients)
    history_id = str((profile or {}).get("historyId") or "")

    indexed, _ = await _walk_query(
        user_id,
        clients,
        query=f"newer_than:{BACKFILL_DAYS}d",
        limit=BACKFILL_MESSAGES,
    )

    async with session_scope() as session:
        await sync_state_repo.mark_success(
            session,
            user_id,
            SERVICE,
            items_indexed=indexed,
            cursor=cursors.with_value(cursors.HISTORY_ID, history_id) or None,
        )
    log.info(
        "sync.full_resync", service=SERVICE, user_id=user_id,
        reason=reason, indexed=indexed,
    )
    return {"service": SERVICE, "mode": "full", "reason": reason, "indexed": indexed}


async def _incremental(
    user_id: str, clients: Any, cursor: dict[str, Any]
) -> dict[str, Any]:
    from app.services import gmail as gmail_api

    start = str(cursors.get(cursor, cursors.HISTORY_ID))
    token = cursors.get(cursor, cursors.PAGE_TOKEN)
    latest = start
    indexed = 0
    removed = 0
    pages = 0

    while pages < MAX_PAGES:
        await acquire_google(user_id, "gmail.history.list", share="background")
        try:
            response = await gmail_api.history_list(
                clients,
                start_history_id=start,
                page_token=token,
                max_results=PAGE,
            )
        except Exception as exc:  # noqa: BLE001 - 410 is a resync, not a failure
            if http_status(exc) == 410:
                return await _full_resync(user_id, clients, reason="history_gone")
            raise

        changed_refs, deleted_refs = _history_refs(response)
        changed, unchanged, counts = await _message_rows(user_id, clients, changed_refs)
        row_ids = await _commit(user_id, changed, unchanged, counts, deleted_refs)
        # The upsert has committed. Only now is it safe to move the cursor.
        fan_to_embed(user_id, TABLE, row_ids)
        indexed += len(changed_refs)
        removed += len(deleted_refs)
        latest = str(response.get("historyId") or latest)
        token = response.get("nextPageToken")
        pages += 1
        if not token:
            break

    # A page token still in hand means the bound cut the walk short, so the
    # cursor keeps it: the next run resumes from that page instead of replaying
    # the whole history range.
    next_cursor = (
        {cursors.HISTORY_ID: start, cursors.PAGE_TOKEN: token}
        if token
        else {cursors.HISTORY_ID: latest}
    )
    async with session_scope() as session:
        await sync_state_repo.mark_success(
            session, user_id, SERVICE, items_indexed=indexed, cursor=next_cursor
        )
    if token:
        incremental.apply_async(args=[user_id], countdown=5, queue="sync")

    log.info(
        "sync.incremental",
        service=SERVICE,
        user_id=user_id,
        indexed=indexed,
        deleted=removed,
        pages=pages,
        more=bool(token),
    )
    return {
        "service": SERVICE,
        "mode": "incremental",
        "indexed": indexed,
        "deleted": removed,
        "pages": pages,
        "more": bool(token),
    }


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
        if mode == "full" or not cursors.get(cursor, cursors.HISTORY_ID):
            outcome = await _full_resync(
                user_id, clients, reason="no_cursor" if mode != "full" else "requested"
            )
        else:
            outcome = await _incremental(user_id, clients, cursor)
    except Exception as exc:
        error_class = await _note_failure(user_id, exc)
        if is_retryable(error_class):
            raise
        raise _terminal(user_id, exc, error_class) from exc

    if not backfill_complete:
        backfill.apply_async(
            args=[user_id], countdown=30, queue="sync"
        )
    return outcome


async def sync_incremental(user_id: str, mode: str = "incremental") -> dict[str, Any]:
    """One sync pass for one user, in the caller's event loop.

    The public async half of the ``sync.gmail`` task. The Celery task is a sync
    wrapper around this; anything already inside a loop calls this directly
    rather than nesting an ``asyncio.run`` inside a running one.
    """
    return await _run(user_id, mode)


@celery_app.task(base=AppTask, bind=True, name="sync.gmail", queue="sync")
def incremental(
    self: AppTask, user_id: str, mode: str = "incremental"
) -> dict[str, Any]:
    """One user's Gmail pass.

    ``mode`` is ``incremental`` (advance from the cursor) or ``full`` (discard
    it and re-walk, bounded). A missing cursor is a first sync and takes the
    full path on its own.
    """
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

    try:
        clients = await _clients(user_id)
        indexed, token = await _walk_query(
            user_id,
            clients,
            query=f"newer_than:{BACKFILL_DAYS}d",
            limit=BACKFILL_MESSAGES,
            page_token=cursors.get(cursor, cursors.PAGE_TOKEN),
        )
    except Exception as exc:
        error_class = await _note_failure(user_id, exc)
        if is_retryable(error_class):
            raise
        raise _terminal(user_id, exc, error_class) from exc

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
        # Self-chaining keeps each task short and each page independently
        # retryable, which is what makes a 15,000 message first sync survivable.
        backfill.apply_async(args=[user_id], countdown=5, queue="sync")

    log.info(
        "sync.backfill",
        service=SERVICE,
        user_id=user_id,
        indexed=indexed,
        complete=not token,
    )
    return {
        "service": SERVICE,
        "mode": "backfill",
        "indexed": indexed,
        "complete": not token,
    }


@celery_app.task(base=AppTask, bind=True, name="sync.gmail.backfill", queue="sync")
def backfill(self: AppTask, user_id: str) -> dict[str, Any]:
    """One bounded page of history: 90 days, at most 500 messages.

    Chains itself on the page token until the window is exhausted, then marks
    ``backfill_complete``.
    """
    return run_async(_backfill(user_id))


__all__ = [
    "sync_incremental",
    "SERVICE",
    "TABLE",
    "PAGE",
    "MAX_PAGES",
    "BACKFILL_DAYS",
    "BACKFILL_MESSAGES",
    "dispatch_all_users",
    "incremental",
    "backfill",
]
