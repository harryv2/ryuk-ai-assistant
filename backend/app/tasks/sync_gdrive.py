"""Drive sync: `changes.list` from the stored `pageToken`, or a bounded walk.

Drive differs from the other two in one expensive way: the searchable content
is not in the change feed. Every changed file that holds text needs an export
or a download before it can be chunked. So this module is deliberately the most
tight-fisted of the three:

* files whose mime type carries no text are mirrored on metadata alone — the
  name and the folder path still answer "where is that spreadsheet";
* the extract is capped, because the tail of a 200-page PDF matches nothing a
  chat ever asks for;
* the backfill bound is 200 files, not 500.

A removal in the change feed, or a file that has moved to the bin, drops every
chunk of it from the mirror.
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

SERVICE = "gdrive"
TABLE = "gdrive"

PAGE = min(int(settings.SYNC_PAGE_SIZE), 100)
MAX_PAGES = 10

#: The bounds on every non-incremental walk.
BACKFILL_DAYS = 90
BACKFILL_FILES = 200

#: How much text is worth pulling out of one file.
MAX_EXTRACT_CHARS = 200_000

#: Mime types with text worth extracting. Everything else is mirrored on its
#: metadata alone.
TEXTUAL_MIME_PREFIXES = (
    "text/",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.spreadsheet",
    "application/pdf",
    "application/rtf",
    "application/json",
    "application/xml",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.oasis.opendocument",
    "application/msword",
)

ROW_FIELDS = (
    "file_id",
    "name",
    "mime_type",
    "owner_email",
    "is_shared",
    "web_view_link",
    "folder_path",
    "size_bytes",
    "modified_at",
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
    name="sync.gdrive.dispatch_all_users",
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


def _has_text(mime_type: str | None) -> bool:
    """Whether pulling this file's contents is worth a quota unit."""
    if not mime_type:
        return False
    return any(mime_type.startswith(prefix) for prefix in TEXTUAL_MIME_PREFIXES)


def _content_hash(base: dict[str, Any], index: int, text: str) -> Any:
    return fingerprint_parts(
        "sync.gdrive",
        base["file_id"],
        str(index),
        base.get("name") or "",
        text,
    )


async def _file_rows(
    user_id: str, clients: Any, raw_files: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Normalise, extract, chunk. Returns ``(changed, unchanged, chunk_counts)``."""
    from app.search.chunking import chunk
    from app.services import gdrive as gdrive_api

    if not raw_files:
        return [], [], {}

    refs = [raw.get("id") for raw in raw_files if raw.get("id")]
    async with session_scope() as session:
        known = await mirror.existing_hashes(session, user_id, TABLE, refs)

    changed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for raw in raw_files:
        parsed = gdrive_api.normalise_file(raw)
        base = {field: parsed.get(field) for field in ROW_FIELDS}
        base["file_id"] = base["file_id"] or raw.get("id")
        if not base["file_id"]:
            continue
        base["name"] = base.get("name") or "(untitled)"
        base["is_shared"] = bool(base.get("is_shared"))

        text = ""
        if _has_text(base.get("mime_type")):
            await acquire_google(user_id, "gdrive.files.export", share="background")
            try:
                text = (await gdrive_api.extract_text(clients, raw)) or ""
            except Exception as exc:  # noqa: BLE001 - metadata is still worth having
                if classify_error(exc) in {"NOT_FOUND", "INVALID", "PRECONDITION"}:
                    log.info(
                        "sync.extract_skipped",
                        user_id=user_id,
                        file_id=base["file_id"],
                        error=str(exc)[:200],
                    )
                    text = ""
                else:
                    raise
        text = text[:MAX_EXTRACT_CHARS]

        pieces = chunk(text) if text else [""]
        pieces = pieces or [""]
        counts[base["file_id"]] = len(pieces)
        for index, piece in enumerate(pieces):
            row = {
                **base,
                "chunk_index": index,
                "content_excerpt": piece,
                "content_hash": _content_hash(base, index, piece),
            }
            if known.get((base["file_id"], index)) == row["content_hash"]:
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
    """One transaction. The cursor moves only after this has committed."""
    async with session_scope() as session:
        changed_ids = await mirror.upsert(session, user_id, TABLE, changed)
        if unchanged:
            await mirror.upsert(session, user_id, TABLE, unchanged)
        for ref, keep in counts.items():
            await mirror.delete_extra_chunks(session, user_id, TABLE, ref, keep)
        if deleted:
            await mirror.delete_by_refs(session, user_id, TABLE, deleted)
    return changed_ids


def _split_changes(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """A change page, split into files to index and files to drop."""
    files: list[dict[str, Any]] = []
    deleted: list[str] = []
    for change in items:
        file_id = change.get("fileId") or (change.get("file") or {}).get("id")
        if not file_id:
            continue
        payload = change.get("file") or {}
        if change.get("removed") or payload.get("trashed") or payload.get("explicitlyTrashed"):
            deleted.append(file_id)
            continue
        payload.setdefault("id", file_id)
        files.append(payload)
    return files, deleted


# --------------------------------------------------------------------------- #
# Incremental sync
# --------------------------------------------------------------------------- #


async def _walk_files(
    user_id: str,
    clients: Any,
    *,
    query: str,
    limit: int,
    page_token: str | None = None,
) -> tuple[int, str | None]:
    """Walk `files.list` for a query, bounded by ``limit`` files."""
    from app.services import gdrive as gdrive_api

    indexed = 0
    token = page_token
    while indexed < limit:
        await acquire_google(user_id, "gdrive.files.list", share="background")
        response = await gdrive_api.files_list(
            clients,
            query=query,
            page_token=token,
            page_size=min(PAGE, limit - indexed),
        )
        files = [f for f in (response.get("files") or []) if f.get("id")]
        if files:
            changed, unchanged, counts = await _file_rows(user_id, clients, files)
            row_ids = await _commit(user_id, changed, unchanged, counts, [])
            fan_to_embed(user_id, TABLE, row_ids)
            indexed += len(files)
        token = response.get("nextPageToken")
        if not token:
            return indexed, None
    return indexed, token


async def _full_resync(user_id: str, clients: Any, *, reason: str) -> dict[str, Any]:
    """A bounded re-walk when the page token is no longer usable.

    The fresh start page token is taken **first**, so a file changed during the
    walk is picked up next cycle rather than lost.
    """
    from app.services import gdrive as gdrive_api

    await acquire_google(user_id, "gdrive.about.get", share="background")
    start_token = await gdrive_api.start_page_token(clients)

    indexed, _ = await _walk_files(
        user_id,
        clients,
        query="trashed = false",
        limit=BACKFILL_FILES,
    )

    async with session_scope() as session:
        await sync_state_repo.mark_success(
            session,
            user_id,
            SERVICE,
            items_indexed=indexed,
            cursor={cursors.PAGE_TOKEN: str(start_token)} if start_token else None,
        )
    log.info(
        "sync.full_resync", service=SERVICE, user_id=user_id,
        reason=reason, indexed=indexed,
    )
    return {"service": SERVICE, "mode": "full", "reason": reason, "indexed": indexed}


async def _incremental(
    user_id: str, clients: Any, cursor: dict[str, Any]
) -> dict[str, Any]:
    from app.services import gdrive as gdrive_api

    token = str(cursors.get(cursor, cursors.PAGE_TOKEN))
    indexed = 0
    removed = 0
    pages = 0
    next_token: str | None = token

    while pages < MAX_PAGES:
        await acquire_google(user_id, "gdrive.changes.list", share="background")
        try:
            response = await gdrive_api.changes_list(
                clients, page_token=next_token, page_size=PAGE
            )
        except Exception as exc:  # noqa: BLE001 - 410 is a resync, not a failure
            if http_status(exc) == 410:
                return await _full_resync(user_id, clients, reason="page_token_gone")
            raise

        files, deleted = _split_changes(list(response.get("changes") or []))
        changed, unchanged, counts = await _file_rows(user_id, clients, files)
        row_ids = await _commit(user_id, changed, unchanged, counts, deleted)
        # Committed. Only now may the cursor move.
        fan_to_embed(user_id, TABLE, row_ids)
        indexed += len(files)
        removed += len(deleted)
        pages += 1

        page_token = response.get("nextPageToken")
        new_start = response.get("newStartPageToken")
        cursor_token = page_token or new_start
        async with session_scope() as session:
            await sync_state_repo.mark_success(
                session,
                user_id,
                SERVICE,
                items_indexed=len(files),
                cursor={cursors.PAGE_TOKEN: str(cursor_token)} if cursor_token else None,
            )
        next_token = page_token
        if not page_token:
            break

    log.info(
        "sync.incremental",
        service=SERVICE,
        user_id=user_id,
        indexed=indexed,
        deleted=removed,
        pages=pages,
        more=bool(next_token),
    )
    if next_token:
        incremental.apply_async(args=[user_id], countdown=5, queue="sync")
    return {
        "service": SERVICE,
        "mode": "incremental",
        "indexed": indexed,
        "deleted": removed,
        "pages": pages,
        "more": bool(next_token),
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
        if mode == "full" or not cursors.get(cursor, cursors.PAGE_TOKEN):
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
        backfill.apply_async(args=[user_id], countdown=30, queue="sync")
    return outcome


async def sync_incremental(user_id: str, mode: str = "incremental") -> dict[str, Any]:
    """One sync pass for one user, in the caller's event loop.

    The public async half of the ``sync.gdrive`` task. The Celery task is a sync
    wrapper around this; anything already inside a loop calls this directly
    rather than nesting an ``asyncio.run`` inside a running one.
    """
    return await _run(user_id, mode)


@celery_app.task(base=AppTask, bind=True, name="sync.gdrive", queue="sync")
def incremental(
    self: AppTask, user_id: str, mode: str = "incremental"
) -> dict[str, Any]:
    """One user's Drive pass, advancing from the stored change ``pageToken``."""
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

    # Drive's query language wants an RFC 3339 literal, not a bind parameter.
    cutoff = (utcnow() - dt.timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        clients = await _clients(user_id)
        indexed, token = await _walk_files(
            user_id,
            clients,
            query=f"trashed = false and modifiedTime > '{cutoff}'",
            limit=BACKFILL_FILES,
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


@celery_app.task(base=AppTask, bind=True, name="sync.gdrive.backfill", queue="sync")
def backfill(self: AppTask, user_id: str) -> dict[str, Any]:
    """One bounded page of the walk: 90 days of changes, at most 200 files.

    Chains itself on the page token until the window is exhausted.
    """
    return run_async(_backfill(user_id))


__all__ = [
    "sync_incremental",
    "SERVICE",
    "TABLE",
    "PAGE",
    "BACKFILL_DAYS",
    "BACKFILL_FILES",
    "MAX_EXTRACT_CHARS",
    "dispatch_all_users",
    "incremental",
    "backfill",
]
