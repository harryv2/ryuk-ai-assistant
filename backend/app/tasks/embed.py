"""The embed queue: vectors for mirror rows whose content changed.

Three things keep the bill down, in order of how much they save:

1. **The upsert.** ``mirror._upsert_set`` keeps the existing vector when the
   ``content_hash`` came back unchanged, so a re-synced row that did not change
   arrives here already embedded and is skipped outright.
2. **The Redis ``emb`` cache**, keyed by model and text. Two people forwarded
   the same newsletter; the second copy is free.
3. **Batching**, 128 chunks per API call.

What the text of a row is, whether it changed, and how a batch of texts becomes
a batch of vectors all live in :mod:`app.search.embedder` — this module used to
carry its own copy of that logic and the copy rotted the day the mirror tables
were renamed, failing on import while every test stayed green. Now it only
decides *which rows* to hand over.

The task is idempotent by construction: it re-reads the rows it was given, so a
redelivery embeds whatever the text is *now*, not whatever it was when the
message was queued.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.repositories import mirror
from app.db.session import session_scope
from app.search import embedder
from app.tasks import EMBED_CHUNK, chunked
from app.tasks.celery_app import AppTask, celery_app, run_async

log = get_logger(__name__)


async def _embed_batch(
    user_id: str, table: str, row_ids: Sequence[str], *, force: bool = False
) -> dict[str, Any]:
    spec = mirror.spec_for(table)  # refuses an unknown table
    ids = [r for r in dict.fromkeys(row_ids) if r]
    if not ids:
        return {"table": table, "requested": 0, "embedded": 0, "skipped": 0}

    async with session_scope() as session:
        result = await session.execute(
            select(spec.model).where(spec.model.user_id == user_id, spec.model.id.in_(ids))
        )
        rows = list(result.scalars().all())

        if force:
            # ``embed_rows`` skips a row whose hash is unchanged and whose
            # vector exists. Clearing the vector is how "re-embed anyway" is
            # said in that vocabulary.
            for row in rows:
                row.embedding = None

        report = await embedder.embed_rows(session, user_id, table, rows)

    log.info(
        "embed.batch",
        user_id=user_id,
        table=table,
        requested=len(ids),
        embedded=report.embedded,
        skipped=report.skipped_unchanged + report.skipped_empty,
    )
    return {
        "table": table,
        "requested": len(ids),
        "embedded": report.embedded,
        "skipped": report.skipped_unchanged + report.skipped_empty,
    }


@celery_app.task(
    base=AppTask,
    bind=True,
    name="embed.embed_batch",
    queue="embed",
    max_retries=5,
)
def embed_batch(
    self: AppTask,
    user_id: str,
    table: str,
    row_ids: list[str],
    force: bool = False,
) -> dict[str, Any]:
    """Vectorise these mirror rows. Rows already carrying a vector are skipped.

    ``table`` is ``gmail`` | ``gcal`` | ``gdrive``; ``row_ids`` are primary keys
    of that table, which is what ``mirror.upsert_*`` hands back.
    """
    return run_async(_embed_batch(user_id, table, row_ids, force=force))


def fan_to_embed(user_id: str, table: str, row_ids: Sequence[str]) -> int:
    """Queue these rows for embedding, 128 at a time.

    Called by the sync tasks after their upsert has committed. Returns how many
    batches were enqueued.
    """
    batches = 0
    for group in chunked(list(row_ids), EMBED_CHUNK):
        embed_batch.apply_async(args=[user_id, table, group], queue="embed")
        batches += 1
    return batches


async def _backfill_missing(
    user_id: str | None, table: str | None, limit: int
) -> dict[str, Any]:
    tables = [table] if table else list(mirror.CONNECTOR_SPECS)
    queued: dict[str, int] = {}
    async with session_scope() as session:
        for name in tables:
            rows = await mirror.rows_needing_embedding(
                session, user_id, name, limit=limit
            )
            by_user: dict[str, list[str]] = {}
            for row in rows:
                by_user.setdefault(row.user_id, []).append(row.id)
            count = 0
            for owner, ids in by_user.items():
                for group in chunked(ids, EMBED_CHUNK):
                    embed_batch.apply_async(
                        args=[owner, name, group], queue="embed"
                    )
                    count += len(group)
            queued[name] = count
    log.info("embed.backfill_missing", queued=queued, user_id=user_id)
    return {"queued": queued}


@celery_app.task(
    base=AppTask,
    bind=True,
    name="embed.backfill_missing",
    queue="embed",
    user_arg=None,
    max_retries=3,
)
def backfill_missing(
    self: AppTask,
    user_id: str | None = None,
    table: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Sweep rows with no vector — the debris an embedder outage leaves behind.

    ``user_id=None`` sweeps every user; this is a maintenance path, never a
    request path.
    """
    return run_async(_backfill_missing(user_id, table, limit))


__all__ = [
    "embed_batch",
    "backfill_missing",
    "fan_to_embed",
]
