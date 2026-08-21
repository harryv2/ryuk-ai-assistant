"""Turning mirror rows into vectors, as few times as possible.

Three savings stack, in this order, before anything reaches the API:

1. **Unchanged content is never re-embedded.** ``content_hash`` is a uuid5 of
   the exact text we embedded last time. A resync that brings back the same
   body brings back the same hash, and the row is skipped without a request.
2. **Redis remembers the vector by fingerprint.** Two people forwarded the same
   newsletter; the second one costs a Redis GET. This is why the cache is keyed
   on ``content_hash`` rather than on the row id — the row is per user, the
   content is not.
3. **Batches of 128.** The API's limit per request, and the difference between
   one round trip and a hundred and twenty-eight of them.

In steady state the first two mean the ``emb`` namespace runs above 95% hits,
which is the number ``/metrics`` reports as ``cache_hit_rate{scope="sync"}``.
"""

from __future__ import annotations

import time
import uuid
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import cache, llm
from app.core.ids import fingerprint
from app.core.logging import get_logger
from app.db.repositories import mirror
from app.search import chunking

log = get_logger(__name__)

#: The embedding API's maximum inputs per request.
BATCH_SIZE = 128

#: uuid5 namespaces, one per mirror table, so the same text in a subject line
#: and in a filename never collides.
HASH_NAMESPACES = {
    "gmail": "sync.gmail",
    "gcal": "sync.gcal",
    "gdrive": "sync.gdrive",
}


@dataclass
class EmbedReport:
    """What one pass did. Goes straight into the Celery task's log line."""

    table: str
    scanned: int = 0
    embedded: int = 0
    cache_hits: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    requests: int = 0
    took_ms: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def api_texts(self) -> int:
        """Texts that actually cost money."""
        return self.embedded - self.cache_hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "scanned": self.scanned,
            "embedded": self.embedded,
            "cache_hits": self.cache_hits,
            "skipped_unchanged": self.skipped_unchanged,
            "skipped_empty": self.skipped_empty,
            "requests": self.requests,
            "took_ms": self.took_ms,
            "errors": self.errors,
        }


def text_for_row(table: str, row: Any) -> str:
    """The exact string that gets embedded for one mirror row."""
    return chunking.embed_text_for_row(table, row)


def content_hash_for(table: str, text: str) -> uuid.UUID:
    """The fingerprint stored on ``content_hash``.

    Same content, same uuid, on any machine, forever — which is the whole
    mechanism behind "skip rows that did not change".
    """
    namespace = HASH_NAMESPACES.get(table.lower().removeprefix("sync_"), f"sync.{table}")
    return fingerprint(namespace, text)


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _needs_embedding(table: str, row: Any) -> tuple[bool, str, uuid.UUID | None]:
    """``(embed it?, text, hash)`` for one row.

    A row is skipped when it already carries a vector and its stored
    ``content_hash`` still matches the text we would send. Anything else — a
    new row, a changed body, a missing vector — is work.
    """
    text = text_for_row(table, row).strip()
    if not text:
        return (False, "", None)
    digest = content_hash_for(table, text)
    stored = _get(row, "content_hash")
    has_vector = _get(row, "embedding") is not None
    if has_vector and stored is not None and str(stored) == str(digest):
        return (False, text, digest)
    return (True, text, digest)


async def _vectors_for(
    texts: Sequence[str],
    digests: Sequence[uuid.UUID],
    report: EmbedReport,
) -> list[list[float]]:
    """Vectors for a batch: Redis first, the API for whatever is left."""
    out: list[list[float] | None] = [None] * len(texts)

    cached = await _cache_get_many(digests)
    todo: list[int] = []
    for index, vector in enumerate(cached):
        if vector is None:
            todo.append(index)
        else:
            out[index] = vector
            report.cache_hits += 1

    if todo:
        # ``use_cache=False``: llm.embed's own cache is keyed on the text, and
        # we have already asked the same question keyed on the content hash.
        fetched = await llm.embed([texts[i] for i in todo], use_cache=False)
        report.requests += 1
        for index, vector in zip(todo, fetched, strict=True):
            out[index] = vector
        await _cache_set_many(
            [(digests[i], out[i]) for i in todo if out[i] is not None]
        )

    dim = llm.embedding_dim()
    return [vector if vector is not None else [0.0] * dim for vector in out]


async def _cache_get_many(digests: Sequence[uuid.UUID]) -> list[list[float] | None]:
    """One round trip for a whole batch, falling back to one key at a time."""
    if not digests:
        return []
    try:
        client = await cache.get_redis()
        raw_rows = await client.mget([cache.key("emb", d) for d in digests])
    except Exception:
        out: list[list[float] | None] = []
        for digest in digests:
            out.append(await cache.get_embedding_by_fp(digest))
        return out

    vectors: list[list[float] | None] = []
    hits = 0
    for raw in raw_rows:
        vector: list[float] | None = None
        if raw:
            buffer = array("f")
            try:
                buffer.frombytes(raw)
                vector = buffer.tolist()
                hits += 1
            except ValueError:
                vector = None
        vectors.append(vector)
    if hits:
        await cache.record_hit("emb")
    if len(vectors) - hits:
        await cache.record_miss("emb")
    return vectors


async def _cache_set_many(pairs: Iterable[tuple[uuid.UUID, Sequence[float]]]) -> None:
    for digest, vector in pairs:
        await cache.set_embedding_by_fp(digest, vector)


async def embed_rows(
    session: AsyncSession,
    user_id: str,
    table: str,
    rows: Sequence[Any],
    *,
    batch_size: int = BATCH_SIZE,
    write_hash: bool = False,
) -> EmbedReport:
    """Vectorise the rows given, in batches, and write the vectors back.

    ``write_hash=True`` also stores the recomputed ``content_hash``; the sync
    tasks set it on upsert, so the default leaves it alone.
    """
    started = time.perf_counter()
    report = EmbedReport(table=table, scanned=len(rows))
    if not rows:
        return report

    pending: list[tuple[str, str, uuid.UUID]] = []  # (row id, text, hash)
    for row in rows:
        wanted, text, digest = _needs_embedding(table, row)
        if digest is None:
            report.skipped_empty += 1
            continue
        if not wanted:
            report.skipped_unchanged += 1
            continue
        pending.append((str(_get(row, "id")), text, digest))

    for start in range(0, len(pending), max(1, batch_size)):
        window = pending[start : start + max(1, batch_size)]
        # One vector per distinct hash: a batch of near-duplicate marketing
        # mail is one API input, not forty.
        unique: dict[uuid.UUID, str] = {}
        for _, text, digest in window:
            unique.setdefault(digest, text)
        digests = list(unique)
        vectors = await _vectors_for([unique[d] for d in digests], digests, report)
        by_hash = dict(zip(digests, vectors, strict=True))

        pairs = [(row_id, by_hash[digest]) for row_id, _, digest in window]
        await mirror.set_embeddings(session, user_id, table, pairs)
        if write_hash:
            await _write_hashes(session, user_id, table, window)
        report.embedded += len(window)

    report.took_ms = int((time.perf_counter() - started) * 1000)
    log.info("embedder.batch", **report.as_dict(), user_id=user_id)
    return report


async def _write_hashes(
    session: AsyncSession,
    user_id: str,
    table: str,
    window: Sequence[tuple[str, str, uuid.UUID]],
) -> None:
    from sqlalchemy import bindparam, update

    spec = mirror.spec_for(table)
    model = spec.model
    statement = (
        update(model)
        .where(model.id == bindparam("row_id"), model.user_id == user_id)
        .values(content_hash=bindparam("digest"))
    )
    await session.execute(
        statement,
        [{"row_id": row_id, "digest": digest} for row_id, _, digest in window],
    )


async def embed_pending(
    session: AsyncSession,
    user_id: str | None,
    table: str,
    *,
    limit: int = 500,
    batch_size: int = BATCH_SIZE,
) -> EmbedReport:
    """Embed whatever is still missing a vector in one mirror table.

    ``user_id=None`` sweeps every user and is for the embed queue only. Rows
    are fetched per user so ``mirror.set_embeddings`` keeps its tenant guard.
    """
    rows = await mirror.rows_needing_embedding(session, user_id, table, limit=limit)
    if user_id is not None:
        return await embed_rows(session, user_id, table, rows, batch_size=batch_size)

    report = EmbedReport(table=table)
    by_user: dict[str, list[Any]] = {}
    for row in rows:
        by_user.setdefault(str(_get(row, "user_id")), []).append(row)
    for owner, owned in by_user.items():
        one = await embed_rows(session, owner, table, owned, batch_size=batch_size)
        report.scanned += one.scanned
        report.embedded += one.embedded
        report.cache_hits += one.cache_hits
        report.skipped_unchanged += one.skipped_unchanged
        report.skipped_empty += one.skipped_empty
        report.requests += one.requests
        report.took_ms += one.took_ms
    return report


async def embed_all(
    session: AsyncSession,
    user_id: str | None,
    *,
    limit: int = 500,
    batch_size: int = BATCH_SIZE,
) -> dict[str, EmbedReport]:
    """One pass over all three mirror tables."""
    out: dict[str, EmbedReport] = {}
    for table in ("gmail", "gcal", "gdrive"):
        out[table] = await embed_pending(
            session, user_id, table, limit=limit, batch_size=batch_size
        )
    return out


async def embed_query(text: str, *, use_cache: bool = True) -> list[float]:
    """The probe's single embedding.

    Cached on the query string, so a repeated question — "what's on my calendar
    next week" typed twice in a morning — costs a Redis GET and about a
    millisecond.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return [0.0] * llm.embedding_dim()
    return await llm.embed_one(cleaned, use_cache=use_cache)


__all__ = [
    "BATCH_SIZE",
    "HASH_NAMESPACES",
    "EmbedReport",
    "content_hash_for",
    "embed_all",
    "embed_pending",
    "embed_query",
    "embed_rows",
    "text_for_row",
]
