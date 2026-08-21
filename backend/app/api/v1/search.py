"""``GET /api/v1/search`` — the retrieval layer with the lid off.

The chat path does not use this. It exists for two jobs: working out why an
answer was wrong, and the offline Precision@5 evaluation. So it returns **every
score component** rather than one number — a relevance figure you cannot
decompose is a figure you cannot act on.

The part worth reading twice is which numbers decide anything.

``rrf`` and ``final`` **order**. They are rank-derived: a document that came
first in both arms scores ``1/61 + 1/62 ≈ 0.0328``, and so does the best of
three bad matches. Compare that to ``FLOOR_READ`` 0.55 and every candidate in
the system fails, forever.

``cn`` and ``evidence`` **decide**. ``cn`` is cosine normalised per corpus —
z-scored against this user's own distribution for that service, clamped to
0..1 — and the floors and the ambiguity margin are computed on it alone.
``evidence`` is the escape hatch similarity cannot cover: an English query does
not embed near a Turkish body, but the sender really is ``bilet@thy.com``.

One embedding is computed here and reused across all three services.
``hybrid.search`` deliberately does not embed for you: the lexical arm needs no
vector, so the probe fires it at t=0 and lets the embedding request overlap.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any

from fastapi import APIRouter, Query

from app.api.v1.schemas import SERVICES
from app.auth.deps import CurrentUser, SessionDep
from app.core.errors import AppError
from app.core.logging import get_logger
from app.search import hybrid

log = get_logger(__name__)
router = APIRouter(tags=["search"])


def _wanted(raw: str | None) -> list[str]:
    """The ``services=`` list, defaulting to all three."""
    if not raw:
        return list(SERVICES)
    asked = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [name for name in asked if name not in SERVICES]
    if unknown:
        raise AppError.validation(
            "That is not a service we mirror.", services=unknown, known=list(SERVICES)
        )
    return [name for name in SERVICES if name in set(asked)] or list(SERVICES)


def _moment(value: str | None, field: str) -> dt.datetime | None:
    """An RFC 3339 stamp with an offset. A naive one is refused: 'since
    midnight' means a different instant in every timezone."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AppError.validation(
            f"{field} must be an RFC 3339 timestamp.", **{field: value}
        ) from exc
    if parsed.tzinfo is None:
        raise AppError.validation(
            f"{field} needs an explicit offset.", **{field: value}
        )
    return parsed.astimezone(dt.UTC)


def _filters(
    service: str,
    *,
    since: dt.datetime | None,
    until: dt.datetime | None,
    sender: str | None,
    attendee: str | None,
    mime: str | None,
) -> dict[str, Any]:
    """The metadata prefilter for one corpus.

    A filter that means nothing for a service is left off rather than rejected:
    ``from=`` is Gmail's, ``attendee=`` is Calendar's, ``mime=`` is Drive's, and
    a caller searching all three should not have to send three requests.
    """
    out: dict[str, Any] = {}
    if since is not None:
        out["since"] = since
    if until is not None:
        out["until"] = until
    if service == "gmail" and sender:
        out["from"] = sender
    if service == "gcal" and attendee:
        out["attendee"] = attendee
    if service == "gdrive" and mime:
        out["mime"] = mime
    return out


@router.get("/search")
async def search(
    session: SessionDep,
    user: CurrentUser,
    q: str = Query(..., min_length=1, max_length=1000),
    services: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
    freshness: str = Query("cached", pattern="^(cached|live)$"),
    since: str | None = Query(None),
    until: str | None = Query(None),
    sender: str | None = Query(None, alias="from"),
    attendee: str | None = Query(None),
    mime: str | None = Query(None),
    explain: bool = Query(False),
) -> dict[str, Any]:
    """Search the mirror and show the working."""
    corpora = _wanted(services)
    window = (_moment(since, "since"), _moment(until, "until"))
    started = time.perf_counter()

    embedding: list[float] | None = None
    embedding_ms = 0
    cached_vector = False
    try:
        from app.core import cache
        from app.llm import embed_model_id
        from app.search.embedder import embed_query

        cached_vector = await cache.get_embedding(embed_model_id(), q) is not None
        mark = time.perf_counter()
        embedding = await embed_query(q)
        embedding_ms = int((time.perf_counter() - mark) * 1000)
    except Exception as exc:
        # No vector is a worse search, not a failed one: the lexical arm still
        # answers, and saying so beats a 500 on a debug endpoint.
        log.warning("search.embedding_failed", error=str(exc))

    if freshness == "live":
        await _read_through(
            session, user.id, corpora, q, limit, str(user.timezone or "UTC")
        )

    results = await hybrid.search_many(
        session,
        user.id,
        corpora,
        q,
        filters={
            name: _filters(
                name,
                since=window[0],
                until=window[1],
                sender=sender,
                attendee=attendee,
                mime=mime,
            )
            for name in corpora
        },
        limit=limit,
        embedding=embedding,
        explain=explain,
    )

    every_hit = [hit for result in results.values() for hit in result.hits]
    return {
        "q": q,
        "took_ms": int((time.perf_counter() - started) * 1000),
        "embedding_ms": embedding_ms,
        "embedding_cached": cached_vector,
        "freshness": freshness,
        "services": {name: results[name].to_dict() for name in corpora},
        "decision": hybrid.decide(every_hit),
    }


#: The search op per corpus. Running one with ``freshness="live"`` is what
#: pulls a narrow, targeted page from Google into the mirror — one
#: ``messages.list``, not a resync — and the mirror is then what we score.
LIVE_OPS: dict[str, str] = {
    "gmail": "gmail.search_emails",
    "gcal": "gcal.search_events",
    "gdrive": "drive.search_files",
}


async def _read_through(
    session: Any, user_id: str, corpora: list[str], q: str, limit: int, tz: str
) -> None:
    """``freshness=live`` — pull from Google first, then search what landed.

    Slower, and honest about it. This goes through the ops rather than around
    them, so the live read is the same narrow one the chat path performs. A
    service that will not answer degrades to the mirror: a hit that is fifteen
    minutes old beats no hit on a debug endpoint, and ``/sync/status`` is where
    the lag gets reported.

    A dead Google *grant* is different from a dead Google *service* and is not
    swallowed — the client has to send the person back through consent.
    """
    from app.google.client import clients_for
    from app.ops import registry
    from app.ops.base import OpContext

    clients = await clients_for(session, user_id)
    ctx = OpContext(
        user_id=user_id,
        conversation_id="",
        run_id="",
        session=session,
        google=clients,
        now=dt.datetime.now(dt.UTC),
        tz=tz,
    )

    async def one(corpus: str) -> None:
        op = registry.get(LIVE_OPS.get(corpus, ""))
        if op is None:
            return
        await op.run(ctx, {"query": q, "limit": limit, "freshness": "live"})

    outcomes = await asyncio.gather(
        *(one(name) for name in corpora), return_exceptions=True
    )
    for name, outcome in zip(corpora, outcomes, strict=True):
        if isinstance(outcome, AppError) and outcome.code == "GOOGLE_REAUTH_REQUIRED":
            raise outcome
        if isinstance(outcome, BaseException):
            log.warning("search.live_read_failed", service=name, error=str(outcome))


__all__ = ["router"]
