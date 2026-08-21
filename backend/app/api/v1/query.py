"""``POST /api/v1/query`` — the one entry point.

A natural-language message in; an answer out, plus — if the message implied a
side effect — a *prepared* action and the card that gates it. **Nothing in this
module writes to Google.** A query prepares, a person approves, a worker
executes. That split is enforced by the database (``actions.requires_input_id``
is ``NOT NULL``), not by good intentions here.

Two transports, one body of events. The default buffers and returns the run's
terminal state as one JSON object. ``?stream=ndjson`` holds the connection open
and writes the same envelopes the SSE channel carries, one per line, so the
whole flow is visible from ``curl`` without a browser.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._shared import hydrate, hydrate_run
from app.api.v1.schemas import QueryRequest
from app.auth.deps import CurrentUser, SessionDep
from app.config import settings
from app.core import cache, ratelimit
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import User
from app.orchestrator import events, runner

log = get_logger(__name__)
router = APIRouter(tags=["query"])

#: How long a ``client_request_id`` keeps pointing at the run it started. Long
#: enough to cover a double-tapped send button and a retried request, short
#: enough that the same handle means a new question tomorrow.
IDEMPOTENCY_TTL_S = 600

#: The contextvar name the orchestrator publishes its run id under. The NDJSON
#: relay needs the id before ``handle_query`` returns and there is no other way
#: to learn it — a task gets a *copy* of the context, so we keep the copy.
RUN_ID_VAR = "orchestrator_run_id"

#: How long to wait for that id before giving up on a live relay and simply
#: answering when the run finishes.
RUN_ID_WAIT_S = 3.0

#: Whatever `runner.run_query` accepts today. `freshness` is the planner's
#: decision per step rather than a run-wide switch, so it is forwarded only if
#: the orchestrator grows a parameter for it.
_RUNNER_ARGS = frozenset(inspect.signature(runner.run_query).parameters)


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def _limit_headers(remaining: int, reset_at: datetime) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_HOUR),
        "X-RateLimit-Remaining": str(max(0, int(remaining))),
        "X-RateLimit-Reset": str(int(reset_at.timestamp())),
    }


def _too_many(request: Request, reset_at: datetime) -> JSONResponse:
    """The 429, with the headers a client needs to back off correctly."""
    retry_after = max(1, int((reset_at - datetime.now(UTC)).total_seconds()))
    error = AppError(
        "RATE_LIMITED",
        f"That is {settings.RATE_LIMIT_PER_HOUR} queries this hour. "
        f"Try again in {retry_after // 60 + 1} minutes.",
        details={
            "limit": settings.RATE_LIMIT_PER_HOUR,
            "remaining": 0,
            "reset_at": reset_at.isoformat(),
            "retry_after_s": retry_after,
        },
    )
    headers = _limit_headers(0, reset_at)
    headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=429,
        content=error.to_response(getattr(request.state, "request_id", None)),
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def _idem_key(user_id: str, client_request_id: str) -> str:
    return f"idem:{user_id}:{client_request_id}"


async def _replay(
    session: AsyncSession, user_id: str, client_request_id: str | None
) -> dict[str, Any] | None:
    """The run this handle already started, if it is still inside the window."""
    if not client_request_id:
        return None
    held = await cache.get_json(_idem_key(user_id, client_request_id), namespace="query")
    run_id = (held or {}).get("run_id") if isinstance(held, dict) else None
    if not run_id:
        return None
    body = await hydrate_run(session, user_id, str(run_id))
    if body is not None:
        log.info("query.replayed", user_id=user_id, run_id=run_id)
    return body


async def _remember(user_id: str, client_request_id: str | None, run_id: str) -> None:
    if not client_request_id or not run_id:
        return
    await cache.set_json(
        _idem_key(user_id, client_request_id),
        {"run_id": run_id},
        ttl=IDEMPOTENCY_TTL_S,
        namespace="query",
    )


# --------------------------------------------------------------------------- #
# Running one turn
# --------------------------------------------------------------------------- #


def _actor_for(user: User, body: QueryRequest) -> runner.Actor:
    """The person, with this request's overrides applied.

    The browser knows where someone is *right now*; the stored profile knows
    where they live. When they differ — travelling — the request wins, for this
    query only, because "next Tuesday" means a different week in each.
    """
    return runner.Actor(
        id=user.id,
        email=str(user.email or ""),
        display_name=str(user.display_name or ""),
        timezone=body.timezone or str(user.timezone or "UTC"),
        week_start=int(body.week_start or user.work_week_start or 1),
    )


def _kwargs(body: QueryRequest, request_id: str | None) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "conversation_id": body.conversation_id,
        "request_id": request_id,
    }
    if "freshness" in _RUNNER_ARGS:
        extra["freshness"] = body.freshness
    return extra


async def _run(
    session: AsyncSession, user: User, body: QueryRequest, request_id: str | None
) -> runner.QueryResult:
    return await runner.run_query(
        session, _actor_for(user, body), body.query, **_kwargs(body, request_id)
    )


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #


@router.post("/query")
async def post_query(
    body: QueryRequest,
    request: Request,
    session: SessionDep,
    user: CurrentUser,
    stream: str = Query("off", pattern="^(off|ndjson)$"),
) -> Any:
    """Answer one message.

    The rate limiter is charged first, before any work: a refused request must
    cost a Redis round trip, not an embedding and a planning call.
    """
    replayed = await _replay(session, user.id, body.client_request_id)
    if replayed is not None:
        _, remaining, reset_at = await ratelimit.check_query_limit(user.id, consume=False)
        return JSONResponse(content=replayed, headers=_limit_headers(remaining, reset_at))

    allowed, remaining, reset_at = await ratelimit.check_query_limit(user.id)
    if not allowed:
        return _too_many(request, reset_at)
    headers = _limit_headers(remaining, reset_at)
    request_id = getattr(request.state, "request_id", None)

    if stream == "ndjson":
        return StreamingResponse(
            _ndjson(session, user, body, request_id),
            media_type="application/x-ndjson",
            headers={
                **headers,
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    outcome = await _run(session, user, body, request_id)
    await _remember(user.id, body.client_request_id, outcome.run_id)
    return JSONResponse(
        content=await hydrate(session, user.id, outcome), headers=headers
    )


# --------------------------------------------------------------------------- #
# The NDJSON variant
# --------------------------------------------------------------------------- #


def _run_id_from(ctx: contextvars.Context) -> str | None:
    """The run id the task set inside its own copy of the context.

    A ``Context`` is a mapping, so reading it while the task is running in it is
    safe — unlike ``Context.run``, which refuses to be entered twice.
    """
    for var, value in ctx.items():
        if var.name == RUN_ID_VAR and value:
            return str(value)
    return None


async def _ndjson(
    session: AsyncSession, user: User, body: QueryRequest, request_id: str | None
) -> AsyncIterator[str]:
    """Run the turn, relaying its events line by line.

    Every line is the same envelope the SSE channel carries — same ``v``, same
    ``seq``, same ``type``, same ``data`` — so a client that can parse one
    transport can parse the other. The last line is always ``run.complete`` or
    ``error``; if the connection drops before then the run keeps going and
    ``GET /runs/{id}/events`` picks it back up.
    """
    ctx = contextvars.copy_context()
    task = asyncio.create_task(_run(session, user, body, request_id), context=ctx)

    run_id: str | None = None
    waited = 0.0
    while waited < RUN_ID_WAIT_S:
        run_id = _run_id_from(ctx)
        if run_id or task.done():
            break
        await asyncio.sleep(0.01)
        waited += 0.01

    if run_id:
        try:
            async for message in events.subscribe(run_id, from_seq=1):
                if message.get("type") == "ping":
                    continue
                yield json.dumps(message, default=str) + "\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("query.relay_failed", run_id=run_id, error=str(exc))

    try:
        outcome = await task
    except AppError as exc:
        yield json.dumps(
            events.envelope(run_id or "", "error", {**exc.to_dict(request_id), "partial": False}, 0),
            default=str,
        ) + "\n"
        return
    except Exception as exc:
        log.exception("query.stream_failed")
        wrapped = AppError("INTERNAL", details={"cause": type(exc).__name__})
        yield json.dumps(
            events.envelope(run_id or "", "error", {**wrapped.to_dict(request_id), "partial": False}, 0),
            default=str,
        ) + "\n"
        return

    await _remember(user.id, body.client_request_id, outcome.run_id)
    if not run_id:
        # The relay never attached — no id in time, or a run that finished
        # first. The client still gets a terminal event rather than a stream
        # that simply stops.
        yield json.dumps(
            events.envelope(
                outcome.run_id,
                "run.complete",
                {
                    "status": outcome.status,
                    "message_id": outcome.message_id,
                    "answer_style": outcome.answer_style,
                    "planner_tier": outcome.planner_tier,
                    "usage": outcome.usage,
                    "timings": outcome.timings,
                    "degraded": outcome.degraded,
                },
                0,
            ),
            default=str,
        ) + "\n"


__all__ = ["router"]
