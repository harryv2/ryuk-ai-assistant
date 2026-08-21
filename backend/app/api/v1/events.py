"""Server-Sent Events. The trace panel is only convincing if these land as they
happen.

Two channels.

**The run channel**, ``GET /runs/{run_id}/events``, carries one run from
``run.started`` to ``run.complete`` or ``error``. It is read-only — opening it
starts nothing — and it replays: ``seq`` is allocated by a Redis ``INCR``, so it
is gapless at the publisher, and a client that spots a jump reconnects with
``Last-Event-ID`` and gets the missing events out of the buffer.

**The conversation channel**, ``GET /conversations/{id}/events``, carries the
two events that arrive after the run that prepared them has already finished:
``action.done`` and ``action.failed``. Approving a write queues a Celery job,
and the job can land minutes later, by which time the run's own stream is long
closed.

A ``: ping`` comment goes out every fifteen seconds on both. An idle stream with
no bytes on it is one a proxy will close, and a closed stream looks exactly like
a hung run.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.auth.deps import CurrentUserId, SessionDep
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.repositories import conversations as conv_repo
from app.db.repositories import runs as run_repo
from app.orchestrator import events as bus

log = get_logger(__name__)
router = APIRouter(tags=["events"])

#: Anything in front of us — nginx, a dev proxy, a corporate middlebox — will
#: drop a connection that has been silent for a while.
HEARTBEAT_S = 15.0

SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: Told to the browser once, at the top of the stream: how long to wait before
#: reconnecting. `EventSource` honours it.
RETRY_MS = 2000


def _from_seq(request: Request, explicit: int | None) -> int:
    """Where to resume.

    ``Last-Event-ID`` is what a browser ``EventSource`` sends by itself, and it
    holds the last seq the client *applied* — so the replay starts at the one
    after it. ``from_seq`` is the same thing spelled out, for clients that are
    not ``EventSource``, and it is inclusive.
    """
    if explicit is not None:
        return max(0, explicit)
    header = request.headers.get("Last-Event-ID")
    if not header:
        return 0
    try:
        return max(0, int(header)) + 1
    except ValueError:
        log.info("events.bad_last_event_id", value=header[:40])
        return 0


def _types(raw: str | None) -> set[str] | None:
    """The ``types=`` filter, or ``None`` for everything."""
    if not raw:
        return None
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return wanted or None


async def _with_heartbeat(
    source: AsyncIterator[dict[str, Any]], request: Request
) -> AsyncIterator[str]:
    """Relay a subscription as SSE frames, keeping the connection warm.

    The reader runs as its own task so a quiet channel still produces bytes:
    without that, the heartbeat could only fire between events, which is
    precisely when there are none.
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def reader() -> None:
        try:
            async for message in source:
                await queue.put(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("events.reader_failed", error=str(exc))
        finally:
            await queue.put(None)

    task = asyncio.create_task(reader())
    yield f"retry: {RETRY_MS}\n\n"
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                if await request.is_disconnected():
                    break
                yield ": ping\n\n"
                continue
            if message is None:
                break
            yield bus.sse_frame(message)
    finally:
        task.cancel()
        close = getattr(source, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):  # teardown is best effort
                await close()


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    session: SessionDep,
    user_id: CurrentUserId,
    from_seq: int | None = Query(None, ge=0),
    types: str | None = Query(None),
) -> StreamingResponse:
    """Every event for one run.

    A run belonging to somebody else is a 404, not a 403 — a 403 would confirm
    the id exists. The check happens before the headers go out, because once
    ``200 text/event-stream`` is committed there is no status code left to
    change: from that point a failure is an ``error`` *event*.
    """
    run = await run_repo.get_run(session, user_id, run_id)
    if run is None:
        raise AppError("NOT_FOUND", "No run with that id.", details={"run_id": run_id})

    stream = bus.subscribe(run_id, from_seq=_from_seq(request, from_seq), types=_types(types))
    return StreamingResponse(
        _with_heartbeat(stream, request),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/conversations/{conversation_id}/events")
async def conversation_events(
    conversation_id: str,
    request: Request,
    session: SessionDep,
    user_id: CurrentUserId,
) -> StreamingResponse:
    """Outcomes that arrive after the run that prepared them has finished.

    An approved action executes in a worker seconds or minutes later. The run's
    own channel has closed by then, so ``action.done`` and ``action.failed``
    land here instead.
    """
    conversation = await conv_repo.get_conversation(session, user_id, conversation_id)
    if conversation is None:
        raise AppError(
            "NOT_FOUND",
            "No conversation with that id.",
            details={"conversation_id": conversation_id},
        )

    stream = bus.subscribe_conversation(conversation_id)
    return StreamingResponse(
        _with_heartbeat(stream, request),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


__all__ = ["router"]
