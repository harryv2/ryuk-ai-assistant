"""SSE over Redis pub/sub, with a replay buffer behind it.

One channel per run. Every event carries the same envelope and a `seq` that
starts at 1 and increases by exactly one, allocated by a Redis ``INCR`` so it is
monotonic and gapless at the publisher even when several workers write to the
same run.

That guarantee is the whole contract with the browser:

* ``seq == last + 1`` — apply it;
* ``seq <= last`` — a duplicate from a reconnect, drop it. Every event is safe
  to drop and not all are safe to apply twice, because ``content.delta``
  appends;
* ``seq > last + 1`` — events were missed. Reconnect with ``Last-Event-ID`` and
  the server replays from the buffer.

The buffer is a capped stream: the last 500 events, fifteen minutes. Inside that
window a replay is exact; outside it the client falls back to
``GET /conversations/{id}``, which reads Postgres. SSE is an accelerator, never
the source of truth — which is also why nothing here raises. A run whose events
cannot be published still runs, still writes its rows and still answers.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Final

from app.core.logging import get_logger

log = get_logger(__name__)

ENVELOPE_VERSION: Final[int] = 1
BUFFER_SIZE: Final[int] = 500
BUFFER_TTL_S: Final[int] = 15 * 60
PING_INTERVAL_S: Final[float] = 15.0

# The event union on a run's channel.
RUN_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "run.started",
        "progress",
        "probe.done",
        "intent",
        "plan.step",
        "step.started",
        "step.retrying",
        "step.finished",
        "content.delta",
        "token",
        "input.raised",
        "action.prepared",
        "run.paused",
        "run.complete",
        "error",
    }
)

# Two events that fire after the run that created them has finished, because
# approving an action queues work to a worker.
CONVERSATION_EVENTS: Final[frozenset[str]] = frozenset({"action.done", "action.failed"})

TERMINAL_EVENTS: Final[frozenset[str]] = frozenset({"run.complete", "error"})


def _key(kind: str, ident: str) -> str:
    from app.core import cache

    return cache.key(kind, ident)


def channel(run_id: str) -> str:
    return _key("ev:run", run_id)


def buffer_key(run_id: str) -> str:
    return _key("buf:run", run_id)


def seq_key(run_id: str) -> str:
    return _key("seq:run", run_id)


def conversation_channel(conversation_id: str) -> str:
    return _key("ev:conv", conversation_id)


def envelope(run_id: str, type: str, data: dict[str, Any], seq: int) -> dict[str, Any]:
    return {
        "v": ENVELOPE_VERSION,
        "seq": seq,
        "type": type,
        "run_id": run_id,
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "data": data or {},
    }


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _loads(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


async def next_seq(run_id: str) -> int:
    """Allocate the next sequence number for a run. Monotonic across workers."""
    from app.core import cache

    client = await cache.get_redis()
    key = seq_key(run_id)
    value = int(await client.incr(key))
    if value == 1:
        await client.expire(key, BUFFER_TTL_S * 4)
    return value


async def publish(run_id: str, type: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Put one event on a run's channel, and in its replay buffer.

    Never raises. A broken Redis costs the live stream, not the run.
    """
    if not run_id or not type:
        return None
    try:
        from app.core import cache

        client = await cache.get_redis()
        seq = int(await client.incr(seq_key(run_id)))
        message = envelope(run_id, type, data or {}, seq)
        payload = _dumps(message)

        pipe = client.pipeline()
        pipe.publish(channel(run_id), payload)
        pipe.xadd(buffer_key(run_id), {"e": payload}, maxlen=BUFFER_SIZE, approximate=True)
        pipe.expire(buffer_key(run_id), BUFFER_TTL_S)
        pipe.expire(seq_key(run_id), BUFFER_TTL_S * 4)
        await pipe.execute()
        return message
    except Exception as exc:  # noqa: BLE001 - the stream is an accelerator
        log.warning("events.publish_failed", run_id=run_id, type=type, error=str(exc))
        return None


async def publish_many(run_id: str, events: list[tuple[str, dict[str, Any]]]) -> None:
    for type, data in events:
        await publish(run_id, type, data)


async def publish_conversation(
    conversation_id: str,
    type: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
) -> None:
    """`action.done` and `action.failed`, which outlive the run that staged them.

    Delivered on the conversation channel and, when we know it, on the channel
    of the run that prepared the action — a client watching either one sees it.
    """
    try:
        from app.core import cache

        client = await cache.get_redis()
        seq = int(await client.incr(_key("seq:conv", conversation_id)))
        message = envelope(run_id or "", type, data or {}, seq)
        message["conversation_id"] = conversation_id
        await client.publish(conversation_channel(conversation_id), _dumps(message))
    except Exception as exc:  # noqa: BLE001
        log.warning("events.publish_conversation_failed", type=type, error=str(exc))
    if run_id:
        await publish(run_id, type, data)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


async def replay(run_id: str, from_seq: int = 0, *, limit: int = BUFFER_SIZE) -> list[dict[str, Any]]:
    """Buffered events with ``seq >= from_seq``, oldest first.

    An empty list when the buffer has aged out is not an error — it is the
    signal for the client to fall back to the durable record.
    """
    try:
        from app.core import cache

        client = await cache.get_redis()
        entries = await client.xrange(buffer_key(run_id), count=limit)
    except Exception as exc:  # noqa: BLE001
        log.warning("events.replay_failed", run_id=run_id, error=str(exc))
        return []

    out: list[dict[str, Any]] = []
    for _entry_id, fields in entries or []:
        raw = fields.get(b"e") if isinstance(fields, dict) else None
        if raw is None and isinstance(fields, dict):
            raw = fields.get("e")
        message = _loads(raw)
        if message and int(message.get("seq", 0)) >= int(from_seq or 0):
            out.append(message)
    return out


async def subscribe(
    run_id: str,
    *,
    from_seq: int = 0,
    types: set[str] | None = None,
    idle_timeout_s: float = 300.0,
) -> AsyncIterator[dict[str, Any]]:
    """Every event for a run, replaying anything missed first.

    Yields until the run settles — ``run.complete`` or ``error`` — or until the
    stream has been idle long enough that nobody is coming back for it.
    """
    seen = int(from_seq or 0) - 1

    for message in await replay(run_id, from_seq):
        seq = int(message.get("seq", 0))
        if seq <= seen:
            continue
        seen = seq
        if types is None or message.get("type") in types:
            yield message
        if message.get("type") in TERMINAL_EVENTS:
            return

    try:
        from app.core import cache

        client = await cache.get_redis()
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel(run_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("events.subscribe_failed", run_id=run_id, error=str(exc))
        return

    idle = 0.0
    try:
        while True:
            try:
                raw = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=PING_INTERVAL_S
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dropped connection ends the stream
                log.warning("events.stream_dropped", run_id=run_id, error=str(exc))
                return

            if raw is None:
                idle += PING_INTERVAL_S
                if idle >= idle_timeout_s:
                    return
                yield {"type": "ping", "run_id": run_id, "seq": seen, "data": {}}
                continue

            idle = 0.0
            message = _loads(raw.get("data"))
            if not message:
                continue
            seq = int(message.get("seq", 0))
            if seq <= seen:
                continue  # a duplicate from a reconnect
            seen = seq
            if types is None or message.get("type") in types:
                yield message
            if message.get("type") in TERMINAL_EVENTS:
                return
    finally:
        try:
            await pubsub.unsubscribe(channel(run_id))
            await pubsub.aclose()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


async def subscribe_conversation(
    conversation_id: str, *, idle_timeout_s: float = 300.0
) -> AsyncIterator[dict[str, Any]]:
    """`action.done` and `action.failed` for a whole conversation."""
    try:
        from app.core import cache

        client = await cache.get_redis()
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(conversation_channel(conversation_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("events.subscribe_conv_failed", conversation_id=conversation_id, error=str(exc))
        return

    idle = 0.0
    try:
        while True:
            raw = await pubsub.get_message(ignore_subscribe_messages=True, timeout=PING_INTERVAL_S)
            if raw is None:
                idle += PING_INTERVAL_S
                if idle >= idle_timeout_s:
                    return
                continue
            idle = 0.0
            message = _loads(raw.get("data"))
            if message:
                yield message
    finally:
        try:
            await pubsub.unsubscribe(conversation_channel(conversation_id))
            await pubsub.aclose()
        except Exception:  # noqa: BLE001
            pass


def sse_frame(message: dict[str, Any]) -> str:
    """One event in SSE wire format.

    ``id:`` is the seq, so a browser ``EventSource`` sends ``Last-Event-ID`` on
    reconnect by itself; ``event:`` is the type, so ``addEventListener`` works.
    """
    if message.get("type") == "ping":
        return ": ping\n\n"
    return (
        f"id: {message.get('seq', 0)}\n"
        f"event: {message.get('type', 'message')}\n"
        f"data: {_dumps(message)}\n\n"
    )


async def clear(run_id: str) -> None:
    """Drop a run's buffer and counter. Only for tests and for account deletion."""
    try:
        from app.core import cache

        client = await cache.get_redis()
        await client.delete(buffer_key(run_id), seq_key(run_id))
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "BUFFER_SIZE",
    "BUFFER_TTL_S",
    "CONVERSATION_EVENTS",
    "ENVELOPE_VERSION",
    "RUN_EVENTS",
    "TERMINAL_EVENTS",
    "channel",
    "clear",
    "conversation_channel",
    "envelope",
    "next_seq",
    "publish",
    "publish_conversation",
    "publish_many",
    "replay",
    "sse_frame",
    "subscribe",
    "subscribe_conversation",
]
