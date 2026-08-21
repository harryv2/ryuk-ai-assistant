"""node_executions — one row per step of a plan.

The same rows are read at three moments: as a live progress feed, as the trace
panel, and as the record afterwards. Nothing here is deleted or rewritten; a
replan writes a new ``round`` instead.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import cast, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import NodeExecution, NodeStatus
from app.db.repositories import ensure_utc, utcnow

#: A node in one of these is finished and will not move again.
TERMINAL_STATUSES = frozenset(
    {
        NodeStatus.SUCCEEDED,
        NodeStatus.FAILED,
        NodeStatus.SKIPPED,
        NodeStatus.TIMEOUT,
        NodeStatus.CANCELLED,
    }
)


async def insert_steps(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    conversation_id: str,
    steps: Sequence[dict[str, Any]],
    *,
    round: int = 0,
    start_seq: int | None = None,
) -> list[NodeExecution]:
    """Write a whole plan in one go.

    Each item is the planner's step shape: ``{id, op, args, depends_on, ...}``.
    ``args`` are stored post-templating. Returns the rows in plan order.
    """
    if not steps:
        return []

    seq = start_seq
    if seq is None:
        result = await session.execute(
            select(func.coalesce(func.max(NodeExecution.seq), -1) + 1).where(
                NodeExecution.run_id == run_id, NodeExecution.user_id == user_id
            )
        )
        seq = int(result.scalar_one())

    rows: list[NodeExecution] = []
    for offset, step in enumerate(steps):
        node_id = step.get("node_id") or step.get("id")
        op = step.get("op")
        if not node_id or not op:
            raise AppError(
                "VALIDATION_ERROR",
                "Every step needs an id and an op.",
                http=422,
                details={"step": step},
            )
        rows.append(
            NodeExecution(
                id=step.get("node_execution_id") or new_id(),
                run_id=run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                message_id=step.get("message_id"),
                seq=seq + offset,
                node_id=str(node_id),
                op=str(op),
                round=round,
                args=dict(step.get("args") or {}),
                depends_on=list(step.get("depends_on") or []),
                status=NodeStatus(step.get("status", NodeStatus.PENDING)),
            )
        )

    session.add_all(rows)
    await session.flush()
    return rows


async def get_step(
    session: AsyncSession, user_id: str, node_execution_id: str
) -> NodeExecution | None:
    """One step by its row id."""
    result = await session.execute(
        select(NodeExecution).where(
            NodeExecution.id == node_execution_id, NodeExecution.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def get_step_by_node(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    node_id: str,
    *,
    round: int | None = None,
) -> NodeExecution | None:
    """One step by the readable name the planner gave it.

    ``round=None`` returns the latest round, which is what ``{{step.path}}``
    references should resolve against.
    """
    stmt = select(NodeExecution).where(
        NodeExecution.run_id == run_id,
        NodeExecution.user_id == user_id,
        NodeExecution.node_id == node_id,
    )
    if round is not None:
        stmt = stmt.where(NodeExecution.round == round)
    stmt = stmt.order_by(NodeExecution.round.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_steps(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    *,
    round: int | None = None,
    statuses: Iterable[NodeStatus | str] | None = None,
) -> list[NodeExecution]:
    """Every step of a run, in display order."""
    stmt = select(NodeExecution).where(
        NodeExecution.run_id == run_id, NodeExecution.user_id == user_id
    )
    if round is not None:
        stmt = stmt.where(NodeExecution.round == round)
    if statuses is not None:
        stmt = stmt.where(
            NodeExecution.status.in_([NodeStatus(s) for s in statuses])
        )
    stmt = stmt.order_by(NodeExecution.round.asc(), NodeExecution.seq.asc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_steps_for_message(
    session: AsyncSession, user_id: str, message_id: str
) -> list[NodeExecution]:
    """The trace under one assistant message. Without ``message_id`` a paused
    run's trace would appear under both of its messages."""
    result = await session.execute(
        select(NodeExecution)
        .where(NodeExecution.message_id == message_id, NodeExecution.user_id == user_id)
        .order_by(NodeExecution.seq.asc())
    )
    return list(result.scalars().all())


async def list_steps_for_conversation(
    session: AsyncSession, user_id: str, conversation_id: str, *, limit: int = 200
) -> list[NodeExecution]:
    """Direct trace query without two hops — that is what the denormalised
    ``conversation_id`` is for."""
    result = await session.execute(
        select(NodeExecution)
        .where(
            NodeExecution.conversation_id == conversation_id,
            NodeExecution.user_id == user_id,
        )
        .order_by(NodeExecution.started_at.desc().nullslast())
        .limit(limit)
    )
    return list(result.scalars().all())


async def mark_started(
    session: AsyncSession,
    user_id: str,
    node_execution_id: str,
    *,
    args: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> NodeExecution | None:
    """The dispatcher picked this step up.

    ``args`` is written again here because bind() resolves references just
    before the call, and the row must hold what actually ran.
    """
    values: dict[str, Any] = {
        "status": NodeStatus.RUNNING,
        "started_at": ensure_utc(now) or utcnow(),
        "finished_at": None,
    }
    if args is not None:
        values["args"] = args
    result = await session.execute(
        update(NodeExecution)
        .where(
            NodeExecution.id == node_execution_id,
            NodeExecution.user_id == user_id,
        )
        .values(**values)
        .returning(NodeExecution)
    )
    return result.scalar_one_or_none()


async def mark_finished(
    session: AsyncSession,
    user_id: str,
    node_execution_id: str,
    status: NodeStatus | str,
    *,
    result: dict[str, Any] | None = None,
    outcome: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> NodeExecution | None:
    """Close a step.

    ``result`` is the ``to_ui()`` output — a snapshot, so it must not shift when
    Gmail changes. ``outcome`` carries the non-success detail:
    ``{reason, class, code, message}``.
    """
    values: dict[str, Any] = {
        "status": NodeStatus(status),
        "finished_at": ensure_utc(now) or utcnow(),
    }
    if result is not None:
        values["result"] = result
    if outcome is not None:
        values["outcome"] = outcome
    row = await session.execute(
        update(NodeExecution)
        .where(NodeExecution.id == node_execution_id, NodeExecution.user_id == user_id)
        .values(**values)
        .returning(NodeExecution)
    )
    return row.scalar_one_or_none()


async def update_step(
    session: AsyncSession,
    user_id: str,
    node_execution_id: str,
    *,
    status: NodeStatus | str | None = None,
    result: dict[str, Any] | None = None,
    outcome: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
    message_id: str | None = None,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
) -> NodeExecution | None:
    """The general-purpose write. Only the fields you pass are touched."""
    values: dict[str, Any] = {}
    if status is not None:
        values["status"] = NodeStatus(status)
    if result is not None:
        values["result"] = result
    if outcome is not None:
        values["outcome"] = outcome
    if args is not None:
        values["args"] = args
    if message_id is not None:
        values["message_id"] = message_id
    if started_at is not None:
        values["started_at"] = ensure_utc(started_at)
    if finished_at is not None:
        values["finished_at"] = ensure_utc(finished_at)
    if not values:
        return await get_step(session, user_id, node_execution_id)

    row = await session.execute(
        update(NodeExecution)
        .where(NodeExecution.id == node_execution_id, NodeExecution.user_id == user_id)
        .values(**values)
        .returning(NodeExecution)
    )
    return row.scalar_one_or_none()


async def record_retry(
    session: AsyncSession,
    user_id: str,
    node_execution_id: str,
    *,
    error_class: str,
    google_status: int | None = None,
    backoff_ms: int | None = None,
    at: dt.datetime | None = None,
) -> NodeExecution | None:
    """Append one attempt to ``retries``.

    Done as a JSONB concat in SQL so two concurrent retries cannot clobber each
    other's entry. Attempt count is ``jsonb_array_length(retries) + 1``.
    """
    entry = {
        "at": (ensure_utc(at) or utcnow()).isoformat(),
        "error_class": error_class,
        "google_status": google_status,
        "backoff_ms": backoff_ms,
    }
    row = await session.execute(
        update(NodeExecution)
        .where(NodeExecution.id == node_execution_id, NodeExecution.user_id == user_id)
        .values(
            retries=func.coalesce(NodeExecution.retries, cast([], JSONB)).op("||")(
                cast([entry], JSONB)
            )
        )
        .returning(NodeExecution)
    )
    return row.scalar_one_or_none()


async def attach_message(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    message_id: str,
    *,
    node_execution_ids: Sequence[str] | None = None,
) -> int:
    """Say which assistant message reported these nodes.

    Called when the message is emitted. With no explicit list it claims every
    step of the run that no message has claimed yet, which is exactly the
    pause-and-resume case.
    """
    stmt = update(NodeExecution).where(
        NodeExecution.run_id == run_id, NodeExecution.user_id == user_id
    )
    if node_execution_ids is not None:
        if not node_execution_ids:
            return 0
        stmt = stmt.where(NodeExecution.id.in_(list(node_execution_ids)))
    else:
        stmt = stmt.where(NodeExecution.message_id.is_(None))
    result = await session.execute(stmt.values(message_id=message_id))
    return int(result.rowcount or 0)


async def cancel_pending(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    *,
    reason: str = "run_cancelled",
) -> int:
    """Close out anything still open when a run stops."""
    result = await session.execute(
        update(NodeExecution)
        .where(
            NodeExecution.run_id == run_id,
            NodeExecution.user_id == user_id,
            NodeExecution.status.notin_(list(TERMINAL_STATUSES)),
        )
        .values(
            status=NodeStatus.CANCELLED,
            outcome={"reason": reason, "class": "CANCELLED"},
            finished_at=utcnow(),
        )
    )
    return int(result.rowcount or 0)


async def next_round(session: AsyncSession, user_id: str, run_id: str) -> int:
    """The round number a replan should write under."""
    result = await session.execute(
        select(func.coalesce(func.max(NodeExecution.round), 0) + 1).where(
            NodeExecution.run_id == run_id, NodeExecution.user_id == user_id
        )
    )
    return int(result.scalar_one())


async def results_by_node(
    session: AsyncSession, user_id: str, run_id: str
) -> dict[str, dict[str, Any] | None]:
    """``{node_id: result}`` for the latest round of each node — the bag that
    ``dispatch.bind()`` resolves ``{{step.path}}`` against."""
    steps = await list_steps(session, user_id, run_id)
    out: dict[str, dict[str, Any] | None] = {}
    for step in steps:  # ordered by round asc, so later rounds win
        out[step.node_id] = step.result
    return out


__all__ = [
    "TERMINAL_STATUSES",
    "insert_steps",
    "get_step",
    "get_step_by_node",
    "list_steps",
    "list_steps_for_message",
    "list_steps_for_conversation",
    "mark_started",
    "mark_finished",
    "update_step",
    "record_retry",
    "attach_message",
    "cancel_pending",
    "next_round",
    "results_by_node",
]
