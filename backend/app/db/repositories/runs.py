"""Runs — one execution of classify -> plan -> execute -> synthesize.

A run can produce several assistant messages (a clarification, then the answer),
which is how pause and resume work without rewriting history.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import NodeExecution, NodeStatus, Run, RunStatus
from app.db.repositories import ensure_utc, utcnow

#: Once a run reaches one of these it never moves again.
TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.TIMEOUT, RunStatus.CANCELLED}
)

#: What a worker picks up after a restart.
LIVE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.AWAITING_INPUT})


async def create_run(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    trigger_message_id: str,
    *,
    intent: dict[str, Any] | None = None,
    planner_tier: int | None = None,
    run_id: str | None = None,
    now: dt.datetime | None = None,
) -> Run:
    """Open a run for the user message that triggered it.

    Insert order is user message, run, assistant message — so neither side of
    the circular FK ever needs a row that does not exist.
    """
    run = Run(
        id=run_id or new_id(),
        conversation_id=conversation_id,
        user_id=user_id,
        trigger_message_id=trigger_message_id,
        intent=intent,
        planner_tier=planner_tier,
        status=RunStatus.RUNNING,
        started_at=ensure_utc(now) or utcnow(),
    )
    session.add(run)
    await session.flush()
    return run


async def get_run(session: AsyncSession, user_id: str, run_id: str) -> Run | None:
    """One run, scoped to its owner."""
    result = await session.execute(
        select(Run).where(Run.id == run_id, Run.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def require_run(session: AsyncSession, user_id: str, run_id: str) -> Run:
    """Same, but a miss is a 404."""
    run = await get_run(session, user_id, run_id)
    if run is None:
        raise AppError(
            "NOT_FOUND", "That run does not exist.", http=404, details={"run_id": run_id}
        )
    return run


async def list_runs(
    session: AsyncSession,
    user_id: str,
    *,
    conversation_id: str | None = None,
    status: RunStatus | str | None = None,
    limit: int = 50,
    before: dt.datetime | None = None,
) -> list[Run]:
    """Newest first."""
    stmt = select(Run).where(Run.user_id == user_id)
    if conversation_id is not None:
        stmt = stmt.where(Run.conversation_id == conversation_id)
    if status is not None:
        stmt = stmt.where(Run.status == RunStatus(status))
    cutoff = ensure_utc(before)
    if cutoff is not None:
        stmt = stmt.where(Run.started_at < cutoff)
    stmt = stmt.order_by(Run.started_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def latest_run(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Run | None:
    """The most recent run in a thread — what a follow-up answers against."""
    result = await session.execute(
        select(Run)
        .where(Run.user_id == user_id, Run.conversation_id == conversation_id)
        .order_by(Run.started_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_resumable(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    older_than: dt.datetime | None = None,
    limit: int = 200,
) -> list[Run]:
    """Runs a worker should pick back up after a restart.

    Hits the partial index on ``runs(status)``. ``user_id=None`` scans every
    user and is for the orchestration sweeper only.
    """
    stmt = select(Run).where(Run.status.in_(list(LIVE_STATUSES)))
    if user_id is not None:
        stmt = stmt.where(Run.user_id == user_id)
    cutoff = ensure_utc(older_than)
    if cutoff is not None:
        stmt = stmt.where(Run.started_at < cutoff)
    stmt = stmt.order_by(Run.started_at.asc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def set_intent(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    intent: dict[str, Any],
    *,
    planner_tier: int | None = None,
) -> Run | None:
    """Record the classifier output, including the resolved window."""
    values: dict[str, Any] = {"intent": intent}
    if planner_tier is not None:
        values["planner_tier"] = planner_tier
    result = await session.execute(
        update(Run)
        .where(Run.id == run_id, Run.user_id == user_id)
        .values(**values)
        .returning(Run)
    )
    return result.scalar_one_or_none()


async def set_status(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    status: RunStatus | str,
    *,
    error: dict[str, Any] | None = None,
    finished_at: dt.datetime | None = None,
) -> Run | None:
    """Move a run's status. Terminal statuses stamp ``finished_at``.

    A run that has already finished is not moved again — a late timeout must
    not overwrite a completed run.
    """
    target = RunStatus(status)
    values: dict[str, Any] = {"status": target}
    if error is not None:
        values["error"] = error
    if target in TERMINAL_STATUSES:
        values["finished_at"] = ensure_utc(finished_at) or utcnow()
    elif finished_at is not None:
        values["finished_at"] = ensure_utc(finished_at)
    else:
        values["finished_at"] = None

    result = await session.execute(
        update(Run)
        .where(
            Run.id == run_id,
            Run.user_id == user_id,
            Run.status.notin_(list(TERMINAL_STATUSES)),
        )
        .values(**values)
        .returning(Run)
    )
    return result.scalar_one_or_none()


async def mark_awaiting_input(
    session: AsyncSession, user_id: str, run_id: str
) -> Run | None:
    """The run is paused on a blocking prompt."""
    return await set_status(session, user_id, run_id, RunStatus.AWAITING_INPUT)


async def mark_resumed(session: AsyncSession, user_id: str, run_id: str) -> Run | None:
    """The prompt was answered; the run is live again."""
    result = await session.execute(
        update(Run)
        .where(
            Run.id == run_id,
            Run.user_id == user_id,
            Run.status == RunStatus.AWAITING_INPUT,
        )
        .values(status=RunStatus.RUNNING, finished_at=None)
        .returning(Run)
    )
    return result.scalar_one_or_none()


async def mark_complete(
    session: AsyncSession, user_id: str, run_id: str
) -> Run | None:
    """Done, cleanly."""
    return await set_status(session, user_id, run_id, RunStatus.COMPLETE)


async def mark_failed(
    session: AsyncSession, user_id: str, run_id: str, error: dict[str, Any]
) -> Run | None:
    """Done, badly. ``error`` is the AppError shape."""
    return await set_status(session, user_id, run_id, RunStatus.FAILED, error=error)


async def set_planner_tier(
    session: AsyncSession, user_id: str, run_id: str, tier: int
) -> None:
    """1 template | 2 composed | 3 replan | 4 step_loop."""
    await session.execute(
        update(Run)
        .where(Run.id == run_id, Run.user_id == user_id)
        .values(planner_tier=tier)
    )


async def add_token_usage(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    usage: dict[str, Any],
) -> Run | None:
    """Accumulate {prompt, completion, model, usd} across the run's LLM calls.

    Merged in Python under the row lock so a second call does not lose the
    first one's numbers.
    """
    locked = await session.execute(
        select(Run).where(Run.id == run_id, Run.user_id == user_id).with_for_update()
    )
    run = locked.scalar_one_or_none()
    if run is None:
        return None

    merged: dict[str, Any] = dict(run.token_usage or {})
    for key in ("prompt", "completion", "total"):
        if key in usage:
            merged[key] = int(merged.get(key, 0)) + int(usage[key] or 0)
    if "usd" in usage:
        merged["usd"] = round(float(merged.get("usd", 0.0)) + float(usage["usd"] or 0.0), 6)
    if "model" in usage:
        models = merged.get("models") or ([merged["model"]] if merged.get("model") else [])
        if usage["model"] not in models:
            models.append(usage["model"])
        merged["models"] = models
        merged["model"] = usage["model"]
    for key, value in usage.items():
        if key not in {"prompt", "completion", "total", "usd", "model"}:
            merged[key] = value

    run.token_usage = merged
    await session.flush()
    return run


async def failed_services(
    session: AsyncSession, user_id: str, run_id: str
) -> list[str]:
    """Which services fell over this run. Derived, not stored — hits the
    expression index on ``split_part(op, '.', 1)``.

    Skipped nodes are excluded on purpose. A skip means the step never ran —
    the run paused on a question, a gate said no, or the thing it reads from
    produced nothing. Naming the skipped step's service blames a service that
    was never called: a run that stopped to ask "which meeting?" would report
    Calendar as down on every reload of the page, forever. When a dependency
    genuinely failed, *that* node names the service through its own row.
    """
    service = func.split_part(NodeExecution.op, ".", 1)
    result = await session.execute(
        select(service)
        .distinct()
        .where(
            NodeExecution.run_id == run_id,
            NodeExecution.user_id == user_id,
            NodeExecution.status.in_([NodeStatus.FAILED, NodeStatus.TIMEOUT]),
            # A step WE built wrong never blames the service it names. The
            # in-memory twin (DispatchResult.failed_services) applies the
            # same rule through _fault_of.
            func.coalesce(NodeExecution.outcome["class"].astext, "").notin_(
                ["INVALID", "VALIDATION_ERROR", "NOT_FOUND"]
            ),
        )
    )
    return list(result.scalars().all())


async def replan_rounds(session: AsyncSession, user_id: str, run_id: str) -> int:
    """``max(round)`` over the run's nodes. Derived, not stored."""
    result = await session.execute(
        select(func.coalesce(func.max(NodeExecution.round), 0)).where(
            NodeExecution.run_id == run_id, NodeExecution.user_id == user_id
        )
    )
    return int(result.scalar_one())


__all__ = [
    "TERMINAL_STATUSES",
    "LIVE_STATUSES",
    "create_run",
    "get_run",
    "require_run",
    "list_runs",
    "latest_run",
    "list_resumable",
    "set_intent",
    "set_status",
    "mark_awaiting_input",
    "mark_resumed",
    "mark_complete",
    "mark_failed",
    "set_planner_tier",
    "add_token_usage",
    "failed_services",
    "replan_rounds",
]
