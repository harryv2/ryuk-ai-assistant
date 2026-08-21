"""One turn -> the wire body.

``query.py`` and ``prompts.py`` both answer with the shape ``docs/API.md``
describes for ``POST /api/v1/query``, so it is built in one place. A drift
between the two shows up in the UI as a card that never appears.

The cards and actions are always **read back from the database** rather than
taken off the orchestrator's result. That is not belt and braces: a confirm
card can be superseded by a newer run between the write and the read, and the
answer a person is looking at has to show the row's current status, not the
status it had a moment ago. It is also the same code path a reopened
conversation takes, which is how the two stay identical.

The row-to-dict work lives in :mod:`app.api.v1.schemas`; this module only
decides which rows to fetch.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas import (
    action_dto,
    entity_dto,
    prompt_dto,
    run_body,
    step_dto,
)
from app.core import logging as applog
from app.db.models import Action, InputStatus, Message, MessageRole, PendingInput, Run
from app.db.repositories import entities as entity_repo
from app.db.repositories import runs as run_repo
from app.db.repositories import steps as step_repo


async def _prompts_for(
    session: AsyncSession, user_id: str, *, ids: list[str], run_id: str | None
) -> list[PendingInput]:
    """The cards this turn raised, in their current state.

    Falls back to the run's own open cards when the orchestrator did not report
    ids. A paused run whose card is missing from the response is a dead end on
    screen: the question was asked and there is no way to answer it.
    """
    if ids:
        stmt = select(PendingInput).where(
            PendingInput.user_id == user_id, PendingInput.id.in_(ids)
        )
    elif run_id:
        stmt = select(PendingInput).where(
            PendingInput.user_id == user_id,
            PendingInput.run_id == run_id,
            PendingInput.status == InputStatus.PENDING,
        )
    else:
        return []
    rows = (await session.execute(stmt.order_by(PendingInput.created_at.asc()))).scalars().all()
    return list(rows)


async def _actions_for(
    session: AsyncSession, user_id: str, *, ids: list[str], gate_ids: list[str]
) -> list[Action]:
    """The writes this turn staged. An action nobody can see is one nobody can
    approve, so a missing id list falls back to whatever the cards gate."""
    if ids:
        stmt = select(Action).where(Action.user_id == user_id, Action.id.in_(ids))
    elif gate_ids:
        stmt = select(Action).where(
            Action.user_id == user_id, Action.requires_input_id.in_(gate_ids)
        )
    else:
        return []
    rows = (await session.execute(stmt.order_by(Action.created_at.asc()))).scalars().all()
    return list(rows)


log = applog.get_logger(__name__)


async def hydrate(session: AsyncSession, user_id: str, outcome: Any) -> dict[str, Any]:
    """The response body for one turn."""
    run_id = getattr(outcome, "run_id", None) or None
    conversation_id = getattr(outcome, "conversation_id", None) or None

    prompt_rows = await _prompts_for(
        session,
        user_id,
        ids=[str(i) for i in (getattr(outcome, "input_ids", None) or []) if i],
        run_id=run_id,
    )
    action_rows = await _actions_for(
        session,
        user_id,
        ids=[str(i) for i in (getattr(outcome, "action_ids", None) or []) if i],
        gate_ids=[p.id for p in prompt_rows],
    )

    expiry = {p.id: p.expires_at for p in prompt_rows}
    actions = [
        action_dto(a, expires_at=expiry.get(a.requires_input_id)) for a in action_rows
    ]
    by_gate: dict[str, dict[str, Any]] = {}
    for shaped in actions:
        gate = shaped.get("requires_input_id")
        if gate:
            by_gate.setdefault(gate, shaped)
    prompts = [
        prompt_dto(p, conversation_id=conversation_id, action=by_gate.get(p.id))
        for p in prompt_rows
    ]

    steps = [step_dto(s) for s in (getattr(outcome, "steps", None) or [])]
    if not steps and run_id:
        rows = await step_repo.list_steps(session, user_id, run_id)
        steps = [step_dto(row) for row in rows]

    entities = [entity_dto(e) for e in (getattr(outcome, "entities", None) or [])]
    if not entities and conversation_id:
        rows = await entity_repo.list_entities(session, user_id, conversation_id, limit=20)
        entities = [entity_dto(row) for row in rows]

    return run_body(
        outcome,
        pending_inputs=prompts,
        actions=actions,
        steps=steps,
        entities=entities,
    )


class _StoredRun:
    """A settled run read back from Postgres, shaped like a turn's result.

    Used by the idempotency replay on ``POST /query`` and by the timeout
    recovery path. The durable record is the authority; the in-memory result is
    only ever a faster copy of it.
    """

    def __init__(self, run: Run, message: Any) -> None:
        self.run_id = run.id
        self.conversation_id = run.conversation_id
        self.message_id = getattr(message, "id", "") or ""
        self.status = run.status.value if hasattr(run.status, "value") else str(run.status)
        self.intent = run.intent
        self.planner_tier = run.planner_tier
        self.usage = run.token_usage or {}
        self.timings = {}
        self.answer_style = (run.intent or {}).get("answer_style") if run.intent else None
        self.route = "replay"
        self.content = list(getattr(message, "content", None) or [])
        self.answer = ""
        self.degraded: list[Any] = []
        self.steps: list[Any] = []
        self.entities: list[Any] = []
        self.input_ids: list[str] = []
        self.action_ids: list[str] = []


async def hydrate_run(
    session: AsyncSession, user_id: str, run_id: str
) -> dict[str, Any] | None:
    """The body for a run that has already happened, straight from the tables."""
    run = await run_repo.get_run(session, user_id, run_id)
    if run is None:
        return None
    message = (
        await session.execute(
            select(Message)
            .where(
                Message.run_id == run_id,
                Message.user_id == user_id,
                Message.role == MessageRole.ASSISTANT,
            )
            .order_by(Message.seq.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    outcome = _StoredRun(run, message)
    degraded = await run_repo.failed_services(session, user_id, run_id)
    outcome.degraded = list(degraded or [])
    return await hydrate(session, user_id, outcome)


__all__ = ["hydrate", "hydrate_run"]


async def hydrate_fresh(user_id: str, outcome: Any) -> dict[str, Any]:
    """:func:`hydrate`, on a session of its own, and never fatal.

    By the time this runs the work has already committed: the prompt is
    answered, the plan resumed, the actions are queued. Everything left is
    turning that into JSON.

    Two things make reading it back fragile. The request's own session has been
    expired by the commit, so touching a loaded object triggers a lazy refresh —
    which on an async session raises rather than quietly issuing a query. And a
    brand-new session has to check a connection out of the pool, which is itself
    IO and can raise from outside the greenlet bridge.

    Neither is a reason to tell the caller their request failed. If the read-back
    cannot be done, answer from what the runner already handed us in memory and
    say so in the payload, rather than returning a 500 for work that succeeded.
    """
    from app.db.session import get_sessionmaker

    try:
        maker = get_sessionmaker()
        async with maker() as fresh:
            return await hydrate(fresh, user_id, outcome)
    except BaseExceptionGroup as group:  # a TaskGroup wraps its failures
        # asyncpg's pool raises from inside a task group, so the failure arrives
        # as a BaseExceptionGroup — which `except Exception` does NOT catch.
        # Missing that is how a guarded call still 500s.
        exc = group.exceptions[0] if group.exceptions else group
        log.warning(
            "hydrate.read_back_failed",
            cause=exc.__class__.__name__,
            run_id=getattr(outcome, "run_id", None),
        )
        return _from_outcome(outcome, degraded_note=exc.__class__.__name__)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        log.warning(
            "hydrate.read_back_failed",
            cause=exc.__class__.__name__,
            run_id=getattr(outcome, "run_id", None),
        )
        return _from_outcome(outcome, degraded_note=exc.__class__.__name__)


def _from_outcome(outcome: Any, *, degraded_note: str | None = None) -> dict[str, Any]:
    """The response built only from what is already in memory.

    Cards and actions come back as the ids the runner reported. The client
    refetches the conversation to render them, which is the same path a reopened
    thread takes — so this degrades to one extra request, not to a broken screen.
    """
    payload = {
        "conversation_id": getattr(outcome, "conversation_id", None),
        "message_id": getattr(outcome, "message_id", None),
        "run_id": getattr(outcome, "run_id", None),
        "status": getattr(outcome, "status", "complete"),
        "answer": getattr(outcome, "text", "") or "",
        "text": getattr(outcome, "text", "") or "",
        "content": list(getattr(outcome, "blocks", []) or []),
        "intent": getattr(outcome, "intent", None),
        "planner_tier": getattr(outcome, "planner_tier", None),
        "steps": list(getattr(outcome, "steps", []) or []),
        "actions": [],
        "pending_inputs": [],
        "action_ids": list(getattr(outcome, "action_ids", []) or []),
        "input_ids": list(getattr(outcome, "input_ids", []) or []),
        "degraded": list(getattr(outcome, "degraded_detail", []) or []),
        "degraded_services": list(getattr(outcome, "degraded", []) or []),
        "timings": {},
        "usage": getattr(outcome, "usage", {}) or {},
        "needs_refetch": True,
    }
    if degraded_note:
        payload["read_back"] = degraded_note
    return payload


__all__ = ["hydrate", "hydrate_fresh", "hydrate_run"]
