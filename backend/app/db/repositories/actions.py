"""actions — every side effect the system intends to perform.

Nothing reaches Google except through a row here, and every row points at the
prompt gating it (``requires_input_id`` is NOT NULL). ``needs_confirm`` ops
prepare a row and stop; approval is what moves it.

The unique index on ``dedupe_key`` is partial — draft, approved, running only.
Once something is done, cancelled, expired or failed, an identical request is a
legitimate new one: a resend after cancelling, the same reminder next week.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import Action, ActionStatus, PendingInput, Message
from app.db.repositories import ensure_utc, utcnow

#: Statuses the partial unique index covers.
IN_FLIGHT_STATUSES = frozenset(
    {ActionStatus.DRAFT, ActionStatus.APPROVED, ActionStatus.RUNNING}
)

#: Statuses an action can never leave.
TERMINAL_STATUSES = frozenset(
    {
        ActionStatus.DONE,
        ActionStatus.FAILED,
        ActionStatus.CANCELLED,
        ActionStatus.EXPIRED,
    }
)

#: The only moves allowed.
ALLOWED_TRANSITIONS: dict[ActionStatus, frozenset[ActionStatus]] = {
    ActionStatus.DRAFT: frozenset(
        {ActionStatus.APPROVED, ActionStatus.CANCELLED, ActionStatus.EXPIRED}
    ),
    ActionStatus.APPROVED: frozenset(
        {ActionStatus.RUNNING, ActionStatus.CANCELLED, ActionStatus.EXPIRED}
    ),
    ActionStatus.RUNNING: frozenset(
        {ActionStatus.DONE, ActionStatus.FAILED, ActionStatus.APPROVED}
    ),
    ActionStatus.DONE: frozenset(),
    ActionStatus.FAILED: frozenset({ActionStatus.APPROVED}),
    ActionStatus.CANCELLED: frozenset(),
    ActionStatus.EXPIRED: frozenset(),
}


async def find_in_flight(
    session: AsyncSession, user_id: str, dedupe_key: uuid.UUID
) -> Action | None:
    """The draft/approved/running action with this fingerprint, if one exists."""
    result = await session.execute(
        select(Action).where(
            Action.user_id == user_id,
            Action.dedupe_key == dedupe_key,
            Action.status.in_(list(IN_FLIGHT_STATUSES)),
        )
    )
    return result.scalar_one_or_none()


async def create_action(
    session: AsyncSession,
    user_id: str,
    message_id: str,
    *,
    requires_input_id: str,
    op: str,
    payload: dict[str, Any],
    dedupe_key: uuid.UUID,
    node_execution_id: str | None = None,
    external_ref: str | None = None,
    action_id: str | None = None,
) -> Action:
    """Prepare a write. It does not execute — that needs approval.

    If an identical action is already in flight, that row is returned instead of
    a second one being made. The savepoint is there so losing the race does not
    take the assistant message down with it.
    """
    existing = await find_in_flight(session, user_id, dedupe_key)
    if existing is not None:
        return existing

    now = utcnow()
    action = Action(
        id=action_id or new_id(),
        message_id=message_id,
        user_id=user_id,
        node_execution_id=node_execution_id,
        requires_input_id=requires_input_id,
        op=op,
        payload=payload,
        revisions=[],
        dedupe_key=dedupe_key,
        status=ActionStatus.DRAFT,
        external_ref=external_ref,
        attempts=0,
        created_at=now,
        updated_at=now,
    )
    try:
        async with session.begin_nested():
            session.add(action)
            await session.flush()
    except IntegrityError:
        session.expunge(action)
        raced = await find_in_flight(session, user_id, dedupe_key)
        if raced is None:
            raise
        return raced
    return action


async def get_action(
    session: AsyncSession, user_id: str, action_id: str
) -> Action | None:
    """One action, scoped to its owner."""
    result = await session.execute(
        select(Action).where(Action.id == action_id, Action.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def require_action(session: AsyncSession, user_id: str, action_id: str) -> Action:
    """Same, but a miss is a 404."""
    action = await get_action(session, user_id, action_id)
    if action is None:
        raise AppError(
            "NOT_FOUND",
            "That action does not exist.",
            http=404,
            details={"action_id": action_id},
        )
    return action


async def list_actions(
    session: AsyncSession,
    user_id: str,
    *,
    status: ActionStatus | str | None = None,
    message_id: str | None = None,
    requires_input_id: str | None = None,
    op: str | None = None,
    conversation_id: str | None = None,
    limit: int = 50,
) -> list[Action]:
    """Actions for this user, newest first.

    ``conversation_id`` narrows to writes staged in one conversation — the
    scope a planner must see. A draft staged in another chat is not "already
    staged" here, and telling a planner about it makes it say so.
    """
    stmt = select(Action).where(Action.user_id == user_id)
    if conversation_id is not None:
        stmt = stmt.join(Message, Message.id == Action.message_id).where(
            Message.conversation_id == conversation_id
        )
    if status is not None:
        stmt = stmt.where(Action.status == ActionStatus(status))
    if message_id is not None:
        stmt = stmt.where(Action.message_id == message_id)
    if requires_input_id is not None:
        stmt = stmt.where(Action.requires_input_id == requires_input_id)
    if op is not None:
        stmt = stmt.where(Action.op == op)
    stmt = stmt.order_by(Action.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_for_message(
    session: AsyncSession, user_id: str, message_id: str
) -> list[Action]:
    """The actions a message's content blocks point at."""
    return await list_actions(session, user_id, message_id=message_id, limit=100)


async def list_for_prompt(
    session: AsyncSession, user_id: str, input_id: str
) -> list[Action]:
    """Everything one card is gating."""
    return await list_actions(session, user_id, requires_input_id=input_id, limit=100)


async def expiry_of(
    session: AsyncSession, user_id: str, action_id: str
) -> dt.datetime | None:
    """When this action stops being approvable. Derived, not stored — it is the
    gating prompt's ``expires_at``."""
    result = await session.execute(
        select(PendingInput.expires_at)
        .join(Action, Action.requires_input_id == PendingInput.id)
        .where(Action.id == action_id, Action.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def revise_action(
    session: AsyncSession,
    user_id: str,
    action_id: str,
    patch: dict[str, Any],
    *,
    replace: bool = False,
) -> Action:
    """"Edit" on a confirm card.

    The old payload is pushed onto ``revisions`` so the trail of what was
    proposed survives. Only a draft can be revised — once approved, the payload
    is what will execute.
    """
    locked = await session.execute(
        select(Action)
        .where(Action.id == action_id, Action.user_id == user_id)
        .with_for_update()
    )
    action = locked.scalar_one_or_none()
    if action is None:
        raise AppError(
            "NOT_FOUND",
            "That action does not exist.",
            http=404,
            details={"action_id": action_id},
        )
    if action.status != ActionStatus.DRAFT:
        raise AppError(
            "PROMPT_NOT_PENDING",
            "That action can no longer be edited.",
            http=409,
            details={"action_id": action_id, "status": action.status.value},
        )

    now = utcnow()
    action.revisions = list(action.revisions or []) + [
        {"payload": action.payload, "replaced_at": now.isoformat()}
    ]
    action.payload = dict(patch) if replace else {**action.payload, **patch}
    action.updated_at = now
    await session.flush()
    return action


async def transition(
    session: AsyncSession,
    user_id: str,
    action_id: str,
    to_status: ActionStatus | str,
    *,
    expected: ActionStatus | str | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    external_ref: str | None = None,
    job_id: str | None = None,
    bump_attempts: bool = False,
    executed_at: dt.datetime | None = None,
) -> Action:
    """Move an action, guarded.

    The expected status goes into the WHERE clause, so two workers cannot both
    take the same action. A move the state machine does not allow raises rather
    than writing.
    """
    target = ActionStatus(to_status)
    current = await require_action(session, user_id, action_id)
    source = ActionStatus(expected) if expected is not None else current.status

    if target not in ALLOWED_TRANSITIONS.get(source, frozenset()):
        raise AppError(
            "PROMPT_NOT_PENDING",
            f"An action cannot go from {source.value} to {target.value}.",
            http=409,
            details={
                "action_id": action_id,
                "status": current.status.value,
                "requested": target.value,
            },
        )

    now = utcnow()
    values: dict[str, Any] = {"status": target, "updated_at": now}
    if result is not None:
        values["result"] = result
    if error is not None:
        values["error"] = error
    if external_ref is not None:
        values["external_ref"] = external_ref
    if job_id is not None:
        values["job_id"] = job_id
    if bump_attempts:
        values["attempts"] = Action.attempts + 1
    if target in {ActionStatus.DONE, ActionStatus.FAILED}:
        values["executed_at"] = ensure_utc(executed_at) or now

    row = await session.execute(
        update(Action)
        .where(
            Action.id == action_id,
            Action.user_id == user_id,
            Action.status == source,
        )
        .values(**values)
        .returning(Action)
    )
    moved = row.scalar_one_or_none()
    if moved is None:
        fresh = await require_action(session, user_id, action_id)
        raise AppError(
            "PROMPT_NOT_PENDING",
            "That action has already moved on.",
            http=409,
            details={"action_id": action_id, "status": fresh.status.value},
        )
    await session.refresh(moved)
    return moved


async def approve_action(session: AsyncSession, user_id: str, action_id: str) -> Action:
    """The person said yes. This is the only thing that unlocks execution, and
    it costs zero LLM calls."""
    return await transition(
        session, user_id, action_id, ActionStatus.APPROVED, expected=ActionStatus.DRAFT
    )


async def claim_action(
    session: AsyncSession, user_id: str, action_id: str, *, job_id: str | None = None
) -> Action:
    """A worker takes an approved action. Guarded, so only one worker wins."""
    return await transition(
        session,
        user_id,
        action_id,
        ActionStatus.RUNNING,
        expected=ActionStatus.APPROVED,
        job_id=job_id,
        bump_attempts=True,
    )


async def complete_action(
    session: AsyncSession,
    user_id: str,
    action_id: str,
    result: dict[str, Any],
    *,
    external_ref: str | None = None,
) -> Action:
    """It happened. Whatever Google gave back goes in ``result``."""
    return await transition(
        session,
        user_id,
        action_id,
        ActionStatus.DONE,
        expected=ActionStatus.RUNNING,
        result=result,
        external_ref=external_ref,
    )


async def fail_action(
    session: AsyncSession, user_id: str, action_id: str, error: dict[str, Any]
) -> Action:
    """It did not happen. A retry re-approves rather than reusing this row's
    running state."""
    return await transition(
        session,
        user_id,
        action_id,
        ActionStatus.FAILED,
        expected=ActionStatus.RUNNING,
        error=error,
    )


async def cancel_action(
    session: AsyncSession,
    user_id: str,
    action_id: str,
    *,
    reason: str = "user_declined",
) -> Action:
    """"Not now". The dedupe key frees up, so asking again later works."""
    current = await require_action(session, user_id, action_id)
    return await transition(
        session,
        user_id,
        action_id,
        ActionStatus.CANCELLED,
        expected=current.status,
        error={"reason": reason},
    )


async def set_external_ref(
    session: AsyncSession, user_id: str, action_id: str, external_ref: str
) -> None:
    """The Gmail draft we created up front, so the preview and the send are the
    same object."""
    await session.execute(
        update(Action)
        .where(Action.id == action_id, Action.user_id == user_id)
        .values(external_ref=external_ref, updated_at=utcnow())
    )


async def cancel_for_prompt(
    session: AsyncSession,
    user_id: str,
    input_id: str,
    *,
    status: ActionStatus = ActionStatus.CANCELLED,
    reason: str = "prompt_cancelled",
) -> int:
    """Close out every action a card was gating when the card is declined,
    superseded or expired."""
    result = await session.execute(
        update(Action)
        .where(
            Action.user_id == user_id,
            Action.requires_input_id == input_id,
            Action.status.in_([ActionStatus.DRAFT, ActionStatus.APPROVED]),
        )
        .values(status=status, error={"reason": reason}, updated_at=utcnow())
    )
    return int(result.rowcount or 0)


async def expire_actions(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    now: dt.datetime | None = None,
    limit: int = 1000,
) -> int:
    """Expire drafts whose gating card has run out of time.

    ``user_id=None`` sweeps every user — for ``maintenance.expire_prompts``
    only.
    """
    at = ensure_utc(now) or utcnow()
    stale = (
        select(Action.id)
        .join(PendingInput, PendingInput.id == Action.requires_input_id)
        .where(
            Action.status.in_([ActionStatus.DRAFT, ActionStatus.APPROVED]),
            PendingInput.expires_at <= at,
        )
        .limit(limit)
    )
    if user_id is not None:
        stale = stale.where(Action.user_id == user_id)

    result = await session.execute(
        update(Action)
        .where(Action.id.in_(stale))
        .values(
            status=ActionStatus.EXPIRED,
            error={"reason": "prompt_expired"},
            updated_at=at,
        )
    )
    return int(result.rowcount or 0)


async def count_by_status(session: AsyncSession, user_id: str) -> dict[str, int]:
    """``{status: count}`` for this user — what the header badge reads."""
    result = await session.execute(
        select(Action.status, func.count())
        .where(Action.user_id == user_id)
        .group_by(Action.status)
    )
    return {status.value: int(count) for status, count in result.all()}


__all__ = [
    "IN_FLIGHT_STATUSES",
    "TERMINAL_STATUSES",
    "ALLOWED_TRANSITIONS",
    "find_in_flight",
    "create_action",
    "get_action",
    "require_action",
    "list_actions",
    "list_for_message",
    "list_for_prompt",
    "expiry_of",
    "revise_action",
    "transition",
    "approve_action",
    "claim_action",
    "complete_action",
    "fail_action",
    "cancel_action",
    "set_external_ref",
    "cancel_for_prompt",
    "expire_actions",
    "count_by_status",
]
