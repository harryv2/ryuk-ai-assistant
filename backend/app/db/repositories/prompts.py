"""pending_inputs — everything the system needs from the person.

Which John, send this?, which week, pick the attachments: one mechanism for all
of it. Rows are never deleted, only moved through statuses, which is what lets a
reopened chat show the card in its answered state instead of a frozen snapshot.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import InputKind, InputStatus, NodeExecution, PendingInput, Run
from app.db.repositories import ensure_utc, utcnow

#: How long a card stays answerable if the caller does not say.
DEFAULT_TTL = dt.timedelta(hours=24)

#: Statuses a prompt can never leave.
TERMINAL_STATUSES = frozenset(
    {
        InputStatus.ANSWERED,
        InputStatus.CANCELLED,
        InputStatus.EXPIRED,
        InputStatus.SUPERSEDED,
    }
)


async def create_prompt(
    session: AsyncSession,
    user_id: str,
    run_id: str,
    message_id: str,
    *,
    kind: InputKind | str,
    prompt: dict[str, Any],
    value_schema: dict[str, Any],
    options: Sequence[dict[str, Any]] | None = None,
    blocking: bool = True,
    node_execution_id: str | None = None,
    conversation_id: str | None = None,
    op: str | None = None,
    expires_at: dt.datetime | None = None,
    ttl: dt.timedelta | None = None,
    prompt_id: str | None = None,
    supersede: bool = True,
) -> PendingInput:
    """Raise a card.

    ``blocking=True`` pauses the run; answering resumes it. ``blocking=False``
    means the run finished and this is waiting on a yes.

    Supersede rule: a new prompt in the same conversation with the same kind and
    the same originating op marks any still-pending twin ``superseded``, in this
    same transaction.
    """
    now = utcnow()
    expiry = ensure_utc(expires_at) or (now + (ttl or DEFAULT_TTL))

    row = PendingInput(
        id=prompt_id or new_id(),
        run_id=run_id,
        message_id=message_id,
        user_id=user_id,
        node_execution_id=node_execution_id,
        kind=InputKind(kind),
        blocking=blocking,
        prompt=prompt,
        value_schema=value_schema,
        options=list(options) if options is not None else None,
        status=InputStatus.PENDING,
        expires_at=expiry,
        created_at=now,
    )
    session.add(row)
    await session.flush()

    if supersede:
        origin_op = op
        if origin_op is None and node_execution_id is not None:
            found = await session.execute(
                select(NodeExecution.op).where(NodeExecution.id == node_execution_id)
            )
            origin_op = found.scalar_one_or_none()
        if origin_op is not None:
            target_conversation = conversation_id
            if target_conversation is None:
                found_conv = await session.execute(
                    select(Run.conversation_id).where(Run.id == run_id)
                )
                target_conversation = found_conv.scalar_one_or_none()
            if target_conversation is not None:
                await supersede_matching(
                    session,
                    user_id,
                    target_conversation,
                    kind=row.kind,
                    op=origin_op,
                    keep_id=row.id,
                )

    return row


async def supersede_matching(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    kind: InputKind | str,
    op: str,
    keep_id: str | None = None,
) -> int:
    """Mark still-pending prompts of the same (kind, op) in this conversation
    ``superseded``. Returns how many were closed."""
    runs_in_conversation = select(Run.id).where(
        Run.conversation_id == conversation_id, Run.user_id == user_id
    )
    nodes_with_op = select(NodeExecution.id).where(
        NodeExecution.user_id == user_id,
        NodeExecution.conversation_id == conversation_id,
        NodeExecution.op == op,
    )
    stmt = update(PendingInput).where(
        PendingInput.user_id == user_id,
        PendingInput.status == InputStatus.PENDING,
        PendingInput.kind == InputKind(kind),
        PendingInput.run_id.in_(runs_in_conversation),
        PendingInput.node_execution_id.in_(nodes_with_op),
    )
    if keep_id is not None:
        stmt = stmt.where(PendingInput.id != keep_id)
    result = await session.execute(stmt.values(status=InputStatus.SUPERSEDED))
    return int(result.rowcount or 0)


async def get_prompt(
    session: AsyncSession, user_id: str, prompt_id: str
) -> PendingInput | None:
    """One card, scoped to its owner."""
    result = await session.execute(
        select(PendingInput).where(
            PendingInput.id == prompt_id, PendingInput.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def require_prompt(
    session: AsyncSession, user_id: str, prompt_id: str
) -> PendingInput:
    """Same, but a miss is a 404."""
    row = await get_prompt(session, user_id, prompt_id)
    if row is None:
        raise AppError(
            "NOT_FOUND",
            "That prompt does not exist.",
            http=404,
            details={"input_id": prompt_id},
        )
    return row


async def list_prompts(
    session: AsyncSession,
    user_id: str,
    *,
    status: InputStatus | str | None = None,
    run_id: str | None = None,
    message_id: str | None = None,
    conversation_id: str | None = None,
    limit: int = 50,
) -> list[PendingInput]:
    """Cards for this user, newest first."""
    stmt = select(PendingInput).where(PendingInput.user_id == user_id)
    if status is not None:
        stmt = stmt.where(PendingInput.status == InputStatus(status))
    if run_id is not None:
        stmt = stmt.where(PendingInput.run_id == run_id)
    if message_id is not None:
        stmt = stmt.where(PendingInput.message_id == message_id)
    if conversation_id is not None:
        stmt = stmt.where(
            PendingInput.run_id.in_(
                select(Run.id).where(
                    Run.conversation_id == conversation_id, Run.user_id == user_id
                )
            )
        )
    stmt = stmt.order_by(PendingInput.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_for_message(
    session: AsyncSession, user_id: str, message_id: str
) -> list[PendingInput]:
    """The cards a message's content blocks point at."""
    result = await session.execute(
        select(PendingInput)
        .where(PendingInput.message_id == message_id, PendingInput.user_id == user_id)
        .order_by(PendingInput.created_at.asc())
    )
    return list(result.scalars().all())


async def get_blocking_for_run(
    session: AsyncSession, user_id: str, run_id: str
) -> PendingInput | None:
    """The card a paused run is waiting on. ``run_id`` is denormalised onto this
    table precisely because resume is the critical path."""
    result = await session.execute(
        select(PendingInput)
        .where(
            PendingInput.run_id == run_id,
            PendingInput.user_id == user_id,
            PendingInput.status == InputStatus.PENDING,
            PendingInput.blocking.is_(True),
        )
        .order_by(PendingInput.created_at.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def answer_prompt(
    session: AsyncSession,
    user_id: str,
    prompt_id: str,
    response: Any,
    *,
    now: dt.datetime | None = None,
) -> PendingInput:
    """Record the person's answer.

    The guard is in the WHERE clause, so two clicks on the same card cannot both
    win. Validating ``response`` against ``value_schema`` is the caller's job —
    that schema is the validation authority, and it lives above this layer.
    """
    at = ensure_utc(now) or utcnow()
    result = await session.execute(
        update(PendingInput)
        .where(
            PendingInput.id == prompt_id,
            PendingInput.user_id == user_id,
            PendingInput.status == InputStatus.PENDING,
            PendingInput.expires_at > at,
        )
        .values(status=InputStatus.ANSWERED, response=response, answered_at=at)
        .returning(PendingInput)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row

    existing = await require_prompt(session, user_id, prompt_id)
    if existing.status == InputStatus.PENDING:
        # Still pending means it timed out rather than being taken by someone else.
        await session.execute(
            update(PendingInput)
            .where(PendingInput.id == prompt_id, PendingInput.user_id == user_id)
            .values(status=InputStatus.EXPIRED)
        )
        raise AppError(
            "PROMPT_NOT_PENDING",
            "That card has expired.",
            http=409,
            details={"input_id": prompt_id, "status": InputStatus.EXPIRED.value},
        )
    raise AppError(
        "PROMPT_NOT_PENDING",
        "That card has already been dealt with.",
        http=409,
        details={"input_id": prompt_id, "status": existing.status.value},
    )


async def cancel_prompt(
    session: AsyncSession, user_id: str, prompt_id: str
) -> PendingInput:
    """"Not now"."""
    result = await session.execute(
        update(PendingInput)
        .where(
            PendingInput.id == prompt_id,
            PendingInput.user_id == user_id,
            PendingInput.status == InputStatus.PENDING,
        )
        .values(status=InputStatus.CANCELLED, answered_at=utcnow())
        .returning(PendingInput)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return row

    existing = await require_prompt(session, user_id, prompt_id)
    raise AppError(
        "PROMPT_NOT_PENDING",
        "That card has already been dealt with.",
        http=409,
        details={"input_id": prompt_id, "status": existing.status.value},
    )


async def expire_prompts(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    now: dt.datetime | None = None,
    limit: int = 1000,
) -> list[PendingInput]:
    """Close out cards nobody answered in time.

    ``user_id=None`` sweeps every user — for ``maintenance.expire_prompts``
    only. Returns the rows it closed so the caller can expire the actions those
    prompts were gating.
    """
    at = ensure_utc(now) or utcnow()
    due = select(PendingInput.id).where(
        PendingInput.status == InputStatus.PENDING, PendingInput.expires_at <= at
    )
    if user_id is not None:
        due = due.where(PendingInput.user_id == user_id)
    due = due.limit(limit)

    result = await session.execute(
        update(PendingInput)
        .where(PendingInput.id.in_(due))
        .values(status=InputStatus.EXPIRED)
        .returning(PendingInput)
    )
    return list(result.scalars().all())


async def count_pending(session: AsyncSession, user_id: str) -> int:
    """How many cards are waiting on this person."""
    result = await session.execute(
        select(func.count())
        .select_from(PendingInput)
        .where(
            PendingInput.user_id == user_id, PendingInput.status == InputStatus.PENDING
        )
    )
    return int(result.scalar_one())


__all__ = [
    "DEFAULT_TTL",
    "TERMINAL_STATUSES",
    "create_prompt",
    "supersede_matching",
    "get_prompt",
    "require_prompt",
    "list_prompts",
    "list_for_message",
    "get_blocking_for_run",
    "answer_prompt",
    "cancel_prompt",
    "expire_prompts",
    "count_pending",
]
