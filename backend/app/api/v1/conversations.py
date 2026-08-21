"""Reading a thread back.

This is the durable record and the fallback for a dropped SSE stream: whatever
the stream cost you, this has. Postgres is the source of truth; the event
channel is only ever an accelerator.

Cards and actions are stored **by reference**, never by value, which is what
makes a reopened conversation honest. An answered choice shows the pick, a sent
email shows Sent, one nobody answered shows expired — because the block carries
an id and the id is joined at read time. A ref with nothing behind it is a row
that is gone: the block is dropped and logged, never rendered as an empty box.

The step trace is off by default. A long thread's ``node_executions`` are large
and almost nobody opens the panel, so ``GET /runs/{id}/steps`` loads them when
somebody does.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas import (
    action_dto,
    conversation_summary_dto,
    decode_cursor,
    encode_cursor,
    entity_dto,
    iso,
    message_dto,
    prompt_dto,
    run_dto,
    step_dto,
)
from app.auth.deps import CurrentUser, SessionDep
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import (
    Action,
    Conversation,
    InputStatus,
    Message,
    MessageRole,
    PendingInput,
    Run,
)
from app.db.repositories import conversations as conv_repo
from app.db.repositories import entities as entity_repo
from app.db.repositories import runs as run_repo
from app.db.repositories import steps as step_repo

log = get_logger(__name__)
router = APIRouter(tags=["conversations"])


# --------------------------------------------------------------------------- #
# GET /conversations
# --------------------------------------------------------------------------- #


def _title_from(message: Message | None) -> str | None:
    """The first user message, flattened enough to be a title."""
    if message is None:
        return None
    for block in message.content or []:
        if not isinstance(block, dict):
            continue
        data = block.get("data") or {}
        for key in ("markdown", "text", "body"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                trimmed = " ".join(value.split())
                return trimmed if len(trimmed) <= 80 else trimmed[:79] + "…"
    return None


@router.get("/conversations")
async def list_conversations(
    session: SessionDep,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None),
    archived: bool = Query(False),
) -> dict[str, Any]:
    """The thread list, newest activity first.

    Keyset paginated on ``(last_message_at, id)`` so the partial index
    ``(user_id, last_message_at DESC) WHERE archived_at IS NULL`` serves it
    directly. Offset paging over a list that changes while you read it shows
    duplicates and skips rows; a cursor does not.
    """
    before, before_id = decode_cursor(cursor)

    stmt = select(Conversation).where(Conversation.user_id == user.id)
    stmt = (
        stmt.where(Conversation.archived_at.isnot(None))
        if archived
        else stmt.where(Conversation.archived_at.is_(None))
    )
    if before is not None:
        # The id breaks the tie when two threads share a timestamp, which is
        # what stops a row being shown twice or skipped at a page boundary.
        stmt = stmt.where(
            (Conversation.last_message_at < before)
            | ((Conversation.last_message_at == before) & (Conversation.id < (before_id or "")))
        )
    stmt = stmt.order_by(Conversation.last_message_at.desc(), Conversation.id.desc())

    rows = list((await session.execute(stmt.limit(limit + 1))).scalars().all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    ids = [row.id for row in rows]

    messages: dict[str, int] = {}
    pending: dict[str, int] = {}
    titles: dict[str, str | None] = {}
    last_runs: dict[str, Run] = {}

    if ids:
        counted = await session.execute(
            select(Message.conversation_id, func.count())
            .where(Message.conversation_id.in_(ids), Message.user_id == user.id)
            .group_by(Message.conversation_id)
        )
        messages = {cid: int(n) for cid, n in counted.all()}

        waiting = await session.execute(
            select(Run.conversation_id, func.count())
            .select_from(PendingInput)
            .join(Run, Run.id == PendingInput.run_id)
            .where(
                Run.conversation_id.in_(ids),
                PendingInput.user_id == user.id,
                PendingInput.status == InputStatus.PENDING,
            )
            .group_by(Run.conversation_id)
        )
        pending = {cid: int(n) for cid, n in waiting.all()}

        recent = await session.execute(
            select(Run)
            .where(Run.conversation_id.in_(ids), Run.user_id == user.id)
            .order_by(Run.conversation_id, Run.started_at.desc())
        )
        for run in recent.scalars().all():
            last_runs.setdefault(run.conversation_id, run)

        for row in rows:
            if row.title is None:
                first = await conv_repo.first_user_message(session, user.id, row.id)
                titles[row.id] = _title_from(first)

    items = [
        conversation_summary_dto(
            row,
            fallback_title=titles.get(row.id),
            message_count=messages.get(row.id, 0),
            pending_input_count=pending.get(row.id, 0),
            last_run=last_runs.get(row.id),
        )
        for row in rows
    ]
    next_cursor = (
        encode_cursor(rows[-1].last_message_at, rows[-1].id) if rows and has_more else None
    )
    return {"items": items, "next_cursor": next_cursor, "has_more": has_more}


# --------------------------------------------------------------------------- #
# PATCH / DELETE /conversations/{id}
# --------------------------------------------------------------------------- #


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    session: SessionDep,
    user: CurrentUser,
    title: str = Body(..., embed=True, max_length=200),
) -> dict[str, Any]:
    """Give a thread a name of your own.

    An empty title clears it, which puts the thread back on the derived name
    from its first message rather than leaving a blank row in the list.
    """
    await conv_repo.require_conversation(session, user.id, conversation_id)
    cleaned = title.strip()
    await conv_repo.set_title(session, user.id, conversation_id, cleaned or None)
    await session.commit()
    return {"id": conversation_id, "title": cleaned or None}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    session: SessionDep,
    user: CurrentUser,
) -> dict[str, Any]:
    """Take a thread off the list.

    Archived, not erased. The runs, actions and audit rows underneath it are
    the record of things that actually happened to somebody's mailbox, and
    those should outlive a tidy-up of the sidebar.
    """
    await conv_repo.require_conversation(session, user.id, conversation_id)
    await conv_repo.archive_conversation(session, user.id, conversation_id)
    await session.commit()
    return {"id": conversation_id, "archived": True}


# --------------------------------------------------------------------------- #
# GET /conversations/{id}
# --------------------------------------------------------------------------- #


async def _hydrate_blocks(
    content: Any,
    prompts: dict[str, dict[str, Any]],
    actions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve every ref in a message's content against the rows behind it."""
    out: list[dict[str, Any]] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind, ref = block.get("type"), block.get("ref")
        if kind == "input" and ref is not None:
            resolved = prompts.get(str(ref))
            if resolved is None:
                log.info("api.dropped_block", block="input", ref=ref)
                continue
            out.append({"type": "input", "ref": ref, "data": resolved})
            continue
        if kind == "action" and ref is not None:
            resolved = actions.get(str(ref))
            if resolved is None:
                log.info("api.dropped_block", block="action", ref=ref)
                continue
            out.append({"type": "action", "ref": ref, "data": resolved})
            continue
        out.append(block)
    return out


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    session: SessionDep,
    user: CurrentUser,
    include_trace: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    before_seq: int | None = Query(None, ge=1),
) -> dict[str, Any]:
    """The full thread, with every referenced object resolved."""
    conversation = await conv_repo.get_conversation(session, user.id, conversation_id)
    if conversation is None:
        raise AppError(
            "NOT_FOUND",
            "No conversation with that id.",
            details={"conversation_id": conversation_id},
        )

    stmt = select(Message).where(
        Message.conversation_id == conversation_id,
        Message.user_id == user.id,
        Message.hidden.is_(False),
    )
    if before_seq is not None:
        stmt = stmt.where(Message.seq < before_seq)
    # Newest first for the limit, then flipped: "the most recent 50" is the page
    # a client wants, and it has to arrive in reading order.
    newest = list(
        (await session.execute(stmt.order_by(Message.seq.desc()).limit(limit + 1)))
        .scalars()
        .all()
    )
    has_more = len(newest) > limit
    rows = list(reversed(newest[:limit]))

    prompt_rows = await _prompts_in(session, user.id, conversation_id)
    action_rows = await _actions_in(
        session,
        user.id,
        message_ids=[row.id for row in rows],
        input_ids=[p.id for p in prompt_rows],
    )

    expiry = {p.id: p.expires_at for p in prompt_rows}
    actions = {
        a.id: action_dto(a, expires_at=expiry.get(a.requires_input_id)) for a in action_rows
    }
    by_gate: dict[str, dict[str, Any]] = {}
    for shaped in actions.values():
        gate = shaped.get("requires_input_id")
        if gate:
            by_gate.setdefault(gate, shaped)
    prompts = {
        p.id: prompt_dto(p, conversation_id=conversation_id, action=by_gate.get(p.id))
        for p in prompt_rows
    }

    runs = await run_repo.list_runs(session, user.id, conversation_id=conversation_id, limit=100)

    messages: list[dict[str, Any]] = []
    for row in rows:
        trace = None
        if include_trace and row.role == MessageRole.ASSISTANT:
            # `message_id` is which assistant message reported the node. Without
            # it a paused run's trace would appear under both of its messages.
            steps = await step_repo.list_steps_for_message(session, user.id, row.id)
            trace = [step_dto(step) for step in steps]
        shaped = message_dto(row, trace=trace)
        shaped["content"] = await _hydrate_blocks(row.content, prompts, actions)
        messages.append(shaped)

    entities = await entity_repo.list_entities(session, user.id, conversation_id, limit=50)
    first = await conv_repo.first_user_message(session, user.id, conversation_id)

    return {
        "id": conversation.id,
        "title": conversation.title or _title_from(first) or "New conversation",
        "title_is_derived": conversation.title is None,
        "created_at": iso(conversation.created_at),
        "last_message_at": iso(conversation.last_message_at),
        "archived_at": iso(conversation.archived_at),
        "messages": messages,
        "runs": [run_dto(run) for run in runs],
        "pending_inputs": list(prompts.values()),
        "actions": list(actions.values()),
        "entities": [entity_dto(e) for e in entities],
        "has_more": has_more,
    }


async def _prompts_in(
    session: AsyncSession, user_id: str, conversation_id: str
) -> list[PendingInput]:
    """Every card this thread has raised, whatever state it ended in."""
    rows = await session.execute(
        select(PendingInput)
        .join(Run, Run.id == PendingInput.run_id)
        .where(Run.conversation_id == conversation_id, PendingInput.user_id == user_id)
        .order_by(PendingInput.created_at.asc())
    )
    return list(rows.scalars().all())


async def _actions_in(
    session: AsyncSession,
    user_id: str,
    *,
    message_ids: list[str],
    input_ids: list[str],
) -> list[Action]:
    """The writes this page has to resolve.

    Reached two ways because both are true and neither is complete on its own:
    ``message_id`` is where the block sits, and ``requires_input_id`` is the
    card that gates it. A revised action moves to a new card without moving
    message.
    """
    if not message_ids and not input_ids:
        return []
    clauses = []
    if message_ids:
        clauses.append(Action.message_id.in_(message_ids))
    if input_ids:
        clauses.append(Action.requires_input_id.in_(input_ids))
    rows = await session.execute(
        select(Action)
        .where(Action.user_id == user_id, or_(*clauses))
        .order_by(Action.created_at.asc())
    )
    return list(rows.scalars().all())


# --------------------------------------------------------------------------- #
# GET /runs/{id}/steps
# --------------------------------------------------------------------------- #


@router.get("/runs/{run_id}/steps")
async def run_steps(
    run_id: str,
    session: SessionDep,
    user: CurrentUser,
    message_id: str | None = Query(None),
) -> dict[str, Any]:
    """Trace detail, loaded when somebody expands it — not with the thread."""
    run = await run_repo.get_run(session, user.id, run_id)
    if run is None:
        raise AppError("NOT_FOUND", "No run with that id.", details={"run_id": run_id})

    rows = (
        await step_repo.list_steps_for_message(session, user.id, message_id)
        if message_id
        else await step_repo.list_steps(session, user.id, run_id)
    )
    return {
        "run_id": run_id,
        "conversation_id": run.conversation_id,
        "status": run_dto(run)["status"],
        "steps": [step_dto(row) for row in rows],
        "count": len(rows),
    }


__all__ = ["router"]
