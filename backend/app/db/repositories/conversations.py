"""Conversations and the messages inside them.

``seq`` is allocated in the same transaction as the insert. We take a row lock
on the conversation first so two concurrent turns serialise instead of racing;
the unique constraint on (conversation_id, seq) is still there as the backstop,
and ``add_message`` retries once if it fires.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import cast, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import Conversation, Message, MessageRole
from app.db.repositories import ensure_utc, utcnow

# --------------------------------------------------------------------------- #
# conversations
# --------------------------------------------------------------------------- #


async def create_conversation(
    session: AsyncSession,
    user_id: str,
    *,
    title: str | None = None,
    conversation_id: str | None = None,
    now: dt.datetime | None = None,
) -> Conversation:
    """A new thread. ``title=None`` means the display falls back to the first
    message."""
    at = ensure_utc(now) or utcnow()
    conversation = Conversation(
        id=conversation_id or new_id(),
        user_id=user_id,
        title=title,
        last_message_at=at,
        created_at=at,
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def get_conversation(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Conversation | None:
    """One conversation, scoped to its owner."""
    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def require_conversation(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Conversation:
    """Same, but a miss is a 404 rather than a None to check."""
    conversation = await get_conversation(session, user_id, conversation_id)
    if conversation is None:
        raise AppError(
            "NOT_FOUND",
            "That conversation does not exist.",
            http=404,
            details={"conversation_id": conversation_id},
        )
    return conversation


async def list_conversations(
    session: AsyncSession,
    user_id: str,
    *,
    limit: int = 30,
    before: dt.datetime | None = None,
    include_archived: bool = False,
) -> list[Conversation]:
    """Newest activity first. Keyset paginated on ``last_message_at``."""
    stmt = select(Conversation).where(Conversation.user_id == user_id)
    if not include_archived:
        stmt = stmt.where(Conversation.archived_at.is_(None))
    cutoff = ensure_utc(before)
    if cutoff is not None:
        stmt = stmt.where(Conversation.last_message_at < cutoff)
    stmt = stmt.order_by(Conversation.last_message_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def set_title(
    session: AsyncSession, user_id: str, conversation_id: str, title: str | None
) -> Conversation | None:
    """Rename. NULL puts the display back on the first message."""
    result = await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .values(title=title)
        .returning(Conversation)
    )
    return result.scalar_one_or_none()


async def archive_conversation(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Conversation | None:
    """Hide a thread. No rows are deleted."""
    result = await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .values(archived_at=utcnow())
        .returning(Conversation)
    )
    return result.scalar_one_or_none()


async def unarchive_conversation(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Conversation | None:
    """Put it back in the list."""
    result = await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .values(archived_at=None)
        .returning(Conversation)
    )
    return result.scalar_one_or_none()


async def touch_conversation(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    at: dt.datetime | None = None,
) -> None:
    """Move ``last_message_at`` forward. Never backwards — a late write for an
    older message must not reorder the list."""
    when = ensure_utc(at) or utcnow()
    await session.execute(
        update(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.last_message_at < when,
        )
        .values(last_message_at=when)
    )


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #


async def next_seq(session: AsyncSession, user_id: str, conversation_id: str) -> int:
    """Allocate the next per-conversation sequence number, inside this
    transaction.

    Locks the conversation row so concurrent turns queue up behind each other
    rather than both reading the same max.
    """
    locked = await session.execute(
        select(Conversation.id)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .with_for_update()
    )
    if locked.scalar_one_or_none() is None:
        raise AppError(
            "NOT_FOUND",
            "That conversation does not exist.",
            http=404,
            details={"conversation_id": conversation_id},
        )
    result = await session.execute(
        select(func.coalesce(func.max(Message.seq), 0) + 1).where(
            Message.conversation_id == conversation_id
        )
    )
    return int(result.scalar_one())


async def add_message(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    role: MessageRole | str,
    content: Sequence[dict[str, Any]],
    run_id: str | None = None,
    hidden: bool = False,
    message_id: str | None = None,
    seq: int | None = None,
    now: dt.datetime | None = None,
    touch: bool = True,
) -> Message:
    """Insert one message and move the conversation's sort key forward.

    Retries once on a seq collision, which is what the unique constraint is
    there to catch.
    """
    at = ensure_utc(now) or utcnow()
    attempts = 2 if seq is None else 1
    last_error: IntegrityError | None = None

    for _ in range(attempts):
        allocated = seq if seq is not None else await next_seq(
            session, user_id, conversation_id
        )
        message = Message(
            id=message_id or new_id(),
            conversation_id=conversation_id,
            user_id=user_id,
            run_id=run_id,
            seq=allocated,
            role=MessageRole(role),
            content=list(content),
            hidden=hidden,
            created_at=at,
        )
        # A savepoint, not the outer transaction: a lost race must not throw
        # away the assistant message and its prompts that we wrote alongside.
        try:
            async with session.begin_nested():
                session.add(message)
                await session.flush()
        except IntegrityError as exc:
            session.expunge(message)
            last_error = exc
            continue
        if touch and not hidden:
            await touch_conversation(session, user_id, conversation_id, at)
        return message

    raise AppError(
        "INTERNAL",
        "Could not allocate a message sequence number.",
        http=500,
        details={"conversation_id": conversation_id},
    ) from last_error


async def get_message(
    session: AsyncSession, user_id: str, message_id: str
) -> Message | None:
    """One message, scoped to its owner."""
    result = await session.execute(
        select(Message).where(Message.id == message_id, Message.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def list_messages(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    limit: int = 200,
    after_seq: int | None = None,
    include_hidden: bool = False,
) -> list[Message]:
    """The thread in order."""
    stmt = select(Message).where(
        Message.conversation_id == conversation_id, Message.user_id == user_id
    )
    if not include_hidden:
        stmt = stmt.where(Message.hidden.is_(False))
    if after_seq is not None:
        stmt = stmt.where(Message.seq > after_seq)
    stmt = stmt.order_by(Message.seq.asc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_recent_messages(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    limit: int = 12,
    include_hidden: bool = True,
) -> list[Message]:
    """The tail of the thread, oldest first — what the planner prompt gets."""
    stmt = select(Message).where(
        Message.conversation_id == conversation_id, Message.user_id == user_id
    )
    if not include_hidden:
        stmt = stmt.where(Message.hidden.is_(False))
    stmt = stmt.order_by(Message.seq.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(reversed(result.scalars().all()))


async def list_messages_for_run(
    session: AsyncSession, user_id: str, run_id: str
) -> list[Message]:
    """Every assistant message a run produced — a clarification, then the answer."""
    result = await session.execute(
        select(Message)
        .where(Message.run_id == run_id, Message.user_id == user_id)
        .order_by(Message.seq.asc())
    )
    return list(result.scalars().all())


async def set_message_run_id(
    session: AsyncSession, user_id: str, message_id: str, run_id: str
) -> None:
    """Attach a message to its run. The trigger message is written before the
    run exists, so it is linked afterwards."""
    await session.execute(
        update(Message)
        .where(Message.id == message_id, Message.user_id == user_id)
        .values(run_id=run_id)
    )


async def set_content(
    session: AsyncSession,
    user_id: str,
    message_id: str,
    content: Sequence[dict[str, Any]],
) -> Message | None:
    """Replace the block list — used when streamed prose is finalised."""
    result = await session.execute(
        update(Message)
        .where(Message.id == message_id, Message.user_id == user_id)
        .values(content=list(content))
        .returning(Message)
    )
    return result.scalar_one_or_none()


async def append_block(
    session: AsyncSession,
    user_id: str,
    message_id: str,
    block: dict[str, Any],
) -> Message | None:
    """Push one block onto the end of ``content``.

    Done in SQL so a concurrent writer cannot clobber the list.
    """
    result = await session.execute(
        update(Message)
        .where(Message.id == message_id, Message.user_id == user_id)
        .values(content=Message.content.op("||")(cast([block], JSONB)))
        .returning(Message)
    )
    return result.scalar_one_or_none()


async def set_hidden(
    session: AsyncSession, user_id: str, message_id: str, hidden: bool
) -> None:
    """Move a message in or out of the on-screen thread. It stays in LLM context
    either way."""
    await session.execute(
        update(Message)
        .where(Message.id == message_id, Message.user_id == user_id)
        .values(hidden=hidden)
    )


async def count_messages(
    session: AsyncSession, user_id: str, conversation_id: str
) -> int:
    """How many messages the thread holds."""
    result = await session.execute(
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id, Message.user_id == user_id)
    )
    return int(result.scalar_one())


async def first_user_message(
    session: AsyncSession, user_id: str, conversation_id: str
) -> Message | None:
    """The fallback display title when ``conversations.title`` is NULL."""
    result = await session.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.user_id == user_id,
            Message.role == MessageRole.USER,
        )
        .order_by(Message.seq.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


__all__ = [
    "create_conversation",
    "get_conversation",
    "require_conversation",
    "list_conversations",
    "set_title",
    "archive_conversation",
    "unarchive_conversation",
    "touch_conversation",
    "next_seq",
    "add_message",
    "get_message",
    "list_messages",
    "list_recent_messages",
    "list_messages_for_run",
    "set_message_run_id",
    "set_content",
    "append_block",
    "set_hidden",
    "count_messages",
    "first_user_message",
]
