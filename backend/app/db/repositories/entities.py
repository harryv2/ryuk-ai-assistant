"""conversation_entities — what this conversation has referred to.

Written whenever a node surfaces something to the user, so "that email about
the proposal" or "move the meeting with John" resolves against ~20 rows instead
of parsing five runs' worth of result blobs.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import new_id
from app.db.models import ConversationEntity
from app.db.repositories import ensure_utc, utcnow

#: The four kinds a chip can be.
ENTITY_TYPES = frozenset({"email", "event", "file", "person"})


async def upsert_entity(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    entity_type: str,
    entity_ref: str,
    label: str,
    meta: dict[str, Any] | None = None,
    run_id: str | None = None,
    last_seen_at: dt.datetime | None = None,
) -> ConversationEntity:
    """Record one reference, or bump the one already there."""
    rows = await upsert_entities(
        session,
        user_id,
        conversation_id,
        [
            {
                "entity_type": entity_type,
                "entity_ref": entity_ref,
                "label": label,
                "meta": meta,
            }
        ],
        run_id=run_id,
        last_seen_at=last_seen_at,
    )
    return rows[0]


async def upsert_entities(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    items: Sequence[dict[str, Any]],
    *,
    run_id: str | None = None,
    last_seen_at: dt.datetime | None = None,
) -> list[ConversationEntity]:
    """Record a batch — a search result set is one call, not twenty.

    Duplicates within the batch are collapsed first, because Postgres will not
    let one statement hit the same conflict target twice.
    """
    if not items:
        return []

    seen = ensure_utc(last_seen_at) or utcnow()
    collapsed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        entity_type = str(item["entity_type"])
        entity_ref = str(item["entity_ref"])
        collapsed[(entity_type, entity_ref)] = {
            "id": new_id(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "run_id": item.get("run_id", run_id),
            "entity_type": entity_type,
            "entity_ref": entity_ref,
            "label": str(item.get("label") or entity_ref),
            "meta": item.get("meta"),
            "last_seen_at": ensure_utc(item.get("last_seen_at")) or seen,
        }

    stmt = pg_insert(ConversationEntity).values(list(collapsed.values()))
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            ConversationEntity.conversation_id,
            ConversationEntity.entity_type,
            ConversationEntity.entity_ref,
        ],
        set_={
            "label": stmt.excluded.label,
            "meta": func.coalesce(stmt.excluded.meta, ConversationEntity.meta),
            "run_id": func.coalesce(stmt.excluded.run_id, ConversationEntity.run_id),
            "last_seen_at": func.greatest(
                stmt.excluded.last_seen_at, ConversationEntity.last_seen_at
            ),
        },
    ).returning(ConversationEntity)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_entity(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    entity_type: str,
    entity_ref: str,
) -> ConversationEntity | None:
    """One chip by its provider id or email address."""
    result = await session.execute(
        select(ConversationEntity).where(
            ConversationEntity.user_id == user_id,
            ConversationEntity.conversation_id == conversation_id,
            ConversationEntity.entity_type == entity_type,
            ConversationEntity.entity_ref == entity_ref,
        )
    )
    return result.scalar_one_or_none()


async def list_entities(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    *,
    entity_type: str | None = None,
    limit: int = 20,
) -> list[ConversationEntity]:
    """Most recently referred to first. This is the shortlist a follow-up
    resolves against."""
    stmt = select(ConversationEntity).where(
        ConversationEntity.user_id == user_id,
        ConversationEntity.conversation_id == conversation_id,
    )
    if entity_type is not None:
        stmt = stmt.where(ConversationEntity.entity_type == entity_type)
    stmt = stmt.order_by(ConversationEntity.last_seen_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_entities(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    term: str,
    *,
    entity_type: str | None = None,
    limit: int = 10,
) -> list[ConversationEntity]:
    """Loose match on the label or the ref — "the proposal", "john@"."""
    pattern = f"%{term.strip()}%"
    stmt = select(ConversationEntity).where(
        ConversationEntity.user_id == user_id,
        ConversationEntity.conversation_id == conversation_id,
        or_(
            ConversationEntity.label.ilike(pattern),
            ConversationEntity.entity_ref.ilike(pattern),
        ),
    )
    if entity_type is not None:
        stmt = stmt.where(ConversationEntity.entity_type == entity_type)
    stmt = stmt.order_by(ConversationEntity.last_seen_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def touch_entity(
    session: AsyncSession,
    user_id: str,
    conversation_id: str,
    entity_type: str,
    entity_ref: str,
    *,
    at: dt.datetime | None = None,
) -> None:
    """Referred to again, nothing else changed."""
    await session.execute(
        update(ConversationEntity)
        .where(
            ConversationEntity.user_id == user_id,
            ConversationEntity.conversation_id == conversation_id,
            ConversationEntity.entity_type == entity_type,
            ConversationEntity.entity_ref == entity_ref,
        )
        .values(last_seen_at=ensure_utc(at) or utcnow())
    )


async def trim_entities(
    session: AsyncSession, user_id: str, conversation_id: str, *, keep: int = 60
) -> int:
    """Keep the conversation's chip list short. Oldest references go first."""
    survivors = (
        select(ConversationEntity.id)
        .where(
            ConversationEntity.user_id == user_id,
            ConversationEntity.conversation_id == conversation_id,
        )
        .order_by(ConversationEntity.last_seen_at.desc())
        .limit(keep)
    )
    result = await session.execute(
        delete(ConversationEntity).where(
            ConversationEntity.user_id == user_id,
            ConversationEntity.conversation_id == conversation_id,
            ConversationEntity.id.notin_(survivors),
        )
    )
    return int(result.rowcount or 0)


__all__ = [
    "ENTITY_TYPES",
    "upsert_entity",
    "upsert_entities",
    "get_entity",
    "list_entities",
    "search_entities",
    "touch_entity",
    "trim_entities",
]
