"""audit_log, and the failed-task bookkeeping that sits next to it.

Audit rows record what changed outside our system: emails sent, events deleted,
files shared, tokens granted and revoked, data purged. They never change and
never cascade, which is why they are not part of ``actions``.

Bodies are never stored. ``payload_hash`` is a uuid5 of the full content, so we
can prove this exact email was sent without keeping the text.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import fingerprint, new_id
from app.db.models import AuditLog, JobFailedTask
from app.db.repositories import ensure_utc, utcnow

#: Who did it.
ACTORS = frozenset({"user", "system", "worker"})

#: uuid5 namespace label for audit payload fingerprints.
PAYLOAD_NAMESPACE = "audit.payload"


def payload_fingerprint(payload: Any) -> uuid.UUID:
    """uuid5 over the canonical JSON of a payload.

    Sorted keys and no incidental whitespace, so the same content always gives
    the same id no matter how the dict was built.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return fingerprint(PAYLOAD_NAMESPACE, canonical)


async def write_audit(
    session: AsyncSession,
    user_id: str,
    *,
    actor: str,
    action: str,
    status: str,
    conversation_id: str | None = None,
    resource_id: str | None = None,
    payload: Any | None = None,
    payload_hash: uuid.UUID | None = None,
    payload_visible: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
    at: dt.datetime | None = None,
) -> AuditLog:
    """Append one row.

    Pass ``payload`` to have the fingerprint computed here, or ``payload_hash``
    if you already have it. ``payload_visible`` is the part that is safe to
    show — recipients, subject — never the body.
    """
    if payload_hash is None and payload is not None:
        payload_hash = payload_fingerprint(payload)

    row = AuditLog(
        user_id=user_id,
        conversation_id=conversation_id,
        actor=actor,
        action=action,
        resource_id=resource_id,
        payload_hash=payload_hash,
        payload_visible=payload_visible,
        status=status,
        error=error,
        ip=ip,
        user_agent=user_agent,
        created_at=ensure_utc(at) or utcnow(),
    )
    session.add(row)
    await session.flush()
    return row


async def list_audit(
    session: AsyncSession,
    user_id: str,
    *,
    action: str | None = None,
    conversation_id: str | None = None,
    status: str | None = None,
    since: dt.datetime | None = None,
    before: dt.datetime | None = None,
    limit: int = 100,
) -> list[AuditLog]:
    """Newest first. Keyset paginated on ``created_at``."""
    stmt = select(AuditLog).where(AuditLog.user_id == user_id)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if conversation_id is not None:
        stmt = stmt.where(AuditLog.conversation_id == conversation_id)
    if status is not None:
        stmt = stmt.where(AuditLog.status == status)
    start = ensure_utc(since)
    if start is not None:
        stmt = stmt.where(AuditLog.created_at >= start)
    end = ensure_utc(before)
    if end is not None:
        stmt = stmt.where(AuditLog.created_at < end)
    stmt = stmt.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def find_by_payload(
    session: AsyncSession, user_id: str, payload_hash: uuid.UUID
) -> list[AuditLog]:
    """"Was this exact thing sent?" — answered without keeping the text."""
    result = await session.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id, AuditLog.payload_hash == payload_hash)
        .order_by(AuditLog.created_at.desc())
    )
    return list(result.scalars().all())


async def count_actions(
    session: AsyncSession,
    user_id: str,
    *,
    since: dt.datetime | None = None,
) -> dict[str, int]:
    """``{action: count}`` — what the activity summary reads."""
    stmt = select(AuditLog.action, func.count()).where(AuditLog.user_id == user_id)
    start = ensure_utc(since)
    if start is not None:
        stmt = stmt.where(AuditLog.created_at >= start)
    result = await session.execute(stmt.group_by(AuditLog.action))
    return {action: int(count) for action, count in result.all()}


# --------------------------------------------------------------------------- #
# job_failed_tasks
# --------------------------------------------------------------------------- #


async def record_failed_task(
    session: AsyncSession,
    user_id: str | None,
    *,
    task_name: str,
    queue: str,
    error_class: str,
    attempts: int,
    task_input: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    traceback: str | None = None,
    celery_task_id: str | None = None,
    at: dt.datetime | None = None,
) -> JobFailedTask:
    """A job that exhausted its retries.

    ``user_id`` is nullable here because some tasks are not on behalf of anyone
    — a beat job, an admin sweep.
    """
    now = ensure_utc(at) or utcnow()
    row = JobFailedTask(
        id=new_id(),
        user_id=user_id,
        task_name=task_name,
        queue=queue,
        task_input=task_input,
        error_class=error_class,
        error=error,
        traceback=traceback,
        attempts=attempts,
        celery_task_id=celery_task_id,
        status="open",
        first_failed_at=now,
        last_failed_at=now,
    )
    session.add(row)
    await session.flush()
    return row


async def list_failed_tasks(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    status: str = "open",
    error_class: str | None = None,
    limit: int = 100,
) -> list[JobFailedTask]:
    """The dead-letter queue. ``user_id=None`` lists every user's — for
    ``maintenance.sweep_dlq``."""
    stmt = select(JobFailedTask).where(JobFailedTask.status == status)
    if user_id is not None:
        stmt = stmt.where(JobFailedTask.user_id == user_id)
    if error_class is not None:
        stmt = stmt.where(JobFailedTask.error_class == error_class)
    stmt = stmt.order_by(JobFailedTask.last_failed_at.asc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def count_failed_tasks(
    session: AsyncSession, user_id: str | None = None, *, status: str = "open"
) -> int:
    """The number on /sync/status."""
    stmt = (
        select(func.count())
        .select_from(JobFailedTask)
        .where(JobFailedTask.status == status)
    )
    if user_id is not None:
        stmt = stmt.where(JobFailedTask.user_id == user_id)
    result = await session.execute(stmt)
    return int(result.scalar_one())


async def close_failed_task(
    session: AsyncSession,
    user_id: str | None,
    task_id: str,
    *,
    status: str = "replayed",
) -> JobFailedTask | None:
    """The sweeper replayed it, or a person decided to ignore it."""
    stmt = update(JobFailedTask).where(JobFailedTask.id == task_id)
    if user_id is not None:
        stmt = stmt.where(JobFailedTask.user_id == user_id)
    result = await session.execute(stmt.values(status=status).returning(JobFailedTask))
    return result.scalar_one_or_none()


async def touch_failed_task(
    session: AsyncSession, user_id: str | None, task_id: str
) -> None:
    """It failed again after a replay."""
    stmt = update(JobFailedTask).where(JobFailedTask.id == task_id)
    if user_id is not None:
        stmt = stmt.where(JobFailedTask.user_id == user_id)
    await session.execute(
        stmt.values(
            last_failed_at=utcnow(), attempts=JobFailedTask.attempts + 1, status="open"
        )
    )


__all__ = [
    "ACTORS",
    "PAYLOAD_NAMESPACE",
    "payload_fingerprint",
    "write_audit",
    "list_audit",
    "find_by_payload",
    "count_actions",
    "record_failed_task",
    "list_failed_tasks",
    "count_failed_tasks",
    "close_failed_task",
    "touch_failed_task",
]
