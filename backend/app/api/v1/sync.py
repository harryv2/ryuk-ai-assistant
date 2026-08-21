"""Refreshing our copy of Gmail, Calendar and Drive.

Background sync runs every fifteen minutes on Celery beat, smeared with
``countdown = hash(user_id) % 900`` so a million users do not all hit Google on
the same second. These two endpoints are for forcing a pass and for seeing
where the last one got to.

Nothing here returns a 409. A queued sync that turns out to be redundant costs a
worker a second; a request refused mid-outage costs the person their trust in
the button. Both outcomes are reported plainly in ``queued`` and ``skipped``
instead.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.v1.schemas import SERVICES, SyncTriggerRequest, iso, sync_service_dto
from app.auth.deps import CurrentUser, SessionDep
from app.config import settings
from app.core import cache
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import JobFailedTask
from app.db.repositories import sync_state as state_repo
from app.db.repositories import users as users_repo

log = get_logger(__name__)
router = APIRouter(tags=["sync"])

#: The Celery task per service, and the backfill twin of each.
TASKS: dict[str, tuple[str, str]] = {
    "gmail": ("sync.gmail", "sync.gmail.backfill"),
    "gcal": ("sync.gcal", "sync.gcal.backfill"),
    "gdrive": ("sync.gdrive", "sync.gdrive.backfill"),
}
QUEUE = "sync"

#: ``mode=full`` discards the cursor and re-walks from scratch. That is
#: expensive enough at Google's end to be worth a leash.
FULL_RESYNC_EVERY_S = 3600


def _full_key(user_id: str, service: str) -> str:
    return f"full:{user_id}:{service}"


@router.post("/sync/trigger", status_code=202)
async def trigger(
    body: SyncTriggerRequest, session: SessionDep, user: CurrentUser
) -> dict[str, Any]:
    """Queue a sync pass now."""
    token = await users_repo.get_token(session, user.id)
    from app.auth import token_store

    if token_store.needs_reauth(token):
        raise AppError(
            "GOOGLE_REAUTH_REQUIRED",
            "Reconnect your Google account before syncing.",
        )

    now = datetime.now(UTC)
    queued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for service in body.wanted():
        state = await state_repo.ensure_state(session, user.id, service)

        open_until = state.circuit_open_until
        if open_until is not None and open_until > now:
            skipped.append(
                {"service": service, "reason": "circuit_open", "until": iso(open_until)}
            )
            continue

        if body.mode == "full":
            key = _full_key(user.id, service)
            if await cache.get_json(key, namespace="sync") is not None:
                raise AppError(
                    "RATE_LIMITED",
                    "A full resync is allowed once an hour per service.",
                    details={"service": service, "retry_after_s": FULL_RESYNC_EVERY_S},
                )
            await cache.set_json(key, {"at": iso(now)}, ttl=FULL_RESYNC_EVERY_S, namespace="sync")

        incremental, backfill = TASKS[service]
        name = backfill if body.mode == "backfill" else incremental
        kwargs = {"user_id": user.id}
        if body.mode != "backfill":
            kwargs["mode"] = "full" if body.mode == "full" else "incremental"

        try:
            from app.tasks.celery_app import celery_app

            handle = await asyncio.to_thread(
                celery_app.send_task, name, kwargs=kwargs, queue=QUEUE
            )
            queued.append(
                {
                    "service": service,
                    "mode": body.mode,
                    "task_id": getattr(handle, "id", None),
                    "queue": QUEUE,
                }
            )
        except Exception as exc:
            # A dead broker is worth reporting per service rather than 500ing
            # the whole request: the other two may well have queued.
            log.warning("sync.enqueue_failed", service=service, error=str(exc))
            skipped.append(
                {"service": service, "reason": "broker_unavailable", "detail": str(exc)[:200]}
            )

    await session.commit()
    return {"queued": queued, "skipped": skipped, "poll": "/api/v1/sync/status"}


@router.get("/sync/status")
async def status(session: SessionDep, user: CurrentUser) -> dict[str, Any]:
    """How fresh our copy is, per service.

    Freshness is the worst service, not the average: an answer built from three
    corpora is only as current as the stalest one, and averaging hides exactly
    the case worth surfacing.
    """
    now = datetime.now(UTC)
    target = int(settings.SYNC_INTERVAL_MIN) * 60

    rows = {
        str(getattr(row.service, "value", row.service)): row
        for row in await state_repo.get_all_states(session, user.id)
    }

    services: dict[str, Any] = {}
    lags: list[int] = []
    for name in SERVICES:
        row = rows.get(name)
        if row is None:
            # Never synced is not the same as stale. It gets its own state
            # rather than a lag of zero, which would read as perfectly fresh.
            services[name] = {
                "last_synced_at": None,
                "last_success_at": None,
                "lag_seconds": None,
                "items_indexed": 0,
                "backfill_complete": False,
                "cursor_present": False,
                "consecutive_failures": 0,
                "circuit_open_until": None,
                "last_error": None,
                "healthy": False,
            }
            continue
        shaped = sync_service_dto(row, now=now, target_seconds=target)
        services[name] = shaped
        if shaped["lag_seconds"] is not None:
            lags.append(int(shaped["lag_seconds"]))

    worst = max(lags) if lags else None

    dlq_open = (
        await session.execute(
            select(func.count())
            .select_from(JobFailedTask)
            .where(JobFailedTask.user_id == user.id, JobFailedTask.status == "open")
        )
    ).scalar_one()
    dlq_oldest = (
        await session.execute(
            select(func.min(JobFailedTask.first_failed_at)).where(
                JobFailedTask.user_id == user.id, JobFailedTask.status == "open"
            )
        )
    ).scalar_one()

    token = await users_repo.get_token(session, user.id)
    from app.auth import token_store

    return {
        "services": services,
        "freshness": {
            "worst_lag_seconds": worst,
            "target_seconds": target,
            # An honest report that the mirror is staler than the goal. A
            # superlative question inside this window wants freshness="live".
            "within_target": worst is not None and worst <= target,
        },
        "dlq": {"open": int(dlq_open or 0), "oldest_at": iso(dlq_oldest)},
        "next_scheduled_at": iso(
            now.replace(second=0, microsecond=0)
            + _until_next_slot(now, settings.SYNC_INTERVAL_MIN)
        ),
        "needs_reauth": token_store.needs_reauth(token),
    }


def _until_next_slot(now: datetime, every_minutes: int) -> timedelta:
    """How long until beat's next tick. Beat runs on the wall clock, not on
    when this user last synced."""
    step = max(1, int(every_minutes))
    return timedelta(minutes=step - (now.minute % step))


__all__ = ["router"]
