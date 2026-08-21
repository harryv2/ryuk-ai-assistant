"""Maintenance: tokens, expiry, the dead-letter sweeper, pruning, gauges, and
the crash sweeper.

Everything here is periodic, interruptible and the lowest priority in the fleet.
None of it is on a person's critical path — but every one of them is the reason
something else on that path keeps working: a token that never expires under a
user, a confirm card that does not sit pending forever, a transient failure that
gets a second chance without anybody being asked about it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import func, select, update

from app.config import settings
from app.core import cache
from app.core.logging import get_logger
from app.db.models import (
    ActionStatus,
    JobFailedTask,
    NodeExecution,
    OAuthToken,
    RunStatus,
    SyncState,
    User,
)
from app.db.repositories import actions as actions_repo
from app.db.repositories import audit as audit_repo
from app.db.repositories import conversations as conversations_repo
from app.db.repositories import mirror
from app.db.repositories import prompts as prompts_repo
from app.db.repositories import runs as runs_repo
from app.db.repositories import steps as steps_repo
from app.db.repositories import users as users_repo
from app.db.session import session_scope
from app.tasks import classify_error, utcnow
from app.tasks.celery_app import AppTask, celery_app, run_async

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# 1. Tokens
# --------------------------------------------------------------------------- #

#: Google access tokens live an hour. Renewing 15 minutes out leaves room for
#: two failures before anything a person does starts returning 401.
REFRESH_MARGIN = dt.timedelta(minutes=15)

#: Past this many failures in a row the grant is the problem, not the network.
MAX_REFRESH_FAILURES = 5

#: Tokens per pass. At */10m this is 3,000 an hour, well past what a single
#: region needs.
REFRESH_LIMIT = 500


async def _refresh_one(user_id: str, provider: str) -> str:
    """Renew one token under a row lock.

    ``FOR UPDATE SKIP LOCKED`` is what makes several workers safe: whoever gets
    the row does the refresh, everybody else moves to the next one instead of
    queueing behind it or refreshing it a second time.
    """
    async with session_scope() as session:
        locked = await session.execute(
            select(OAuthToken)
            .where(
                OAuthToken.user_id == user_id,
                OAuthToken.provider == provider,
                OAuthToken.revoked_at.is_(None),
                OAuthToken.expires_at.is_not(None),
                OAuthToken.expires_at <= utcnow() + REFRESH_MARGIN,
                OAuthToken.refresh_failures < MAX_REFRESH_FAILURES,
            )
            .with_for_update(skip_locked=True)
        )
        token = locked.scalar_one_or_none()
        if token is None:
            return "skipped"  # another worker has it, or it no longer qualifies

        from app.auth.token_store import refresh_access_token

        try:
            # Act on the row this transaction just locked — a second lookup
            # inside the helper could pick a row someone else is refreshing.
            await refresh_access_token(session, user_id, provider=provider, row=token)
        except Exception as exc:  # noqa: BLE001 - the class decides what happens
            error_class = classify_error(exc)
            if error_class == "AUTH_REVOKED" or "invalid_grant" in str(exc):
                await users_repo.mark_token_revoked(session, user_id, provider)
                log.warning("token.revoked", user_id=user_id, provider=provider)
                return "revoked"
            failures = await users_repo.bump_refresh_failures(session, user_id, provider)
            log.warning(
                "token.refresh_failed",
                user_id=user_id,
                provider=provider,
                failures=failures,
                error=str(exc)[:300],
            )
            return "failed"
    return "refreshed"


async def _refresh_tokens(limit: int) -> dict[str, int]:
    async with session_scope() as session:
        due = await users_repo.list_tokens_expiring(
            session,
            None,
            before=utcnow() + REFRESH_MARGIN,
            max_failures=MAX_REFRESH_FAILURES,
            limit=limit,
        )
        candidates = [(t.user_id, t.provider) for t in due]

    tally: dict[str, int] = {"refreshed": 0, "revoked": 0, "failed": 0, "skipped": 0}
    for user_id, provider in candidates:
        tally[await _refresh_one(user_id, provider)] += 1

    if candidates:
        log.info("maintenance.refresh_tokens", due=len(candidates), **tally)
    return tally


@celery_app.task(
    base=AppTask,
    bind=True,
    name="maintenance.refresh_tokens",
    queue="maintenance",
    user_arg=None,
    max_retries=3,
)
def refresh_tokens(self: AppTask, limit: int = REFRESH_LIMIT) -> dict[str, int]:
    """Renew anything expiring inside 15 minutes.

    ``token_store`` re-encrypts under the current key while it is writing
    anyway, which is what makes key rotation a background job rather than a
    maintenance window.
    """
    return run_async(_refresh_tokens(limit))


# --------------------------------------------------------------------------- #
# 2. Prompt expiry
# --------------------------------------------------------------------------- #


async def _delete_draft(user_id: str, external_ref: str) -> None:
    """Best effort: drop the Gmail draft behind an expired card.

    Leaving it would put a draft in somebody's mailbox that our UI no longer
    shows and nobody meant to keep. A failure here is logged, never raised —
    the expiry itself has already been recorded.
    """
    try:
        from app.google.client import clients_for
        from app.services import gmail as gmail_api

        async with session_scope() as session:
            clients = await clients_for(session, user_id)
        await gmail_api.delete_draft(clients, external_ref)
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.info(
            "maintenance.draft_delete_failed",
            user_id=user_id,
            external_ref=external_ref,
            error=str(exc)[:200],
        )


async def _expire_prompts(limit: int) -> dict[str, int]:
    drafts: list[tuple[str, str]] = []
    async with session_scope() as session:
        expired = await prompts_repo.expire_prompts(session, None, limit=limit)
        cancelled = 0
        for prompt in expired:
            for action in await actions_repo.list_for_prompt(
                session, prompt.user_id, prompt.id
            ):
                if action.external_ref and action.op.startswith("gmail."):
                    drafts.append((action.user_id, action.external_ref))
            cancelled += await actions_repo.cancel_for_prompt(
                session,
                prompt.user_id,
                prompt.id,
                status=ActionStatus.EXPIRED,
                reason="prompt_expired",
            )
        # Backstop for anything whose card expired in an earlier pass.
        orphans = await actions_repo.expire_actions(session, None)

    for user_id, ref in drafts:
        await _delete_draft(user_id, ref)

    if expired or orphans:
        log.info(
            "maintenance.expire_prompts",
            prompts=len(expired),
            actions=cancelled,
            orphans=orphans,
            drafts_deleted=len(drafts),
        )
    return {"prompts": len(expired), "actions": cancelled, "orphans": orphans}


@celery_app.task(
    base=AppTask,
    bind=True,
    name="maintenance.expire_prompts",
    queue="maintenance",
    user_arg=None,
    max_retries=3,
)
def expire_prompts(self: AppTask, limit: int = 1000) -> dict[str, int]:
    """Close out cards nobody answered, and everything they were gating."""
    return run_async(_expire_prompts(limit))


# --------------------------------------------------------------------------- #
# 3. The dead-letter sweeper
# --------------------------------------------------------------------------- #

#: Failure classes worth replaying, and how long to wait first. The ones that
#: are missing are missing on purpose: AUTH_REVOKED needs the person to
#: reauthorise, PRECONDITION means the remote object moved and the plan is
#: stale, NOT_FOUND is an answer, and INVALID is our bug — replaying it fails
#: identically forever.
REPLAYABLE: dict[str, dt.timedelta] = {
    "TRANSIENT": dt.timedelta(minutes=10),
    "RATE_LIMITED": dt.timedelta(minutes=30),
    "QUOTA_EXHAUSTED": dt.timedelta(hours=24),
    "AUTH_EXPIRED": dt.timedelta(minutes=10),
    "UNKNOWN": dt.timedelta(hours=1),
}

#: A job that has failed this many times is not going to start working.
MAX_REPLAY_ATTEMPTS = 8


async def _sweep_dlq(limit: int) -> dict[str, int]:
    now = utcnow()
    replayed = 0
    held = 0
    abandoned = 0
    async with session_scope() as session:
        open_rows = await audit_repo.list_failed_tasks(session, None, status="open",
                                                       limit=limit)
        # A task whose user no longer exists can never succeed — replaying it
        # produces a fresh FK failure every half hour, forever. Deleting a
        # test account left dozens of these cycling; check once per sweep.
        owner_ids = {row.user_id for row in open_rows if row.user_id}
        living: set[str] = set()
        if owner_ids:
            found = await session.execute(
                select(User.id).where(User.id.in_(owner_ids))
            )
            living = set(found.scalars().all())
        for row in open_rows:
            if row.user_id and row.user_id not in living:
                await audit_repo.close_failed_task(
                    session, row.user_id, row.id, status="abandoned"
                )
                abandoned += 1
                continue
            wait = REPLAYABLE.get(row.error_class)
            if wait is None or int(row.attempts or 0) >= MAX_REPLAY_ATTEMPTS:
                held += 1
                continue
            if row.last_failed_at + wait > now:
                continue  # not yet — never replay inside the backoff window

            payload = row.task_input or {}
            celery_app.send_task(
                row.task_name,
                args=list(payload.get("args") or []),
                kwargs=dict(payload.get("kwargs") or {}),
                queue=row.queue,
            )
            await audit_repo.close_failed_task(
                session, row.user_id, row.id, status="replayed"
            )
            replayed += 1

    log.info("maintenance.sweep_dlq", open=len(open_rows), replayed=replayed,
             held=held, abandoned=abandoned)
    return {"open": len(open_rows), "replayed": replayed, "held": held,
            "abandoned": abandoned}


@celery_app.task(
    base=AppTask,
    bind=True,
    name="maintenance.sweep_dlq",
    queue="maintenance",
    user_arg=None,
    max_retries=2,
)
def sweep_dlq(self: AppTask, limit: int = 200) -> dict[str, int]:
    """Replay the replayable half of ``job_failed_tasks``.

    Celery has no dead-letter queue; this is ours. The shape of what is left
    behind is the diagnosis: all AUTH_REVOKED is a consent-screen problem, all
    RATE_LIMITED is a quota problem, all INVALID is us.
    """
    return run_async(_sweep_dlq(limit))


# --------------------------------------------------------------------------- #
# 4. Pruning
# --------------------------------------------------------------------------- #

#: Node results are a snapshot for the step trace in the chat. After this long nobody is
#: reading them and they are the largest JSONB in the database.
RESULT_RETENTION_DAYS = 90


async def _prune_sync(batch: int) -> dict[str, int]:
    cutoff = utcnow() - dt.timedelta(days=int(settings.SYNC_BACKFILL_DAYS))
    dropped: dict[str, int] = {}
    async with session_scope() as session:
        for table in mirror.SPECS:
            dropped[table] = await mirror.prune(
                session, None, table, cutoff, limit=batch
            )

    results_cutoff = utcnow() - dt.timedelta(days=RESULT_RETENTION_DAYS)
    async with session_scope() as session:
        doomed = (
            select(NodeExecution.id)
            .where(
                NodeExecution.started_at < results_cutoff,
                NodeExecution.result.is_not(None),
            )
            .limit(batch)
        )
        cleared = await session.execute(
            update(NodeExecution)
            .where(NodeExecution.id.in_(doomed))
            .values(result=None)
        )
        dropped["node_results"] = int(cleared.rowcount or 0)

    log.info("maintenance.prune_sync", cutoff=cutoff.isoformat(), **dropped)
    return dropped


@celery_app.task(
    base=AppTask,
    bind=True,
    name="maintenance.prune_sync",
    queue="maintenance",
    user_arg=None,
    max_retries=2,
)
def prune_sync(self: AppTask, batch: int = 5000) -> dict[str, int]:
    """Drop mirror rows outside the backfill window and blank old node results.

    Bounded per pass on purpose: this is a heavy delete and it runs at 03:00.
    """
    return run_async(_prune_sync(batch))


# --------------------------------------------------------------------------- #
# 5. Freshness gauges
# --------------------------------------------------------------------------- #

#: The freshness SLO: the mirror should never be more than one cycle behind.
FRESHNESS_TARGET_S = 900

#: Where /metrics reads the last computed gauge from.
FRESHNESS_KEY = "sync"


async def _freshness() -> dict[str, Any]:
    now = utcnow()
    # Postgres does the arithmetic: one aggregate, no rows crossing the wire.
    lag = func.extract("epoch", func.now() - SyncState.last_success_at)
    async with session_scope() as session:
        rows = await session.execute(
            select(
                SyncState.service,
                func.count(),
                func.max(lag),
                func.avg(lag),
                func.count(SyncState.last_success_at),
            ).group_by(SyncState.service)
        )
        services: dict[str, Any] = {}
        for service, total, worst, mean, with_success in rows.all():
            worst_s = float(worst) if worst is not None else None
            services[str(getattr(service, "value", service))] = {
                "states": int(total),
                "never_synced": int(total) - int(with_success or 0),
                "worst_lag_seconds": worst_s,
                "mean_lag_seconds": float(mean) if mean is not None else None,
                "within_target": worst_s is not None and worst_s <= FRESHNESS_TARGET_S,
            }

        failed = await session.execute(
            select(JobFailedTask.error_class, func.count())
            .where(JobFailedTask.status == "open")
            .group_by(JobFailedTask.error_class)
        )
        dlq = {str(cls): int(count) for cls, count in failed.all()}

    gauge = {
        "at": now.isoformat(),
        "target_seconds": FRESHNESS_TARGET_S,
        "services": services,
        "job_failed_tasks_open": dlq,
    }
    await cache.set_json(FRESHNESS_KEY, gauge, ttl=1800, namespace="metrics")
    log.info("metrics.freshness", **{k: v for k, v in gauge.items() if k != "at"})
    return gauge


@celery_app.task(
    base=AppTask,
    bind=True,
    name="metrics.freshness",
    queue="maintenance",
    user_arg=None,
    max_retries=1,
)
def freshness(self: AppTask) -> dict[str, Any]:
    """One aggregate over ``sync_state``, parked in Redis for ``/metrics``.

    Not a work task — it changes nothing. It is here because a freshness SLO
    nobody measures is a freshness SLO nobody has.
    """
    return run_async(_freshness())


# --------------------------------------------------------------------------- #
# 6. The crash sweeper
# --------------------------------------------------------------------------- #

#: How long past its hard deadline a `running` run has to be before we call the
#: process that owned it gone. Generous, because killing a live run is worse
#: than leaving a dead one an extra minute.
STUCK_AFTER_S = max(int(settings.HARD_DEADLINE_MS / 1000) * 3, 120)

STOPPED_TEXT = (
    "That request stopped before it finished — the process running it went "
    "away. Nothing was changed. Ask me again and I will start over."
)


async def _publish_run(run_id: str, event: str, data: dict[str, Any]) -> None:
    try:
        from app.orchestrator.events import publish

        await publish(run_id, event, data)
    except Exception as exc:  # noqa: BLE001 - a sweeper never fails on fan-out
        log.warning("sweep.publish_failed", run_id=run_id, error=str(exc)[:200])


async def _close_dead_run(user_id: str, run_id: str, conversation_id: str) -> None:
    """A `running` run whose worker is gone: stop it, say so, unfreeze the UI."""
    async with session_scope() as session:
        await runs_repo.set_status(
            session,
            user_id,
            run_id,
            RunStatus.TIMEOUT,
            error={"reason": "worker_lost", "class": "TRANSIENT"},
        )
        await steps_repo.cancel_pending(session, user_id, run_id, reason="worker_lost")
        for message in await conversations_repo.list_messages_for_run(
            session, user_id, run_id
        ):
            for action in await actions_repo.list_for_message(
                session, user_id, message.id
            ):
                if action.status in (ActionStatus.DRAFT, ActionStatus.APPROVED):
                    await actions_repo.cancel_action(
                        session, user_id, action.id, reason="run_timeout"
                    )
        await conversations_repo.add_message(
            session,
            user_id,
            conversation_id,
            role="assistant",
            content=[{"type": "text", "data": {"markdown": STOPPED_TEXT}}],
            run_id=run_id,
        )
    await _publish_run(
        run_id, "run.complete", {"run_id": run_id, "status": "timeout",
                                 "reason": "worker_lost"}
    )
    log.warning("sweep.run_timeout", user_id=user_id, run_id=run_id)


async def _sweep_stuck(limit: int) -> dict[str, int]:
    now = utcnow()
    tally = {"timed_out": 0, "resumed": 0, "cancelled": 0, "left_alone": 0}

    async with session_scope() as session:
        live = await runs_repo.list_resumable(
            session, None, older_than=now - dt.timedelta(seconds=STUCK_AFTER_S),
            limit=limit,
        )
        snapshot = [
            (r.id, r.user_id, r.conversation_id, r.status, r.started_at) for r in live
        ]

    for run_id, user_id, conversation_id, status, _started in snapshot:
        if status == RunStatus.RUNNING:
            await _close_dead_run(user_id, run_id, conversation_id)
            tally["timed_out"] += 1
            continue

        # awaiting_input is the correct resting state for a run that asked a
        # question. It is stuck only when the card it is waiting on is no
        # longer pending — the request that answered it died in between.
        async with session_scope() as session:
            cards = await prompts_repo.list_prompts(
                session, user_id, run_id=run_id, limit=20
            )
        blocking = [c for c in cards if c.blocking]
        if not blocking:
            tally["left_alone"] += 1
            continue
        card = blocking[0]

        if card.status.value == "pending":
            tally["left_alone"] += 1
        elif card.status.value == "answered":
            celery_app.send_task(
                "orchestration.resume_run",
                args=[user_id, run_id],
                kwargs={"input_id": card.id},
                queue="orchestration",
            )
            tally["resumed"] += 1
        else:  # expired, cancelled, superseded — nothing left to wait for
            async with session_scope() as session:
                await runs_repo.set_status(
                    session,
                    user_id,
                    run_id,
                    RunStatus.CANCELLED,
                    error={"reason": f"input_{card.status.value}"},
                )
                await steps_repo.cancel_pending(
                    session, user_id, run_id, reason=f"input_{card.status.value}"
                )
            await _publish_run(
                run_id,
                "run.complete",
                {"run_id": run_id, "status": "cancelled",
                 "reason": f"input_{card.status.value}"},
            )
            tally["cancelled"] += 1

    if any(v for k, v in tally.items() if k != "left_alone"):
        log.info("maintenance.sweep_stuck", examined=len(snapshot), **tally)
    return tally


@celery_app.task(
    base=AppTask,
    bind=True,
    name="orchestration.sweep_stuck",
    queue="orchestration",
    user_arg=None,
    max_retries=2,
)
def sweep_stuck(self: AppTask, limit: int = 200) -> dict[str, int]:
    """Runs left behind by a dead worker, and paused runs whose answer was lost.

    Two cases that look alike and are not: ``running`` past the hard deadline
    means nobody is going to finish it; ``awaiting_input`` means somebody is
    thinking, which is not a fault unless the card it is waiting on has already
    been dealt with.
    """
    return run_async(_sweep_stuck(limit))


__all__ = [
    "REFRESH_MARGIN",
    "MAX_REFRESH_FAILURES",
    "REPLAYABLE",
    "MAX_REPLAY_ATTEMPTS",
    "RESULT_RETENTION_DAYS",
    "FRESHNESS_TARGET_S",
    "STUCK_AFTER_S",
    "refresh_tokens",
    "expire_prompts",
    "sweep_dlq",
    "prune_sync",
    "freshness",
    "sweep_stuck",
]
