"""The Celery application: queues, policies, the beat schedule, and the bridge
that lets a synchronous task run our asynchronous code.

Five queues, because the workloads have nothing in common:

===============  =========================================================
``sync``         paging Google and upserting the mirror. The bulk of the work
``embed``        batching text to the embedding provider. A different limiter
``actions``      an approved, irreversible write. A person is waiting
``orchestration``resuming a paused run. Latency sensitive
``maintenance``  tokens, expiry, the DLQ sweeper, pruning. Interruptible
===============  =========================================================

``acks_late`` with ``prefetch 1`` and ``reject_on_worker_lost`` is only safe
because every task here is idempotent: sync upserts on a natural key and moves
its cursor after the upsert commits, embedding is a pure function of the row,
and ``actions.execute`` is gated by a conditional status update plus the Sent
check. Getting that backwards is how the same email goes out twice.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from celery import Celery, Task
from celery.schedules import crontab
from celery.signals import worker_process_shutdown, worker_process_init
from kombu import Queue

from app.config import settings
from app.core.logging import configure as configure_logging
from app.core.logging import get_logger
from app.tasks import TASK_MODULES, classify_error, error_payload

log = get_logger(__name__)

T = TypeVar("T")

QUEUE_NAMES: tuple[str, ...] = (
    "sync",
    "embed",
    "actions",
    "orchestration",
    "maintenance",
)

#: Redis has no real acknowledgement, so Celery re-delivers anything not
#: completed inside this window. It has to exceed the longest task, which is a
#: backfill page — capped at SYNC_PAGE_SIZE precisely so it cannot get close.
VISIBILITY_TIMEOUT_S = 900

#: Kept under the visibility timeout: a task that outlives its lease would be
#: delivered to a second worker while the first is still running it.
SOFT_TIME_LIMIT_S = 600
HARD_TIME_LIMIT_S = 780


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #

celery_app = Celery(
    "alpha_law",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=list(TASK_MODULES),
)

celery_app.conf.update(
    # -- delivery guarantees ------------------------------------------------
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    broker_transport_options={
        "visibility_timeout": VISIBILITY_TIMEOUT_S,
        "fanout_prefix": True,
        "fanout_patterns": True,
    },
    result_backend_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_S},
    broker_connection_retry_on_startup=True,
    # -- queues -------------------------------------------------------------
    task_queues=tuple(Queue(name) for name in QUEUE_NAMES),
    task_default_queue="maintenance",
    task_routes={
        "sync.*": {"queue": "sync"},
        "embed.*": {"queue": "embed"},
        "actions.*": {"queue": "actions"},
        "orchestration.*": {"queue": "orchestration"},
        "maintenance.*": {"queue": "maintenance"},
        "metrics.*": {"queue": "maintenance"},
    },
    # -- payloads and clocks ------------------------------------------------
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    task_soft_time_limit=SOFT_TIME_LIMIT_S,
    task_time_limit=HARD_TIME_LIMIT_S,
    worker_max_tasks_per_child=200,
    worker_hijack_root_logger=False,
    worker_send_task_events=True,
    task_send_sent_event=True,
)

celery_app.conf.beat_schedule = {
    "sync-dispatch-all-users": {
        "task": "sync.dispatch_all_users",
        "schedule": crontab(minute="*/15"),
        "options": {"queue": "sync", "expires": 600},
    },
    "maintenance-refresh-tokens": {
        "task": "maintenance.refresh_tokens",
        "schedule": crontab(minute="*/10"),
        "options": {"queue": "maintenance", "expires": 540},
    },
    "maintenance-expire-prompts": {
        "task": "maintenance.expire_prompts",
        "schedule": crontab(minute=0),
        "options": {"queue": "maintenance", "expires": 3300},
    },
    "maintenance-sweep-dlq": {
        "task": "maintenance.sweep_dlq",
        "schedule": crontab(minute="*/30"),
        "options": {"queue": "maintenance", "expires": 1500},
    },
    "maintenance-prune-sync": {
        "task": "maintenance.prune_sync",
        "schedule": crontab(hour=3, minute=0),
        "options": {"queue": "maintenance", "expires": 3600},
    },
    "metrics-freshness": {
        "task": "metrics.freshness",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "maintenance", "expires": 240},
    },
    # Not in contracts.md's six: a stuck run is a visibly frozen UI, so it gets
    # a short cadence of its own.
    "orchestration-sweep-stuck": {
        "task": "orchestration.sweep_stuck",
        "schedule": crontab(minute="*/2"),
        "options": {"queue": "orchestration", "expires": 110},
    },
}


# --------------------------------------------------------------------------- #
# The async bridge
# --------------------------------------------------------------------------- #
#
# Every task is a synchronous function around asynchronous code. The obvious
# `asyncio.run(...)` per task is wrong here: it builds and tears down an event
# loop each time, and both the SQLAlchemy pool and the Redis client hold
# connections bound to the loop that opened them. The second task would find a
# pool full of sockets belonging to a loop that no longer exists.
#
# So each worker process gets one long-lived loop on a daemon thread, and tasks
# hand it coroutines. The pool survives, and connection reuse is real.

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _spawn_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=_run, name="celery-asyncio", daemon=True).start()
    return loop


def event_loop() -> asyncio.AbstractEventLoop:
    """This process's loop, started on first use."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = _spawn_loop()
        return _loop


def run_async(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run a coroutine from inside a synchronous task and return its result.

    The exception a coroutine raises comes back out of here unchanged, which is
    what lets ``autoretry_for`` and ``on_failure`` see the real failure.
    """
    future = asyncio.run_coroutine_threadsafe(coro, event_loop())
    try:
        return future.result(timeout if timeout is not None else HARD_TIME_LIMIT_S)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError("the task's coroutine outran its time limit") from None


@worker_process_init.connect
def _configure_worker(**_: Any) -> None:
    configure_logging()


@worker_process_shutdown.connect
def _close_worker(**_: Any) -> None:
    """Give the pool and the Redis client a chance to close cleanly."""
    global _loop
    loop = _loop
    if loop is None or loop.is_closed():
        return
    try:
        from app.core import cache
        from app.db.session import shutdown_engine

        async def _close() -> None:
            await shutdown_engine()
            await cache.close()

        asyncio.run_coroutine_threadsafe(_close(), loop).result(10)
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        log.warning("worker.shutdown_unclean", error=str(exc))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        _loop = None


# --------------------------------------------------------------------------- #
# The base task
# --------------------------------------------------------------------------- #


class NonRetryable(Exception):
    """A failure a retry cannot fix: a revoked grant, a stale precondition, a
    bad argument. Raising this skips the retry ladder and goes straight to the
    dead-letter row, where a person can see it."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str = "INVALID",
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.details = details or {}
        if cause is not None:
            self.__cause__ = cause


class AppTask(Task):
    """Retry policy and the dead-letter row, in one place.

    ``autoretry_for`` covers everything, and :meth:`retry` refuses the ones a
    retry cannot fix. Backoff is exponential with full jitter, so a fleet that
    hits a shared quota ceiling does not come back in one synchronised wave.
    """

    autoretry_for = (Exception,)
    # Honoured by Celery 5.3+; :meth:`retry` below enforces the same rule
    # regardless, so the policy does not depend on the version.
    dont_autoretry_for = (NonRetryable,)
    throws = (NonRetryable,)

    max_retries = 5
    retry_backoff = 2
    retry_backoff_max = 600
    retry_jitter = True

    acks_late = True
    track_started = True

    #: Which positional argument carries ``user_id``, for the DLQ row. ``None``
    #: for a task that is not on behalf of anybody.
    user_arg: int | None = 0

    # -- retry ---------------------------------------------------------------

    def retry(self, *args: Any, **kwargs: Any) -> Any:
        exc = kwargs.get("exc")
        if _is_non_retryable(exc):
            raise exc  # type: ignore[misc]
        return super().retry(*args, **kwargs)

    # -- failure -------------------------------------------------------------

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Retries are exhausted, or the failure was terminal. Write the row a
        person will read.

        Nothing in here may raise: a failing failure handler loses the failure.
        """
        error_class = getattr(exc, "error_class", None) or classify_error(exc)
        payload = {"args": list(args or ()), "kwargs": dict(kwargs or {})}
        try:
            run_async(
                _record_failure(
                    task_name=self.name,
                    queue=_queue_of(self),
                    user_id=_user_id_of(self, args, kwargs),
                    task_input=payload,
                    error_class=str(error_class),
                    error=error_payload(exc),
                    traceback=str(einfo)[:20000] if einfo is not None else None,
                    attempts=int(getattr(self.request, "retries", 0) or 0) + 1,
                    celery_task_id=task_id,
                ),
                timeout=30,
            )
        except Exception as write_error:  # noqa: BLE001 - see docstring
            log.error(
                "task.dlq_write_failed",
                task=self.name,
                task_id=task_id,
                error=str(write_error),
            )
        log.error(
            "task.failed",
            task=self.name,
            task_id=task_id,
            error_class=str(error_class),
            error=str(exc)[:500],
        )


def _is_non_retryable(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, NonRetryable):
        return True
    return isinstance(getattr(exc, "__cause__", None), NonRetryable)


def _queue_of(task: Task) -> str:
    """The queue this task ran on, for the dead-letter row."""
    delivery = getattr(task.request, "delivery_info", None) or {}
    routing_key = delivery.get("routing_key") if isinstance(delivery, dict) else None
    if routing_key in QUEUE_NAMES:
        return str(routing_key)
    queue = getattr(task, "queue", None)
    if queue in QUEUE_NAMES:
        return str(queue)
    prefix = (task.name or "").split(".", 1)[0]
    return prefix if prefix in QUEUE_NAMES else str(celery_app.conf.task_default_queue)


def _user_id_of(
    task: Task, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    """Whose job this was, if it was anybody's."""
    from app.core.ids import is_id

    value = (kwargs or {}).get("user_id")
    if isinstance(value, str) and is_id(value):
        return value
    index = getattr(task, "user_arg", 0)
    if index is None:
        return None
    args = tuple(args or ())
    if index < len(args) and isinstance(args[index], str) and is_id(args[index]):
        return str(args[index])
    return None


async def _record_failure(
    *,
    task_name: str,
    queue: str,
    user_id: str | None,
    task_input: dict[str, Any],
    error_class: str,
    error: dict[str, Any],
    traceback: str | None,
    attempts: int,
    celery_task_id: str | None,
) -> None:
    """Insert, or bump, one ``job_failed_tasks`` row.

    Deduplicated on ``(task_name, task_input)``. Without that, one user with a
    revoked token writes 96 rows a day and the table stops being read.
    """
    from sqlalchemy import select

    from app.db.models import JobFailedTask
    from app.db.repositories import audit as audit_repo
    from app.db.session import session_scope

    async with session_scope() as session:
        existing = await session.execute(
            select(JobFailedTask)
            .where(
                JobFailedTask.status == "open",
                JobFailedTask.task_name == task_name,
                JobFailedTask.task_input == task_input,
            )
            .limit(1)
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            await audit_repo.touch_failed_task(session, row.user_id, row.id)
            return
        await audit_repo.record_failed_task(
            session,
            user_id,
            task_name=task_name,
            queue=queue,
            error_class=error_class,
            attempts=attempts,
            task_input=task_input,
            error=error,
            traceback=traceback,
            celery_task_id=celery_task_id,
        )


# --------------------------------------------------------------------------- #
# The beat fan-out
# --------------------------------------------------------------------------- #


@celery_app.task(
    base=AppTask,
    bind=True,
    name="sync.dispatch_all_users",
    queue="sync",
    user_arg=None,
    max_retries=2,
)
def dispatch_all_users(self: AppTask) -> dict[str, Any]:
    """Every 15 minutes: hand each service its own fan-out.

    It lives here rather than in one of the three sync modules because it
    belongs to none of them. It sends by name, so nothing is imported and there
    is no cycle.
    """
    dispatched = []
    for service in ("gmail", "gcal", "gdrive"):
        name = f"sync.{service}.dispatch_all_users"
        celery_app.send_task(name, queue="sync", expires=600)
        dispatched.append(name)
    log.info("sync.dispatch_all_users", dispatched=dispatched)
    return {"dispatched": dispatched}


__all__ = [
    "celery_app",
    "AppTask",
    "NonRetryable",
    "run_async",
    "event_loop",
    "QUEUE_NAMES",
    "VISIBILITY_TIMEOUT_S",
    "dispatch_all_users",
]
