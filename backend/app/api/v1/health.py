"""Liveness, readiness, metrics.

These three sit at the root rather than under ``/api/v1``: a load balancer
should not have to know your API version. None of them needs a session.

The distinction between the first two is the one that gets confused most often.
``/healthz`` answers from process memory and touches nothing, so it can only
fail by not answering — which is exactly what a liveness probe should mean.
``/readyz`` checks what a request actually needs, and a failure there takes this
instance out of rotation without killing it.

One deliberate asymmetry: a failed **Celery** check does not fail readiness. The
API serves every read without a worker, and pulling the whole tier out of the
load balancer because a queue is down turns a degradation into an outage. It
reports ``ok: false`` and readiness stays ``ready``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import func, select, text

from app import __version__
from app.config import settings
from app.core.logging import get_logger
from app.db.models import JobFailedTask, NodeExecution, SyncState
from app.db.session import session_scope

log = get_logger(__name__)
router = APIRouter(tags=["health"])

_STARTED = time.monotonic()

#: Every readiness check gets this long and no longer. A probe that hangs is a
#: probe that has already failed.
CHECK_BUDGET_S = 0.5

#: Latency quantiles reported per op, straight out of ``node_executions`` — the
#: same rows the step trace reads, so the dashboard and the UI cannot disagree.
QUANTILES = (0.5, 0.95, 0.99)

PROMETHEUS_TYPE = "text/plain; version=0.0.4; charset=utf-8"


# --------------------------------------------------------------------------- #
# /healthz
# --------------------------------------------------------------------------- #


@router.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Am I running."""
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.ENV,
        "uptime_s": int(time.monotonic() - _STARTED),
    }


# --------------------------------------------------------------------------- #
# /readyz
# --------------------------------------------------------------------------- #


def _ok(started: float, detail: str) -> dict[str, Any]:
    return {"ok": True, "latency_ms": int((time.perf_counter() - started) * 1000), "detail": detail}


def _bad(started: float, detail: str) -> dict[str, Any]:
    return {"ok": False, "latency_ms": int((time.perf_counter() - started) * 1000), "detail": detail}


async def _check_postgres() -> tuple[dict[str, Any], dict[str, Any]]:
    """``SELECT 1`` and the pgvector extension, on one connection.

    They share a connection because one session is one connection and cannot
    serve two queries at once; splitting them would buy a millisecond and cost
    a pool slot.
    """
    started = time.perf_counter()
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
            postgres = _ok(started, "SELECT 1")

            mark = time.perf_counter()
            found = (
                await session.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
            vector = (
                _ok(mark, "extension present")
                if found
                else _bad(mark, "the vector extension is not installed")
            )
        return postgres, vector
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
        return _bad(started, detail), _bad(started, "not checked — postgres is down")


async def _check_redis() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from app.core import cache

        client = await cache.get_redis()
        await client.ping()
        return _ok(started, "PING")
    except Exception as exc:
        return _bad(started, f"{type(exc).__name__}: {str(exc)[:120]}")


def _ping_workers() -> dict[str, Any]:
    """Blocking, so it runs in a thread. Celery's control channel is sync."""
    from app.tasks.celery_app import QUEUE_NAMES, celery_app

    replies = celery_app.control.ping(timeout=CHECK_BUDGET_S) or []
    count = len(replies)
    if not count:
        return {"ok": False, "detail": "no workers answered"}
    return {"ok": True, "detail": f"{count} worker{'' if count == 1 else 's'}, queues: " + " ".join(QUEUE_NAMES)}


async def _check_celery() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_ping_workers), timeout=CHECK_BUDGET_S * 2
        )
        return {**result, "latency_ms": int((time.perf_counter() - started) * 1000)}
    except TimeoutError:
        return _bad(started, "the broker did not answer in time")
    except Exception as exc:
        return _bad(started, f"{type(exc).__name__}: {str(exc)[:120]}")


def _head_revision() -> str | None:
    """The newest migration on disk. Sync file IO, so it runs in a thread."""
    root = Path(__file__).resolve().parents[3]
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "alembic"))
        return ScriptDirectory.from_config(config).get_current_head()
    except Exception as exc:
        log.warning("readyz.alembic_unreadable", error=str(exc))
        return None


async def _check_migrations() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with session_scope() as session:
            applied = (
                await session.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar_one_or_none()
    except Exception as exc:
        return _bad(started, f"{type(exc).__name__}: {str(exc)[:120]}")

    head = await asyncio.to_thread(_head_revision)
    if applied is None:
        return _bad(started, "no migrations have been applied")
    if head is None:
        return _ok(started, f"at {applied} (head unknown)")
    if applied != head:
        return _bad(started, f"at {applied}, head is {head}")
    return _ok(started, f"at head {applied}")


@router.get("/readyz")
async def readyz() -> Response:
    """Can I serve."""
    postgres_pair, redis_check, celery_check, migrations = await asyncio.gather(
        _check_postgres(), _check_redis(), _check_celery(), _check_migrations()
    )
    postgres, pgvector = postgres_pair

    checks = {
        "postgres": postgres,
        "pgvector": pgvector,
        "redis": redis_check,
        "celery": celery_check,
        "migrations": migrations,
    }
    # Celery is reported but does not gate: reads work without a worker.
    required = ("postgres", "pgvector", "redis", "migrations")
    ready = all(checks[name]["ok"] for name in required)

    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


# --------------------------------------------------------------------------- #
# /metrics
# --------------------------------------------------------------------------- #


def _metric(lines: list[str], name: str, help_text: str, kind: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")


def _label(value: Any) -> str:
    """A label value with the three characters Prometheus cannot take escaped."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


async def _cache_lines(lines: list[str]) -> None:
    from app.core import cache

    stats = await cache.stats()
    _metric(lines, "orchestrator_cache_hit_ratio", "Redis cache hits over lookups.", "gauge")
    lines.append(f"orchestrator_cache_hit_ratio {stats.get('hit_rate', 0.0)}")
    _metric(lines, "orchestrator_cache_hits_total", "Redis cache hits by key class.", "counter")
    for namespace, counts in (stats.get("namespaces") or {}).items():
        lines.append(
            f'orchestrator_cache_hits_total{{kind="{_label(namespace)}"}} {int(counts.get("hits", 0))}'
        )
    _metric(lines, "orchestrator_cache_misses_total", "Redis cache misses by key class.", "counter")
    for namespace, counts in (stats.get("namespaces") or {}).items():
        lines.append(
            f'orchestrator_cache_misses_total{{kind="{_label(namespace)}"}} {int(counts.get("misses", 0))}'
        )


async def _op_lines(session: Any, lines: list[str]) -> None:
    """p50, p95 and p99 per op, computed in Postgres.

    Reading the durations out and sorting them in Python would work at a
    thousand rows and fall over at a million. ``percentile_disc`` picks a real
    observed value rather than interpolating between two, which is what you
    want for a latency you intend to explain.
    """
    seconds = func.extract("epoch", NodeExecution.finished_at - NodeExecution.started_at)
    rows = (
        await session.execute(
            select(
                NodeExecution.op,
                func.count().label("n"),
                *(
                    func.percentile_disc(q).within_group(seconds.asc()).label(f"p{q}")
                    for q in QUANTILES
                ),
            )
            .where(
                NodeExecution.started_at.isnot(None),
                NodeExecution.finished_at.isnot(None),
            )
            .group_by(NodeExecution.op)
        )
    ).all()
    if not rows:
        return

    _metric(
        lines,
        "orchestrator_op_duration_seconds",
        "Step duration by op, from node_executions.",
        "summary",
    )
    for row in rows:
        op = _label(row[0])
        for quantile, value in zip(QUANTILES, row[2:], strict=True):
            lines.append(
                f'orchestrator_op_duration_seconds{{op="{op}",quantile="{quantile}"}} '
                f"{float(value or 0.0):.4f}"
            )
        lines.append(f'orchestrator_op_duration_seconds_count{{op="{op}"}} {int(row[1])}')


async def _dlq_lines(session: Any, lines: list[str]) -> None:
    open_count = (
        await session.execute(
            select(func.count()).select_from(JobFailedTask).where(JobFailedTask.status == "open")
        )
    ).scalar_one()
    _metric(
        lines,
        "orchestrator_dlq_open",
        "Background jobs that exhausted their retries and are waiting.",
        "gauge",
    )
    lines.append(f"orchestrator_dlq_open {int(open_count)}")


async def _lag_lines(session: Any, lines: list[str]) -> None:
    rows = (
        await session.execute(
            select(
                SyncState.service,
                func.max(func.extract("epoch", func.now() - SyncState.last_success_at)),
            )
            .where(SyncState.last_success_at.isnot(None))
            .group_by(SyncState.service)
        )
    ).all()
    if not rows:
        return
    _metric(
        lines,
        "orchestrator_mirror_lag_seconds",
        "Worst seconds since a successful sync, by service.",
        "gauge",
    )
    for service, lag in rows:
        name = _label(getattr(service, "value", service))
        lines.append(f'orchestrator_mirror_lag_seconds{{service="{name}"}} {float(lag or 0.0):.1f}')


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> Response:
    """Prometheus text format.

    Every figure is derived at read time from rows that already exist. There is
    no metrics table, and no counter that can drift away from what happened.
    In production this endpoint is bound to the internal listener only.
    """
    lines: list[str] = []
    _metric(lines, "orchestrator_uptime_seconds", "Seconds since this process started.", "gauge")
    lines.append(f"orchestrator_uptime_seconds {int(time.monotonic() - _STARTED)}")

    try:
        await _cache_lines(lines)
    except Exception as exc:
        log.warning("metrics.cache_failed", error=str(exc))

    try:
        async with session_scope() as session:
            await _op_lines(session, lines)
            await _dlq_lines(session, lines)
            await _lag_lines(session, lines)
    except Exception as exc:
        log.warning("metrics.db_failed", error=str(exc))

    return PlainTextResponse("\n".join(lines) + "\n", media_type=PROMETHEUS_TYPE)


__all__ = ["router"]
