"""One screen answering "is the background machinery actually running?"

    make jobs        # from the repo root
    .venv/bin/python scripts/jobs_status.py   # from backend/

Checks, in order: the two processes (worker, beat), the queues they feed
from, each service's mirror freshness, and what the dead-letter table holds.
Every line is a fact read live — nothing here is cached or inferred.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import subprocess
import sys

sys.path.insert(0, ".")

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def processes() -> None:
    out = subprocess.run(["pgrep", "-fl", "celery"], capture_output=True, text=True).stdout
    worker = [l for l in out.splitlines() if " worker" in l]
    beat = [l for l in out.splitlines() if " beat" in l]
    print(f"{OK if worker else BAD} worker  {'pid ' + worker[0].split()[0] if worker else 'NOT RUNNING — sync, embedding and approved writes are all stopped'}")
    print(f"{OK if beat else BAD} beat    {'pid ' + beat[0].split()[0] if beat else 'NOT RUNNING — nothing schedules the 15-minute sync; the mirror will quietly go stale'}")


def worker_ping() -> None:
    from app.tasks.celery_app import celery_app

    replies = celery_app.control.inspect(timeout=2.0).ping() or {}
    for name in replies:
        print(f"{OK} worker answers as {name}")
    if not replies:
        print(f"{BAD} no worker answered a ping (broker down, or worker wedged)")


def queues() -> None:
    import redis

    from app.config import settings

    r = redis.Redis.from_url(settings.REDIS_URL)
    for q in ("sync", "embed", "actions", "orchestration", "maintenance"):
        depth = r.llen(q)
        mark = OK if depth < 50 else WARN
        print(f"{mark} queue {q:<14} {depth} waiting")


async def mirror_and_dlq() -> None:
    import asyncpg

    from app.config import settings

    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    c = await asyncpg.connect(url)
    now = await c.fetchval("select now()")
    rows = await c.fetch(
        "select service, last_success_at, items_indexed, consecutive_failures,"
        " circuit_open_until from sync_state order by service"
    )
    if not rows:
        print(f"{WARN} no sync_state rows — no account has connected yet")
    for r in rows:
        lag = (now - r["last_success_at"]).total_seconds() if r["last_success_at"] else None
        stale = lag is None or lag > 1800
        breaker = r["circuit_open_until"] and r["circuit_open_until"] > now
        mark = BAD if breaker else (WARN if stale else OK)
        lag_s = f"{int(lag)}s ago" if lag is not None else "never"
        note = " BREAKER OPEN" if breaker else (" stale (>30 min)" if stale else "")
        print(f"{mark} sync {r['service']:<7} last success {lag_s:<12} {r['items_indexed']} items, {r['consecutive_failures']} consecutive failures{note}")

    dlq = await c.fetch(
        "select task_name, count(*) n, max(last_failed_at) latest from job_failed_tasks"
        " where status = 'open' and last_failed_at > now() - interval '24 hours'"
        " group by task_name order by latest desc"
    )
    if dlq:
        for r in dlq:
            print(f"{WARN} dead-letter {r['task_name']:<24} {r['n']} open (latest {r['latest']:%H:%M:%S})")
    else:
        print(f"{OK} dead-letter table: nothing new in 24h")
    await c.close()


if __name__ == "__main__":
    print("── processes ──")
    processes()
    print("── broker ──")
    worker_ping()
    queues()
    print("── mirror ──")
    asyncio.run(mirror_and_dlq())
