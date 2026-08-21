"""Two limiters.

**The query limiter** — a sliding window per user over the last hour, so a
person cannot burn the LLM budget in one minute. A Redis sorted set holds one
member per query; the Lua script trims, counts and adds in a single atomic
step, so two concurrent requests can never both squeeze past the last slot.

**The Google quota governor** — Google rates its APIs in *units per second*,
not calls per second: a Gmail ``messages.list`` costs 5 units, a
``messages.send`` costs 100. The ceiling is 250 units/sec per user. A token
bucket in Redis models exactly that, refilled by elapsed time inside the same
Lua script that spends from it.

The bucket is split in two. Background sync gets 70 percent of the budget and
interactive work gets 30 percent, in separate buckets, so a large backfill can
never starve the person typing in the chat box. :func:`acquire_google` sleeps —
with jitter, so a hundred waiting tasks do not all wake at the same
millisecond — until its share has the units.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime
from typing import Any, Final, Literal

from redis.exceptions import RedisError

from app.core import cache
from app.core.errors import AppError
from app.core.ids import new_id
from app.core.logging import get_logger

log = get_logger(__name__)

Share = Literal["background", "interactive"]

# Google's per-user ceiling, and how we divide it.
GOOGLE_UNITS_PER_SEC: Final[float] = 250.0
SHARES: Final[dict[str, float]] = {"background": 0.70, "interactive": 0.30}

# Per-method unit costs. Gmail publishes these; Calendar and Drive are quoted
# in requests, so they are modelled at a flat cost that keeps the same bucket
# honest. Anything unlisted falls back to DEFAULT_UNITS.
DEFAULT_UNITS: Final[int] = 5
METHOD_UNITS: Final[dict[str, int]] = {
    # Gmail — published quota units
    "gmail.messages.list": 5,
    "gmail.messages.get": 5,
    "gmail.messages.attachments.get": 5,
    "gmail.messages.insert": 25,
    "gmail.messages.import": 25,
    "gmail.messages.send": 100,
    "gmail.messages.modify": 5,
    "gmail.messages.trash": 5,
    "gmail.messages.untrash": 5,
    "gmail.messages.delete": 10,
    "gmail.messages.batchModify": 50,
    "gmail.messages.batchDelete": 50,
    "gmail.drafts.create": 10,
    "gmail.drafts.update": 15,
    "gmail.drafts.send": 100,
    "gmail.drafts.list": 5,
    "gmail.drafts.get": 5,
    "gmail.drafts.delete": 10,
    "gmail.threads.list": 10,
    "gmail.threads.get": 10,
    "gmail.threads.modify": 10,
    "gmail.threads.delete": 20,
    "gmail.labels.list": 1,
    "gmail.labels.get": 1,
    "gmail.labels.create": 5,
    "gmail.history.list": 2,
    "gmail.users.getProfile": 1,
    "gmail.users.watch": 100,
    "gmail.users.stop": 50,
    # Calendar
    "gcal.events.list": 5,
    "gcal.events.get": 3,
    "gcal.events.insert": 20,
    "gcal.events.update": 20,
    "gcal.events.patch": 20,
    "gcal.events.delete": 20,
    "gcal.events.move": 20,
    "gcal.events.instances": 5,
    "gcal.freebusy.query": 10,
    "gcal.calendarList.list": 3,
    "gcal.calendars.get": 3,
    # Drive
    "gdrive.files.list": 5,
    "gdrive.files.get": 3,
    "gdrive.files.export": 20,
    "gdrive.files.create": 20,
    "gdrive.files.update": 20,
    "gdrive.files.copy": 20,
    "gdrive.files.delete": 20,
    "gdrive.permissions.create": 25,
    "gdrive.permissions.list": 5,
    "gdrive.permissions.delete": 20,
    "gdrive.changes.list": 5,
    "gdrive.about.get": 1,
}

# --- Lua ---------------------------------------------------------------------
# Sliding window: trim, count, admit, and report when the window frees a slot.
_SLIDING_WINDOW_LUA: Final[str] = """
local key      = KEYS[1]
local now_ms   = tonumber(ARGV[1])
local window   = tonumber(ARGV[2])
local limit    = tonumber(ARGV[3])
local member   = ARGV[4]
local cost     = tonumber(ARGV[5])
local consume  = tonumber(ARGV[6])

redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window)
local used = redis.call('ZCARD', key)
local allowed = 0
if used + cost <= limit then
  allowed = 1
  if consume == 1 then
    for i = 1, cost do
      redis.call('ZADD', key, now_ms, member .. '-' .. i)
    end
    used = used + cost
  end
end
redis.call('PEXPIRE', key, window)

local reset_ms = now_ms + window
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if oldest[2] then
  reset_ms = tonumber(oldest[2]) + window
end
local remaining = limit - used
if remaining < 0 then remaining = 0 end
return {allowed, remaining, reset_ms, used}
"""

# Token bucket: refill by elapsed time, then spend. One atomic step, so two
# workers cannot both read the same tokens and both spend them.
_TOKEN_BUCKET_LUA: Final[str] = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])     -- units per second
local capacity = tonumber(ARGV[2])     -- burst ceiling, in units
local now_ms   = tonumber(ARGV[3])
local want     = tonumber(ARGV[4])

if want > capacity then want = capacity end

local state  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now_ms
end

local elapsed = (now_ms - ts) / 1000.0
if elapsed < 0 then elapsed = 0 end
tokens = tokens + elapsed * rate
if tokens > capacity then tokens = capacity end

local allowed = 0
local wait_ms = 0
if tokens >= want then
  tokens = tokens - want
  allowed = 1
else
  wait_ms = math.ceil(((want - tokens) / rate) * 1000)
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
redis.call('PEXPIRE', key, math.ceil((capacity / rate) * 1000) + 5000)
return {allowed, wait_ms, math.floor(tokens)}
"""

_scripts: dict[str, tuple[Any, Any]] = {}


async def _script(name: str, body: str):
    """Register a Lua script once per client; redis-py handles EVALSHA."""
    client = await cache.get_redis()
    cached = _scripts.get(name)
    if cached is None or cached[0] is not client:
        cached = (client, client.register_script(body))
        _scripts[name] = cached
    return cached[1]


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- the per-user query limiter ---------------------------------------------


def _limit_per_hour() -> int:
    from app.config import settings

    return int(getattr(settings, "RATE_LIMIT_PER_HOUR", 100))


async def check_query_limit(
    user_id: str, cost: int = 1, consume: bool = True
) -> tuple[bool, int, datetime]:
    """Is this user allowed another query right now?

    Returns ``(allowed, remaining, reset_at)``. ``reset_at`` is when the oldest
    query in the window ages out, which is the moment a slot frees up.

    ``cost`` above 1 charges a query several slots — an expensive replan loop
    can be made to count for more than a one-shot answer.

    If Redis is unreachable the caller is let through. A cache outage must not
    take the product down; the Google quota governor is the hard ceiling that
    actually protects anything.
    """
    limit = _limit_per_hour()
    window_ms = 3_600_000
    now_ms = _now_ms()
    charge = max(1, int(cost))
    bucket = cache.key("rl", f"q:{user_id}")

    try:
        script = await _script("sliding", _SLIDING_WINDOW_LUA)
        result = await script(
            keys=[bucket],
            args=[
                now_ms,
                window_ms,
                limit,
                f"{now_ms}-{new_id()}",
                charge,
                1 if consume else 0,
            ],
        )
        allowed = bool(int(result[0]))
        remaining = int(result[1])
        reset_ms = int(result[2])
    except (RedisError, OSError) as exc:
        log.warning("ratelimit.unavailable", user_id=user_id, error=str(exc))
        return True, limit, datetime.fromtimestamp((now_ms + window_ms) / 1000, tz=UTC)

    return allowed, remaining, datetime.fromtimestamp(reset_ms / 1000, tz=UTC)


async def enforce_query_limit(user_id: str, cost: int = 1) -> tuple[int, datetime]:
    """Charge the limiter, raising ``RATE_LIMITED`` when the user is over.

    Returns ``(remaining, reset_at)`` so the caller can set the
    ``X-RateLimit-*`` headers.
    """
    allowed, remaining, reset_at = await check_query_limit(user_id, cost=cost)
    if not allowed:
        retry_after = max(1, int((reset_at - datetime.now(UTC)).total_seconds()))
        raise AppError(
            "RATE_LIMITED",
            f"That is {_limit_per_hour()} queries this hour. Try again in "
            f"{retry_after // 60 + 1} minutes.",
            details={
                "limit": _limit_per_hour(),
                "remaining": 0,
                "reset_at": reset_at.isoformat(),
                "retry_after_s": retry_after,
            },
        )
    return remaining, reset_at


async def peek_query_limit(user_id: str) -> tuple[bool, int, datetime]:
    """Read the limiter without spending a slot."""
    return await check_query_limit(user_id, cost=1, consume=False)


async def reset_query_limit(user_id: str) -> None:
    """Clear a user's window. Tests and support operations."""
    try:
        client = await cache.get_redis()
        await client.delete(cache.key("rl", f"q:{user_id}"))
    except (RedisError, OSError):
        pass


# --- the Google quota governor ----------------------------------------------


def units_for(method: str) -> int:
    """Unit cost of a Google method, e.g. ``gmail.messages.send`` -> 100."""
    if method in METHOD_UNITS:
        return METHOD_UNITS[method]
    # Tolerate a fully qualified name like "gmail.users.messages.send".
    parts = method.split(".")
    for start in range(len(parts)):
        candidate = ".".join(parts[start:])
        if candidate in METHOD_UNITS:
            return METHOD_UNITS[candidate]
        prefixed = f"{parts[0]}.{candidate}"
        if prefixed in METHOD_UNITS:
            return METHOD_UNITS[prefixed]
    return DEFAULT_UNITS


def _share_rate(share: str) -> float:
    from app.config import settings

    ceiling = float(getattr(settings, "GOOGLE_UNITS_PER_SEC", GOOGLE_UNITS_PER_SEC))
    fraction = SHARES.get(share)
    if fraction is None:
        raise AppError(
            "INTERNAL",
            f"Unknown quota share {share!r}.",
            details={"shares": sorted(SHARES)},
        )
    return ceiling * fraction


def _max_wait_s() -> float:
    from app.config import settings

    return float(getattr(settings, "GOOGLE_QUOTA_MAX_WAIT_S", 30.0))


async def try_acquire_google(
    user_id: str,
    method: str,
    units: int | None = None,
    share: Share = "interactive",
) -> tuple[bool, float, int]:
    """One non-blocking attempt.

    Returns ``(granted, wait_s, tokens_left)``. ``wait_s`` is how long until
    the bucket would hold enough — 0 when granted.
    """
    cost = int(units) if units is not None else units_for(method)
    cost = max(1, cost)
    rate = _share_rate(share)
    # A one second burst. Enough for a real batch, small enough that Google's
    # own per-second window is never blown through.
    capacity = max(rate, float(cost))
    bucket = cache.key("gq", f"{user_id}:{share}")

    try:
        script = await _script("bucket", _TOKEN_BUCKET_LUA)
        result = await script(keys=[bucket], args=[rate, capacity, _now_ms(), cost])
    except (RedisError, OSError) as exc:
        # No governor available. Pace by hand rather than hammering Google.
        log.warning("quota.unavailable", user_id=user_id, method=method,
                    error=str(exc))
        await asyncio.sleep(cost / rate)
        return True, 0.0, 0

    granted = bool(int(result[0]))
    wait_s = int(result[1]) / 1000.0
    left = int(result[2])
    return granted, wait_s, left


async def acquire_google(
    user_id: str,
    method: str,
    units: int | None = None,
    share: Share = "interactive",
) -> int:
    """Wait until this call fits inside the user's Google quota, then charge it.

    Sleeps with jitter while the bucket is dry. Raises ``RATE_LIMITED`` if the
    wait would exceed ``GOOGLE_QUOTA_MAX_WAIT_S`` — a caller that has queued
    that long should fail loudly rather than hold a request open.

    Returns the units charged, so the caller can log them.
    """
    cost = int(units) if units is not None else units_for(method)
    cost = max(1, cost)
    deadline = time.monotonic() + _max_wait_s()
    waited = 0.0

    while True:
        granted, wait_s, left = await try_acquire_google(user_id, method, cost, share)
        if granted:
            if waited > 0.25:
                log.info(
                    "quota.waited",
                    user_id=user_id,
                    method=method,
                    units=cost,
                    share=share,
                    waited_s=round(waited, 3),
                    tokens_left=left,
                )
            return cost

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AppError(
                "RATE_LIMITED",
                "Google's rate limit is full right now. Try again in a moment.",
                details={
                    "method": method,
                    "units": cost,
                    "share": share,
                    "waited_s": round(waited, 3),
                },
            )

        # Full jitter over the computed wait, floored so we never spin, and
        # capped by what is left of the deadline.
        sleep_s = min(max(random.uniform(0.02, max(wait_s, 0.05)), 0.02), remaining)
        await asyncio.sleep(sleep_s)
        waited += sleep_s


async def google_quota_state(user_id: str) -> dict[str, dict[str, float]]:
    """What both buckets hold right now. For /metrics and the sync bar."""
    out: dict[str, dict[str, float]] = {}
    now_ms = _now_ms()
    try:
        client = await cache.get_redis()
        for share in SHARES:
            rate = _share_rate(share)
            raw = await client.hgetall(cache.key("gq", f"{user_id}:{share}"))
            tokens = rate
            if raw:
                stored = raw.get(b"tokens") or raw.get("tokens")
                ts = raw.get(b"ts") or raw.get("ts")
                try:
                    tokens = float(stored)
                    elapsed = max(0.0, (now_ms - float(ts)) / 1000.0)
                    tokens = min(rate, tokens + elapsed * rate)
                except (TypeError, ValueError):
                    tokens = rate
            out[share] = {
                "rate_per_sec": round(rate, 2),
                "capacity": round(rate, 2),
                "tokens": round(tokens, 2),
                "utilisation": round(1 - (tokens / rate), 4) if rate else 0.0,
            }
    except (RedisError, OSError):
        return {
            share: {"rate_per_sec": _share_rate(share), "capacity": _share_rate(share),
                    "tokens": _share_rate(share), "utilisation": 0.0}
            for share in SHARES
        }
    return out


__all__ = [
    "Share",
    "GOOGLE_UNITS_PER_SEC",
    "SHARES",
    "METHOD_UNITS",
    "DEFAULT_UNITS",
    "units_for",
    "check_query_limit",
    "enforce_query_limit",
    "peek_query_limit",
    "reset_query_limit",
    "acquire_google",
    "try_acquire_google",
    "google_quota_state",
]
