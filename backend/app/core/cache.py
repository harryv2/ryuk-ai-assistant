"""Async Redis wrapper.

Two things live here:

* a small JSON cache with TTLs, and
* the embedding cache, keyed by the fingerprint of the text that was embedded,
  so the same email chunk is never sent to the embedding model twice.

The cache is optional by design. If Redis is down, every read misses and every
write is dropped — the caller gets ``None`` and does the real work. Nothing in
the request path fails because a cache was unreachable.

Hit and miss counters are kept both in-process (cheap) and in Redis (shared
across workers). ``/metrics`` reads :func:`stats`.
"""

from __future__ import annotations

import asyncio
import json
from array import array
from collections import defaultdict
from typing import Any, Final, Iterable, Sequence
from uuid import UUID

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.ids import canonical_json, fingerprint
from app.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_TTL: Final[int] = 300
EMBEDDING_TTL: Final[int] = 60 * 60 * 24 * 30  # a month; content_hash never lies
_METRICS_KEY: Final[str] = "metrics:cache"

_client: Redis | None = None
_lock = asyncio.Lock()

# In-process mirrors of the counters, so /metrics still says something useful
# when Redis is the thing that is broken.
_hits: defaultdict[str, int] = defaultdict(int)
_misses: defaultdict[str, int] = defaultdict(int)
_errors: defaultdict[str, int] = defaultdict(int)


# -- client -----------------------------------------------------------------


def _prefix() -> str:
    from app.config import settings

    return str(getattr(settings, "REDIS_PREFIX", "alpha"))


def key(namespace: str, name: str | UUID) -> str:
    """Fully qualified Redis key."""
    return f"{_prefix()}:{namespace}:{name}"


async def get_redis() -> Redis:
    """The shared connection pool. Created once per process."""
    global _client
    if _client is not None:
        return _client
    async with _lock:
        if _client is None:
            from app.config import settings

            url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
            _client = aioredis.from_url(
                url,
                decode_responses=False,
                health_check_interval=30,
                socket_connect_timeout=3,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=int(getattr(settings, "REDIS_MAX_CONNECTIONS", 50)),
            )
    return _client


async def close() -> None:
    """Shut the pool down. Called on app shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:  # pragma: no cover - shutdown is best effort
            pass
        _client = None


async def ping() -> bool:
    """True when Redis answers. Used by /readyz."""
    try:
        client = await get_redis()
        return bool(await client.ping())
    except (RedisError, OSError):
        return False


# -- counters ---------------------------------------------------------------


async def _bump(field: str, namespace: str, amount: int = 1) -> None:
    if amount <= 0:
        return
    try:
        client = await get_redis()
        pipe = client.pipeline(transaction=False)
        pipe.hincrby(key("m", _METRICS_KEY), field, amount)
        pipe.hincrby(key("m", _METRICS_KEY), f"{field}:{namespace}", amount)
        await pipe.execute()
    except (RedisError, OSError):
        pass


async def record_hit(namespace: str) -> None:
    _hits[namespace] += 1
    await _bump("hits", namespace)


async def record_miss(namespace: str) -> None:
    _misses[namespace] += 1
    await _bump("misses", namespace)


def _record_error(namespace: str, exc: Exception) -> None:
    _errors[namespace] += 1
    if _errors[namespace] in (1, 10, 100, 1000):
        log.warning("cache.unavailable", namespace=namespace, error=str(exc))


async def stats() -> dict[str, Any]:
    """Counters for /metrics: totals, per namespace, and the hit rate."""
    shared: dict[str, int] = {}
    try:
        client = await get_redis()
        raw = await client.hgetall(key("m", _METRICS_KEY))
        for field, value in raw.items():
            name = field.decode() if isinstance(field, bytes) else str(field)
            try:
                shared[name] = int(value)
            except (TypeError, ValueError):
                continue
    except (RedisError, OSError):
        shared = {}

    hits = shared.get("hits", sum(_hits.values()))
    misses = shared.get("misses", sum(_misses.values()))
    total = hits + misses
    namespaces = sorted(set(_hits) | set(_misses) | {
        name.split(":", 1)[1]
        for name in shared
        if name.startswith(("hits:", "misses:")) and ":" in name
    })
    per_namespace = {
        ns: {
            "hits": shared.get(f"hits:{ns}", _hits.get(ns, 0)),
            "misses": shared.get(f"misses:{ns}", _misses.get(ns, 0)),
        }
        for ns in namespaces
    }
    return {
        "hits": hits,
        "misses": misses,
        "errors": sum(_errors.values()),
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "namespaces": per_namespace,
    }


async def reset_stats() -> None:
    """Zero the counters. Tests and manual operations only."""
    _hits.clear()
    _misses.clear()
    _errors.clear()
    try:
        client = await get_redis()
        await client.delete(key("m", _METRICS_KEY))
    except (RedisError, OSError):
        pass


# -- JSON cache -------------------------------------------------------------


async def get_json(name: str, namespace: str = "kv") -> Any | None:
    """Read a JSON value. ``None`` means miss, unreachable, or a bad payload."""
    try:
        client = await get_redis()
        raw = await client.get(key(namespace, name))
    except (RedisError, OSError) as exc:
        _record_error(namespace, exc)
        await record_miss(namespace)
        return None
    if raw is None:
        await record_miss(namespace)
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        await record_miss(namespace)
        return None
    await record_hit(namespace)
    return value


async def set_json(
    name: str,
    value: Any,
    ttl: int | None = DEFAULT_TTL,
    namespace: str = "kv",
) -> bool:
    """Write a JSON value with a TTL. False when the write did not land."""
    try:
        payload = canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        log.warning("cache.unserialisable", namespace=namespace, error=str(exc))
        return False
    try:
        client = await get_redis()
        if ttl and ttl > 0:
            await client.set(key(namespace, name), payload, ex=int(ttl))
        else:
            await client.set(key(namespace, name), payload)
        return True
    except (RedisError, OSError) as exc:
        _record_error(namespace, exc)
        return False


async def delete(name: str, namespace: str = "kv") -> bool:
    try:
        client = await get_redis()
        return bool(await client.delete(key(namespace, name)))
    except (RedisError, OSError) as exc:
        _record_error(namespace, exc)
        return False


async def incr(name: str, amount: int = 1, ttl: int | None = None,
               namespace: str = "counter") -> int:
    """Increment a counter, returning the new value (0 when Redis is down)."""
    try:
        client = await get_redis()
        full = key(namespace, name)
        value = await client.incrby(full, amount)
        if ttl and value == amount:
            await client.expire(full, int(ttl))
        return int(value)
    except (RedisError, OSError) as exc:
        _record_error(namespace, exc)
        return 0


# -- embedding cache --------------------------------------------------------


def embedding_fingerprint(model: str, text: str) -> UUID:
    """The cache key for one piece of text under one embedding model."""
    return fingerprint(f"embedding.{model}", text)


def _pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(raw: bytes) -> list[float]:
    out = array("f")
    out.frombytes(raw)
    return out.tolist()


async def get_embedding(model: str, text: str) -> list[float] | None:
    """The cached vector for this text, or None."""
    return await get_embedding_by_fp(embedding_fingerprint(model, text))


async def get_embedding_by_fp(fp: UUID | str) -> list[float] | None:
    """The cached vector for a fingerprint already computed by the caller."""
    try:
        client = await get_redis()
        raw = await client.get(key("emb", fp))
    except (RedisError, OSError) as exc:
        _record_error("emb", exc)
        await record_miss("emb")
        return None
    if not raw:
        await record_miss("emb")
        return None
    try:
        vector = _unpack(raw)
    except ValueError:
        await record_miss("emb")
        return None
    await record_hit("emb")
    return vector


async def set_embedding(
    model: str, text: str, vector: Sequence[float], ttl: int = EMBEDDING_TTL
) -> bool:
    return await set_embedding_by_fp(embedding_fingerprint(model, text), vector, ttl)


async def set_embedding_by_fp(
    fp: UUID | str, vector: Sequence[float], ttl: int = EMBEDDING_TTL
) -> bool:
    try:
        client = await get_redis()
        await client.set(key("emb", fp), _pack(vector), ex=int(ttl))
        return True
    except (RedisError, OSError) as exc:
        _record_error("emb", exc)
        return False


async def get_embeddings(model: str, texts: Sequence[str]) -> list[list[float] | None]:
    """Batch read, in the same order as ``texts``. One round trip."""
    if not texts:
        return []
    fps = [embedding_fingerprint(model, t) for t in texts]
    try:
        client = await get_redis()
        rows = await client.mget([key("emb", fp) for fp in fps])
    except (RedisError, OSError) as exc:
        _record_error("emb", exc)
        _misses["emb"] += len(texts)
        return [None] * len(texts)

    out: list[list[float] | None] = []
    for raw in rows:
        vector: list[float] | None = None
        if raw:
            try:
                vector = _unpack(raw)
            except ValueError:
                vector = None
        out.append(vector)

    hit_count = sum(1 for v in out if v is not None)
    miss_count = len(out) - hit_count
    _hits["emb"] += hit_count
    _misses["emb"] += miss_count
    await _bump("hits", "emb", hit_count)
    await _bump("misses", "emb", miss_count)
    return out


async def set_embeddings(
    model: str,
    pairs: Iterable[tuple[str, Sequence[float]]],
    ttl: int = EMBEDDING_TTL,
) -> int:
    """Batch write. Returns how many were stored."""
    items = list(pairs)
    if not items:
        return 0
    try:
        client = await get_redis()
        pipe = client.pipeline(transaction=False)
        for text, vector in items:
            pipe.set(key("emb", embedding_fingerprint(model, text)), _pack(vector),
                     ex=int(ttl))
        await pipe.execute()
        return len(items)
    except (RedisError, OSError) as exc:
        _record_error("emb", exc)
        return 0


__all__ = [
    "DEFAULT_TTL",
    "EMBEDDING_TTL",
    "get_redis",
    "close",
    "ping",
    "key",
    "get_json",
    "set_json",
    "delete",
    "incr",
    "stats",
    "reset_stats",
    "record_hit",
    "record_miss",
    "embedding_fingerprint",
    "get_embedding",
    "get_embedding_by_fp",
    "set_embedding",
    "set_embedding_by_fp",
    "get_embeddings",
    "set_embeddings",
]
