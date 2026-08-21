"""Fixtures for the integration suite.

What is real here: Postgres, the Alembic migration, the SQLAlchemy models, the
repositories, the FastAPI app and every line of orchestration between them.

What is not: the network. ``respx`` intercepts httpx, so every call to Google
and to OpenAI is served from ``tests/fixtures``. An unrecognised URL is a loud
failure naming the path, never a real socket. Redis is an in-process fake for
the same reason.

Three fixtures carry most of the weight:

``google``
    Records every Google request and lets a test break one service on purpose —
    ``google.fail("gcal", 503)``, ``google.rate_limit("gmail")``,
    ``google.expire_auth(refresh_ok=False)``. ``google.mutations`` is the list
    of calls that changed something in the user's account, which is how
    "nothing reached Google" is asserted rather than assumed.

``llm``
    Serves canned plans and counts calls. ``llm.completions`` is the honest LLM
    call count — embeddings are counted separately, exactly as
    ``docs/SAMPLE_QUERIES.md`` counts them.

``client``
    An ASGI client over the real app with the auth dependency overridden for a
    seeded user. ASGI traffic does not go through httpx's transport, so respx
    never sees it and the app's own outbound calls are still intercepted.

The modules under test are being written in parallel against
``docs/contracts.md``. Where one is not on disk yet, the fixture that needs it
skips with the import error attached. A skip here means "not built yet"; it
never means "passed".
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import inspect
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import pytest

# --------------------------------------------------------------------------- #
# Environment, before anything imports app.config
# --------------------------------------------------------------------------- #

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

#: 32 bytes, base64. The same key `tests/conftest.py` uses.
TEST_TOKEN_KEY = "3q2+796tvu/erb7v3q2+796tvu/erb7v3q2+796tvu8="

os.environ.setdefault("ENV", "test")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")
os.environ.setdefault("OPENAI_MODEL", "gpt-4.1-mini")
os.environ.setdefault("OPENAI_EMBED_MODEL", "text-embedding-3-small")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("SESSION_SECRET", "test-session-secret")
os.environ.setdefault("TOKEN_ENCRYPTION_KEY", TEST_TOKEN_KEY)
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/15")
# Every connection is closed when its session ends, so a pool cannot outlive the
# event loop that made it. Integration tests create a loop per test.
os.environ["DB_NULL_POOL"] = "1"

TEST_DB_URL = os.environ.get("DATABASE_URL_TEST") or os.environ.get("TEST_DATABASE_URL")
if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL

SKIP_NO_DB = (
    "DATABASE_URL_TEST is not set. The integration suite needs a real Postgres "
    "with pgvector — the generated columns, the HNSW indexes and the partial "
    "unique index on actions.dedupe_key cannot be exercised against anything "
    "else. Start one and point at it, e.g.\n"
    "  docker compose up -d db\n"
    "  DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5432/"
    "orchestrator_test pytest tests/integration"
)

# --------------------------------------------------------------------------- #
# Fixture payloads
# --------------------------------------------------------------------------- #

try:  # `tests` is a package, but tolerate a bare `tests/` on sys.path too
    from tests.fixtures import google_responses as gr
    from tests.fixtures import llm_responses as lr
except ImportError:  # pragma: no cover - layout fallback
    from fixtures import google_responses as gr  # type: ignore[no-redef]
    from fixtures import llm_responses as lr  # type: ignore[no-redef]

FROZEN_NOW = gr.NOW


# --------------------------------------------------------------------------- #
# Small helpers, importable from the test modules
# --------------------------------------------------------------------------- #


def _import(path: str):
    """Import a module, or raise ImportError with the path in the message."""
    return importlib.import_module(path)


def require(path: str, *names: str):
    """Import a module the suite depends on, or skip with the reason.

    The contract fixes the module path. A missing module means that part of the
    system is not written yet, and the honest report is a skip that says which
    import failed — not a green test.
    """
    try:
        module = _import(path)
    except ImportError as exc:
        pytest.skip(f"{path} is not importable yet: {exc}")
    for name in names:
        if not hasattr(module, name):
            pytest.skip(f"{path} has no {name!r} yet")
    return module


def first_attr(module, *names: str, default=None):
    """The first of ``names`` that exists on ``module``."""
    for name in names:
        found = getattr(module, name, None)
        if found is not None:
            return found
    return default


def blob(value: Any) -> str:
    """Everything an object says, as one lower-case string, for keyword checks."""
    try:
        return json.dumps(value, default=str).lower()
    except (TypeError, ValueError):
        return str(value).lower()


def mentions(value: Any, *keywords: str) -> bool:
    text = blob(value)
    return any(k.lower() in text for k in keywords)


def parse_ts(value: Any) -> datetime | None:
    """A timestamp from JSON or from the database, as tz-aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass
class Span:
    """One step's wall-clock window, from ``node_executions``."""

    node_id: str
    op: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    depends_on: tuple[str, ...] = ()

    def overlaps(self, other: "Span") -> bool:
        if not (self.started_at and self.finished_at and other.started_at and other.finished_at):
            return False
        return max(self.started_at, other.started_at) < min(self.finished_at, other.finished_at)

    def __str__(self) -> str:  # pragma: no cover - only used in failure output
        return (
            f"{self.node_id} ({self.op}, {self.status}) "
            f"{self.started_at} -> {self.finished_at}"
        )


def spans_of(rows: Iterable[Any]) -> dict[str, Span]:
    """``{node_id: Span}`` from node_execution rows or from response steps."""
    out: dict[str, Span] = {}
    for row in rows:
        get = row.get if isinstance(row, dict) else lambda n, d=None: getattr(row, n, d)
        node_id = get("node_id")
        if not node_id:
            continue
        span = Span(
            node_id=str(node_id),
            op=str(get("op", "")),
            status=str(getattr(get("status", ""), "value", get("status", ""))),
            started_at=parse_ts(get("started_at")),
            finished_at=parse_ts(get("finished_at")),
            depends_on=tuple(get("depends_on") or ()),
        )
        # A replan writes a new round; the later row is the one that counts.
        out[span.node_id] = span
    return out


def assert_ran_concurrently(spans: dict[str, Span], a: str, b: str) -> None:
    """The actual parallelism proof: two steps whose windows overlap.

    Both were started before either finished. Run one after the other and this
    is false however fast they were, which is why it is the assertion and
    "it looked quick" is not.
    """
    missing = [n for n in (a, b) if n not in spans]
    assert not missing, f"no such step(s) {missing}; ran: {sorted(spans)}"
    first, second = spans[a], spans[b]
    assert first.overlaps(second), (
        "expected these two steps to run at the same time, but their windows do "
        f"not overlap:\n  {first}\n  {second}"
    )


def assert_ran_in_order(spans: dict[str, Span], before: str, after: str) -> None:
    """A dependency edge, honoured: the dependant did not start early."""
    missing = [n for n in (before, after) if n not in spans]
    assert not missing, f"no such step(s) {missing}; ran: {sorted(spans)}"
    first, second = spans[before], spans[after]
    assert first.finished_at and second.started_at, (
        f"one of these never ran:\n  {first}\n  {second}"
    )
    assert second.started_at >= first.finished_at, (
        f"{after} started before {before} finished:\n  {first}\n  {second}"
    )


def degraded_services(payload: dict[str, Any]) -> list[str]:
    """Service names out of a query response's degraded block.

    ``docs/API.md`` calls the field ``degraded`` and fills it with
    ``{"service": ..., "reason": ...}``. A plain list of names is accepted too,
    and so is ``degraded_services``, because that detail is not worth a failing
    test either way.
    """
    raw = payload.get("degraded")
    if raw is None:
        raw = payload.get("degraded_services")
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = list(raw.values()) or list(raw.keys())
    out: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            name = item.get("service") or item.get("name")
            if name:
                out.append(str(name))
    return out


def answer_text(payload: dict[str, Any]) -> str:
    """Everything the user would read, flattened."""
    parts: list[str] = []
    if isinstance(payload.get("text"), str):
        parts.append(payload["text"])
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        data = block.get("data") or {}
        for key in ("markdown", "text", "body"):
            if isinstance(data.get(key), str):
                parts.append(data[key])
    for card in payload.get("pending_inputs") or []:
        prompt = (card or {}).get("prompt") or {}
        for key in ("question", "help_text"):
            if isinstance(prompt.get(key), str):
                parts.append(prompt[key])
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Postgres
# --------------------------------------------------------------------------- #


def _sync_url(url: str) -> str:
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def _async_url(url: str) -> str:
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    return url


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test DSN. Skips the whole suite when it is not configured.

    Guarded: the fixture drops and rebuilds the schema, so it refuses to run
    against a database whose name does not look like a test database.
    """
    if not TEST_DB_URL:
        pytest.skip(SKIP_NO_DB)
    name = urlsplit(TEST_DB_URL).path.lstrip("/")
    if "test" not in name.lower():
        pytest.skip(
            f"refusing to run against database {name!r}: this suite drops and "
            "rebuilds the schema, so the database name has to contain 'test'"
        )
    return TEST_DB_URL


@pytest.fixture(scope="session")
def migrated_database(database_url: str) -> str:
    """A database at the head migration, built once for the session.

    The migration is what is under test here as much as the models are: it is
    the thing that creates the generated ``tsv`` columns, the
    ``attendee_email_list`` function behind ``sync_gcal.attendee_emails``, the
    HNSW indexes, and the partial unique index on ``actions.dedupe_key``.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg", reason="alembic runs over psycopg, not asyncpg")

    # app.config may already be imported by an earlier test module; point the
    # live settings object at the test database as well as the environment.
    os.environ["DATABASE_URL"] = database_url
    with contextlib.suppress(ImportError):
        from app.config import get_settings, settings

        settings.DATABASE_URL = database_url
        get_settings.cache_clear()

    engine = sqlalchemy.create_engine(_sync_url(database_url), future=True)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP SCHEMA IF EXISTS public CASCADE")
            connection.exec_driver_sql("CREATE SCHEMA public")
    except Exception as exc:  # pragma: no cover - a dead database is a skip
        engine.dispose()
        pytest.skip(f"cannot reach {database_url}: {exc}")
    engine.dispose()

    from alembic import command
    from alembic.config import Config

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", _sync_url(database_url).replace("%", "%%"))
    command.upgrade(config, "head")
    return database_url


@pytest.fixture
async def engine(migrated_database: str):
    """A per-test async engine. NullPool, so nothing survives the event loop."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    made = create_async_engine(_async_url(migrated_database), poolclass=NullPool, future=True)
    try:
        yield made
    finally:
        await made.dispose()


@pytest.fixture(autouse=True)
async def clean_database(engine):
    """Empty every table before each test, and reset the app's engine.

    Truncate rather than a rolled-back transaction: the code under test commits
    on its own, and a test that cannot see its own writes is worse than useless.
    """
    from sqlalchemy import text

    async with engine.begin() as connection:
        rows = await connection.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
            )
        )
        tables = [r[0] for r in rows]
        if tables:
            await connection.execute(
                text("TRUNCATE " + ", ".join(f'"{t}"' for t in tables) + " RESTART IDENTITY CASCADE")
            )

    await _reset_app_engine()
    yield
    await _reset_app_engine()


async def _reset_app_engine() -> None:
    """Dispose the app's engine, then drop it, so the next test builds a fresh one.

    Nulling the reference without disposing leaks the asyncpg pool: its
    connections stay open, bound to an event loop that pytest-asyncio has since
    closed. Nothing fails immediately — the next test builds its own engine and
    passes — so the damage shows up later, as tests that pass alone and fail in
    a full run. That reads as flakiness and sends you looking in the wrong file.

    `dispose()` is IO, so this has to be awaited rather than done in a sync
    helper, which is why the fixture that calls it is async.
    """
    with contextlib.suppress(ImportError):
        from app.db import session as db_session

        engine = getattr(db_session, "_engine", None)
        if engine is not None:
            with contextlib.suppress(Exception):
                await engine.dispose()
        db_session._engine = None
        db_session._sessionmaker = None


@pytest.fixture
async def db(engine):
    """A session for the test's own reads and writes. Commits are the caller's."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


# --------------------------------------------------------------------------- #
# Redis, in process
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Enough Redis for the cache and the counters, with no socket.

    The Lua scripts — the sliding-window query limiter and the Google token
    bucket — are answered "granted". Both return ``[allowed, n, n]``, so one
    permissive shape covers them. A test that wants to see a limiter say no
    should drive ``app.core.ratelimit`` directly rather than through this.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.hashes: dict[str, dict[str, Any]] = {}
        self.expiries: dict[str, float] = {}
        #: channel -> the pubsubs currently listening on it
        self.channels: dict[str, list[_FakePubSub]] = {}
        #: stream key -> [(entry_id, {field: value})]
        self.streams: dict[str, list[tuple[bytes, dict[bytes, bytes]]]] = {}
        self._seq = 0

    # -- strings ---------------------------------------------------------- #
    async def get(self, key):
        return self.store.get(_k(key))

    async def getdel(self, key):
        return self.store.pop(_k(key), None)

    async def setex(self, key, seconds, value):
        return await self.set(key, value, ex=seconds)

    async def set(self, key, value, ex=None, px=None, nx=False, xx=False):
        key = _k(key)
        if nx and key in self.store:
            return None
        if xx and key not in self.store:
            return None
        self.store[key] = value if isinstance(value, bytes) else str(value).encode()
        return True

    async def mget(self, keys):
        return [self.store.get(_k(k)) for k in keys]

    async def delete(self, *keys):
        gone = 0
        for key in keys:
            gone += 1 if self.store.pop(_k(key), None) is not None else 0
            self.hashes.pop(_k(key), None)
        return gone

    async def exists(self, *keys):
        return sum(1 for k in keys if _k(k) in self.store)

    async def expire(self, key, seconds):
        return True

    async def incrby(self, key, amount=1):
        key = _k(key)
        current = int(self.store.get(key, b"0"))
        current += int(amount)
        self.store[key] = str(current).encode()
        return current

    incr = incrby

    # -- hashes ----------------------------------------------------------- #
    async def hincrby(self, key, field, amount=1):
        bucket = self.hashes.setdefault(_k(key), {})
        bucket[_s(field)] = bucket.get(_s(field), 0) + int(amount)
        return bucket[_s(field)]

    async def hgetall(self, key):
        return {
            k.encode(): str(v).encode() for k, v in self.hashes.get(_k(key), {}).items()
        }

    async def hget(self, key, field):
        bucket = self.hashes.get(_k(key), {})
        if _s(field) not in bucket:
            return None
        return str(bucket[_s(field)]).encode()

    async def hmget(self, key, *fields):
        # redis-py takes either hmget(key, "a", "b") or hmget(key, ["a", "b"]).
        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = tuple(fields[0])
        bucket = self.hashes.get(_k(key), {})
        return [
            str(bucket[_s(f)]).encode() if _s(f) in bucket else None for f in fields
        ]

    async def hset(self, key, field=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(_k(key), {})
        written = 0
        if field is not None:
            written += 1 if _s(field) not in bucket else 0
            bucket[_s(field)] = _s(value)
        for k, v in (mapping or {}).items():
            written += 1 if _s(k) not in bucket else 0
            bucket[_s(k)] = _s(v)
        return written

    async def hdel(self, key, *fields):
        bucket = self.hashes.get(_k(key), {})
        return sum(1 for f in fields if bucket.pop(_s(f), None) is not None)

    # -- streams, the SSE replay buffer ------------------------------------ #
    async def xadd(self, key, fields, maxlen=None, approximate=True, id="*"):
        self._seq += 1
        entry_id = f"{int(datetime.now(tz=UTC).timestamp() * 1000)}-{self._seq}".encode()
        entries = self.streams.setdefault(_k(key), [])
        entries.append(
            (
                entry_id,
                {
                    (k if isinstance(k, bytes) else str(k).encode()): (
                        v if isinstance(v, bytes) else str(v).encode()
                    )
                    for k, v in fields.items()
                },
            )
        )
        if maxlen is not None and len(entries) > maxlen:
            del entries[: len(entries) - maxlen]
        return entry_id

    async def xrange(self, key, min="-", max="+", count=None):
        entries = list(self.streams.get(_k(key), []))
        return entries[:count] if count else entries

    # -- pub/sub, the live SSE channel ------------------------------------- #
    async def publish(self, channel, message):
        payload = message if isinstance(message, bytes) else str(message).encode()
        listeners = self.channels.get(_k(channel), [])
        for pubsub in listeners:
            pubsub.deliver(_k(channel), payload)
        return len(listeners)

    def pubsub(self, ignore_subscribe_messages=False):
        return _FakePubSub(self)

    # -- sorted sets, used by the sliding window when Lua is unavailable --- #
    async def zadd(self, key, mapping):
        return len(mapping)

    async def zcard(self, key):
        return 0

    async def zremrangebyscore(self, key, minimum, maximum):
        return 0

    # -- plumbing --------------------------------------------------------- #
    async def ping(self):
        return True

    async def aclose(self):
        return None

    close = aclose

    def pipeline(self, transaction=False):
        return _FakePipeline(self)

    def register_script(self, body):
        """Answer a Lua script in the shape that script's caller reads.

        Every script here says "go ahead", but they do not all say it in the
        same shape, and a caller that reads a field the reply does not have is
        a crash rather than a permissive answer. So the reply is chosen by
        which script this is: the two circuit breakers in `app/google/retry.py`
        return four values, the sliding-window limiters return three.
        """
        source = body if isinstance(body, str) else str(body)

        async def run(keys=(), args=(), client=None):
            if "'closed'" in source:  # retry._ALLOW_LUA -> {allowed, state, ms, failures}
                return [1, b"closed", 0, 0]
            if "threshold" in source and "open_until" in source:
                # retry._FAIL_LUA -> {opened, failures, open_until, open_ms}
                return [0, 1, 0, 0]
            # The limiters: {allowed, remaining, reset_at_ms}
            return [1, 99, int(datetime.now(tz=UTC).timestamp() * 1000) + 3_600_000]

        return run


class _FakePubSub:
    """One subscriber. `publish` pushes straight onto its queue, no socket."""

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.subscribed: set[str] = set()
        self.queue: list[dict[str, Any]] = []

    def deliver(self, channel: str, payload: bytes) -> None:
        self.queue.append(
            {"type": "message", "channel": channel.encode(), "data": payload}
        )

    async def subscribe(self, *channels):
        for channel in channels:
            name = _k(channel)
            self.subscribed.add(name)
            self.redis.channels.setdefault(name, []).append(self)

    async def unsubscribe(self, *channels):
        names = [_k(c) for c in channels] or list(self.subscribed)
        for name in names:
            self.subscribed.discard(name)
            listeners = self.redis.channels.get(name, [])
            if self in listeners:
                listeners.remove(self)

    async def get_message(self, ignore_subscribe_messages=False, timeout=None):
        # Nothing buffered means nothing is coming: the fake has no background
        # producer, so returning None immediately is the honest answer and lets
        # the caller's idle timeout do its job without a real sleep.
        return self.queue.pop(0) if self.queue else None

    async def aclose(self):
        await self.unsubscribe()

    close = aclose

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


class _FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.queue: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.queue.append((name, args, kwargs))
            return self

        return record

    async def execute(self):
        out = []
        for name, args, kwargs in self.queue:
            out.append(await getattr(self.redis, name)(*args, **kwargs))
        self.queue.clear()
        return out

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def _k(key) -> str:
    return key.decode() if isinstance(key, bytes) else str(key)


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@pytest.fixture(autouse=True)
def redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Replace the Redis client everywhere `app.core.cache` hands it out."""
    fake = FakeRedis()

    try:
        from app.core import cache
    except ImportError:  # pragma: no cover - cache is core, but be honest
        return fake

    async def get_redis():
        return fake

    monkeypatch.setattr(cache, "_client", fake, raising=False)
    monkeypatch.setattr(cache, "get_redis", get_redis)

    with contextlib.suppress(ImportError):
        from app.core import ratelimit

        monkeypatch.setattr(ratelimit, "_scripts", {}, raising=False)

    return fake


# --------------------------------------------------------------------------- #
# The network, intercepted
# --------------------------------------------------------------------------- #


@dataclass
class Recorded:
    """One intercepted request."""

    method: str
    url: str
    path: str
    params: dict[str, str]
    headers: dict[str, str]
    body: bytes | None
    service: str
    status: int = 0

    @property
    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body)
        except (ValueError, UnicodeDecodeError):
            return None

    @property
    def authorized(self) -> bool:
        return "bearer" in (self.headers.get("authorization", "").lower())

    def __str__(self) -> str:  # pragma: no cover - failure output only
        return f"{self.method} {self.path}"


#: Requests that change something in the user's Google account. "Nothing
#: reached Google" means this list is empty — a search or a read does not count.
_MUTATION_PATHS = (
    "/messages/send",
    "/drafts/send",
    "/sendmail",
    "/permissions",
)


def _service_of(path: str) -> str:
    if "/gmail/" in path:
        return "gmail"
    if "/calendar/" in path:
        return "gcal"
    if "/drive/" in path:
        return "gdrive"
    if path.endswith("/token") or "oauth" in path:
        return "oauth"
    return "other"


def _is_mutation(record: Recorded) -> bool:
    if record.service == "oauth":
        return False
    if any(p in record.path for p in _MUTATION_PATHS):
        return True
    if record.method in {"POST", "PUT", "PATCH", "DELETE"}:
        # Creating or updating a draft is deliberately not a mutation: a draft
        # is reversible and is created *before* the confirm card so the person
        # can see what they are approving. Deleting one is the "Not now" path.
        if "/drafts" in record.path:
            return False
        return True
    return False


@dataclass
class _Failure:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    times: int | None = None  # None = until restored
    method: str | None = None
    contains: str | None = None

    def matches(self, record: "Recorded") -> bool:
        if self.times is not None and self.times <= 0:
            return False
        if self.method and record.method != self.method.upper():
            return False
        if self.contains and self.contains not in record.path:
            return False
        return True

    def take(self) -> None:
        if self.times is not None:
            self.times -= 1


class GoogleMock:
    """Recorded Google, plus the switches the reliability tests need."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self._failures: dict[str, list[_Failure]] = {}
        self._auth_expired = False
        self._refresh_ok = True
        self._refreshes = 0

    # -- reading what happened -------------------------------------------- #
    def calls(self, service: str | None = None, *, method: str | None = None,
              contains: str | None = None) -> list[Recorded]:
        out = self.requests
        if service:
            out = [r for r in out if r.service == service]
        if method:
            out = [r for r in out if r.method == method.upper()]
        if contains:
            out = [r for r in out if contains in r.path]
        return list(out)

    @property
    def mutations(self) -> list[Recorded]:
        """Everything that changed the user's account. Should be empty until
        somebody presses a button."""
        return [r for r in self.requests if _is_mutation(r)]

    @property
    def sends(self) -> list[Recorded]:
        return [r for r in self.requests if "send" in r.path]

    @property
    def refreshes(self) -> int:
        return self._refreshes

    def count(self, service: str | None = None, **kwargs) -> int:
        return len(self.calls(service, **kwargs))

    # -- breaking things on purpose --------------------------------------- #
    def fail(self, service: str, status: int = 503, *, body: Any = None,
             headers: dict[str, str] | None = None, times: int | None = None,
             method: str | None = None, contains: str | None = None) -> None:
        """Calls to ``service`` return ``status`` until restored.

        ``times`` limits it to the next N matching calls, which is how "fails
        twice then recovers" is written. ``method`` and ``contains`` narrow it to
        one operation — breaking a calendar *update* while leaving reads alone,
        for instance, which is what a 412 from someone else's edit looks like.
        """
        chosen = body if body is not None else _DEFAULT_ERROR_BODIES.get(status)
        self._failures.setdefault(service, []).append(
            _Failure(
                status=status,
                body=chosen,
                headers=headers or {},
                times=times,
                method=method,
                contains=contains,
            )
        )

    def rate_limit(self, service: str, *, retry_after: int = 2, times: int | None = None) -> None:
        """429 with a Retry-After, which the backoff is meant to honour."""
        self.fail(
            service,
            429,
            body=gr.ERROR_429,
            headers={"Retry-After": str(retry_after)},
            times=times,
        )

    def expire_auth(self, *, refresh_ok: bool = True) -> None:
        """Every API call 401s. With ``refresh_ok=False`` the refresh grant is
        dead too, which is the reconnect case rather than the retry case."""
        self._auth_expired = True
        self._refresh_ok = refresh_ok

    def restore(self, service: str | None = None) -> None:
        if service is None:
            self._failures.clear()
            self._auth_expired = False
            self._refresh_ok = True
        else:
            self._failures.pop(service, None)

    # -- the handler ------------------------------------------------------ #
    def handle(self, record: Recorded):
        import httpx

        if record.service == "oauth" and record.path.endswith("/token"):
            self._refreshes += 1
            if not self._refresh_ok:
                record.status = 400
                return httpx.Response(400, json=gr.ERROR_INVALID_GRANT)
            self._auth_expired = False
            record.status = 200
            return httpx.Response(200, json=gr.TOKEN_REFRESHED)

        if self._auth_expired and record.service in {"gmail", "gcal", "gdrive"}:
            record.status = 401
            return httpx.Response(401, json=gr.ERROR_401)

        pending = self._failures.get(record.service) or []
        for failure in list(pending):
            if failure.times is not None and failure.times <= 0:
                pending.remove(failure)
                continue
            if not failure.matches(record):
                continue
            failure.take()
            record.status = failure.status
            return httpx.Response(
                failure.status, json=failure.body, headers=failure.headers
            )

        answer = gr.resolve(record.method, record.path, record.params, record.body)
        if answer is None:
            raise AssertionError(
                f"no recorded Google payload for {record.method} {record.path}\n"
                "Add one to tests/fixtures/google_responses.py — a test must "
                "never fall through to the network."
            )
        status, body = answer
        record.status = status
        if body is None:
            return httpx.Response(status)
        if isinstance(body, dict) and "_text" in body:
            return httpx.Response(status, text=body["_text"])
        return httpx.Response(status, json=body)


_DEFAULT_ERROR_BODIES = {
    401: gr.ERROR_401,
    410: gr.ERROR_410,
    412: gr.ERROR_412,
    429: gr.ERROR_429,
    503: gr.ERROR_503,
}


#: The heading `app/orchestrator/prompts.py` puts the turn's own question under.
_QUERY_HEADINGS = ("THE QUERY", "THE ORIGINAL QUERY")


def _asked_question(payload: dict[str, Any]) -> str:
    """The question this turn is asking, without the cached system prefix.

    Scenario matching has to see the query and nothing else. The planner's
    system prompt is a stable, cacheable prefix that carries worked examples —
    one of them is literally `query: "move the meeting with John"` — so matching
    over the whole prompt picks that example for every query that ever runs.
    The volatile half puts the real question under `THE QUERY`, which is the
    anchor used here; failing that, the last user message.
    """
    messages = [m for m in payload.get("messages", []) if isinstance(m, dict)]
    user_text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "user"
    )
    for heading in _QUERY_HEADINGS:
        head, sep, rest = user_text.partition(heading)
        if not sep:
            continue
        # The query is the indented block that follows, up to the next blank
        # line or the next all-caps heading.
        out: list[str] = []
        for line in rest.splitlines()[1:]:
            if not line.strip():
                break
            if line[:1] not in {" ", "\t"}:
                break
            out.append(line.strip())
        if out:
            return "\n".join(out)
    return user_text


class LLMMock:
    """Canned plans and prose, plus the only LLM call count that matters."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.completions: list[dict[str, Any]] = []
        self.embeddings: list[dict[str, Any]] = []
        self.served: list[Any] = []
        self._pinned: list[str] = []
        self._error: tuple[int, Any] | None = None
        self._scenario: str | None = None

    # -- counts ----------------------------------------------------------- #
    @property
    def calls(self) -> int:
        """Completion calls. The probe's embedding is not an LLM call — that is
        how ``docs/SAMPLE_QUERIES.md`` counts, and how the budget is spent."""
        return len(self.completions)

    @property
    def embedding_calls(self) -> int:
        return len(self.embeddings)

    def prompt(self, index: int = -1) -> str:
        """Everything sent as messages on one completion call, concatenated."""
        if not self.completions:
            return ""
        body = self.completions[index]
        return "\n".join(
            str(m.get("content", "")) for m in body.get("messages", []) if isinstance(m, dict)
        )

    @property
    def prompts(self) -> list[str]:
        return [self.prompt(i) for i in range(len(self.completions))]

    def json_calls(self) -> list[dict[str, Any]]:
        """Planner-shaped calls: the ones asking for a JSON object back."""
        return [c for c in self.completions if c.get("response_format")]

    def prose_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.completions if not c.get("response_format")]

    # -- steering --------------------------------------------------------- #
    def use(self, *scenarios: str) -> None:
        """Pin the next completion(s) to a named scenario.

        Needed where two scenarios share a query — #5 and #11 are the same
        sentence, and only the injected Calendar failure tells them apart.
        """
        for name in scenarios:
            if name not in lr.PLANS:
                raise KeyError(f"no such scenario {name!r}; have {sorted(lr.PLANS)}")
        self._pinned.extend(scenarios)

    def fail(self, status: int = 503, body: Any = None) -> None:
        self._error = (status, body or {"error": {"message": "model unavailable"}})

    # -- the handler ------------------------------------------------------ #
    def handle(self, record: Recorded):
        import httpx

        payload = record.json or {}

        if record.path.endswith("/embeddings"):
            self.embeddings.append(payload)
            record.status = 200
            body = lr.embeddings_response(
                payload.get("input", []),
                model=payload.get("model", lr.DEFAULT_EMBED_MODEL),
                dim=int(payload.get("dimensions") or lr.EMBED_DIM),
            )
            return httpx.Response(200, json=body)

        if not record.path.endswith("/chat/completions"):
            raise AssertionError(
                f"unexpected OpenAI endpoint {record.path}. The system makes four "
                "calls and they all go to /chat/completions or /embeddings."
            )

        self.completions.append(payload)
        if self._error is not None:
            status, body = self._error
            record.status = status
            return httpx.Response(status, json=body)

        prompt = "\n".join(
            str(m.get("content", "")) for m in payload.get("messages", []) if isinstance(m, dict)
        )
        wants_json = bool(payload.get("response_format"))

        if wants_json:
            slug = self._pinned.pop(0) if self._pinned else lr.scenario_for(
                _asked_question(payload) or prompt
            )
            self._scenario = slug or self._scenario
            content: Any = lr.PLANS.get(slug, lr.FALLBACK_ANSWER) if slug else (
                lr.FALLBACK_ANSWER
            )
        else:
            # The synthesis prompt carries results, and may not quote the
            # question any more — so the prose follows the plan this run was
            # given rather than trying to recognise the query a second time.
            content = lr.prose_for(prompt, self._scenario)

        self.served.append(content)
        record.status = 200
        model = payload.get("model", lr.DEFAULT_MODEL)
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=lr.sse_chunks(content, model=model),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=lr.chat_completion(content, model=model))


@pytest.fixture(autouse=True)
def network(monkeypatch: pytest.MonkeyPatch):
    """Intercept every httpx call. Nothing leaves the process.

    ASGI traffic from the test client does not go through an httpx transport, so
    the app itself is untouched — only its outbound calls are.
    """
    respx = pytest.importorskip("respx", reason="respx intercepts the outbound httpx calls")
    import httpx

    google = GoogleMock()
    llm = LLMMock()
    seen: list[Recorded] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        split = urlsplit(url)
        record = Recorded(
            method=request.method.upper(),
            url=url,
            path=split.path,
            params={k: v[0] for k, v in parse_qs(split.query).items()},
            headers={k.lower(): v for k, v in request.headers.items()},
            body=request.content or None,
            service="openai" if "openai" in split.netloc else _service_of(split.path),
        )
        seen.append(record)
        if record.service == "openai" or "openai" in split.netloc:
            llm.requests.append(record)
            return llm.handle(record)
        google.requests.append(record)
        return google.handle(record)

    # httplib2 is what google-api-python-client uses, and respx cannot see it.
    # Fail loudly rather than opening a socket.
    with contextlib.suppress(ImportError):
        import httplib2

        def _no_httplib2(*args, **kwargs):  # pragma: no cover - a guard, not a path
            raise AssertionError(
                "a Google call went out over httplib2 (google-api-python-client). "
                "The contract puts Google behind app/google/client.py on httpx, "
                "which is what respx can intercept and what the async path needs."
            )

        monkeypatch.setattr(httplib2.Http, "request", _no_httplib2, raising=False)

    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=dispatch)
        yield _Network(router=router, google=google, llm=llm, seen=seen)


@dataclass
class _Network:
    router: Any
    google: GoogleMock
    llm: LLMMock
    seen: list[Recorded]


@pytest.fixture
def google(network) -> GoogleMock:
    return network.google


@pytest.fixture
def llm(network) -> LLMMock:
    return network.llm


@pytest.fixture
def embed() -> Callable[[str], list[float]]:
    """The same vectors the mocked embedding endpoint returns.

    Seeding the mirror with these means a search in a test scores exactly what
    it would score in production against the same corpus.
    """
    return lambda text: lr.fake_embedding(text, lr.EMBED_DIM)


# --------------------------------------------------------------------------- #
# Users, tokens and the seeded mirror
# --------------------------------------------------------------------------- #


def _encrypt(value: str) -> bytes:
    """Seal a token the way the app seals one.

    ``app.auth.token_store`` is asked first, because it owns the decryption side
    and therefore owns any additional authenticated data the blob is bound to;
    encrypting here without it would produce something that only fails at
    decrypt time, a long way from the cause. ``app.core.crypto`` is the fallback,
    and a marked blob the fallback's fallback.
    """
    for path, names in (
        ("app.auth.token_store", ("encrypt", "seal", "encrypt_token")),
        ("app.core.crypto", ("encrypt", "seal", "encrypt_token")),
    ):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        for name in names:
            fn = getattr(module, name, None)
            if fn is None or not callable(fn):
                continue
            with contextlib.suppress(Exception):
                out = fn(value)
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
    return b"\x00" * 12 + base64.b64encode(value.encode()) + b"\x00" * 16


async def make_user(
    session,
    *,
    email: str,
    display_name: str,
    timezone: str = gr.USER_TZ,
    week_start: int = 1,
):
    """A user with a live Google grant, committed."""
    users = require("app.db.repositories.users")
    user = await users.create_user(
        session,
        email,
        display_name=display_name,
        timezone=timezone,
        work_week_start=week_start,
    )
    await users.upsert_token(
        session,
        user.id,
        access_token_enc=_encrypt(f"ya29.access-{user.id}"),
        refresh_token_enc=_encrypt(f"1//refresh-{user.id}"),
        scopes=[
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/drive",
        ],
        expires_at=FROZEN_NOW + timedelta(minutes=45),
        provider_account_id=f"sub-{user.id}",
    )
    await session.commit()
    # Detached, with its attributes still loaded. A test reads `user.id` long
    # after the session has been expired to pick up the app's commits, and an
    # attached-but-expired instance would try to reload itself there — which in
    # async SQLAlchemy is a MissingGreenlet, ten frames from anything to do with
    # the test. Detaching costs nothing and removes that whole class of noise.
    session.expunge(user)
    return user


@pytest.fixture
async def user(db):
    """The user every scenario in ``docs/SAMPLE_QUERIES.md`` is written for.

    Deliberately shadows the `FakeUser` in the root conftest: inside the
    integration suite a user is a row, not a dataclass. The attributes a test
    reads — ``id``, ``email``, ``timezone`` — are the same either way.
    """
    return await make_user(db, email=gr.USER_EMAIL, display_name=gr.USER_NAME)


@pytest.fixture
async def other_user(db):
    """A second tenant, with deliberately similar data. Never user A."""
    return await make_user(
        db,
        email="rival@alphalaw.test",
        display_name="Rival Ozturk",
        timezone="Europe/Istanbul",
    )


async def seed_mirror(
    session,
    user_id: str,
    embed_fn: Callable[[str], list[float]],
    *,
    gmail: bool = True,
    gcal: bool = True,
    gdrive: bool = True,
) -> dict[str, int]:
    """Fill the pgvector mirror with the fixture corpus, embeddings and all.

    This is what a completed sync would have left behind. Doing it directly
    keeps a query test about the query and not about the sync.
    """
    mirror = require("app.db.repositories.mirror")
    counts: dict[str, int] = {}
    if gmail:
        rows = gr.gmail_mirror_rows(embed_fn)
        counts["gmail"] = len(await mirror.upsert_gmail(session, user_id, rows))
    if gcal:
        rows = gr.gcal_mirror_rows(embed_fn)
        counts["gcal"] = len(await mirror.upsert_gcal(session, user_id, rows))
    if gdrive:
        rows = gr.gdrive_mirror_rows(embed_fn)
        counts["gdrive"] = len(await mirror.upsert_gdrive(session, user_id, rows))
    await session.commit()
    return counts


@pytest.fixture
async def mirrored(db, user, embed):
    """The corpus, mirrored for the main user."""
    await seed_mirror(db, user.id, embed)
    return user


@pytest.fixture
async def sync_ready(db, user):
    """`sync_state` rows for all three services, as a first sync would leave them."""
    state = require("app.db.repositories.sync_state")
    for service in ("gmail", "gcal", "gdrive"):
        await state.ensure_state(db, user.id, service)
    await db.commit()
    return user


# --------------------------------------------------------------------------- #
# The app
# --------------------------------------------------------------------------- #

_AUTH_MODULES = ("app.auth.deps", "app.api.deps", "app.api.v1.deps")
_AUTH_NAMES = (
    "current_user",
    "get_current_user",
    "require_user",
    "require_current_user",
    "authenticated_user",
    "session_user",
    "user_from_session",
    "current_user_id",
    "get_current_user_id",
    "require_user_id",
)


def _wants_id(fn: Callable[..., Any]) -> bool:
    """True when the dependency hands back a user id rather than a row."""
    if any(word in getattr(fn, "__name__", "") for word in ("_id", "user_id")):
        return True
    try:
        annotation = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):  # pragma: no cover
        return False
    return "str" in str(annotation)


class Identity:
    """Who the next request is from.

    One mutable holder rather than a fixed override, because the isolation tests
    need two signed-in users at once. Each client sets this on its way out, so a
    request always resolves to the user that client belongs to.
    """

    def __init__(self, user: Any = None) -> None:
        self.user = user

    def as_seen_by(self, fn: Callable[..., Any]) -> Any:
        return self.user.id if _wants_id(fn) else self.user


def _install_auth(app, identity: Identity) -> int:
    """Override whatever dependency the routers use to identify the caller.

    The contract fixes the module (``app/auth/deps.py``) but not the function
    name, so every plausible name is overridden. Returns how many were found;
    zero means the routers are not wired to a dependency this can reach, and the
    fixture skips rather than testing an app nobody is signed in to.
    """
    installed = 0
    for path in _AUTH_MODULES:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        for name in _AUTH_NAMES:
            fn = getattr(module, name, None)
            if fn is None or not callable(fn):
                continue

            async def override(_fn=fn):
                return identity.as_seen_by(_fn)

            app.dependency_overrides[fn] = override
            installed += 1
    return installed


def _session_cookie(user) -> dict[str, str]:
    """A signed session cookie, if the app exposes a way to mint one."""
    for path in ("app.auth.deps", "app.auth.google_oauth", "app.api.v1.auth"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        for name in ("issue_session", "make_session", "sign_session", "new_session_value"):
            fn = getattr(module, name, None)
            if fn is None:
                continue
            with contextlib.suppress(Exception):
                value = fn(user.id)
                if inspect.isawaitable(value):
                    continue
                if isinstance(value, str):
                    from app.config import settings

                    return {getattr(settings, "SESSION_COOKIE_NAME", "alpha_session"): value}
    return {}


@pytest.fixture
def identity(user) -> Identity:
    """Whose request the app is serving. Clients set it as they send."""
    return Identity(user)


@pytest.fixture
def app(identity):
    """The real FastAPI app, with a user signed in."""
    module = require("app.main", "app")
    application = module.app
    installed = _install_auth(application, identity)
    if not installed:
        pytest.skip(
            "no auth dependency found to override — tried "
            + ", ".join(f"{m}.{n}" for m in _AUTH_MODULES for n in _AUTH_NAMES[:3])
            + " and the rest of the candidates in _AUTH_NAMES"
        )
    try:
        yield application
    finally:
        application.dependency_overrides.clear()


def _client_for(app, identity: Identity, who):
    """An ASGI client that signs every request as ``who``.

    No lifespan is run: the engine and the redis pool are lazy singletons, and
    skipping it keeps `require_production_secrets()` out of the test path.
    """
    import httpx

    async def sign(request: httpx.Request) -> None:
        identity.user = who

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=30.0,
        cookies=_session_cookie(who),
        event_hooks={"request": [sign]},
    )


@pytest.fixture
async def client(app, identity, user):
    """The main user's client."""
    async with _client_for(app, identity, user) as http:
        yield http


@pytest.fixture
async def client_for(app, identity):
    """A client factory, so two tenants can be signed in at the same time."""
    made = []

    async def build(who):
        http = _client_for(app, identity, who)
        made.append(http)
        return http

    yield build
    for http in made:
        await http.aclose()


# --------------------------------------------------------------------------- #
# Driving a query
# --------------------------------------------------------------------------- #


async def post_query(
    client,
    text: str,
    *,
    conversation_id: str | None = None,
    timezone: str = gr.USER_TZ,
    freshness: str | None = None,
    expect: int | tuple[int, ...] = 200,
    **extra: Any,
) -> dict[str, Any]:
    """POST /api/v1/query and hand back the parsed body.

    A non-2xx that was not asked for fails here with the error envelope in the
    message, which is far easier to read than a KeyError three lines later.
    """
    body: dict[str, Any] = {"query": text, "timezone": timezone}
    if conversation_id:
        body["conversation_id"] = conversation_id
    if freshness:
        body["freshness"] = freshness
    body.update(extra)

    response = await client.post("/api/v1/query", json=body)
    wanted = (expect,) if isinstance(expect, int) else expect
    assert response.status_code in wanted, (
        f"POST /api/v1/query {text!r} -> {response.status_code}, expected "
        f"{wanted}: {response.text[:1200]}"
    )
    return response.json() if response.content else {}


async def respond_to_prompt(
    client, input_id: str, value: Any, *, expect: int | tuple[int, ...] = (200, 202)
) -> dict[str, Any]:
    """Answer a card. 200 resumes a paused run, 202 queues an approved write."""
    response = await client.post(f"/api/v1/prompts/{input_id}/respond", json={"value": value})
    wanted = (expect,) if isinstance(expect, int) else expect
    assert response.status_code in wanted, (
        f"POST /prompts/{input_id}/respond -> {response.status_code}, expected "
        f"{wanted}: {response.text[:1200]}"
    )
    return response.json() if response.content else {}


async def _fresh(session) -> None:
    """End this session's transaction so the next read sees the app's commits.

    Expiring is what makes a re-read actually re-read; the cost is that ORM
    objects loaded earlier are now stale handles. Take the values you need off a
    row as soon as you load it, rather than holding the row across one of these.
    """
    await session.rollback()
    session.expire_all()


async def load_steps(session, user_id: str, run_id: str) -> dict[str, Span]:
    """The run's steps, straight from ``node_executions``.

    The authoritative record — the response's ``steps`` array is a projection of
    these rows, and a concurrency claim should be checked against the rows.
    """
    steps = require("app.db.repositories.steps")
    await _fresh(session)
    rows = await steps.list_steps(session, user_id, run_id)
    return spans_of(rows)


async def load_step_rows(session, user_id: str, run_id: str) -> list[Any]:
    """The same rows, unflattened — for ``outcome``, ``retries``, ``round``."""
    steps = require("app.db.repositories.steps")
    await _fresh(session)
    return await steps.list_steps(session, user_id, run_id)


async def load_run(session, user_id: str, run_id: str):
    runs = require("app.db.repositories.runs")
    await _fresh(session)
    return await runs.get_run(session, user_id, run_id)


async def load_actions(session, user_id: str, **kwargs) -> list[Any]:
    actions = require("app.db.repositories.actions")
    await _fresh(session)
    return await actions.list_actions(session, user_id, **kwargs)


async def load_prompts(session, user_id: str, **kwargs) -> list[Any]:
    prompts = require("app.db.repositories.prompts")
    await _fresh(session)
    return await prompts.list_prompts(session, user_id, **kwargs)


def status_of(row: Any) -> str:
    """A status column's value, whether it came back as an enum or a string."""
    value = getattr(row, "status", row)
    return str(getattr(value, "value", value))


_TERMINAL_RUN_STATUSES = {"complete", "failed", "timeout", "cancelled"}

_RESUME_ENTRYPOINTS = (
    ("app.orchestrator.dispatch", ("resume_run", "resume", "continue_run")),
    ("app.tasks.orchestration", ("resume_run_async", "resume_run", "resume")),
    ("app.orchestrator.route", ("resume_run", "resume")),
)

_ACTION_ENTRYPOINTS = (
    ("app.tasks.actions", ("execute_action", "execute_async", "run_action", "execute")),
    ("app.ops.registry", ()),
)


async def call_entrypoint(fn, session, user_id: str, ref: str | None = None, **extra) -> Any:
    """Call a worker entrypoint with whatever arguments it declares.

    The contract names the tasks (``orchestration.resume_run``,
    ``actions.execute``, ``sync.gmail``) but not their signatures. Anything whose
    leading parameters are drawn from the set below is driven correctly;
    anything else raises :class:`LookupError` so the caller can say so plainly
    rather than guessing.
    """
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        raise LookupError(f"{fn!r} has no inspectable signature") from None

    supplied: dict[str, Any] = {
        "session": session,
        "db": session,
        "user_id": user_id,
        "run_id": ref,
        "action_id": ref,
        "input_id": ref,
        "id": ref,
        **extra,
    }
    args = []
    for name in params:
        if name in {"self", "cls"}:
            continue
        if name in supplied and supplied[name] is not None:
            args.append(supplied[name])
        else:
            break

    if not args:
        raise LookupError(
            f"cannot call {getattr(fn, '__qualname__', fn)}{tuple(params)} — none of "
            f"its leading parameters are in {sorted(supplied)}"
        )
    result = fn(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _try_entrypoint(fn, session, user_id: str, ref: str) -> bool:
    with contextlib.suppress(LookupError):
        await call_entrypoint(fn, session, user_id, ref)
        return True
    return False


async def _drive(kind: str, session, user_id: str, ref: str) -> bool:
    """Run the worker half in-process.

    Celery is not running in a test, and eager mode is no good either — the
    tasks are sync wrappers that call ``asyncio.run``, which cannot be re-entered
    from inside a live loop. So the async entrypoint underneath is called
    directly. If a run or an action moves on its own, this never fires.
    """
    table = _RESUME_ENTRYPOINTS if kind == "resume" else _ACTION_ENTRYPOINTS
    for path, names in table:
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        for name in names:
            fn = getattr(module, name, None)
            if fn is None:
                continue
            fn = getattr(fn, "__wrapped__", fn)  # unwrap a celery task
            if not inspect.iscoroutinefunction(fn):
                continue
            with contextlib.suppress(Exception):
                if await _try_entrypoint(fn, session, user_id, ref):
                    return True
    return False


async def settle_run(session, user_id: str, run_id: str, *, timeout: float = 5.0):
    """Wait for a run to stop moving, driving the worker in-process if needed.

    Returns the run row. A run that is still ``running`` when the timeout is up
    comes back as it is, so the caller's assertion reports the real state rather
    than a timeout error.
    """
    import asyncio

    driven = False
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        run = await load_run(session, user_id, run_id)
        if run is None or status_of(run) in _TERMINAL_RUN_STATUSES:
            return run
        if not driven:
            driven = await _drive("resume", session, user_id, run_id)
            if driven:
                continue
        if asyncio.get_running_loop().time() >= deadline:
            return run
        await asyncio.sleep(0.05)


async def execute_approved_actions(session, user_id: str, *, timeout: float = 5.0) -> int:
    """Run whatever the approval queued, in this process.

    Returns how many actions left ``approved``/``running``. Zero means either
    nothing was queued or the endpoint executed inline, and the caller's
    assertions about what reached Google decide which.
    """
    import asyncio

    actions = require("app.db.repositories.actions")
    moved = 0
    deadline = asyncio.get_running_loop().time() + timeout

    while True:
        await _fresh(session)
        waiting = [
            a
            for a in await actions.list_actions(session, user_id, limit=100)
            if status_of(a) in {"approved", "running"}
        ]
        if not waiting:
            return moved
        progressed = False
        for action in waiting:
            if await _drive("action", session, user_id, action.id):
                moved += 1
                progressed = True
        if not progressed or asyncio.get_running_loop().time() >= deadline:
            return moved
        await asyncio.sleep(0.05)


#: Module-level state a restart would have lost. Cleared by `simulate_restart`.
_VOLATILE_HINTS = (
    "cache",
    "channel",
    "subscriber",
    "queue",
    "live",
    "pending",
    "stream",
    "bus",
    "runs",
    "state",
    "sessions",
)


def simulate_restart() -> list[str]:
    """Throw away everything a process restart would have thrown away.

    The database keeps the plan; nothing else is allowed to. Whatever the
    orchestrator was holding in module globals — the SSE channels, any live-run
    registry, the engine's pool — goes here, so a resume after this has nothing
    to work from but ``node_executions``.

    Returns the names it cleared, which is useful in a failure message.
    """
    cleared: list[str] = []
    for path in (
        "app.orchestrator.dispatch",
        "app.orchestrator.events",
        "app.orchestrator.route",
        "app.orchestrator.render",
    ):
        module = sys.modules.get(path)
        if module is None:
            continue
        for name, value in vars(module).items():
            if not name.startswith("_"):
                continue
            if not any(hint in name.lower() for hint in _VOLATILE_HINTS):
                continue
            if isinstance(value, dict):
                value.clear()
                cleared.append(f"{path}.{name}")
            elif isinstance(value, (list, set)):
                value.clear()
                cleared.append(f"{path}.{name}")
    # Sync helper: cannot await a dispose here. Drop the references so the
    # next builder makes a fresh engine; the autouse fixture does the
    # actual dispose on its own async path.
    with contextlib.suppress(ImportError):
        from app.db import session as db_session

        db_session._engine = None
        db_session._sessionmaker = None
    return cleared


def blocking_prompt(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The card a paused run is waiting on."""
    for card in payload.get("pending_inputs") or []:
        if card.get("blocking"):
            return card
    return None


def confirm_card(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The non-blocking confirm card that gates a prepared write."""
    for card in payload.get("pending_inputs") or []:
        if card.get("kind") == "confirm" and not card.get("blocking"):
            return card
    return None


_YES_KEYS = ("approve", "approved", "confirm", "confirmed", "ok", "yes", "send")


def confirm_value(
    card: dict[str, Any], *, approve: bool = True, patch: dict[str, Any] | None = None
) -> Any:
    """The value a Send it / Not now / Edit click produces.

    Read off the card's own ``value_schema`` rather than assumed: that schema is
    the validation authority, and ``additionalProperties: false`` means guessing
    the key name wrongly is a 422 rather than a click.
    """
    schema = card.get("value_schema") or {}
    properties = schema.get("properties") or {}

    if schema.get("type") == "boolean":
        return approve
    if schema.get("enum"):
        options = schema["enum"]
        return options[0] if approve else options[-1]

    key = next((k for k in _YES_KEYS if k in properties), None) or "approve"
    value: dict[str, Any] = {key: approve}
    if patch is not None and ("patch" in properties or not properties):
        value["patch"] = patch
    return value


def pytest_collection_modifyitems(config, items):
    """Everything in this directory needs a live database."""
    for item in items:
        if "integration" in str(getattr(item, "fspath", "")):
            item.add_marker(pytest.mark.integration)


__all__ = [
    "FROZEN_NOW",
    "Recorded",
    "Span",
    "GoogleMock",
    "LLMMock",
    "FakeRedis",
    "answer_text",
    "assert_ran_concurrently",
    "assert_ran_in_order",
    "blob",
    "blocking_prompt",
    "call_entrypoint",
    "confirm_card",
    "confirm_value",
    "degraded_services",
    "execute_approved_actions",
    "first_attr",
    "gr",
    "load_actions",
    "load_prompts",
    "load_run",
    "load_step_rows",
    "load_steps",
    "lr",
    "make_user",
    "mentions",
    "parse_ts",
    "post_query",
    "require",
    "respond_to_prompt",
    "seed_mirror",
    "settle_run",
    "simulate_restart",
    "spans_of",
    "status_of",
]
