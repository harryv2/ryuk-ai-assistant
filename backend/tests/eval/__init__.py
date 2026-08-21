"""The evaluation harness — datasets, adapters and the arithmetic behind them.

Three scripts sit on top of this module:

* ``intent_accuracy.py``  — does the classifier say the right thing?
* ``precision_at_k.py``   — does retrieval put the right documents first?
* ``latency.py``          — how long does each of those take?

Everything they share lives here: dataset loading, percentiles, table
rendering, window resolution, the search backends and the classifier adapters.
It is one module rather than four because ``tests/eval`` owns exactly the eight
files listed in its README and no more.

Two rules run through all of it.

**A filter is never dropped silently.** If a dataset row asks for
``from=sarah@company.com`` and the bound implementation has nowhere to put it,
the harness raises. A dropped filter widens a search, and a widened search
scores *better* on a bad system — which would make this harness a liar.

**Nothing here invents a number.** Where a backend cannot supply something —
``cn``, an evidence flag, a stage timing — the metric that needs it is reported
as unavailable, with the reason, rather than approximated.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import datetime as _dt
import os
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "ARMS",
    "CANNED_ERRORS",
    "DATASETS_DIR",
    "DEFAULT_API_BASE",
    "EVAL_DIR",
    "EVAL_USER_EMAIL",
    "FIXED_NOW",
    "FIXED_TZ",
    "FIXED_WEEK_START",
    "FRONT_DOOR_MODULE",
    "FRONT_DOOR_NAMES",
    "OUT_DIR",
    "ROUTE_MODULE",
    "ROUTE_NAMES",
    "SEARCH_MODULES",
    "SERVICES",
    "AdapterError",
    "ApiClassifier",
    "CannedClassifier",
    "Classifier",
    "Dataset",
    "Hit",
    "HttpSearchBackend",
    "HybridModuleBackend",
    "LiveClassifier",
    "MirrorBackend",
    "Prediction",
    "QueryParams",
    "RunTrace",
    "SearchBackend",
    "SearchResult",
    "Stat",
    "bind",
    "build_classifier",
    "build_search_backend",
    "deliver",
    "env_fingerprint",
    "explain_failure",
    "fmt",
    "git_revision",
    "load_jsonl",
    "markdown_table",
    "mean",
    "normalise_intent",
    "parse_now",
    "percentile",
    "read_json",
    "render_table",
    "resolve_window",
    "run_async",
    "run_query",
    "sse_events",
    "summarise",
    "write_json",
]

# --------------------------------------------------------------------------- #
# Locations and the fixed context
# --------------------------------------------------------------------------- #

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
OUT_DIR = EVAL_DIR / "out"

#: The instant every worked example in docs/SAMPLE_QUERIES.md is evaluated at.
#: Passing ``--now`` overrides it; the default keeps date arithmetic checkable
#: by hand against that document.
# The corpus is seeded with dates relative to the day the seeder ran, so the
# eval clock must be that same day — `make eval` runs seed and harness
# together. EVAL_NOW pins it explicitly when re-running against an old corpus.
FIXED_NOW = os.environ.get("EVAL_NOW") or (
    _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
)
FIXED_TZ = "America/New_York"
FIXED_WEEK_START = 1
EVAL_USER_EMAIL = "demo@example.com"

DEFAULT_API_BASE = os.environ.get("EVAL_API_BASE", "http://localhost:8000")


class AdapterError(RuntimeError):
    """The harness could not bind to the code it is supposed to measure.

    Always carries what it looked for, so the fix is obvious from the message
    rather than from reading this file.
    """


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    """A .jsonl dataset: its metadata line and its labelled rows."""

    path: Path
    meta: dict[str, Any]
    rows: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def filter(self, ids: Sequence[str] | None, tag: str | None = None) -> Dataset:
        rows = self.rows
        if ids:
            wanted = set(ids)
            rows = [r for r in rows if r["id"] in wanted]
            missing = wanted - {r["id"] for r in rows}
            if missing:
                raise SystemExit(f"no such rows in {self.path.name}: {', '.join(sorted(missing))}")
        if tag:
            rows = [r for r in rows if tag in r.get("tags", []) or r.get("type") == tag]
        return Dataset(self.path, self.meta, rows)


def load_jsonl(path: str | Path) -> Dataset:
    """Read a dataset. Lines carrying ``_schema`` are metadata, not rows."""
    path = Path(path)
    if not path.is_absolute():
        path = DATASETS_DIR / path
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # a broken dataset is a build error
                raise SystemExit(f"{path}:{lineno}: {exc}") from exc
            if "_schema" in obj:
                meta = obj
            else:
                rows.append(obj)
    if not rows:
        raise SystemExit(f"{path} has no labelled rows")
    return Dataset(path, meta, rows)


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


@dataclass
class Stat:
    """A latency (or any other) distribution, summarised."""

    n: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": round(self.mean, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value at or above ``q`` of the data.

    Nearest-rank rather than interpolated because a p99 that never occurred is
    not a latency anyone experienced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def summarise(values: Sequence[float]) -> Stat:
    if not values:
        return Stat()
    return Stat(
        n=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, 0.50),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        min=min(values),
        max=max(values),
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, right: Iterable[int] = ()) -> str:
    """A fixed-width text table. Column widths from the content, no dependencies."""
    right_set = set(right)
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if i in right_set else cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(list(headers)), sep, *(line(r) for r in body)])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """The same table as GitHub-flavoured markdown, for RESULTS.md."""
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def fmt(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


# --------------------------------------------------------------------------- #
# Time windows
# --------------------------------------------------------------------------- #


def parse_now(text: str | None) -> datetime:
    if not text:
        text = FIXED_NOW
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _zone(tz: str):
    from zoneinfo import ZoneInfo

    return ZoneInfo(tz)


def _fallback_window(phrase: str, tz: str, week_start: int, now: datetime) -> tuple[datetime, datetime] | None:
    """The handful of phrases the datasets use, resolved locally.

    Used only when ``app.orchestrator.temporal`` is not importable. The rules
    are the ones in docs/contracts.md — local wall time first, UTC second, and
    every window half-open — so the two agree where they overlap. ``--strict-temporal``
    refuses to use this and demands the real module.
    """
    zone = _zone(tz)
    local = now.astimezone(zone)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    key = " ".join(phrase.lower().split())

    def utc(a: datetime, b: datetime) -> tuple[datetime, datetime]:
        return a.astimezone(UTC), b.astimezone(UTC)

    if key == "today":
        return utc(midnight, midnight + timedelta(days=1))
    if key == "tomorrow":
        return utc(midnight + timedelta(days=1), midnight + timedelta(days=2))
    if key == "yesterday":
        return utc(midnight - timedelta(days=1), midnight)
    if key in ("this week", "next week", "last week"):
        # week_start is 1=Mon..7=Sun; isoweekday() is the same numbering.
        delta = (local.isoweekday() - week_start) % 7
        start = midnight - timedelta(days=delta)
        if key == "next week":
            start += timedelta(days=7)
        elif key == "last week":
            start -= timedelta(days=7)
        return utc(start, start + timedelta(days=7))
    if key in ("this month", "last month"):
        first = midnight.replace(day=1)
        if key == "last month":
            end = first
            start = (first - timedelta(days=1)).replace(day=1)
        else:
            start = first
            end = (first + timedelta(days=32)).replace(day=1)
        return utc(start, end)
    return None


def resolve_window(
    spec: Any,
    *,
    now: datetime,
    tz: str = FIXED_TZ,
    week_start: int = FIXED_WEEK_START,
    strict: bool = False,
) -> tuple[datetime, datetime] | None:
    """Turn a dataset ``window`` into a half-open [start, end) pair in UTC.

    Accepts a phrase ("last month"), an explicit offset (``{"days_back": 7}``)
    or literal bounds (``{"start": ..., "end": ...}``). Phrases go to
    ``app.orchestrator.temporal.resolve`` when it is importable, which is the
    point — the eval should exercise the real rule, not a copy of it.
    """
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        if "days_back" in spec:
            return now - timedelta(days=float(spec["days_back"])), now
        if "start" in spec and "end" in spec:
            return parse_now(spec["start"]), parse_now(spec["end"])
        raise ValueError(f"unrecognised window spec: {spec!r}")

    phrase = str(spec)
    try:
        from app.orchestrator import temporal  # type: ignore[import-not-found]
    except Exception:
        temporal = None
    if temporal is not None and hasattr(temporal, "resolve"):
        window = temporal.resolve(phrase, tz, week_start, now)
        if window is not None:
            return _as_utc(window.start), _as_utc(window.end)
        if strict:
            raise AdapterError(f"app.orchestrator.temporal.resolve({phrase!r}) returned None")
    elif strict:
        raise AdapterError(
            "--strict-temporal was passed but app.orchestrator.temporal.resolve is not importable"
        )

    resolved = _fallback_window(phrase, tz, week_start, now)
    if resolved is None:
        raise ValueError(f"no rule for the window phrase {phrase!r}")
    return resolved


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Binding to code that is being written in parallel
# --------------------------------------------------------------------------- #


def bind(modules: str | Sequence[str], candidates: Sequence[str]):
    """First callable in ``candidates`` on the first of ``modules`` that has it.

    Named entry points are tried in order so the harness keeps working while the
    code it measures is still being written, and fails with everything it looked
    for when none of them is there. Same idea and the same candidate lists as
    ``tests/conftest.load_any``, which the rest of the suite uses — they should
    not disagree about where a thing lives.
    """
    import importlib

    if isinstance(modules, str):
        modules = [modules]
    found: list[tuple[str, Any]] = []
    absent: list[str] = []
    broken: list[str] = []
    for path in modules:
        try:
            found.append((path, importlib.import_module(path)))
        except ModuleNotFoundError as exc:
            if exc.name and (path == exc.name or path.startswith(exc.name + ".")):
                absent.append(path)  # not written yet
            else:
                broken.append(f"{path} ({exc})")  # written, but a dependency is missing
        except ImportError as exc:
            broken.append(f"{path} ({exc})")

    for path, module in found:
        for name in candidates:
            fn = getattr(module, name, None)
            if callable(fn):
                return fn, f"{path}.{name}"

    detail = []
    if found:
        have = sorted(
            n for _, m in found for n in vars(m) if not n.startswith("_") and callable(vars(m)[n])
        )
        detail.append(f"present but exposing {', '.join(have) or 'nothing callable'}")
    if absent:
        detail.append(f"not written yet: {', '.join(absent)}")
    if broken:
        detail.append(f"would not import: {'; '.join(broken)}")
    raise AdapterError(
        f"looked for {', '.join(candidates)} on {', '.join(modules)} — " + "; ".join(detail)
    )


def deliver(fn, offered: Mapping[str, Any], *, required: Sequence[str] = ()) -> dict[str, Any]:
    """Keep the offered kwargs the callable can actually accept.

    Anything in ``required`` that has nowhere to go raises. That is the rule
    about never dropping a filter, enforced in one place.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(offered)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(offered)
    accepted = set(signature.parameters)
    kept = {k: v for k, v in offered.items() if k in accepted}
    missing = [k for k in required if k not in kept and offered.get(k) is not None]
    if missing:
        raise AdapterError(
            f"{getattr(fn, '__qualname__', fn)} accepts {sorted(accepted)} and cannot take "
            f"{missing} — the harness will not run a query with a filter it could not pass"
        )
    return kept


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

SERVICES = ("gmail", "gcal", "gdrive")

ARMS = ("hybrid", "vector", "lexical")

#: Where the harness looks for the code it measures, and under what names. The
#: candidate lists are the same ones `tests/conftest.load_any` uses, so the eval
#: and the unit suite cannot end up disagreeing about where a thing lives.
SEARCH_MODULES = ("app.search.hybrid", "app.search.probe")
FRONT_DOOR_MODULE = "app.orchestrator.front_door"
FRONT_DOOR_NAMES = (
    "decide", "run", "front_door", "handle", "try_front_door", "match", "classify", "try_match", "resolve",
)
ROUTE_MODULE = "app.orchestrator.route"
ROUTE_NAMES = ("plan", "route", "classify", "build_plan", "make_plan", "run")


@dataclass
class QueryParams:
    """The prefilters one dataset row asks for, already resolved."""

    services: tuple[str, ...] = SERVICES
    from_email: str | None = None
    attendee: str | None = None
    mime: str | None = None
    since: datetime | None = None
    until: datetime | None = None

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        now: datetime,
        tz: str = FIXED_TZ,
        week_start: int = FIXED_WEEK_START,
        strict_temporal: bool = False,
    ) -> QueryParams:
        params = dict(row.get("params") or {})
        window = resolve_window(
            params.get("window"), now=now, tz=tz, week_start=week_start, strict=strict_temporal
        )
        services = tuple(params.get("services") or SERVICES)
        unknown = [s for s in services if s not in SERVICES]
        if unknown:
            raise SystemExit(f"{row.get('id')}: unknown service(s) {unknown}")
        return cls(
            services=services,
            from_email=params.get("from"),
            attendee=params.get("attendee"),
            mime=params.get("mime"),
            since=window[0] if window else None,
            until=window[1] if window else None,
        )

    def for_service(self, service: str) -> dict[str, Any]:
        """The subset that applies to one corpus. A filter that does not apply
        to a corpus is not a dropped filter — ``from`` means nothing to
        Calendar — but one that applies and cannot be delivered is."""
        out: dict[str, Any] = {}
        if self.since is not None:
            out["since"] = self.since
        if self.until is not None:
            out["until"] = self.until
        if service == "gmail" and self.from_email:
            out["from_email"] = self.from_email
        if service == "gcal" and self.attendee:
            out["attendee_emails"] = [self.attendee]
        if service == "gdrive" and self.mime:
            out["mime_type"] = self.mime
        return out

    #: which corpus each filter belongs to; a filter set on one of these means
    #: the query is about that corpus, whatever else the row lists.
    OWNER = (("from_email", "gmail"), ("attendee", "gcal"), ("mime", "gdrive"))

    def applies_to(self, service: str) -> bool:
        """False when a row's filters make a corpus irrelevant — a ``from``
        filter means the query is about mail, whatever else it lists."""
        return not any(
            getattr(self, field) and service != corpus for field, corpus in self.OWNER
        )


@dataclass
class Hit:
    """One retrieved document, normalised across backends."""

    service: str
    ref: str  # provider id: message_id | event_id | file_id
    row_id: str | None = None  # our nanoid, when the backend gives one
    title: str = ""
    when: datetime | None = None
    cn: float | None = None
    score: float | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "ref": self.ref,
            "title": self.title,
            "cn": self.cn,
            "score": self.score,
            "evidence": self.evidence,
        }


@dataclass
class SearchResult:
    hits: list[Hit]
    took_ms: float
    stages: dict[str, float] = field(default_factory=dict)


class SearchBackend:
    """What precision_at_k and latency both talk to."""

    name = "abstract"
    supports_arms = False
    supplies_cn = False

    async def setup(self) -> None:
        return None

    async def search(
        self, query: str, service: str, params: QueryParams, *, limit: int, arm: str = "hybrid"
    ) -> SearchResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def describe(self) -> str:
        return self.name


class MirrorBackend(SearchBackend):
    """Straight onto ``app.db.repositories.mirror.hybrid_search``.

    This is the backend the ablation runs on, because the repository layer
    takes the two arms as separate inputs: pass an embedding and no ``text``
    for vector-only, ``text`` and no embedding for lexical-only, both for the
    fused search. No flag has to be invented to turn an arm off.
    """

    name = "mirror"
    supports_arms = True
    supplies_cn = False  # cn is computed a layer up, in app.search.hybrid

    def __init__(self, *, user_email: str = EVAL_USER_EMAIL, user_id: str | None = None):
        self._user_email = user_email
        self._user_id = user_id
        self._session_ctx = None
        self._session = None
        self._embed_cache: dict[str, list[float]] = {}

    async def setup(self) -> None:
        from app.db.repositories import users as users_repo
        from app.db.session import session_scope

        self._session_ctx = session_scope()
        self._session = await self._session_ctx.__aenter__()
        if self._user_id is None:
            user = await users_repo.get_user_by_email(self._session, self._user_email)
            if user is None:
                raise AdapterError(
                    f"no user with email {self._user_email!r} — seed the corpus first, "
                    f"or pass --user-id"
                )
            self._user_id = user.id

    async def close(self) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None

    async def _embed(self, text: str) -> list[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        from app.core import llm

        vector = await llm.embed_one(text)
        self._embed_cache[text] = vector
        return vector

    async def search(
        self, query: str, service: str, params: QueryParams, *, limit: int, arm: str = "hybrid"
    ) -> SearchResult:
        from app.db.repositories import mirror

        started = time.perf_counter()
        stages: dict[str, float] = {}
        text = query.strip()
        filters = params.for_service(service)
        if service == "gcal":
            filters["not_cancelled"] = True

        embedding = None
        if text and arm in ("hybrid", "vector"):
            embed_started = time.perf_counter()
            embedding = await self._embed(text)
            stages["embed_ms"] = (time.perf_counter() - embed_started) * 1000
        if text and arm in ("hybrid", "lexical"):
            filters["text"] = text

        sql_started = time.perf_counter()
        try:
            rows = await mirror.hybrid_search(
                self._session, self._user_id, service, embedding, filters, limit
            )
        except Exception:
            # One transaction spans the whole run, so a failed statement would
            # poison every query after it. Roll back and let the caller see the
            # real error.
            await self._session.rollback()
            raise
        stages["sql_ms"] = (time.perf_counter() - sql_started) * 1000

        hits = [self._to_hit(service, row) for row in rows]
        return SearchResult(hits, (time.perf_counter() - started) * 1000, stages)

    @staticmethod
    def _to_hit(service: str, row: Mapping[str, Any]) -> Hit:
        ref_col, title_col, time_col = {
            "gmail": ("message_id", "subject", "received_at"),
            "gcal": ("event_id", "title", "starts_at"),
            "gdrive": ("file_id", "name", "modified_at"),
        }[service]
        return Hit(
            service=service,
            ref=str(row.get(ref_col) or ""),
            row_id=row.get("id"),
            title=str(row.get(title_col) or ""),
            when=row.get(time_col),
            score=row.get("score"),
            scores={"cos": row.get("cos"), "lex": row.get("lex"), "score": row.get("score")},
        )

    def describe(self) -> str:
        return f"mirror (app.db.repositories.mirror.hybrid_search, user_id={self._user_id})"


class HybridModuleBackend(SearchBackend):
    """``app.search.hybrid`` in process — the scoring layer, with cn and evidence.

    Bound by introspection because the module is written in parallel with this
    one. The names it looks for are listed in the README; if none is present it
    says so and names them.
    """

    name = "hybrid"
    supplies_cn = True

    def __init__(self, *, user_email: str = EVAL_USER_EMAIL, user_id: str | None = None):
        self._user_email = user_email
        self._user_id = user_id
        self._session_ctx = None
        self._session = None
        self._fn = None
        self._label = ""
        self._embeddings: dict[str, Any] = {}

    async def setup(self) -> None:
        from app.db.repositories import users as users_repo
        from app.db.session import session_scope

        self._fn, self._label = bind(
            SEARCH_MODULES, ("search", "hybrid_search", "run", "query", "probe")
        )
        self._session_ctx = session_scope()
        self._session = await self._session_ctx.__aenter__()
        if self._user_id is None:
            user = await users_repo.get_user_by_email(self._session, self._user_email)
            if user is None:
                raise AdapterError(f"no user with email {self._user_email!r}")
            self._user_id = user.id
        self.supports_arms = _accepts(self._fn, ("arm", "arms", "mode"))

    async def close(self) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None

    async def _embed(self, query: str):
        """One embedding per distinct query, exactly as the probe pays for it.

        The scoring layer takes the vector as an argument — the app embeds
        once per turn and hands it in. A harness that omits it is silently
        measuring the lexical arm alone and calling it hybrid.
        """
        if query in self._embeddings:
            return self._embeddings[query]
        from app.llm import router as llm_router

        vector = await llm_router.embed_one(query)
        self._embeddings[query] = vector
        return vector

    async def search(
        self, query: str, service: str, params: QueryParams, *, limit: int, arm: str = "hybrid"
    ) -> SearchResult:
        filters = params.for_service(service)
        embedding = await self._embed(query) if query.strip() else None
        offered: dict[str, Any] = {
            "embedding": embedding,
            "session": self._session,
            "user_id": self._user_id,
            "q": query,
            "query": query,
            "text": query,
            "service": service,
            "table": service,
            "corpus": service,
            "limit": limit,
            "filters": filters,
            **filters,
        }
        if self.supports_arms:
            offered.update({"arm": arm, "mode": arm})
        required = [k for k in filters if k != "not_cancelled"]
        # A filter satisfied via the `filters` dict is delivered; only demand a
        # direct parameter when the callable takes no filters mapping at all.
        if _accepts(self._fn, ("filters",)):
            required = []
        kwargs = deliver(self._fn, offered, required=required)
        if not any(k in kwargs for k in ("q", "query", "text")):
            raise AdapterError(f"{self._label} takes no query parameter the harness recognises")

        started = time.perf_counter()
        try:
            result = self._fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            await self._session.rollback()
            raise
        took = (time.perf_counter() - started) * 1000
        return SearchResult([_coerce_hit(service, h) for h in _hit_rows(result)], took)

    def describe(self) -> str:
        arms = "with arm selection" if self.supports_arms else "no arm selection (ablation unavailable)"
        return f"{self._label} ({arms}, user_id={self._user_id})"


class HttpSearchBackend(SearchBackend):
    """``GET /api/v1/search`` over HTTP — the documented debug and eval endpoint.

    Measures the whole request, so its latency includes FastAPI, the session
    and the network. Use it to check the deployed thing; use ``mirror`` or
    ``hybrid`` to measure the retrieval layer on its own.
    """

    name = "http"
    supplies_cn = True

    def __init__(self, base_url: str = DEFAULT_API_BASE, *, cookie: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._cookie = cookie or os.environ.get("EVAL_SESSION_COOKIE")
        self._client = None

    async def setup(self) -> None:
        import httpx

        headers = {}
        if self._cookie:
            headers["Cookie"] = self._cookie
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=30.0)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def search(
        self, query: str, service: str, params: QueryParams, *, limit: int, arm: str = "hybrid"
    ) -> SearchResult:
        if arm != "hybrid":
            raise AdapterError(
                "GET /api/v1/search has no arm parameter; run the ablation on --backend=mirror"
            )
        query_params: dict[str, Any] = {"q": query, "services": service, "limit": limit}
        if params.since is not None:
            query_params["since"] = params.since.isoformat().replace("+00:00", "Z")
        if params.until is not None:
            query_params["until"] = params.until.isoformat().replace("+00:00", "Z")
        if service == "gmail" and params.from_email:
            query_params["from"] = params.from_email
        if service == "gcal" and params.attendee:
            query_params["attendee"] = params.attendee
        if service == "gdrive" and params.mime:
            query_params["mime"] = params.mime

        started = time.perf_counter()
        response = await self._client.get("/api/v1/search", params=query_params)
        took = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise AdapterError(f"GET /api/v1/search -> {response.status_code}: {response.text[:200]}")
        payload = response.json()
        block = (payload.get("services") or {}).get(service) or {}
        hits = [_coerce_hit(service, h) for h in block.get("hits", [])]
        stages = {
            "embedding_ms": float(payload.get("embedding_ms") or 0.0),
            "service_ms": float(block.get("took_ms") or 0.0),
            "server_ms": float(payload.get("took_ms") or 0.0),
        }
        return SearchResult(hits, took, stages)

    def describe(self) -> str:
        return f"http ({self.base_url}/api/v1/search)"


def _accepts(fn, names: Sequence[str]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return True
    return any(n in signature.parameters for n in names)


def _hit_rows(result: Any) -> list[Any]:
    """Pull the hit list out of whatever shape the bound callable returned."""
    if result is None:
        return []
    if isinstance(result, Mapping):
        for key in ("hits", "results", "candidates", "rows"):
            if key in result:
                return list(result[key])
        return []
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return list(result)
    for key in ("hits", "results", "candidates"):
        if hasattr(result, key):
            return list(getattr(result, key))
    raise AdapterError(f"cannot find a hit list in a {type(result).__name__}")


def _coerce_hit(service: str, raw: Any) -> Hit:
    """Normalise one hit. Handles the API's shape, a plain mapping and an object."""
    data = raw if isinstance(raw, Mapping) else getattr(raw, "__dict__", {}) or {}

    def get(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
            value = getattr(raw, name, None)
            if value is not None:
                return value
        return default

    ref_block = get("ref", default=None) or {}
    ref = None
    if isinstance(ref_block, Mapping):
        ref = ref_block.get("message_id") or ref_block.get("event_id") or ref_block.get("file_id")
    ref = ref or get("message_id", "event_id", "file_id", "external_id", "provider_id")
    scores = get("scores", default=None) or {}
    if not isinstance(scores, Mapping):
        scores = {}
    when = get("when", "received_at", "starts_at", "modified_at")
    if isinstance(when, str):
        try:
            when = parse_now(when)
        except ValueError:
            when = None
    return Hit(
        service=service,
        ref=str(ref or get("id", default="") or ""),
        row_id=get("id"),
        title=str(get("title", "subject", "name", default="") or ""),
        when=when,
        cn=_as_float(scores.get("cn") if scores else get("cn")),
        score=_as_float(scores.get("final") if scores else None) or _as_float(get("score", "final")),
        scores=dict(scores),
        evidence=list(get("evidence", default=[]) or []),
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_search_backend(
    kind: str, *, base_url: str = DEFAULT_API_BASE, user_email: str = EVAL_USER_EMAIL,
    user_id: str | None = None, cookie: str | None = None,
) -> SearchBackend:
    if kind == "mirror":
        return MirrorBackend(user_email=user_email, user_id=user_id)
    if kind == "hybrid":
        return HybridModuleBackend(user_email=user_email, user_id=user_id)
    if kind == "http":
        return HttpSearchBackend(base_url, cookie=cookie)
    raise SystemExit(f"unknown search backend {kind!r} — mirror | hybrid | http")


# --------------------------------------------------------------------------- #
# Intent classification
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    """What the classifier said, normalised to the fields the dataset labels."""

    intent: str | None = None
    services: list[str] = field(default_factory=list)
    entity_keys: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    has_write: bool = False
    ambiguous: bool = False
    refs_prior_turn: bool = False
    route: str | None = None
    confidence: float | None = None
    llm_calls: int | None = None
    latency_ms: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "services": self.services,
            "entity_keys": self.entity_keys,
            "has_write": self.has_write,
            "ambiguous": self.ambiguous,
            "refs_prior_turn": self.refs_prior_turn,
            "route": self.route,
            "confidence": self.confidence,
            "llm_calls": self.llm_calls,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


#: Entity keys that only ever appear because something in the conversation was
#: reused. Used to derive refs_prior_turn when the classifier does not say so.
CONTEXT_KEYS = frozenset(
    {"deixis", "resolved_from", "carried_from_run", "ordinal", "referent", "context_ref"}
)


class Classifier:
    name = "abstract"

    async def setup(self) -> None:
        return None

    async def classify(self, row: Mapping[str, Any], *, now: datetime) -> Prediction:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def describe(self) -> str:
        return self.name


def normalise_intent(payload: Any, *, elapsed_ms: float = 0.0) -> Prediction:
    """Read a plan, a front-door decision or a bare intent object into a Prediction."""
    if payload is None:
        return Prediction(error="classifier returned None", latency_ms=elapsed_ms)
    if hasattr(payload, "to_dict") and not isinstance(payload, Mapping):
        payload = payload.to_dict()  # front_door.Decision spells itself
    if not isinstance(payload, Mapping):
        payload = getattr(payload, "__dict__", None) or {
            k: getattr(payload, k) for k in dir(payload) if not k.startswith("_")
        }

    verb = payload.get("type")
    intent_obj = payload.get("intent")
    if intent_obj is None and "name" in payload and "services" in payload:
        intent_obj = payload
    if not isinstance(intent_obj, Mapping):
        intent_obj = {} if intent_obj is None else dict(getattr(intent_obj, "__dict__", {}) or {})

    steps = payload.get("steps") or []
    entities = intent_obj.get("entities") or payload.get("entities") or {}
    if not isinstance(entities, Mapping):
        entities = {}

    name = intent_obj.get("name") or payload.get("intent_name")
    if name is None:
        # Terminal front-door routes ARE the classification.
        route_name = str(payload.get("route") or "")
        if route_name in ("chit_chat", "ui_verb"):
            name = route_name
        elif route_name == "open_card":
            name = "ui_verb"  # answering the card on screen is a UI verb
        elif route_name == "capability":
            name = "unsupported"
    if name is None and verb == "answer":
        # An `answer` verb with no intent is the front door refusing to plan:
        # chit-chat when nothing was retrieved for, unsupported otherwise.
        name = "chit_chat" if payload.get("source", "").startswith("chit") else "unsupported"

    ambiguous = bool(
        intent_obj.get("ambiguous")
        or payload.get("ambiguous")
        or any(_step_op(s) == "ask.user" for s in steps)
        or payload.get("needs_input") is not None
    )
    source = str(intent_obj.get("source") or payload.get("source") or "") or None
    refs_prior = bool(
        payload.get("refs_prior_turn")
        or intent_obj.get("refs_prior_turn")
        or (source or "").endswith("intent_carry")
        or (CONTEXT_KEYS & set(entities))
        or "{{context." in json.dumps(steps, default=str)
    )
    services = list(intent_obj.get("services") or payload.get("services") or [])
    has_write = bool(intent_obj.get("has_write") or payload.get("has_write"))

    return Prediction(
        intent=name,
        services=[str(s) for s in services],
        entity_keys=sorted(str(k) for k in entities),
        entities=dict(entities),
        has_write=has_write,
        ambiguous=ambiguous,
        refs_prior_turn=refs_prior,
        route=source,
        confidence=_as_float(intent_obj.get("confidence")),
        latency_ms=elapsed_ms,
        raw=dict(payload) if isinstance(payload, Mapping) else {},
    )


def _step_op(step: Any) -> str:
    if isinstance(step, Mapping):
        return str(step.get("op") or "")
    return str(getattr(step, "op", "") or "")


class LiveClassifier(Classifier):
    """The real front door, then the real planner. One LLM call at most.

    Binds ``app.orchestrator.front_door`` and ``app.orchestrator.route`` by
    introspection for the same reason the search backend does.
    """

    name = "live"

    def __init__(self, *, user_email: str = EVAL_USER_EMAIL, user_id: str | None = None,
                 tz: str = FIXED_TZ, week_start: int = FIXED_WEEK_START):
        self._user_email = user_email
        self._user_id = user_id
        self._tz = tz
        self._week_start = week_start
        self._front_door = None
        self._front_door_label = ""
        self._route = None
        self._route_label = ""
        self._session_ctx = None
        self._session = None

    async def setup(self) -> None:
        from app.db.repositories import users as users_repo
        from app.db.session import session_scope

        try:
            self._front_door, self._front_door_label = bind(FRONT_DOOR_MODULE, FRONT_DOOR_NAMES)
        except AdapterError:
            self._front_door = None  # a missing front door is not fatal; the planner covers it
        self._route, self._route_label = bind(ROUTE_MODULE, ROUTE_NAMES)
        self._session_ctx = session_scope()
        self._session = await self._session_ctx.__aenter__()
        if self._user_id is None:
            user = await users_repo.get_user_by_email(self._session, self._user_email)
            self._user_id = user.id if user else None

    async def close(self) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None

    def _offer(self, row: Mapping[str, Any], now: datetime) -> dict[str, Any]:
        context = row.get("context") or {}
        return {
            "query": row["query"],
            "text": row["query"],
            "message": row["query"],
            "q": row["query"],
            "session": self._session,
            "user_id": self._user_id,
            "conversation_id": None,
            "now": now,
            "tz": self._tz,
            "week_start": self._week_start,
            "history": context.get("prior_turns") or [],
            "context": context,
            "prior_intent": context.get("prior_intent"),
            # what `front_door.decide` calls it — shaped like the runner's
            # `_last_intent()`, which hands the previous turn's intent dict.
            "last_intent": (
                {"name": context.get("prior_intent")} if context.get("prior_intent") else None
            ),
            # A dataset row that says a card is on screen gets that card
            # offered the way the runner offers pending prompts.
            "open_prompts": [context["open_card"]] if context.get("open_card") else [],
        }

    async def classify(self, row: Mapping[str, Any], *, now: datetime) -> Prediction:
        offered = self._offer(row, now)
        started = time.perf_counter()
        try:
            payload = None
            if self._front_door is not None:
                payload = await _maybe_await(self._front_door(**deliver(self._front_door, offered)))
            if _is_miss(payload):
                payload = await _maybe_await(self._route(**deliver(self._route, offered)))
            elapsed = (time.perf_counter() - started) * 1000
            prediction = normalise_intent(payload, elapsed_ms=elapsed)
        except Exception as exc:  # a failed classification is data, not a crash
            return Prediction(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return prediction

    def describe(self) -> str:
        front = self._front_door_label or "no front door bound"
        return f"live ({front} -> {self._route_label})"


def _is_miss(payload: Any) -> bool:
    """A front-door result that means 'not mine, keep going'."""
    if payload is None or payload is False:
        return True
    if isinstance(payload, str):
        return True  # a bare string is prose, not a classification
    handled = getattr(payload, "handled", None)
    if handled is False:
        return True  # front_door.Decision with route == miss
    if isinstance(payload, Mapping):
        if payload.get("type") in ("miss", "pass", "decline"):
            return True
        if payload.get("matched") is False:
            return True
    return False


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class ApiClassifier(Classifier):
    """``POST /api/v1/query`` and read the ``intent`` SSE event.

    The truest measurement — it is the path a user takes — and the slowest.
    Writes are prepared, never executed: nothing leaves for Google without an
    approval this harness never gives.
    """

    name = "api"

    def __init__(self, base_url: str = DEFAULT_API_BASE, *, cookie: str | None = None,
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._cookie = cookie or os.environ.get("EVAL_SESSION_COOKIE")
        self._timeout = timeout
        self._client = None

    async def setup(self) -> None:
        import httpx

        headers = {"Cookie": self._cookie} if self._cookie else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self._timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def classify(self, row: Mapping[str, Any], *, now: datetime) -> Prediction:
        started = time.perf_counter()
        try:
            run = await run_query(self._client, row["query"], timeout=self._timeout)
        except Exception as exc:
            return Prediction(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        elapsed = run.marks.get("intent", (time.perf_counter() - started) * 1000)
        payload = run.intent or {"type": "answer", "source": run.answer_source or ""}
        prediction = normalise_intent(payload, elapsed_ms=elapsed)
        prediction.llm_calls = run.llm_calls
        return prediction

    def describe(self) -> str:
        return f"api ({self.base_url}/api/v1/query)"


#: Deterministic wrong answers for --dry-run. Each one exercises a branch of the
#: report: an intent confusion, a service-set miss, a missing entity key, a
#: missed ambiguity flag and a false alarm. Fixed rather than random so CI
#: output does not move between runs.
CANNED_ERRORS: dict[str, dict[str, Any]] = {
    "i003": {"intent": "email_search", "services": ["gmail"]},        # awkward phrasing confusion
    "i011": {"intent": "email_detail"},                               # search vs detail
    "i023": {"intent": "drive_filter"},                               # search vs filter, the near pair
    "i027": {"intent": "calendar_list", "services": ["gcal"]},        # multi-service collapsed to one
    "i031": {"services": ["gcal", "gdrive"]},                         # service set too wide
    "i035": {"entity_drop": ["*"]},                                   # no entities at all (none expected)
    "i036": {"entity_drop": ["sender_domain"]},                       # entity key missed
    "i041": {"ambiguous": False},                                     # MISSED ambiguity — the dangerous one
    "i048": {"ambiguous": False},                                     # a second missed ambiguity
    "i042": {"ambiguous": True},                                      # false alarm, the cheap direction
    "i056": {"intent": "chit_chat", "services": []},                  # the eager-matcher trap, failed
    "i061": {"intent": "cancel_flight", "services": ["gmail", "gcal"]},  # out of scope taken as in scope
    "i008": {"refs_prior_turn": True},                                # context claimed where there is none
}


class CannedClassifier(Classifier):
    """--dry-run: the labels, with a fixed set of deliberate mistakes.

    It exists so the report itself can be exercised in CI without an API key
    and without a database. **Its numbers are not measurements** and every
    surface that prints them says so.
    """

    name = "canned"

    def __init__(self, errors: Mapping[str, dict[str, Any]] | None = None):
        self._errors = dict(errors if errors is not None else CANNED_ERRORS)

    async def classify(self, row: Mapping[str, Any], *, now: datetime) -> Prediction:
        expected = row["expected"]
        error = self._errors.get(row["id"], {})
        entity_keys = list(expected.get("entity_keys") or [])
        drop = error.get("entity_drop")
        if drop == ["*"]:
            entity_keys = []
        elif drop:
            entity_keys = [k for k in entity_keys if k not in drop]
        entities = {k: v for k, v in (expected.get("entities") or {}).items() if k in entity_keys}
        # A believable classifier is a fast one on the router path and a slower
        # one when it has to call a model; the shape matters for the report.
        route = expected.get("route") or "planner"
        latency = 8.0 if route in ("chit_chat", "ui_verb") else 150.0 if route.startswith("rule_router") else 590.0
        return Prediction(
            intent=error.get("intent", expected["intent"]),
            services=list(error.get("services", expected["services"])),
            entity_keys=sorted(entity_keys),
            entities=entities,
            has_write=bool(error.get("has_write", expected["has_write"])),
            ambiguous=bool(error.get("ambiguous", expected["flag_ambiguity"])),
            refs_prior_turn=bool(error.get("refs_prior_turn", expected["refs_prior_turn"])),
            route=route,
            confidence=0.9,
            llm_calls=0 if route in ("chit_chat", "ui_verb") or route.startswith("rule_router") else 1,
            latency_ms=latency,
        )

    def describe(self) -> str:
        return f"canned ({len(self._errors)} deliberate errors — NOT a measurement)"


def build_classifier(kind: str, **kwargs: Any) -> Classifier:
    if kind == "canned":
        return CannedClassifier()
    if kind == "live":
        return LiveClassifier(**{k: v for k, v in kwargs.items() if k in
                                 ("user_email", "user_id", "tz", "week_start")})
    if kind == "api":
        return ApiClassifier(**{k: v for k, v in kwargs.items() if k in ("base_url", "cookie", "timeout")})
    raise SystemExit(f"unknown classifier {kind!r} — live | api | canned")


# --------------------------------------------------------------------------- #
# The SSE client, shared by the api classifier and by latency.py
# --------------------------------------------------------------------------- #


@dataclass
class RunTrace:
    """One trip through POST /api/v1/query, timed at every event that matters."""

    run_id: str | None = None
    marks: dict[str, float] = field(default_factory=dict)  # event -> ms from request start
    intent: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int | None = None
    status: str | None = None
    answer_source: str | None = None
    total_ms: float = 0.0
    error: str | None = None


async def sse_events(client, url: str, *, timeout: float = 60.0):  # noqa: ASYNC109 — httpx read timeout, not a cancel scope
    """Yield parsed ``data:`` payloads from an SSE stream."""
    async with client.stream("GET", url, timeout=timeout) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")[:200]
            raise AdapterError(f"GET {url} -> {response.status_code}: {body}")
        buffer: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                buffer.append(line[5:].strip())
                continue
            if line == "" and buffer:
                chunk = "\n".join(buffer)
                buffer.clear()
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    continue


async def run_query(client, query: str, *, conversation_id: str | None = None,
                    timeout: float = 60.0) -> RunTrace:  # noqa: ASYNC109 — httpx read timeout
    """POST the query, follow its event stream, and time every stage.

    The marks are the ones docs/DESIGN.md §8.1 publishes as ``ttfp_seconds``:
    run.started, probe.done, intent, input.raised, first step finished, first
    prose token, run.complete.
    """
    trace = RunTrace()
    started = time.perf_counter()
    body: dict[str, Any] = {"query": query}
    if conversation_id:
        body["conversation_id"] = conversation_id
    response = await client.post("/api/v1/query", json=body, timeout=timeout)
    if response.status_code >= 400:
        trace.error = f"POST /api/v1/query -> {response.status_code}: {response.text[:200]}"
        trace.total_ms = (time.perf_counter() - started) * 1000
        return trace
    payload = response.json()
    trace.run_id = payload.get("run_id") or payload.get("id")
    trace.marks["accepted"] = (time.perf_counter() - started) * 1000
    if not trace.run_id:
        # A synchronous answer (chit-chat, a UI verb) never opens a stream.
        trace.status = payload.get("status", "complete")
        trace.intent = payload.get("intent")
        trace.llm_calls = _llm_calls(payload)
        trace.answer_source = payload.get("source")
        trace.total_ms = (time.perf_counter() - started) * 1000
        return trace

    first_of = {"probe.done", "intent", "input.raised", "step.finished", "token", "content.delta"}
    async for event in sse_events(client, f"/api/v1/runs/{trace.run_id}/events", timeout=timeout):
        kind = event.get("type")
        at = (time.perf_counter() - started) * 1000
        data = event.get("data") or {}
        if kind in first_of and kind not in trace.marks:
            trace.marks[kind] = at
        if kind == "run.started":
            trace.marks.setdefault("run.started", at)
        elif kind == "intent":
            trace.intent = data.get("intent", data)
        elif kind == "step.finished":
            trace.steps.append(data)
        elif kind in ("run.complete", "run.paused", "error"):
            trace.marks[kind] = at
            trace.status = data.get("status") or ("failed" if kind == "error" else kind.split(".")[1])
            trace.llm_calls = _llm_calls(data)
            break
    trace.total_ms = (time.perf_counter() - started) * 1000
    return trace


def _llm_calls(payload: Mapping[str, Any]) -> int | None:
    usage = payload.get("token_usage") or payload.get("usage") or {}
    if isinstance(usage, Mapping) and "calls" in usage:
        return int(usage["calls"])
    return payload.get("llm_calls")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_json(name: str, payload: Mapping[str, Any]) -> Path:
    """Drop a metrics blob in tests/eval/out/ for precision_at_k to assemble."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_json(name: str) -> dict[str, Any] | None:
    path = OUT_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def git_revision() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVAL_DIR, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_fingerprint() -> dict[str, Any]:
    """What was true when the numbers were produced. Goes in every output file."""
    try:
        from app.config import settings

        thresholds = {
            "FLOOR_READ": settings.FLOOR_READ,
            "MARGIN": settings.MARGIN,
            "FLOOR_WRITE": settings.FLOOR_WRITE,
        }
        models = {"chat": settings.OPENAI_MODEL, "embed": settings.OPENAI_EMBED_MODEL}
    except Exception:
        thresholds = {"FLOOR_READ": 0.55, "MARGIN": 0.15, "FLOOR_WRITE": 0.80}
        models = {"chat": "unknown", "embed": "unknown"}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "git": git_revision(),
        "python": sys.version.split()[0],
        "thresholds": thresholds,
        "models": models,
    }


def run_async(coro):
    """One entry point for the scripts, so they all handle Ctrl-C the same way."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None


def explain_failure(exc: BaseException, *, what: str, traceback_wanted: bool) -> int:
    """Print why a harness could not run, and return the exit code.

    A benchmark that dies in a traceback tells you less than one that says which
    thing it could not reach. ``--traceback`` restores the traceback for the
    case where the harness itself is the bug.
    """
    if traceback_wanted:
        raise exc
    hint = ""
    if isinstance(exc, ImportError):
        hint = "  the module it measures does not exist yet, or its dependencies are not installed"
    elif isinstance(exc, (ConnectionError, OSError)):
        hint = "  nothing answered — is `docker compose up -d` done and migrated?"
    elif isinstance(exc, AdapterError):
        hint = "  the harness bound to nothing it recognised; the message above lists what it looked for"
    print(f"\n{what} could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    print("  re-run with --traceback for the full stack.", file=sys.stderr)
    return 2
