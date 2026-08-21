"""The probe: search before you plan.

A planner that has not seen the user's data has to guess whether the thing
exists, guess the query string, guess whether there is one of them or three,
and cannot see an ambiguity until execution — by which time noticing it costs a
second model call. The probe removes four of those five problems for the price
of one embedding and three indexed queries.

What it does, in order:

* Skip the embedding entirely when the query is **structural** — "what's on my
  calendar next week" has no topic, only filters, and a vector adds nothing to
  a time-window scan.
* Fire the **lexical arm at t=0**, before the embedding comes back, because
  full text needs no vector. The two overlap, which is where most of the
  latency saving lives.
* Run the three corpora **concurrently**, one connection each.
* Collapse chunks to one row per message / event / file, so the planner sees
  one candidate with one reference path.
* Run the **regex extractors** over the top candidates, so a PNR reaches the
  plan as ``{{search.gmail[0].extracted.pnr}}`` — a value the model never held
  and therefore cannot invent.
* **Wave 2**: deterministic follow-ups, only when wave 1 actually found
  something. The participants of the top event, the threads behind the top
  emails and their replies, files named after the top event. No model involved
  in deciding to do any of it.
* Stamp every candidate with the mirror's ``synced_at`` and report anything
  **degraded**, so an answer built on a stale corpus can say so.

The whole pass is recorded as step ``seq=0``, ``node_id="search"``, which is
what makes it visible in the trace panel beside the steps the planner asked
for.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.db.repositories import mirror, sync_state
from app.search import aliases as alias_table
from app.search import embedder, extractors, hybrid
from app.search.hybrid import Hit, ServiceResult, Terms

log = get_logger(__name__)

#: The step this pass is recorded as.
NODE_ID = "search"
OP_NAME = "search.probe"
SEQ = 0

#: Candidates kept per corpus for the planner.
TOP_N = 5
#: Candidates the extractors run over. They are cheap, but not free, and
#: nothing below the top few is ever referenced by a plan.
EXTRACT_TOP_N = 3
#: Rows each arm brings back before fusion.
CANDIDATES = hybrid.DEFAULT_CANDIDATES

#: How far back a query with no time phrase looks. A booking made a year ago
#: is still the flight you are cancelling today.
DEFAULT_LOOKBACK_DAYS = 365
#: Calendar looks a little way back and a long way forward.
CALENDAR_LOOKBACK_DAYS = 30


def _words(block: str) -> frozenset[str]:
    """A word set written as one readable block rather than a list literal."""
    return frozenset(block.split())


#: Words that mean "everything in this corpus", not "this particular thing".
_PLURAL_CUES = re.compile(
    r"\b(all|every|list|emails|messages|files|documents|docs|events|meetings|"
    r"summar(?:y|ise|ize)|everything|any)\b",
    re.I,
)

#: Words that carry no topic. What is left after removing these, the time
#: phrases and the service nouns is the actual subject of the query.
_STOPWORDS = _words(
    """a an and are as at be been by can could did do does for from get give had has have
    how i if in into is it its just me my of on or our please show tell that the their them
    then there these they this to us was we were what when where which who will with would
    you your s t am pm"""
)

#: Nouns that name a corpus rather than a topic.
_SERVICE_WORDS = _words(
    """calendar diary schedule agenda meeting meetings event events appointment appointments
    mail email emails message messages inbox drive file files document documents doc docs
    folder folders spreadsheet spreadsheets presentation deck"""
)

#: Verbs the front door and the planner recognise; not a topic either.
_VERB_WORDS = _words(
    """find search look show list open read send draft reply forward move cancel delete
    create make schedule reschedule push prepare summarise summarize check what whats"""
)

#: Time words. ``temporal.py`` has already turned these into a window, so they
#: are filters by the time the probe sees them, not subject matter.
_TIME_WORDS = _words(
    """today tonight tomorrow yesterday now soon later week weeks weekend weekends month
    months year years day days morning afternoon evening night hour hours minute minutes
    next last this coming upcoming past previous following recent recently ago until since
    before after between early late monday tuesday wednesday thursday friday saturday sunday
    mon tue tues wed thu thur thurs fri sat sun january february march april may june july
    august september october november december jan feb mar apr jun jul aug sep sept oct nov
    dec quarter q1 q2 q3 q4"""
)


# --------------------------------------------------------------------------- #
# Reading whatever the pre-pass produced
# --------------------------------------------------------------------------- #


def _get(source: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute or key. The pre-pass is another
    module's output and this module refuses to be brittle about its spelling."""
    for name in names:
        if isinstance(source, Mapping):
            if name in source and source[name] is not None:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return default


def _as_window(value: Any) -> tuple[dt.datetime, dt.datetime] | None:
    """``(start, end)`` out of a ``Window``, a dict or a pair."""
    if value is None:
        return None
    start = _get(value, "start", "from", "since")
    end = _get(value, "end", "to", "until")
    if start is None or end is None:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = value
        else:
            return None
    start = _coerce_dt(start)
    end = _coerce_dt(end)
    if start is None or end is None:
        return None
    return (start, end)


def _coerce_dt(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    return None


@dataclass
class Prepass:
    """The pre-pass, read into the handful of things the probe needs.

    Built from whatever ``app.orchestrator.prepass`` hands over — a dataclass,
    a Pydantic model, a dict, or in the simplest case the raw query string.
    """

    query: str = ""
    search_text: str = ""
    windows: dict[str, tuple[dt.datetime, dt.datetime]] = field(default_factory=dict)
    primary_window: tuple[dt.datetime, dt.datetime] | None = None
    terms: Terms = field(default_factory=Terms)
    services: tuple[str, ...] = hybrid.SERVICES
    expect: str = "one"
    filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    structural: bool | None = None
    now: dt.datetime | None = None
    tz: str = "UTC"
    raw: Any = None

    @classmethod
    def of(cls, prepass: Any, *, now: dt.datetime | None = None) -> Prepass:
        if isinstance(prepass, Prepass):
            return prepass
        if isinstance(prepass, str):
            prepass = {"query": prepass}

        query = str(_get(prepass, "query", "text", "message", "q", default="") or "")
        search_text = str(
            _get(prepass, "search_text", "residual", "topic", "clean_query", default="") or ""
        )

        raw_windows = _get(prepass, "windows", "resolved_windows", default={}) or {}
        windows: dict[str, tuple[dt.datetime, dt.datetime]] = {}
        if isinstance(raw_windows, Mapping):
            for name, window in raw_windows.items():
                pair = _as_window(window)
                if pair:
                    windows[str(name)] = pair

        primary = _as_window(
            _get(prepass, "window", "primary_window", "resolved_window")
        )
        if primary is None and windows:
            primary = next(iter(windows.values()))

        services = _get(prepass, "services", "corpora", default=None)
        if services:
            keys = tuple(dict.fromkeys(hybrid.service_key(s) for s in services))
        else:
            keys = hybrid.SERVICES

        expect = str(_get(prepass, "expect", "arity", default="") or "").lower()
        if expect not in {"one", "many"}:
            expect = "many" if _PLURAL_CUES.search(query) else "one"

        filters = _get(prepass, "filters", "service_filters", default={}) or {}
        if not isinstance(filters, Mapping):
            filters = {}
        per_service: dict[str, dict[str, Any]] = {}
        for name, value in filters.items():
            try:
                key = hybrid.service_key(name)
            except Exception:
                per_service = {s: dict(filters) for s in keys}
                break
            if isinstance(value, Mapping):
                per_service[key] = dict(value)

        return cls(
            query=query,
            search_text=search_text or query,
            windows=windows,
            primary_window=primary,
            terms=_terms_from(prepass, query),
            services=keys,
            expect=expect,
            filters=per_service,
            structural=_get(prepass, "structural", "is_structural"),
            now=_coerce_dt(_get(prepass, "now")) or now,
            tz=str(_get(prepass, "tz", "timezone", default="UTC") or "UTC"),
            raw=prepass,
        )


def _terms_from(prepass: Any, query: str) -> Terms:
    """Alias tokens, sender domains and literals, from the pre-pass or the query."""
    supplied = _get(prepass, "terms")
    if isinstance(supplied, Terms):
        return supplied.merge(Terms.from_query(query))

    tokens: list[str] = []
    domains: list[str] = []
    patterns: list[str] = []
    block = _get(prepass, "aliases", "alias", "alias_group", "vendor", default=None)
    blocks = block if isinstance(block, (list, tuple)) else [block] if block else []
    for item in blocks:
        if isinstance(item, Mapping):
            tokens.extend(str(t) for t in (item.get("tokens") or []))
            domains.extend(str(d) for d in (item.get("sender_domains") or []))
            patterns.extend(str(p) for p in (item.get("code_patterns") or []))
        elif isinstance(item, str):
            group = alias_table.find(item)
            if group is not None:
                tokens.extend(group.tokens)
                domains.extend(group.sender_domains)
                patterns.extend(group.code_patterns)

    literals = _get(prepass, "literals", "entities", default={}) or {}
    ids: list[str] = []
    emails: list[str] = []
    filenames: list[str] = []
    if isinstance(literals, Mapping):
        for name in ("ids", "message_ids", "event_ids", "file_ids", "refs"):
            value = literals.get(name)
            if isinstance(value, (list, tuple, set)):
                ids.extend(str(v) for v in value)
            elif value:
                ids.append(str(value))
        for name in ("emails", "addresses", "people"):
            value = literals.get(name)
            if isinstance(value, (list, tuple, set)):
                emails.extend(str(v) for v in value)
            elif value:
                emails.append(str(value))
        for name in ("filenames", "files", "names"):
            value = literals.get(name)
            if isinstance(value, (list, tuple, set)):
                filenames.extend(str(v) for v in value)
            elif value:
                filenames.append(str(value))

    supplied_terms = Terms(
        aliases=tuple(dict.fromkeys(alias_table.normalise(t) for t in tokens if t)),
        sender_domains=tuple(dict.fromkeys(d.lower() for d in domains if d)),
        emails=tuple(dict.fromkeys(e.lower() for e in emails if e)),
        ids=tuple(dict.fromkeys(ids)),
        filenames=tuple(dict.fromkeys(filenames)),
        code_patterns=tuple(dict.fromkeys(patterns)),
    )
    return Terms.from_query(query, supplied_terms)


# --------------------------------------------------------------------------- #
# Structural queries
# --------------------------------------------------------------------------- #


def topic_words(query: str, terms: Terms | None = None) -> list[str]:
    """The words in a query that are actually a subject.

    Strip the verbs, the corpus nouns, the pronouns, the resolved time phrases
    and any vendor alias already expanded into filters. What is left is what a
    vector could help with.
    """
    text = alias_table.normalise(query)
    for token in (terms.aliases if terms else ()):
        normalised = alias_table.normalise(token)
        if normalised:
            text = text.replace(normalised, " ")
    words = [w for w in text.split() if w]
    return [
        word
        for word in words
        if word not in _STOPWORDS
        and word not in _SERVICE_WORDS
        and word not in _VERB_WORDS
        and word not in _TIME_WORDS
        and not word.isdigit()
        and len(word) > 1
    ]


def is_structural(pre: Prepass) -> bool:
    """True when the query is filters only and an embedding would add nothing.

    "What's on my calendar next week where john@company.com is invited" is a
    date range and a GIN index hit. Embedding it costs 55 ms and a request, and
    changes no answer.
    """
    if pre.structural is not None:
        return bool(pre.structural)
    if pre.terms.emails or pre.terms.ids:
        # An address or an id is an exact filter; the rest of the sentence is
        # usually scaffolding.
        return not topic_words(pre.search_text, pre.terms)
    if not (pre.windows or pre.primary_window or pre.filters):
        return False
    return not topic_words(pre.search_text, pre.terms)


# --------------------------------------------------------------------------- #
# Filters per corpus
# --------------------------------------------------------------------------- #


def window_filters(
    service: str,
    pre: Prepass,
    now: dt.datetime,
) -> dict[str, Any]:
    """The metadata prefilter for one corpus.

    A resolved window applies to the calendar always. It applies to mail and
    files only when it points at the past — "next week" filtered against
    ``received_at`` would exclude every email there has ever been.
    """
    key = hybrid.service_key(service)
    out: dict[str, Any] = dict(pre.filters.get(key) or {})

    window = pre.primary_window
    if key == "gcal":
        if window:
            out.setdefault("since", window[0])
            out.setdefault("until", window[1])
        else:
            out.setdefault("since", now - dt.timedelta(days=CALENDAR_LOOKBACK_DAYS))
        out.setdefault("not_cancelled", True)
        return out

    lookback = int(getattr(settings, "SYNC_BACKFILL_DAYS", DEFAULT_LOOKBACK_DAYS) or
                   DEFAULT_LOOKBACK_DAYS)
    lookback = max(lookback, DEFAULT_LOOKBACK_DAYS)
    if window and window[0] <= now:
        out.setdefault("since", window[0])
        out.setdefault("until", window[1])
    else:
        out.setdefault("since", now - dt.timedelta(days=lookback))
    return out


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    """Everything the planner is told about the user's data, and nothing else."""

    query: str = ""
    services: dict[str, list[Hit]] = field(default_factory=dict)
    stats: dict[str, ServiceResult] = field(default_factory=dict)
    windows: dict[str, tuple[dt.datetime, dt.datetime]] = field(default_factory=dict)
    extracted: dict[str, list[Any]] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    ambiguity: dict[str, Any] | None = None
    degraded: list[dict[str, Any]] = field(default_factory=list)
    synced_at: dict[str, dt.datetime | None] = field(default_factory=dict)
    wave2: dict[str, Any] = field(default_factory=dict)
    #: The turn's query vector, kept so the rest of the run can reuse it. The
    #: budget is one embedding per turn: a plan step that searches the same
    #: mirror is asking about the same question, so it rides on this rather
    #: than paying a second round trip to embed a sub-phrase of it.
    embedding: list[float] | None = None
    embedding_ms: int = 0
    embedding_skipped: bool = False
    took_ms: int = 0
    expect: str = "one"
    now: dt.datetime | None = None

    # -- reading it ------------------------------------------------------- #

    @property
    def gmail(self) -> list[Hit]:
        return self.services.get("gmail", [])

    @property
    def gcal(self) -> list[Hit]:
        return self.services.get("gcal", [])

    @property
    def gdrive(self) -> list[Hit]:
        return self.services.get("gdrive", [])

    @property
    def all_hits(self) -> list[Hit]:
        return [hit for key in hybrid.SERVICES for hit in self.services.get(key, [])]

    @property
    def counts(self) -> dict[str, int]:
        return {key: len(self.services.get(key, [])) for key in hybrid.SERVICES}

    @property
    def found_anything(self) -> bool:
        """Is there a candidate worth planning against at all?"""
        return any(hit.qualifies() for hit in self.all_hits)

    def top(self, service: str | None = None) -> Hit | None:
        hits = self.services.get(hybrid.service_key(service), []) if service else self.all_hits
        if service is None:
            hits = sorted(hits, key=lambda h: -hybrid.effective_cn(h))
        return hits[0] if hits else None

    def qualifying(self, floor: float | None = None) -> list[Hit]:
        return [hit for hit in self.all_hits if hit.qualifies(floor)]

    # -- what other modules consume --------------------------------------- #

    def to_bindings(self) -> dict[str, Any]:
        """What ``{{search.gmail[0].extracted.pnr}}`` resolves against."""
        out: dict[str, Any] = {
            key: [hit.to_binding() for hit in self.services.get(key, [])]
            for key in hybrid.SERVICES
        }
        out["ambiguity"] = self.ambiguity
        out["decision"] = self.decision
        out["extracted"] = self.extracted
        out["wave2"] = self.wave2
        return out

    def to_event(self) -> dict[str, Any]:
        """The ``probe.done`` SSE payload."""
        leaders = []
        for key in hybrid.SERVICES:
            hits = self.services.get(key, [])
            if hits:
                leaders.append(
                    {
                        "service": key,
                        "label": hits[0].title or hits[0].snippet[:60],
                        "cn": round(hybrid.effective_cn(hits[0]), 2),
                        "evidence": list(hits[0].evidence),
                    }
                )
        return {
            "took_ms": self.took_ms,
            "embedding_ms": self.embedding_ms,
            "candidates": self.counts,
            "top": leaders,
            "extracted": {k: v for k, v in self.extracted.items() if v},
            "ambiguous": bool(self.ambiguity),
            "degraded": self.degraded,
        }

    def to_step(self, *, seq: int = SEQ, status: str = "succeeded") -> dict[str, Any]:
        """The ``node_executions`` row for this pass.

        Written as step ``seq=0`` named ``search``, so the trace panel shows
        the grounding pass above the steps the planner asked for, with the same
        shape as every other step.
        """
        return {
            "id": NODE_ID,
            "node_id": NODE_ID,
            "op": OP_NAME,
            "seq": seq,
            "status": status,
            "depends_on": [],
            "args": {
                "query": self.query,
                "services": list(self.services),
                "expect": self.expect,
                "embedding_skipped": self.embedding_skipped,
                "windows": {
                    name: [start.isoformat(), end.isoformat()]
                    for name, (start, end) in self.windows.items()
                },
            },
            # The trace shows what was found and how sure we are, not every
            # snippet. The full candidates went to the planner and live in the
            # probe.done event; a node row does not need a second copy.
            "result": {
                "took_ms": self.took_ms,
                "embedding_ms": self.embedding_ms,
                "counts": self.counts,
                "decision": self.decision,
                "ambiguity": self.ambiguity,
                "extracted": {k: v for k, v in self.extracted.items() if v},
                "degraded": self.degraded,
                "top": [
                    {
                        "path": self.path(key, index),
                        "label": hit.title or hit.snippet[:60],
                        "cn": round(hybrid.effective_cn(hit), 3),
                        "evidence": list(hit.evidence),
                        **{k: v for k, v in hit.ref.items() if v is not None},
                    }
                    for key in hybrid.SERVICES
                    for index, hit in enumerate(self.services.get(key, [])[:2])
                ],
                "wave2": {
                    "participants": len(self.wave2.get("participants") or []),
                    "threads": len(self.wave2.get("threads") or {}),
                    "replies": len(self.wave2.get("replies") or []),
                    "files_for_event": len(self.wave2.get("files_for_event") or []),
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "took_ms": self.took_ms,
            "embedding_ms": self.embedding_ms,
            "embedding_skipped": self.embedding_skipped,
            "expect": self.expect,
            "counts": self.counts,
            "services": {
                key: result.to_dict() for key, result in self.stats.items()
            },
            "decision": self.decision,
            "ambiguity": self.ambiguity,
            "extracted": self.extracted,
            "degraded": self.degraded,
            "synced_at": {
                key: value.isoformat() if value else None
                for key, value in self.synced_at.items()
            },
            "wave2": self.wave2,
        }

    # -- what the planner reads ------------------------------------------- #

    def to_prompt_block(self, budget: int = 900) -> str:
        """The CANDIDATES block, inside a token budget.

        Every candidate is labelled with the path a plan references it by, so
        the model copies a path instead of retyping a value. Detail is spent
        top-down: the leaders get their extracted fields, the tail gets one
        line, and anything that does not fit is dropped rather than truncated
        mid-fact.
        """
        limit = max(120, int(budget)) * 4  # ~4 characters per token
        header = self._header_line()
        footers = self._footer_lines()

        blocks: list[tuple[int, str, str]] = []
        for key in hybrid.SERVICES:
            for index, hit in enumerate(self.services.get(key, [])):
                blocks.append((index, self._full_block(key, index, hit),
                               self._short_block(key, index, hit)))

        searched = ", ".join(self.services) or "gmail, gcal, gdrive"
        if not blocks:
            body = f"  (nothing found; searched {searched})"
            return "\n".join([header, body, *footers])

        if not self.found_anything:
            # Absence is an answer. Showing near-misses in full detail invites
            # the planner to act on one, so they come back as context, clearly
            # labelled, and the honest reply is "I could not find it".
            near = [short for _, _, short in blocks[:3]]
            return "\n".join(
                [
                    header,
                    f"  NOTHING cleared the relevance floor in {searched}. "
                    "Answer that you could not find it rather than acting on the below.",
                    *near,
                    *footers,
                ]
            )

        chosen: list[str] = []
        used = len(header) + sum(len(f) + 1 for f in footers)
        for rank, full, short in blocks:
            text = full if rank < 2 else short
            if used + len(text) + 1 > limit:
                text = short
            if used + len(text) + 1 > limit:
                break
            chosen.append(text)
            used += len(text) + 1

        return "\n".join([header, *chosen, *footers])

    # -- rendering helpers ------------------------------------------------ #

    def _header_line(self) -> str:
        stamps = [value for value in self.synced_at.values() if value]
        if stamps and self.now:
            lag = max(0, int((self.now - min(stamps)).total_seconds() // 60))
            freshness = f"mirror synced {lag} min ago"
        else:
            freshness = "mirror freshness unknown"
        return (
            f"CANDIDATES ({freshness}). Reference these by path — never retype a value."
        )

    def _footer_lines(self) -> list[str]:
        lines: list[str] = []
        for name, (start, end) in self.windows.items():
            lines.append(f"WINDOW {name}: {start.isoformat()} .. {end.isoformat()}")
        if self.ambiguity:
            candidates = ", ".join(
                str(c.get("path")) for c in self.ambiguity.get("candidates", [])
            )
            lines.append(
                f"AMBIGUOUS: {self.ambiguity['service']} top two within "
                f"{self.ambiguity['margin']:.2f} — ask the user before acting "
                f"({candidates})"
            )
        people = self.wave2.get("participants") or []
        if people:
            names = ", ".join(
                f"{p.get('name') or p.get('email')} <{p.get('email')}>" for p in people[:6]
            )
            lines.append(f"PARTICIPANTS of {self.wave2.get('participants_of', 'top event')}: {names}")
        for entry in self.degraded:
            detail = entry.get("detail") or entry.get("reason")
            lines.append(f"DEGRADED {entry['service']}: {detail}")
        return lines

    def path(self, service: str, index: int) -> str:
        return f"search.{hybrid.service_key(service)}[{index}]"

    def _label(self, service: str, index: int, hit: Hit) -> str:
        flags = ",".join(hit.evidence) if hit.evidence else "—"
        return (
            f"[{self.path(service, index)}] cn {hybrid.effective_cn(hit):.2f} {flags}"
        )

    def _meta_line(self, hit: Hit) -> str:
        parts: list[str] = []
        if hit.from_email:
            parts.append(f"from {hit.from_email}")
        if hit.when:
            parts.append(hit.when.strftime("%Y-%m-%d %H:%M UTC"))
        for name, value in hit.ref.items():
            if value is not None and name != "chunk_index":
                parts.append(f"{name}={value}")
        return "  " + " · ".join(parts)

    def _full_block(self, service: str, index: int, hit: Hit) -> str:
        lines = [self._label(service, index, hit)]
        title = hit.title or hit.snippet[:80] or "(untitled)"
        lines.append(f"  {title}")
        lines.append(self._meta_line(hit))
        if hit.snippet and hit.title:
            lines.append(f"  “{hit.snippet[:160]}”")
        if hit.extracted:
            fields = " · ".join(
                f"{name}={_flatten(value)}"
                for name, value in hit.extracted.items()
                if name not in {"links", "phones", "dates"}
            )
            if fields:
                lines.append(f"  extracted: {fields}")
        return "\n".join(lines)

    def _short_block(self, service: str, index: int, hit: Hit) -> str:
        when = hit.when.strftime("%Y-%m-%d") if hit.when else ""
        title = (hit.title or hit.snippet[:60] or "(untitled)")[:80]
        ref = next(
            (f"{k}={v}" for k, v in hit.ref.items() if v and k != "chunk_index"), ""
        )
        return f"{self._label(service, index, hit)} {title} · {when} · {ref}"


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_flatten(v) for v in value[:3])
    if isinstance(value, Mapping):
        return str(value.get("display") or value.get("url") or value.get("id") or value)
    return str(value)


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


async def probe(
    session: AsyncSession,
    user_id: str,
    prepass: Any,
    *,
    now: dt.datetime | None = None,
    limit: int = TOP_N,
    candidates: int = CANDIDATES,
    wave2: bool = True,
) -> ProbeResult:
    """One embedding, three hybrid searches, extractors, then follow-ups.

    Never raises for a corpus that fails: a dead service becomes a line in
    ``degraded`` and the answer is built from the rest. The only thing that can
    end this call badly is the database being gone entirely, which is not a
    degraded mode — it is ``/readyz`` failing and the pod leaving the pool.
    """
    started = dt.datetime.now(dt.UTC)
    pre = Prepass.of(prepass, now=now)
    anchor = pre.now or now or started
    result = ProbeResult(
        query=pre.query,
        windows=dict(pre.windows),
        expect=pre.expect,
        now=anchor,
    )

    structural = is_structural(pre)
    result.embedding_skipped = structural

    text = pre.search_text or pre.query
    per_service_filters = {key: window_filters(key, pre, anchor) for key in pre.services}

    # The lexical arm needs no vector, so it starts now and the embedding
    # request overlaps with it rather than blocking it.
    lexical_jobs: list[Callable[[AsyncSession], Awaitable[list[dict[str, Any]]]]] = [
        (
            lambda s, key=key: hybrid.lexical_arm(
                s, user_id, key, text, per_service_filters[key], candidates
            )
        )
        for key in pre.services
    ]
    lexical_task = asyncio.create_task(
        hybrid.gather_reads(session, lexical_jobs, return_exceptions=True)
    )

    embedding: list[float] | None = None
    embedding_started = dt.datetime.now(dt.UTC)
    if not structural and text.strip():
        try:
            embedding = await embedder.embed_query(text)
        except Exception as exc:
            log.warning("probe.embedding_failed", error=str(exc), user_id=user_id)
            result.degraded.append(
                {
                    "service": "embedding",
                    "reason": "embedding_failed",
                    "detail": "searched on keywords only",
                }
            )
            embedding = None
        result.embedding_ms = int(
            (dt.datetime.now(dt.UTC) - embedding_started).total_seconds() * 1000
        )
        result.embedding = list(embedding) if embedding else None

    lexical_outcomes = await lexical_task

    vector_outcomes: list[Any] = [None] * len(pre.services)
    if embedding:
        vector_jobs: list[Callable[[AsyncSession], Awaitable[list[dict[str, Any]]]]] = [
            (
                lambda s, key=key: hybrid.vector_arm(
                    s, user_id, key, embedding or [], per_service_filters[key], candidates
                )
            )
            for key in pre.services
        ]
        vector_outcomes = await hybrid.gather_reads(
            session, vector_jobs, return_exceptions=True
        )

    for index, key in enumerate(pre.services):
        lexical = lexical_outcomes[index] if index < len(lexical_outcomes) else []
        vector = vector_outcomes[index] if index < len(vector_outcomes) else []
        errors: list[str] = []
        if isinstance(lexical, BaseException):
            errors.append(f"lexical: {lexical}")
            lexical = []
        if isinstance(vector, BaseException):
            errors.append(f"vector: {vector}")
            vector = []

        hits = hybrid.fuse_arms(
            key,
            vector or [],
            lexical or [],
            terms=pre.terms,
            query=text,
            limit=limit,
            now=anchor,
        )
        result.services[key] = hits
        result.stats[key] = ServiceResult(
            service=key,
            hits=hits,
            vector_candidates=len(vector or []),
            fts_candidates=len(lexical or []),
            error="; ".join(errors) or None,
        )
        if errors:
            log.warning("probe.corpus_failed", service=key, errors=errors, user_id=user_id)
            result.degraded.append(
                {"service": key, "reason": "search_failed", "detail": errors[0]}
            )

    _run_extractors(result, pre, anchor)
    await _stamp_freshness(session, user_id, result, anchor)

    if wave2 and any(hit.qualifies() for hit in result.all_hits):
        await _wave_two(session, user_id, result, pre, anchor, candidates)
        _run_extractors(result, pre, anchor)

    _decide(result, pre)
    result.took_ms = int((dt.datetime.now(dt.UTC) - started).total_seconds() * 1000)

    log.info(
        "probe.done",
        user_id=user_id,
        took_ms=result.took_ms,
        embedding_ms=result.embedding_ms,
        embedding_skipped=result.embedding_skipped,
        counts=result.counts,
        verdict=result.decision.get("verdict"),
    )
    return result


def _run_extractors(result: ProbeResult, pre: Prepass, now: dt.datetime) -> None:
    """Regex over the top candidates, and a roll-up of what came out."""
    rollup: dict[str, list[Any]] = {}
    for key in hybrid.SERVICES:
        for hit in result.services.get(key, [])[:EXTRACT_TOP_N]:
            if not hit.extracted:
                hit.extracted = extractors.extract_from_candidate(
                    hit.row or hit, now=now, codes=pre.terms.code_patterns
                )
            for name, value in hit.extracted.items():
                if name.startswith("_"):
                    continue
                bucket = rollup.setdefault(name, [])
                for item in (value if isinstance(value, list) else [value]):
                    if item not in bucket:
                        bucket.append(item)
    result.extracted = rollup


def _decide(result: ProbeResult, pre: Prepass) -> None:
    """The verdict, and the ambiguity card's contents if there is one.

    Computed per corpus, because two close calendar events are a question and
    two close emails in a different corpus are not the same question.
    """
    # Per corpus, never across them. A booking email and the calendar entry for
    # the same flight are both cn 1.0 and a hundredth apart, and they are not
    # two answers to one question — they are two halves of one. Mixing the
    # corpora would raise an ambiguity card asking the user to choose between
    # their email and their meeting.
    leader = max(
        (key for key in hybrid.SERVICES if result.services.get(key)),
        key=lambda key: (
            hybrid.effective_cn(result.services[key][0]),
            hybrid.raw_cn(result.services[key][0]),
        ),
        default=None,
    )
    result.decision = hybrid.decide(
        result.services.get(leader, []) if leader else [], expect=pre.expect
    )
    result.decision["service"] = leader
    result.ambiguity = None
    if pre.expect != "one":
        return

    for key in hybrid.SERVICES:
        hits = result.services.get(key, [])
        if len(hits) < 2 or not hits[0].qualifies():
            continue
        if not hybrid.is_ambiguous(hits, expect="one"):
            continue
        _, _, gap = hybrid.top_gap(hits)
        result.ambiguity = {
            "service": key,
            "margin": round(gap or 0.0, 4),
            "threshold": float(getattr(settings, "MARGIN", 0.15)),
            # Only candidates that cleared the floor. A card offering a choice
            # between two bookings and a newsletter is not a choice.
            "candidates": [
                {
                    "path": result.path(key, index),
                    "id": hit.id,
                    "label": hit.title or hit.snippet[:60],
                    "when": hit.when.isoformat() if hit.when else None,
                    "cn": round(hybrid.effective_cn(hit), 3),
                    **{k: v for k, v in hit.ref.items() if v is not None},
                }
                for index, hit in enumerate(hits[:4])
                if hit.qualifies()
            ],
        }
        break


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #

#: Beyond twice the sync interval the mirror is stale enough to say so.
def _stale_after_seconds() -> float:
    return float(getattr(settings, "SYNC_INTERVAL_MIN", 15) or 15) * 60.0 * 2


async def _stamp_freshness(
    session: AsyncSession,
    user_id: str,
    result: ProbeResult,
    now: dt.datetime,
) -> None:
    """Stamp candidates with ``synced_at`` and list what is degraded.

    An answer built on a corpus that last synced two hours ago is still a
    useful answer. An answer that does not admit it is worse than an error.
    """
    try:
        freshness = await sync_state.freshness(session, user_id, now=now)
    except Exception as exc:
        log.warning("probe.freshness_failed", error=str(exc), user_id=user_id)
        return

    stale_after = _stale_after_seconds()
    for key in hybrid.SERVICES:
        state = freshness.get(key)
        if state is None:
            result.synced_at[key] = None
            if key in result.services:
                result.degraded.append(
                    {
                        "service": key,
                        "reason": "never_synced",
                        "detail": "this service has not been mirrored yet",
                    }
                )
            continue

        last_success = state.get("last_success_at")
        result.synced_at[key] = last_success
        for hit in result.services.get(key, []):
            hit.synced_at = last_success

        if state.get("circuit_open"):
            result.degraded.append(
                {
                    "service": key,
                    "reason": "circuit_open",
                    "detail": f"{state.get('consecutive_failures', 0)} consecutive failures",
                }
            )
            continue
        if last_success is None:
            result.degraded.append(
                {
                    "service": key,
                    "reason": "never_synced",
                    "detail": "no successful sync yet",
                }
            )
            continue
        lag = state.get("lag_seconds")
        if lag is not None and lag > stale_after:
            result.degraded.append(
                {
                    "service": key,
                    "reason": "stale",
                    # The renderer needs the number, not just the sentence — it
                    # reports the worst lag across services, not each one.
                    "lag_seconds": int(lag),
                    "detail": f"last successful sync {int(lag // 60)} minutes ago",
                }
            )


# --------------------------------------------------------------------------- #
# Wave 2
# --------------------------------------------------------------------------- #


async def _wave_two(
    session: AsyncSession,
    user_id: str,
    result: ProbeResult,
    pre: Prepass,
    now: dt.datetime,
    candidates: int,
) -> None:
    """Deterministic follow-ups on what wave 1 found.

    Not a second guess at the query — a set of joins that are always worth
    making once something has cleared the floor:

    * the participants of the top event, so "the meeting with John" can be
      answered without a second search;
    * the threads behind the top emails, and the replies on them, because the
      newest message in a thread is usually the one that matters;
    * files whose name matches the top event's title, which is how a meeting
      finds its agenda.

    Rules, not judgement, so this costs no model call and always does the same
    thing for the same input.
    """
    wave: dict[str, Any] = {}

    top_event = next((h for h in result.gcal if h.qualifies()), None)
    if top_event is not None:
        attendees = top_event.row.get("attendees") or []
        people: list[dict[str, Any]] = []
        if isinstance(attendees, Sequence) and not isinstance(attendees, (str, bytes)):
            for person in attendees:
                if isinstance(person, Mapping) and person.get("email"):
                    people.append(
                        {
                            "email": str(person.get("email")).lower(),
                            "name": person.get("name"),
                            "response_status": person.get("response_status"),
                            "optional": bool(person.get("optional")),
                        }
                    )
        organiser = top_event.row.get("organizer_email")
        if organiser and not any(p["email"] == str(organiser).lower() for p in people):
            people.insert(0, {"email": str(organiser).lower(), "name": None,
                              "response_status": "organizer", "optional": False})
        if people:
            wave["participants"] = people
            wave["participants_of"] = top_event.title or top_event.event_id

    thread_ids: list[str] = []
    for hit in result.gmail[:2]:
        if hit.qualifies() and hit.thread_id and hit.thread_id not in thread_ids:
            thread_ids.append(str(hit.thread_id))

    jobs: list[Callable[[AsyncSession], Awaitable[Any]]] = []
    job_names: list[str] = []
    for thread_id in thread_ids:
        jobs.append(lambda s, tid=thread_id: hybrid.get_thread(s, user_id, tid))
        job_names.append(f"thread:{thread_id}")

    title = (top_event.title if top_event is not None else "") or ""
    if title.strip():
        jobs.append(
            lambda s, t=title: hybrid.search(
                s, user_id, "gdrive", t,
                window_filters("gdrive", pre, now),
                limit=3, embedding=None, terms=pre.terms, candidates=candidates, now=now,
            )
        )
        job_names.append("files_for_event")

    if not jobs:
        result.wave2 = wave
        return

    outcomes = await hybrid.gather_reads(session, jobs, return_exceptions=True)

    threads: dict[str, list[dict[str, Any]]] = {}
    replies: list[dict[str, Any]] = []
    for name, outcome in zip(job_names, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.warning("probe.wave2_failed", job=name, error=str(outcome), user_id=user_id)
            continue
        if name.startswith("thread:"):
            thread_id = name.split(":", 1)[1]
            messages = [
                {
                    "message_id": row.get("message_id"),
                    "subject": row.get("subject"),
                    "from_email": row.get("from_email"),
                    "received_at": row["received_at"].isoformat()
                    if isinstance(row.get("received_at"), dt.datetime)
                    else row.get("received_at"),
                    "snippet": " ".join(str(row.get("body_clean") or "").split())[:160],
                }
                for row in outcome
            ]
            threads[thread_id] = messages
            anchor_id = next(
                (h.message_id for h in result.gmail if h.thread_id == thread_id), None
            )
            seen_anchor = anchor_id is None
            for message in messages:
                if message["message_id"] == anchor_id:
                    seen_anchor = True
                    continue
                if seen_anchor:
                    replies.append({"thread_id": thread_id, **message})
        elif name == "files_for_event":
            existing = {hit.id for hit in result.gdrive}
            related = []
            for hit in outcome:
                hit.wave = 2
                related.append(hit)
                if hit.id not in existing:
                    result.services.setdefault("gdrive", []).append(hit)
            if related:
                wave["files_for_event"] = [hit.to_binding() for hit in related]

    if threads:
        wave["threads"] = threads
    if replies:
        wave["replies"] = replies
    result.wave2 = wave


# --------------------------------------------------------------------------- #
# Convenience for callers that have no pre-pass yet
# --------------------------------------------------------------------------- #


async def probe_query(
    session: AsyncSession,
    user_id: str,
    query: str,
    *,
    now: dt.datetime | None = None,
    services: Sequence[str] | None = None,
    limit: int = TOP_N,
) -> ProbeResult:
    """Run the probe straight off a query string. Debugging and evaluation."""
    pre = Prepass.of({"query": query, "services": list(services or hybrid.SERVICES)}, now=now)
    return await probe(session, user_id, pre, now=now, limit=limit)


async def entity_rows(
    session: AsyncSession,
    user_id: str,
    service: str,
    refs: Sequence[str],
) -> list[dict[str, Any]]:
    """Mirror rows for provider ids the conversation already mentioned.

    Used when the pre-pass resolved "that email" to a ``conversation_entities``
    row: the id is known, so there is nothing to search for.
    """
    if not refs:
        return []
    key = hybrid.service_key(service)
    # Same projection the search path returns, so a row fetched by id and a row
    # that was searched for are the same shape to everything downstream.
    return await mirror.list_filtered(
        session,
        user_id,
        key,
        {"external_ids": list(refs), "chunk_index": 0},
        limit=max(len(refs), 1),
    )


__all__ = [
    "CANDIDATES",
    "EXTRACT_TOP_N",
    "NODE_ID",
    "OP_NAME",
    "SEQ",
    "TOP_N",
    "Prepass",
    "ProbeResult",
    "entity_rows",
    "is_structural",
    "probe",
    "probe_query",
    "topic_words",
    "window_filters",
]
