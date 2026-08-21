"""Hybrid search: two arms, one fusion, and a hard line between ordering and
deciding.

The shape of a query, from the top:

1. **Metadata prefilter.** Always ``user_id`` — no query in this system can
   cross a tenant — plus whatever the caller knows: a date range, a sender, an
   attendee, a mime type, a label set.
2. **Two arms over that filtered set, in parallel.** Vector cosine through the
   HNSW index, and Postgres full text through the GIN index on ``tsv``. The
   lexical arm needs no embedding, so a caller that has not got one yet can
   fire it at t=0 and let the two overlap.
3. **RRF at k=60, for ordering only.** Two second places beat one first place,
   which is the property reciprocal rank fusion exists for.
4. **Temporal shaping.** Mail decays — ``score * exp(-age_days / 30)``. Events
   lean forward: something coming up matters more than something finished.

The order a caller finally sees is ``(evidence, cn * decay * boost)`` and the
reported ``final`` is ``rrf * decay * boost``, kept separate because they answer
different questions. Ordering on ``final`` alone does not survive contact with
a real mailbox: consecutive RRF ranks differ by about a ten-thousandth while
decay swings by a factor of three across a fortnight, so the newest candidate
would win every time and search would be the inbox sorted by date. RRF still
does what it is good at — merging two rankings — and nothing decides anything
on it.

And then the part that matters most:

**Decisions are never made on the fused score.** A document that came first in
both arms scores 2/61 ≈ 0.0328. Compare that to ``FLOOR_READ`` 0.55 and every
candidate in the system fails, forever. RRF is rank-derived — the best of three
bad matches gets exactly the same number as a perfect one. So relevance and
ambiguity are decided on ``cn`` (cosine normalised per corpus, z-scored,
clamped to 0..1) together with an ``evidence`` flag, and ``cn`` is what the
planner sees.

Evidence is the escape hatch for the case similarity cannot solve: an English
query does not embed near a Turkish body, but the sender really is
``bilet@thy.com`` and the subject really does say TK1984. An exact match on an
id, a sender, a filename or an alias token forces ``cn`` to 1.0.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.repositories import mirror
from app.search import aliases as alias_table
from app.search.chunking import filename_words

log = get_logger(__name__)

#: Reciprocal rank fusion constant. 60 is the value from the original paper and
#: the one every published comparison uses; it is not tuned here and saying so
#: is more honest than implying it was.
RRF_K = 60

#: How many rows each arm brings back before fusion.
DEFAULT_CANDIDATES = 40

#: Mail half-life-ish constant, in days. ``exp(-age / 30)``.
MAIL_DECAY_DAYS = 30.0
#: Files change less often than mail arrives, so they fade more slowly.
FILE_DECAY_DAYS = 90.0
#: How fast the forward boost fades as an event moves further out.
EVENT_HORIZON_DAYS = 45.0
#: The most a soon-ish event can gain.
EVENT_MAX_BOOST = 0.6

#: How far a z-score has to travel for cn to move half a unit. Chosen so a
#: candidate a bit over one standard deviation above its corpus lands near the
#: 0.87 in docs/API.md rather than saturating at 1.0.
CN_SPREAD = 4.0

EVIDENCE_EXACT_ID = "EXACT_ID"
EVIDENCE_EXACT_SENDER = "EXACT_SENDER"
EVIDENCE_EXACT_FILENAME = "EXACT_FILENAME"
EVIDENCE_ALIAS_TOKEN = "ALIAS_TOKEN_IN_SUBJECT"

#: Every flag in this set forces cn to 1.0.
EXACT_EVIDENCE: frozenset[str] = frozenset(
    {EVIDENCE_EXACT_ID, EVIDENCE_EXACT_SENDER, EVIDENCE_EXACT_FILENAME, EVIDENCE_ALIAS_TOKEN}
)

SERVICES: tuple[str, ...] = ("gmail", "gcal", "gdrive")

_SERVICE_ALIASES = {
    "gmail": "gmail", "mail": "gmail", "email": "gmail", "sync_gmail": "gmail",
    "gcal": "gcal", "calendar": "gcal", "cal": "gcal", "events": "gcal", "sync_gcal": "gcal",
    "gdrive": "gdrive", "drive": "gdrive", "files": "gdrive", "sync_gdrive": "gdrive",
}

#: Friendly filter names the ops and the debug endpoint use, mapped onto the
#: whitelist in each mirror spec.
_FILTER_ALIASES: dict[str, dict[str, str]] = {
    "gmail": {
        "from": "from_email", "sender": "from_email", "senders": "from_emails",
        "from_emails": "from_emails", "to": "to_email", "recipient": "to_email",
        "label": "labels", "start": "since", "end": "until", "after": "since",
        "before": "until", "ids": "message_ids", "subject": "subject_contains",
    },
    "gcal": {
        "attendee": "attendee_emails", "attendees": "attendee_emails",
        "organizer": "organizer_email", "start": "since", "end": "until",
        "after": "since", "before": "until", "ids": "event_ids",
    },
    "gdrive": {
        "mime": "mime_type", "mimes": "mime_types", "owner": "owner_email",
        "start": "since", "end": "until", "after": "since", "before": "until",
        "ids": "file_ids", "name": "name_contains", "folder": "folder_prefix",
    },
}

_SINGLE_TO_LIST = {
    "gmail": {"labels"},
    "gcal": {"attendee_emails"},
    "gdrive": set(),
}


def service_key(service: str) -> str:
    """``"calendar"`` and ``"sync_gcal"`` both mean ``"gcal"``."""
    key = _SERVICE_ALIASES.get(str(service).strip().lower())
    if key is None:
        raise AppError(
            "VALIDATION_ERROR",
            f"Unknown service {service!r}.",
            http=422,
            details={"service": service, "known": list(SERVICES)},
        )
    return key


def _floor_read() -> float:
    return float(getattr(settings, "FLOOR_READ", 0.55))


def _floor_write() -> float:
    return float(getattr(settings, "FLOOR_WRITE", 0.80))


def _margin() -> float:
    return float(getattr(settings, "MARGIN", 0.15))


# --------------------------------------------------------------------------- #
# Terms: what "exact" means for this query
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Terms:
    """The literals the pre-pass pulled out of the query.

    These are what turn a similarity number into evidence. They come from the
    pre-pass, never from the model, which is the point: the model cannot talk
    us into believing a match is exact.
    """

    aliases: tuple[str, ...] = ()
    sender_domains: tuple[str, ...] = ()
    emails: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    code_patterns: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(
            self.aliases or self.sender_domains or self.emails or self.ids or self.filenames
        )

    @classmethod
    def from_query(cls, query: str | None, extra: Terms | None = None) -> Terms:
        """Everything derivable from the query text alone."""
        from app.search.extractors import find_emails

        text = query or ""
        groups = alias_table.detect(text)
        tokens: list[str] = []
        domains: list[str] = []
        patterns: list[str] = []
        for group in groups:
            tokens.extend(group.tokens_in(text) or [alias_table.normalise(group.canonical)])
            for token in group.tokens:
                normalised = alias_table.normalise(token)
                if normalised and normalised not in tokens:
                    tokens.append(normalised)
            domains.extend(group.sender_domains)
            patterns.extend(group.code_patterns)
        emails = tuple(str(m.value) for m in find_emails(text))
        built = cls(
            aliases=tuple(dict.fromkeys(tokens)),
            sender_domains=tuple(dict.fromkeys(domains)),
            emails=emails,
            code_patterns=tuple(dict.fromkeys(patterns)),
        )
        return built.merge(extra) if extra else built

    def merge(self, other: Terms | None) -> Terms:
        if other is None:
            return self
        return Terms(
            aliases=tuple(dict.fromkeys(self.aliases + other.aliases)),
            sender_domains=tuple(dict.fromkeys(self.sender_domains + other.sender_domains)),
            emails=tuple(dict.fromkeys(self.emails + other.emails)),
            ids=tuple(dict.fromkeys(self.ids + other.ids)),
            filenames=tuple(dict.fromkeys(self.filenames + other.filenames)),
            code_patterns=tuple(dict.fromkeys(self.code_patterns + other.code_patterns)),
        )


# --------------------------------------------------------------------------- #
# A hit
# --------------------------------------------------------------------------- #


@dataclass
class Hit:
    """One candidate, with every score component kept separate.

    ``cn`` and ``evidence`` decide. ``rrf`` and ``final`` order. Keeping both on
    the object is what lets ``/api/v1/search`` show why something ranked where
    it did instead of asserting that it should have.
    """

    id: str
    service: str
    ref: dict[str, Any]
    title: str = ""
    snippet: str = ""
    when: dt.datetime | None = None
    from_email: str | None = None

    cos: float = 0.0
    lex: float = 0.0
    cn: float = 0.0
    cn_raw: float = 0.0
    vec_rank: int | None = None
    fts_rank: int | None = None
    rrf: float = 0.0
    decay: float = 1.0
    boost: float = 1.0
    final: float = 0.0
    #: The presented order: ``cn_raw * decay * boost``, under an evidence tier.
    sort: float = 0.0

    evidence: list[str] = field(default_factory=list)
    extracted: dict[str, Any] = field(default_factory=dict)
    synced_at: dt.datetime | None = None
    wave: int = 1
    row: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- the identifiers a plan references -------------------------------- #

    @property
    def message_id(self) -> str | None:
        return self.ref.get("message_id")

    @property
    def thread_id(self) -> str | None:
        return self.ref.get("thread_id")

    @property
    def event_id(self) -> str | None:
        return self.ref.get("event_id")

    @property
    def file_id(self) -> str | None:
        return self.ref.get("file_id")

    @property
    def is_exact(self) -> bool:
        return any(flag in EXACT_EVIDENCE for flag in self.evidence)

    @property
    def scores(self) -> dict[str, Any]:
        """Every component, separately. Never collapse these into one number."""
        return self.to_dict()["scores"]

    def qualifies(self, floor: float | None = None) -> bool:
        return qualifies(self, floor=floor)

    def to_dict(self) -> dict[str, Any]:
        """The ``/api/v1/search`` hit shape."""
        scores: dict[str, Any] = {
            "cosine": round(self.cos, 4),
            "cn": round(self.cn, 4),
            "cn_raw": round(self.cn_raw, 4),
            "ts_rank": round(self.lex, 4),
            "vec_rank": self.vec_rank,
            "fts_rank": self.fts_rank,
            "rrf": round(self.rrf, 6),
            "decay": round(self.decay, 4),
            "final": round(self.final, 6),
            "sort": round(self.sort, 6),
        }
        if self.service == "gcal":
            scores["boost"] = round(self.boost, 4)
        return {
            "id": self.id,
            "service": self.service,
            "ref": dict(self.ref),
            "title": self.title,
            "snippet": self.snippet,
            "when": self.when.isoformat() if self.when else None,
            "from": self.from_email,
            "scores": scores,
            "evidence": list(self.evidence),
            "extracted": dict(self.extracted),
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
        }

    def to_binding(self) -> dict[str, Any]:
        """What ``{{search.gmail[0].…}}`` resolves against.

        Flat, plain types, and it carries the provider ids rather than our row
        id, because a plan step calls Google with the former.
        """
        out: dict[str, Any] = {
            "id": self.id,
            "service": self.service,
            "title": self.title,
            "snippet": self.snippet,
            "when": self.when.isoformat() if self.when else None,
            "cn": round(self.cn, 4),
            "evidence": list(self.evidence),
            "extracted": dict(self.extracted),
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
        }
        out.update({k: v for k, v in self.ref.items() if v is not None})
        if self.from_email:
            out["from_email"] = self.from_email
        for name in ("subject", "name", "starts_at", "ends_at", "etag", "web_view_link",
                     "attendees", "organizer_email", "location", "labels", "mime_type"):
            value = self.row.get(name)
            if value is not None:
                out.setdefault(name, value.isoformat() if isinstance(value, dt.datetime) else value)
        return out


@dataclass
class ServiceResult:
    """Hits plus the counters ``/api/v1/search`` prints beside them."""

    service: str
    hits: list[Hit] = field(default_factory=list)
    took_ms: int = 0
    prefiltered: int | None = None
    vector_candidates: int = 0
    fts_candidates: int = 0
    error: str | None = None

    @property
    def returned(self) -> int:
        return len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "took_ms": self.took_ms,
            "vector_candidates": self.vector_candidates,
            "fts_candidates": self.fts_candidates,
            "returned": self.returned,
            "hits": [hit.to_dict() for hit in self.hits],
        }
        if self.prefiltered is not None:
            out["prefiltered"] = self.prefiltered
        if self.error:
            out["error"] = self.error
        return out


# --------------------------------------------------------------------------- #
# Pure ranking: fusion, normalisation, floors, ambiguity, time
# --------------------------------------------------------------------------- #


def rrf_fuse(
    rankings: Sequence[Sequence[Any]] | Sequence[Any],
    *more: Sequence[Any],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Any, float]]:
    """Reciprocal rank fusion over any number of ranked id lists.

    ``score(d) = Σ 1 / (k + rank(d))`` over the lists ``d`` appears in, with
    ranks 1-based. A document missing from a list contributes nothing from it —
    not a penalty, and not ``1/k``; it simply is not there.

    Accepts either one sequence of rankings or several ranking arguments.
    Ties break on the id, so the same input always gives the same order.
    """
    lists: list[Sequence[Any]] = []
    if more:
        lists = [rankings, *more]  # type: ignore[list-item]
    elif rankings and isinstance(rankings[0], (list, tuple)):
        lists = list(rankings)  # type: ignore[arg-type]
    elif rankings:
        lists = [rankings]  # type: ignore[list-item]

    scores: dict[Any, float] = {}
    order: list[Any] = []
    for index, ranking in enumerate(lists):
        weight = 1.0 if weights is None else float(weights[index])
        for position, item in enumerate(ranking or (), start=1):
            key = item[0] if isinstance(item, (tuple, list)) else item
            if key not in scores:
                scores[key] = 0.0
                order.append(key)
            scores[key] += weight / (k + position)

    return sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))


def cn_transform(cosines: Sequence[float], spread: float = CN_SPREAD) -> Callable[[float], float]:
    """A function mapping a raw cosine to ``cn`` for this corpus.

    Z-score against the candidate pool, then ``0.5 + z / spread`` clamped to
    0..1. Per corpus because absolute cosine means different things over mail,
    over calendar titles and over file names — a calendar entry is six words
    long and nothing embeds close to it.

    Degenerate pools — fewer than three candidates, or every cosine identical —
    fall back to the clamped raw cosine. Inventing a ranking out of no variance
    is worse than admitting there is none.
    """
    values = [float(c) for c in cosines]
    count = len(values)
    if count < 3:
        return _clamp01
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / count
    deviation = math.sqrt(variance)
    if deviation < 1e-9:
        return _clamp01

    def transform(cosine: float) -> float:
        return _clamp01(0.5 + ((float(cosine) - mean) / deviation) / spread)

    return transform


def cn_scores(cosines: Sequence[float], spread: float = CN_SPREAD) -> list[float]:
    """``cn`` for a list of raw cosines, order preserved."""
    transform = cn_transform(cosines, spread)
    return [transform(c) for c in cosines]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def _read(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def has_exact_evidence(candidate: Any) -> bool:
    """True when a candidate carries any exact-match flag.

    Accepts the flag as a list, as a set, as a bare string, or as the
    ``EXACT(sender-domain, code-pattern)`` shorthand the documents use.
    """
    evidence = _read(candidate, "evidence")
    if not evidence:
        return False
    if isinstance(evidence, str):
        text = evidence.upper()
        return "EXACT" in text
    if isinstance(evidence, Mapping):
        return any(bool(v) for v in evidence.values())
    if isinstance(evidence, Iterable):
        return any("EXACT" in str(flag).upper() or str(flag).upper() in EXACT_EVIDENCE
                   for flag in evidence)
    return bool(evidence)


def effective_cn(candidate: Any) -> float:
    """``cn``, with exact evidence forcing 1.0.

    This is the number every threshold in the system is compared against, and
    the reason a Turkish booking confirmation that embeds badly still wins:
    the sender domain and the subject code are facts, and a similarity score is
    an opinion.
    """
    if has_exact_evidence(candidate):
        return 1.0
    value = _read(candidate, "cn")
    if value is None:
        value = _read(candidate, "cosine", _read(candidate, "cos", 0.0))
    try:
        return _clamp01(float(value))
    except (TypeError, ValueError):
        return 0.0


def qualifies(candidate: Any, *, floor: float | None = None) -> bool:
    """Is this candidate worth showing (or, at ``FLOOR_WRITE``, writing against)?

    A disjunction, not a similarity test: ``cn >= floor`` **or** exact
    evidence. Both halves are needed. Without the floor, everything qualifies;
    without the evidence clause, an email in a language the query is not
    written in never does.
    """
    limit = _floor_read() if floor is None else float(floor)
    return effective_cn(candidate) >= limit


def meets_floor(candidate: Any, *, floor: float | None = None) -> bool:
    """Alias of :func:`qualifies`."""
    return qualifies(candidate, floor=floor)


def raw_cn(candidate: Any) -> float:
    """``cn`` before evidence flattened it, when the candidate carries one."""
    value = _read(candidate, "cn_raw")
    if value is None:
        value = _read(candidate, "cn", 0.0)
    try:
        return _clamp01(float(value))
    except (TypeError, ValueError):
        return 0.0


def top_gap(hits: Sequence[Any]) -> tuple[float, float | None, float | None]:
    """``(top cn, runner-up cn, gap)`` for a candidate list.

    The gap is measured on ``cn`` after evidence is applied. When evidence has
    flattened the leaders to an exact tie at 1.0 — the airline sent both the
    booking and the marketing email, so both match the sender exactly — the
    similarity underneath decides whether they are really a tie. Without that
    step every corpus containing two emails from one vendor would look
    ambiguous, and the person would be asked to choose between their booking
    and an advertisement.
    """
    scored = sorted(
        ((effective_cn(hit), raw_cn(hit)) for hit in hits or ()),
        key=lambda pair: (-pair[0], -pair[1]),
    )
    if not scored:
        return (0.0, None, None)
    if len(scored) == 1:
        return (scored[0][0], None, None)
    top, second = scored[0], scored[1]
    gap = top[0] - second[0]
    if gap <= 0.0:
        gap = abs(top[1] - second[1])
    return (top[0], second[0], gap)


def is_ambiguous(
    hits: Sequence[Any],
    *,
    expect: str = "one",
    margin: float | None = None,
) -> bool:
    """Are the top two too close to call?

    Only when the query wants exactly one thing. "Find emails about the budget"
    with three near-identical candidates is a list, not a question — asking
    there would be asking about something the person did not ask about.

    The gap is measured on ``cn``, never on the fused score, so a leader that
    matches on the sender and a runner-up that matches on nothing are not a tie
    however close their fused scores are. Exactly ``MARGIN`` apart is decided:
    the test is ``gap < margin``.
    """
    if str(expect).lower() not in {"one", "1", "single"}:
        return False
    _, runner_up, gap = top_gap(hits)
    if runner_up is None or gap is None:
        return False
    limit = _margin() if margin is None else float(margin)
    return gap < limit


def ambiguous(hits: Sequence[Any], *, expect: str = "one",
              margin: float | None = None) -> bool:
    """Alias of :func:`is_ambiguous`."""
    return is_ambiguous(hits, expect=expect, margin=margin)


def _age_days(value: float | dt.datetime | None, now: dt.datetime | None,
              *, forward: bool = False) -> float:
    """Days between ``value`` and ``now``, however the caller expressed it."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dt.datetime):
        anchor = now or dt.datetime.now(dt.UTC)
        moment = value if value.tzinfo else value.replace(tzinfo=dt.UTC)
        anchor = anchor if anchor.tzinfo else anchor.replace(tzinfo=dt.UTC)
        delta = (moment - anchor) if forward else (anchor - moment)
        return delta.total_seconds() / 86400.0
    raise TypeError(f"cannot read an age from {value!r}")


def temporal_decay(
    score: float,
    age: float | dt.datetime | None,
    *,
    now: dt.datetime | None = None,
    half_life_days: float = MAIL_DECAY_DAYS,
) -> float:
    """``score * exp(-age_days / 30)``. Mail, and more gently, files.

    A tilt, not an override. A much better match five days old still beats a
    weak one from yesterday — otherwise search would just be the mailbox sorted
    by ``received_at``, which the user already has.

    ``age`` is days, or the timestamp to measure from.
    """
    days = _age_days(age, now)
    if days <= 0:  # future-dated mail is not fresher than now
        days = 0.0
    scale = max(1e-9, float(half_life_days))
    return float(score) * math.exp(-days / scale)


def decay(score: float, age: float | dt.datetime | None, *,
          now: dt.datetime | None = None,
          half_life_days: float = MAIL_DECAY_DAYS) -> float:
    """Alias of :func:`temporal_decay`."""
    return temporal_decay(score, age, now=now, half_life_days=half_life_days)


def forward_boost(
    score: float,
    horizon: float | dt.datetime | None,
    *,
    now: dt.datetime | None = None,
    max_boost: float = EVENT_MAX_BOOST,
    fade_days: float = EVENT_HORIZON_DAYS,
) -> float:
    """Calendar's answer to decay: the future is what matters.

    ``horizon`` is days from now (negative for the past) or the event's start.
    A future event is multiplied by ``1 + max_boost * exp(-days / fade_days)``,
    strongest for something imminent and fading to nothing by next year. A past
    event decays like mail, so "my next flight" cannot return last year's.

    Applying mail's decay to a calendar would rank the meeting that finished an
    hour ago above the one the user is about to walk into.
    """
    days = _age_days(horizon, now, forward=True)
    if days >= 0:
        return float(score) * (1.0 + float(max_boost) * math.exp(-days / max(1e-9, fade_days)))
    return temporal_decay(score, -days, half_life_days=MAIL_DECAY_DAYS)


def decide(
    hits: Sequence[Any],
    *,
    expect: str = "one",
    floor: float | None = None,
    write_floor: float | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """The verdict block: confident, ambiguous, or absent.

    Every number in it is computed on ``cn``. None of them is computed on the
    fused score, and the block says so by carrying the thresholds it used.
    """
    read_floor = _floor_read() if floor is None else float(floor)
    strict_floor = _floor_write() if write_floor is None else float(write_floor)
    gap_limit = _margin() if margin is None else float(margin)

    top, runner_up, gap = top_gap(hits)
    close = is_ambiguous(hits, expect=expect, margin=gap_limit)
    above_read = bool(hits) and top >= read_floor

    if not above_read:
        verdict = "absent"
    elif close:
        verdict = "ambiguous"
    else:
        verdict = "confident"

    return {
        "top_cn": round(top, 4),
        "runner_up_cn": round(runner_up, 4) if runner_up is not None else None,
        "margin": round(gap, 4) if gap is not None else None,
        "floor_read": read_floor,
        "floor_write": strict_floor,
        "margin_threshold": gap_limit,
        "above_read_floor": above_read,
        "above_write_floor": bool(hits) and top >= strict_floor,
        "ambiguous": close,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def _row_ids(service: str, row: Mapping[str, Any]) -> list[str]:
    if service == "gmail":
        return [str(v) for v in (row.get("message_id"), row.get("thread_id")) if v]
    if service == "gcal":
        return [str(v) for v in (row.get("event_id"), row.get("recurring_event_id")) if v]
    return [str(v) for v in (row.get("file_id"),) if v]


def _row_title(service: str, row: Mapping[str, Any]) -> str:
    if service == "gmail":
        return str(row.get("subject") or "")
    if service == "gcal":
        return str(row.get("title") or "")
    return str(row.get("name") or "")


def _row_sender(service: str, row: Mapping[str, Any]) -> str | None:
    if service == "gmail":
        return row.get("from_email")
    if service == "gcal":
        return row.get("organizer_email")
    return row.get("owner_email")


#: What makes a filename an identifier rather than a description.
#:
#: One ordinary word never identifies a file, however long — "presentation" is
#: twelve characters and names half the Drive. Two words might; a word carrying
#: digits almost certainly does, because serials, dates and version codes are
#: how files actually get told apart.
_NAME_MIN_TOKENS: Final[int] = 2


def _is_identifier(name: str) -> bool:
    """Whether a filename is distinctive enough to identify a file by itself."""
    if not name:
        return False
    tokens = name.split()
    if len(tokens) >= _NAME_MIN_TOKENS:
        return True
    # A lone token still counts when it carries digits — "20260112081205",
    # "TK1234", "v3". Those are identifiers; "presentation" is not.
    return any(ch.isdigit() for ch in name)


def _names_the_file(name_norm: str, stem_norm: str, query_norm: str) -> bool:
    """Whether the query is actually naming *this* file.

    Plain containment is not enough, and getting that wrong is expensive. A
    file called "presentation" is a substring of
    "xuv_3xo_presentation_20260112081205", so a substring test hands it
    `EXACT_FILENAME` — which pins its score at 1.0 and makes four unrelated
    files tie with the one that was actually named. The person then gets asked
    to choose between them after typing the full filename.

    So: equality always counts, and appearing inside a longer query counts only
    when the name is distinctive enough to be an identifier rather than a
    common word — several tokens, or long enough that the coincidence is
    implausible — and only on token boundaries.
    """
    for candidate in (name_norm, stem_norm):
        if not candidate:
            continue
        if candidate == query_norm:
            return True
        if not _is_identifier(candidate):
            continue
        # Token boundaries, so "port" does not match "airport".
        padded_query = f" {query_norm} "
        if f" {candidate} " in padded_query:
            return True
    return False


def evidence_for(
    service: str,
    row: Mapping[str, Any],
    terms: Terms | None,
    query: str | None = None,
) -> list[str]:
    """The exact-match flags this row earns, in the order the API documents.

    Four ways to be exact, and each one is a fact rather than a similarity:
    the query named the id; the sender is the address or the vendor's domain;
    the filename is what was typed; an alias token is in the subject.
    """
    flags: list[str] = []
    terms = terms or Terms()
    query_norm = alias_table.normalise(query)

    ids = {str(i) for i in terms.ids}
    if ids:
        for identifier in _row_ids(service, row):
            if identifier in ids:
                flags.append(EVIDENCE_EXACT_ID)
                break
    elif query_norm:
        for identifier in _row_ids(service, row):
            if len(identifier) >= 8 and alias_table.normalise(identifier) in query_norm:
                flags.append(EVIDENCE_EXACT_ID)
                break

    sender = _row_sender(service, row)
    if sender:
        address = str(sender).lower()
        domain = address.rsplit("@", 1)[-1]
        known_domain = any(
            domain == d or domain.endswith("." + d) for d in terms.sender_domains
        )
        if address in {e.lower() for e in terms.emails} or known_domain:
            flags.append(EVIDENCE_EXACT_SENDER)

    title = _row_title(service, row)
    if service == "gdrive" and title:
        name_norm = alias_table.normalise(title)
        stem_norm = alias_table.normalise(filename_words(title))
        wanted = {alias_table.normalise(f) for f in terms.filenames if f}
        named = {n for n in (name_norm, stem_norm) if n and n in wanted}
        # A term the query happened to contain is not the same as the file's
        # name. "presentation" is a word in half the Drive; matching it must
        # not carry the weight of somebody typing the whole filename.
        if any(_is_identifier(n) for n in named):
            flags.append(EVIDENCE_EXACT_FILENAME)
        elif query_norm and name_norm and _names_the_file(name_norm, stem_norm, query_norm):
            flags.append(EVIDENCE_EXACT_FILENAME)

    if title and (terms.aliases or terms.code_patterns):
        title_norm = alias_table.normalise(title)
        if any(_token_in(title_norm, token) for token in terms.aliases):
            flags.append(EVIDENCE_ALIAS_TOKEN)
        elif any(_matches_code(title, pattern) for pattern in terms.code_patterns):
            # "Istanbul → JFK, TK1988" carries no alias word, only the vendor's
            # own code. That is the same fact by another spelling.
            flags.append(EVIDENCE_ALIAS_TOKEN)

    return flags


def _matches_code(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return False


def _token_in(haystack_norm: str, token: str) -> bool:
    token_norm = alias_table.normalise(token)
    if not token_norm or not haystack_norm:
        return False
    if len(token_norm) < 2:
        return False
    padded = f" {haystack_norm} "
    return f" {token_norm} " in padded


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def normalise_filters(service: str, filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Map friendly filter names onto the mirror spec's whitelist.

    ``user_id`` is not a filter here — the repository always applies it and
    there is deliberately no way to ask it not to.
    """
    key = service_key(service)
    spec = mirror.spec_for(key)
    renames = _FILTER_ALIASES[key]
    listify = _SINGLE_TO_LIST[key]

    out: dict[str, Any] = {}
    for raw_name, value in (filters or {}).items():
        if value is None or raw_name in {"user_id", "text", "query", "q"}:
            if raw_name in {"text", "query", "q"} and value:
                out["text"] = str(value)
            continue
        name = renames.get(raw_name, raw_name)
        if name not in spec.filters:
            raise AppError(
                "VALIDATION_ERROR",
                f"{raw_name!r} is not a filter on {key}.",
                http=422,
                details={"filter": raw_name, "known": sorted(set(spec.filters) | set(renames))},
            )
        if name in listify and not isinstance(value, (list, tuple, set)):
            value = [value]
        out[name] = list(value) if isinstance(value, (set, tuple)) else value
    return out


# --------------------------------------------------------------------------- #
# Running the arms
# --------------------------------------------------------------------------- #


async def gather_reads[T](
    session: AsyncSession,
    jobs: Sequence[Callable[[AsyncSession], Awaitable[T]]],
    *,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Run read-only jobs concurrently, each on its own connection.

    One ``AsyncSession`` is one connection and cannot serve two queries at
    once, so genuine parallelism needs a session per job. Siblings are built
    from the caller's own bind, so nothing here can end up pointed at a
    different database than the caller is using.

    When the bind is not an async engine — a caller that handed us a session
    made some other way — the jobs run in order on the caller's session. Same
    answers, no parallelism, and no surprises.

    Siblings do not see the caller's uncommitted writes. That is correct for
    the mirror, which is written by the sync tasks and committed before any
    search runs.
    """
    if not jobs:
        return []
    if len(jobs) == 1:
        if return_exceptions:
            try:
                return [await jobs[0](session)]
            except Exception as exc:
                return [exc]
        return [await jobs[0](session)]

    bind = getattr(session, "bind", None)
    if not isinstance(bind, AsyncEngine):
        out: list[T | BaseException] = []
        for job in jobs:
            try:
                out.append(await job(session))
            except Exception as exc:
                if not return_exceptions:
                    raise
                out.append(exc)
        return out

    async def run(job: Callable[[AsyncSession], Awaitable[T]]) -> T:
        sibling = AsyncSession(bind=bind, expire_on_commit=False, autoflush=False)
        try:
            return await job(sibling)
        finally:
            await sibling.close()

    return list(
        await asyncio.gather(*(run(job) for job in jobs), return_exceptions=return_exceptions)
    )


async def vector_arm(
    session: AsyncSession,
    user_id: str,
    service: str,
    embedding: Sequence[float],
    filters: Mapping[str, Any] | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[dict[str, Any]]:
    """Cosine over the prefiltered set, best first. One row per document."""
    if embedding is None or len(embedding) == 0:
        return []
    key = service_key(service)
    work = dict(normalise_filters(key, filters))
    work.pop("text", None)
    return await mirror.hybrid_search(
        session, user_id, key, list(embedding), work, limit=candidates,
        candidates=candidates, w_vec=1.0, w_lex=0.0,
    )


async def lexical_arm(
    session: AsyncSession,
    user_id: str,
    service: str,
    text: str | None,
    filters: Mapping[str, Any] | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[dict[str, Any]]:
    """Postgres full text over the same prefiltered set.

    Needs no embedding, which is why the probe fires it at t=0 and lets the
    embedding request overlap with it.
    """
    key = service_key(service)
    work = dict(normalise_filters(key, filters))
    query_text = (text or work.get("text") or "").strip()
    if not query_text:
        return []
    work["text"] = query_text
    return await mirror.hybrid_search(
        session, user_id, key, None, work, limit=candidates,
        candidates=candidates, w_vec=0.0, w_lex=1.0,
    )


async def count_prefiltered(
    session: AsyncSession,
    user_id: str,
    service: str,
    filters: Mapping[str, Any] | None = None,
) -> int:
    """How many rows the metadata prefilter left. Debug endpoint only."""
    key = service_key(service)
    spec = mirror.spec_for(key)
    work = dict(normalise_filters(key, filters))
    work.pop("text", None)
    params: dict[str, Any] = {}
    where = mirror._build_where(spec, spec.table, work, params)
    from sqlalchemy import text as sql_text

    statement = sql_text(
        f"SELECT count(*) FROM {spec.table} WHERE {spec.table}.user_id = :user_id{where}"
    )
    result = await session.execute(statement, {"user_id": user_id, **params})
    return int(result.scalar_one())


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


def _ref_of(service: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if service == "gmail":
        return {
            "message_id": row.get("message_id"),
            "thread_id": row.get("thread_id"),
            "chunk_index": row.get("chunk_index"),
        }
    if service == "gcal":
        return {
            "event_id": row.get("event_id"),
            "calendar_id": row.get("calendar_id"),
            "recurring_event_id": row.get("recurring_event_id"),
        }
    return {"file_id": row.get("file_id"), "chunk_index": row.get("chunk_index")}


def document_key(service: str, row: Mapping[str, Any]) -> str:
    """One key per message / event / file, so chunks collapse to one candidate."""
    if service == "gmail":
        return f"gmail:{row.get('message_id')}"
    if service == "gcal":
        return f"gcal:{row.get('calendar_id')}:{row.get('event_id')}"
    return f"gdrive:{row.get('file_id')}"


def _when_of(service: str, row: Mapping[str, Any]) -> dt.datetime | None:
    field_name = {"gmail": "received_at", "gcal": "starts_at", "gdrive": "modified_at"}[service]
    value = row.get(field_name)
    return value if isinstance(value, dt.datetime) else None


def _snippet_of(service: str, row: Mapping[str, Any], width: int = 240) -> str:
    if service == "gmail":
        body = row.get("body_clean") or ""
    elif service == "gcal":
        body = " · ".join(
            str(part) for part in (row.get("location"), row.get("description")) if part
        )
    else:
        body = row.get("content_excerpt") or ""
    text = " ".join(str(body).split())
    return text[: width - 1] + "…" if len(text) > width else text


def fuse_arms(
    service: str,
    vector_rows: Sequence[Mapping[str, Any]],
    lexical_rows: Sequence[Mapping[str, Any]],
    *,
    terms: Terms | None = None,
    query: str | None = None,
    limit: int = 10,
    now: dt.datetime | None = None,
) -> list[Hit]:
    """Merge the two arms into ranked hits. Pure Python, no I/O.

    Documents, not chunks: the two arms can pick different chunks of the same
    message, and the planner must see one candidate with one path, not two.
    """
    key = service_key(service)
    anchor = now or dt.datetime.now(dt.UTC)

    rows: dict[str, dict[str, Any]] = {}
    vec_rank: dict[str, int] = {}
    fts_rank: dict[str, int] = {}
    cos: dict[str, float] = {}
    lex: dict[str, float] = {}

    for position, row in enumerate(vector_rows or (), start=1):
        doc = document_key(key, row)
        if doc not in vec_rank:
            vec_rank[doc] = position
            rows[doc] = dict(row)
            cos[doc] = float(row.get("cos") or 0.0)

    for position, row in enumerate(lexical_rows or (), start=1):
        doc = document_key(key, row)
        if doc not in fts_rank:
            fts_rank[doc] = position
            lex[doc] = float(row.get("lex") or 0.0)
        if doc not in rows:
            rows[doc] = dict(row)

    if not rows:
        return []

    # cn is normalised against the vector arm's distribution for this corpus.
    pool = [cos[doc] for doc in vec_rank]
    transform = cn_transform(pool)
    # A document the vector arm never returned has no measured cosine. It is
    # not zero — it is "below the cutoff" — so it inherits the weakest cosine
    # the pool did produce, and only its evidence can lift it from there.
    floor_cos = min(pool) if pool else 0.0
    # The text arm gets its own normalisation ALWAYS, not only when the vector
    # arm came back empty. Computing it conditionally made `cn` — and with it
    # the presented order, which is `cn * decay` — a pure function of cosine
    # whenever the vector arm returned anything, which is nearly always. The
    # fusion was computed and then discarded at the sort, so "hybrid" scored
    # identically to "vector" and a search for "invoice" put three proposals
    # above "Vendor invoice 4471.pdf" — the one document whose text rank was
    # an exact hit.
    lex_pool = [lex[doc] for doc in fts_rank]
    lex_transform = cn_transform(lex_pool) if lex_pool else None

    fused = dict(
        rrf_fuse(
            [
                [doc for doc, _ in sorted(vec_rank.items(), key=lambda kv: kv[1])],
                [doc for doc, _ in sorted(fts_rank.items(), key=lambda kv: kv[1])],
            ]
        )
    )

    hits: list[Hit] = []
    for doc, row in rows.items():
        raw_cos = cos.get(doc, floor_cos)
        # Each arm states its own confidence and the stronger one carries the
        # document: a row the text arm ranked first is a strong candidate even
        # when its cosine is unremarkable, and vice versa. Taking the max is
        # what makes the two arms actually fuse in the order the reader sees.
        vector_cn = transform(raw_cos) if pool else 0.0
        lexical_cn = (
            lex_transform(lex[doc])
            if lex_transform is not None and doc in fts_rank
            else 0.0
        )
        cn_raw = max(vector_cn, lexical_cn)

        flags = evidence_for(key, row, terms, query)
        cn_value = 1.0 if flags else cn_raw
        when = _when_of(key, row)

        if key == "gcal":
            decay_value = 1.0
            boost_value = (
                forward_boost(1.0, when, now=anchor) if when is not None else 1.0
            )
        else:
            half_life = MAIL_DECAY_DAYS if key == "gmail" else FILE_DECAY_DAYS
            decay_value = (
                temporal_decay(1.0, when, now=anchor, half_life_days=half_life)
                if when is not None
                else 1.0
            )
            boost_value = 1.0

        base = fused.get(doc, 0.0)
        # The presented order. `final` is the documented fused score and is
        # reported unchanged, but ordering on it alone puts a marketing email
        # from the right airline above the booking it is advertising against:
        # consecutive RRF ranks differ by a ten-thousandth, and decay swings by
        # a factor of three over a fortnight, so date would decide everything.
        # So the same two things that decide — evidence, then the un-flattened
        # cn — also shape the order, with time on top of them.
        shaped = cn_raw * decay_value * boost_value
        hits.append(
            Hit(
                id=str(row.get("id")),
                service=key,
                ref=_ref_of(key, row),
                title=_row_title(key, row),
                snippet=_snippet_of(key, row),
                when=when,
                from_email=_row_sender(key, row),
                cos=raw_cos if doc in cos else 0.0,
                lex=lex.get(doc, 0.0),
                cn=cn_value,
                cn_raw=cn_raw,
                vec_rank=vec_rank.get(doc),
                fts_rank=fts_rank.get(doc),
                rrf=base,
                decay=decay_value,
                boost=boost_value,
                final=base * decay_value * boost_value,
                sort=shaped,
                evidence=flags,
                row=dict(row),
            )
        )

    hits.sort(key=lambda hit: (0 if hit.is_exact else 1, -hit.sort, -hit.final, hit.id))
    return hits[:limit] if limit and limit > 0 else hits


# --------------------------------------------------------------------------- #
# The public search
# --------------------------------------------------------------------------- #


async def enumerate_arm(
    session: AsyncSession,
    user_id: str,
    service: str,
    filters: Mapping[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Every row matching the filters, in the corpus's own time order.

    The unscored read. Used when a request has filters but nothing to rank by,
    which is what every "list my X for Y" question actually is.
    """
    key = service_key(service)
    work = dict(normalise_filters(key, filters))
    work.pop("text", None)
    return await mirror.list_filtered(
        session, user_id, key, work, limit=max(limit, 1)
    )


async def search(
    session: AsyncSession,
    user_id: str,
    service: str,
    query: str | None = None,
    filters: Mapping[str, Any] | None = None,
    limit: int = 10,
    embedding: Sequence[float] | None = None,
    *,
    terms: Terms | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    now: dt.datetime | None = None,
    arm: str = "hybrid",
) -> list[Hit]:
    """One corpus, both arms, fused and scored. The everyday entry point.

    ``arm`` turns one side off — ``"vector"`` or ``"lexical"`` — which is how
    the offline evaluation measures what the fusion is actually buying.
    """
    result = await search_detailed(
        session, user_id, service, query, filters, limit, embedding,
        terms=terms, candidates=candidates, now=now, arm=arm,
    )
    return result.hits


async def search_detailed(
    session: AsyncSession,
    user_id: str,
    service: str,
    query: str | None = None,
    filters: Mapping[str, Any] | None = None,
    limit: int = 10,
    embedding: Sequence[float] | None = None,
    *,
    terms: Terms | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    now: dt.datetime | None = None,
    explain: bool = False,
    arm: str = "hybrid",
) -> ServiceResult:
    """:func:`search`, plus the counters ``/api/v1/search`` prints.

    ``explain`` adds the prefiltered row count, which costs one extra query and
    is why the chat path never asks for it.
    """
    started = dt.datetime.now(dt.UTC)
    key = service_key(service)
    prepared = normalise_filters(key, filters)
    text = (query or prepared.get("text") or "").strip()
    prepared.pop("text", None)

    wanted = str(arm or "hybrid").lower()
    if wanted not in {"hybrid", "both", "vector", "lexical", "fts", "text"}:
        raise AppError(
            "VALIDATION_ERROR",
            f"Unknown arm {arm!r}.",
            http=422,
            details={"arm": arm, "known": ["hybrid", "vector", "lexical"]},
        )
    # Evidence is read off the query, not off whichever arm is switched on, so
    # an ablation run still scores the same flags as the full search.
    evidence_text = text
    if wanted in {"lexical", "fts", "text"}:
        embedding = None
    if wanted == "vector":
        text = ""

    has_vector = embedding is not None and len(embedding) > 0
    jobs: list[Callable[[AsyncSession], Awaitable[list[dict[str, Any]]]]] = []
    if has_vector:
        jobs.append(
            lambda s: vector_arm(s, user_id, key, embedding or [], prepared, candidates)
        )
    if text:
        jobs.append(lambda s: lexical_arm(s, user_id, key, text, prepared, candidates))
    if not jobs:
        # No query text and no vector — this is an ENUMERATION, not a search.
        # "What is on my calendar next week" carries a window and nothing else,
        # and both scored arms need something to score against. Returning empty
        # answers a perfectly answerable question with "nothing found", which is
        # the worst failure we have: confidently wrong, and silent.
        # So read the window plainly, in the corpus's own time order. There is
        # no ranking question when the filter IS the query.
        rows = await enumerate_arm(session, user_id, key, prepared, limit)
        now_at = now or dt.datetime.now(dt.UTC)
        hits = [
            Hit(
                id=str(row.get("id")),
                service=key,
                ref=_ref_of(key, row),
                title=_row_title(key, row),
                snippet=_snippet_of(key, row),
                when=row.get(mirror.spec_for(key).time_col),
                from_email=_row_sender(key, row),
                cn=1.0,
                cn_raw=1.0,
                final=1.0,
                sort=1.0,
                evidence=["FILTER_MATCH"],
                row=dict(row),
            )
            for row in rows
        ]
        return ServiceResult(
            service=key,
            hits=hits[:limit],
            took_ms=int((dt.datetime.now(dt.UTC) - started).total_seconds() * 1000)
            if "started" in dir()
            else 0,
        )

    outcomes = await gather_reads(session, jobs)
    vector_rows: list[dict[str, Any]] = []
    lexical_rows: list[dict[str, Any]] = []
    index = 0
    if has_vector:
        vector_rows = outcomes[index] or []  # type: ignore[assignment]
        index += 1
    if text:
        lexical_rows = outcomes[index] or []  # type: ignore[assignment]

    resolved_terms = terms if terms is not None else Terms.from_query(evidence_text)
    hits = fuse_arms(
        key, vector_rows, lexical_rows,
        terms=resolved_terms, query=evidence_text, limit=limit, now=now,
    )

    result = ServiceResult(
        service=key,
        hits=hits,
        vector_candidates=len(vector_rows),
        fts_candidates=len(lexical_rows),
        took_ms=int((dt.datetime.now(dt.UTC) - started).total_seconds() * 1000),
    )
    if explain:
        result.prefiltered = await count_prefiltered(session, user_id, key, prepared)
    return result


async def search_many(
    session: AsyncSession,
    user_id: str,
    services: Sequence[str],
    query: str | None = None,
    filters: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int = 10,
    embedding: Sequence[float] | None = None,
    *,
    terms: Terms | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    now: dt.datetime | None = None,
    explain: bool = False,
) -> dict[str, ServiceResult]:
    """The same search over several corpora at once, one connection each.

    A corpus that fails comes back as a ``ServiceResult`` with an ``error`` and
    no hits — one dead service degrades the answer, it does not lose it.
    """
    keys = [service_key(s) for s in services]
    per_service = filters or {}

    jobs = [
        (
            lambda s, key=key: search_detailed(
                s, user_id, key, query, per_service.get(key), limit, embedding,
                terms=terms, candidates=candidates, now=now, explain=explain,
            )
        )
        for key in keys
    ]
    outcomes = await gather_reads(session, jobs, return_exceptions=True)

    out: dict[str, ServiceResult] = {}
    for key, outcome in zip(keys, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.warning("search.arm_failed", service=key, error=str(outcome))
            out[key] = ServiceResult(service=key, error=str(outcome))
        else:
            out[key] = outcome  # type: ignore[assignment]
    return out


async def get_thread(
    session: AsyncSession,
    user_id: str,
    thread_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Every message in one thread, oldest first, one row per message.

    Threads are not a Gmail idea — a Slack thread and the comments under a Jira
    issue are the same container — so this reads the whole `message` shape and
    lets the id decide which connector answers.
    """
    return await mirror.list_filtered(
        session,
        user_id,
        "message",
        {"thread_id": thread_id, "chunk_index": 0},
        limit=limit,
        newest_first=False,
    )


async def count_rows(session: AsyncSession, user_id: str, service: str) -> int:
    """Rows mirrored for one user in one corpus."""
    spec = mirror.spec_for(service_key(service))
    # The table is the shape now, so there is no `kind` to narrow by — only
    # the connector, when this spec names one.
    clauses = [spec.model.user_id == user_id, *mirror._scope(spec)]
    if spec.connector is not None:
        clauses.append(spec.model.connector == spec.connector)
    result = await session.execute(
        select(func.count()).select_from(spec.model).where(*clauses)
    )
    return int(result.scalar_one())


__all__ = [
    "CN_SPREAD",
    "DEFAULT_CANDIDATES",
    "EVENT_HORIZON_DAYS",
    "EVIDENCE_ALIAS_TOKEN",
    "EVIDENCE_EXACT_FILENAME",
    "EVIDENCE_EXACT_ID",
    "EVIDENCE_EXACT_SENDER",
    "EXACT_EVIDENCE",
    "FILE_DECAY_DAYS",
    "MAIL_DECAY_DAYS",
    "RRF_K",
    "SERVICES",
    "Hit",
    "ServiceResult",
    "Terms",
    "ambiguous",
    "cn_scores",
    "cn_transform",
    "count_prefiltered",
    "count_rows",
    "decay",
    "decide",
    "document_key",
    "effective_cn",
    "evidence_for",
    "forward_boost",
    "fuse_arms",
    "gather_reads",
    "get_thread",
    "has_exact_evidence",
    "is_ambiguous",
    "lexical_arm",
    "meets_floor",
    "normalise_filters",
    "qualifies",
    "rrf_fuse",
    "search",
    "search_detailed",
    "search_many",
    "service_key",
    "temporal_decay",
    "vector_arm",
]
