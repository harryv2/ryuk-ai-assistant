"""What an op is, and the machinery every op shares.

An op is one step of a plan. The planner names it, `validate.py` checks the
arguments against `args_model` before anything runs, `dispatch.py` gives it an
:class:`OpContext` and awaits `run`, and the step trace renders it through
`progress_label` and `to_llm`.

The five interfaces at the top — :class:`InputRequest`, :class:`OpResult`,
:class:`Op`, :class:`ConfirmableOp`, :class:`OpContext` — are the contract in
`docs/contracts.md`, spelled exactly. Everything below the fold is shared
plumbing so that twenty-odd ops do not each grow their own copy of the filter
allowlist, the ambiguity test, or the reference-time coercions.

Three rules that decide most of the code here:

* **A search op reads our mirror, never Google.** `freshness: "live"` buys one
  targeted Google list first, which lands in the mirror, and then the search
  runs against the mirror like any other.
* **An unknown filter key is an error.** Silently dropping it would widen a
  search, and a widened search on a write path is how the wrong email gets
  cancelled.
* **A `needs_confirm` op never performs its side effect in `run`.** It prepares
  — creates the draft, pins the etag, resolves the target — and `execute` is
  called later, after a person has said yes.
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.errors import AppError

from app.core.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


@dataclass
class InputRequest:
    """Something the system needs from the person before it can go on."""

    kind: str  # confirm | choice | multi_choice | text | form | date_range
    question: str
    value_schema: dict
    options: list[dict] | None = None
    #: A form's field specs, in draw order — labels and per-field options
    #: included. The schema is the validation authority; this is what the
    #: client renders. Without it a choice field has nowhere to keep its
    #: labels and degrades to a free-text box asking for an id.
    fields: list[dict] | None = None
    blocking: bool = True


@dataclass
class OpResult:
    """What one step produced.

    ``data`` is the node's result: the thing `{{step.x}}` references resolve
    against, and what `to_llm` trims for the next model call.
    """

    data: dict
    needs_replan: bool = False
    replan_reason: str | None = None
    needs_input: InputRequest | None = None


@runtime_checkable
class GoogleClients(Protocol):
    """The per-user Google handles hanging off :class:`OpContext`.

    Built once per run by `app.google.client` with that user's token already
    attached, which is why the methods below take no ``user_id``. Ops reach
    them through :func:`google_call`, which tolerates either convention.
    """

    gmail: Any
    gcal: Any
    drive: Any


class Op:
    """One executable step.

    Subclasses set the class attributes and implement `run`. The three flags
    are read by `validate.py` before execution: `is_local` means no network,
    `is_write` means a side effect, `needs_confirm` means a person has to say
    yes and therefore `run` must stop at prepare.
    """

    name: str = ""
    args_model: type[BaseModel] = BaseModel
    output_fields: list[str] = []
    is_local: bool = False
    is_write: bool = False
    needs_confirm: bool = False
    timeout_s: float = 6.0
    max_attempts: int = 2

    #: One line of help for the planner catalogue.
    summary: str = ""

    # -- identity ---------------------------------------------------------- #

    @property
    def service(self) -> str:
        """``"gmail"`` out of ``"gmail.search_emails"``.

        The same value `node_executions` indexes on `split_part(op,'.',1)`, and
        the key the dispatcher's per-service semaphore is taken from.
        """
        return self.name.split(".", 1)[0]

    # -- execution --------------------------------------------------------- #

    async def run(self, ctx: "OpContext", args: dict) -> OpResult:
        raise NotImplementedError(f"{self.name} has no run()")

    # -- presentation ------------------------------------------------------ #

    def progress_label(self, args: dict) -> str:
        """The line the step trace shows while this step is running."""
        return self.summary or self.name

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        """The node's result, trimmed to something worth spending tokens on."""
        return trim_for_llm(data, budget)

    def catalogue_line(self) -> str:
        """One line for the planner prompt."""
        return catalogue_line_for(self)

    # -- argument parsing -------------------------------------------------- #

    def parse(self, args: dict) -> Any:
        """`args_model` applied, with a Pydantic failure turned into an
        `AppError` the dispatcher can record on the node."""
        try:
            return self.args_model(**(args or {}))
        except AppError:
            raise
        except Exception as exc:  # pydantic.ValidationError and friends
            raise AppError(
                "VALIDATION_ERROR",
                f"{self.name} was given arguments it cannot use.",
                http=422,
                details={"op": self.name, "problem": _short(str(exc), 600)},
            ) from exc


class ConfirmableOp(Op):
    """A write a person has to approve.

    `run` prepares and returns the payload; the dispatcher writes the
    `pending_inputs` row and the `actions` row from `confirm_question` and
    `preview`, and stops. `execute` runs later, from the actions worker, once
    the prompt has been answered.
    """

    needs_confirm = True
    is_write = True

    def preview(self, payload: dict) -> dict:
        """What the confirm card shows. Never the whole payload — the fields a
        person needs to decide with."""
        return dict(payload)

    def confirm_question(self, payload: dict) -> str:
        return f"Go ahead with {self.name}?"

    async def execute(self, ctx: "OpContext", payload: dict) -> dict:
        raise NotImplementedError(f"{self.name} has no execute()")


@dataclass
class OpContext:
    """Everything an op is allowed to reach."""

    user_id: str
    conversation_id: str
    run_id: str
    session: AsyncSession
    google: GoogleClients
    now: dt.datetime
    tz: str
    #: The vector the probe already made for this turn's query, when there is
    #: one. A turn buys exactly one embedding, and a step that asks in the
    #: turn's own words reuses this rather than buying a near-duplicate. Left
    #: None outside a run, where there is no probe and the op embeds for itself.
    probe_embedding: list[float] | None = None
    #: The turn's own query text. This is what makes the vector above safe to
    #: reuse: a step asking in these words means the same thing, a step asking
    #: with an identifier the turn never said does not. See `turn_embedding`.
    probe_query: str = ""


# --------------------------------------------------------------------------- #
# Retrieval thresholds
# --------------------------------------------------------------------------- #

#: How often one corpus may be refreshed live for one user. Long enough that a
#: multi-step plan buys a single Google call, short enough that a person who
#: just booked something and asks about it gets the truth.
LIVE_REFRESH_EVERY_S: Final[int] = 60

#: How many freshly-fetched rows to vectorise inline. Enough for a page of
#: results; beyond that the background queue can catch up.
EMBED_ON_BACKFILL: Final[int] = 100

FLOOR_READ: float = settings.FLOOR_READ
MARGIN: float = settings.MARGIN
FLOOR_WRITE: float = settings.FLOOR_WRITE

#: Evidence flags that mean the match was on something exact — an id, a sender
#: address, a filename, or a vendor alias token in the subject. A candidate
#: carrying one of these is admitted whatever its `cn` is: the floor is a
#: disjunction, not a gate.
EXACT_FLAGS: frozenset[str] = frozenset(
    {
        "EXACT",
        "EXACT_ID",
        "EXACT_SENDER",
        "EXACT_SENDER_DOMAIN",
        "EXACT_FILENAME",
        "EXACT_FILENAME_TOKEN",
        "ALIAS_TOKEN_IN_SUBJECT",
        "ALIAS_TOKEN_IN_TITLE",
        "FILTER_MATCH",
    }
)

# Fallback calibration, used only when the retrieval layer hands back a raw
# cosine and no `cn`. Below LOW nothing is worth showing, above HIGH nothing
# gets better. Hand-set, like the floors themselves.
_CN_LOW: float = 0.20
_CN_HIGH: float = 0.85


def is_exact(evidence: Any) -> bool:
    """True when any evidence flag on a candidate is an exact match."""
    if not evidence:
        return False
    if isinstance(evidence, str):
        evidence = [evidence]
    for flag in evidence:
        token = str(flag).upper()
        if token in EXACT_FLAGS or token.startswith("EXACT"):
            return True
    return False


def admits(hit: dict) -> bool:
    """Is this candidate worth showing at all?

    `cn >= FLOOR_READ` **or** an exact evidence flag. A Turkish-language
    booking confirmation scores 0.41 against an English query and is still the
    right email when the PNR matches.
    """
    return float(hit.get("cn") or 0.0) >= FLOOR_READ or is_exact(hit.get("evidence"))


def ambiguity_test(hits: Sequence[dict]) -> str | None:
    """The reason this set is too unclear to act on, or ``None``.

    ``"absent"``   nothing cleared the floor.
    ``"margin"``   the top two are within `MARGIN` and nothing separates them.

    An exact flag on the leader settles a near-tie — but only when the
    runner-up does not carry the same one. Two Turkish Airlines confirmations
    both matching on the alias token, 0.84 against 0.79, are still two
    bookings: the evidence that would break the tie is evidence they share.
    """
    if not hits:
        return "absent"
    top = hits[0]
    if not admits(top):
        return "absent"
    if len(hits) > 1:
        runner_up = hits[1]
        gap = float(top.get("cn") or 0.0) - float(runner_up.get("cn") or 0.0)

        # The evidence that would break the tie is evidence they share.
        #
        # Two events both literally containing "John" are two candidates, and
        # the cosine distance between them is noise: these are three-word
        # titles, and a vector arm will happily put 0.78 on one and 0.55 on the
        # other for no reason a person would accept. Where the leader and the
        # runner-up match on the same exact signal, that signal cannot also be
        # the thing that separates them — whatever the gap says.
        shared = _shared_exact(top.get("evidence"), runner_up.get("evidence"))
        if shared:
            return "margin"

        decisive = is_exact(top.get("evidence")) and not is_exact(runner_up.get("evidence"))
        if gap < MARGIN and not decisive:
            return "margin"
    return None


def _exact_flags(evidence: Any) -> set[str]:
    """The exact-match flags on a candidate, as a set."""
    if not evidence:
        return set()
    flags = [evidence] if isinstance(evidence, str) else list(evidence)
    return {str(f) for f in flags if is_exact(f)}


def _shared_exact(left: Any, right: Any) -> set[str]:
    """Exact flags both candidates carry — the ones that cannot separate them."""
    return _exact_flags(left) & _exact_flags(right)


def choice_input(
    question: str,
    hits: Sequence[dict],
    *,
    id_field: str,
    help_text: str | None = None,
    kind: str = "choice",
    max_options: int = 6,
) -> InputRequest:
    """A choice card carrying the real candidates, not a paraphrase of them."""
    options: list[dict] = []
    for hit in list(hits)[:max_options]:
        ident = str(hit.get(id_field) or hit.get("ref") or "")
        if not ident:
            continue
        options.append(
            {
                "id": ident,
                "label": hit.get("label") or ident,
                "meta": {
                    k: v
                    for k, v in (hit.get("meta") or {}).items()
                    if v not in (None, "", [], {})
                },
            }
        )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {id_field: {"type": "string", "enum": [o["id"] for o in options]}},
        "required": [id_field],
    }
    request = InputRequest(kind=kind, question=question, value_schema=schema, options=options)
    if help_text:
        request.value_schema = {**schema, "description": help_text}
    return request


# --------------------------------------------------------------------------- #
# Small conversions the whole registry needs
# --------------------------------------------------------------------------- #

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ensure_utc(value: dt.datetime | None) -> dt.datetime | None:
    """Tz-aware UTC, always. A naive datetime is read as UTC rather than
    guessed at — guessing is how a meeting moves by five hours."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def parse_dt(value: Any) -> dt.datetime | None:
    """An ISO string, an epoch, or a datetime — as tz-aware UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return ensure_utc(value)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return ensure_utc(dt.datetime.fromisoformat(text))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return ensure_utc(dt.datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise AppError(
        "VALIDATION_ERROR",
        f"{value!r} is not a date or a time.",
        http=422,
        details={"value": str(value)},
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise AppError("VALIDATION_ERROR", f"{value!r} is not a yes or a no.", http=422)


def as_list(value: Any) -> list[Any]:
    """One value or many, always as a list. Comma-separated strings included,
    because a model that has been asked for a list will sometimes send one."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, str) and "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def iso(value: Any) -> str | None:
    parsed = value if isinstance(value, dt.datetime) else parse_dt(value)
    return ensure_utc(parsed).isoformat() if parsed else None


def window_bounds(value: Any) -> tuple[dt.datetime, dt.datetime]:
    """``{"start": ..., "end": ...}`` as a half-open UTC pair.

    A list of two is accepted as well, since that is a shape models produce.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    elif isinstance(value, dict):
        start = value.get("start") or value.get("from") or value.get("min")
        end = value.get("end") or value.get("to") or value.get("max")
    else:
        raise AppError(
            "VALIDATION_ERROR",
            "A window needs a start and an end.",
            http=422,
            details={"got": _short(str(value), 120)},
        )
    lo, hi = parse_dt(start), parse_dt(end)
    if lo is None or hi is None:
        raise AppError(
            "VALIDATION_ERROR",
            "A window needs a start and an end.",
            http=422,
            details={"start": str(start), "end": str(end)},
        )
    if hi <= lo:
        raise AppError(
            "VALIDATION_ERROR",
            "A window ends before it starts.",
            http=422,
            details={"start": iso(lo), "end": iso(hi)},
        )
    return lo, hi


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def excerpt_of(text: str | None, limit: int = 240) -> str:
    return _short(text or "", limit)


def duration_minutes(start: Any, end: Any) -> int | None:
    lo, hi = parse_dt(start), parse_dt(end)
    if lo is None or hi is None:
        return None
    return max(0, int(round((hi - lo).total_seconds() / 60.0)))


# --------------------------------------------------------------------------- #
# Result shaping
# --------------------------------------------------------------------------- #


def jsonable(value: Any) -> Any:
    """Anything a row can hold, as something `json.dumps` accepts.

    `node_executions.result` is JSONB, so datetimes and UUIDs have to be text
    by the time a result leaves an op.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dt.datetime):
        return ensure_utc(value).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def project_fields(row: dict, fields: Sequence[str] | None) -> dict:
    """Keep only the named fields, plus the ones every consumer needs."""
    if not fields:
        return row
    keep = set(fields) | {"ref", "label", "cn", "evidence"}
    return {k: v for k, v in row.items() if k in keep}


def trim_for_llm(data: dict, budget: int = 900) -> dict:
    """`data` shrunk until it serialises inside ``budget`` characters.

    Lists lose their tail, long strings lose their middle. Scalars are never
    dropped: a count of 41 is one of the cheapest and most useful things the
    model can be told.
    """
    if budget <= 0:
        return {}
    payload = jsonable(data)
    if not isinstance(payload, dict):
        return {"value": payload}
    if len(json.dumps(payload, ensure_ascii=False)) <= budget:
        return payload

    scalars = {k: v for k, v in payload.items() if not isinstance(v, (list, dict))}
    lists = {k: v for k, v in payload.items() if isinstance(v, list)}
    dicts = {k: v for k, v in payload.items() if isinstance(v, dict)}

    out: dict[str, Any] = dict(scalars)
    for key, value in dicts.items():
        out[key] = trim_for_llm(value, max(120, budget // 4))

    room = budget - len(json.dumps(out, ensure_ascii=False))
    for key, items in lists.items():
        kept: list[Any] = []
        for item in items:
            piece = _trim_item(item, 220)
            cost = len(json.dumps(piece, ensure_ascii=False)) + 2
            if cost > room and kept:
                break
            kept.append(piece)
            room -= cost
        out[key] = kept
        if len(kept) < len(items):
            out[f"{key}_shown"] = len(kept)

    # The pass above is greedy and can still overshoot on a single fat row, so
    # squeeze until it actually fits: shorter strings first, then fewer rows.
    for width in (160, 90, 50, 24):
        if len(json.dumps(out, ensure_ascii=False)) <= budget:
            break
        for key in lists:
            out[key] = [_trim_item(item, width) for item in out.get(key, [])]
    while len(json.dumps(out, ensure_ascii=False)) > budget:
        longest = max(lists, key=lambda k: len(out.get(k, [])), default=None)
        if longest is None or not out.get(longest):
            break
        out[longest] = out[longest][:-1]
        out[f"{longest}_shown"] = len(out[longest])
    return out


def _trim_item(item: Any, budget: int) -> Any:
    if isinstance(item, str):
        return _short(item, budget)
    if isinstance(item, list):
        return [_trim_item(v, budget // 2) for v in item[:5]]
    if isinstance(item, dict):
        out: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, str):
                out[key] = _short(value, budget)
            elif isinstance(value, (list, dict)):
                out[key] = _trim_item(value, max(60, budget // 3))
            else:
                out[key] = value
        return out
    return item


def catalogue_line_for(op: Op) -> str:
    """``name(arg, arg) -> field, field  · note``, one line, for the prompt."""
    args = ", ".join(op.args_model.model_fields.keys())
    out = ", ".join(op.output_fields)
    notes: list[str] = []
    if op.is_local:
        notes.append("local")
    if op.needs_confirm:
        notes.append("CONFIRM")
    elif op.is_write:
        notes.append("write")
    note = f" [{' '.join(notes)}]" if notes else ""
    tail = f" — {op.summary}" if op.summary else ""
    return f"{op.name}({args}) -> {out}{note}{tail}"


# --------------------------------------------------------------------------- #
# Google, through whatever shape the service wrappers ended up with
# --------------------------------------------------------------------------- #


async def google_call(
    ctx: OpContext,
    service: str,
    names: Sequence[str],
    /,
    **kwargs: Any,
) -> Any:
    """Call the first method in ``names`` that the wrapper actually has.

    ``ctx.google`` is built per user, so the wrappers do not need a ``user_id``
    — but a wrapper written to the repository convention takes one first, and
    this passes it when the signature asks for it. Keyword arguments the
    wrapper does not declare are dropped rather than raising, so an optional
    extra like ``send_updates`` never breaks a call.
    """
    client = getattr(ctx.google, service, None)
    if client is None:
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "That Google service is not connected.",
            details={"service": service},
        )
    fn = None
    for candidate in names:
        found = getattr(client, candidate, None)
        if callable(found):
            fn = found
            break
    if fn is None:
        raise AppError(
            "INTERNAL",
            f"The {service} wrapper cannot {names[0]}.",
            details={"service": service, "tried": list(names)},
        )

    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_all = any(p.kind is p.VAR_KEYWORD for p in params.values())
        wants_user = "user_id" in params
        call_kwargs = kwargs if accepts_all else {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):  # a builtin or a mock without a signature
        wants_user, call_kwargs = False, dict(kwargs)

    result = fn(ctx.user_id, **call_kwargs) if wants_user else fn(**call_kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _reason_of(exc: BaseException) -> str:
    """A short, loggable label for why a live refresh gave up."""
    code = getattr(exc, "code", None) or getattr(exc, "error_class", None)
    return str(code or type(exc).__name__)[:40]


def has_google(ctx: OpContext, service: str, names: Sequence[str]) -> bool:
    """Whether a live call is even possible, without making one."""
    client = getattr(ctx.google, service, None)
    return client is not None and any(callable(getattr(client, n, None)) for n in names)


# --------------------------------------------------------------------------- #
# Retrieval — one adapter over app.search.hybrid, one over the extractors
# --------------------------------------------------------------------------- #

_HYBRID: tuple[str, Callable[..., Any]] | None = None
_EMBED: Callable[..., Any] | None = None
_EXTRACTORS: Any = None
_EXTRACTORS_LOADED = False

CORPUS_OF: dict[str, str] = {"gmail": "gmail", "gcal": "gcal", "drive": "gdrive", "gdrive": "gdrive"}

#: What one row of each corpus is called, for a sentence aimed at a person.
_NOUN_OF: dict[str, str] = {"gmail": "an email", "gcal": "a meeting", "gdrive": "a file"}


def service_of(op_name: str) -> str:
    """The service a step reaches, under the one name the rest of the app uses.

    Some ops are registered under two namespaces — Drive answers to both
    ``drive.*`` and ``gdrive.*``, the calendar to ``calendar.*`` and ``gcal.*``
    — and the planner may pick either. Everything downstream (the intent, the
    service labels, the entity types) knows only the canonical name, so any
    code that turns an op into a service has to come through here. Reading the
    namespace raw is how a valid plan gets rejected and how a Drive file quietly
    fails to be remembered.

    A namespace that names no service — ``ask``, ``meta``, ``llm`` — is returned
    as it is, which is what the callers that filter on those names expect.
    """
    namespace = str(op_name).split(".", 1)[0]
    return CORPUS_OF.get(namespace, ALIASES.get(namespace, namespace))


# Namespaces that reach a service but hold no corpus of their own.
ALIASES: dict[str, str] = {"calendar": "gcal", "mail": "gmail"}

#: The identifying column of each mirror corpus.
REF_FIELD: dict[str, str] = {"gmail": "message_id", "gcal": "event_id", "gdrive": "file_id"}

#: The column a time window filters on.
TIME_FIELD: dict[str, str] = {"gmail": "received_at", "gcal": "starts_at", "gdrive": "modified_at"}

#: The long text field, kept out of results unless ``body`` is asked for.
BODY_FIELD: dict[str, str] = {
    "gmail": "body_clean",
    "gcal": "description",
    "gdrive": "content_excerpt",
}


def _resolve_hybrid() -> tuple[str, Callable[..., Any]]:
    """`app.search.hybrid` if it is there, the mirror repository if it is not.

    Both do the same job — a prefiltered vector arm and a lexical arm, fused —
    and the ops do not care which one answered.
    """
    global _HYBRID
    if _HYBRID is not None:
        return _HYBRID
    for module_name, fn_names in (
        ("app.search.hybrid", ("search", "hybrid_search", "run", "search_corpus")),
        ("app.db.repositories.mirror", ("hybrid_search",)),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for fn_name in fn_names:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                _HYBRID = (module_name, fn)
                return _HYBRID
    raise AppError("INTERNAL", "There is no retrieval backend to search with.")


def _is_turn_wording(step_text: str, turn_text: str) -> bool:
    """Is this step asking in words the turn itself used?

    Word-level containment, not substring: "budget" is in "emails from sarah
    about the budget", while "TK1984" is in no part of "cancel my Turkish
    Airlines flight".
    """
    step_words = set(re.findall(r"\w+", step_text.lower()))
    if not step_words:
        return False
    return step_words <= set(re.findall(r"\w+", turn_text.lower()))


async def turn_embedding(ctx: OpContext, text: str | None) -> list[float] | None:
    """The vector this search should use for its semantic arm.

    One turn, one embedding — the budget the whole design is built on. The probe
    has already paid for a vector of the user's question, so a step that asks in
    the turn's own words ("budget", out of "emails from sarah about the budget")
    rides on that vector instead of buying a near-duplicate of it.

    A step asking with a word the turn never said is a different matter. When
    the planner writes `query: "TK1984"` it has read that flight number off a
    booking email; it is an identifier, and an identifier is an exact-match
    signal that belongs to the lexical arm. Handing such a step the turn's
    vector would score every calendar event by how much it looks like "cancel my
    Turkish Airlines flight", which buries the exact hit among near-ties — so
    the semantic arm sits this one out and the lexical arm answers alone.

    Outside a run there is no probe to inherit from, and the op embeds for
    itself.
    """
    inherited = getattr(ctx, "probe_embedding", None)
    if not inherited:
        return await embed_query(text) if text else None
    if not text:
        return None
    turn_text = getattr(ctx, "probe_query", "") or ""
    if not turn_text or _is_turn_wording(text, turn_text):
        return list(inherited)
    return None


async def embed_query(text: str) -> list[float] | None:
    """One vector for the query string, cached by content upstream."""
    global _EMBED
    query = (text or "").strip()
    if not query:
        return None
    if _EMBED is None:
        for module_name, fn_names in (
            ("app.search.embedder", ("embed_query", "embed_one", "embed_text")),
            ("app.core.llm", ("embed_one",)),
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for fn_name in fn_names:
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    _EMBED = fn
                    break
            if _EMBED is not None:
                break
    if _EMBED is None:
        return None
    try:
        vector = _EMBED(query)
        if inspect.isawaitable(vector):
            vector = await vector
    except AppError:
        raise
    except Exception:
        # A retrieval run without the vector arm is a worse search, not a
        # failed step: the lexical arm still answers.
        return None
    if vector and isinstance(vector[0], (list, tuple)):
        vector = vector[0]
    return list(vector) if vector else None


def extractors_module() -> Any:
    global _EXTRACTORS, _EXTRACTORS_LOADED
    if not _EXTRACTORS_LOADED:
        _EXTRACTORS_LOADED = True
        try:
            _EXTRACTORS = importlib.import_module("app.search.extractors")
        except ImportError:
            _EXTRACTORS = None
    return _EXTRACTORS


# The safety net: the same shapes `app.search.extractors` pulls out, used only
# when that module is not importable. Compiled once, at import.
_LINK: re.Pattern[str] = re.compile(r"(https?://[^\s<>\"')]+)")

_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pnr", re.compile(r"\b(?:booking(?: ref(?:erence)?)?|PNR|confirmation(?: code)?)\D{0,12}\b([A-Z0-9]{5,7})\b", re.I)),
    ("flight_no", re.compile(r"\b([A-Z]{2}\d{1,4})\b(?=\s|,|\.|$)")),
    ("ticket_no", re.compile(r"\bticket\D{0,10}(\d{3}[- ]?\d{10})\b", re.I)),
    ("order_id", re.compile(r"\border(?:\s*(?:id|no|number|#))?\D{0,6}\b([A-Z0-9][A-Z0-9-]{5,19})\b", re.I)),
    ("amount", re.compile(r"\b((?:USD|EUR|GBP|INR|\$|€|£)\s?\d[\d,]*(?:\.\d{2})?)\b")),
    ("support_email", re.compile(r"\b((?:cancel|support|help|service|care|reservations?)[\w.\-+]*@[\w.\-]+\.\w{2,})\b", re.I)),
    ("email", re.compile(r"\b([\w.\-+]+@[\w.\-]+\.\w{2,})\b")),
    ("link", _LINK),
    ("phone", re.compile(r"(\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4})")),
    ("date", re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.I)),
    ("route", re.compile(r"\b([A-Z]{3})\s*(?:->|→|-|to)\s*([A-Z]{3})\b")),
]


def run_extractors(
    text: str,
    fields: Sequence[str] | None = None,
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    """Regex first, always. Twenty patterns cost 2 ms; a model call costs 400.

    Returns only what actually matched, so a caller can tell a miss from an
    empty string. ``subject`` is passed through when the extractor takes one —
    a booking reference lives in the subject line at least as often as in the
    body, and scoring it as a subject match is what tells the two apart.
    """
    body = text or ""
    if not body.strip() and not (subject or "").strip():
        return {}

    module = extractors_module()
    if module is not None:
        for fn_name in ("extract", "extract_all", "run", "scan"):
            fn = getattr(module, fn_name, None)
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters
                kwargs: dict[str, Any] = {}
                if subject and "subject" in params:
                    kwargs["subject"] = subject
                if fields:
                    for key in ("fields", "only", "keys", "want"):
                        if key in params:
                            kwargs[key] = list(fields)
                            break
                found = fn(body, **kwargs)
            except AppError:
                raise
            except Exception:
                continue
            if isinstance(found, dict):
                return _clean_extracted(found, fields)
    head = (subject or "").strip()
    return _clean_extracted(_fallback_extract(f"{head}\n{body}" if head else body), fields)


def _fallback_extract(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pattern in _FALLBACK_PATTERNS:
        match = pattern.search(body)
        if match is None:
            continue
        if name == "route":
            out["route"] = f"{match.group(1)} → {match.group(2)}"
            out.setdefault("origin", match.group(1))
            out.setdefault("destination", match.group(2))
        else:
            out[name] = match.group(1)
    links = _LINK.findall(body)
    if links:
        out["links"] = links[:5]
    return out


def _clean_extracted(found: dict[str, Any], fields: Sequence[str] | None) -> dict[str, Any]:
    out = {k: v for k, v in found.items() if v not in (None, "", [], {})}
    if fields:
        wanted = set(fields)
        out = {k: v for k, v in out.items() if k in wanted}
    return jsonable(out)


def hit_to_dict(hit: Any) -> dict[str, Any]:
    """One retrieval hit as the flat dict a plan reference resolves against.

    `app.search.hybrid.Hit` already knows how to flatten itself — `to_binding`
    is that method — and it keeps the mirror row it came from on `.row`. Taking
    both means a hit carries its provider ids *and* the columns a later step
    filters on, with the binding view winning where they disagree.
    """
    binding = getattr(hit, "to_binding", None)
    if callable(binding):
        try:
            flat = dict(binding())
        except TypeError:
            flat = {}
        if flat:
            raw = getattr(hit, "row", None)
            merged = {**(raw if isinstance(raw, dict) else {}), **flat}
            for name in ("cos", "lex", "cn", "final", "score", "service"):
                value = getattr(hit, name, None)
                if value is not None:
                    merged.setdefault(name, value)
            return merged
    return row_to_dict(hit)


def row_to_dict(row: Any) -> dict[str, Any]:
    """A mapping, a dataclass-ish hit, or an ORM row — as a plain dict."""
    if isinstance(row, dict):
        return dict(row)
    for attr in ("model_dump", "to_dict", "_asdict", "dict"):
        fn = getattr(row, attr, None)
        if callable(fn):
            try:
                value = fn()
                if isinstance(value, dict):
                    return dict(value)
            except TypeError:
                pass
    data = getattr(row, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {"value": row}


def derive_cn(row: dict) -> float:
    """`cn` if the retrieval layer supplied one, otherwise from the cosine.

    Ordering and deciding are separate: a fused RRF score says "it came first",
    which is not the same as "it is right", so it is never used here.
    """
    for key in ("cn", "confidence", "norm_score"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    cos = row.get("cos")
    if isinstance(cos, (int, float)):
        span = _CN_HIGH - _CN_LOW
        return max(0.0, min(1.0, (float(cos) - _CN_LOW) / span))
    score = row.get("score")
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))
    return 0.0


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{2,}")


def derive_evidence(corpus: str, row: dict, query: str | None, filters: dict) -> list[str]:
    """Exact-match flags, computed here when the retrieval layer did not.

    Four kinds, the four the design names: the id itself, the sender address, a
    filename, and a vendor alias token sitting in a subject or a title.
    """
    existing = row.get("evidence")
    if existing:
        return [str(e) for e in as_list(existing)]

    flags: list[str] = []
    ref = str(row.get(REF_FIELD.get(corpus, "id")) or "")
    tokens = {t.lower() for t in _TOKEN.findall(query or "")}

    if ref and ref.lower() in tokens:
        flags.append("EXACT_ID")
    if corpus == "gmail":
        sender = str(row.get("from_email") or "").lower()
        pinned = {str(v).lower() for v in as_list(filters.get("from_email")) + as_list(filters.get("from_emails"))}
        if sender and (sender in pinned or sender in tokens):
            flags.append("EXACT_SENDER")
        elif sender and "@" in sender and sender.split("@", 1)[1] in tokens:
            flags.append("EXACT_SENDER_DOMAIN")
        subject = str(row.get("subject") or "").lower()
        if subject and _alias_token_hit(subject, tokens):
            flags.append("ALIAS_TOKEN_IN_SUBJECT")
    elif corpus == "gcal":
        title = str(row.get("title") or "").lower()
        if title and _alias_token_hit(title, tokens):
            flags.append("ALIAS_TOKEN_IN_TITLE")
    elif corpus == "gdrive":
        name = str(row.get("name") or "").lower()
        if name and name in tokens:
            flags.append("EXACT_FILENAME")
        elif name and _alias_token_hit(name, tokens):
            flags.append("EXACT_FILENAME_TOKEN")
    return flags


def _alias_token_hit(haystack: str, tokens: set[str]) -> bool:
    """A distinctive token from the query appearing verbatim in the text.

    Distinctive means at least four characters and carrying a digit or a
    capital in the original — "TK1984" and "Acme" qualify, "flight" does not.
    """
    for token in tokens:
        if len(token) < 4 or token in _STOPWORDS:
            continue
        if token in haystack:
            return True
    return False


_STOPWORDS: frozenset[str] = frozenset(
    """about after again against alle also anything book booking cancel change email emails
    event events file files find flight flights from have here into just last late later
    meeting meetings message messages more most much need needs next note notes only other
    over please recent reschedule same schedule search send sent show some soon than that
    them then there these they this those time today tomorrow week weeks what when where
    which will with within would year yesterday your""".split()
)


def label_for(corpus: str, row: dict) -> str:
    """The one line a person would recognise this row by."""
    if corpus == "gmail":
        return str(row.get("subject") or row.get("from_email") or "(no subject)")
    if corpus == "gcal":
        return str(row.get("title") or "(untitled event)")
    return str(row.get("name") or "(unnamed file)")


def meta_for(corpus: str, row: dict) -> dict[str, Any]:
    """What a choice card shows under the label."""
    if corpus == "gmail":
        return jsonable(
            {
                "from": row.get("from_name") or row.get("from_email"),
                "when": row.get("received_at"),
                "excerpt": excerpt_of(row.get("body_clean"), 120),
            }
        )
    if corpus == "gcal":
        return jsonable(
            {
                "when": row.get("starts_at"),
                "where": row.get("location"),
                "guests": len(as_list(row.get("attendee_emails"))) or None,
            }
        )
    return jsonable(
        {
            "when": row.get("modified_at"),
            "where": row.get("folder_path"),
            "type": row.get("mime_type"),
        }
    )


def shape_hit(
    corpus: str,
    raw: Any,
    *,
    query: str | None,
    filters: dict,
    body: bool,
    extract: bool,
) -> dict[str, Any]:
    """One retrieval row as the dict a plan reference resolves against."""
    row = row_to_dict(raw)
    inner = row.get("row") or row.get("fields") or row.get("source")
    if isinstance(inner, dict):
        row = {**inner, **{k: v for k, v in row.items() if k not in ("row", "fields", "source")}}

    body_field = BODY_FIELD.get(corpus, "body_clean")
    text = str(row.get(body_field) or "")
    hit = {k: v for k, v in row.items() if k not in ("embedding", "tsv", "content_hash")}
    hit.pop("row", None)

    if not body:
        hit.pop(body_field, None)
    hit["excerpt"] = excerpt_of(text or label_for(corpus, row))
    hit["ref"] = row.get(REF_FIELD.get(corpus, "id"))
    hit["label"] = label_for(corpus, row)
    hit["cn"] = round(derive_cn(row), 4)
    hit["evidence"] = derive_evidence(corpus, row, query, filters)
    hit["meta"] = meta_for(corpus, row)
    if corpus == "gcal":
        hit["duration_minutes"] = duration_minutes(row.get("starts_at"), row.get("ends_at"))
    if extract:
        already = row.get("extracted")
        if isinstance(already, dict) and already:
            # The probe already ran the rules over this excerpt. Running them
            # again would cost nothing and change nothing, so keep its answer.
            hit["extracted"] = jsonable(already)
        else:
            found = run_extractors(text, subject=label_for(corpus, row))
            if found:
                hit["extracted"] = found
    return jsonable(hit)


async def hybrid(
    ctx: OpContext,
    corpus: str,
    *,
    query: str | None,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """One hybrid search over one mirror corpus.

    The call is bound by name against whatever signature the retrieval layer
    ended up with, so `app.search.hybrid` owns its own argument order without
    twenty ops needing to agree with it.
    """
    module_name, fn = _resolve_hybrid()
    text = (query or "").strip() or None
    embedding = await turn_embedding(ctx, text)

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}

    pool: dict[str, Any] = {
        "session": ctx.session,
        "db": ctx.session,
        "user_id": ctx.user_id,
        "corpus": corpus,
        "table": corpus,
        "service": corpus,
        "source": corpus,
        "query": text,
        "text": text,
        "q": text,
        "embedding": embedding,
        "vector": embedding,
        "emb": embedding,
        "filters": dict(filters),
        "filter": dict(filters),
        "limit": limit,
        "top_k": limit,
        "k": limit,
        "now": ctx.now,
        "tz": ctx.tz,
    }
    # A backend with no query parameter takes the lexical string inside the
    # filter dict — which is how the mirror repository reads it.
    if not any(name in params for name in ("query", "text", "q")) and text:
        pool["filters"] = {**pool["filters"], "text": text}
        pool["filter"] = pool["filters"]

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in params.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name not in pool:
            if param.default is param.empty:
                raise AppError(
                    "INTERNAL",
                    f"{module_name} wants an argument the ops cannot supply.",
                    details={"argument": name},
                )
            continue
        if param.kind is param.POSITIONAL_ONLY:
            args.append(pool[name])
        else:
            kwargs[name] = pool[name]

    if not params:  # a test double without an inspectable signature
        kwargs = {"session": ctx.session, "user_id": ctx.user_id, "corpus": corpus}

    rows = fn(*args, **kwargs)
    if inspect.isawaitable(rows):
        rows = await rows
    return [hit_to_dict(r) for r in (rows or [])]


async def hybrid_many(
    ctx: OpContext,
    corpora: Sequence[str],
    *,
    query: str | None,
    filters: dict[str, Any] | None = None,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """The same query over several corpora, genuinely in parallel.

    One `AsyncSession` is one connection and cannot serve two queries at once,
    so the fan-out belongs to the retrieval layer, which builds a sibling
    session per corpus. Without that layer the corpora are read in order —
    slower, same answers.
    """
    module_name, _ = _resolve_hybrid()
    text = (query or "").strip() or None
    shared = dict(filters or {})

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = None
    many = getattr(module, "search_many", None) if module is not None else None

    if callable(many):
        embedding = await turn_embedding(ctx, text)
        outcome = many(
            ctx.session,
            ctx.user_id,
            list(corpora),
            text,
            {c: dict(shared) for c in corpora},
            limit,
            embedding,
            now=ctx.now,
        )
        if inspect.isawaitable(outcome):
            outcome = await outcome
        out: dict[str, list[dict[str, Any]]] = {}
        for corpus in corpora:
            result = (outcome or {}).get(corpus)
            hits = getattr(result, "hits", result) or []
            out[corpus] = [hit_to_dict(h) for h in hits]
        return out

    return {
        corpus: await hybrid(ctx, corpus, query=text, filters=dict(shared), limit=limit)
        for corpus in corpora
    }


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


@dataclass
class FilterPlan:
    """Filters split by where they can be enforced."""

    sql: dict[str, Any] = field(default_factory=dict)
    post: dict[str, Any] = field(default_factory=dict)


def build_filters(op_name: str, corpus: str, raw: dict[str, Any], allowed: dict[str, str | None]) -> FilterPlan:
    """Validate every key against the op's allowlist, then split it.

    An unknown key is refused with the allowlist in the error, because a
    silently dropped filter turns a precise search into a broad one and nobody
    finds out until the wrong thing has been drafted.
    """
    plan = FilterPlan()
    for key, value in (raw or {}).items():
        if value is None or value == [] or value == "":
            continue
        if key not in allowed:
            raise AppError(
                "VALIDATION_ERROR",
                f"{key!r} is not something {op_name} can filter on.",
                http=422,
                details={"op": op_name, "filter": key, "allowed": sorted(allowed)},
            )
        target = allowed[key]
        if target is None:
            plan.post[key] = value
        elif target == "window":
            lo, hi = window_bounds(value)
            plan.sql["since"] = lo
            plan.sql["until"] = hi
        elif target in ("since", "until", "ends_after"):
            plan.sql[target] = parse_dt(value)
        elif target.endswith("[]"):
            plan.sql[target[:-2]] = [str(v) for v in as_list(value)]
        elif target.startswith("bool:"):
            plan.sql[target[5:]] = as_bool(value)
        else:
            plan.sql[target] = value
    return plan


def apply_post_filters(corpus: str, hits: list[dict], post: dict[str, Any]) -> list[dict]:
    """The predicates SQL cannot express, applied in Python.

    ``participants`` is the one that matters: "emails with these people" is an
    OR across sender and recipients, and the mirror's whitelist has no OR. One
    query and a pass in Python beats two queries.
    """
    if not post:
        return hits
    out = hits
    for key, value in post.items():
        if key == "participants":
            wanted = {str(v).lower() for v in as_list(value)}
            out = [h for h in out if wanted & _participants_of(corpus, h)]
        elif key == "exclude_refs":
            skip = {str(v) for v in as_list(value)}
            out = [h for h in out if str(h.get("ref")) not in skip]
        elif key == "min_cn":
            floor = float(value)
            out = [h for h in out if float(h.get("cn") or 0.0) >= floor]
        elif key == "is_unread":
            want = as_bool(value)
            out = [h for h in out if ("UNREAD" in {str(l).upper() for l in as_list(h.get("labels"))}) is want]
        elif key == "has_link":
            want = as_bool(value)
            out = [h for h in out if bool((h.get("extracted") or {}).get("links")) is want]
        elif key == "text_contains":
            needle = str(value).lower()
            out = [
                h
                for h in out
                if needle in f"{h.get('label','')} {h.get('excerpt','')}".lower()
            ]
    return out


def _participants_of(corpus: str, hit: dict) -> set[str]:
    if corpus == "gmail":
        people = as_list(hit.get("from_email")) + as_list(hit.get("to_emails"))
    elif corpus == "gcal":
        people = as_list(hit.get("attendee_emails")) + as_list(hit.get("organizer_email"))
    else:
        people = as_list(hit.get("owner_email"))
    return {str(p).lower() for p in people}


def order_hits(hits: list[dict], order_by: str, order_dir: str, allowed: dict[str, str]) -> list[dict]:
    """Sort by one of the op's declared orderings. ``relevance`` is `cn`."""
    if order_by not in allowed:
        raise AppError(
            "VALIDATION_ERROR",
            f"{order_by!r} is not an ordering this op offers.",
            http=422,
            details={"order_by": order_by, "allowed": sorted(allowed)},
        )
    key = allowed[order_by]
    reverse = str(order_dir).lower() != "asc"

    def sort_key(hit: dict) -> tuple[int, Any]:
        value = hit.get(key)
        if value is None:
            return (1, "")
        if isinstance(value, str):
            parsed = None
            if len(value) >= 10 and value[4] == "-" and value[7] == "-":
                try:
                    parsed = parse_dt(value)
                except AppError:
                    parsed = None
            return (0, parsed.timestamp() if parsed else value.lower())
        return (0, value)

    return sorted(hits, key=sort_key, reverse=reverse)


# --------------------------------------------------------------------------- #
# The search op
# --------------------------------------------------------------------------- #


class SearchArgs(BaseModel):
    """The shape every search op takes.

    Extra top-level keys are folded into ``filter`` rather than rejected,
    because the planner writes `{"from_email": ...}` as often as it writes
    `{"filter": {"from_email": ...}}` and both mean the same thing. What the
    fold cannot do is invent a filter: an unknown key still fails against the
    op's allowlist, with the allowlist in the error.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    query: str | None = None
    filter: dict[str, Any] = Field(default_factory=dict)
    project: list[str] | None = None
    order_by: str = "relevance"
    order_dir: str = "desc"
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    body: bool = False
    expect: str = "many"
    freshness: str = "cached"

    @model_validator(mode="after")
    def _fold_extras(self) -> "SearchArgs":
        extras = dict(self.__pydantic_extra__ or {})
        if extras:
            merged = dict(extras)
            merged.update(self.filter or {})
            object.__setattr__(self, "filter", merged)
            self.__pydantic_extra__.clear()
        if self.expect not in ("one", "many"):
            raise ValueError("expect is either 'one' or 'many'")
        if self.freshness not in ("cached", "live"):
            raise ValueError("freshness is either 'cached' or 'live'")
        return self


class SearchOp(Op):
    """A read over our pgvector mirror of one Google corpus.

    Subclasses declare the corpus, the filter allowlist and the orderings. The
    body is here once: validate, refresh if the step asked to be live, search,
    post-filter, order, page, project, and — when the step expects exactly one
    thing — apply the ambiguity test and raise a choice card instead of
    guessing.
    """

    #: Post-filters that may be dropped, once, when they empty a search that
    #: still has a query. People-filters earn this: the address in them came
    #: out of `resolve.person`, and a wrong resolution — "John" read as a
    #: newsletter's sender — must not erase two meetings that carry John in
    #: the title. The query stays, so the name still does the constraining;
    #: the result says which filter was dropped so the answer can too.
    rescuable_filters: frozenset[str] = frozenset()

    is_local = True
    timeout_s = 3.0
    args_model = SearchArgs
    corpus: str = "gmail"
    filter_spec: dict[str, str | None] = {}
    order_spec: dict[str, str] = {}
    entity_type: str = "email"
    output_fields = ["hits", "count", "has_more", "next_offset"]

    #: How many extra rows to pull so a Python post-filter has something to
    #: keep and `has_more` is an answer rather than a guess.
    overfetch: int = 3

    # -- hooks ------------------------------------------------------------- #

    async def refresh_live(self, ctx: OpContext, args: Any, filters: FilterPlan) -> dict | None:
        """One targeted Google list, landing in the mirror. Optional."""
        return None

    async def _mirror_is_stale(self, ctx: OpContext) -> bool:
        """Whether this corpus is too far behind to answer from on its own.

        Two things make it stale: the last successful sync is older than the
        freshness target, or sync is currently failing, in which case the age
        will only grow. A mirror with rows but no sync record is left alone —
        an empty one is already handled by the cache-miss branch.

        Guarded in Redis for a short window so a plan with several steps over
        the same corpus makes one live call rather than one per step. The
        breaker and the quota bucket sit in front of the call itself, so this
        guard is about not being wasteful rather than about safety.
        """
        from app.core import cache
        from app.db.repositories import sync_state as sync_repo

        try:
            state = await sync_repo.get_state(ctx.session, ctx.user_id, self.corpus)
        except Exception:  # noqa: BLE001 - never fail a read over a freshness check
            return False

        if state is None or state.last_success_at is None:
            # No bookkeeping means we do not know the age — and for a mirror
            # that *has* rows, "unknown" is not a reason to spend a Google call
            # on every read. The never-synced case is already covered: an empty
            # mirror takes the branch above this one.
            stale = False
        elif int(getattr(state, "consecutive_failures", 0) or 0) > 0:
            stale = True
        else:
            age = (dt.datetime.now(dt.UTC) - state.last_success_at).total_seconds()
            stale = age > settings.SYNC_INTERVAL_MIN * 60

        if not stale:
            return False

        # One refresh per corpus per window, not one per step.
        try:
            client = await cache.get_redis()
            fresh = await client.set(
                cache.key("lr", f"{ctx.user_id}:{self.corpus}"),
                "1",
                ex=LIVE_REFRESH_EVERY_S,
                nx=True,
            )
            return bool(fresh)
        except Exception:  # noqa: BLE001 - no redis is not a reason to skip it
            return True

    async def _embed_backfilled(self, ctx: OpContext) -> None:
        """Vectorise what the live refresh just wrote, before reading it back.

        A row arrives from Google with a title and a body and no vector. The
        keyword arm can still find it by a word it literally contains, but the
        vector arm cannot see it at all — so "what are my meetings next week"
        misses an event called "John Wick Visit Offline", because *meetings* is
        not a word in it. That reads as the search being broken, and from where
        the person is sitting it is.

        Normally the embed queue does this in the background. On this path
        there is no time for background: the row was fetched because somebody
        is waiting on an answer that depends on it. One request for a handful
        of rows is worth it; the alternative is a confidently empty reply.
        """
        from app.search.embedder import embed_pending

        try:
            await embed_pending(ctx.session, ctx.user_id, self.corpus, limit=EMBED_ON_BACKFILL)
        except Exception as exc:  # noqa: BLE001 - a missing vector is not a failure
            # The keyword arm still works, so the read continues either way.
            log.warning("op.backfill_embed_failed", op=self.name, error=str(exc))

    def ambiguity_question(self, args: Any, hits: list[dict]) -> str:
        return "Which one did you mean?"

    def not_found_question(self, args: Any) -> str:
        """What to ask when the search came back empty.

        Names what was looked for, so the answer is a correction rather than a
        guess at what went wrong.
        """
        looked_for = str(getattr(args, "query", "") or "").strip()
        noun = _NOUN_OF.get(self.corpus, "anything")
        if looked_for:
            return f"I could not find {noun} matching “{looked_for}”. How would you describe it?"
        return f"I could not find {noun} like that. How would you describe it?"

    # -- run --------------------------------------------------------------- #

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        plan = build_filters(self.name, self.corpus, parsed.filter, self.filter_spec)

        degraded: list[str] = []
        if parsed.freshness == "live":
            try:
                refreshed = await self.refresh_live(ctx, parsed, plan)
            except AppError as exc:
                # `freshness: "live"` is the planner saying the answer has to be
                # current — "what time is tomorrow's meeting" cannot be served
                # from a mirror that may be fifteen minutes behind. So a failure
                # here is NOT quietly swallowed: a stale time stated as fact is
                # worse than admitting the calendar is unreachable. The step
                # fails, and the answer says which service went missing.
                #
                # This is the opposite of the cache-miss path below, and the
                # difference is who asked for the call.
                refreshed = None
                degraded.append(f"live_refresh:{exc.code}")
            else:
                if refreshed is None:
                    degraded.append("live_refresh_unavailable")

        want = parsed.offset + parsed.limit
        limit = max(want + 1, want * self.overfetch if plan.post else want + 1)

        async def read() -> list[Any]:
            return await hybrid(
                ctx,
                self.corpus,
                query=parsed.query,
                filters=plan.sql,
                limit=limit,
            )

        rows = await read()

        # When to go past the mirror to Google itself.
        #
        # A cache miss is the obvious case: a meeting booked five minutes ago
        # is simply not in the mirror yet, and "no events matched" is a wrong
        # answer rather than a slow one.
        #
        # A *stale* mirror is the case that hides. It returns rows, so nothing
        # looks wrong — the answer is just quietly out of date, and the reader
        # has no way to tell. Past the freshness target, or while sync is
        # failing, the rows on hand are not good enough to answer from alone.
        if parsed.freshness != "live" and (not rows or await self._mirror_is_stale(ctx)):
            try:
                refreshed = await self.refresh_live(ctx, parsed, plan)
            except Exception as exc:  # noqa: BLE001 - deliberately everything
                # Here the refresh is OUR idea, not the planner's: the read was
                # going to happen against the mirror regardless, and this is an
                # attempt to improve it. So nothing Google does may fail it.
                #
                # Catching `AppError` alone was a hole you could drive a bus
                # through — `GoogleAPIError` does not inherit from it, so every
                # real 401 and 503 sailed past and killed the step one line
                # before the code that reads rows we had already synced. That is
                # "Calendar is unavailable" about a calendar sitting in our own
                # Postgres, which is the complaint that led here.
                degraded.append(f"live_backfill:{_reason_of(exc)}")
            else:
                # A dict saying `fetched: 0` is still a dict, and truthiness
                # alone would buy a second identical query for nothing.
                fetched = int((refreshed or {}).get("fetched") or 0)
                if fetched:
                    degraded.append(f"live_backfill:{fetched}")
                    await self._embed_backfilled(ctx)
                    rows = await read()

        hits = [
            shape_hit(
                self.corpus,
                row,
                query=parsed.query,
                filters=plan.sql,
                body=parsed.body,
                extract=parsed.body or self.corpus == "gmail",
            )
            for row in rows
        ]
        unfiltered = hits
        hits = apply_post_filters(self.corpus, hits, plan.post)
        # A row a people-filter SELECTED has already answered the person part
        # of the question. "Move the meeting with John" resolves John to an
        # address and filters on it; the surviving rows are John's meetings by
        # construction — but their titles say "Design sync", score 0.25
        # against the word "John", and the floor throws away exactly what the
        # filter found. The filter vouches for them: same keys, same logic as
        # the rescue below, pointing the other way.
        vouching = {k for k in {**plan.sql, **plan.post} if k in self.rescuable_filters}
        if vouching and hits:
            for hit in hits:
                flags = list(hit.get("evidence") or [])
                if "FILTER_MATCH" not in flags:
                    hit["evidence"] = [*flags, "FILTER_MATCH"]
        dropped_filters: list[str] = []
        if not hits and unfiltered and parsed.query:
            sacrificial = sorted(k for k in plan.post if k in self.rescuable_filters)
            if sacrificial:
                kept = {k: v for k, v in plan.post.items() if k not in self.rescuable_filters}
                rescued = apply_post_filters(self.corpus, unfiltered, kept)
                if rescued:
                    hits = rescued
                    dropped_filters = sacrificial
                    degraded.append("filter_dropped:" + ",".join(sacrificial))
        # Best match first, before anything reorders for presentation. The
        # ambiguity test below asks "is the leader clearly the leader?", which
        # is a question about match quality — and `order_by: starts_at` would
        # otherwise hand it the *earliest* row and let a weak match masquerade
        # as the top hit. "Move the meeting with John" then reports no such
        # meeting while two of them sit in the results.
        by_score = sorted(hits, key=lambda h: float(h.get("cn") or 0.0), reverse=True)

        hits = order_hits(hits, parsed.order_by, parsed.order_dir, self.order_spec)

        total_found = len(hits)
        page = hits[parsed.offset : parsed.offset + parsed.limit]

        if parsed.expect == "one":
            ranked = by_score[parsed.offset : parsed.offset + parsed.limit]
            reason = ambiguity_test(ranked)
            # Only candidates that clear the bar go on the card. Offering a row
            # the search itself does not believe in — "Hello World" against a
            # query for John — turns a two-way pick into a three-way guess and
            # makes the good options look less certain than they are.
            plausible = [h for h in ranked if admits(h)] or ranked[:1]
            if reason == "margin":
                return OpResult(
                    data={
                        "hits": [project_fields(h, parsed.project) for h in plausible],
                        "count": len(plausible),
                        "ambiguous": True,
                        "reason": reason,
                        "corpus": self.corpus,
                    },
                    needs_input=choice_input(
                        self.ambiguity_question(parsed, plausible),
                        plausible,
                        id_field=REF_FIELD[self.corpus],
                        help_text="Two candidates scored within the margin, so this is a guess "
                        "unless you say which."
                        if not dropped_filters
                        else "These match by name; none of them lists the person I resolved, "
                        "so say which one you meant.",
                    ),
                )
            if reason == "absent":
                # Nothing matched, and a step downstream was going to use this.
                # Returning empty here ends the run on "cannot resolve
                # {{...hits[0].event_id}}" — true, useless, and a dead end.
                #
                # Ask instead. Describing the thing again in their own words is
                # something a person can actually do, unlike supplying an id,
                # and the card carries the ordinary "search differently" and
                # "cancel" ways out.
                return OpResult(
                    data={
                        "hits": [],
                        "count": 0,
                        "has_more": False,
                        "next_offset": parsed.offset,
                        "found": False,
                        "reason": "absent",
                        "corpus": self.corpus,
                        "query": parsed.query,
                    },
                    needs_input=InputRequest(
                        kind="text",
                        question=self.not_found_question(parsed),
                        value_schema={"type": "string", "minLength": 2},
                    ),
                )

        shown = [project_fields(h, parsed.project) for h in page]
        data: dict[str, Any] = {
            "hits": shown,
            "count": len(shown),
            "has_more": total_found > parsed.offset + len(page),
            "next_offset": parsed.offset + len(page),
            "corpus": self.corpus,
            "query": parsed.query,
        }
        if degraded:
            data["degraded"] = degraded
        if dropped_filters:
            data["dropped_filters"] = dropped_filters
        if parsed.expect == "one" and shown:
            # A step that expects one thing is referenced as `{{step.field}}`,
            # not `{{step.hits[0].field}}`. Both work.
            data = {**shown[0], **data, "found": True}
        return OpResult(data=data)

    # -- presentation ------------------------------------------------------ #

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        slim = dict(data)
        hits = []
        for hit in (data.get("hits") or [])[:8]:
            keep = {
                k: v
                for k, v in hit.items()
                if k in ("ref", "label", "cn", "evidence", "excerpt", "extracted")
                or k in (REF_FIELD.get(self.corpus, "id"), TIME_FIELD.get(self.corpus, ""))
            }
            hits.append(keep)
        slim["hits"] = hits
        return trim_for_llm(slim, budget)


__all__ = [
    "EXACT_FLAGS",
    "FLOOR_READ",
    "FLOOR_WRITE",
    "MARGIN",
    "ConfirmableOp",
    "FilterPlan",
    "GoogleClients",
    "InputRequest",
    "Op",
    "OpContext",
    "OpResult",
    "SearchArgs",
    "SearchOp",
    "admits",
    "ambiguity_test",
    "apply_post_filters",
    "as_bool",
    "as_list",
    "build_filters",
    "catalogue_line_for",
    "choice_input",
    "duration_minutes",
    "embed_query",
    "turn_embedding",
    "ensure_utc",
    "excerpt_of",
    "google_call",
    "has_google",
    "hybrid",
    "is_exact",
    "iso",
    "jsonable",
    "label_for",
    "order_hits",
    "parse_dt",
    "project_fields",
    "row_to_dict",
    "run_extractors",
    "shape_hit",
    "trim_for_llm",
    "utcnow",
    "window_bounds",
    "REF_FIELD",
    "TIME_FIELD",
]
