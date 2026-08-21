"""Execution. One asyncio task per step, each awaiting only its own edges.

Not level batching. A plan where `a` takes 40 ms and `b` takes 400 ms and `c`
depends only on `a` should start `c` at 40 ms, not at 400 — level batching makes
every step in a level wait for the slowest one in it, which throws away most of
what having a DAG bought. So every step gets a task, every task awaits the
events of its own dependencies, and the shape of the graph decides the order.

What else lives here:

* :func:`bind` — the reference resolver. The planner writes
  ``{{booking.extracted.pnr}}`` and this puts the real value there at the moment
  the step runs. A reference that cannot resolve **raises**; it never binds to
  ``None``, because "booking None" in a draft email is the failure this whole
  design exists to prevent.
* per-service semaphores, so a fan-out cannot take the whole quota;
* two attempts, in-request, and only for the two error classes where trying
  again is honest — releasing the semaphore around the sleep, because holding a
  slot while doing nothing is how a backoff becomes a queue;
* a soft and a hard deadline measured in **executing** time, so a run that sat
  overnight on a question resumes with its whole budget;
* pausing on a question, signalling a replan, and staging a write instead of
  performing one.

A failed step never cancels a sibling. It cancels what depends on it, and the
answer says what was lost.
"""

from __future__ import annotations

import asyncio
import copy
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator, Awaitable, Callable, Final

from app.core.errors import AppError
from app.core.logging import get_logger
from app.ops.base import service_of
from app.orchestrator import temporal

log = get_logger(__name__)

# How many steps may be in flight against each service at once. Google's is
# small on purpose: the quota is shared across the whole project, and a wide
# fan-out from one user is exactly how one tenant blinds everybody else.
SEMAPHORE_LIMITS: Final[dict[str, int]] = {
    "gmail": 3,
    "gcal": 3,
    "gdrive": 3,
    "local": 8,
    "openai": 4,
}

SOFT_DEADLINE_MS: Final[int] = 2500
HARD_DEADLINE_MS: Final[int] = 12000

# In-request retries are capped in attempts *and* in added latency. A third
# attempt landing at four seconds is strictly worse than a degraded answer at
# two, because a person is watching.
MAX_ATTEMPTS: Final[int] = 2
RETRY_LATENCY_CAP_MS: Final[int] = 1500
MAX_REPLAN_ROUNDS: Final[int] = 2

# The only two classes where trying again in front of a user is honest. A rate
# limit or an exhausted quota will still be a rate limit in 400 ms; a
# precondition failure means the world changed and repeating the call would
# overwrite something the user has not seen.
RETRYABLE_CLASSES: Final[frozenset[str]] = frozenset({"TRANSIENT", "AUTH_EXPIRED"})

#: The key `app/tasks/actions.py` reads to find an action's predecessors. Same
#: constant, spelled once there and once here: dispatch must not import the
#: Celery package, which imports the orchestrator.
SEQUENCE_KEY: Final[str] = "_sequence_after"


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


class BindError(AppError):
    """A reference that does not resolve. Loud, and before the step runs."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("VALIDATION_ERROR", message, details=details or None)


_REFERENCE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[\s*(\*|-?\d+)\s*\]")
_MISSING = object()


def _split_filters(body: str) -> tuple[str, list[str]]:
    """Split ``path|filter(a, b)|other`` without cutting inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    cleaned = [p.strip() for p in parts if p.strip()]
    if not cleaned:
        return "", []
    return cleaned[0], cleaned[1:]


def _tokens(path: str) -> list[tuple[str, str]]:
    """``hits[0].id`` becomes ``[(name, hits), (index, 0), (name, id)]``."""
    out: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(path):
        if match.group(1) is not None:
            out.append(("name", match.group(1)))
        else:
            out.append(("index", match.group(2)))
    return out


def _lookup(container: Any, name: str) -> Any:
    """A step result travels as a dict here and as an object there."""
    if isinstance(container, dict):
        return container.get(name, _MISSING)
    value = getattr(container, name, _MISSING)
    return value


def _resolve_path(path: str, scope: Any, raw: str) -> Any:
    """Walk one path through the scope, or raise saying exactly where it stopped."""
    tokens = _tokens(path)
    if not tokens or tokens[0][0] != "name":
        raise BindError(f"cannot resolve {raw}: {path!r} is not a reference path")

    root = tokens[0][1]
    current = _lookup(scope, root)
    if current is _MISSING:
        steps = _lookup(scope, "steps")
        current = _lookup(steps, root) if steps is not _MISSING else _MISSING
    if current is _MISSING:
        raise BindError(
            f"cannot resolve {raw}: there is no {root!r} here — no such step ran, "
            "and it is not one of search, windows, probe or user",
            reference=raw,
            root=root,
        )

    walked = root
    for kind, token in tokens[1:]:
        if kind == "name":
            if current is None:
                raise BindError(
                    f"cannot resolve {raw}: {walked!r} is null, so it has no {token!r}",
                    reference=raw,
                )
            found = _lookup(current, token)
            if found is _MISSING:
                raise BindError(
                    f"cannot resolve {raw}: {walked!r} has no {token!r} — "
                    "the value is not there, and binding it to null is not an option",
                    reference=raw,
                    missing=token,
                )
            current = found
            walked = f"{walked}.{token}"
            continue

        if token == "*":
            if current is None:
                current = []
            if not isinstance(current, (list, tuple)):
                raise BindError(
                    f"cannot resolve {raw}: {walked!r} is not a list, so [*] means nothing",
                    reference=raw,
                )
            rest = path.split("[*]", 1)[1].lstrip(".")
            items = list(current)
            if not rest:
                return items
            return [_resolve_path(f"__item__.{rest}", {"__item__": item}, raw) for item in items]

        if not isinstance(current, (list, tuple)):
            raise BindError(
                f"cannot resolve {raw}: {walked!r} is not a list, so [{token}] means nothing",
                reference=raw,
            )
        index = int(token)
        if index >= len(current) or index < -len(current):
            raise BindError(
                f"cannot resolve {raw}: index {index} is out of range — "
                f"{walked!r} holds {len(current)} item(s)",
                reference=raw,
                index=index,
                length=len(current),
            )
        current = current[index]
        walked = f"{walked}[{index}]"

    return current


# -- filters ----------------------------------------------------------------


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise BindError(f"{value!r} is not a time, so it cannot be shifted") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


_SHIFT = re.compile(r"^([+-])(\d+)([dhms])$")


def _apply_filter(value: Any, spec: str, scope: Any, ctx: "BindContext", raw: str) -> Any:
    name, _, argument = spec.partition("(")
    name = name.strip()
    argument = argument.rstrip(")").strip()

    shift = _SHIFT.match(name)
    if shift:
        sign, count, unit = shift.group(1), int(shift.group(2)), shift.group(3)
        delta = {"d": timedelta(days=count), "h": timedelta(hours=count),
                 "m": timedelta(minutes=count), "s": timedelta(seconds=count)}[unit]
        moment = _as_datetime(value)
        return moment - delta if sign == "-" else moment + delta

    if name in ("day_start", "start_of_day"):
        local = _as_datetime(value).astimezone(temporal._zone(ctx.tz))  # noqa: SLF001
        return local.replace(hour=0, minute=0, second=0, microsecond=0)
    if name in ("day_end", "end_of_day"):
        local = _as_datetime(value).astimezone(temporal._zone(ctx.tz))  # noqa: SLF001
        return local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if name == "iso":
        return _as_datetime(value).isoformat()
    if name == "date":
        return _as_datetime(value).astimezone(temporal._zone(ctx.tz)).date().isoformat()  # noqa: SLF001
    if name == "resolve_time":
        window = temporal.resolve(str(value), ctx.tz, ctx.week_start, ctx.now)
        if window is None:
            raise BindError(
                f"cannot resolve {raw}: {value!r} is not a time phrase I can read"
            )
        return window.start.isoformat()
    if name == "lower":
        return str(value).lower()
    if name == "upper":
        return str(value).upper()
    if name == "strip":
        return str(value).strip()
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    if name == "count" or name == "length":
        return len(value or [])
    if name == "first":
        return (value or [None])[0]
    if name == "last":
        return (value or [None])[-1]
    if name == "unique":
        seen: list[Any] = []
        for item in value or []:
            if item not in seen:
                seen.append(item)
        return seen
    if name == "join":
        separator = argument.strip("\"'") or ", "
        return separator.join(str(v) for v in (value or []))
    if name == "exclude":
        drop = _resolve_argument(argument, scope, ctx, raw)
        drops = drop if isinstance(drop, (list, tuple, set)) else [drop]
        lowered = {str(d).lower() for d in drops}
        return [v for v in (value or []) if str(v).lower() not in lowered]
    if name == "as_options":
        keys = [k.strip() for k in argument.split(",") if k.strip()]
        id_key = keys[0] if keys else "id"
        label_key = keys[1] if len(keys) > 1 else "label"
        meta_keys = keys[2:]
        options = []
        for item in value or []:
            option = {
                "id": _lookup(item, id_key),
                "label": _lookup(item, label_key),
            }
            meta = {k: _lookup(item, k) for k in meta_keys}
            meta = {k: v for k, v in meta.items() if v is not _MISSING}
            if meta:
                option["meta"] = meta
            options.append(
                {k: (None if v is _MISSING else v) for k, v in option.items()}
            )
        return options

    raise BindError(f"cannot resolve {raw}: there is no {name!r} filter")


def _resolve_argument(argument: str, scope: Any, ctx: "BindContext", raw: str) -> Any:
    """A filter argument is either a literal or another reference."""
    text = argument.strip()
    if not text:
        return None
    inner = _REFERENCE.fullmatch(text)
    if inner:
        text = inner.group(1).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    try:
        return _resolve_path(text, scope, raw)
    except BindError:
        return text


@dataclass(frozen=True)
class BindContext:
    now: datetime
    tz: str
    week_start: int


def _resolve_reference(body: str, scope: Any, ctx: BindContext, raw: str) -> Any:
    path, filters = _split_filters(body)
    value = _resolve_path(path, scope, raw)
    for spec in filters:
        value = _apply_filter(value, spec, scope, ctx, raw)
    return value


def bind(
    value: Any,
    scope: Any,
    *,
    now: datetime | None = None,
    tz: str = "UTC",
    week_start: int = 1,
) -> Any:
    """Resolve every reference in an argument tree, at the moment it is needed.

    A string that is *exactly* one reference keeps the referenced value's type —
    ``{{mail.count}}`` binds to the integer 3, not to ``"3"``, and
    ``{{mail.hits[*].id}}`` binds to a real list. A reference inside a sentence
    is interpolated. Anything that will not resolve raises.
    """
    context = BindContext(now or datetime.now(UTC), tz or "UTC", int(week_start or 1))
    return _bind(value, scope, context)


def _bind(value: Any, scope: Any, ctx: BindContext) -> Any:
    if isinstance(value, str):
        whole = _REFERENCE.fullmatch(value.strip())
        if whole:
            return _resolve_reference(whole.group(1).strip(), scope, ctx, value.strip())

        def replace(match: "re.Match[str]") -> str:
            resolved = _resolve_reference(
                match.group(1).strip(), scope, ctx, match.group(0)
            )
            return "" if resolved is None else str(resolved)

        return _REFERENCE.sub(replace, value)

    if isinstance(value, dict):
        # `{"time_phrase": "tomorrow"}` is resolved here rather than at plan
        # time because a paused run can sit overnight, and "tomorrow" has to
        # mean tomorrow from the moment the step actually runs.
        if "time_phrase" in value and len(value) <= 2:
            phrase = _bind(value["time_phrase"], scope, ctx)
            anchor = value.get("anchor")
            window = temporal.resolve(
                str(phrase),
                ctx.tz,
                ctx.week_start,
                ctx.now,
                anchor=_as_datetime(anchor) if anchor else None,
            )
            if window is None:
                raise BindError(
                    f"cannot resolve the time phrase {phrase!r} — it names no window "
                    "I can compute, so the step has no dates to run against",
                    time_phrase=str(phrase),
                )
            return {
                "start": window.start,
                "end": window.end,
                "tz": window.tz,
                "interpretation": window.interpretation,
            }
        return {key: _bind(child, scope, ctx) for key, child in value.items()}

    if isinstance(value, list):
        return [_bind(child, scope, ctx) for child in value]
    if isinstance(value, tuple):
        return tuple(_bind(child, scope, ctx) for child in value)
    return value


# Names other modules and the tests reach for.
bind_references = bind
bind_args = bind
resolve_references = bind


def json_safe(value: Any) -> Any:
    """Make a bound argument tree storable in ``node_executions.args``."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def evaluate_gate(gate: dict[str, Any], scope: Any, ctx: BindContext) -> bool:
    """A cheap alternative to a replan: skip the step unless the test passes."""
    test = str(gate.get("test") or "exists")
    try:
        left = _bind(gate.get("left"), scope, ctx)
    except BindError:
        # A gate whose subject does not exist is a gate that did not pass.
        return test in ("empty", "not_exists")
    right = _bind(gate.get("right"), scope, ctx) if "right" in gate else None

    if test == "exists":
        return bool(left) or left == 0
    if test in ("empty", "not_exists"):
        return not left and left != 0
    if test == "count_gt":
        return len(left or []) > int(right or 0)
    if test == "equals":
        return left == right
    if test == "contains":
        if isinstance(left, str):
            return str(right) in left
        return right in (left or [])
    if test == "within":
        window = right if isinstance(right, dict) else {}
        moment = _as_datetime(left)
        return _as_datetime(window["start"]) <= moment < _as_datetime(window["end"])
    if test == "before":
        return _as_datetime(left) < _as_datetime(right)
    if test == "after":
        return _as_datetime(left) > _as_datetime(right)
    return True


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget:
    """Wall time spent *executing*, which is not the same as elapsed time.

    A run that paused for a question and resumed tomorrow has spent no budget
    while it waited. Measuring elapsed time instead would hand a resumed run a
    deadline that expired overnight, and it would time out before doing
    anything at all.
    """

    def __init__(self, soft_ms: int = SOFT_DEADLINE_MS, hard_ms: int = HARD_DEADLINE_MS) -> None:
        self.soft_ms = soft_ms
        self.hard_ms = hard_ms
        self._spent = 0.0
        self._started: float | None = None

    def resume(self) -> "Budget":
        if self._started is None:
            self._started = time.monotonic()
        return self

    def pause(self) -> None:
        if self._started is not None:
            self._spent += (time.monotonic() - self._started) * 1000.0
            self._started = None

    @property
    def spent_ms(self) -> float:
        running = (time.monotonic() - self._started) * 1000.0 if self._started else 0.0
        return self._spent + running

    @property
    def soft_expired(self) -> bool:
        return self.spent_ms >= self.soft_ms

    @property
    def hard_expired(self) -> bool:
        return self.spent_ms >= self.hard_ms

    def remaining_hard_s(self) -> float:
        return max(0.0, (self.hard_ms - self.spent_ms) / 1000.0)

    def __enter__(self) -> "Budget":
        return self.resume()

    def __exit__(self, *_exc: Any) -> None:
        self.pause()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    """One node's whole story: what ran, how it ended, and how long it took."""

    node_id: str
    op: str
    status: str = "pending"
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    retries: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def attempts(self) -> int:
        return len(self.retries) + 1

    @property
    def duration_ms(self) -> int | None:
        if not self.started_at or not self.finished_at:
            return None
        return int((self.finished_at - self.started_at).total_seconds() * 1000)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "status": self.status,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "outcome": self.outcome,
        }


@dataclass
class PausedOn:
    """The question a run stopped on."""

    node_id: str
    op: str
    input_id: str | None
    request: Any
    blocking: bool = True



#: Failure classes and reasons that are OUR doing, not Google's.
_OUR_FAULT = {"INVALID", "VALIDATION_ERROR", "NOT_FOUND"}
_OUR_REASONS = {"unresolved_reference", "bad_args", "gate_false", "plan_invalid"}


def _fault_of(o: "StepOutcome") -> str:
    """Whose fault a failed step was: `service`, `plan` or `dependency`.

    Splitting the op name and calling everything a service outage tells someone
    "Gmail is not responding" when the real cause was a plan referencing a
    field the op never returns. That is our bug wearing Google's name, and it
    is worse than saying nothing — it sends them to check a status page.

    Module level, and used by every producer of a degraded entry. It lived
    inside one function once, so the *other* producer had no classification at
    all and stamped a service name on every failure regardless.
    """
    detail = o.outcome or {}
    if detail.get("reason") == "dependency_failed":
        return "dependency"
    if detail.get("class") in _OUR_FAULT or detail.get("reason") in _OUR_REASONS:
        return "plan"
    if o.status == "skipped":
        return "dependency"
    return "service"


@dataclass
class DispatchResult:
    status: str  # complete | paused | replan | failed
    results: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[str, StepOutcome] = field(default_factory=dict)
    staged: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[dict[str, Any]] = field(default_factory=list)
    #: Nodes that never ran because the run stopped to ask the person a
    #: question. They have not failed and must never be reported as degraded:
    #: the plan is mid-flight, waiting on an answer.
    awaiting: frozenset[str] = frozenset()
    paused_on: PausedOn | None = None
    replan_reason: str | None = None
    took_ms: int = 0

    @property
    def failed_services(self) -> list[str]:
        """Services a step genuinely could not get an answer out of.

        Failed and timed out only — never skipped. A skipped step never ran:
        the run paused on a question, a gate said no, or its dependency
        produced nothing. This list is shipped to the browser as bare service
        names and the client widens a bare name into "unavailable", so a skip
        in here puts "Calendar — unavailable" over a run whose calendar step
        worked and whose next step is a question we asked on purpose. The repo
        twin (`runs.failed_services`) follows the same rule for reloads.
        """
        return sorted(
            {
                outcome.op.split(".", 1)[0]
                for outcome in self.outcomes.values()
                if outcome.status in ("failed", "timeout")
                and _fault_of(outcome) == "service"
                and outcome.op.split(".", 1)[0] not in ("ask", "meta")
            }
        )

    def degraded_context(self) -> dict[str, Any]:
        """The structured shape the synthesizer is given for a partial answer."""
        failed = []
        for o in self.outcomes.values():
            if o.status not in ("failed", "timeout"):
                continue
            fault = _fault_of(o)
            failed.append(
                {
                    "node": o.node_id,
                    # Only a real service failure names a service. A plan fault
                    # leaves it None so nothing downstream can blame Google for
                    # a step we built wrong.
                    "service": service_of(o.op) if fault == "service" else None,
                    "fault": fault,
                    "class": (o.outcome or {}).get("class"),
                    "code": (o.outcome or {}).get("code"),
                    "attempts": o.attempts,
                    "message": (o.outcome or {}).get("message"),
                }
            )
        skipped = [
            {"node": o.node_id, "because": (o.outcome or {}).get("depends_on")}
            for o in self.outcomes.values()
            if o.status == "skipped" and o.node_id not in self.awaiting
        ]
        succeeded = [
            {"node": o.node_id, "op": o.op} for o in self.outcomes.values() if o.ok
        ]
        return {
            "degraded": bool(failed or skipped),
            "failed": failed,
            "skipped": skipped,
            "succeeded": succeeded,
        }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass
class Hooks:
    """Everything the dispatcher needs from the outside world.

    The dispatcher owns concurrency and nothing else. Writing rows and
    publishing events belong to the runner, which owns the transaction — so
    they arrive as callbacks rather than as imports.
    """

    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] = _noop
    on_step_started: Callable[[StepOutcome], Awaitable[None]] = _noop
    on_step_finished: Callable[[StepOutcome], Awaitable[None]] = _noop
    on_retry: Callable[[StepOutcome, dict[str, Any]], Awaitable[None]] = _noop
    raise_input: Callable[[dict[str, Any], Any], Awaitable[str | None]] | None = None
    stage_write: Callable[[dict[str, Any], dict[str, Any]], Awaitable[str | None]] | None = None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _classify(exc: BaseException) -> str:
    """Name the failure class, using the Google classifier when it applies."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "TRANSIENT"
    if isinstance(exc, BindError):
        return "INVALID"
    if isinstance(exc, AppError):
        return {
            "GOOGLE_UNAVAILABLE": "TRANSIENT",
            "RATE_LIMITED": "RATE_LIMITED",
            "GOOGLE_REAUTH_REQUIRED": "AUTH_REVOKED",
            "VALIDATION_ERROR": "INVALID",
            "NOT_FOUND": "NOT_FOUND",
        }.get(exc.code, "UNKNOWN")
    try:
        from app.google.retry import classify

        return str(classify(exc))
    except Exception:  # noqa: BLE001 - the classifier is another module's file
        return "UNKNOWN"


def _backoff(error_class: str, attempt: int) -> float:
    try:
        from app.google.retry import backoff

        return float(backoff(error_class, attempt))
    except Exception:  # noqa: BLE001 - full jitter, same shape, as a fallback
        import random

        return random.uniform(0, min(1.5, 0.4 * (2 ** (attempt - 1))))


def _status_code(exc: BaseException) -> int | None:
    for name in ("status_code", "status", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    return status if isinstance(status, int) else None


def _outcome_of(exc: BaseException, error_class: str) -> dict[str, Any]:
    reason = {
        "TRANSIENT": "google_unavailable",
        "RATE_LIMITED": "rate_limited",
        "QUOTA_EXHAUSTED": "quota_exhausted",
        "AUTH_EXPIRED": "auth_expired",
        "AUTH_REVOKED": "auth_revoked",
        "PRECONDITION": "precondition_failed",
        "NOT_FOUND": "not_found",
        "INVALID": "invalid_arguments",
    }.get(error_class, "failed")
    message = getattr(exc, "message", None) or str(exc)
    return {
        "reason": reason,
        "class": error_class,
        "code": _status_code(exc),
        "message": str(message)[:500],
    }


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


#: `gcal.search_events` names its service `gcal`; the drive ops say `drive`
#: while the transport is keyed `gdrive`.
_TRANSPORT_OF: Final[dict[str, str]] = {
    "gmail": "gmail",
    "gcal": "gcal",
    "calendar": "gcal",
    "drive": "gdrive",
    "gdrive": "gdrive",
}


def _transport_retries(ctx: Any, op_name: str) -> list[dict[str, Any]]:
    """Retries the Google transport made for itself, in node-row shape.

    Scoped to the one service this step talks to. Steps run concurrently and
    the transports are shared for the whole run, so a whole-run counter cannot
    say which step a retry belonged to — read Calendar's list for a Calendar
    step and Drive's retries cannot be credited to it.

    Anything that is not a real clients object — the stand-in used when Google
    is not connected, a test double — has no retries to report.
    """
    service = _TRANSPORT_OF.get(str(op_name).split(".", 1)[0])
    if service is None:
        return []
    try:
        transport = (getattr(ctx, "google", None).transports or {}).get(service)
        return list(getattr(transport, "attempts", None) or [])
    except Exception:  # noqa: BLE001 - a trace detail never fails a step
        return []


def _semaphore_key(op: Any, op_name: str) -> str:
    if getattr(op, "uses_llm", False):
        return "openai"
    service = op_name.split(".", 1)[0]
    if service in SEMAPHORE_LIMITS:
        return service if not getattr(op, "is_local", False) else "local"
    return "local"


def make_semaphores() -> dict[str, asyncio.Semaphore]:
    return {name: asyncio.Semaphore(limit) for name, limit in SEMAPHORE_LIMITS.items()}


async def dispatch(
    plan: dict[str, Any],
    ctx: Any,
    *,
    scope: dict[str, Any],
    registry: Any = None,
    hooks: Hooks | None = None,
    budget: Budget | None = None,
    completed: dict[str, Any] | None = None,
    semaphores: dict[str, asyncio.Semaphore] | None = None,
    week_start: int | None = None,
) -> DispatchResult:
    """Run a validated plan. One task per step, awaiting only its own edges."""
    from app.orchestrator.validate import _registry  # same accessor, one place

    ops = _registry(registry)
    hooks = hooks or Hooks()
    budget = (budget or Budget()).resume()
    semaphores = semaphores or make_semaphores()
    started = time.monotonic()

    # asyncpg connections are NOT safe for concurrent use, and every step task
    # below runs at the same time. Sharing one AsyncSession across them raises
    # "another operation is in progress" the moment two steps overlap — which is
    # exactly when the design is working. So: writes go through a lock (they are
    # short), and each step's op gets a session of its own.
    write_lock = asyncio.Lock()

    # Resolved once, here, rather than per step. Importing `app.db.session` and
    # building the sessionmaker are synchronous and take a couple of hundred
    # milliseconds cold — inside a step task that is not a slow step, it is the
    # whole event loop stopping, which stalls every *other* step too. Doing it
    # before the tasks start also means the fallback is decided once, and said
    # out loud once, instead of being swallowed silently on every attempt.
    session_maker = None
    if getattr(ctx, "session", None) is not None:
        try:
            from app.db.session import get_sessionmaker

            session_maker = get_sessionmaker()
        except Exception as exc:  # noqa: BLE001 - one session still works
            log.warning("dispatch.no_sessionmaker", error=str(exc))

    @asynccontextmanager
    async def step_context() -> "AsyncIterator[Any]":
        """A ctx clone with its own session, so steps really can overlap."""
        if session_maker is None:
            yield ctx  # single-session fallback: correct, just serialised
            return
        async with session_maker() as own:
            clone = copy.copy(ctx)
            try:
                clone.session = own
            except Exception:  # noqa: BLE001 - a frozen ctx cannot be cloned
                yield ctx
                return
            yield clone

    steps = [dict(step) for step in (plan.get("steps") or [])]
    by_id: dict[str, dict[str, Any]] = {str(step["id"]): step for step in steps}
    order = [str(step["id"]) for step in steps]

    # `week_start` is a user setting and `OpContext` does not carry one, so it
    # is passed in rather than guessed. Guessing it moves "next week" by a day.
    bind_ctx = BindContext(
        getattr(ctx, "now", None) or datetime.now(UTC),
        getattr(ctx, "tz", None) or "UTC",
        int(week_start or getattr(ctx, "week_start", 1) or 1),
    )

    # The bag every reference resolves against. Step results are added to it as
    # they land, which is safe because a step only reads its own ancestors and
    # those are finished before it starts.
    live_scope: dict[str, Any] = dict(scope or {})
    results: dict[str, Any] = dict(live_scope.get("steps") or {})
    for node_id, value in (completed or {}).items():
        results[node_id] = value
    live_scope["steps"] = results
    for node_id, value in results.items():
        live_scope.setdefault(node_id, value)

    outcomes: dict[str, StepOutcome] = {}
    done: dict[str, asyncio.Event] = {node_id: asyncio.Event() for node_id in order}
    staged: list[dict[str, Any]] = []
    #: node id -> the action it staged, so a dependant can name its predecessor.
    staged_by_node: dict[str, str] = {}
    paused: list[PausedOn] = []
    replan: list[str] = []
    stop = asyncio.Event()  # set when the run pauses; no new step may start

    def record(node_id: str, op_name: str) -> StepOutcome:
        outcome = outcomes.get(node_id)
        if outcome is None:
            outcome = StepOutcome(node_id=node_id, op=op_name)
            outcomes[node_id] = outcome
        return outcome

    async def finish(
        node: StepOutcome,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        node.status = status
        node.finished_at = datetime.now(UTC)
        if result is not None:
            node.result = result
        if outcome is not None:
            node.outcome = outcome
        done[node.node_id].set()
        async with write_lock:
            await hooks.on_step_finished(node)

    async def run_one(node_id: str) -> None:
        step = by_id[node_id]
        op_name = str(step.get("op") or "")
        outcome = record(node_id, op_name)

        # Already have a result from an earlier round or a paused run.
        if node_id in results and results[node_id] is not None:
            outcome.status = "succeeded"
            outcome.result = results[node_id]
            done[node_id].set()
            return

        deps = [str(d) for d in (step.get("depends_on") or []) if str(d) in done]
        if deps:
            await asyncio.wait([asyncio.create_task(done[d].wait()) for d in deps])

        broken = [d for d in deps if not outcomes.get(d, StepOutcome(d, "")).ok]
        if broken:
            # A dependency did not produce what this step reads. It never ran,
            # so this is a skip and not a Google error — the distinction matters
            # because a skip must not count against the circuit breaker.
            await finish(
                outcome,
                "skipped",
                outcome={
                    "reason": "dependency_failed",
                    "depends_on": broken[0],
                    "message": f"{broken[0]} did not produce a result, so this could not run",
                },
            )
            return

        if stop.is_set():
            await finish(
                outcome,
                "skipped",
                outcome={"reason": "run_paused", "message": "the run stopped on a question"},
            )
            return

        op = ops.get(op_name)
        if op is None:
            await finish(
                outcome,
                "failed",
                outcome={"reason": "unknown_op", "class": "INVALID",
                         "message": f"{op_name} is not a registered op"},
            )
            return

        if budget.hard_expired:
            await finish(
                outcome,
                "timeout",
                outcome={"reason": "hard_deadline", "class": "TRANSIENT",
                         "message": "the run ran out of time before this step started"},
            )
            return

        # Past the soft deadline we stop starting work the answer can live
        # without, rather than spending a budget we already know we will miss.
        if budget.soft_expired and (step.get("optional") or step.get("defer")):
            await finish(
                outcome,
                "skipped",
                outcome={"reason": "soft_deadline", "class": "TRANSIENT",
                         "message": "skipped to keep the answer inside its time budget"},
            )
            return

        try:
            bound = _bind(step.get("args") or {}, live_scope, bind_ctx)
        except BindError as exc:
            await finish(
                outcome,
                "failed",
                outcome={"reason": "unresolved_reference", "class": "INVALID",
                         "message": exc.message},
            )
            # Same deal as an op rejecting its arguments below: the plan wrote
            # a reference that does not resolve, and the planner can fix it if
            # it hears what broke — one repair round beats a degraded answer.
            replan.append(f"{node_id} ({op_name}) has a broken reference: {exc.message}")
            return

        gate = step.get("gate")
        if isinstance(gate, dict) and gate:
            try:
                passed = evaluate_gate(gate, live_scope, bind_ctx)
            except Exception:  # noqa: BLE001 - an unreadable gate is a failed gate
                passed = False
            if not passed:
                await finish(
                    outcome,
                    "skipped",
                    outcome={"reason": "gate_not_met", "gate": gate,
                             "message": "the condition on this step was not met"},
                )
                return

        outcome.args = json_safe(bound)
        outcome.started_at = datetime.now(UTC)
        outcome.status = "running"
        async with write_lock:
            await hooks.on_step_started(outcome)

        key = _semaphore_key(op, op_name)
        semaphore = semaphores.get(key) or semaphores.setdefault(key, asyncio.Semaphore(4))
        max_attempts = min(int(getattr(op, "max_attempts", MAX_ATTEMPTS) or MAX_ATTEMPTS), MAX_ATTEMPTS)
        timeout_s = float(getattr(op, "timeout_s", 6.0) or 6.0)
        retry_ms = 0.0
        attempt = 0
        google_retried = False
        op_result: Any = None
        failure: BaseException | None = None

        while True:
            attempt += 1
            failure = None
            async with semaphore:
                try:
                    async with step_context() as step_ctx:
                        seen = _transport_retries(step_ctx, op_name)
                        op_result = await asyncio.wait_for(
                            op.run(step_ctx, bound),
                            timeout=min(timeout_s, budget.remaining_hard_s() or timeout_s),
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - classified immediately below
                    failure = exc
                finally:
                    # Two tiers retry, and only one of them was writing itself
                    # down. A 429 the transport absorbed and recovered from is
                    # the difference between "Calendar is slow" and "Calendar is
                    # fine", and a trace that does not show it cannot be used to
                    # tell those apart. Reported through the same hook the
                    # dispatcher's own retries use, so it lands on the node row
                    # and on the wire by the path that already works.
                    fresh = _transport_retries(step_ctx, op_name)[len(seen) :]
                    google_retried = bool(fresh)
                    for entry in fresh:
                        outcome.retries.append(entry)
                        try:
                            await hooks.on_retry(
                                outcome,
                                {**entry, "attempt": attempt, "of": max_attempts, "tier": "google"},
                            )
                        except Exception:  # noqa: BLE001 - a trace detail, never the answer
                            pass
            if failure is None:
                break

            error_class = _classify(failure)
            if attempt >= max_attempts or error_class not in RETRYABLE_CLASSES:
                break

            # The transport already tried this one again, closer to the failure
            # and out of the same user-facing budget. Two tiers each taking
            # their own two attempts is four calls at a service that has
            # answered the same way twice, so this tier defers.
            if google_retried:
                break

            # The in-request tier is capped in attempts *and* in added latency.
            # The shared backoff function is the worker's policy — full jitter
            # over a doubling window, which is right when nobody is watching
            # and much too long when somebody is. Check the room before
            # sleeping, and spend at most what is left of it.
            room_ms = RETRY_LATENCY_CAP_MS - retry_ms
            if room_ms < 50 or budget.hard_expired:
                break
            delay = min(_backoff(error_class, attempt), room_ms / 1000.0)

            entry = {
                "at": datetime.now(UTC).isoformat(),
                "error_class": error_class,
                "google_status": _status_code(failure),
                "backoff_ms": int(delay * 1000),
            }
            outcome.retries.append(entry)
            await hooks.on_retry(outcome, {**entry, "attempt": attempt, "of": max_attempts})
            retry_ms += delay * 1000
            # The semaphore is already released: holding a service slot while
            # doing nothing is how one backoff becomes everybody's queue.
            await asyncio.sleep(delay)

        if failure is not None:
            error_class = _classify(failure)
            status = "timeout" if isinstance(failure, (asyncio.TimeoutError, TimeoutError)) else "failed"
            await finish(outcome, status, outcome=_outcome_of(failure, error_class))
            if error_class == "INVALID":
                # The op refused the arguments the plan wrote. That is OUR
                # mistake, and the planner can fix it if it hears what the op
                # said — so ask for the repair round instead of shipping a
                # degraded answer about a step we authored wrong. The round
                # cap and the pause/replan precedence both still apply.
                message = getattr(failure, "message", None) or str(failure)
                replan.append(f"{node_id} ({op_name}) rejected its arguments: {message}")
            log.warning(
                "dispatch.step_failed",
                node_id=node_id,
                op=op_name,
                error_class=error_class,
                attempts=outcome.attempts,
                # Without these two a failed step is a dead end in the log: the
                # class alone does not say which argument or which call broke.
                error=str(failure)[:500],
                error_type=type(failure).__name__,
            )
            return

        data = dict(getattr(op_result, "data", None) or {})

        # A question is a step. Raising it pauses the run, and answering it
        # resumes from here at zero model calls.
        request = getattr(op_result, "needs_input", None)
        if request is not None:
            input_id = None
            if hooks.raise_input is not None:
                input_id = await hooks.raise_input(step, request)
            blocking = bool(getattr(request, "blocking", True))
            paused.append(PausedOn(node_id, op_name, input_id, request, blocking))
            if blocking:
                stop.set()
                outcome.status = "running"
                outcome.result = {"input_id": input_id, "awaiting": True}
                done[node_id].set()
                await hooks.on_step_finished(outcome)
                return
            data.setdefault("input_id", input_id)

        if getattr(op_result, "needs_replan", False):
            replan.append(str(getattr(op_result, "replan_reason", "") or "a step asked to replan"))

        # A confirmable op has done the reversible half of its work — a Gmail
        # draft exists, and the user can see it. The irreversible half waits
        # for a person.
        if getattr(op, "needs_confirm", False) and hooks.stage_write is not None:
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else bound
            preview = data.get("preview")
            if preview is None and hasattr(op, "preview"):
                try:
                    preview = op.preview(payload)
                except Exception:  # noqa: BLE001 - a preview must never fail a step
                    preview = None
            question = data.get("question")
            if question is None and hasattr(op, "confirm_question"):
                try:
                    question = op.confirm_question(payload)
                except Exception:  # noqa: BLE001
                    question = None
            # A staged write whose step depends on another staged write must
            # not be executed until that one is done. The dependency edge in
            # the plan is the ordering ("move the meeting, then tell the
            # guests"), and the actions worker enforces it by refusing to run
            # while a predecessor is unfinished — but only if it is told who
            # the predecessor is, which is what this writes down. Without it
            # the guests get told about a move that failed.
            after = [
                staged_by_node[dep]
                for dep in (step.get("depends_on") or [])
                if dep in staged_by_node
            ]
            if after:
                payload = {**payload, SEQUENCE_KEY: after}

            action_id = await hooks.stage_write(
                step,
                {
                    "op": op_name,
                    "payload": payload,
                    "preview": preview,
                    "question": question,
                    "external_ref": data.get("external_ref") or data.get("draft_id"),
                },
            )
            if action_id:
                data.setdefault("action_id", action_id)
                staged_by_node[node_id] = action_id
                staged.append(
                    {
                        "action_id": action_id,
                        "node_id": node_id,
                        "op": op_name,
                        "preview": preview,
                        "question": question,
                    }
                )

        results[node_id] = data
        live_scope[node_id] = data
        await finish(outcome, "succeeded", result=data)

    tasks = [asyncio.create_task(run_one(node_id), name=f"step:{node_id}") for node_id in order]
    try:
        finished, pending = await asyncio.wait(tasks, timeout=budget.remaining_hard_s() or None)
        if pending:
            # The hard deadline. Everything still running is stopped, and every
            # one of them says so in the trace rather than vanishing.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for node_id in order:
                outcome = outcomes.get(node_id)
                if outcome is None or outcome.status in ("pending", "running"):
                    outcome = record(node_id, str(by_id[node_id].get("op") or ""))
                    if outcome.status not in ("succeeded", "failed", "skipped"):
                        outcome.status = "timeout"
                        outcome.finished_at = datetime.now(UTC)
                        outcome.outcome = {
                            "reason": "hard_deadline",
                            "class": "TRANSIENT",
                            "message": "the run hit its hard deadline and was stopped",
                        }
                        await hooks.on_step_finished(outcome)
        for task in finished:
            exc = task.exception()
            if exc is not None:  # a bug in the dispatcher, not in a step
                log.error("dispatch.task_crashed", error=str(exc))
    finally:
        budget.pause()

    blocking_pause = next((p for p in paused if p.blocking), None)
    if blocking_pause is not None:
        status = "paused"
    elif replan:
        status = "replan"
    elif all(o.status in ("failed", "timeout") for o in outcomes.values()) and outcomes:
        status = "failed"
    else:
        status = "complete"

    # Everything parked behind the question, transitively. A node skipped
    # because the run stopped, and anything that then skipped because *it*
    # produced nothing, is waiting — not broken.
    awaiting: set[str] = set()
    if blocking_pause is not None:
        awaiting = {p.node_id for p in paused}
        while True:
            grew = False
            for o in outcomes.values():
                if o.status != "skipped" or o.node_id in awaiting:
                    continue
                detail = o.outcome or {}
                if detail.get("reason") == "run_paused" or detail.get("depends_on") in awaiting:
                    awaiting.add(o.node_id)
                    grew = True
            if not grew:
                break

    degraded = []
    for o in outcomes.values():
        if o.status not in ("failed", "timeout", "skipped"):
            continue
        if o.node_id in awaiting:
            continue
        fault = _fault_of(o)
        detail = o.outcome or {}
        degraded.append(
            {
                # Only a real service failure names a service. A plan fault
                # leaves it None so the renderer cannot blame Google for it.
                "service": o.op.split(".", 1)[0] if fault == "service" else None,
                "fault": fault,
                "node": o.node_id,
                "op": o.op,
                "reason": detail.get("reason", "failed"),
                "class": detail.get("class"),
                "code": detail.get("code"),
                "detail": detail.get("message"),
            }
        )

    return DispatchResult(
        status=status,
        results=results,
        outcomes=outcomes,
        staged=staged,
        degraded=degraded,
        awaiting=frozenset(awaiting),
        paused_on=blocking_pause or (paused[0] if paused else None),
        replan_reason=replan[0] if replan else None,
        took_ms=int((time.monotonic() - started) * 1000),
    )


__all__ = [
    "HARD_DEADLINE_MS",
    "MAX_ATTEMPTS",
    "MAX_REPLAN_ROUNDS",
    "RETRYABLE_CLASSES",
    "RETRY_LATENCY_CAP_MS",
    "SEMAPHORE_LIMITS",
    "SOFT_DEADLINE_MS",
    "BindContext",
    "BindError",
    "Budget",
    "DispatchResult",
    "Hooks",
    "PausedOn",
    "StepOutcome",
    "bind",
    "bind_args",
    "bind_references",
    "dispatch",
    "evaluate_gate",
    "json_safe",
    "make_semaphores",
    "resolve_references",
]
