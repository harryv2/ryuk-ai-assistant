"""Shared fixtures and small adapters for the whole test suite.

Three things live here.

**A frozen clock.** Every temporal test anchors on the same instant the
documentation uses: Thursday 20 August 2026, 13:12:04 UTC — ISO week 34. A test
that reads the wall clock is a test that fails on a Sunday, so nothing here ever
does.

**Fake users, ops and plans.** A user in Asia/Kolkata (a half-hour offset, which
catches a whole class of timezone shortcuts), a small op registry with real
Pydantic argument models, and factory helpers for plans and steps.

**Adapters.** The modules under test are being written in parallel against
`docs/contracts.md`. The contract fixes module paths and behaviour; it does not
fix every helper's exact name or parameter list. `load_any` and `call` bridge
that last inch: they find the function and pass only the arguments it actually
declares. They never soften an assertion — every test still has to get the right
answer, and a function that cannot be found fails loudly with the list of names
that were tried.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# The frozen clock
# --------------------------------------------------------------------------

# Thursday 20 August 2026, ISO week 34. The same instant docs/SAMPLE_QUERIES.md
# uses for every worked example, so a test and the document can be read side by
# side.
FROZEN_NOW = datetime(2026, 8, 20, 13, 12, 4, tzinfo=UTC)

TZ_UTC = "UTC"
TZ_NY = "America/New_York"
TZ_KOLKATA = "Asia/Kolkata"
TZ_LONDON = "Europe/London"

MONDAY = 1
SUNDAY = 7

# Thresholds. Repeated here as literals on purpose: if someone edits the
# defaults in config.py, these tests should notice rather than quietly follow.
FLOOR_READ = 0.55
MARGIN = 0.15
FLOOR_WRITE = 0.80


def at(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    tz: str = TZ_UTC,
) -> datetime:
    """A tz-aware local wall time. Never build a naive datetime in a test."""
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz))


def local(moment: datetime, tz: str) -> datetime:
    """The same instant, read in `tz`."""
    return moment.astimezone(ZoneInfo(tz))


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

_MISSING = object()


def load_any(module_paths: str | list[str], names: str | list[str]) -> Any:
    """Find the first of `names` on the first of `module_paths` that has it.

    The contract fixes where a thing lives and what it does. Where a helper's
    exact name is still settling between parallel modules, list the candidates
    in the order the contract implies. Failure is loud and names everything that
    was tried, so a genuine miss reads as a miss and not as a skipped test.
    """
    if isinstance(module_paths, str):
        module_paths = [module_paths]
    if isinstance(names, str):
        names = [names]

    tried: list[str] = []
    absent: list[str] = []  # the module itself is not written yet
    broken: list[str] = []  # the module is there and would not import
    modules = []

    for path in module_paths:
        try:
            modules.append((path, importlib.import_module(path)))
        except ModuleNotFoundError as exc:
            if exc.name and (path == exc.name or path.startswith(exc.name + ".")):
                absent.append(path)
            else:
                broken.append(f"{path}: {exc}")
        except ImportError as exc:
            broken.append(f"{path}: {exc}")

    for path, module in modules:
        for name in names:
            tried.append(f"{path}.{name}")
            found = getattr(module, name, _MISSING)
            if found is not _MISSING:
                return found

    if not modules and absent and not broken:
        # Not written yet. Say so and move on — a build-order fact, not a
        # defect. `-ra` in the pytest options prints the reason, so nothing
        # disappears quietly.
        pytest.skip(
            "not implemented yet: " + ", ".join(absent),
            allow_module_level=True,
        )

    raise AttributeError(
        "could not find any of "
        + ", ".join(tried or names)
        + (("; import failures: " + " | ".join(broken)) if broken else "")
        + (("; missing modules: " + ", ".join(absent)) if absent else "")
    )


def sync(value: Any) -> Any:
    """Run a coroutine to completion if that is what came back, else pass it through.

    Some of these helpers may end up async because they read the mirror. The
    unit tests do not care either way.
    """
    if inspect.isawaitable(value):
        return asyncio.run(_await(value))
    return value


async def _await(value: Any) -> Any:
    return await value


def call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call `fn` with only the arguments its signature declares.

    Extra keywords are dropped rather than raising, which lets one test cover a
    function whether or not it takes, say, an injected registry. Positional
    arguments are trimmed the same way. The values that matter to the assertion
    are always passed positionally, so nothing important can be silently lost.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return sync(fn(*args, **kwargs))

    takes_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    takes_var_pos = any(p.kind is p.VAR_POSITIONAL for p in params.values())

    if not takes_var_kw:
        allowed = {
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    if not takes_var_pos:
        positional_slots = sum(
            1
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and name not in kwargs
        )
        args = args[:positional_slots]

    return sync(fn(*args, **kwargs))


def get(obj: Any, name: str, default: Any = _MISSING) -> Any:
    """Read `name` off an object or a dict, whichever it turned out to be."""
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
    else:
        found = getattr(obj, name, _MISSING)
        if found is not _MISSING:
            return found
    if default is _MISSING:
        raise AssertionError(f"{obj!r} has no {name!r}")
    return default


def text_of(obj: Any) -> str:
    """Everything an object says about itself, lower-cased, for keyword checks."""
    parts = [str(obj)]
    for name in ("code", "message", "reason", "detail", "details", "errors", "error", "kind"):
        value = get(obj, name, None)
        if value is not None:
            parts.append(json.dumps(value, default=str))
    return " ".join(parts).lower()


def mentions(obj: Any, *keywords: str) -> bool:
    """True when anything the object says contains one of `keywords`."""
    blob = text_of(obj)
    return any(k.lower() in blob for k in keywords)


# --------------------------------------------------------------------------
# Rejection and acceptance helpers, used by the plan validation tests
# --------------------------------------------------------------------------


def rejected(fn: Any, *args: Any, because: tuple[str, ...] = (), **kwargs: Any) -> Any:
    """Assert the call rejects, and that it says why.

    A validator may raise or may hand back a result carrying errors; both are
    reasonable and both are accepted. What is not accepted is a rejection that
    does not name its reason — `because` must show up in the message, so this
    still pins down *which* rule fired and not merely that something went wrong.
    """
    try:
        result = call(fn, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any failure shape is a rejection
        assert not because or mentions(exc, *because), (
            f"rejected, but the reason does not mention any of {because}: {exc!r}"
        )
        return exc

    ok = get(result, "ok", None)
    errors = get(result, "errors", None)
    assert ok is False or errors, f"expected a rejection, got {result!r}"
    assert not because or mentions(result, *because), (
        f"rejected, but the reason does not mention any of {because}: {result!r}"
    )
    return result


def accepted(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Assert the call is accepted, and hand back whatever it produced."""
    result = call(fn, *args, **kwargs)
    ok = get(result, "ok", None)
    errors = get(result, "errors", None)
    assert ok is not False, f"expected acceptance, got {result!r}"
    assert not errors, f"expected acceptance, got errors {errors!r}"
    return result


# --------------------------------------------------------------------------
# The fake user
# --------------------------------------------------------------------------


@dataclass
class FakeUser:
    id: str = "u_7QkR2mXvB4nLd9TsW"
    email: str = "priya@alpha-law.test"
    display_name: str = "Priya Raman"
    timezone: str = TZ_KOLKATA
    work_week_start: int = MONDAY


@pytest.fixture
def now() -> datetime:
    """The frozen instant. Thursday 20 August 2026, ISO week 34."""
    return FROZEN_NOW


@pytest.fixture
def user() -> FakeUser:
    """Asia/Kolkata, Monday week start. The half-hour offset is the point."""
    return FakeUser()


@pytest.fixture
def conversation_id() -> str:
    return "cnv_8Ln5TqWm3XdRb7ZkV"


@pytest.fixture
def run_id() -> str:
    return "run_3Fd9RbXn2QsLt6WkJ"


# --------------------------------------------------------------------------
# A fake op registry
# --------------------------------------------------------------------------
#
# Real Pydantic models, because the validator's second job is checking a step's
# args against the op's schema and a hand-rolled stub would not exercise that.


class SearchEmailsArgs(BaseModel):
    query: str
    window: dict | None = None
    from_email: str | None = None
    limit: int = 10


class GetEmailArgs(BaseModel):
    message_id: str


class SendEmailArgs(BaseModel):
    to: list[str] = Field(min_length=1)
    subject: str
    body: str


class DraftEmailArgs(BaseModel):
    to: list[str] = Field(min_length=1)
    subject: str
    body: str


class SearchEventsArgs(BaseModel):
    window: dict
    attendee_emails_any: list[str] | None = None
    status_in: list[str] | None = None
    order_by: str = "starts_at"


class GetEventArgs(BaseModel):
    event_id: str


class UpdateEventArgs(BaseModel):
    event_id: str
    etag: str | None = None
    starts_at: str
    duration_minutes: int | None = None
    send_updates: str = "all"


class SearchFilesArgs(BaseModel):
    query: str
    mime_type: str | None = None
    window: dict | None = None


class AskUserArgs(BaseModel):
    kind: str
    question: str
    value_schema: dict
    help_text: str | None = None
    fields: list[dict] | None = None
    options: list[dict] | None = None
    blocking: bool = True


class SummarizeArgs(BaseModel):
    items: list = Field(default_factory=list)
    style: str = "bullets"


@dataclass
class FakeOp:
    """Everything the validator and the binder read off an op.

    Mirrors `app.ops.base.Op` in `docs/contracts.md`: the argument model, the
    fields a `{{step.x}}` reference may name, and the three flags that decide
    whether a step may run at all.
    """

    name: str
    args_model: type[BaseModel]
    output_fields: list[str]
    is_local: bool = False
    is_write: bool = False
    needs_confirm: bool = False
    timeout_s: float = 6.0
    max_attempts: int = 2

    @property
    def service(self) -> str:
        return self.name.split(".", 1)[0]

    def catalogue_line(self) -> str:
        return f"{self.name}({', '.join(self.args_model.model_fields)}) -> {self.output_fields}"

    def progress_label(self, args: dict) -> str:
        return f"running {self.name}"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        return data


def _build_registry() -> dict[str, FakeOp]:
    ops = [
        FakeOp("gmail.search_emails", SearchEmailsArgs, ["hits", "count"], is_local=True),
        FakeOp(
            "gmail.get_email",
            GetEmailArgs,
            ["message_id", "thread_id", "subject", "from_email", "body", "extracted"],
        ),
        FakeOp(
            "gmail.send_email",
            SendEmailArgs,
            ["message_id", "thread_id"],
            is_write=True,
            needs_confirm=True,
        ),
        FakeOp(
            "gmail.draft_email",
            DraftEmailArgs,
            ["draft_id", "message_id"],
            is_write=True,
            needs_confirm=True,
        ),
        FakeOp("gcal.search_events", SearchEventsArgs, ["hits", "count"], is_local=True),
        FakeOp(
            "gcal.get_event",
            GetEventArgs,
            ["event_id", "etag", "title", "starts_at", "ends_at", "duration_minutes", "attendees"],
        ),
        FakeOp(
            "gcal.update_event",
            UpdateEventArgs,
            ["event_id", "etag", "starts_at", "ends_at"],
            is_write=True,
            needs_confirm=True,
        ),
        FakeOp("gdrive.search_files", SearchFilesArgs, ["hits", "count"], is_local=True),
        FakeOp("ask.user", AskUserArgs, ["value"], is_local=True),
        FakeOp("meta.summarize", SummarizeArgs, ["text"], is_local=True),
    ]
    return {op.name: op for op in ops}


@pytest.fixture
def op_registry() -> dict[str, FakeOp]:
    """A small registry shaped like `app.ops.registry.REGISTRY`."""
    return _build_registry()


# --------------------------------------------------------------------------
# Plan and step factories
# --------------------------------------------------------------------------


def make_step(
    step_id: str,
    op: str,
    args: dict | None = None,
    depends_on: list[str] | None = None,
    **overrides: Any,
) -> dict:
    """One step of a plan, with every field the contract declares filled in."""
    step = {
        "id": step_id,
        "op": op,
        "args": args if args is not None else {},
        "depends_on": depends_on or [],
        "expect": "many",
        "optional": False,
        "freshness": "cached",
        "speculate": False,
    }
    step.update(overrides)
    return step


def make_plan(
    steps: list[dict],
    *,
    name: str = "calendar_list",
    services: list[str] | None = None,
    has_write: bool = False,
    confidence: float = 0.92,
    answer_style: str = "template:event_list",
    **overrides: Any,
) -> dict:
    """A whole plan, in the exact shape the planner returns."""
    if services is None:
        services = sorted({s["op"].split(".", 1)[0] for s in steps} - {"ask", "meta"}) or ["gcal"]
    plan = {
        "type": "plan",
        "intent": {
            "name": name,
            "services": services,
            "has_write": has_write,
            "confidence": confidence,
        },
        "answer_style": answer_style,
        "steps": steps,
    }
    plan.update(overrides)
    return plan


@pytest.fixture
def plan_factory():
    return make_plan


@pytest.fixture
def step_factory():
    return make_step


@pytest.fixture
def read_plan() -> dict:
    """The simplest thing that should validate: one local read."""
    return make_plan(
        [
            make_step(
                "events",
                "gcal.search_events",
                {"window": {"start": "{{windows.next_week.start}}", "end": "{{windows.next_week.end}}"}},
            )
        ]
    )


# --------------------------------------------------------------------------
# A realistic Google error
# --------------------------------------------------------------------------

GMAIL_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"


def _response(status: int, body: Any, headers: dict[str, str] | None = None):
    import httpx

    request = httpx.Request("GET", GMAIL_URL)
    return httpx.Response(status, json=body, headers=headers or {}, request=request)


def _as_error(response, service: str, method: str) -> Exception:
    """Turn a response into whatever the Google path raises for it."""
    try:
        from app.google.retry import GoogleAPIError

        return GoogleAPIError.from_response(response, service=service, method=method)
    except ImportError:  # pragma: no cover - only before google/retry.py lands
        import httpx

        return httpx.HTTPStatusError(
            f"{response.status_code}", request=response.request, response=response
        )


def google_error(
    status: int,
    reason: str | None = None,
    message: str | None = None,
    *,
    headers: dict[str, str] | None = None,
    domain: str = "usageLimits",
    service: str = "gmail",
    method: str = "users.messages.list",
) -> Exception:
    """A Google API error built from a real error envelope.

    The shape straight off the wire::

        {"error": {"code": 403, "message": "...", "status": "PERMISSION_DENIED",
                   "errors": [{"domain": "usageLimits", "reason": "rateLimitExceeded",
                               "message": "..."}]}}

    The response goes through the same parser the live client uses, so a test
    exercises the envelope reading as well as the classification.
    """
    body: dict[str, Any] = {
        "error": {
            "code": status,
            "message": message or reason or "request failed",
            "status": _STATUS_NAMES.get(status, "UNKNOWN"),
        }
    }
    if reason:
        body["error"]["errors"] = [
            {"domain": domain, "reason": reason, "message": message or reason}
        ]
        body["error"]["status"] = _REASON_STATUS.get(reason, body["error"]["status"])
    return _as_error(_response(status, body, headers), service, method)


def oauth_error(
    error: str = "invalid_grant",
    description: str = "Token has been expired or revoked.",
    status: int = 400,
) -> Exception:
    """The flatter envelope Google's OAuth endpoints use.

    ``{"error": "invalid_grant", "error_description": "..."}`` — this is the one
    that separates a stale access token from a grant the user has taken away.
    """
    body = {"error": error, "error_description": description}
    return _as_error(_response(status, body), "oauth", "token.refresh")


_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    412: "FAILED_PRECONDITION",
    422: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    502: "BAD_GATEWAY",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}

_REASON_STATUS = {
    "rateLimitExceeded": "RESOURCE_EXHAUSTED",
    "userRateLimitExceeded": "RESOURCE_EXHAUSTED",
    "dailyLimitExceeded": "RESOURCE_EXHAUSTED",
    "quotaExceeded": "RESOURCE_EXHAUSTED",
    "authError": "UNAUTHENTICATED",
    "insufficientPermissions": "PERMISSION_DENIED",
    "conditionNotMet": "FAILED_PRECONDITION",
    "notFound": "NOT_FOUND",
    "badRequest": "INVALID_ARGUMENT",
}


@pytest.fixture
def make_google_error():
    return google_error


@pytest.fixture
def make_oauth_error():
    return oauth_error


# --------------------------------------------------------------------------
# Encryption keys
# --------------------------------------------------------------------------

KEY_A = "3q2+796tvu/erb7v3q2+796tvu/erb7v3q2+796tvu8="  # 32 bytes, base64
KEY_B = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="  # 32 different bytes


@pytest.fixture
def crypto_keys():
    """Give `app.core.crypto` a known key, and put the settings back afterwards.

    Yields a small object with `use(key, version, retired)` so a test can rotate
    keys mid-way and check that an old blob still opens — or, with the wrong
    key, that it does not.

    Values are written straight into the settings object's `__dict__`. Some of
    these names are read with `getattr(..., default)` rather than declared as
    fields, and a normal `setattr` on a Pydantic model refuses anything it does
    not know about.
    """
    from app.config import settings
    from app.core import crypto

    originals: dict[str, tuple[bool, Any]] = {}

    def put(name: str, value: Any) -> None:
        if name not in originals:
            originals[name] = (name in settings.__dict__, settings.__dict__.get(name))
        settings.__dict__[name] = value

    class Keys:
        def use(self, key: str, version: int = 1, retired: dict[int, str] | None = None) -> None:
            put("TOKEN_ENCRYPTION_KEY", key)
            put("TOKEN_ENCRYPTION_KEY_VERSION", version)
            put("TOKEN_KEY_VERSION", version)
            put(
                "TOKEN_ENCRYPTION_KEYS",
                json.dumps({str(v): k for v, k in (retired or {}).items()}),
            )
            crypto.reload_keys()

    keys = Keys()
    keys.use(KEY_A, 1)
    yield keys

    for name, (existed, old) in originals.items():
        if existed:
            settings.__dict__[name] = old
        else:
            settings.__dict__.pop(name, None)
    crypto.reload_keys()


# --------------------------------------------------------------------------
# Session-wide safety net
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _no_real_credentials():
    """Make sure a stray unit test cannot reach a real service.

    Placeholders, not blanks: a blank key sometimes means "read the default from
    somewhere else", and a unit test should never get that far anyway.
    """
    os.environ.setdefault("ENV", "test")
    os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")
    os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
    os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
    os.environ.setdefault("SESSION_SECRET", "test-session-secret")
    os.environ.setdefault("TOKEN_ENCRYPTION_KEY", KEY_A)
    yield


__all__ = [
    "FROZEN_NOW",
    "TZ_UTC",
    "TZ_NY",
    "TZ_KOLKATA",
    "TZ_LONDON",
    "MONDAY",
    "SUNDAY",
    "FLOOR_READ",
    "MARGIN",
    "FLOOR_WRITE",
    "FakeOp",
    "FakeUser",
    "accepted",
    "at",
    "call",
    "get",
    "google_error",
    "load_any",
    "local",
    "oauth_error",
    "make_plan",
    "make_step",
    "mentions",
    "rejected",
    "sync",
    "text_of",
]
