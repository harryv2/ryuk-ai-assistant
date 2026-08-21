"""Failure classes, backoff, and the per user-and-service circuit breaker.

Google fails in ways that want different answers. A 503 wants another go in a
moment. A 429 wants a longer wait, and it usually tells you how long. A 401
wants a fresh access token and exactly one more try. A 412 means the etag we
held is stale, so the only useful move is to refetch the thing once. A daily
quota is over is not a retry at all — it is tomorrow.

So every failure is put in one class first, and the class decides everything
after: whether to try again, how many times, and how long to wait.

Waiting uses **full jitter** — ``random.uniform(0, min(base * 2**attempt, cap))``.
Not "backoff plus a bit of noise": the whole delay is random inside a growing
window. That is what stops a hundred tasks that failed on the same second from
retrying on the same second.

The circuit breaker sits one level up. Backoff protects a single call; the
breaker protects the user from a service that is simply down. Five consecutive
failures against one service for one person opens it for five minutes. Every
further failure doubles that, up to thirty minutes. When the open period ends
exactly one call is let through as a probe: it closes the breaker if it works
and reopens, longer, if it does not.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, Final, TypeVar

import httpx
from redis.exceptions import RedisError

from app.core import cache
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class ErrorClass(StrEnum):
    """The nine ways a Google call can go wrong."""

    TRANSIENT = "TRANSIENT"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_REVOKED = "AUTH_REVOKED"
    PRECONDITION = "PRECONDITION"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Policy:
    """What one error class is allowed to do.

    ``base`` and ``cap`` are seconds. ``max_attempts`` counts *retries*, not
    total calls: 5 means the first call plus five more.
    """

    retryable: bool
    base: float
    cap: float
    max_attempts: int
    #: refetch the resource before retrying — only PRECONDITION does this
    refetch: bool = False


POLICY: Final[dict[ErrorClass, Policy]] = {
    ErrorClass.TRANSIENT: Policy(True, base=1.0, cap=60.0, max_attempts=5),
    ErrorClass.RATE_LIMITED: Policy(True, base=5.0, cap=300.0, max_attempts=8),
    # A dead access token is fixed by refreshing it, not by waiting.
    ErrorClass.AUTH_EXPIRED: Policy(True, base=0.0, cap=0.0, max_attempts=1),
    # The etag we sent is stale. Read the current version, then try once more.
    ErrorClass.PRECONDITION: Policy(True, base=0.0, cap=0.0, max_attempts=1, refetch=True),
    ErrorClass.QUOTA_EXHAUSTED: Policy(False, base=0.0, cap=0.0, max_attempts=0),
    ErrorClass.AUTH_REVOKED: Policy(False, base=0.0, cap=0.0, max_attempts=0),
    ErrorClass.NOT_FOUND: Policy(False, base=0.0, cap=0.0, max_attempts=0),
    ErrorClass.INVALID: Policy(False, base=0.0, cap=0.0, max_attempts=0),
    ErrorClass.UNKNOWN: Policy(False, base=0.0, cap=0.0, max_attempts=0),
}

#: Failure classes that say something about the *service*, so they move the
#: circuit breaker. A 404 or a bad argument is about the request, not Google.
BREAKER_CLASSES: Final[frozenset[ErrorClass]] = frozenset(
    {
        ErrorClass.TRANSIENT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.QUOTA_EXHAUSTED,
        ErrorClass.UNKNOWN,
    }
)

#: How each class surfaces at the API boundary.
APP_ERROR_CODE: Final[dict[ErrorClass, str]] = {
    ErrorClass.TRANSIENT: "GOOGLE_UNAVAILABLE",
    ErrorClass.RATE_LIMITED: "RATE_LIMITED",
    ErrorClass.QUOTA_EXHAUSTED: "GOOGLE_UNAVAILABLE",
    ErrorClass.AUTH_EXPIRED: "GOOGLE_REAUTH_REQUIRED",
    ErrorClass.AUTH_REVOKED: "GOOGLE_REAUTH_REQUIRED",
    ErrorClass.PRECONDITION: "GOOGLE_UNAVAILABLE",
    ErrorClass.NOT_FOUND: "NOT_FOUND",
    ErrorClass.INVALID: "VALIDATION_ERROR",
    ErrorClass.UNKNOWN: "GOOGLE_UNAVAILABLE",
}


# --------------------------------------------------------------------------- #
# The error we raise for a non-2xx Google response
# --------------------------------------------------------------------------- #


class GoogleAPIError(Exception):
    """A Google response we could not use.

    Carries the status, the machine-readable ``errors[].reason`` from Google's
    envelope, and the ``Retry-After`` value when there was one — which is
    everything :func:`classify` and :func:`backoff` need.
    """

    def __init__(
        self,
        status: int,
        *,
        reason: str | None = None,
        message: str | None = None,
        service: str | None = None,
        method: str | None = None,
        url: str | None = None,
        retry_after: float | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = int(status)
        self.reason = reason or ""
        self.message = message or f"Google returned {status}."
        self.service = service
        self.method = method
        self.url = url
        self.retry_after = retry_after
        self.body = body
        self.headers = headers or {}
        super().__init__(f"{status} {self.reason or ''} {self.message}".strip())

    # -- derived ------------------------------------------------------------ #

    @property
    def error_class(self) -> ErrorClass:
        return classify(self)

    def to_app_error(self) -> AppError:
        cls = self.error_class
        return AppError(
            APP_ERROR_CODE[cls],
            details={
                "service": self.service,
                "method": self.method,
                "google_status": self.status,
                "google_reason": self.reason,
                "error_class": str(cls),
                "detail": self.message[:400],
            },
        )

    def as_outcome(self) -> dict[str, Any]:
        """The ``node_executions.outcome`` shape."""
        return {
            "reason": self.reason or "google_error",
            "class": str(self.error_class),
            "code": self.status,
            "message": self.message[:400],
        }

    @classmethod
    def from_response(
        cls,
        response: httpx.Response,
        *,
        service: str | None = None,
        method: str | None = None,
    ) -> "GoogleAPIError":
        """Read Google's error envelope off a response.

        The envelope is ``{"error": {"code", "message", "status", "errors":
        [{"domain", "reason", "message"}]}}``. The OAuth endpoints use a flatter
        ``{"error": "invalid_grant", "error_description": "..."}``, which is
        also handled — that one is the difference between AUTH_EXPIRED and
        AUTH_REVOKED.
        """
        reason = ""
        message = ""
        body: Any = None
        try:
            body = response.json()
        except Exception:  # not JSON: keep the text, such as it is
            body = (response.text or "")[:500]

        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or "")
                errors = err.get("errors")
                if isinstance(errors, list) and errors:
                    first = errors[0]
                    if isinstance(first, dict):
                        reason = str(first.get("reason") or "")
                        message = message or str(first.get("message") or "")
                if not reason:
                    reason = str(err.get("status") or "")
            elif isinstance(err, str):
                # OAuth style
                reason = err
                message = str(body.get("error_description") or err)

        return cls(
            response.status_code,
            reason=reason,
            message=message or f"Google returned {response.status_code}.",
            service=service,
            method=method,
            url=str(response.request.url) if response.request is not None else None,
            retry_after=parse_retry_after(response.headers.get("Retry-After")),
            body=body,
            headers=dict(response.headers),
        )


class CircuitOpen(AppError):
    """Raised instead of calling Google while the breaker is open."""

    def __init__(self, user_id: str, service: str, retry_after_s: float) -> None:
        super().__init__(
            "GOOGLE_UNAVAILABLE",
            f"{_SERVICE_LABEL.get(service, service)} has been failing, so calls are "
            "paused for a moment.",
            details={
                "service": service,
                "error_class": str(ErrorClass.TRANSIENT),
                "retry_after_s": round(retry_after_s, 1),
                "circuit": "open",
            },
        )
        self.service = service
        self.user_id = user_id
        self.retry_after_s = retry_after_s


_SERVICE_LABEL: Final[dict[str, str]] = {
    "gmail": "Gmail",
    "gcal": "Google Calendar",
    "gdrive": "Google Drive",
}


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

# Google's `errors[].reason`, lower-cased. The reason is more precise than the
# status: a 429 can be a per-second rate limit (wait) or a daily quota (stop),
# and only the reason tells them apart.
_REASON_CLASS: Final[dict[str, ErrorClass]] = {
    # slow down
    "ratelimitexceeded": ErrorClass.RATE_LIMITED,
    "userratelimitexceeded": ErrorClass.RATE_LIMITED,
    "sharingratelimitexceeded": ErrorClass.RATE_LIMITED,
    "concurrentlimitexceeded": ErrorClass.RATE_LIMITED,
    "resource_exhausted": ErrorClass.RATE_LIMITED,
    # come back tomorrow
    "dailylimitexceeded": ErrorClass.QUOTA_EXHAUSTED,
    "dailylimitexceededunreg": ErrorClass.QUOTA_EXHAUSTED,
    "quotaexceeded": ErrorClass.QUOTA_EXHAUSTED,
    "limitexceeded": ErrorClass.QUOTA_EXHAUSTED,
    "storagequotaexceeded": ErrorClass.QUOTA_EXHAUSTED,
    "usagelimits": ErrorClass.QUOTA_EXHAUSTED,
    # the access token is stale
    "autherror": ErrorClass.AUTH_EXPIRED,
    "authenticationfailure": ErrorClass.AUTH_EXPIRED,
    "invalid_token": ErrorClass.AUTH_EXPIRED,
    "invalidcredentials": ErrorClass.AUTH_EXPIRED,
    "unauthenticated": ErrorClass.AUTH_EXPIRED,
    "expired": ErrorClass.AUTH_EXPIRED,
    # the grant itself is gone — refreshing will not help
    "invalid_grant": ErrorClass.AUTH_REVOKED,
    "unauthorized_client": ErrorClass.AUTH_REVOKED,
    "insufficientpermissions": ErrorClass.AUTH_REVOKED,
    "insufficientfilepermissions": ErrorClass.AUTH_REVOKED,
    "appnotauthorizedtofile": ErrorClass.AUTH_REVOKED,
    "accessnotconfigured": ErrorClass.AUTH_REVOKED,
    "domainpolicy": ErrorClass.AUTH_REVOKED,
    "forbidden": ErrorClass.AUTH_REVOKED,
    "permission_denied": ErrorClass.AUTH_REVOKED,
    # what we held is out of date
    "conditionnotmet": ErrorClass.PRECONDITION,
    "preconditionfailed": ErrorClass.PRECONDITION,
    "failedprecondition": ErrorClass.PRECONDITION,
    "failed_precondition": ErrorClass.PRECONDITION,
    "fullsyncrequired": ErrorClass.PRECONDITION,
    "etagmismatch": ErrorClass.PRECONDITION,
    # gone
    "notfound": ErrorClass.NOT_FOUND,
    "filenotfound": ErrorClass.NOT_FOUND,
    "not_found": ErrorClass.NOT_FOUND,
    # our fault
    "invalid": ErrorClass.INVALID,
    "invalidargument": ErrorClass.INVALID,
    "invalid_argument": ErrorClass.INVALID,
    "invalidparameter": ErrorClass.INVALID,
    "invalidquery": ErrorClass.INVALID,
    "badrequest": ErrorClass.INVALID,
    "parseerror": ErrorClass.INVALID,
    "required": ErrorClass.INVALID,
    "invalidsharingrequest": ErrorClass.INVALID,
    # theirs, and temporary
    "backenderror": ErrorClass.TRANSIENT,
    "internalerror": ErrorClass.TRANSIENT,
    "internal": ErrorClass.TRANSIENT,
    "transienterror": ErrorClass.TRANSIENT,
    "serviceunavailable": ErrorClass.TRANSIENT,
    "unavailable": ErrorClass.TRANSIENT,
    "deadlineexceeded": ErrorClass.TRANSIENT,
    "deadline_exceeded": ErrorClass.TRANSIENT,
    "aborted": ErrorClass.TRANSIENT,
}

_STATUS_CLASS: Final[dict[int, ErrorClass]] = {
    400: ErrorClass.INVALID,
    401: ErrorClass.AUTH_EXPIRED,
    403: ErrorClass.AUTH_REVOKED,
    404: ErrorClass.NOT_FOUND,
    405: ErrorClass.INVALID,
    409: ErrorClass.PRECONDITION,
    410: ErrorClass.PRECONDITION,
    412: ErrorClass.PRECONDITION,
    413: ErrorClass.INVALID,
    416: ErrorClass.INVALID,
    422: ErrorClass.INVALID,
    428: ErrorClass.PRECONDITION,
    429: ErrorClass.RATE_LIMITED,
    500: ErrorClass.TRANSIENT,
    502: ErrorClass.TRANSIENT,
    503: ErrorClass.TRANSIENT,
    504: ErrorClass.TRANSIENT,
}

# AppError codes, for when a wrapped error comes back round.
_APP_CODE_CLASS: Final[dict[str, ErrorClass]] = {
    "RATE_LIMITED": ErrorClass.RATE_LIMITED,
    "GOOGLE_REAUTH_REQUIRED": ErrorClass.AUTH_REVOKED,
    "GOOGLE_UNAVAILABLE": ErrorClass.TRANSIENT,
    "ORCHESTRATION_TIMEOUT": ErrorClass.TRANSIENT,
    "NOT_FOUND": ErrorClass.NOT_FOUND,
    "VALIDATION_ERROR": ErrorClass.INVALID,
}


def classify_status(status: int, reason: str | None = None) -> ErrorClass:
    """Class for a status and an optional Google ``reason``.

    The reason wins where the two disagree — that is the whole reason Google
    sends it.
    """
    if reason:
        hit = _REASON_CLASS.get(reason.strip().lower())
        if hit is not None:
            return hit
    by_status = _STATUS_CLASS.get(int(status))
    if by_status is not None:
        return by_status
    if 500 <= int(status) < 600:
        return ErrorClass.TRANSIENT
    if 400 <= int(status) < 500:
        return ErrorClass.INVALID
    return ErrorClass.UNKNOWN


def classify(exc: Exception) -> ErrorClass:
    """Put any exception from the Google path into one class."""
    if isinstance(exc, CircuitOpen):
        return ErrorClass.TRANSIENT
    if isinstance(exc, GoogleAPIError):
        return classify_status(exc.status, exc.reason)
    if isinstance(exc, AppError):
        return _APP_CODE_CLASS.get(exc.code, ErrorClass.UNKNOWN)
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_status(exc.response.status_code)
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return ErrorClass.TRANSIENT
    if isinstance(exc, httpx.TooManyRedirects):
        return ErrorClass.INVALID
    if isinstance(exc, httpx.InvalidURL):
        return ErrorClass.INVALID
    if isinstance(exc, httpx.HTTPError):
        return ErrorClass.TRANSIENT
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
        return ErrorClass.TRANSIENT
    return ErrorClass.UNKNOWN


def retryable(cls: ErrorClass) -> bool:
    """Is another attempt worth making?"""
    return POLICY[ErrorClass(cls)].retryable


def max_attempts(cls: ErrorClass) -> int:
    """How many retries this class allows."""
    return POLICY[ErrorClass(cls)].max_attempts


def needs_refetch(cls: ErrorClass) -> bool:
    """True when the retry must read the resource again first (etag is stale)."""
    return POLICY[ErrorClass(cls)].refetch


def parse_retry_after(value: str | None) -> float | None:
    """``Retry-After`` as seconds. Accepts a count or an HTTP date."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def backoff(cls: ErrorClass, attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    Full jitter: ``random.uniform(0, min(base * 2**attempt, cap))``. The whole
    delay is random inside a window that doubles, so a herd of tasks that
    failed together does not retry together.

    ``retry_after`` is honoured when Google sent one: the wait is never shorter
    than Google asked for.
    """
    policy = POLICY[ErrorClass(cls)]
    n = max(0, int(attempt))
    if policy.base <= 0:
        delay = 0.0
    else:
        ceiling = min(policy.base * (2**n), policy.cap)
        delay = random.uniform(0, ceiling)
    if retry_after is not None and retry_after > 0:
        # Never sooner than we were told; never longer than this class's cap.
        delay = min(max(delay, float(retry_after)), max(policy.cap, float(retry_after)))
    return round(delay, 3)


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #

BREAKER_THRESHOLD: Final[int] = 5
BREAKER_OPEN_S: Final[float] = 300.0  # first open: five minutes
BREAKER_MAX_OPEN_S: Final[float] = 1800.0  # ceiling: thirty minutes
BREAKER_PROBE_S: Final[float] = 30.0  # how long one probe holds the slot

# Read the state, and claim the half-open probe slot if this caller gets there
# first. One round trip, so two workers cannot both think they are the probe.
_ALLOW_LUA: Final[
    str
] = """
local key       = KEYS[1]
local now       = tonumber(ARGV[1])
local probe_ms  = tonumber(ARGV[2])

local st = redis.call('HMGET', key, 'open_until', 'probe_until', 'failures', 'open_ms')
local open_until  = tonumber(st[1]) or 0
local probe_until = tonumber(st[2]) or 0
local failures    = tonumber(st[3]) or 0
local open_ms     = tonumber(st[4]) or 0

if open_until <= 0 then
  return {1, 'closed', 0, failures}
end

if now < open_until then
  return {0, 'open', open_until - now, failures}
end

-- The open period has elapsed: one caller becomes the probe.
if probe_until > now then
  return {0, 'half_open', probe_until - now, failures}
end
redis.call('HSET', key, 'probe_until', now + probe_ms)
redis.call('PEXPIRE', key, math.ceil(open_ms) + 60000)
return {1, 'probe', 0, failures}
"""

# Count the failure, and open (or reopen, doubled) when it is time.
_FAIL_LUA: Final[
    str
] = """
local key       = KEYS[1]
local now       = tonumber(ARGV[1])
local threshold = tonumber(ARGV[2])
local base_ms   = tonumber(ARGV[3])
local cap_ms    = tonumber(ARGV[4])

local st = redis.call('HMGET', key, 'failures', 'open_ms', 'open_until')
local failures   = (tonumber(st[1]) or 0) + 1
local open_ms    = tonumber(st[2]) or 0
local open_until = tonumber(st[3]) or 0

local opened = 0
if failures >= threshold or open_until > 0 then
  -- Already open once: this was the probe, so double it. Otherwise start.
  if open_ms <= 0 then open_ms = base_ms else open_ms = open_ms * 2 end
  if open_ms > cap_ms then open_ms = cap_ms end
  open_until = now + open_ms
  failures = 0
  opened = 1
end

redis.call('HSET', key, 'failures', failures, 'open_ms', open_ms,
           'open_until', open_until, 'probe_until', 0)
redis.call('PEXPIRE', key, math.ceil(cap_ms) + 60000)
return {opened, failures, open_until, open_ms}
"""

_scripts: dict[str, tuple[Any, Any]] = {}


async def _script(name: str, body: str):
    client = await cache.get_redis()
    cached = _scripts.get(name)
    if cached is None or cached[0] is not client:
        cached = (client, client.register_script(body))
        _scripts[name] = cached
    return cached[1]


def _breaker_key(user_id: str, service: str) -> str:
    return cache.key("cb", f"{user_id}:{service}")


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class BreakerDecision:
    """What the breaker said, and why."""

    allowed: bool
    state: str  # closed | open | half_open | probe | unknown
    retry_after_s: float
    failures: int

    @property
    def is_probe(self) -> bool:
        return self.state == "probe"


def _field(result: Any, index: int, default: Any = None) -> Any:
    """One value out of a Lua reply, or the default if it is not there.

    The breaker reads its own script's output, and reading it must not be able
    to fail: an unexpected reply shape has to end in "allowed", exactly like an
    unreachable Redis does. Indexing blind is how a bookkeeping detail becomes
    a failed send.
    """
    try:
        return result[index]
    except (TypeError, IndexError, KeyError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def breaker_allow(user_id: str, service: str) -> BreakerDecision:
    """May we call this service for this user right now?

    When Redis is unreachable the answer is yes. A breaker that cannot be read
    must not become an outage of its own — which covers a reply this cannot
    make sense of just as much as a socket that is not there.
    """
    try:
        script = await _script("cb_allow", _ALLOW_LUA)
        result = await script(
            keys=[_breaker_key(user_id, service)],
            args=[_now_ms(), int(BREAKER_PROBE_S * 1000)],
        )
    except (RedisError, OSError) as exc:
        log.warning("breaker.unavailable", user_id=user_id, service=service, error=str(exc))
        return BreakerDecision(True, "unknown", 0.0, 0)

    allowed = _field(result, 0)
    if allowed is None:
        log.warning("breaker.unreadable", user_id=user_id, service=service, reply=str(result)[:200])
        return BreakerDecision(True, "unknown", 0.0, 0)

    raw_state = _field(result, 1, "unknown")
    state = raw_state.decode() if isinstance(raw_state, bytes) else str(raw_state)
    return BreakerDecision(
        allowed=bool(_as_int(allowed)),
        state=state,
        retry_after_s=max(0.0, _as_int(_field(result, 2)) / 1000.0),
        failures=_as_int(_field(result, 3)),
    )


async def breaker_record_failure(
    user_id: str, service: str, error_class: ErrorClass | None = None
) -> BreakerDecision:
    """Count a failure against the service. Opens the breaker when it is due."""
    if error_class is not None and ErrorClass(error_class) not in BREAKER_CLASSES:
        # A 404 or a bad argument says nothing about Google's health.
        return BreakerDecision(True, "closed", 0.0, 0)
    try:
        script = await _script("cb_fail", _FAIL_LUA)
        result = await script(
            keys=[_breaker_key(user_id, service)],
            args=[
                _now_ms(),
                BREAKER_THRESHOLD,
                int(BREAKER_OPEN_S * 1000),
                int(BREAKER_MAX_OPEN_S * 1000),
            ],
        )
    except (RedisError, OSError):
        return BreakerDecision(True, "unknown", 0.0, 0)

    opened = bool(_as_int(_field(result, 0)))
    failures = _as_int(_field(result, 1))
    open_until = _as_int(_field(result, 2))
    open_ms = _as_int(_field(result, 3))
    if opened:
        log.warning(
            "breaker.opened",
            user_id=user_id,
            service=service,
            open_for_s=round(open_ms / 1000, 1),
            error_class=str(error_class) if error_class else None,
        )
    return BreakerDecision(
        allowed=not opened,
        state="open" if opened else "closed",
        retry_after_s=max(0.0, (open_until - _now_ms()) / 1000.0) if opened else 0.0,
        failures=failures,
    )


async def breaker_record_success(user_id: str, service: str) -> None:
    """Close the breaker. One good call is enough — that is the point of a probe."""
    try:
        client = await cache.get_redis()
        await client.delete(_breaker_key(user_id, service))
    except (RedisError, OSError):
        pass


async def breaker_state(user_id: str, service: str) -> dict[str, Any]:
    """What the breaker holds right now. For /metrics and the sync bar."""
    try:
        client = await cache.get_redis()
        raw = await client.hgetall(_breaker_key(user_id, service))
    except (RedisError, OSError):
        return {"service": service, "state": "unknown", "failures": 0, "open_for_s": 0.0}

    def _num(field: str) -> float:
        value = raw.get(field.encode()) or raw.get(field)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    now = _now_ms()
    open_until = _num("open_until")
    failures = int(_num("failures"))
    if open_until <= 0:
        state = "closed"
    elif now < open_until:
        state = "open"
    else:
        state = "half_open"
    return {
        "service": service,
        "state": state,
        "failures": failures,
        "open_for_s": round(max(0.0, (open_until - now) / 1000.0), 1),
        "open_window_s": round(_num("open_ms") / 1000.0, 1),
    }


async def breaker_reset(user_id: str, service: str) -> None:
    """Clear a breaker by hand. Support operations and tests."""
    await breaker_record_success(user_id, service)


# --------------------------------------------------------------------------- #
# A generic retry loop, for callers outside the request path
# --------------------------------------------------------------------------- #


async def retry_call(
    fn: Callable[[], Awaitable[T]],
    *,
    user_id: str | None = None,
    service: str | None = None,
    on_retry: Callable[[ErrorClass, int, float, Exception], None] | None = None,
    limit: int | None = None,
) -> T:
    """Call ``fn`` until it works or its failure class says stop.

    Each class keeps its own attempt count, so a call that sees one 401 and
    then three 503s is still inside both budgets. Sync tasks use this; the
    request path uses the loop inside ``client.py``, which also has to refresh
    tokens and charge quota per attempt.
    """
    used: dict[ErrorClass, int] = {}
    while True:
        try:
            result = await fn()
        except Exception as exc:
            cls = classify(exc)
            attempt = used.get(cls, 0)
            allowance = max_attempts(cls) if limit is None else min(max_attempts(cls), limit)
            if not retryable(cls) or attempt >= allowance:
                if user_id and service:
                    await breaker_record_failure(user_id, service, cls)
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = backoff(cls, attempt, retry_after)
            used[cls] = attempt + 1
            if on_retry is not None:
                on_retry(cls, attempt, delay, exc)
            log.info(
                "google.retry",
                service=service,
                error_class=str(cls),
                attempt=attempt + 1,
                backoff_ms=int(delay * 1000),
                error=str(exc)[:200],
            )
            if user_id and service:
                await breaker_record_failure(user_id, service, cls)
            if delay:
                await asyncio.sleep(delay)
            continue
        if user_id and service and used:
            await breaker_record_success(user_id, service)
        return result


__all__ = [
    "ErrorClass",
    "Policy",
    "POLICY",
    "BREAKER_CLASSES",
    "APP_ERROR_CODE",
    "GoogleAPIError",
    "CircuitOpen",
    "BreakerDecision",
    "BREAKER_THRESHOLD",
    "BREAKER_OPEN_S",
    "BREAKER_MAX_OPEN_S",
    "BREAKER_PROBE_S",
    "classify",
    "classify_status",
    "retryable",
    "max_attempts",
    "needs_refetch",
    "backoff",
    "parse_retry_after",
    "breaker_allow",
    "breaker_record_failure",
    "breaker_record_success",
    "breaker_state",
    "breaker_reset",
    "retry_call",
]
