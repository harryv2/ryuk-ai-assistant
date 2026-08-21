"""One failure type for every model call, and one set of class names.

`app.google.retry.ErrorClass` already sorts Google API failures into eight
classes, and everything downstream — retry, backoff, circuit breaker, what the
user is told — reads the class and nothing else. Model APIs fail the same ways,
so they get the same names rather than a second vocabulary nobody can hold in
their head at once.

The names reused from `app.google.retry`:

    TRANSIENT · RATE_LIMITED · QUOTA_EXHAUSTED · AUTH_EXPIRED · AUTH_REVOKED
    INVALID · NOT_FOUND · UNKNOWN

Two more that only models produce:

    TIMEOUT           the request ran past the per-attempt budget
    CONTENT_FILTERED  the provider refused on safety grounds

Every provider maps its own SDK's exceptions onto these through
`Provider.classify_error`, so the router never imports an SDK and never has to
know that OpenAI raises `RateLimitError` while Gemini raises something else.

One thing to keep straight: `AUTH_EXPIRED` and `AUTH_REVOKED` here are about
*our* API key for a model provider, not the end user's Google token. They
become `INTERNAL` at the API boundary — a server misconfiguration — and never
`GOOGLE_REAUTH_REQUIRED`, which would show the user a reconnect button for a
problem no amount of reconnecting can fix.
"""

from __future__ import annotations

import random
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from app.core.errors import AppError

if TYPE_CHECKING:  # pragma: no cover - import cycle: base imports this module
    from app.llm.base import Usage


class ErrorClass(StrEnum):
    """What went wrong, in the only vocabulary the system uses."""

    TRANSIENT = "TRANSIENT"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_REVOKED = "AUTH_REVOKED"
    INVALID = "INVALID"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    UNKNOWN = "UNKNOWN"


# Trying the *same* provider again could plausibly work. The router's in-provider
# retry is deliberately narrower than this — see `router._attempt` — because a
# rate limit wants a different provider, not the same one a second later.
RETRYABLE: Final[frozenset[ErrorClass]] = frozenset(
    {ErrorClass.TRANSIENT, ErrorClass.RATE_LIMITED, ErrorClass.TIMEOUT}
)

# What LLM_FALLBACK_ON defaults to. These are the classes where a different
# provider is likely to succeed at the same request.
DEFAULT_FALLBACK_ON: Final[frozenset[ErrorClass]] = frozenset(
    {
        ErrorClass.RATE_LIMITED,
        ErrorClass.TRANSIENT,
        ErrorClass.QUOTA_EXHAUSTED,
        ErrorClass.TIMEOUT,
    }
)

# Classes that say "this provider is unhealthy" and so count towards opening its
# circuit breaker. Deliberately excluded:
#   INVALID, CONTENT_FILTERED, NOT_FOUND — our request is the problem, not the
#     provider, and the next request may be perfectly fine.
#   AUTH_EXPIRED, AUTH_REVOKED — a bad API key fails instantly, so skipping the
#     provider saves no latency, and an open breaker would hide the real reason
#     behind "unavailable" in the logs.
HEALTH_CLASSES: Final[frozenset[ErrorClass]] = frozenset(
    {
        ErrorClass.TRANSIENT,
        ErrorClass.TIMEOUT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.QUOTA_EXHAUSTED,
        ErrorClass.UNKNOWN,
    }
)

# ErrorClass -> the AppError code the API boundary should show.
_APP_CODES: Final[dict[ErrorClass, str]] = {
    ErrorClass.TRANSIENT: "INTERNAL",
    ErrorClass.RATE_LIMITED: "RATE_LIMITED",
    ErrorClass.QUOTA_EXHAUSTED: "RATE_LIMITED",
    ErrorClass.TIMEOUT: "ORCHESTRATION_TIMEOUT",
    ErrorClass.AUTH_EXPIRED: "INTERNAL",
    ErrorClass.AUTH_REVOKED: "INTERNAL",
    ErrorClass.INVALID: "INTERNAL",
    ErrorClass.NOT_FOUND: "INTERNAL",
    ErrorClass.CONTENT_FILTERED: "VALIDATION_ERROR",
    ErrorClass.UNKNOWN: "INTERNAL",
}

# What a person should read when the class is all we have.
_MESSAGES: Final[dict[ErrorClass, str]] = {
    ErrorClass.TRANSIENT: "The model had a temporary problem.",
    ErrorClass.RATE_LIMITED: "The model is rate limited right now. Try again in a moment.",
    ErrorClass.QUOTA_EXHAUSTED: "We are out of model quota for now.",
    ErrorClass.TIMEOUT: "The model took too long to answer.",
    ErrorClass.AUTH_EXPIRED: "The model API key is not working.",
    ErrorClass.AUTH_REVOKED: "The model API key has been revoked.",
    ErrorClass.INVALID: "The model rejected the request.",
    ErrorClass.NOT_FOUND: "That model does not exist.",
    ErrorClass.CONTENT_FILTERED: "The model declined to answer that.",
    ErrorClass.UNKNOWN: "The model call failed.",
}


def coerce(value: Any) -> ErrorClass:
    """Any name — a string, an ErrorClass, a Google ErrorClass — as one of ours.

    Anything unrecognised becomes ``UNKNOWN``. That includes Google's
    ``PRECONDITION``, which has no meaning for a model call: there is no etag to
    lose a race with.
    """
    if isinstance(value, ErrorClass):
        return value
    try:
        return ErrorClass(str(value).strip().upper())
    except ValueError:
        return ErrorClass.UNKNOWN


def is_retryable(cls: Any) -> bool:
    """True when the same provider is worth another try."""
    return coerce(cls) in RETRYABLE


def counts_against_health(cls: Any) -> bool:
    """True when this failure should count towards opening the breaker."""
    return coerce(cls) in HEALTH_CLASSES


def backoff(cls: Any, attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """Seconds to wait before attempt ``attempt + 1``. Full jitter.

    Full jitter, not equal jitter, for the reason `DESIGN.md` §6.3 gives: equal
    jitter packs every retry into the upper half of the window, so a herd that
    collided once collides again there. ``attempt`` is 1-based.
    """
    if not is_retryable(cls):
        return 0.0
    ceiling = min(cap, base * (2 ** max(0, attempt - 1)))
    return random.uniform(0, ceiling)


def class_for_status(status: int | None) -> ErrorClass:
    """The class for a bare HTTP status.

    Providers use this as the last step of `classify_error`, after they have
    looked at whatever their SDK says. Every provider maps the same numbers the
    same way, so this lives here rather than twice.
    """
    if status is None:
        return ErrorClass.UNKNOWN
    if status in (408, 504):
        return ErrorClass.TIMEOUT
    if status == 429:
        return ErrorClass.RATE_LIMITED
    if status == 401:
        return ErrorClass.AUTH_EXPIRED
    if status == 403:
        return ErrorClass.AUTH_REVOKED
    if status == 404:
        return ErrorClass.NOT_FOUND
    if status in (400, 422):
        return ErrorClass.INVALID
    if status >= 500:
        return ErrorClass.TRANSIENT
    return ErrorClass.UNKNOWN


class LLMError(Exception):
    """A model call that failed, with everything needed to decide what next.

    Carries the provider and model that failed, the class of failure, the
    underlying SDK exception, and whether retrying the same provider is worth
    it. ``usage`` is set when the failed attempt still burned tokens, so the
    cost of a failure is not invisible.
    """

    def __init__(
        self,
        error_class: Any,
        message: str | None = None,
        *,
        provider: str = "",
        model: str = "",
        cause: BaseException | None = None,
        retryable: bool | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.error_class: ErrorClass = coerce(error_class)
        self.provider = provider
        self.model = model
        self.cause = cause
        self.status = status
        self.retryable = is_retryable(self.error_class) if retryable is None else retryable
        self.details: dict[str, Any] = details or {}
        self.usage = usage
        self.message = message or _MESSAGES[self.error_class]
        # `model` is normally the full reference already ("openai:gpt-4.1-mini"),
        # so only reach for the provider when it is not.
        where = model or provider or "llm"
        if provider and model and ":" not in model:
            where = f"{provider}:{model}"
        super().__init__(f"{where} {self.error_class}: {self.message}")

    # -- construction ---------------------------------------------------------

    @classmethod
    def wrap(
        cls,
        exc: BaseException,
        *,
        error_class: Any,
        provider: str = "",
        model: str = "",
        message: str | None = None,
        **details: Any,
    ) -> LLMError:
        """An SDK exception as an LLMError, keeping a readable slice of it."""
        if isinstance(exc, LLMError):
            return exc
        return cls(
            error_class,
            message,
            provider=provider,
            model=model,
            cause=exc,
            details={"cause": type(exc).__name__, "detail": str(exc)[:500], **details},
        )

    # -- rendering ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The shape that goes into a structured log line or a step outcome."""
        return {
            "provider": self.provider,
            "model": self.model,
            "class": str(self.error_class),
            "retryable": self.retryable,
            "message": self.message,
            "details": self.details,
        }

    def to_app_error(self) -> AppError:
        """The same failure at the API boundary.

        Note what is *not* here: no LLM failure ever becomes
        `GOOGLE_REAUTH_REQUIRED`. Our API key being wrong is our problem, and
        telling the user to reconnect their Google account would send them off
        to fix something that is not broken.
        """
        details = {
            "provider": self.provider,
            "model": self.model,
            "class": str(self.error_class),
            **self.details,
        }
        return AppError(_APP_CODES[self.error_class], self.message, details=details)


__all__ = [
    "DEFAULT_FALLBACK_ON",
    "HEALTH_CLASSES",
    "RETRYABLE",
    "ErrorClass",
    "LLMError",
    "backoff",
    "class_for_status",
    "coerce",
    "counts_against_health",
    "is_retryable",
]
