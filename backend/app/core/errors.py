"""The one error type.

Every failure that reaches the API boundary is an :class:`AppError`. It carries
a code from a fixed table, a message safe to show a person, an HTTP status and
an optional details dict. The response body shape is fixed by the contract::

    {"error": {"code", "message", "details", "request_id"}}
"""

from __future__ import annotations

from typing import Any, Final

# code -> default HTTP status. The table is closed; a new code is a contract
# change, not a local decision.
CODES: Final[dict[str, int]] = {
    "VALIDATION_ERROR": 422,
    "NOT_AUTHENTICATED": 401,
    "GOOGLE_REAUTH_REQUIRED": 428,
    "RATE_LIMITED": 429,
    "PROMPT_NOT_PENDING": 409,
    "PROMPT_VALUE_INVALID": 422,
    "ORCHESTRATION_TIMEOUT": 504,
    "GOOGLE_UNAVAILABLE": 503,
    "NOT_FOUND": 404,
    "INTERNAL": 500,
}

# Messages used when a raiser does not supply one.
DEFAULT_MESSAGES: Final[dict[str, str]] = {
    "VALIDATION_ERROR": "That request did not make sense.",
    "NOT_AUTHENTICATED": "Sign in to continue.",
    "GOOGLE_REAUTH_REQUIRED": "Reconnect your Google account to continue.",
    "RATE_LIMITED": "Too many requests. Try again shortly.",
    "PROMPT_NOT_PENDING": "That card has already been answered.",
    "PROMPT_VALUE_INVALID": "That answer does not fit what was asked.",
    "ORCHESTRATION_TIMEOUT": "This took too long and was stopped.",
    "GOOGLE_UNAVAILABLE": "Google is not responding right now.",
    "NOT_FOUND": "Not found.",
    "INTERNAL": "Something went wrong on our side.",
}


class AppError(Exception):
    """A failure with a code, a status and a body the client can render."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        http: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if code not in CODES:
            # An unknown code must never become a silent 400. Keep the original
            # code visible in details so nothing is lost, then fall back.
            details = {**(details or {}), "unknown_code": code}
            code = "INTERNAL"
        self.code = code
        self.message = message or DEFAULT_MESSAGES[code]
        self.http = http if http is not None else CODES[code]
        self.details = details or {}
        super().__init__(f"{self.code}: {self.message}")

    # -- construction helpers -------------------------------------------------

    @classmethod
    def validation(cls, message: str, **details: Any) -> "AppError":
        return cls("VALIDATION_ERROR", message, details=details or None)

    @classmethod
    def not_found(cls, what: str, ref: str | None = None) -> "AppError":
        details = {"resource": what}
        if ref is not None:
            details["ref"] = ref
        return cls("NOT_FOUND", f"No {what} with that id.", details=details)

    @classmethod
    def internal(cls, message: str | None = None, **details: Any) -> "AppError":
        return cls("INTERNAL", message, details=details or None)

    # -- rendering ------------------------------------------------------------

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        """The `error` object on its own."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "request_id": request_id,
        }

    def to_response(self, request_id: str | None = None) -> dict[str, Any]:
        """The full response body: ``{"error": {...}}``."""
        return {"error": self.to_dict(request_id)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AppError(code={self.code!r}, http={self.http}, "
            f"message={self.message!r}, details={self.details!r})"
        )


def from_exception(exc: BaseException) -> AppError:
    """Wrap any exception as an AppError, passing existing ones through."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        return AppError("ORCHESTRATION_TIMEOUT", details={"cause": type(exc).__name__})
    return AppError(
        "INTERNAL",
        details={"cause": type(exc).__name__, "detail": str(exc)[:500]},
    )


def error_response(exc: BaseException, request_id: str | None = None) -> dict[str, Any]:
    """Body for any exception, wrapping it first if needed."""
    return from_exception(exc).to_response(request_id)


__all__ = [
    "CODES",
    "DEFAULT_MESSAGES",
    "AppError",
    "from_exception",
    "error_response",
]
