"""structlog set-up: one JSON object per line, with a request id on every line.

``configure()`` runs once at start-up — from ``main.py`` for the API and from
``celery_app.py`` for the workers. After that ``get_logger(__name__)``
anywhere gives a bound logger.

The request id lives in a contextvar, so it follows the request into every
``await`` and into any task spawned from it, without being threaded through
call signatures.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

import structlog

from app.core.ids import new_id

# -- context ----------------------------------------------------------------

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

_configured = False


def get_request_id() -> str | None:
    return request_id_var.get()


def set_request_id(request_id: str | None = None) -> Token:
    """Put a request id in context. Returns the token to reset with."""
    return request_id_var.set(request_id or new_id())


def reset_request_id(token: Token) -> None:
    request_id_var.reset(token)


@contextmanager
def request_context(
    request_id: str | None = None,
    run_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[str]:
    """Bind ids for the duration of a block, then put them back as they were."""
    tokens: list[tuple[ContextVar, Token]] = []
    rid = request_id or new_id()
    tokens.append((request_id_var, request_id_var.set(rid)))
    if run_id is not None:
        tokens.append((run_id_var, run_id_var.set(run_id)))
    if user_id is not None:
        tokens.append((user_id_var, user_id_var.set(user_id)))
    try:
        yield rid
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _add_context(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
    """Copy the contextvars onto every event that does not set them itself."""
    for key, var in (
        ("request_id", request_id_var),
        ("run_id", run_id_var),
        ("user_id", user_id_var),
    ):
        value = var.get()
        if value is not None and key not in event:
            event[key] = value
    return event


def _rename_level(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
    if "level" in event:
        event["severity"] = event["level"]
    return event


# -- configuration ----------------------------------------------------------

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    _add_context,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
]


def configure(level: str | int | None = None, json_logs: bool | None = None) -> None:
    """Configure structlog and the stdlib root logger. Safe to call twice."""
    global _configured

    from app.config import settings  # local: config may import core.errors

    if level is None:
        level = getattr(settings, "LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if not isinstance(level, int):
        level = logging.INFO

    if json_logs is None:
        json_logs = bool(getattr(settings, "LOG_JSON", True))

    renderer: Any = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            _rename_level,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            *_SHARED_PROCESSORS,
            _rename_level,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Libraries that are chatty at INFO and say nothing we need.
    for noisy, noisy_level in (
        ("uvicorn.access", logging.WARNING),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("openai", logging.WARNING),
        ("sqlalchemy.engine", logging.WARNING),
        ("asyncio", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(noisy_level)

    _configured = True


def get_logger(name: str | None = None) -> Any:
    """A bound structlog logger. Configures on first use if nobody has yet."""
    if not _configured:
        try:
            configure()
        except Exception:  # pragma: no cover - never let logging break a request
            structlog.configure(
                processors=[*_SHARED_PROCESSORS, structlog.processors.JSONRenderer()],
                logger_factory=structlog.PrintLoggerFactory(),
            )
    return structlog.get_logger(name) if name else structlog.get_logger()


__all__ = [
    "configure",
    "get_logger",
    "request_id_var",
    "run_id_var",
    "user_id_var",
    "get_request_id",
    "set_request_id",
    "reset_request_id",
    "request_context",
]
