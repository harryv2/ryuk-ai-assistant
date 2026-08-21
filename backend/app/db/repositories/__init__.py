"""Repositories. One module per aggregate.

Two rules hold everywhere in here:

1. ``session`` is the first parameter, ``user_id`` the first real argument.
   A handful of sweeper functions accept ``user_id=None`` to mean "every user";
   they are only ever called from Celery maintenance tasks, never from a
   request path, and each one says so in its docstring.
2. Every datetime written is tz-aware UTC.

Repositories flush but do not commit. The caller owns the transaction, because
an assistant message and the prompts and actions its content references have to
land together.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, TypeVar

T = TypeVar("T")


def utcnow() -> dt.datetime:
    """Tz-aware now, in UTC. The only clock the database layer reads."""
    return dt.datetime.now(dt.UTC)


def ensure_utc(value: dt.datetime | None) -> dt.datetime | None:
    """Reject naive datetimes; normalise anything else to UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("naive datetime reached the database layer")
    return value.astimezone(dt.UTC)


def clean(values: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is the sentinel ``UNSET``."""
    return {k: v for k, v in values.items() if v is not UNSET}


class _Unset:
    """Distinguishes 'not supplied' from 'set to NULL' in update helpers."""

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()

__all__ = [
    "utcnow",
    "ensure_utc",
    "clean",
    "UNSET",
    "users",
    "conversations",
    "runs",
    "steps",
    "prompts",
    "actions",
    "entities",
    "mirror",
    "sync_state",
    "audit",
]

# Imported last: the submodules import the helpers above from this package.
from app.db.repositories import (  # noqa: E402,F401
    actions,
    audit,
    conversations,
    entities,
    mirror,
    prompts,
    runs,
    steps,
    sync_state,
    users,
)
