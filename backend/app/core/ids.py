"""Ids, fingerprints and canonical JSON.

Three rules the whole system leans on:

* every primary key is a nanoid of exactly 21 characters, made here, never by
  the database;
* every fingerprint (``content_hash``, ``dedupe_key``, ``payload_hash``) is a
  uuid5, so the same content always gives the same value on any machine, in
  any process, forever;
* the string that goes into a fingerprint is produced by
  :func:`canonical_json`, so key order and whitespace can never change it.
"""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final

# The nanoid alphabet: 64 URL-safe characters. A power of two, which means a
# random byte maps to a character with no rejection loop and no modulo bias.
ALPHABET: Final[str] = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
ID_LENGTH: Final[int] = 21

# Root namespace for every uuid5 we make. Derived from a fixed URL so it is
# stable across deploys and reproducible from this source file alone.
NAMESPACE_ROOT: Final[uuid.UUID] = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://alpha-law.local/fingerprint/v1"
)

# Separator between namespace and payload. Unit separator, never appears in a
# namespace name and is escaped by json.dumps inside a canonical payload.
_SEP: Final[str] = "\x1f"


def new_id() -> str:
    """A fresh nanoid: 21 characters, ~126 bits of entropy."""
    raw = secrets.token_bytes(ID_LENGTH)
    return "".join(ALPHABET[b & 0x3F] for b in raw)


def is_id(value: object) -> bool:
    """True when ``value`` looks like one of our ids."""
    return (
        isinstance(value, str)
        and len(value) == ID_LENGTH
        and all(c in ALPHABET for c in value)
    )


def fingerprint(ns: str, text: str) -> uuid.UUID:
    """A uuid5 of ``text`` inside the namespace ``ns``.

    ``ns`` keeps unrelated things apart: the same string hashed as
    ``"gmail.body"`` and as ``"action.payload"`` gives two different uuids.
    """
    return uuid.uuid5(NAMESPACE_ROOT, f"{ns}{_SEP}{text}")


def fingerprint_parts(ns: str, *parts: Any) -> uuid.UUID:
    """Fingerprint of several values joined in order.

    Used for composite keys such as ``dedupe_key = uuid5(user|op|payload|conv)``.
    Each part is canonicalised first, so dicts are order-insensitive.
    """
    joined = _SEP.join(
        p if isinstance(p, str) else canonical_json(p) for p in parts
    )
    return fingerprint(ns, joined)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return format(obj, "f")
    if isinstance(obj, (set, frozenset)):
        return sorted(canonical_json(v) for v in obj)
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"cannot canonicalise {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace — the input to a fingerprint."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )


__all__ = [
    "ALPHABET",
    "ID_LENGTH",
    "NAMESPACE_ROOT",
    "new_id",
    "is_id",
    "fingerprint",
    "fingerprint_parts",
    "canonical_json",
]
