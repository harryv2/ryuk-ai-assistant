"""The audit trail.

One row for anything that changed outside our system: an email sent, an event
deleted, a file shared, a token granted or revoked, an account purged.

Bodies are never stored. ``payload_hash`` is a uuid5 of the canonical payload,
so you can prove *this exact email* was sent without keeping a word of it.
``payload_visible`` keeps only the envelope — recipients and subject — which is
what an audit view actually needs to show.

Written with Core SQL rather than the ORM model on purpose: this module sits
under ``app.core`` and must not depend on ``app.db.models``.
"""

from __future__ import annotations

import ipaddress
from datetime import UTC, datetime
from typing import Any, Final, Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import canonical_json, fingerprint
from app.core.logging import get_logger

log = get_logger(__name__)

# Keys copied into payload_visible, in this order. Everything else is hashed
# and dropped.
VISIBLE_KEYS: Final[tuple[str, ...]] = (
    "to",
    "cc",
    "bcc",
    "recipients",
    "attendees",
    "subject",
    "title",
)

# Recipient-ish keys get normalised to a list of addresses.
_ADDRESS_KEYS: Final[frozenset[str]] = frozenset(
    {"to", "cc", "bcc", "recipients", "attendees"}
)

_MAX_VISIBLE_ADDRESSES: Final[int] = 50
_MAX_SUBJECT_CHARS: Final[int] = 300

#: uuid5 namespace for payload hashes. The same constant as
#: ``app.db.repositories.audit.PAYLOAD_NAMESPACE`` on purpose: one payload must
#: hash to one value whichever path wrote the row.
PAYLOAD_NAMESPACE: Final[str] = "audit.payload"

_INSERT: Final[str] = """
INSERT INTO audit_log (
    user_id, conversation_id, actor, action, resource_id,
    payload_hash, payload_visible, status, error, ip, user_agent, created_at
) VALUES (
    :user_id, :conversation_id, :actor, :action, :resource_id,
    CAST(:payload_hash AS UUID), CAST(:payload_visible AS JSONB), :status,
    CAST(:error AS JSONB), CAST(:ip AS INET), :user_agent, :created_at
)
RETURNING id
"""


def _addresses(value: Any) -> list[str]:
    """Pull email addresses out of whatever shape the payload used."""
    if value is None:
        return []
    if isinstance(value, str):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, dict):
        items = [value]
    else:
        items = [value]

    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            address = item.get("email") or item.get("address") or item.get("value")
            if address:
                out.append(str(address))
        elif item is not None:
            out.append(str(item))
    return out[:_MAX_VISIBLE_ADDRESSES]


def visible_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Recipients and subject only. Never a body, never an attachment."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in VISIBLE_KEYS:
        if key not in payload or payload[key] in (None, "", [], {}):
            continue
        if key in _ADDRESS_KEYS:
            addresses = _addresses(payload[key])
            if addresses:
                out[key] = addresses
        else:
            out[key] = str(payload[key])[:_MAX_SUBJECT_CHARS]
    return out


def payload_fingerprint(payload: Any) -> UUID | None:
    """uuid5 of the full payload. Key order and whitespace cannot change it."""
    if payload is None:
        return None
    try:
        return fingerprint(PAYLOAD_NAMESPACE, canonical_json(payload))
    except (TypeError, ValueError):
        return fingerprint(PAYLOAD_NAMESPACE, repr(payload))


def _normalise_error(error: Any) -> str | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        error = {"type": type(error).__name__, "message": str(error)[:500]}
    elif isinstance(error, str):
        error = {"message": error[:500]}
    try:
        return canonical_json(error)
    except (TypeError, ValueError):
        return canonical_json({"message": str(error)[:500]})


def _normalise_ip(ip: Any) -> str | None:
    if not ip:
        return None
    candidate = str(ip).split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


async def record(
    session: AsyncSession,
    user_id: str,
    actor: str,
    action: str,
    resource_id: str | None = None,
    payload: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    status: str = "ok",
    error: Any = None,
    ip: str | None = None,
    ua: str | None = None,
) -> int | None:
    """Append one audit row.

    Runs inside the caller's transaction and flushes, but does not commit —
    the audit row lands with whatever it is describing, or not at all.

    An audit failure never breaks the operation being audited: the error is
    logged and ``None`` comes back.
    """
    params = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "actor": actor[:16],
        "action": action[:64],
        "resource_id": str(resource_id)[:255] if resource_id is not None else None,
        "payload_hash": str(payload_fingerprint(payload))
        if payload is not None
        else None,
        "payload_visible": canonical_json(visible_fields(payload))
        if payload is not None
        else None,
        "status": str(status)[:16],
        "error": _normalise_error(error),
        "ip": _normalise_ip(ip),
        "user_agent": str(ua)[:1000] if ua else None,
        "created_at": datetime.now(UTC),
    }
    try:
        # A savepoint, so a rejected audit row cannot poison the transaction
        # that carries the thing being audited.
        async with session.begin_nested():
            result = await session.execute(text(_INSERT), params)
            row_id = result.scalar_one()
        return int(row_id)
    except Exception as exc:  # noqa: BLE001 - auditing must never be the failure
        log.error(
            "audit.write_failed",
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            error=str(exc),
        )
        return None


async def record_many(
    session: AsyncSession, rows: Sequence[dict[str, Any]]
) -> list[int]:
    """Several rows in one go. Each dict takes :func:`record`'s keyword names."""
    written: list[int] = []
    for row in rows:
        row_id = await record(session, **row)
        if row_id is not None:
            written.append(row_id)
    return written


__all__ = [
    "VISIBLE_KEYS",
    "PAYLOAD_NAMESPACE",
    "record",
    "record_many",
    "visible_fields",
    "payload_fingerprint",
]
