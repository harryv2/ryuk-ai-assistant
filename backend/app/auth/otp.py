"""The one-time code sent to an email address.

Codes live in Redis, never in Postgres: they are worthless after ten minutes and
a row per attempt is a table that only ever grows.

In development every code is 123456. That is a real decision, not a leftover —
there is no mailer wired up, and a code nobody can read is a door nobody can open.
It is behind `OTP_DEV_MODE`, the app shouts about it at startup, and `send_code`
is the single function to replace when a mailer exists.
"""

from __future__ import annotations

import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any, Final

from app.core import logging as applog
from app.core.cache import get_redis
from app.config import settings

log = applog.get_logger(__name__)

DEV_CODE: Final[str] = "123456"
TTL_SECONDS: Final[int] = 600
MAX_ATTEMPTS: Final[int] = 5
RESEND_LIMIT: Final[int] = 3
RESEND_WINDOW_SECONDS: Final[int] = 600


def dev_mode() -> bool:
    return bool(getattr(settings, "OTP_DEV_MODE", True))


def _key(email: str) -> str:
    return f"otp:{email.strip().lower()}"


def _resend_key(email: str) -> str:
    return f"otp:resend:{email.strip().lower()}"


@dataclass(frozen=True)
class Issued:
    email: str
    expires_in: int
    dev_code: str | None  #: filled only in dev mode, so the UI can show it


@dataclass(frozen=True)
class Check:
    ok: bool
    reason: str | None = None
    attempts_remaining: int = 0


async def issue(email: str) -> Issued:
    """Make a code for this address and store it. Replaces any earlier one."""
    email = email.strip().lower()
    code = DEV_CODE if dev_mode() else f"{secrets.randbelow(1_000_000):06d}"
    redis = await get_redis()
    await redis.set(
        _key(email),
        json.dumps({"code": code, "attempts": 0}),
        ex=TTL_SECONDS,
    )
    await send_code(email, code)
    return Issued(email=email, expires_in=TTL_SECONDS, dev_code=code if dev_mode() else None)


async def send_code(email: str, code: str) -> None:
    """Deliver the code. THE ONE FUNCTION TO REPLACE when a mailer exists.

    Today it logs. In dev mode the code is the constant anyway, so nothing is
    leaked that the sign-in screen does not already say out loud.
    """
    if dev_mode():
        log.info("otp.issued_dev", email=email, code=code)
        return
    log.warning(
        "otp.no_mailer",
        email=email,
        detail="OTP_DEV_MODE is off but no mailer is configured; the code cannot be delivered",
    )


async def check(email: str, code: str) -> Check:
    """Verify a code, once. A correct code is consumed and cannot be reused."""
    email = email.strip().lower()
    redis = await get_redis()
    raw = await redis.get(_key(email))
    if raw is None:
        return Check(False, "expired", 0)

    state: dict[str, Any] = json.loads(raw if isinstance(raw, str) else raw.decode())
    attempts = int(state.get("attempts", 0)) + 1
    expected = str(state.get("code", ""))

    if hmac.compare_digest(expected, str(code).strip()):
        await redis.delete(_key(email))
        return Check(True)

    if attempts >= MAX_ATTEMPTS:
        # Burn the code rather than leaving it to be guessed at leisure.
        await redis.delete(_key(email))
        return Check(False, "too_many_attempts", 0)

    state["attempts"] = attempts
    ttl = await redis.ttl(_key(email))
    await redis.set(_key(email), json.dumps(state), ex=max(int(ttl or 0), 1))
    return Check(False, "wrong_code", MAX_ATTEMPTS - attempts)


async def may_resend(email: str) -> bool:
    """Rate limit on asking for another code."""
    redis = await get_redis()
    key = _resend_key(email)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, RESEND_WINDOW_SECONDS)
    return int(count) <= RESEND_LIMIT


async def clear(email: str) -> None:
    redis = await get_redis()
    await redis.delete(_key(email), _resend_key(email))


__all__ = [
    "DEV_CODE",
    "MAX_ATTEMPTS",
    "TTL_SECONDS",
    "Check",
    "Issued",
    "check",
    "clear",
    "dev_mode",
    "issue",
    "may_resend",
    "send_code",
]
