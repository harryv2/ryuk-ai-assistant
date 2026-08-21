"""The session cookie, and the dependencies every endpoint uses.

The cookie is a signed statement of one fact: which user this browser is. It
holds no token material and no profile data, so a stolen cookie buys an
attacker exactly what a stolen cookie should — a session, revocable by rotating
``SESSION_SECRET``.

Format::

    <user_id>:<issued_at_epoch_seconds>:<base64url(hmac_sha256(secret, "<user_id>:<issued_at>"))>

HMAC-SHA256 over ``SESSION_SECRET``, compared with :func:`hmac.compare_digest`.
Ids are nanoids and the timestamp is digits, so ``:`` never appears inside a
field and the split is unambiguous.

Flags on the way out: ``HttpOnly``, ``SameSite=Lax``, ``Path=/``, and
``Secure`` everywhere except development. Lax rather than Strict because the
OAuth callback is a top-level cross-site GET landing back on us, and Strict
would drop the cookie we just set on the redirect that follows.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from collections.abc import AsyncIterator
from typing import Annotated, Final

from fastapi import Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import User
from app.db.repositories import users as users_repo
from app.db.session import session_scope

log = get_logger(__name__)

# The short-lived cookie that binds an in-flight OAuth handshake to this
# browser. It carries a nonce, nothing else, and dies with the handshake.
OAUTH_COOKIE_SUFFIX: Final[str] = "_oauth"
OAUTH_COOKIE_MAX_AGE: Final[int] = 600  # matches the redis ttl on the state

# A clock that runs a little fast on the signing host must not invalidate every
# cookie it mints.
_CLOCK_SKEW_S: Final[int] = 60

_dev_secret: bytes | None = None


# --------------------------------------------------------------------------- #
# names and flags
# --------------------------------------------------------------------------- #


def session_cookie_name() -> str:
    """The session cookie's name, from settings so it can be namespaced."""
    return str(getattr(settings, "SESSION_COOKIE_NAME", "alpha_session"))


def oauth_cookie_name() -> str:
    """The handshake cookie's name — the session cookie plus ``_oauth``."""
    return session_cookie_name() + OAUTH_COOKIE_SUFFIX


def _secure_cookies() -> bool:
    """``Secure`` off in development, where there is no TLS on localhost."""
    return not settings.is_dev


def _secret() -> bytes:
    """The signing key.

    Outside development a missing ``SESSION_SECRET`` is a refusal, not a
    fallback — an unsigned session is worse than no session. In development we
    mint one per process and say so, so restarts sign you out rather than
    silently sharing a well-known key.
    """
    global _dev_secret
    raw = str(getattr(settings, "SESSION_SECRET", "") or "")
    if raw:
        return raw.encode("utf-8")
    if not settings.is_dev:
        raise AppError(
            "INTERNAL",
            "SESSION_SECRET is not configured.",
            details={"setting": "SESSION_SECRET"},
        )
    if _dev_secret is None:
        _dev_secret = os.urandom(32)
        log.warning("auth.session_secret_missing", detail="using a per-process dev key")
    return _dev_secret


def max_age() -> int:
    return int(getattr(settings, "SESSION_MAX_AGE_S", 60 * 60 * 24 * 30))


# --------------------------------------------------------------------------- #
# signing
# --------------------------------------------------------------------------- #


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign(payload: str) -> str:
    digest = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64u(digest)


def issue_session(user_id: str, *, issued_at: int | None = None) -> str:
    """Mint a cookie value for ``user_id``."""
    stamp = int(issued_at if issued_at is not None else time.time())
    payload = f"{user_id}:{stamp}"
    return f"{payload}:{_sign(payload)}"


def verify_session(value: str | None) -> str | None:
    """The user id inside a cookie value, or ``None`` if it does not hold up.

    Returns ``None`` for every kind of failure — malformed, tampered with,
    expired, or issued in the future. The caller turns that into
    ``NOT_AUTHENTICATED``; nothing here leaks which of those it was.
    """
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    user_id, stamp, signature = parts
    if not user_id or not stamp.isdigit():
        return None
    try:
        expected = _sign(f"{user_id}:{stamp}")
    except AppError:
        return None
    if not hmac.compare_digest(expected, signature):
        return None

    issued_at = int(stamp)
    now = int(time.time())
    if issued_at > now + _CLOCK_SKEW_S:
        return None
    if now - issued_at > max_age():
        return None
    return user_id


# --------------------------------------------------------------------------- #
# reading and writing cookies
# --------------------------------------------------------------------------- #


def read_session(request: Request) -> str | None:
    """The signed-in user id carried by this request, if any."""
    return verify_session(request.cookies.get(session_cookie_name()))


def set_session_cookie(response: Response, user_id: str) -> None:
    """Sign this browser in as ``user_id``."""
    response.set_cookie(
        key=session_cookie_name(),
        value=issue_session(user_id),
        max_age=max_age(),
        httponly=True,
        secure=_secure_cookies(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Sign this browser out. Same flags, or the browser keeps the old one."""
    response.delete_cookie(
        key=session_cookie_name(),
        httponly=True,
        secure=_secure_cookies(),
        samesite="lax",
        path="/",
    )


def set_oauth_cookie(response: Response, binding: str) -> None:
    """Bind an in-flight OAuth handshake to this browser."""
    response.set_cookie(
        key=oauth_cookie_name(),
        value=binding,
        max_age=OAUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=_secure_cookies(),
        samesite="lax",
        path="/",
    )


def read_oauth_cookie(request: Request) -> str | None:
    return request.cookies.get(oauth_cookie_name())


def clear_oauth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=oauth_cookie_name(),
        httponly=True,
        secure=_secure_cookies(),
        samesite="lax",
        path="/",
    )


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #


async def get_session() -> AsyncIterator[AsyncSession]:
    """An ``AsyncSession`` in one transaction — commit on success, rollback on
    error.

    The whole request is that transaction, which is what lets an assistant
    message and the prompts and actions its content references land together.
    Use this one dependency everywhere in the API: two of them in one request
    would be two transactions, and the pair could half-commit.
    """
    async with session_scope() as session:
        yield session


def current_user_id(request: Request) -> str:
    """The signed-in user id, or ``NOT_AUTHENTICATED``.

    Cookie only, no database round trip — for endpoints that just need the
    tenant key.
    """
    user_id = read_session(request)
    if user_id is None:
        raise AppError("NOT_AUTHENTICATED")
    return user_id


def optional_user_id(request: Request) -> str | None:
    """The signed-in user id, or ``None``. For endpoints open to strangers."""
    return read_session(request)


async def current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    """The signed-in :class:`~app.db.models.User` row.

    A cookie naming a user who no longer exists — disconnected, purged — is
    treated exactly like no cookie at all.
    """
    user_id = current_user_id(request)
    user = await users_repo.get_user(session, user_id)
    if user is None:
        log.info("auth.session_user_missing", user_id=user_id)
        raise AppError("NOT_AUTHENTICATED")
    return user


SessionDep = Annotated[AsyncSession, Depends(get_session)]
CurrentUser = Annotated[User, Depends(current_user)]
CurrentUserId = Annotated[str, Depends(current_user_id)]

__all__ = [
    "OAUTH_COOKIE_MAX_AGE",
    "OAUTH_COOKIE_SUFFIX",
    "session_cookie_name",
    "oauth_cookie_name",
    "max_age",
    "issue_session",
    "verify_session",
    "read_session",
    "set_session_cookie",
    "clear_session_cookie",
    "set_oauth_cookie",
    "read_oauth_cookie",
    "clear_oauth_cookie",
    "get_session",
    "current_user",
    "current_user_id",
    "optional_user_id",
    "SessionDep",
    "CurrentUser",
    "CurrentUserId",
]
