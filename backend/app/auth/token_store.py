"""The only place a token is decrypted.

Everything that talks to Google asks here for an access token and gets back a
plain string it may use for the next few minutes. What happens in between —
decrypting the blob, noticing it is about to expire, spending the refresh
token, re-encrypting under the current key — is nobody else's business.

Two failure modes matter and they are told apart deliberately:

* ``invalid_grant`` from the refresh endpoint means the grant is gone. The user
  revoked it in their Google account, or the refresh token was rotated out from
  under us. Nothing we can retry. We stamp ``revoked_at``, count the failure
  and raise ``GOOGLE_REAUTH_REQUIRED`` (428) so the UI shows a reconnect button.
* Anything else — a timeout, a 500, a network blip — is Google being briefly
  unavailable. We count the failure and raise ``GOOGLE_UNAVAILABLE`` (503). The
  grant is untouched.

The bookkeeping for a *failed* refresh is written in its own session and
committed there, so a request that is about to unwind still leaves the record
behind. A *successful* refresh is written in the caller's session: if that
transaction rolls back we simply refresh again next time, which costs one call
and is always safe.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Final

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core import crypto
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import OAuthToken
from app.db.repositories import users as users_repo
from app.db.session import session_scope

log = get_logger(__name__)

TOKEN_ENDPOINT: Final[str] = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT: Final[str] = "https://oauth2.googleapis.com/revoke"

# Refresh anything expiring inside this window rather than handing out a token
# that dies mid-request.
REFRESH_SKEW_S: Final[int] = 300  # five minutes

# After this many consecutive failures we stop trying and ask for a reconnect.
MAX_REFRESH_FAILURES: Final[int] = 5

HTTP_TIMEOUT_S: Final[float] = 10.0

# Errors from the token endpoint that mean "this grant is dead", not "try again".
_DEAD_GRANT: Final[frozenset[str]] = frozenset(
    {"invalid_grant", "invalid_token", "unauthorized_client"}
)
# Errors that mean our own client configuration is wrong. Retrying cannot help
# and it is not the user's problem to fix.
_MISCONFIGURED: Final[frozenset[str]] = frozenset(
    {"invalid_client", "unsupported_grant_type"}
)

# One refresh per user per process. Ten concurrent steps hitting an expired
# token should spend one refresh, not ten.
_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _lock_for(user_id: str, provider: str) -> asyncio.Lock:
    key = (user_id, provider)
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _expiry_from(expires_in: Any) -> dt.datetime | None:
    """``expires_in`` seconds, as an absolute UTC instant."""
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return None
    return _utcnow() + dt.timedelta(seconds=seconds)


def _is_stale(expires_at: dt.datetime | None, *, skew_s: int = REFRESH_SKEW_S) -> bool:
    """True when a token is expired or close enough that it may as well be.

    No expiry recorded counts as stale: we would rather spend a refresh than
    hand out something we cannot vouch for.
    """
    if expires_at is None:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=dt.UTC)
    return expires_at <= _utcnow() + dt.timedelta(seconds=skew_s)


def needs_reauth(row: OAuthToken | None) -> bool:
    """Would the next Google call return 428? Answer without making one.

    ``/api/v1/auth/me`` reads this so the banner appears before the user hits
    the wall.
    """
    if row is None or row.revoked_at is not None:
        return True
    if row.refresh_failures >= MAX_REFRESH_FAILURES:
        return True
    if row.refresh_token_enc is None and _is_stale(row.expires_at):
        return True
    return False


def _reauth(reason: str, **details: Any) -> AppError:
    return AppError(
        "GOOGLE_REAUTH_REQUIRED",
        "Reconnect your Google account to continue.",
        details={"reason": reason, **details},
    )


# --------------------------------------------------------------------------- #
# failure bookkeeping, in its own transaction
# --------------------------------------------------------------------------- #


async def _record_failure(
    user_id: str, provider: str, *, revoked: bool
) -> int:
    """Count a failed refresh, and stamp ``revoked_at`` when the grant is dead.

    Runs in a session of its own so the record survives the caller's rollback.
    Never raises: bookkeeping must not replace the error it is describing.
    """
    try:
        async with session_scope() as book:
            failures = await users_repo.bump_refresh_failures(book, user_id, provider)
            if revoked:
                await users_repo.mark_token_revoked(book, user_id, provider)
            return failures
    except Exception as exc:  # noqa: BLE001 - the real error is the caller's
        log.error(
            "auth.refresh_bookkeeping_failed",
            user_id=user_id,
            provider=provider,
            error=str(exc),
        )
        return 0


# --------------------------------------------------------------------------- #
# the token endpoint
# --------------------------------------------------------------------------- #


async def _post_refresh(refresh_token: str) -> dict[str, Any]:
    """Spend a refresh token. Raises the classified AppError on any failure."""
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
    }
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.post(
                TOKEN_ENDPOINT,
                data=form,
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "Google's token endpoint did not respond.",
            details={"cause": type(exc).__name__},
        ) from exc

    if response.status_code == 200:
        try:
            return dict(response.json())
        except ValueError as exc:
            raise AppError(
                "GOOGLE_UNAVAILABLE",
                "Google's token endpoint returned something we could not read.",
            ) from exc

    try:
        body = dict(response.json())
    except ValueError:
        body = {}
    error = str(body.get("error") or "")
    description = str(body.get("error_description") or "")[:200]

    if error in _DEAD_GRANT:
        raise _reauth("invalid_grant", google_error=error, detail=description)
    if error in _MISCONFIGURED:
        raise AppError(
            "INTERNAL",
            "Our Google client credentials were rejected.",
            details={"google_error": error, "detail": description},
        )
    raise AppError(
        "GOOGLE_UNAVAILABLE",
        "Google could not refresh the connection right now.",
        details={
            "status": response.status_code,
            "google_error": error or None,
            "detail": description or None,
        },
    )


# --------------------------------------------------------------------------- #
# the public surface
# --------------------------------------------------------------------------- #


async def refresh_access_token(
    session: AsyncSession,
    user_id: str,
    provider: str = "google",
    *,
    row: OAuthToken | None = None,
) -> str:
    """Spend the refresh token and store the result. Returns the new token.

    Callers normally want :func:`get_access_token`, which only lands here when
    the stored token is actually stale.
    """
    token_row = row if row is not None else await users_repo.get_token(
        session, user_id, provider
    )
    if token_row is None:
        raise _reauth("no_grant")
    if token_row.revoked_at is not None:
        raise _reauth("revoked")
    if token_row.refresh_failures >= MAX_REFRESH_FAILURES:
        raise _reauth("too_many_failures", failures=int(token_row.refresh_failures))
    if token_row.refresh_token_enc is None:
        raise _reauth("no_refresh_token")

    refresh_token = crypto.decrypt(
        token_row.refresh_token_enc, token_row.key_version
    )

    try:
        payload = await _post_refresh(refresh_token)
    except AppError as exc:
        dead = exc.code == "GOOGLE_REAUTH_REQUIRED"
        failures = await _record_failure(user_id, provider, revoked=dead)
        log.warning(
            "auth.refresh_failed",
            user_id=user_id,
            provider=provider,
            code=exc.code,
            revoked=dead,
            failures=failures,
        )
        if not dead and failures >= MAX_REFRESH_FAILURES:
            # Enough consecutive failures that "try again later" has stopped
            # being true. Ask for a reconnect instead of looping forever.
            raise _reauth("too_many_failures", failures=failures) from exc
        raise

    access_token = str(payload.get("access_token") or "")
    if not access_token:
        await _record_failure(user_id, provider, revoked=False)
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "Google's refresh response carried no access token.",
        )

    # Re-encrypt under the current key while we are writing anyway; that is how
    # a key rotation drains without a downtime window.
    blob, key_version = crypto.encrypt_with_version(access_token)
    expires_at = _expiry_from(payload.get("expires_in"))

    # Google rotates refresh tokens on some grants. When it hands us a new one,
    # keep it; the repository coalesces a NULL so ours is never wiped.
    new_refresh = payload.get("refresh_token")
    if new_refresh:
        refresh_blob, _ = crypto.encrypt_with_version(str(new_refresh))
        await users_repo.upsert_token(
            session,
            user_id,
            provider=provider,
            access_token_enc=blob,
            refresh_token_enc=refresh_blob,
            scopes=_scope_list(payload, token_row),
            expires_at=expires_at,
            provider_account_id=token_row.provider_account_id,
            key_version=key_version,
        )
    else:
        await users_repo.update_access_token(
            session,
            user_id,
            provider=provider,
            access_token_enc=blob,
            expires_at=expires_at,
            key_version=key_version,
        )

    log.info(
        "auth.refreshed",
        user_id=user_id,
        provider=provider,
        expires_at=expires_at.isoformat() if expires_at else None,
    )
    return access_token


def _scope_list(payload: dict[str, Any], row: OAuthToken) -> list[str]:
    """Scopes from the refresh response, falling back to what we already hold."""
    granted = str(payload.get("scope") or "").split()
    return granted or list(row.scopes or [])


async def get_access_token(
    session: AsyncSession, user_id: str, provider: str = "google"
) -> str:
    """A usable access token for ``user_id``.

    Refreshes when the stored one expires within five minutes. Raises
    ``GOOGLE_REAUTH_REQUIRED`` (428) when the grant is gone, and
    ``GOOGLE_UNAVAILABLE`` (503) when Google is merely having a moment.
    """
    row = await users_repo.get_token(session, user_id, provider)
    if row is None:
        raise _reauth("no_grant")
    if row.revoked_at is not None:
        raise _reauth("revoked")

    if not _is_stale(row.expires_at):
        return crypto.decrypt(row.access_token_enc, row.key_version)

    async with _lock_for(user_id, provider):
        # Someone may have refreshed while we waited for the lock.
        fresh = await users_repo.get_token(session, user_id, provider)
        if fresh is None:
            raise _reauth("no_grant")
        if fresh.revoked_at is not None:
            raise _reauth("revoked")
        if not _is_stale(fresh.expires_at):
            return crypto.decrypt(fresh.access_token_enc, fresh.key_version)
        return await refresh_access_token(session, user_id, provider, row=fresh)


async def revoke_remote(token: str) -> bool:
    """Tell Google to drop a grant. ``False`` when the call did not land.

    Used by ``DELETE /api/v1/auth/google``. Local state is cleared either way —
    a disconnect the user asked for must not depend on Google answering.
    """
    if not token:
        return False
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.post(
                REVOKE_ENDPOINT,
                data={"token": token},
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        log.warning("auth.revoke_unreachable", error=type(exc).__name__)
        return False
    # 200 is done. 400 with invalid_token means it was already gone, which is
    # the state we were asking for.
    if response.status_code == 200:
        return True
    if response.status_code == 400:
        try:
            body = dict(response.json())
        except ValueError:
            body = {}
        return str(body.get("error") or "") == "invalid_token"
    log.warning("auth.revoke_rejected", status=response.status_code)
    return False


async def refresh_token_plaintext(
    session: AsyncSession, user_id: str, provider: str = "google"
) -> str | None:
    """The stored refresh token, decrypted.

    Exists for one caller — the disconnect endpoint, which has to hand it to
    Google's revoke endpoint. Nothing else may use it, and it never leaves the
    process.
    """
    row = await users_repo.get_token(session, user_id, provider)
    if row is None:
        return None
    if row.refresh_token_enc is not None:
        return crypto.decrypt(row.refresh_token_enc, row.key_version)
    try:
        return crypto.decrypt(row.access_token_enc, row.key_version)
    except AppError:
        return None


__all__ = [
    "TOKEN_ENDPOINT",
    "REVOKE_ENDPOINT",
    "REFRESH_SKEW_S",
    "MAX_REFRESH_FAILURES",
    "get_access_token",
    "refresh_access_token",
    "refresh_token_plaintext",
    "revoke_remote",
    "needs_reauth",
]
