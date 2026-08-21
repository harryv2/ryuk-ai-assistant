"""Connecting and disconnecting a Google account.

Google redirects to the **backend**, never to the UI, because the client secret
lives here. The backend exchanges the code, sets the session cookie and then
bounces the browser to the app.

Two things this module will not do. It never puts token material in a response
— access and refresh tokens are AES-256-GCM blobs in ``oauth_tokens`` and only
``auth/token_store.py`` can read them. And it never renders an error page: a
failed handshake redirects back to the app with ``?auth_error=<reason>``, so
the SPA owns the message.

The handshake itself — PKCE, the single-use ``state``, the scope check — lives
in :mod:`app.auth.google_oauth`. This file is the HTTP shell around it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas import iso
from app.auth import deps, google_oauth, token_store
from app.auth.deps import CurrentUser, SessionDep
from app.config import settings
from app.core import audit, ratelimit
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.repositories import actions as action_repo
from app.db.repositories import mirror as mirror_repo
from app.db.repositories import prompts as prompt_repo
from app.db.repositories import users as users_repo

log = get_logger(__name__)
router = APIRouter(tags=["auth"])

#: Statuses whose Gmail drafts are still sitting in the person's mailbox.
LIVE_ACTION_STATUSES = ("draft", "approved")


def _app_url(path: str, **params: str) -> str:
    """Somewhere in the SPA. ``path`` has already been checked as relative."""
    base = settings.APP_URL.rstrip("/")
    query = f"?{urlencode(params)}" if params else ""
    return f"{base}{path}{query}"


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip() or None
    return request.client.host if request.client else None


# --------------------------------------------------------------------------- #
# GET /auth/google
# --------------------------------------------------------------------------- #


@router.get("/auth/google")
async def start(request: Request, next: str = Query("/")) -> RedirectResponse:
    """Send the browser to Google's consent screen.

    One route, two jobs, and the difference is whether anybody is signed in.
    A stranger is signing in *with* Google. Somebody already signed in is
    connecting a workspace *to* the account they are using — so their id rides
    along in the handshake and the connection lands there, whatever address the
    workspace turns out to use.

    ``next`` must be a relative path. ``//evil.example`` and
    ``https://evil.example`` are the same open redirect wearing different hats,
    and :func:`google_oauth.safe_next` refuses both.
    """
    started = await google_oauth.begin(
        next_path=next,
        link_to_user_id=deps.optional_user_id(request),
    )
    response = RedirectResponse(started.url, status_code=302)
    deps.set_oauth_cookie(response, started.binding)
    return response


# --------------------------------------------------------------------------- #
# GET /auth/google/callback
# --------------------------------------------------------------------------- #


@router.get("/auth/google/callback")
async def callback(
    request: Request,
    session: SessionDep,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
) -> RedirectResponse:
    """Google's redirect target. Not called by a client directly."""
    if error:
        log.info("auth.declined", reason=error[:60])
        return _bounce(_app_url("/", auth_error=error))

    if not code or not state:
        return _bounce(_app_url("/", auth_error="incomplete"))

    try:
        grant = await google_oauth.handle_callback(
            session,
            code=code,
            state=state,
            binding=deps.read_oauth_cookie(request),
            ip=_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
        )
    except AppError as exc:
        log.warning("auth.callback_failed", code=exc.code)
        return _bounce(_app_url("/", auth_error=exc.code.lower()))

    await session.commit()

    response = RedirectResponse(_app_url(grant.next_path), status_code=302)
    deps.set_session_cookie(response, grant.user_id)
    deps.clear_oauth_cookie(response)
    log.info("auth.signed_in", user_id=grant.user_id, new_user=grant.is_new_user)
    return response


def _bounce(url: str) -> RedirectResponse:
    """Back to the app, with the handshake cookie cleared either way."""
    response = RedirectResponse(url, status_code=302)
    deps.clear_oauth_cookie(response)
    return response


# --------------------------------------------------------------------------- #
# GET /auth/me
# --------------------------------------------------------------------------- #


@router.get("/auth/me")
async def me(session: SessionDep, user: CurrentUser) -> dict[str, Any]:
    """Who is signed in, and whether the Google grant is healthy.

    ``needs_reauth: true`` means the next query would return 428 — show the
    reconnect banner before the person hits the wall rather than after.
    """
    token = await users_repo.get_token(session, user.id)
    _, remaining, _reset = await ratelimit.check_query_limit(user.id, consume=False)

    return {
        "user": {
            "id": user.id,
            "email": str(user.email),
            "display_name": user.display_name,
            "has_password": bool(user.password_hash),
            "timezone": user.timezone,
            "work_week_start": int(user.work_week_start),
            "created_at": iso(user.created_at),
        },
        "google": {
            "connected": token is not None and token.revoked_at is None,
            # Which workspace, not which account — since they can differ, the
            # page has to say which one it is actually reading.
            "account_email": getattr(token, "account_email", None),
            "provider_account_id": getattr(token, "provider_account_id", None),
            "scopes": list(getattr(token, "scopes", None) or []),
            "expires_at": iso(getattr(token, "expires_at", None)),
            "needs_reauth": token_store.needs_reauth(token),
        },
        "limits": {
            "queries_per_hour": settings.RATE_LIMIT_PER_HOUR,
            "remaining_this_hour": max(0, int(remaining)),
        },
    }


# --------------------------------------------------------------------------- #
# DELETE /auth/google
# --------------------------------------------------------------------------- #


@router.delete("/auth/google")
async def disconnect(
    request: Request,
    response: Response,
    session: SessionDep,
    user: CurrentUser,
    purge: bool = Query(True),
) -> dict[str, Any]:
    """Disconnect, and optionally drop the mirror.

    This is a disconnect, not an account deletion. Conversations, messages,
    runs and audit rows survive — "no hard delete" holds everywhere except the
    mirror, which is a cache by definition and rebuilds from a resync.

    Order matters. The prepared writes are cancelled and their Gmail drafts
    deleted **first**, while the grant still works; revoking before that would
    leave half-written cancellations in the person's Drafts folder with no way
    left to reach them.
    """
    cancelled_actions, cancelled_prompts = await _stand_down(session, user.id)

    revoked_remotely = False
    refresh = await token_store.refresh_token_plaintext(session, user.id)
    if refresh:
        revoked_remotely = await token_store.revoke_remote(refresh)

    now = datetime.now(UTC)
    await users_repo.mark_token_revoked(session, user.id, at=now)
    await users_repo.delete_token(session, user.id, provider="google")

    purged: dict[str, int] = {}
    if purge:
        purged = await mirror_repo.purge_user(session, user.id)
        from sqlalchemy import delete as sql_delete

        from app.db.models import SyncState

        result = await session.execute(
            sql_delete(SyncState).where(SyncState.user_id == user.id)
        )
        purged["sync_state"] = int(result.rowcount or 0)

    conversations_kept = await _count_conversations(session, user.id)

    await audit.record(
        session,
        user.id,
        actor="user",
        action="auth.revoke",
        payload={"purged": purged, "revoked_remotely": revoked_remotely},
        status="ok",
        ip=_client_ip(request),
        ua=request.headers.get("User-Agent"),
    )
    await session.commit()

    deps.clear_session_cookie(response)
    log.info("auth.disconnected", user_id=user.id, purged=bool(purge))
    return {
        "disconnected": True,
        "revoked_at": iso(now),
        "revoked_remotely": revoked_remotely,
        "purged": purged,
        "cancelled_actions": cancelled_actions,
        "cancelled_prompts": cancelled_prompts,
        "conversations_kept": conversations_kept,
    }


async def _stand_down(session: AsyncSession, user_id: str) -> tuple[int, int]:
    """Close every open card and cancel every write it was gating."""
    open_cards = await prompt_repo.list_prompts(
        session, user_id, status="pending", limit=500
    )
    cancelled_actions = 0
    for card in open_cards:
        gated = await action_repo.list_for_prompt(session, user_id, card.id)
        await _discard_drafts(session, user_id, gated)
        cancelled_actions += await action_repo.cancel_for_prompt(
            session, user_id, card.id, reason="disconnected"
        )
        await prompt_repo.cancel_prompt(session, user_id, card.id)

    # A write can outlive its card — an approved action whose prompt was
    # answered days ago. Sweep those too, or a worker picks one up after the
    # grant is gone and fails loudly for no reason.
    for status in LIVE_ACTION_STATUSES:
        for action in await action_repo.list_actions(
            session, user_id, status=status, limit=500
        ):
            await _discard_drafts(session, user_id, [action])
            await action_repo.cancel_action(
                session, user_id, action.id, reason="disconnected"
            )
            cancelled_actions += 1

    return cancelled_actions, len(open_cards)


async def _discard_drafts(session: AsyncSession, user_id: str, rows: list[Any]) -> None:
    """Delete the Gmail drafts behind cancelled writes. Best effort."""
    from app.api.v1.prompts import DRAFT_BACKED

    targets = [r for r in rows if r.op in DRAFT_BACKED and r.external_ref]
    if not targets:
        return
    try:
        from app.google.client import clients_for
        from app.services import gmail as gmail_service

        clients = await clients_for(session, user_id)
    except Exception as exc:
        log.warning("auth.draft_cleanup_skipped", user_id=user_id, error=str(exc))
        return
    for row in targets:
        try:
            await gmail_service.delete_draft(clients, str(row.external_ref))
        except Exception as exc:
            log.warning("auth.draft_not_deleted", action_id=row.id, error=str(exc))


async def _count_conversations(session: AsyncSession, user_id: str) -> int:
    from sqlalchemy import func, select

    from app.db.models import Conversation

    result = await session.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
    )
    return int(result.scalar_one())


# --------------------------------------------------------------------------- #
# POST /auth/logout
# --------------------------------------------------------------------------- #


@router.post("/auth/logout", status_code=204)
async def logout(response: Response) -> Response:
    """Sign this browser out. The Google grant is untouched — signing out is
    not disconnecting, and conflating the two loses a 180-day backfill."""
    deps.clear_session_cookie(response)
    response.status_code = 204
    return response


__all__ = ["router"]


# --------------------------------------------------------------------------- #
# Email + password, with a code to the address
#
# Google sign-in stays exactly as it is above. This is a second door, for people
# who do not want to connect a Google account, and so that several accounts can
# be used on one machine.
#
# Two rules run through all of it:
#
#   * A wrong password and an unknown address answer identically. Anything else
#     turns the sign-in form into a way to find out who has an account here.
#   * The code proves the mailbox. That is the same thing Google proves, which is
#     why an account created through Google may add a password later without any
#     further ceremony — it is not a takeover, it is the same person.
# --------------------------------------------------------------------------- #

from pydantic import BaseModel, EmailStr, Field  # noqa: E402

from app.auth import otp as otp_store  # noqa: E402
from app.auth import passwords  # noqa: E402
from app.core.ids import new_id  # noqa: E402


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class EmailOnly(BaseModel):
    email: EmailStr


class OtpSubmission(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=12)


#: The same answer for a wrong password and an address nobody has registered.
_REFUSED = "That email and password do not match."


#: Sign-in attempts allowed per address, and per caller, in fifteen minutes.
#: Separate from the 100-queries-an-hour product limit — that one meters use,
#: this one stops guessing.
_AUTH_ATTEMPTS = 10
_AUTH_WINDOW_S = 900


async def _guard(request: Request, email: str, what: str) -> None:
    """Rate limit by address AND by caller.

    Both are needed. Per-address alone lets one host work through a list of
    addresses; per-caller alone lets a botnet share the work on one address.
    """
    from app.core.cache import get_redis

    redis = await get_redis()
    for scope in (email.lower(), _client_ip(request) or "-"):
        key = f"auth:{what}:{scope}"
        count = await redis.incr(key)
        if int(count) == 1:
            await redis.expire(key, _AUTH_WINDOW_S)
        if int(count) > _AUTH_ATTEMPTS:
            ttl = await redis.ttl(key)
            raise AppError(
                "RATE_LIMITED",
                "Too many attempts. Try again shortly.",
                details={"retry_after": max(1, int(ttl or _AUTH_WINDOW_S))},
            )


def _pending(issued: otp_store.Issued) -> dict[str, Any]:
    body: dict[str, Any] = {
        "email": issued.email,
        "otp_required": True,
        "expires_in": issued.expires_in,
        "attempts_allowed": otp_store.MAX_ATTEMPTS,
    }
    if issued.dev_code:
        # Say it plainly rather than making someone dig through the logs for a
        # code the server already decided is a constant.
        body["dev_mode"] = True
        body["dev_code"] = issued.dev_code
        body["dev_note"] = "Development mode — the code is always 123456."
    return body


@router.post("/auth/register")
async def register(
    body: Credentials, request: Request, session: SessionDep
) -> dict[str, Any]:
    """Start creating an account. Finishes at /auth/verify-otp."""
    email = body.email.lower()
    await _guard(request, email, "register")

    problem = passwords.strength_problem(body.password)
    if problem:
        raise AppError("VALIDATION_ERROR", problem, details={"field": "password"})

    existing = await users_repo.get_user_by_email(session, email)
    if existing is not None and existing.password_hash:
        # Already taken. Saying so would confirm the address to anyone asking, so
        # answer exactly as if we had just created it. The code goes to the real
        # owner, who is the only one who can finish.
        await otp_store.issue(email)
        return _pending(otp_store.Issued(email=email, expires_in=otp_store.TTL_SECONDS,
                                         dev_code=otp_store.DEV_CODE if otp_store.dev_mode() else None))

    # The row is created now, unverified. Nothing about it grants access until a
    # code has been entered — `email_verified` stays false and there is no cookie
    # — so this is a placeholder, not an account. Keeping the password hash here
    # rather than in Redis means a password never sits in a cache.
    await users_repo.stage_password_signup(
        session,
        email=email,
        password_hash=passwords.hash_password(body.password),
        user_id=new_id(),
    )
    issued = await otp_store.issue(email)
    await audit.record(session, user_id=(existing.id if existing else "-"), actor="user",
                       action="auth.register", resource_id=email, payload={"email": email})
    await session.commit()
    return _pending(issued)


@router.post("/auth/login")
async def login(body: Credentials, request: Request, session: SessionDep) -> dict[str, Any]:
    """Check the password, then send a code. The cookie is set at verify-otp."""
    email = body.email.lower()
    await _guard(request, email, "login")

    user = await users_repo.get_user_by_email(session, email)
    # Runs the hash either way, so a missing account costs the same as a wrong
    # password and the timing says nothing.
    if not passwords.verify_password(body.password, user.password_hash if user else None):
        raise AppError("NOT_AUTHENTICATED", _REFUSED)

    issued = await otp_store.issue(email)
    await audit.record(session, user_id=user.id, actor="user", action="auth.login",
                       resource_id=email, payload={"email": email})
    await session.commit()
    return _pending(issued)


@router.post("/auth/verify-otp")
async def verify_otp(
    body: OtpSubmission, request: Request, response: Response, session: SessionDep
) -> dict[str, Any]:
    """The code is right: create or wake the account, and set the cookie."""
    email = body.email.lower()
    await _guard(request, email, "verify")

    result = await otp_store.check(email, body.code)
    if not result.ok:
        user = await users_repo.get_user_by_email(session, email)
        await audit.record(session, user_id=(user.id if user else "-"), actor="user",
                           action="auth.otp_failed", resource_id=email,
                           payload={"reason": result.reason}, status="failed")
        await session.commit()
        message = {
            "expired": "That code has expired. Ask for a new one.",
            "too_many_attempts": "Too many wrong codes. Ask for a new one.",
        }.get(result.reason or "", "That code is not right.")
        raise AppError(
            "PROMPT_VALUE_INVALID", message,
            details={"reason": result.reason, "attempts_remaining": result.attempts_remaining},
        )

    user = await users_repo.get_user_by_email(session, email)
    if user is None:
        raise AppError("NOT_AUTHENTICATED", "Start again from the sign-in screen.")

    await users_repo.mark_signed_in(session, user.id)
    await audit.record(session, user_id=user.id, actor="user", action="auth.otp_verified",
                       resource_id=email, payload={"email": email})
    await session.commit()

    deps.set_session_cookie(response, user.id)
    return await me(session=session, user=user)


@router.post("/auth/resend-otp")
async def resend_otp(body: EmailOnly, request: Request, session: SessionDep) -> dict[str, Any]:
    email = body.email.lower()
    await _guard(request, email, "resend")
    if not await otp_store.may_resend(email):
        raise AppError("RATE_LIMITED", "Too many codes requested. Wait a few minutes.")
    return _pending(await otp_store.issue(email))
