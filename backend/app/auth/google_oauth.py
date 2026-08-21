"""Connecting a Google account: the redirect out, and the callback back.

Authorization code with PKCE (S256). The client secret already authenticates
the token exchange for a confidential web client, so PKCE is belt and braces —
it costs one hash and it closes the code-interception window the day someone
moves this redirect to a public client.

The handshake in full:

1. :func:`begin` mints a ``state`` nanoid and a PKCE verifier, holds them in
   Redis for ten minutes against a nonce we also hand back as a short-lived
   cookie, and returns the Google URL to redirect to.
2. Google sends the browser back with ``code`` and ``state``.
3. :func:`handle_callback` **deletes the state before doing anything else**, so
   a replayed callback finds nothing and gets a 422. It then checks the nonce
   matches the cookie, exchanges the code, verifies the granted scopes are
   actually the ones we asked for, reads the profile and the calendar
   timezone, upserts the user and the encrypted token row, writes an
   ``auth.grant`` audit row and queues the first backfill.

Nothing here ever returns token material to a caller.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Final, Iterable
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core import audit, cache, crypto
from app.core.errors import AppError
from app.core.ids import canonical_json, new_id
from app.core.logging import get_logger
from app.db.repositories import sync_state as sync_state_repo
from app.db.repositories import users as users_repo

log = get_logger(__name__)

AUTH_ENDPOINT: Final[str] = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT: Final[str] = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT: Final[str] = "https://www.googleapis.com/oauth2/v3/userinfo"
CALENDAR_SETTINGS_ENDPOINT: Final[str] = (
    "https://www.googleapis.com/calendar/v3/users/me/settings"
)

STATE_TTL_S: Final[int] = 600  # ten minutes, same as the handshake cookie
STATE_NAMESPACE: Final[str] = "oauth"
HTTP_TIMEOUT_S: Final[float] = 10.0

# What we ask for. Read scopes fill the mirror; write scopes prepare drafts and
# calendar changes that a person still has to approve.
SCOPES: Final[tuple[str, ...]] = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
)

# Google lets a person untick individual scopes on the consent screen. These
# are the ones the system genuinely cannot work without — dropping any of them
# produces a product that looks broken in a confusing way, so we fail at the
# door instead.
REQUIRED_SCOPES: Final[tuple[str, ...]] = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
)

# Asked for, but the system degrades rather than breaks without them.
OPTIONAL_SCOPES: Final[tuple[str, ...]] = tuple(
    s for s in SCOPES if s not in REQUIRED_SCOPES
)

# Google normalises and widens scopes: it hands back ``userinfo.email`` for
# ``email``, and a broad scope in place of the narrow ones it covers. Checking
# for string equality against what we asked for would fail on a grant that is
# in fact wider than we need. This maps "a granted scope" to "what it covers".
_COVERS: Final[dict[str, frozenset[str]]] = {
    "https://mail.google.com/": frozenset(
        {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
        }
    ),
    "https://www.googleapis.com/auth/gmail.modify": frozenset(
        {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.send",
        }
    ),
    "https://www.googleapis.com/auth/gmail.compose": frozenset(
        {"https://www.googleapis.com/auth/gmail.send"}
    ),
    "https://www.googleapis.com/auth/drive": frozenset(
        {
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        }
    ),
    "https://www.googleapis.com/auth/calendar": frozenset(
        {
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.settings.readonly",
        }
    ),
    "https://www.googleapis.com/auth/userinfo.email": frozenset({"email"}),
    "https://www.googleapis.com/auth/userinfo.profile": frozenset({"profile"}),
}

# Calendar's weekStart is 0=Sunday, 1=Monday, 2=Saturday. Ours is ISO:
# 1=Monday .. 7=Sunday.
_WEEK_START: Final[dict[str, int]] = {"0": 7, "1": 1, "2": 6}

# The three services a fresh grant backfills, and the task that does each.
BACKFILL_TASKS: Final[tuple[tuple[str, str], ...]] = (
    ("gmail", "sync.gmail"),
    ("gcal", "sync.gcal"),
    ("gdrive", "sync.gdrive"),
)
SYNC_QUEUE: Final[str] = "sync"


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuthStart:
    """What the ``GET /api/v1/auth/google`` handler needs to send a browser off."""

    url: str
    state: str
    binding: str  # goes in the short-lived handshake cookie
    next_path: str


@dataclass(frozen=True)
class Grant:
    """A completed connection. No token material, by construction."""

    user_id: str
    email: str
    display_name: str | None
    timezone: str
    work_week_start: int
    provider_account_id: str | None
    scopes: list[str]
    expires_at: dt.datetime | None
    next_path: str
    is_new_user: bool
    queued: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small pieces
# --------------------------------------------------------------------------- #


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _pkce_pair() -> tuple[str, str]:
    """A PKCE verifier and its S256 challenge."""
    verifier = _b64u(secrets.token_bytes(64))  # 86 chars, inside the 43..128 range
    challenge = _b64u(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def safe_next(next_path: str | None) -> str:
    """Where to land after the callback, validated as same-origin.

    Only a relative path is allowed. ``//evil.example`` and
    ``https://evil.example`` are both open redirects wearing different hats, so
    both are refused.
    """
    if not next_path:
        return "/"
    candidate = str(next_path).strip()
    if not candidate:
        return "/"
    if any(ch in candidate for ch in ("\r", "\n", "\t", "\\")):
        raise AppError.validation("That redirect target is not allowed.", next=candidate)
    if not candidate.startswith("/") or candidate.startswith("//"):
        raise AppError.validation(
            "The redirect target must be a relative path.", next=candidate
        )
    return candidate


def _expand(scope: str) -> frozenset[str]:
    return frozenset({scope}) | _COVERS.get(scope, frozenset())


def missing_scopes(granted: Iterable[str]) -> list[str]:
    """Which required scopes this grant does not cover, widening as it goes."""
    covered: set[str] = set()
    for scope in granted:
        covered |= _expand(scope)
    return [s for s in REQUIRED_SCOPES if s not in covered]


def _state_key(state: str) -> str:
    return cache.key(STATE_NAMESPACE, f"state:{state}")


def _expiry_from(expires_in: Any) -> dt.datetime | None:
    """``expires_in`` seconds, as an absolute UTC instant.

    A missing or unreadable value gives ``None`` rather than "now", so the
    token store treats it as stale and refreshes instead of handing out
    something that expired the moment it was written.
    """
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)


# --------------------------------------------------------------------------- #
# step one: the redirect out
# --------------------------------------------------------------------------- #


async def begin(
    *,
    next_path: str = "/",
    force_consent: bool = True,
    login_hint: str | None = None,
    link_to_user_id: str | None = None,
) -> AuthStart:
    """Start the handshake. Returns the URL to 302 to, and the cookie binding.

    ``force_consent`` sends ``prompt=consent``, which is what makes Google
    issue a refresh token. Leave it on for a first grant. Turn it off only for
    a returning user whose refresh token we already hold, so they are not asked
    to approve the same thing twice.

    ``link_to_user_id`` is the account already signed in. When it is set, the
    workspace is attached to *that* account whatever address Google comes back
    with — connecting your work Google to a personal login is a normal thing to
    want, and matching on the address would silently strand the data on a
    second account instead.
    """
    target = safe_next(next_path)
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise AppError(
            "INTERNAL",
            "Google sign-in is not configured on this server.",
            details={"setting": "GOOGLE_CLIENT_ID"},
        )

    state = new_id()
    binding = new_id()
    verifier, challenge = _pkce_pair()

    payload = canonical_json(
        {
            "code_verifier": verifier,
            "binding": binding,
            "next": target,
            "link_to_user_id": link_to_user_id or "",
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }
    ).encode("utf-8")

    try:
        client = await cache.get_redis()
        await client.set(_state_key(state), payload, ex=STATE_TTL_S)
    except Exception as exc:  # noqa: BLE001 - redis is not optional for this
        log.error("auth.state_store_failed", error=str(exc))
        raise AppError(
            "INTERNAL", "Could not start sign-in. Try again in a moment."
        ) from exc

    params: dict[str, str] = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "include_granted_scopes": "true",
    }
    if force_consent:
        params["prompt"] = "consent"
    if login_hint:
        params["login_hint"] = login_hint

    return AuthStart(
        url=f"{AUTH_ENDPOINT}?{urlencode(params)}",
        state=state,
        binding=binding,
        next_path=target,
    )


async def _take_state(state: str) -> dict[str, Any]:
    """Read the state and delete it in the same breath.

    A replay finds nothing. That is the whole point, so this happens before any
    other work in the callback.
    """
    key = _state_key(state)
    try:
        client = await cache.get_redis()
        try:
            raw = await client.getdel(key)
        except AttributeError:  # pragma: no cover - very old redis-py
            raw = await client.get(key)
            await client.delete(key)
    except Exception as exc:  # noqa: BLE001
        log.error("auth.state_read_failed", error=str(exc))
        raise AppError(
            "INTERNAL", "Could not finish sign-in. Try again in a moment."
        ) from exc

    if not raw:
        raise AppError.validation(
            "That sign-in link has expired or was already used. Start again."
        )
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError) as exc:
        raise AppError.validation("That sign-in link is not valid.") from exc


# --------------------------------------------------------------------------- #
# step two: the callback
# --------------------------------------------------------------------------- #


async def _exchange_code(code: str, verifier: str) -> dict[str, Any]:
    """Authorization code -> tokens."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code_verifier": verifier,
    }
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.post(
                TOKEN_ENDPOINT, data=form, headers={"Accept": "application/json"}
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
    detail = str(body.get("error_description") or "")[:200]

    if response.status_code >= 500:
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "Google could not complete the sign-in right now.",
            details={"status": response.status_code, "google_error": error or None},
        )
    if error in {"invalid_client", "unauthorized_client"}:
        raise AppError(
            "INTERNAL",
            "Our Google client credentials were rejected.",
            details={"google_error": error, "detail": detail},
        )
    raise AppError.validation(
        "Google would not accept that sign-in. Start again.",
        google_error=error or None,
        detail=detail or None,
    )


async def _fetch_json(url: str, access_token: str) -> dict[str, Any] | None:
    """A GET with the bearer token. ``None`` on anything other than a clean 200."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
    except httpx.HTTPError as exc:
        log.warning("auth.fetch_failed", url=url, cause=type(exc).__name__)
        return None
    if response.status_code != 200:
        log.warning("auth.fetch_rejected", url=url, status=response.status_code)
        return None
    try:
        return dict(response.json())
    except ValueError:
        return None


async def _fetch_profile(access_token: str) -> dict[str, Any]:
    """Who this is. Without an email there is no user, so this one is fatal."""
    profile = await _fetch_json(USERINFO_ENDPOINT, access_token)
    if not profile or not profile.get("email"):
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "Google would not tell us who you are. Try connecting again.",
        )
    return profile


async def _fetch_calendar_prefs(access_token: str) -> tuple[str | None, int | None]:
    """The user's calendar timezone and week start.

    Every date phrase in the system is resolved against this timezone in
    Python, so it is worth one call at sign-in. A failure is not fatal — we
    fall back to UTC and Monday and the user can change both.
    """
    payload = await _fetch_json(CALENDAR_SETTINGS_ENDPOINT, access_token)
    if not payload:
        return None, None

    values: dict[str, str] = {}
    for item in payload.get("items") or []:
        if isinstance(item, dict) and item.get("id") is not None:
            values[str(item["id"])] = str(item.get("value") or "")

    timezone: str | None = values.get("timezone") or None
    if timezone:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            log.warning("auth.unknown_timezone", timezone=timezone)
            timezone = None

    week_start = _WEEK_START.get(values.get("weekStart", ""))
    return timezone, week_start


async def enqueue_backfill(session: AsyncSession, user_id: str) -> list[str]:
    """Create the sync bookkeeping rows and queue the first pass.

    The rows go in inside the caller's transaction; the tasks are sent after,
    by the caller's commit, only in the sense that a task which starts early
    finds the row already there — ``ensure_state`` is a find-or-create and the
    sync tasks call it themselves.

    A broker that is down does not fail a sign-in. Beat picks the user up
    within fifteen minutes either way.
    """
    for service, _task in BACKFILL_TASKS:
        await sync_state_repo.ensure_state(session, user_id, service)

    try:
        from app.tasks.celery_app import celery_app  # local: workers own this module
    except Exception as exc:  # noqa: BLE001 - queue unavailable is not fatal here
        log.warning("auth.backfill_not_queued", user_id=user_id, error=str(exc))
        return []

    queued: list[str] = []
    for _service, task_name in BACKFILL_TASKS:
        try:
            await asyncio.to_thread(
                celery_app.send_task,
                task_name,
                kwargs={"user_id": user_id, "mode": "backfill"},
                queue=SYNC_QUEUE,
            )
            queued.append(task_name)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "auth.backfill_enqueue_failed",
                user_id=user_id,
                task=task_name,
                error=str(exc),
            )
    return queued


async def handle_callback(
    session: AsyncSession,
    *,
    code: str,
    state: str,
    binding: str | None,
    ip: str | None = None,
    user_agent: str | None = None,
) -> Grant:
    """Finish the handshake and land the user.

    Everything from here to the return happens in the caller's transaction, so
    the user row, the token row and the audit row commit together or not at all.
    """
    if not code or not state:
        raise AppError.validation("That sign-in did not come back complete.")

    stored = await _take_state(state)  # deleted before anything else happens

    expected_binding = str(stored.get("binding") or "")
    if not binding or not hmac.compare_digest(expected_binding, str(binding)):
        log.warning("auth.state_binding_mismatch", state=state)
        raise AppError.validation(
            "That sign-in started in a different browser. Start again."
        )

    verifier = str(stored.get("code_verifier") or "")
    next_path = safe_next(str(stored.get("next") or "/"))

    tokens = await _exchange_code(code, verifier)

    access_token = str(tokens.get("access_token") or "")
    if not access_token:
        raise AppError(
            "GOOGLE_UNAVAILABLE", "Google's sign-in response carried no access token."
        )
    refresh_token = str(tokens.get("refresh_token") or "") or None
    granted = str(tokens.get("scope") or "").split() or list(SCOPES)

    absent = missing_scopes(granted)
    if absent:
        log.warning("auth.scopes_declined", missing=absent)
        raise AppError.validation(
            "Some permissions were not granted, so the connection would not work. "
            "Connect again and leave every box ticked.",
            missing_scopes=absent,
        )

    profile = await _fetch_profile(access_token)
    email = str(profile["email"])
    display_name = str(profile["name"]) if profile.get("name") else None
    provider_account_id = str(profile["sub"]) if profile.get("sub") else None

    timezone, week_start = await _fetch_calendar_prefs(access_token)

    # Three ways to land on an account, in order of how sure we are:
    #
    #   1. Someone is signed in and asked to connect — it is theirs, whatever
    #      address the workspace turns out to use.
    #   2. This workspace has been connected before — same person coming back,
    #      even if they have since changed the address on either side.
    #   3. Nobody is signed in and the address matches an account — this is
    #      "Continue with Google" for someone who signed up with a password,
    #      and it has to land on the account they already have, not a copy.
    linked = str(stored.get("link_to_user_id") or "")
    user = await users_repo.get_user(session, linked) if linked else None
    if user is None and provider_account_id:
        user = await users_repo.get_user_by_provider_account(
            session, provider="google", provider_account_id=provider_account_id
        )

    if user is not None:
        is_new_user = False
        await users_repo.adopt_google_profile(
            session,
            user.id,
            display_name=display_name,
            timezone=timezone,
            work_week_start=week_start,
        )
    else:
        is_new_user = await users_repo.get_user_by_email(session, email) is None
        user = await users_repo.upsert_user_by_email(
            session,
            email,
            display_name=display_name,
            timezone=timezone,
            work_week_start=week_start,
        )

    if refresh_token is None:
        held = await users_repo.get_token(session, user.id)
        if held is None or held.refresh_token_enc is None:
            # Without a refresh token the connection dies in an hour and the
            # user has no idea why. Say so now, and send them back through
            # consent, which is what makes Google issue one.
            raise AppError.validation(
                "Google did not grant offline access, so the connection would "
                "expire within the hour. Connect again and approve the consent "
                "screen."
            )

    access_blob, key_version = crypto.encrypt_with_version(access_token)
    refresh_blob = (
        crypto.encrypt(refresh_token) if refresh_token is not None else None
    )
    expires_at = _expiry_from(tokens.get("expires_in"))

    await users_repo.upsert_token(
        session,
        user.id,
        account_email=email,
        provider="google",
        access_token_enc=access_blob,
        refresh_token_enc=refresh_blob,
        scopes=granted,
        expires_at=expires_at,
        provider_account_id=provider_account_id,
        key_version=key_version,
    )

    await audit.record(
        session,
        user.id,
        actor="user",
        action="auth.grant",
        resource_id=provider_account_id,
        payload={"email": email, "scopes": granted, "new_user": is_new_user},
        status="ok",
        ip=ip,
        ua=user_agent,
    )

    queued = await enqueue_backfill(session, user.id)

    log.info(
        "auth.granted",
        user_id=user.id,
        new_user=is_new_user,
        timezone=user.timezone,
        queued=queued,
    )

    return Grant(
        user_id=user.id,
        email=str(user.email),
        display_name=user.display_name,
        timezone=user.timezone,
        work_week_start=int(user.work_week_start),
        provider_account_id=provider_account_id,
        scopes=granted,
        expires_at=expires_at,
        next_path=next_path,
        is_new_user=is_new_user,
        queued=queued,
    )


__all__ = [
    "AUTH_ENDPOINT",
    "TOKEN_ENDPOINT",
    "USERINFO_ENDPOINT",
    "CALENDAR_SETTINGS_ENDPOINT",
    "SCOPES",
    "REQUIRED_SCOPES",
    "OPTIONAL_SCOPES",
    "STATE_TTL_S",
    "BACKFILL_TASKS",
    "SYNC_QUEUE",
    "AuthStart",
    "Grant",
    "begin",
    "handle_callback",
    "enqueue_backfill",
    "missing_scopes",
    "safe_next",
]
