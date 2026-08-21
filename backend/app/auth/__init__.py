"""Signing in, staying signed in, and the Google grant behind it.

Three files, three jobs:

* :mod:`app.auth.google_oauth` — the redirect out to Google, and the callback
  that turns an authorization code into a user row, an encrypted token row and
  a first sync.
* :mod:`app.auth.token_store` — the only place in the system that decrypts a
  token. Hands out a live access token, refreshing when one is about to expire.
* :mod:`app.auth.deps` — the signed session cookie and the FastAPI
  dependencies every endpoint leans on.

Nothing else may decrypt ``oauth_tokens``. Nothing else may mint a session
cookie. Keeping both in one package is what makes that easy to check.
"""

from __future__ import annotations

from app.auth.deps import (
    OAUTH_COOKIE_MAX_AGE,
    SessionDep,
    CurrentUser,
    CurrentUserId,
    clear_oauth_cookie,
    clear_session_cookie,
    current_user,
    current_user_id,
    get_session,
    issue_session,
    oauth_cookie_name,
    optional_user_id,
    read_oauth_cookie,
    read_session,
    session_cookie_name,
    set_oauth_cookie,
    set_session_cookie,
    verify_session,
)
from app.auth.google_oauth import (
    OPTIONAL_SCOPES,
    REQUIRED_SCOPES,
    SCOPES,
    AuthStart,
    Grant,
    begin,
    enqueue_backfill,
    handle_callback,
    missing_scopes,
    safe_next,
)
from app.auth.token_store import (
    REFRESH_SKEW_S,
    MAX_REFRESH_FAILURES,
    get_access_token,
    needs_reauth,
    refresh_access_token,
    revoke_remote,
)

__all__ = [
    # google_oauth
    "SCOPES",
    "REQUIRED_SCOPES",
    "OPTIONAL_SCOPES",
    "AuthStart",
    "Grant",
    "begin",
    "handle_callback",
    "enqueue_backfill",
    "missing_scopes",
    "safe_next",
    # token_store
    "get_access_token",
    "refresh_access_token",
    "revoke_remote",
    "needs_reauth",
    "REFRESH_SKEW_S",
    "MAX_REFRESH_FAILURES",
    # deps
    "get_session",
    "current_user",
    "current_user_id",
    "optional_user_id",
    "issue_session",
    "verify_session",
    "read_session",
    "set_session_cookie",
    "clear_session_cookie",
    "set_oauth_cookie",
    "read_oauth_cookie",
    "clear_oauth_cookie",
    "session_cookie_name",
    "oauth_cookie_name",
    "OAUTH_COOKIE_MAX_AGE",
    "SessionDep",
    "CurrentUser",
    "CurrentUserId",
]
