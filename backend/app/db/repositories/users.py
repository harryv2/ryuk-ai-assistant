"""Users and their OAuth tokens.

Tokens are stored encrypted and are handed back encrypted. Decryption lives
only in ``app.auth.token_store``; nothing here ever sees plaintext.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ids import new_id
from app.db.models import OAuthToken, User
from app.db.repositories import UNSET, ensure_utc, utcnow

# --------------------------------------------------------------------------- #
# users
# --------------------------------------------------------------------------- #


async def get_user(session: AsyncSession, user_id: str) -> User | None:
    """The user row, or None."""
    return await session.get(User, user_id)


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Lookup by email. No user_id argument exists yet at this point — this is
    the one entry point that runs before we know who the caller is."""
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(
    session: AsyncSession,
    email: str,
    *,
    display_name: str | None = None,
    timezone: str = "UTC",
    work_week_start: int = 1,
    user_id: str | None = None,
) -> User:
    """Insert a user. Sign-up path only."""
    now = utcnow()
    user = User(
        id=user_id or new_id(),
        email=email,
        display_name=display_name,
        timezone=timezone,
        work_week_start=work_week_start,
        created_at=now,
        updated_at=now,
    )
    session.add(user)
    await session.flush()
    return user


async def upsert_user_by_email(
    session: AsyncSession,
    email: str,
    *,
    display_name: str | None = None,
    timezone: str | None = None,
    work_week_start: int | None = None,
) -> User:
    """Find-or-create for the OAuth callback.

    An existing row keeps whatever the person has configured; only a missing
    display name is filled in from the provider profile.
    """
    existing = await get_user_by_email(session, email)
    if existing is not None:
        changed = False
        if display_name and not existing.display_name:
            existing.display_name = display_name
            changed = True
        if timezone and existing.timezone == "UTC" and timezone != "UTC":
            existing.timezone = timezone
            changed = True
        if work_week_start and existing.work_week_start != work_week_start:
            existing.work_week_start = work_week_start
            changed = True
        if changed:
            existing.updated_at = utcnow()
            await session.flush()
        return existing

    return await create_user(
        session,
        email,
        display_name=display_name,
        timezone=timezone or "UTC",
        work_week_start=work_week_start or 1,
    )


async def get_user_by_provider_account(
    session: AsyncSession,
    *,
    provider: str,
    provider_account_id: str,
) -> User | None:
    """The account this workspace was last connected to, if any.

    The token row is the durable link between a person here and a workspace at
    Google, and it survives either side changing its address. Matching on it
    first is what stops a returning connection forking into a second account.
    """
    stmt = (
        select(User)
        .join(OAuthToken, OAuthToken.user_id == User.id)
        .where(
            OAuthToken.provider == provider,
            OAuthToken.provider_account_id == provider_account_id,
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


get_user_by_provider_account._no_tenant = True  # type: ignore[attr-defined]


async def adopt_google_profile(
    session: AsyncSession,
    user_id: str,
    *,
    display_name: str | None = None,
    timezone: str | None = None,
    work_week_start: int | None = None,
) -> User | None:
    """Take the settings worth having from a freshly connected workspace.

    Only fills gaps. Somebody who set their own name or timezone here should
    not have it overwritten because a Google profile happened to disagree —
    but a blank one is worth filling, and the calendar knows the timezone and
    the first day of the week better than a default does.
    """
    user = await get_user(session, user_id)
    if user is None:
        return None

    changed = False
    if display_name and not user.display_name:
        user.display_name = display_name
        changed = True
    if timezone and user.timezone == "UTC" and timezone != "UTC":
        user.timezone = timezone
        changed = True
    if work_week_start and user.work_week_start != work_week_start:
        user.work_week_start = work_week_start
        changed = True
    if changed:
        user.updated_at = utcnow()
        await session.flush()
    return user


async def update_user(
    session: AsyncSession,
    user_id: str,
    *,
    display_name: str | None | Any = UNSET,
    timezone: str | Any = UNSET,
    work_week_start: int | Any = UNSET,
) -> User | None:
    """Change the settings that alter how a query is read."""
    values: dict[str, Any] = {}
    if display_name is not UNSET:
        values["display_name"] = display_name
    if timezone is not UNSET:
        values["timezone"] = timezone
    if work_week_start is not UNSET:
        values["work_week_start"] = work_week_start
    if not values:
        return await get_user(session, user_id)

    values["updated_at"] = utcnow()
    result = await session.execute(
        update(User).where(User.id == user_id).values(**values).returning(User)
    )
    return result.scalar_one_or_none()


async def delete_user(session: AsyncSession, user_id: str) -> None:
    """Account disconnect. The only hard delete in the system; every child row
    cascades from here."""
    await session.execute(delete(User).where(User.id == user_id))


# --------------------------------------------------------------------------- #
# oauth_tokens
# --------------------------------------------------------------------------- #


async def get_token(
    session: AsyncSession, user_id: str, provider: str = "google"
) -> OAuthToken | None:
    """The (encrypted) token row for a provider."""
    result = await session.execute(
        select(OAuthToken).where(
            OAuthToken.user_id == user_id, OAuthToken.provider == provider
        )
    )
    return result.scalar_one_or_none()


async def get_live_token(
    session: AsyncSession, user_id: str, provider: str = "google"
) -> OAuthToken | None:
    """The token row only if it has not been revoked."""
    result = await session.execute(
        select(OAuthToken).where(
            OAuthToken.user_id == user_id,
            OAuthToken.provider == provider,
            OAuthToken.revoked_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def upsert_token(
    session: AsyncSession,
    user_id: str,
    *,
    provider: str = "google",
    access_token_enc: bytes,
    refresh_token_enc: bytes | None = None,
    scopes: Sequence[str],
    expires_at: dt.datetime | None = None,
    provider_account_id: str | None = None,
    account_email: str | None = None,
    key_version: int = 1,
) -> OAuthToken:
    """Store a freshly granted or refreshed token.

    A refresh response with no refresh_token must not wipe the one we hold, so
    the update keeps the existing value when the new one is NULL.
    """
    now = utcnow()
    stmt = (
        pg_insert(OAuthToken)
        .values(
            id=new_id(),
            user_id=user_id,
            provider=provider,
            provider_account_id=provider_account_id,
            account_email=account_email,
            access_token_enc=access_token_enc,
            refresh_token_enc=refresh_token_enc,
            key_version=key_version,
            scopes=list(scopes),
            expires_at=ensure_utc(expires_at),
            revoked_at=None,
            refresh_failures=0,
            created_at=now,
            updated_at=now,
        )
        .returning(OAuthToken)
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[OAuthToken.user_id, OAuthToken.provider],
        set_={
            "access_token_enc": stmt.excluded.access_token_enc,
            # a refresh response with no refresh_token must not wipe ours
            "refresh_token_enc": func.coalesce(
                stmt.excluded.refresh_token_enc, OAuthToken.refresh_token_enc
            ),
            "key_version": stmt.excluded.key_version,
            "scopes": stmt.excluded.scopes,
            "expires_at": stmt.excluded.expires_at,
            "provider_account_id": func.coalesce(
                stmt.excluded.provider_account_id, OAuthToken.provider_account_id
            ),
            "account_email": func.coalesce(
                stmt.excluded.account_email, OAuthToken.account_email
            ),
            "revoked_at": None,
            "refresh_failures": 0,
            "updated_at": now,
        },
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def update_access_token(
    session: AsyncSession,
    user_id: str,
    *,
    provider: str = "google",
    access_token_enc: bytes,
    expires_at: dt.datetime | None,
    key_version: int = 1,
) -> OAuthToken | None:
    """Narrow write for the refresh loop: new access token, clock reset."""
    result = await session.execute(
        update(OAuthToken)
        .where(OAuthToken.user_id == user_id, OAuthToken.provider == provider)
        .values(
            access_token_enc=access_token_enc,
            expires_at=ensure_utc(expires_at),
            key_version=key_version,
            refresh_failures=0,
            updated_at=utcnow(),
        )
        .returning(OAuthToken)
    )
    return result.scalar_one_or_none()


async def bump_refresh_failures(
    session: AsyncSession, user_id: str, provider: str = "google"
) -> int:
    """Count a failed refresh. Returns the new count so the caller can decide
    when to give up and ask the person to reconnect."""
    result = await session.execute(
        update(OAuthToken)
        .where(OAuthToken.user_id == user_id, OAuthToken.provider == provider)
        .values(
            refresh_failures=OAuthToken.refresh_failures + 1,
            updated_at=utcnow(),
        )
        .returning(OAuthToken.refresh_failures)
    )
    value = result.scalar_one_or_none()
    return int(value or 0)


async def mark_token_revoked(
    session: AsyncSession,
    user_id: str,
    provider: str = "google",
    *,
    at: dt.datetime | None = None,
) -> None:
    """Google says the grant is gone, or the person disconnected."""
    await session.execute(
        update(OAuthToken)
        .where(OAuthToken.user_id == user_id, OAuthToken.provider == provider)
        .values(revoked_at=ensure_utc(at) or utcnow(), updated_at=utcnow())
    )


async def delete_token(
    session: AsyncSession, user_id: str, provider: str | None = None
) -> None:
    """Hard delete on disconnect."""
    stmt = delete(OAuthToken).where(OAuthToken.user_id == user_id)
    if provider is not None:
        stmt = stmt.where(OAuthToken.provider == provider)
    await session.execute(stmt)


async def list_tokens_expiring(
    session: AsyncSession,
    user_id: str | None = None,
    *,
    before: dt.datetime | None = None,
    max_failures: int = 5,
    limit: int = 500,
) -> list[OAuthToken]:
    """Tokens due for refresh.

    ``user_id=None`` scans every user — for ``maintenance.refresh_tokens`` only.
    """
    cutoff = ensure_utc(before) or (utcnow() + dt.timedelta(minutes=15))
    stmt = (
        select(OAuthToken)
        .where(
            OAuthToken.revoked_at.is_(None),
            OAuthToken.refresh_token_enc.is_not(None),
            OAuthToken.expires_at.is_not(None),
            OAuthToken.expires_at <= cutoff,
            OAuthToken.refresh_failures < max_failures,
        )
        .order_by(OAuthToken.expires_at.asc())
        .limit(limit)
    )
    if user_id is not None:
        stmt = stmt.where(OAuthToken.user_id == user_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_connected_user_ids(
    session: AsyncSession, user_id: str | None = None, *, provider: str = "google"
) -> list[str]:
    """Users with a live grant. ``user_id=None`` means all — for
    ``sync.dispatch_all_users``."""
    stmt = select(OAuthToken.user_id).where(
        OAuthToken.provider == provider, OAuthToken.revoked_at.is_(None)
    )
    if user_id is not None:
        stmt = stmt.where(OAuthToken.user_id == user_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def stage_password_signup(
    session: AsyncSession,
    *,
    email: str,
    password_hash: str,
    user_id: str,
) -> User:
    """Create or update the row a sign-up will finish claiming.

    Unverified, so it grants nothing on its own. Re-registering an address that
    was never verified simply overwrites it: an abandoned attempt must not be
    able to hold an address hostage, and there is nothing to protect yet because
    nobody has ever proved they own it.

    An address that already exists from Google is updated rather than duplicated
    — the code about to be sent proves the same mailbox Google proved, so this is
    the same person adding a second way in.
    """
    existing = await get_user_by_email(session, email)
    if existing is not None:
        if existing.password_hash and existing.email_verified:
            # Verified account with a password: leave it completely alone.
            return existing
        existing.password_hash = password_hash
        await session.flush()
        return existing

    user = User(
        id=user_id,
        email=email,
        display_name=email.split("@", 1)[0],
        password_hash=password_hash,
        email_verified=False,
    )
    session.add(user)
    await session.flush()
    return user


async def mark_signed_in(session: AsyncSession, user_id: str) -> None:
    """A code was entered correctly: the address is proved and they are in."""
    await session.execute(
        update(User)
        .where(User.id == user_id)
        .values(email_verified=True, last_login_at=utcnow(), updated_at=utcnow())
    )
    await session.flush()
