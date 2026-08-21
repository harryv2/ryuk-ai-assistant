"""Password hashing.

PBKDF2-HMAC-SHA256 from the standard library. Argon2 is the better algorithm and
is used automatically when `argon2-cffi` is installed; PBKDF2 is here so the app
has no new hard dependency and still never stores a recoverable password.

The stored value carries its own parameters, so raising the iteration count later
does not invalidate existing passwords — an old hash still verifies with the
numbers it was made with, and is quietly upgraded on the next successful sign-in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Final

#: OWASP's 2023 floor for PBKDF2-HMAC-SHA256 is 600,000.
ITERATIONS: Final[int] = 600_000
SALT_BYTES: Final[int] = 16
KEY_BYTES: Final[int] = 32

try:  # pragma: no cover - depends on what is installed
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerifyMismatchError

    _argon2: PasswordHasher | None = PasswordHasher()
except Exception:  # pragma: no cover
    _argon2 = None
    VerifyMismatchError = InvalidHashError = Exception  # type: ignore[assignment,misc]


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """A verifier for this password. Never reversible, never the same twice."""
    if not password:
        raise ValueError("password must not be empty")
    if _argon2 is not None:
        return _argon2.hash(password)
    salt = os.urandom(SALT_BYTES)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS, KEY_BYTES)
    return f"pbkdf2_sha256${ITERATIONS}${_b64(salt)}${_b64(key)}"


#: Verified against when the account does not exist, so a sign-in attempt for an
#: unknown address costs the same as one for a real address. Without it the
#: response time answers "is this email registered?" for anyone who asks.
_DUMMY: Final[str] = hash_password("dummy-password-for-constant-time-comparison")


def verify_password(password: str, stored: str | None) -> bool:
    """Does this password match? Same cost whether or not the account exists."""
    candidate = stored or _DUMMY
    result = _verify(password, candidate)
    # A missing hash never succeeds, but only after doing the work.
    return bool(result and stored)


def _verify(password: str, stored: str) -> bool:
    if stored.startswith("$argon2") and _argon2 is not None:
        try:
            return bool(_argon2.verify(stored, password))
        except (VerifyMismatchError, InvalidHashError, Exception):
            return False
    try:
        scheme, iterations, salt, expected = stored.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    try:
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _unb64(salt), int(iterations), KEY_BYTES
        )
    except Exception:
        return False
    return hmac.compare_digest(_b64(key), expected)


def needs_rehash(stored: str | None) -> bool:
    """True when a successful sign-in should quietly re-hash at current settings."""
    if not stored:
        return False
    if _argon2 is not None:
        if not stored.startswith("$argon2"):
            return True
        try:
            return bool(_argon2.check_needs_rehash(stored))
        except Exception:
            return False
    if not stored.startswith("pbkdf2_sha256$"):
        return True
    try:
        return int(stored.split("$", 2)[1]) < ITERATIONS
    except (ValueError, IndexError):
        return True


def strength_problem(password: str) -> str | None:
    """Why this password is not acceptable, in words, or None.

    Length does far more for a password than a character-class rule, which mostly
    teaches people to end everything in "1!".
    """
    if len(password) < 10:
        return "Use at least 10 characters."
    if len(password) > 200:
        return "That is longer than 200 characters."
    if password.lower() in {"password12", "1234567890", "qwertyuiop", "passw0rd12"}:
        return "That password is too common."
    return None


def new_password_token() -> str:
    """A random password, for accounts created without one."""
    return secrets.token_urlsafe(24)


__all__ = [
    "hash_password",
    "needs_rehash",
    "new_password_token",
    "strength_problem",
    "verify_password",
]
