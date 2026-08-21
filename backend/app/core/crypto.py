"""AES-256-GCM for the OAuth tokens.

Blob layout, matching ``oauth_tokens.access_token_enc``::

    nonce(12) || ciphertext || tag(16)

Python's AESGCM appends the tag to the ciphertext already, so the blob is just
``nonce + aesgcm.encrypt(...)``.

Keys are held per version so one can be rotated without a downtime window:
write with the current version, keep reading with the old ones until every row
has been re-encrypted. ``oauth_tokens.key_version`` records which key a row
used, and :func:`decrypt` takes that value.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config import settings
from app.core.errors import AppError

NONCE_BYTES: Final[int] = 12
TAG_BYTES: Final[int] = 16
KEY_BYTES: Final[int] = 32  # AES-256

_keys: dict[int, bytes] | None = None
_current_version: int | None = None


def _decode_key(raw: str | bytes, where: str) -> bytes:
    """Base64 -> 32 raw bytes, with a clear message when it is not."""
    if isinstance(raw, bytes):
        material = raw
    else:
        text = raw.strip()
        try:
            material = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            try:
                material = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
            except (binascii.Error, ValueError) as exc:
                raise AppError(
                    "INTERNAL",
                    f"{where} is not valid base64.",
                    details={"setting": where},
                ) from exc
    if len(material) != KEY_BYTES:
        raise AppError(
            "INTERNAL",
            f"{where} must decode to {KEY_BYTES} bytes, got {len(material)}.",
            details={"setting": where, "bytes": len(material)},
        )
    return material


def _parse_key_map(raw: object) -> dict[int, bytes]:
    """Retired keys, given as ``{"1": "<base64>", "2": "<base64>"}``."""
    if not raw:
        return {}
    data = raw
    if isinstance(raw, (str, bytes)):
        text = raw.decode() if isinstance(raw, bytes) else raw
        text = text.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AppError(
                "INTERNAL",
                "TOKEN_ENCRYPTION_KEYS must be a JSON object of version -> base64 key.",
            ) from exc
    if not isinstance(data, dict):
        raise AppError(
            "INTERNAL",
            "TOKEN_ENCRYPTION_KEYS must be a JSON object of version -> base64 key.",
        )
    return {
        int(version): _decode_key(value, f"TOKEN_ENCRYPTION_KEYS[{version}]")
        for version, value in data.items()
    }


def _load() -> tuple[dict[int, bytes], int]:
    global _keys, _current_version
    if _keys is not None and _current_version is not None:
        return _keys, _current_version

    # config exposes the decoded bytes; fall back to decoding the raw setting.
    primary: str | bytes | None
    try:
        primary = settings.token_encryption_key
    except (AttributeError, ValueError):
        primary = getattr(settings, "TOKEN_ENCRYPTION_KEY", None)
    if not primary:
        raise AppError("INTERNAL", "TOKEN_ENCRYPTION_KEY is not configured.")

    version = int(
        getattr(settings, "TOKEN_KEY_VERSION", None)
        or getattr(settings, "TOKEN_ENCRYPTION_KEY_VERSION", None)
        or 1
    )
    keys = _parse_key_map(getattr(settings, "TOKEN_ENCRYPTION_KEYS", None))
    keys[version] = _decode_key(primary, "TOKEN_ENCRYPTION_KEY")

    _keys, _current_version = keys, version
    return keys, version


def reload_keys() -> None:
    """Drop the cached keys. Call after settings change, and in tests."""
    global _keys, _current_version
    _keys = None
    _current_version = None


def current_key_version() -> int:
    """The version new ciphertext is written with — store it on the row."""
    return _load()[1]


def key_versions() -> list[int]:
    """Every version we can still decrypt."""
    return sorted(_load()[0])


def _key_for(version: int) -> bytes:
    keys, _ = _load()
    try:
        return keys[version]
    except KeyError as exc:
        raise AppError(
            "INTERNAL",
            f"No encryption key for version {version}.",
            details={"key_version": version, "known": sorted(keys)},
        ) from exc


def generate_key() -> str:
    """A fresh base64 key, for filling in the setting. Not used at runtime."""
    return base64.b64encode(os.urandom(KEY_BYTES)).decode()


def encrypt(plaintext: str | bytes, aad: bytes | None = None) -> bytes:
    """Encrypt with the current key. Returns ``nonce || ciphertext || tag``."""
    if plaintext is None:
        raise AppError("INTERNAL", "Nothing to encrypt.")
    data = plaintext.encode("utf-8") if isinstance(plaintext, str) else bytes(plaintext)
    keys, version = _load()
    nonce = os.urandom(NONCE_BYTES)
    sealed = AESGCM(keys[version]).encrypt(nonce, data, aad)
    return nonce + sealed


def encrypt_with_version(
    plaintext: str | bytes, aad: bytes | None = None
) -> tuple[bytes, int]:
    """Encrypt and hand back the key version to store alongside the blob."""
    return encrypt(plaintext, aad), current_key_version()


def decrypt(blob: bytes, key_version: int | None = None, aad: bytes | None = None) -> str:
    """Decrypt a blob written by :func:`encrypt`. Returns UTF-8 text."""
    return decrypt_bytes(blob, key_version, aad).decode("utf-8")


def decrypt_bytes(
    blob: bytes | memoryview | bytearray,
    key_version: int | None = None,
    aad: bytes | None = None,
) -> bytes:
    """Decrypt a blob to raw bytes."""
    raw = bytes(blob) if not isinstance(blob, bytes) else blob
    if len(raw) < NONCE_BYTES + TAG_BYTES + 1:
        raise AppError(
            "INTERNAL",
            "Stored token is too short to be valid ciphertext.",
            details={"bytes": len(raw)},
        )
    version = current_key_version() if key_version is None else int(key_version)
    key = _key_for(version)
    nonce, sealed = raw[:NONCE_BYTES], raw[NONCE_BYTES:]
    try:
        return AESGCM(key).decrypt(nonce, sealed, aad)
    except InvalidTag as exc:
        raise AppError(
            "INTERNAL",
            "Stored token could not be decrypted with that key version.",
            details={"key_version": version},
        ) from exc


def needs_rotation(key_version: int) -> bool:
    """True when a row was written with an older key than the current one."""
    return int(key_version) != current_key_version()


__all__ = [
    "NONCE_BYTES",
    "TAG_BYTES",
    "KEY_BYTES",
    "encrypt",
    "encrypt_with_version",
    "decrypt",
    "decrypt_bytes",
    "current_key_version",
    "key_versions",
    "needs_rotation",
    "generate_key",
    "reload_keys",
]
