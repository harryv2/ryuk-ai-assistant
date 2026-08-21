"""Ids, fingerprints, canonical JSON and token encryption.

Small functions, and almost everything else leans on them.

`new_id` makes every primary key in the schema. `fingerprint` makes every
`content_hash`, `dedupe_key` and `payload_hash` — and a `dedupe_key` that is not
perfectly stable is a duplicate email, while one that is not sensitive enough is
a send that silently never happens. `canonical_json` is what feeds a
fingerprint, so key order in a payload dict must not be able to change it.

`crypto` holds the OAuth tokens. The blob layout is fixed by `schema.md` as
`nonce(12) || ciphertext || tag(16)`, and `oauth_tokens.key_version` says which
key opened it, so a key can be rotated without a downtime window.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.core.crypto import (
    KEY_BYTES,
    NONCE_BYTES,
    TAG_BYTES,
    current_key_version,
    decrypt,
    encrypt,
    generate_key,
    needs_rotation,
)
from app.core.errors import AppError
from app.core.ids import (
    ALPHABET,
    ID_LENGTH,
    canonical_json,
    fingerprint,
    fingerprint_parts,
    is_id,
    new_id,
)
from tests.conftest import KEY_A, KEY_B

# ---------------------------------------------------------------------------
# nanoid
# ---------------------------------------------------------------------------


def test_an_id_is_twenty_one_characters():
    assert ID_LENGTH == 21
    assert len(new_id()) == 21


def test_an_id_uses_only_the_declared_alphabet():
    for _ in range(200):
        assert set(new_id()) <= set(ALPHABET)


def test_the_alphabet_is_a_power_of_two():
    # Sixty-four characters means a random byte maps to a character with a
    # single mask, no rejection loop and no modulo bias. Any other size makes
    # some characters slightly more likely than others.
    assert len(ALPHABET) == 64
    assert len(set(ALPHABET)) == 64


def test_the_alphabet_is_url_safe():
    # Ids appear in paths such as /api/v1/prompts/{id}/respond, so nothing in
    # here may need percent-encoding.
    assert set(ALPHABET) <= set(
        "-_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )


def test_ids_do_not_repeat():
    # 21 characters over a 64-character alphabet is about 126 bits. Ten thousand
    # of them colliding would mean the generator is broken, not unlucky.
    made = {new_id() for _ in range(10_000)}
    assert len(made) == 10_000


def test_is_id_accepts_ours_and_rejects_everything_else():
    assert is_id(new_id())
    assert not is_id("too-short")
    assert not is_id(new_id() + "x")  # 22 characters
    assert not is_id("!" * ID_LENGTH)  # right length, wrong alphabet
    assert not is_id(None)
    assert not is_id(12345)


# ---------------------------------------------------------------------------
# canonical_json
# ---------------------------------------------------------------------------


def test_keys_come_out_sorted():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_nested_keys_are_sorted_too():
    one = canonical_json({"outer": {"z": 1, "a": 2}, "first": [{"y": 1, "x": 2}]})
    two = canonical_json({"first": [{"x": 2, "y": 1}], "outer": {"a": 2, "z": 1}})
    assert one == two


def test_there_is_no_whitespace():
    text = canonical_json({"to": ["a@x.com"], "subject": "Hi"})
    assert " " not in text.replace('"Hi"', "").replace('"a@x.com"', "")
    assert ": " not in text and ", " not in text


def test_list_order_is_kept():
    # Order carries meaning in a recipient list, so it is not sorted away.
    assert canonical_json(["b", "a"]) != canonical_json(["a", "b"])


def test_unicode_is_written_out_rather_than_escaped():
    # The Turkish subject line has to fingerprint identically wherever it is
    # hashed, and \u escapes would make that depend on the encoder's mood.
    assert canonical_json({"s": "Uçuş"}) == '{"s":"Uçuş"}'


def test_the_awkward_types_have_a_representation():
    moment = datetime(2026, 8, 20, 13, 12, 4, tzinfo=UTC)
    text = canonical_json(
        {"when": moment, "id": uuid.UUID(int=1), "money": Decimal("812.40"), "raw": b"\x00\xff"}
    )
    parsed = json.loads(text)
    assert parsed["when"].startswith("2026-08-20T13:12:04")
    assert parsed["id"] == "00000000-0000-0000-0000-000000000001"
    assert parsed["money"] == "812.40"
    assert parsed["raw"] == "00ff"


def test_a_value_that_cannot_be_canonicalised_is_refused():
    with pytest.raises(TypeError):
        canonical_json({"fn": lambda: None})


def test_not_a_number_is_refused():
    # NaN is not JSON, and `float('nan') != float('nan')` would make a
    # fingerprint that never matches itself.
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_a_fingerprint_is_a_uuid5():
    value = fingerprint("gmail.body", "hello")
    assert isinstance(value, uuid.UUID)
    assert value.version == 5


def test_the_same_input_always_gives_the_same_fingerprint():
    first = fingerprint("gmail.body", "Your booking is confirmed")
    for _ in range(50):
        assert fingerprint("gmail.body", "Your booking is confirmed") == first


def test_one_character_changes_the_fingerprint():
    a = fingerprint("gmail.body", "Your booking is confirmed")
    b = fingerprint("gmail.body", "Your booking is confirmed.")
    assert a != b


def test_the_namespace_keeps_unrelated_things_apart():
    # The same text hashed as an email body and as an action payload must not
    # collide, or a `content_hash` could be mistaken for a `dedupe_key`.
    assert fingerprint("gmail.body", "x") != fingerprint("action.payload", "x")


def test_the_separator_cannot_be_smuggled_across():
    # Namespace and payload are joined by a separator. If it were an ordinary
    # character, ("ab", "c") and ("a", "bc") would fingerprint the same.
    assert fingerprint("ab", "c") != fingerprint("a", "bc")


def test_fingerprint_parts_ignores_key_order_in_a_payload():
    payload_one = {"to": ["ops@x.com"], "subject": "Cancellation", "body": "..."}
    payload_two = {"body": "...", "subject": "Cancellation", "to": ["ops@x.com"]}
    assert fingerprint_parts("action", "u1", "gmail.send_email", payload_one, "c1") == (
        fingerprint_parts("action", "u1", "gmail.send_email", payload_two, "c1")
    )


def test_the_dedupe_key_recipe_is_stable_and_sensitive():
    # actions.dedupe_key = uuid5(user | op | canonical payload | conversation),
    # under a partial unique index over draft/approved/running.
    payload = {"to": ["cancel@turkishairlines.com"], "subject": "Cancellation — 6F2QK9"}

    def key(user="u1", op="gmail.send_email", body=None, conversation="c1"):
        return fingerprint_parts("action", user, op, body if body is not None else payload, conversation)

    baseline = key()
    assert key() == baseline  # a double submit dedupes
    assert key(user="u2") != baseline  # another tenant is another action
    assert key(conversation="c2") != baseline  # same request, different thread
    assert key(op="gmail.draft_email") != baseline
    assert key(body={**payload, "subject": "Cancellation — R4TQ8M"}) != baseline


def test_a_fingerprint_of_the_empty_string_is_still_a_fingerprint():
    value = fingerprint("gmail.body", "")
    assert isinstance(value, uuid.UUID) and value.version == 5


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


def test_a_token_survives_the_round_trip(crypto_keys):
    token = "ya29.a0AfB_byC" + "x" * 180
    assert decrypt(encrypt(token)) == token


def test_the_blob_has_the_layout_the_schema_declares(crypto_keys):
    # nonce(12) || ciphertext || tag(16), which is what oauth_tokens.
    # access_token_enc is documented to hold.
    plaintext = "refresh-token-value"
    blob = encrypt(plaintext)
    assert NONCE_BYTES == 12 and TAG_BYTES == 16 and KEY_BYTES == 32
    assert len(blob) == NONCE_BYTES + len(plaintext.encode()) + TAG_BYTES


def test_the_same_token_encrypts_differently_every_time(crypto_keys):
    # A fresh nonce each call. Otherwise two users with the same token would
    # have identical ciphertext, which leaks that fact to anyone with the table.
    blobs = {encrypt("same token") for _ in range(20)}
    assert len(blobs) == 20


def test_unicode_and_empty_strings_survive(crypto_keys):
    assert decrypt(encrypt("Uçuş rezervasyonunuz — 6F2QK9")) == "Uçuş rezervasyonunuz — 6F2QK9"
    assert decrypt(encrypt("x")) == "x"


def test_the_wrong_key_cannot_open_it(crypto_keys):
    blob = encrypt("ya29.secret")
    crypto_keys.use(KEY_B, 1)  # same version number, different key material
    with pytest.raises(AppError) as caught:
        decrypt(blob)
    assert caught.value.code == "INTERNAL"


def test_a_tampered_blob_is_refused(crypto_keys):
    blob = bytearray(encrypt("ya29.secret"))
    blob[-1] ^= 0x01  # flip one bit of the tag
    with pytest.raises(AppError):
        decrypt(bytes(blob))


def test_a_tampered_ciphertext_is_refused(crypto_keys):
    blob = bytearray(encrypt("ya29.secret"))
    blob[NONCE_BYTES] ^= 0x01  # flip one bit of the ciphertext
    with pytest.raises(AppError):
        decrypt(bytes(blob))


def test_a_blob_that_is_too_short_is_refused(crypto_keys):
    with pytest.raises(AppError):
        decrypt(b"\x00" * 8)


def test_extra_data_has_to_match(crypto_keys):
    # Binding a blob to the user it belongs to means a row copied between
    # tenants will not decrypt, whatever else went wrong.
    blob = encrypt("ya29.secret", aad=b"u_7QkR2mXvB4nLd9TsW")
    assert decrypt(blob, aad=b"u_7QkR2mXvB4nLd9TsW") == "ya29.secret"
    with pytest.raises(AppError):
        decrypt(blob, aad=b"u_someone_else______")
    with pytest.raises(AppError):
        decrypt(blob)


def test_an_old_key_version_still_opens_an_old_blob(crypto_keys):
    # Rotation without a downtime window: write with the new key, keep reading
    # with the old one until every row has been re-encrypted.
    old_blob = encrypt("written under version 1")
    assert current_key_version() == 1

    crypto_keys.use(KEY_B, version=2, retired={1: KEY_A})
    assert current_key_version() == 2

    assert decrypt(old_blob, key_version=1) == "written under version 1"
    assert decrypt(encrypt("written under version 2")) == "written under version 2"
    assert needs_rotation(1) is True
    assert needs_rotation(2) is False


def test_an_unknown_key_version_is_refused(crypto_keys):
    blob = encrypt("ya29.secret")
    with pytest.raises(AppError):
        decrypt(blob, key_version=99)


def test_a_generated_key_is_thirty_two_bytes():
    key = generate_key()
    assert len(base64.b64decode(key)) == KEY_BYTES
    assert generate_key() != generate_key()
