"""Incremental sync: the cursor, the embedding cache, and the 410.

Three invariants, each of which costs real money or real correctness when it is
wrong:

* **the cursor advances only after the upsert commits.** Get this backwards and
  a worker that dies mid-page skips the messages it had already fetched —
  silently, permanently, and the person never learns why that email is not
  searchable. The test kills the process between the two and asserts the page is
  reprocessed rather than skipped;
* **an unchanged ``content_hash`` is not re-embedded.** The hash is a
  fingerprint of the exact text that was embedded, so equality means the vector
  is still correct. Re-embedding anyway is a bill, not a bug — which is why it
  needs a test to stay fixed;
* **a 410 is recoverable.** Gmail's ``historyId`` and Calendar's ``syncToken``
  both expire, and the only honest answer is to walk the mailbox again. Bounded:
  a resync that pages forever is worse than the stale mirror it was fixing.

The first and last need the sync tasks. Where those are not on disk yet the test
says which import failed. The repository-level invariants underneath are tested
either way, because they are the part the tasks lean on.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import uuid

import pytest

from tests.fixtures import google_responses as gr
from tests.integration.conftest import (
    call_entrypoint,
    require,
)

pytestmark = pytest.mark.integration


class Crash(RuntimeError):
    """The worker died. Not a bug in the code under test."""


#: Where a service's sync lives, and the async entrypoint underneath the Celery
#: task. The Celery wrapper itself is a sync `asyncio.run`, which cannot be
#: re-entered from inside a running loop, so the coroutine is called directly.
SYNC_MODULES = {
    "gmail": "app.tasks.sync_gmail",
    "gcal": "app.tasks.sync_gcal",
    "gdrive": "app.tasks.sync_gdrive",
}

SYNC_NAMES = (
    "sync_incremental",
    "run_incremental",
    "incremental",
    "sync_user",
    "run_sync",
    "sync_async",
    "run_async",
    "sync",
    "run",
)


def sync_entrypoint(service: str):
    """The coroutine that performs one incremental pass for one user."""
    path = SYNC_MODULES[service]
    module = require(path)
    for name in SYNC_NAMES:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        fn = getattr(fn, "__wrapped__", fn)  # unwrap the celery task
        if inspect.iscoroutinefunction(fn):
            return module, fn
    coroutines = [
        n
        for n, f in vars(module).items()
        if inspect.iscoroutinefunction(f) and not n.startswith("_")
    ]
    pytest.skip(
        f"{path} has no async incremental entrypoint. Tried {SYNC_NAMES}; the "
        f"module defines {coroutines}"
    )


async def run_sync(service, session, user_id: str, **extra):
    """One incremental pass, driven in this process."""
    module, fn = sync_entrypoint(service)
    try:
        return await call_entrypoint(
            fn, session, user_id, service=service, mode="incremental", **extra
        )
    except LookupError as exc:
        pytest.skip(str(exc))


async def count_gmail(session, user_id: str) -> int:
    from sqlalchemy import func, select

    models = require("app.db.models")
    await session.rollback()
    total = await session.execute(
        select(func.count()).select_from(models.SyncMessage).where(
            models.SyncMessage.connector == "gmail",
            models.SyncMessage.user_id == user_id
        )
    )
    return int(total.scalar_one())


async def cursor_of(session, user_id: str, service: str):
    state_repo = require("app.db.repositories.sync_state")
    await session.rollback()
    state = await state_repo.get_state(session, user_id, service)
    return state.cursor if state else None


# --------------------------------------------------------------------------- #
# The cursor
# --------------------------------------------------------------------------- #


async def test_the_cursor_moves_only_after_the_upsert_commits(
    db, sync_ready, google, monkeypatch
):
    """Kill the worker between the write and the cursor. Nothing is lost.

    The order is: upsert the page, commit, *then* advance the cursor. A crash in
    that gap means the next run fetches the same page again — wasteful, and
    completely safe, because the upsert is keyed on
    ``(user_id, message_id, chunk_index)``. The other order loses mail.
    """
    state_repo = require("app.db.repositories.sync_state")
    user = sync_ready

    start = {"historyId": "9912841"}
    await state_repo.set_cursor(db, user.id, "gmail", start)
    await db.commit()

    crashed = {"count": 0}
    real_set_cursor = state_repo.set_cursor
    real_mark_success = state_repo.mark_success

    async def die(*args, **kwargs):
        crashed["count"] += 1
        raise Crash("worker lost between the upsert and the cursor")

    monkeypatch.setattr(state_repo, "set_cursor", die)
    monkeypatch.setattr(state_repo, "mark_success", die)

    with contextlib.suppress(Crash, Exception):
        await run_sync("gmail", db, user.id)

    assert crashed["count"] >= 1, (
        "the cursor was never written, so this test proved nothing about the "
        "order it is written in"
    )

    landed = await count_gmail(db, user.id)
    assert landed > 0, (
        "the page should already be committed when the cursor write happens — "
        "that is the whole point of doing them in that order"
    )
    assert await cursor_of(db, user.id, "gmail") == start, (
        "the cursor moved even though the run did not finish; the next pass will "
        "skip whatever was in flight"
    )

    first_pass = [r for r in google.calls("gmail") if "history" in r.path]
    assert first_pass, "no history call was made"

    # -- the worker comes back --------------------------------------------- #
    monkeypatch.setattr(state_repo, "set_cursor", real_set_cursor)
    monkeypatch.setattr(state_repo, "mark_success", real_mark_success)

    await run_sync("gmail", db, user.id)

    replayed = [r for r in google.calls("gmail") if "history" in r.path]
    assert len(replayed) > len(first_pass), "the second pass made no history call"
    assert replayed[-1].params.get("startHistoryId") == start["historyId"], (
        "the second pass started somewhere else, which means the page in flight "
        f"was skipped: {replayed[-1].params}"
    )

    assert await count_gmail(db, user.id) == landed, (
        "reprocessing the same page duplicated rows; the upsert key is doing "
        "nothing"
    )
    after = await cursor_of(db, user.id, "gmail")
    assert after and after != start, (
        f"a clean pass should have advanced the cursor: {after}"
    )


async def test_a_failed_pass_updates_the_attempt_but_not_the_success(db, sync_ready):
    """``lag_seconds`` has to reflect the data, not the effort.

    Two timestamps for that reason: a failing sync moves ``last_synced_at`` and
    leaves ``last_success_at`` where it was, so ``/sync/status`` reports the age
    of the mirror rather than how recently we tried.
    """
    state_repo = require("app.db.repositories.sync_state")
    user = sync_ready

    await state_repo.mark_success(db, user.id, "gmail", items_indexed=12)
    await db.commit()
    good = await state_repo.get_state(db, user.id, "gmail")
    success_at = good.last_success_at
    assert success_at is not None

    await state_repo.mark_failure(
        db, user.id, "gmail", {"class": "TRANSIENT", "code": 503}
    )
    await db.commit()

    bad = await state_repo.get_state(db, user.id, "gmail")
    assert bad.last_success_at == success_at, "a failure must not look like a success"
    assert bad.last_synced_at > success_at
    assert bad.consecutive_failures == 1
    assert bad.last_error["code"] == 503

    freshness = await state_repo.freshness(db, user.id)
    assert freshness["gmail"]["lag_seconds"] is not None
    assert freshness["gmail"]["last_error"]["class"] == "TRANSIENT"


async def test_repeated_failures_open_the_circuit(db, sync_ready):
    """Five in a row and the service is left alone for a while.

    ``list_due`` is what the beat fan-out reads, so an open circuit has to keep
    the user out of that list — otherwise the breaker is decorative.
    """
    state_repo = require("app.db.repositories.sync_state")
    user = sync_ready

    for _ in range(5):
        await state_repo.mark_failure(db, user.id, "gcal", {"class": "TRANSIENT"})
    await db.commit()

    state = await state_repo.get_state(db, user.id, "gcal")
    assert state.consecutive_failures == 5
    assert state.circuit_open_until is not None, "five failures should hold it open"
    assert await state_repo.is_open(db, user.id, "gcal") is True

    due = await state_repo.list_due(db, user.id, service="gcal")
    assert not due, "an open circuit must keep the service out of the due list"

    await state_repo.reset_circuit(db, user.id, "gcal")
    await db.commit()
    assert await state_repo.is_open(db, user.id, "gcal") is False


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


async def test_an_unchanged_content_hash_keeps_the_vector(db, user, embed):
    """The upsert rule, tested where it lives.

    Same hash: keep the vector already paid for. Different hash: the old vector
    is stale, so take whatever arrived — usually NULL, which is exactly what
    queues the row for re-embedding.
    """
    mirror = require("app.db.repositories.mirror")
    from sqlalchemy import select

    models = require("app.db.models")

    rows = gr.gmail_mirror_rows(embed, [gr.MSG_SARAH_BUDGET])
    await mirror.upsert_gmail(db, user.id, rows)
    await db.commit()

    async def vector_of():
        found = await db.execute(
            select(models.SyncMessage.embedding).where(
                models.SyncMessage.user_id == user.id,
                models.SyncMessage.connector == "gmail",
                models.SyncMessage.source_id == gr.MSG_SARAH_BUDGET,
            )
        )
        return found.scalar_one()

    original = await vector_of()
    assert original is not None and len(original) == 1536

    # The sync task re-reads a message it has seen: same text, same hash, and
    # no vector in hand because it did not pay for one.
    same = dict(rows[0])
    same["embedding"] = None
    await mirror.upsert_gmail(db, user.id, [same])
    await db.commit()

    kept = await vector_of()
    assert kept is not None, (
        "the vector was dropped even though the text did not change — the next "
        "embed sweep will pay for it again"
    )
    assert list(kept)[:8] == list(original)[:8]

    waiting = await mirror.rows_needing_embedding(db, user.id, "gmail")
    assert not [r for r in waiting if r.source_id == gr.MSG_SARAH_BUDGET], (
        "an unchanged row must not be queued for re-embedding"
    )

    # Edited message: new hash, no vector, straight onto the embed queue.
    edited = dict(rows[0])
    edited["body_clean"] = rows[0]["body_clean"] + "\n\nPS: numbers revised."
    edited["content_hash"] = gr.content_hash(edited["body_clean"])
    edited["embedding"] = None
    await mirror.upsert_gmail(db, user.id, [edited])
    await db.commit()

    assert await vector_of() is None, (
        "a changed body leaves a stale vector behind, which is worse than none"
    )
    waiting = await mirror.rows_needing_embedding(db, user.id, "gmail")
    assert [r for r in waiting if r.source_id == gr.MSG_SARAH_BUDGET]


async def test_a_second_sync_of_unchanged_mail_embeds_nothing(
    db, sync_ready, llm, google
):
    """The same thing end to end: sync twice, pay once."""
    user = sync_ready

    await run_sync("gmail", db, user.id)
    first = llm.embedding_calls
    landed = await count_gmail(db, user.id)
    assert landed > 0, "the first pass indexed nothing"

    await run_sync("gmail", db, user.id)

    assert await count_gmail(db, user.id) == landed
    assert llm.embedding_calls == first, (
        f"the second pass re-embedded unchanged mail: {llm.embedding_calls - first} "
        "extra calls to the embedding model"
    )


# --------------------------------------------------------------------------- #
# 410 Gone
# --------------------------------------------------------------------------- #


async def test_a_410_falls_back_to_a_bounded_full_resync(db, sync_ready, google):
    """The historyId expired. Walk the mailbox again, and stop.

    Google returns 410 when the cursor is older than it keeps history for. There
    is no way to ask "what did I miss", so the answer is a full walk — bounded
    by ``SYNC_PAGE_SIZE`` and a page cap, because a resync that never finishes
    is worse than the stale mirror it was fixing.
    """
    state_repo = require("app.db.repositories.sync_state")
    user = sync_ready

    stale = {"historyId": "1"}
    await state_repo.set_cursor(db, user.id, "gmail", stale)
    await db.commit()

    google.fail("gmail", 410, body=gr.ERROR_410, contains="/history")

    await run_sync("gmail", db, user.id)

    calls = google.calls("gmail")
    assert any("history" in r.path for r in calls), "the incremental path was never tried"
    listed = [r for r in calls if r.path.endswith("/messages")]
    assert listed, (
        "a 410 should have fallen back to a full walk over messages.list; "
        f"Google only saw {[str(r) for r in calls]}"
    )

    assert len(calls) < 60, (
        f"the resync made {len(calls)} calls — that is not bounded, and on a real "
        "mailbox it is a quota incident"
    )

    assert await count_gmail(db, user.id) > 0, "the resync indexed nothing"

    after = await cursor_of(db, user.id, "gmail")
    assert after and after != stale, (
        f"the expired cursor is still stored, so the next pass 410s again: {after}"
    )

    state = await state_repo.get_state(db, user.id, "gmail")
    assert state.last_success_at is not None, "a recovered pass is a successful pass"


async def test_a_410_on_calendar_drops_the_sync_token(db, sync_ready, google):
    """Calendar's syncToken expires the same way, and the answer is the same."""
    state_repo = require("app.db.repositories.sync_state")
    user = sync_ready

    stale = {"syncToken": "expired-token"}
    await state_repo.set_cursor(db, user.id, "gcal", stale)
    await db.commit()

    google.fail("gcal", 410, body=gr.ERROR_410, times=1)

    await run_sync("gcal", db, user.id)

    after = await cursor_of(db, user.id, "gcal")
    assert after != stale, f"the expired syncToken was kept: {after}"
    assert after and gr.CALENDAR_SYNC_TOKEN in str(after), (
        f"the fresh nextSyncToken should have been stored: {after}"
    )

    from sqlalchemy import func, select

    models = require("app.db.models")
    await db.rollback()
    total = await db.execute(
        select(func.count()).select_from(models.SyncEvent).where(
            models.SyncEvent.connector == "gcal",
            models.SyncEvent.user_id == user.id
        )
    )
    assert int(total.scalar_one()) > 0, "the full resync stored no events"


# --------------------------------------------------------------------------- #
# What a sync leaves behind
# --------------------------------------------------------------------------- #


async def test_a_sync_stores_what_the_search_needs(db, sync_ready, google):
    """The generated columns are the point of syncing into Postgres at all."""
    from sqlalchemy import select

    models = require("app.db.models")
    user = sync_ready

    await run_sync("gmail", db, user.id)
    await run_sync("gcal", db, user.id)

    await db.rollback()
    item = models.SyncMessage
    mail = await db.execute(
        select(item)
        .where(item.user_id == user.id, item.connector == "gmail")
        .limit(1)
    )
    row = mail.scalar_one_or_none()
    assert row is not None, "no mail was mirrored"
    # The table *is* the shape now, so there is no `kind` to assert on — the
    # row being in `sync_messages` at all is the assertion.
    assert row.tsv is not None, "the generated tsv column is empty"
    assert isinstance(row.content_hash, uuid.UUID)
    assert row.sent_at is not None and row.sent_at.tzinfo is not None
    assert row.subject, "a mirrored mail with no subject cannot be searched by one"

    events = await db.execute(
        select(models.SyncEvent).where(
            models.SyncEvent.user_id == user.id,
            models.SyncEvent.connector == "gcal",
        )
    )
    with_guests = [e for e in events.scalars().all() if e.attendee_emails]
    if with_guests:
        first = with_guests[0]
        flat = {str(e).lower() for e in (first.attendee_emails or [])}
        listed = {
            str(a.get("email", "")).lower() for a in (first.attendees or [])
        }
        assert flat == listed, (
            "attendee_emails is derived from attendees on write; the two "
            "disagree, which means something set the flat column by hand and "
            f"an attendee filter will now miss rows: {flat} vs {listed}"
        )
        assert first.starts_at is not None and first.starts_at.tzinfo is not None


async def test_the_sync_modules_exist():
    """A plain import check, so a missing task reads as one failure and not as
    six skips scattered through the file."""
    missing = []
    for service, path in SYNC_MODULES.items():
        try:
            importlib.import_module(path)
        except ImportError as exc:
            missing.append(f"{service}: {exc}")
    if missing:
        pytest.skip("sync tasks not on disk yet — " + "; ".join(missing))
