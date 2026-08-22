"""A closed tab must not kill the run.

The browser drives a turn over the ndjson stream. Closing the tab aborts that
response, and Starlette closes the generator that was relaying events — which
is exactly what these tests do, with ``aclose()``, mid-run. Everything the
person then relies on has to hold anyway:

* the run keeps executing and reaches a terminal state,
* every event lands in the durable record, so ``GET /runs/{id}/events``
  can replay the whole thing to the tab that reopens,
* the idempotency memo is written, so re-sending the same
  ``client_request_id`` replays the finished run instead of starting a twin.

The failure mode this pins down is structural, not hypothetical: the run used
to share the request's database session, so the request's teardown yanked the
session out from under the run at its next write. The fix gives the detached
task its own session; these tests are the proof it stays fixed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

pytestmark = pytest.mark.anyio


def _new_request(query: str, **extra):
    from app.api.v1.schemas import QueryRequest

    return QueryRequest(query=query, **extra)


async def _read_until_started(gen) -> tuple[str, list[dict]]:
    """Consume the stream only as far as ``run.started``, like a browser that
    rendered the first frame and then closed."""
    seen: list[dict] = []
    async for line in gen:
        event = json.loads(line)
        seen.append(event)
        if event.get("type") == "run.started":
            return event["run_id"], seen
    raise AssertionError(
        f"stream ended without run.started; got {[e.get('type') for e in seen]}"
    )


async def _wait_terminal(db, run_id: str, timeout_s: float = 30.0) -> str:
    """Poll the runs table until the orphaned run settles."""
    from sqlalchemy import text

    waited = 0.0
    while waited < timeout_s:
        status = (
            await db.execute(
                text("SELECT status FROM runs WHERE id = :id"), {"id": run_id}
            )
        ).scalar_one_or_none()
        if status is not None and str(status) not in ("running", "queued", "pending"):
            return str(status)
        await asyncio.sleep(0.1)
        waited += 0.1
    raise AssertionError(f"run {run_id} never settled; still {status!r}")


async def _drain_detached(before: set) -> None:
    """Wait for any run task these tests spawned to finish completely.

    The whole point of the feature is that the run outlives its request — but
    it must not outlive the TEST: the conftest disposes the app engine and
    closes the event loop at teardown, and a task still running through
    either of those poisons the next test with cross-loop futures. Waiting is
    the entire cleanup — engine hygiene belongs to ``clean_database``.
    """
    from app.api.v1 import query as query_module

    for task in list(query_module._DETACHED - before):
        with contextlib.suppress(Exception):
            await task


async def test_closing_the_tab_mid_run_does_not_kill_the_run(
    client, db, user, mirrored, llm
):
    """Generator closed after the first event — the run still completes."""
    from app.api.v1 import query as query_module
    from app.orchestrator import events

    body = _new_request("What's on my calendar next week?")
    actor = query_module._actor_for(user, body)
    before = set(query_module._DETACHED)

    gen = query_module._ndjson(actor, body, request_id=None)
    try:
        run_id, _ = await _read_until_started(gen)

        # Starlette's client-disconnect path: close the response generator.
        # The GeneratorExit lands at the current yield, the relay dies, and
        # nothing else may die with it.
        await gen.aclose()

        status = await _wait_terminal(db, run_id)
        assert status == "complete", f"orphaned run ended {status!r}, not complete"

        # Let the task finish its tail before reading the buffer — the run row
        # commits a beat before the terminal event lands in it.
        await _drain_detached(before)

        # The durable record has the whole story for the tab that reopens:
        # replay must include the terminal event the closed tab never saw.
        replayed = await events.replay(run_id, 1)
        types = [e.get("type") for e in replayed]
        assert "run.started" in types
        assert "run.complete" in types, f"no terminal event in replay: {types}"
    finally:
        await _drain_detached(before)
        with contextlib.suppress(Exception):
            await db.rollback()


async def test_the_orphaned_run_still_writes_its_idempotency_memo(
    client, db, user, mirrored, llm
):
    """The memo used to be written by the relay, after the run — code that a
    disconnect never reaches. It belongs to the run task itself."""
    from app.api.v1 import query as query_module

    handle = "tab-closed-mid-run-000001"
    body = _new_request("What's on my calendar next week?", client_request_id=handle)
    actor = query_module._actor_for(user, body)
    before = set(query_module._DETACHED)

    gen = query_module._ndjson(actor, body, request_id=None)
    try:
        run_id, _ = await _read_until_started(gen)
        await gen.aclose()

        # Waiting for the task, not polling the row: the memo is the LAST
        # thing the task writes, so "task done" is the only honest signal.
        await _drain_detached(before)

        replayed = await query_module._replay(db, user.id, handle)
        assert replayed is not None, "idempotency memo missing after disconnect"
        assert replayed.get("run_id") == run_id, (
            "re-sending the same client_request_id would start a second run"
        )
    finally:
        await _drain_detached(before)
        with contextlib.suppress(Exception):
            await db.rollback()


async def test_a_connected_stream_still_ends_with_the_terminal_event(
    client, db, user, mirrored, llm
):
    """The refactor must not cost the happy path: a client that stays gets the
    full stream, terminal event last."""
    from app.api.v1 import query as query_module

    body = _new_request("What's on my calendar next week?")
    actor = query_module._actor_for(user, body)
    before = set(query_module._DETACHED)

    try:
        lines = [json.loads(line) async for line in query_module._ndjson(actor, body, None)]
        types = [e.get("type") for e in lines]
        assert types[0] == "run.started"
        assert types[-1] == "run.complete", f"stream ended badly: {types}"
    finally:
        await _drain_detached(before)
