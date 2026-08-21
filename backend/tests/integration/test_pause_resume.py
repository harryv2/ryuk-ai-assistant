"""Pausing on a question, and coming back to it.

"Move the meeting with John" has three unknowns — which John, which meeting,
when — and the design's answer is that asking is a **step**. ``ask.user`` is an
op like any other, so a paused run is a node in ``running`` and a run in
``awaiting_input``. There is no separate clarification subsystem to keep in
sync, and that is what makes the two properties below possible:

* **the resume costs nothing.** The plan is already on disk. Answering binds one
  value and re-enters the dispatcher — zero LLM calls, which this file asserts
  by counting completions before and after;
* **the resume survives a restart.** The plan lives in ``node_executions``, not
  in a worker's memory, so a resume can be rebuilt from rows alone. The test
  throws away every in-process cache first and then answers the card.

If a run kept its plan in memory it would pass neither: it would either fail to
resume, or re-plan and spend a second call.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import (
    blocking_prompt,
    confirm_value,
    load_actions,
    load_prompts,
    load_run,
    load_step_rows,
    post_query,
    require,
    respond_to_prompt,
    settle_run,
    simulate_restart,
    status_of,
)

pytestmark = pytest.mark.integration

AMBIGUOUS = "Move the meeting with John"

#: The two Johns in the corpus, and the answer the person gives.
JOHN_OKAFOR_EVENT = "3k9m2p_20260825T200000Z"
JOHN_REYES_EVENT = "7t4v8q_20260826T130000Z"


def answer_for(card: dict) -> object:
    """The value a person's click produces, shaped to the card's own schema.

    ``value_schema`` is the validation authority, so the test reads it rather
    than assuming a shape — a `choice` card takes the bare id, a `form` takes an
    object.
    """
    schema = card.get("value_schema") or {}
    kind = card.get("kind")

    if kind in {"choice", "multi_choice"} or schema.get("type") == "string":
        options = card.get("options") or []
        if options:
            chosen = options[0].get("id") or options[0].get("value")
            return [chosen] if kind == "multi_choice" else chosen
        enum = schema.get("enum") or []
        if enum:
            return enum[0] if kind != "multi_choice" else [enum[0]]
        return JOHN_OKAFOR_EVENT

    if kind == "confirm":
        return confirm_value(card)

    if kind == "text":
        return "Friday 3pm"

    # form
    value: dict[str, object] = {}
    for name, spec in (schema.get("properties") or {}).items():
        if spec.get("enum"):
            value[name] = spec["enum"][0]
        elif "time" in name or "when" in name:
            value[name] = "Friday 3pm"
        elif spec.get("type") == "boolean":
            value[name] = True
        else:
            value[name] = "Friday 3pm"
    return value or {"event_id": JOHN_OKAFOR_EVENT, "new_time": "Friday 3pm"}


async def pause_on_the_card(client, db, user, llm):
    """Drive the ambiguous query and hand back (payload, card)."""
    llm.use("move_meeting_john")
    payload = await post_query(client, AMBIGUOUS)

    assert payload["status"] == "awaiting_input", (
        f"expected the run to park on a question, got {payload['status']}: "
        f"{payload.get('text')}"
    )
    card = blocking_prompt(payload)
    assert card is not None, (
        f"awaiting_input with no blocking prompt: {payload.get('pending_inputs')}"
    )
    return payload, card


# --------------------------------------------------------------------------- #
# Parking
# --------------------------------------------------------------------------- #


async def test_an_ambiguous_query_parks_on_a_blocking_card(
    client, db, mirrored, llm, google
):
    """Two meetings with a John, 0.04 apart — under MARGIN, so it asks.

    Both unknowns go in one card. One round trip to the human, not two.
    """
    payload, card = await pause_on_the_card(client, db, mirrored, llm)

    assert llm.calls == 1, "planning the question is the only model call here"
    assert card["kind"] in {"choice", "form", "multi_choice"}, card["kind"]
    assert card["blocking"] is True
    assert card["status"] == "pending"
    assert card.get("value_schema"), "a card without a schema cannot be validated"

    shown = str(card.get("options")) + str(card.get("value_schema"))
    assert JOHN_OKAFOR_EVENT in shown and JOHN_REYES_EVENT in shown, (
        "both candidates should be on the card, and the schema should pin the "
        f"choice to them so a client cannot smuggle in another event:\n{shown}"
    )

    run = await load_run(db, mirrored.id, payload["run_id"])
    assert status_of(run) == "awaiting_input"

    rows = await load_step_rows(db, mirrored.id, payload["run_id"])
    by_node = {r.node_id: r for r in rows}
    assert len(by_node) >= 2, f"the whole plan should be on disk, got {list(by_node)}"

    downstream = [r for r in rows if r.depends_on]
    assert downstream, "the plan should have steps waiting behind the question"
    assert all(status_of(r) in {"pending", "running"} for r in downstream), (
        "nothing behind the question may have run: "
        f"{[(r.node_id, status_of(r)) for r in downstream]}"
    )

    assert not google.mutations, (
        f"a paused write touched Google: {[str(r) for r in google.mutations]}"
    )


async def test_a_parked_run_is_findable_after_a_restart(client, db, mirrored, llm):
    """The partial index on ``runs(status)`` is how a worker picks a run back up.

    Nothing else is needed: the row is enough to find it, and
    ``node_executions`` is enough to continue it.
    """
    runs = require("app.db.repositories.runs")
    payload, _ = await pause_on_the_card(client, db, mirrored, llm)

    simulate_restart()

    await db.rollback()
    resumable = await runs.list_resumable(db, mirrored.id)
    assert payload["run_id"] in {r.id for r in resumable}, (
        "a paused run has to be findable by a worker that has just started; "
        f"list_resumable returned {[r.id for r in resumable]}"
    )


# --------------------------------------------------------------------------- #
# Resuming
# --------------------------------------------------------------------------- #


async def test_answering_the_card_resumes_the_run_for_free(
    client, db, mirrored, llm, google
):
    """The heart of it: the resume path spends nothing.

    The plan is on disk, the value came from a person, and the time phrase is
    parsed in Python. There is nothing left to ask a model.
    """
    payload, card = await pause_on_the_card(client, db, mirrored, llm)
    run_id = payload["run_id"]
    before = llm.calls

    result = await respond_to_prompt(client, card["id"], answer_for(card), expect=200)

    assert llm.calls == before, (
        f"the resume spent {llm.calls - before} extra LLM call(s). Prompts sent "
        f"after the pause: {[p[:160] for p in llm.prompts[before:]]}"
    )
    if "llm_calls" in result:
        assert result["llm_calls"] == 0
    if "resumed_run_id" in result:
        assert result["resumed_run_id"] == run_id
    assert result.get("status") == "answered"

    run = await settle_run(db, mirrored.id, run_id)
    assert status_of(run) == "complete", (
        f"the run should have finished after the answer, it is {status_of(run)}"
    )
    assert llm.calls == before, "settling the run must not have cost a call either"

    rows = await load_step_rows(db, mirrored.id, run_id)
    ran = [r for r in rows if status_of(r) == "succeeded"]
    assert len(ran) >= 2, (
        "the steps behind the question should have run once it was answered: "
        f"{[(r.node_id, status_of(r)) for r in rows]}"
    )

    # The move is a write, so it prepares and stops. Still nothing sent.
    assert not google.mutations or all(
        r.method == "GET" for r in google.mutations
    ), [str(r) for r in google.mutations]

    prompts = await load_prompts(db, mirrored.id)
    answered = [p for p in prompts if p.id == card["id"]]
    assert answered and status_of(answered[0]) == "answered"
    assert answered[0].response is not None, "the answer itself is kept on the row"


async def test_the_resume_is_rebuilt_from_node_executions_alone(
    client, db, mirrored, llm
):
    """Pull the process out from under the run, then answer the card.

    Every in-memory cache the orchestrator holds is cleared and the database
    engine is dropped, so the only thing left describing this run is its rows.
    If the resume works from those, and costs nothing, the plan really is on
    disk — and if it silently re-planned instead, the call count says so.
    """
    payload, card = await pause_on_the_card(client, db, mirrored, llm)
    run_id = payload["run_id"]
    before = llm.calls

    rows_before = await load_step_rows(db, mirrored.id, run_id)
    ids_before = {r.node_id: r.id for r in rows_before}
    assert ids_before, "nothing was persisted for this run"
    for row in rows_before:
        assert row.args is not None, f"{row.node_id} stored no args"
        assert row.round == 0

    cleared = simulate_restart()

    result = await respond_to_prompt(client, card["id"], answer_for(card), expect=200)
    assert result.get("status") == "answered"

    run = await settle_run(db, mirrored.id, run_id)
    assert status_of(run) == "complete", (
        f"the run did not survive the restart (cleared: {cleared}); it is "
        f"{status_of(run)}"
    )
    assert llm.calls == before, (
        "resuming after a restart re-planned instead of reading the rows: "
        f"{[p[:160] for p in llm.prompts[before:]]}"
    )

    rows_after = await load_step_rows(db, mirrored.id, run_id)
    same_round = {r.node_id: r.id for r in rows_after if r.round == 0}
    assert same_round == ids_before, (
        "round 0 was rewritten rather than continued — a resume reuses the rows "
        f"it finds:\n before {ids_before}\n after  {same_round}"
    )


async def test_cancelling_a_blocking_card_cancels_the_run(client, db, mirrored, llm):
    """"Never mind" stops the run and keeps the trace.

    The steps that already ran keep their rows; only the run's status moves.
    """
    payload, card = await pause_on_the_card(client, db, mirrored, llm)
    before = llm.calls

    response = await client.post(f"/api/v1/prompts/{card['id']}/cancel")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "cancelled"

    run = await load_run(db, mirrored.id, payload["run_id"])
    assert status_of(run) == "cancelled"
    assert llm.calls == before, "declining a question costs nothing"

    rows = await load_step_rows(db, mirrored.id, payload["run_id"])
    assert rows, "the trace should survive a cancelled run"

    actions = await load_actions(db, mirrored.id)
    assert all(status_of(a) in {"cancelled", "expired"} for a in actions), (
        f"a cancelled run must not leave a live action behind: "
        f"{[(a.op, status_of(a)) for a in actions]}"
    )


async def test_answering_a_card_twice_is_refused(client, db, mirrored, llm):
    """Two clicks, one answer. The guard is in the WHERE clause, not in a check."""
    _, card = await pause_on_the_card(client, db, mirrored, llm)
    value = answer_for(card)

    await respond_to_prompt(client, card["id"], value, expect=200)
    again = await client.post(f"/api/v1/prompts/{card['id']}/respond", json={"value": value})

    assert again.status_code == 409, again.text
    body = again.json()
    assert body["error"]["code"] == "PROMPT_NOT_PENDING"
    assert body["error"]["details"].get("status") in {
        "answered",
        "cancelled",
        "expired",
        "superseded",
    }


async def test_an_answer_that_does_not_fit_the_schema_is_refused(
    client, db, mirrored, llm
):
    """``value_schema`` is the validation authority, and it is enforced.

    The enum on the card exists so a client cannot name an event the person was
    never shown. That is a security property, not a nicety.
    """
    _, card = await pause_on_the_card(client, db, mirrored, llm)

    response = await client.post(
        f"/api/v1/prompts/{card['id']}/respond",
        json={"value": {"event_id": "somebody-elses-event", "new_time": ""}},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "PROMPT_VALUE_INVALID"

    run_id = (await load_prompts(db, mirrored.id))[0].run_id
    run = await load_run(db, mirrored.id, run_id)
    assert status_of(run) == "awaiting_input", (
        "a rejected answer leaves the run exactly where it was"
    )
