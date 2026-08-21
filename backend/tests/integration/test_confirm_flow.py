"""The two-phase write: prepare, then a person, then execute.

No endpoint sends an email, moves an event or shares a file. A query prepares;
somebody approves; a worker executes. The database is what enforces it rather
than the code being careful — ``actions.requires_input_id`` is NOT NULL and
references ``pending_inputs(id)``, so a confirm-requiring write physically
cannot exist without a prompt gating it.

What is checked here:

* preparing a send creates a **Gmail draft** — reversible, visible in the
  person's own account — and nothing else reaches Google;
* the ``actions`` row exists, is ``draft``, and points at the card;
* approving executes it exactly once;
* approving **twice** still sends once. Two guards, and both are tested: the
  prompt's status check, and the partial unique index on ``dedupe_key``;
* "Not now" cancels the action and deletes the draft;
* "Edit" patches the payload, keeps the old one in ``revisions``, and is refused
  once the action has been approved;
* two writes under one card run in order, and the second is held when the first
  fails. The email announcing a move never goes out when the move did not
  happen.
"""

from __future__ import annotations

import pytest

from tests.fixtures import google_responses as gr
from tests.integration.conftest import (
    confirm_card,
    confirm_value,
    execute_approved_actions,
    load_actions,
    load_prompts,
    load_step_rows,
    post_query,
    require,
    respond_to_prompt,
    status_of,
)

pytestmark = pytest.mark.integration

CANCEL_FLIGHT = "Cancel my Turkish Airlines flight"
PUSH_REVIEW = "Push my Acme review next Thursday to Friday 3pm and tell the attendees"


async def prepare_a_send(client, db, user, llm):
    """Run the flight cancellation and hand back (payload, card, actions)."""
    llm.use("cancel_turkish_flight")
    payload = await post_query(client, CANCEL_FLIGHT)
    card = confirm_card(payload)
    assert card is not None, (
        f"a write with no confirm card: {payload.get('pending_inputs')}"
    )
    actions = await load_actions(db, user.id)
    assert actions, "nothing was written to actions"
    return payload, card, actions


# --------------------------------------------------------------------------- #
# Prepare
# --------------------------------------------------------------------------- #


async def test_a_write_prepares_and_stops(client, db, mirrored, llm, google):
    """The run finishes; the send does not happen.

    A draft is created up front on purpose: it is reversible, and it means the
    card shows the actual message being approved rather than a rendering of what
    we intend to write.
    """
    payload, card, actions = await prepare_a_send(client, db, mirrored, llm)

    assert payload["status"] == "complete", (
        "a non-blocking confirm does not hold the run open — the person can "
        "approve tomorrow"
    )
    assert card["blocking"] is False
    assert card["kind"] == "confirm"
    assert card["status"] == "pending"
    assert card.get("expires_at"), "a card that never expires is a card nobody sweeps"

    assert not google.sends, (
        f"something was sent before anyone approved it: {[str(r) for r in google.sends]}"
    )
    assert not google.mutations, (
        f"a query changed the account: {[str(r) for r in google.mutations]}"
    )
    drafts = google.calls("gmail", contains="/drafts")
    assert drafts, (
        "the send should have been prepared as a real Gmail draft, so the person "
        "can see exactly what they are approving"
    )

    send = next((a for a in actions if a.op == "gmail.send_email"), None)
    assert send is not None, f"no send action: {[a.op for a in actions]}"
    assert status_of(send) == "draft"
    assert send.requires_input_id == card["id"], (
        "requires_input_id is NOT NULL and must point at the card on screen"
    )
    assert send.dedupe_key is not None
    assert send.payload, "the payload is a snapshot of exactly what will execute"
    assert send.external_ref, "external_ref is the Gmail draft that already exists"

    reported = payload.get("actions") or []
    assert reported and reported[0]["status"] == "draft", (
        "the response has to say `draft`; a client that reads this as 'done' "
        f"tells the person a lie: {reported}"
    )
    assert "payload" not in (reported[0] if reported else {}), (
        "the full payload is deliberately not returned — the preview is what "
        "Op.preview() chose to show"
    )


async def test_the_database_refuses_an_ungated_write(db, user):
    """The guarantee, tested at the level that provides it.

    ``requires_input_id`` is NOT NULL. An action with no prompt behind it cannot
    be written, whatever the application layer believes — so a bug in the
    orchestrator cannot produce a write nobody agreed to.

    Everything else about the row is valid, so the only thing this can fail on
    is the column under test.
    """
    from sqlalchemy.exc import DBAPIError, IntegrityError

    conversations = require("app.db.repositories.conversations")
    models = require("app.db.models")
    ids = require("app.core.ids")

    conversation = await conversations.create_conversation(db, user.id, title="gate")
    message = await conversations.add_message(
        db,
        user.id,
        conversation.id,
        role="assistant",
        content=[{"type": "text", "data": {"markdown": "prepared"}}],
    )
    await db.commit()

    action = models.Action(
        id=ids.new_id(),
        message_id=message.id,
        user_id=user.id,
        requires_input_id=None,
        op="gmail.send_email",
        payload={"to": ["nobody@example.com"], "subject": "unauthorised"},
        dedupe_key=ids.fingerprint("test", "ungated"),
        status=models.ActionStatus.DRAFT,
    )
    db.add(action)
    with pytest.raises((IntegrityError, DBAPIError)) as raised:
        await db.flush()
    assert "requires_input_id" in str(raised.value), (
        f"the row failed for some other reason: {raised.value}"
    )
    await db.rollback()


# --------------------------------------------------------------------------- #
# Approve
# --------------------------------------------------------------------------- #


async def test_approving_executes_the_write_once(client, db, mirrored, llm, google):
    """Send it. Zero LLM calls — the decision came from a person."""
    _, card, _ = await prepare_a_send(client, db, mirrored, llm)
    before = llm.calls

    result = await respond_to_prompt(client, card["id"], confirm_value(card))
    assert result.get("status") == "answered"
    if "llm_calls" in result:
        assert result["llm_calls"] == 0
    assert llm.calls == before, "approving a prepared write is not a thinking task"

    await execute_approved_actions(db, mirrored.id)

    sends = google.sends
    assert len(sends) == 1, (
        f"expected exactly one send, got {len(sends)}: {[str(r) for r in sends]}"
    )
    assert sends[0].method == "POST"

    actions = await load_actions(db, mirrored.id, op="gmail.send_email")
    assert actions and status_of(actions[0]) == "done", (
        f"the action should be done: {[(a.op, status_of(a)) for a in actions]}"
    )
    assert actions[0].executed_at is not None
    assert actions[0].result, "whatever Google gave back belongs in result"

    prompts = await load_prompts(db, mirrored.id)
    assert all(status_of(p) != "pending" for p in prompts)


async def test_approving_twice_sends_once(client, db, mirrored, llm, google):
    """The person double-taps Send. Nothing goes out twice.

    First guard: the prompt is no longer pending, so the second call is a 409.
    """
    _, card, _ = await prepare_a_send(client, db, mirrored, llm)
    value = confirm_value(card)

    await respond_to_prompt(client, card["id"], value)
    second = await client.post(f"/api/v1/prompts/{card['id']}/respond", json={"value": value})

    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "PROMPT_NOT_PENDING"
    assert second.json()["error"]["details"].get("status") == "answered"

    await execute_approved_actions(db, mirrored.id)
    assert len(google.sends) == 1, (
        f"the second approval sent again: {[str(r) for r in google.sends]}"
    )


async def test_the_partial_index_dedupes_in_flight_writes_only(
    client, db, mirrored, llm
):
    """The second guard, and the reason the unique index is partial.

    While an identical action is ``draft``/``approved``/``running`` it cannot be
    duplicated. Once it is ``cancelled`` — or done, or failed — an identical
    request is a legitimately new one: a resend after cancelling, the same
    reminder next week. That is a deliberate property of the schema, so it gets
    a test rather than a comment.
    """
    actions_repo = require("app.db.repositories.actions")
    _, card, actions = await prepare_a_send(client, db, mirrored, llm)
    original = next(a for a in actions if a.op == "gmail.send_email")

    # Taken off the row now, while it is still loaded: everything below goes
    # through a rollback, and a stale ORM handle is not worth debugging.
    action_id = original.id
    message_id = original.message_id
    input_id = original.requires_input_id
    op = original.op
    payload = dict(original.payload)
    dedupe_key = original.dedupe_key

    await db.rollback()
    twin = await actions_repo.create_action(
        db,
        mirrored.id,
        message_id,
        requires_input_id=input_id,
        op=op,
        payload=payload,
        dedupe_key=dedupe_key,
    )
    assert twin.id == action_id, (
        "an identical in-flight action must resolve to the row that already "
        "exists rather than making a second one"
    )

    await actions_repo.cancel_action(db, mirrored.id, action_id)
    await db.commit()

    resend = await actions_repo.create_action(
        db,
        mirrored.id,
        message_id,
        requires_input_id=input_id,
        op=op,
        payload=payload,
        dedupe_key=dedupe_key,
    )
    await db.commit()
    assert resend.id != action_id, (
        "after a cancel the same request is allowed again — that is exactly why "
        "the unique index covers only draft, approved and running"
    )


# --------------------------------------------------------------------------- #
# Not now
# --------------------------------------------------------------------------- #


async def test_cancelling_the_card_deletes_the_gmail_draft(
    client, db, mirrored, llm, google
):
    """"Not now" is not just a status change.

    The draft was created in the person's own account, so declining has to clean
    it up. Leaving it behind means a half-written cancellation sitting in their
    Drafts folder forever.
    """
    _, card, _ = await prepare_a_send(client, db, mirrored, llm)

    response = await client.post(f"/api/v1/prompts/{card['id']}/cancel")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "cancelled"

    deletes = google.calls("gmail", method="DELETE", contains="/drafts")
    assert deletes, (
        "the Gmail draft should have been deleted; Google saw "
        f"{[str(r) for r in google.calls('gmail')]}"
    )
    assert gr.GMAIL_DRAFT_ID in deletes[0].path, (
        f"the wrong draft was deleted: {deletes[0].path}"
    )

    actions = await load_actions(db, mirrored.id)
    assert all(status_of(a) == "cancelled" for a in actions), [
        (a.op, status_of(a)) for a in actions
    ]
    assert not google.sends


# --------------------------------------------------------------------------- #
# Edit
# --------------------------------------------------------------------------- #


async def test_editing_patches_the_payload_and_keeps_the_old_one(
    client, db, mirrored, llm, google
):
    """Edit is 0 LLM calls: the change came from the person.

    The old payload goes onto ``revisions`` so the trail of what was proposed
    survives, and a fresh card is raised against the revised payload.
    """
    _, card, actions = await prepare_a_send(client, db, mirrored, llm)
    send = next(a for a in actions if a.op == "gmail.send_email")
    send_id, before_payload = send.id, dict(send.payload)
    before_calls = llm.calls

    patch = {"subject": f"Cancellation request - booking {gr.PNR} (urgent)"}
    result = await respond_to_prompt(
        client, card["id"], confirm_value(card, approve=False, patch=patch), expect=(200, 202)
    )
    assert llm.calls == before_calls, "an edit is not a planning problem"

    actions_after = await load_actions(db, mirrored.id, op="gmail.send_email")
    edited = next(a for a in actions_after if a.id == send_id)
    assert status_of(edited) == "draft", "an edited action is still waiting on a yes"
    assert edited.payload.get("subject") == patch["subject"], edited.payload
    assert edited.revisions, "the previous payload should be on the revisions trail"
    assert edited.revisions[-1]["payload"].get("subject") == before_payload.get("subject")

    next_card = result.get("next_input") or {}
    if next_card:
        assert next_card["status"] == "pending"
        assert next_card["id"] != card["id"], "the old card is spent; a new one asks again"

    updates = google.calls("gmail", contains="/drafts")
    assert any(r.method in {"PUT", "PATCH", "POST"} for r in updates), (
        "the Gmail draft should have been updated in place, so external_ref does "
        "not change and the person sees the edit in their own Drafts folder"
    )
    assert not google.sends


async def test_editing_after_approval_is_refused(client, db, mirrored, llm):
    """Once it is approved the payload is what will execute. It is frozen."""
    actions_repo = require("app.db.repositories.actions")
    _, card, actions = await prepare_a_send(client, db, mirrored, llm)
    send_id = next(a.id for a in actions if a.op == "gmail.send_email")

    await respond_to_prompt(client, card["id"], confirm_value(card))

    await db.rollback()
    errors = require("app.core.errors")
    with pytest.raises(errors.AppError) as raised:
        await actions_repo.revise_action(db, mirrored.id, send_id, {"subject": "too late"})
    assert raised.value.http == 409
    await db.rollback()

    fresh = await load_actions(db, mirrored.id, op="gmail.send_email")
    assert fresh[0].payload.get("subject") != "too late"


# --------------------------------------------------------------------------- #
# Two writes, one card
# --------------------------------------------------------------------------- #


async def sequence_marker(db, user_id: str, run_id: str, first, second) -> str | None:
    """Whatever records "run these in this order", or None if nothing does.

    The queued action can carry it as ``_sequence_after`` so the worker does not
    have to re-read the DAG, or the edge can stay on the node executions the two
    actions came from. Either is a real mechanism. Neither means the ordering is
    a coincidence of scheduling, and coincidences send emails about meetings
    that did not move.
    """
    first_id = first.id
    payload = second.payload or {}
    upstream_node = getattr(first, "node_execution_id", None)
    downstream_node = getattr(second, "node_execution_id", None)

    for key in ("_sequence_after", "sequence_after", "after_action_id", "depends_on"):
        value = payload.get(key)
        if value == first_id:
            return key
        if isinstance(value, (list, tuple)) and first_id in value:
            return key

    rows = await load_step_rows(db, user_id, run_id)
    by_row_id = {r.id: r for r in rows}
    upstream = by_row_id.get(upstream_node)
    downstream = by_row_id.get(downstream_node)
    if upstream and downstream and upstream.node_id in (downstream.depends_on or ()):
        return f"node_executions.depends_on ({downstream.node_id} -> {upstream.node_id})"
    return None


async def prepare_two_writes(client, db, user, llm):
    llm.use("push_acme_review")
    payload = await post_query(client, PUSH_REVIEW)
    card = confirm_card(payload)
    assert card is not None, payload.get("pending_inputs")
    actions = await load_actions(db, user.id)
    return payload, card, actions


async def test_one_card_gates_two_actions_in_order(client, db, mirrored, llm, google):
    """Move the meeting, then tell the guests. One prompt, two actions.

    ``notify`` depends on ``move`` even though the email body needs nothing from
    the move's result. The edge exists to enforce ordering, which is the whole
    point: an email announcing a move must not go out before the move happened.
    """
    payload, card, actions = await prepare_two_writes(client, db, mirrored, llm)

    ops = {a.op for a in actions}
    assert {"gcal.update_event", "gmail.send_email"} <= ops, (
        f"expected both writes to be prepared: {ops}"
    )
    assert len({a.requires_input_id for a in actions}) == 1, (
        "two actions, one card — that is what separate `pending_inputs` and "
        "`actions` tables buy"
    )
    assert all(a.requires_input_id == card["id"] for a in actions)
    assert all(status_of(a) == "draft" for a in actions)

    move = next(a for a in actions if a.op == "gcal.update_event")
    notify = next(a for a in actions if a.op == "gmail.send_email")
    payloads = f"\n  move:   {move.payload}\n  notify: {notify.payload}"

    marker = await sequence_marker(db, mirrored.id, payload["run_id"], move, notify)
    assert marker is not None, (
        "nothing records that the email must wait for the calendar change. "
        "Expected `_sequence_after` on the email's payload, or a dependency "
        f"edge on its node execution.{payloads}"
    )

    assert not google.mutations, [str(r) for r in google.mutations]

    blocks = [b.get("ref") for b in payload.get("content") or [] if b.get("type") == "action"]
    assert len(blocks) >= 2 or len(payload.get("actions") or []) >= 2, (
        "the message should reference both actions, so the card can show two "
        f"previews under one button: {payload.get('content')}"
    )


async def test_the_second_write_is_held_when_the_first_one_fails(
    client, db, mirrored, llm, google
):
    """Somebody else moved the event while the person was deciding.

    Google answers the update with 412 — PRECONDITION, not retryable. The email
    must not go out: it would tell three people about a change that did not
    happen.
    """
    _, card, actions = await prepare_two_writes(client, db, mirrored, llm)

    # Only the update breaks. Reading the event still works, which is what it
    # looks like when somebody else edited it — the etag we hold is stale, the
    # event is fine.
    google.fail("gcal", 412, body=gr.ERROR_412, method="PATCH")
    google.fail("gcal", 412, body=gr.ERROR_412, method="PUT")
    await respond_to_prompt(client, card["id"], confirm_value(card))
    await execute_approved_actions(db, mirrored.id)

    after = await load_actions(db, mirrored.id)
    move = next(a for a in after if a.op == "gcal.update_event")
    notify = next(a for a in after if a.op == "gmail.send_email")

    assert status_of(move) == "failed", (
        f"a 412 is not retryable and should have failed the move: {status_of(move)}"
    )
    assert "412" in str(move.error) or "precondition" in str(move.error).lower(), (
        f"the failure class should name the precondition: {move.error}"
    )

    assert status_of(notify) in {"cancelled", "expired", "draft", "approved"}, (
        f"the email must not have run: {status_of(notify)}"
    )
    assert status_of(notify) != "done"
    assert not google.sends, (
        "the email announcing a move went out even though the move failed: "
        f"{[str(r) for r in google.sends]}"
    )

    if status_of(notify) == "cancelled":
        assert "upstream" in str(notify.error).lower() or "failed" in str(notify.error).lower(), (
            f"say why it was cancelled: {notify.error}"
        )
