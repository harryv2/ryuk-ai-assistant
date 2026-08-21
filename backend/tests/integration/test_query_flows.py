"""The brief's sample queries, end to end.

Four things are checked on every flow, and the third is the one that is hard to
fake:

1. **the plan shape** — which ops ran, and what depended on what;
2. **the answer** — that the words the person reads come from their own data;
3. **which steps ran at the same time** — proved from the ``started_at`` and
   ``finished_at`` stamps in ``node_executions``: two independent steps must
   *overlap*. A sequential dispatcher passes every other assertion in this file
   and fails that one, which is the whole point of writing it that way;
4. **the LLM call count** — the number in the response has to match the number
   of completions that actually left the process. A run that under-reports its
   own cost is lying about the thing the design is proudest of.

Nothing here touches the network. Google and OpenAI are served from
``tests/fixtures``; Postgres is real, so the hybrid search, the generated
``tsv`` columns and the ``attendee_emails`` GIN index are genuinely exercised.
"""

from __future__ import annotations

import pytest

from tests.fixtures import google_responses as gr
from tests.integration.conftest import (
    answer_text,
    assert_ran_concurrently,
    assert_ran_in_order,
    confirm_card,
    load_actions,
    load_run,
    load_steps,
    post_query,
    require,
    seed_mirror,
    status_of,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def reported_calls(payload: dict) -> int | None:
    """What the run says it spent, if it says."""
    usage = payload.get("usage") or {}
    for key in ("llm_calls", "calls"):
        if isinstance(usage.get(key), int):
            return usage[key]
    if isinstance(payload.get("llm_calls"), int):
        return payload["llm_calls"]
    return None


def assert_cost(payload: dict, llm, expected: int) -> None:
    """The honest count: what left the process, and what the run admits to."""
    assert llm.calls == expected, (
        f"expected {expected} completion call(s), got {llm.calls}. "
        f"Prompts sent: {[p[:120] for p in llm.prompts]}"
    )
    claimed = reported_calls(payload)
    if claimed is not None:
        assert claimed == llm.calls, (
            f"the response reports {claimed} LLM calls but {llm.calls} were made"
        )


def ops_of(payload: dict) -> list[str]:
    return [str(s.get("op", "")) for s in payload.get("steps") or []]


def services_of(payload: dict) -> set[str]:
    return {op.split(".", 1)[0] for op in ops_of(payload) if "." in op}


# --------------------------------------------------------------------------- #
# 1. "What's on my calendar next week?"  — 0 LLM calls
# --------------------------------------------------------------------------- #


async def test_calendar_next_week_is_answered_without_a_model(client, db, mirrored, llm):
    """The rule router owns this: one op, one window, nothing to disambiguate.

    Zero completions *and* zero embeddings — a router hit skips the probe
    entirely, which is where the embedding would have been spent.
    """
    payload = await post_query(client, "What's on my calendar next week?")

    assert payload["status"] == "complete"
    assert_cost(payload, llm, 0)
    assert llm.embedding_calls == 0, (
        "a rule-router hit must skip the probe; an embedding was requested"
    )

    assert services_of(payload) <= {"gcal"}, f"only Calendar should run: {ops_of(payload)}"

    text = answer_text(payload)
    for expected in ("Standup", "Design review", "Acme review"):
        assert expected in text, f"{expected!r} missing from the answer:\n{text}"

    if not gr.TOMORROW_IS_NEXT_WEEK:
        assert "Q3 renewal review" not in text, (
            "that meeting is tomorrow, not next week. The window is half-open "
            f"[{gr.NEXT_WEEK_START:%a %d %b}, {gr.NEXT_WEEK_END:%a %d %b}) and "
            f"must not reach back to today:\n{text}"
        )

    run = await load_run(db, mirrored.id, payload["run_id"])
    assert status_of(run) == "complete"


# --------------------------------------------------------------------------- #
# 2. "Find emails from sarah@company.com about the budget"
# --------------------------------------------------------------------------- #


async def test_emails_from_a_named_sender_about_a_topic(client, db, mirrored, llm):
    """One planner call, one gmail step, and the answer is her mail and no one
    else's."""
    payload = await post_query(
        client, "Find emails from sarah@company.com about the budget"
    )

    assert payload["status"] == "complete"
    assert_cost(payload, llm, 1)
    assert llm.embedding_calls == 1, "the probe embeds the query exactly once"
    assert services_of(payload) <= {"gmail"}

    text = answer_text(payload)
    assert "budget" in text.lower()
    assert "turkish" not in text.lower(), (
        "the airline booking shares no vocabulary with a budget question and "
        f"must not surface:\n{text}"
    )

    spans = await load_steps(db, mirrored.id, payload["run_id"])
    assert spans, "the run recorded no steps"
    assert all(s.status == "succeeded" for s in spans.values()), list(map(str, spans.values()))


# --------------------------------------------------------------------------- #
# 3. "Show me PDFs in Drive from last month"
# --------------------------------------------------------------------------- #


async def test_drive_pdfs_from_last_month(client, mirrored, llm):
    """`sync_gdrive` has no created date, so "from last month" can only mean
    *modified* last month. The answer should be July's PDFs and nothing else."""
    payload = await post_query(client, "Show me PDFs in Drive from last month")

    assert payload["status"] == "complete"
    assert_cost(payload, llm, 1)
    assert services_of(payload) <= {"gdrive"}

    text = answer_text(payload)
    assert "Invoice_TK_1984.pdf" in text or "Invoice" in text, text
    assert "Acme - Q3 renewal proposal v4.gdoc" not in text, (
        "that one is a Google Doc modified in August; neither the mime type nor "
        f"the window admits it:\n{text}"
    )


# --------------------------------------------------------------------------- #
# 4. "Cancel my Turkish Airlines flight" — the headline orchestration
# --------------------------------------------------------------------------- #


async def test_cancel_flight_fans_out_then_prepares_a_write(client, db, mirrored, llm, google):
    """Gmail and Calendar in parallel, a sequential dependency, one prepared
    write — on one LLM call.

    The DAG the planner returned:

        booking  ──────────┐
                           ├──> draft ──> send (prepared, gated)
        flight_event ──────┘

    ``booking`` and ``flight_event`` both depend on nothing, so they start in
    the same event-loop tick. ``draft`` waits on ``booking`` only — the calendar
    event is reported to the person but is not an input to the email.
    """
    payload = await post_query(client, "Cancel my Turkish Airlines flight")

    assert payload["status"] in {"complete", "awaiting_input"}
    assert_cost(payload, llm, 1)
    assert llm.embedding_calls == 1, "one embedding for the probe, no more"

    ops = ops_of(payload)
    assert "gmail.get_email" in ops or "gmail.search_emails" in ops, ops
    assert any(op.startswith("gcal.") for op in ops), f"Calendar never ran: {ops}"
    assert "gmail.send_email" in ops, f"the send step is missing from the plan: {ops}"

    # -- the parallelism, from the rows themselves ------------------------- #
    spans = await load_steps(db, mirrored.id, payload["run_id"])
    independent = [n for n, s in spans.items() if not s.depends_on and s.started_at]
    assert len(independent) >= 2, (
        "two steps in this plan depend on nothing and should have been launched "
        f"together; only {independent} ever started"
    )
    assert_ran_concurrently(spans, independent[0], independent[1])

    draft = next((n for n, s in spans.items() if s.op == "gmail.draft_email"), None)
    if draft is not None:
        dependency = spans[draft].depends_on
        assert dependency, "the draft step should depend on the booking it quotes"
        assert_ran_in_order(spans, dependency[0], draft)

    # -- prepared, not performed ------------------------------------------- #
    assert not google.sends, (
        f"a send reached Google before anyone approved it: {[str(r) for r in google.sends]}"
    )
    assert not google.mutations, [str(r) for r in google.mutations]

    card = confirm_card(payload)
    assert card is not None, (
        f"a write without a confirm card: {payload.get('pending_inputs')}"
    )
    assert card["blocking"] is False, (
        "a confirm on a finished run does not block it — the person can walk "
        "away and approve tomorrow"
    )

    actions = await load_actions(db, mirrored.id)
    assert len(actions) >= 1, "no action row was written for the send"
    send = next((a for a in actions if a.op == "gmail.send_email"), actions[0])
    assert status_of(send) == "draft"
    assert send.requires_input_id, "requires_input_id is NOT NULL for a reason"
    assert send.requires_input_id == card["id"]

    text = answer_text(payload)
    assert gr.PNR in text or gr.TICKET_NO in text, (
        f"the booking reference came out of the email by regex, not by the "
        f"model retyping it — it should be in the answer:\n{text}"
    )


async def test_cancel_flight_reads_a_turkish_booking_email(
    client, db, user, embed, llm, google
):
    """Scenario 12: the only booking in the mailbox is in Turkish.

    The vector leg scores it near zero against an English query and the
    ``english`` text-search configuration produces useless lexemes, so round 0
    of the probe finds nothing. Recovery is the escalation ladder — sender
    domain and code pattern — and it costs no extra model call.
    """
    mirror = require("app.db.repositories.mirror")
    await seed_mirror(db, user.id, embed, gcal=True, gdrive=True, gmail=False)
    turkish = gr.gmail_mirror_rows(embed, [gr.MSG_TK_BOOKING_TR, gr.MSG_TK_PROMO])
    await mirror.upsert_gmail(db, user.id, turkish)
    await db.commit()

    llm.use("cancel_turkish_flight_tr")
    payload = await post_query(client, "Cancel my Turkish Airlines flight")

    # The ladder is four SQL queries and a prompt. It costs no model call, which
    # is exactly why it can afford four rungs.
    assert_cost(payload, llm, 1)
    text = answer_text(payload)
    assert gr.PNR in text or gr.TICKET_NO in text, (
        "a PNR is a shape, and shapes survive translation — the extractors "
        f"should have found it in the Turkish body:\n{text}"
    )
    assert not google.sends


# --------------------------------------------------------------------------- #
# 5. "Prepare for tomorrow's meeting with Acme Corp" — three services
# --------------------------------------------------------------------------- #


async def test_meeting_prep_starts_two_services_then_waits_for_the_guest_list(
    client, db, mirrored, llm
):
    """The real sequential dependency in the brief.

        meeting ──> mail ──┐
                           ├──> prose
        docs ──────────────┘

    ``docs`` searches on the company alias rather than on the meeting title
    precisely so it does not have to wait. ``mail`` filters on the meeting's
    attendee list, so it cannot start until the meeting is in hand.
    """
    payload = await post_query(client, "Prepare for tomorrow's meeting with Acme Corp")

    assert payload["status"] == "complete"
    assert services_of(payload) <= {"gcal", "gmail", "gdrive", "meta"}
    assert_cost(payload, llm, 2)  # plan, then prose

    spans = await load_steps(db, mirrored.id, payload["run_id"])
    by_op = {s.op: n for n, s in spans.items()}
    calendar = by_op.get("gcal.search_events")
    drive = by_op.get("gdrive.search_files")
    mail = by_op.get("gmail.search_emails")
    assert calendar and drive and mail, f"expected all three services: {by_op}"

    assert_ran_concurrently(spans, calendar, drive)
    assert_ran_in_order(spans, calendar, mail)

    prose = llm.prose_calls()
    assert prose, "an answer_style of prose should have produced a streamed call"

    text = answer_text(payload)
    assert "Acme" in text
    assert "10:00" in text or "renewal" in text.lower(), text


# --------------------------------------------------------------------------- #
# 6. "...where john@company.com is invited?" — the generated column
# --------------------------------------------------------------------------- #


async def test_attendee_filter_hits_the_generated_column(client, mirrored, llm):
    """An `@` in the query is the router's cheap test for "needs no
    interpretation": the attendee is a literal value, so this costs nothing.

    Underneath it is ``attendee_emails @> ARRAY['john@company.com']`` against a
    generated column with a GIN index — not a ``jsonb_array_elements`` scan.
    """
    payload = await post_query(
        client, "What's on my calendar next week where john@company.com is invited?"
    )

    assert payload["status"] == "complete"
    assert_cost(payload, llm, 0)

    text = answer_text(payload)
    assert "Design review" in text, (
        f"john@company.com is a guest on exactly one event next week:\n{text}"
    )
    assert "Standup" not in text, (
        f"Standup has no John on it and must be filtered out:\n{text}"
    )
    assert "1:1 with John Okafor" not in text, (
        "john.okafor@company.com is a different address from john@company.com; "
        f"a containment filter is exact, not fuzzy:\n{text}"
    )


# --------------------------------------------------------------------------- #
# Conversation context
# --------------------------------------------------------------------------- #


async def test_a_second_query_continues_the_same_conversation(client, mirrored, llm):
    """Two turns, one thread — and the entities the first turn surfaced are on
    record for the second one to resolve against."""
    first = await post_query(client, "Find emails from sarah@company.com about the budget")
    conversation_id = first["conversation_id"]

    second = await post_query(
        client, "What's on my calendar next week?", conversation_id=conversation_id
    )
    assert second["conversation_id"] == conversation_id
    assert second["run_id"] != first["run_id"], "each turn is its own run"

    entities = first.get("entities") or []
    assert any(e.get("entity_type") == "email" for e in entities), (
        "a search that showed emails should have written them to "
        f"conversation_entities: {entities}"
    )
