"""The front door, and the order its matchers run in.

Matchers, in this order and no other:

1. an answer to a card that is open on screen;
2. a UI verb — retry, edit, show more, undo;
3. chit-chat, and only when no card is waiting;
4. a rule router of literal patterns.

The order is the whole design. "ok" is gratitude when nothing is pending and an
approval when a confirm card is waiting, and the only thing that tells them
apart is which matcher gets to look first. Put chit-chat ahead of the card
answer and the system cheerfully replies "anytime" to somebody trying to send an
email — a failure that looks like success, which is the worst kind.

The matchers are exact rather than eager, for the same reason. A chit-chat
matcher that fires on a prefix silently drops the real request appended to a
thank-you. A rule router that fires on a prefix answers half a question. Both
have to consume the whole message or decline. Everything here costs zero LLM
calls, which is only worth having if it is also never wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.conftest import FROZEN_NOW, MONDAY, TZ_NY, call, get, load_any

_FD = "app.orchestrator.front_door"

decide = load_any(_FD, ["decide", "run", "front_door", "handle"])
parse_answer = load_any(_FD, ["parse_answer", "match_answer", "interpret_answer"])
match_ui_verb = load_any(_FD, ["match_ui_verb", "ui_verb", "match_verb"])
match_chit_chat = load_any(_FD, ["match_chit_chat", "match_chitchat", "chit_chat"])
match_rule_router = load_any(
    ["app.orchestrator.front_door", "app.orchestrator.route"],
    ["match_rule_router", "rule_router", "match_rule", "router"],
)


# ---------------------------------------------------------------------------
# Reading what the front door decided
# ---------------------------------------------------------------------------


def kind_of(result) -> str:
    """A lower-case word for the route taken."""
    if result is None:
        return "none"
    if isinstance(result, tuple) and result:
        return str(result[0]).lower()
    for name in ("route", "kind", "type", "matcher", "source", "outcome"):
        found = get(result, name, None)
        if found is not None:
            return str(get(found, "value", found)).lower()
    return str(result).lower()


def value_of(result):
    """The answer value, when the message answered a card."""
    if result is None:
        return None
    answer = get(result, "answer", None)
    if answer is not None:
        return get(answer, "value", answer)
    direct = get(result, "value", None)
    if direct is not None:
        return direct
    data = get(result, "data", None)
    if data is not None:
        return get(data, "value", None)
    return None


def is_chit_chat(result) -> bool:
    return any(word in kind_of(result) for word in ("chit", "smalltalk", "social"))


def is_card_answer(result) -> bool:
    return any(word in kind_of(result) for word in ("card", "answer", "input", "prompt"))


def front(message: str, cards=(), last_intent=None):
    """Run the whole door on one message, with whatever is on screen."""
    return call(
        decide,
        message,
        tz=TZ_NY,
        week_start=MONDAY,
        now=FROZEN_NOW,
        open_prompts=list(cards),
        last_intent=last_intent,
    )


def answer(message: str, card: dict):
    """What one card makes of a message, or None."""
    result = call(parse_answer, message, card, tz=TZ_NY, week_start=MONDAY, now=FROZEN_NOW)
    return None if result is None else get(result, "value", result)


def routed(message: str, last_intent=None):
    return call(
        match_rule_router,
        message,
        tz=TZ_NY,
        week_start=MONDAY,
        now=FROZEN_NOW,
        last_intent=last_intent,
    )


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def confirm_card(question: str = "Send this cancellation email?") -> dict:
    return {
        "id": "pin_9Fd4RbXn2QsLt6WkJ",
        "kind": "confirm",
        "blocking": False,
        "status": "pending",
        "op": "gmail.send_email",
        "prompt": {"question": question},
        "question": question,
        "value_schema": {"type": "boolean"},
        "options": None,
    }


JOHN_OPTIONS = [
    {
        "id": "3k9m2p_20260825T200000Z",
        "label": "1:1 with John Okafor",
        "meta": {"when": "Tue Aug 25, 4:00 PM"},
    },
    {
        "id": "7t4v8q_20260826T130000Z",
        "label": "Vendor sync — John Reyes (Northwind)",
        "meta": {"when": "Wed Aug 26, 9:00 AM"},
    },
]


def choice_card(options=None) -> dict:
    options = options or JOHN_OPTIONS
    return {
        "id": "pin_2Wq7ZbKm4XvNr8TsL",
        "kind": "choice",
        "blocking": True,
        "status": "pending",
        "op": "ask.user",
        "prompt": {"question": "Which meeting?"},
        "question": "Which meeting?",
        "options": options,
        "value_schema": {"type": "string", "enum": [o["id"] for o in options]},
    }


FILE_OPTIONS = [
    {"id": "f_agenda", "label": "Acme Q3 agenda.docx"},
    {"id": "f_pricing", "label": "Acme pricing sheet.xlsx"},
    {"id": "f_notes", "label": "Kickoff notes.pdf"},
]


def multi_choice_card() -> dict:
    return {
        "id": "pin_5Rt8YcLn3MwPq7VjK",
        "kind": "multi_choice",
        "blocking": True,
        "status": "pending",
        "op": "ask.user",
        "prompt": {"question": "Which files should I attach?"},
        "question": "Which files should I attach?",
        "options": FILE_OPTIONS,
        "value_schema": {
            "type": "array",
            "items": {"type": "string", "enum": [o["id"] for o in FILE_OPTIONS]},
        },
    }


def text_card() -> dict:
    return {
        "id": "pin_6Yu9ZdMo4NxQr8WkL",
        "kind": "text",
        "blocking": True,
        "status": "pending",
        "op": "ask.user",
        "prompt": {"question": "What time should it move to?"},
        "question": "What time should it move to?",
        "options": None,
        "value_schema": {"type": "string", "minLength": 3},
    }


# ---------------------------------------------------------------------------
# The order. This is the point of the file.
# ---------------------------------------------------------------------------


def test_ok_with_a_confirm_card_open_approves_it():
    # If chit-chat looked first, this would come back "anytime" and the email
    # would never be sent. The user would have no way to tell.
    result = front("ok", cards=[confirm_card()])

    assert not is_chit_chat(result), f"chit-chat ate an approval: {result!r}"
    assert is_card_answer(result), result
    assert value_of(result) is True


def test_ok_with_nothing_open_is_chit_chat_and_not_an_approval():
    result = front("ok")
    assert is_chit_chat(result), result
    assert value_of(result) is not True


def test_cancel_that_with_a_card_open_declines_rather_than_starting_a_turn():
    # "cancel that" is also a UI verb. With a confirm card on screen it means
    # *do not send it*, not *begin a new conversation about cancelling*.
    result = front("cancel that", cards=[confirm_card()])
    assert is_card_answer(result), result
    assert value_of(result) is False


def test_thanks_with_a_card_open_is_not_an_approval():
    # The matchers are exact in both directions. "That's perfect" is warmth, not
    # consent, and reading it as consent sends an email nobody agreed to.
    result = front("thanks, that's perfect", cards=[confirm_card()])
    assert value_of(result) is not True, f"gratitude read as approval: {result!r}"
    assert not is_card_answer(result), result


def test_thanks_with_nothing_open_is_chit_chat():
    result = front("thanks, that's perfect")
    assert is_chit_chat(result), result
    assert str(get(result, "text", "")).strip(), "chit-chat with no reply text"


def test_chit_chat_is_skipped_entirely_while_a_card_waits():
    # Answering "anytime" to somebody looking at an open question is worse than
    # saying nothing. With a card up, an unmatched message goes to the pipeline.
    result = front("nice", cards=[choice_card()])
    assert not is_chit_chat(result), result


def test_a_request_attached_to_a_thank_you_is_not_chit_chat():
    # docs/SAMPLE_QUERIES.md §15: "thanks for finding that, can you send it?"
    # is not gratitude, it is an instruction with gratitude attached.
    result = front("thanks for finding that, can you send it?", cards=[confirm_card()])
    assert not is_chit_chat(result), result


def test_a_topic_change_while_a_card_is_open_is_not_mangled_into_an_answer():
    # A choice card about two Johns is on screen and the user asks about next
    # week instead. Forcing that into the card would pick a meeting at random
    # and then move it.
    result = front("actually, what's on my calendar next week?", cards=[choice_card()])

    assert value_of(result) is None, f"a new question was read as a card answer: {result!r}"
    assert not is_card_answer(result), result
    assert not is_chit_chat(result), result


def test_a_bare_time_phrase_while_a_card_is_open_does_not_pick_an_option():
    # The open card asks which meeting. "next Tuesday" answers a question that
    # was not asked.
    result = front("next Tuesday", cards=[choice_card()])
    assert value_of(result) not in {o["id"] for o in JOHN_OPTIONS}


def test_a_hard_query_falls_through_to_the_model():
    # The door's other job is knowing when to get out of the way. Every one of
    # these needs ranking, resolution or a write.
    for message in (
        "Move the meeting with John",
        "Cancel my Turkish Airlines flight",
        "Prepare for tomorrow's meeting with Acme Corp",
    ):
        result = front(message)
        assert get(result, "handled", False) is False, (message, result)


# ---------------------------------------------------------------------------
# Answer grammar — confirm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["ok", "OK", "yes", "yep", "yeah", "sure", "do it", "send it", "go ahead", "please do", "y"],
)
def test_confirm_yes(message):
    assert answer(message, confirm_card()) is True


@pytest.mark.parametrize(
    "message", ["no", "nope", "not now", "cancel", "don't send", "n", "never mind", "hold off"]
)
def test_confirm_no(message):
    assert answer(message, confirm_card()) is False


@pytest.mark.parametrize(
    "message",
    [
        "what's on my calendar tomorrow?",
        "who is it going to?",
        "change the subject line first",
        "ok but change the subject line first",
        "yes, but send it tomorrow",
    ],
)
def test_confirm_declines_anything_that_is_not_a_whole_answer(message):
    # The last two are the trap: both begin with a yes and neither is one.
    # Half-understanding "yes, but send it tomorrow" and executing the yes is
    # the failure this whole design exists to prevent.
    assert answer(message, confirm_card()) is None


def test_confirm_shapes_itself_to_the_cards_schema():
    # Some confirm cards want an object rather than a bare boolean, and the
    # card's own `value_schema` is the authority on which.
    card = confirm_card()
    card["value_schema"] = {
        "type": "object",
        "properties": {"approved": {"type": "boolean"}},
        "required": ["approved"],
    }
    assert answer("send it", card) == {"approved": True}
    assert answer("not now", card) == {"approved": False}


def test_a_card_that_is_no_longer_pending_takes_no_answer():
    # Reopening an old chat shows the card in its answered state. Typing "yes"
    # underneath it must not send the email a second time.
    card = confirm_card()
    card["status"] = "answered"
    assert answer("yes", card) is None


# ---------------------------------------------------------------------------
# Answer grammar — choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "index"),
    [
        ("1", 0),
        ("2", 1),
        ("#2", 1),
        ("first", 0),
        ("the first one", 0),
        ("second", 1),
        ("the second one", 1),
        ("option 2", 1),
        ("last", 1),
    ],
)
def test_choice_by_ordinal(message, index):
    assert answer(message, choice_card()) == JOHN_OPTIONS[index]["id"]


@pytest.mark.parametrize(
    ("message", "index"),
    [
        ("okafor", 0),
        ("Okafor", 0),
        ("1:1 with john", 0),
        ("Northwind", 1),
        ("vendor sync", 1),
        ("reyes", 1),
    ],
)
def test_choice_by_a_substring_that_matches_one_label(message, index):
    assert answer(message, choice_card()) == JOHN_OPTIONS[index]["id"]


@pytest.mark.parametrize("message", ["john", "John", "the meeting with john", "meeting"])
def test_choice_declines_a_substring_that_matches_more_than_one(message):
    # "John" is in both labels. Picking the first one and moving it is worse
    # than asking again — the whole card exists because we could not tell.
    assert answer(message, choice_card()) is None


@pytest.mark.parametrize("message", ["3", "0", "fourth", "the purple one", ""])
def test_choice_declines_an_ordinal_that_is_not_there(message):
    assert answer(message, choice_card()) is None


def test_choice_accepts_the_option_id_itself():
    # This is what the button in the UI posts.
    assert answer("3k9m2p_20260825T200000Z", choice_card()) == "3k9m2p_20260825T200000Z"


def test_choice_accepts_the_whole_label():
    assert answer("1:1 with John Okafor", choice_card()) == JOHN_OPTIONS[0]["id"]


# ---------------------------------------------------------------------------
# Answer grammar — multi_choice
# ---------------------------------------------------------------------------


def test_multi_choice_by_ordinals():
    assert answer("1 and 3", multi_choice_card()) == ["f_agenda", "f_notes"]


def test_multi_choice_with_commas():
    assert answer("1, 2", multi_choice_card()) == ["f_agenda", "f_pricing"]


def test_multi_choice_all():
    assert answer("all of them", multi_choice_card()) == ["f_agenda", "f_pricing", "f_notes"]


def test_multi_choice_none():
    # An empty list is a real answer: attach nothing. It is not the same as
    # declining to answer, and the two must not collapse into each other.
    assert answer("none", multi_choice_card()) == []


def test_multi_choice_refuses_a_partly_readable_list():
    # "1 and the other one" is one clear pick and one guess. Taking the clear
    # half and dropping the rest attaches the wrong files.
    assert answer("1 and the other one", multi_choice_card()) is None


def test_multi_choice_declines_a_sentence():
    assert answer("whichever ones you think", multi_choice_card()) is None


# ---------------------------------------------------------------------------
# Answer grammar — text
# ---------------------------------------------------------------------------


def test_text_takes_the_whole_message():
    assert answer("Friday 3pm", text_card()) == "Friday 3pm"


def test_text_declines_something_shorter_than_the_schema_allows():
    # `minLength: 3`. The card is re-raised with help text rather than storing
    # a value that will fail later.
    assert answer("ok", text_card()) is None


def test_text_declines_a_question():
    # A reply that asks its own question is a new turn, not an answer.
    assert answer("what times are free?", text_card()) is None


# ---------------------------------------------------------------------------
# UI verbs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "verb"),
    [
        ("retry", "retry"),
        ("try again", "retry"),
        ("Retry calendar", "retry"),
        ("edit", "edit"),
        ("edit that", "edit"),
        ("show more", "more"),
        ("cancel that", "cancel"),
        ("undo", "undo"),
        ("open that", "open"),
        ("sync now", "sync"),
        ("reconnect", "reconnect"),
    ],
)
def test_ui_verbs_are_recognised(message, verb):
    matched = call(match_ui_verb, message)
    assert matched is not None, message
    assert verb in str(get(matched, "verb", matched)).lower(), (message, matched)


def test_retrying_one_service_names_it():
    # docs/SAMPLE_QUERIES.md §11: pressing "Retry Calendar" re-runs the failed
    # node. It does not re-plan, and it does not re-run Gmail.
    matched = call(match_ui_verb, "retry calendar")
    assert get(matched, "target", None) == "gcal"


@pytest.mark.parametrize(
    "message",
    [
        "what's on my calendar next week?",
        "cancel my Turkish Airlines flight",
        "send the proposal to Sarah and tell her it is late",
        "retry the Acme booking search and then draft a reply",
    ],
)
def test_a_sentence_is_not_a_ui_verb(message):
    # "Cancel my Turkish Airlines flight" starts with a word the UI also uses.
    # Treating it as a button press would cancel the last pending action
    # instead — the wrong thing, silently.
    assert call(match_ui_verb, message) is None


# ---------------------------------------------------------------------------
# Chit-chat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message", ["thanks", "thanks!", "thank you", "thanks, that's perfect", "perfect"]
)
def test_chit_chat_matches_a_whole_message(message):
    result = call(match_chit_chat, message, tz=TZ_NY, week_start=MONDAY, now=FROZEN_NOW)
    assert result is not None, message
    assert str(get(result, "text", "")).strip(), message


@pytest.mark.parametrize(
    "message",
    [
        "thanks — also what's on Friday?",
        "thanks for finding that, can you send it?",
        "perfect, do the same for Northwind",
        "thanks, now cancel my Turkish Airlines flight",
        "thanks! and email bob@x.com to say I am running late",
    ],
)
def test_chit_chat_declines_anything_with_a_request_attached(message):
    # Straight from docs/SAMPLE_QUERIES.md §15. Every one of these begins with
    # gratitude and ends with work. An eager matcher here drops the work.
    assert call(match_chit_chat, message, tz=TZ_NY, week_start=MONDAY, now=FROZEN_NOW) is None


def test_the_chit_chat_reply_is_stable_for_the_same_message():
    # Varied by hash of the message, not at random, so the same thank-you does
    # not get a different answer on a retry.
    first = call(match_chit_chat, "thanks", tz=TZ_NY, week_start=MONDAY, now=FROZEN_NOW)
    second = call(match_chit_chat, "thanks", tz=TZ_NY, week_start=MONDAY, now=FROZEN_NOW)
    assert get(first, "text", None) == get(second, "text", None)


# ---------------------------------------------------------------------------
# The rule router: it fires only when it consumes the whole message
# ---------------------------------------------------------------------------


def intent_name(result) -> str:
    intent = get(result, "intent", None) or result
    return str(get(intent, "name", "")).lower()


def test_the_calendar_window_pattern_fires():
    result = routed("What's on my calendar next week?")
    assert result is not None
    assert intent_name(result) == "calendar_list"


def test_a_router_hit_carries_a_resolved_window():
    # The capture group goes straight to temporal.resolve(). Week 34 plus one is
    # week 35: Mon 24 Aug 00:00 EDT, which is 04:00Z.
    result = routed("What's on my calendar next week?")
    windows = get(result, "windows", {})
    assert windows, result
    window = next(iter(windows.values()))
    assert window.start.astimezone(UTC) == datetime(2026, 8, 24, 4, 0, tzinfo=UTC)
    assert window.end.astimezone(UTC) == datetime(2026, 8, 31, 4, 0, tzinfo=UTC)


def test_a_router_hit_carries_a_whole_plan():
    # A hit skips the planner, so the plan has to be complete here or the run
    # has nothing to dispatch.
    result = routed("What's on my calendar next week?")
    plan = get(result, "plan", None)
    assert plan and plan["steps"], result
    assert plan["steps"][0]["op"] == "gcal.search_events"
    assert plan["intent"]["has_write"] is False


def test_the_attendee_pattern_fires_on_a_literal_address():
    # The `@` is the trigger: an email address needs no resolution, so nothing
    # has to be ranked and the router can take the query at zero cost.
    result = routed("What's on my calendar next week where john@company.com is invited?")
    assert result is not None
    assert "john@company.com" in str(get(result, "intent", result))


def test_the_same_query_with_a_bare_name_does_not_fire():
    # "John" might be two people. That is a ranking problem, and the router
    # does not rank.
    assert routed("What's on my calendar next week where John is invited?") is None


@pytest.mark.parametrize(
    "message",
    [
        # Each of these starts with a phrase the router knows and then keeps
        # going. Matching the prefix would answer half the question and drop
        # the rest without telling anyone.
        "What's on my calendar next week and also draft an email to Sarah about it",
        "What's on my calendar next week? Move anything that clashes with the Acme review",
        "What's on my calendar next week where my out-of-office doc says I am away",
    ],
)
def test_the_router_declines_when_it_cannot_consume_the_whole_message(message):
    assert routed(message) is None, "the router swallowed a compound request"


@pytest.mark.parametrize(
    "message",
    [
        "Move the meeting with John",
        "Cancel my Turkish Airlines flight",
        "That email about the proposal",
        "Prepare for tomorrow's meeting with Acme Corp",
        "Find events next week that conflict with my out-of-office doc",
    ],
)
def test_the_router_declines_anything_that_needs_judgement(message):
    # Ranking, resolution or a write. The router's job is to be certain or to
    # get out of the way.
    assert routed(message) is None


def test_a_window_that_does_not_resolve_is_not_a_hit():
    # A calendar query with no readable window is not something to guess at.
    assert routed("What's on my calendar when the Acme contract lands?") is None


def test_a_bare_time_phrase_carries_the_last_intent():
    # docs/SAMPLE_QUERIES.md §9. "Next Tuesday" on its own is not a query —
    # unless the last turn was a calendar list, in which case it is the same
    # question with a new window, at zero LLM calls.
    carried = routed("Next Tuesday", last_intent={"name": "calendar_list", "services": ["gcal"]})
    assert carried is not None
    windows = get(carried, "windows", {})
    window = next(iter(windows.values()))
    assert window.start.astimezone(UTC) == datetime(2026, 8, 25, 4, 0, tzinfo=UTC)


def test_a_bare_time_phrase_with_no_prior_calendar_turn_declines():
    assert routed("Next Tuesday", last_intent=None) is None
