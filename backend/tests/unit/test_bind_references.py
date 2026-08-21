"""Reference binding.

The planner never retypes a value. It writes `{{booking.extracted.pnr}}` and the
binder puts the real booking reference there at dispatch time. That is the whole
reason a hallucinated PNR cannot reach a draft email: the model does not have
the opportunity to type one.

Which puts the weight on this function. Everything the planner is allowed to
write has to resolve to the right thing, in the right Python type, and anything
it writes that does not exist has to blow up here — loudly, before the step
runs — rather than binding to None and producing a draft addressed to nobody
about booking `None`.

Reference forms, from `docs/contracts.md`:

    {{step.path}}                       {{step.hits[0].id}}
    {{step.hits[*].id}}                 {{search.gmail[0].extracted.pnr}}
    {{windows.<name>.start}}            time_phrase: "tomorrow"
"""

from __future__ import annotations

import pytest

from tests.conftest import FROZEN_NOW, MONDAY, TZ_KOLKATA, call, get, load_any

bind = load_any(
    "app.orchestrator.dispatch",
    ["bind", "bind_references", "bind_args", "resolve_references"],
)
resolve_time = load_any("app.orchestrator.temporal", "resolve")


class Bag(dict):
    """A dict that also answers to attribute access.

    Step results travel as JSON, and different parts of the system hold them as
    dicts or as small objects depending on where they came from. The binder has
    to cope with both, so the fixture data is both.
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - only on a genuine miss
            raise AttributeError(name) from exc


def bag(obj):
    """Deep-convert plain dicts and lists into Bags."""
    if isinstance(obj, dict):
        return Bag({k: bag(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [bag(v) for v in obj]
    return obj


NEXT_WEEK_START = "2026-08-23T18:30:00+00:00"
NEXT_WEEK_END = "2026-08-30T18:30:00+00:00"


@pytest.fixture
def scope():
    """Everything a reference is allowed to point at, part way through a run.

    Two steps have finished (`booking` and `mail`), one question has been
    answered (`disambiguate`), the probe's candidates are on `search`, and the
    pre-pass has left its resolved windows on `windows`.
    """
    steps = {
        "booking": {
            "message_id": "msg_18f2c9a1",
            "thread_id": "thr_44b1",
            "subject": "Your Turkish Airlines booking is confirmed — TK1984",
            "from_email": "noreply@turkishairlines.com",
            "extracted": {
                "pnr": "6F2QK9",
                "flight_no": "TK1",
                "support_email": "cancel@turkishairlines.com",
                "amount": "USD 812.40",
            },
        },
        "mail": {
            "count": 3,
            "hits": [
                {"id": "m_001", "subject": "Q3 budget draft", "cn": 0.88},
                {"id": "m_002", "subject": "Q3 budget — revised", "cn": 0.71},
                {"id": "m_003", "subject": "budget questions", "cn": 0.59},
            ],
        },
        "disambiguate": {"value": {"event_id": "3k9m2p_20260825T200000Z", "new_time": "Friday 3pm"}},
        "event": {"event_id": "3k9m2p_20260825T200000Z", "etag": 'W/"cWx4"', "duration_minutes": 30},
    }
    everything = {
        **steps,
        "steps": steps,
        "search": {
            "gmail": [
                {"id": "m_001", "cn": 0.88, "extracted": {"pnr": "6F2QK9", "flight_no": "TK1"}},
                {"id": "m_009", "cn": 0.61, "extracted": {}},
            ],
            "gcal": [{"id": "evt_77", "cn": 0.79}],
            "gdrive": [],
        },
        "windows": {
            "next_week": {
                "start": NEXT_WEEK_START,
                "end": NEXT_WEEK_END,
                "tz": TZ_KOLKATA,
                "interpretation": "iso week 35, half-open",
            }
        },
        "now": FROZEN_NOW,
        "tz": TZ_KOLKATA,
        "week_start": MONDAY,
    }
    return bag(everything)


def b(value, scope):
    """Bind one value in this scope, with the clock the run is using."""
    return call(bind, value, scope, now=FROZEN_NOW, tz=TZ_KOLKATA, week_start=MONDAY)


def fails(value, scope, *because: str):
    """Binding must raise, and the message must name what could not be found."""
    with pytest.raises(Exception) as caught:  # noqa: PT011 - any failure shape will do
        b(value, scope)
    said = str(caught.value).lower()
    assert any(k.lower() in said for k in because), f"{caught.value!r} does not mention {because}"
    return caught.value


# ---------------------------------------------------------------------------
# {{step.field}}
# ---------------------------------------------------------------------------


def test_a_plain_step_field(scope):
    assert b("{{booking.subject}}", scope) == (
        "Your Turkish Airlines booking is confirmed — TK1984"
    )


def test_a_nested_step_field(scope):
    assert b("{{booking.extracted.pnr}}", scope) == "6F2QK9"


def test_an_answered_question(scope):
    assert b("{{disambiguate.value.event_id}}", scope) == "3k9m2p_20260825T200000Z"


def test_a_whole_value_reference_keeps_its_type(scope):
    # `{{mail.count}}` is the integer 3, not the string "3". An op whose args
    # model says `int` would otherwise reject a perfectly good plan.
    bound = b("{{mail.count}}", scope)
    assert bound == 3
    assert isinstance(bound, int)


def test_a_reference_inside_a_sentence_is_interpolated(scope):
    # From docs/SAMPLE_QUERIES.md §4: the subject line of the cancellation draft.
    subject = b("Cancellation request — booking {{booking.extracted.pnr}}", scope)
    assert subject == "Cancellation request — booking 6F2QK9"


def test_two_references_in_one_string(scope):
    line = b("{{booking.extracted.flight_no}} / {{booking.extracted.pnr}}", scope)
    assert line == "TK1 / 6F2QK9"


def test_text_with_no_reference_is_left_alone(scope):
    assert b("Cancel my Turkish Airlines flight", scope) == "Cancel my Turkish Airlines flight"
    assert b(42, scope) == 42
    assert b(True, scope) is True
    assert b(None, scope) is None


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_an_indexed_hit(scope):
    assert b("{{mail.hits[0].id}}", scope) == "m_001"
    assert b("{{mail.hits[1].subject}}", scope) == "Q3 budget — revised"


def test_a_star_index_returns_a_real_list(scope):
    # This is the one that goes wrong quietly. A binder that formats the list
    # into a string hands `["m_001", "m_002", "m_003"]` to an op expecting
    # list[str], and the failure surfaces three layers away as a Google 400.
    bound = b("{{mail.hits[*].id}}", scope)
    assert isinstance(bound, list), f"expected a list, got {type(bound).__name__}: {bound!r}"
    assert bound == ["m_001", "m_002", "m_003"]
    assert all(isinstance(item, str) for item in bound)


def test_a_star_index_over_an_empty_list_is_an_empty_list(scope):
    # Not an error, and not None. "No Drive files matched" is a legitimate
    # result, and the step downstream should receive an empty list and say so.
    bound = b("{{search.gdrive[*].id}}", scope)
    assert bound == []
    assert isinstance(bound, list)


def test_an_index_past_the_end_raises(scope):
    fails("{{mail.hits[9].id}}", scope, "9", "index", "range", "hits", "mail")


# ---------------------------------------------------------------------------
# {{search...}} and {{windows...}}
# ---------------------------------------------------------------------------


def test_a_probe_candidate(scope):
    assert b("{{search.gmail[0].extracted.pnr}}", scope) == "6F2QK9"
    assert b("{{search.gcal[0].id}}", scope) == "evt_77"


def test_a_resolved_window(scope):
    assert b("{{windows.next_week.start}}", scope) == NEXT_WEEK_START
    assert b("{{windows.next_week.end}}", scope) == NEXT_WEEK_END


def test_a_window_referenced_from_inside_a_nested_argument(scope):
    args = {
        "window": {
            "start": "{{windows.next_week.start}}",
            "end": "{{windows.next_week.end}}",
        },
        "attendee_emails_any": ["john@company.com"],
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
    }
    bound = b(args, scope)
    assert bound["window"] == {"start": NEXT_WEEK_START, "end": NEXT_WEEK_END}
    assert bound["attendee_emails_any"] == ["john@company.com"]
    assert bound["order_by"] == "starts_at"


def test_references_inside_lists_are_bound(scope):
    bound = b({"to": ["{{booking.extracted.support_email}}"], "subject": "x"}, scope)
    assert bound["to"] == ["cancel@turkishairlines.com"]


def test_a_window_that_was_never_resolved_raises(scope):
    fails("{{windows.last_quarter.start}}", scope, "last_quarter", "window", "unknown", "not")


# ---------------------------------------------------------------------------
# time_phrase, resolved at bind time
# ---------------------------------------------------------------------------


def test_a_time_phrase_is_resolved_when_the_step_runs(scope):
    # Resolved here rather than at plan time because a paused run can sit
    # overnight. "Tomorrow" has to mean tomorrow from the moment the step runs,
    # and the run's `now` is what says which moment that is.
    bound = b({"time_phrase": "tomorrow"}, scope)

    expected = call(resolve_time, "tomorrow", TZ_KOLKATA, MONDAY, FROZEN_NOW)
    start = get(bound, "start")
    end = get(bound, "end")

    assert str(start).startswith(str(expected.start)[:19]) or start == expected.start
    assert str(end).startswith(str(expected.end)[:19]) or end == expected.end


def test_a_time_phrase_nested_in_an_argument_is_resolved(scope):
    bound = b({"window": {"time_phrase": "next week"}, "order_by": "starts_at"}, scope)
    window = bound["window"]
    expected = call(resolve_time, "next week", TZ_KOLKATA, MONDAY, FROZEN_NOW)
    assert str(get(window, "start")).startswith(str(expected.start)[:19])
    assert bound["order_by"] == "starts_at"


def test_a_time_phrase_that_means_nothing_raises(scope):
    fails({"time_phrase": "whenever suits you"}, scope, "whenever", "time", "phrase", "resolve")


# ---------------------------------------------------------------------------
# Missing references raise. They never bind to None.
# ---------------------------------------------------------------------------


def test_a_reference_to_a_step_that_never_ran_raises(scope):
    fails("{{nowhere.hits[0].id}}", scope, "nowhere", "unknown", "not found", "no such")


def test_a_reference_to_a_field_that_is_not_there_raises(scope):
    fails("{{booking.cabin_class}}", scope, "cabin_class", "booking", "unknown", "not found")


def test_a_reference_part_way_down_a_path_raises(scope):
    fails(
        "{{booking.extracted.seat.row}}",
        scope,
        "seat",
        "row",
        "extracted",
        "unknown",
        "not found",
    )


def test_a_missing_reference_inside_a_sentence_raises(scope):
    # The dangerous one: this would otherwise render as
    # "Cancellation request — booking None" and get sent.
    fails(
        "Cancellation request — booking {{booking.extracted.ticket_no}}",
        scope,
        "ticket_no",
        "unknown",
        "not found",
    )


def test_a_missing_reference_deep_in_an_argument_tree_raises(scope):
    fails(
        {"to": ["ops@x.com"], "subject": "x", "body": {"lines": ["{{mail.hits[0].snippet}}"]}},
        scope,
        "snippet",
        "unknown",
        "not found",
    )
