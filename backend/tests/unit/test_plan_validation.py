"""Plan validation.

One LLM call produces a plan. Everything after that point is pure Python, and
this is the gate it goes through first. The validator's job is to turn a
plausible-looking blob of JSON into something that is safe to execute, or to
refuse it — cheaply, before any Google call has been made and before any write
has been prepared.

Both halves matter. A validator that rejects too much makes the system look
broken; a validator that lets a write through on an intent that said it was a
read makes it dangerous. So this file covers every rejection rule and the
accepting cases that prove none of them fires on a good plan.

The rules, as `docs/contracts.md` and `docs/SAMPLE_QUERIES.md` §5 state them:

* the op must exist in the registry;
* the args must satisfy the op's own model;
* a `{{step.field}}` reference must point at a step this one depends on,
  transitively — not merely at a step that happens to exist;
* the field must be one the op declares, after the singular/plural correction;
* no cycles;
* at most twelve steps;
* a `needs_confirm` step must have something to confirm;
* no write when the intent said it was a read;
* no service the intent did not name;
* a speculative subtree must be local and read-only.
"""

from __future__ import annotations

from tests.conftest import accepted, load_any, make_plan, make_step, rejected

validate_plan = load_any(
    "app.orchestrator.validate", ["validate_plan", "validate", "check_plan"]
)


def reject(op_registry, plan, because, **extra):
    """Assert the plan is refused, and that the message names the rule that fired."""
    return rejected(validate_plan, plan, because=because, registry=op_registry, **extra)


def accept(op_registry, plan, **extra):
    return accepted(validate_plan, plan, registry=op_registry, **extra)


# ---------------------------------------------------------------------------
# The accepting cases
# ---------------------------------------------------------------------------


def test_a_plain_single_step_read_is_accepted(op_registry, read_plan):
    accept(op_registry, read_plan)


def test_a_parallel_dag_is_accepted(op_registry):
    # "Prepare for tomorrow's meeting with Acme Corp": one calendar lookup, then
    # mail and files fan out from it and never touch each other.
    plan = make_plan(
        [
            make_step(
                "meeting",
                "gcal.search_events",
                {"window": {"start": "{{windows.tomorrow.start}}", "end": "{{windows.tomorrow.end}}"}},
                expect="one",
                freshness="live",
            ),
            make_step(
                "mail",
                "gmail.search_emails",
                {"query": "Acme Corp", "limit": 10},
                depends_on=["meeting"],
            ),
            make_step(
                "docs",
                "gdrive.search_files",
                {"query": "Acme Corp"},
                depends_on=["meeting"],
                optional=True,
            ),
        ],
        name="meeting_prep",
        services=["gcal", "gmail", "gdrive"],
        answer_style="prose",
    )
    accept(op_registry, plan)


def test_a_write_behind_an_ask_is_accepted(op_registry):
    # docs/SAMPLE_QUERIES.md §7. Confidence 0.62 is below the 0.75 line, and the
    # plan is still fine because a blocking question stands in front of the write.
    plan = make_plan(
        [
            make_step(
                "disambiguate",
                "ask.user",
                {
                    "kind": "form",
                    "question": "Which meeting, and when should it move to?",
                    "value_schema": {
                        "type": "object",
                        "properties": {"event_id": {"type": "string"}},
                        "required": ["event_id"],
                    },
                },
                expect="one",
            ),
            make_step(
                "event",
                "gcal.get_event",
                {"event_id": "{{disambiguate.value.event_id}}"},
                depends_on=["disambiguate"],
                expect="one",
                freshness="live",
            ),
            make_step(
                "move",
                "gcal.update_event",
                {
                    "event_id": "{{event.event_id}}",
                    "etag": "{{event.etag}}",
                    "starts_at": "{{disambiguate.value.new_time}}",
                },
                depends_on=["event"],
                expect="one",
                freshness="live",
            ),
        ],
        name="move_event",
        services=["gcal"],
        has_write=True,
        confidence=0.62,
        answer_style="card",
    )
    accept(op_registry, plan)


def test_exactly_twelve_steps_is_accepted(op_registry):
    steps = [
        make_step(f"s{i}", "gmail.search_emails", {"query": f"topic {i}"}) for i in range(12)
    ]
    accept(op_registry, make_plan(steps, services=["gmail"]))


def test_a_reference_to_a_transitive_dependency_is_accepted(op_registry):
    # b depends on a, c depends on b. c may reference a: it is upstream, even
    # though it is not a direct edge.
    plan = make_plan(
        [
            make_step("a", "gmail.search_emails", {"query": "booking"}, expect="one"),
            make_step(
                "b",
                "gmail.get_email",
                {"message_id": "{{a.hits[0].id}}"},
                depends_on=["a"],
                expect="one",
            ),
            make_step(
                "c",
                "gmail.get_email",
                {"message_id": "{{a.hits[1].id}}"},
                depends_on=["b"],
                expect="one",
            ),
        ],
        services=["gmail"],
    )
    accept(op_registry, plan)


# ---------------------------------------------------------------------------
# Unknown op
# ---------------------------------------------------------------------------


def test_an_op_that_does_not_exist_is_rejected(op_registry):
    plan = make_plan(
        [make_step("guess", "gmail.telepathy", {"query": "what I meant"})],
        services=["gmail"],
    )
    reject(op_registry, plan, ("unknown", "no such", "not registered", "registry", "telepathy"))


def test_a_plausible_typo_is_still_an_unknown_op(op_registry):
    # `gmail.search_email` is one character from a real op. It is still not one.
    plan = make_plan(
        [make_step("mail", "gmail.search_email", {"query": "budget"})],
        services=["gmail"],
    )
    reject(op_registry, plan, ("unknown", "no such", "not registered", "registry", "search_email"))


# ---------------------------------------------------------------------------
# Args against the op's own model
# ---------------------------------------------------------------------------


def test_a_missing_required_argument_is_rejected(op_registry):
    plan = make_plan([make_step("mail", "gmail.search_emails", {"limit": 5})], services=["gmail"])
    reject(op_registry, plan, ("query", "required", "missing", "args", "argument", "schema"))


def test_an_argument_of_the_wrong_type_is_rejected(op_registry):
    plan = make_plan(
        [make_step("mail", "gmail.search_emails", {"query": "budget", "limit": "ten"})],
        services=["gmail"],
    )
    reject(op_registry, plan, ("limit", "integer", "int", "type", "args", "argument", "schema"))


def test_an_empty_recipient_list_is_rejected(op_registry):
    # The model says at least one recipient. A draft addressed to nobody is a
    # bug that would otherwise surface as a confusing Google 400.
    plan = make_plan(
        [make_step("send", "gmail.draft_email", {"to": [], "subject": "Hi", "body": "..."})],
        services=["gmail"],
        has_write=True,
    )
    reject(op_registry, plan, ("to", "recipient", "length", "empty", "args", "argument", "schema"))


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_a_reference_to_a_step_that_is_not_upstream_is_rejected(op_registry):
    # `booking` exists, and `draft` does not depend on it. At dispatch time the
    # two would run at the same moment and the reference would bind to nothing.
    plan = make_plan(
        [
            make_step("booking", "gmail.search_emails", {"query": "Turkish Airlines"}, expect="one"),
            make_step(
                "draft",
                "gmail.draft_email",
                {
                    "to": ["cancel@turkishairlines.com"],
                    "subject": "Cancellation",
                    "body": "Booking {{booking.hits[0].id}}",
                },
                depends_on=[],
            ),
        ],
        services=["gmail"],
        has_write=True,
    )
    reject(op_registry, plan, ("upstream", "depends", "dependency", "not a dependency", "booking"))


def test_a_reference_to_a_step_that_does_not_exist_is_rejected(op_registry):
    plan = make_plan(
        [
            make_step(
                "draft",
                "gmail.get_email",
                {"message_id": "{{nowhere.hits[0].id}}"},
                expect="one",
            )
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("nowhere", "unknown", "no such", "reference", "step"))


def test_a_field_the_op_does_not_declare_is_rejected(op_registry):
    # `gmail.search_emails` declares hits and count. It does not declare
    # `messages`, and no amount of pluralisation makes it.
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "budget"}, expect="one"),
            make_step(
                "one",
                "gmail.get_email",
                {"message_id": "{{mail.messages[0].id}}"},
                depends_on=["mail"],
                expect="one",
            ),
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("messages", "field", "output", "declare", "unknown"))


def test_the_singular_of_a_declared_plural_field_is_corrected(op_registry):
    # The planner writes `{{mail.hit[0].id}}` often enough that rejecting it
    # would cost a replan for nothing. The op declares `hits`; this is the same
    # field and is accepted. That correction has to happen before the unknown
    # field check, or the good case never gets there.
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "budget"}, expect="one"),
            make_step(
                "one",
                "gmail.get_email",
                {"message_id": "{{mail.hit[0].id}}"},
                depends_on=["mail"],
                expect="one",
            ),
        ],
        services=["gmail"],
    )
    accept(op_registry, plan)


def test_a_search_and_windows_reference_needs_no_dependency(op_registry):
    # `search` and `windows` come from the probe and the pre-pass. They exist
    # before any step runs, so they are never an upstream violation.
    plan = make_plan(
        [
            make_step(
                "events",
                "gcal.search_events",
                {
                    "window": {
                        "start": "{{windows.next_week.start}}",
                        "end": "{{windows.next_week.end}}",
                    }
                },
            ),
            make_step(
                "booking",
                "gmail.get_email",
                {"message_id": "{{search.gmail[0].id}}"},
                expect="one",
            ),
        ],
        services=["gcal", "gmail"],
    )
    accept(op_registry, plan)


# ---------------------------------------------------------------------------
# Shape of the graph
# ---------------------------------------------------------------------------


def test_a_two_step_cycle_is_rejected(op_registry):
    plan = make_plan(
        [
            make_step("a", "gmail.search_emails", {"query": "one"}, depends_on=["b"]),
            make_step("b", "gmail.search_emails", {"query": "two"}, depends_on=["a"]),
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("cycle", "circular", "acyclic", "loop"))


def test_a_longer_cycle_is_rejected(op_registry):
    plan = make_plan(
        [
            make_step("a", "gmail.search_emails", {"query": "one"}, depends_on=["c"]),
            make_step("b", "gmail.search_emails", {"query": "two"}, depends_on=["a"]),
            make_step("c", "gmail.search_emails", {"query": "three"}, depends_on=["b"]),
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("cycle", "circular", "acyclic", "loop"))


def test_a_step_that_depends_on_itself_is_rejected(op_registry):
    plan = make_plan(
        [make_step("a", "gmail.search_emails", {"query": "one"}, depends_on=["a"])],
        services=["gmail"],
    )
    reject(op_registry, plan, ("cycle", "circular", "acyclic", "loop", "itself", "self"))


def test_a_dependency_on_a_step_that_does_not_exist_is_rejected(op_registry):
    plan = make_plan(
        [make_step("a", "gmail.search_emails", {"query": "one"}, depends_on=["ghost"])],
        services=["gmail"],
    )
    reject(op_registry, plan, ("ghost", "unknown", "no such", "depends", "step"))


def test_two_steps_with_the_same_id_are_rejected(op_registry):
    # `node_executions` is unique on (run_id, node_id, round), so a duplicate id
    # would collide in the database rather than merely confuse the binder.
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "one"}),
            make_step("mail", "gmail.search_emails", {"query": "two"}),
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("duplicate", "unique", "twice", "same id", "mail"))


def test_thirteen_steps_is_rejected(op_registry):
    steps = [
        make_step(f"s{i}", "gmail.search_emails", {"query": f"topic {i}"}) for i in range(13)
    ]
    reject(
        op_registry,
        make_plan(steps, services=["gmail"]),
        ("steps", "too many", "12", "twelve", "max", "limit"),
    )


def test_a_plan_with_no_steps_at_all_is_rejected(op_registry):
    # An empty plan is not an answer. The planner should have returned
    # {"type": "answer", ...} instead, and silently completing with nothing
    # would look to the user like the system did the work.
    reject(op_registry, make_plan([], services=["gmail"]), ("steps", "empty", "no step", "least"))


# ---------------------------------------------------------------------------
# Writes and confirmation
# ---------------------------------------------------------------------------


def test_a_needs_confirm_flag_on_a_read_step_is_rejected(op_registry):
    # There is nothing to confirm. `actions.requires_input_id` is NOT NULL, so
    # this would either create a prompt gating no action, or an action row with
    # no side effect behind it. Both are nonsense; neither should reach the
    # database.
    plan = make_plan(
        [make_step("mail", "gmail.search_emails", {"query": "budget"}, needs_confirm=True)],
        services=["gmail"],
    )
    reject(op_registry, plan, ("confirm", "action", "write", "side effect", "nothing"))


def test_a_write_step_when_the_intent_says_read_only_is_rejected(op_registry):
    # The intent is what the user is shown and what `runs.intent` records. A
    # plan that writes under a read intent is either a planner mistake or a
    # prompt injection, and the two are indistinguishable from here.
    plan = make_plan(
        [
            make_step(
                "send",
                "gmail.send_email",
                {"to": ["bob@x.com"], "subject": "Running late", "body": "Sorry!"},
            )
        ],
        services=["gmail"],
        has_write=False,
    )
    reject(op_registry, plan, ("write", "has_write", "read"))


def test_a_write_with_low_confidence_and_no_question_is_rejected(op_registry):
    # docs/SAMPLE_QUERIES.md §7: below 0.75 a write needs an `ask.user` in front
    # of it. Here there is none.
    plan = make_plan(
        [
            make_step(
                "move",
                "gcal.update_event",
                {"event_id": "evt_1", "etag": "W/x", "starts_at": "2026-08-28T19:00:00Z"},
                expect="one",
            )
        ],
        services=["gcal"],
        has_write=True,
        confidence=0.62,
    )
    reject(op_registry, plan, ("confidence", "ask", "confirm", "0.75", "question", "low"))


def test_the_same_write_with_high_confidence_is_accepted(op_registry):
    plan = make_plan(
        [
            make_step(
                "move",
                "gcal.update_event",
                {"event_id": "evt_1", "etag": "W/x", "starts_at": "2026-08-28T19:00:00Z"},
                expect="one",
            )
        ],
        services=["gcal"],
        has_write=True,
        confidence=0.93,
    )
    accept(op_registry, plan)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


def test_a_service_the_intent_did_not_name_is_rejected(op_registry):
    # The intent said Calendar only. A Drive step here means the plan and the
    # intent disagree, and the intent is the thing the user was shown.
    plan = make_plan(
        [
            make_step(
                "events",
                "gcal.search_events",
                {"window": {"start": "{{windows.next_week.start}}", "end": "{{windows.next_week.end}}"}},
            ),
            make_step("docs", "gdrive.search_files", {"query": "out of office"}),
        ],
        services=["gcal"],
    )
    reject(op_registry, plan, ("service", "gdrive", "drive", "intent"))


def test_the_local_services_are_always_allowed(op_registry):
    # `ask.user` and `meta.*` are ours, not Google's. Requiring the intent to
    # list them would mean every ambiguity card changed the intent's services.
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "Acme"}),
            make_step("digest", "meta.summarize", {"items": []}, depends_on=["mail"]),
        ],
        services=["gmail"],
    )
    accept(op_registry, plan)


# ---------------------------------------------------------------------------
# Speculation
# ---------------------------------------------------------------------------
#
# A speculative step runs before we know it is needed. That is only safe when
# nothing it does can be seen from outside: no Google call, no write, no quota
# spent, nothing to undo. Our own mirror is fair game.


def test_speculating_on_a_local_read_is_accepted(op_registry):
    plan = make_plan(
        [make_step("mail", "gmail.search_emails", {"query": "Acme"}, speculate=True)],
        services=["gmail"],
    )
    accept(op_registry, plan)


def test_speculating_on_a_google_call_is_rejected(op_registry):
    # `gmail.get_email` goes to Google. Running it on a guess spends quota on
    # work that may be thrown away, and the quota is shared across the project.
    plan = make_plan(
        [make_step("one", "gmail.get_email", {"message_id": "m1"}, speculate=True, expect="one")],
        services=["gmail"],
    )
    reject(op_registry, plan, ("speculat", "local", "google", "remote", "mirror"))


def test_speculating_on_a_write_is_rejected(op_registry):
    plan = make_plan(
        [
            make_step(
                "send",
                "gmail.send_email",
                {"to": ["bob@x.com"], "subject": "Hi", "body": "..."},
                speculate=True,
            )
        ],
        services=["gmail"],
        has_write=True,
    )
    reject(op_registry, plan, ("speculat", "write", "side effect", "irreversible"))


def test_speculating_on_a_subtree_that_reaches_a_google_call_is_rejected(op_registry):
    # The root is local and harmless. What hangs off it is not, and the subtree
    # is what gets run.
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "Acme"}, speculate=True),
            make_step(
                "one",
                "gmail.get_email",
                {"message_id": "{{mail.hits[0].id}}"},
                depends_on=["mail"],
                speculate=True,
                expect="one",
            ),
        ],
        services=["gmail"],
    )
    reject(op_registry, plan, ("speculat", "local", "google", "remote", "subtree", "mirror"))


def test_speculating_on_a_subtree_that_reaches_a_write_is_rejected(op_registry):
    plan = make_plan(
        [
            make_step("mail", "gmail.search_emails", {"query": "Turkish Airlines"}, speculate=True),
            make_step(
                "draft",
                "gmail.draft_email",
                {
                    "to": ["cancel@turkishairlines.com"],
                    "subject": "Cancellation",
                    "body": "{{mail.hits[0].id}}",
                },
                depends_on=["mail"],
                speculate=True,
            ),
        ],
        services=["gmail"],
        has_write=True,
    )
    reject(op_registry, plan, ("speculat", "write", "side effect", "subtree", "irreversible"))
