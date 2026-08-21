"""When Google falls over: partial answers, named gaps, no 500s.

"Gmail succeeded, Calendar failed" is the case the brief calls out, and the
design's answer is that a broken service is a *fact in the answer* rather than
an exception on the way out. So:

* the failing node retries inside the in-request budget — two attempts, capped
  at 1.5 s of added latency, because a person is watching a cursor blink;
* it lands in ``failed`` with an ``outcome`` naming the class and the code;
* everything that depended on it is ``skipped``, not ``failed`` — it never ran,
  and a skip must not count against the circuit breaker;
* ``runs.status`` stays ``complete``. Rolling the whole turn to ``failed``
  because one node broke would throw away the results the person can still use;
* the answer says which service is down, in its own words, before anything else.

The last one is only worth asserting if it can fail. The canned prose in
``tests/fixtures/llm_responses.py`` names a service **only when the synthesis
prompt carried the failure**, so an orchestrator that quietly drops its
``degraded`` block produces an answer that does not mention Calendar — and this
file says so.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import (
    answer_text,
    degraded_services,
    load_run,
    load_step_rows,
    post_query,
    require,
    status_of,
)

pytestmark = pytest.mark.integration

ACME_PREP = "Prepare for tomorrow's meeting with Acme Corp"


def outcome_of(row) -> dict:
    return row.outcome or {}


def retries_of(row) -> list:
    return list(row.retries or [])


def by_service(rows, service: str) -> list:
    return [r for r in rows if r.op.split(".", 1)[0] == service]


# --------------------------------------------------------------------------- #
# 503 — the headline case
# --------------------------------------------------------------------------- #


async def test_calendar_503_degrades_the_answer_instead_of_failing_the_run(
    client, db, mirrored, llm, google
):
    """Calendar is down; Drive is fine; the answer says so.

    The planner marked the meeting lookup ``freshness: live`` because the
    question is about *tomorrow* and a mirror up to fifteen minutes stale does
    not get to state a meeting time as fact. So this one really does call
    Google, and Google really does 503.
    """
    llm.use("acme_meeting_prep_degraded")
    google.fail("gcal", 503)

    payload = await post_query(client, ACME_PREP, freshness="live")

    assert payload["status"] == "complete", (
        "one broken node does not fail the turn — the Drive results are still "
        f"worth having: {payload.get('error')}"
    )

    rows = await load_step_rows(db, mirrored.id, payload["run_id"])
    assert rows, "no steps were recorded"
    calendar = by_service(rows, "gcal")
    assert calendar, f"no calendar step ran: {[r.op for r in rows]}"

    failed = [r for r in calendar if status_of(r) in {"failed", "timeout"}]
    assert failed, (
        f"Calendar returned 503 and the step should have failed: "
        f"{[(r.node_id, status_of(r)) for r in calendar]}"
    )
    detail = outcome_of(failed[0])
    assert detail, "a failed node has to say why"
    assert "503" in str(detail) or "unavailable" in str(detail).lower(), detail

    attempts = retries_of(failed[0])
    assert attempts, (
        "the in-request tier retries once before giving up; nothing was recorded "
        f"in retries: {failed[0].retries}"
    )
    assert len(attempts) <= 2, (
        "a user-facing read does not get to spend eleven seconds being brave — "
        f"the in-request budget is two attempts: {attempts}"
    )
    assert any("503" in str(a) or "TRANSIENT" in str(a).upper() for a in attempts), attempts

    calls = google.count("gcal")
    assert calls >= 2, f"Calendar was never retried, only called {calls} time(s)"
    assert calls <= 4, (
        f"{calls} calls to a service that is returning 503 — the in-request tier "
        "is two attempts, capped at 1.5 s of added latency"
    )

    # -- the dependant was skipped, not failed ----------------------------- #
    skipped = [r for r in rows if status_of(r) == "skipped"]
    assert skipped, (
        "the mail search filters on the meeting's guest list, so it could not "
        f"bind and should have been skipped: {[(r.node_id, status_of(r)) for r in rows]}"
    )
    reason = outcome_of(skipped[0])
    assert reason, "a skipped node has to say what it was waiting for"
    assert "depend" in str(reason).lower() or any(
        r.node_id in str(reason) for r in failed
    ), f"the skip should name its dependency: {reason}"

    succeeded = [r for r in rows if status_of(r) == "succeeded"]
    assert succeeded, "Drive was healthy; something should have come back"

    # -- the run says which service is gone -------------------------------- #
    named = degraded_services(payload)
    assert "gcal" in named, (
        f"the response should name the service that failed: {payload.get('degraded')}"
    )

    runs = require("app.db.repositories.runs")
    await db.rollback()
    derived = await runs.failed_services(db, mirrored.id, payload["run_id"])
    assert "gcal" in derived, (
        f"failed_services is derived from node_executions, not stored: {derived}"
    )

    # -- and the answer says it in words ----------------------------------- #
    text = answer_text(payload)
    assert "calendar" in text.lower() or "gcal" in text.lower(), (
        "the person has to be told which part of the answer is missing and why. "
        f"The answer was:\n{text}"
    )

    prose = llm.prose_calls()
    if prose:
        context = llm.prompts[-1].lower()
        assert "gcal" in context or "calendar" in context, (
            "the synthesizer was not told about the failure, so it could not "
            "have opened with it. The failure has to reach the prompt as "
            "structured context, not as an absence."
        )


async def test_a_degraded_run_still_answers_from_the_services_that_worked(
    client, db, mirrored, llm, google
):
    """The point of degrading rather than failing: the rest of the answer."""
    llm.use("acme_meeting_prep_degraded")
    google.fail("gcal", 503)

    payload = await post_query(client, ACME_PREP, freshness="live")
    text = answer_text(payload)

    assert "Acme" in text, f"the Drive results should still be in the answer:\n{text}"
    assert payload.get("actions") in (None, []), "nothing was prepared, and nothing sent"
    assert not google.mutations, [str(r) for r in google.mutations]

    run = await load_run(db, mirrored.id, payload["run_id"])
    assert status_of(run) == "complete"
    assert run.finished_at is not None


# --------------------------------------------------------------------------- #
# 429 — a rate limit is not a failure
# --------------------------------------------------------------------------- #


async def test_a_429_with_retry_after_is_classified_and_retried(
    client, db, mirrored, llm, google
):
    """One 429, then the service recovers.

    Google's body carries ``errors[0].reason = "userRateLimitExceeded"`` and the
    response carries ``Retry-After``. Both are the classifier's input: this is
    RATE_LIMITED, which is retryable, and it is not the same thing as
    QUOTA_EXHAUSTED even though both arrive as 429.
    """
    llm.use("acme_meeting_prep_degraded")
    google.rate_limit("gcal", retry_after=1, times=1)

    payload = await post_query(client, ACME_PREP, freshness="live")
    assert payload["status"] == "complete"

    assert google.count("gcal") >= 2, (
        "a retryable 429 should have been tried again, not surfaced: "
        f"{[str(r) for r in google.calls('gcal')]}"
    )

    rows = await load_step_rows(db, mirrored.id, payload["run_id"])
    calendar = by_service(rows, "gcal")
    assert calendar, [r.op for r in rows]

    recorded = str([retries_of(r) for r in calendar] + [outcome_of(r) for r in calendar])
    assert "429" in recorded or "RATE_LIMITED" in recorded.upper(), (
        f"the attempt should be on the record with its class: {recorded}"
    )

    # It recovered on the second attempt, so the run is not degraded.
    if status_of(calendar[0]) == "succeeded":
        assert "gcal" not in degraded_services(payload), (
            "a service that recovered is not degraded"
        )


async def test_a_rate_limited_run_is_still_a_200(client, db, mirrored, llm, google):
    """Even when the retry does not save it, the person gets an answer.

    A 429 that outlasts the in-request budget degrades the run the same way a
    503 does. What must not happen is a 500.
    """
    llm.use("acme_meeting_prep_degraded")
    google.rate_limit("gcal", retry_after=2)

    payload = await post_query(client, ACME_PREP, freshness="live")

    assert payload["status"] in {"complete", "awaiting_input"}
    run = await load_run(db, mirrored.id, payload["run_id"])
    assert status_of(run) != "failed", (
        f"a rate limit on one service is not a failed turn: {run.error}"
    )
    assert "gcal" in degraded_services(payload) or answer_text(payload), (
        "either name the degraded service or say something useful; silence is "
        "the one option that is not available"
    )


# --------------------------------------------------------------------------- #
# 401 — the grant is gone
# --------------------------------------------------------------------------- #


async def test_a_401_that_survives_a_refresh_asks_the_person_to_reconnect(
    client, db, mirrored, llm, google
):
    """The session is fine; the Google grant is not.

    That is a different thing from being signed out, and it gets a different
    answer: reconnect, not sign in. The run parks rather than failing, because
    the person can fix this and the plan is still good.
    """
    llm.use("acme_meeting_prep_degraded")
    google.expire_auth(refresh_ok=False)

    response = await client.post(
        "/api/v1/query",
        json={"query": ACME_PREP, "timezone": "America/New_York", "freshness": "live"},
    )
    assert response.status_code in {200, 428}, (
        f"a dead grant is a 428 or a parked run, never a 500: {response.status_code} "
        f"{response.text[:500]}"
    )
    body = response.json()

    assert google.refreshes >= 1, (
        "the access token should have been refreshed before giving up — a 401 "
        "is AUTH_EXPIRED until the refresh itself fails"
    )

    if response.status_code == 428:
        assert body["error"]["code"] == "GOOGLE_REAUTH_REQUIRED"
        assert "reconnect" in body["error"]["message"].lower(), body["error"]
        return

    assert body["status"] == "awaiting_input", (
        f"expected the run to park on a reconnect prompt, got {body['status']}"
    )
    run = await load_run(db, mirrored.id, body["run_id"])
    assert status_of(run) == "awaiting_input", (
        f"the run must not be marked failed — the person can fix this: {status_of(run)}"
    )

    cards = body.get("pending_inputs") or []
    assert cards, "awaiting_input with nothing to answer"
    text = (answer_text(body) + str(cards)).lower()
    assert "reconnect" in text or "reauth" in text or "google" in text, (
        f"the card has to say what to do about it: {cards}"
    )


async def test_a_401_that_a_refresh_fixes_is_invisible(client, db, mirrored, llm, google):
    """An expired access token is routine. The person should never see it."""
    llm.use("acme_meeting_prep_degraded")
    google.expire_auth(refresh_ok=True)

    payload = await post_query(client, ACME_PREP, freshness="live")

    assert google.refreshes >= 1, "the token should have been refreshed"
    assert payload["status"] == "complete", (
        f"a refreshable 401 is not a degradation: {payload.get('degraded')}"
    )
    text = answer_text(payload).lower()
    assert "reconnect" not in text, f"nothing to tell the person about:\n{text}"
