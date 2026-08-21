"""Error classification, backoff and the circuit breaker.

Classification is the decision. Everything downstream — whether to retry, how
long to wait, whether the breaker trips, whether the job goes to the dead letter
queue, what the user is told — reads the class and nothing else. So getting the
class right from a real Google payload is the single most load-bearing thing in
`app/google/retry.py`, and it is most of this file.

The table, from `docs/DESIGN.md` §6.1:

    TRANSIENT         500/502/503/504, reset, read timeout    retry
    RATE_LIMITED      429, 403 rateLimitExceeded              retry, honour Retry-After
    QUOTA_EXHAUSTED   403 dailyLimitExceeded/quotaExceeded    not in-request
    AUTH_EXPIRED      401 invalid credentials                 refresh, retry once
    AUTH_REVOKED      invalid_grant, insufficientPermissions  no
    PRECONDITION      412 conditionNotMet                     re-read first
    NOT_FOUND         404                                     no — it is an answer
    INVALID           400 badRequest, 422                     no — it is our bug

Every error here is built from a real response envelope and parsed by the same
code the live client uses, so these tests cover the envelope reading as well as
the classification.

Backoff is full jitter, `uniform(0, min(cap, base * 2**attempt))`, not equal
jitter. Equal jitter guarantees a minimum wait, which packs every retry into the
upper half of the window and lets a herd re-collide there. The test below proves
which one is implemented rather than taking it on trust.
"""

from __future__ import annotations

import inspect

import pytest

from tests.conftest import call, get, google_error, load_any, oauth_error

_RETRY = "app.google.retry"

ErrorClass = load_any(_RETRY, "ErrorClass")
classify = load_any(_RETRY, "classify")
backoff = load_any(_RETRY, "backoff")
retryable = load_any(_RETRY, "retryable")
parse_retry_after = load_any(_RETRY, ["parse_retry_after", "retry_after", "retry_after_seconds"])


def cls_of(exc) -> str:
    """The class name, whether an enum member or a bare string comes back."""
    result = call(classify, exc)
    return str(get(result, "value", result))


# ---------------------------------------------------------------------------
# The eight classes, from payloads Google actually sends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_five_hundred_is_transient(status):
    assert cls_of(google_error(status, "backendError", "Backend Error")) == "TRANSIENT"


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError("Connection reset by peer"),
        TimeoutError("The read operation timed out"),
        ConnectionError("Remote end closed connection without response"),
    ],
)
def test_a_broken_connection_is_transient(exc):
    # Nothing came back at all, so there is no status to read. These still have
    # to land somewhere sensible, and a socket that died mid-request is the
    # most retryable thing there is.
    assert cls_of(exc) == "TRANSIENT"


def test_a_429_is_rate_limited():
    assert cls_of(google_error(429, "rateLimitExceeded", "Rate Limit Exceeded")) == "RATE_LIMITED"


@pytest.mark.parametrize("reason", ["rateLimitExceeded", "userRateLimitExceeded"])
def test_a_403_that_means_slow_down_is_rate_limited(reason):
    # Google overloads 403. The reason string is the only thing separating
    # "wait a moment" from "you are out of quota until midnight" from "you no
    # longer have permission", and the three want completely different handling.
    assert cls_of(google_error(403, reason, "Rate Limit Exceeded")) == "RATE_LIMITED"


@pytest.mark.parametrize("reason", ["dailyLimitExceeded", "quotaExceeded"])
def test_a_403_that_means_out_of_quota_is_quota_exhausted(reason):
    assert cls_of(google_error(403, reason, "Daily Limit Exceeded")) == "QUOTA_EXHAUSTED"


def test_the_reason_beats_the_status():
    # A bare 403 is a permission problem. The same 403 carrying
    # `rateLimitExceeded` is not, and reading only the status would revoke a
    # perfectly good token because the mailbox was busy.
    assert cls_of(google_error(403)) == "AUTH_REVOKED"
    assert cls_of(google_error(403, "rateLimitExceeded")) == "RATE_LIMITED"


def test_a_401_is_auth_expired():
    error = google_error(401, "authError", "Invalid Credentials", domain="global")
    assert cls_of(error) == "AUTH_EXPIRED"


def test_a_403_insufficient_permissions_is_auth_revoked():
    # The user took a scope away. Retrying cannot help and would just burn
    # quota; the answer is a reconnect button.
    error = google_error(403, "insufficientPermissions", "Insufficient Permission")
    assert cls_of(error) == "AUTH_REVOKED"


def test_an_invalid_grant_from_the_refresh_endpoint_is_auth_revoked():
    # The OAuth endpoints use a flatter envelope than the API ones:
    # {"error": "invalid_grant", "error_description": "..."}. This is the line
    # between "refresh the token" and "ask the user to reconnect".
    assert cls_of(oauth_error("invalid_grant")) == "AUTH_REVOKED"


def test_a_412_is_a_precondition_failure():
    # Our If-Match against `sync_gcal.etag`. Somebody moved the event while the
    # confirm card was on screen.
    error = google_error(412, "conditionNotMet", "Precondition Failed")
    assert cls_of(error) == "PRECONDITION"


def test_a_full_sync_required_is_a_precondition_failure():
    # Calendar's way of saying the sync token is too old to use.
    error = google_error(410, "fullSyncRequired", "Sync token is no longer valid")
    assert cls_of(error) == "PRECONDITION"


def test_a_404_is_not_found():
    # Excluded from the error rate on purpose: "that event is gone" is an
    # answer, and counting it as a failure would make the dashboard lie.
    assert cls_of(google_error(404, "notFound", "Not Found")) == "NOT_FOUND"


@pytest.mark.parametrize(
    ("status", "reason"),
    [(400, "badRequest"), (400, "invalidArgument"), (422, "invalid")],
)
def test_a_bad_request_is_invalid(status, reason):
    # Our bug, or an argument the planner invented. Retrying an argument error
    # produces the same argument error at a cost.
    assert cls_of(google_error(status, reason, "Bad Request")) == "INVALID"


def test_something_nobody_mapped_is_unknown():
    assert cls_of(ValueError("no idea what this is")) == "UNKNOWN"
    assert cls_of(RuntimeError("something odd happened in a library")) == "UNKNOWN"


def test_every_class_in_the_contract_exists():
    names = {member.name for member in ErrorClass}
    assert names >= {
        "TRANSIENT",
        "RATE_LIMITED",
        "QUOTA_EXHAUSTED",
        "AUTH_EXPIRED",
        "AUTH_REVOKED",
        "PRECONDITION",
        "NOT_FOUND",
        "INVALID",
        "UNKNOWN",
    }


def test_the_error_carries_what_the_trace_panel_needs():
    # `node_executions.outcome` is {reason, class, code, message}, and the
    # answer text is built from it. An error that loses the reason on the way
    # up leaves the user with "something went wrong".
    error = google_error(429, "userRateLimitExceeded", "Too many requests")
    assert get(error, "status", None) == 429
    assert "userRateLimitExceeded".lower() in str(get(error, "reason", "")).lower()


# ---------------------------------------------------------------------------
# retryable()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("TRANSIENT", True),
        ("RATE_LIMITED", True),
        ("AUTH_EXPIRED", True),  # once, after a token refresh
        ("QUOTA_EXHAUSTED", False),  # not in-request; the worker tries later
        ("AUTH_REVOKED", False),
        ("NOT_FOUND", False),
        ("INVALID", False),
    ],
)
def test_retryable_per_class(name, expected):
    assert call(retryable, ErrorClass[name]) is expected


def test_a_stale_etag_is_never_repeated_blind():
    # A precondition failure means what we held is out of date. If it is retried
    # at all, the resource has to be read again first — repeating the same
    # If-Match would either fail identically or, worse, succeed against a
    # version the user has not seen.
    try:
        needs_refetch = load_any(_RETRY, ["needs_refetch", "refetch_first", "must_refetch"])
    except AttributeError:
        assert call(retryable, ErrorClass["PRECONDITION"]) is False
        return
    assert call(needs_refetch, ErrorClass["PRECONDITION"]) is True
    for other in ("TRANSIENT", "RATE_LIMITED", "NOT_FOUND", "INVALID"):
        assert call(needs_refetch, ErrorClass[other]) is False


def test_an_unclassifiable_error_is_not_retried():
    # "A write we cannot classify is a write we do not repeat." The failure mode
    # is a duplicate email, which no amount of retrying can take back.
    params = inspect.signature(retryable).parameters
    write_param = next(
        (p for p in params if p in ("is_write", "write", "writing", "idempotent")), None
    )
    if write_param is not None:
        assert call(retryable, ErrorClass["UNKNOWN"], **{write_param: True}) is False
    else:
        assert call(retryable, ErrorClass["UNKNOWN"]) is False


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

SAMPLES = 400
HARD_CAP_S = 60.0


def samples(name: str, attempt: int) -> list[float]:
    return [float(call(backoff, ErrorClass[name], attempt)) for _ in range(SAMPLES)]


def test_backoff_is_never_negative_and_never_past_the_cap():
    for attempt in range(8):
        for value in samples("TRANSIENT", attempt):
            assert 0.0 <= value <= HARD_CAP_S, (attempt, value)


def test_backoff_stays_capped_however_many_attempts():
    # 2**20 seconds is eleven days. The cap is what stops an exponential from
    # becoming an outage of its own.
    for value in samples("TRANSIENT", 20):
        assert value <= HARD_CAP_S


def test_the_window_widens_with_each_attempt():
    early = max(samples("TRANSIENT", 0))
    later = max(samples("TRANSIENT", 4))
    assert later > early


def test_backoff_is_full_jitter_and_not_equal_jitter():
    # Full jitter draws from [0, window]. Equal jitter draws from
    # [window/2, window] and so can never produce a short wait. Over 400 draws
    # the smallest one settles it: with full jitter it lands near zero, with
    # equal jitter it cannot go below half the largest.
    drawn = samples("TRANSIENT", 5)
    smallest, largest = min(drawn), max(drawn)

    assert largest > 0.0
    assert smallest < largest / 2.0, "looks like equal jitter, not full jitter"
    assert smallest < largest * 0.05, "the lower half of the window is not being used"

    # And it really is spread out, rather than a fixed sleep with noise on it.
    assert len({round(v, 3) for v in drawn}) > SAMPLES // 2


def test_two_calls_do_not_agree():
    # If they did, every client in a herd would wake up together, which is the
    # exact failure the jitter is there to prevent.
    assert len({call(backoff, ErrorClass["RATE_LIMITED"], 3) for _ in range(50)}) > 1


def test_a_rate_limit_backs_off_harder_than_a_blip():
    # A 503 is one server having a bad moment. A 429 is a shared quota, where
    # coming back quickly makes it worse for everybody.
    rate = max(samples("RATE_LIMITED", 3))
    transient = max(samples("TRANSIENT", 3))
    assert rate > transient


def test_a_class_that_is_not_retried_waits_for_nothing():
    for name in ("NOT_FOUND", "INVALID", "AUTH_REVOKED", "QUOTA_EXHAUSTED"):
        assert float(call(backoff, ErrorClass[name], 0)) == 0.0


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------


def _backoff_takes_retry_after() -> str | None:
    params = inspect.signature(backoff).parameters
    return next((p for p in params if "retry_after" in p or p == "hint"), None)


@pytest.mark.parametrize(("header", "expected"), [("7", 7.0), ("0", 0.0), ("120", 120.0)])
def test_a_numeric_retry_after_is_read_as_seconds(header, expected):
    assert float(call(parse_retry_after, header)) == pytest.approx(expected)


def test_a_missing_retry_after_is_none():
    assert call(parse_retry_after, None) is None
    assert call(parse_retry_after, "") is None


def test_a_nonsense_retry_after_does_not_blow_up():
    # Anything can arrive in a header. A crash here would turn a rate limit
    # into a 500.
    assert call(parse_retry_after, "soon") is None


def test_a_retry_after_header_reaches_the_error():
    error = google_error(429, "rateLimitExceeded", headers={"Retry-After": "7"})
    assert float(get(error, "retry_after", 0.0) or 0.0) == pytest.approx(7.0)


def test_a_retry_after_beats_the_computed_backoff():
    param = _backoff_takes_retry_after()
    if param is None:
        pytest.skip("backoff() takes no Retry-After hint; the caller applies it")

    # Google said seven seconds. Coming back in 0.4 seconds because the jitter
    # rolled low is how a rate limit becomes a ban.
    waits = [
        float(call(backoff, ErrorClass["RATE_LIMITED"], 0, **{param: 7.0})) for _ in range(SAMPLES)
    ]
    assert min(waits) >= 7.0 - 1e-6


def test_no_retry_after_means_the_ordinary_curve():
    param = _backoff_takes_retry_after()
    if param is None:
        pytest.skip("backoff() takes no Retry-After hint")
    waits = [
        float(call(backoff, ErrorClass["RATE_LIMITED"], 0, **{param: None})) for _ in range(SAMPLES)
    ]
    assert min(waits) < 7.0


# ---------------------------------------------------------------------------
# The circuit breaker
# ---------------------------------------------------------------------------
#
# Per (user, service), because the usual cause is one user's revoked token or
# one user's pathological mailbox, and a global breaker would let one bad tenant
# blind everybody.
#
# The transitions themselves live in Redis, in two Lua scripts that run as one
# round trip so two workers cannot both believe they are the half-open probe.
# That makes the full closed → open → half-open → closed walk an integration
# test, and it is at the bottom of this file, skipped when there is no Redis to
# talk to. What is unit-testable is the policy those scripts are driven by: the
# threshold, the ladder of open windows, which failures count at all, and what
# happens when Redis itself is the thing that is down.

BREAKER_THRESHOLD = load_any(_RETRY, ["BREAKER_THRESHOLD", "BREAKER_FAILURES"])
BREAKER_OPEN_S = load_any(_RETRY, ["BREAKER_OPEN_S", "BREAKER_BASE_OPEN_S"])
BREAKER_MAX_OPEN_S = load_any(_RETRY, ["BREAKER_MAX_OPEN_S", "BREAKER_CAP_OPEN_S"])
BREAKER_CLASSES = load_any(_RETRY, ["BREAKER_CLASSES", "BREAKER_ERROR_CLASSES"])
breaker_allow = load_any(_RETRY, ["breaker_allow", "allow"])
breaker_record_failure = load_any(_RETRY, ["breaker_record_failure", "record_failure"])
breaker_record_success = load_any(_RETRY, ["breaker_record_success", "record_success"])

USER = "u_7QkR2mXvB4nLd9TsW"


def test_five_consecutive_failures_is_the_threshold():
    assert int(BREAKER_THRESHOLD) == 5


def test_the_first_open_window_is_five_minutes():
    assert float(BREAKER_OPEN_S) == 300.0


def test_the_window_doubles_to_a_thirty_minute_ceiling():
    # 5, 10, 20, 30, 30. Doubling without a ceiling means a service that has
    # been down all morning is never checked again; a ceiling of thirty minutes
    # costs two probes an hour and keeps recovery automatic.
    window = float(BREAKER_OPEN_S)
    ladder = []
    for _ in range(5):
        ladder.append(min(window, float(BREAKER_MAX_OPEN_S)) / 60.0)
        window *= 2
    assert ladder == [5.0, 10.0, 20.0, 30.0, 30.0]


@pytest.mark.parametrize("name", ["TRANSIENT", "RATE_LIMITED", "QUOTA_EXHAUSTED", "UNKNOWN"])
def test_a_failure_that_is_about_google_moves_the_breaker(name):
    assert ErrorClass[name] in BREAKER_CLASSES


@pytest.mark.parametrize("name", ["NOT_FOUND", "INVALID", "PRECONDITION", "AUTH_REVOKED"])
def test_a_failure_that_is_about_the_request_does_not(name):
    # Five deleted events in a row says nothing about Calendar's health.
    # Counting them would open the breaker on a user whose service is fine.
    assert ErrorClass[name] not in BREAKER_CLASSES


async def test_a_not_found_never_trips_the_breaker():
    # Returns before it ever reaches Redis, which is also why this is a unit test.
    decision = await breaker_record_failure(USER, "gmail", ErrorClass["NOT_FOUND"])
    assert bool(get(decision, "allowed", True)) is True
    assert "open" not in str(get(decision, "state", "closed")).lower()


async def test_a_breaker_that_cannot_be_read_lets_the_call_through(monkeypatch):
    # A breaker that cannot be reached must not become an outage of its own.
    # Failing closed here would mean one Redis blip stops every Google call in
    # the fleet.
    from app.core import cache

    async def no_redis():
        raise OSError("redis is not answering")

    monkeypatch.setattr(cache, "get_redis", no_redis)

    decision = await breaker_allow(USER, "gmail")
    assert bool(get(decision, "allowed", False)) is True

    failure = await breaker_record_failure(USER, "gmail", ErrorClass["TRANSIENT"])
    assert bool(get(failure, "allowed", False)) is True

    # And closing it must not raise either — a success during a Redis outage is
    # still a success.
    await breaker_record_success(USER, "gmail")


async def test_the_open_breaker_error_reads_as_transient():
    # `CircuitOpen` is what the dispatcher raises instead of calling Google.
    # It has to classify as transient, or a paused service would look to the
    # rest of the system like a permanent failure.
    circuit_open = load_any(_RETRY, ["CircuitOpen"])
    exc = circuit_open(USER, "gmail", 42.0)
    assert cls_of(exc) == "TRANSIENT"
    assert get(exc, "code", "") == "GOOGLE_UNAVAILABLE"
    assert get(exc, "details", {}).get("retry_after_s") == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# The transitions, against a real Redis
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis_or_skip():
    """A live Redis, or skip. The Lua scripts cannot run anywhere else."""
    from redis.exceptions import RedisError

    from app.core import cache

    try:
        client = await cache.get_redis()
        await client.ping()
    except (RedisError, OSError) as exc:
        pytest.skip(f"no redis: {exc}")
    return client


@pytest.mark.integration
async def test_five_failures_open_it_and_one_success_closes_it(redis_or_skip):
    service = "gmail_breaker_test"
    breaker_reset = load_any(_RETRY, ["breaker_reset"])
    await breaker_reset(USER, service)

    for _ in range(int(BREAKER_THRESHOLD) - 1):
        await breaker_record_failure(USER, service, ErrorClass["TRANSIENT"])
        assert bool(get(await breaker_allow(USER, service), "allowed", False)) is True

    opened = await breaker_record_failure(USER, service, ErrorClass["TRANSIENT"])
    assert bool(get(opened, "allowed", True)) is False
    assert bool(get(await breaker_allow(USER, service), "allowed", True)) is False

    # One good call is enough. That is the point of a probe.
    await breaker_record_success(USER, service)
    assert bool(get(await breaker_allow(USER, service), "allowed", False)) is True

    await breaker_reset(USER, service)
