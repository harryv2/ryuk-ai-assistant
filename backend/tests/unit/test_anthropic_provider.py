"""The Anthropic adapter, checked against the things that actually break.

No network, no API key, no SDK client. `AnthropicProvider` takes a client in its
constructor, so every test here hands it a double that records what the provider
tried to send and replies with a response shaped like a real one. What is being
tested is the translation in both directions: our four calls into Anthropic's
request shape, and Anthropic's failures into our `ErrorClass` names.

Six things earn most of this file, because each one is a way to be wrong that
does not look wrong:

**A refusal is an HTTP 200.** ``stop_reason == "refusal"`` raises no exception
and the message still has content blocks. A provider that reads the content
first hands a refusal to the planner as though it were an answer.

**The exception hierarchy is not flat.** `APITimeoutError` is a subclass of
`APIConnectionError`, and every 4xx class is a subclass of `APIStatusError`. The
order of the `isinstance` chain in `classify_error` is load-bearing: one clause
in the wrong place turns every 400 into TRANSIENT and the router starts
retrying a request that can never work.

**A rate limit knows how long to wait.** The number is in a header, and it goes
in the log line that explains the fallback.

**Anthropic cannot embed.** Not slowly, not badly — at all. `embed` must raise
and must say what to set instead, because the alternative is a vector-shaped
value going into a `VECTOR(1536)` column that a different model filled.

**Four parameters are now 400s.** ``budget_tokens``, ``temperature``, ``top_p``
and ``top_k`` were removed on claude-opus-5, claude-sonnet-5 and claude-fable-5.
Sending any of them fails the whole call, so the request is scanned for all
four.

**A stream must not be a buffer.** `router.stream_json` exists so the
orchestrator can start step one while step four is still being written. A
provider that joins the chunks and yields once still passes a naive test, so the
test here records the order in which chunks were produced and consumed and
insists that the two interleave.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from app.llm.base import Provider
from app.llm.errors import DEFAULT_FALLBACK_ON, ErrorClass, LLMError

anthropic = pytest.importorskip(
    "anthropic",
    reason="The Anthropic SDK is not installed. `pip install anthropic`, or "
    "`pip install -e backend` to pick it up from pyproject.toml.",
)

from app.llm.providers.anthropic_provider import (  # noqa: E402  (must follow importorskip)
    DEFAULT_MODEL,
    EFFORTS,
    PRICES,
    PROVIDER_NAME,
    AnthropicProvider,
    price,
    provider,
)

# The parameters that were removed from the current models. Any one of them in a
# request is a 400 on the whole call, not a warning.
REMOVED_PARAMETERS = ("budget_tokens", "temperature", "top_p", "top_k")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {"type": "object", "properties": {"op": {"type": "string"}}},
        },
    },
    "required": ["answer"],
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def a_usage(
    *, fresh: int = 200, output: int = 60, cache_read: int = 0, cache_write: int = 0
) -> SimpleNamespace:
    """Anthropic's four counts. `input_tokens` excludes both cache numbers."""
    return SimpleNamespace(
        input_tokens=fresh,
        output_tokens=output,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def a_message(
    *,
    text: str = '{"answer": "yes"}',
    stop_reason: str = "end_turn",
    stop_details: Any = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """A response shaped like the real one, including the request id."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage if usage is not None else a_usage(),
        _request_id="req_0123456789",
    )


class FakeMessages:
    """`client.messages`, recording every request it is handed."""

    def __init__(
        self,
        *,
        response: Any = None,
        chunks: tuple[str, ...] = (),
        final: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.chunks = chunks
        self.final = final
        self.error = error
        self.requests: list[dict[str, Any]] = []
        # Who did what, in order. The streaming test reads this.
        self.timeline: list[tuple[str, int]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response

    def stream(self, **request: Any) -> FakeStream:
        """Not a coroutine. The real SDK returns a context manager here too."""
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return FakeStream(self.chunks, self.final, self.timeline)


class FakeStream:
    """`async with client.messages.stream(...) as stream:`."""

    def __init__(
        self, chunks: tuple[str, ...], final: Any, timeline: list[tuple[str, int]]
    ) -> None:
        self._chunks = chunks
        self._final = final
        self._timeline = timeline
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        self.closed = True
        return False

    @property
    def text_stream(self) -> Any:
        async def chunks() -> Any:
            for index, piece in enumerate(self._chunks):
                self._timeline.append(("produced", index))
                yield piece

        return chunks()

    async def get_final_message(self) -> Any:
        return self._final


def a_provider(**kwargs: Any) -> tuple[AnthropicProvider, FakeMessages]:
    messages = FakeMessages(**kwargs)
    return AnthropicProvider(client=SimpleNamespace(messages=messages)), messages


@pytest.fixture(autouse=True)
def _plain_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run against the shipped defaults, not against whoever's shell this is.

    The knobs below are not fields on `Settings`, so the provider reads them
    from the environment. A developer with `ANTHROPIC_CACHE=off` exported should
    not change what these tests assert.
    """
    for name in (
        "ANTHROPIC_CACHE",
        "ANTHROPIC_MAX_TOKENS",
        "ANTHROPIC_STREAM_MAX_TOKENS",
        "ANTHROPIC_TIMEOUT_S",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Building SDK errors
# ---------------------------------------------------------------------------


def http_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status, headers=headers or {}, request=request, content=b"{}")


def status_error(
    kind: type, status: int, message: str, headers: dict[str, str] | None = None
) -> Any:
    return kind(message, response=http_response(status, headers), body=None)


def connection_error(kind: type) -> Any:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if kind is anthropic.APITimeoutError:
        return kind(request)
    return kind(message="Connection error.", request=request)


# ---------------------------------------------------------------------------
# The shape of the adapter
# ---------------------------------------------------------------------------


def test_the_adapter_is_a_provider():
    instance = provider()
    assert isinstance(instance, Provider)
    assert instance.name == PROVIDER_NAME == "anthropic"


def test_the_default_model_is_opus_five_and_it_is_priced():
    assert DEFAULT_MODEL == "claude-opus-5"
    assert PRICES["claude-opus-5"] == (5.00, 25.00)
    assert PRICES["claude-sonnet-5"] == (3.00, 15.00)
    assert PRICES["claude-haiku-4-5"] == (1.00, 5.00)
    assert PRICES["claude-fable-5"] == (10.00, 50.00)


def test_a_model_name_never_carries_the_provider_prefix():
    """The router strips the prefix before calling, so a price lookup on the
    bare name has to work — and the full reference must not."""
    assert price("claude-opus-5", 1_000_000, 0) == 5.00
    assert price("anthropic:claude-opus-5", 1_000_000, 0) == 0.0


def test_prefix_caching_is_advertised():
    instance, _ = a_provider(response=a_message())
    assert instance.supports_prefix_caching() is True


# ---------------------------------------------------------------------------
# A refusal is a 200
# ---------------------------------------------------------------------------

REFUSED_TEXT = '{"answer": "here is the thing the model refused to say"}'


async def test_a_refusal_raises_content_filtered_and_never_returns_the_content():
    instance, _ = a_provider(
        response=a_message(
            text=REFUSED_TEXT,
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="harmful_content", explanation="no"),
        )
    )

    with pytest.raises(LLMError) as caught:
        await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    err = caught.value
    assert err.error_class == ErrorClass.CONTENT_FILTERED
    assert err.provider == "anthropic"

    # Nothing the model was refusing to say may leak out through the failure.
    everywhere = " ".join([str(err), err.message, json.dumps(err.details, default=str)])
    assert "refused to say" not in everywhere

    # Why it refused is useful and is kept.
    assert err.details["category"] == "harmful_content"
    assert err.details["request_id"] == "req_0123456789"


async def test_a_refusal_still_reports_what_it_cost():
    """The prompt was read and billed even though nothing usable came back."""
    instance, _ = a_provider(
        response=a_message(
            text=REFUSED_TEXT,
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="harmful_content", explanation=None),
            usage=a_usage(fresh=1500, output=12),
        )
    )
    with pytest.raises(LLMError) as caught:
        await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    assert caught.value.usage is not None
    assert caught.value.usage.prompt_tokens == 1500


def test_a_refusal_does_not_buy_a_second_opinion():
    """CONTENT_FILTERED must stay out of the fallback set. A model that refused
    on safety grounds is not a model having a bad minute, and asking the next
    provider the same question just pays twice for the same no."""
    assert ErrorClass.CONTENT_FILTERED not in DEFAULT_FALLBACK_ON


async def test_stop_details_is_only_read_when_the_reason_is_refusal():
    """It is null for every other stop reason. Reading it unguarded is an
    AttributeError on the happy path, which is the worst place for one."""
    instance, _ = a_provider(response=a_message(stop_reason="end_turn", stop_details=None))
    result = await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)
    assert result.finish_reason == "stop"
    assert result.parsed == {"answer": "yes"}


# ---------------------------------------------------------------------------
# Naming the failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (status_error(anthropic.BadRequestError, 400, "bad schema"), "INVALID"),
        (status_error(anthropic.AuthenticationError, 401, "bad key"), "AUTH_REVOKED"),
        (status_error(anthropic.PermissionDeniedError, 403, "not allowed"), "AUTH_REVOKED"),
        (status_error(anthropic.NotFoundError, 404, "no such model"), "NOT_FOUND"),
        (status_error(anthropic.RateLimitError, 429, "slow down"), "RATE_LIMITED"),
        (status_error(anthropic.APIStatusError, 500, "server error"), "TRANSIENT"),
        (status_error(anthropic.APIStatusError, 503, "unavailable"), "TRANSIENT"),
        (status_error(anthropic.APIStatusError, 529, "overloaded"), "TRANSIENT"),
        (connection_error(anthropic.APIConnectionError), "TRANSIENT"),
        (connection_error(anthropic.APITimeoutError), "TIMEOUT"),
        (TimeoutError("took too long"), "TIMEOUT"),
        (RuntimeError("something else entirely"), "UNKNOWN"),
    ],
)
def test_every_sdk_exception_gets_the_right_class(exc, expected):
    instance, _ = a_provider()
    assert instance.classify_error(exc) == expected


def test_a_timeout_is_not_swallowed_by_the_connection_clause():
    """APITimeoutError inherits from APIConnectionError. If the broad clause
    ran first, a stalled request would be TRANSIENT and get retried on the same
    provider instead of counted against the deadline."""
    instance, _ = a_provider()
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)
    assert instance.classify_error(connection_error(anthropic.APITimeoutError)) == "TIMEOUT"
    assert instance.classify_error(connection_error(anthropic.APIConnectionError)) == "TRANSIENT"


def test_a_four_hundred_is_not_swallowed_by_the_status_clause():
    """Every 4xx class subclasses APIStatusError, so the specific clauses have
    to come first or a malformed request reads as a server problem."""
    instance, _ = a_provider()
    assert issubclass(anthropic.BadRequestError, anthropic.APIStatusError)
    assert (
        instance.classify_error(status_error(anthropic.BadRequestError, 400, "nope")) == "INVALID"
    )


def test_a_rate_limit_carries_retry_after_through():
    instance, _ = a_provider()
    exc = status_error(anthropic.RateLimitError, 429, "slow down", {"retry-after": "42"})

    assert instance.classify_error(exc) == "RATE_LIMITED"
    assert exc.retry_after_s == 42.0
    assert exc.response.headers["retry-after"] == "42"


def test_a_rate_limit_without_the_header_is_still_a_rate_limit():
    instance, _ = a_provider()
    exc = status_error(anthropic.RateLimitError, 429, "slow down")
    assert instance.classify_error(exc) == "RATE_LIMITED"
    assert exc.retry_after_s is None


def test_running_out_of_credit_is_quota_not_a_bad_request():
    """The one case where the status code lies. A 400 saying "credit balance is
    too low" is not our request being wrong, and it is worth trying the next
    provider — which INVALID would not do."""
    instance, _ = a_provider()
    broke = status_error(anthropic.BadRequestError, 400, "Your credit balance is too low")

    assert instance.classify_error(broke) == "QUOTA_EXHAUSTED"
    assert ErrorClass.QUOTA_EXHAUSTED in DEFAULT_FALLBACK_ON
    assert ErrorClass.INVALID not in DEFAULT_FALLBACK_ON


def test_classify_error_never_raises():
    """It runs on the failure path. An exception here loses the real failure."""
    instance, _ = a_provider()

    class Hostile(Exception):
        def __str__(self) -> str:
            raise ValueError("even reading me fails")

    assert instance.classify_error(Hostile()) == "UNKNOWN"
    assert instance.classify_error(None) == "UNKNOWN"  # type: ignore[arg-type]


def test_our_own_error_keeps_its_class():
    instance, _ = a_provider()
    mine = LLMError(ErrorClass.CONTENT_FILTERED, "already classified")
    assert instance.classify_error(mine) == "CONTENT_FILTERED"


# ---------------------------------------------------------------------------
# Embeddings, which do not exist
# ---------------------------------------------------------------------------


async def test_embed_refuses_and_says_what_to_set_instead():
    instance, _ = a_provider()

    with pytest.raises(LLMError) as caught:
        await instance.embed("claude-opus-5", ["some text"], 1536)

    message = caught.value.message
    assert "EMBED_MODEL" in message
    assert "no embeddings" in message.lower()
    # A message that names the constraint but not the fix is half a message.
    assert "openai" in message.lower()
    assert caught.value.provider == "anthropic"


@pytest.mark.parametrize("texts", [[], ["one"], ["one", "two"]])
async def test_embed_refuses_for_every_input_including_none_at_all(texts):
    """No quiet no-op on the empty list either. The failure this guards is
    silent: anything vector-shaped from the wrong model lands in a VECTOR(1536)
    column, scores against the real vectors, and returns wrong rows with no
    error anywhere to say why."""
    instance, _ = a_provider()
    with pytest.raises(LLMError):
        await instance.embed("claude-opus-5", texts, 1536)


# ---------------------------------------------------------------------------
# The request we build
# ---------------------------------------------------------------------------


async def sent_request(**kwargs: Any) -> dict[str, Any]:
    """Run one complete_json and hand back what reached the SDK."""
    instance, messages = a_provider(response=a_message())
    await instance.complete_json(
        kwargs.get("model", "claude-opus-5"),
        kwargs.get("system", "a long system prefix that would be cached"),
        kwargs.get("user", "the question"),
        kwargs.get("schema", SCHEMA),
        kwargs.get("max_tokens", 1200),
    )
    return messages.requests[0]


def every_key(node: Any) -> list[str]:
    """Every key at every depth, so a forbidden parameter cannot hide in a
    nested block."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(key)
            found.extend(every_key(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(every_key(item))
    return found


async def test_the_schema_goes_in_output_config_format():
    request = await sent_request()

    fmt = request["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert sorted(fmt["schema"]["required"]) == ["answer", "steps"]

    # The deprecated top-level parameter must not appear.
    assert "output_format" not in request
    # Not forced tool use any more, either.
    assert "tools" not in request
    assert "tool_choice" not in request


async def test_effort_rides_inside_output_config():
    request = await sent_request()
    assert request["output_config"]["effort"] in EFFORTS


async def test_the_cache_breakpoint_sits_at_the_end_of_the_system_prompt():
    """Not top level. Top level auto-caches the *last* cacheable block, which is
    the user's question — the one thing that changes every call — so the prefix
    would never match and cache_read_input_tokens would sit at zero."""
    request = await sent_request()

    system = request["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert "the question" not in json.dumps(system)


async def test_the_request_sends_none_of_the_four_removed_parameters():
    """budget_tokens, temperature, top_p and top_k all return 400 on the current
    models. The schema subtree is dropped before scanning, because a property in
    somebody's schema may legitimately be called "temperature"."""
    request = copy.deepcopy(await sent_request())
    request.get("output_config", {}).get("format", {}).pop("schema", None)

    keys = every_key(request)
    for parameter in REMOVED_PARAMETERS:
        assert parameter not in keys, f"{parameter} was removed from these models and now 400s"


async def test_thinking_is_sent_as_a_type_and_nothing_else():
    request = await sent_request()
    thinking = request.get("thinking")
    if thinking is not None:
        assert set(thinking) == {"type"}
        assert thinking["type"] in ("adaptive", "disabled", "enabled")


async def test_the_system_prompt_is_a_parameter_not_a_message_role():
    request = await sent_request()

    assert "system" in request
    assert [m["role"] for m in request["messages"]] == ["user"]
    assert request["messages"][0]["content"] == "the question"


async def test_there_is_no_assistant_prefill():
    """Prefill was removed on all current models and returns 400."""
    request = await sent_request()
    assert all(m["role"] != "assistant" for m in request["messages"])


async def test_max_tokens_is_raised_off_the_routers_lowball_default():
    """router.complete_json defaults to 1200. These models think first, and a
    truncated answer surfaces as INVALID, which does not fall back — so a cap
    that is too low costs the entire call."""
    request = await sent_request(max_tokens=1200)
    assert request["max_tokens"] >= 16000


async def test_a_caller_asking_for_more_than_the_floor_gets_what_it_asked_for():
    request = await sent_request(max_tokens=30000)
    assert request["max_tokens"] == 30000


async def test_prose_streaming_asks_for_no_schema():
    instance, messages = a_provider(chunks=("hello ", "there"), final=a_message(text="hello there"))
    async for _ in instance.stream_text("claude-opus-5", "sys", "user"):
        pass

    request = messages.requests[0]
    assert "format" not in request.get("output_config", {})
    assert request["max_tokens"] >= 64000


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_streaming_hands_each_chunk_over_as_it_arrives():
    """The point of the whole streaming path. If the provider buffered, the
    timeline would read produced, produced, produced, consumed, consumed,
    consumed — every chunk made before the caller saw any of them."""
    chunks = ('{"ans', 'wer": ', '"yes"}')
    instance, messages = a_provider(chunks=chunks, final=a_message(text="".join(chunks)))

    seen: list[str] = []
    stream = instance.stream_json("claude-opus-5", "sys", "user", SCHEMA, 1200)
    async for index, piece in _enumerate(stream):
        messages.timeline.append(("consumed", index))
        seen.append(piece)

    assert seen == list(chunks)
    assert messages.timeline == [
        ("produced", 0),
        ("consumed", 0),
        ("produced", 1),
        ("consumed", 1),
        ("produced", 2),
        ("consumed", 2),
    ]


async def test_streaming_reports_its_tokens_after_the_last_chunk():
    """Vendors count at the end, which is why a TextStream and not a bare
    generator: a generator object has nowhere to put the number."""
    instance, _ = a_provider(
        chunks=("a", "b"),
        final=a_message(text="ab", usage=a_usage(fresh=900, output=40, cache_read=1100)),
    )
    stream = instance.stream_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    assert getattr(stream, "usage", None) is None  # not known yet
    async for _ in stream:
        pass

    assert stream.usage is not None
    assert stream.usage.prompt_tokens == 2000
    assert stream.usage.cached_prompt_tokens == 1100
    assert stream.usage.completion_tokens == 40
    assert stream.usage.model == "anthropic:claude-opus-5"


async def test_a_json_stream_cut_off_mid_object_fails_loudly():
    """Half a JSON object is not a shorter answer, it is a broken one — and by
    now the router is committed, so the only honest move is to raise."""
    instance, _ = a_provider(
        chunks=('{"ans',), final=a_message(text='{"ans', stop_reason="max_tokens")
    )

    with pytest.raises(LLMError) as caught:
        async for _ in instance.stream_json("claude-opus-5", "sys", "user", SCHEMA, 1200):
            pass

    assert caught.value.error_class == ErrorClass.INVALID


async def test_prose_cut_off_at_the_limit_is_not_fatal():
    """A long answer that ran out of room is still an answer worth showing."""
    instance, _ = a_provider(
        chunks=("a long ", "answer that ran out"),
        final=a_message(text="a long answer that ran out", stop_reason="max_tokens"),
    )
    pieces = [p async for p in instance.stream_text("claude-opus-5", "sys", "user")]
    assert pieces == ["a long ", "answer that ran out"]


async def test_a_refusal_at_the_end_of_a_stream_is_still_a_refusal():
    instance, _ = a_provider(
        chunks=("partial",),
        final=a_message(
            text="partial",
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="harmful_content", explanation=None),
        ),
    )

    with pytest.raises(LLMError) as caught:
        async for _ in instance.stream_json("claude-opus-5", "sys", "user", SCHEMA, 1200):
            pass

    assert caught.value.error_class == ErrorClass.CONTENT_FILTERED


# ---------------------------------------------------------------------------
# Reading the response
# ---------------------------------------------------------------------------


async def test_a_truncated_answer_comes_back_as_length():
    """router.complete_json turns "length" into a hard failure. Any other
    spelling and a half-written plan gets handed on as though it were whole."""
    instance, _ = a_provider(response=a_message(text='{"ans', stop_reason="max_tokens"))
    result = await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    assert result.finish_reason == "length"
    assert result.parsed is None


async def test_the_json_is_parsed_once_here_not_twice():
    instance, _ = a_provider(response=a_message(text='{"answer": "yes", "steps": []}'))
    result = await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    assert result.parsed == {"answer": "yes", "steps": []}
    assert result.finish_reason == "stop"


async def test_cached_tokens_are_counted_inside_the_prompt_total_not_beside_it():
    """Anthropic reports input_tokens *excluding* the cache; our Usage says
    cached_prompt_tokens is a subset of prompt_tokens. So the three add up."""
    instance, _ = a_provider(
        response=a_message(usage=a_usage(fresh=200, output=60, cache_read=1800, cache_write=100))
    )
    result = await instance.complete_json("claude-opus-5", "sys", "user", SCHEMA, 1200)

    usage = result.usage
    assert usage.prompt_tokens == 2100
    assert usage.cached_prompt_tokens == 1800
    assert usage.completion_tokens == 60
    assert usage.provider == "anthropic"
    assert usage.usd > 0

    # A cache read costs a tenth, a cache write costs a premium. Both matter,
    # and neither is what a flat discount would have produced.
    assert usage.usd == pytest.approx(price("claude-opus-5", 200, 60, 1800, 100))


async def test_an_unlisted_model_costs_zero_rather_than_a_wrong_number():
    instance, _ = a_provider(response=a_message())
    result = await instance.complete_json("claude-something-unreleased", "s", "u", SCHEMA, 1200)
    assert result.usage.usd == 0.0


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------


async def _enumerate(stream: Any) -> Any:
    """`enumerate` for an async iterator, which the builtin does not do."""
    index = 0
    async for item in stream:
        yield index, item
        index += 1
