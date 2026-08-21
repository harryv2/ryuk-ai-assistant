"""The router: who gets asked, and what happens when they fail.

No network, no SDK, no API key. Every provider here is a small object that
implements `app.llm.base.Provider` and answers from a script — which is the
whole claim the package makes, so a fake that satisfies the protocol is not a
shortcut, it is the test.

What is pinned down:

* the primary answers when it is healthy, and nobody else is called;
* a rate-limited primary hands over to the fallback, and the answer comes from
  the fallback;
* an `INVALID` failure does not fall back — a request the first model could not
  read is a request the second one cannot read either, and trying twice buys a
  second rejection and a second bill;
* a provider whose circuit breaker is open is skipped without a request;
* both attempts land in the usage ledger, so a fallback firing costs something
  visible rather than nothing;
* `ModelRef.parse` on a prefixed name, a bare name and a malformed one;
* a stream that breaks **after** the first token raises instead of quietly
  finishing on another provider.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from app.llm import base, router
from app.llm import usage as usage_tracker
from app.llm.base import ChatResult, ModelRef, Provider, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError

PRIMARY = "fakeone:model-a"
SECONDARY = "faketwo:model-b"


# --------------------------------------------------------------------------
# A provider that answers from a script
# --------------------------------------------------------------------------


class FakeSDKError(Exception):
    """What a vendor SDK raises, as far as this test is concerned.

    The router never sees the class; it sees whatever `classify_error` says
    about it. That indirection is the point of the layer, so the fakes go
    through it rather than raising `LLMError` directly.
    """

    def __init__(self, error_class: ErrorClass, message: str = "boom") -> None:
        self.error_class = error_class
        super().__init__(message)


class FakeProvider:
    """One vendor. Counts its calls and fails exactly when told to.

    `fail_with` is the class of failure this provider raises instead of
    answering. `fail_after_chunks` breaks a stream part way through, which is
    the one case where falling back is forbidden.
    """

    def __init__(
        self,
        name: str,
        *,
        answer: dict[str, Any] | None = None,
        text: str = "hello from ",
        fail_with: ErrorClass | None = None,
        fail_after_chunks: int | None = None,
        usage_on_failure: Usage | None = None,
        prompt_tokens: int = 120,
        completion_tokens: int = 30,
        caching: bool = True,
    ) -> None:
        self.name = name
        self.answer = answer if answer is not None else {"said": name}
        self.text = text
        self.fail_with = fail_with
        self.fail_after_chunks = fail_after_chunks
        self.usage_on_failure = usage_on_failure
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.caching = caching

        self.calls: list[str] = []
        self.embed_calls: list[list[str]] = []

    # -- helpers -----------------------------------------------------------

    def _usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            model=self.name,
            provider=self.name,
            usd=0.0,
        )

    def _maybe_fail(self) -> None:
        if self.fail_with is None:
            return
        if self.usage_on_failure is not None:
            raise LLMError(
                self.fail_with,
                f"{self.name} refused",
                provider=self.name,
                usage=self.usage_on_failure,
            )
        raise FakeSDKError(self.fail_with, f"{self.name} refused")

    # -- the protocol ------------------------------------------------------

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> ChatResult:
        self.calls.append(f"complete_json:{model}")
        self._maybe_fail()
        return ChatResult(
            text="",
            parsed={**self.answer, "model": model},
            usage=self._usage(),
            finish_reason="stop",
        )

    def stream_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self.calls.append(f"stream_json:{model}")
        return self._stream(['{"said":"', self.name, '"}'])

    def stream_text(self, model: str, system: str, user: str) -> AsyncIterator[str]:
        self.calls.append(f"stream_text:{model}")
        return self._stream([self.text, self.name, "."])

    def _stream(self, pieces: Sequence[str]) -> TextStream:
        out = TextStream()
        provider = self

        async def run() -> AsyncIterator[str]:
            if provider.fail_after_chunks is None:
                provider._maybe_fail()
            for index, piece in enumerate(pieces):
                if provider.fail_after_chunks is not None and index >= provider.fail_after_chunks:
                    provider._maybe_fail()
                yield piece
            out.usage = provider._usage()

        return out.feed(run())

    async def embed(
        self,
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[list[list[float]], Usage]:
        self.embed_calls.append(list(texts))
        self._maybe_fail()
        return [[0.5] * dimensions for _ in texts], self._usage()

    def classify_error(self, exc: Exception) -> str:
        if isinstance(exc, FakeSDKError):
            return str(exc.error_class)
        return str(ErrorClass.UNKNOWN)

    def supports_prefix_caching(self) -> bool:
        return self.caching


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_redis(monkeypatch: pytest.MonkeyPatch):
    """No Redis in a unit test, so the breaker uses its in-process mirror.

    `router.breaker_open_for` falls back to `_local_breakers` when Redis is
    unreachable, which is deliberate: failing open on a circuit breaker would
    mean every worker hammering a dead provider at once. Here it just makes the
    breaker deterministic and free.
    """
    from app.core import cache
    from redis.exceptions import ConnectionError as RedisConnectionError

    async def unreachable() -> Any:
        raise RedisConnectionError("no redis in a unit test")

    monkeypatch.setattr(cache, "get_redis", unreachable)
    router._local_breakers.clear()
    yield
    router._local_breakers.clear()


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch: pytest.MonkeyPatch):
    """The one retry the router does is a real `asyncio.sleep`. Not here."""
    monkeypatch.setattr(router, "backoff", lambda *a, **k: 0.0)


@pytest.fixture
def llm_settings():
    """Write settings straight into the singleton, and put them back after.

    `settings.__dict__` rather than `setattr` for the reason `crypto_keys` in
    conftest gives: several of these are read with `getattr(..., default)` and
    a plain assignment on a Pydantic model refuses a name it does not declare.

    It also teaches `ModelRef.parse` the two fake vendor names, because a model
    string is only resolvable if the parser knows its prefix — settings that
    name a provider nothing has heard of would fail before any of this got a
    chance to be tested.
    """
    from app.config import settings

    saved_aliases = dict(base.PROVIDER_ALIASES)
    for name in ("fakeone", "faketwo"):
        base.add_provider_name(name)

    originals: dict[str, tuple[bool, Any]] = {}

    def put(**values: Any) -> None:
        for name, value in values.items():
            if name not in originals:
                originals[name] = (name in settings.__dict__, settings.__dict__.get(name))
            settings.__dict__[name] = value

    put(
        LLM_MODEL=PRIMARY,
        LLM_FALLBACK_MODELS=SECONDARY,
        LLM_PROSE_MODEL="",
        LLM_DEFAULT_PROVIDER="openai",
        LLM_FALLBACK_ON="RATE_LIMITED,TRANSIENT,QUOTA_EXHAUSTED,TIMEOUT",
        LLM_TIMEOUT_S=5.0,
        EMBED_MODEL="fakeone:embed-a",
        EMBED_DIMENSIONS=8,
    )
    yield put

    for name, (existed, old) in originals.items():
        if existed:
            settings.__dict__[name] = old
        else:
            settings.__dict__.pop(name, None)
    base.PROVIDER_ALIASES.clear()
    base.PROVIDER_ALIASES.update(saved_aliases)


@pytest.fixture
def providers(llm_settings):
    """Two registered fake vendors, cleaned out of every registry afterwards.

    `router.register` also teaches `ModelRef.parse` the name — one registry, not
    two — so the alias table has to be put back as well.
    """
    saved_overrides = dict(router._overrides)
    saved_instances = dict(router.PROVIDERS)
    saved_aliases = dict(base.PROVIDER_ALIASES)

    made: dict[str, FakeProvider] = {}

    def make(name: str, **kwargs: Any) -> FakeProvider:
        provider = FakeProvider(name, **kwargs)
        router.register(name, lambda p=provider: p)
        made[name] = provider
        return provider

    yield make

    router._overrides.clear()
    router._overrides.update(saved_overrides)
    router.PROVIDERS.clear()
    router.PROVIDERS.update(saved_instances)
    base.PROVIDER_ALIASES.clear()
    base.PROVIDER_ALIASES.update(saved_aliases)


@pytest.fixture
def pair(providers):
    """The usual arrangement: a healthy primary and a healthy fallback."""
    return providers("fakeone"), providers("faketwo")


async def collect(stream: AsyncIterator[str]) -> list[str]:
    return [piece async for piece in stream]


# --------------------------------------------------------------------------
# The chain, when nothing is wrong
# --------------------------------------------------------------------------


class TestPrimaryFirst:
    def test_the_fakes_satisfy_the_protocol(self, pair):
        """If this ever fails, every other test in the file is theatre."""
        first, second = pair
        assert isinstance(first, Provider)
        assert isinstance(second, Provider)

    def test_resolve_puts_the_primary_first(self, llm_settings):
        chain = router.resolve()
        assert [str(ref) for ref in chain] == [PRIMARY, SECONDARY]

    def test_resolve_drops_a_duplicate_fallback(self, llm_settings):
        llm_settings(LLM_FALLBACK_MODELS=f"{SECONDARY},{PRIMARY},{SECONDARY}")
        assert [str(ref) for ref in router.resolve()] == [PRIMARY, SECONDARY]

    async def test_healthy_primary_answers_and_nobody_else_is_called(self, pair):
        first, second = pair

        answer = await router.complete_json("sys", "user")

        assert answer["said"] == "fakeone"
        assert answer["model"] == "model-a"  # the bare name, not the reference
        assert first.calls == ["complete_json:model-a"]
        assert second.calls == []

    async def test_the_answer_is_streamed_from_the_primary(self, pair):
        first, second = pair

        pieces = await collect(router.stream_text("sys", "user"))

        assert "".join(pieces) == "hello from fakeone."
        assert second.calls == []


# --------------------------------------------------------------------------
# The chain, when the primary fails
# --------------------------------------------------------------------------


class TestFallback:
    async def test_rate_limited_primary_hands_over_to_the_fallback(self, providers):
        first = providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        second = providers("faketwo")

        answer = await router.complete_json("sys", "user")

        assert answer["said"] == "faketwo"
        assert first.calls == ["complete_json:model-a"]
        assert second.calls == ["complete_json:model-b"]

    @pytest.mark.parametrize(
        "error_class",
        [ErrorClass.RATE_LIMITED, ErrorClass.QUOTA_EXHAUSTED, ErrorClass.TIMEOUT],
    )
    async def test_every_class_in_the_fallback_set_falls_back(self, providers, error_class):
        providers("fakeone", fail_with=error_class)
        second = providers("faketwo")

        answer = await router.complete_json("sys", "user")

        assert answer["said"] == "faketwo"
        assert second.calls == ["complete_json:model-b"]

    async def test_a_transient_primary_is_retried_once_before_moving_on(self, providers):
        """TRANSIENT is the one class worth asking the same provider twice."""
        first = providers("fakeone", fail_with=ErrorClass.TRANSIENT)
        second = providers("faketwo")

        answer = await router.complete_json("sys", "user")

        assert first.calls == ["complete_json:model-a", "complete_json:model-a"]
        assert answer["said"] == "faketwo"
        assert second.calls == ["complete_json:model-b"]

    async def test_invalid_does_not_fall_back_and_raises_immediately(self, providers):
        first = providers("fakeone", fail_with=ErrorClass.INVALID)
        second = providers("faketwo")

        with pytest.raises(LLMError) as caught:
            await router.complete_json("sys", "user")

        assert caught.value.error_class == ErrorClass.INVALID
        assert caught.value.model == PRIMARY
        assert first.calls == ["complete_json:model-a"]  # tried once, not twice
        assert second.calls == []  # never asked

    @pytest.mark.parametrize(
        "error_class",
        [ErrorClass.CONTENT_FILTERED, ErrorClass.AUTH_EXPIRED, ErrorClass.AUTH_REVOKED],
    )
    async def test_the_other_terminal_classes_do_not_fall_back_either(self, providers, error_class):
        providers("fakeone", fail_with=error_class)
        second = providers("faketwo")

        with pytest.raises(LLMError) as caught:
            await router.complete_json("sys", "user")

        assert caught.value.error_class == error_class
        assert second.calls == []

    async def test_the_last_failure_is_raised_when_the_whole_chain_fails(self, providers):
        providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        second = providers("faketwo", fail_with=ErrorClass.QUOTA_EXHAUSTED)

        with pytest.raises(LLMError) as caught:
            await router.complete_json("sys", "user")

        assert caught.value.error_class == ErrorClass.QUOTA_EXHAUSTED
        assert caught.value.model == SECONDARY
        assert second.calls == ["complete_json:model-b"]


# --------------------------------------------------------------------------
# The circuit breaker
# --------------------------------------------------------------------------


class TestBreaker:
    async def test_an_open_breaker_skips_the_provider_without_calling_it(self, pair):
        first, second = pair
        router._local_breakers["fakeone"] = router._LocalBreaker(
            failures=router.BREAKER_THRESHOLD,
            open_until=time.time() + router.BREAKER_COOLDOWN_S,
        )

        answer = await router.complete_json("sys", "user")

        assert answer["said"] == "faketwo"
        assert first.calls == []  # not a failed call: no call at all
        assert second.calls == ["complete_json:model-b"]

    async def test_it_opens_after_five_consecutive_failures(self, providers):
        first = providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        providers("faketwo")

        for _ in range(router.BREAKER_THRESHOLD):
            await router.complete_json("sys", "user")

        assert len(first.calls) == router.BREAKER_THRESHOLD
        assert await router.breaker_open_for("fakeone") is not None

        await router.complete_json("sys", "user")
        assert len(first.calls) == router.BREAKER_THRESHOLD  # skipped, not retried

    async def test_a_success_closes_it(self, pair):
        first, _ = pair
        router._local_breakers["fakeone"] = router._LocalBreaker(failures=3)

        await router.complete_json("sys", "user")

        assert "fakeone" not in router._local_breakers
        assert first.calls == ["complete_json:model-a"]

    async def test_an_invalid_request_does_not_count_against_the_provider(self, providers):
        """Our request was the problem, not the vendor. Opening the breaker
        there would hide a real outage behind a schema bug."""
        providers("fakeone", fail_with=ErrorClass.INVALID)
        providers("faketwo")

        for _ in range(router.BREAKER_THRESHOLD + 2):
            with pytest.raises(LLMError):
                await router.complete_json("sys", "user")

        assert await router.breaker_open_for("fakeone") is None

    async def test_every_provider_skipped_is_a_failure_not_a_silence(self, pair):
        for name in ("fakeone", "faketwo"):
            router._local_breakers[name] = router._LocalBreaker(
                failures=router.BREAKER_THRESHOLD,
                open_until=time.time() + router.BREAKER_COOLDOWN_S,
            )

        with pytest.raises(LLMError) as caught:
            await router.complete_json("sys", "user")

        assert caught.value.error_class == ErrorClass.TRANSIENT
        assert caught.value.details["skipped"] == [PRIMARY, SECONDARY]


# --------------------------------------------------------------------------
# What it cost
# --------------------------------------------------------------------------


class TestUsage:
    async def test_both_attempts_are_recorded_when_a_fallback_fires(self, providers):
        """A fallback that looks free is a fallback nobody notices firing."""
        burned = Usage(
            prompt_tokens=900,
            completion_tokens=0,
            model=PRIMARY,
            provider="fakeone",
            usd=0.0,
        )
        providers("fakeone", fail_with=ErrorClass.RATE_LIMITED, usage_on_failure=burned)
        providers("faketwo", prompt_tokens=900, completion_tokens=40)

        async with usage_tracker.track_usage() as ledger:
            answer = await router.complete_json("sys", "user")

        assert answer["said"] == "faketwo"
        assert ledger.calls == 2

        failed, succeeded = ledger.entries
        assert failed.ok is False
        assert failed.usage.model == PRIMARY
        assert failed.usage.prompt_tokens == 900  # the rate limit still read the prompt
        assert succeeded.ok is True
        assert succeeded.usage.model == SECONDARY

        totals = ledger.totals()
        assert totals["calls"] == 2
        assert totals["prompt"] == 1800  # both attempts, not just the one that worked
        assert totals["model"] == SECONDARY  # who actually answered

    async def test_a_failure_that_burned_nothing_still_counts_as_a_call(self, providers):
        providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        providers("faketwo")

        async with usage_tracker.track_usage() as ledger:
            await router.complete_json("sys", "user")

        assert ledger.calls == 2
        assert ledger.entries[0].usage.total_tokens == 0
        assert ledger.entries[0].ok is False

    async def test_usage_always_names_the_model_in_the_canonical_spelling(self, pair):
        async with usage_tracker.track_usage() as ledger:
            await router.complete_json("sys", "user")

        assert ledger.entries[0].usage.model == PRIMARY
        assert ledger.entries[0].usage.provider == "fakeone"

    async def test_a_stream_is_counted_too(self, pair):
        async with usage_tracker.track_usage() as ledger:
            await collect(router.stream_text("sys", "user"))

        assert ledger.calls == 1
        assert ledger.entries[0].usage.model == PRIMARY


# --------------------------------------------------------------------------
# Parsing a model string
# --------------------------------------------------------------------------


class TestModelRef:
    def test_a_prefixed_string_names_the_provider(self):
        ref = ModelRef.parse("gemini:gemini-2.5-flash")
        assert ref.provider == "gemini"
        assert ref.name == "gemini-2.5-flash"
        assert str(ref) == "gemini:gemini-2.5-flash"

    def test_a_model_name_may_contain_colons(self):
        ref = ModelRef.parse("openai:ft:gpt-4.1-mini:acme")
        assert ref.provider == "openai"
        assert ref.name == "ft:gpt-4.1-mini:acme"

    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("google:x", "gemini"),
            ("oai:x", "openai"),
            ("GEMINI:x", "gemini"),
            ("claude:x", "anthropic"),
        ],
    )
    def test_the_spellings_people_reach_for_all_work(self, written, expected):
        assert ModelRef.parse(written).provider == expected

    def test_a_bare_name_belongs_to_the_default_provider(self):
        ref = ModelRef.parse("gpt-4.1-mini")
        assert ref.provider == "openai"
        assert ref.name == "gpt-4.1-mini"

        ref = ModelRef.parse("gemini-2.5-flash", "gemini")
        assert ref.provider == "gemini"
        assert ref.name == "gemini-2.5-flash"

    def test_a_bare_name_follows_the_configured_default(self, llm_settings):
        llm_settings(LLM_DEFAULT_PROVIDER="gemini", LLM_MODEL="gemini-2.5-flash")
        assert str(router.resolve()[0]) == "gemini:gemini-2.5-flash"

    @pytest.mark.parametrize(
        ("bad", "why"),
        [
            ("nosuchvendor:some-model", "no provider called"),
            ("openai:", "names a provider but no model"),
            (":gpt-4.1-mini", "no provider called"),
            ("", "No model was given"),
            ("   ", "No model was given"),
        ],
    )
    def test_a_malformed_string_is_an_error_not_a_guess(self, bad, why):
        """An unknown prefix quietly becoming an OpenAI call is the kind of
        thing you find out about in a bill."""
        with pytest.raises(LLMError) as caught:
            ModelRef.parse(bad)

        assert caught.value.error_class == ErrorClass.INVALID
        assert why.lower() in caught.value.message.lower()

    def test_an_unknown_default_provider_is_an_error_too(self):
        with pytest.raises(LLMError):
            ModelRef.parse("some-model", "nosuchvendor")

    def test_the_router_says_which_providers_it_has(self, llm_settings):
        llm_settings(LLM_MODEL="nosuchvendor:model")
        with pytest.raises(LLMError) as caught:
            router.resolve()

        assert "This build serves" in caught.value.message
        assert "openai" in caught.value.message

    def test_registering_a_provider_teaches_the_parser_its_name(self, providers):
        providers("brandnew")
        assert ModelRef.parse("brandnew:some-model").provider == "brandnew"


# --------------------------------------------------------------------------
# Streaming commits at the first token
# --------------------------------------------------------------------------


class TestStreamingCommits:
    async def test_a_break_before_the_first_token_may_still_fall_back(self, providers):
        first = providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        second = providers("faketwo")

        pieces = await collect(router.stream_text("sys", "user"))

        assert "".join(pieces) == "hello from faketwo."
        assert first.calls == ["stream_text:model-a"]
        assert second.calls == ["stream_text:model-b"]

    async def test_a_break_after_the_first_token_raises_instead_of_switching(self, providers):
        """Two halves of two answers spliced into one chat bubble is worse than
        one honest failure, so the fallback closes the moment the caller has
        seen a chunk — even for a class that would otherwise fall back."""
        first = providers(
            "fakeone",
            fail_with=ErrorClass.RATE_LIMITED,
            fail_after_chunks=1,
        )
        second = providers("faketwo")

        seen: list[str] = []
        with pytest.raises(LLMError) as caught:
            async for piece in router.stream_text("sys", "user"):
                seen.append(piece)

        assert seen == ["hello from "]  # what the user already saw
        assert caught.value.error_class == ErrorClass.RATE_LIMITED
        assert caught.value.details["committed"] is True
        assert first.calls == ["stream_text:model-a"]
        assert second.calls == []  # never asked to finish somebody else's sentence

    async def test_the_same_rule_holds_for_a_streamed_plan(self, providers):
        first = providers("fakeone", fail_with=ErrorClass.TRANSIENT, fail_after_chunks=2)
        second = providers("faketwo")

        seen: list[str] = []
        with pytest.raises(LLMError):
            async for piece in router.stream_json("sys", "user"):
                seen.append(piece)

        assert seen == ['{"said":"', "fakeone"]
        assert len(first.calls) == 1  # not even the TRANSIENT retry
        assert second.calls == []

    async def test_a_committed_stream_still_files_what_it_cost(self, providers):
        providers("fakeone", fail_with=ErrorClass.RATE_LIMITED, fail_after_chunks=1)
        providers("faketwo")

        async with usage_tracker.track_usage() as ledger:
            with pytest.raises(LLMError):
                await collect(router.stream_text("sys", "user"))

        assert ledger.calls == 1
        assert ledger.entries[0].ok is False
        assert ledger.entries[0].usage.model == PRIMARY


# --------------------------------------------------------------------------
# Embeddings, which are not like the others
# --------------------------------------------------------------------------


class TestEmbeddingsNeverFallBack:
    async def test_the_embedding_chain_is_one_model_long(self, llm_settings):
        chain = router.resolve(purpose=router.PURPOSE_EMBED)
        assert [str(ref) for ref in chain] == ["fakeone:embed-a"]

    async def test_a_failed_embedding_is_a_failure_not_a_different_model(self, providers):
        """A fallback here would write vectors nothing can be compared with."""
        first = providers("fakeone", fail_with=ErrorClass.RATE_LIMITED)
        second = providers("faketwo")

        with pytest.raises(LLMError) as caught:
            await router.embed(["some text"], use_cache=False)

        assert caught.value.error_class == ErrorClass.RATE_LIMITED
        assert first.embed_calls == [["some text"]]
        assert second.embed_calls == []

    async def test_empty_strings_get_a_zero_vector_instead_of_a_request(self, providers):
        first = providers("fakeone")

        vectors = await router.embed(["", "  ", "real"], use_cache=False)

        assert first.embed_calls == [["real"]]
        assert vectors[0] == [0.0] * 8
        assert vectors[1] == [0.0] * 8
        assert vectors[2] == [0.5] * 8

    async def test_a_wrong_width_vector_never_reaches_pgvector(self, providers, llm_settings):
        providers("fakeone")
        llm_settings(EMBED_DIMENSIONS=1536)  # the provider will answer with 8

        class WrongWidth(FakeProvider):
            async def embed(self, model, texts, dimensions):
                return [[0.5] * 8 for _ in texts], self._usage()

        router.register("fakeone", lambda: WrongWidth("fakeone"))

        with pytest.raises(LLMError) as caught:
            await router.embed(["text"], use_cache=False)

        assert caught.value.error_class == ErrorClass.INVALID
        assert caught.value.details == {"expected": 1536, "got": 8}


class TestEmbedModelContract:
    def test_the_id_is_the_string_that_goes_in_the_column(self, llm_settings):
        assert router.embed_model_id() == "fakeone:embed-a"
        assert len(router.embed_model_id()) <= 64  # VARCHAR(64)

    def test_matching_rows_are_allowed_through(self, llm_settings):
        router.assert_same_embed_model("fakeone:embed-a")

    def test_rows_from_another_model_are_refused_loudly(self, llm_settings):
        with pytest.raises(LLMError) as caught:
            router.assert_same_embed_model("openai:text-embedding-3-large")

        assert caught.value.error_class == ErrorClass.INVALID
        assert "not comparable" in caught.value.message
        assert "Re-embed" in caught.value.message

    def test_a_row_that_never_recorded_a_model_is_refused_too(self, llm_settings):
        with pytest.raises(LLMError) as caught:
            router.assert_same_embed_model("")

        assert "unrecorded model" in caught.value.message
