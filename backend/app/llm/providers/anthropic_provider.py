"""The Anthropic adapter: our four calls, spoken to `client.messages`.

Third provider behind `app.llm.base.Provider`, alongside OpenAI and Gemini. It
knows how to phrase a request for Anthropic and how to name Anthropic's failures
in our vocabulary. It knows nothing about fallback chains, breakers, retries or
ledgers — those live in `app.llm.router` and `app.llm.usage`, once, for all
three.

What is different here
----------------------

**Structured output is not tool use.** It used to be that the way to make Claude
return JSON was to hand it a single tool and force a call. That is over. The
schema goes in ``output_config.format`` and the first text block of the reply is
JSON that matches it. See `app.llm.anthropic_schema` for the translation, which
is nearly a copy — Anthropic wants the same ``required``-everything,
``additionalProperties: false`` shape OpenAI's strict mode does.

**Streaming is plain text.** Because ``output_config.format`` makes the answer a
text block that happens to hold JSON, there is no partial-JSON event type to
reassemble. We hand the caller raw chunks as they land and let it parse
incrementally, which is the whole point of `router.stream_json`.

**Caching is asked for, not automatic.** OpenAI caches a long prefix on its own.
Anthropic caches only where a ``cache_control`` breakpoint says to, the prefix
must be about 1024 tokens before anything is stored, and the render order is
tools, then system, then messages. Our system prefix is deliberately long, so the
breakpoint goes at the end of it and the user's question — the part that changes
every call — falls after it. Check `usage.cache_read_input_tokens`: a persistent
zero means something in the prefix is moving between calls.

**No embeddings.** Anthropic does not have an embeddings endpoint, and there is
no polite way to fake one. `embed` raises and says what to do instead.

Three parameters that are now 400s
----------------------------------

``budget_tokens``, ``temperature``, ``top_p`` and ``top_k`` were all removed on
claude-opus-5, claude-sonnet-5 and claude-fable-5. Nothing here sends them, and
nothing here should start. Thinking depth is ``thinking`` plus
``output_config.effort``; there is no temperature to turn down.

Configuration
-------------

Read through `_setting`, which checks `app.config.settings` first and the
environment second, so this file keeps working whether or not the names below
have been added to `Settings` (today they have not, and `Settings` ignores extra
fields, so the environment is what answers).

    ANTHROPIC_API_KEY          not required if `ant auth login` has been run
    ANTHROPIC_BASE_URL         for a proxy or a gateway
    ANTHROPIC_TIMEOUT_S        seconds; the router's own deadline is shorter
    ANTHROPIC_EFFORT           low | medium | high | xhigh | max   (default medium)
    ANTHROPIC_THINKING         off | adaptive | default            (default off)
    ANTHROPIC_CACHE            system | auto | off                 (default system)
    ANTHROPIC_MAX_TOKENS       floor for a non-streamed answer     (default 16000)
    ANTHROPIC_STREAM_MAX_TOKENS  floor for a streamed answer       (default 64000)
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any, Final

import anthropic

from app.core.logging import get_logger
from app.llm.anthropic_schema import to_anthropic_schema
from app.llm.base import ChatResult, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError, class_for_status

log = get_logger(__name__)

PROVIDER_NAME: Final[str] = "anthropic"

# The model when nobody names one. Everything else in the app spells models
# "anthropic:claude-opus-5"; by the time we are called the prefix is gone.
DEFAULT_MODEL: Final[str] = "claude-opus-5"

# USD per million tokens, (input, output). A rough cost signal, not billing —
# same rule as `app.llm.usage`: a model that is not here costs zero, which shows
# up as a suspicious zero rather than a confidently wrong number.
PRICES: Final[dict[str, tuple[float, float]]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

# What a cached token costs against the input rate. Reading from the cache is
# cheap; writing to it costs a premium once. `app.llm.usage` assumes a flat 0.25
# for everyone, which is why the dollar figure is worked out here instead and
# handed over already filled in — the ledger only prices a call we left at zero.
CACHE_READ_RATE: Final[float] = 0.10
CACHE_WRITE_RATE: Final[float] = 1.25

EFFORTS: Final[frozenset[str]] = frozenset({"low", "medium", "high", "xhigh", "max"})

# Sending {"type": "disabled"} alongside either of these is a 400.
_EFFORTS_REQUIRING_THINKING: Final[frozenset[str]] = frozenset({"xhigh", "max"})

# Anthropic's reasons for stopping, in our words. `router.complete_json` treats
# "length" as a hard failure, so `max_tokens` has to arrive under that name or a
# truncated answer would be handed on as if it were whole.
_FINISH: Final[dict[str, str]] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "pause",
}

# What a 400 says when the account has simply run out of money. This is the one
# case where the status code lies about the cause: it is not a bad request.
_OUT_OF_CREDIT: Final[tuple[str, ...]] = (
    "credit balance",
    "insufficient credit",
    "insufficient_quota",
    "billing",
    "spend limit",
    "purchase credits",
)

_JSON_NUDGE: Final[str] = (
    "\n\nAnswer with a single JSON object and nothing else. No prose, no code fences."
)


# ------------------------------------------------------------------- settings


def _setting(name: str, default: Any = None) -> Any:
    """One knob, from config if it is there and from the environment if not.

    `Settings` is declared with ``extra="ignore"``, so an ``ANTHROPIC_*`` line in
    `.env` does not become an attribute until somebody adds the field to
    `app.config`. Checking both means this provider works either way, and starts
    reading `.env` for free on the day those fields are added.
    """
    try:
        from app.config import settings

        value = getattr(settings, name, None)
        if value is not None and value != "":
            return value
    except Exception:  # config is not importable in a bare unit test
        pass

    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw


def _float_setting(name: str, default: float) -> float:
    try:
        return float(_setting(name, default))
    except (TypeError, ValueError):
        return default


def _int_setting(name: str, default: int) -> int:
    try:
        return int(float(_setting(name, default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- the provider


class AnthropicProvider:
    """Anthropic behind `app.llm.base.Provider`."""

    name: str = PROVIDER_NAME

    def __init__(self, client: Any | None = None) -> None:
        # No client until the first call. Importing this module must not need an
        # API key: `router._build` imports a provider to find out whether it is
        # installed, and a process configured for OpenAI only should not trip
        # over a missing ANTHROPIC_API_KEY on the way past.
        self._client: Any | None = client
        self._lock = threading.Lock()

    # -- the client -----------------------------------------------------------

    def client(self) -> Any:
        """The shared `AsyncAnthropic`, built on first use.

        ``max_retries=0`` on purpose. The SDK will happily retry 408, 409, 429
        and 5xx for us, and every one of those is a decision the router has
        already made differently: it wants a rate limit to move to another
        provider, not to sleep here inside a deadline it is holding.
        """
        existing = self._client
        if existing is not None:
            return existing

        with self._lock:
            if self._client is not None:
                return self._client

            kwargs: dict[str, Any] = {
                "max_retries": 0,
                # Seconds in Python. Kept well above the router's per-attempt
                # budget so the router's clock is the one that matters, and set
                # explicitly so the SDK does not refuse a large `max_tokens`
                # request on the grounds that it might outlast a default.
                "timeout": _float_setting("ANTHROPIC_TIMEOUT_S", 120.0),
            }
            api_key = _setting("ANTHROPIC_API_KEY")
            if api_key:
                kwargs["api_key"] = str(api_key)
            base_url = _setting("ANTHROPIC_BASE_URL")
            if base_url:
                kwargs["base_url"] = str(base_url)

            try:
                self._client = anthropic.AsyncAnthropic(**kwargs)
            except Exception as exc:
                # An unset ANTHROPIC_API_KEY is not proof there are no
                # credentials — the SDK also reads ANTHROPIC_AUTH_TOKEN and a
                # signed-in profile. So let it decide, and translate whatever it
                # says into something an operator can act on.
                raise LLMError(
                    ErrorClass.AUTH_EXPIRED,
                    "The Anthropic client could not be built. Set ANTHROPIC_API_KEY.",
                    provider=PROVIDER_NAME,
                    cause=exc,
                    details={"detail": str(exc)[:200]},
                ) from exc
            return self._client

    async def aclose(self) -> None:
        """Shut the HTTP client down. Called from `router.close`."""
        client, self._client = self._client, None
        if client is None:
            return
        closer = getattr(client, "close", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result

    # -- one request, one JSON object -----------------------------------------

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> ChatResult:
        """One call, one JSON object, already parsed.

        The schema goes in ``output_config.format`` and the first text block is
        guaranteed to match it, so `ChatResult.parsed` is filled in and the
        router does not parse the same text twice.
        """
        name = model or DEFAULT_MODEL
        request = self._request(
            name,
            system,
            user,
            max_tokens=self._max_tokens(max_tokens, streaming=False),
            schema=schema,
        )

        try:
            response = await self.client().messages.create(**request)
        except Exception as exc:
            self._log_failure("complete_json", name, exc)
            raise

        self._refuse_if_refused(response, name, "complete_json")

        usage = self._usage(response, name)
        text = self._first_text(response)
        finish = _FINISH.get(str(response.stop_reason or ""), str(response.stop_reason or ""))

        parsed: dict[str, Any] | None = None
        if schema is not None and finish != "length" and text:
            # With a schema in force the text is JSON. Parse it here so the
            # router can skip its repair pass; if the model surprised us, hand
            # the text over and let `parse_json_object` do its work.
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict):
                parsed = candidate

        return ChatResult(text=text, parsed=parsed, usage=usage, finish_reason=finish)

    # -- the same JSON, in pieces ---------------------------------------------

    def stream_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Raw chunks of the JSON answer, never buffered to completion.

        Buffering here would undo the reason `router.stream_json` exists: the
        orchestrator starts step one while step four is still being written.
        """
        name = model or DEFAULT_MODEL
        request = self._request(
            name,
            system,
            user,
            max_tokens=self._max_tokens(max_tokens, streaming=True),
            schema=schema,
        )
        return self._open_stream(name, request, what="stream_json", truncation_is_fatal=True)

    def stream_text(self, model: str, system: str, user: str) -> AsyncIterator[str]:
        """Prose, chunk by chunk. No schema, so no ``output_config.format``."""
        name = model or DEFAULT_MODEL
        request = self._request(
            name,
            system,
            user,
            max_tokens=self._max_tokens(0, streaming=True),
            schema=None,
            want_json=False,
        )
        return self._open_stream(name, request, what="stream_text", truncation_is_fatal=False)

    def _open_stream(
        self,
        model: str,
        request: dict[str, Any],
        *,
        what: str,
        truncation_is_fatal: bool,
    ) -> AsyncIterator[str]:
        """Both streaming calls, which differ only in what a cut-off answer means.

        A `TextStream` rather than a bare async generator, because the token
        counts do not exist until after the last chunk and a generator object has
        nowhere to put them.
        """
        out = TextStream()

        async def run() -> AsyncIterator[str]:
            client = self.client()
            try:
                async with client.messages.stream(**request) as stream:
                    async for chunk in stream.text_stream:
                        yield chunk
                    final = await _maybe_await(stream.get_final_message())
            except LLMError:
                raise
            except Exception as exc:
                self._log_failure(what, model, exc)
                raise

            self._refuse_if_refused(final, model, what)
            out.usage = self._usage(final, model)

            if str(final.stop_reason or "") == "max_tokens":
                if truncation_is_fatal:
                    # Half a JSON object is not a smaller answer, it is a broken
                    # one. The router is already committed by now, so this
                    # surfaces as a visible failure rather than a quiet splice.
                    raise LLMError(
                        ErrorClass.INVALID,
                        "The model ran out of room before finishing its answer.",
                        provider=PROVIDER_NAME,
                        model=model,
                        details={"stage": what, "max_tokens": request.get("max_tokens")},
                        usage=out.usage,
                    )
                log.warning(
                    "anthropic.truncated",
                    stage=what,
                    model=model,
                    max_tokens=request.get("max_tokens"),
                )

        return out.feed(run())

    # -- embeddings, which do not exist ---------------------------------------

    async def embed(
        self,
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[list[list[float]], Usage]:
        """Always raises. Anthropic has no embeddings endpoint.

        Not an oversight and not something a newer SDK will add for us. The one
        thing this must never do is return something vector-shaped: the mirror's
        `VECTOR(1536)` columns were filled by one model, and a vector from
        anywhere else scores against them as confident nonsense.
        """
        raise LLMError(
            ErrorClass.INVALID,
            (
                "Anthropic has no embeddings API, so it cannot serve EMBED_MODEL. "
                "Point EMBED_MODEL at an openai: or gemini: model — for example "
                "openai:text-embedding-3-small. Chat can still run on Anthropic; "
                "only embeddings cannot. Changing EMBED_MODEL means re-embedding "
                "the whole mirror with scripts/reembed.py, because vectors from "
                "two different models are not comparable."
            ),
            provider=PROVIDER_NAME,
            model=model,
            details={"asked_for": len(texts), "dimensions": dimensions},
        )

    # -- naming the failures --------------------------------------------------

    def classify_error(self, exc: Exception) -> str:
        """An Anthropic exception as one of our `ErrorClass` names.

        Most specific first, which matters more than it looks: `APITimeoutError`
        is a subclass of `APIConnectionError`, and every 4xx class is a subclass
        of `APIStatusError`, so a broad clause first would swallow the lot.

        Never raises. It runs on the failure path, and an exception here would
        lose the failure it was called to describe.
        """
        try:
            return str(self._classify(exc))
        except Exception:  # a classifier that throws is worse than a vague one
            return str(ErrorClass.UNKNOWN)

    def _classify(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, LLMError):
            return exc.error_class

        if isinstance(exc, anthropic.BadRequestError):
            # A 400 is normally our fault and worth failing on. An empty wallet
            # arrives as one too, and that one wants the fallback provider.
            return ErrorClass.QUOTA_EXHAUSTED if _mentions_credit(exc) else ErrorClass.INVALID
        if isinstance(exc, anthropic.AuthenticationError):
            return ErrorClass.AUTH_REVOKED
        if isinstance(exc, anthropic.PermissionDeniedError):
            return ErrorClass.AUTH_REVOKED
        if isinstance(exc, anthropic.NotFoundError):
            return ErrorClass.NOT_FOUND
        if isinstance(exc, anthropic.RateLimitError):
            self._note_retry_after(exc)
            return ErrorClass.QUOTA_EXHAUSTED if _mentions_credit(exc) else ErrorClass.RATE_LIMITED
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and status >= 500:
                # 529 is "overloaded", which is the busiest kind of temporary.
                return ErrorClass.TRANSIENT
            return class_for_status(status) if isinstance(status, int) else ErrorClass.INVALID

        # Order matters: APITimeoutError inherits from APIConnectionError.
        if isinstance(exc, anthropic.APITimeoutError):
            return ErrorClass.TIMEOUT
        if isinstance(exc, anthropic.APIConnectionError):
            return ErrorClass.TRANSIENT
        if isinstance(exc, TimeoutError):
            return ErrorClass.TIMEOUT
        if isinstance(exc, anthropic.APIError):
            return ErrorClass.UNKNOWN
        return ErrorClass.UNKNOWN

    def _note_retry_after(self, exc: Exception) -> None:
        """Log how long Anthropic wants us to wait, and leave it on the exception.

        `classify_error` can only return a class name, so this is where the hint
        gets recorded. The router does not sleep on a rate limit anyway — it
        moves to the next provider — but the number belongs in the log line that
        explains why.
        """
        seconds: float | None = None
        try:
            headers = getattr(getattr(exc, "response", None), "headers", None)
            raw = headers.get("retry-after") if headers is not None else None
            if raw is not None:
                seconds = float(raw)
        except (TypeError, ValueError, AttributeError):
            seconds = None

        with suppress(Exception):
            exc.retry_after_s = seconds  # type: ignore[attr-defined]
        log.warning(
            "anthropic.rate_limited",
            retry_after_s=seconds,
            request_id=_request_id(exc),
        )

    def supports_prefix_caching(self) -> bool:
        """True, with an asterisk.

        Unlike OpenAI, Anthropic does not cache a long prefix by itself: nothing
        is stored unless a ``cache_control`` breakpoint asks for it, which is
        what `_request` does at the end of the system block. The prefix also has
        to reach roughly 1024 tokens before anything is kept — under that it
        silently does not cache, with no error and no warning.

        The answer is still True, because what the caller does with it is right
        either way: put the fixed catalogue first and the user's question last.
        """
        return True

    # -- building the request -------------------------------------------------

    def _request(
        self,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
        schema: dict[str, Any] | None,
        want_json: bool = True,
    ) -> dict[str, Any]:
        """Everything `messages.create` and `messages.stream` both need.

        Note what is absent: no ``temperature``, no ``top_p``, no ``top_k``, no
        ``budget_tokens``, no assistant prefill. All five were removed on the
        current models and all five now come back as a 400.
        """
        system_text = (system or "").strip()
        if want_json and schema is None:
            # No schema to enforce the shape, so ask in words. A fixed suffix,
            # so the cached prefix stays byte-identical between calls.
            system_text = (system_text + _JSON_NUDGE).strip()

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }

        cache_mode = str(_setting("ANTHROPIC_CACHE", "system")).strip().lower()
        if system_text:
            if cache_mode == "system":
                # The breakpoint goes at the end of the system prompt, so the
                # cached prefix stops exactly where the volatile part starts.
                # Top-level auto-caching would put it after the user's question
                # instead, and a prefix that ends in a different question every
                # time is a prefix that never matches.
                request["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                request["system"] = system_text
        if cache_mode == "auto":
            request["cache_control"] = {"type": "ephemeral"}

        effort = self._effort()
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": to_anthropic_schema(schema),
            }
        request["output_config"] = output_config

        thinking = self._thinking(effort)
        if thinking is not None:
            request["thinking"] = thinking
        return request

    def _effort(self) -> str:
        """How hard to think, from `ANTHROPIC_EFFORT`.

        Defaults to medium rather than Anthropic's high. The planner call is on
        the latency budget and lower effort is the cheap way to buy that back —
        cheaper, and better behaved, than switching thinking off.
        """
        raw = str(_setting("ANTHROPIC_EFFORT", "medium")).strip().lower()
        if raw in EFFORTS:
            return raw
        log.warning("anthropic.bad_effort", value=raw, using="medium", allowed=sorted(EFFORTS))
        return "medium"

    def _thinking(self, effort: str) -> dict[str, str] | None:
        """The ``thinking`` block, or None to leave the parameter off entirely.

        Worth being precise about, because "off" and "not set" are not the same
        thing here. On claude-opus-5 thinking is **on by default**: omitting the
        parameter runs adaptive. Actually turning it off takes an explicit
        ``{"type": "disabled"}``, which is what ``ANTHROPIC_THINKING=off`` sends
        — the default, because the planner call is latency-sensitive.

        Never ``budget_tokens``. That was the old way to bound thinking and it is
        a 400 on every model we can name.
        """
        raw = str(_setting("ANTHROPIC_THINKING", "off")).strip().lower()

        if raw in ("adaptive", "on", "true", "1", "yes"):
            return {"type": "adaptive"}
        if raw in ("default", "auto", "unset"):
            return None
        if raw in ("off", "disabled", "false", "0", "no"):
            if effort in _EFFORTS_REQUIRING_THINKING:
                # Disabled thinking at xhigh or max is a 400. Effort is the more
                # deliberate setting of the two, so it wins.
                log.warning("anthropic.thinking_kept", reason="effort_too_high", effort=effort)
                return None
            return {"type": "disabled"}

        log.warning("anthropic.bad_thinking", value=raw, using="off")
        return {"type": "disabled"} if effort not in _EFFORTS_REQUIRING_THINKING else None

    def _max_tokens(self, asked: int, *, streaming: bool) -> int:
        """The caller's ceiling, raised to a floor that is not a lowball.

        `router.complete_json` defaults to 1200, which was sized for a model that
        answers straight away. These models think first, and an answer cut off
        mid-object comes back as `INVALID`, which does not fall back — so a cap
        that is too low costs the whole call rather than a bit of quality.

        Set the floor to 0 to honour whatever the caller asked for exactly.
        """
        floor = (
            _int_setting("ANTHROPIC_STREAM_MAX_TOKENS", 64000)
            if streaming
            else _int_setting("ANTHROPIC_MAX_TOKENS", 16000)
        )
        return max(int(asked or 0), max(0, floor)) or 16000

    # -- reading the response -------------------------------------------------

    def _refuse_if_refused(self, response: Any, model: str, stage: str) -> None:
        """Stop on a refusal before touching the content.

        A refusal is an HTTP 200 with nothing usable in it, not an exception, so
        nothing else will catch it. ``stop_details`` is null for every other stop
        reason, which is why it is only read once we know the reason is this one.
        """
        if str(getattr(response, "stop_reason", "") or "") != "refusal":
            return

        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)

        raise LLMError(
            ErrorClass.CONTENT_FILTERED,
            "The model declined to answer that.",
            provider=PROVIDER_NAME,
            model=model,
            details={
                "stage": stage,
                "category": category,
                "explanation": str(explanation)[:200] if explanation else None,
                "request_id": _request_id(response),
            },
            usage=self._usage(response, model),
        )

    @staticmethod
    def _first_text(response: Any) -> str:
        """The first text block. With a schema in force, that is the JSON."""
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                return str(getattr(block, "text", "") or "")
        return ""

    def _usage(self, response: Any, model: str) -> Usage:
        """Anthropic's four token counts, in the shape our ledger expects.

        Anthropic reports ``input_tokens`` **excluding** anything served from or
        written to the cache. Our `Usage.cached_prompt_tokens` is a subset of
        ``prompt_tokens``, not an addition, so the three are added back together
        and the cached share recorded inside the total.

        The dollar figure is worked out here, which is the opposite of what the
        OpenAI provider does — it leaves ``usd`` at zero so prices live in one
        file. That does not work for Anthropic: `app.llm.usage` has no Anthropic
        table, it assumes a single flat cache discount for every vendor, and it
        has nowhere to put the premium a cache *write* costs. So the number is
        worked out where the four counts are, and the ledger keeps whatever a
        provider supplies. Move it back to `app.llm.usage` the day that table
        can tell a cache read from a cache write.
        """
        raw = getattr(response, "usage", None)
        fresh = _count(raw, "input_tokens")
        output = _count(raw, "output_tokens")
        cache_read = _count(raw, "cache_read_input_tokens")
        cache_write = _count(raw, "cache_creation_input_tokens")

        prompt = fresh + cache_read + cache_write
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=output,
            model=f"{PROVIDER_NAME}:{model}",
            provider=PROVIDER_NAME,
            usd=price(model, fresh, output, cache_read, cache_write),
            cached_prompt_tokens=cache_read,
        )

    def _log_failure(self, stage: str, model: str, exc: Exception) -> None:
        """One line naming the request, because that is what support asks for."""
        if isinstance(exc, LLMError):
            return  # already described itself on the way past
        log.warning(
            "anthropic.call_failed",
            stage=stage,
            model=model,
            error=type(exc).__name__,
            status=getattr(exc, "status_code", None),
            request_id=_request_id(exc),
            detail=str(exc)[:200],
        )


# -------------------------------------------------------------------- pricing


def price(
    model: str,
    fresh_input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """What one call cost, in dollars. An unlisted model costs zero.

    Takes the bare model name. ``fresh_input_tokens`` is what Anthropic calls
    ``input_tokens`` — the part that was neither read from nor written to the
    cache — so the three input numbers add up rather than overlap.
    """
    rates = _rates(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return (
        max(0, fresh_input_tokens) * input_rate
        + max(0, cache_read_tokens) * input_rate * CACHE_READ_RATE
        + max(0, cache_write_tokens) * input_rate * CACHE_WRITE_RATE
        + max(0, output_tokens) * output_rate
    ) / 1_000_000


def _rates(model: str) -> tuple[float, float] | None:
    """Longest name that matches, so a suffixed variant still prices correctly."""
    name = str(model or "").strip()
    if name in PRICES:
        return PRICES[name]
    for known in sorted(PRICES, key=len, reverse=True):
        if name.startswith(known):
            return PRICES[known]
    return None


# -------------------------------------------------------------------- helpers


async def _maybe_await(value: Any) -> Any:
    """`get_final_message` is a coroutine on the async stream and a value on the
    sync one. Accepting both keeps a hand-rolled test double simple."""
    return await value if inspect.isawaitable(value) else value


def _count(usage: Any, field: str) -> int:
    try:
        return max(0, int(getattr(usage, field, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _request_id(obj: Any) -> str | None:
    """The request id, wherever this object happens to keep it.

    Public despite the underscore on a response. On an exception the SDK exposes
    it as ``request_id``; failing both, the response header still has it. Worth
    logging on every failure — it is the only handle support has.
    """
    for attribute in ("_request_id", "request_id"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    try:
        headers = getattr(getattr(obj, "response", None), "headers", None)
        if headers is not None:
            return headers.get("request-id") or headers.get("x-request-id")
    except AttributeError:
        pass
    return None


def _mentions_credit(exc: Exception) -> bool:
    """True when the error is really "you have run out of money"."""
    text = str(exc).lower()
    return any(phrase in text for phrase in _OUT_OF_CREDIT)


def provider() -> AnthropicProvider:
    """Factory. `router._build` looks for this after `PROVIDER`."""
    return AnthropicProvider()


__all__ = [
    "CACHE_READ_RATE",
    "CACHE_WRITE_RATE",
    "DEFAULT_MODEL",
    "EFFORTS",
    "PRICES",
    "PROVIDER_NAME",
    "AnthropicProvider",
    "price",
    "provider",
]
