"""The OpenAI adapter: our four calls, spoken to `chat.completions` and `embeddings`.

One of the providers behind `app.llm.base.Provider`. It knows how to phrase a
request for OpenAI and how to say what OpenAI's failures are called in our
vocabulary. It knows nothing about fallback chains, breakers, retry budgets or
ledgers — all of that lives in `app.llm.router` and `app.llm.usage`, once, for
every vendor. Nothing in the app imports this file; the router builds it when
`LLM_MODEL` starts with ``openai:``.

What is worth knowing here
--------------------------

**Structured output has two modes and they are not the same.** With a schema we
send ``response_format={"type": "json_schema", ..., "strict": true}``, and the
answer is guaranteed to match the schema — so `ChatResult.parsed` is filled in
and the router does not parse the same text twice. Without one we send
``{"type": "json_object"}``, which guarantees only that the text is valid JSON;
that mode also 400s unless the word "json" appears in the prompt, which is what
`_JSON_NUDGE` is for.

**Strict mode is a subset of JSON Schema.** Every object must carry
``additionalProperties: false`` and list every one of its properties in
``required``. `to_strict_schema` does that translation, making a property that
was optional nullable instead, because that is how strict mode spells optional.
``allOf`` has no strict equivalent — write the schema without it.

**Streaming is never buffered.** `router.stream_json` exists so the orchestrator
can start step one while step four is still being written; joining the deltas
here to hand back a whole object would quietly undo it. Chunks go out as they
land, and the token counts arrive after the last one, which is why both
streaming calls return a `TextStream` rather than a bare async generator.

**Caching is automatic.** OpenAI caches a prompt prefix over about 1024 tokens
with nothing asked for, and reports the reused part as
``usage.prompt_tokens_details.cached_tokens``. That number is a *subset* of
``prompt_tokens``, which is exactly what `Usage.cached_prompt_tokens` means, so
it goes straight across. A persistent zero means the prefix is moving between
calls — put the fixed catalogue first and the user's question last.

The embedding constraint
------------------------

`embed` is the one call here whose output outlives the request. Vectors go into
`VECTOR(1536)` columns that one specific model filled, and a vector from a
different model is not comparable with them even at the same width — the search
returns rows, they are just the wrong rows, with no error to say so. So the
model that produced a row is recorded in its `embed_model` column, `dimensions`
is honoured exactly or the call fails, and changing `EMBED_MODEL` means
re-embedding the whole mirror rather than mixing two vocabularies in one index.
This adapter never quietly returns a vector of the wrong width.

Configuration
-------------

Read through `_setting`, which asks `app.config.settings` first and the
environment second, so a knob works whether or not it has been added to
`Settings` yet.

    OPENAI_API_KEY        required
    OPENAI_BASE_URL       for a proxy or a gateway
    OPENAI_TIMEOUT_S      seconds on the SDK client; the router's own budget is
                          usually the shorter of the two
    OPENAI_TEMPERATURE    default 0.0, and not sent to models that refuse it
    OPENAI_STREAM_USAGE   default on; turn it off for a gateway that rejects
                          `stream_options`, at the cost of streamed token counts
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import os
import re
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final

import openai
from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.llm.base import ChatResult, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError, class_for_status

log = get_logger(__name__)

PROVIDER_NAME: Final[str] = "openai"

# The model when nobody names one. Everything else in the app spells models
# "openai:gpt-4.1-mini"; by the time we are called the prefix is gone.
DEFAULT_MODEL: Final[str] = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL: Final[str] = "text-embedding-3-small"

# Inputs per embeddings request. The endpoint accepts more, but a batch that
# fails takes everything in it down with it, and 128 short chunks is already one
# round trip's worth of work.
EMBED_BATCH: Final[int] = 128

# Only the text-embedding-3 family can be asked for a narrower vector. Anything
# else returns its native width, and we check rather than hope.
_DIMENSION_FAMILIES: Final[tuple[str, ...]] = ("text-embedding-3",)

# Models that reject an explicit temperature: the reasoning families run at 1
# and a request naming any other value is a 400.
_NO_TEMPERATURE: Final[tuple[str, ...]] = ("o1", "o3", "o4", "gpt-5")

# The same families accept `reasoning_effort`. Left to their default they think
# for tens of seconds before the first byte of JSON, which is how a 20-second
# planner budget turns into a coin flip. The planning call does not need deep
# reasoning — the probe already gathered the evidence — so the app asks for
# the low setting and lets OPENAI_REASONING_EFFORT override it.
_REASONING_FAMILIES: Final[tuple[str, ...]] = ("o1", "o3", "o4", "gpt-5")

# `json_object` mode refuses a prompt that never says "json". A fixed suffix, so
# the cached prefix stays byte-identical between calls.
_JSON_NUDGE: Final[str] = (
    "\n\nAnswer with a single JSON object and nothing else. No prose, no code fences."
)

# What a 429 says when the account is out of money rather than going too fast.
# The status is the same; what to do about it is not.
_OUT_OF_QUOTA: Final[tuple[str, ...]] = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
    "billing hard limit",
)

# What a 400 says when it is really a refusal.
_CONTENT_POLICY: Final[tuple[str, ...]] = (
    "content_policy_violation",
    "content_filter",
    "your request was rejected as a result of our safety system",
)

# Schema keywords strict mode has no meaning for. Dropped rather than passed on,
# because an unknown keyword is a 400 for the whole request.
_UNSUPPORTED_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"default", "examples", "$comment", "deprecated", "readOnly", "writeOnly", "additionalItems"}
)

_SCHEMA_NAME_OK: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_-]")


# ------------------------------------------------------------------- settings


def _setting(name: str, default: Any = None) -> Any:
    """One knob, from config if it is there and from the environment if not.

    `Settings` is declared with ``extra="ignore"``, so an ``OPENAI_*`` line in
    `.env` does not become an attribute until somebody adds the field to
    `app.config`. Checking both means this file works either way.
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


def _bool_setting(name: str, default: bool) -> bool:
    value = _setting(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- the provider


class OpenAIProvider:
    """OpenAI behind `app.llm.base.Provider`."""

    name: str = PROVIDER_NAME

    def __init__(self, client: Any | None = None) -> None:
        # No client until the first call. `router._build` imports this module to
        # find out whether the provider is installed, and a process configured
        # for Gemini should not trip over a missing OPENAI_API_KEY on the way
        # past. The argument is for tests, which pass a double.
        self._client: Any | None = client
        self._lock = threading.Lock()

    # -- the client -----------------------------------------------------------

    def client(self) -> Any:
        """The shared `AsyncOpenAI`, built on first use.

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

            api_key = str(_setting("OPENAI_API_KEY", "") or "")
            if not api_key and not os.environ.get("OPENAI_API_KEY"):
                raise LLMError(
                    ErrorClass.AUTH_EXPIRED,
                    "OPENAI_API_KEY is not set, so the OpenAI provider cannot be used.",
                    provider=PROVIDER_NAME,
                )

            kwargs: dict[str, Any] = {
                "max_retries": 0,
                # Seconds. Kept as its own knob so an operator can give the SDK
                # more room than the router's per-attempt budget; whichever is
                # shorter is the one that fires, and both end up as TIMEOUT.
                "timeout": _float_setting("OPENAI_TIMEOUT_S", 60.0),
            }
            if api_key:
                kwargs["api_key"] = api_key
            base_url = _setting("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = str(base_url)

            try:
                self._client = AsyncOpenAI(**kwargs)
            except Exception as exc:
                raise LLMError(
                    ErrorClass.AUTH_EXPIRED,
                    "The OpenAI client could not be built. Check OPENAI_API_KEY.",
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
        """One call, one JSON object back.

        With a schema the answer is guaranteed to match it, so it is parsed here
        and handed over in `ChatResult.parsed`. Without one the text is valid
        JSON but nothing more, and the router's `parse_json_object` does the
        rest — including the fenced-code-block repair this mode still needs.
        """
        name = model or DEFAULT_MODEL
        request = self._request(name, system, user, schema=schema, max_tokens=max_tokens)

        try:
            response = await self.client().chat.completions.create(**request)
        except Exception as exc:
            self._log_failure("complete_json", name, exc)
            raise

        usage = self._usage(getattr(response, "usage", None), name)
        choice = self._first_choice(response, name, "complete_json", usage)
        message = getattr(choice, "message", None)
        finish = str(getattr(choice, "finish_reason", "") or "")

        self._refuse_if_refused(finish, message, name, "complete_json", usage)

        text = str(getattr(message, "content", "") or "")
        parsed: dict[str, Any] | None = None
        if schema is not None and finish != "length" and text:
            # Strict mode makes the text JSON that matches the schema. Parse it
            # here to save the router a second pass; if the model surprised us,
            # hand the text over and let `parse_json_object` do its work.
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict):
                parsed = candidate

        # `finish_reason` goes through unchanged: OpenAI already spells the one
        # value the router acts on — "length" — the way the router expects.
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
        """Raw deltas of the JSON answer, never buffered to completion.

        The caller parses incrementally so it can dispatch step one before step
        four has arrived. Joining every chunk gives the same text
        `complete_json` would have returned.
        """
        name = model or DEFAULT_MODEL
        request = self._request(name, system, user, schema=schema, max_tokens=max_tokens)
        return self._open_stream(name, request, what="stream_json", truncation_is_fatal=True)

    def stream_text(self, model: str, system: str, user: str) -> AsyncIterator[str]:
        """Prose, chunk by chunk. No schema, no JSON, no ``response_format``."""
        name = model or DEFAULT_MODEL
        request = self._request(name, system, user, schema=None, max_tokens=0, want_json=False)
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
        counts do not exist until after the last chunk and a generator object
        has nowhere to put them.
        """
        payload = dict(request)
        payload["stream"] = True
        if _bool_setting("OPENAI_STREAM_USAGE", True):
            # Without this the streamed response carries no usage at all and
            # every streamed call files a zero in the ledger.
            payload["stream_options"] = {"include_usage": True}

        out = TextStream()

        async def run() -> AsyncIterator[str]:
            client = self.client()
            try:
                stream = await client.chat.completions.create(**payload)
            except Exception as exc:
                self._log_failure(what, model, exc)
                raise

            finish = ""
            raw_usage: Any = None
            try:
                async for chunk in stream:
                    # Usage rides on a final chunk that has no choices.
                    if getattr(chunk, "usage", None):
                        raw_usage = chunk.usage
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice = choices[0]
                    reason = getattr(choice, "finish_reason", None)
                    if reason:
                        finish = str(reason)
                    delta = getattr(choice, "delta", None)
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield piece
            except LLMError:
                raise
            except Exception as exc:
                self._log_failure(what, model, exc)
                raise
            finally:
                await _aclose(stream)

            out.usage = self._usage(raw_usage, model)

            if finish == "content_filter":
                raise LLMError(
                    ErrorClass.CONTENT_FILTERED,
                    "The model declined to answer that.",
                    provider=PROVIDER_NAME,
                    model=model,
                    details={"stage": what},
                    usage=out.usage,
                )
            if finish == "length":
                if truncation_is_fatal:
                    # Half a JSON object is not a smaller answer, it is a broken
                    # one. The router is committed by now, so this surfaces as a
                    # visible failure rather than a quiet splice.
                    raise LLMError(
                        ErrorClass.INVALID,
                        "The model ran out of room before finishing its answer.",
                        provider=PROVIDER_NAME,
                        model=model,
                        details={
                            "stage": what,
                            "max_tokens": payload.get("max_completion_tokens"),
                        },
                        usage=out.usage,
                    )
                log.warning(
                    "openai.truncated",
                    stage=what,
                    model=model,
                    max_tokens=payload.get("max_completion_tokens"),
                )

        return out.feed(run())

    # -- embeddings -----------------------------------------------------------

    async def embed(
        self,
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[list[list[float]], Usage]:
        """One vector per string, in the order given, every one ``dimensions`` long.

        Batched at `EMBED_BATCH` inputs per request, and the batches are stitched
        back together in the order they were asked for — the endpoint returns an
        ``index`` on every row and this sorts by it rather than trusting the
        order it happened to reply in. One text and one vector line up, or the
        wrong email is what the search finds.

        A width that is not what was asked for raises instead of returning. The
        vectors go into a `VECTOR(1536)` column beside vectors from one specific
        model, and a wrong-width or wrong-model vector is a corrupted mirror, not
        a poor answer.
        """
        name = model or DEFAULT_EMBED_MODEL
        items = [text if isinstance(text, str) else str(text) for text in texts]
        if not items:
            return [], self._embed_usage(None, name)

        blank = next((i for i, text in enumerate(items) if not text.strip()), None)
        if blank is not None:
            # The endpoint 400s on an empty input. The router turns empties into
            # zero vectors before it gets here, so reaching this means somebody
            # called the provider directly, and a clear message beats a 400.
            raise LLMError(
                ErrorClass.INVALID,
                "An embedding input was empty. Empty text has no vector.",
                provider=PROVIDER_NAME,
                model=name,
                details={"index": blank, "count": len(items)},
            )

        want = int(dimensions) if dimensions else 0
        vectors: list[list[float]] = []
        prompt_tokens = 0

        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start : start + EMBED_BATCH]
            request: dict[str, Any] = {
                "model": name,
                "input": batch,
                "encoding_format": "float",
            }
            if want and _supports_dimensions(name):
                request["dimensions"] = want

            try:
                response = await self.client().embeddings.create(**request)
            except Exception as exc:
                self._log_failure("embed", name, exc)
                raise

            rows = list(getattr(response, "data", None) or [])
            if len(rows) != len(batch):
                raise LLMError(
                    ErrorClass.INVALID,
                    "The embeddings endpoint returned the wrong number of vectors.",
                    provider=PROVIDER_NAME,
                    model=name,
                    details={"asked": len(batch), "got": len(rows)},
                )
            rows.sort(key=lambda row: _count(row, "index"))

            for offset, row in enumerate(rows):
                vector = [float(value) for value in (getattr(row, "embedding", None) or [])]
                if want and len(vector) != want:
                    raise LLMError(
                        ErrorClass.INVALID,
                        (
                            f"{name} returned a {len(vector)}-dimension vector where "
                            f"{want} was asked for. Storing it would corrupt the index."
                        ),
                        provider=PROVIDER_NAME,
                        model=name,
                        details={
                            "expected": want,
                            "got": len(vector),
                            "index": start + offset,
                            "supports_dimensions": _supports_dimensions(name),
                        },
                    )
                vectors.append(vector)

            prompt_tokens += _count(getattr(response, "usage", None), "prompt_tokens")

        return vectors, self._embed_usage(prompt_tokens, name)

    # -- naming the failures --------------------------------------------------

    def classify_error(self, exc: Exception) -> str:
        """An OpenAI exception as one of our `ErrorClass` names.

        Never raises. It runs on the failure path, and an exception here would
        lose the failure it was called to describe.
        """
        try:
            return str(self._classify(exc))
        except Exception:  # a classifier that throws is worse than a vague one
            return str(ErrorClass.UNKNOWN)

    def _classify(self, exc: Exception) -> ErrorClass:
        """Most specific first, which matters more than it looks.

        `APITimeoutError` is a subclass of `APIConnectionError`, and every 4xx
        class is a subclass of `APIStatusError`, so a broad clause placed first
        would swallow the lot.
        """
        if isinstance(exc, LLMError):
            # Already named — a refusal or a truncation we raised ourselves.
            return exc.error_class

        if isinstance(exc, openai.RateLimitError):
            self._note_retry_after(exc)
            # A 429 is either "too fast" or "out of money", and only the body
            # says which. The first wants a moment; the second wants a different
            # provider, and no amount of waiting will help it.
            if _mentions(exc, _OUT_OF_QUOTA):
                return ErrorClass.QUOTA_EXHAUSTED
            return ErrorClass.RATE_LIMITED
        if isinstance(exc, openai.AuthenticationError):
            return ErrorClass.AUTH_REVOKED
        if isinstance(exc, openai.PermissionDeniedError):
            return ErrorClass.AUTH_REVOKED
        if isinstance(exc, openai.NotFoundError):
            return ErrorClass.NOT_FOUND
        if isinstance(exc, openai.BadRequestError):
            if _mentions(exc, _CONTENT_POLICY):
                return ErrorClass.CONTENT_FILTERED
            if _mentions(exc, _OUT_OF_QUOTA):
                return ErrorClass.QUOTA_EXHAUSTED
            # Everything else at 400 is our request: a schema strict mode will
            # not take, a model that cannot do JSON, a context overflow. The
            # next provider would reject it too, which is why INVALID does not
            # fall back.
            return ErrorClass.INVALID
        if isinstance(exc, openai.UnprocessableEntityError):
            return ErrorClass.INVALID
        if isinstance(exc, openai.ConflictError):
            return ErrorClass.TRANSIENT
        if isinstance(exc, openai.InternalServerError):
            return ErrorClass.TRANSIENT
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and status >= 500:
                return ErrorClass.TRANSIENT
            return class_for_status(status if isinstance(status, int) else None)

        # Order matters: APITimeoutError inherits from APIConnectionError.
        if isinstance(exc, openai.APITimeoutError):
            return ErrorClass.TIMEOUT
        if isinstance(exc, openai.APIConnectionError):
            return ErrorClass.TRANSIENT
        if isinstance(exc, TimeoutError):
            return ErrorClass.TIMEOUT
        return ErrorClass.UNKNOWN

    def _note_retry_after(self, exc: Exception) -> None:
        """Log how long OpenAI wants us to wait, and leave it on the exception.

        `classify_error` can only return a class name, so this is where the hint
        gets recorded. The router does not sleep on a rate limit anyway — it
        moves to the next model in the chain — but the number belongs in the log
        line that explains why it did.
        """
        seconds = _retry_after_seconds(exc)
        with contextlib.suppress(Exception):  # a frozen exception type is not worth failing over
            exc.retry_after_s = seconds  # type: ignore[attr-defined]
        log.warning(
            "openai.rate_limited",
            retry_after_s=seconds,
            code=_error_code(exc),
            request_id=_request_id(exc),
        )

    def supports_prefix_caching(self) -> bool:
        """True, and nothing has to be asked for.

        OpenAI caches a prompt prefix longer than about 1024 tokens by itself and
        reports the hit in ``usage.prompt_tokens_details.cached_tokens``. What
        the caller should do about it is order the prompt so the fixed part —
        the op catalogue, the rules — comes first and the user's question last.
        """
        return True

    # -- building the request -------------------------------------------------

    def _request(
        self,
        model: str,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None,
        max_tokens: int,
        want_json: bool = True,
    ) -> dict[str, Any]:
        """Everything both the streamed and the non-streamed call need."""
        system_text = (system or "").strip()
        response_format = self._response_format(schema) if want_json else None

        # `json_object` mode 400s unless the prompt says "json" somewhere.
        wants_nudge = response_format is not None and response_format["type"] == "json_object"
        if wants_nudge and "json" not in f"{system_text} {user}".lower():
            system_text = f"{system_text}{_JSON_NUDGE}" if system_text else _JSON_NUDGE.strip()

        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user})

        request: dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            request["response_format"] = response_format
        if max_tokens:
            # `max_tokens` is deprecated and the reasoning models reject it
            # outright; `max_completion_tokens` is the name that works on all of
            # them. Note it bounds reasoning tokens too, where there are any.
            request["max_completion_tokens"] = int(max_tokens)

        temperature = _float_setting("OPENAI_TEMPERATURE", 0.0)
        if _supports_temperature(model):
            request["temperature"] = temperature
        if _supports_reasoning_effort(model):
            effort = str(_setting("OPENAI_REASONING_EFFORT", "low") or "").strip()
            if effort:
                request["reasoning_effort"] = effort
        return request

    def _response_format(self, schema: dict[str, Any] | None) -> dict[str, Any]:
        """Strict structured output when there is a schema, plain JSON when not.

        The difference is worth being clear about: with a schema the shape is
        guaranteed by the decoder, so a missing field is impossible rather than
        unlikely. Without one, "valid JSON" is the whole promise.
        """
        if schema is None:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _schema_name(schema),
                "schema": to_strict_schema(schema),
                "strict": True,
            },
        }

    # -- reading the response -------------------------------------------------

    def _first_choice(self, response: Any, model: str, stage: str, usage: Usage) -> Any:
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            raise LLMError(
                ErrorClass.TRANSIENT,
                "The model returned no answer at all.",
                provider=PROVIDER_NAME,
                model=model,
                details={"stage": stage, "request_id": _request_id(response)},
                usage=usage,
            )
        return choices[0]

    def _refuse_if_refused(
        self,
        finish_reason: str,
        message: Any,
        model: str,
        stage: str,
        usage: Usage,
    ) -> None:
        """Stop on a refusal before touching the content.

        A refusal is an HTTP 200 with nothing usable in it, not an exception, so
        nothing else will catch it. It arrives two ways: ``content_filter`` as
        the finish reason, and — under a strict schema — a populated ``refusal``
        field with ``content`` left null.
        """
        refusal = getattr(message, "refusal", None)
        if finish_reason != "content_filter" and not refusal:
            return

        raise LLMError(
            ErrorClass.CONTENT_FILTERED,
            "The model declined to answer that.",
            provider=PROVIDER_NAME,
            model=model,
            details={
                "stage": stage,
                "finish_reason": finish_reason,
                "refusal": str(refusal)[:200] if refusal else None,
            },
            usage=usage,
        )

    def _usage(self, raw: Any, model: str) -> Usage:
        """OpenAI's token counts in the shape our ledger expects.

        ``cached_tokens`` is part of ``prompt_tokens`` on OpenAI's side and part
        of ``prompt_tokens`` on ours, so it crosses over unchanged. The dollar
        figure is left at zero deliberately: `app.llm.usage` has the OpenAI price
        table and fills in anything a provider left blank, so prices live in one
        file instead of two that can disagree.
        """
        details = _attr(raw, "prompt_tokens_details")
        return Usage(
            prompt_tokens=_count(raw, "prompt_tokens"),
            completion_tokens=_count(raw, "completion_tokens"),
            model=f"{PROVIDER_NAME}:{model}",
            provider=PROVIDER_NAME,
            usd=0.0,
            cached_prompt_tokens=_count(details, "cached_tokens") if details is not None else 0,
        )

    def _embed_usage(self, prompt_tokens: int | None, model: str) -> Usage:
        """Embeddings bill input only, so the completion side is always zero."""
        return Usage(
            prompt_tokens=max(0, int(prompt_tokens or 0)),
            completion_tokens=0,
            model=f"{PROVIDER_NAME}:{model}",
            provider=PROVIDER_NAME,
            usd=0.0,
        )

    def _log_failure(self, stage: str, model: str, exc: Exception) -> None:
        """One line naming the request, because that is what support asks for."""
        if isinstance(exc, LLMError):
            return
        log.warning(
            "openai.call_failed",
            stage=stage,
            model=model,
            error=type(exc).__name__,
            status=getattr(exc, "status_code", None),
            code=_error_code(exc),
            request_id=_request_id(exc),
            detail=str(exc)[:200],
        )


# ------------------------------------------------------------ strict schemas


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """An ordinary JSON Schema as the subset OpenAI's strict mode accepts.

    Three rules, applied all the way down:

    * every object gets ``additionalProperties: false``;
    * every property is listed in ``required`` — strict mode has no optional
      keys, so a property that was optional becomes nullable instead, which is
      how the docs say to spell "may be absent";
    * ``oneOf`` becomes ``anyOf`` and a handful of annotation keywords are
      dropped, because an unrecognised keyword fails the whole request.

    ``allOf`` has no strict equivalent and is left alone — a schema using it will
    be rejected, and merging branches behind the caller's back would be a worse
    surprise than the 400.

    The input is never modified; the caller keeps whatever it passed in.
    """
    if not isinstance(schema, dict):
        raise LLMError(
            ErrorClass.INVALID,
            "A response schema must be a JSON Schema object.",
            provider=PROVIDER_NAME,
            details={"type": type(schema).__name__},
        )

    root = _strict(copy.deepcopy(schema))
    if root.get("type") not in (None, "object") or "properties" not in root:
        raise LLMError(
            ErrorClass.INVALID,
            (
                "OpenAI strict mode needs an object at the root of the schema. "
                "Wrap the value in one property."
            ),
            provider=PROVIDER_NAME,
            details={"root_type": root.get("type")},
        )
    return root


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {k: v for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}

    if "oneOf" in out:
        out["anyOf"] = out.pop("oneOf")
    for key in ("anyOf", "allOf", "prefixItems"):
        if isinstance(out.get(key), list):
            out[key] = [_strict(item) for item in out[key]]
    for key in ("items", "contains", "not"):
        if key in out:
            out[key] = _strict(out[key])
    for key in ("$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {name: _strict(value) for name, value in out[key].items()}

    properties = out.get("properties")
    if isinstance(properties, dict):
        was_required = set(out.get("required") or [])
        rewritten: dict[str, Any] = {}
        for prop_name, prop in properties.items():
            child = _strict(prop)
            if prop_name not in was_required:
                child = _nullable(child)
            rewritten[prop_name] = child
        out["properties"] = rewritten
        out["required"] = list(rewritten)
        out["additionalProperties"] = False
        out.setdefault("type", "object")

    return out


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    """A schema that also accepts null. How strict mode spells "optional"."""
    if not isinstance(node, dict):
        return node
    kind = node.get("type")
    if isinstance(kind, str):
        return node if kind == "null" else {**node, "type": [kind, "null"]}
    if isinstance(kind, list):
        return node if "null" in kind else {**node, "type": [*kind, "null"]}
    if isinstance(node.get("anyOf"), list):
        branches = node["anyOf"]
        if any(branch.get("type") == "null" for branch in branches if isinstance(branch, dict)):
            return node
        return {**node, "anyOf": [*branches, {"type": "null"}]}
    # A bare $ref or a schema with no type: wrap it rather than guess.
    return {"anyOf": [node, {"type": "null"}]}


def _schema_name(schema: dict[str, Any]) -> str:
    """A name for the schema. OpenAI allows letters, digits, ``_`` and ``-``."""
    raw = str(schema.get("title") or schema.get("name") or "answer")
    cleaned = _SCHEMA_NAME_OK.sub("_", raw).strip("_")[:64]
    return cleaned or "answer"


# -------------------------------------------------------------------- helpers


def _supports_dimensions(model: str) -> bool:
    """True for the models that can be asked for a narrower vector."""
    name = str(model or "")
    return any(name.startswith(family) for family in _DIMENSION_FAMILIES)


def _supports_temperature(model: str) -> bool:
    """False for the reasoning families, which 400 on any temperature but 1."""
    name = str(model or "")
    return not any(name.startswith(family) for family in _NO_TEMPERATURE)


def _supports_reasoning_effort(model: str) -> bool:
    """True for the families that take `reasoning_effort`; others 400 on it."""
    name = str(model or "")
    return any(name.startswith(family) for family in _REASONING_FAMILIES)


def _attr(obj: Any, field: str) -> Any:
    """One field, whether the SDK handed back an object or a plain dict.

    Test doubles and gateway responses are dicts often enough that reaching for
    both is cheaper than being surprised by one.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _count(obj: Any, field: str) -> int:
    try:
        return max(0, int(_attr(obj, field) or 0))
    except (TypeError, ValueError):
        return 0


def _error_code(exc: Exception) -> str | None:
    """OpenAI's machine-readable code — ``insufficient_quota`` and friends."""
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code"):
            return str(error["code"])
    return None


def _mentions(exc: Exception, phrases: Sequence[str]) -> bool:
    """True when the code or the message contains one of these."""
    haystack = f"{_error_code(exc) or ''} {exc}".lower()
    return any(phrase in haystack for phrase in phrases)


def _retry_after_seconds(exc: Exception) -> float | None:
    """The ``retry-after`` header as a number, if the response carried one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    for header in ("retry-after", "retry-after-ms", "x-ratelimit-reset-requests"):
        try:
            raw = headers.get(header)
        except AttributeError:
            return None
        if not raw:
            continue
        try:
            value = float(str(raw).rstrip("s"))
        except ValueError:
            continue
        return value / 1000.0 if header.endswith("-ms") else value
    return None


def _request_id(obj: Any) -> str | None:
    """The request id, wherever this object happens to keep it.

    Worth logging on every failure — it is the only handle support has.
    """
    for attribute in ("request_id", "_request_id"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    headers = getattr(getattr(obj, "response", None), "headers", None)
    if headers is not None:
        try:
            return headers.get("x-request-id") or headers.get("request-id")
        except AttributeError:
            return None
    return None


async def _aclose(stream: Any) -> None:
    """Close a stream, however it ended. Closing a broken one may be messy."""
    closer = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception:  # nothing useful to do about a failed close
        pass


def provider() -> OpenAIProvider:
    """Factory. `router._build` looks for this after `PROVIDER`."""
    return OpenAIProvider()


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_MODEL",
    "EMBED_BATCH",
    "PROVIDER_NAME",
    "OpenAIProvider",
    "provider",
    "to_strict_schema",
]
