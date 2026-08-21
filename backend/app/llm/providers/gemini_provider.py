"""The Gemini adapter: our four calls, spoken to Google's model API.

Second provider behind `app.llm.base.Provider`. It knows how to phrase a request
for Gemini and how to name Gemini's failures in our vocabulary. It knows nothing
about fallback chains, breakers, retries or ledgers — those live in
`app.llm.router` and `app.llm.usage`, once, for every provider.

Two ways to reach the API, one set of words
-------------------------------------------

Everything is built as the REST body Google documents — camel case,
`generationConfig`, `systemInstruction` — and then sent one of two ways:

* **the `google-genai` SDK** (``from google import genai``, the async client),
  when it is installed;
* **plain `httpx`** against ``generativelanguage.googleapis.com``, when it is
  not.

Today it is httpx: `google-genai` is not in `backend/pyproject.toml`, and the
`google` namespace that *is* installed is `google-api-python-client`, which is a
different thing entirely. The REST path is therefore the one that runs, and it
is written against the published request and response shapes rather than against
a remembered SDK signature. Add the dependency and the SDK path takes over on
the next restart with no other change — `GEMINI_TRANSPORT=rest` pins it back.

Both paths return the same JSON shapes, because the SDK's objects are the REST
message names in snake case, so one set of readers (`_text_of`, `_usage_of`,
`_finish_of`) copes with either.

Things worth knowing before changing anything here
--------------------------------------------------

**The system prompt is not part of the user turn.** It goes in
`systemInstruction`. Merging it into the user text costs the model the
distinction and costs us any hope of prompt caching.

**Streaming is never buffered.** `stream_json` yields raw text as it lands, and
the caller parses incrementally — that is the whole reason the orchestrator can
start step one while step four is still being written.

**Thinking eats the output budget.** On 2.5 models, thoughts and answer come out
of the same `maxOutputTokens`. A literal ``max_tokens=1200`` can be spent
entirely on thinking and return an empty answer with `finishReason: MAX_TOKENS`,
which looks exactly like a model that had nothing to say. So thinking is turned
off for the planner where the model allows it (`gemini-2.5-pro` does not allow
it), and the output budget has a floor. See `_thinking` and `_max_output_tokens`.

**Thought text is not answer text.** Parts marked `thought` are skipped. Letting
them through would splice reasoning into the middle of a JSON plan.

Embeddings, and the one thing that must not be papered over
-----------------------------------------------------------

Chat models are swappable. **Embedding models are not.** The pgvector columns are
`VECTOR(1536)`, filled by one specific model, and a vector from a different model
is not comparable with them even at the same width — the two models put meaning
in different places. A cosine search across a mix returns confident nonsense: no
error, no warning, just the wrong emails.

So: every mirror row records the model that embedded it in `embed_model`
(`sync_messages`, `sync_events`, `sync_files`, `VARCHAR(64) NOT NULL`), the search
path checks that column against the query's model before trusting a distance,
and **changing `EMBED_MODEL` means re-embedding the whole mirror** — in batches,
with the management command, not opportunistically.

`gemini-embedding-001` does support `output_dimensionality=1536`, so switching to
it does not change the column type. It changes every value in it. Same shape,
different meaning.

Two Gemini specifics that fall out of that:

* Vectors at any width other than 3072 come back **not normalised**, because the
  model is trained so that a truncated vector is still meaningful once you fix
  its length. We normalise here. Storing un-normalised vectors alongside
  normalised ones is the same class of bug as mixing models.
* The **task type** is part of what a vector means, exactly like the model name.
  `GEMINI_EMBED_TASK_TYPE` defaults to `SEMANTIC_SIMILARITY` because our
  interface cannot tell a query from a document, and changing it later means
  re-embedding just as surely as changing the model does.

Configuration
-------------

Read through `_setting`, which asks `app.config.settings` first and the
environment second, so a knob works before anybody adds the field to `Settings`
(which is declared `extra="ignore"`, so `.env` alone will not create one).

    GOOGLE_AI_API_KEY        the model API key. Not the OAuth client — that one
                             is for the user's Gmail; this one is for Gemini.
    GEMINI_API_KEY           accepted as an alias, and GOOGLE_API_KEY too: they
                             are what the SDK and Google's samples read
    GEMINI_BASE_URL          default https://generativelanguage.googleapis.com
    GEMINI_API_VERSION       default v1beta
    GEMINI_TRANSPORT         auto | sdk | rest                 (default auto)
    GEMINI_TIMEOUT_S         HTTP timeout; the router's deadline is shorter
    GEMINI_THINKING_BUDGET   thinking tokens, 0 is off          (default 0)
    GEMINI_MAX_TOKENS        floor for a non-streamed answer    (default 2048)
    GEMINI_STREAM_MAX_TOKENS floor for a streamed answer        (default 8192)
    GEMINI_EMBED_TASK_TYPE   default SEMANTIC_SIMILARITY
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import aclosing
from typing import Any, Final, Protocol

import httpx

from app.core.logging import get_logger
from app.llm.base import ChatResult, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError, class_for_status
from app.llm.schema_translate import Translation, translate

log = get_logger(__name__)

PROVIDER_NAME: Final[str] = "gemini"

# The model when nobody names one. Everything else in the app spells models
# "gemini:gemini-2.5-flash"; by the time we are called the prefix is gone.
DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_EMBED_MODEL: Final[str] = "gemini-embedding-001"

API_HOST: Final[str] = "https://generativelanguage.googleapis.com"
API_VERSION: Final[str] = "v1beta"

# `batchEmbedContents` takes at most this many texts per call. The router
# already batches at 96, which sits under it; this is the belt to that braces,
# for anyone who calls the provider directly.
EMBED_BATCH: Final[int] = 100

# gemini-embedding-001's native width. Ask for anything narrower and the vector
# comes back the right length but not unit length — see the module docstring.
FULL_DIMENSIONS: Final[int] = 3072

# Model families that do not think, and so need no room set aside for it.
# Anything not on this list is assumed to think, because every family since has.
_NON_THINKING: Final[tuple[str, ...]] = ("gemini-1.", "gemini-2.0")

# Room set aside for thoughts when the model thinks and will not say how much.
# A guess, and a cheap one: unused output tokens are not billed.
THINKING_HEADROOM: Final[int] = 4096

# Gemini's reasons for stopping, in our words. `router.complete_json` treats
# "length" as a hard failure, so MAX_TOKENS has to arrive under that name or a
# half-written plan would be handed on as though it were whole.
_FINISH: Final[dict[str, str]] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
    "LANGUAGE": "other",
    "OTHER": "other",
}

# Finish reasons that mean the model refused rather than finished.
_REFUSALS: Final[frozenset[str]] = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)

# A 429 that will clear in a minute, versus one that will not clear today. Both
# fall back to another provider; they differ in what we tell the operator and in
# whether the same provider is worth another go.
_PER_MINUTE: Final[tuple[str, ...]] = ("perminute", "per minute", "requests per minute")
_OUT_OF_QUOTA: Final[tuple[str, ...]] = (
    "perday",
    "per day",
    "daily limit",
    "free tier",
    "free_tier",
    "billing",
    "billing_not_active",
    "out of credit",
)

# A 400 that is really an authentication problem. Google returns INVALID_ARGUMENT
# for a bad key, and calling that a malformed request would send an operator
# looking at the prompt.
_BAD_KEY: Final[tuple[str, ...]] = ("api key not valid", "api_key_invalid", "invalid api key")

_JSON_NUDGE: Final[str] = (
    "\n\nAnswer with a single JSON object and nothing else. No prose, no code fences."
)


# ------------------------------------------------------------------- settings


def _setting(name: str, default: Any = None) -> Any:
    """One knob, from config if it is there and from the environment if not.

    `Settings` is declared with ``extra="ignore"``, so a ``GEMINI_*`` line in
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
        return int(_setting(name, default))
    except (TypeError, ValueError):
        return default


def _api_key() -> str:
    """The model API key, under any of the names people give it.

    `GOOGLE_AI_API_KEY` is ours and the one `.env.example` documents; the other
    two are what the SDK and Google's own samples read, and someone will set one
    of those and wonder why nothing happened.
    """
    for name in ("GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = _setting(name)
        if value:
            return str(value)
    raise LLMError(
        ErrorClass.AUTH_EXPIRED,
        "Gemini has no API key. Set GOOGLE_AI_API_KEY.",
        provider=PROVIDER_NAME,
    )


def _model_name(model: str) -> str:
    """The bare model name.

    Google's own docs write it both ways — ``gemini-2.5-flash`` and
    ``models/gemini-2.5-flash`` — and the REST path adds the prefix itself, so a
    configured `LLM_MODEL=gemini:models/gemini-2.5-flash` would otherwise ask
    for `models/models/gemini-2.5-flash`.
    """
    name = (model or "").strip()
    return name[len("models/") :] if name.startswith("models/") else name


def _base_url() -> str:
    return str(_setting("GEMINI_BASE_URL", API_HOST)).rstrip("/")


def _api_version() -> str:
    return str(_setting("GEMINI_API_VERSION", API_VERSION)).strip("/")


# ------------------------------------------------------------------ transports


class _Transport(Protocol):
    """How the request actually leaves the process. SDK or httpx, same words."""

    kind: str

    async def generate(self, model: str, body: dict[str, Any]) -> Any: ...

    def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[Any]: ...

    async def embed(
        self, model: str, texts: Sequence[str], dimensions: int, task_type: str
    ) -> tuple[list[list[float]], Any]: ...

    async def aclose(self) -> None: ...


class _RestTransport:
    """The REST API over httpx. No SDK, no surprises, one dependency we have.

    `max_retries` has no equivalent to turn off here, which is the point: every
    retry decision belongs to the router, which is holding a deadline this code
    knows nothing about.
    """

    kind = "rest"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._lock = threading.Lock()

    def client(self) -> httpx.AsyncClient:
        existing = self._client
        if existing is not None:
            return existing
        with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=f"{_base_url()}/{_api_version()}",
                    # Well above the router's per-attempt budget, so the
                    # router's clock is the one that decides.
                    timeout=httpx.Timeout(_float_setting("GEMINI_TIMEOUT_S", 120.0)),
                    headers={
                        # Header, not a query parameter: a key in a URL ends up
                        # in logs, proxies and error reports.
                        "x-goog-api-key": _api_key(),
                        "content-type": "application/json",
                    },
                )
            return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def generate(self, model: str, body: dict[str, Any]) -> Any:
        response = await self.client().post(f"/models/{model}:generateContent", json=body)
        _raise_for_status(response, model)
        return response.json()

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[Any]:
        """Server-sent events, one `GenerateContentResponse` per `data:` line.

        ``?alt=sse`` matters: without it the endpoint answers with one long JSON
        array that only parses once it is complete, which is the opposite of
        streaming.
        """
        request = self.client().build_request(
            "POST", f"/models/{model}:streamGenerateContent", params={"alt": "sse"}, json=body
        )
        response = await self.client().send(request, stream=True)
        try:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response, model)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("gemini.stream_chunk_unreadable", model=model, head=payload[:120])
        finally:
            await response.aclose()

    async def embed(
        self, model: str, texts: Sequence[str], dimensions: int, task_type: str
    ) -> tuple[list[list[float]], Any]:
        body = {
            "requests": [
                {
                    # Each sub-request repeats the model. The API requires it,
                    # and it is the fully qualified name, not the bare one.
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": dimensions,
                }
                for text in texts
            ]
        }
        response = await self.client().post(f"/models/{model}:batchEmbedContents", json=body)
        _raise_for_status(response, model)
        payload = response.json()
        vectors = [
            [float(v) for v in (_field(item, "values") or [])]
            for item in (_field(payload, "embeddings") or [])
        ]
        return vectors, _field(payload, "usageMetadata", "usage_metadata")


class _SdkTransport:
    """The `google-genai` async client, when it is installed.

    The SDK's objects are the REST messages in snake case, so the request built
    for REST needs only its keys renamed on the way in, and the readers below
    cope with what comes back either way.
    """

    kind = "sdk"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._lock = threading.Lock()

    def client(self) -> Any:
        existing = self._client
        if existing is not None:
            return existing
        with self._lock:
            if self._client is None:
                # The ignore is for the day it is not installed: `google` is a
                # namespace package that other Google libraries also live in, so
                # the import resolves far enough for a type checker to notice
                # that `genai` is missing from it.
                from google import genai  # type: ignore[attr-defined]

                self._client = genai.Client(api_key=_api_key())
            return self._client

    async def aclose(self) -> None:
        # The SDK owns its own connection pool and offers no close on the
        # client, so there is nothing to shut down here.
        self._client = None

    async def generate(self, model: str, body: dict[str, Any]) -> Any:
        contents, config = _sdk_call(body)
        return await self.client().aio.models.generate_content(
            model=model, contents=contents, config=config
        )

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[Any]:
        contents, config = _sdk_call(body)
        stream = self.client().aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        # Different releases of the SDK have this return the iterator and
        # return a coroutine that yields the iterator. Both are handled rather
        # than guessed at.
        if hasattr(stream, "__await__"):
            stream = await stream
        async for chunk in stream:
            yield chunk

    async def embed(
        self, model: str, texts: Sequence[str], dimensions: int, task_type: str
    ) -> tuple[list[list[float]], Any]:
        result = await self.client().aio.models.embed_content(
            model=model,
            contents=list(texts),
            config={"output_dimensionality": dimensions, "task_type": task_type},
        )
        vectors = [
            [float(v) for v in (_field(item, "values") or [])]
            for item in (_field(result, "embeddings") or [])
        ]
        return vectors, _field(result, "usageMetadata", "usage_metadata", "metadata")


def _sdk_call(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A REST body as the SDK's ``contents`` and ``config``.

    Plain strings rather than message dicts wherever the SDK accepts them: there
    is exactly one user turn and one system instruction, and a string cannot be
    coerced wrongly.
    """
    contents = ""
    for turn in body.get("contents", []):
        for part in _field(turn, "parts") or []:
            contents += str(_field(part, "text") or "")

    gen = body.get("generationConfig", {})
    config: dict[str, Any] = {}

    system = ""
    for part in _field(body.get("systemInstruction"), "parts") or []:
        system += str(_field(part, "text") or "")
    if system:
        config["system_instruction"] = system

    if gen.get("maxOutputTokens"):
        config["max_output_tokens"] = gen["maxOutputTokens"]
    if gen.get("responseMimeType"):
        config["response_mime_type"] = gen["responseMimeType"]
    if gen.get("responseSchema"):
        config["response_schema"] = _snake_schema(gen["responseSchema"])
    thinking = gen.get("thinkingConfig")
    if thinking:
        config["thinking_config"] = {"thinking_budget": thinking["thinkingBudget"]}
    return contents, config


def _snake_schema(schema: Any) -> Any:
    """The translated schema with the SDK's spelling of the two-word fields.

    `app.llm.schema_translate` emits the REST spelling, because that is the wire
    format. The SDK's `types.Schema` is a pydantic model whose field names are
    snake case; whether it also accepts the camel aliases has varied by release,
    so this renames rather than relies on it.
    """
    if isinstance(schema, list):
        return [_snake_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    renames = {
        "propertyOrdering": "property_ordering",
        "minItems": "min_items",
        "maxItems": "max_items",
    }
    out: dict[str, Any] = {}
    for key, value in schema.items():
        name = renames.get(str(key), str(key))
        if key == "properties" and isinstance(value, dict):
            out[name] = {k: _snake_schema(v) for k, v in value.items()}
        elif key == "items":
            out[name] = _snake_schema(value)
        else:
            out[name] = value
    return out


# -------------------------------------------------------------- the provider


class GeminiProvider:
    """Gemini behind `app.llm.base.Provider`."""

    name: str = PROVIDER_NAME

    def __init__(self, transport: Any | None = None) -> None:
        # No transport until the first call. Importing this module must not need
        # an API key: `router._build` imports a provider to find out whether it
        # is usable at all, and a process configured for OpenAI only should not
        # trip over a missing GOOGLE_AI_API_KEY on the way past.
        self._transport: Any | None = transport
        self._lock = threading.Lock()
        # Reasons a schema could not be enforced, so each is said once.
        self._warned: set[str] = set()

    # -- the transport --------------------------------------------------------

    def transport(self) -> Any:
        """The SDK if it is installed and wanted, httpx otherwise."""
        existing = self._transport
        if existing is not None:
            return existing

        with self._lock:
            if self._transport is not None:
                return self._transport

            wanted = str(_setting("GEMINI_TRANSPORT", "auto")).strip().lower()
            if wanted != "rest" and _sdk_available():
                self._transport = _SdkTransport()
            elif wanted == "sdk":
                raise LLMError(
                    ErrorClass.INVALID,
                    "GEMINI_TRANSPORT=sdk but google-genai is not installed. "
                    "Add google-genai to backend/pyproject.toml, or unset it to use REST.",
                    provider=PROVIDER_NAME,
                )
            else:
                self._transport = _RestTransport()

            log.info("gemini.transport", transport=self._transport.kind)
            return self._transport

    async def aclose(self) -> None:
        """Shut the HTTP client down. Called from `router.close`."""
        transport, self._transport = self._transport, None
        if transport is not None:
            await transport.aclose()

    # -- one request, one JSON object -----------------------------------------

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> ChatResult:
        """One call, one JSON object.

        `responseMimeType` is `application/json` and the schema — when it
        survives translation — is `responseSchema`, so the answer is JSON by
        construction rather than by asking nicely. It is still parsed here when
        it parses, so the router does not do the same work twice.
        """
        name = _model_name(model or DEFAULT_MODEL)
        body = self._body(
            name,
            system,
            user,
            schema=schema,
            max_output_tokens=self._max_output_tokens(max_tokens, name, streaming=False),
            want_json=True,
        )

        try:
            response = await self.transport().generate(name, body)
        except Exception as exc:
            self._log_failure("complete_json", name, exc)
            raise

        usage = _usage_of(response, name)
        _refuse_if_filtered(response, name, "complete_json", usage)

        text = _text_of(response)
        finish = _finish_of(response)

        parsed: dict[str, Any] | None = None
        if finish != "length" and text:
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
        name = _model_name(model or DEFAULT_MODEL)
        body = self._body(
            name,
            system,
            user,
            schema=schema,
            max_output_tokens=self._max_output_tokens(max_tokens, name, streaming=True),
            want_json=True,
        )
        return self._open_stream(name, body, what="stream_json", truncation_is_fatal=True)

    def stream_text(self, model: str, system: str, user: str) -> AsyncIterator[str]:
        """Prose, chunk by chunk. No JSON mime type, no schema."""
        name = _model_name(model or DEFAULT_MODEL)
        body = self._body(
            name,
            system,
            user,
            schema=None,
            max_output_tokens=self._max_output_tokens(0, name, streaming=True),
            want_json=False,
        )
        return self._open_stream(name, body, what="stream_text", truncation_is_fatal=False)

    def _open_stream(
        self,
        model: str,
        body: dict[str, Any],
        *,
        what: str,
        truncation_is_fatal: bool,
    ) -> AsyncIterator[str]:
        """Both streaming calls, which differ only in what a cut-off answer means.

        A `TextStream` rather than a bare async generator, because the token
        counts arrive after the last chunk and a generator object has nowhere to
        put them.
        """
        out = TextStream()

        async def run() -> AsyncIterator[str]:
            finish = ""
            blocked = ""

            try:
                # `aclosing` so that a caller who stops early — the router does,
                # on a timeout — releases the HTTP response now rather than
                # whenever the generator happens to be collected.
                async with aclosing(self.transport().stream(model, body)) as chunks:
                    async for chunk in chunks:
                        blocked = blocked or _block_reason(chunk)
                        meta = _field(chunk, "usageMetadata", "usage_metadata")
                        if meta is not None:
                            # Gemini repeats the running totals on every chunk,
                            # so filing them as they arrive means a stream that
                            # breaks at token four hundred still says what it
                            # burned.
                            out.usage = _usage_from(meta, model)
                        reason = _raw_finish(chunk)
                        if reason:
                            finish = reason
                        piece = _text_of(chunk)
                        if piece:
                            yield piece
            except LLMError:
                raise
            except Exception as exc:
                self._log_failure(what, model, exc)
                raise

            if out.usage is None:
                out.usage = _usage_from(None, model)

            if blocked or finish in _REFUSALS:
                raise LLMError(
                    ErrorClass.CONTENT_FILTERED,
                    _refusal_words(blocked or finish),
                    provider=PROVIDER_NAME,
                    model=model,
                    details={"stage": what, "reason": blocked or finish},
                    usage=out.usage,
                )

            if finish == "MAX_TOKENS":
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
                            "max_tokens": body["generationConfig"].get("maxOutputTokens"),
                        },
                        usage=out.usage,
                    )
                log.warning(
                    "gemini.truncated",
                    stage=what,
                    model=model,
                    max_tokens=body["generationConfig"].get("maxOutputTokens"),
                )

        return out.feed(run())

    # -- embeddings -----------------------------------------------------------

    async def embed(
        self,
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[list[list[float]], Usage]:
        """One vector per string, in the order given, all `dimensions` long.

        `outputDimensionality` is honoured, which is what lets
        `gemini-embedding-001` fill a `VECTOR(1536)` column without the column
        changing. It does not let the *rows* stay: a vector from this model is
        not comparable with one from OpenAI, so switching `EMBED_MODEL` means
        re-embedding the whole mirror. See the module docstring.

        Anything other than the model's native 3072 comes back un-normalised and
        is normalised here — Google's own instruction, and the reason a
        truncated vector is meaningful at all.
        """
        name = _model_name(model or DEFAULT_EMBED_MODEL)
        items = [t if isinstance(t, str) else str(t) for t in texts]
        if not items:
            return [], Usage(0, 0, f"{PROVIDER_NAME}:{name}", PROVIDER_NAME, 0.0)

        task_type = str(_setting("GEMINI_EMBED_TASK_TYPE", "SEMANTIC_SIMILARITY")).upper()
        vectors: list[list[float]] = []
        counted = 0

        for start in range(0, len(items), EMBED_BATCH):
            batch = items[start : start + EMBED_BATCH]
            try:
                got, meta = await self.transport().embed(name, batch, dimensions, task_type)
            except Exception as exc:
                self._log_failure("embed", name, exc)
                raise

            if len(got) != len(batch):
                # The router checks this as well. Both are cheap, and a mirror
                # row that ends up with somebody else's vector is not something
                # anything downstream can detect.
                raise LLMError(
                    ErrorClass.INVALID,
                    "Gemini returned a different number of vectors than we asked for.",
                    provider=PROVIDER_NAME,
                    model=name,
                    details={"asked": len(batch), "got": len(got)},
                )
            vectors.extend(_unit(v) if dimensions != FULL_DIMENSIONS else v for v in got)
            counted += _int(
                _field(meta, "promptTokenCount", "prompt_token_count", "totalTokenCount")
            )

        for vector in vectors:
            if len(vector) != dimensions:
                # The router checks this too. It is here as well because a wrong
                # width reaching pgvector is a corrupted mirror, not a bad answer,
                # and the cheapest place to stop it is closest to the wire.
                raise LLMError(
                    ErrorClass.INVALID,
                    "Gemini returned a vector of the wrong width.",
                    provider=PROVIDER_NAME,
                    model=name,
                    details={"expected": dimensions, "got": len(vector)},
                )

        return vectors, _embed_usage(counted, name, items)

    # -- naming the failures --------------------------------------------------

    def classify_error(self, exc: Exception) -> str:
        """A Gemini failure as one of our `ErrorClass` names.

        Never raises. It runs on the failure path, and an exception here would
        lose the failure it was called to describe.
        """
        try:
            return str(_classify(exc))
        except Exception:  # a classifier that throws is worse than a vague one
            return str(ErrorClass.UNKNOWN)

    def supports_prefix_caching(self) -> bool:
        """False, and not because Gemini has no caching.

        Gemini has two kinds. *Implicit* caching gives 2.5 models a discount when
        a repeated prefix happens to hit, with no way to ask for it and no
        promise that it will. *Explicit* caching is a separate resource with its
        own lifecycle: `cachedContents.create` with the model, the shared prefix
        and a TTL, a handle back, `cachedContent` on every request that wants it,
        a minimum prefix of roughly a thousand tokens, storage billed by the
        hour, a new cache for every model version, and a delete when you are
        done.

        Neither is "a stable prefix is cheaper on the next call for free", which
        is what this method promises the router. Saying True would have the
        router treat a prefix it never registered as already paid for.

        Turning it on would mean: caching the catalogue prefix per model version
        at startup, storing the handle, refreshing it before the TTL runs out,
        deleting it on shutdown, and falling back to an uncached request when the
        handle has expired — a lifecycle, not a flag. Until that exists, False.
        """
        return False

    # -- building the request -------------------------------------------------

    def _body(
        self,
        model: str,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None,
        max_output_tokens: int,
        want_json: bool,
    ) -> dict[str, Any]:
        """The REST request body, which the SDK path then renames into its own.

        The system prompt goes in `systemInstruction`, on its own, where it
        belongs: it is a different thing from the user's question and Gemini
        treats it as one.
        """
        instruction = (system or "").strip()
        generation: dict[str, Any] = {"maxOutputTokens": max_output_tokens}

        if want_json:
            generation["responseMimeType"] = "application/json"
            translated = self._translate(schema, model)
            if translated is not None and translated.usable:
                generation["responseSchema"] = translated.schema
            elif translated is not None:
                # Nothing Gemini can enforce, so the shape goes into the words.
                instruction = f"{instruction}\n\n{translated.prompt_hint()}".strip()
            else:
                instruction = f"{instruction}{_JSON_NUDGE}".strip()

        thinking = self._thinking(model)
        if thinking is not None:
            generation["thinkingConfig"] = thinking

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        if instruction:
            body["systemInstruction"] = {"parts": [{"text": instruction}]}
        return body

    def _translate(self, schema: dict[str, Any] | None, model: str) -> Translation | None:
        """The caller's JSON Schema in Gemini's dialect, with a line in the log.

        A schema that cannot be enforced is worth a warning, because otherwise
        nobody ever finds out that the planner's grammar stopped being checked on
        this provider. Once per reason, though: the schemas are static, so the
        same warning on every request would be a wall of the same sentence.
        """
        if not schema:
            return None

        translated = translate(schema)
        if not translated.usable:
            if translated.reason not in self._warned:
                self._warned.add(translated.reason)
                log.warning(
                    "gemini.schema_not_enforceable",
                    model=model,
                    reason=translated.reason,
                    notes=translated.notes[:5],
                )
        elif translated.dropped:
            log.debug(
                "gemini.schema_relaxed",
                model=model,
                dropped=translated.dropped,
                notes=translated.notes[:5],
            )
        return translated

    def _thinking(self, model: str) -> dict[str, Any] | None:
        """How much of the answer budget the model may spend on thinking.

        Zero for the flash models, which is what `GEMINI_THINKING_BUDGET`
        defaults to: the planner is filling in a schema, not solving a puzzle,
        and thoughts come out of the same allowance the answer does.

        `gemini-2.5-pro` cannot turn thinking off — a budget of 0 is a 400 — so
        it is sent nothing and left to decide for itself. Models before 2.5 have
        no `thinkingConfig` at all and would reject the field.
        """
        if "2.5" not in model:
            return None
        budget = _int_setting("GEMINI_THINKING_BUDGET", 0)
        if "pro" in model:
            return {"thinkingBudget": budget} if budget > 0 else None
        return {"thinkingBudget": max(0, budget)}

    def _max_output_tokens(self, asked: int, model: str, *, streaming: bool) -> int:
        """The caller's budget, floored, plus room for thinking when it is on.

        `max_tokens` is written for a model where the number buys answer. On a
        thinking model the thoughts are spent from the same allowance, so a
        literal 1200 can come back empty with `MAX_TOKENS` — an answer that never
        existed, reported as an answer that was cut off. The floor is what stops
        that being the common case, and the headroom is what stops it wherever
        thinking is on and its size is not ours to choose: `-pro`, and any model
        newer than the ones named here.
        """
        floor = (
            _int_setting("GEMINI_STREAM_MAX_TOKENS", 8192)
            if streaming
            else _int_setting("GEMINI_MAX_TOKENS", 2048)
        )
        budget = max(int(asked or 0), floor)

        thinking = self._thinking(model)
        asked_for = int(thinking.get("thinkingBudget", 0)) if thinking else 0
        if asked_for > 0:
            return budget + asked_for
        if thinking is not None:  # thinking is off, so the whole budget is answer
            return budget
        if any(model.startswith(prefix) for prefix in _NON_THINKING):
            return budget
        return budget + THINKING_HEADROOM

    # -- logging --------------------------------------------------------------

    def _log_failure(self, stage: str, model: str, exc: Exception) -> None:
        """One line per failure, before it goes back to the router.

        The router logs the decision it makes — fall back, retry, give up. This
        logs what the vendor actually said, which is the part you want when the
        decision was wrong.
        """
        if isinstance(exc, LLMError):
            log.warning(
                "gemini.failed",
                stage=stage,
                model=model,
                error_class=str(exc.error_class),
                reason=exc.message,
                **{k: v for k, v in exc.details.items() if k in ("status", "retry_after_s")},
            )
            return
        log.warning(
            "gemini.failed",
            stage=stage,
            model=model,
            error=type(exc).__name__,
            detail=str(exc)[:300],
        )


# ------------------------------------------------------- reading what came back
#
# One set of readers for both transports. The SDK's objects carry the same field
# names as the REST JSON, in snake case instead of camel, so every read names
# both spellings and `_field` takes whichever is there.


def _field(node: Any, *names: str) -> Any:
    """A field off a dict or an object, under any of the names given."""
    if node is None:
        return None
    if isinstance(node, dict):
        for name in names:
            if name in node:
                return node[name]
        return None
    for name in names:
        value = getattr(node, name, None)
        if value is not None:
            return value
    return None


def _enum_name(value: Any) -> str:
    """An enum member, a string, or None as a plain upper-case name.

    The SDK hands back `FinishReason.MAX_TOKENS`; REST hands back
    `"MAX_TOKENS"`. Both need to end up as the same word.
    """
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).rsplit(".", 1)[-1].strip().upper()


def _candidates(response: Any) -> Iterable[Any]:
    return _field(response, "candidates") or []


def _text_of(response: Any) -> str:
    """The answer text of one response or one streamed chunk.

    Parts marked `thought` are skipped. They are the model's reasoning, they are
    not the answer, and letting them through would put prose in the middle of a
    JSON plan.
    """
    pieces: list[str] = []
    for candidate in _candidates(response):
        for part in _field(_field(candidate, "content"), "parts") or []:
            if _field(part, "thought"):
                continue
            text = _field(part, "text")
            if isinstance(text, str) and text:
                pieces.append(text)
    return "".join(pieces)


def _raw_finish(response: Any) -> str:
    """Gemini's own finish reason, in its own words. "" when it has not stopped."""
    for candidate in _candidates(response):
        reason = _enum_name(_field(candidate, "finishReason", "finish_reason"))
        if reason and reason != "FINISH_REASON_UNSPECIFIED":
            return reason
    return ""


def _finish_of(response: Any) -> str:
    """The finish reason in our words. `router.complete_json` reads "length"."""
    reason = _raw_finish(response)
    if not reason:
        return "stop"
    return _FINISH.get(reason, reason.lower())


def _block_reason(response: Any) -> str:
    """Why the *prompt* was refused, if it was. Empty when it was not.

    A blocked prompt comes back with no candidates at all, so this is the only
    place the refusal is visible.
    """
    feedback = _field(response, "promptFeedback", "prompt_feedback")
    return _enum_name(_field(feedback, "blockReason", "block_reason"))


def _refusal_words(reason: str) -> str:
    """What to tell a person when Gemini declines."""
    if reason == "RECITATION":
        return "The model stopped because the answer was reproducing its training data."
    if reason == "SPII":
        return "The model declined because the answer would contain personal identifiers."
    return "The model declined to answer that."


def _refuse_if_filtered(response: Any, model: str, stage: str, usage: Usage) -> None:
    """Raise when the answer is a refusal rather than an answer."""
    blocked = _block_reason(response)
    finish = _raw_finish(response)
    reason = blocked or (finish if finish in _REFUSALS else "")
    if not reason:
        return
    raise LLMError(
        ErrorClass.CONTENT_FILTERED,
        _refusal_words(reason),
        provider=PROVIDER_NAME,
        model=model,
        details={"stage": stage, "reason": reason},
        usage=usage,
    )


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _usage_of(response: Any, model: str) -> Usage:
    return _usage_from(_field(response, "usageMetadata", "usage_metadata"), model)


def _usage_from(meta: Any, model: str) -> Usage:
    """`usageMetadata` as our `Usage`.

    Thinking tokens are added to the completion side. Gemini reports them apart
    from `candidatesTokenCount`, and they are billed at the output rate, so
    leaving them out would make a thinking model look half price.

    `cachedContentTokenCount` is a subset of the prompt count, not an addition
    to it — the same convention `Usage` documents — so it is clamped rather than
    trusted.
    """
    prompt = _int(_field(meta, "promptTokenCount", "prompt_token_count"))
    completion = _int(_field(meta, "candidatesTokenCount", "candidates_token_count"))
    thoughts = _int(_field(meta, "thoughtsTokenCount", "thoughts_token_count"))
    cached = _int(_field(meta, "cachedContentTokenCount", "cached_content_token_count"))

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion + thoughts,
        model=f"{PROVIDER_NAME}:{model}",
        provider=PROVIDER_NAME,
        usd=0.0,  # priced by app.llm.usage, which has the table
        cached_prompt_tokens=min(cached, prompt),
    )


def _embed_usage(counted: int, model: str, texts: Sequence[str]) -> Usage:
    """What an embedding call cost, counted when Gemini says and estimated when not.

    `batchEmbedContents` does not report token counts. An estimate is used
    instead — four characters to a token — because the alternative is a zero,
    and a zero would make a full re-embed of the mirror look free. It is a cost
    *signal*, in the same spirit as the price table in `app.llm.usage`, and it
    is not the bill.
    """
    tokens = counted or sum(max(1, math.ceil(len(text) / 4)) for text in texts)
    return Usage(
        prompt_tokens=tokens,
        completion_tokens=0,
        model=f"{PROVIDER_NAME}:{model}",
        provider=PROVIDER_NAME,
        usd=0.0,
        cached_prompt_tokens=0,
    )


def _unit(vector: list[float]) -> list[float]:
    """A vector scaled to length one.

    gemini-embedding-001 returns unit vectors only at its native 3072. Ask for
    1536 and the values are meaningful but the length is not one, which matters
    the moment anything compares them by inner product or assumes cosine and
    dot product are the same number.
    """
    length = math.sqrt(sum(v * v for v in vector))
    if length == 0.0 or math.isclose(length, 1.0, rel_tol=1e-6):
        return vector
    return [v / length for v in vector]


# ------------------------------------------------------------ naming a failure


def _classify(exc: Exception) -> ErrorClass:
    """The mapping. Called only by `GeminiProvider.classify_error`, which guards it."""
    if isinstance(exc, LLMError):
        return exc.error_class
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return ErrorClass.TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        return _class_for(response.status_code, *_status_and_message(_safe_json(response)))
    if isinstance(exc, httpx.TransportError):
        # Connection refused, DNS, a reset mid-body: the request never got an
        # answer, so another one might.
        return ErrorClass.TRANSIENT

    # The SDK's `google.genai.errors.APIError` carries `.code` (the HTTP status),
    # `.status` (the google.rpc name) and `.message`. Read them off by name
    # rather than importing the SDK: this runs on the failure path, in a process
    # that may not have the SDK installed at all.
    status = _int(getattr(exc, "code", None) or getattr(exc, "status_code", None)) or None
    name = str(getattr(exc, "status", "") or "")
    message = str(getattr(exc, "message", "") or exc)
    if status or name:
        return _class_for(status, name, message)

    # Nothing structured left. The status name is usually still in the text.
    return _class_for(None, _status_name_in(message), message)


def _class_for(status: int | None, name: str, message: str) -> ErrorClass:
    """One HTTP status plus one `google.rpc` status name, as our class.

    Two places this deliberately parts company with `errors.class_for_status`:

    * **401 is `AUTH_REVOKED`, not `AUTH_EXPIRED`.** A Gemini API key does not
      expire on a schedule; it is wrong, or it has been turned off. Nothing
      waits for it to come back.
    * **504 / `DEADLINE_EXCEEDED` is `TRANSIENT`, not `TIMEOUT`.** It is Google's
      deadline, not ours, and it earns the one in-provider retry that a blip
      deserves. Our own clock running out is still `TIMEOUT`, raised by the
      router.
    """
    name = (name or "").strip().upper()
    text = " ".join((message or "").lower().split())

    if name == "RESOURCE_EXHAUSTED" or status == 429:
        return ErrorClass.QUOTA_EXHAUSTED if _is_out_of_quota(text) else ErrorClass.RATE_LIMITED
    if name in ("UNAVAILABLE", "DEADLINE_EXCEEDED") or status in (503, 504):
        return ErrorClass.TRANSIENT
    if name in ("UNAUTHENTICATED", "PERMISSION_DENIED") or status in (401, 403):
        return ErrorClass.AUTH_REVOKED
    if name in ("INVALID_ARGUMENT", "FAILED_PRECONDITION", "OUT_OF_RANGE") or status == 400:
        if any(word in text for word in _BAD_KEY):
            return ErrorClass.AUTH_REVOKED
        return ErrorClass.INVALID
    if name == "NOT_FOUND" or status == 404:
        return ErrorClass.NOT_FOUND
    if name in ("INTERNAL", "UNKNOWN", "ABORTED", "CANCELLED") or (status or 0) >= 500:
        return ErrorClass.TRANSIENT
    return class_for_status(status)


def _is_out_of_quota(text: str) -> bool:
    """Whether a 429 is a burst limit or a wall.

    A per-minute limit clears while the user is still waiting; a per-day one or
    an unpaid account does not. Both fall back to the other provider — the
    difference is what the operator reads afterwards, and whether this provider
    is worth asking again in a second.

    Ambiguous cases read as a rate limit, which is the gentler of the two.
    """
    squashed = text.replace(" ", "")
    if any(word.replace(" ", "") in squashed for word in _PER_MINUTE):
        return False
    return any(word.replace(" ", "") in squashed for word in _OUT_OF_QUOTA)


_STATUS_NAMES: Final[tuple[str, ...]] = (
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "UNAVAILABLE",
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
    "INVALID_ARGUMENT",
    "FAILED_PRECONDITION",
    "OUT_OF_RANGE",
    "NOT_FOUND",
    "INTERNAL",
    "ABORTED",
    "CANCELLED",
    "UNKNOWN",
)


def _status_name_in(text: str) -> str:
    """The `google.rpc` status name inside a message, when that is all there is."""
    upper = (text or "").upper()
    for name in _STATUS_NAMES:
        if name in upper:
            return name
    return ""


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return {"error": {"message": response.text[:500]}}


def _status_and_message(payload: Any) -> tuple[str, str]:
    """The `status` and `message` out of Google's error envelope."""
    error = _field(payload, "error") or {}
    return str(_field(error, "status") or ""), str(_field(error, "message") or "")


def _raise_for_status(response: httpx.Response, model: str) -> None:
    """A non-2xx REST answer as an `LLMError` that already knows its class.

    Raised here rather than as an `httpx.HTTPStatusError` so the retry hint
    Google sends survives into the log line that explains the fallback. The
    router does not sleep on a rate limit — it moves to the next provider — but
    the number is what tells an operator whether that was one busy second or an
    exhausted account.
    """
    if response.status_code < 400:
        return

    payload = _safe_json(response)
    name, message = _status_and_message(payload)
    error_class = _class_for(response.status_code, name, message)

    details: dict[str, Any] = {"status": response.status_code, "google_status": name or None}
    hint = _retry_after(response, payload)
    if hint is not None:
        details["retry_after_s"] = hint

    raise LLMError(
        error_class,
        message.strip() or f"Gemini answered {response.status_code}.",
        provider=PROVIDER_NAME,
        model=model,
        status=response.status_code,
        details={k: v for k, v in details.items() if v is not None},
    )


def _retry_after(response: httpx.Response, payload: Any) -> float | None:
    """How long Google wants us to wait, from the header or from `RetryInfo`."""
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass

    error = _field(payload, "error") or {}
    for detail in _field(error, "details") or []:
        delay = _field(detail, "retryDelay", "retry_delay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------- assembly


def _sdk_available() -> bool:
    """Whether `google-genai` is installed.

    Deliberately not an import at the top of the file: `router._build` imports
    this module to find out whether Gemini is usable at all, and an ImportError
    there reads as "the provider is not installed" — which would take the REST
    path down with the SDK it does not need.
    """
    from importlib.util import find_spec

    try:
        return find_spec("google.genai") is not None
    except (ImportError, ValueError):
        return False


def provider() -> GeminiProvider:
    """What `app.llm.router` calls to build the provider."""
    return GeminiProvider()


__all__ = ["DEFAULT_EMBED_MODEL", "DEFAULT_MODEL", "PROVIDER_NAME", "GeminiProvider", "provider"]
