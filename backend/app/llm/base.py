"""What a provider is, and the values providers hand back.

A provider is an adapter around one vendor's SDK. It knows how to turn our four
calls into that vendor's request shape and how to turn that vendor's exceptions
into our `ErrorClass` names. It knows nothing about fallback chains, circuit
breakers, retries, pricing or usage accounting — all of that lives in
`app.llm.router` and `app.llm.usage`, once, for every provider.

Everything a provider needs from us is in this module, so a new provider is one
file that imports `app.llm.base` and nothing else of ours.

Writing a provider
------------------

Implement `Provider`. Four rules that are easy to get wrong:

1. **`model` is the bare name.** By the time the router calls you it has already
   chosen the provider, so you get ``"gpt-4.1-mini"``, never
   ``"openai:gpt-4.1-mini"``.
2. **Do not retry, do not sleep, do not catch and swallow.** Raise. The router
   owns the retry budget and it cannot honour a deadline you are sleeping
   through.
3. **`stream_json` and `stream_text` are called, not awaited.** They return an
   async iterator straight away. Return a `TextStream` so the token usage can
   come back after the last chunk — a bare async generator cannot carry it.
4. **`classify_error` never raises.** It is called on the failure path; an
   exception there loses the original failure.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.llm.errors import ErrorClass, LLMError

# Spellings people reach for that mean a provider we have, mapped to the one
# canonical name. The prefix is the only thing that picks a provider, so it is
# worth being forgiving about how it is written rather than failing on "google:"
# when "gemini:" was meant.
PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "googleai": "gemini",
    "google-ai": "gemini",
    "google_genai": "gemini",
    "genai": "gemini",
    "vertex": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def known_providers() -> list[str]:
    """Every provider name the parser will accept, canonical spellings only."""
    return sorted(set(PROVIDER_ALIASES.values()))


def add_provider_name(name: str) -> str:
    """Teach the parser a provider registered at run time.

    `router.register` calls this. Without it a freshly registered provider
    would be reachable by the router and rejected by the parser, which is a
    confusing way to find out that "the registry" was two registries.
    """
    key = (name or "").strip().lower()
    if not key:
        raise LLMError(ErrorClass.INVALID, "A provider needs a name.")
    PROVIDER_ALIASES.setdefault(key, key)
    return key


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A provider and a model name, which is all "openai:gpt-4.1-mini" is.

    Parsing lives in `parse` and nowhere else. If you find yourself splitting a
    model string on ``":"`` somewhere in the app, that is the bug.
    """

    provider: str
    name: str

    @classmethod
    def parse(cls, s: str, default_provider: str = "openai") -> ModelRef:
        """``"gemini:gemini-2.5-flash"`` or a bare name plus a default provider.

        A bare name — no colon — belongs to ``default_provider``, which is what
        `LLM_DEFAULT_PROVIDER` is for. An unknown prefix is an error rather than
        a silent fall-through to the default: a mistyped or unconfigured prefix
        quietly becoming an OpenAI call is the kind of thing you find out about
        in a bill.
        """
        raw = (s or "").strip()
        if not raw:
            raise LLMError(ErrorClass.INVALID, "No model was given.")

        if ":" in raw:
            prefix, _, name = raw.partition(":")
            key = prefix.strip().lower()
            provider = PROVIDER_ALIASES.get(key)
            if provider is None:
                raise LLMError(
                    ErrorClass.INVALID,
                    f"There is no provider called {prefix!r}.",
                    model=raw,
                    details={"known": known_providers()},
                )
        else:
            name = raw
            key = (default_provider or "openai").strip().lower()
            provider = PROVIDER_ALIASES.get(key)
            if provider is None:
                raise LLMError(
                    ErrorClass.INVALID,
                    f"LLM_DEFAULT_PROVIDER is set to {default_provider!r}, which is not a provider.",
                    model=raw,
                    details={"known": known_providers()},
                )

        name = name.strip()
        if not name:
            raise LLMError(
                ErrorClass.INVALID,
                f"{raw!r} names a provider but no model.",
                provider=provider,
                model=raw,
            )
        return cls(provider=provider, name=name)

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


@dataclass
class Usage:
    """What one call cost, in tokens and in dollars.

    ``model`` is the full reference — ``"openai:gpt-4.1-mini"`` — not the bare
    name. One canonical spelling for a model everywhere: in config, in this
    field, and in the ``embed_model`` column of the mirror tables.

    ``cached_prompt_tokens`` is a *subset* of ``prompt_tokens``, not an
    addition to it. That is how both vendors report it, and pricing here
    follows the same convention.
    """

    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str
    usd: float
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "cached_prompt": self.cached_prompt_tokens,
            "model": self.model,
            "provider": self.provider,
            "usd": round(self.usd, 6),
        }

    @classmethod
    def empty(cls, ref: ModelRef | str) -> Usage:
        """A zeroed record, for a call that failed before it burned anything."""
        model = str(ref)
        provider = ref.provider if isinstance(ref, ModelRef) else model.partition(":")[0]
        return cls(0, 0, model, provider, 0.0)


@dataclass
class ChatResult:
    """One non-streamed completion.

    ``parsed`` is the JSON object when the provider could parse it itself —
    several SDKs hand back structured output directly, and re-parsing text they
    already parsed is wasted work. When it is ``None`` the router parses
    ``text`` with `parse_json_object`.
    """

    text: str
    parsed: dict[str, Any] | None
    usage: Usage
    finish_reason: str


class TextStream:
    """An async iterator of text chunks that can also carry the token usage.

    Vendors report token counts at the *end* of a stream, after the last chunk.
    An async generator object has no ``__dict__``, so a provider cannot simply
    hang the numbers off one. This is the box that makes it possible::

        def stream_text(self, model, system, user):
            out = TextStream()

            async def run():
                stream = await client.responses.stream(...)
                async for chunk in stream:
                    yield chunk.text
                out.usage = Usage(...)      # set it last, from inside

            return out.feed(run())

    The router reads ``.usage`` once the stream ends, however it ended. A
    provider that returns a plain async generator still works — its tokens just
    do not get counted, which is worse but not broken.
    """

    __slots__ = ("_source", "usage")

    def __init__(self, source: AsyncIterator[str] | None = None) -> None:
        self.usage: Usage | None = None
        self._source: AsyncIterator[str] | None = source

    def feed(self, source: AsyncIterator[str]) -> TextStream:
        """Attach the chunk source and return self, so providers can one-line it."""
        self._source = source
        return self

    def __aiter__(self) -> TextStream:
        return self

    async def __anext__(self) -> str:
        if self._source is None:
            raise StopAsyncIteration
        return await self._source.__anext__()

    async def aclose(self) -> None:
        """Shut the underlying generator down. Safe to call more than once."""
        source, self._source = self._source, None
        closer = getattr(source, "aclose", None)
        if closer is not None:
            await closer()


@runtime_checkable
class Provider(Protocol):
    """The whole of what the router needs from a vendor."""

    name: str

    async def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> ChatResult:
        """One request, one JSON object back."""
        ...

    def stream_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """The same JSON, chunk by chunk as it arrives. Joining every chunk
        gives the complete text; parsing is the caller's business."""
        ...

    def stream_text(self, model: str, system: str, user: str) -> AsyncIterator[str]:
        """Prose, chunk by chunk."""
        ...

    async def embed(
        self,
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> tuple[list[list[float]], Usage]:
        """One vector per string, in the order given, all of ``dimensions`` long.

        Raise rather than return a short vector or a different length. A wrong
        dimension reaching pgvector is a corrupted mirror, not a bad answer.
        """
        ...

    def classify_error(self, exc: Exception) -> str:
        """This SDK's exception as one of our `ErrorClass` names. Never raises."""
        ...

    def supports_prefix_caching(self) -> bool:
        """True when a stable prompt prefix is cheaper on a repeat call.

        Worth knowing because it changes how prompts should be ordered — the
        fixed catalogue first, the user's question last — and because cached
        prompt tokens are priced differently.
        """
        ...


def parse_json_object(raw: str, *, where: str = "complete_json", model: str = "") -> dict[str, Any]:
    """Model output as a JSON object, repairing the damage models usually do.

    Fenced code blocks and a sentence of preamble are both common enough to be
    worth handling. Anything past that is `INVALID`: the same prompt will
    produce the same mess on a retry, so failing loudly beats paying twice.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    if not text:
        raise LLMError(
            ErrorClass.INVALID,
            "The model returned nothing.",
            model=model,
            details={"stage": where},
        )

    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(
                ErrorClass.INVALID,
                "The model did not return JSON.",
                model=model,
                details={"stage": where, "head": text[:200]},
            ) from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(
                ErrorClass.INVALID,
                "The model returned malformed JSON.",
                model=model,
                details={"stage": where, "head": text[:200]},
            ) from exc

    if not isinstance(value, dict):
        raise LLMError(
            ErrorClass.INVALID,
            "The model returned JSON that is not an object.",
            model=model,
            details={"stage": where, "type": type(value).__name__},
        )
    return value


__all__ = [
    "PROVIDER_ALIASES",
    "ChatResult",
    "ModelRef",
    "Provider",
    "TextStream",
    "Usage",
    "add_provider_name",
    "known_providers",
    "parse_json_object",
]
