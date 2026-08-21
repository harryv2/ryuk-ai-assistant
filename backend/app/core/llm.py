"""The old front door to the model layer. The layer itself now lives in `app.llm`.

Everything real moved to `app/llm/` — `base.py` (what a provider is),
`router.py` (who gets asked, and what happens when they fail), `usage.py` (what
it cost) and one adapter per vendor under `providers/`. This module stayed
behind so that the twelve places already importing it keep working:

    from app.core import llm
    plan = await llm.complete_json(system, user, max_tokens)

New code should import `app.llm` directly and use the four calls in their own
spelling. Nothing here adds behaviour; it only keeps the old argument order and
the old exception type.

Three differences worth knowing, because they are the whole point of the move:

* **No provider is named anywhere.** The old module imported `openai` at the
  top and could not have done anything else. This one imports nothing until a
  call is made, and which vendor serves that call is `LLM_MODEL`. See
  `docs/MODELS.md`.
* **Failures come out as `AppError`, exactly as they used to.** The router
  raises `app.llm.errors.LLMError`, which carries the provider, the model and
  the failure class; this module converts it at the boundary because every
  caller here already catches `AppError`. Reach for `app.llm` when you want the
  class.
* **`track_usage()` is still a plain `with`.** `app.llm.usage.track_usage()` is
  an async context manager; `app.orchestrator.runner` uses a synchronous one and
  there is no reason to rewrite it, so the version here sets the same contextvar
  from a normal `with` block and hands back the same numbers in the old shape.

The embedding rule, restated because this is where it used to be enforced
------------------------------------------------------------------------

`embed` never falls back to another model. Chat models are interchangeable and
embedding models are not: every `VECTOR(1536)` column in the mirror was filled
by one specific model, and a vector from a different one is not comparable with
it even at the same width. The cosine distance is still a number; the number is
nonsense, and nothing about it looks wrong. So the embedding chain is one model
long, each mirror row records the model that embedded it in `embed_model`, and
changing `EMBED_MODEL` means re-embedding the whole mirror with
`backend/scripts/reembed.py`. Gemini's `gemini-embedding-001` will produce 1536
dimensions on request, so the column type survives the switch — the rows do not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import Token
from dataclasses import dataclass, field
from typing import Any, Final

from app.core.errors import AppError
from app.core.logging import get_logger
from app.llm import base, errors, router
from app.llm import usage as usage_tracker
from app.llm.base import ChatResult, ModelRef, Provider, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError
from app.llm.router import (
    assert_same_embed_model,
    breaker_reset,
    embed_dimensions,
    embed_model_id,
    get_provider,
    register,
    resolve,
)
from app.llm.usage import Ledger

log = get_logger(__name__)

# The contextvar `app.llm.usage` files calls under. Reached into on purpose:
# that module offers only an *async* context manager to set it, and the runner
# needs a synchronous one. Everything else here goes through the public API.
_ledger_var = usage_tracker._ledger

# Kept so `from app.core.llm import EMBEDDING_DIM` still resolves. The live
# value is `EMBED_DIMENSIONS`, read through `embedding_dim()`.
EMBEDDING_DIM: Final[int] = 1536

# The old flat price table, rebuilt from the per-provider ones in
# `app.llm.usage`. Bare names only, as it always was; `cost_usd` below takes
# either spelling.
PRICING: Final[dict[str, tuple[float, float]]] = {
    name: rate for table in usage_tracker.PRICES.values() for name, rate in table.items()
}


# -- errors ------------------------------------------------------------------


def _as_app_error(exc: BaseException, where: str) -> AppError:
    """Whatever failed, as the error type this module has always raised."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, LLMError):
        app_error = exc.to_app_error()
        app_error.details.setdefault("stage", where)
        return app_error
    return AppError(
        "INTERNAL",
        "The model call failed.",
        details={"stage": where, "cause": type(exc).__name__, "detail": str(exc)[:500]},
    )


# -- usage accounting --------------------------------------------------------


@dataclass
class TokenUsage:
    """What one run spent, in the shape `runs.token_usage` has always stored.

    A thin reading of `app.llm.usage.Ledger`. The ledger holds one entry per
    attempt — including the attempts that failed — so a fallback firing is
    visible in `calls` and in `usd` rather than free.

    `prompt` and `completion` count chat tokens; embedding tokens are their own
    field, as they were, because they are priced at the input rate and mixing
    them into `prompt` makes a planner call look enormous.
    """

    ledger: Ledger = field(default_factory=Ledger)

    # -- the old attribute surface -------------------------------------------

    @property
    def prompt(self) -> int:
        return sum(e.usage.prompt_tokens for e in self.ledger.entries if e.kind == "chat")

    @property
    def completion(self) -> int:
        return sum(e.usage.completion_tokens for e in self.ledger.entries if e.kind == "chat")

    @property
    def embedding(self) -> int:
        return sum(e.usage.total_tokens for e in self.ledger.entries if e.kind == "embed")

    @property
    def calls(self) -> int:
        return self.ledger.calls

    @property
    def usd(self) -> float:
        return self.ledger.usd

    @property
    def model(self) -> str | None:
        """The model that actually answered — the last chat call that worked,
        not the first one that was rate limited."""
        chosen = self.ledger.totals()["model"]
        return str(chosen) if chosen else None

    @property
    def provider(self) -> str | None:
        chosen = self.ledger.totals()["provider"]
        return str(chosen) if chosen else None

    @property
    def models(self) -> dict[str, int]:
        """Tokens per model reference, every attempt counted."""
        out: dict[str, int] = {}
        for entry in self.ledger.entries:
            out[entry.usage.model] = out.get(entry.usage.model, 0) + entry.usage.total_tokens
        return out

    def add(self, model: str, prompt: int, completion: int, embedding: int = 0) -> None:
        """Record a call by hand. Only tests and the odd script need this."""
        from app.config import settings

        default = str(getattr(settings, "LLM_DEFAULT_PROVIDER", "") or "openai")
        ref = ModelRef.parse(model, default)
        kind: usage_tracker.Kind = "embed" if embedding else "chat"
        self.ledger.add(
            Usage(
                prompt_tokens=prompt or embedding,
                completion_tokens=completion,
                model=str(ref),
                provider=ref.provider,
                usd=0.0,
            ),
            kind=kind,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "embedding": self.embedding,
            "calls": self.calls,
            "model": self.model,
            "provider": self.provider,
            "models": self.models,
            "usd": round(self.usd, 6),
        }


def _wrap(ledger: Ledger | None) -> TokenUsage:
    return TokenUsage(ledger=ledger if ledger is not None else Ledger())


def start_usage() -> Token:
    """Begin a fresh tally. Returns the token to reset with."""
    return _ledger_var.set(Ledger())


def reset_usage(token: Token) -> None:
    _ledger_var.reset(token)


def current_usage() -> TokenUsage | None:
    """The tally for this context, or None when nobody started one."""
    ledger = usage_tracker.current()
    return None if ledger is None else TokenUsage(ledger=ledger)


def usage_dict() -> dict[str, Any]:
    """The tally as it goes into ``runs.token_usage``."""
    return _wrap(usage_tracker.current()).to_dict()


@contextmanager
def track_usage() -> Iterator[TokenUsage]:
    """Count every model call made inside this block.

    ::

        with llm.track_usage() as usage:
            plan = await llm.complete_json(system, user, 1200)
        run.token_usage = usage.to_dict()

    Synchronous on purpose — `app.orchestrator.runner` wraps a whole run in a
    plain `with`. `app.llm.usage.track_usage()` is the async one; both drive the
    same contextvar, so a call made under either is counted by either.
    """
    ledger = Ledger()
    token = _ledger_var.set(ledger)
    try:
        yield TokenUsage(ledger=ledger)
    finally:
        _ledger_var.reset(token)


def cost_usd(model: str, prompt: int, completion: int) -> float:
    """Dollar cost of one call. Unknown models cost 0.

    Takes either spelling — ``"openai:gpt-4.1-mini"`` or the bare
    ``"gpt-4.1-mini"``. Prefer the full reference: a bare name has to be looked
    for in every provider's table, and two vendors could name a model the same
    thing.
    """
    return usage_tracker.price(model, prompt, completion)


# -- the client, the models, the shutdown hook -------------------------------


def get_client() -> Any:
    """The shared `AsyncOpenAI`, for the one or two places that still want it.

    Only meaningful when OpenAI is one of the configured providers. Everything
    that goes through the four calls below should not care which SDK is behind
    them, which is the point of the package this now forwards to.
    """
    try:
        provider = get_provider("openai")
        client = getattr(provider, "client", None)
        if client is None:
            raise AppError("INTERNAL", "The OpenAI provider has no client to hand out.")
        return client()
    except LLMError as exc:
        raise _as_app_error(exc, "get_client") from exc


async def close() -> None:
    """Shut every built provider down. Called from the app's lifespan hook."""
    await router.close()


def chat_model() -> str:
    """The model that serves the planner, as a full reference.

    Full reference, not the bare name it used to be: one canonical spelling for
    a model in config, in the usage ledger and in the `embed_model` column beats
    three spellings that have to be normalised before they can be compared.
    """
    return str(resolve(None, router.PURPOSE_PLAN)[0])


def prose_model() -> str:
    """The model that writes the answer. `LLM_PROSE_MODEL`, or `LLM_MODEL`."""
    return str(resolve(None, router.PURPOSE_PROSE)[0])


def embed_model() -> str:
    """The embedding model, as a full reference.

    This exact string is what goes in each mirror row's `embed_model` column.
    """
    return embed_model_id()


def embedding_dim() -> int:
    """Vector width. Must match ``VECTOR(1536)`` in the schema."""
    return embed_dimensions()


# -- JSON repair -------------------------------------------------------------


def parse_json(raw: str, where: str = "complete_json") -> dict[str, Any]:
    """Parse model output as a JSON object, repairing the usual damage.

    `app.orchestrator.route.PlanScanner.finish` calls this and catches
    `AppError` to salvage a truncated plan, so the conversion here is load
    bearing rather than tidiness.
    """
    try:
        return base.parse_json_object(raw, where=where)
    except LLMError as exc:
        raise _as_app_error(exc, where) from exc


# -- the four calls ----------------------------------------------------------
#
# The argument order is the old one: `max_tokens` third and positional, because
# `app.orchestrator.route` passes it that way. `app.llm.router` puts `schema`
# there instead. `_third` sorts out which one arrived so a call written against
# either spelling lands correctly.


def _third(value: Any, max_tokens: int) -> tuple[dict[str, Any] | None, int]:
    """The third positional argument, as (schema, max_tokens)."""
    if isinstance(value, dict):
        return value, max_tokens
    if value is None:
        return None, max_tokens
    return None, int(value)


async def complete_json(
    system: str,
    user: str,
    max_tokens: int | dict[str, Any] = 1200,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    schema: dict[str, Any] | None = None,
    purpose: str = router.PURPOSE_PLAN,
) -> dict[str, Any]:
    """One request, one JSON object back. This is the planner's only call.

    `temperature` is accepted and ignored. It was an OpenAI-only knob; two of
    the three providers either do not have it or reject it outright, so the
    router does not carry it and the providers set their own.
    """
    from_third, limit = _third(max_tokens, 1200)
    try:
        return await router.complete_json(
            system,
            user,
            schema if schema is not None else from_third,
            model,
            limit,
            purpose=purpose,
        )
    except LLMError as exc:
        raise _as_app_error(exc, "complete_json") from exc


async def stream_json(
    system: str,
    user: str,
    max_tokens: int | dict[str, Any] = 1200,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    schema: dict[str, Any] | None = None,
    purpose: str = router.PURPOSE_PLAN,
) -> AsyncIterator[str]:
    """The JSON answer as it arrives, chunk by raw chunk.

    Joining every yield gives the complete JSON text. Reassembly and parsing
    belong to the caller, which is what lets the planner start on step one
    before step four exists.

    Falls back to another provider only until the first chunk has been handed
    over. After that a failure is a failure — two halves of two different plans
    spliced into one object is worse than a visible error.
    """
    from_third, limit = _third(max_tokens, 1200)
    stream = router.stream_json(
        system, user, schema if schema is not None else from_third, model, limit, purpose=purpose
    )
    try:
        async for piece in stream:
            yield piece
    except LLMError as exc:
        raise _as_app_error(exc, "stream_json") from exc


async def stream_text(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    purpose: str = router.PURPOSE_PROSE,
) -> AsyncIterator[str]:
    """The prose answer, streamed. Each yield is a piece of text to display.

    `max_tokens` and `temperature` are accepted and ignored, for the same reason
    as in `complete_json`: the length ceiling for prose is the provider's, and
    there is no temperature to set on some of these models at all.

    Defaults to the prose purpose, so `LLM_PROSE_MODEL` can point the writing at
    a different model from the planning without touching a call site.
    """
    stream = router.stream_text(system, user, model, purpose=purpose)
    try:
        async for piece in stream:
            yield piece
    except LLMError as exc:
        raise _as_app_error(exc, "stream_text") from exc


async def embed(
    texts: Sequence[str],
    *,
    model: str | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """One vector per string, in the order given.

    No fallback chain, ever — see the module docstring. Empty strings get a zero
    vector rather than a request, and anything already in the embedding cache is
    not sent again.
    """
    try:
        return await router.embed(texts, model, use_cache=use_cache)
    except LLMError as exc:
        raise _as_app_error(exc, "embed") from exc


async def embed_one(
    text: str,
    *,
    model: str | None = None,
    use_cache: bool = True,
) -> list[float]:
    """One string, one vector. The probe's single embedding call."""
    try:
        return await router.embed_one(text, model, use_cache=use_cache)
    except LLMError as exc:
        raise _as_app_error(exc, "embed_one") from exc


__all__ = [
    "EMBEDDING_DIM",
    "PRICING",
    "ChatResult",
    "ErrorClass",
    "LLMError",
    "Ledger",
    "ModelRef",
    "Provider",
    "TextStream",
    "TokenUsage",
    "Usage",
    "assert_same_embed_model",
    "base",
    "breaker_reset",
    "chat_model",
    "close",
    "complete_json",
    "cost_usd",
    "current_usage",
    "embed",
    "embed_dimensions",
    "embed_model",
    "embed_model_id",
    "embed_one",
    "embedding_dim",
    "errors",
    "get_client",
    "get_provider",
    "parse_json",
    "prose_model",
    "register",
    "reset_usage",
    "resolve",
    "router",
    "start_usage",
    "stream_json",
    "stream_text",
    "track_usage",
    "usage_dict",
    "usage_tracker",
]
