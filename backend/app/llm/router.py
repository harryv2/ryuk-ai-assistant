"""The four calls the app makes, and everything that decides who answers them.

This module is the point of the package. Nothing outside it knows that OpenAI
exists, or that Gemini does, or that Anthropic does. Call sites say what they
want::

    plan = await llm.complete_json(system, user)
    async for piece in llm.stream_text(system, user):
        ...

and a line of config decides who serves it::

    LLM_MODEL=anthropic:claude-opus-5
    LLM_FALLBACK_MODELS=openai:gpt-4.1-mini

Switching provider is that edit and a restart. No call site changes, because no
call site ever named a provider. The three prefixes are ``openai:``, ``gemini:``
and ``anthropic:`` — `docs/MODELS.md` has the model names and the prices.

How a call is served
--------------------

`resolve` turns a model string and a purpose into a chain: the primary, then the
configured fallbacks, deduplicated. The chain is then walked:

* A provider whose circuit breaker is **open is skipped**, not tried. Five
  consecutive failures opens it for sixty seconds, then one probe call decides
  whether it closes — the same shape as the per-user Google breaker in
  `app.google.retry`, kept in Redis instead of a table because there is no row
  to hang it on.
* Inside one provider, **one extra attempt** on `TRANSIENT` only, full jitter.
  A rate limit does not want the same provider again a second later; it wants a
  different one, which is what the fallback is for.
* Each attempt gets its own timeout, counted as `TIMEOUT`.
* A failure whose class is in **`LLM_FALLBACK_ON`** logs a warning naming both
  models and the class, then moves to the next link.
* A failure whose class is **not** in that set — `INVALID`, `CONTENT_FILTERED`,
  `AUTH_EXPIRED`, `AUTH_REVOKED` — fails immediately, without burning the
  fallback. A malformed request is malformed on every provider; sending it to
  Gemini buys a second, differently-worded rejection and another bill.
* Every attempt, including the ones that failed, is recorded in
  `app.llm.usage`, so a fallback firing shows up as money.

Streaming commits at the first token
------------------------------------

`stream_json` and `stream_text` may fall back freely **until the first chunk
reaches the caller**. After that we are committed: if the stream dies at token
four hundred we raise, and we do not quietly restart on another provider. The
alternative is emitting two partial answers into one chat bubble, where the
second half of an OpenAI sentence is finished by Gemini and nobody can tell why
the answer contradicts itself. A visible failure beats an invisible splice.

Embeddings do not fall back. Ever.
----------------------------------

Chat models are interchangeable. **Embedding models are not.** The pgvector
columns are `VECTOR(1536)` filled by one specific model, and a vector from a
different model is not comparable to them — not because the dimensions differ,
but because the two models put meaning in different places. Cosine distance
between them is a number, and the number is nonsense. It does not error, it
does not look wrong, it just quietly returns the wrong emails.

So:

* `resolve(..., purpose="embed")` returns **one** model. There is no fallback
  chain for embeddings, because a fallback would silently poison the mirror.
* Every mirror row records the model that embedded it, in `embed_model`
  (`sync_messages`, `sync_events`, `sync_files` — `VARCHAR(64) NOT NULL`). Use
  `embed_model_id()` to fill it: it is the same canonical string as
  `EMBED_MODEL`, e.g. ``"openai:text-embedding-3-small"``.
* The search path calls `assert_same_embed_model` before trusting a distance,
  and gets a loud error instead of a plausible wrong answer.
* Changing `EMBED_MODEL` means **re-embedding the whole mirror**. Nothing else
  will do. There is a management command for it that works in batches and
  reports progress.
* Gemini's `gemini-embedding-001` does support `output_dimensionality=1536`, so
  the column type does not change when you switch. The rows still all have to
  be rebuilt. Same shape, different meaning.
* **Anthropic cannot embed at all.** There is no embeddings endpoint in that
  API, so ``EMBED_MODEL=anthropic:...`` is not a slow path or a missing adapter
  — it is a setting that can never work. `app.config` rejects it at startup and
  the provider raises if it is somehow reached anyway. Chat on Anthropic with
  embeddings on OpenAI is the normal, supported arrangement.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from redis.exceptions import RedisError

from app.core.logging import get_logger
from app.llm import usage as usage_tracker
from app.llm.base import (
    ChatResult,
    ModelRef,
    Provider,
    Usage,
    add_provider_name,
    parse_json_object,
)
from app.llm.errors import (
    DEFAULT_FALLBACK_ON,
    ErrorClass,
    LLMError,
    backoff,
    coerce,
    counts_against_health,
)

log = get_logger(__name__)

# What `purpose` may say. The planner and the router call are "plan"; the
# synthesis call that writes the answer is "prose" and may be pointed at a
# different model with LLM_PROSE_MODEL; embeddings are their own thing.
PURPOSE_PLAN: Final[str] = "plan"
PURPOSE_PROSE: Final[str] = "prose"
PURPOSE_EMBED: Final[str] = "embed"

_PROSE_NAMES: Final[frozenset[str]] = frozenset(
    {"prose", "answer", "render", "synthesis", "synthesise", "text"}
)
_EMBED_NAMES: Final[frozenset[str]] = frozenset({"embed", "embedding", "embeddings"})

# Circuit breaker, same shape as the Google one: five consecutive failures opens
# it, one minute later a single probe decides whether it closes.
BREAKER_THRESHOLD: Final[int] = 5
BREAKER_COOLDOWN_S: Final[float] = 60.0
_PROBE_TTL_S: Final[int] = 30
_BREAKER_TTL_S: Final[int] = 3600

# Vendors cap how many strings one embedding request may carry, and the caps
# differ. This sits under both of them, so providers never have to page.
EMBED_BATCH: Final[int] = 96

_DEFAULTS: Final[dict[str, Any]] = {
    "LLM_MODEL": "openai:gpt-4.1-mini",
    "LLM_FALLBACK_MODELS": "gemini:gemini-2.5-flash",
    "LLM_PROSE_MODEL": "",
    "LLM_DEFAULT_PROVIDER": "openai",
    "LLM_FALLBACK_ON": "RATE_LIMITED,TRANSIENT,QUOTA_EXHAUSTED,TIMEOUT",
    "LLM_TIMEOUT_S": 20.0,
    "EMBED_MODEL": "openai:text-embedding-3-small",
    "EMBED_DIMENSIONS": 1536,
}


def _setting(name: str, default: Any = None) -> Any:
    """One setting, with the package's own default when config has not got it.

    Read through here rather than off `settings` directly so this package keeps
    working if it is imported before, or without, the config additions.
    """
    from app.config import settings

    value = getattr(settings, name, None)
    if value is None or value == "":
        fallback = _DEFAULTS.get(name, default)
        return default if fallback is None else fallback
    return value


# ---------------------------------------------------------------- the registry


# A provider module is imported the first time somebody asks for that provider,
# so a process with only an OPENAI_API_KEY never touches the Gemini or Anthropic
# SDK and never needs a key for either. Each entry lists the modules to look in;
# the first one that imports wins.
#
# A provider module must expose one of, tried in this order:
#     PROVIDER      a ready instance
#     provider()    a factory returning one
#     <ClassName>   the class named below, taking no required arguments
_FACTORIES: dict[str, tuple[tuple[str, ...], str]] = {
    "openai": (("app.llm.openai_provider", "app.llm.providers.openai_provider"), "OpenAIProvider"),
    "gemini": (("app.llm.gemini_provider", "app.llm.providers.gemini_provider"), "GeminiProvider"),
    "anthropic": (
        ("app.llm.providers.anthropic_provider", "app.llm.anthropic_provider"),
        "AnthropicProvider",
    ),
}

# The pip package a provider module needs before it will import. Only used to
# finish the sentence when that import fails: "not installed" is true but
# unhelpful to read at three in the morning, and the fix is one line long.
# Gemini is absent on purpose — it talks REST over httpx, which is already a
# hard dependency, so there is no separate SDK to be missing.
_PACKAGES: Final[dict[str, str]] = {
    "openai": "openai",
    "anthropic": "anthropic",
}

PROVIDERS: dict[str, Provider] = {}
_registry_lock = threading.Lock()
_overrides: dict[str, Callable[[], Provider]] = {}


def register(name: str, factory: Callable[[], Provider]) -> None:
    """Point a provider name at a factory of your own.

    For tests, and for adding a provider without editing this file — a third
    vendor is one module that implements `Provider` plus one call to this. The
    name is taught to `ModelRef.parse` at the same time, so
    ``"<name>:some-model"`` starts resolving immediately.

    Takes effect at once, dropping any instance already built for that name.
    """
    key = add_provider_name(name)
    with _registry_lock:
        _overrides[key] = factory
        PROVIDERS.pop(key, None)


def get_provider(name: str) -> Provider:
    """The provider instance for a name, built on first use and kept."""
    key = name.strip().lower()
    existing = PROVIDERS.get(key)
    if existing is not None:
        return existing

    with _registry_lock:
        existing = PROVIDERS.get(key)
        if existing is not None:
            return existing
        instance = _build(key)
        PROVIDERS[key] = instance
        return instance


def _build(key: str) -> Provider:
    override = _overrides.get(key)
    if override is not None:
        return override()

    entry = _FACTORIES.get(key)
    if entry is None:
        raise LLMError(
            ErrorClass.INVALID,
            f"There is no provider called {key!r}. Add one with llm.register({key!r}, ...).",
            provider=key,
            details={"known": sorted(set(_FACTORIES) | set(_overrides))},
        )

    modules, class_name = entry
    module = None
    import_errors: dict[str, str] = {}
    for path in modules:
        try:
            module = __import__(path, fromlist=["_"])
            break
        except ImportError as exc:
            import_errors[path] = str(exc)[:200]
    if module is None:
        package = _PACKAGES.get(key)
        fix = f" Install it with `pip install {package}`." if package else ""
        raise LLMError(
            ErrorClass.INVALID,
            f"The {key} provider is not installed.{fix}",
            provider=key,
            details={"looked_in": list(modules), "errors": import_errors, "package": package},
        )

    try:
        if hasattr(module, "PROVIDER"):
            return module.PROVIDER
        if hasattr(module, "provider"):
            return module.provider()
        return getattr(module, class_name)()
    except LLMError:
        raise
    except Exception as exc:  # a missing API key usually lands here
        raise LLMError.wrap(
            exc,
            error_class=ErrorClass.AUTH_EXPIRED,
            provider=key,
            message=f"The {key} provider could not start up.",
        ) from exc


async def close() -> None:
    """Shut every built provider down. Called from the app's lifespan hook."""
    with _registry_lock:
        instances = list(PROVIDERS.items())
        PROVIDERS.clear()
    for name, instance in instances:
        closer = getattr(instance, "aclose", None) or getattr(instance, "close", None)
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # shutdown is best effort
            log.warning("llm.close_failed", provider=name, error=str(exc)[:200])


# ------------------------------------------------------------------ the chain


def _default_provider() -> str:
    return str(_setting("LLM_DEFAULT_PROVIDER", "openai"))


def _fallback_models() -> list[str]:
    raw = str(_setting("LLM_FALLBACK_MODELS", ""))
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _fallback_on() -> frozenset[ErrorClass]:
    """Which failure classes are worth trying somebody else for.

    A name that is not an `ErrorClass` is dropped rather than quietly folded
    into `UNKNOWN`, which would turn a typo into "fall back on anything".
    """
    raw = str(_setting("LLM_FALLBACK_ON", ""))
    names = {piece.strip().upper() for piece in raw.split(",") if piece.strip()}
    known = {member.value for member in ErrorClass}
    unknown = names - known
    if unknown:
        log.warning("llm.fallback_on_unknown", names=sorted(unknown))
    chosen = {ErrorClass(name) for name in names & known}
    return frozenset(chosen) or DEFAULT_FALLBACK_ON


def _timeout_s() -> float:
    return float(_setting("LLM_TIMEOUT_S", 20.0))


def _parse_ref(spec: str, default_provider: str) -> ModelRef:
    """`ModelRef.parse`, with the list of providers put into the sentence.

    `base.ModelRef.parse` files the known names under ``details["known"]``,
    where a structured log line finds them and a person reading a traceback does
    not. A mistyped prefix is the most common way to get this config wrong and
    the whole fix is that list of names, so the list belongs in the message
    somebody will actually read.
    """
    try:
        return ModelRef.parse(spec, default_provider)
    except LLMError as err:
        known = err.details.get("known")
        if not isinstance(known, list) or not known:
            raise
        raise LLMError(
            err.error_class,
            f"{err.message} This build serves: {', '.join(str(k) for k in known)}.",
            provider=err.provider,
            model=err.model or spec,
            cause=err.cause,
            details=err.details,
        ) from err


def resolve(model: str | None = None, purpose: str = PURPOSE_PLAN) -> list[ModelRef]:
    """The primary model followed by its fallbacks, deduplicated.

    ``purpose`` picks which setting supplies the primary when the caller did not
    name one: ``"prose"`` prefers `LLM_PROSE_MODEL` and falls back to
    `LLM_MODEL`; ``"embed"`` uses `EMBED_MODEL`; anything else is `LLM_MODEL`.

    Embeddings come back as a chain of exactly one. See the module docstring —
    quietly answering with a different embedding model is the one failure this
    package will not perform.
    """
    kind = (purpose or PURPOSE_PLAN).strip().lower()
    default_provider = _default_provider()

    if kind in _EMBED_NAMES:
        primary = model or str(_setting("EMBED_MODEL"))
        return [_parse_ref(primary, default_provider)]

    if model:
        primary = model
    elif kind in _PROSE_NAMES:
        primary = str(_setting("LLM_PROSE_MODEL", "")) or str(_setting("LLM_MODEL"))
    else:
        primary = str(_setting("LLM_MODEL"))

    chain = [_parse_ref(primary, default_provider)]
    for name in _fallback_models():
        ref = _parse_ref(name, default_provider)
        if ref not in chain:
            chain.append(ref)
    return chain


# -------------------------------------------------------------- circuit breaker


@dataclass
class _LocalBreaker:
    """The in-process mirror, used only while Redis is unreachable.

    Redis is the shared truth so every worker sees one breaker. When Redis is
    down the cache layer's rule is fail-open, but failing open on *this* would
    mean hammering a dead provider from every worker at once, which is the exact
    thing a breaker exists to stop. So we keep a local copy.
    """

    failures: int = 0
    open_until: float = 0.0


_local_breakers: dict[str, _LocalBreaker] = {}


def _breaker_key(provider: str) -> str:
    from app.core import cache

    return cache.key("llmcb", provider)


def _probe_key(provider: str) -> str:
    from app.core import cache

    return cache.key("llmcb", f"{provider}:probe")


def _as_float(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw.decode() if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError):
        return 0.0


def _as_int(raw: Any) -> int:
    return int(_as_float(raw))


async def breaker_open_for(provider: str) -> float | None:
    """Seconds until this provider is worth trying again, or None if it is now.

    None means "go ahead". A number means skip it — the chain moves on rather
    than spending a timeout finding out what we already know.
    """
    from app.core import cache

    now = time.time()
    try:
        client = await cache.get_redis()
        failures_raw, open_until_raw = await client.hmget(  # type: ignore[misc]
            _breaker_key(provider), "failures", "open_until"
        )
    except (RedisError, OSError):
        return _local_open_for(provider, now)

    open_until = _as_float(open_until_raw)
    if open_until > now:
        return open_until - now

    if _as_int(failures_raw) >= BREAKER_THRESHOLD:
        # The window has passed and the breaker never closed, so it is half-open:
        # exactly one call gets through to find out, and the rest keep skipping.
        try:
            claimed = await client.set(_probe_key(provider), b"1", nx=True, ex=_PROBE_TTL_S)
        except (RedisError, OSError):
            return None
        if not claimed:
            return float(_PROBE_TTL_S)
        log.info("llm.breaker_probe", provider=provider)
    return None


def _local_open_for(provider: str, now: float) -> float | None:
    state = _local_breakers.get(provider)
    if state is None:
        return None
    if state.open_until > now:
        return state.open_until - now
    if state.failures >= BREAKER_THRESHOLD:
        # Half-open: let this one through as the probe, and leave the counter one
        # short of the threshold so a single failure re-opens immediately.
        state.failures = BREAKER_THRESHOLD - 1
    return None


async def _breaker_success(provider: str) -> None:
    from app.core import cache

    _local_breakers.pop(provider, None)
    try:
        client = await cache.get_redis()
        await client.delete(_breaker_key(provider), _probe_key(provider))
    except (RedisError, OSError):
        pass


async def _breaker_failure(provider: str, error_class: ErrorClass) -> None:
    from app.core import cache

    if not counts_against_health(error_class):
        return

    now = time.time()
    state = _local_breakers.setdefault(provider, _LocalBreaker())
    state.failures += 1
    if state.failures >= BREAKER_THRESHOLD:
        state.open_until = now + BREAKER_COOLDOWN_S

    try:
        client = await cache.get_redis()
        pipe = client.pipeline(transaction=False)
        pipe.hincrby(_breaker_key(provider), "failures", 1)
        pipe.expire(_breaker_key(provider), _BREAKER_TTL_S)
        pipe.delete(_probe_key(provider))
        results = await pipe.execute()
        failures = int(results[0] or 0)
        if failures >= BREAKER_THRESHOLD:
            await client.hset(_breaker_key(provider), "open_until", str(now + BREAKER_COOLDOWN_S))
            log.warning(
                "llm.breaker_open",
                provider=provider,
                failures=failures,
                cooldown_s=BREAKER_COOLDOWN_S,
                error_class=str(error_class),
            )
    except (RedisError, OSError):
        pass


async def breaker_reset(provider: str) -> None:
    """Close a provider's breaker by hand. Tests and operators."""
    await _breaker_success(provider)


# ------------------------------------------------------------ one attempt


def _classify(provider: Provider, ref: ModelRef, exc: BaseException) -> LLMError:
    """Any exception as an LLMError that knows who raised it."""
    if isinstance(exc, LLMError):
        if not exc.provider:
            exc.provider = ref.provider
        if not exc.model:
            exc.model = str(ref)
        return exc
    try:
        error_class = coerce(provider.classify_error(exc))  # type: ignore[arg-type]
    except Exception:  # a classifier that throws must not lose the real failure
        error_class = ErrorClass.UNKNOWN
    return LLMError.wrap(exc, error_class=error_class, provider=ref.provider, model=str(ref))


def _timed_out(ref: ModelRef, what: str, budget: float, exc: BaseException | None) -> LLMError:
    return LLMError(
        ErrorClass.TIMEOUT,
        f"{ref} took longer than {budget:g}s to answer.",
        provider=ref.provider,
        model=str(ref),
        cause=exc,
        details={"stage": what, "timeout_s": budget},
    )


def _stamp(usage: Usage, ref: ModelRef) -> Usage:
    """Force the canonical model spelling, so providers cannot get it wrong."""
    usage.model = str(ref)
    usage.provider = ref.provider
    return usage


async def _attempt[T](
    provider: Provider,
    ref: ModelRef,
    call: Callable[[Provider, ModelRef], Awaitable[tuple[T, Usage]]],
    *,
    what: str,
    kind: usage_tracker.Kind,
) -> T:
    """One provider, at most two tries, one timeout each.

    The second try only happens for `TRANSIENT` — a blip. Everything else either
    wants a different provider (which is the caller's job) or wants to stop.
    """
    budget = _timeout_s()
    for attempt in (1, 2):
        failure: LLMError
        try:
            async with asyncio.timeout(budget):
                value, used = await call(provider, ref)
        except TimeoutError as exc:
            failure = _timed_out(ref, what, budget, exc)
        except Exception as exc:
            failure = _classify(provider, ref, exc)
        else:
            usage_tracker.record(_stamp(used, ref), kind=kind, ok=True)
            await _breaker_success(ref.provider)
            return value

        if failure.usage is not None:
            usage_tracker.record(_stamp(failure.usage, ref), kind=kind, ok=False)
        else:
            usage_tracker.record_failure(ref, kind=kind)

        if attempt == 1 and failure.error_class == ErrorClass.TRANSIENT:
            delay = backoff(failure.error_class, attempt)
            log.warning(
                "llm.retry",
                model=str(ref),
                provider=ref.provider,
                stage=what,
                delay_s=round(delay, 3),
                error_class=str(failure.error_class),
            )
            await asyncio.sleep(delay)
            continue

        await _breaker_failure(ref.provider, failure.error_class)
        raise failure

    raise LLMError(ErrorClass.UNKNOWN, "The retry loop fell through.", model=str(ref))


# ------------------------------------------------------------ walking the chain


async def _run_chain[T](
    chain: Sequence[ModelRef],
    call: Callable[[Provider, ModelRef], Awaitable[tuple[T, Usage]]],
    *,
    what: str,
    kind: usage_tracker.Kind = "chat",
) -> T:
    """Try each model in turn until one answers or one refuses for good."""
    fallback_on = _fallback_on()
    last: LLMError | None = None
    skipped: list[str] = []

    for index, ref in enumerate(chain):
        more_to_try = index + 1 < len(chain)

        try:
            provider = get_provider(ref.provider)
        except LLMError as err:
            # Never made a request, so this is not a fallback decision — the
            # provider simply is not usable in this process.
            last = err
            log.warning("llm.provider_unavailable", model=str(ref), reason=err.message)
            continue

        open_for = await breaker_open_for(ref.provider)
        if open_for is not None:
            skipped.append(str(ref))
            log.warning(
                "llm.provider_skipped",
                model=str(ref),
                provider=ref.provider,
                stage=what,
                reason="circuit_open",
                retry_in_s=round(open_for, 1),
            )
            continue

        try:
            return await _attempt(provider, ref, call, what=what, kind=kind)
        except LLMError as err:
            last = err
            if err.error_class in fallback_on and more_to_try:
                log.warning(
                    "llm.fallback",
                    stage=what,
                    from_model=str(ref),
                    to_model=str(chain[index + 1]),
                    error_class=str(err.error_class),
                    reason=err.message,
                )
                continue
            raise

    if last is not None:
        raise last
    raise LLMError(
        ErrorClass.TRANSIENT,
        "Every model provider is unavailable right now.",
        details={"stage": what, "chain": [str(r) for r in chain], "skipped": skipped},
    )


async def _pump(
    stream: AsyncIterator[str],
    provider: Provider,
    ref: ModelRef,
    what: str,
) -> AsyncIterator[str]:
    """Chunks out of a provider stream, with our errors and our clock.

    The timeout is per chunk, not per stream: a long answer is fine, a stalled
    one is not, and the two are indistinguishable from a single deadline over
    the whole thing.
    """
    budget = _timeout_s()
    iterator = stream.__aiter__()
    while True:
        try:
            async with asyncio.timeout(budget):
                piece = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise _timed_out(ref, what, budget, exc) from exc
        except Exception as exc:
            raise _classify(provider, ref, exc) from exc
        if piece:
            yield piece


async def _aclose(stream: Any) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):  # closing a broken stream is allowed to be messy
        await closer()


def _record_stream_usage(stream: Any, ref: ModelRef, *, ok: bool) -> None:
    """File what a stream cost, however it ended.

    A provider that returned a bare async generator has nowhere to put its token
    counts, so the entry is a zero. That is worth recording anyway: it keeps
    `calls` honest, which is what makes a fallback visible in the numbers.
    """
    used = getattr(stream, "usage", None)
    if not isinstance(used, Usage):
        used = Usage.empty(ref)
    usage_tracker.record(_stamp(used, ref), kind="chat", ok=ok)


async def _stream_chain(
    chain: Sequence[ModelRef],
    open_stream: Callable[[Provider, ModelRef], Any],
    *,
    what: str,
) -> AsyncIterator[str]:
    """The chain walk again, with the one rule streams add.

    Falling back is free until the caller has seen a chunk. After that it is
    forbidden — see the module docstring. `committed` is that line.
    """
    fallback_on = _fallback_on()
    last: LLMError | None = None
    skipped: list[str] = []

    for index, ref in enumerate(chain):
        more_to_try = index + 1 < len(chain)

        try:
            provider = get_provider(ref.provider)
        except LLMError as err:
            last = err
            log.warning("llm.provider_unavailable", model=str(ref), reason=err.message)
            continue

        open_for = await breaker_open_for(ref.provider)
        if open_for is not None:
            skipped.append(str(ref))
            log.warning(
                "llm.provider_skipped",
                model=str(ref),
                provider=ref.provider,
                stage=what,
                reason="circuit_open",
                retry_in_s=round(open_for, 1),
            )
            continue

        failure: LLMError | None = None
        for attempt in (1, 2):
            stream: Any = None
            committed = False
            failure = None
            try:
                # Opening gets the per-attempt budget of its own. It cannot
                # cover the whole stream — a long answer is not a stalled one —
                # so `_pump` takes over the clock once the chunks start.
                async with asyncio.timeout(_timeout_s()):
                    stream = open_stream(provider, ref)
                    if inspect.isawaitable(stream):
                        stream = await stream
                async for piece in _pump(stream, provider, ref, what):
                    committed = True
                    yield piece
            except TimeoutError as exc:
                failure = _timed_out(ref, what, _timeout_s(), exc)
            except Exception as exc:
                failure = _classify(provider, ref, exc)
            finally:
                _record_stream_usage(stream, ref, ok=failure is None)
                await _aclose(stream)

            if failure is None:
                await _breaker_success(ref.provider)
                return

            if committed:
                # Past the point of no return. Two halves of two answers is
                # worse than one honest failure.
                await _breaker_failure(ref.provider, failure.error_class)
                log.error(
                    "llm.stream_broken",
                    stage=what,
                    model=str(ref),
                    error_class=str(failure.error_class),
                    reason=failure.message,
                    committed=True,
                )
                failure.details["committed"] = True
                raise failure

            if attempt == 1 and failure.error_class == ErrorClass.TRANSIENT:
                delay = backoff(failure.error_class, attempt)
                log.warning(
                    "llm.retry",
                    model=str(ref),
                    provider=ref.provider,
                    stage=what,
                    delay_s=round(delay, 3),
                    error_class=str(failure.error_class),
                )
                await asyncio.sleep(delay)
                continue

            await _breaker_failure(ref.provider, failure.error_class)
            break

        last = failure
        if failure is not None and failure.error_class in fallback_on and more_to_try:
            log.warning(
                "llm.fallback",
                stage=what,
                from_model=str(ref),
                to_model=str(chain[index + 1]),
                error_class=str(failure.error_class),
                reason=failure.message,
            )
            continue
        if failure is not None:
            raise failure

    if last is not None:
        raise last
    raise LLMError(
        ErrorClass.TRANSIENT,
        "Every model provider is unavailable right now.",
        details={"stage": what, "chain": [str(r) for r in chain], "skipped": skipped},
    )


# ------------------------------------------------------------- the four calls


async def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any] | None = None,
    model: str | None = None,
    max_tokens: int = 1200,
    *,
    purpose: str = PURPOSE_PLAN,
) -> dict[str, Any]:
    """One request, one JSON object back. The planner's call.

    ``schema`` is passed to the provider when it has a structured-output mode.
    Providers that do not still get told to answer in JSON, and the object is
    parsed here either way.
    """
    chain = resolve(model, purpose)

    async def call(provider: Provider, ref: ModelRef) -> tuple[dict[str, Any], Usage]:
        result: ChatResult = await provider.complete_json(
            ref.name, system, user, schema, max_tokens
        )
        used = _stamp(result.usage, ref)

        if result.finish_reason == "length":
            raise LLMError(
                ErrorClass.INVALID,
                "The model ran out of room before finishing its answer.",
                provider=ref.provider,
                model=str(ref),
                details={"stage": "complete_json", "max_tokens": max_tokens},
                usage=used,
            )
        if isinstance(result.parsed, dict):
            return result.parsed, used
        try:
            return parse_json_object(result.text, where="complete_json", model=str(ref)), used
        except LLMError as err:
            # The tokens were spent even though the answer was unusable.
            err.usage = used
            raise

    return await _run_chain(chain, call, what="complete_json")


async def stream_json(
    system: str,
    user: str,
    schema: dict[str, Any] | None = None,
    model: str | None = None,
    max_tokens: int = 1200,
    *,
    purpose: str = PURPOSE_PLAN,
) -> AsyncIterator[str]:
    """The JSON answer as it arrives, exactly as the model produced it.

    Joining every chunk gives the complete text. Parsing is the caller's, which
    is what lets the orchestrator start on step one before step four has been
    written.

    Falls back only until the first chunk is handed over.
    """
    chain = resolve(model, purpose)

    def open_stream(provider: Provider, ref: ModelRef) -> Any:
        return provider.stream_json(ref.name, system, user, schema, max_tokens)

    async for piece in _stream_chain(chain, open_stream, what="stream_json"):
        yield piece


async def stream_text(
    system: str,
    user: str,
    model: str | None = None,
    *,
    purpose: str = PURPOSE_PROSE,
) -> AsyncIterator[str]:
    """The prose answer, streamed to the chat box.

    Defaults to the prose purpose, so `LLM_PROSE_MODEL` can point the writing at
    a better model than the planning without touching a call site.

    Falls back only until the first chunk is handed over.
    """
    chain = resolve(model, purpose)

    def open_stream(provider: Provider, ref: ModelRef) -> Any:
        return provider.stream_text(ref.name, system, user)

    async for piece in _stream_chain(chain, open_stream, what="stream_text"):
        yield piece


async def embed(
    texts: Sequence[str],
    model: str | None = None,
    *,
    use_cache: bool = True,
) -> list[list[float]]:
    """One vector per string, in the order given.

    No fallback chain — read the module docstring if that seems inconsistent.
    If the embedding provider is down, embedding fails; it does not quietly
    produce vectors that cannot be compared with the ones already stored.

    Empty strings get a zero vector instead of a request: the model would reject
    them, and zero scores zero against everything, which is the honest answer.
    Anything already in the embedding cache is not sent again — the cache key
    includes the model, so two models never share an entry.
    """
    if not texts:
        return []

    from app.core import cache

    ref = resolve(model, PURPOSE_EMBED)[0]
    model_id = str(ref)
    dimensions = embed_dimensions()
    items = [t if isinstance(t, str) else str(t) for t in texts]

    out: list[list[float] | None] = [None] * len(items)
    todo: list[int] = []
    for index, text in enumerate(items):
        if text.strip():
            todo.append(index)
        else:
            out[index] = [0.0] * dimensions

    if use_cache and todo:
        cached = await cache.get_embeddings(model_id, [items[i] for i in todo])
        remaining: list[int] = []
        for index, vector in zip(todo, cached, strict=True):
            if vector is None:
                remaining.append(index)
            else:
                out[index] = vector
        todo = remaining

    fetched: list[tuple[str, list[float]]] = []
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start : start + EMBED_BATCH]
        payload = [items[i] for i in batch]

        async def call(
            provider: Provider, ref: ModelRef, payload: list[str] = payload
        ) -> tuple[list[list[float]], Usage]:
            vectors, used = await provider.embed(ref.name, payload, dimensions)
            if len(vectors) != len(payload):
                raise LLMError(
                    ErrorClass.INVALID,
                    "The embedding model returned the wrong number of vectors.",
                    provider=ref.provider,
                    model=str(ref),
                    details={"asked": len(payload), "got": len(vectors)},
                    usage=used,
                )
            for vector in vectors:
                if len(vector) != dimensions:
                    raise LLMError(
                        ErrorClass.INVALID,
                        "The embedding model returned the wrong dimension.",
                        provider=ref.provider,
                        model=str(ref),
                        details={"expected": dimensions, "got": len(vector)},
                        usage=used,
                    )
            return vectors, used

        vectors = await _run_chain([ref], call, what="embed", kind="embed")
        for index, vector in zip(batch, vectors, strict=True):
            out[index] = vector
            fetched.append((items[index], vector))

    if use_cache and fetched:
        await cache.set_embeddings(model_id, fetched)

    return [vector if vector is not None else [0.0] * dimensions for vector in out]


async def embed_one(text: str, model: str | None = None, *, use_cache: bool = True) -> list[float]:
    """One string, one vector. The probe's single call."""
    vectors = await embed([text], model, use_cache=use_cache)
    return vectors[0]


# ------------------------------------------------------- the embedding contract


def embed_model_id() -> str:
    """The canonical name of the embedding model in use.

    This exact string goes in the `embed_model` column of every mirror row
    whose vector we just produced: ``"openai:text-embedding-3-small"``. It fits
    `VARCHAR(64)` and it is the same spelling `EMBED_MODEL` uses, so a row can
    be compared against config without normalising anything.
    """
    return str(resolve(None, PURPOSE_EMBED)[0])


def embed_dimensions() -> int:
    """How long a vector is. Must match the `VECTOR(n)` columns."""
    return int(_setting("EMBED_DIMENSIONS", 1536))


def assert_same_embed_model(row_model: str | None, *, where: str = "search") -> None:
    """Refuse to compare vectors that came from different models.

    Call this on the search path with whatever `embed_model` the rows carry,
    before trusting a distance. Two models can agree on 1536 dimensions and
    still disagree completely about what each dimension means, so a cosine
    score across them is not a weak signal — it is a made-up one, and it looks
    exactly like a real one.

    The fix is never a cast or a coalesce. It is re-embedding the mirror with
    the model now configured.
    """
    want = embed_model_id()
    have = (row_model or "").strip()
    if have == want:
        return

    if have:
        default_provider = _default_provider()
        try:
            if ModelRef.parse(have, default_provider) == ModelRef.parse(want, default_provider):
                return
        except LLMError:
            pass

    raise LLMError(
        ErrorClass.INVALID,
        (
            f"These rows were embedded with {have or 'an unrecorded model'}, but the query "
            f"vector came from {want}. Vectors from different models are not comparable. "
            f"Re-embed the mirror before searching."
        ),
        details={"stage": where, "rows_embedded_with": have or None, "query_model": want},
    )


__all__ = [
    "BREAKER_COOLDOWN_S",
    "BREAKER_THRESHOLD",
    "EMBED_BATCH",
    "PROVIDERS",
    "PURPOSE_EMBED",
    "PURPOSE_PLAN",
    "PURPOSE_PROSE",
    "assert_same_embed_model",
    "breaker_open_for",
    "breaker_reset",
    "close",
    "complete_json",
    "embed",
    "embed_dimensions",
    "embed_model_id",
    "embed_one",
    "get_provider",
    "register",
    "resolve",
    "stream_json",
    "stream_text",
]
