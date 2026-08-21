"""What a run spent on models, counted where the calls happen.

The runner wraps a run in `track_usage()` and every call inside it — however
deep, on whichever provider, however many fallbacks it took — adds itself to the
same tally. Nothing has to be threaded through a call signature, and two runs in
one worker never mix, because the tally lives in a contextvar.

    async with track_usage() as ledger:
        plan = await llm.complete_json(system, user)
        ...
    run.token_usage = ledger.totals()

Failed attempts count too. If OpenAI rate-limits us after reading the whole
prompt and Gemini answers instead, the run paid for both, and `totals()` says
so. A fallback that looks free is a fallback nobody notices firing.

On the prices below: they are a **rough cost signal, not billing**. Vendor
prices change without asking us, per-model, sometimes mid-quarter. A model that
is not listed costs zero, which shows up as a suspicious zero in the usage
record rather than as a confidently wrong number.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from app.llm.base import ModelRef, Usage

Kind = Literal["chat", "embed"]

# USD per million tokens, as (prompt, completion), per provider. Embedding
# models bill input only, so their completion rate is 0.
PRICES: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "openai": {
        "gpt-5": (1.25, 10.00),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5-nano": (0.05, 0.40),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "o4-mini": (1.10, 4.40),
        "text-embedding-3-small": (0.02, 0.0),
        "text-embedding-3-large": (0.13, 0.0),
    },
    "gemini": {
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-embedding-001": (0.15, 0.0),
        "text-embedding-004": (0.0, 0.0),
    },
}

# What a cached prompt token costs as a fraction of a fresh one. Both vendors
# discount reused prefixes; both currently land near a quarter of list price.
CACHED_DISCOUNT: Final[dict[str, float]] = {"openai": 0.25, "gemini": 0.25}


def rates(model: str) -> tuple[float, float]:
    """(prompt, completion) USD per million tokens for a model reference.

    Takes a full reference — ``"openai:gpt-4.1-mini"``. Falls back to the
    longest matching name so a dated snapshot like
    ``"gpt-4.1-mini-2025-04-14"`` prices as ``gpt-4.1-mini`` and not as
    ``gpt-4.1``.
    """
    provider, _, name = str(model).partition(":")
    if not name:  # a bare name; we cannot tell whose it is, so try every table
        name = provider
        tables = list(PRICES.values())
    else:
        tables = [PRICES.get(provider, {})]

    for table in tables:
        if name in table:
            return table[name]
    for table in tables:
        for known in sorted(table, key=len, reverse=True):
            if name.startswith(known):
                return table[known]
    return (0.0, 0.0)


def price(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> float:
    """Dollar cost of one call. Unknown models cost 0.

    ``cached_prompt_tokens`` is part of ``prompt_tokens``, so it is discounted
    out of the total rather than added on top.
    """
    prompt_rate, completion_rate = rates(model)
    if prompt_rate == 0.0 and completion_rate == 0.0:
        return 0.0

    provider = str(model).partition(":")[0]
    discount = CACHED_DISCOUNT.get(provider, 1.0)
    cached = max(0, min(int(cached_prompt_tokens), int(prompt_tokens)))
    fresh = max(0, int(prompt_tokens) - cached)

    return (
        (fresh / 1_000_000) * prompt_rate
        + (cached / 1_000_000) * prompt_rate * discount
        + (max(0, int(completion_tokens)) / 1_000_000) * completion_rate
    )


@dataclass(frozen=True, slots=True)
class Entry:
    """One call, whether or not it worked."""

    usage: Usage
    kind: Kind
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {**self.usage.to_dict(), "kind": self.kind, "ok": self.ok}


@dataclass
class Ledger:
    """Every model call made inside one `track_usage()` block."""

    entries: list[Entry] = field(default_factory=list)

    def add(self, usage: Usage, *, kind: Kind = "chat", ok: bool = True) -> Usage:
        """Record one call. Fills in the dollar cost when the provider left it 0."""
        if not usage.usd:
            usage.usd = price(
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_prompt_tokens,
            )
        self.entries.append(Entry(usage=usage, kind=kind, ok=ok))
        return usage

    # -- reading it back ------------------------------------------------------

    @property
    def calls(self) -> int:
        return len(self.entries)

    @property
    def usd(self) -> float:
        return sum(e.usage.usd for e in self.entries)

    def totals(self) -> dict[str, Any]:
        """The shape `runs.token_usage` stores.

        ``model`` and ``provider`` name whichever chat model actually produced
        the answer — the last one that succeeded, not the one that was tried
        first and rate-limited. Embedding calls never claim the field, because
        "which model wrote this" is a question about the chat model.
        """
        prompt = sum(e.usage.prompt_tokens for e in self.entries)
        completion = sum(e.usage.completion_tokens for e in self.entries)

        chosen: Usage | None = None
        for entry in reversed(self.entries):
            if entry.ok and entry.kind == "chat":
                chosen = entry.usage
                break
        if chosen is None and self.entries:
            chosen = self.entries[-1].usage

        return {
            "prompt": prompt,
            "completion": completion,
            "model": chosen.model if chosen else None,
            "provider": chosen.provider if chosen else None,
            "usd": round(self.usd, 6),
            "calls": self.calls,
        }

    def breakdown(self) -> list[dict[str, Any]]:
        """Every call in order. For the trace panel and for working out why a
        run cost what it cost."""
        return [e.to_dict() for e in self.entries]


_ledger: ContextVar[Ledger | None] = ContextVar("llm_usage", default=None)


@asynccontextmanager
async def track_usage() -> AsyncIterator[Ledger]:
    """Count every model call made inside this block.

    Nests safely: an inner block gets its own ledger and the outer one is put
    back untouched when it exits.
    """
    ledger = Ledger()
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


def current() -> Ledger | None:
    """The ledger for this context, or None when nobody started one."""
    return _ledger.get()


def record(usage: Usage, *, kind: Kind = "chat", ok: bool = True) -> Usage:
    """Add one call to the ledger, if there is one.

    Outside a `track_usage()` block this still prices the call and returns it,
    it just has nowhere to file it. Calls made outside a run — a warm-up, a
    script — should not need a ledger to be legal.
    """
    ledger = _ledger.get()
    if ledger is not None:
        return ledger.add(usage, kind=kind, ok=ok)
    if not usage.usd:
        usage.usd = price(
            usage.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cached_prompt_tokens,
        )
    return usage


def record_failure(ref: ModelRef | str, usage: Usage | None = None, *, kind: Kind = "chat") -> None:
    """File what a failed attempt cost, so a fallback is not free in the numbers.

    Most failures burn nothing and record a zero, which is still worth having:
    it is what makes ``calls`` larger than the number of answers.
    """
    record(usage or Usage.empty(ref), kind=kind, ok=False)


def totals() -> dict[str, Any]:
    """The current ledger's totals, or an empty tally outside a block."""
    ledger = _ledger.get()
    return ledger.totals() if ledger else Ledger().totals()


__all__ = [
    "CACHED_DISCOUNT",
    "PRICES",
    "Entry",
    "Ledger",
    "current",
    "price",
    "rates",
    "record",
    "record_failure",
    "totals",
    "track_usage",
]
