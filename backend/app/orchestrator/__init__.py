"""The orchestrator: front door, pre-pass, one planning call, dispatch, render.

The order a query moves through, and what each stage costs:

``front_door``  pure Python. An answer to an open card, a UI verb, chit-chat, a
                capability question, or one of about fifteen rule-router shapes.
                A hit here answers the whole turn at **0 LLM calls**.
``prepass``     pure Python. Time phrases resolved by ``temporal``, literals
                lifted by regex, vendor aliases expanded, mime words mapped.
``probe``       one embedding, three parallel hybrid searches over our own
                mirror, then regex extractors over the excerpts. **0 LLM calls.**
``route``       **the one** planning call. Streams, so the intent and the first
                steps are on the wire before the last one has been written.
``validate``    pure Python. Ops exist, args fit, references resolve, no cycles,
                writes are gated, services match the intent.
``dispatch``    one asyncio task per step, each awaiting only its own
                dependencies. Per-service semaphores. Writes are prepared, never
                executed.
``render``      a card or a template at 0 calls, or one streamed prose call.

``events`` carries all of it to the browser over Redis pub/sub, and ``entities``
records what was surfaced so "that email about the proposal" resolves later
without a search.
"""

from __future__ import annotations

__all__ = [
    "dispatch",
    "entities",
    "events",
    "front_door",
    "prepass",
    "prompts",
    "render",
    "route",
    "runner",
    "temporal",
    "validate",
]
