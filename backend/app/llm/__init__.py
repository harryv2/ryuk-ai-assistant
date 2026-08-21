"""The model layer. Four calls, any provider, one config line to switch.

    from app import llm

    plan = await llm.complete_json(system, user)
    async for piece in llm.stream_text(system, user):
        ...
    vectors = await llm.embed(chunks)

Nothing above names a provider, and nothing in the app does. `LLM_MODEL` picks
who answers; `LLM_FALLBACK_MODELS` picks who answers when the first one cannot.
Moving the whole app from OpenAI to Gemini is one line of `.env` and a restart.

    LLM_MODEL=gemini:gemini-2.5-pro
    LLM_FALLBACK_MODELS=openai:gpt-4.1
    GOOGLE_AI_API_KEY=...

Hand written on purpose. No LangChain, no LlamaIndex, no agent framework — the
whole indirection layer is `base.py` (what a provider is), `router.py` (who gets
asked, and what happens when they fail) and `usage.py` (what it cost).

One thing this layer will not do
--------------------------------

**It will not fall back to a different embedding model.** Chat models are
interchangeable; embedding models are not. The pgvector columns are
`VECTOR(1536)` filled by one specific model, and a vector from another model is
not comparable with them even at the same width — the two models put meaning in
different dimensions. A cosine search across a mix returns confident nonsense:
no error, no warning, just the wrong emails.

So embeddings resolve to exactly one model with no fallback chain; every mirror
row records the model that embedded it in `embed_model`; the search path calls
`assert_same_embed_model` and gets a loud failure instead of a plausible wrong
answer; and changing `EMBED_MODEL` means re-embedding the whole mirror, in
batches, with the management command.

Gemini's `gemini-embedding-001` does support `output_dimensionality=1536`, so
the column type survives the switch. The rows do not.
"""

from app.llm.base import ChatResult, ModelRef, Provider, TextStream, Usage
from app.llm.errors import ErrorClass, LLMError
from app.llm.router import (
    assert_same_embed_model,
    complete_json,
    embed,
    embed_dimensions,
    embed_model_id,
    embed_one,
    register,
    resolve,
    stream_json,
    stream_text,
)
from app.llm.usage import Ledger, totals, track_usage

__all__ = [
    "ChatResult",
    "ErrorClass",
    "LLMError",
    "Ledger",
    "ModelRef",
    "Provider",
    "TextStream",
    "Usage",
    "assert_same_embed_model",
    "complete_json",
    "embed",
    "embed_dimensions",
    "embed_model_id",
    "embed_one",
    "register",
    "resolve",
    "stream_json",
    "stream_text",
    "totals",
    "track_usage",
]
