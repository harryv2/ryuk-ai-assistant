# Models

Three providers sit behind one interface: **OpenAI**, **Gemini** and
**Anthropic**. No call site in this app names a provider. Code says what it
wants —

```python
plan = await llm.complete_json(system, user)
async for piece in llm.stream_text(system, user):
    ...
```

— and one line of config decides who serves it. Switching vendors is that line
and a restart.

Everything here lives in `backend/app/llm/`:

| file | what it holds |
| --- | --- |
| `base.py` | the `Provider` protocol, `ModelRef`, `Usage`, `ChatResult`, `TextStream` |
| `router.py` | the fallback chain, the circuit breaker, the four public calls |
| `errors.py` | the ten `ErrorClass` names every provider maps onto |
| `usage.py` | the per-run ledger and the price tables |
| `providers/openai_provider.py` | OpenAI, over `chat.completions` and `embeddings` |
| `providers/gemini_provider.py` | Gemini, over `generateContent` |
| `providers/anthropic_provider.py` | Anthropic, over `messages` |

---

## 1. Model strings

Every model is written `<provider>:<name>`.

```
openai:gpt-5.6-terra
gemini:gemini-3.5-flash
anthropic:claude-opus-5
```

The prefix is the **only** thing that picks a provider. A name with no prefix
belongs to `LLM_DEFAULT_PROVIDER`. An unknown prefix is an error at startup, not
a silent fall-through to the default — a mistyped prefix quietly becoming an
OpenAI call is the kind of thing you find out about in a bill.

These spellings are all accepted and all mean the same three providers:

| canonical | also accepted |
| --- | --- |
| `openai` | `oai`, `gpt` |
| `gemini` | `google`, `googleai`, `google-ai`, `google_genai`, `genai`, `vertex` |
| `anthropic` | `claude` |

Parsing happens in `ModelRef.parse` and nowhere else. If you find code splitting
a model string on `":"`, that is the bug.

---

## 2. The models

Prices are **USD per million tokens, (input / output)**. They are a rough cost
signal for the run ledger, not billing. A model that is not in the table costs
zero, which shows up as a suspicious zero rather than a confidently wrong
number.

### Chat

| provider | model string | input | output | notes |
| --- | ---: | ---: | ---: | --- |
| Anthropic | `anthropic:claude-opus-5` | $5.00 | $25.00 | 1M context. The default when you move to Anthropic. |
| Anthropic | `anthropic:claude-sonnet-5` | $3.00 | $15.00 | 1M context. Most of Opus at a bit over half the price. |
| Anthropic | `anthropic:claude-haiku-4-5` | $1.00 | $5.00 | 200K context. The cheap one; note the smaller window. |
| Anthropic | `anthropic:claude-fable-5` | $10.00 | $50.00 | 1M context. |
| OpenAI | `openai:gpt-5.6-terra` | — | — | **The app default.** What this deployment runs. |
| OpenAI | `openai:gpt-5` | $1.25 | $10.00 | |
| OpenAI | `openai:gpt-5-mini` | $0.25 | $2.00 | |
| OpenAI | `openai:gpt-5-nano` | $0.05 | $0.40 | |
| Gemini | `gemini:gemini-3.5-flash` | — | — | **The default fallback.** |
| Gemini | `gemini:gemini-2.5-pro` | $1.25 | $10.00 | Cannot turn thinking off. |
| Gemini | `gemini:gemini-2.5-flash` | $0.30 | $2.50 | |

Prices move and new ids appear faster than a table in a repository does. The
ones marked `—` are what this deployment is configured with; treat every figure
here as indicative and check the provider's own page before quoting a cost.
Nothing in the code reads this table — `app/llm/pricing.py` is the only thing
that affects a bill, and an unknown id there costs `0.0` rather than guessing.

**Never append a date suffix to a Claude model id.** `claude-opus-5` is the id.
`claude-opus-5-20260514` is a 404. The price lookup does match the longest known
prefix, so a suffixed name still prices correctly if one ever appears — but the
API call itself will have already failed.

### Embeddings

| provider | model string | input | dimensions |
| --- | ---: | ---: | ---: |
| OpenAI | `openai:text-embedding-3-small` | $0.02 | 1536 (configurable) |
| OpenAI | `openai:text-embedding-3-large` | $0.13 | 3072, truncatable to 1536 |
| Gemini | `gemini:gemini-embedding-001` | $0.15 | 3072 native, 1536 on request |
| Anthropic | — | — | **none. See §6.** |

---

## 3. Switching provider

One variable:

```bash
LLM_MODEL=anthropic:claude-opus-5
```

That is the whole change for chat. Nothing else moves — not a call site, not a
prompt, not a schema. The provider module is imported the first time a model
string names it, so a deployment that never says `anthropic:` never loads the
Anthropic SDK and never needs a key for it.

A fuller move, with a different vendor as the safety net:

```bash
LLM_MODEL=anthropic:claude-opus-5
LLM_FALLBACK_MODELS=openai:gpt-5.6-terra
ANTHROPIC_API_KEY=sk-ant-...
```

Every knob:

| variable | default | what it does |
| --- | --- | --- |
| `LLM_MODEL` | `openai:gpt-5.6-terra` | The primary for planning and, unless overridden, prose. |
| `LLM_FALLBACK_MODELS` | `gemini:gemini-2.5-flash` | Comma separated, tried in order. |
| `LLM_PROSE_MODEL` | *(empty)* | A different model for writing the answer than for planning it. |
| `LLM_DEFAULT_PROVIDER` | `openai` | Who owns a model string with no prefix. |
| `LLM_TIMEOUT_S` | `20` | Per attempt, and per chunk while streaming. |
| `OPENAI_REASONING_EFFORT` | `low` | Sent to the reasoning families (o-series, gpt-5*) only. Their default depth thinks for tens of seconds before the first byte of JSON; `low` keeps the planner inside its budget. Blank sends nothing. |
| `LLM_FALLBACK_ON` | see §4 | Which failure classes are worth trying somebody else for. |
| `EMBED_MODEL` | `openai:text-embedding-3-small` | See §6 before touching this. |
| `EMBED_DIMENSIONS` | `1536` | Must match every `VECTOR(n)` column. |

Provider keys: `OPENAI_API_KEY`, `GOOGLE_AI_API_KEY`, `ANTHROPIC_API_KEY`.
`require_production_secrets()` insists on `OPENAI_API_KEY` outside development
whatever else is configured — `EMBED_MODEL` defaults to OpenAI and embeddings
never fall back — and adds `ANTHROPIC_API_KEY` to that list only when
`LLM_MODEL`, `LLM_FALLBACK_MODELS` or `LLM_PROSE_MODEL` actually names
Anthropic. A deployment that never says `anthropic:` is never asked for a key it
has no use for.

### Anthropic-only knobs

| variable | default | what it does |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | *(empty)* | From console.anthropic.com. |
| `ANTHROPIC_EFFORT` | `medium` | `low` · `medium` · `high` · `xhigh` · `max`. Anthropic's own default is `high`; ours is lower because the planner call sits inside `SOFT_DEADLINE_MS`. A bad value fails at startup. |
| `ANTHROPIC_THINKING` | `false` | `false` sends `{"type": "disabled"}`; `true` sends `{"type": "adaptive"}`. |
| `ANTHROPIC_CACHE` | `system` | `system` puts the cache breakpoint at the end of the system prompt; `auto` uses the top-level parameter; `off` disables. Environment only — not a `Settings` field. |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Floor for a non-streamed answer. `0` honours the caller exactly. |
| `ANTHROPIC_STREAM_MAX_TOKENS` | `64000` | Floor for a streamed answer. |
| `ANTHROPIC_TIMEOUT_S` | `120` | On the SDK client, in seconds. The router's own budget is the shorter one and is the one that matters. |
| `ANTHROPIC_BASE_URL` | *(unset)* | For a proxy or a gateway. |

Two things about `ANTHROPIC_THINKING` that are easy to get backwards:

* **"Not set" is not "off".** On `claude-opus-5`, omitting `thinking` runs
  adaptive thinking. Turning it off takes an explicit `{"type": "disabled"}`,
  which is what `false` sends.
* **At `xhigh` and `max` effort, disabled thinking is a 400.** The provider drops
  the parameter and logs a warning rather than sending a request that cannot
  work. Effort is the more deliberate of the two settings, so effort wins.

---

## 4. The fallback chain

`resolve()` builds a chain: the primary, then each entry of
`LLM_FALLBACK_MODELS` in order, deduplicated. The chain is then walked.

1. A provider whose **circuit breaker is open is skipped**, not tried. Five
   consecutive failures opens it for sixty seconds; then one probe call decides
   whether it closes. State lives in Redis so every worker sees the same
   breaker, with an in-process mirror for when Redis is unreachable.
2. Inside one provider, **one extra attempt on `TRANSIENT` only**, with full
   jitter. A rate limit does not want the same provider again a second later; it
   wants a different one, which is what the chain is for.
3. Each attempt gets its own `LLM_TIMEOUT_S`. Running past it is `TIMEOUT`.
4. A failure whose class is in `LLM_FALLBACK_ON` logs a warning naming both
   models and the class, then moves to the next link.
5. A failure whose class is **not** in that set fails on the spot.
6. Every attempt, including the failed ones, is recorded in the run's usage
   ledger. A fallback that fired shows up as money.

### Which classes fall back

`LLM_FALLBACK_ON` defaults to:

```
RATE_LIMITED,TRANSIENT,QUOTA_EXHAUSTED,TIMEOUT
```

| class | falls back? | why |
| --- | --- | --- |
| `TRANSIENT` | yes | 5xx, a dropped connection, Anthropic's 529 "overloaded". Somebody else is probably fine. |
| `RATE_LIMITED` | yes | 429. Our budget with *this* vendor is spent; another vendor's is not. |
| `QUOTA_EXHAUSTED` | yes | Out of credit. Same reasoning, slower to fix. |
| `TIMEOUT` | yes | This provider is slow right now. |
| `INVALID` | **no** | A malformed request is malformed everywhere. Sending it on buys a second rejection and a second bill. |
| `CONTENT_FILTERED` | **no** | A refusal on safety grounds is not a bad minute. Asking the next model the same question pays twice for the same no. |
| `AUTH_EXPIRED` / `AUTH_REVOKED` | **no** | Our API key is wrong. That is a deployment problem and it fails instantly, so hiding it behind a fallback only delays the fix. |
| `NOT_FOUND` | **no** | The model name is wrong. It will be wrong on the retry too. |
| `UNKNOWN` | **no** | We do not know what happened, so we do not know that repeating it is safe. |

**Set a key for every model in the chain.** The error the caller sees is the
*last* link's, not the first's. Point `LLM_MODEL` at Anthropic while leaving the
default `LLM_FALLBACK_MODELS=gemini:gemini-2.5-flash` with no `GOOGLE_AI_API_KEY`
set, and the first Anthropic rate limit surfaces as
`gemini AUTH_EXPIRED: Gemini has no API key` — technically true, and it hides
the rate limit that actually started it. Either set the key or point the
fallback somewhere you have one.

Only `TRANSIENT`, `TIMEOUT`, `RATE_LIMITED`, `QUOTA_EXHAUSTED` and `UNKNOWN`
count towards opening a circuit breaker. `INVALID`, `CONTENT_FILTERED` and
`NOT_FOUND` say our request was the problem, not the provider, and the `AUTH_*`
pair would only hide the real reason behind "unavailable" in the logs.

### How each vendor's failures are named

| our class | OpenAI | Gemini | Anthropic |
| --- | --- | --- | --- |
| `INVALID` | `BadRequestError` | 400 | `BadRequestError` |
| `AUTH_REVOKED` | 401 / 403 | 403 | `AuthenticationError`, `PermissionDeniedError` |
| `NOT_FOUND` | `NotFoundError` | 404 | `NotFoundError` |
| `RATE_LIMITED` | `RateLimitError` | 429 | `RateLimitError` (`retry-after` header logged) |
| `QUOTA_EXHAUSTED` | insufficient quota | quota exceeded | 400 or 429 mentioning credit or billing |
| `TRANSIENT` | 5xx, connection error | 5xx | `APIStatusError` ≥ 500 (529 = overloaded), `APIConnectionError` |
| `TIMEOUT` | `APITimeoutError` | read timeout | `APITimeoutError` |
| `CONTENT_FILTERED` | refusal | `SAFETY` finish reason | `stop_reason == "refusal"` |

Anthropic's exception hierarchy has two traps worth knowing about, because both
are silent when you get them wrong: `APITimeoutError` **subclasses**
`APIConnectionError`, and every 4xx class **subclasses** `APIStatusError`. The
order of the `isinstance` chain is load-bearing, not cosmetic.

A refusal is worse than a trap: it arrives as **HTTP 200** with content blocks
attached. Nothing raises. A provider that reads the content before checking
`stop_reason` hands a refusal to the planner as though it were an answer.

---

## 5. Streaming commits at the first token

`stream_json` and `stream_text` may fall back freely **until the first chunk
reaches the caller**. After that we are committed: if the stream dies at token
four hundred we raise, and we do not quietly restart on another provider.

The alternative is emitting two partial answers into one chat bubble, where the
second half of an OpenAI sentence is finished by Claude and nobody can tell why
the answer contradicts itself. A visible failure beats an invisible splice.

Two consequences:

* The per-attempt timeout becomes a **per-chunk** timeout once chunks start. A
  long answer is fine; a stalled one is not, and a single deadline over the
  whole stream cannot tell them apart.
* A JSON stream that ends at `max_tokens` is a failure, not a short answer —
  half an object is broken, not smaller. A prose stream that ends the same way
  is fine and is logged, because a long answer that ran out of room is still an
  answer.

Chunks are never buffered to completion before being handed on. That is the
whole reason the streaming path exists: the orchestrator starts on step one
while step four is still being written.

---

## 6. Embeddings

> **Anthropic has no embeddings API at all.**
>
> Not a missing adapter, not a slow path, not something a newer SDK will add.
> The API does not have the endpoint. `EMBED_MODEL=anthropic:...` is a
> configuration that can never work, so `app/config.py` rejects it at startup
> with a message saying what to set instead, and `AnthropicProvider.embed`
> raises if it is somehow reached anyway.
>
> Chat on Claude with embeddings on OpenAI is the normal arrangement:
>
> ```bash
> LLM_MODEL=anthropic:claude-opus-5
> EMBED_MODEL=openai:text-embedding-3-small
> ```

### Chat models are swappable. Embedding models are not.

Every `VECTOR(1536)` column in the mirror was filled by **one specific model**.
A vector from a different model is not comparable with those, and the reason is
not the width:

* **OpenAI's `text-embedding-3-small` produces 1536 dimensions. Gemini's
  `gemini-embedding-001` can also produce 1536.** The column type does not
  change when you switch.
* **The vectors are still not interchangeable.** Two models can agree on 1536
  dimensions and disagree completely about what each dimension means. A cosine
  distance across them is not a weak signal — it is a made-up one, and it looks
  exactly like a real one. No error, no warning, no obviously wrong output. Just
  the wrong emails, quietly, forever.

So:

* **`resolve(purpose="embed")` returns a chain of exactly one.** Embeddings never
  fall back. If the embedding provider is down, embedding fails; it does not
  produce vectors that cannot be compared with the ones already stored.
* **Every mirror row records the model that embedded it** in its `embed_model`
  column — the same canonical string as `EMBED_MODEL`, e.g.
  `openai:text-embedding-3-small`.
* **The search path calls `assert_same_embed_model`** before trusting a
  distance, and gets a loud error instead of a plausible wrong answer.
* **Changing `EMBED_MODEL` means re-embedding the whole mirror.** Nothing else
  will do — not a cast, not a coalesce, not a migration. Every row has to be
  rebuilt with the model now configured, and until it is, search is comparing
  two vocabularies.

Budget for it. This is the one setting in the whole file that is not a restart.

> **Two ends of this are still owed by the sync and search paths.** The column,
> the index, the helpers and the re-embed command are all in place, but nothing
> writes `embed_model` on the normal sync path and nothing reads it on the normal
> search path yet:
>
> * `app/db/repositories/mirror.py::set_embeddings` writes `embedding` and not
>   `embed_model`, so a freshly synced row carries `''` and looks un-embedded to
>   `scripts/reembed.py`. The fix is one more column in that `UPDATE`, filled
>   from `app.llm.embed_model_id()`.
> * `mirror.hybrid_search` does not call `assert_same_embed_model` (or filter on
>   `embed_model = :model`), so today a mixed mirror would return the wrong rows
>   quietly rather than raising.
>
> Until both land, `python -m scripts.reembed --table all` is also the way to
> stamp rows that the sync path left blank.

### Switching the embedding model

```bash
# 1. See what it would cost. Reads the mirror, calls nothing, writes nothing.
docker compose exec api python -m scripts.reembed --table all \
    --model gemini:gemini-embedding-001 --dry-run

# 2. Do it. Batched, resumable, prints progress and a running dollar figure.
docker compose exec api python -m scripts.reembed --table all \
    --model gemini:gemini-embedding-001 --yes

# 3. Point the app at the same model and restart the api and the workers.
#    EMBED_MODEL=gemini:gemini-embedding-001
```

Search is wrong between steps 2 and 3 — and between 1 and 2, if you set
`EMBED_MODEL` first. Once the search path calls `assert_same_embed_model` it will
be wrong *loudly*, raising rather than returning rubbish; until then it is wrong
quietly, which is worse. Either way, pick the quiet hour.

`scripts/reembed.py` in full:

| flag | what it does |
| --- | --- |
| `--table gmail\|gcal\|gdrive\|all` | which mirror table. Default `all`. |
| `--model <ref>` | what to embed with. Default: whatever `EMBED_MODEL` says. |
| `--batch-size N` | texts per request. Default 96, under both vendors' caps. |
| `--dry-run` | count the work, estimate the cost, call nothing, write nothing. |
| `--user <id>` | one account only. Needs `--force`. |
| `--limit N` | stop after N rows. Needs `--force`. |
| `--force` | run even though it would leave the mirror holding two models. |
| `--yes` | do not ask before starting. |

Three things it does that are worth knowing:

* **It refuses to start a run that would leave the mirror mixed** — a table left
  out, one `--user` out of many, a `--limit` smaller than the work — and prints
  which rows would be left behind and why that breaks cosine search. `--force`
  overrides, for a deliberate two-stage run.
* **The vector and the model name are written in one statement.** Two statements
  can be interrupted between them, and a new vector whose `embed_model` still
  names the old model is precisely the silent wrongness the column exists to
  prevent.
* **It is resumable.** A row leaves the work list the moment its update commits,
  so a run that dies at 60% picks up at 60%. Rows whose text is now empty cannot
  be embedded at all: their stale vector is cleared and `embed_model` blanked,
  rather than left claiming a model that never saw them.

---

## 7. Structured output

All three providers can be told to return JSON matching a schema, and all three
spell it differently. `router.complete_json` hides that; what follows matters
only if you are reading a provider.

| | how the schema is sent | what the schema has to look like |
| --- | --- | --- |
| OpenAI | `response_format={"type": "json_schema", …, "strict": true}` | Every object needs `additionalProperties: false` and every property in `required`. `allOf` has no strict equivalent. |
| Gemini | `responseSchema` in the generation config | A different vocabulary entirely; `schema_translate.py` rebuilds the schema. |
| Anthropic | `output_config.format = {"type": "json_schema", "schema": …}` | Same deal as OpenAI strict, so `anthropic_schema.py` is nearly a copy. |

Notes on the Anthropic path specifically:

* **It is not forced tool use any more.** Handing Claude one tool and forcing a
  call was the old way. The current way is `output_config.format`, and the
  first text block of the reply is JSON that matches the schema.
* **The deprecated top-level `output_format` parameter must not be used.**
* **`output_config.format` is incompatible with document citations** — sending
  both is a 400.
* **Streaming is ordinary text.** Because the schema makes the answer a text
  block that happens to hold JSON, there is no partial-JSON event type to
  reassemble.
* Anything a schema asks for that Anthropic cannot express — number ranges,
  string patterns, list sizes, a schema that refers to itself — is **stripped
  and rewritten as a sentence in that node's `description`**, never raised. A
  rule the validator cannot enforce becomes a rule the model can read, and the
  call still happens. `INVALID` does not fall back, so raising there would have
  cost the entire answer.

### Parameters that were removed from Claude 5 models

On `claude-opus-5`, `claude-sonnet-5` and `claude-fable-5` these are gone and
now return **HTTP 400**:

```
budget_tokens    temperature    top_p    top_k    assistant prefill
```

There is no temperature to turn down. Thinking depth is `thinking` plus
`output_config.effort`. `backend/tests/unit/test_anthropic_provider.py` scans
every request the provider builds for all four parameter names, at every depth,
so this cannot creep back in.

---

## 8. Prompt caching

A long, stable prefix is much cheaper on a repeat call — but only two of the
three vendors will do it without being asked, and each asks differently.

| | how it works | minimum prefix | do we use it? |
| --- | --- | --- | --- |
| **OpenAI** | Automatic. Caches a prefix over about 1024 tokens with nothing requested, reports the reused part as `usage.prompt_tokens_details.cached_tokens`. | ~1024 tokens | yes, by default |
| **Anthropic** | Explicit. Nothing is cached unless a `cache_control` breakpoint asks for it. Same ~1024 token minimum; under that it silently does not cache, with no error. Maximum four breakpoints. Render order is tools, then system, then messages. | ~1024 tokens | yes — one breakpoint at the end of the system prompt |
| **Gemini** | A separate explicit context-caching API with its own cached-content objects and their own lifetime. | — | **no** |

### What that means for our system prefix

The planner prompt is built in two halves with a hard line between them, and the
stable half — role and rules, the op catalogue, the plan grammar, the reference
forms, the worked examples — runs to about **2,400 tokens, byte-identical on
every request**. It is over 1,024 tokens *on purpose*: that is the threshold both
OpenAI and Anthropic gate caching at, and under it you get nothing. See
`DESIGN.md` §L3.

So the same prompt is billed three different ways:

* **OpenAI** notices the repeat by itself and charges the reused 2,400 tokens at
  a quarter of list. Nothing to switch on.
* **Anthropic** charges full price until a `cache_control` breakpoint asks for
  the cache — then reads cost about a tenth, and the write that seeds it about
  1.25×.
* **Gemini** should be budgeted at **full price for all ~2,400 tokens on every
  planner call**. It has two kinds of caching and neither is the one the other
  two do: *implicit* caching may discount a repeated prefix on 2.5 models, with
  no way to ask for it and no promise it will happen, and *explicit* caching is a
  separate resource — create a `cachedContents` object with a TTL, pass the
  handle on every request, refresh it before it expires, delete it on shutdown,
  and make a new one for every model version. That is a lifecycle, not a flag,
  and we do not run it. At `gemini-2.5-flash`'s $0.30/M the prefix is about
  $0.0007 a call — small, but it is *every* call, and it is the part of the cost
  table that does not shrink as traffic grows.

The ordering rule is the same whichever vendor is serving: **fixed catalogue
first, the user's question last**, and nothing volatile before the breakpoint —
no timestamp, no request id, no user data. One moving token in the prefix
invalidates the whole thing. `supports_prefix_caching()` returns True for OpenAI
and Anthropic and False for Gemini, which is exactly this difference.

The Anthropic breakpoint goes at the **end of the system block**, not at the top
level. The top-level `cache_control` parameter auto-caches the *last* cacheable
block, which is the user's question — the one thing that changes every call. The
prefix would then never match and `cache_read_input_tokens` would sit at zero
forever. `ANTHROPIC_CACHE=auto` gives you that literal top-level behaviour if
you want it; `system` (the default) is the one that actually caches.

Whichever provider is in use, the prompt should be ordered the same way: **the
fixed catalogue first, the user's question last.** `supports_prefix_caching()`
returns True for OpenAI and Anthropic for exactly this reason.

**Checking that it works:** watch `cache_read_input_tokens` (Anthropic) or
`cached_tokens` (OpenAI) in the usage ledger. A persistent zero means something
in the prefix is moving between calls. The usual culprits are a timestamp in the
system prompt, a JSON blob serialised with unsorted keys, or a tool list built
in a different order each time.

Cache pricing is not a flat discount on Anthropic: a cache **read** costs about
a tenth of the input rate, and a cache **write** costs about 1.25×. The Anthropic
provider therefore computes its own dollar figure instead of leaving it to
`usage.py`, whose table has one flat discount per vendor and nowhere to record
the difference. See `AnthropicProvider._usage`.

---

## 9. Adding a fourth provider

One file that implements `app.llm.base.Provider`, plus one line. The protocol is
six methods:

```python
async def complete_json(model, system, user, schema, max_tokens) -> ChatResult
def       stream_json(model, system, user, schema, max_tokens) -> AsyncIterator[str]
def       stream_text(model, system, user)                     -> AsyncIterator[str]
async def embed(model, texts, dimensions)   -> tuple[list[list[float]], Usage]
def       classify_error(exc)               -> str
def       supports_prefix_caching()         -> bool
```

Four rules that are easy to get wrong:

1. **`model` is the bare name.** The router has already chosen the provider, so
   you get `claude-opus-5`, never `anthropic:claude-opus-5`.
2. **Do not retry, do not sleep, do not swallow.** Raise. The router owns the
   retry budget and cannot honour a deadline you are sleeping through.
3. **The streaming calls are called, not awaited.** Return a `TextStream` so the
   token counts can arrive after the last chunk.
4. **`classify_error` never raises.** It runs on the failure path; an exception
   there loses the failure it was called to describe.

Then either add an entry to `_FACTORIES` in `router.py`, or register at run
time:

```python
llm.register("mistral", lambda: MistralProvider())
```

`register` teaches `ModelRef.parse` the name at the same time, so
`mistral:whatever` starts resolving immediately.
