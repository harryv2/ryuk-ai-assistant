# Database Schema

Postgres 17 + pgvector. Extensions: `vector`, `pgcrypto`, `citext`.

**Locked.** Changes go through a migration, not an edit here.

## Conventions

- All ids are **nanoid, 21 chars, generated in the service**. No database defaults.
- The only UUIDs are content fingerprints (`content_hash`, `dedupe_key`, `payload_hash`) where the value must be a function of the content. `uuid5(namespace, canonical_string)`.
- `user_id` on child tables is a deliberate exception to normalisation: every repository method takes it as the first argument, so no query can cross tenants by accident.
- Anything computable from other columns is not stored. See **Derived, not stored** at the end.
- Prefixes: no prefix = app data, source of truth. `sync_` = disposable mirror of whatever is connected. `job_` = queue bookkeeping.

---

## Enums

```sql
CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system');

CREATE TYPE run_status AS ENUM (
  'running', 'awaiting_input', 'complete', 'failed', 'timeout', 'cancelled'
);

CREATE TYPE node_status AS ENUM (
  'pending', 'running', 'succeeded', 'failed', 'skipped', 'timeout', 'cancelled'
);

CREATE TYPE input_kind AS ENUM (
  'confirm', 'choice', 'multi_choice', 'text', 'form', 'date_range'
);

CREATE TYPE input_status AS ENUM (
  'pending', 'answered', 'cancelled', 'expired', 'superseded'
);

CREATE TYPE action_status AS ENUM (
  'draft', 'approved', 'running', 'done', 'failed', 'cancelled', 'expired'
);

CREATE TYPE sync_service AS ENUM ('gmail', 'gcal', 'gdrive');
```

---

## `users`

Identity, plus the two settings that change how a query is read.

```sql
CREATE TABLE users (
    id               CHAR(21) PRIMARY KEY,
    email            CITEXT NOT NULL UNIQUE,
    display_name     TEXT,
    timezone         VARCHAR(64) NOT NULL DEFAULT 'UTC',   -- IANA
    work_week_start  SMALLINT NOT NULL DEFAULT 1,          -- 1=Mon, 7=Sun
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

## `oauth_tokens`

One row per user per provider. Adding another provider later is a row, not a migration.

```sql
CREATE TABLE oauth_tokens (
    id                   CHAR(21) PRIMARY KEY,
    user_id              CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider             VARCHAR(32) NOT NULL DEFAULT 'google',
    provider_account_id  VARCHAR(128),   -- Google `sub`, Slack user id, ...

    access_token_enc     BYTEA NOT NULL, -- AES-256-GCM: nonce(12) || ct || tag(16)
    refresh_token_enc    BYTEA,          -- not every provider issues one
    key_version          SMALLINT NOT NULL DEFAULT 1,
    scopes               TEXT[] NOT NULL,
    expires_at           TIMESTAMPTZ,
    revoked_at           TIMESTAMPTZ,
    refresh_failures     SMALLINT NOT NULL DEFAULT 0,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, provider)
);

CREATE INDEX ON oauth_tokens (expires_at) WHERE revoked_at IS NULL;
CREATE UNIQUE INDEX ON oauth_tokens (provider, provider_account_id)
    WHERE provider_account_id IS NOT NULL;
```

Tokens never leave via any endpoint. Decryption lives only in `auth/token_store.py`.

## `conversations`

```sql
CREATE TABLE conversations (
    id               CHAR(21) PRIMARY KEY,
    user_id          CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title            TEXT,           -- NULL = derive from the first message
    archived_at      TIMESTAMPTZ,
    last_message_at  TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON conversations (user_id, last_message_at DESC)
    WHERE archived_at IS NULL;
```

`title IS NULL` means the user has not set one, so the display falls back to the first message. `last_message_at` is a deliberate exception — the list sorts on it and `max()` per row on every page is the wrong trade.

## `messages`

Who said what, in what order. Nothing about execution.

```sql
CREATE TABLE messages (
    id               CHAR(21) PRIMARY KEY,
    conversation_id  CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id          CHAR(21) NOT NULL,
    run_id           CHAR(21),          -- assistant turns; FK added after runs exists
    seq              INTEGER NOT NULL,  -- per conversation, deterministic order

    role             message_role NOT NULL,
    content          JSONB NOT NULL,
    hidden           BOOLEAN NOT NULL DEFAULT false,   -- in LLM context, not on screen

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE INDEX ON messages (conversation_id, seq);
CREATE INDEX ON messages (run_id) WHERE run_id IS NOT NULL;
```

`content` is an ordered list of blocks. Prose inline; anything with a lifecycle by reference:

```json
[
  { "type": "text",   "data": { "markdown": "I found your booking (TK1234)..." } },
  { "type": "input",  "ref":  "pi_V1StGXR8_Z5jdHi6B" },
  { "type": "action", "ref":  "ac_bZ9kL2mQx7RtY4nWs" }
]
```

The reference is the on-screen ordering, which is why it is not a duplicate of the child's
`message_id` — that exists for cascade and for "all prompts for this user".

`seq` is allocated in the same transaction as the insert. The unique constraint catches a
race; the writer retries.

## `runs`

One execution of classify -> plan -> execute -> synthesize. This is the brief's
`conversations` row, correctly named. A run can produce several assistant messages — a
clarification, then the answer — which is how pause and resume work without rewriting
history.

```sql
CREATE TABLE runs (
    id                  CHAR(21) PRIMARY KEY,
    conversation_id     CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id             CHAR(21) NOT NULL,
    trigger_message_id  CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,

    intent              JSONB,       -- classifier output, incl. resolved_window
    planner_tier        SMALLINT,    -- 1 template | 2 composed | 3 replan | 4 step_loop

    status              run_status NOT NULL DEFAULT 'running',
    error               JSONB,
    token_usage         JSONB,       -- {prompt, completion, model, usd}

    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ
);

CREATE INDEX ON runs (conversation_id, started_at DESC);
CREATE INDEX ON runs (user_id, started_at DESC);
CREATE INDEX ON runs (status) WHERE status IN ('running', 'awaiting_input');
```

The partial index is how a worker finds runs to resume after a restart.

## `node_executions`

One row per step. Live progress feed, step list, and the record afterwards — the same
rows read at three moments.

```sql
CREATE TABLE node_executions (
    id               CHAR(21) PRIMARY KEY,
    run_id           CHAR(21) NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    conversation_id  CHAR(21) NOT NULL,
    user_id          CHAR(21) NOT NULL,
    message_id       CHAR(21) REFERENCES messages(id) ON DELETE SET NULL,

    seq              SMALLINT NOT NULL,      -- display order within the run
    node_id          VARCHAR(64) NOT NULL,   -- 'gmail_search', unique in the plan
    op               VARCHAR(64) NOT NULL,   -- 'gmail.search_emails'
    round            SMALLINT NOT NULL DEFAULT 0,
    args             JSONB NOT NULL,         -- stored post-templating
    depends_on       TEXT[] NOT NULL DEFAULT '{}',

    status           node_status NOT NULL DEFAULT 'pending',
    result           JSONB,     -- to_ui() output; to_llm() trims this at read time
    outcome          JSONB,     -- non-success detail: {reason, class, code, message}
    retries          JSONB,     -- [{at, error_class, google_status, backoff_ms}]

    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    UNIQUE (run_id, node_id, round)
);

CREATE INDEX ON node_executions (run_id, seq);
CREATE INDEX ON node_executions (message_id, seq) WHERE message_id IS NOT NULL;
CREATE INDEX ON node_executions (conversation_id, started_at DESC);
CREATE INDEX ON node_executions (user_id, started_at DESC);
CREATE INDEX ON node_executions ((split_part(op, '.', 1)), status);
```

`message_id` is which assistant message reported this node — set when the message is
emitted. Without it, a paused run's trace would appear under both of its messages.

## `pending_inputs`

Anything the system needs from the person: which John, send this?, which week, pick the
attachments. One mechanism for all of it.

```sql
CREATE TABLE pending_inputs (
    id                CHAR(21) PRIMARY KEY,
    run_id            CHAR(21) NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    message_id        CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id           CHAR(21) NOT NULL,
    node_execution_id CHAR(21) REFERENCES node_executions(id) ON DELETE SET NULL,

    kind              input_kind NOT NULL,
    blocking          BOOLEAN NOT NULL DEFAULT false,

    prompt            JSONB NOT NULL,   -- {question, help_text}
    value_schema      JSONB NOT NULL,   -- JSON Schema — the validation authority
    options           JSONB,            -- [{id, label, meta}] display only

    status            input_status NOT NULL DEFAULT 'pending',
    response          JSONB,
    expires_at        TIMESTAMPTZ NOT NULL,
    answered_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON pending_inputs (user_id, status);
CREATE INDEX ON pending_inputs (message_id);
CREATE INDEX ON pending_inputs (expires_at) WHERE status = 'pending';
```

`blocking = true` pauses the run; answering resumes it. `blocking = false` means the run
finished and this is waiting on a yes.

Rows are never deleted — status changes. That is what lets a reopened chat show the card in
its answered state rather than a frozen snapshot.

**Supersede rule:** when a new run in the same conversation creates a prompt with the same
`kind` and the same originating `op`, any still-`pending` prompt matching that pair is marked
`superseded` in the same transaction.

## `actions`

Every side effect the system intends to perform. Nothing reaches Google except through a row
here. `requires_input_id` is NOT NULL, so the database guarantees no confirm-requiring write
exists without a prompt gating it.

```sql
CREATE TABLE actions (
    id                CHAR(21) PRIMARY KEY,
    message_id        CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id           CHAR(21) NOT NULL,
    node_execution_id CHAR(21) REFERENCES node_executions(id) ON DELETE SET NULL,
    requires_input_id CHAR(21) NOT NULL REFERENCES pending_inputs(id),

    op                VARCHAR(64) NOT NULL,   -- 'gmail.send_email'
    payload           JSONB NOT NULL,         -- exactly what will execute
    revisions         JSONB NOT NULL DEFAULT '[]',  -- [{payload, replaced_at}]
    dedupe_key        UUID NOT NULL,          -- uuid5(user_id|op|payload|conversation)

    status            action_status NOT NULL DEFAULT 'draft',
    external_ref      VARCHAR(255),           -- Gmail draft created up front
    result            JSONB,
    error             JSONB,
    attempts          SMALLINT NOT NULL DEFAULT 0,
    job_id            VARCHAR(64),

    executed_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX ON actions (dedupe_key)
    WHERE status IN ('draft', 'approved', 'running');

CREATE INDEX ON actions (user_id, status);
CREATE INDEX ON actions (message_id);
CREATE INDEX ON actions (requires_input_id);
```

The unique index is **partial on purpose**. Dedup applies to in-flight actions only. Once
something is `done`, `cancelled`, `expired` or `failed`, an identical one is a legitimate new
request — a resend after cancelling, the same reminder next week.

## `conversation_entities`

What this conversation has referred to. Serves "that email about the proposal" and
"move the meeting with John". Written whenever a node surfaces something to the user, so a
follow-up resolves against ~20 rows instead of parsing five runs' worth of result blobs.

```sql
CREATE TABLE conversation_entities (
    id               CHAR(21) PRIMARY KEY,
    conversation_id  CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id          CHAR(21) NOT NULL,
    run_id           CHAR(21) REFERENCES runs(id) ON DELETE SET NULL,

    entity_type      VARCHAR(16) NOT NULL,   -- email | event | file | person
    entity_ref       VARCHAR(255) NOT NULL,  -- provider id or email address
    label            TEXT NOT NULL,          -- "Acme Q3 proposal — pricing"
    meta             JSONB,                  -- sender, date — what the chip shows

    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, entity_type, entity_ref)
);

CREATE INDEX ON conversation_entities (conversation_id, last_seen_at DESC);
```

## `audit_log`

What changed outside our system: emails sent, events deleted, files shared, tokens granted
and revoked, data purged.

Separate from `actions` because `actions` is working state that updates and cascades away
with a deleted conversation, while these rows never change and never cascade. It also covers
things that are not actions at all — OAuth grant, disconnect, an admin operation.

Bodies are never stored. `payload_hash` is a uuid5 of the full content, so you can prove this
exact email was sent without keeping the text.

```sql
CREATE TABLE audit_log (
    id               BIGSERIAL PRIMARY KEY,
    user_id          CHAR(21) NOT NULL,
    conversation_id  CHAR(21),
    actor            VARCHAR(16) NOT NULL,   -- user | system | worker
    action           VARCHAR(64) NOT NULL,   -- gmail.send_email, auth.revoke
    resource_id      VARCHAR(255),
    payload_hash     UUID,
    payload_visible  JSONB,                  -- recipients, subject
    status           VARCHAR(16) NOT NULL,
    error            JSONB,
    ip               INET,
    user_agent       TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON audit_log (user_id, created_at DESC);
CREATE INDEX ON audit_log (action, created_at DESC);
```

`BIGSERIAL` deliberately — append-only, highest volume, and a sequence beats a generated id.

---

# `sync_` — the Google mirror

A cache. Drop these and a resync rebuilds everything. Nothing here is a source of truth.

**Every `sync_` table carries `embed_model`.** It records the exact model that produced
that row's `embedding` — `openai:text-embedding-3-small`, the same string `EMBED_MODEL`
uses. Fill it from `app.llm.embed_model_id()`. `''` means the row has no vector yet.

It is there because chat models are swappable and embedding models are not. Two models
can both produce 1536 dimensions and still disagree completely about what each dimension
means, so a cosine distance across a mix is not a weak signal — it is a made-up one that
looks exactly like a real one. No error, no warning, just the wrong emails. Recording the
model is what lets the search path call `app.llm.assert_same_embed_model` and fail loudly
instead. Each table therefore also indexes `(user_id, embed_model)`: search prefilters on
`user_id` already, so putting `embed_model` beside it makes the check an index-only scan
rather than a read of every row the prefilter matched.

Changing `EMBED_MODEL` means re-embedding every row — `backend/scripts/reembed.py`, which
works in batches and writes the new name into `embed_model` as it goes. See
`docs/MODELS.md` §6.

## `sync_messages` · `sync_events` · `sync_files`

The mirror: a disposable cache of what the connectors hold, with a vector index
over it. Drop all three and a backfill rebuilds every row.

**One table per shape, not per source.** Gmail, Outlook mail, Slack, Teams and
Jira comments are all `sync_messages`. Google and Outlook calendars are both
`sync_events`. Drive, OneDrive, Dropbox and Notion are all `sync_files`. A new
connector picks the shape it fits and needs no DDL — which is the whole point:
three tables cover an open-ended set of sources.

These were one table (`sync_items`) discriminated by `kind`. That fixed the
original problem — three near-identical tables, three sets of indexes, three
arms in every fan-out — but it charged twice for it:

* **Columns could not be honest.** `ends_at` sat NULL on every email. The time
  a thing happened was `occurred_at` whether it was sent, scheduled or
  modified. Worst, recipients lived in a JSONB `participants` blob, so
  *"mail **to** Sarah"* compiled to a per-row function call that no index could
  serve, while *"mail **involving** Sarah"* was a fast GIN hit. Different
  questions, and only one of them was indexed.
* **One ANN index served every shape.** A vector search for events walked an
  index holding all your mail and files, took the top-k and filtered by kind
  afterwards. As mail volume grows it dominates the neighbourhood and event
  searches quietly under-return.

### The shared spine

Identical in all three, so one search layer reads them without knowing which:

```sql
id            CHAR(21) PRIMARY KEY,
user_id       CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
connector     VARCHAR(32)  NOT NULL,   -- where it came from
source_id     VARCHAR(255) NOT NULL,   -- the id that source gave it
scope_key     VARCHAR(255) NOT NULL DEFAULT '',
chunk_index   SMALLINT     NOT NULL DEFAULT 0,
labels        TEXT[],
url           TEXT,
attributes    JSONB NOT NULL DEFAULT '{}'::jsonb,
content_hash  UUID NOT NULL,           -- unchanged means skip re-embedding
embedding     VECTOR(1536),
embed_model   VARCHAR(64) NOT NULL DEFAULT '',
updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()

UNIQUE (user_id, connector, scope_key, source_id, chunk_index)
```

`scope_key` is the namespace that makes `source_id` unique inside a connector —
a mailbox, a Slack workspace, a calendar. For events it is load-bearing rather
than cosmetic: **Google event ids are unique only within a calendar**, so two
calendars can hand out the same id for different meetings.

### What each shape adds

| `sync_messages` | `sync_events` | `sync_files` |
|---|---|---|
| `subject` `body` | `title` `description` | `name` `content` |
| `from_email` `from_name` | `organizer_email` | `owner_email` |
| **`to_emails`** `cc_emails` | **`attendee_emails`** `attendees` | **`shared_with_emails`** |
| `participant_emails` *(generated)* | | |
| `sent_at` `thread_id` | `starts_at` `ends_at` `all_day` | `modified_at` |
| `is_unread` `has_attachments` | `location` `status` `recurring_event_id` | `mime_type` `size_bytes` `folder_id` `folder_path` `is_shared` |

The bolded columns are `CITEXT[]` with GIN indexes — case-insensitive
containment. `"What's on my calendar next week where john@company.com is
invited?"` is one index hit and **zero model calls**.

`participant_emails` on messages is `GENERATED ALWAYS` from from/to/cc, so
"involving X" and "addressed to X" are both indexed and can never disagree.
On events the flat `attendee_emails` is derived from `attendees` on write for
the same reason.

### Indexes

Each table carries its own, which is the point of the split:

```sql
-- per table: own HNSW, so top-k events means top-k events
CREATE INDEX {t}_embedding_idx ON {t} USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX {t}_tsv_idx    ON {t} USING GIN (tsv);
CREATE INDEX {t}_labels_idx ON {t} USING GIN (labels);

-- messages
CREATE INDEX sync_messages_to_idx   ON sync_messages USING GIN (to_emails);
CREATE INDEX sync_messages_cc_idx   ON sync_messages USING GIN (cc_emails);
CREATE INDEX sync_messages_time_idx ON sync_messages (user_id, connector, sent_at DESC);

-- events
CREATE INDEX sync_events_attendees_idx ON sync_events USING GIN (attendee_emails);
CREATE INDEX sync_events_live_idx      ON sync_events (user_id, starts_at)
    WHERE status IS DISTINCT FROM 'cancelled';

-- files: "PDFs from last month" is a mime + time question
CREATE INDEX sync_files_mime_idx ON sync_files (user_id, mime_type, modified_at DESC)
    WHERE mime_type IS NOT NULL;
```

`tsv` weights the title above the body (`A` over `B`), which is what makes
*"find the mail **with subject** X"* behave differently from *"find mail
**about** X"*.

### Where the next connector goes

> **A field you filter on is a column. A field you only display is an
> `attributes` key.**

Burying a queryable dimension in JSONB is exactly how `to` ended up unindexed.
By that rule Jira issues want a fourth table when they arrive — `assignee` and
`due_at` are things people filter on — while Jira comments are simply
`sync_messages` rows whose `thread_id` is the issue key. That is what lets one
search cover mail, chat and tickets at once.

## `sync_state`

Where each sync got to. `cursor` holds whatever the service uses — Gmail's `historyId`,
Calendar's `syncToken`, Drive's `pageToken`. Advanced only after the upsert commits, so a
crash reprocesses rather than skips.

```sql
CREATE TABLE sync_state (
    id                    CHAR(21) PRIMARY KEY,
    user_id               CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    service               sync_service NOT NULL,

    cursor                JSONB,
    backfill_complete     BOOLEAN NOT NULL DEFAULT false,
    backfill_cursor       JSONB,

    last_synced_at        TIMESTAMPTZ,   -- last attempt
    last_success_at       TIMESTAMPTZ,   -- drives the lag figure on /sync/status
    last_error            JSONB,
    items_indexed         INT NOT NULL DEFAULT 0,

    consecutive_failures  SMALLINT NOT NULL DEFAULT 0,
    circuit_open_until    TIMESTAMPTZ,

    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, service)
);
```

---

## `job_failed_tasks`

Jobs that exhausted their retries. A sweeper replays the ones whose failure class is
transient; the rest wait for a person. The count shows on `/sync/status`.

```sql
CREATE TABLE job_failed_tasks (
    id              CHAR(21) PRIMARY KEY,
    user_id         CHAR(21),
    task_name       VARCHAR(128) NOT NULL,
    queue           VARCHAR(32) NOT NULL,
    task_input      JSONB,                  -- {args, kwargs}
    error_class     VARCHAR(32) NOT NULL,   -- TRANSIENT | RATE_LIMITED | ...
    error           JSONB,
    traceback       TEXT,
    attempts        SMALLINT NOT NULL,
    celery_task_id  VARCHAR(64),
    status          VARCHAR(16) NOT NULL DEFAULT 'open',  -- open|replayed|ignored
    first_failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_failed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON job_failed_tasks (status, last_failed_at);
CREATE INDEX ON job_failed_tasks (user_id, status);
```

---

## Derived, not stored

| Wanted | How |
|---|---|
| Services that failed this run | `SELECT DISTINCT split_part(op,'.',1) FROM node_executions WHERE run_id=$1 AND status IN ('failed','skipped','timeout')` |
| What the LLM saw for a node | `Op.to_llm(result)` |
| Confirm card preview | `Op.preview(payload)` |
| Step progress line | `Op.progress_label(args)` |
| Node attempts | `coalesce(jsonb_array_length(retries),0) + 1` |
| Node duration | `finished_at - started_at` |
| Run latency | `finished_at - started_at`, or first assistant message minus `started_at` while paused |
| Replan rounds | `max(round)` over the run's nodes |
| Node's service | `split_part(op,'.',1)`, expression-indexed |
| Action expiry | the gating prompt's `expires_at` |
| Conversation display title | `title`, else the first user message |
| Which card renderer | mapped from `pending_inputs.kind` |

## Deliberate exceptions

| | Why |
|---|---|
| `user_id` on child tables | Every repository method takes it first; no query can cross tenants by accident |
| `node_executions.conversation_id` | Direct trace queries without two hops |
| `conversations.last_message_at` | The list sorts on it; `max()` per row on every page is the wrong trade |
| `node_executions.result`, `actions.payload` | Snapshots. They record what happened and must not shift when Gmail changes |
| `pending_inputs.run_id` | Reachable via message, but resume is the critical path |

## Implementation rules

1. **One transaction.** An assistant message and every prompt and action its `content`
   references are written together. A ref that does not resolve is dropped on read and
   logged, never rendered as an empty box.
2. **Circular FK.** `messages.run_id` -> `runs.id` and `runs.trigger_message_id` ->
   `messages.id`. Insert order is user message, run, assistant message, so no row ever needs
   a value that does not exist. Alembic needs `use_alter=True` on one side.
3. **No hard delete.** `conversations.archived_at` hides a thread; rows stay. Actual deletion
   only on account disconnect.

**15 tables** — 10 app, 4 sync, 1 job.
