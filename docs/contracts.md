# Build Contracts

Every module codes against these. Read `docs/schema.md` for the tables.

## Stack

- Python 3.12, FastAPI, SQLAlchemy 2.0 async + asyncpg, Alembic, Pydantic v2,
  pydantic-settings, Celery 5 + Redis, openai SDK, structlog
- Frontend: Vite + React 18 + TypeScript + Tailwind. No component library.
- Everything runs via `docker compose up`.

## Conventions

- All ids: nanoid, 21 chars, generated in the service. `app.core.ids.new_id()`.
- Fingerprints (`content_hash`, `dedupe_key`, `payload_hash`): `app.core.ids.fingerprint(ns, text) -> uuid.UUID` using uuid5.
- All datetimes stored UTC, tz-aware. Never naive.
- Every repository function takes `user_id` as its FIRST argument. No exceptions.
- Async everywhere in the API path. Celery tasks are sync wrappers around async via `asyncio.run`.
- Errors raise `app.core.errors.AppError(code, message, http, details)`.

## Package layout — who owns what

```
backend/app/
  main.py  config.py
  core/       ids.py crypto.py errors.py cache.py ratelimit.py audit.py logging.py
  db/         base.py session.py models.py repositories/*.py
  auth/       google_oauth.py token_store.py deps.py
  google/     client.py retry.py quota.py
  services/   gmail.py gcal.py gdrive.py          # thin API wrappers
  search/     embedder.py chunking.py hybrid.py probe.py
  ops/        base.py registry.py gmail_ops.py gcal_ops.py drive_ops.py meta_ops.py
  orchestrator/
              temporal.py prepass.py front_door.py route.py validate.py
              dispatch.py render.py events.py entities.py
  tasks/      celery_app.py sync_gmail.py sync_gcal.py sync_gdrive.py
              embed.py actions.py maintenance.py
  api/v1/     query.py auth.py sync.py prompts.py conversations.py events.py search.py
frontend/src/
  App.tsx main.tsx index.css
  lib/api.ts lib/sse.ts lib/types.ts
  components/Chat.tsx TracePanel.tsx PromptCard.tsx ActionCard.tsx
             ActionCard.tsx Activity.tsx Chat.tsx Composer.tsx Empty.tsx GoogleMark.tsx History.tsx Integrations.tsx MessageList.tsx PromptCard.tsx SignIn.tsx SyncBar.tsx SyncStatus.tsx UsageRing.tsx UserMenu.tsx Widget.tsx
```

## `app.core.ids`

```python
def new_id() -> str: ...                       # nanoid, 21 chars
def fingerprint(ns: str, text: str) -> UUID: ...   # uuid5
```

## `app.core.errors`

```python
class AppError(Exception):
    def __init__(self, code: str, message: str, http: int = 400, details: dict | None = None)

# codes: VALIDATION_ERROR 422 · NOT_AUTHENTICATED 401 · GOOGLE_REAUTH_REQUIRED 428
#        RATE_LIMITED 429 · PROMPT_NOT_PENDING 409 · PROMPT_VALUE_INVALID 422
#        ORCHESTRATION_TIMEOUT 504 · GOOGLE_UNAVAILABLE 503 · NOT_FOUND 404 · INTERNAL 500
```

## `app.google.retry`

```python
class ErrorClass(StrEnum):
    TRANSIENT = "TRANSIENT"; RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"; AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_REVOKED = "AUTH_REVOKED"; PRECONDITION = "PRECONDITION"
    NOT_FOUND = "NOT_FOUND"; INVALID = "INVALID"; UNKNOWN = "UNKNOWN"

def classify(exc: Exception) -> ErrorClass: ...
def backoff(cls: ErrorClass, attempt: int) -> float: ...   # full jitter
def retryable(cls: ErrorClass) -> bool: ...
```

## `app.ops.base` — THE central interface

```python
@dataclass
class InputRequest:
    kind: str                 # confirm|choice|multi_choice|text|form|date_range
    question: str
    value_schema: dict
    options: list[dict] | None = None
    blocking: bool = True

@dataclass
class OpResult:
    data: dict
    needs_replan: bool = False
    replan_reason: str | None = None
    needs_input: InputRequest | None = None

class Op:
    name: str                      # "gmail.search_emails"
    args_model: type[BaseModel]
    output_fields: list[str]       # what {{step.X}} may reference
    is_local: bool = False         # reads our mirror, not Google
    is_write: bool = False
    needs_confirm: bool = False
    timeout_s: float = 6.0
    max_attempts: int = 2

    async def run(self, ctx: "OpContext", args: dict) -> OpResult: ...
    def progress_label(self, args: dict) -> str: ...
    def to_llm(self, data: dict, budget: int = 900) -> dict: ...
    def catalogue_line(self) -> str: ...          # one line for the planner prompt

class ConfirmableOp(Op):
    needs_confirm = True
    def preview(self, payload: dict) -> dict: ...
    def confirm_question(self, payload: dict) -> str: ...
    async def execute(self, ctx, payload: dict) -> dict: ...   # the irreversible part

@dataclass
class OpContext:
    user_id: str
    conversation_id: str
    run_id: str
    session: AsyncSession
    google: "GoogleClients"
    now: datetime
    tz: str
```

`registry.py` exposes `REGISTRY: dict[str, Op]`, `catalogue() -> str`, `get(name)`.

### The op inventory — 28 ops

The planner never sees this list. `catalogue()` builds its prompt section from
`REGISTRY` at call time, so an op that exists is offerable and an op that is
deleted stops being offered, with nothing to keep in step by hand.

`local` reads our mirror rather than Google. `write` has a side effect.
`confirm` cannot execute without a `pending_inputs` row gating it — a rule the
database enforces through `actions.requires_input_id NOT NULL`, not something
the planner is trusted to remember.

| service | ops |
|---|---|
| **gmail** | `search_emails` `get_emails` · `draft_email` `send_email`* `update_labels` |
| **gcal** | `search_events` `get_events` `match_event` `find_conflicts` `find_free_slots` · `create_event`* `update_event`* `delete_event`* |
| **gdrive** | `search_files` `get_files` · `share_file`* `create_folder` `move_file` |
| **ask** | `ask.user` — a question, as a step in the plan |
| **chat** | `chat.set_title` — name the conversation on its first turn |
| **resolve** | `resolve.person` `resolve.reference` — "which John", "that email" |
| **data / llm / page / search / action** | `data.filter` `llm.map` `llm.extract` `page.more` `search.all` `action.revise` |

\* asks before it acts.

The brief asks for five methods per service — search, get, and the writes. All
fifteen are here; the rest exist because the orchestrator needs them to avoid
model calls it would otherwise have to make.

### Freshness: when a read goes to Google

`freshness: "cached"` reads the mirror. `freshness: "live"` calls Google, and
the planner sets it before a write so an etag is not stale.

A cached read still reaches past the mirror in two cases, because answering
from a mirror that cannot answer is worse than being slow:

* **the mirror returns nothing** — a meeting booked five minutes ago is not in
  it yet, and "no events matched" would be a wrong answer rather than a slow
  one;
* **the mirror is behind** — no successful sync inside `SYNC_INTERVAL_MIN`, or
  sync is currently failing. A stale mirror returns rows, so nothing looks
  wrong; the answer is just quietly out of date.

Both are guarded to one refresh per corpus per minute, so a five-step plan
buys one Google call rather than five. Rows fetched this way are embedded
inline before the re-read — a row with no vector is invisible to the vector
arm, which is how "meetings next week" misses an event called *John Wick Visit
Offline*. Never having synced at all is not treated as staleness: an empty
mirror is already covered by the first case.

## The plan JSON the model returns

```json
{
  "type": "plan",
  "intent": {"name": "...", "services": ["gmail"], "has_write": false, "confidence": 0.9},
  "answer_style": "card | template:<name> | prose",
  "steps": [
    {"id": "readable_name", "op": "gmail.search_emails",
     "args": {...}, "depends_on": [], "expect": "one|many",
     "optional": false, "freshness": "cached|live",
     "gate": {"left": "...", "test": "exists|empty|count_gt|within|before|equals|contains", "right": "..."},
     "speculate": false,
     "defer": {"reason": "...", "budget": 2}}
  ]
}
```

Other verbs: `{"type":"answer","text":"..."}` · `{"type":"revise","action_id":"..","patch":{..}}`
· `{"type":"answer_input","input_id":"..","value":..}`

Reference forms resolved by `dispatch.bind()`:
`{{step.path}}` · `{{step.hits[0].id}}` · `{{step.hits[*].id}}` · `{{search.gmail[0].extracted.pnr}}`
· `{{windows.<name>.start}}` · `time_phrase: "tomorrow"` (resolved at bind time)

## `app.orchestrator.temporal`

```python
@dataclass
class Window:
    start: datetime; end: datetime; tz: str; interpretation: str

def resolve(phrase: str, tz: str, week_start: int, now: datetime) -> Window | None: ...
def scan(text: str, tz: str, week_start: int, now: datetime) -> dict[str, Window]: ...
```

Rules: "next <weekday>" = that weekday in the FOLLOWING iso week (on a Tuesday,
"next Tuesday" is +7d). "next week" = Mon..Sun of iso week+1 honouring week_start.
Windows half-open [start, end). All maths on local wall time via zoneinfo, then UTC.

## `app.orchestrator.events` — SSE

```python
async def publish(run_id: str, type: str, data: dict) -> None: ...
async def subscribe(run_id: str) -> AsyncIterator[dict]: ...
```

Event types: `run.started · progress · probe.done · intent · plan.step · step.started
· step.retrying · step.finished · content.delta · input.raised · action.prepared
· token · run.paused · run.complete · error`
Conversation channel: `action.done · action.failed`

Envelope: `{"v":1,"seq":<int>,"type":"...","run_id":"...","ts":"iso","data":{...}}`

## API surface

```
POST   /api/v1/query                      {query, conversation_id?}
GET    /api/v1/runs/{run_id}/events       SSE
GET    /api/v1/auth/google                -> 302
GET    /api/v1/auth/google/callback
GET    /api/v1/auth/me
DELETE /api/v1/auth/google
POST   /api/v1/sync/trigger               {services?, mode}
GET    /api/v1/sync/status
GET    /api/v1/prompts?status=pending
POST   /api/v1/prompts/{id}/respond       {value}
POST   /api/v1/prompts/{id}/cancel
GET    /api/v1/conversations
GET    /api/v1/conversations/{id}
GET    /api/v1/search                     debug + eval, returns score components
GET    /healthz  /readyz  /metrics
```

Error body: `{"error":{"code","message","details","request_id"}}`

## Celery

Queues: `sync` `embed` `actions` `orchestration` `maintenance`
Beat: `sync.dispatch_all_users` */15m · `maintenance.refresh_tokens` */10m
· `maintenance.expire_prompts` hourly · `maintenance.sweep_dlq` */30m
· `maintenance.prune_sync` daily 03:00 · `metrics.freshness` */5m

Settings: `task_acks_late=True`, `worker_prefetch_multiplier=1`,
`task_reject_on_worker_lost=True`, `visibility_timeout=900`.

## Frontend types (`lib/types.ts`)

Mirror the API responses. SSE via `EventSource` on `/api/v1/runs/{id}/events`.
Two columns: the chat list left, the conversation right. The steps render
inline under the turn they belong to — pending, running, settled — and collapse
to a one-line summary when the run ends. Cards render from `pending_inputs.kind`,
and a `choice` answer is submitted in the shape `value_schema` asks for, which
is usually `{"<ref>_id": "..."}` rather than a bare id.
Plain button text: "Send it", "Not now", "Edit".
