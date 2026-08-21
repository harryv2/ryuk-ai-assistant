# API

Version `v1`. Everything under `/api/v1`, except the three probes at the root.

This document is the contract a client codes against. It matches
`docs/contracts.md` (the locked interface) and `docs/schema.md` (the tables).
Where a field maps straight onto a column, the column is named.

- [Basics](#basics)
- [Error envelope](#error-envelope)
- [Rate limits](#rate-limits)
- [POST /api/v1/query](#post-apiv1query)
- [GET /api/v1/runs/{run_id}/events](#get-apiv1runsrun_idevents)
- [The two-phase write flow](#the-two-phase-write-flow)
- [The `value_schema` contract](#the-value_schema-contract)
- [Prompts](#prompts)
- [Conversations](#conversations)
- [Auth](#auth)
- [Sync](#sync)
- [GET /api/v1/search](#get-apiv1search)
- [Health and metrics](#health-and-metrics)
- [Known limits](#known-limits)

---

## Basics

**Base URL.** `http://localhost:5173` in development — the browser only ever
talks to Vite, which proxies `/api` to the API container. The API's own port is
exposed to the compose network but never published, so one origin is the whole
surface. All examples below use
`$BASE`:

```bash
export BASE=http://localhost:5173
```

**Auth.** A signed session cookie, `alpha_session`, set by the OAuth callback.
Every endpoint except `/api/v1/auth/google`, `/api/v1/auth/google/callback` and
the root probes requires it. Missing or expired cookie is `401
NOT_AUTHENTICATED`. A cookie that is still valid while the *Google* grant is
dead is `428 GOOGLE_REAUTH_REQUIRED` — a different thing, and the client must
send the user back through the consent screen rather than through sign-in.

Google access and refresh tokens never appear in any response body. They are
AES-256-GCM blobs in `oauth_tokens`, decrypted only inside
`auth/token_store.py`.

**Content type.** `application/json; charset=utf-8` for requests and responses.
Two endpoints stream: `POST /api/v1/query?stream=ndjson` returns
`application/x-ndjson`, and `GET /api/v1/runs/{run_id}/events` returns
`text/event-stream`.

**Request ids.** Send `X-Request-ID` and it is echoed back and stamped on every
log line for the request. Omit it and the server generates one. It is also
returned in the error envelope as `request_id`. Sending your own is how you
correlate a `POST /query` with the SSE stream you open a moment later.

Every response also carries `X-Elapsed-Ms`, the server-side wall time.

**Ids.** All resource ids are nanoid, 21 characters, from
`app.core.ids.new_id()`. They are opaque — do not parse them. The only UUIDs
in the system are content fingerprints (`content_hash`, `dedupe_key`,
`payload_hash`) and those are internal.

**Time.** Every timestamp in a request or a response is RFC 3339 with an
explicit offset, and the server emits UTC (`...Z`). Never send a naive
timestamp; it is rejected with `VALIDATION_ERROR`. Natural-language time in a
query string ("next Tuesday") is resolved server-side against the user's
`users.timezone` and `users.work_week_start`, and the window it resolved to is
returned so the client can show its work.

**Idempotency.** Reads are idempotent by nature. The one write path —
approving a prepared action — is idempotent through `actions.dedupe_key`
(uuid5 over user, op, canonical payload and conversation) with a partial unique
index covering only `draft`, `approved` and `running`. Approving twice does not
send twice. Outbound mail additionally carries an `X-Orchestrator-Idem` header,
so a worker that retries after an ambiguous failure checks Sent for that header
before trying again.

---

## Error envelope

Every failure, from every endpoint, has the same body:

```json
{
  "error": {
    "code": "PROMPT_VALUE_INVALID",
    "message": "That answer does not fit what was asked.",
    "details": {
      "errors": [
        { "path": "$.approve", "msg": "'yes' is not of type 'boolean'" }
      ]
    },
    "request_id": "NKgjLgOI3xAxLtdLLB0pX"
  }
}
```

`message` is safe to render to a person. `details` is for the developer and its
shape varies by code; it is always an object, never null.

| Code | HTTP | When | Client should |
|---|---|---|---|
| `VALIDATION_ERROR` | 422 | The body or query string did not parse or validate. `details.errors` is a list of `{loc, msg, type}`. | Fix the request. Do not retry unchanged. |
| `NOT_AUTHENTICATED` | 401 | No session cookie, or it is expired or tampered with. | Send the user to `/api/v1/auth/google`. |
| `GOOGLE_REAUTH_REQUIRED` | 428 | The session is fine but the Google grant is revoked, or the refresh token failed enough times that `oauth_tokens.refresh_failures` tripped. | Show a reconnect banner, link to `/api/v1/auth/google`. |
| `RATE_LIMITED` | 429 | 100 queries per user per hour, or the Google quota governor shed an interactive request. `Retry-After` is set. | Back off for `Retry-After` seconds. |
| `PROMPT_NOT_PENDING` | 409 | The prompt was already answered, cancelled, expired, or superseded by a newer prompt of the same kind and op. `details.status` says which. | Refresh the card from `GET /api/v1/prompts`. Do not retry. |
| `PROMPT_VALUE_INVALID` | 422 | The posted `value` failed jsonschema validation against `pending_inputs.value_schema`. `details.errors` lists the failing JSON paths. | Fix the value. The schema is in the prompt object. |
| `ORCHESTRATION_TIMEOUT` | 504 | The run passed `HARD_DEADLINE_MS` (12s default). Partial results are already in the run and on the SSE stream. | Read `GET /api/v1/conversations/{id}` for what did land. |
| `GOOGLE_UNAVAILABLE` | 503 | Google returned 5xx past the in-request retry budget, or the per-(user, service) circuit breaker is open. `details.service` and `details.retry_after_s`. | Retry after the hint. Other services may still be fine. |
| `NOT_FOUND` | 404 | No such conversation, run, or prompt **for this user**. Cross-tenant reads are indistinguishable from missing rows, on purpose. | Stop. |
| `INTERNAL` | 500 | Anything unhandled. In production `message` is generic; the detail is in the logs under `request_id`. | Retry once, then surface. |

The table is closed. `app/core/errors.py` refuses to construct an unknown code
— it downgrades to `INTERNAL` and records the attempted code in
`details.unknown_code`, so a typo shows up in a log rather than inventing a new
public contract.

---

## Rate limits

100 queries per user per hour (`RATE_LIMIT_PER_HOUR`), enforced by a Redis
sliding window keyed on `user_id`. Only `POST /api/v1/query` counts. Reads of
conversations, prompts and sync status do not.

Every `POST /api/v1/query` response carries:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 96
X-RateLimit-Reset: 1755712800
```

Separately, a Redis token bucket models Google's 250 units/sec, split 70%
background sync / 30% reserved for interactive traffic. When the interactive
share is exhausted the request fails fast with `RATE_LIMITED` rather than
queueing behind a sync job.

---

## POST /api/v1/query

The one entry point. Takes a natural-language query, returns an answer — and,
if the query implied a side effect, a *prepared* action plus the prompt that
gates it. Nothing reaches Google as a write from this endpoint. Ever.

### Request

`POST /api/v1/query`

Query parameters:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `stream` | `off` \| `ndjson` | `off` | `off` buffers and returns one JSON object when the run settles. `ndjson` streams the event union as newline-delimited JSON over the same POST. See [the streaming variant](#the-ndjson-streaming-variant). |

Body:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `query` | string, 1–2000 chars | yes | What the person typed. Sent verbatim — the server does the normalising. |
| `conversation_id` | string (21) | no | Continue an existing thread. Omit to start a new one; the id of the new conversation is in the response. A `conversation_id` belonging to another user is `404`, not `403`. |
| `timezone` | IANA string | no | Overrides `users.timezone` for this query only. Use it when the browser's zone differs from the stored one (travelling). Invalid zone is `VALIDATION_ERROR`. |
| `week_start` | int 1–7 | no | Overrides `users.work_week_start` for this query only. 1 = Monday. Changes what "next week" means. |
| `freshness` | `auto` \| `cached` \| `live` | no, default `auto` | `cached` answers from the pgvector mirror only (fastest, up to 15 min stale). `live` forces a read-through to Google for the steps that support it. `auto` lets the planner decide — it picks `live` for superlatives ("the *latest* email from…") and for anything a write will target. |
| `client_request_id` | string ≤ 64 | no | Your own idempotency handle. Re-posting the same `client_request_id` within 10 minutes returns the original run instead of starting a second one. Protects against a double-tapped send button. |

```json
{
  "query": "Cancel my Turkish Airlines flight",
  "conversation_id": "KF336Xj1yXQ90iQZ31fN2",
  "timezone": "Europe/Istanbul",
  "freshness": "auto"
}
```

### Response — 200

The buffered form. This is the run's terminal state: `complete` if it finished,
`awaiting_input` if it paused on a blocking question.

```json
{
  "run_id": "4faGRd0_Gx4BuZxj2frxK",
  "conversation_id": "KF336Xj1yXQ90iQZ31fN2",
  "message_id": "ObkgZf-bc4Mzn_eVs08H3",
  "status": "complete",
  "planner_tier": 2,

  "intent": {
    "name": "cancel_flight",
    "services": ["gmail", "gcal"],
    "has_write": true,
    "confidence": 0.91,
    "entities": { "airline": "Turkish Airlines", "pnr": "TK1234" },
    "windows": {}
  },

  "answer_style": "card",
  "text": "I found your Turkish Airlines booking (TK1234) in an email from Ozan Demir on 15 Jul.\n\n- Calendar event **Istanbul → JFK, TK1988** on Wed 5 Nov, 10:30\n- Drafted a cancellation email to support@turkishairlines.com\n\nWant me to send it?",
  "content": [
    { "type": "text",   "data": { "markdown": "I found your Turkish Airlines booking (TK1234) in an email from Ozan Demir on 15 Jul." } },
    { "type": "action", "ref": "7v1wpk18mA4xFMiZgTznp" },
    { "type": "input",  "ref": "AwmTv620oCkmGlRtvD4hU" }
  ],

### Content blocks

Four kinds. `text` is markdown. `action` and `input` are refs, resolved at read
time against the rows behind them, so a reopened conversation shows what really
happened rather than what we intended.

`widget` is a structured answer the UI draws itself:

```json
{
  "type": "widget",
  "v": 1,
  "widget": "event_list",
  "text": "**Mon Aug 24 to Sun Aug 30** — 6 events\n\n…",
  "replaces_text": true,
  "data": { "items": [{ "id": "e_1", "title": "Platform planning", "starts_at": "…" }] }
}
```

**Never markup.** A widget is data; the client owns every pixel. Model-authored
HTML in a page holding somebody's mailbox is an XSS hole, and sanitising it is
a game you lose slowly.

**`text` is not optional.** It is what a screen reader reads, what a
copy-paste copies, and what a client renders when it meets a `widget` or a `v`
it does not know. Messages are durable: one written today has to still render
in a year.

**`replaces_text`** says whether the widget *is* the answer. `true` for a
template — its rows drawn instead of printed, so showing both is the answer
twice. `false` for one the synthesiser offered, where the prose says something
the widget does not and dropping it would lose the answer.

Kinds are `event_list`, `email_list`, `file_list`, `free_slots` (built from
rows an op returned) and `table`, `list`, `stat`, `key_values`, `timeline`,
`comparison`, `chips` (composable primitives the model may choose). A kind the
client does not know falls back to `text`.

**Interaction is two verbs.** An `ask` button puts a sentence in the composer
as if the person typed it — so it goes down the ordinary query path and a write
still stops at its confirmation card. An `open` button follows a link. Nothing
executes: a button that could act directly would be a second write path around
`actions.requires_input_id`, which the database enforces. A model-authored
`open` url must be one that appeared in that run's own data.

  "actions": [
    {
      "id": "7v1wpk18mA4xFMiZgTznp",
      "op": "gmail.send_email",
      "status": "draft",
      "requires_input_id": "AwmTv620oCkmGlRtvD4hU",
      "external_ref": "r-8841207733012994001",
      "preview": {
        "to": ["support@turkishairlines.com"],
        "subject": "Cancellation request — booking TK1234",
        "body_excerpt": "Hello,\n\nI would like to cancel booking reference TK1234 (Istanbul → JFK, TK1988, 5 November)…",
        "attachments": []
      },
      "payload_fields": ["to", "cc", "subject", "body", "reply_to_message_id"],
      "expires_at": "2026-08-21T09:14:22Z",
      "created_at": "2026-08-20T09:14:22Z"
    }
  ],

  "pending_inputs": [
    {
      "id": "AwmTv620oCkmGlRtvD4hU",
      "kind": "confirm",
      "blocking": false,
      "prompt": {
        "question": "Send the cancellation email to support@turkishairlines.com?",
        "help_text": "Nothing has been sent. This is a Gmail draft until you say so."
      },
      "value_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["approve"],
        "properties": {
          "approve": { "type": "boolean" },
          "patch": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "to":      { "type": "array", "items": { "type": "string", "format": "email" }, "minItems": 1 },
              "cc":      { "type": "array", "items": { "type": "string", "format": "email" } },
              "subject": { "type": "string", "maxLength": 300 },
              "body":    { "type": "string", "maxLength": 20000 }
            }
          },
          "note": { "type": "string", "maxLength": 2000 }
        }
      },
      "options": null,
      "status": "pending",
      "expires_at": "2026-08-21T09:14:22Z",
      "created_at": "2026-08-20T09:14:22Z"
    }
  ],

  "steps": [
    {
      "node_id": "find_booking", "op": "gmail.search_emails", "seq": 0, "round": 0,
      "status": "succeeded", "depends_on": [],
      "args": { "query": "Turkish Airlines booking confirmation", "limit": 5 },
      "result_summary": { "count": 3, "top": "Your Turkish Airlines booking TK1234" },
      "started_at": "2026-08-20T09:14:21.194Z", "finished_at": "2026-08-20T09:14:21.402Z",
      "duration_ms": 208, "attempts": 1
    },
    {
      "node_id": "find_event", "op": "gcal.search_events", "seq": 1, "round": 0,
      "status": "succeeded", "depends_on": [],
      "args": { "q": "TK1988", "time_min": "2026-08-20T00:00:00Z" },
      "result_summary": { "count": 1, "top": "Istanbul → JFK, TK1988" },
      "started_at": "2026-08-20T09:14:21.194Z", "finished_at": "2026-08-20T09:14:21.377Z",
      "duration_ms": 183, "attempts": 1
    },
    {
      "node_id": "draft_cancel", "op": "gmail.draft_email", "seq": 2, "round": 0,
      "status": "succeeded", "depends_on": ["find_booking", "find_event"],
      "args": { "to": ["support@turkishairlines.com"], "subject": "Cancellation request — booking TK1234" },
      "result_summary": { "action_id": "7v1wpk18mA4xFMiZgTznp", "draft_id": "r-8841207733012994001" },
      "started_at": "2026-08-20T09:14:21.404Z", "finished_at": "2026-08-20T09:14:22.061Z",
      "duration_ms": 657, "attempts": 1
    }
  ],

  "entities": [
    { "entity_type": "email", "entity_ref": "18f2c9a4b7e10d33", "label": "Your Turkish Airlines booking TK1234", "meta": { "from": "no-reply@turkishairlines.com", "date": "2026-07-15" } },
    { "entity_type": "event", "entity_ref": "6k9m2p4q8r1s3t5u7v9w", "label": "Istanbul → JFK, TK1988", "meta": { "starts_at": "2026-11-05T07:30:00Z" } }
  ],

  "degraded": [],

  "usage": {
    "llm_calls": 1,
    "model": "gpt-5.6-terra",
    "prompt_tokens": 1842,
    "completion_tokens": 264,
    "usd": 0.00093
  },

  "timings": {
    "front_door_ms": 3,
    "prepass_ms": 11,
    "probe_ms": 112,
    "plan_ms": 468,
    "dispatch_ms": 867,
    "render_ms": 6,
    "total_ms": 1467
  }
}
```

#### Field notes

- **`status`** mirrors `runs.status`: `running` never appears in a buffered
  response, so you will see `complete`, `awaiting_input`, `failed`, `timeout`
  or `cancelled`.
- **`content`** is exactly `messages.content` — an ordered block list. Prose is
  inline; anything with a lifecycle is a `ref`. Resolve `ref` against the
  `actions` and `pending_inputs` arrays in the same response. A ref with no
  matching object was dropped on read (the row is gone) and must not be
  rendered as an empty box.
- **`text`** is the text blocks flattened to markdown, for a client that only
  wants prose. It is derived, not stored.
- **`actions`** is the assignment's `actions_taken`. Read the `status` before
  telling a person anything happened: `draft` means prepared and *not* done.
  `external_ref` on a `gmail.send_email` action is the Gmail draft that already
  exists in the user's account — visible to them, not sent.
- **`payload_fields`** lists which payload keys the confirm card may edit. The
  full payload is deliberately not returned; the `preview` is what `Op.preview()`
  chose to show.
- **`steps`** is `node_executions` for this run, `seq` ordered. `result_summary`
  is a trimmed `result`; the full blob is in the trace stream, not here.
- **`degraded`** is non-empty when a service failed but the run still produced
  an answer. Example: `[{"service": "gdrive", "reason": "circuit_open",
  "detail": "5 consecutive failures", "until": "2026-08-20T09:19:00Z"}]`. This
  is how "Gmail succeeded, Calendar failed" surfaces — as a partial answer with
  a named gap, not a 500.
- **`usage.llm_calls`** is the honest count for this run. Hard cap 5
  (`MAX_LLM_CALLS_PER_RUN`). A rule-routed read is 0, a single-call template
  read is 1, prose synthesis adds a second.
- **`timings.total_ms`** is server-side. P95 under 2s holds for the read class.
  Two-call prose reads land around 3.1s.

### Response — 200, paused on a question

When the run cannot proceed without the person — two plausible Johns, an
unclear week — the ambiguity is a step (`ask.user`) rather than a failure. The
run stops, `status` is `awaiting_input`, and the pending input is `blocking`:

```json
{
  "run_id": "7zfghYBGrldrtA3nh3gYT",
  "conversation_id": "pSbhuNXd5vHdRJROrRXL8",
  "message_id": "ntIFR9d8KYsqp8Xox-Gom",
  "status": "awaiting_input",
  "planner_tier": 2,
  "intent": {
    "name": "move_event", "services": ["gcal"], "has_write": true, "confidence": 0.74,
    "entities": { "person": "John" }, "windows": {}
  },
  "answer_style": "card",
  "text": "Two people called John have meetings with you. Which one?",
  "content": [
    { "type": "text",  "data": { "markdown": "Two people called John have meetings with you. Which one?" } },
    { "type": "input", "ref": "QbYCF5Qz21W4LYLS5D401" }
  ],
  "actions": [],
  "pending_inputs": [
    {
      "id": "QbYCF5Qz21W4LYLS5D401",
      "kind": "choice",
      "blocking": true,
      "prompt": { "question": "Which John?", "help_text": "Both have an upcoming meeting with you." },
      "value_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "string",
        "enum": ["john.mercer@acme.com", "jkowalski@northbridge.io"]
      },
      "options": [
        { "id": "john.mercer@acme.com",     "label": "John Mercer (Acme Corp)",   "meta": { "next_event": "Acme quarterly review — Thu 21 Aug, 14:00", "emails_90d": 41 } },
        { "id": "jkowalski@northbridge.io", "label": "John Kowalski (Northbridge)", "meta": { "next_event": "Northbridge sync — Mon 25 Aug, 09:30", "emails_90d": 6 } }
      ],
      "status": "pending",
      "expires_at": "2026-08-21T09:31:05Z",
      "created_at": "2026-08-20T09:31:05Z"
    }
  ],
  "steps": [
    { "node_id": "find_john", "op": "gmail.resolve_person", "seq": 0, "round": 0, "status": "succeeded",
      "depends_on": [], "args": { "name": "John" },
      "result_summary": { "count": 2, "top_cn": 0.66, "runner_up_cn": 0.61, "margin": 0.05 },
      "duration_ms": 74, "attempts": 1 },
    { "node_id": "ask_which_john", "op": "ask.user", "seq": 1, "round": 0, "status": "succeeded",
      "depends_on": ["find_john"], "args": { "kind": "choice" },
      "result_summary": { "input_id": "QbYCF5Qz21W4LYLS5D401" },
      "duration_ms": 9, "attempts": 1 },
    { "node_id": "move_it", "op": "gcal.update_event", "seq": 2, "round": 0, "status": "pending",
      "depends_on": ["ask_which_john"], "args": {}, "result_summary": null,
      "duration_ms": null, "attempts": 0 }
  ],
  "entities": [],
  "degraded": [],
  "usage": { "llm_calls": 1, "model": "gpt-5.6-terra", "prompt_tokens": 1290, "completion_tokens": 118, "usd": 0.00051 },
  "timings": { "front_door_ms": 2, "prepass_ms": 9, "probe_ms": 104, "plan_ms": 441, "dispatch_ms": 88, "render_ms": 4, "total_ms": 648 }
}
```

The margin here is 0.05, under `MARGIN` (0.15), which is why it asked. Answer
it with `POST /api/v1/prompts/{id}/respond` and the run resumes from the
`ask_which_john` node at **0 LLM calls** — the plan is already on disk, only
the binding was missing.

### Errors

`VALIDATION_ERROR` 422 · `NOT_AUTHENTICATED` 401 ·
`GOOGLE_REAUTH_REQUIRED` 428 · `RATE_LIMITED` 429 ·
`ORCHESTRATION_TIMEOUT` 504 · `GOOGLE_UNAVAILABLE` 503 · `NOT_FOUND` 404
(unknown `conversation_id`) · `INTERNAL` 500.

### curl

```bash
curl -sS -X POST "$BASE/api/v1/query" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-cancel-flight-001' \
  -b cookies.txt \
  -d '{
        "query": "Cancel my Turkish Airlines flight",
        "conversation_id": "KF336Xj1yXQ90iQZ31fN2"
      }' | jq
```

### The ndjson streaming variant

`POST /api/v1/query?stream=ndjson` returns `application/x-ndjson` and holds the
connection open for the life of the run. Each line is one complete JSON object
in **exactly the SSE envelope** described in the next section — same `v`, same
`seq`, same `type`, same `data`. One event union, two transports; a client that
can parse one can parse the other.

```bash
curl -N -sS -X POST "$BASE/api/v1/query?stream=ndjson" \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"query":"What is on my calendar next week?"}'
```

```
{"v":1,"seq":1,"type":"run.started","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.005Z","data":{"conversation_id":"pSbhuNXd5vHdRJROrRXL8","message_id":"QGR716zbKm-wcMY76xhbx","query":"What is on my calendar next week?"}}
{"v":1,"seq":2,"type":"progress","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.008Z","data":{"phase":"prepass","label":"Working out what \"next week\" means"}}
{"v":1,"seq":3,"type":"probe.done","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.121Z","data":{"took_ms":113,"candidates":{"gmail":0,"gcal":9,"gdrive":0}}}
{"v":1,"seq":4,"type":"intent","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.128Z","data":{"name":"list_events","services":["gcal"],"has_write":false,"confidence":0.97,"windows":{"next_week":{"start":"2026-08-24T00:00:00+03:00","end":"2026-08-31T00:00:00+03:00","tz":"Europe/Istanbul","interpretation":"Mon 24 Aug – Sun 30 Aug"}}}}
{"v":1,"seq":5,"type":"plan.step","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.130Z","data":{"node_id":"list_week","op":"gcal.search_events","depends_on":[],"expect":"many","optional":false,"freshness":"cached"}}
{"v":1,"seq":6,"type":"step.started","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.132Z","data":{"node_id":"list_week","op":"gcal.search_events","label":"Reading your calendar for 24–30 Aug"}}
{"v":1,"seq":7,"type":"step.finished","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.271Z","data":{"node_id":"list_week","status":"succeeded","duration_ms":139,"summary":{"count":6}}}
{"v":1,"seq":8,"type":"run.complete","run_id":"0yKfhUaFl8XEChIJtodxN","ts":"2026-08-20T09:40:00.283Z","data":{"status":"complete","message_id":"QGR716zbKm-wcMY76xhbx","answer_style":"template:event_list","usage":{"llm_calls":0,"usd":0},"timings":{"total_ms":283}}}
```

That run cost 0 LLM calls: the rule router recognised it, temporal resolved the
window in Python, and the answer rendered from the `event_list` template.

The last line of an ndjson stream is always `run.complete` or `error`. If the
connection drops mid-run the run keeps going server-side — reattach with
`GET /api/v1/runs/{run_id}/events`.

---

## GET /api/v1/runs/{run_id}/events

Server-Sent Events for one run. Open it immediately after (or in parallel with)
`POST /api/v1/query` when you are not using the ndjson variant. Read-only; it
starts nothing.

```
GET /api/v1/runs/4faGRd0_Gx4BuZxj2frxK/events
Accept: text/event-stream
Last-Event-ID: 7
```

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `from_seq` | int | — | Replay from this seq inclusive. Alternative to `Last-Event-ID` for clients that are not `EventSource`. |
| `types` | comma list | all | Only deliver these event types. `types=progress,step.started,step.finished,run.complete` is a reasonable chat-only filter. |

Response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, `Connection: keep-alive`.

The wire format:

```
retry: 2000

id: 1
event: run.started
data: {"v":1,"seq":1,"type":"run.started","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.005Z","data":{...}}

: ping
```

- `id:` is the `seq`, so a browser `EventSource` sends `Last-Event-ID`
  automatically on reconnect.
- `event:` is the type, so `addEventListener("step.finished", …)` works.
- `data:` is the full envelope — `v`, `seq`, `type`, `run_id`, `ts`, `data` —
  duplicating the type and seq inside the payload so the ndjson transport is
  byte-identical in content.
- `: ping` comment every 15 seconds keeps intermediaries from closing an idle
  stream. It has no `id:` and does not advance `seq`.
- The stream closes after `run.complete` or `error`.

### The seq gap-detection contract

This is the part worth reading twice.

1. `seq` is an integer starting at **1**, incremented by exactly 1 for every
   event published on a run's channel. It is allocated by a Redis `INCR` on the
   channel key, so it is monotonic and gapless *at the publisher*, even with
   several workers writing to the same run.
2. A client holds `last_seq`, initially 0. On each event:
   - `seq == last_seq + 1` → normal. Apply it, advance.
   - `seq <= last_seq` → a duplicate from a reconnect. **Drop it.** Every event
     is safe to drop but not all are safe to apply twice (`content.delta`
     appends).
   - `seq > last_seq + 1` → **you missed events.** Do not guess. Reconnect with
     `Last-Event-ID: <last_seq>`; the server replays from the buffer.
3. The buffer is a capped Redis stream per run: last 500 events, 15-minute TTL.
   Inside that window a replay is exact. Outside it, the server cannot replay
   and sends a single `error` event with `data.code = "STREAM_EXPIRED"` and
   then closes. The client's recovery is `GET /api/v1/conversations/{id}`,
   which returns the settled state from Postgres — the durable record. SSE is
   an accelerator, never the source of truth.
4. `seq` is per **run**. It restarts at 1 for the next run in the same
   conversation. Do not compare seqs across runs.
5. The conversation channel (`action.done`, `action.failed`) has its own
   sequence, described at the end of this section.

### The event union

Fourteen types on the run channel. Every one carries the same envelope; only
`data` differs.

#### `run.started`
First event, always seq 1.
```json
{"v":1,"seq":1,"type":"run.started","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.005Z",
 "data":{"conversation_id":"KF336Xj1yXQ90iQZ31fN2","message_id":"ObkgZf-bc4Mzn_eVs08H3",
         "query":"Cancel my Turkish Airlines flight","timezone":"Europe/Istanbul"}}
```
Render progress the moment this lands — about 5ms after the POST.

#### `progress`
Coarse phase ticks for the status line. Not every phase emits one.
```json
{"v":1,"seq":2,"type":"progress","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.011Z",
 "data":{"phase":"probe","label":"Looking through mail, calendar and files","pct":10}}
```
`phase` ∈ `front_door` `prepass` `probe` `plan` `dispatch` `render`. `pct` is a
hint for a bar, not a promise.

#### `probe.done`
The retrieval pass finished — one embedding plus three parallel hybrid
searches, no LLM call. This is the event that lets the UI paint candidate chips
at about 200ms, before the planner has said anything.
```json
{"v":1,"seq":3,"type":"probe.done","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.118Z",
 "data":{"took_ms":112,"embedding_ms":39,
   "candidates":{"gmail":5,"gcal":2,"gdrive":0},
   "top":[
     {"service":"gmail","label":"Your Turkish Airlines booking TK1234","cn":0.87,"evidence":["ALIAS_TOKEN_IN_SUBJECT"]},
     {"service":"gcal","label":"Istanbul → JFK, TK1988","cn":0.81,"evidence":["EXACT_ID"]}],
   "extracted":{"pnr":["TK1234"],"flight_no":["TK1988"]},
   "ambiguous":false}}
```

#### `intent`
The single planning call returned. Arrives around 590ms.
```json
{"v":1,"seq":4,"type":"intent","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.186Z",
 "data":{"name":"cancel_flight","services":["gmail","gcal"],"has_write":true,"confidence":0.91,
         "answer_style":"card","planner_tier":2,
         "entities":{"airline":"Turkish Airlines","pnr":"TK1234"},
         "windows":{}}}
```

#### `plan.step`
One per step in the DAG, emitted before dispatch so the UI can draw
the whole graph greyed out and fill it in.
```json
{"v":1,"seq":5,"type":"plan.step","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.188Z",
 "data":{"node_id":"draft_cancel","op":"gmail.draft_email","depends_on":["find_booking","find_event"],
         "expect":"one","optional":false,"freshness":"live","is_write":true,"needs_confirm":true}}
```

#### `step.started`
```json
{"v":1,"seq":7,"type":"step.started","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.194Z",
 "data":{"node_id":"find_booking","op":"gmail.search_emails","round":0,
         "label":"Searching mail for the Turkish Airlines booking"}}
```
`label` is `Op.progress_label(args)` — written by the op, safe to show.

#### `step.retrying`
Emitted between attempts, so a slow step visibly explains itself instead of
looking hung.
```json
{"v":1,"seq":9,"type":"step.retrying","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.640Z",
 "data":{"node_id":"find_event","op":"gcal.search_events","attempt":1,"of":2,
         "error_class":"TRANSIENT","google_status":503,"backoff_ms":412,
         "label":"Calendar hiccuped, trying once more"}}
```
`error_class` is from the taxonomy: `TRANSIENT` `RATE_LIMITED`
`QUOTA_EXHAUSTED` `AUTH_EXPIRED` `AUTH_REVOKED` `PRECONDITION` `NOT_FOUND`
`INVALID` `UNKNOWN`. In-request retries are capped at 2 attempts and 1.5s of
added latency; the full policy only runs in the worker.

#### `step.finished`
Terminal for a node, whatever happened.
```json
{"v":1,"seq":10,"type":"step.finished","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.402Z",
 "data":{"node_id":"find_booking","op":"gmail.search_emails","status":"succeeded","round":0,
         "duration_ms":208,"attempts":1,
         "summary":{"count":3,"top":"Your Turkish Airlines booking TK1234"}}}
```
Non-success carries `outcome`:
```json
{"v":1,"seq":11,"type":"step.finished","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:21.930Z",
 "data":{"node_id":"pull_docs","op":"gdrive.search_files","status":"failed","round":0,
         "duration_ms":736,"attempts":2,
         "outcome":{"reason":"circuit_open","class":"TRANSIENT","code":503,
                    "message":"Drive is not responding; skipped this step",
                    "until":"2026-08-20T09:19:00Z"}}}
```
`status` ∈ `succeeded` `failed` `skipped` `timeout` `cancelled`. A `failed`
step on an `optional` node does not fail the run — it lands in `degraded`.

#### `content.delta`
Assistant-visible text, appended to a block of the message being built. This is
what a chat client renders.
```json
{"v":1,"seq":14,"type":"content.delta","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:22.140Z",
 "data":{"message_id":"ObkgZf-bc4Mzn_eVs08H3","block_index":0,"text":"I found your Turkish Airlines booking ("}}
```
Deltas for a given `block_index` arrive in order and concatenate. Applying one
twice corrupts the text, which is the whole reason for the seq rule above.
First prose token lands around 1.6s.

#### `token`
Raw model tokens, for the "watch it think" view. Emitted only
when the run was started by a client that asked for the trace (`types` filter
including `token`). A chat client ignores these entirely — they are the
pre-render stream, not the answer.
```json
{"v":1,"seq":13,"type":"token","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:22.131Z",
 "data":{"call":"synthesize","text":"I found"}}
```

#### `input.raised`
A question appeared. Render the card from `kind`.
```json
{"v":1,"seq":15,"type":"input.raised","run_id":"7zfghYBGrldrtA3nh3gYT","ts":"2026-08-20T09:31:05.402Z",
 "data":{"input_id":"QbYCF5Qz21W4LYLS5D401","kind":"choice","blocking":true,
         "node_id":"ask_which_john","message_id":"ntIFR9d8KYsqp8Xox-Gom",
         "prompt":{"question":"Which John?","help_text":"Both have an upcoming meeting with you."},
         "value_schema":{"type":"string","enum":["john.mercer@acme.com","jkowalski@northbridge.io"]},
         "options":[
           {"id":"john.mercer@acme.com","label":"John Mercer (Acme Corp)","meta":{"next_event":"Acme quarterly review — Thu 21 Aug, 14:00"}},
           {"id":"jkowalski@northbridge.io","label":"John Kowalski (Northbridge)","meta":{"next_event":"Northbridge sync — Mon 25 Aug, 09:30"}}],
         "expires_at":"2026-08-21T09:31:05Z"}}
```
The ambiguity card is on screen by about 650ms.

#### `action.prepared`
A write has been staged. Nothing has been sent.
```json
{"v":1,"seq":16,"type":"action.prepared","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:22.070Z",
 "data":{"action_id":"7v1wpk18mA4xFMiZgTznp","op":"gmail.send_email","status":"draft",
         "requires_input_id":"AwmTv620oCkmGlRtvD4hU","external_ref":"r-8841207733012994001",
         "preview":{"to":["support@turkishairlines.com"],
                    "subject":"Cancellation request — booking TK1234",
                    "body_excerpt":"Hello,\n\nI would like to cancel booking reference TK1234…"},
         "expires_at":"2026-08-21T09:14:22Z"}}
```

#### `run.paused`
The run stopped on a blocking input and released its worker slot. No further
run events until the prompt is answered; answering starts a **new** stream on
the same `run_id` continuing the same `seq`.
```json
{"v":1,"seq":17,"type":"run.paused","run_id":"7zfghYBGrldrtA3nh3gYT","ts":"2026-08-20T09:31:05.410Z",
 "data":{"reason":"awaiting_input","input_id":"QbYCF5Qz21W4LYLS5D401",
         "resumable_until":"2026-08-21T09:31:05Z","completed_nodes":2,"remaining_nodes":1}}
```

#### `run.complete`
Last event of a settled run.
```json
{"v":1,"seq":18,"type":"run.complete","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:22.472Z",
 "data":{"status":"complete","message_id":"ObkgZf-bc4Mzn_eVs08H3","answer_style":"card",
         "planner_tier":2,
         "usage":{"llm_calls":1,"model":"gpt-5.6-terra","prompt_tokens":1842,"completion_tokens":264,"usd":0.00093},
         "timings":{"probe_ms":112,"plan_ms":468,"dispatch_ms":867,"total_ms":1467},
         "degraded":[]}}
```

#### `error`
The run failed, or the stream cannot continue. Terminal either way.
```json
{"v":1,"seq":12,"type":"error","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:14:23.001Z",
 "data":{"code":"GOOGLE_UNAVAILABLE","message":"Google is not responding right now.",
         "retryable":true,"details":{"service":"gmail","retry_after_s":30},
         "partial":true,"message_id":"ObkgZf-bc4Mzn_eVs08H3"}}
```
`data.code` is from the error-code table, plus one stream-only code:
`STREAM_EXPIRED`, meaning the replay buffer no longer holds the events you
missed — fall back to `GET /api/v1/conversations/{id}`.
`partial: true` means an assistant message was still written with whatever did
succeed.

### The conversation channel

Two events fire after the run that created them has already finished, because
approving an action queues work to a Celery worker. They are delivered on the
run channel of the run that prepared the action, and on any stream currently
open for that conversation.

#### `action.done`
```json
{"v":1,"seq":19,"type":"action.done","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:15:07.882Z",
 "data":{"action_id":"7v1wpk18mA4xFMiZgTznp","op":"gmail.send_email","status":"done",
         "result":{"message_id":"18f31c07a9d4e221","thread_id":"18f2c9a4b7e10d33",
                   "sent_at":"2026-08-20T09:15:07Z"},
         "attempts":1,"audit_id":91824}}
```

#### `action.failed`
```json
{"v":1,"seq":19,"type":"action.failed","run_id":"4faGRd0_Gx4BuZxj2frxK","ts":"2026-08-20T09:15:11.204Z",
 "data":{"action_id":"7v1wpk18mA4xFMiZgTznp","op":"gmail.send_email","status":"failed",
         "error":{"class":"AUTH_REVOKED","code":401,
                  "message":"Your Google connection was revoked. Reconnect and I will resend."},
         "attempts":3,"retryable":false,"audit_id":91825}}
```

### Errors

`NOT_AUTHENTICATED` 401 · `NOT_FOUND` 404 (no such run for this user) ·
`INTERNAL` 500. Errors *before* the stream opens use the normal JSON envelope
with the right status. Once `200 text/event-stream` is committed, failures
arrive as an `error` **event**, because you cannot change a status code after
the headers are gone.

### curl

```bash
curl -N -sS "$BASE/api/v1/runs/4faGRd0_Gx4BuZxj2frxK/events" \
  -H 'Accept: text/event-stream' \
  -b cookies.txt

# reattach after a drop, from the last seq you applied
curl -N -sS "$BASE/api/v1/runs/4faGRd0_Gx4BuZxj2frxK/events?from_seq=8" -b cookies.txt

# chat-only filter, no trace noise
curl -N -sS "$BASE/api/v1/runs/4faGRd0_Gx4BuZxj2frxK/events?types=progress,content.delta,input.raised,action.prepared,run.complete" -b cookies.txt
```

---

## The two-phase write flow

The least obvious part of this API, and the part that matters most: **no
endpoint sends an email, moves an event, or shares a file.** A query prepares;
a person approves; a worker executes. Six stages.

The database enforces it rather than trusting the code:
`actions.requires_input_id` is `NOT NULL` and references `pending_inputs(id)`,
so a confirm-requiring write physically cannot exist without a prompt gating
it.

```
POST /query ─┬─> 1 PREPARE   actions(draft) + pending_inputs(pending), one txn
             └─> 2 PROMPT    action.prepared + input.raised on the stream
                                     │
                          person clicks Send it
                                     ▼
POST /prompts/{id}/respond ─┬─> 3 APPROVE      prompt -> answered
                            ├─> 4 REVALIDATE   six checks, pure Python, 0 LLM calls
                            └─> 5 EXECUTE      action -> approved -> queued -> running
                                     │
                              Celery `actions` queue
                                     ▼
                              6 REPORT   action.done / action.failed + audit_log
```

### 1. Prepare

Happens inside the run. A `ConfirmableOp` does the reversible half of its job —
`gmail.send_email` actually creates a **Gmail draft**, so the user can see it in
their own account — then writes, in one transaction:

- an `actions` row, `status = draft`, `payload` = exactly what will execute,
  `dedupe_key` = uuid5 over user, op, canonical payload and conversation,
  `external_ref` = the draft id;
- a `pending_inputs` row, `kind = confirm`, `blocking = false` (the run is
  finished; this is waiting on a yes), `value_schema` = the JSON Schema the
  answer must satisfy;
- the assistant `messages` row whose `content` references both by id.

`blocking = false` is the important flag. The run completes and returns. The
person can walk away and approve tomorrow; nothing is holding a worker.

A write may only target a candidate whose `cn` is above `FLOOR_WRITE` (0.80).
Below that the op raises an `ask.user` step instead of preparing anything.

The `POST /query` response for this run is the first example in this document.

### 2. Prompt

Two ways to get the card:

```bash
curl -sS "$BASE/api/v1/prompts?status=pending" -b cookies.txt | jq
```

or the `input.raised` and `action.prepared` events on the stream. Both carry
the same `value_schema`. Render three buttons: **Send it**, **Not now**,
**Edit**.

### 3. Approve

```bash
curl -sS -X POST "$BASE/api/v1/prompts/AwmTv620oCkmGlRtvD4hU/respond" \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"value": {"approve": true}}'
```

Zero LLM calls. The value is validated against the stored `value_schema`, the
prompt moves to `answered`, and the gated action moves to `approved`.

**Not now** is `{"value": {"approve": false}}` — prompt `answered`, action
`cancelled`, Gmail draft deleted. **Edit** is
`{"value": {"approve": false, "patch": {...}}}`; see
[Editing before approving](#editing-before-approving).

### 4. Revalidate

Before the action is queued, six checks run in pure Python. Any failure aborts
and nothing is queued.

| Check | Failure |
|---|---|
| Prompt is still `pending` and `expires_at` is in the future | `409 PROMPT_NOT_PENDING`, `details.status` ∈ `answered` `cancelled` `expired` `superseded` |
| The posted value validates against `value_schema` | `422 PROMPT_VALUE_INVALID`, `details.errors` |
| The gated action is still `draft` | `409 PROMPT_NOT_PENDING`, `details.action_status` |
| `dedupe_key` is still unique among `draft`/`approved`/`running` | `409 PROMPT_NOT_PENDING`, `details.duplicate_action_id` — an identical write is already in flight |
| The user's Google grant is live and scoped for the op | `428 GOOGLE_REAUTH_REQUIRED` |
| The target still exists (event `etag` unchanged, draft not deleted by hand) | `409 PROMPT_NOT_PENDING`, `details.reason = "target_changed"`; the client re-asks |

The `etag` check is why `sync_events.attributes->>'etag'` exists: an update or delete is sent
with `If-Match`, so a calendar event someone else moved in the meantime fails
loudly instead of silently clobbering.

### 5. Execute

`actions.status` goes `approved` → `running`, `job_id` is set, and the payload
is handed to the Celery `actions` queue. The HTTP response returns immediately
— **202** — because a Gmail send can take a second and the person should not
watch a spinner for it.

```json
{
  "input_id": "AwmTv620oCkmGlRtvD4hU",
  "status": "answered",
  "answered_at": "2026-08-20T09:15:02.771Z",
  "value": { "approve": true },
  "resumed_run_id": null,
  "action": {
    "id": "7v1wpk18mA4xFMiZgTznp",
    "op": "gmail.send_email",
    "status": "running",
    "job_id": "c7e1d0f2-3a4b-4c5d-8e9f-0a1b2c3d4e5f",
    "queued_at": "2026-08-20T09:15:02.774Z",
    "watch": "/api/v1/runs/4faGRd0_Gx4BuZxj2frxK/events"
  },
  "llm_calls": 0
}
```

The worker runs the irreversible half — `ConfirmableOp.execute()` — under the
full retry policy: full-jitter backoff by error class, the per-(user, service)
circuit breaker (5 consecutive failures opens for 5 minutes, doubling to a
30-minute cap, half-open with a single probe), and the quota governor. The
outbound message carries `X-Orchestrator-Idem: <dedupe_key>`, so a retry after
an ambiguous timeout searches Sent for that header before sending again.

### 6. Report

On success: `actions.status = done`, `result` and `executed_at` written, an
`audit_log` row appended (recipients and subject in `payload_visible`, the body
only as a `payload_hash` uuid5 — you can prove what was sent without storing
it), and `action.done` published.

On terminal failure: `actions.status = failed`, `error` written,
`audit_log` row with the failure, `action.failed` published, and if the class
was `TRANSIENT` the job lands in `job_failed_tasks` for the half-hourly
sweeper. Because the unique index on `dedupe_key` is partial — covering only
`draft`, `approved` and `running` — a `failed` or `cancelled` action does not
block an identical retry. Resending after cancelling is a legitimate new
request, and the schema says so.

Poll or stream:

```bash
curl -sS "$BASE/api/v1/conversations/KF336Xj1yXQ90iQZ31fN2" -b cookies.txt \
  | jq '.actions[] | {id, status, result}'
```

### Editing before approving

The **Edit** button sends a patch. The server applies it to
`actions.payload`, appends the previous payload to `actions.revisions`,
recomputes `dedupe_key`, cancels the old prompt as `answered`, and raises a
**new** confirm prompt against the revised payload. Still 0 LLM calls — the
edit came from the person, so there is nothing to plan.

```bash
curl -sS -X POST "$BASE/api/v1/prompts/AwmTv620oCkmGlRtvD4hU/respond" \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
        "value": {
          "approve": false,
          "patch": {
            "subject": "Cancellation request — booking TK1234 (urgent)",
            "body": "Hello,\n\nPlease cancel booking TK1234 (Istanbul → JFK, TK1988, 5 November). I am travelling and need written confirmation today.\n\nThank you."
          }
        }
      }'
```

```json
{
  "input_id": "AwmTv620oCkmGlRtvD4hU",
  "status": "answered",
  "answered_at": "2026-08-20T09:16:40.118Z",
  "value": { "approve": false, "patch": { "subject": "…", "body": "…" } },
  "action": {
    "id": "7v1wpk18mA4xFMiZgTznp",
    "op": "gmail.send_email",
    "status": "draft",
    "revision": 1,
    "external_ref": "r-8841207733012994001"
  },
  "next_input": {
    "id": "1uzmfJcrbEbG_VPXhvC72",
    "kind": "confirm",
    "blocking": false,
    "prompt": {
      "question": "Send the revised cancellation email to support@turkishairlines.com?",
      "help_text": "Edited just now. Still a draft."
    },
    "value_schema": { "type": "object", "required": ["approve"], "properties": { "approve": { "type": "boolean" }, "patch": { "type": "object" }, "note": { "type": "string" } } },
    "status": "pending",
    "expires_at": "2026-08-21T09:16:40Z"
  },
  "llm_calls": 0
}
```

The Gmail draft is updated in place, so `external_ref` does not change and the
person sees the edit in their own Drafts folder.

### What can go wrong, and what it looks like

| Situation | Response |
|---|---|
| The person taps Send twice | Second call `409 PROMPT_NOT_PENDING`, `details.status = "answered"`. Nothing sent twice. |
| Two tabs, both approve | First wins. Second gets `409`. The partial unique index on `dedupe_key` is the backstop even if both pass the prompt check. |
| The card sat for two days | `409 PROMPT_NOT_PENDING`, `details.status = "expired"`. `PROMPT_TTL_MIN` is 24 hours; `maintenance.expire_prompts` runs hourly. The action expires with it. |
| A newer run prepared the same kind of write | Old prompt is `superseded` in the same transaction that created the new one. `409`, `details.superseded_by`. |
| Google was down at execute time | Action stays `running` through the retry policy, then `failed`. `action.failed` on the stream, `job_failed_tasks` row, sweeper retries transient classes. |
| The user revoked access between prepare and approve | `428 GOOGLE_REAUTH_REQUIRED` at revalidate. The action stays `draft` and is approvable again after reconnecting. |

---

## The `value_schema` contract

Every `pending_inputs` row carries a `value_schema` — a JSON Schema
(draft 2020-12) that **is** the validation authority. When a client posts to
`/api/v1/prompts/{id}/respond`, the server does exactly this:

```python
jsonschema.validate(instance=body["value"], schema=row.value_schema,
                    cls=jsonschema.Draft202012Validator,
                    format_checker=Draft202012Validator.FORMAT_CHECKER)
```

Nothing else. There is no per-kind validation branch in the API layer, no
`if kind == "choice"` anywhere in `api/v1/prompts.py`.

**The consequence worth stating plainly:** adding a new prompt kind needs no
API change, no new endpoint, no client release. An op writes a different
`value_schema` into the row and the same `respond` endpoint validates it
correctly. `kind` exists only so the frontend knows which of six card
renderers to use; the *contract* lives in the schema.

`format` assertion is on, so `"format": "email"` and `"format": "date-time"`
are enforced rather than advisory. Unknown keywords are ignored by the
validator, as the spec says, so an op can annotate a schema with UI hints
(`"x-widget": "textarea"`) without breaking validation.

On failure the server returns `422 PROMPT_VALUE_INVALID` with each error's JSON
path:

```json
{
  "error": {
    "code": "PROMPT_VALUE_INVALID",
    "message": "That answer does not fit what was asked.",
    "details": {
      "errors": [
        { "path": "$.approve", "msg": "'yes' is not of type 'boolean'" },
        { "path": "$.patch.to[0]", "msg": "'support@' is not a 'email'" }
      ],
      "schema_ref": "AwmTv620oCkmGlRtvD4hU"
    },
    "request_id": "NKgjLgOI3xAxLtdLLB0pX"
  }
}
```

### The six kinds

`kind` maps to a card renderer. These are the schemas the shipped ops write —
they are examples of the pattern, not a closed list the server enforces.

| `kind` | Typical `value_schema` | Value that validates |
|---|---|---|
| `confirm` | `{"type":"object","required":["approve"],"properties":{"approve":{"type":"boolean"},"patch":{"type":"object"},"note":{"type":"string","maxLength":2000}}}` | `{"approve": true}` |
| `choice` | `{"type":"string","enum":["john.mercer@acme.com","jkowalski@northbridge.io"]}` | `"john.mercer@acme.com"` |
| `multi_choice` | `{"type":"array","items":{"type":"string","enum":["1a2b","3c4d","5e6f"]},"minItems":1,"maxItems":10,"uniqueItems":true}` | `["1a2b","5e6f"]` |
| `text` | `{"type":"string","minLength":1,"maxLength":500}` | `"Push it to Thursday please"` |
| `form` | `{"type":"object","required":["title","starts_at"],"additionalProperties":false,"properties":{"title":{"type":"string","maxLength":200},"starts_at":{"type":"string","format":"date-time"},"duration_min":{"type":"integer","minimum":15,"maximum":480,"default":30},"attendees":{"type":"array","items":{"type":"string","format":"email"}}}}` | `{"title":"Acme review","starts_at":"2026-08-27T13:00:00Z","duration_min":45}` |
| `date_range` | `{"type":"object","required":["start","end"],"additionalProperties":false,"properties":{"start":{"type":"string","format":"date-time"},"end":{"type":"string","format":"date-time"}}}` | `{"start":"2026-08-24T00:00:00+03:00","end":"2026-08-31T00:00:00+03:00"}` |

`options` is display only. It is **not** the validation source — a `choice`
prompt's enum is in the schema, and a client that posts an option id absent
from the enum gets `PROMPT_VALUE_INVALID`, which is exactly right.

---

## Prompts

### `GET /api/v1/prompts`

Everything the system is waiting on, or has waited on.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `status` | `pending` \| `answered` \| `cancelled` \| `expired` \| `superseded` \| `all` | `pending` | Filter on `pending_inputs.status`. |
| `conversation_id` | string (21) | — | Only prompts from this thread. |
| `run_id` | string (21) | — | Only prompts from this run. |
| `limit` | int 1–100 | 50 | |

**200**

```json
{
  "items": [
    {
      "id": "AwmTv620oCkmGlRtvD4hU",
      "run_id": "4faGRd0_Gx4BuZxj2frxK",
      "conversation_id": "KF336Xj1yXQ90iQZ31fN2",
      "message_id": "ObkgZf-bc4Mzn_eVs08H3",
      "node_execution_id": "c-rMTDBMV-WCxxq8Grclz",
      "kind": "confirm",
      "blocking": false,
      "prompt": { "question": "Send the cancellation email to support@turkishairlines.com?", "help_text": "Nothing has been sent. This is a Gmail draft until you say so." },
      "value_schema": { "type": "object", "required": ["approve"], "properties": { "approve": { "type": "boolean" }, "patch": { "type": "object" }, "note": { "type": "string" } } },
      "options": null,
      "status": "pending",
      "response": null,
      "expires_at": "2026-08-21T09:14:22Z",
      "answered_at": null,
      "created_at": "2026-08-20T09:14:22Z",
      "action": {
        "id": "7v1wpk18mA4xFMiZgTznp",
        "op": "gmail.send_email",
        "status": "draft",
        "preview": { "to": ["support@turkishairlines.com"], "subject": "Cancellation request — booking TK1234", "body_excerpt": "Hello,\n\nI would like to cancel booking reference TK1234…" }
      }
    }
  ],
  "count": 1
}
```

`action` is present only when a row in `actions` has this prompt as its
`requires_input_id`. A `choice` prompt that only disambiguates has `action:
null`.

Errors: `NOT_AUTHENTICATED` 401 · `VALIDATION_ERROR` 422 (bad `status` value).

```bash
curl -sS "$BASE/api/v1/prompts?status=pending" -b cookies.txt | jq
```

### `POST /api/v1/prompts/{id}/respond`

Answer a prompt. This single endpoint covers all six kinds, both blocking
(resume the run) and non-blocking (approve the write), because the validation
authority is the row's `value_schema`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `value` | any | yes | Must validate against the prompt's `value_schema`. `null` is only legal if the schema permits it. |

**Non-blocking (a write approval)** → `202 Accepted`, body as shown in
[stage 5](#5-execute).

**Blocking (an ambiguity)** → `200 OK`, and the run resumes at 0 LLM calls:

```bash
curl -sS -X POST "$BASE/api/v1/prompts/QbYCF5Qz21W4LYLS5D401/respond" \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"value": "john.mercer@acme.com"}'
```

```json
{
  "input_id": "QbYCF5Qz21W4LYLS5D401",
  "status": "answered",
  "answered_at": "2026-08-20T09:32:11.043Z",
  "value": "john.mercer@acme.com",
  "resumed_run_id": "7zfghYBGrldrtA3nh3gYT",
  "action": null,
  "llm_calls": 0,
  "watch": "/api/v1/runs/7zfghYBGrldrtA3nh3gYT/events"
}
```

Reattach to the stream at your last seq to see the rest of the plan execute.
The `seq` continues from where `run.paused` left it.

Errors: `PROMPT_NOT_PENDING` 409 · `PROMPT_VALUE_INVALID` 422 ·
`GOOGLE_REAUTH_REQUIRED` 428 · `NOT_FOUND` 404 · `NOT_AUTHENTICATED` 401 ·
`GOOGLE_UNAVAILABLE` 503.

### `POST /api/v1/prompts/{id}/cancel`

Dismiss a prompt without answering it — the "Not now" that does not even record
a decision, and the way a client abandons a stuck blocking prompt.

No body.

**200**

```json
{
  "input_id": "AwmTv620oCkmGlRtvD4hU",
  "status": "cancelled",
  "cancelled_at": "2026-08-20T09:20:00.512Z",
  "action": { "id": "7v1wpk18mA4xFMiZgTznp", "op": "gmail.send_email", "status": "cancelled" },
  "run_status": "complete"
}
```

Cancelling a prompt that gates an action cancels the action too, and deletes
the Gmail draft it created. Cancelling a **blocking** prompt cancels the run
(`runs.status = cancelled`); the steps that already ran keep their rows, so the
trace is intact.

Rows are never deleted — the status changes. A reopened chat shows the card in
its cancelled state rather than a frozen snapshot.

Errors: `PROMPT_NOT_PENDING` 409 · `NOT_FOUND` 404 · `NOT_AUTHENTICATED` 401.

```bash
curl -sS -X POST "$BASE/api/v1/prompts/AwmTv620oCkmGlRtvD4hU/cancel" -b cookies.txt | jq
```

---

## Conversations

### `GET /api/v1/conversations`

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `limit` | int 1–100 | 20 | |
| `cursor` | opaque string | — | From `next_cursor`. Keyset pagination on `(last_message_at, id)` — the partial index `(user_id, last_message_at DESC) WHERE archived_at IS NULL` serves it. |
| `archived` | bool | `false` | `true` returns archived threads instead. |

**200**

```json
{
  "items": [
    {
      "id": "KF336Xj1yXQ90iQZ31fN2",
      "title": "Cancel my Turkish Airlines flight",
      "title_is_derived": true,
      "last_message_at": "2026-08-20T09:15:07Z",
      "created_at": "2026-08-20T09:14:20Z",
      "archived_at": null,
      "message_count": 2,
      "pending_input_count": 1,
      "last_run": { "id": "4faGRd0_Gx4BuZxj2frxK", "status": "complete", "intent": "cancel_flight" }
    },
    {
      "id": "pSbhuNXd5vHdRJROrRXL8",
      "title": "Q3 planning",
      "title_is_derived": false,
      "last_message_at": "2026-08-19T16:02:41Z",
      "created_at": "2026-08-19T15:48:10Z",
      "archived_at": null,
      "message_count": 8,
      "pending_input_count": 0,
      "last_run": { "id": "7zfghYBGrldrtA3nh3gYT", "status": "complete", "intent": "list_events" }
    }
  ],
  "next_cursor": "eyJsIjoiMjAyNi0wOC0xOVQxNjowMjo0MVoiLCJpIjoicFNiaHVOWGQ1dkhkUkpST3JSWEw4In0",
  "has_more": true
}
```

`title_is_derived: true` means `conversations.title IS NULL` and the title
shown is the first user message. Nothing is stored for it.

Errors: `NOT_AUTHENTICATED` 401 · `VALIDATION_ERROR` 422.

```bash
curl -sS "$BASE/api/v1/conversations?limit=20" -b cookies.txt | jq
```

### `GET /api/v1/conversations/{id}`

The full thread, with every referenced object resolved. This is the durable
record and the SSE fallback: whatever a dropped stream cost you, this has.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `include_trace` | bool | `false` | Include `node_executions` per message. Off by default — a long thread's trace is large. |
| `limit` | int 1–200 | 50 | Most recent N messages. |
| `before_seq` | int | — | Page backwards through `messages.seq`. |

**200**

```json
{
  "id": "KF336Xj1yXQ90iQZ31fN2",
  "title": "Cancel my Turkish Airlines flight",
  "title_is_derived": true,
  "created_at": "2026-08-20T09:14:20Z",
  "last_message_at": "2026-08-20T09:15:07Z",
  "archived_at": null,

  "messages": [
    {
      "id": "NKgjLgOI3xAxLtdLLB0pX",
      "seq": 1,
      "role": "user",
      "content": [{ "type": "text", "data": { "markdown": "Cancel my Turkish Airlines flight" } }],
      "run_id": null,
      "created_at": "2026-08-20T09:14:20Z"
    },
    {
      "id": "ObkgZf-bc4Mzn_eVs08H3",
      "seq": 2,
      "role": "assistant",
      "content": [
        { "type": "text",   "data": { "markdown": "I found your Turkish Airlines booking (TK1234) in an email from Ozan Demir on 15 Jul." } },
        { "type": "action", "ref": "7v1wpk18mA4xFMiZgTznp" },
        { "type": "input",  "ref": "AwmTv620oCkmGlRtvD4hU" }
      ],
      "run_id": "4faGRd0_Gx4BuZxj2frxK",
      "created_at": "2026-08-20T09:14:22Z"
    }
  ],

  "runs": [
    {
      "id": "4faGRd0_Gx4BuZxj2frxK",
      "trigger_message_id": "NKgjLgOI3xAxLtdLLB0pX",
      "status": "complete",
      "planner_tier": 2,
      "intent": { "name": "cancel_flight", "services": ["gmail", "gcal"], "has_write": true, "confidence": 0.91 },
      "token_usage": { "prompt": 1842, "completion": 264, "model": "gpt-5.6-terra", "usd": 0.00093 },
      "error": null,
      "started_at": "2026-08-20T09:14:21.000Z",
      "finished_at": "2026-08-20T09:14:22.472Z",
      "duration_ms": 1472
    }
  ],

  "pending_inputs": [
    { "id": "AwmTv620oCkmGlRtvD4hU", "kind": "confirm", "blocking": false, "status": "answered",
      "prompt": { "question": "Send the cancellation email to support@turkishairlines.com?" },
      "value_schema": { "type": "object", "required": ["approve"], "properties": { "approve": { "type": "boolean" } } },
      "options": null, "response": { "approve": true },
      "answered_at": "2026-08-20T09:15:02Z", "expires_at": "2026-08-21T09:14:22Z" }
  ],

  "actions": [
    { "id": "7v1wpk18mA4xFMiZgTznp", "op": "gmail.send_email", "status": "done",
      "requires_input_id": "AwmTv620oCkmGlRtvD4hU",
      "preview": { "to": ["support@turkishairlines.com"], "subject": "Cancellation request — booking TK1234" },
      "result": { "message_id": "18f31c07a9d4e221", "thread_id": "18f2c9a4b7e10d33" },
      "error": null, "attempts": 1, "revision": 0,
      "executed_at": "2026-08-20T09:15:07Z" }
  ],

  "entities": [
    { "id": "jDFOaP9o6qFamCf5LJ8vr", "entity_type": "email", "entity_ref": "18f2c9a4b7e10d33",
      "label": "Your Turkish Airlines booking TK1234",
      "meta": { "from": "no-reply@turkishairlines.com", "date": "2026-07-15" },
      "last_seen_at": "2026-08-20T09:14:21Z" },
    { "id": "L_hrhwZtXSKNanzIghCNi", "entity_type": "event", "entity_ref": "6k9m2p4q8r1s3t5u7v9w",
      "label": "Istanbul → JFK, TK1988",
      "meta": { "starts_at": "2026-11-05T07:30:00Z", "calendar_id": "primary" },
      "last_seen_at": "2026-08-20T09:14:21Z" }
  ],

  "has_more": false
}
```

`entities` is `conversation_entities` — what this thread has referred to. It is
what makes "that email about the proposal" resolve against ~20 rows instead of
re-parsing five runs' worth of result blobs, and it is the reason a follow-up
costs no extra retrieval.

With `include_trace=true`, each assistant message gains a `trace` array of its
`node_executions` (the ones whose `message_id` is that message — which is why
a paused run's trace does not appear twice).

Errors: `NOT_FOUND` 404 · `NOT_AUTHENTICATED` 401 · `VALIDATION_ERROR` 422.

```bash
curl -sS "$BASE/api/v1/conversations/KF336Xj1yXQ90iQZ31fN2?include_trace=true" -b cookies.txt | jq
```

---

## Auth

### `GET /api/v1/auth/google`

Starts the OAuth flow. Not an API call — a browser redirect.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `next` | relative path | `/` | Where to land after the callback. Must be same-origin and relative; an absolute URL is rejected with `VALIDATION_ERROR` (open-redirect guard). |

**302** to `accounts.google.com/o/oauth2/v2/auth` with `access_type=offline`,
`prompt=consent`, `include_granted_scopes=true`, a signed `state` carrying a
CSRF nonce and `next`, and these scopes:

```
openid
https://www.googleapis.com/auth/userinfo.email
https://www.googleapis.com/auth/userinfo.profile
https://www.googleapis.com/auth/gmail.modify
https://www.googleapis.com/auth/calendar
https://www.googleapis.com/auth/drive
```

`gmail.modify` rather than `gmail.send` because the system creates and updates
drafts before anything is sent.

```bash
# open in a browser; curl only shows you the redirect
curl -sSi "$BASE/api/v1/auth/google" | head -n 3
```

### `GET /api/v1/auth/google/callback`

Google's redirect target. Not called by a client directly.

| Query parameter | Meaning |
|---|---|
| `code` | Authorization code. |
| `state` | The signed nonce from the start of the flow. |
| `error` | Present when the user declined. |

On success: the code is exchanged, tokens are encrypted (AES-256-GCM, nonce ‖
ciphertext ‖ tag) into `oauth_tokens`, the `users` row is created or updated,
an initial backfill is queued on the `sync` queue, an `audit_log` row records
`auth.grant`, the `alpha_session` cookie is set (`HttpOnly`, `SameSite=Lax`,
`Secure` outside development), and the response is **302** to `next`.

On failure it redirects to `next` with `?auth_error=<reason>` rather than
rendering an error page, so the SPA owns the message.

Errors: `VALIDATION_ERROR` 422 (bad or replayed `state`) ·
`GOOGLE_UNAVAILABLE` 503 (token endpoint down).

### `GET /api/v1/auth/me`

Who is signed in, and whether the Google grant is healthy. The first call a
client makes.

**200**

```json
{
  "user": {
    "id": "jDFOaP9o6qFamCf5LJ8vr",
    "email": "ayse@northbridge.io",
    "display_name": "Ayşe Yılmaz",
    "timezone": "Europe/Istanbul",
    "work_week_start": 1,
    "created_at": "2026-06-02T11:20:00Z"
  },
  "google": {
    "connected": true,
    "provider_account_id": "104928374651029384756",
    "scopes": [
      "openid",
      "https://www.googleapis.com/auth/userinfo.email",
      "https://www.googleapis.com/auth/userinfo.profile",
      "https://www.googleapis.com/auth/gmail.modify",
      "https://www.googleapis.com/auth/calendar",
      "https://www.googleapis.com/auth/drive"
    ],
    "expires_at": "2026-08-20T10:02:00Z",
    "needs_reauth": false
  },
  "limits": { "queries_per_hour": 100, "remaining_this_hour": 96 }
}
```

No token material appears here or anywhere else. `needs_reauth: true` means the
next query would return `428` — show the banner before the user hits it.

Errors: `NOT_AUTHENTICATED` 401.

```bash
curl -sS "$BASE/api/v1/auth/me" -b cookies.txt | jq
```

### `DELETE /api/v1/auth/google`

Disconnect. The only destructive endpoint in the API, and it is deliberate.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `purge` | bool | `true` | `true` deletes the mirror (`sync_messages`, `sync_events`, `sync_files`, `sync_state`) as well as revoking. `false` revokes and keeps the mirror, so reconnecting does not re-backfill 180 days. |

What happens: the refresh token is revoked at
`oauth2.googleapis.com/revoke`; `oauth_tokens.revoked_at` is set and the
ciphertext columns are overwritten; every `draft` and `approved` action is
cancelled and its Gmail drafts deleted; pending prompts move to `cancelled`;
with `purge=true` the `sync_` rows are deleted; an `audit_log` row records
`auth.revoke`; the session cookie is cleared.

Conversations, messages, runs and audit rows survive. This is a disconnect, not
an account deletion — "no hard delete" holds everywhere except the mirror,
which is a cache by definition.

**200**

```json
{
  "disconnected": true,
  "revoked_at": "2026-08-20T11:00:03Z",
  "purged": { "gmail": 18422, "gcal": 1204, "gdrive": 3311 },
  "cancelled_actions": 1,
  "cancelled_prompts": 2,
  "conversations_kept": 14
}
```

Errors: `NOT_AUTHENTICATED` 401 · `GOOGLE_UNAVAILABLE` 503 (revoke endpoint
unreachable — local state is still cleared and the response says
`"revoked_remotely": false`).

```bash
curl -sS -X DELETE "$BASE/api/v1/auth/google?purge=false" -b cookies.txt | jq
```

---

## Sync

Background sync runs every 15 minutes via Celery beat, smeared with
`countdown = hash(user_id) % 900` so a million users do not all hit Google on
the same second. These endpoints are for forcing it and for seeing where it
got to.

### `POST /api/v1/sync/trigger`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `services` | array of `gmail` \| `gcal` \| `gdrive` | no | Default: all three. |
| `mode` | `incremental` \| `backfill` \| `full` | no, default `incremental` | `incremental` advances from the stored cursor (Gmail `historyId`, Calendar `syncToken`, Drive `pageToken`). `backfill` continues the historical walk from `backfill_cursor` over `SYNC_BACKFILL_DAYS` (180). `full` discards the cursor and re-walks from scratch — expensive, and rate-limited to once an hour per service. |

```json
{ "services": ["gmail", "gcal"], "mode": "incremental" }
```

**202**

```json
{
  "queued": [
    { "service": "gmail", "mode": "incremental", "task_id": "b41f0e2a-7c33-4b0e-9a1f-2d5c6e7f8091", "queue": "sync" },
    { "service": "gcal",  "mode": "incremental", "task_id": "c52a1f3b-8d44-4c1f-8b20-3e6d7f809102", "queue": "sync" }
  ],
  "skipped": [],
  "poll": "/api/v1/sync/status"
}
```

`skipped` is non-empty when a service is already syncing or its circuit breaker
is open:

```json
{
  "queued": [],
  "skipped": [
    { "service": "gdrive", "reason": "circuit_open", "until": "2026-08-20T09:19:00Z" },
    { "service": "gmail",  "reason": "already_running", "task_id": "b41f0e2a-7c33-4b0e-9a1f-2d5c6e7f8091" }
  ],
  "poll": "/api/v1/sync/status"
}
```

Nothing is a 409. A queued sync that turns out to be redundant is a wasted
worker second, and a request that is refused mid-outage is a bad experience;
reporting both plainly beats an error code.

Errors: `VALIDATION_ERROR` 422 · `NOT_AUTHENTICATED` 401 ·
`GOOGLE_REAUTH_REQUIRED` 428 · `RATE_LIMITED` 429 (`mode=full` more than once
an hour).

```bash
curl -sS -X POST "$BASE/api/v1/sync/trigger" \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"services":["gmail","gcal","gdrive"],"mode":"incremental"}' | jq
```

### `GET /api/v1/sync/status`

Reads `sync_state` plus the open count from `job_failed_tasks`.

**200**

```json
{
  "services": {
    "gmail": {
      "last_synced_at": "2026-08-20T09:07:41Z",
      "last_success_at": "2026-08-20T09:07:41Z",
      "lag_seconds": 421,
      "items_indexed": 18422,
      "backfill_complete": true,
      "cursor_present": true,
      "consecutive_failures": 0,
      "circuit_open_until": null,
      "last_error": null,
      "healthy": true
    },
    "gcal": {
      "last_synced_at": "2026-08-20T09:07:44Z",
      "last_success_at": "2026-08-20T09:07:44Z",
      "lag_seconds": 418,
      "items_indexed": 1204,
      "backfill_complete": true,
      "cursor_present": true,
      "consecutive_failures": 0,
      "circuit_open_until": null,
      "last_error": null,
      "healthy": true
    },
    "gdrive": {
      "last_synced_at": "2026-08-20T09:07:52Z",
      "last_success_at": "2026-08-20T08:52:50Z",
      "lag_seconds": 1312,
      "items_indexed": 3311,
      "backfill_complete": false,
      "cursor_present": true,
      "consecutive_failures": 2,
      "circuit_open_until": null,
      "last_error": {
        "class": "RATE_LIMITED",
        "code": 429,
        "message": "User rate limit exceeded",
        "at": "2026-08-20T09:07:52Z"
      },
      "healthy": false
    }
  },
  "freshness": { "worst_lag_seconds": 1312, "target_seconds": 900, "within_target": false },
  "dlq": { "open": 1, "oldest_at": "2026-08-20T08:41:12Z" },
  "next_scheduled_at": "2026-08-20T09:22:41Z"
}
```

`lag_seconds` is `now - last_success_at`, which is why `last_success_at` exists
separately from `last_synced_at`: a failing sync updates the attempt but not
the success, and the lag figure must reflect the data, not the effort.
`within_target: false` is the honest report that the mirror is staler than the
15-minute goal — the answer to a superlative query in that window should use
`freshness: "live"`.

Errors: `NOT_AUTHENTICATED` 401.

```bash
curl -sS "$BASE/api/v1/sync/status" -b cookies.txt | jq '.freshness'
```

---

## GET /api/v1/search

The retrieval layer with the lid off. Not used by the chat path — this is for
debugging a bad answer and for the offline Precision@5 evaluation. It returns
every score component so you can see *why* something ranked where it did.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `q` | string | required | The search text. Embedded once and reused across all three services. |
| `services` | comma list of `gmail` `gcal` `gdrive` | all | |
| `limit` | int 1–50 | 10 | Per service. |
| `freshness` | `cached` \| `live` | `cached` | `live` reads through to Google first, then searches. Slower and honest about it. |
| `since` / `until` | RFC 3339 | — | Metadata prefilter on `received_at` / `starts_at` / `modified_at`. |
| `from` | email | — | Gmail only. Exact prefilter on `from_email`. |
| `attendee` | email | — | Calendar only. GIN index hit on `attendee_emails`. |
| `mime` | string | — | Drive only, e.g. `application/pdf`. |
| `explain` | bool | `false` | Include `plan` — the actual SQL strategy and row counts at each stage. |

**200**

```json
{
  "q": "turkish airlines booking",
  "took_ms": 108,
  "embedding_ms": 39,
  "embedding_cached": false,
  "services": {
    "gmail": {
      "took_ms": 63,
      "prefiltered": 18422,
      "vector_candidates": 40,
      "fts_candidates": 27,
      "returned": 3,
      "hits": [
        {
          "id": "jIRi4gX42rxe3yWWTDdAR",
          "ref": { "message_id": "18f2c9a4b7e10d33", "thread_id": "18f2c9a4b7e10d33", "chunk_index": 0 },
          "title": "Your Turkish Airlines booking TK1234",
          "snippet": "Dear Ayşe Yılmaz, your booking is confirmed. PNR TK1234. Istanbul (IST) → New York (JFK), TK1988, 5 November 10:30…",
          "when": "2026-07-15T09:12:00Z",
          "from": "no-reply@turkishairlines.com",
          "scores": {
            "cosine": 0.7421,
            "cn": 0.87,
            "ts_rank": 0.2913,
            "vec_rank": 1,
            "fts_rank": 2,
            "rrf": 0.032522,
            "decay": 0.3012,
            "final": 0.009796
          },
          "evidence": ["ALIAS_TOKEN_IN_SUBJECT"],
          "extracted": { "pnr": "TK1234", "flight_no": "TK1988", "date": "2026-11-05" }
        }
      ]
    },
    "gcal": {
      "took_ms": 41,
      "prefiltered": 1204,
      "vector_candidates": 40,
      "fts_candidates": 6,
      "returned": 1,
      "hits": [
        {
          "id": "9E2P6oUeBmNlTVKhLkzDk",
          "ref": { "event_id": "6k9m2p4q8r1s3t5u7v9w", "calendar_id": "primary" },
          "title": "Istanbul → JFK, TK1988",
          "snippet": "PNR TK1234. Terminal 1, gate posted 40 min before departure.",
          "when": "2026-11-05T07:30:00Z",
          "scores": {
            "cosine": 0.6903,
            "cn": 0.81,
            "ts_rank": 0.3412,
            "vec_rank": 1,
            "fts_rank": 1,
            "rrf": 0.032787,
            "decay": 1.0,
            "boost": 1.0,
            "final": 0.032787
          },
          "evidence": ["EXACT_ID"],
          "extracted": { "pnr": "TK1234" }
        }
      ]
    },
    "gdrive": { "took_ms": 37, "prefiltered": 3311, "vector_candidates": 40, "fts_candidates": 0, "returned": 0, "hits": [] }
  },
  "decision": {
    "top_cn": 0.87,
    "runner_up_cn": 0.41,
    "margin": 0.46,
    "floor_read": 0.55,
    "floor_write": 0.80,
    "margin_threshold": 0.15,
    "above_read_floor": true,
    "above_write_floor": true,
    "ambiguous": false,
    "verdict": "confident"
  }
}
```

#### Reading the score components

- **`cosine`** — raw pgvector cosine similarity, `1 - (embedding <=> $query)`.
- **`cn`** — cosine normalised per corpus: z-scored against that user's
  distribution for the service, then clamped to 0..1. **This is the number the
  floors and the ambiguity margin are computed on**, together with `evidence`.
- **`ts_rank`** — Postgres `ts_rank_cd` over the weighted `tsv` (subject/title
  weight A, body weight B).
- **`vec_rank` / `fts_rank`** — 1-based positions in the two candidate lists.
- **`rrf`** — reciprocal rank fusion, `1/(60 + vec_rank) + 1/(60 + fts_rank)`
  with k=60. Above: `1/61 + 1/62 = 0.032522`. **Ordering only.**
- **`decay`** — `exp(-age_days / 30)` for mail. The booking is 36 days old:
  `exp(-1.2) = 0.3012`.
- **`boost`** — calendar only; future events are boosted so "my next flight"
  does not surface last year's.
- **`final`** — `rrf × decay × boost`. The sort key. Above:
  `0.032522 × 0.3012 = 0.009796`.

**The floors are never computed on `rrf` or `final`.** RRF is rank-derived: the
best of three bad matches gets the same 0.032522 as a perfect one. A rank-based
number cannot tell you whether anything is actually relevant, so relevance
(`FLOOR_READ` 0.55, `FLOOR_WRITE` 0.80) and ambiguity (`MARGIN` 0.15) are
decided on `cn` plus the `evidence` flag — `EXACT_ID`, `EXACT_SENDER`,
`EXACT_FILENAME`, `ALIAS_TOKEN_IN_SUBJECT`.

These thresholds are hand-set, not calibrated against a labelled set. They are
a defensible starting point, not a tuned result, and the honest way to improve
them is to collect data.

`decision.verdict` ∈ `confident` (top above `FLOOR_READ`, margin clear),
`ambiguous` (two candidates within `MARGIN` — the chat path raises an
`ask.user` step), `absent` (nothing above `FLOOR_READ` — the chat path says so
rather than guessing).

Errors: `VALIDATION_ERROR` 422 · `NOT_AUTHENTICATED` 401 ·
`GOOGLE_REAUTH_REQUIRED` 428 (`freshness=live` only) ·
`GOOGLE_UNAVAILABLE` 503 (`freshness=live` only).

```bash
curl -sS -G "$BASE/api/v1/search" \
  --data-urlencode 'q=turkish airlines booking' \
  --data-urlencode 'services=gmail,gcal' \
  --data-urlencode 'limit=5' \
  --data-urlencode 'explain=true' \
  -b cookies.txt | jq '.decision, .services.gmail.hits[0].scores'
```

---

## Health and metrics

These three live at the root, not under `/api/v1`, because a load balancer
should not have to know your API version. None of them require a session.

### `GET /healthz`

Liveness. Answers from process memory; touches nothing.

**200**

```json
{ "status": "ok", "version": "0.1.0", "env": "development", "uptime_s": 8412 }
```

Only fails by not answering, which is exactly what a liveness probe should
mean.

### `GET /readyz`

Readiness. Checks the things a request needs, in parallel, with a 500ms budget
each.

**200**

```json
{
  "status": "ready",
  "checks": {
    "postgres":  { "ok": true, "latency_ms": 3,  "detail": "SELECT 1" },
    "pgvector":  { "ok": true, "latency_ms": 1,  "detail": "extension present" },
    "redis":     { "ok": true, "latency_ms": 1,  "detail": "PING" },
    "celery":    { "ok": true, "latency_ms": 12, "detail": "3 workers, queues: sync embed actions orchestration maintenance" },
    "migrations":{ "ok": true, "latency_ms": 2,  "detail": "at head 0007_conversation_entities" }
  }
}
```

**503** when any check fails — same body shape, `"status": "not_ready"`, and
the failing check carries `"ok": false` with the reason. A failed `celery`
check does **not** fail readiness: the API serves reads without a worker, and
taking the whole tier out of the load balancer because a queue is down would
turn a degradation into an outage. It reports `"ok": false` and readiness stays
`ready`.

### `GET /metrics`

Prometheus text format, `text/plain; version=0.0.4`.

```
# HELP orchestrator_run_duration_seconds End-to-end run latency.
# TYPE orchestrator_run_duration_seconds histogram
orchestrator_run_duration_seconds_bucket{class="read",le="1.0"} 8123
orchestrator_run_duration_seconds_bucket{class="read",le="2.0"} 9761
orchestrator_run_duration_seconds_bucket{class="write_prepare",le="2.0"} 1442
# HELP orchestrator_llm_calls_total LLM calls, by purpose.
# TYPE orchestrator_llm_calls_total counter
orchestrator_llm_calls_total{purpose="plan"} 9812
orchestrator_llm_calls_total{purpose="synthesize"} 3204
# HELP orchestrator_front_door_total Queries answered with zero LLM calls.
# TYPE orchestrator_front_door_total counter
orchestrator_front_door_total{route="rule_router"} 4411
orchestrator_front_door_total{route="open_card"} 1288
# HELP orchestrator_probe_seconds Retrieval probe latency.
# TYPE orchestrator_probe_seconds histogram
orchestrator_probe_seconds_bucket{le="0.15"} 11902
# HELP orchestrator_google_errors_total Google API errors by class.
# TYPE orchestrator_google_errors_total counter
orchestrator_google_errors_total{service="gmail",class="TRANSIENT"} 31
# HELP orchestrator_mirror_lag_seconds Seconds since last successful sync.
# TYPE orchestrator_mirror_lag_seconds gauge
orchestrator_mirror_lag_seconds{service="gmail"} 421
# HELP orchestrator_cache_hits_total Redis cache hits by key class.
# TYPE orchestrator_cache_hits_total counter
orchestrator_cache_hits_total{kind="embedding"} 20114
orchestrator_cache_misses_total{kind="embedding"} 4288
```

In production this endpoint is bound to the internal listener only.

```bash
curl -sS "$BASE/healthz" | jq
curl -sS "$BASE/readyz"  | jq
curl -sS "$BASE/metrics" | head -n 20
```

---

## Known limits

Stated here because an API document that only lists happy paths is not
documentation.

- **Plausible-but-wrong retrieval is not detected.** The system detects
  *ambiguity* (two close candidates) and *absence* (nothing above the floor).
  It does not detect a confident single hit that happens to be the wrong
  thing. `GET /api/v1/search` is the tool for finding out that it happened.
- **The thresholds are uncalibrated.** `FLOOR_READ` 0.55, `MARGIN` 0.15,
  `FLOOR_WRITE` 0.80 are hand-set.
- **The mirror can be 15 minutes stale.** Superlative queries ("the latest…",
  "my next…") should pass `freshness: "live"`. `GET /api/v1/sync/status` tells
  you the actual lag.
- **An `event` row has no attachments column.** "No agenda" therefore means no
  Drive link in the description and a short description — not a checked
  attachment list.
- **A `file` row has no created date.** "PDFs from last month" means *modified*
  last month.
- **Depth beyond three genuine hops hands back to the user** as an `ask.user`
  step rather than planning a fourth round.
- **Latency.** P95 under 2s holds for the read class — about 150ms for a
  rule-routed answer, about 800ms for a single-call template read. Two-call
  prose reads land around 3.1s. What is defended is time to first meaningful
  pixel: 5ms progress, 200ms candidate chips, 590ms intent, 650ms ambiguity
  card, 780ms list, about 1.6s first prose token. There is no claim of P99
  under 2s across the board.

---

## Machine-readable spec

`docs/openapi.json` is generated from the running app:

```bash
cd backend && python -m scripts.export_openapi
```

`docs/postman_collection.json` is a Postman v2.1 collection covering every
endpoint here, with `{{base_url}}` and `{{conversation_id}}` variables and test
scripts that chain a query into a prompt response.
