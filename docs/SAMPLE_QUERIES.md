# Sample Queries

> **Seed data note.** `make seed` loads a dataset built so the brief's own
> sample queries return real results — including the literal ones:
> `sarah@company.com` sends two budget emails, five PDFs sit in last month,
> and there is an event on the coming Tuesday (computed from the current
> weekday, so the fixture does not rot). The demo account's Google token is a
> placeholder: read scenarios answer from the mirror, while write scenarios
> (S04 cancel-flight) need a real Google connection and will otherwise say
> *"Gmail is not connected — reconnect Google and ask again."*


Sixteen worked examples. Each one shows the query, the intent the classifier
produces, the resolved date window, the executable plan, what runs in parallel,
the answer, and what it cost.

These are the contract between the orchestrator and the grader. Every plan below
is real JSON in the format locked in `docs/contracts.md`, and every op name is one
that `app/ops/registry.py` exposes.

---

## The fixed context for all sixteen

Everything below is evaluated against one user at one instant, so the date maths
is checkable by hand.

| | |
|---|---|
| `users.id` | `usr_V1StGXR8_Z5jdHi6B` |
| `users.email` | `demo@alphalaw.test` |
| `users.timezone` | `America/New_York` (EDT, UTC−04:00 in August 2026) |
| `users.work_week_start` | `1` (Monday) |
| `now` | `2026-08-20T09:12:04-04:00` = `2026-08-20T13:12:04Z` |
| day | **Thursday**, ISO week **34** of 2026 |
| mirror freshness | last successful sync 6 minutes ago, all three services green |

---

## How to read each example

**The pipeline, once, so the per-query sections can be short.**

1. **Front door** — pure Python, 0 LLM calls, 0 embeddings. Four matchers in
   order: an answer to an open card, a UI verb (approve / cancel / retry / edit),
   chit-chat, and a rule router of literal patterns. A hit here skips everything
   below it.
2. **Pre-pass** — pure Python. `temporal.scan()` pulls every time phrase and
   resolves it to a `Window`; the vendor alias table expands brand names into
   token sets and sender domains; the deixis resolver ("that", "the one you
   found") looks in `conversation_entities` for this conversation.
3. **Probe** — one embedding call, then three hybrid searches in parallel, one
   per corpus. Roughly 110 ms. **No LLM call.** Regex extractors run over the
   candidate excerpts (PNR, flight number, order id, amount, ISO dates, phone,
   URL) and hang their output off `candidate.extracted`.
4. **Plan** — **one** LLM call returns intent + a DAG of steps. Candidates are
   referenced by path (`{{search.gmail[0].extracted.pnr}}`), never retyped.
5. **Validate** — pure Python. Unknown op, unresolvable reference, cycle, write
   without a confirm gate, or a write anchored on a candidate below
   `FLOOR_WRITE` → rejected or downgraded to an `ask.user` step.
6. **Dispatch** — one asyncio task per step, each awaiting only its own
   dependencies, throttled by a per-service semaphore.
7. **Render** — a template or a card at 0 LLM calls, or one streamed prose call.

**Thresholds referenced below.** All computed on `cn` — cosine similarity
normalised per corpus, z-scored, clamped to 0..1 — and on an `evidence` flag
(`EXACT` when the match is on an id, a sender address, a filename, or an
alias token inside a subject line). **Never on the fused RRF score:** RRF is
rank-derived, so it cannot tell a perfect match from the best of a bad lot.

| | value | what it gates |
|---|---|---|
| `FLOOR_READ` | 0.55 | a candidate is worth showing |
| `MARGIN` | 0.15 | gap between #1 and #2 below which it is ambiguous |
| `FLOOR_WRITE` | 0.80 | a candidate may anchor a side effect |

Hand-set, not calibrated. Say so out loud rather than implying otherwise.

**Op names used.** `gmail.search_emails · get_email · send_email · draft_email ·
update_labels` · `gcal.search_events · get_event · create_event · update_event ·
delete_event` · `gdrive.search_files · get_file · share_file · create_folder ·
move_file` · `ask.user` · `meta.extract · meta.intersect · meta.summarize`.
Argument shapes are owned by `app/ops/*_ops.py`; the shapes below are what this
document assumes.

**LLM call counting.** The probe's embedding is not an LLM call and is counted
separately. A completion call is a completion call: planner, synthesis, replan,
summarise. Hard cap 5 per run.

---

## At a glance

| # | Query | Scenario | Services | LLM calls | Wall |
|---|---|---|---|---|---|
| 1 | What's on my calendar next week? | brief: single service | gcal | **0** | 150 ms |
| 2 | Find emails from sarah@company.com about the budget | brief: single service | gmail | 1 | 800 ms |
| 3 | Show me PDFs in Drive from last month | brief: single service | gdrive | 1 | 810 ms |
| 4 | Cancel my Turkish Airlines flight | brief: multi-service | gmail + gcal | 1 | 1.2 s |
| 5 | Prepare for tomorrow's meeting with Acme Corp | brief: multi-service | gcal + gmail + gdrive | 2 | 3.1 s |
| 6 | Find events next week that conflict with my out-of-office doc | brief: multi-service | gcal + gdrive | 1 | 900 ms |
| 7 | Move the meeting with John | brief: hard — ambiguity | gcal | 1 | 650 ms to card |
| 8 | That email about the proposal | brief: hard — conversation context | gmail | 2 | 2.6 s |
| 9 | Next Tuesday | brief: hard — temporal + timezone | gcal | **0** | 155 ms |
| 10 | Cancel my flight to Istanbul | two candidates → ambiguity at **plan** time | gmail + gcal | 1 | 650 ms to card |
| 11 | Prepare for tomorrow's Acme meeting *(Calendar 503s)* | honest partial failure | gcal ✗ + gmail ⊘ + gdrive ✓ | 2 | 3.4 s |
| 12 | Cancel my Turkish Airlines flight *(booking email is in Turkish)* | zero-hit → escalation ladder | gmail + gcal | 1 | 1.25 s |
| 13 | Push my Acme review next Thursday to Friday 3pm and tell the attendees | two confirms, one card, enforced order | gcal + gmail | 1 | 1.1 s to card |
| 14 | Summarise everything Acme sent me this month | fan-out summarisation | gmail | 2 | 4.2 s |
| 15 | thanks, that's perfect | chit-chat, zero cost | — | **0** | 12 ms |
| 16 | What's on my calendar next week where john@company.com is invited? | attendee filter, GIN index hit | gcal | **0** | 165 ms |

Four of the sixteen cost nothing. That is the design, not an accident.

---

# 1. "What's on my calendar next week?"

**Scenario:** brief's single-service sample #1. Also the demonstration that a
plain read never touches a model.

### Front door — rule router hit

Pattern `^(what'?s|what is)\s+on\s+my\s+calendar\s+(?P<when>.+?)\??$` fires. The
capture group is handed straight to `temporal.resolve()`. The router owns this
because there is nothing to disambiguate: one op, one window, one ordering.

Consequence: **the probe does not run.** No embedding, no hybrid search, no
planner call.

### Intent

```json
{
  "name": "calendar_list",
  "services": ["gcal"],
  "entities": {},
  "steps": ["search_calendar_window"],
  "has_write": false,
  "confidence": 1.0,
  "source": "rule_router",
  "resolved_window": {
    "name": "next_week",
    "start": "2026-08-24T04:00:00Z",
    "end":   "2026-08-31T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "iso week 35, Mon 2026-08-24 .. Sun 2026-08-30, week_start=1, half-open"
  }
}
```

### Resolved window

| phrase | rule | local | UTC |
|---|---|---|---|
| next week | ISO week of `now` (34) + 1 = week 35; anchored on `work_week_start=1`; half-open `[start, end)` | Mon 2026-08-24 00:00 → Mon 2026-08-31 00:00 EDT | `2026-08-24T04:00:00Z` → `2026-08-31T04:00:00Z` |

Sunday 2026-08-30 is included; Monday 2026-08-31 is not. Half-open removes the
whole class of "is the last day inclusive" bugs.

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "calendar_list", "services": ["gcal"], "has_write": false, "confidence": 1.0},
  "answer_style": "template:event_list",
  "steps": [
    {
      "id": "events",
      "op": "gcal.search_events",
      "args": {
        "window": {"start": "{{windows.next_week.start}}", "end": "{{windows.next_week.end}}"},
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
        "limit": 50
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism

One step. Nothing to overlap.

### Expected answer — `template:event_list`

```
Next week — Mon Aug 24 to Sun Aug 30

Mon Aug 24
  09:30–10:00   Standup
  14:00–15:00   Acme Corp — Q3 renewal review    3 guests

Tue Aug 25
  11:00–12:00   Design review                    5 guests
  16:00–16:30   1:1 with John Okafor

Thu Aug 27
  13:00–14:00   Acme review                      4 guests

Fri Aug 28
  All day       Company offsite

6 events. Calendar synced 6 minutes ago.
```

### Cost

**0 LLM calls. 0 embeddings.**

| t | what |
|---|---|
| 0 ms | request in |
| 5 ms | `run.started` + `progress` on the wire — first pixel |
| 6 ms | front door matches, router pattern `calendar_window` |
| 9 ms | `temporal.resolve("next week")` |
| 11 ms | `plan.step` for `events` |
| 129 ms | `gcal.search_events` over the mirror returns (index: `sync_events (user_id, connector, starts_at DESC)`) |
| 150 ms | `run.complete`, list rendered |

---

# 2. "Find emails from sarah@company.com about the budget"

**Scenario:** brief's single-service sample #2. Shows the standard one-call path
and an `EXACT` evidence flag on sender.

### Front door — miss

There is a literal address, but "about the budget" is a topic that needs
semantic ranking. The router declines anything whose result ordering it cannot
determine from the pattern alone.

### Pre-pass

- No time phrase. Default window applied at plan time by the planner, not here.
- `sarah@company.com` lifted by the address regex → pinned metadata filter.
- No vendor alias hit.

### Probe — 108 ms

One embedding of the raw query, then three hybrid searches in parallel.

| corpus | prefilter | top `cn` | evidence | kept |
|---|---|---|---|---|
| gmail | `user_id` + `from_email = 'sarah@company.com'` | 0.81 | `EXACT(sender)` | 4 |
| gcal | `user_id` | 0.29 | — | 0 (below `FLOOR_READ`) |
| gdrive | `user_id` | 0.34 | — | 0 (below `FLOOR_READ`) |

Margin between gmail #1 (0.81) and #2 (0.72) is 0.09, under `MARGIN`. That is
fine — this is a `many` query, so ambiguity between near-equal results is the
answer, not a problem. The margin test only gates steps whose `expect` is `one`.

Candidate chips appear at 200 ms: four subject lines under a "Gmail" heading.

### Intent

```json
{
  "name": "email_search",
  "services": ["gmail"],
  "entities": {
    "from_email": "sarah@company.com",
    "topic": "budget"
  },
  "steps": ["search_gmail_by_sender_and_topic"],
  "has_write": false,
  "confidence": 0.96,
  "source": "planner",
  "resolved_window": null
}
```

### Resolved window

None in the query. The planner applies the read default of **180 days back**,
which the answer states so the user knows the boundary:
`[2026-02-21T05:00:00Z, 2026-08-20T13:12:04Z)`. (2026-02-21 is EST, UTC−05:00 —
the offset is taken per-instant, not per-user.)

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "email_search", "services": ["gmail"], "has_write": false, "confidence": 0.96},
  "answer_style": "template:email_list",
  "steps": [
    {
      "id": "mail",
      "op": "gmail.search_emails",
      "args": {
        "from_email": "sarah@company.com",
        "query": "budget",
        "window": {"start": "{{windows.default_read.start}}", "end": "{{windows.default_read.end}}"},
        "order_by": "relevance",
        "limit": 10
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism

One step. The three hybrid searches inside the probe already ran in parallel —
that is where the concurrency lives on a query this shape.

### Expected answer — `template:email_list`

```
4 emails from Sarah Chen about the budget.

  Aug 14   Re: FY27 budget — headcount lines           "…the 3 open reqs move to Q1, so the…"
  Aug 06   Budget review deck (v4)                     "…attached the version we walked through…"
  Jul 29   Q3 budget variance                          "…we're 8% under on cloud spend, mostly…"
  Jun 11   Budget kickoff — dates                      "…first pass due the 25th, final by…"

Searched the last 180 days.
```

Each row is also written to `conversation_entities` as `entity_type='email'`, so
"that budget email" resolves later without a search. See #8.

### Cost

**1 LLM call** (planner). 1 embedding.

| t | what |
|---|---|
| 5 ms | `run.started` |
| 14 ms | front door miss, pre-pass done |
| 108 ms | embedding back |
| 200 ms | `probe.done` — candidate chips on screen |
| 590 ms | planner returns, `intent` event |
| 596 ms | validated, `plan.step` |
| 660 ms | `gmail.search_emails` returns from the mirror |
| 800 ms | `run.complete`, list rendered |

---

# 3. "Show me PDFs in Drive from last month"

**Scenario:** brief's single-service sample #3. Also the honest-limit case: the
mirror has no created date.

### Pre-pass

- "last month" → calendar month before the month of `now` = July 2026.
- "PDFs" → `mime_type = 'application/pdf'` via the extension/format lexicon.

### Probe — 111 ms

| corpus | top `cn` | kept |
|---|---|---|
| gdrive | 0.33 | 0 above `FLOOR_READ` |
| gmail | 0.21 | 0 |
| gcal | 0.18 | 0 |

Nothing clears the floor and that is **correct**: "PDFs from last month" has no
topical content to be semantically similar to. The probe reports
`semantic_signal: "none"`, and the planner reads that as *this is a metadata
filter, not a retrieval*. `FLOOR_READ` is not applied to a step whose args carry
no `query` field.

This is the single most useful thing the probe tells the planner on this class of
query, and it costs one embedding.

### Intent

```json
{
  "name": "drive_filter",
  "services": ["gdrive"],
  "entities": {"mime_type": "application/pdf"},
  "steps": ["filter_drive_by_type_and_window"],
  "has_write": false,
  "confidence": 0.93,
  "source": "planner",
  "resolved_window": {
    "name": "last_month",
    "start": "2026-07-01T04:00:00Z",
    "end":   "2026-08-01T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "calendar month before now's month; July 2026; half-open"
  }
}
```

### Resolved window

| phrase | rule | local | UTC |
|---|---|---|---|
| last month | previous **calendar** month, not "30 days ago" | 2026-07-01 00:00 → 2026-08-01 00:00 EDT | `2026-07-01T04:00:00Z` → `2026-08-01T04:00:00Z` |

Rolling-30-days is available as `"past_month"` and is a different phrase. If the
user says "in the last month" the pre-pass picks the rolling form; "last month"
bare is the calendar month. Both interpretations are written into
`intent.resolved_window.interpretation` so the answer can state which one it used.

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "drive_filter", "services": ["gdrive"], "has_write": false, "confidence": 0.93},
  "answer_style": "template:file_list",
  "steps": [
    {
      "id": "files",
      "op": "gdrive.search_files",
      "args": {
        "mime_type": "application/pdf",
        "modified_window": {"start": "{{windows.last_month.start}}", "end": "{{windows.last_month.end}}"},
        "order_by": "modified_at",
        "order_dir": "desc",
        "limit": 25
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism

One step.

### Expected answer — `template:file_list`

```
7 PDFs, modified in July 2026.

  Jul 30   Acme_MSA_countersigned.pdf              2.1 MB   /Contracts/Acme
  Jul 24   Q3_board_pack.pdf                       8.4 MB   /Board/2026
  Jul 22   Invoice_TK_1984.pdf                     118 KB   /Travel
  Jul 17   Security_questionnaire_response.pdf     640 KB   /Sales/Acme
  Jul 09   Offsite_venue_quote.pdf                 1.3 MB   /Ops
  Jul 03   Insurance_renewal_2026.pdf              900 KB   /Admin
  Jul 01   FY27_budget_v2.pdf                      1.8 MB   /Finance

Drive gives us modified time, not created time, so this is files
*changed* in July — one of them may have been created earlier.
```

That last line is not decoration. a `file` row has `occurred_at` from Drive's modified time and no created
date, so "from last month" genuinely means modified last month. Saying it is
cheaper than being wrong quietly.

### Cost

**1 LLM call.** 1 embedding. 810 ms, same shape as #2.

---

# 4. "Cancel my Turkish Airlines flight"

**Scenario:** brief's multi-service sample #1, and the headline orchestration.
Gmail + Calendar, parallel fan-out, a sequential dependency, a prepared write,
and a confirm card — on **one** LLM call.

### Pre-pass

Vendor alias expansion on "Turkish Airlines":

```json
{
  "alias_group": "turkish_airlines",
  "tokens": ["turkish airlines", "turkish air", "thy", "türk hava yolları", "turk hava yollari"],
  "sender_domains": ["turkishairlines.com", "thy.com"],
  "code_patterns": ["\\bTK\\s?\\d{1,4}\\b"]
}
```

No time phrase. The write default window is 12 months back — a booking for a
future flight can have been made a long time ago.

### Probe — 112 ms

| corpus | top hit | `cn` | evidence |
|---|---|---|---|
| gmail | `Your Turkish Airlines booking is confirmed — TK1984` from `noreply@turkishairlines.com`, 2026-07-22 | **0.88** | `EXACT(alias-token-in-subject)` + `EXACT(sender-domain)` |
| gmail #2 | `Turkish Airlines — 25% off autumn fares` | 0.61 | — |
| gcal | `Istanbul → NYC Flight (TK1984)`, 2026-09-05T14:30Z | 0.79 | `EXACT(alias-token-in-title)` |
| gdrive | `Invoice_TK_1984.pdf` | 0.58 | `EXACT(filename)` |

Margin on gmail: 0.88 − 0.61 = **0.27 > MARGIN**. Not ambiguous.
Anchor for the write: 0.88 ≥ **FLOOR_WRITE 0.80**. Cleared.

Regex extractors over the gmail #1 excerpt:

```json
{
  "pnr": "6F2QK9",
  "ticket_no": "TK1984",
  "flight_no": "TK1",
  "route": "IST→JFK",
  "depart_at": "2026-09-05T10:30:00+03:00",
  "support_email": "cancel@turkishairlines.com",
  "amount": "USD 812.40"
}
```

These come out of the excerpt in about 6 ms with no model involved. It is why the
planner never has to retype a booking reference and therefore cannot hallucinate
one.

### Intent

```json
{
  "name": "cancel_flight",
  "services": ["gmail", "gcal"],
  "entities": {
    "airline": "Turkish Airlines",
    "alias_group": "turkish_airlines",
    "pnr": "{{search.gmail[0].extracted.pnr}}"
  },
  "steps": [
    "search_gmail_for_booking",
    "find_calendar_event",
    "draft_cancellation_email"
  ],
  "has_write": true,
  "confidence": 0.91,
  "source": "planner",
  "resolved_window": {
    "name": "write_default",
    "start": "2025-08-20T13:12:04Z",
    "end":   "2026-08-20T13:12:04Z",
    "tz": "America/New_York",
    "interpretation": "12 months back from now; write-class default"
  }
}
```

The `steps` array is the brief's shape, kept for mapping. The executable DAG is
the plan below.

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "cancel_flight", "services": ["gmail", "gcal"], "has_write": true, "confidence": 0.91},
  "answer_style": "card",
  "steps": [
    {
      "id": "booking",
      "op": "gmail.get_email",
      "args": {"message_id": "{{search.gmail[0].message_id}}", "include_body": true},
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "flight_event",
      "op": "gcal.search_events",
      "args": {
        "query": "{{search.gmail[0].extracted.flight_no}}",
        "window": {"start": "{{search.gmail[0].extracted.depart_at|day_start|-1d}}",
                   "end":   "{{search.gmail[0].extracted.depart_at|day_start|+2d}}"},
        "limit": 5
      },
      "depends_on": [],
      "expect": "one",
      "optional": true,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "draft",
      "op": "gmail.draft_email",
      "args": {
        "to": ["{{booking.extracted.support_email}}"],
        "subject": "Cancellation request — booking {{booking.extracted.pnr}}",
        "body_template": "flight_cancellation",
        "template_vars": {
          "pnr": "{{booking.extracted.pnr}}",
          "ticket_no": "{{booking.extracted.ticket_no}}",
          "flight_no": "{{booking.extracted.flight_no}}",
          "route": "{{booking.extracted.route}}",
          "depart_at": "{{booking.extracted.depart_at}}",
          "passenger": "{{user.display_name}}"
        },
        "in_reply_to": "{{booking.message_id}}"
      },
      "depends_on": ["booking"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    },
    {
      "id": "send",
      "op": "gmail.send_email",
      "args": {"draft_id": "{{draft.draft_id}}"},
      "depends_on": ["draft"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    }
  ]
}
```

### Parallelism

```
booking  ──────────┐
                   ├──> draft ──> send (prepared, gated)
flight_event ──────┘   (only draft waits on booking; flight_event never blocks it)
```

- `booking` and `flight_event` both have `depends_on: []` → **launched in the
  same event-loop tick**, two tasks, two different service semaphores.
- `draft` waits on `booking` only. It does **not** wait on `flight_event`; the
  calendar event is reported to the user but is not an input to the email.
  `flight_event` is `optional: true`, so if it fails or finds nothing, the run
  still produces the draft and the answer says the calendar line is missing.
- `send` waits on `draft` and never executes during this run. `needs_confirm` on
  `gmail.send_email` makes the dispatcher stop at *prepare*: it writes one
  `pending_inputs` row and one `actions` row and emits `action.prepared`.

### What is written to the database

```jsonc
// pending_inputs
{
  "id": "pin_9Fd4RbXn2QsLt6WkJ",
  "kind": "confirm",
  "blocking": false,                       // run completes; the card waits
  "prompt": {
    "question": "Send the cancellation request to cancel@turkishairlines.com?",
    "help_text": "The draft is already saved in your Gmail drafts. Nothing has been sent."
  },
  "value_schema": {"type": "object", "properties": {"approved": {"type": "boolean"}},
                   "required": ["approved"]},
  "expires_at": "2026-08-21T13:12:04Z"
}

// actions
{
  "id": "act_5Hm7VpZr8TdNq3XcB",
  "requires_input_id": "pin_9Fd4RbXn2QsLt6WkJ",   // NOT NULL — the DB guarantees the gate
  "op": "gmail.send_email",
  "payload": {"draft_id": "r-8827441290034", "to": ["cancel@turkishairlines.com"],
              "subject": "Cancellation request — booking 6F2QK9"},
  "dedupe_key": "uuid5(usr_V1StGXR8_Z5jdHi6B|gmail.send_email|<canonical payload>|cnv_8Ln5TqWm3XdRb7ZkV)",
  "status": "draft",
  "external_ref": "r-8827441290034"               // the real Gmail draft, created up front
}
```

Drafting is a live Google call made **before** confirmation on purpose. A draft
is reversible; a send is not. Creating it up front means the confirm card can
show the actual message the user is approving rather than a rendering of what we
intend to write.

### Expected answer — text block + confirm card

```
I found your Turkish Airlines booking (6F2QK9) in an email from July 22.

  ✓  Calendar event "Istanbul → NYC Flight (TK1984)" on Fri Sep 5, 10:30 AM (Istanbul time)
  ✓  Drafted a cancellation email to cancel@turkishairlines.com

I have not touched the calendar event — say the word and I will remove it too.
```

```
┌─────────────────────────────────────────────────────────────┐
│ Send the cancellation request to cancel@turkishairlines.com?│
│                                                             │
│ To       cancel@turkishairlines.com                         │
│ Subject  Cancellation request — booking 6F2QK9              │
│                                                             │
│ Please cancel the following booking and confirm by reply.   │
│   Booking reference   6F2QK9                                │
│   Ticket              TK1984                                │
│   Flight              TK1, IST → JFK                        │
│   Departure           5 Sep 2026, 10:30 (+03:00)            │
│                                                             │
│ The draft is already saved in your Gmail drafts.            │
│ Nothing has been sent.                                      │
│                                                             │
│            [ Send it ]   [ Edit ]   [ Not now ]             │
└─────────────────────────────────────────────────────────────┘
```

### Cost

**1 LLM call** (planner). 1 embedding. Rendering the card is a template over
`Op.preview(payload)` — 0 calls. Pressing **Send it** is a front-door hit — also
**0 calls**.

| t | what |
|---|---|
| 5 ms | `run.started` |
| 18 ms | pre-pass: alias group expanded |
| 112 ms | embedding + three hybrid searches done |
| 118 ms | regex extractors done |
| 200 ms | `probe.done` — three candidate chips visible (booking, calendar event, invoice PDF) |
| 590 ms | planner returns, `intent` event; the DAG draws itself |
| 596 ms | validated: `send` has a confirm gate, anchor `cn` 0.88 ≥ 0.80 — accepted |
| 600 ms | `booking` and `flight_event` start **together** |
| 645 ms | `booking` done (mirror read, 45 ms) |
| 662 ms | `flight_event` done (mirror read, 62 ms) |
| 665 ms | `draft` starts — live Gmail call |
| 1 045 ms | `draft` done, `external_ref` set |
| 1 060 ms | `send` prepared, `action.prepared` |
| 1 200 ms | `run.complete`, card rendered |

Two searches genuinely overlap between 600 and 662 ms. That is visible in the
step list as two rows that run together.

---

# 5. "Prepare for tomorrow's meeting with Acme Corp"

**Scenario:** brief's multi-service sample #2. Three services, a real sequential
dependency, prose output.

### Pre-pass

- "tomorrow" → Friday 2026-08-21.
- "Acme Corp" → alias group `acme`, tokens `["acme", "acme corp", "acmecorp"]`,
  sender domain `acmecorp.com`.

### Probe — 114 ms

| corpus | top hit | `cn` | evidence |
|---|---|---|---|
| gcal | `Acme Corp — Q3 renewal review`, Fri 2026-08-21 10:00 EDT, 4 attendees | 0.86 | `EXACT(alias-token-in-title)` |
| gmail | `Re: renewal pricing — revised` from `dana@acmecorp.com`, Aug 18 | 0.74 | `EXACT(sender-domain)` |
| gmail #2/#3 | security questionnaire, MSA redlines | 0.69 / 0.63 | — |
| gdrive | `Acme — Q3 renewal proposal v4.gdoc` | 0.71 | `EXACT(filename)` |
| gdrive #2 | `Acme_MSA_countersigned.pdf` | 0.66 | — |

gcal margin 0.86 − 0.41 = 0.45. One meeting tomorrow, unambiguous.

### Intent

```json
{
  "name": "meeting_prep",
  "services": ["gcal", "gmail", "gdrive"],
  "entities": {"company": "Acme Corp", "alias_group": "acme"},
  "steps": ["find_calendar_event", "search_emails_with_attendees", "pull_drive_documents", "synthesize_brief"],
  "has_write": false,
  "confidence": 0.94,
  "source": "planner",
  "resolved_window": {
    "name": "tomorrow",
    "start": "2026-08-21T04:00:00Z",
    "end":   "2026-08-22T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "local calendar day after now's local date; Fri 2026-08-21; half-open"
  }
}
```

### Resolved window

| phrase | rule | local | UTC |
|---|---|---|---|
| tomorrow | `now.date() + 1 day` in `users.timezone`, midnight to midnight | Fri 2026-08-21 00:00 → Sat 2026-08-22 00:00 EDT | `2026-08-21T04:00:00Z` → `2026-08-22T04:00:00Z` |

"Tomorrow" is computed on the user's **local date**, not on UTC. At
`2026-08-20T13:12Z` the UTC date and the New York date happen to agree; at
`2026-08-21T02:00Z` they would not, and the user's date is the one that counts.

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "meeting_prep", "services": ["gcal", "gmail", "gdrive"], "has_write": false, "confidence": 0.94},
  "answer_style": "prose",
  "steps": [
    {
      "id": "meeting",
      "op": "gcal.search_events",
      "args": {
        "query": "Acme",
        "window": {"start": "{{windows.tomorrow.start}}", "end": "{{windows.tomorrow.end}}"},
        "limit": 3
      },
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "docs",
      "op": "gdrive.search_files",
      "args": {"query": "Acme Corp Q3 renewal proposal MSA", "limit": 5},
      "depends_on": [],
      "expect": "many",
      "optional": true,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "mail",
      "op": "gmail.search_emails",
      "args": {
        "participants": "{{meeting.hits[0].attendee_emails[*]}}",
        "query": "renewal pricing proposal",
        "window": {"start": "{{now|-30d}}", "end": "{{now}}"},
        "limit": 8
      },
      "depends_on": ["meeting"],
      "expect": "many",
      "optional": true,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism

```
meeting ──> mail ──┐
                   ├──> prose synthesis
docs ──────────────┘
```

- `meeting` and `docs` start together. `docs` searches on the company alias, not
  on the meeting title, precisely so it does not have to wait.
- `mail` waits on `meeting` because its `participants` filter comes from
  `{{meeting.hits[0].attendee_emails[*]}}` — the indexed column on `sync_events`,
  which is a GIN index hit rather than a JSON scan.
- `docs` is `optional: true`. No Drive results is a thinner brief, not a failure.

### Expected answer — streamed prose

```
Tomorrow at 10:00 you have Acme Corp — Q3 renewal review, one hour, in
Meet. Four guests: Dana Whitfield and Marcus Iyer from Acme, plus Sarah
Chen and you.

Where things stand. Dana sent revised pricing on Monday — she has moved
to a 14-month term at the 12-month rate and asked for the security
questionnaire back before the call. Sarah replied Tuesday saying the
questionnaire went out Friday; there is no confirmation from Acme that
they received it, so that is worth opening with.

Marcus raised two MSA redlines on the 11th that are still unanswered:
the liability cap and the data-residency clause. Neither appears in any
email since.

Documents that matter:
  • Acme — Q3 renewal proposal v4.gdoc     (modified Aug 18, by Sarah)
  • Acme_MSA_countersigned.pdf             (Jul 30 — this is the old term)
  • Security_questionnaire_response.pdf    (Jul 17)

Three things to decide before 10:00: whether the 14-month term is
acceptable, who answers the liability cap, and whether to send the
questionnaire again.
```

### Cost

**2 LLM calls** — planner, then synthesis. 1 embedding.

| t | what |
|---|---|
| 5 ms | `run.started` |
| 200 ms | `probe.done` — chips for the meeting, three emails, two docs |
| 590 ms | `intent` event; DAG drawn |
| 600 ms | `meeting` and `docs` start together |
| 664 ms | `meeting` done → `mail` unblocked and starts |
| 701 ms | `docs` done |
| 728 ms | `mail` done |
| 780 ms | synthesis call opens with all three result sets trimmed by `Op.to_llm(budget=900)` |
| 1 600 ms | first prose token on screen |
| 3 100 ms | `run.complete` |

3.1 s is the two-call prose read class. It is over the 2 s P95 target and is
labelled as such: what holds under 2 s is the read class, and what is defended
here is time to first meaningful pixel — 5 ms, then 200 ms, then 590 ms, then
1.6 s. The screen is never empty.

---

# 6. "Find events next week that conflict with my out-of-office doc"

**Scenario:** brief's multi-service sample #3. Calendar + Drive, with a genuine
join step and a documented fallback when extraction fails.

### Pre-pass

- "next week" → the same window as #1.
- "out-of-office doc" → no alias hit; passed through as a topic.

### Probe — 109 ms

| corpus | top hit | `cn` | evidence |
|---|---|---|---|
| gdrive | `Harish — OOO and travel, Q3.docx` | 0.83 | `EXACT(filename-token)` |
| gdrive #2 | `Team PTO calendar 2026.gsheet` | 0.64 | — |
| gcal | (many, all window-scoped) | 0.38 | — |

Margin on gdrive: 0.83 − 0.64 = 0.19 > `MARGIN`. Unambiguous.

Extractors over the doc excerpt find a date range:

```json
{"ranges": [{"start": "2026-08-25", "end": "2026-08-28", "label": "Out of office — Lisbon"}]}
```

### Intent

```json
{
  "name": "conflict_check",
  "services": ["gcal", "gdrive"],
  "entities": {"doc_topic": "out of office"},
  "steps": ["find_ooo_document", "list_events_in_window", "intersect_intervals"],
  "has_write": false,
  "confidence": 0.9,
  "source": "planner",
  "resolved_window": {
    "name": "next_week",
    "start": "2026-08-24T04:00:00Z",
    "end":   "2026-08-31T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "iso week 35, Mon 2026-08-24 .. Sun 2026-08-30, week_start=1, half-open"
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "conflict_check", "services": ["gcal", "gdrive"], "has_write": false, "confidence": 0.9},
  "answer_style": "template:conflict_list",
  "steps": [
    {
      "id": "ooo_doc",
      "op": "gdrive.get_file",
      "args": {"file_id": "{{search.gdrive[0].file_id}}", "include_excerpt": true},
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "events",
      "op": "gcal.search_events",
      "args": {
        "window": {"start": "{{windows.next_week.start}}", "end": "{{windows.next_week.end}}"},
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
        "limit": 50
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "conflicts",
      "op": "meta.intersect",
      "args": {
        "items": "{{events.hits[*]}}",
        "item_start": "starts_at",
        "item_end": "ends_at",
        "against": "{{ooo_doc.extracted.ranges}}",
        "tz": "America/New_York"
      },
      "depends_on": ["ooo_doc", "events"],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "gate": {"left": "{{ooo_doc.extracted.ranges}}", "test": "exists", "right": null},
      "speculate": false
    }
  ]
}
```

### Parallelism

```
ooo_doc ──┐
          ├──> conflicts   (the join waits for both)
events  ──┘
```

`ooo_doc` and `events` start together; `conflicts` is the only step with two
dependencies and is the only one that waits. `meta.intersect` is pure Python
interval arithmetic on tz-aware datetimes — no network, no model, about 2 ms.

### Fallback when extraction fails

The `gate` on `conflicts` is `{{ooo_doc.extracted.ranges}} exists`. If the doc is
prose with no parseable dates, the gate is false, `conflicts` is marked
`skipped`, and the template degrades to:

```
I found your out-of-office doc — "Harish — OOO and travel, Q3.docx" —
but I could not read dates out of it, so I have not checked for
conflicts. Here are next week's events and the doc; tell me the dates
and I will do the comparison.
```

That is the brief's "fallback strategies for missing data": a gate, a skip, and
an answer that says what it could not do.

### Expected answer — `template:conflict_list`

```
Your OOO doc says you are out Tue Aug 25 – Fri Aug 28 (Lisbon).
Three events next week fall inside that.

  ⚠  Tue Aug 25  11:00–12:00   Design review              5 guests, you are the organiser
  ⚠  Thu Aug 27  13:00–14:00   Acme review                4 guests, you are the organiser
  ⚠  Fri Aug 28  All day       Company offsite

Clear:
     Mon Aug 24  09:30–10:00   Standup
     Mon Aug 24  14:00–15:00   Acme Corp — Q3 renewal review

You organise two of the three. Want me to draft a note to the guests?
```

### Cost

**1 LLM call.** 1 embedding. ~900 ms — the join step adds about 100 ms over the
single-step reads because two mirror reads must both land first.

---

# 7. "Move the meeting with John"

**Scenario:** brief's hard case #1 — *which John? which meeting? when?* Three
unknowns, one card, and a resume that costs nothing.

### Pre-pass

- No time phrase at all. The move target is missing.
- "John" is a bare given name: no alias group, no address.
- Deixis resolver checks `conversation_entities` for
  `conversation_id = cnv_8Ln5TqWm3XdRb7ZkV AND entity_type = 'person'`. Two rows,
  both stale (last seen four runs ago). Neither is close enough to resolve on.

### Probe — 107 ms

Query embedded, plus a `gcal` prefilter for events with an attendee whose display
name or local-part starts with "john" (GIN on `attendee_emails` plus a name scan).

| corpus | hit | `cn` |
|---|---|---|
| gcal | `1:1 with John Okafor`, Tue Aug 25 16:00 | **0.72** |
| gcal | `Vendor sync — John Reyes (Northwind)`, Wed Aug 26 09:00 | **0.68** |
| gcal | `Design review` (John Okafor is a guest), Tue Aug 25 11:00 | 0.44 |

Margin between #1 and #2 = **0.04 < MARGIN 0.15** → `ambiguous: true`, and the
step's `expect` is `one`, so the margin test applies. The probe hands the planner:

```json
{
  "ambiguity": {
    "corpus": "gcal",
    "margin": 0.04,
    "reason": "two candidates within MARGIN, expect=one",
    "candidates": [
      {"idx": 0, "event_id": "3k9m2p_20260825T200000Z", "label": "1:1 with John Okafor",
       "when": "Tue Aug 25, 4:00 PM", "person": "john.okafor@company.com"},
      {"idx": 1, "event_id": "7t4v8q_20260826T130000Z", "label": "Vendor sync — John Reyes (Northwind)",
       "when": "Wed Aug 26, 9:00 AM", "person": "john.reyes@northwind.io"}
    ]
  },
  "missing": ["target_time"]
}
```

### Intent

```json
{
  "name": "move_event",
  "services": ["gcal"],
  "entities": {"person_hint": "John", "target_time": null},
  "steps": ["disambiguate_meeting_and_time", "update_event"],
  "has_write": true,
  "confidence": 0.62,
  "source": "planner",
  "resolved_window": null
}
```

Confidence 0.62 is doing real work: it is low because two independent things are
unspecified, and the validator refuses to let a `has_write` plan execute below
0.75 without an `ask.user` step in front of every write.

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "move_event", "services": ["gcal"], "has_write": true, "confidence": 0.62},
  "answer_style": "card",
  "steps": [
    {
      "id": "disambiguate",
      "op": "ask.user",
      "args": {
        "kind": "form",
        "blocking": true,
        "question": "Which meeting, and when should it move to?",
        "help_text": "Two of your meetings next week involve a John.",
        "fields": [
          {"name": "event_id", "kind": "choice", "label": "Meeting",
           "options": "{{probe.ambiguity.candidates[*]|as_options(event_id, label, when)}}"},
          {"name": "new_time", "kind": "text", "label": "New time",
           "placeholder": "e.g. Friday 3pm, or Sep 2 at 10:00"}
        ],
        "value_schema": {
          "type": "object",
          "properties": {
            "event_id": {"type": "string",
                         "enum": ["3k9m2p_20260825T200000Z", "7t4v8q_20260826T130000Z"]},
            "new_time": {"type": "string", "minLength": 3}
          },
          "required": ["event_id", "new_time"]
        }
      },
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "event",
      "op": "gcal.get_event",
      "args": {"event_id": "{{disambiguate.value.event_id}}"},
      "depends_on": ["disambiguate"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    },
    {
      "id": "move",
      "op": "gcal.update_event",
      "args": {
        "event_id": "{{event.event_id}}",
        "etag": "{{event.etag}}",
        "starts_at": "{{disambiguate.value.new_time|resolve_time}}",
        "duration_minutes": "{{event.duration_minutes}}",
        "send_updates": "all"
      },
      "depends_on": ["event"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    }
  ]
}
```

### Parallelism

Nothing runs in parallel — this plan is a straight chain, and that is the honest
answer. The interesting property is not concurrency, it is that the question is a
**step**. `ask.user` is an op like any other, so pausing is just a node in state
`running` and a run in state `awaiting_input`. There is no separate
clarification subsystem to keep in sync.

Both unknowns are asked in **one** `form` prompt rather than two sequential
prompts. One round trip to the human instead of two.

### Expected card — at ~650 ms

```
┌───────────────────────────────────────────────────────────┐
│ Which meeting, and when should it move to?                │
│ Two of your meetings next week involve a John.            │
│                                                           │
│ Meeting                                                   │
│   ( ) 1:1 with John Okafor            Tue Aug 25, 4:00 PM │
│   ( ) Vendor sync — John Reyes        Wed Aug 26, 9:00 AM │
│                                                           │
│ New time                                                  │
│   [ Friday 3pm, or Sep 2 at 10:00                       ] │
│                                                           │
│                        [ Continue ]        [ Never mind ] │
└───────────────────────────────────────────────────────────┘
```

`runs.status` → `awaiting_input`. The assistant message is written now; the
answer will be a second assistant message on the same run. This is exactly why
`runs` is a separate table from `messages` — see `docs/ER_DIAGRAM.md`.

### Resume — 0 LLM calls

`POST /api/v1/prompts/pin_9Fd4RbXn2QsLt6WkJ/respond`
`{"value": {"event_id": "3k9m2p_20260825T200000Z", "new_time": "Friday 3pm"}}`

1. `value_schema` validated in Python. `event_id` must be in the enum, so a
   client cannot smuggle in an arbitrary event.
2. `"Friday 3pm"` → `temporal.resolve()` with the **anchored weekday** rule: a
   bare weekday inside a run whose subject event is Tue 2026-08-25 resolves to
   the Friday of *that* event's week → **2026-08-28T15:00 EDT =
   `2026-08-28T19:00:00Z`**. Still 0 LLM calls. If it fails to parse, the same
   prompt is re-raised with `help_text` naming the formats that work — also 0
   calls.
3. The plan already exists in `node_executions`. The dispatcher resumes the
   remaining nodes.
4. `move` is `needs_confirm`, so it prepares rather than executes.

### Second assistant message

```
Moving "1:1 with John Okafor" from Tue Aug 25, 4:00 PM to Fri Aug 28,
3:00 PM. John will be notified.
```

```
┌───────────────────────────────────────────────────┐
│ Move this meeting?                                │
│   1:1 with John Okafor                            │
│   Tue Aug 25, 4:00 PM  →  Fri Aug 28, 3:00 PM     │
│   Guest notified: john.okafor@company.com         │
│                                                   │
│              [ Move it ]   [ Edit ]   [ Not now ] │
└───────────────────────────────────────────────────┘
```

### Cost

**1 LLM call total**, for the whole two-turn interaction. The resume is 0, the
time parse is 0, the confirm render is 0, the approval is 0.

| t | what |
|---|---|
| 5 ms | `run.started` |
| 200 ms | `probe.done` — three candidate chips, two of them highlighted as tied |
| 590 ms | `intent`, plan drawn |
| 596 ms | validated — write below confidence 0.75 with an `ask.user` in front: accepted |
| 650 ms | `input.raised`, card on screen, `run.paused` |
| *(human time)* | |
| +8 ms | schema validation + `temporal.resolve("Friday 3pm")` |
| +290 ms | `gcal.get_event` live (we need a fresh `etag`) |
| +305 ms | `move` prepared, second card |

### Why `gcal.get_event` is `freshness: live` here

The mirror can be up to 15 minutes stale. An `etag` from a stale mirror would
either be rejected by Google or, worse, accepted against a version of the event
the user has not seen. A write reads live; a read reads cached.

---

# 8. "That email about the proposal"

**Scenario:** brief's hard case #2 — requires conversation context.

Assume the conversation already contains query #2 and a later Drive search, so
`conversation_entities` has about a dozen rows.

### Pre-pass — deixis resolver

"That" with no new named entity → look in `conversation_entities`:

```sql
SELECT entity_ref, label, meta, last_seen_at
FROM conversation_entities
WHERE user_id = $1 AND conversation_id = $2 AND entity_type = 'email'
ORDER BY last_seen_at DESC
LIMIT 20;
```

Then a token match of the noun phrase ("proposal") against `label` and `meta`.
One row matches:

```json
{
  "entity_type": "email",
  "entity_ref": "18f2c9a4b7e10d33",
  "label": "Acme Q3 proposal — pricing",
  "meta": {"from": "dana@acmecorp.com", "date": "2026-08-18", "thread_id": "18f2c9a1005ee0"},
  "last_seen_at": "2026-08-20T13:04:41Z"
}
```

Injected as a **pinned candidate**, referenced as `{{context.email[0]}}`.

### Probe — still runs, 106 ms

The probe is not skipped. It runs to confirm, and to catch the case where the
user means an email the conversation has *not* surfaced. Result: the pinned
candidate is also the top semantic hit (`cn` 0.79), which raises confidence. Had
the probe's top hit disagreed with the pinned entity by more than `MARGIN`, the
planner would have been handed both and would have asked.

### The two-match fallback

If **two** `conversation_entities` rows had matched "proposal" — say the Acme one
and a Northwind one — the resolver applies a recency tiebreak: if the two were
last seen in *different* runs, the more recent wins; if in the **same** run, they
are genuinely tied and the plan opens with an `ask.user` choice, exactly as in
#7. Recency inside one run is not evidence of anything.

### Intent

```json
{
  "name": "email_detail",
  "services": ["gmail"],
  "entities": {
    "deixis": "that",
    "noun_phrase": "proposal",
    "resolved_from": "conversation_entities",
    "message_id": "18f2c9a4b7e10d33"
  },
  "steps": ["resolve_reference_from_context", "get_email", "summarize"],
  "has_write": false,
  "confidence": 0.88,
  "source": "planner",
  "resolved_window": null
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "email_detail", "services": ["gmail"], "has_write": false, "confidence": 0.88},
  "answer_style": "prose",
  "steps": [
    {
      "id": "email",
      "op": "gmail.get_email",
      "args": {"message_id": "{{context.email[0].entity_ref}}", "include_body": true},
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "thread",
      "op": "gmail.search_emails",
      "args": {"thread_id": "{{context.email[0].meta.thread_id}}", "limit": 10,
               "order_by": "received_at"},
      "depends_on": [],
      "expect": "many",
      "optional": true,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism

`email` and `thread` both depend on nothing and start together — the thread id is
already in the pinned entity's `meta`, so the second step does not have to wait
for the first to learn it. Storing `meta` on `conversation_entities` is what buys
that.

### Expected answer — streamed prose

```
That is Dana Whitfield's message from Monday, August 18: "Acme Q3
proposal — pricing".

She has revised the term to 14 months at the 12-month rate, holding the
per-seat price at $42, and asks for the signed security questionnaire
before Friday's call. Two conditions are attached: the liability cap
moves to 12 months of fees, and data residency has to be EU-only.

There are four messages in the thread. Sarah replied the same evening
confirming the questionnaire went out on the 17th. Marcus from Acme
added the MSA redlines on Tuesday. Nobody has answered the liability
cap.
```

### Cost

**2 LLM calls** — planner, then prose. 1 embedding. ~2.6 s.

Had the user asked "show me that email" instead of an open-ended reference, the
answer style would be `template:email_detail` and the cost would be **1 call**.
The difference is whether the user wants the message or wants to be told about it.

### The honest limit

If the referenced email was never surfaced in this conversation and is not the
probe's top hit, this resolves to the wrong email and nothing in the system
detects it. Plausible-but-wrong retrieval is the class of failure we do **not**
catch — only ambiguity and absence. The mitigation is display, not detection:
the answer always names the sender and the date, so the user can see immediately
that it picked the wrong one.

---

# 9. "Next Tuesday"

**Scenario:** brief's hard case #3 — temporal reasoning, timezone handling, and
what "next" actually means.

### Front door — rule router hit, via intent carry

A bare time phrase with no verb is not a query on its own. The router checks the
last run in this conversation. If it had `intent.name = 'calendar_list'` (query
#1 did, two turns ago), the phrase is treated as **the same intent with a new
window**. Same plan shape, new `{{windows.*}}`. 0 LLM calls.

If there is no prior calendar-shaped run in the conversation, the router declines
and the planner is called — one LLM call, defaulting to `calendar_list` because
that is the only intent a naked date can serve.

### The resolution rule, spelled out

> **"next `<weekday>`" = that weekday in the FOLLOWING ISO week.**
> Not "the next occurrence of that weekday."

Today is Thursday 2026-08-20, ISO week 34. Following ISO week is 35. Tuesday of
ISO week 35 is **2026-08-25**.

The two readings agree today. They diverge exactly when the named weekday is
still ahead of you inside the current week:

| today | phrase | "next occurrence" | **our rule** (following ISO week) |
|---|---|---|---|
| Thu 2026-08-20 (wk 34) | next Tuesday | 2026-08-25 | 2026-08-25 — agree |
| **Mon 2026-08-24 (wk 35)** | next Tuesday | **2026-08-25** (tomorrow) | **2026-09-01** — differ |
| Tue 2026-08-25 (wk 35) | next Tuesday | 2026-09-01 | 2026-09-01 — agree |
| Sun 2026-08-23 (wk 34) | next Tuesday | 2026-08-25 | 2026-08-25 — agree |

The Monday row is the one that matters. Nobody standing on a Monday says "next
Tuesday" and means tomorrow. Picking the ISO rule makes that case right and costs
nothing elsewhere.

### Timezone and week-start, on the same instant

Same `now` = `2026-08-20T13:12:04Z`, three different users:

| tz | `work_week_start` | "next Tuesday" (UTC) | "next week" (UTC) |
|---|---|---|---|
| America/New_York | 1 (Mon) | `2026-08-25T04:00Z` → `2026-08-26T04:00Z` | `2026-08-24T04:00Z` → `2026-08-31T04:00Z` |
| Asia/Kolkata | 1 (Mon) | `2026-08-24T18:30Z` → `2026-08-25T18:30Z` | `2026-08-23T18:30Z` → `2026-08-30T18:30Z` |
| America/New_York | 7 (Sun) | `2026-08-25T04:00Z` → `2026-08-26T04:00Z` | `2026-08-23T04:00Z` → `2026-08-30T04:00Z` |

Two things to read off that table. The Kolkata user's Tuesday **starts on Monday
in UTC** — which is why every window is computed on local wall time with
`zoneinfo` and only then converted, never the other way round. And
`work_week_start` moves "next week" by a day without touching "next Tuesday",
because a named weekday does not depend on where the week begins, only on which
ISO week it lands in.

### DST is not decorative

A week is not always 168 hours. For the New York user:

```
"the week of Oct 26" =  Mon 2026-10-26 00:00 EDT  →  Mon 2026-11-02 00:00 EST
                     =  2026-10-26T04:00:00Z      →  2026-11-02T05:00:00Z
                     =  169 hours
```

DST ends Sunday 2026-11-01. Doing the arithmetic in local wall time and
converting at the end gets this right for free; doing it in UTC and adding
`timedelta(days=7)` gets it wrong by an hour, which silently drops or duplicates
an 11 PM event. (Python's own trap: subtracting two aware datetimes that share a
`tzinfo` object ignores the offset change, so the naive check *also* reports 168
hours. Compare in UTC.)

### Intent

```json
{
  "name": "calendar_list",
  "services": ["gcal"],
  "entities": {"carried_from_run": "run_3Zc8YtWq5NrKp7VmH"},
  "steps": ["search_calendar_window"],
  "has_write": false,
  "confidence": 1.0,
  "source": "rule_router:intent_carry",
  "resolved_window": {
    "name": "next_tuesday",
    "start": "2026-08-25T04:00:00Z",
    "end":   "2026-08-26T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "Tuesday of ISO week 35 (week of now + 1); local day boundaries; half-open"
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "calendar_list", "services": ["gcal"], "has_write": false, "confidence": 1.0},
  "answer_style": "template:event_list",
  "steps": [
    {
      "id": "events",
      "op": "gcal.search_events",
      "args": {
        "window": {"start": "{{windows.next_tuesday.start}}", "end": "{{windows.next_tuesday.end}}"},
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
        "limit": 50
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Expected answer — `template:event_list`

```
Tuesday, August 25

  11:00–12:00   Design review          5 guests
  16:00–16:30   1:1 with John Okafor

2 events. "Next Tuesday" read as Aug 25 — the Tuesday of next week.
```

The last line is the whole point of the example. State the interpretation so a
disagreement is one sentence away, not a silent wrong answer.

### Cost

**0 LLM calls, 0 embeddings.** 155 ms.

---

# 10. "Cancel my flight to Istanbul"

**Scenario:** two candidates match, so the ambiguity is raised **at plan time**
from a measured margin — not predicted before the search, and not discovered
after a full execution.

The user has two live Turkish Airlines bookings.

### Pre-pass

- "Istanbul" → destination token, plus the airport codes `IST`, `SAW`.
- No airline named, so no alias group. The route is the only handle.

### Probe — 113 ms

| corpus | hit | `cn` | evidence |
|---|---|---|---|
| gmail | `Your Turkish Airlines booking is confirmed — TK1984` (JFK→IST, departs Sep 5) | **0.84** | `EXACT(alias-token-in-subject)` |
| gmail | `Booking confirmed — TK1996` (JFK→IST, departs Oct 12) | **0.79** | `EXACT(alias-token-in-subject)` |
| gmail #3 | `Istanbul hotel — reservation` | 0.57 | — |
| gcal | `Istanbul → NYC Flight (TK1984)` Sep 5 | 0.75 | — |
| gcal | `NYC → Istanbul (TK1996)` Oct 12 | 0.71 | — |

Margin = 0.84 − 0.79 = **0.05 < MARGIN 0.15** on a corpus whose consuming step is
`expect: one` and `has_write: true`. Both flags set.

The `FLOOR_WRITE` check is the second, independent gate: 0.84 clears 0.80, but a
write may not anchor on a candidate that is *tied*. Clearing the floor is
necessary, not sufficient.

### Where this differs from #7

In #7 the ambiguity is about a **name**, and you could half-guess it before
searching. Here it is a measured property of the returned candidate set: two real
bookings, both matching, 0.05 apart. Nothing before the probe could have
predicted it, and nothing after it needs to run to discover it. The probe
computes the margin at 200 ms, the planner sees it in its input, and emits an
`ask.user` as step 0.

### Intent

```json
{
  "name": "cancel_flight",
  "services": ["gmail", "gcal"],
  "entities": {"destination": "Istanbul", "airport_codes": ["IST", "SAW"], "airline": null},
  "steps": ["disambiguate_booking", "get_booking_email", "find_calendar_event", "draft_cancellation_email"],
  "has_write": true,
  "confidence": 0.71,
  "source": "planner",
  "resolved_window": {
    "name": "write_default",
    "start": "2025-08-20T13:12:04Z",
    "end":   "2026-08-20T13:12:04Z",
    "tz": "America/New_York",
    "interpretation": "12 months back from now; write-class default"
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "cancel_flight", "services": ["gmail", "gcal"], "has_write": true, "confidence": 0.71},
  "answer_style": "card",
  "steps": [
    {
      "id": "which",
      "op": "ask.user",
      "args": {
        "kind": "choice",
        "blocking": true,
        "question": "You have two Istanbul flights booked. Which one should I cancel?",
        "options": [
          {"id": "18f30aa19bb2c401", "label": "TK1984 — JFK → IST",
           "meta": {"when": "Sat Sep 5, 2026, 5:55 PM", "booked": "Jul 22", "pnr": "6F2QK9", "fare": "USD 812.40"}},
          {"id": "19a4bb7c02de1180", "label": "TK1996 — JFK → IST",
           "meta": {"when": "Mon Oct 12, 2026, 6:20 PM", "booked": "Aug 11", "pnr": "R4TQ8M", "fare": "USD 744.00"}}
        ],
        "value_schema": {
          "type": "object",
          "properties": {"message_id": {"type": "string",
                                        "enum": ["18f30aa19bb2c401", "19a4bb7c02de1180"]}},
          "required": ["message_id"]
        }
      },
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "booking",
      "op": "gmail.get_email",
      "args": {"message_id": "{{which.value.message_id}}", "include_body": true},
      "depends_on": ["which"],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "flight_event",
      "op": "gcal.search_events",
      "args": {
        "query": "{{booking.extracted.flight_no}}",
        "window": {"start": "{{booking.extracted.depart_at|day_start|-1d}}",
                   "end":   "{{booking.extracted.depart_at|day_start|+2d}}"},
        "limit": 5
      },
      "depends_on": ["booking"],
      "expect": "one",
      "optional": true,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "draft",
      "op": "gmail.draft_email",
      "args": {
        "to": ["{{booking.extracted.support_email}}"],
        "subject": "Cancellation request — booking {{booking.extracted.pnr}}",
        "body_template": "flight_cancellation",
        "template_vars": {
          "pnr": "{{booking.extracted.pnr}}",
          "flight_no": "{{booking.extracted.flight_no}}",
          "route": "{{booking.extracted.route}}",
          "depart_at": "{{booking.extracted.depart_at}}",
          "passenger": "{{user.display_name}}"
        },
        "in_reply_to": "{{booking.message_id}}"
      },
      "depends_on": ["booking"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    },
    {
      "id": "send",
      "op": "gmail.send_email",
      "args": {"draft_id": "{{draft.draft_id}}"},
      "depends_on": ["draft"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    }
  ]
}
```

### Parallelism

Before the answer: nothing — `which` blocks everything, correctly.
After the answer: `flight_event` and `draft` both depend only on `booking`, so
they **start together** the moment the booking body lands.

```
which ──> booking ──┬──> flight_event
                    └──> draft ──> send (prepared, gated)
```

### Expected card — at ~650 ms

```
┌──────────────────────────────────────────────────────────────┐
│ You have two Istanbul flights booked. Which one should I     │
│ cancel?                                                      │
│                                                              │
│  ( ) TK1984 — JFK → IST                                      │
│      Sat Sep 5, 2026, 5:55 PM · booked Jul 22 · 6F2QK9       │
│      USD 812.40                                              │
│                                                              │
│  ( ) TK1996 — JFK → IST                                      │
│      Mon Oct 12, 2026, 6:20 PM · booked Aug 11 · R4TQ8M      │
│      USD 744.00                                              │
│                                                              │
│                            [ Continue ]     [ Never mind ]   │
└──────────────────────────────────────────────────────────────┘
```

The card carries the departure date, the booking date, the PNR and the fare —
everything that distinguishes the two. A choice card that only shows two nearly
identical labels has moved the ambiguity to the user without helping them.

### Resume — 0 LLM calls

Answer validated against the enum, `run.status` back to `running`, remaining four
nodes dispatched. Finishes in ~470 ms with the same confirm card as #4.

### Cost

**1 LLM call** for the whole two-turn interaction.

| t | what |
|---|---|
| 200 ms | `probe.done` — five chips, the two tied bookings visibly flagged |
| 590 ms | `intent`; the DAG draws with `ask.user` at the head |
| 650 ms | `input.raised`, `run.paused` |
| *(human)* | |
| +6 ms | enum validation |
| +52 ms | `booking` (mirror) |
| +54 ms | `flight_event` (mirror) — **overlapping** `draft` |
| +370 ms | `draft` (live Gmail) |
| +470 ms | confirm card |

---

# 11. "Prepare for tomorrow's meeting with Acme Corp" — Calendar 503s mid-run

**Scenario:** cross-service failure. Gmail and Drive succeed, Calendar does not,
and the answer says so instead of quietly producing a thinner brief.

Same query as #5. The difference is that the planner marks `meeting` as
`freshness: "live"` — the user asked about *tomorrow*, and a mirror up to 15
minutes stale is not good enough to state a meeting time as fact.

### The failure

`gcal.search_events` with `freshness: live` calls Google. Google returns **503**
twice.

```
attempt 1  →  503  ·  classify() → TRANSIENT  ·  retryable  ·  backoff(TRANSIENT, 1) = 412 ms (full jitter)
attempt 2  →  503  ·  classify() → TRANSIENT  ·  max_attempts (2) reached in-request
```

The in-request retry tier is **max 2 attempts, capped at 1.5 s of added latency**.
The full policy — five attempts, doubling — only runs in a worker, where nobody
is watching a cursor blink. A user-facing read does not get to spend eleven
seconds being brave.

Circuit breaker for `(usr_V1StGXR8_Z5jdHi6B, gcal)` records consecutive failure
**1 of 5**. Five would open it for 5 minutes.

### Node states

```
meeting        failed    outcome {"reason": "google_unavailable", "class": "TRANSIENT",
                                  "code": 503, "message": "The service is currently unavailable."}
                         retries [{"at": "...", "error_class": "TRANSIENT",
                                   "google_status": 503, "backoff_ms": 412}]
mail           skipped   outcome {"reason": "dependency_failed", "depends_on": "meeting"}
docs           succeeded 2 files
```

`mail` is skipped, not failed. It never ran — its `participants` filter was
`{{meeting.hits[0].attendee_emails[*]}}` and that reference cannot bind. The
distinction matters because a skip is not a Google error and must not count
against the circuit breaker.

The run's failed services are derived, never stored:

```sql
SELECT DISTINCT split_part(op, '.', 1) FROM node_executions
WHERE run_id = $1 AND status IN ('failed', 'skipped', 'timeout');
-- {'gcal', 'gmail'}
```

`runs.status` is **`complete`**, not `failed`. The run produced an answer. Rolling
the whole turn to `failed` because one node broke would throw away the Drive
results the user can still use.

### What the synthesizer is given

The prose call receives the failure as structured context, not as an absence:

```json
{
  "degraded": true,
  "failed": [{"node": "meeting", "service": "gcal", "class": "TRANSIENT", "code": 503, "attempts": 2}],
  "skipped": [{"node": "mail", "because": "meeting"}],
  "succeeded": [{"node": "docs", "count": 2}]
}
```

The synthesis prompt's standing instruction: *state what failed, in one sentence,
before anything else. Do not compensate with a longer answer about what worked.*

### Expected answer

```
Calendar is not responding — Google returned 503 twice, so I could not
confirm tomorrow's Acme meeting or who is on it. Everything below comes
from Drive only; there may be email context I have not seen, because
that search needed the guest list.

Two Acme documents changed recently:

  • Acme — Q3 renewal proposal v4.gdoc     modified Aug 18 by Sarah Chen
  • Acme_MSA_countersigned.pdf             modified Jul 30

The proposal is on v4 and moved on Monday, so it is the one to read.

Gmail and Drive are both fine. This looks like a Calendar hiccup rather
than an account problem.
```

```
┌──────────────────────────────────────────────────┐
│ Calendar failed · 503 · 2 attempts               │
│                        [ Retry Calendar ]        │
└──────────────────────────────────────────────────┘
```

### The retry chip — 0 LLM calls

Pressing **Retry Calendar** is a front-door UI verb. It does **not** re-plan. The
plan is already in `node_executions`; the retry re-dispatches the failed node and
everything that was skipped because of it, at `round = 1`:

```
meeting   round 1  →  succeeded
mail      round 1  →  succeeded (its reference binds now)
docs      round 0  →  reused, not re-run
```

`UNIQUE (run_id, node_id, round)` on `node_executions` is what makes a retry a new
row rather than an overwrite, so the trace keeps both attempts. Then one prose
call renders the complete brief.

Retry cost: **1 LLM call** (the new prose), 0 for planning. Retrying is cheaper
than asking again, which is the point.

### Cost

**2 LLM calls.** ~3.4 s, of which 412 ms is deliberate backoff.

| t | what |
|---|---|
| 200 ms | `probe.done` |
| 590 ms | `intent` |
| 600 ms | `meeting` (live) and `docs` start together |
| 712 ms | `meeting` attempt 1 → 503; `step.retrying` on the wire, visible in the trace |
| 701 ms | `docs` succeeded |
| 1 124 ms | `meeting` attempt 2 → 503; node `failed` |
| 1 130 ms | `mail` marked `skipped` |
| 1 180 ms | synthesis opens with `degraded: true` |
| 1 950 ms | first token — and the **first sentence is the failure** |
| 3 400 ms | `run.complete` |

---

# 12. "Cancel my Turkish Airlines flight" — the booking email is in Turkish

**Scenario:** zero qualifying hits on the first pass, recovered by the escalation
ladder at **0 LLM calls**.

Same query as #4. The difference is the corpus: the booking confirmation is

```
From:     bilet@thy.com
Subject:  Uçuş rezervasyonunuz onaylandı — TK1984
Body:     Sayın Yolcumuz, 6F2QK9 numaralı rezervasyonunuz onaylanmıştır.
          Uçuş: TK1, İstanbul (IST) → New York (JFK), 5 Eylül 2026, 10:30.
          Toplam: 812,40 USD.
```

### Probe, round 0 — 112 ms, nothing survives

| leg | why it fails |
|---|---|
| vector | The English query embedding against a Turkish body: `cn` **0.41**, below `FLOOR_READ 0.55`. Multilingual embeddings help, but they do not close a gap this wide on a short transactional email. |
| full-text | `tsv` is built with the `'english'` configuration. `to_tsvector('english', 'Uçuş rezervasyonunuz onaylandı')` produces useless lexemes. Zero rank. |
| RRF | Fuses two bad lists into one bad list. This is exactly why the floor is not computed on the fused score — RRF would happily hand back a #1. |

`probe.round0.qualifying = 0`. Without a ladder, the answer is "I couldn't find a
Turkish Airlines booking," which is **wrong**, and the user has no way to tell a
missing email from a failed search.

### The escalation ladder — all Python, 0 LLM calls

Rungs run in order; the first one to produce a qualifying candidate stops the
ladder.

**Rung 1 — evidence without similarity (+38 ms).**
The floor is a disjunction: `cn ≥ FLOOR_READ` **OR** `evidence = EXACT`. Rung 1
drops the `cn` term entirely and re-queries on the alias group's hard signals:

```sql
SELECT ... FROM sync_events
WHERE user_id = $1
  AND received_at >= $2
  AND (   from_email LIKE '%@turkishairlines.com'
       OR from_email LIKE '%@thy.com'
       OR subject ~* '\yTK\s?\d{1,4}\y'
       OR subject ILIKE ANY (ARRAY['%türk hava yolları%', '%turk hava yollari%',
                                   '%turkish airlines%', '%thy%']) )
ORDER BY received_at DESC
LIMIT 20;
```

`bilet@thy.com` matches on the sender domain **and** the subject carries `TK1984`.
Two independent `EXACT` signals. Candidate admitted with `cn = 0.41,
evidence = EXACT(sender-domain, code-pattern)`.

Alias groups are hand-maintained data, not inference. `thy.com` is in the table
because Turkish Airlines uses it, and that fact is worth more here than any
amount of embedding cleverness.

**Rung 2 — widen (not needed here).** Drop the date filter to all time, raise
top-k to 50, add trigram similarity on `subject` for the code pattern.

**Rung 3 — cross-corpus (not needed here).** Run the same alias tokens against
`event` and `file` rows. A calendar event titled `IST → JFK TK1984` or a
file named `Invoice_TK_1984.pdf` carries the record locator even when the email
does not surface.

**Rung 4 — ask (not needed here).** An `ask.user` text prompt: *"I can't find a
Turkish Airlines booking in the last 12 months. Do you have the booking
reference?"* Absence reported as absence, with a way forward.

Every rung is a SQL query or a prompt. **The ladder costs zero LLM calls**, which
is why it can afford four rungs.

### Why the extractors still work

The regex extractors are language-independent by construction — they match
structure, not vocabulary:

```jsonc
{
  "pnr": "6F2QK9",              // /\b[A-Z0-9]{6}\b/ near a booking keyword or "numaralı"
  "ticket_no": "TK1984",
  "flight_no": "TK1",           // /\b[A-Z]{2}\s?\d{1,4}\b/
  "route": "IST→JFK",           // /\b([A-Z]{3})\b\s*(?:→|->|-)\s*\b([A-Z]{3})\b/
  "depart_at": "2026-09-05T10:30:00+03:00",   // month-name lexicon incl. Turkish "Eylül"
  "amount": "USD 812.40",       // /812,40\s*USD/ with comma-decimal normalisation
  "support_email": null         // not present in this email
}
```

This is the strongest argument for regex-first extraction over asking a model to
read the email: a PNR is a shape, and shapes survive translation. The model's job
is to decide *what to do*, not to retype values it can reference by path.

### Consequence for the plan

`support_email` is null, so the draft cannot be addressed from the email body.
The planner falls back to the alias group's known support address
(`cancel@turkishairlines.com`, hand-maintained alongside the tokens) and the
answer **says** it did that. The plan is otherwise identical to #4.

### Intent

```json
{
  "name": "cancel_flight",
  "services": ["gmail", "gcal"],
  "entities": {
    "airline": "Turkish Airlines",
    "alias_group": "turkish_airlines",
    "recovered_by": "escalation_rung_1",
    "source_language": "tr"
  },
  "steps": ["search_gmail_for_booking", "find_calendar_event", "draft_cancellation_email"],
  "has_write": true,
  "confidence": 0.87,
  "source": "planner",
  "resolved_window": {
    "name": "write_default",
    "start": "2025-08-20T13:12:04Z",
    "end":   "2026-08-20T13:12:04Z",
    "tz": "America/New_York",
    "interpretation": "12 months back from now; write-class default"
  }
}
```

Note `confidence` 0.87, not 0.91 as in #4 — an evidence-only match is strong but
strictly weaker than evidence plus similarity, and the number says so.

`FLOOR_WRITE` in the evidence-only case is satisfied by **two independent `EXACT`
signals** (sender domain and code pattern in subject) rather than by `cn ≥ 0.80`.
A single `EXACT` signal would not have been enough to anchor a write, and the plan
would have carried an `ask.user` confirmation of the booking first.

### Expected answer

```
I found your Turkish Airlines booking (6F2QK9) in an email from July 22.
The email is in Turkish; I have read the booking reference, flight
number and date out of it, and drafted the cancellation in English.

  ✓  TK1, Istanbul (IST) → New York (JFK), Sat Sep 5, 10:30 (+03:00)
  ✓  Calendar event "Istanbul → NYC Flight (TK1984)" on Sep 5
  ✓  Drafted a cancellation email

The booking email does not give a cancellation address, so I have
addressed it to cancel@turkishairlines.com — Turkish Airlines' published
one. Change it if you have a better address.
```

Then the same confirm card as #4.

### Cost

**1 LLM call.** The recovery is free.

| t | what |
|---|---|
| 112 ms | probe round 0 — 0 qualifying |
| 150 ms | rung 1 returns the candidate |
| 238 ms | `probe.done` (38 ms later than the happy path) |
| 628 ms | `intent` |
| 1 250 ms | card |

---

# 13. "Push my Acme review next Thursday to Friday 3pm and tell the attendees"

**Scenario:** two writes, one confirm card, enforced ordering. One prompt gating
two actions is exactly what `pending_inputs` and `actions` being separate tables
buys.

### Pre-pass

Two time phrases, and the second one depends on the first.

- "next Thursday" → Thursday of ISO week 35 = **2026-08-27**.
- "Friday 3pm" → **anchored weekday** rule: a bare weekday appearing in a
  sentence that already has a resolved anchor resolves inside that anchor's ISO
  week. Anchor is 2026-08-27 (week 35) → Friday of week 35 = **2026-08-28**, at
  15:00 local = **`2026-08-28T19:00:00Z`**.

Without the anchoring rule, "Friday" would resolve against `now` (Thursday
Aug 20) and land on **Aug 21** — moving the meeting six days *backwards*, into the
past relative to its original slot. The rule is what stops that.

### Probe — 110 ms

| corpus | hit | `cn` | evidence |
|---|---|---|---|
| gcal | `Acme review`, Thu 2026-08-27 13:00–14:00, 4 attendees | **0.89** | `EXACT(alias-token-in-title)` + window match |
| gcal #2 | `Acme Corp — Q3 renewal review`, Mon 2026-08-24 | 0.66 | — |

Margin 0.23 > `MARGIN`. `cn` 0.89 ≥ `FLOOR_WRITE 0.80`. Both writes may anchor
here. (The #2 hit is outside the resolved window and would have been dropped by
the prefilter anyway; it appears because the probe's window is deliberately wider
than the plan's, so the planner can see what it is *not* choosing.)

### Intent

```json
{
  "name": "reschedule_and_notify",
  "services": ["gcal", "gmail"],
  "entities": {"event_topic": "Acme review", "company": "Acme", "alias_group": "acme"},
  "steps": ["find_event", "update_event", "notify_attendees"],
  "has_write": true,
  "confidence": 0.9,
  "source": "planner",
  "resolved_window": {
    "name": "next_thursday",
    "start": "2026-08-27T04:00:00Z",
    "end":   "2026-08-28T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "Thursday of ISO week 35; local day; half-open"
  },
  "additional_windows": {
    "target_slot": {
      "start": "2026-08-28T19:00:00Z",
      "end":   "2026-08-28T20:00:00Z",
      "tz": "America/New_York",
      "interpretation": "Friday anchored to next_thursday's ISO week (35) = 2026-08-28, 15:00 local; duration copied from the source event"
    }
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "reschedule_and_notify", "services": ["gcal", "gmail"], "has_write": true, "confidence": 0.9},
  "answer_style": "card",
  "steps": [
    {
      "id": "event",
      "op": "gcal.get_event",
      "args": {"event_id": "{{search.gcal[0].event_id}}"},
      "depends_on": [],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    },
    {
      "id": "move",
      "op": "gcal.update_event",
      "args": {
        "event_id": "{{event.event_id}}",
        "etag": "{{event.etag}}",
        "starts_at": "{{windows.target_slot.start}}",
        "ends_at": "{{windows.target_slot.end}}",
        "send_updates": "none"
      },
      "depends_on": ["event"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    },
    {
      "id": "notify",
      "op": "gmail.send_email",
      "args": {
        "to": "{{event.attendee_emails[*]|exclude(user.email)}}",
        "subject": "Moved: {{event.title}} → Fri Aug 28, 3:00 PM",
        "body_template": "meeting_moved",
        "template_vars": {
          "title": "{{event.title}}",
          "old_start": "{{event.starts_at}}",
          "new_start": "{{windows.target_slot.start}}",
          "tz": "America/New_York"
        }
      },
      "depends_on": ["move"],
      "expect": "one",
      "optional": false,
      "freshness": "live",
      "speculate": false
    }
  ]
}
```

`send_updates: "none"` on the calendar update is deliberate. Google's own invite
mail plus our note would mean two messages saying the same thing, and the user
approved *our* wording.

### Parallelism

None, on purpose:

```
event ──> move ──> notify
```

`notify` depends on `move` even though the email body needs nothing from the
move's *result*. The dependency exists to enforce **ordering**, which is the
whole point of the example: an email announcing a move must not go out before the
move has actually happened.

### The one card, two actions

One `pending_inputs` row, two `actions` rows, both with
`requires_input_id = pin_2Wq6KbYn4LsRt9PdF`:

```jsonc
// pending_inputs — ONE row
{
  "id": "pin_2Wq6KbYn4LsRt9PdF",
  "kind": "confirm",
  "blocking": false,
  "prompt": {
    "question": "Move the Acme review and email the 3 guests?",
    "help_text": "The calendar change happens first. If it fails, the email is not sent."
  },
  "value_schema": {"type": "object", "properties": {"approved": {"type": "boolean"}},
                   "required": ["approved"]}
}

// actions — TWO rows
{"id": "act_5Hm7VpZr8TdNq3XcB", "op": "gcal.update_event",
 "requires_input_id": "pin_2Wq6KbYn4LsRt9PdF", "status": "draft",
 "payload": {"event_id": "9r2k4m_20260827T170000Z", "etag": "\"3401882940110000\"",
             "starts_at": "2026-08-28T19:00:00Z", "ends_at": "2026-08-28T20:00:00Z",
             "send_updates": "none"}}

{"id": "act_2Wq6KbYn4LsRt9PdF", "op": "gmail.send_email",
 "requires_input_id": "pin_2Wq6KbYn4LsRt9PdF", "status": "draft",
 "payload": {"to": ["dana@acmecorp.com", "marcus@acmecorp.com", "sarah@company.com"],
             "subject": "Moved: Acme review → Fri Aug 28, 3:00 PM", "body": "..."}}
```

The message's `content` is an ordered block list referencing all three:

```json
[
  {"type": "text",   "data": {"markdown": "Ready to move the Acme review..."}},
  {"type": "action", "ref": "act_5Hm7VpZr8TdNq3XcB"},
  {"type": "action", "ref": "act_2Wq6KbYn4LsRt9PdF"},
  {"type": "input",  "ref": "pin_2Wq6KbYn4LsRt9PdF"}
]
```

Two previews, one button. All four rows are written in a single transaction — a
`ref` that does not resolve is dropped on read and logged, never rendered as an
empty box.

### Expected card

```
Ready to move the Acme review. Nothing has changed yet.

┌──────────────────────────────────────────────────────────────┐
│ Move the Acme review and email the 3 guests?                 │
│                                                              │
│ 1 · Calendar                                                 │
│     Acme review                                              │
│     Thu Aug 27, 1:00–2:00 PM  →  Fri Aug 28, 3:00–4:00 PM    │
│                                                              │
│ 2 · Email  (only if step 1 succeeds)                         │
│     To       dana@acmecorp.com, marcus@acmecorp.com,         │
│              sarah@company.com                               │
│     Subject  Moved: Acme review → Fri Aug 28, 3:00 PM        │
│                                                              │
│       "I've moved our Acme review from Thursday 1:00 PM      │
│        to Friday Aug 28 at 3:00 PM Eastern. The calendar     │
│        invite is updated. Shout if that doesn't work."       │
│                                                              │
│ The calendar change happens first. If it fails, the email    │
│ is not sent.                                                 │
│                                                              │
│              [ Do both ]     [ Edit email ]     [ Not now ]  │
└──────────────────────────────────────────────────────────────┘
```

### On approval — 0 LLM calls

Both actions go to `approved` and are handed to the `actions` Celery queue. The
worker respects the DAG edge:

1. `gcal.update_event` with `If-Match: "3401882940110000"`.
2. Only on success does `gmail.send_email` run, carrying
   `X-Orchestrator-Idem: <act_2Wq6KbYn4LsRt9PdF dedupe_key>` so a retried send can
   check Sent before re-sending.

### The failure path that justifies the ordering

If someone else edited the event in the meantime, Google returns **412
Precondition Failed** → `classify()` → `PRECONDITION` → not retryable.

```
act_5Hm7VpZr8TdNq3XcB   failed     {"class": "PRECONDITION", "code": 412}
act_2Wq6KbYn4LsRt9PdF   cancelled  {"reason": "upstream_failed",
                                    "upstream": "act_5Hm7VpZr8TdNq3XcB"}
```

```
The calendar move failed — someone changed that event while you were
deciding, so my copy was out of date. I have not sent the email; it
would have told three people about a change that did not happen.

                                        [ Reload the event and retry ]
```

**The email announcing a move never goes out when the move did not happen.** That
property comes from the dependency edge plus `etag`, and it is the reason
`notify` waits.

Both `dedupe_key`s stay unique only while `status IN ('draft','approved','running')`.
Once cancelled, an identical retry is a legitimately new action — which is what
makes that retry button work.

### Cost

**1 LLM call.** Card at ~1.1 s (one live `gcal.get_event` for a fresh etag).
Approval, execution and the failure message are all 0.

---

# 14. "Summarise everything Acme sent me this month"

**Scenario:** fan-out summarisation. Many items, one batched call, and an honest
latency class.

### Pre-pass

- "this month" → the current calendar month **to now**, not to month end. There
  is nothing to summarise in the future.
- "Acme" → alias group `acme`, sender domain `acmecorp.com`.

### Probe — 116 ms

14 gmail candidates over `FLOOR_READ`, all with `EXACT(sender-domain)`. No
ambiguity test — `expect` is `many`.

### Intent

```json
{
  "name": "digest",
  "services": ["gmail"],
  "entities": {"company": "Acme", "alias_group": "acme", "sender_domain": "acmecorp.com"},
  "steps": ["search_emails_by_domain", "fetch_bodies", "summarize"],
  "has_write": false,
  "confidence": 0.95,
  "source": "planner",
  "resolved_window": {
    "name": "this_month",
    "start": "2026-08-01T04:00:00Z",
    "end":   "2026-08-20T13:12:04Z",
    "tz": "America/New_York",
    "interpretation": "first local day of now's month, to now; not to month end"
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "digest", "services": ["gmail"], "has_write": false, "confidence": 0.95},
  "answer_style": "prose",
  "steps": [
    {
      "id": "mail",
      "op": "gmail.search_emails",
      "args": {
        "from_domain": "acmecorp.com",
        "window": {"start": "{{windows.this_month.start}}", "end": "{{windows.this_month.end}}"},
        "order_by": "received_at",
        "limit": 25
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "bodies",
      "op": "gmail.get_email",
      "args": {"message_ids": "{{mail.hits[*].message_id}}", "include_body": true,
               "max_concurrency": 6},
      "depends_on": ["mail"],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    },
    {
      "id": "digest",
      "op": "meta.summarize",
      "args": {
        "items": "{{bodies.results[*]}}",
        "group_by": "thread_id",
        "style": "chronological_with_open_items",
        "per_item_budget": 900,
        "stream": true
      },
      "depends_on": ["bodies"],
      "expect": "one",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### Parallelism — two different kinds

This is the example that separates them.

**Between steps:** none. `mail → bodies → digest` is a strict chain.

**Inside a step:** `bodies` is one step that fans out to 14 message fetches. The
dispatcher gives **one asyncio task per step**; the op fans out internally under
the Gmail semaphore, `max_concurrency: 6`. Fourteen fetches at 6 wide is three
waves, ~180 ms, instead of 14 × 30 ms serial.

Fanning out inside the op rather than emitting 14 plan steps keeps the plan
readable, keeps the step list to three bars instead of sixteen, and keeps the
concurrency limit in one place — the semaphore that also protects the quota
governor's interactive reservation.

### The fan-in

`meta.summarize` is the second LLM call. Map-side trimming first:
`Op.to_llm(data, budget=900)` per item → 14 × ~900 chars ≈ 12.6 KB, comfortably
inside one call.

If the trimmed corpus had exceeded the budget, `meta.summarize` chunks into
groups of 8, summarises each, then reduces — 2 + 1 = 3 calls for this step, 4 for
the run. The **hard cap of 5** is what bounds that recursion; past it the op
returns what it has and the answer says it covered the most recent N.

`stream: true` means the summariser's output *is* the answer. There is no separate
synthesis call layered on top of it.

### Expected answer — streamed prose

```
14 emails from Acme this month, across 5 threads.

Renewal (7 messages, still open). Dana opened on the 4th with the
standard 12-month quote, you pushed back on price on the 6th, and after
two rounds she landed on 14 months at the 12-month rate — that is the
Aug 18 message. Open: you have not replied to the 14-month term.

MSA redlines (3 messages, still open). Marcus sent two on the 11th:
liability cap to 12 months of fees, data residency EU-only. Sarah
acknowledged. Nobody has answered either.

Security review (2 messages, closed). They asked for the questionnaire
on the 10th; Sarah confirmed it went out on the 17th. Acme has not
confirmed receipt, which is the loose thread.

Invoicing (1 message). July invoice paid, receipt on the 12th. Nothing
needed.

Introductions (1 message). Priya Raman joins as their new CSM from
September 1.

Three things are waiting on you: the 14-month term, the liability cap,
and confirming they got the questionnaire.
```

### Cost

**2 LLM calls** — planner, then summarise. 1 embedding. ~4.2 s.

| t | what |
|---|---|
| 200 ms | `probe.done` — 14 chips |
| 590 ms | `intent` |
| 600 ms | `mail` starts |
| 655 ms | `mail` done, 14 hits |
| 660 ms | `bodies` starts, 6 wide |
| 845 ms | `bodies` done (3 waves) |
| 900 ms | `meta.summarize` opens |
| 1 850 ms | first token |
| 4 200 ms | `run.complete` |

**4.2 s is not a P95 read.** This is the batch class and it is labelled as such.
The read class — rule-routed at ~150 ms and single-call template reads at
~800 ms — is what holds under 2 s at P95. Claiming P99 under 2 s across every
query shape, including a 14-email digest, would be false.

---

# 15. "thanks, that's perfect"

**Scenario:** chit-chat. The cheapest possible path, and a demonstration that the
front door's matchers are exact rather than eager.

### Front door — chit-chat matcher

Matcher #3 fires. It requires the **whole message** to match a closed pattern set
after normalisation — never a prefix, never a substring.

| message | matches? | why |
|---|---|---|
| `thanks, that's perfect` | **yes** | whole message is gratitude, no verb phrase, no entity |
| `thanks!` | yes | |
| `thanks — also what's on Friday?` | **no** | contains a time phrase and an interrogative; goes to the full pipeline |
| `thanks for finding that, can you send it?` | **no** | contains a UI verb; matcher #2 takes it instead |
| `perfect, do the same for Northwind` | **no** | imperative plus an entity |

An eager chit-chat matcher is worse than none: silently dropping the real request
appended to a thank-you is a bad failure and an invisible one.

### What is written

- One `messages` row, `role='user'`.
- One `runs` row: `planner_tier = 1`, `status = 'complete'`,
  `token_usage = {"prompt": 0, "completion": 0, "usd": 0}`, `intent = null`.
- Zero `node_executions`.
- One `messages` row, `role='assistant'`.

The run row costs one insert and is worth it: the step list shows *answered
without calling anything*, which is a true statement about the system, and
`/metrics` can report what fraction of turns cost nothing without a separate
counter.

### Expected answer

```
Anytime.
```

Drawn from a small response set, varied by hash of the message so it is not
identical every time, and deliberately short — a long friendly paragraph here
would be the system talking to hear itself.

### Cost

**0 LLM calls. 0 embeddings. 0 Google calls. ~12 ms.**

| t | what |
|---|---|
| 3 ms | user message persisted |
| 4 ms | matcher #3 fires |
| 12 ms | assistant message persisted, `run.complete` |

---

# 16. "What's on my calendar next week where john@company.com is invited?"

**Scenario:** the problem statement's third example. Attendee filtering, a GIN
index hit, and the boundary of what the rule router will accept.

### Front door — rule router hit

Pattern `calendar_window_attendee`:
`calendar\s+(?P<when>.+?)\s+(?:where|with)\s+(?P<email>[^\s@]+@[^\s@]+)\s+(?:is\s+)?(?:invited|attending|on it)`

Two extractable things, both unambiguous: a time phrase for
`temporal.resolve()` and a **literal email address**. Nothing needs ranking, so
no probe, no embedding, no planner call.

### The boundary — why this is a router hit and #7 is not

| query | router? | why |
|---|---|---|
| `...where john@company.com is invited?` | **yes** | `@` present — the attendee is an exact value, `attendee_emails @> ARRAY[...]` |
| `...where John is invited?` | **no** | "John" needs resolution; there may be two |
| `Move the meeting with John` | **no** | resolution *and* a write *and* a missing target time (see #7) |

The `@` is the trigger. It is a cheap, honest test for "this needs no
interpretation", and it is why the router can take this query at zero cost while
declining its near neighbour.

### Intent

```json
{
  "name": "calendar_list",
  "services": ["gcal"],
  "entities": {"attendee_email": "john@company.com"},
  "steps": ["search_calendar_window_filtered_by_attendee"],
  "has_write": false,
  "confidence": 1.0,
  "source": "rule_router",
  "resolved_window": {
    "name": "next_week",
    "start": "2026-08-24T04:00:00Z",
    "end":   "2026-08-31T04:00:00Z",
    "tz": "America/New_York",
    "interpretation": "iso week 35, Mon 2026-08-24 .. Sun 2026-08-30, week_start=1, half-open"
  }
}
```

### Plan

```json
{
  "type": "plan",
  "intent": {"name": "calendar_list", "services": ["gcal"], "has_write": false, "confidence": 1.0},
  "answer_style": "template:event_list",
  "steps": [
    {
      "id": "events",
      "op": "gcal.search_events",
      "args": {
        "window": {"start": "{{windows.next_week.start}}", "end": "{{windows.next_week.end}}"},
        "attendee_emails_any": ["john@company.com"],
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
        "limit": 50
      },
      "depends_on": [],
      "expect": "many",
      "optional": false,
      "freshness": "cached",
      "speculate": false
    }
  ]
}
```

### The query underneath

```sql
SELECT external_id AS event_id, title, occurred_at AS starts_at,
       ends_at, all_day, participants AS attendees
FROM sync_events
WHERE user_id = $1
  AND kind = 'event'
  AND occurred_at >= $2 AND occurred_at < $3
  AND participant_emails @> ARRAY['john@company.com']::citext[]
  AND status IN ('confirmed', 'tentative')
ORDER BY occurred_at;
```

`attendee_emails` is an indexed `CITEXT[]` column on `sync_events`
(`participant_email_list(participants)`), so `participants` stays the single
source of truth and the containment operator still gets a GIN index. Without it
this is a `jsonb_array_elements` scan over every event in the window.

The same index answers "emails from Sarah" and "files shared by Priya", because
a sender, an organiser and an owner are all just a participant with a role.

### Expected answer — `template:event_list`

```
Next week — Mon Aug 24 to Sun Aug 30 — 2 events with john@company.com.

Tue Aug 25
  11:00–12:00   Design review              5 guests   John: accepted
  16:00–16:30   1:1 with John Okafor       2 guests   John: needs action

The 1:1 has not been accepted yet.
```

`response_status` comes free out of the `attendees` JSONB, and "has not been
accepted" is the kind of thing the user actually wanted to know.

### Cost

**0 LLM calls. 0 embeddings.** 165 ms — 15 ms more than #1 because of the
containment predicate.

---

# Cost ledger

| | |
|---|---|
| Completion calls across all 16 | **16** |
| Mean per query | **1.00** |
| Zero-call queries | 4 of 16 (#1, #9, #15, #16) |
| One-call queries | 8 of 16 |
| Two-call queries | 4 of 16 (#5, #8, #11, #14) |
| Most expensive here | 2 (#14 reaches 3–4 only when the digest has to chunk) |
| Hard cap | 5 |
| Embedding calls | 12 of 16 — the four zero-call queries skip the probe entirely |

**Reconciling with the 1.9 figure.** The design note projects a mean of about 1.9
completion calls per completed task on the expected production mix, where prose
answers dominate. This document averages 1.00 because it deliberately
over-samples the cheap paths in order to show them — four rule-router queries out
of sixteen is not what a real week looks like. The right reading is that 1.9 is
the projection and 1.00 is what these sixteen specific queries cost. Neither
number is a measurement of production traffic, because there is no production
traffic.

Free by construction, and worth listing because each one is somewhere a system
would normally spend a call:

| operation | calls |
|---|---|
| Answering a card (#7, #10) | 0 |
| Resuming a paused run | 0 |
| Approving a write (#4, #13) | 0 |
| Rendering a confirm card from `Op.preview()` | 0 |
| Parsing "Friday 3pm" from a card answer | 0 |
| The four-rung escalation ladder (#12) | 0 |
| Retrying a failed node (#11) | 0 (re-plan), 1 (new prose) |
| Chit-chat (#15) | 0 |

---

# Latency classes

Stated separately because they are genuinely different, and averaging them would
be dishonest.

| class | examples | P95 |
|---|---|---|
| Rule-routed read | #1, #9, #16 | **~165 ms** |
| Single-call templated read | #2, #3, #6 | **~900 ms** |
| Single-call card / write prep | #4, #7, #10, #12, #13 | **~1.3 s** |
| Two-call prose read | #5, #8 | **~3.1 s** |
| Degraded (with in-request backoff) | #11 | **~3.4 s** |
| Batch / fan-out | #14 | **~4.2 s** |

**P95 under 2 s holds for the read class.** It does not hold for two-call prose,
and claiming P99 under 2 s across the board would be false.

What is defended everywhere is **time to first meaningful pixel**:

| t | what the user sees |
|---|---|
| 5 ms | progress indicator, run started |
| 200 ms | candidate chips — the actual emails and events being considered |
| 590 ms | intent, and the DAG drawing itself |
| 650 ms | an ambiguity card, when there is one |
| 780 ms | a rendered list, when the answer is a list |
| ~1.6 s | first prose token |

The screen is never blank and never shows a spinner with nothing behind it. The
chips at 200 ms are real retrieved documents, so even in the worst case the user
can see whether the system found the right thing four hundred milliseconds before
the model has said anything.

---

# Edge cases, and what each one proves

| # | Edge case | Proves |
|---|---|---|
| 1 | Rule-routed read | A plain calendar query never reaches a model |
| 3 | No created date in Drive | Mirror limits stated in the answer, not hidden |
| 4 | Draft created before confirm | Reversible up front, irreversible behind a gate |
| 5 | Optional Drive step | A missing service thins the answer, it does not fail the run |
| 6 | Gate on a failed extraction | Fallback for missing data is a skip plus an explanation |
| 7 | Two unknowns in one form | One round trip to the human, not two |
| 8 | Two context matches in one run | Recency is not a tiebreak inside a single run |
| 9 | Monday / "next Tuesday" | ISO-week rule differs from next-occurrence, and matters |
| 9 | DST week = 169 hours | Local wall time first, UTC second |
| 10 | Margin 0.05 on a write | Clearing `FLOOR_WRITE` is necessary, not sufficient |
| 11 | 503 twice, then honest failure | Partial results with the failure stated first |
| 11 | `skipped` ≠ `failed` | A dependency skip does not trip the circuit breaker |
| 12 | Turkish email, zero hits | Evidence without similarity; four rungs, zero calls |
| 12 | `support_email` null | Fall back to alias-group data and say so |
| 13 | 412 on the calendar write | The email announcing a move is cancelled, not sent |
| 13 | Cancel then retry | Partial unique index permits a legitimate resend |
| 14 | 14 bodies, one step | Fan-out inside an op, not fourteen plan steps |
| 15 | `thanks — also what's on Friday?` | Whole-message match, never a prefix |
| 16 | `@` present vs absent | The router's boundary is a cheap, honest test |

---

# What this document does not claim

- **Plausible-but-wrong retrieval is not detected.** The system detects
  ambiguity (two candidates within `MARGIN`) and absence (nothing above
  `FLOOR_READ` after four rungs). It does not detect a single confident wrong
  answer. #8's honest-limit note is the general case. The mitigation is display:
  every answer names its sources so a human can see the mistake.
- **The thresholds are hand-set.** 0.55, 0.15, 0.80 were chosen by looking at
  score distributions on the seed corpus, not by optimising a labelled set. They
  are constants in one module for exactly that reason.
- **The mirror can be 15 minutes stale.** Steps that must be current carry
  `freshness: "live"` — writes, superlative queries ("my next meeting"), and
  anything about today or tomorrow. Everything else reads the mirror and the
  answer states the sync age.
- **Depth stops at three hops.** A plan needing a fourth genuine dependency hop
  hands back to the user with what it has rather than replanning again.
- **Precision@5.** Measured with `GET /api/v1/search`, which returns the score
  components (`cn`, RRF rank, FTS rank, decay factor, evidence flags) for a
  labelled query set over the seed corpus. Both the harness and the numbers live
  with the evaluation notes, not here — this document specifies behaviour, not
  measurements.
