# Demo Script — 5:00

Shot by shot. Eight segments, 300 seconds, no filler.

The thing being demonstrated is **orchestration**, not chat. Every segment shows
something a single API call cannot do: parallel steps, a real dependency, a pause
that costs nothing to resume, two writes ordered against each other, and a service
failing without the answer lying about it.

Read the pre-flight checklist at the bottom **before** you start recording. The
one that bites is the 7-day refresh token.

---

## Screen layout

Record at **1920×1080, 30 fps**. Browser at 1440×900 zoomed to 110%, centred, so
text is legible after compression.

```
┌──────────────────────────── browser ─────────────────────────────┐
│ ┌ chats ────────┐ ┌──────────────── conversation ──────────────┐ │
│ │ + New chat    │ │                   Cancel my Turkish flight │ │
│ │   Search      │ │                                            │ │
│ │               │ │  ●●● Working…                              │ │
│ │  Turkish …    │ │   │ ● Looking through your mail            │ │
│ │  Next week    │ │   │ ● Checking your calendar    running    │ │
│ │  Sarah budget │ │   │ ○ Updating the draft                   │ │
│ │               │ │                                            │ │
│ │               │ │  I found Türk Hava Yolları — TK1234 …      │ │
│ │               │ │  ┌ confirm ──────────────────────────────┐ │ │
│ │               │ │  │ Send this cancellation?  [Send it]    │ │ │
│ │               │ │  └───────────────────────────────────────┘ │ │
│ │ ▸ Harish S.   │ │  [ Message Ryuk                      ◔ ↑ ] │ │
│ └───────────────┘ └────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

Keep a terminal ready but **off screen** until segment 7. Bring it in as a small
overlay in the bottom-right, about 600×200, then dismiss it.

**The step list is the shot.** It draws itself as the plan arrives — dimmed
while pending, pulsing while running — and collapses to "N steps" when the run
finishes. Expanding it shows each step and its duration, which is where the
cost argument lives: most turns settle on **one** model call, and the calendar
list on **zero**.

Sync freshness and item counts live under **Your information** in the account
menu; Integrations shows what is connected. Neither is on screen by default —
open them deliberately when the script calls for it.

---

# Segment 1 — Cold open · 0:00 – 0:20 (20 s)

**On screen.** The app, already loaded, empty conversation. Do not show a logo
card, a title slide, or a face. Start on the product.

**Say** (≈50 words / ~20 s — no screen action here, so it is wall-to-wall):

> This is a chat box over Gmail, Calendar and Drive. The interesting part isn't
> the chat — it's what happens between pressing enter and the answer. A planner
> builds an execution graph, runs the steps in parallel, and asks before it
> changes anything. Watch the panel on the right.

**Why this shot.** Sets the grader's attention on the step list in the first
twenty seconds, so every later segment reads as evidence rather than decoration.

**Do not** explain the architecture here. Show it in segment 4 and let the panel
do the talking.

---

# Segment 2 — Connect Google, sync fills in · 0:20 – 0:50 (30 s)

**On screen.**

| t | action |
|---|---|
| 0:20 | Click **Connect Google**. |
| 0:22 | Google consent screen — already-consented account, so it is one click. |
| 0:26 | Redirect back. Your information shows three services, none checked yet. |
| 0:28 | Backfill starts automatically. Counters tick: `Gmail 0 → 340`, `Calendar 0 → 96`, `Drive 0 → 74`. |
| 0:44 | All three go green. `Gmail ● just now · 340`, and so on. |

**Say** (≈58 words / ~23 s, over the filling counters):

> OAuth in, and a backfill starts. It's pulling messages, events and files into
> our own Postgres — chunked, embedded, and indexed with pgvector. Every search
> after this hits that mirror, not Google, which is how a three-service query
> comes back in under a second. It re-syncs every fifteen minutes, and the bar
> tells you how stale it is.

**Have ready.** The backfill must complete in roughly sixteen seconds on camera.
With the seed corpus that is realistic on a warm machine; verify it in a dry run.
If it is slower, **cut** the middle of the fill in post — do not speed-ramp, it
looks like a hidden failure.

**Why this shot.** Proves the mirror exists and is populated before any query
runs, so nothing later can be dismissed as a canned response.

---

# Segment 3 — Simple query, establish the step list · 0:50 – 1:15 (25 s)

**Type.**

```
What's on my calendar next week?
```

**On screen.**

| t | what appears |
|---|---|
| +5 ms | Progress line in the step list. |
| +11 ms | One step: `gcal.search_events`. |
| +150 ms | Grouped event list, Mon Aug 24 – Sun Aug 30. |
| — | Trace footer reads **`LLM calls: 0 · 150 ms`**. Let it sit for two full seconds. |

**Say** (≈40 words / ~16 s, leaving 9 s to type and to sit on the counter):

> Next week. A hundred and fifty milliseconds, and zero model calls — a plain
> calendar read is a regex and a date rule. Note the window: Monday the
> twenty-fourth through Sunday the thirtieth, in my timezone, off my week-start
> setting.

**Why this shot.** Establishes what the step list means on something trivial, so
segment 4 does not have to explain the UI and the orchestration at the same time.
It also lands the zero-call claim early, on a query nobody can dispute.

**Point the cursor at** the `LLM calls: 0` counter as you say "zero model calls".

---

# Segment 4 — Flight cancellation · 1:15 – 2:25 (70 s)

The centrepiece. Five things happen and each one is visible.

**Type.**

```
Cancel my Turkish Airlines flight
```

**On screen, in order.** Do not talk over the first 200 ms — let it land.

| t | what appears | what to point at |
|---|---|---|
| +5 ms | progress | |
| **+200 ms** | **Candidate chips**: `TK1984 booking — turkishairlines.com`, `Istanbul → NYC Flight`, `Invoice_TK_1984.pdf` | **the chips** |
| +590 ms | Intent lands: `cancel_flight · gmail + gcal · write`. The DAG **draws itself** in the panel. | the graph |
| +600 ms | `gmail.get_email` and `gcal.search_events` bars start **on the same row, overlapping** | the two bars |
| +665 ms | `gmail.draft_email` starts — it waited for the booking | the edge into it |
| +1.06 s | `action.prepared` | |
| +1.2 s | Answer text, then the **confirm card** | the card |

**Say** (≈139 words / ~56 s spoken, leaving ~14 s of deliberate pauses):

> *(silent until the chips appear)*
>
> Those chips are the actual emails and calendar entries it's considering —
> retrieved at two hundred milliseconds, before the model has said a word. One
> embedding, three searches in parallel over pgvector and Postgres full-text.
>
> *(at 590 ms)* Now the plan. One call, and it comes back with a graph, not a
> list.
>
> *(at 600 ms)* Two searches at the same time — no reason for either to wait. The
> draft below them does wait, because it needs the booking reference.
>
> And it never retyped that reference. Regex pulled `6F2QK9` out of the email
> during the search, and the plan points at it by path — so it can't invent a
> booking code.
>
> *(at the card)* It drafted the cancellation, saved it in Gmail, and stopped.
> Nothing has been sent. That's the rule for every write: prepare it, show it,
> wait.
>
> One model call for all of that.

**Scroll the card** so the grader sees the real PNR, flight number and route in
the preview. Do **not** click Send.

**Why this shot.** It is the brief's own example, and it demonstrates parallel
execution, a genuine dependency, regex extraction feeding a write, and a gated
side effect — on one LLM call. If only one segment survives an edit, keep this one.

---

# Segment 5 — Two flights, ambiguity at 650 ms, free resume · 2:25 – 3:05 (40 s)

**Type.**

```
Cancel my flight to Istanbul
```

**On screen.**

| t | what appears |
|---|---|
| +200 ms | Five chips — **two are flagged as tied** |
| +590 ms | Intent, and the DAG has `ask.user` **at the head** |
| **+650 ms** | Choice card: TK1984 Sep 5 vs TK1996 Oct 12, each with departure, booking date, PNR and fare |
| — | Run status shows **`awaiting_input`** |

Click **TK1984**, then **Continue**.

| t | what appears |
|---|---|
| +6 ms | validated |
| +55 ms | `gmail.get_email` and `gcal.search_events` overlapping again |
| +470 ms | Same confirm card as segment 4 |
| — | Counter still reads **`LLM calls: 1`** |

**Say** (≈63 words / ~25 s spoken, leaving ~15 s for typing and the click):

> Two Istanbul bookings, five hundredths apart — inside the ambiguity margin. So
> instead of guessing, it asked. Six hundred and fifty milliseconds.
>
> And the question is a step in the plan, not a special case. So when I answer —
>
> *(click, then point at the counter)*
>
> — it picks up where it left off and the counter doesn't move. Answering,
> resuming, approving: all free, because the plan already exists.

**Point the cursor at** `LLM calls: 1` for a beat after the resume completes. That
number not changing is the entire segment.

**Why this shot.** Ambiguity is the brief's hardest listed case. This shows it
detected from a measured margin, surfaced fast, and resolved without paying twice.

---

# Segment 6 — Two writes, one card, enforced order · 3:05 – 3:50 (45 s)

**Type.**

```
Push my Acme review next Thursday to Friday 3pm and tell the attendees
```

**On screen.**

| t | what appears |
|---|---|
| +590 ms | Intent: `reschedule_and_notify · gcal + gmail · write` |
| +600 ms | Chain in the panel: `gcal.get_event → gcal.update_event → gmail.send_email` — **no parallel bars this time** |
| +1.1 s | **One** card with **two** numbered previews: calendar change, then the email, labelled *only if step 1 succeeds* |

**Say** (≈93 words / ~37 s spoken, leaving ~8 s for typing and the card):

> Two dates in one sentence. "Next Thursday" is the twenty-seventh; "Friday" then
> resolves inside *that* week — the twenty-eighth, not tomorrow.
>
> Two writes now, and look: nothing runs in parallel. The email depends on the
> calendar update deliberately. It doesn't need data from it — it needs it to have
> *happened*. The invite lands before the announcement, or the announcement
> doesn't go out.
>
> One card, one button, two actions. And the database enforces the gate: an action
> row can't exist without a prompt row. That's a `NOT NULL` foreign key, not a
> convention.

**Optional, only if you are ahead of schedule.** Click **Edit email**, change one
word, and show the card update — `actions.revisions` keeps the previous payload.
Budget 8 s. Cut it first if you are behind.

Do **not** click **Do both**.

**Why this shot.** Ordering between side effects is the property that separates an
orchestrator from a function caller, and the `NOT NULL` FK is a concrete, checkable
design claim rather than a promise.

---

# Segment 7 — Kill Calendar mid-run · 3:50 – 4:40 (50 s)

The honesty segment. Rehearse the timing — you have about a two-second window.

**Setup.** Terminal overlay, bottom-right, command already typed and **not**
entered:

```
docker compose stop gcal-proxy
```

**On screen.**

| t | action |
|---|---|
| 3:50 | Terminal slides in. Press enter. `Stopping alpha-law-gcal-proxy … done` |
| 3:54 | Terminal slides out. Your information's Calendar dot goes **amber**. |
| 3:56 | Type the query. |

```
Prepare for tomorrow's meeting with Acme Corp
```

| t | what appears |
|---|---|
| +200 ms | chips from all three services |
| +600 ms | `gcal.search_events` starts — marked `live`, because it is about tomorrow |
| **+712 ms** | **`step.retrying`** — the bar turns amber, `503 · attempt 1 of 2` |
| +701 ms | `gdrive.search_files` **succeeds** alongside it |
| +1.12 s | `gcal.search_events` → **red, failed** |
| +1.13 s | `gmail.search_emails` → **grey, skipped**, tooltip `dependency_failed: meeting` |
| +1.95 s | First prose token — **and the first sentence is the failure** |
| +3.4 s | Complete, plus a `Retry Calendar` chip |

**Say** (≈57 words / ~23 s spoken — the rest of the segment is screen action):

> I've just killed Calendar mid-flight.
>
> *(at the amber bar)* Two attempts, four hundred milliseconds of backoff, then it
> stops — because a person is waiting. The full retry policy runs in a worker.
>
> Drive succeeded. Gmail is grey, not red: skipped, because its filter needed the
> guest list Calendar never returned. A skip isn't an error.
>
> *(at the first sentence)* And the answer leads with what broke.

**Then bring the container back and click the chip:**

```
docker compose start gcal-proxy
```

| t | what appears |
|---|---|
| — | Calendar dot green |
| click | `gcal.search_events` **round 1** — a new row, the failed one stays visible |
| — | `gmail.search_emails` round 1 runs; `gdrive.search_files` is **reused, not re-run** |
| — | Complete brief |

**Say** (≈27 words / ~11 s):

> Retry doesn't re-plan. Same graph — it re-runs the failed node and what it
> blocked, and reuses what worked. Both attempts stay in the trace.

**Why this shot.** The brief calls out "Gmail succeeds, Calendar fails" as a thing
that must be handled gracefully. This shows it happening live, with the failure
stated first and a recovery that does not pay for the plan twice.

---

# Segment 8 — Close · 4:40 – 5:00 (20 s)

**On screen.** Scroll the conversation up so all five queries are visible at once
with their trace footers: `0`, `1`, `1`, `1`, `2`.

**Say** (≈46 words / ~18 s, leaving 2 s to scroll):

> Five queries. Zero, one, one, one, two model calls. The model is the expensive
> part, so everything else — searching, dates, asking, approving, retrying — is
> plain Python.
>
> What it can't do is tell when it's confidently wrong. That's why it always shows
> you its sources.

**End on the step list**, not on a thank-you slide.

**Why this ending.** Naming the one failure mode the system does not handle, in the
last five seconds, is more persuasive than any feature would be. It also matches
the honest-limits section in `docs/SAMPLE_QUERIES.md`, so a grader who reads both
finds the same claim twice.

---

# Timing summary

| # | Segment | Start | Length |
|---|---|---|---|
| 1 | Cold open | 0:00 | 20 s |
| 2 | Connect + sync | 0:20 | 30 s |
| 3 | Simple query, step list | 0:50 | 25 s |
| 4 | **Flight cancellation** | 1:15 | **70 s** |
| 5 | Two flights, ambiguity, free resume | 2:25 | 40 s |
| 6 | Two writes, one card, enforced order | 3:05 | 45 s |
| 7 | **Kill Calendar, partial failure, retry** | 3:50 | **50 s** |
| 8 | Close | 4:40 | 20 s |
| | | | **5:00** |

Every **Say** block is costed at **150 words per minute** — an unhurried
demo-voice pace, not a presentation pace. The remainder of each segment is screen
action: typing, waiting for a run, clicking a card. If you naturally speak faster,
do not fill the gap; the pauses are where the grader actually reads the trace
panel.

**If you overrun**, cut in this order and no other:

1. The **Edit email** beat in segment 6 (−8 s).
2. Segment 2 down to 20 s by cutting the middle of the backfill (−10 s).
3. Segment 1 down to 12 s (−8 s).
4. Segment 3 down to 18 s — but keep the shot of `LLM calls: 0` (−7 s).

Never cut segment 4 or segment 7. They are the two the brief actually asks for.

---

# Pre-flight checklist

Run through this within the hour before recording. Every line has failed at least
once.

### Containers

- [ ] `docker compose ps` — `api`, `worker`, `beat`, `postgres`, `redis`,
      `frontend`, `gcal-proxy` all `running (healthy)`.
- [ ] `curl -s localhost:5173/healthz` → `{"status":"ok"}`
- [ ] `curl -s localhost:5173/readyz` → db and redis both `ok`.
- [ ] `docker compose logs --since 5m api | grep -i error` → empty.
- [ ] `docker compose start gcal-proxy` — in case a previous rehearsal left it
      stopped. **This is the single most common way the recording dies.**

### Seeded data

The demo needs these exact fixtures present. Verify by querying, not by hoping.

- [ ] Gmail ≈ 340 messages, including:
      - `Your Turkish Airlines booking is confirmed — TK1984`, from
        `noreply@turkishairlines.com`, PNR `6F2QK9`, dated ~4 weeks back
      - a **second** Turkish Airlines booking `TK1996`, PNR `R4TQ8M`, so
        segment 5 has something to be ambiguous about
      - the Acme renewal thread: ≥ 7 messages from `@acmecorp.com` this month
- [ ] Calendar ≈ 96 events, including:
      - `Istanbul → NYC Flight (TK1984)` matching the booking date
      - `Acme review` on **next Thursday**, 1:00–2:00 PM, with 3 external guests
      - `Acme Corp — Q3 renewal review` **tomorrow** at 10:00, for segment 7
      - 5–7 events inside next week, so segment 3's list looks lived-in and not
        like three rows of test data
- [ ] Drive ≈ 74 files, including `Acme — Q3 renewal proposal v4`,
      `Acme_MSA_countersigned.pdf`, `Invoice_TK_1984.pdf`
- [ ] Every fixture date is **relative to today**, regenerated by the seeder.
      A booking dated last year makes segment 4 look broken.

```sql
-- one query, run it
SELECT 'message' AS shape, connector, count(*) FROM sync_messages WHERE user_id = :u GROUP BY 2
UNION ALL SELECT 'event', connector, count(*) FROM sync_events   WHERE user_id = :u GROUP BY 2
UNION ALL SELECT 'file',  connector, count(*) FROM sync_files    WHERE user_id = :u GROUP BY 2
          WHERE user_id = :u AND embedding IS NULL;
-- the last row must be 0
```

### Sync

- [ ] `GET /api/v1/sync/status` — all three services `last_success_at` within
      15 minutes, `consecutive_failures = 0`, `circuit_open_until` null.
- [ ] `job_failed_tasks` where `status = 'open'` → **0 rows**. A non-zero badge in
      the Your information is a distraction you will have to explain.
- [ ] Embedding backlog empty: no `sync_*` row with `embedding IS NULL`.

### The user

- [ ] `users.timezone` matches the **recording machine's** timezone. If they
      disagree, the on-screen clock and the resolved windows will contradict each
      other and someone will notice.
- [ ] `users.work_week_start = 1`, so "next week" starts on Monday and matches
      what you say out loud.
- [ ] `users.display_name` set — it appears in the cancellation draft.
- [ ] Today is **not** a Friday, Saturday or Sunday. Segment 6 says "next
      Thursday" and segment 7 says "tomorrow"; on a Friday, "tomorrow" is a
      Saturday with an empty calendar. **Record Monday to Thursday.**

### Environment

- [ ] Browser 1440×900, zoom 110%, one tab, no extensions, no bookmark bar.
- [ ] System notifications **off**. Slack quit, not minimised.
- [ ] Terminal overlay pre-positioned bottom-right, large font, `docker compose
      stop gcal-proxy` typed and unentered.
- [ ] Clear the demo conversation so the transcript starts empty.
- [ ] Mic level checked on 10 seconds of actual speech, not a tap.
- [ ] Screen recorder at 1080p30, capturing system audio **off** (you do not want
      a Docker beep in the take).

### Dry run

- [ ] Run all five queries end to end once. Confirm the LLM-call counters read
      **0, 1, 1, 1, 2**. If any of them is higher, something is re-planning and you
      want to know before you are talking over it.
- [ ] Time segment 4 with a stopwatch. If the confirm card takes longer than
      ~1.5 s, the mirror is cold — run the query once to warm it, then clear the
      conversation.
- [ ] **Then clear the conversation again** so the recorded take is a first run.

---

# The 7-day token, and other ways this breaks

**Test-mode refresh tokens expire after 7 days.** While the Google Cloud project's
OAuth consent screen is in **Testing** publishing status, every refresh token
Google issues is invalidated after seven days. When it goes, the next Google call
returns `invalid_grant`, which `classify()` maps to `AUTH_REVOKED`, which surfaces
as `GOOGLE_REAUTH_REQUIRED` (HTTP 428) and a "reconnect Google" banner. Nothing is
broken; the grant is simply gone.

**What this means for recording.** Grant consent, then record **within a few
days** — not a week later, and not the morning after a long weekend. If you shot
segments 1–4 on a Monday and come back the following Tuesday for segments 5–8,
Google will have expired the token in between and the second half will not run.

If it happens mid-take:

1. Reconnect via `/api/v1/auth/google`. The mirror is untouched — `sync_*` rows
   survive a token expiry, so there is no backfill to wait for.
2. Confirm `oauth_tokens.revoked_at IS NULL` and `refresh_failures = 0`.
3. Re-record the segment. Do not try to splice around it.

To remove the seven-day limit entirely, move the consent screen to **In
production**. That requires verification for the scopes this app uses, which takes
longer than the assignment does — so plan around the limit rather than fighting it.

### The other three

**Rate limits.** Repeated dry runs can push the Gmail API toward its quota. The
governor models 250 units/sec split 70/30 background/interactive, so background
sync backs off before your interactive query does — but if you see
`RATE_LIMITED` in the logs, pause the beat schedule for the recording:

```
docker compose stop beat
```

Do this **after** segment 2's backfill and remember to note that sync is paused if
anyone asks why the bar's age keeps climbing.

**A cold mirror.** The first query after a restart pays for connection setup and
an empty query-plan cache — 300 to 400 ms extra, enough to make segment 3's
"a hundred and fifty milliseconds" a lie. Always warm it, then clear the
conversation.

**The date drift.** If the seed fixtures were generated more than a day or two
ago, "tomorrow" and "next Thursday" will point at empty days. Re-run the seeder on
the morning of the recording. It is idempotent.
