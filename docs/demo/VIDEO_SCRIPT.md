# Demo video — narration script

The rendered cut is [`demo.mp4`](demo.mp4) (5:15, background music, captioned — the caption sits in its own band below the app window).
Each flow appears in two stages — the live run with its steps streaming, then
the finished answer with the step list expanded. This script matches the cut
scene for scene; read it over the video or paste it into HeyGen for an
avatar-narrated version. Composition source: [`composition/`](composition/).

| # | At | Scene | Frames |
|---|----|-------|--------|
| 1 | 0:00 | Title | — |
| 2 | 0:10 | The setup — seeded through OAuth | — |
| 3 | 0:18 | Sign-in screen recording (2×) | `login-flow.mp4` |
| 4 | 0:39 | Seeded Gmail / Calendar ×2 / Drive | `../gsuite-screenshots/g1–g4` |
| 5 | 1:09 | The product | `01_home.png` |
| 6 | 1:15 | Calendar next week | `02a` → `02b` |
| 7 | 1:29 | Sarah's budget emails | `03a` → `03b` |
| 8 | 1:43 | Last month's PDFs | `04a` → `04b` |
| 9 | 1:57 | Turkish Airlines cancel | `05a` → `05b` |
| 10 | 2:24 | Acme meeting prep | `06a` → `06b` |
| 11 | 2:47 | Out-of-office conflicts | `07a` → `07b` |
| 12 | 3:05 | Move the meeting with John | `08a` → `08b` → `09_time_form` |
| 13 | 3:36 | "That email" — context, two turns | `10b` → `10c` |
| 14 | 3:58 | Next Tuesday | `11a` → `11b` |
| 15 | 4:12 | Compose | `12a` → `12b` |
| 16 | 4:30 | "Send it" — approval gate | `13a` → `13b` |
| 17 | 4:50 | Under the hood | — |

---|----|-------|------------|
| 1 | 0:00 | Title | — |
| 2 | 0:10 | The setup — seeded through OAuth | — |
| 3 | 0:18 | Seeded Gmail | `../gsuite-screenshots/g1_gmail.png` |
| 4 | 0:28 | Seeded Calendar (next week) | `g2.png` |
| 5 | 0:37 | Seeded Calendar (tomorrow) | `g3.png` |
| 6 | 0:44 | Seeded Drive | `g4.png` |
| 7 | 0:52 | The product | `01_home.png` |
| 8 | 1:00 | Calendar next week | `02_calendar_next_week.png` |
| 9 | 1:16 | Sarah's budget emails | `03_sarah_budget_emails.png` |
| 10 | 1:32 | Last month's PDFs | `04_drive_pdfs_last_month.png` |
| 11 | 1:48 | Turkish Airlines cancel | `05_turkish_airlines_cancel.png` |
| 12 | 2:18 | Acme meeting prep | `06_acme_meeting_prep.png` |
| 13 | 2:42 | Out-of-office conflicts | `07_ooo_conflicts.png` |
| 14 | 3:02 | Move the meeting with John | `08_move_john_ambiguous.png` |
| 15 | 3:26 | The follow-up | `09_move_john_time_form.png` |
| 16 | 3:38 | "That email" — context | `10_context_that_email.png` |
| 17 | 3:58 | Next Tuesday | `11_next_tuesday.png` |
| 18 | 4:14 | Compose | `12_compose_draft.png` |
| 19 | 4:34 | "Send it" — approval gate | `13_send_confirm_card.png` |
| 20 | 4:58 | Under the hood | — |

---

**1 — Title.** "This is Ryuk, an assistant that works across Gmail, Google
Calendar and Drive from one chat box. Underneath: an intent classifier, an LLM
query planner that builds a dependency graph, parallel execution against a
pgvector mirror, and two-phase writes — nothing irreversible without approval."

**2 — The setup.** "Everything you'll see runs against a real Google account.
The demo data was seeded through the app's own stored OAuth token — the same
grant, the same APIs."

**3–6 — The seed, in Google's own UI.** "Here it is in Gmail: a Turkish
Airlines booking with a PNR, Sarah's two budget threads, Acme's agenda and
proposal. On the calendar: next week's meetings, including two with John and
one with Sarah next Tuesday, and tomorrow's Acme call. In Drive: the proposal
and budget PDFs, dated last month, plus an out-of-office note."

**7 — The product.** "The app itself is one chat box. Every answer shows the
plan it ran — each step, with timings."

**8 — Calendar.** "What's on my calendar next week. The window resolves in my
timezone, the mirror answers in well under a second, and the answer states the
week it read."

**9 — Mail search.** "Find emails from sarah at company dot com about the
budget. Hybrid search — vector similarity plus a sender filter. Both of her
budget threads, nothing else."

**10 — Drive search.** "Show me PDFs in Drive from last month. A mime filter
plus a resolved date window — and the answer says which window it searched."

**11 — The flagship.** "Cancel my Turkish Airlines flight. Gmail search finds
the booking and extracts the PNR — ABC123. The flight's calendar event is
located. A cancellation email to the airline is drafted, quoting that PNR.
And then it stops: the draft exists, the send waits for approval. That card is
the two-phase write model in one screen."

**12 — Meeting prep.** "Prepare for tomorrow's meeting with Acme Corp. Three
services in parallel — the event and its guests, the agenda and proposal
emails, the proposal PDF from Drive — synthesized into one briefing."

**13 — Conflicts.** "Find events next week that conflict with my
out-of-office doc. It reads the doc, understands the dates, and lists exactly
the two meetings that clash."

**14 — Ambiguity.** "Move the meeting with John. Which John, which meeting?
It resolves John to the person actually on my invitations and offers his real
meetings as choices. Ambiguity is a step in the plan — pausing costs nothing."

**15 — The follow-up.** "One tap picks the meeting. The only genuinely
unknowable thing — the new time — is asked as a typed date field."

**16 — Context.** "Who sent that email about the proposal? 'That email'
resolves against what this conversation already found. Daniel Lee, from
acmecorp dot com — no re-search."

**17 — Temporal.** "What's on my calendar next Tuesday. Timezone-aware,
week-boundary-aware: Tuesday the twenty-fifth, exactly one event."

**18 — Compose.** "Compose an email and write the subject and body yourself.
The model writes it, the draft lands in Gmail drafts, and the answer says
plainly: nothing has been sent."

**19 — The gate.** "Send it. The follow-up finds the draft in conversation
context, stages the send, and raises the approval card naming the recipient.
Approve and it goes. Decline and it doesn't."

**20 — Close.** "Nine for nine on the brief's sample queries. Orchestration
built from scratch — no frameworks. pgvector hybrid search under 500
milliseconds. Celery keeps the mirror fresh every fifteen minutes. And every
write waits for a human. That's Ryuk."
