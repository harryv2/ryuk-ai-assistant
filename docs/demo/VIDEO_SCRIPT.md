# Demo video — script and storyboard

~5 minutes at a normal speaking pace. One scene per screenshot in this folder;
paste the narration into HeyGen scene by scene and set each PNG as the scene's
background. Timings are what the narration takes aloud — trim to taste.

| # | Screenshot | Duration | What is on screen |
|---|------------|----------|-------------------|
| 1 | `01_home.png` | 0:20 | The empty app |
| 2 | `02_calendar_next_week.png` | 0:20 | Calendar list answer |
| 3 | `03_sarah_budget_emails.png` | 0:20 | Filtered mail search |
| 4 | `04_drive_pdfs_last_month.png` | 0:20 | Drive search with a window |
| 5 | `05_turkish_airlines_cancel.png` | 0:35 | Three services, one confirm card |
| 6 | `06_acme_meeting_prep.png` | 0:30 | Cross-service briefing |
| 7 | `07_ooo_conflicts.png` | 0:25 | Doc-driven conflict check |
| 8 | `08_move_john_ambiguous.png` | 0:30 | Ambiguity as choice chips |
| 9 | `09_move_john_time_form.png` | 0:15 | The follow-up question |
| 10 | `10_context_that_email.png` | 0:25 | "That email" resolved from context |
| 11 | `11_next_tuesday.png` | 0:20 | Temporal reasoning |
| 12 | `12_compose_draft.png` | 0:25 | Composed draft, nothing sent |
| 13 | `13_send_confirm_card.png` | 0:25 | "Send it" → approval gate |

---

**Scene 1 — the product.** "This is Ryuk, an assistant that works across
Gmail, Google Calendar and Drive from one chat box. Under it: an intent
classifier, an LLM query planner that builds a dependency graph, parallel
execution against a pgvector mirror of the account, and live Google API
fallback when the mirror is behind. Everything you'll see runs against a real
Google account."

**Scene 2 — a simple read.** "What's on my calendar next week. The planner
resolves 'next week' in my timezone, reads the mirror, and answers in well
under a second — no LLM call needed for a shape this common. Every answer
shows its steps, with timings."

**Scene 3 — filtered search.** "Find emails from sarah about the budget.
Hybrid search: vector similarity plus keyword filtering on the sender. Both of
Sarah's budget threads, nothing else."

**Scene 4 — Drive with a time window.** "Show me PDFs from last month. A mime
filter plus a resolved date window — and the answer states the window it
searched, so a disagreement is one sentence away."

**Scene 5 — the flagship multi-service flow.** "Cancel my Turkish Airlines
flight. Gmail search finds the booking confirmation and extracts the PNR —
ABC123. The calendar event for the flight is located. A cancellation email to
the airline is drafted, quoting that PNR. And then it stops: writes are
two-phase. The draft exists, the send waits for approval. Nothing irreversible
happens without a click."

**Scene 6 — meeting prep.** "Prepare for tomorrow's meeting with Acme Corp.
Three services in parallel: the event with its guests, the agenda email, the
proposal email, and the proposal PDF from Drive — synthesized into one
briefing."

**Scene 7 — documents driving calendar logic.** "Find events next week that
conflict with my out-of-office doc. It reads the doc from Drive, understands
the dates I'm away, and lists exactly the two meetings that clash."

**Scene 8 — ambiguity, handled properly.** "Move the meeting with John. Which
John? Which meeting? Instead of guessing or asking an open question, it
resolves John to the person actually on my invitations and offers his real
meetings as choices. Ambiguity is a step in the plan — pausing costs nothing."

**Scene 9 — the resume.** "One tap picks the meeting; the only genuinely
unknowable thing — the new time — is asked as a typed date field, not free
text."

**Scene 10 — conversation context.** "Earlier in this chat I asked about the
proposal. Now just: who sent that email about the proposal. 'That email'
resolves against what the conversation has already referred to — no re-search,
no re-asking."

**Scene 11 — temporal reasoning.** "What's on my calendar next Tuesday.
Timezone-aware, week-boundary-aware — Tuesday the 25th, exactly one event."

**Scene 12 — composing.** "Compose an email about the federal integration —
write the subject and body yourself. The model writes it, the draft lands in
Gmail drafts, and the answer says so plainly: nothing has been sent."

**Scene 13 — the send gate.** "Send it. The follow-up turn finds the draft in
conversation context, stages the send, and raises the approval card naming the
recipient. Approve and it goes; decline and it doesn't. That's the whole
product: grounded answers, parallel orchestration, and writes that never
outrun the person."
