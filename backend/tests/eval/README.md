# Evaluation harness

The four numbers the brief grades are produced here, by three scripts over two
labelled datasets.

| Brief line | Points | Produced by |
|---|---|---|
| Intent classification accuracy | 10 | `intent_accuracy.py` |
| Embedding & search quality: Precision@5 > 0.8 | 10 | `precision_at_k.py` |
| Relevance metrics | 2 | `precision_at_k.py` (MRR, nDCG@5, Recall@10, per-type split, ablation) |
| Performance < 500 ms | 3 | `latency.py --path search` |

`latency.py --path query` additionally evidences the P95-under-2s read claim,
which is a different measurement over a different path and is never averaged
into the search figure.

```
tests/eval/
  __init__.py              shared kit: datasets, percentiles, tables, backends, adapters
  datasets/
    intents.jsonl          63 labelled queries, 18 intents, 7 languages
    relevance.jsonl        35 queries, 101 graded judgements
  intent_accuracy.py       accuracy, per-intent P/R/F1, confusions, missed ambiguities
  precision_at_k.py        P@1/@3/@5, R@10, MRR, nDCG@5, the ablation, writes RESULTS.md
  latency.py               search path and full query path, separately
  RESULTS.md               the committed output
  out/                     metric JSON, written at run time (not committed)
```

---

## Running it

```bash
# inside the api container, or anywhere with backend/ on PYTHONPATH
python -m tests.eval.intent_accuracy                  # -> out/intent_accuracy.json
python -m tests.eval.precision_at_k                   # -> out/precision_at_k.json
python -m tests.eval.latency                          # -> out/latency.json
python -m tests.eval.precision_at_k --write-results   # -> RESULTS.md
```

Nothing here needs pytest; they are scripts, and they exit non-zero when you
give them a threshold to hold:

```bash
python -m tests.eval.intent_accuracy --fail-under 0.85 --max-missed-ambiguities 0
python -m tests.eval.precision_at_k  --fail-under 0.80
```

The `make eval` target that RESULTS.md refers to is those four lines:

```make
eval: ## Run the evaluation harness and regenerate tests/eval/RESULTS.md
	$(API) python -m tests.eval.intent_accuracy
	$(API) python -m tests.eval.precision_at_k
	$(API) python -m tests.eval.latency
	$(API) python -m tests.eval.precision_at_k --write-results
```

### Without a database or an API key

```bash
python -m tests.eval.intent_accuracy --dry-run   # canned classifier, fixed mistakes
python -m tests.eval.precision_at_k --self-test  # checks the metric maths, 20 assertions
```

Both are for CI. `--dry-run` produces a full report from canned responses so a
broken table is caught without spending a token; every line it prints is
labelled **CANNED — NOT A MEASUREMENT**, and `RESULTS.md` carries the same
warning if it is assembled from a dry run.

`--self-test` checks the arithmetic against hand-computed values — including
the two cases that were wrong the first time this harness ran: precision with
fewer relevant documents than `k`, and one document credited twice.

---

## What the datasets contain

### `datasets/intents.jsonl`

63 labelled queries. One JSON object per line; the line carrying `_schema` is
metadata and is skipped by the loader.

```json
{"id": "i009",
 "query": "Find emails from sarah@company.com about the budget",
 "lang": "en",
 "tags": ["brief_sample", "single_service", "filter"],
 "context": {"prior_intent": null, "prior_turns": []},
 "expected": {
   "intent": "email_search",
   "services": ["gmail"],
   "entity_keys": ["from_email", "topic"],
   "entities": {"from_email": "sarah@company.com", "topic": "budget"},
   "has_write": false,
   "flag_ambiguity": false,
   "refs_prior_turn": false,
   "route": "planner"},
 "note": "..."}
```

`entity_keys` is what is scored. `entities` carries values only where the value
is a literal lift from the query, and is scored separately as entity-value
accuracy — a classifier that returns `from_email: "sarah"` has produced the
right key and the wrong value, and those are different failures. `route` is the
front-door path we expect and is **reported, not scored**: which matcher fires
is an implementation choice, and pinning it in a graded dataset would freeze
the front door.

**The taxonomy** — 18 intents. Eleven come from the worked examples in
`docs/SAMPLE_QUERIES.md`; the other seven cover ops in the registry that the
sixteen examples happen not to reach.

| intent | services | in the docs | queries |
|---|---|---|---|
| `calendar_list` | gcal | yes (#1, #9, #16) | 9 |
| `email_search` | gmail | yes (#2) | 6 |
| `email_detail` | gmail | yes (#8) | 3 |
| `drive_filter` | gdrive | yes (#3) | 4 |
| `drive_search` | gdrive | new | 3 |
| `meeting_prep` | gcal + gmail + gdrive | yes (#5, #11) | 4 |
| `conflict_check` | gcal + gdrive | yes (#6) | 3 |
| `availability` | gcal | new | 2 |
| `digest` | gmail | yes (#14) | 4 |
| `cancel_flight` | gmail + gcal | yes (#4, #10, #12) | 4 |
| `move_event` | gcal | yes (#7) | 4 |
| `reschedule_and_notify` | gcal + gmail | yes (#13) | 2 |
| `email_compose` | gmail (+ gdrive/gcal) | new | 3 |
| `event_create` | gcal | new | 2 |
| `file_share` | gdrive | new | 2 |
| `chit_chat` | — | yes (#15) | 2 |
| `ui_verb` | — | yes (front door) | 3 |
| `unsupported` | — | new | 3 |

Coverage that is there on purpose:

* **All six of the brief's sample queries plus its problem-statement example**
  (7 rows, tagged `brief_sample`), and **all three hard cases** (tagged
  `brief_hard`).
* **Six non-English queries** — Spanish, French, German, Japanese, Turkish and
  romanised Hindi. The Turkish and Japanese ones matter most: the Postgres text
  arm is `english`-configured, so those are the queries where the vector arm is
  the only thing that can work.
* **Five awkward ones** — `cal next wk?`, `emails sarah budget`, `pdfs last
  month`. Telegraphic, no verb, no address.
* **Four that must raise an ambiguity card** and four near-identical ones that
  must not. `Move the meeting with John` (flag) against `push my 1:1 with John
  Park to Friday` (do not flag) is the pair that matters: the flag has to track
  the query, not the intent name.
* **Ten that lean on the conversation**, including `Next Tuesday` twice — once
  after a calendar turn, where it carries the previous intent, and once in an
  empty conversation, where it does not. Same string, different label.
* **Three out-of-scope**, one of which (`book me a flight to Istanbul next
  Friday`) is a near-miss for `cancel_flight` and is the interesting one.
* **Two traps** the front door must not take: `thanks — also what's on Friday?`
  is `calendar_list`, not chit-chat, and `perfect, do the same for Northwind`
  is a digest with one entity swapped.

### `datasets/relevance.jsonl`

35 queries, 101 graded judgements.

```json
{"id": "r009", "type": "sender", "query": "budget",
 "params": {"services": ["gmail"], "from": "sarah@company.com"},
 "relevant": [
   {"service": "gmail", "id": "g-budget-draft", "grade": 2, "title": "Q4 budget draft for review"},
   {"service": "gmail", "id": "g-budget-headcount", "grade": 2, "title": "Re: Q4 budget — headcount lines"}],
 "note": "..."}
```

* **grade 2** — what the query is actually asking for.
* **grade 1** — genuinely related, fine to see in the list, wrong at the top.
* Precision and recall count grade >= 1. nDCG uses the grades, gain `2^g - 1`.

| type | queries | what it exercises |
|---|---|---|
| `semantic` | 8 | ranking with no filters at all |
| `sender` | 5 | exact `from_email` prefilter — including one where no subject contains the query word, so the text arm returns nothing |
| `date_window` | 5 | half-open windows resolved against `now`, one rolling and four calendar |
| `attendee` | 4 | the `attendee_emails` GIN index, including the brief's `john@company.com` query |
| `mime` | 4 | Drive type prefilters, two of them with no query text |
| `cross_lingual` | 4 | Turkish and Spanish queries against mostly-English documents |
| `multi_service` | 3 | the probe's three-corpus briefing, judged directly |
| `absence` | 2 | queries with no referent — the floor's job |

`params.window` is a phrase (`"last month"`), a rolling offset
(`{"days_back": 7}`) or literal bounds. Phrases resolve through
`app.orchestrator.temporal.resolve` when it is importable, which is deliberate:
the eval should exercise the real rule rather than a copy of it. There is a
local fallback for the handful of phrases used here, and `--strict-temporal`
refuses it.

**Listing queries are judged completely.** For `pdfs from last month` every PDF
inside the window is graded, not just the interesting ones — otherwise a
correct result would be scored as an error for surfacing a document nobody
labelled.

---

## The fixture manifest

`relevance.jsonl` keys on the provider ids below: `sync_gmail.message_id`,
`sync_gcal.event_id`, `sync_gdrive.file_id`. **This is the contract with the
seeder.** Dates are relative to `now`, which the seeder regenerates, so the
dataset does not go stale.

If the seeder assigns opaque ids instead, `--match=title` scores on the
normalised subject/title/name in the `title` field, and `--match=auto` (the
default) tries the id first and falls back to the title.

### Gmail — 18 messages

| `message_id` | subject | from | when |
|---|---|---|---|
| `g-tk1984-confirm` | Your Turkish Airlines booking is confirmed — TK1984 | noreply@turkishairlines.com | now − 28d |
| `g-tk1984-invoice` | Invoice for booking TK1984 | billing@turkishairlines.com | now − 27d |
| `g-tk1996-confirm` | Turkish Airlines booking confirmed — TK1996 | noreply@turkishairlines.com | now − 21d |
| `g-thy-tr-booking` | THY rezervasyon onayınız — TK2010 | bilet@thy.com | now − 14d |
| `g-acme-kickoff` | Q3 renewal — kickoff and timeline | sarah.chen@acmecorp.com | now − 18d |
| `g-acme-pricing` | Re: Q3 renewal — pricing questions | mike.ross@acmecorp.com | now − 11d |
| `g-acme-redlines` | MSA redlines attached | legal@acmecorp.com | now − 6d |
| `g-acme-agenda` | Agenda for Thursday's renewal review | sarah.chen@acmecorp.com | now − 2d |
| `g-acme-countersigned` | Countersigned MSA | legal@acmecorp.com | now − 1d |
| `g-budget-draft` | Q4 budget draft for review | sarah@company.com | now − 9d |
| `g-budget-headcount` | Re: Q4 budget — headcount lines | sarah@company.com | now − 4d |
| `g-budget-forecast` | Budget forecast v2 | tom@company.com | now − 5d |
| `g-ooo-notice` | Out of office 24–28 August | demo@alphalaw.test | now − 3d |
| `g-john-mercer-move` | Can we move our Tuesday sync? | john.mercer@northwind.example | now − 2d |
| `g-john-park-invoice` | Invoice question | john.park@company.com | now − 8d |
| `g-northwind-nda` | Northwind NDA for signature | contracts@northwind.example | now − 13d |
| `g-standup-notes` | Standup notes | team@company.com | now − 1d |
| `g-newsletter-devops` | The DevOps Weekly, issue 212 | news@devopsweekly.example | now − 2d |

`g-thy-tr-booking` must have a **Turkish body** — that is the whole point of it.
Something like: *THY rezervasyonunuz onaylandı. PNR K7WQ2N. İstanbul (IST) →
New York (JFK), TK2010.*

The three `@acmecorp.com` senders are deliberate: `r010` filters on one address
inside a domain that has three, and `i036` in the intent set filters on the
domain.

### Calendar — 14 events

| `event_id` | title | when | attendees |
|---|---|---|---|
| `c-tk1984-flight` | Istanbul → NYC Flight (TK1984) | now + 15d, 10:30 | — |
| `c-acme-renewal-tomorrow` | Acme Corp — Q3 renewal review | tomorrow 10:00–11:00 | sarah.chen@, mike.ross@ |
| `c-acme-review-thu` | Acme review | next Thu 13:00–14:00 | sarah.chen@, mike.ross@, legal@acmecorp.com |
| `c-design-review` | Design review | next Tue 14:00–15:00 | john@company.com |
| `c-all-hands` | All hands | next Wed 16:00–17:00 | john@company.com |
| `c-john-mercer-sync` | Weekly sync — John Mercer | next Tue 15:00–15:30 | john.mercer@northwind.example |
| `c-john-park-1on1` | 1:1 with John Park | next Wed 11:00–11:30 | john.park@company.com |
| `c-dentist` | Dentist | next Thu 08:00–09:00 | — |
| `c-standup-mon` … `c-standup-fri` | Daily standup | next week weekdays 09:15–09:30 | team@company.com |
| `c-board-prep` | Q3 board prep | this week Fri 15:00 | — |

The two Johns are the point. `c-john-mercer-sync` and `c-john-park-1on1` are in
the same week, and `Move the meeting with John` has to notice that.

### Drive — 10 files

| `file_id` | name | mime | modified |
|---|---|---|---|
| `d-acme-proposal-v4` | Acme — Q3 renewal proposal v4 | `…google-apps.document` | now − 3d |
| `d-acme-msa-pdf` | Acme_MSA_countersigned.pdf | `application/pdf` | now − 1d |
| `d-invoice-tk1984-pdf` | Invoice_TK_1984.pdf | `application/pdf` | now − 27d |
| `d-ooo-plan` | Out of office — August coverage plan | `…google-apps.document` | now − 5d |
| `d-budget-model` | FY26 budget model | `…google-apps.spreadsheet` | now − 9d |
| `d-headcount-sheet` | Headcount plan FY26 | `…google-apps.spreadsheet` | now − 33d |
| `d-board-deck-pdf` | Q3 board deck.pdf | `application/pdf` | now − 40d |
| `d-security-review-pdf` | Vendor security review.pdf | `application/pdf` | now − 38d |
| `d-northwind-nda-pdf` | Northwind_NDA.pdf | `application/pdf` | now − 13d |
| `d-brand-guidelines-pdf` | Brand guidelines.pdf | `application/pdf` | now − 200d |

`d-ooo-plan` must say **24–28 August** in its text, because `r032` judges every
event in that window as conflicting.

Filler around these is welcome and does not disturb anything — the judgements
name what is relevant, and everything else in the corpus is a distractor by
definition. `docs/DEMO.md` asks for ≈340 messages, ≈96 events and ≈74 files;
the 42 rows above are the ones the eval depends on.

---

## What the harness binds to

The modules under measurement are being written in parallel with this one, so
the adapters bind by introspection and **fail loudly with the list of names
they looked for** rather than silently measuring something else.

### Search

| backend | binds to | ablation | supplies `cn` |
|---|---|---|---|
| `mirror` (default) | `app.db.repositories.mirror.hybrid_search` | **yes** | no |
| `hybrid` | `app.search.hybrid`, then `app.search.probe` → first of `search`, `hybrid_search`, `run`, `query`, `probe` | if it takes `arm`/`arms`/`mode` | yes |
| `http` | `GET /api/v1/search` | no | yes |

The ablation runs on `mirror` because the repository function already takes the
two arms as separate inputs — an embedding with no `text` filter is vector-only,
a `text` filter with no embedding is keyword-only, both is the fusion. No flag
had to be invented to turn an arm off.

`hybrid` is the one to use once `app/search/hybrid.py` exists, because it is the
layer that computes `cn` and the evidence flags, and without `cn` the absence
and ambiguity checks cannot run at all. The harness offers it these keyword
arguments and passes whichever the signature accepts: `session`, `user_id`,
`q`/`query`/`text`, `service`/`table`/`corpus`, `limit`, `filters`, and the
filter keys directly (`since`, `until`, `from_email`, `attendee_emails`,
`mime_type`). A hit may be a mapping or an object; it is read for `ref`
(or `message_id`/`event_id`/`file_id`), `title`/`subject`/`name`, and
`scores.cn` / `scores.final` / `evidence`.

**A filter is never dropped.** If a row asks for `from=sarah@company.com` and
the bound callable has nowhere to put it, the harness raises. Dropping it would
widen the search, and a widened search scores *better* on a bad system.

### Intent

| backend | binds to |
|---|---|
| `live` (default) | `app.orchestrator.front_door` → first of `run`, `front_door`, `handle`, `try_front_door`, `match`, `classify`, `try_match`, `resolve`; then `app.orchestrator.route` → first of `plan`, `route`, `classify`, `build_plan`, `make_plan`, `run` |
| `api` | `POST /api/v1/query`, reading the `intent` SSE event |
| `canned` | `--dry-run` |

Those candidate lists live in one place — `SEARCH_MODULES`, `FRONT_DOOR_NAMES`
and `ROUTE_NAMES` in `__init__.py` — and are the same lists
`tests/conftest.load_any` uses for the unit suite, so the eval and the unit
tests cannot end up disagreeing about where a thing lives. The frozen clock is
shared too: `FIXED_NOW` here is `conftest.FROZEN_NOW`, which is the instant
every worked example in `docs/SAMPLE_QUERIES.md` is evaluated at.

A front-door result that is `None`, `False`, `{"type": "miss"}` or
`{"matched": false}` means "not mine", and the planner is called next. The
result — a plan, a front-door decision or a bare intent object — is normalised
to the labelled fields. Two of those fields are derived when the classifier
does not state them outright:

* **ambiguity** — true if the intent says so, if any step's op is `ask.user`,
  or if the result carries a `needs_input`.
* **prior-turn reference** — true if `intent.source` ends in `intent_carry`, if
  the entities contain any of `deixis`, `resolved_from`, `carried_from_run`,
  `ordinal`, `referent`, or if a step references `{{context.…}}`.

If `route.py` states these directly, the direct value wins.

---

## How each number is defined

Definitions matter more than the numbers, because most of the ways to inflate
one of these is a definition nobody wrote down.

**Precision@k** divides by `min(k, relevant)`, not by `k`. Most queries here
have one or two relevant documents per corpus, and a strict `/k` denominator
caps such a query at P@5 = 0.2 no matter how good the ranking is — a metric a
perfect system cannot score 1.0 on is not measuring the system. The strict
figure is computed and printed next to it, so the choice is visible rather than
assumed.

**Scoring is per (query, corpus)**, macro-averaged. A query with gold in Gmail
and Calendar is scored once per corpus. Corpora with no gold for that query are
not scored — an empty Drive list for a query that was never about Drive is not a
precision failure. The three corpora are not merged into one ranking, because
merging means comparing scores across corpora, and `cn` exists precisely because
those scores are not comparable.

**Each document counts once.** The same `(service, ref)` twice is dropped from
the ranking; a second hit resolving to a judgement already credited scores 0.
Without that, a chunked row that failed to collapse inflates precision and
Recall@10 can exceed 1.0. It did, on the first run.

**Absence rows** have no gold and are excluded from the means. They are scored
on whether the top hit stayed under `FLOOR_READ`, which needs `cn`; on a backend
that does not expose it the check reports as unavailable rather than guessing.

**Filter-only rows** — the two with no query text — are reported in their own
bucket. They exercise the metadata prefilter with no ranking involved. Note that
`mirror.hybrid_search` returns nothing when it has neither an embedding nor
query text, so if the retrieval layer routes a filter-only query through it,
those rows come back empty. That is a real gap and the harness shows it rather
than hiding it behind an average.

**Search latency** is the retrieval layer with no model in the loop: one
embedding plus the corpora searched concurrently. The first pass over the
dataset is discarded as a warm-up — connection setup and a cold embedding cache
describe a machine nobody is using — and reported separately as `cold_ms`.

**Query latency** is reported by class (`router` 0 calls, `template` 1,
`prose` 2) taken from what the run actually reported, never averaged across
classes, plus the time-to-first-pixel marks. Write-intent queries are excluded
by default: preparing one creates a Gmail draft and rows in `actions`, and a
benchmark should not do that sixty times. `--include-writes` opts in.

---

## What this harness does not measure

**Plausible-but-wrong retrieval.** Every metric here answers "is the right
document in the list". None of them answers "is the wrong document at the top
being believed". A confident wrong answer with the right document ranked second
still scores well. That failure mode is invisible to Precision@5 by
construction, and it is the one the design says it does not detect either.

**Whether the thresholds are right.** `FLOOR_READ`, `MARGIN` and `FLOOR_WRITE`
are hand-set. This dataset is far too small to fit them on — see the note at the
bottom of RESULTS.md for what calibrating them would actually take.

**Answer quality.** Nothing here reads the prose the synthesiser produces. A run
that retrieves perfectly and then writes a bad summary scores full marks.

**Real mailboxes.** ~42 planted documents built to exercise specific behaviours
is an upper bound on precision over fifteen years of real mail.

---

## Adding a row

Add a line to the dataset. Nothing else needs touching.

For an intent row, the field that takes thought is `flag_ambiguity`: it means
*this query cannot be answered without asking*, not *this query is vague*. If a
sensible person would act on it without checking, it is false. Getting that
wrong in the dataset makes the missed-ambiguity count meaningless, and that
count is the one that matters most.

For a relevance row, judge **every** document a correct system would return,
not only the one you were thinking of. An incomplete judgement scores a right
answer as an error, which is the most common way an eval set quietly stops
being useful.
