# Evaluation results

Generated 2026-08-21T18:13:31Z from commit `82bfb19`.

Every number here comes from one of the three harnesses in this directory, run
against a seeded database. Nothing is hand-entered. To reproduce, see
[How to regenerate](#how-to-regenerate) at the bottom.

| | |
|---|---|
| Intent dataset | `datasets/intents.jsonl` — 63 labelled queries across 18 intents, 7 languages |
| Relevance dataset | `datasets/relevance.jsonl` — 35 queries, 101 graded judgements |
| Thresholds in force | `FLOOR_READ` 0.55 · `MARGIN` 0.15 · `FLOOR_WRITE` 0.80 |
| Models | chat `gpt-5.6-terra` · embeddings `text-embedding-3-small` |
| Fixed context | `demo@example.com`, `America/New_York`, week starts Monday, now = 2026-08-20T13:12:04Z |

---

## 1. Intent classification

| metric | value | what it means |
|---|---|---|
| Accuracy | 0.222 | 14/63 exact intent-name matches |
| Macro F1 | 0.127 | unweighted across intents, so rare intents count |
| Service-set F1 | 0.879 | micro over service labels |
| Service-set exact | 0.778 | the whole set right |
| Entity-key F1 | 0.000 | keys the planner produced vs labelled |
| Prior-turn detection | 0.841 | did it know the query leans on the conversation |
| Ambiguities missed | 2 | of 4 — the dangerous direction |
| False alarms | 5 | asked when it did not need to — the cheap direction |

Per intent:

| intent | n | precision | recall | F1 |
|---|---|---|---|---|
| availability | 2 | 0.000 | 0.000 | 0.000 |
| calendar_list | 9 | 1.000 | 0.778 | 0.875 |
| cancel_flight | 4 | 0.000 | 0.000 | 0.000 |
| chit_chat | 2 | 0.000 | 0.000 | 0.000 |
| conflict_check | 3 | 0.000 | 0.000 | 0.000 |
| digest | 4 | 0.000 | 0.000 | 0.000 |
| drive_filter | 4 | 0.000 | 0.000 | 0.000 |
| drive_search | 3 | 0.000 | 0.000 | 0.000 |
| email_compose | 3 | 0.000 | 0.000 | 0.000 |
| email_detail | 3 | 0.000 | 0.000 | 0.000 |
| email_search | 6 | 1.000 | 0.833 | 0.909 |
| event_create | 2 | 0.000 | 0.000 | 0.000 |
| file_share | 2 | 0.000 | 0.000 | 0.000 |
| meeting_prep | 4 | 0.000 | 0.000 | 0.000 |
| move_event | 4 | 0.000 | 0.000 | 0.000 |
| reschedule_and_notify | 2 | 0.000 | 0.000 | 0.000 |
| ui_verb | 3 | 0.000 | 0.000 | 0.000 |
| unsupported | 3 | 0.400 | 0.667 | 0.500 |

Worst confusions:

| expected | got | n |
|---|---|---|
| ui_verb | <none> | 3 |
| email_detail | <none> | 2 |
| meeting_prep | prepare_acme_meeting | 2 |
| chit_chat | unsupported | 2 |
| calendar_list | <none> | 1 |
| calendar_list | unsupported | 1 |

---

## 2. Retrieval quality

| metric | value | target |
|---|---|---|
| **Precision@5** | **0.000** | > 0.80 (brief, 10 pts) |
| Precision@1 | 0.000 | — |
| Precision@3 | 0.000 | — |
| Recall@10 | 0.000 | — |
| MRR | 0.000 | — |
| nDCG@5 | 0.000 | — |
| Precision@5, strict /k | 0.000 | see note |
| Search p95 | 14 ms | < 500 ms (brief, 3 pts) |
| Search p99 | 18 ms | — |

Scored over 43 (query, corpus) pairs from 31 queries. Precision divides by `min(k, relevant)`: most queries here have one or two relevant documents per corpus, and a strict `/k` denominator caps such a query at P@5 = 0.2 however good the ranking is. The strict figure is in the table above so the choice is visible rather than assumed.

By query type:

| type | queries | P@1 | P@5 | R@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| attendee | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cross_lingual | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| date_window | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mime | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| multi_service | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| semantic | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sender | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Ablation — vector arm alone, text arm alone, and the two fused:

| arm | P@1 | P@5 | R@10 | MRR | nDCG@5 | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 21 |
| keyword | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 9 |
| hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 14 |

---

## 3. Latency

**Search path** — one embedding plus the hybrid searches, no model in the loop.

| measure | p50 | p95 | p99 | target |
|---|---|---|---|---|
| per query, all corpora in parallel | 6 ms | 19 ms | 20 ms | < 500 ms |

Where the time goes:

| stage | p50 | p95 |
|---|---|---|
| embed_ms | 0.0 ms | 0.0 ms |
| sql_ms | 5.0 ms | 15.9 ms |

**Full query path** — POST to the answer, by class. Averaging these classes together would describe nothing, so they are not averaged.

| class | runs | p50 | p95 | p99 |
|---|---|---|---|---|

Read class (router + template) p95 **0 ms** against a 2 s target. Two-call prose reads do not fit in 2 s and are not claimed to.

---

## How to regenerate

```bash
docker compose up -d                 # postgres, redis, api, worker, beat
make migrate                         # schema
python -m scripts.seed --user demo@example.com   # the fixture corpus
make eval                            # the three harnesses, in order
```

`make eval` runs:

```bash
python -m tests.eval.intent_accuracy                 # -> out/intent_accuracy.json
python -m tests.eval.precision_at_k                  # -> out/precision_at_k.json
python -m tests.eval.latency                         # -> out/latency.json
python -m tests.eval.precision_at_k --write-results  # -> RESULTS.md (this file)
```

The last command assembles this page from the three JSON files. A section whose
file is missing renders as a "not in this run" placeholder rather than as stale
numbers from a previous run.

Preconditions, each of which the harness complains about specifically when it
is not met:

* a user row for `demo@example.com` with the fixture corpus embedded — no
  `sync_*` row for that user may have `embedding IS NULL`. `tests/eval/README.md`
  lists exactly what the seeder must plant;
* `OPENAI_API_KEY` set, for the query embeddings;
* the API running for `latency.py --path query` and for `--backend http`, with
  `EVAL_SESSION_COOKIE` set to an authenticated session.

Without any of that:

```bash
python -m tests.eval.intent_accuracy --dry-run    # canned responses, labelled as such
python -m tests.eval.precision_at_k --self-test   # assertions on the metric maths
python -m tests.eval.precision_at_k --arms keyword  # the text arm alone, no embeddings
```

---

## What these numbers do not say

**The thresholds are hand-set, not calibrated.** `FLOOR_READ` 0.55, `MARGIN`
0.15 and `FLOOR_WRITE` 0.80 were chosen by looking at score distributions on the
seed corpus. They were not fitted to this dataset, and this dataset cannot fit
them — 101 judgements over
35 queries is an order of magnitude short.

Calibrating them properly means four things, and none of them is a code change:

1. **Label about 500 (query, candidate) pairs per corpus**, not per system.
   Mail, events and files have visibly different `cn` distributions — mail
   bodies are long and varied, event titles are short and formulaic, filenames
   are almost identifiers — so one number over three corpora is a compromise
   between three different right answers.
2. **Pick a precision target per decision, not per system.** `FLOOR_READ`
   governs "is this worth showing", where a false positive costs the user a
   glance. `FLOOR_WRITE` governs "may this anchor a side effect", where a false
   positive sends an email about the wrong booking. Those two should not be
   fitted to the same target, and the gap between 0.55 and 0.80 is currently a
   guess at how much wider the second one should be.
3. **Hold out a split.** Fitting thresholds on the same set you then report
   precision on produces a number that means nothing. The queries in
   `relevance.jsonl` are entirely a test set; a calibration set has to be
   collected separately.
4. **Re-fit on a schedule.** `cn` is z-scored against a user's own score
   distribution, so the thresholds are coupled to corpus size and shape. A
   threshold fitted on a 400-message mailbox is not the same threshold on a
   40,000-message one, and nothing currently notices that.

That is roughly a day of labelling and an afternoon of fitting, and
`docs/DESIGN.md` §9 lists it as the single highest-value quality change
available. It has not been done, and no number on this page should be read as
if it had.

**Plausible-but-wrong retrieval is not measured here.** These metrics say how
often the right document is in the list. They say nothing about the case where
one confident wrong document is at the top and the user believes it. That
failure mode is invisible to Precision@5 by construction, because a wrong
answer with the right document ranked second still scores well — and it is the
same failure the design says it cannot detect either. The mitigation is display,
not detection: every answer names its sources.

**The corpus is seeded, not real.** ~42 planted documents built to exercise
specific retrieval behaviours, plus filler. Precision on a curated corpus is an
upper bound on precision over a real mailbox with fifteen years of newsletters
in it.

**Latency is measured on one machine.** Docker on a laptop, warm caches, no
network between the API and Postgres. The *shape* is informative — where the
milliseconds go, whether the fan-out is genuinely concurrent — and the absolute
numbers are not a production SLO.

**Answer quality is not scored at all.** Nothing here reads the prose the
synthesiser produces. A run that retrieves perfectly and then writes a bad
summary gets full marks on this page.
