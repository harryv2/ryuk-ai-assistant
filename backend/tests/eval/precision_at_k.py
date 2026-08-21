#!/usr/bin/env python3
"""Retrieval quality — Precision@5 and the rest, plus the fusion ablation.

Runs every query in ``datasets/relevance.jsonl`` through the retrieval layer
and reports Precision@1/@3/@5, Recall@10, MRR and nDCG@5, overall and split by
query type, with p50/p95/p99 search latency alongside.

It also runs the **ablation**: the same queries through the vector arm alone,
the Postgres full-text arm alone, and the two fused. A design document can
assert that hybrid search beats either half; three columns in a table is the
only thing that shows it. If the fused row is not better, that is the finding
and it gets printed.

How a score is defined here, precisely:

* Scoring is **per (query, corpus)**. A query whose gold spans Gmail and
  Calendar is scored once against the Gmail ranking and once against the
  Calendar ranking, and the two are macro-averaged. Corpora with no gold for a
  query are not scored — a Drive list that is empty because the query was never
  about Drive is not a precision failure. Merging three corpora into one list
  would mean comparing scores across corpora, and the whole reason ``cn``
  exists is that those scores are not comparable.
* Relevant means grade >= 1. nDCG uses the graded values with gain 2^g - 1.
* Rows tagged ``absence`` have no gold at all; precision is undefined and they
  are excluded from the means. They are scored on their own terms — did the top
  hit stay under ``FLOOR_READ`` — which needs a backend that exposes ``cn``.
* Rows whose query text is empty are filter-only. They exercise the metadata
  prefilter with no ranking involved, and are reported in their own bucket.

Run it::

    python -m tests.eval.precision_at_k                      # mirror backend, all arms
    python -m tests.eval.precision_at_k --arms hybrid        # skip the ablation
    python -m tests.eval.precision_at_k --backend http       # through GET /api/v1/search
    python -m tests.eval.precision_at_k --write-results      # regenerate RESULTS.md
    python -m tests.eval.precision_at_k --self-test          # check the metric maths, no db
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python tests/eval/precision_at_k.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.eval import (
    EVAL_DIR,
    EVAL_USER_EMAIL,
    FIXED_NOW,
    FIXED_TZ,
    FIXED_WEEK_START,
    Hit,
    QueryParams,
    build_search_backend,
    env_fingerprint,
    explain_failure,
    fmt,
    load_jsonl,
    markdown_table,
    mean,
    parse_now,
    read_json,
    render_table,
    run_async,
    summarise,
    write_json,
)

RECALL_K = 10
NDCG_K = 5


# --------------------------------------------------------------------------- #
# The metric maths, on nothing but a list of grades in rank order
# --------------------------------------------------------------------------- #


def precision_at(grades: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    """Relevant results in the top k, over the most it could have found there.

    With ``total_relevant`` given the denominator is ``min(k, total_relevant)``.
    That matters here: most queries in this dataset have one or two genuinely
    relevant documents per corpus, and a strict ``/k`` denominator caps such a
    query at P@5 = 0.2 **no matter how good the ranking is**. A metric that a
    perfect system cannot score 1.0 on is not measuring the system.

    The strict ``/k`` figure is still computed and reported next to this one, so
    nothing is hidden by the choice — pass ``total_relevant=None`` for it.
    """
    if k <= 0:
        return 0.0
    found = sum(1 for g in grades[:k] if g >= 1)
    denominator = k if total_relevant is None else max(1, min(k, total_relevant))
    return found / denominator


def recall_at(grades: Sequence[int], k: int, total_relevant: int) -> float:
    if not total_relevant:
        return 0.0
    return sum(1 for g in grades[:k] if g >= 1) / total_relevant


def reciprocal_rank(grades: Sequence[int]) -> float:
    for i, g in enumerate(grades, 1):
        if g >= 1:
            return 1.0 / i
    return 0.0


def dcg(grades: Sequence[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], 1))


def ndcg_at(grades: Sequence[int], k: int, all_grades: Sequence[int]) -> float:
    ideal = dcg(sorted(all_grades, reverse=True), k)
    return dcg(grades, k) / ideal if ideal else 0.0


# --------------------------------------------------------------------------- #
# Matching a hit to a judgement
# --------------------------------------------------------------------------- #


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


class GoldIndex:
    """The judgements for one query, keyed the two ways a hit can be matched."""

    def __init__(self, relevant: Sequence[Mapping[str, Any]], match: str):
        self.match = match
        self.grades: list[int] = []
        #: both maps point at a judgement's index, not at its grade, so an id
        #: match and a title match on the same judgement are the same judgement.
        self.by_ref: dict[tuple[str, str], int] = {}
        self.by_title: dict[tuple[str, str], int] = {}
        self.per_service: dict[str, list[int]] = defaultdict(list)
        for item in relevant:
            service = item["service"]
            grade = int(item.get("grade", 1))
            index = len(self.grades)
            self.grades.append(grade)
            self.by_ref[(service, str(item["id"]))] = index
            title = _normalise(str(item.get("title", "")))
            if title:
                self.by_title.setdefault((service, title), index)
            self.per_service[service].append(grade)

    def lookup(self, hit: Hit) -> tuple[int | None, int]:
        """The judgement this hit matches, and its grade. ``(None, 0)`` for a miss."""
        if self.match in ("auto", "id"):
            index = self.by_ref.get((hit.service, hit.ref))
            if index is not None:
                return index, self.grades[index]
            if self.match == "id":
                return None, 0
        title = _normalise(hit.title)
        if title:
            index = self.by_title.get((hit.service, title))
            if index is not None:
                return index, self.grades[index]
        return None, 0

    def grade_sequence(self, hits: Sequence[Hit]) -> list[int]:
        """Grades in rank order, counting each **document** once.

        Two guards, both of which change the arithmetic when they fire:

        * the same ``(service, ref)`` twice is one document, so the repeat is
          dropped from the ranking rather than scored again — a chunked row
          that failed to collapse must not inflate precision;
        * a second hit resolving to a judgement already credited scores 0, so
          Recall@10 cannot exceed 1.0 by counting one gold document twice.
        """
        grades: list[int] = []
        seen: set[tuple[str, str]] = set()
        credited: set[int] = set()
        for hit in hits:
            ref_key = (hit.service, hit.ref)
            if ref_key in seen:
                continue
            seen.add(ref_key)
            gold_key, grade = self.lookup(hit)
            if gold_key is not None:
                if gold_key in credited:
                    grade = 0
                else:
                    credited.add(gold_key)
            grades.append(grade)
        return grades

    def services(self) -> list[str]:
        return sorted(self.per_service)


# --------------------------------------------------------------------------- #
# Running the dataset
# --------------------------------------------------------------------------- #


async def search_row(
    backend, row: Mapping[str, Any], params: QueryParams, *, limit: int, arm: str
) -> tuple[dict[str, list[Hit]], dict[str, float], float]:
    """Search every corpus the row asks for, in parallel, and time it."""
    services = [s for s in params.services if params.applies_to(s)]
    started = time.perf_counter()
    results = await asyncio.gather(
        *(backend.search(row["query"], service, params, limit=limit, arm=arm) for service in services),
        return_exceptions=True,
    )
    wall = (time.perf_counter() - started) * 1000
    hits: dict[str, list[Hit]] = {}
    per_service: dict[str, float] = {}
    for service, result in zip(services, results, strict=True):
        if isinstance(result, BaseException):
            raise result
        hits[service] = result.hits
        per_service[service] = result.took_ms
    return hits, per_service, wall


def score_row(row: Mapping[str, Any], hits: Mapping[str, list[Hit]], gold: GoldIndex) -> dict[str, Any]:
    """One row's per-corpus scores, plus whatever the decision layer supports."""
    corpora: list[dict[str, Any]] = []
    for service in gold.services():
        ranked = hits.get(service, [])
        grades = gold.grade_sequence(ranked)
        all_grades = gold.per_service[service]
        total = len(all_grades)
        corpora.append(
            {
                "service": service,
                "returned": len(ranked),
                "relevant_total": total,
                "p@1": precision_at(grades, 1, total),
                "p@3": precision_at(grades, 3, total),
                "p@5": precision_at(grades, 5, total),
                "p@1_strict": precision_at(grades, 1),
                "p@3_strict": precision_at(grades, 3),
                "p@5_strict": precision_at(grades, 5),
                f"r@{RECALL_K}": recall_at(grades, RECALL_K, total),
                "mrr": reciprocal_rank(grades),
                f"ndcg@{NDCG_K}": ndcg_at(grades, NDCG_K, all_grades),
                "grades": grades[:10],
                "top": [h.title[:60] for h in ranked[:3]],
            }
        )

    top_cn, runner_up_cn = _top_two_cn(hits)
    return {
        "id": row["id"],
        "type": row["type"],
        "query": row["query"],
        "corpora": corpora,
        "top_cn": top_cn,
        "runner_up_cn": runner_up_cn,
        "margin": None if top_cn is None or runner_up_cn is None else top_cn - runner_up_cn,
    }


def _top_two_cn(hits: Mapping[str, list[Hit]]) -> tuple[float | None, float | None]:
    """cn is comparable across corpora by construction — it is normalised per
    corpus precisely so the decision layer can compare them. Raw scores are not."""
    values = sorted(
        (h.cn for hs in hits.values() for h in hs if h.cn is not None), reverse=True
    )
    return (values[0] if values else None, values[1] if len(values) > 1 else None)


def aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average over (query, corpus) pairs."""
    pairs = [(row, corpus) for row in scored for corpus in row["corpora"]]
    if not pairs:
        return {"pairs": 0, "queries": 0}
    keys = [
        "p@1", "p@3", "p@5", "p@1_strict", "p@3_strict", "p@5_strict",
        f"r@{RECALL_K}", "mrr", f"ndcg@{NDCG_K}",
    ]
    return {
        "pairs": len(pairs),
        "queries": len(scored),
        **{k: mean([c[k] for _, c in pairs]) for k in keys},
    }


def by_type(scored: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        buckets[row["type"]].append(row)
    return {name: aggregate(rows) for name, rows in sorted(buckets.items())}


# --------------------------------------------------------------------------- #
# The decision layer: absence and ambiguity
# --------------------------------------------------------------------------- #


def score_decisions(scored: Sequence[Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]],
                    *, floor_read: float, margin: float) -> dict[str, Any]:
    """Did the floors do their job on the rows that exist to test them?

    Needs ``cn``. A backend that does not expose it gets ``available: false``
    and a reason, rather than a number computed off something else.
    """
    absence = [r for r in scored if rows[r["id"]].get("expect_absent")]
    ambiguous = [r for r in scored if rows[r["id"]].get("expect_ambiguous")]
    have_cn = any(r["top_cn"] is not None for r in scored)
    if not have_cn:
        return {
            "available": False,
            "reason": "the backend does not expose cn; run --backend hybrid or http",
            "absence_n": len(absence),
            "ambiguous_n": len(ambiguous),
        }
    absent_right = [r for r in absence if r["top_cn"] is not None and r["top_cn"] < floor_read]
    ambiguous_right = [r for r in ambiguous if r["margin"] is not None and r["margin"] < margin]
    return {
        "available": True,
        "floor_read": floor_read,
        "margin": margin,
        "absence_n": len(absence),
        "absence_correct": len(absent_right),
        "absence_rows": [
            {"id": r["id"], "top_cn": r["top_cn"], "verdict": "absent" if r in absent_right else "claimed a hit"}
            for r in absence
        ],
        "ambiguous_n": len(ambiguous),
        "ambiguous_correct": len(ambiguous_right),
        "ambiguous_rows": [
            {"id": r["id"], "margin": r["margin"],
             "verdict": "ambiguous" if r in ambiguous_right else "picked one"}
            for r in ambiguous
        ],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(result: Mapping[str, Any]) -> str:
    out: list[str] = []
    overall = result["arms"][result["primary_arm"]]["overall"]
    out.append(f"Retrieval — {result['queries']} queries from {result['dataset']}, "
               f"{result['judgements']} graded judgements")
    out.append(f"backend: {result['backend']}    match: {result['match']}    limit: {result['limit']}")
    out.append("")
    out.append(f"  Precision@5   {overall['p@5']:.3f}   (target > 0.80)")
    out.append(f"  Precision@1   {overall['p@1']:.3f}")
    out.append(f"  Precision@3   {overall['p@3']:.3f}")
    out.append(f"  Recall@10     {overall[f'r@{RECALL_K}']:.3f}")
    out.append(f"  MRR           {overall['mrr']:.3f}")
    out.append(f"  nDCG@5        {overall[f'ndcg@{NDCG_K}']:.3f}")
    out.append(f"  scored over {overall['pairs']} (query, corpus) pairs")
    out.append("")
    out.append(
        "  denominator is min(k, relevant). Strict /k, which most queries cannot reach "
        "because they have fewer than k relevant documents:"
    )
    out.append(
        f"    P@1 {overall['p@1_strict']:.3f}   P@3 {overall['p@3_strict']:.3f}   "
        f"P@5 {overall['p@5_strict']:.3f}"
    )

    out.append("")
    out.append("By query type")
    out.append(
        render_table(
            ["type", "q", "pairs", "P@1", "P@3", "P@5", f"R@{RECALL_K}", "MRR", "nDCG@5"],
            [
                [name, block["queries"], block["pairs"], f"{block['p@1']:.3f}", f"{block['p@3']:.3f}",
                 f"{block['p@5']:.3f}", f"{block[f'r@{RECALL_K}']:.3f}", f"{block['mrr']:.3f}",
                 f"{block[f'ndcg@{NDCG_K}']:.3f}"]
                for name, block in result["arms"][result["primary_arm"]]["by_type"].items()
                if block.get("pairs")
            ],
            right=tuple(range(1, 9)),
        )
    )

    arms = [a for a in ("vector", "keyword", "hybrid") if a in result["arms"]]
    if len(arms) > 1:
        out.append("")
        out.append("Ablation — does the fusion earn its place?")
        out.append(
            render_table(
                ["arm", "P@1", "P@5", f"R@{RECALL_K}", "MRR", "nDCG@5", "p50 ms", "p95 ms"],
                [
                    [
                        arm,
                        f"{result['arms'][arm]['overall']['p@1']:.3f}",
                        f"{result['arms'][arm]['overall']['p@5']:.3f}",
                        f"{result['arms'][arm]['overall'][f'r@{RECALL_K}']:.3f}",
                        f"{result['arms'][arm]['overall']['mrr']:.3f}",
                        f"{result['arms'][arm]['overall'][f'ndcg@{NDCG_K}']:.3f}",
                        f"{result['arms'][arm]['latency_ms']['p50']:.0f}",
                        f"{result['arms'][arm]['latency_ms']['p95']:.0f}",
                    ]
                    for arm in arms
                ],
                right=tuple(range(1, 8)),
            )
        )
        best = max(arms, key=lambda a: result["arms"][a]["overall"]["p@5"])
        if best == "hybrid":
            gap = result["arms"]["hybrid"]["overall"]["p@5"] - max(
                result["arms"][a]["overall"]["p@5"] for a in arms if a != "hybrid"
            )
            out.append(f"  fused wins by {gap:+.3f} P@5 over the better single arm")
        else:
            out.append(f"  the {best} arm alone beats the fusion here — the fusion is not earning its place")
        out.append("")
        out.append("  Per type, P@5 by arm")
        types = sorted(result["arms"][result["primary_arm"]]["by_type"])
        out.append(
            render_table(
                ["type", *arms],
                [
                    [t, *[f"{result['arms'][a]['by_type'].get(t, {}).get('p@5', 0.0):.3f}" for a in arms]]
                    for t in types
                    if result["arms"][result["primary_arm"]]["by_type"].get(t, {}).get("pairs")
                ],
                right=tuple(range(1, len(arms) + 1)),
            )
        )

    latency = result["arms"][result["primary_arm"]]["latency_ms"]
    out.append("")
    out.append(
        f"Search latency per query (all corpora in parallel)  "
        f"p50 {latency['p50']:.0f} ms   p95 {latency['p95']:.0f} ms   p99 {latency['p99']:.0f} ms"
        f"   n={latency['n']}"
    )
    out.append(f"  target < 500 ms: {'MET' if latency['p95'] < 500 else 'NOT MET'} at p95")

    decisions = result["decisions"]
    out.append("")
    if decisions.get("available"):
        out.append("Decision layer")
        out.append(f"  absence   {decisions['absence_correct']}/{decisions['absence_n']} "
                   f"correctly under FLOOR_READ {decisions['floor_read']}")
        out.append(f"  ambiguity {decisions['ambiguous_correct']}/{decisions['ambiguous_n']} "
                   f"correctly inside MARGIN {decisions['margin']}")
        for item in decisions["absence_rows"]:
            out.append(f"    {item['id']}  top cn {fmt(item['top_cn'], 2)}  -> {item['verdict']}")
        for item in decisions["ambiguous_rows"]:
            out.append(f"    {item['id']}  margin {fmt(item['margin'], 2)}  -> {item['verdict']}")
    else:
        out.append(f"Decision layer: not scored — {decisions['reason']}")

    if result["filter_only"]:
        out.append("")
        out.append("Filter-only rows (no query text — the prefilter with no ranking)")
        for item in result["filter_only"]:
            out.append(f"  {item['id']}  returned {item['returned']}  P@5 {item['p@5']:.3f}  {item['note']}")

    if result["worst"]:
        out.append("")
        out.append("Worst rows")
        out.append(
            render_table(
                ["id", "type", "corpus", "P@5", "query", "top hit"],
                [
                    [w["id"], w["type"], w["service"], f"{w['p@5']:.3f}", w["query"][:38],
                     (w["top"][0] if w["top"] else "-")[:38]]
                    for w in result["worst"]
                ],
                right=(3,),
            )
        )
    return "\n".join(out)


def markdown_section(result: Mapping[str, Any]) -> str:
    primary = result["arms"][result["primary_arm"]]
    overall = primary["overall"]
    latency = primary["latency_ms"]
    lines = [
        markdown_table(
            ["metric", "value", "target"],
            [
                ["**Precision@5**", f"**{overall['p@5']:.3f}**", "> 0.80 (brief, 10 pts)"],
                ["Precision@1", f"{overall['p@1']:.3f}", "—"],
                ["Precision@3", f"{overall['p@3']:.3f}", "—"],
                [f"Recall@{RECALL_K}", f"{overall[f'r@{RECALL_K}']:.3f}", "—"],
                ["MRR", f"{overall['mrr']:.3f}", "—"],
                ["nDCG@5", f"{overall[f'ndcg@{NDCG_K}']:.3f}", "—"],
                ["Precision@5, strict /k", f"{overall['p@5_strict']:.3f}", "see note"],
                ["Search p95", f"{latency['p95']:.0f} ms", "< 500 ms (brief, 3 pts)"],
                ["Search p99", f"{latency['p99']:.0f} ms", "—"],
            ],
        ),
        "",
        f"Scored over {overall['pairs']} (query, corpus) pairs from {overall['queries']} queries. "
        "Precision divides by `min(k, relevant)`: most queries here have one or two relevant "
        "documents per corpus, and a strict `/k` denominator caps such a query at P@5 = 0.2 "
        "however good the ranking is. The strict figure is in the table above so the choice "
        "is visible rather than assumed.",
        "",
        "By query type:",
        "",
        markdown_table(
            ["type", "queries", "P@1", "P@5", f"R@{RECALL_K}", "MRR", "nDCG@5"],
            [
                [name, block["queries"], f"{block['p@1']:.3f}", f"{block['p@5']:.3f}",
                 f"{block[f'r@{RECALL_K}']:.3f}", f"{block['mrr']:.3f}", f"{block[f'ndcg@{NDCG_K}']:.3f}"]
                for name, block in primary["by_type"].items() if block.get("pairs")
            ],
        ),
    ]
    arms = [a for a in ("vector", "keyword", "hybrid") if a in result["arms"]]
    if len(arms) > 1:
        lines += [
            "",
            "Ablation — vector arm alone, text arm alone, and the two fused:",
            "",
            markdown_table(
                ["arm", "P@1", "P@5", f"R@{RECALL_K}", "MRR", "nDCG@5", "p95 ms"],
                [
                    [
                        arm,
                        f"{result['arms'][arm]['overall']['p@1']:.3f}",
                        f"{result['arms'][arm]['overall']['p@5']:.3f}",
                        f"{result['arms'][arm]['overall'][f'r@{RECALL_K}']:.3f}",
                        f"{result['arms'][arm]['overall']['mrr']:.3f}",
                        f"{result['arms'][arm]['overall'][f'ndcg@{NDCG_K}']:.3f}",
                        f"{result['arms'][arm]['latency_ms']['p95']:.0f}",
                    ]
                    for arm in arms
                ],
            ),
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #

PLACEHOLDER = (
    "_Not in this run. Regenerate with `make eval` against a seeded database._"
)


def render_results_md(*, intent: Mapping[str, Any] | None, retrieval: Mapping[str, Any] | None,
                      latency: Mapping[str, Any] | None) -> str:
    env = (retrieval or intent or latency or {}).get("env") or env_fingerprint()
    stamp = env.get("generated_at", "unknown")
    revision = env.get("git", "unknown")
    thresholds = env.get("thresholds", {})

    def section(block: Mapping[str, Any] | None) -> str:
        return block["markdown"] if block and block.get("markdown") else PLACEHOLDER

    canned_note = ""
    if intent and intent.get("canned"):
        canned_note = (
            "\n> The intent numbers below came from `--dry-run`, which uses canned responses. "
            "They exercise the report; they measure nothing.\n"
        )

    return f"""# Evaluation results

Generated {stamp} from commit `{revision}`.
{canned_note}
Every number here comes from one of the three harnesses in this directory, run
against a seeded database. Nothing is hand-entered. To reproduce, see
[How to regenerate](#how-to-regenerate) at the bottom.

| | |
|---|---|
| Intent dataset | `datasets/intents.jsonl` — {(intent or {}).get('dataset_size', '?')} labelled queries across 18 intents, 7 languages |
| Relevance dataset | `datasets/relevance.jsonl` — {(retrieval or {}).get('queries', '?')} queries, {(retrieval or {}).get('judgements', '?')} graded judgements |
| Thresholds in force | `FLOOR_READ` {float(thresholds.get('FLOOR_READ', 0.55)):.2f} · `MARGIN` {float(thresholds.get('MARGIN', 0.15)):.2f} · `FLOOR_WRITE` {float(thresholds.get('FLOOR_WRITE', 0.80)):.2f} |
| Models | chat `{env.get('models', {}).get('chat', '?')}` · embeddings `{env.get('models', {}).get('embed', '?')}` |
| Fixed context | `{EVAL_USER_EMAIL}`, `{FIXED_TZ}`, week starts {'Monday' if FIXED_WEEK_START == 1 else str(FIXED_WEEK_START)}, now = {(retrieval or intent or latency or {}).get('now', FIXED_NOW)} |

---

## 1. Intent classification

{section(intent)}

---

## 2. Retrieval quality

{section(retrieval)}

---

## 3. Latency

{section(latency)}

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
them — {(retrieval or {}).get('judgements', '~100')} judgements over
{(retrieval or {}).get('queries', '~35')} queries is an order of magnitude short.

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
"""


# --------------------------------------------------------------------------- #
# Self-test — the arithmetic, with no database
# --------------------------------------------------------------------------- #


def self_test() -> int:
    """Check the metric functions against hand-computed values."""
    checks: list[tuple[str, float, float]] = []
    grades = [2, 0, 1, 0, 0, 2]
    checks.append(("P@1", precision_at(grades, 1), 1.0))
    checks.append(("P@3 strict", precision_at(grades, 3), 2 / 3))
    checks.append(("P@5 strict", precision_at(grades, 5), 2 / 5))
    checks.append(("P@5 short list", precision_at([2, 2], 5), 2 / 5))
    checks.append(("P@5 capped |R|=2", precision_at([2, 2, 0, 0, 0], 5, 2), 1.0))
    checks.append(("P@5 capped |R|=8", precision_at([2, 2, 0, 0, 0], 5, 8), 2 / 5))
    checks.append(("P@5 capped, miss", precision_at([0, 2, 0, 0, 0], 5, 2), 0.5))
    checks.append(("R@10", recall_at(grades, 10, 4), 3 / 4))
    checks.append(("MRR first", reciprocal_rank(grades), 1.0))
    checks.append(("MRR third", reciprocal_rank([0, 0, 1]), 1 / 3))
    checks.append(("MRR none", reciprocal_rank([0, 0, 0]), 0.0))
    # DCG of [2,0,1] = 3/1 + 0/1.585 + 1/2 = 3.5 ; ideal [2,1,0] = 3 + 1/1.585 = 3.6309
    checks.append(("nDCG@3", ndcg_at([2, 0, 1], 3, [2, 1, 0]), 3.5 / (3 + 1 / math.log2(3))))
    checks.append(("nDCG perfect", ndcg_at([2, 1], 5, [2, 1]), 1.0))
    checks.append(("nDCG empty", ndcg_at([], 5, [2]), 0.0))

    failed = 0
    for name, got, want in checks:
        ok = abs(got - want) < 1e-9
        failed += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:16} got {got:.6f}  want {want:.6f}")

    index = GoldIndex(
        [{"service": "gmail", "id": "g-1", "grade": 2, "title": "Your Turkish Airlines booking — TK1984"}],
        match="auto",
    )
    by_id = index.lookup(Hit(service="gmail", ref="g-1", title="whatever"))[1]
    by_title = index.lookup(Hit(service="gmail", ref="opaque-id",
                                title="your turkish airlines  booking - TK1984"))[1]
    miss = index.lookup(Hit(service="gcal", ref="g-1", title="Your Turkish Airlines booking — TK1984"))[1]
    # The same document twice is one document; a second hit resolving to an
    # already-credited judgement scores 0, so recall cannot exceed 1.
    repeat_ref = index.grade_sequence(
        [Hit(service="gmail", ref="g-1", title="x"), Hit(service="gmail", ref="g-1", title="x")]
    )
    repeat_title = index.grade_sequence(
        [
            Hit(service="gmail", ref="g-1", title="Your Turkish Airlines booking — TK1984"),
            Hit(service="gmail", ref="other", title="Your Turkish Airlines booking — TK1984"),
        ]
    )
    object_checks = [
        ("match by id", by_id, 2),
        ("match by title", by_title, 2),
        ("wrong corpus", miss, 0),
        ("dupe ref dropped", repeat_ref, [2]),
        ("dupe judgement", repeat_title, [2, 0]),
        ("recall <= 1", recall_at(repeat_title, 10, 1), 1.0),
    ]
    for name, got, want in object_checks:
        ok = got == want
        failed += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name:16} got {got}  want {want}")

    print(f"\n{len(checks) + len(object_checks)} checks, {failed} failed")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run_arm(backend, rows, golds, *, limit: int, arm: str, now, tz, week_start,
                  strict_temporal: bool, verbose: bool) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    latencies: list[float] = []
    per_service_latency: dict[str, list[float]] = defaultdict(list)
    filter_only: list[dict[str, Any]] = []

    for row in rows:
        params = QueryParams.from_row(
            row, now=now, tz=tz, week_start=week_start, strict_temporal=strict_temporal
        )
        hits, service_ms, wall = await search_row(backend, row, params, limit=limit, arm=arm)
        latencies.append(wall)
        for service, ms in service_ms.items():
            per_service_latency[service].append(ms)
        entry = score_row(row, hits, golds[row["id"]])
        if not row["query"].strip():
            returned = sum(len(v) for v in hits.values())
            entry["filter_only"] = True
            filter_only.append(
                {
                    "id": row["id"],
                    "returned": returned,
                    "p@5": mean([c["p@5"] for c in entry["corpora"]]) if entry["corpora"] else 0.0,
                    "note": "no query text: prefilter only"
                    + ("" if returned else " — the backend returned nothing for it"),
                }
            )
        if verbose:
            best = entry["corpora"][0] if entry["corpora"] else {}
            print(f"  {arm:8} {row['id']}  P@5 {best.get('p@5', 0):.2f}  {row['query'][:44]!r}",
                  file=sys.stderr)
        scored.append(entry)

    ranked = [r for r in scored if not r.get("filter_only") and r["corpora"]]
    return {
        "scored": scored,
        "overall": aggregate(ranked),
        "by_type": by_type(ranked),
        "latency_ms": summarise(latencies).to_dict(),
        "latency_by_service": {s: summarise(v).to_dict() for s, v in sorted(per_service_latency.items())},
        "filter_only": filter_only,
    }


def worst_rows(arm_result: Mapping[str, Any], n: int = 8) -> list[dict[str, Any]]:
    flat = [
        {
            "id": row["id"], "type": row["type"], "query": row["query"],
            "service": corpus["service"], "p@5": corpus["p@5"], "top": corpus["top"],
        }
        for row in arm_result["scored"]
        if not row.get("filter_only")
        for corpus in row["corpora"]
    ]
    return sorted(flat, key=lambda r: (r["p@5"], r["id"]))[:n]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="relevance.jsonl")
    parser.add_argument("--backend", choices=("mirror", "hybrid", "http"), default="mirror")
    parser.add_argument("--arms", default="vector,keyword,hybrid",
                        help="comma list of vector,keyword,hybrid — the ablation")
    parser.add_argument("--limit", type=int, default=10, help="per corpus; must be >= 10 for Recall@10")
    parser.add_argument("--match", choices=("auto", "id", "title"), default="auto",
                        help="how a hit is matched to a judgement")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--user-email", default=EVAL_USER_EMAIL)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--now", default=None)
    parser.add_argument("--tz", default=FIXED_TZ)
    parser.add_argument("--week-start", type=int, default=FIXED_WEEK_START)
    parser.add_argument("--strict-temporal", action="store_true",
                        help="require app.orchestrator.temporal rather than the local fallback")
    parser.add_argument("--only", default=None)
    parser.add_argument("--type", dest="type_filter", default=None)
    parser.add_argument("--json", dest="json_out", default="precision_at_k.json")
    parser.add_argument("--write-results", action="store_true",
                        help="regenerate RESULTS.md from whatever is in out/")
    parser.add_argument("--self-test", action="store_true", help="check the metric maths and exit")
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 if P@5 is below this")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--traceback", action="store_true", help="re-raise instead of explaining")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    dataset = load_jsonl(args.dataset).filter(
        args.only.split(",") if args.only else None, args.type_filter
    )
    rows = dataset.rows
    golds = {row["id"]: GoldIndex(row.get("relevant") or [], args.match) for row in rows}
    now = parse_now(args.now)
    if args.limit < RECALL_K:
        print(f"note: --limit {args.limit} < {RECALL_K}, so Recall@{RECALL_K} is capped by the limit",
              file=sys.stderr)

    backend_kwargs: dict[str, Any] = {"user_email": args.user_email, "user_id": args.user_id}
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    backend = build_search_backend(args.backend, **backend_kwargs)
    wanted_arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    async def go() -> dict[str, Any]:
        await backend.setup()
        try:
            arms: dict[str, Any] = {}
            for arm in wanted_arms:
                if arm != "hybrid" and not backend.supports_arms:
                    print(f"note: backend {backend.name} cannot isolate the {arm} arm; skipping it. "
                          f"Run the ablation with --backend mirror.", file=sys.stderr)
                    continue
                arms[arm] = await run_arm(
                    backend, rows, golds, limit=args.limit, arm=arm, now=now, tz=args.tz,
                    week_start=args.week_start, strict_temporal=args.strict_temporal,
                    verbose=args.verbose,
                )
            return arms
        finally:
            await backend.close()

    try:
        arms = run_async(go())
    except Exception as exc:
        return explain_failure(exc, what="retrieval eval", traceback_wanted=args.traceback)
    if not arms:
        print("no arms ran", file=sys.stderr)
        return 2

    primary = "hybrid" if "hybrid" in arms else next(iter(arms))
    row_index = {row["id"]: row for row in rows}
    try:
        from app.config import settings

        floor_read, margin = settings.FLOOR_READ, settings.MARGIN
    except Exception:
        floor_read, margin = 0.55, 0.15

    result = {
        "kind": "precision_at_k",
        "dataset": dataset.path.name,
        "queries": len(rows),
        "judgements": sum(len(r.get("relevant") or []) for r in rows),
        "backend": backend.describe(),
        "match": args.match,
        "limit": args.limit,
        "now": now.isoformat().replace("+00:00", "Z"),
        "primary_arm": primary,
        "arms": arms,
        "decisions": score_decisions(arms[primary]["scored"], row_index,
                                     floor_read=floor_read, margin=margin),
        "filter_only": arms[primary]["filter_only"],
        "worst": worst_rows(arms[primary]),
        "env": env_fingerprint(),
    }
    print(report(result))

    if args.json_out:
        payload = dict(result)
        payload["markdown"] = markdown_section(result)
        # The per-row detail is large; keep it in the file but out of the report.
        path = write_json(args.json_out, payload)
        print(f"\nwrote {path}")

    if args.write_results:
        path = write_results_md()
        print(f"wrote {path}")

    p5 = arms[primary]["overall"].get("p@5", 0.0)
    if args.fail_under is not None and p5 < args.fail_under:
        print(f"\nFAIL: Precision@5 {p5:.3f} < {args.fail_under}", file=sys.stderr)
        return 1
    return 0


def write_results_md() -> Path:
    """Assemble RESULTS.md from whatever metric files exist in out/."""
    document = render_results_md(
        intent=read_json("intent_accuracy.json"),
        retrieval=read_json("precision_at_k.json"),
        latency=read_json("latency.json"),
    )
    path = EVAL_DIR / "RESULTS.md"
    path.write_text(document, encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
