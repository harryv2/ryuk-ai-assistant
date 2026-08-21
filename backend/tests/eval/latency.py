#!/usr/bin/env python3
"""Latency — the search path and the full query path, measured separately.

Two claims in the brief have two different answers, so they get two different
measurements and are never averaged together:

* **Search under 500 ms.** The retrieval layer on its own: one embedding, three
  hybrid searches in parallel over the mirror. Measured by ``--path search``.
* **P95 under 2 s for a read.** The whole trip a user takes: POST the query,
  follow the event stream, stop when the run completes. Measured by
  ``--path query``.

Averaging those two produces a number that describes nothing. Averaging a
rule-routed read (0 LLM calls) with a two-call prose read produces a number
that describes nothing either, so the query path reports **by class**:

| class | what it is | LLM calls |
|---|---|---|
| ``router`` | the front door answered — rule router, chit-chat, a UI verb | 0 |
| ``template`` | one planner call, the answer renders from a template or a card | 1 |
| ``prose`` | planner plus a streamed prose answer | 2 |

The class comes from what the run actually reported, not from what the dataset
guessed, so a query that was supposed to be free and cost a call shows up in
the class it landed in.

The query path also reports **time to first meaningful pixel** — the marks
DESIGN.md §8.1 publishes as ``ttfp_seconds``. A prose answer that starts at
1.6 s and finishes at 3.1 s reads as fast; a blank screen for 800 ms reads as
slow, and only one of those is visible in a total.

Run it::

    python -m tests.eval.latency --path search              # retrieval only, no API
    python -m tests.eval.latency --path query               # through the running API
    python -m tests.eval.latency --path both --repeat 5

Writes are **excluded from the query path by default**: preparing one creates a
Gmail draft and rows in ``actions``, which is a side effect on a real mailbox
and not something a benchmark should do sixty times. ``--include-writes`` opts
in.
"""

from __future__ import annotations

import argparse
import os
import asyncio
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python tests/eval/latency.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.eval import (
    DEFAULT_API_BASE,
    EVAL_USER_EMAIL,
    FIXED_TZ,
    FIXED_WEEK_START,
    QueryParams,
    build_search_backend,
    env_fingerprint,
    explain_failure,
    load_jsonl,
    markdown_table,
    mean,
    parse_now,
    render_table,
    run_async,
    run_query,
    summarise,
    write_json,
)

#: The marks a run publishes, in the order they can occur. Deltas between
#: consecutive marks are where the time actually went.
MARKS = (
    "accepted",
    "run.started",
    "probe.done",
    "intent",
    "input.raised",
    "step.finished",
    "content.delta",
    "token",
    "run.complete",
    "run.paused",
)

TARGET_SEARCH_MS = 500.0
TARGET_READ_MS = 2000.0


# --------------------------------------------------------------------------- #
# The search path
# --------------------------------------------------------------------------- #


async def measure_search(backend, rows: Sequence[Mapping[str, Any]], *, repeat: int, limit: int,
                         now, tz: str, week_start: int, warmup: bool, verbose: bool) -> dict[str, Any]:
    """Time the retrieval layer, per query and per stage.

    The first pass over the dataset is discarded when ``warmup`` is set. It is
    always slower — connection setup, an empty query-embedding cache, cold
    HNSW pages — and reporting it inside the same distribution as the warm
    passes describes a machine nobody is using.
    """
    per_query_ms: list[float] = []
    per_service_ms: dict[str, list[float]] = defaultdict(list)
    per_stage_ms: dict[str, list[float]] = defaultdict(list)
    cold_ms: list[float] = []
    by_service_count: dict[int, list[float]] = defaultdict(list)

    async def one(row: Mapping[str, Any], collect: bool) -> None:
        params = QueryParams.from_row(row, now=now, tz=tz, week_start=week_start)
        services = [s for s in params.services if params.applies_to(s)]
        started = time.perf_counter()
        results = await asyncio.gather(
            *(backend.search(row["query"], s, params, limit=limit, arm="hybrid") for s in services)
        )
        wall = (time.perf_counter() - started) * 1000
        if not collect:
            cold_ms.append(wall)
            return
        per_query_ms.append(wall)
        by_service_count[len(services)].append(wall)
        for service, result in zip(services, results, strict=True):
            per_service_ms[service].append(result.took_ms)
            for stage, value in result.stages.items():
                per_stage_ms[stage].append(value)
        if verbose:
            print(f"  {wall:7.1f} ms  {len(services)} corpora  {row['query'][:48]!r}", file=sys.stderr)

    if warmup:
        for row in rows:
            await one(row, collect=False)
    for _ in range(repeat):
        for row in rows:
            await one(row, collect=True)

    return {
        "backend": backend.describe(),
        "queries": len(rows),
        "repeats": repeat,
        "limit": limit,
        "total_ms": summarise(per_query_ms).to_dict(),
        "cold_ms": summarise(cold_ms).to_dict() if cold_ms else None,
        "by_service": {s: summarise(v).to_dict() for s, v in sorted(per_service_ms.items())},
        "by_stage": {s: summarise(v).to_dict() for s, v in sorted(per_stage_ms.items())},
        "by_corpus_count": {
            str(k): summarise(v).to_dict() for k, v in sorted(by_service_count.items())
        },
        "target_ms": TARGET_SEARCH_MS,
        "meets_target_p95": summarise(per_query_ms).p95 < TARGET_SEARCH_MS if per_query_ms else False,
    }


# --------------------------------------------------------------------------- #
# The full query path
# --------------------------------------------------------------------------- #


def classify_run(trace) -> str:
    calls = trace.llm_calls
    if trace.status in ("failed", "error") or trace.error:
        return "failed"
    if trace.status == "paused" or "input.raised" in trace.marks:
        return "paused"
    if calls is None:
        return "unknown"
    if calls == 0:
        return "router"
    if calls == 1:
        return "template"
    return "prose"


async def measure_query_path(rows: Sequence[Mapping[str, Any]], *, base_url: str, repeat: int,
                             cookie: str | None, timeout: float,  # noqa: ASYNC109 — httpx timeout
                             verbose: bool) -> dict[str, Any]:
    import httpx

    headers = {"Cookie": cookie} if cookie else {}
    totals_by_class: dict[str, list[float]] = defaultdict(list)
    marks_by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_totals: list[float] = []
    llm_calls: list[int] = []
    per_row: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout) as client:
        for _ in range(repeat):
            for row in rows:
                trace = await run_query(client, row["query"], timeout=timeout)
                if trace.error:
                    failures.append({"id": row["id"], "error": trace.error})
                    continue
                klass = classify_run(trace)
                totals_by_class[klass].append(trace.total_ms)
                all_totals.append(trace.total_ms)
                if trace.llm_calls is not None:
                    llm_calls.append(trace.llm_calls)
                for mark, at in trace.marks.items():
                    marks_by_class[klass][mark].append(at)
                per_row.append(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "class": klass,
                        "total_ms": round(trace.total_ms, 1),
                        "llm_calls": trace.llm_calls,
                        "marks": {k: round(v, 1) for k, v in sorted(trace.marks.items())},
                    }
                )
                if verbose:
                    print(f"  {trace.total_ms:7.1f} ms  {klass:8} calls={trace.llm_calls}  "
                          f"{row['query'][:44]!r}", file=sys.stderr)

    read_classes = ("router", "template")
    read_totals = [ms for k in read_classes for ms in totals_by_class.get(k, [])]
    return {
        "base_url": base_url,
        "queries": len(rows),
        "repeats": repeat,
        "runs": len(all_totals),
        "overall_ms": summarise(all_totals).to_dict(),
        "by_class": {
            klass: {
                "n": len(values),
                "total_ms": summarise(values).to_dict(),
                "marks_ms": {
                    mark: summarise(marks_by_class[klass][mark]).to_dict()
                    for mark in MARKS
                    if marks_by_class[klass].get(mark)
                },
            }
            for klass, values in sorted(totals_by_class.items())
        },
        "read_class_ms": summarise(read_totals).to_dict(),
        "target_read_ms": TARGET_READ_MS,
        "meets_read_target_p95": summarise(read_totals).p95 < TARGET_READ_MS if read_totals else False,
        "llm_calls": {
            "mean": round(mean(llm_calls), 2) if llm_calls else None,
            "median": statistics.median(llm_calls) if llm_calls else None,
            "max": max(llm_calls) if llm_calls else None,
            "n": len(llm_calls),
        },
        "rows": per_row,
        "failures": failures,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report_search(block: Mapping[str, Any]) -> str:
    out = ["Search path — retrieval only, no model in the loop"]
    out.append(f"  backend: {block['backend']}")
    out.append(f"  {block['queries']} queries x {block['repeats']} repeats, limit {block['limit']} per corpus")
    total = block["total_ms"]
    out.append("")
    out.append(f"  p50 {total['p50']:7.1f} ms")
    out.append(f"  p95 {total['p95']:7.1f} ms     target < {block['target_ms']:.0f} ms: "
               f"{'MET' if block['meets_target_p95'] else 'NOT MET'}")
    out.append(f"  p99 {total['p99']:7.1f} ms")
    out.append(f"  mean {total['mean']:6.1f} ms   min {total['min']:.1f}   max {total['max']:.1f}   n={total['n']}")
    if block.get("cold_ms"):
        cold = block["cold_ms"]
        out.append(f"  cold pass (discarded): p50 {cold['p50']:.1f} ms, p95 {cold['p95']:.1f} ms")

    if block["by_stage"]:
        out.append("")
        out.append("  Where the time goes")
        out.append(
            render_table(
                ["stage", "n", "p50", "p95", "p99"],
                [[stage, s["n"], f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}"]
                 for stage, s in block["by_stage"].items()],
                right=(1, 2, 3, 4),
            )
        )
    if block["by_service"]:
        out.append("")
        out.append("  Per corpus")
        out.append(
            render_table(
                ["corpus", "n", "p50", "p95", "p99"],
                [[service, s["n"], f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}"]
                 for service, s in block["by_service"].items()],
                right=(1, 2, 3, 4),
            )
        )
    if block["by_corpus_count"]:
        out.append("")
        out.append("  By how many corpora ran in parallel — flat means the fan-out is genuinely concurrent")
        out.append(
            render_table(
                ["corpora", "n", "p50", "p95"],
                [[k, s["n"], f"{s['p50']:.1f}", f"{s['p95']:.1f}"]
                 for k, s in block["by_corpus_count"].items()],
                right=(0, 1, 2, 3),
            )
        )
    return "\n".join(out)


def report_query(block: Mapping[str, Any]) -> str:
    out = ["Full query path — POST /api/v1/query, following the event stream to completion"]
    out.append(f"  {block['base_url']}    {block['runs']} runs "
               f"({block['queries']} queries x {block['repeats']})")
    read = block["read_class_ms"]
    out.append("")
    out.append(f"  read class (router + template):  p50 {read['p50']:.0f} ms   "
               f"p95 {read['p95']:.0f} ms   p99 {read['p99']:.0f} ms   n={read['n']}")
    out.append(f"  target P95 < {block['target_read_ms']:.0f} ms: "
               f"{'MET' if block['meets_read_target_p95'] else 'NOT MET'}")
    out.append("")
    out.append("  By class")
    out.append(
        render_table(
            ["class", "n", "p50", "p95", "p99", "max"],
            [
                [klass, data["n"], f"{data['total_ms']['p50']:.0f}", f"{data['total_ms']['p95']:.0f}",
                 f"{data['total_ms']['p99']:.0f}", f"{data['total_ms']['max']:.0f}"]
                for klass, data in block["by_class"].items()
            ],
            right=(1, 2, 3, 4, 5),
        )
    )
    calls = block["llm_calls"]
    if calls["mean"] is not None:
        out.append("")
        out.append(f"  LLM calls per completed run: mean {calls['mean']}  median {calls['median']}  "
                   f"max {calls['max']}  (hard cap 5)")

    out.append("")
    out.append("  Time to first meaningful pixel, p50 by class")
    marks_present = [m for m in MARKS
                     if any(data["marks_ms"].get(m) for data in block["by_class"].values())]
    out.append(
        render_table(
            ["class", *marks_present],
            [
                [klass, *[f"{data['marks_ms'][m]['p50']:.0f}" if data["marks_ms"].get(m) else "-"
                          for m in marks_present]]
                for klass, data in block["by_class"].items()
            ],
            right=tuple(range(1, len(marks_present) + 1)),
        )
    )
    if block["failures"]:
        out.append("")
        out.append(f"  Failures ({len(block['failures'])})")
        for failure in block["failures"][:10]:
            out.append(f"    {failure['id']}  {failure['error']}")
    return "\n".join(out)


def markdown_section(search: Mapping[str, Any] | None, query: Mapping[str, Any] | None) -> str:
    lines: list[str] = []
    if search:
        total = search["total_ms"]
        lines += [
            "**Search path** — one embedding plus the hybrid searches, no model in the loop.",
            "",
            markdown_table(
                ["measure", "p50", "p95", "p99", "target"],
                [["per query, all corpora in parallel", f"{total['p50']:.0f} ms",
                  f"{total['p95']:.0f} ms", f"{total['p99']:.0f} ms",
                  f"< {search['target_ms']:.0f} ms"]],
            ),
            "",
            "Where the time goes:",
            "",
            markdown_table(
                ["stage", "p50", "p95"],
                [[stage, f"{s['p50']:.1f} ms", f"{s['p95']:.1f} ms"]
                 for stage, s in search["by_stage"].items()],
            ),
        ]
    if query:
        read = query["read_class_ms"]
        lines += [
            "",
            "**Full query path** — POST to the answer, by class. Averaging these classes together "
            "would describe nothing, so they are not averaged.",
            "",
            markdown_table(
                ["class", "runs", "p50", "p95", "p99"],
                [[klass, data["n"], f"{data['total_ms']['p50']:.0f} ms",
                  f"{data['total_ms']['p95']:.0f} ms", f"{data['total_ms']['p99']:.0f} ms"]
                 for klass, data in query["by_class"].items()],
            ),
            "",
            f"Read class (router + template) p95 **{read['p95']:.0f} ms** against a 2 s target. "
            "Two-call prose reads do not fit in 2 s and are not claimed to.",
        ]
        marks_present = [m for m in MARKS
                         if any(d["marks_ms"].get(m) for d in query["by_class"].values())]
        if marks_present:
            lines += [
                "",
                "Time to first meaningful pixel (p50, ms):",
                "",
                markdown_table(
                    ["class", *marks_present],
                    [[klass, *[f"{d['marks_ms'][m]['p50']:.0f}" if d["marks_ms"].get(m) else "—"
                               for m in marks_present]]
                     for klass, d in query["by_class"].items()],
                ),
            ]
    return "\n".join(lines) if lines else "_No latency run._"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", choices=("search", "query", "both"), default="both")
    parser.add_argument("--search-dataset", default="relevance.jsonl")
    parser.add_argument("--query-dataset", default="intents.jsonl")
    parser.add_argument("--backend", choices=("mirror", "hybrid", "http"), default="mirror",
                        help="which layer the search path measures")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--cookie",
        default=os.environ.get("EVAL_SESSION_COOKIE"),
        help="session cookie for the API (or EVAL_SESSION_COOKIE)",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-warmup", action="store_true", help="include the cold pass in the distribution")
    parser.add_argument("--include-writes", action="store_true",
                        help="also send write-intent queries through the query path (prepares real drafts)")
    parser.add_argument("--user-email", default=EVAL_USER_EMAIL)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--now", default=None)
    parser.add_argument("--tz", default=FIXED_TZ)
    parser.add_argument("--week-start", type=int, default=FIXED_WEEK_START)
    parser.add_argument("--only", default=None)
    parser.add_argument("--json", dest="json_out", default="latency.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--traceback", action="store_true", help="re-raise instead of explaining")
    args = parser.parse_args(argv)

    now = parse_now(args.now)
    search_block: dict[str, Any] | None = None
    query_block: dict[str, Any] | None = None

    if args.path in ("search", "both"):
        dataset = load_jsonl(args.search_dataset).filter(args.only.split(",") if args.only else None)
        backend_kwargs: dict[str, Any] = {"user_email": args.user_email, "user_id": args.user_id}
        if args.backend == "http":
            backend_kwargs["base_url"] = args.base_url
        backend = build_search_backend(args.backend, **backend_kwargs)

        async def go_search() -> dict[str, Any]:
            await backend.setup()
            try:
                return await measure_search(
                    backend, dataset.rows, repeat=args.repeat, limit=args.limit, now=now,
                    tz=args.tz, week_start=args.week_start, warmup=not args.no_warmup,
                    verbose=args.verbose,
                )
            finally:
                await backend.close()

        try:
            search_block = run_async(go_search())
            print(report_search(search_block))
        except Exception as exc:
            code = explain_failure(exc, what="search path", traceback_wanted=args.traceback)
            if args.path == "search":
                return code

    if args.path in ("query", "both"):
        dataset = load_jsonl(args.query_dataset).filter(args.only.split(",") if args.only else None)
        rows = dataset.rows
        if not args.include_writes:
            rows = [r for r in rows if not r["expected"]["has_write"]]
        # Front-door rows depend on an open card or a prior turn that a
        # standalone POST does not have, so they would measure the wrong thing.
        rows = [r for r in rows if r["expected"]["route"] != "ui_verb"]
        if search_block is not None:
            print("")

        async def go_query() -> dict[str, Any]:
            return await measure_query_path(
                rows, base_url=args.base_url, repeat=args.repeat, cookie=args.cookie,
                timeout=args.timeout, verbose=args.verbose,
            )

        try:
            query_block = run_async(go_query())
            print(report_query(query_block))
        except Exception as exc:
            code = explain_failure(exc, what=f"query path against {args.base_url}",
                                   traceback_wanted=args.traceback)
            if args.path == "query":
                return code

    if args.json_out and (search_block or query_block):
        payload = {
            "kind": "latency",
            "now": now.isoformat().replace("+00:00", "Z"),
            "search": search_block,
            "query_path": query_block,
            "env": env_fingerprint(),
            "markdown": markdown_section(search_block, query_block),
        }
        path = write_json(args.json_out, payload)
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
