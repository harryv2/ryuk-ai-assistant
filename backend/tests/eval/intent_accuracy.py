#!/usr/bin/env python3
"""Intent classification accuracy — the brief's first ten points.

Runs every labelled query in ``datasets/intents.jsonl`` through the classifier
and reports:

* overall accuracy, and accuracy per slice (multilingual, awkward, ambiguous,
  context-dependent, out-of-scope);
* precision, recall and F1 per intent, with support;
* the worst confusions, because "84% accurate" hides whether the 16% is
  ``email_search`` vs ``email_detail`` (harmless) or ``unsupported`` vs
  ``cancel_flight`` (not);
* service-set F1, micro-averaged over service labels plus the exact-set rate;
* **missed ambiguities** — a query the dataset says should raise a card and the
  classifier answered anyway. This is the dangerous direction and it is
  reported on its own line. Missing one means the system guessed which John.

Run it::

    python -m tests.eval.intent_accuracy --dry-run          # no API key, no db
    python -m tests.eval.intent_accuracy                    # live classifier
    python -m tests.eval.intent_accuracy --backend api      # through the HTTP API

``--dry-run`` uses canned responses with a fixed set of deliberate mistakes, so
CI can catch a broken report without spending a token. Its numbers are not
measurements and every line that prints them says so.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python tests/eval/intent_accuracy.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.eval import (
    Prediction,
    build_classifier,
    env_fingerprint,
    explain_failure,
    fmt,
    load_jsonl,
    markdown_table,
    mean,
    parse_now,
    render_table,
    run_async,
    summarise,
    write_json,
)

SLICES = ("brief_sample", "multilingual", "awkward", "ambiguous", "context", "negative", "front_door")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Prediction]) -> dict[str, Any]:
    """Everything the report prints, computed once."""
    n = len(rows)
    correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    support: Counter[str] = Counter()
    predicted_count: Counter[str] = Counter()
    hits_per_intent: Counter[str] = Counter()

    svc_tp = svc_fp = svc_fn = 0
    svc_exact = 0
    ent_tp = ent_fp = ent_fn = 0
    ent_exact = 0
    ent_value_total = ent_value_right = 0

    amb_tp = amb_fp = amb_fn = amb_tn = 0
    missed_ambiguities: list[dict[str, str]] = []
    false_alarms: list[dict[str, str]] = []

    ctx_right = ctx_total = 0
    write_right = 0
    route_agree = route_total = 0
    errors: list[dict[str, str]] = []
    slice_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    latencies: list[float] = []
    llm_calls: list[int] = []
    wrong: list[dict[str, Any]] = []

    for row, pred in zip(rows, predictions, strict=True):
        expected = row["expected"]
        want_intent = expected["intent"]
        got_intent = pred.intent or "<none>"
        support[want_intent] += 1
        predicted_count[got_intent] += 1
        is_right = got_intent == want_intent
        if is_right:
            correct += 1
            hits_per_intent[want_intent] += 1
        else:
            confusion[(want_intent, got_intent)] += 1
            wrong.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "expected": want_intent,
                    "got": got_intent,
                    "note": row.get("note", ""),
                }
            )
        if pred.error:
            errors.append({"id": row["id"], "error": pred.error})

        for tag in row.get("tags", []):
            if tag in SLICES:
                slice_counts[tag][1] += 1
                slice_counts[tag][0] += int(is_right)

        want_services = set(expected["services"])
        got_services = set(pred.services)
        svc_tp += len(want_services & got_services)
        svc_fp += len(got_services - want_services)
        svc_fn += len(want_services - got_services)
        svc_exact += int(want_services == got_services)

        want_keys = set(expected.get("entity_keys") or [])
        got_keys = set(pred.entity_keys)
        ent_tp += len(want_keys & got_keys)
        ent_fp += len(got_keys - want_keys)
        ent_fn += len(want_keys - got_keys)
        ent_exact += int(want_keys == got_keys)
        for key, value in (expected.get("entities") or {}).items():
            ent_value_total += 1
            got_value = pred.entities.get(key)
            if got_value is not None and _same_value(value, got_value):
                ent_value_right += 1

        should_flag = bool(expected["flag_ambiguity"])
        did_flag = bool(pred.ambiguous)
        if should_flag and did_flag:
            amb_tp += 1
        elif should_flag and not did_flag:
            amb_fn += 1
            missed_ambiguities.append({"id": row["id"], "query": row["query"], "got": got_intent})
        elif not should_flag and did_flag:
            amb_fp += 1
            false_alarms.append({"id": row["id"], "query": row["query"], "got": got_intent})
        else:
            amb_tn += 1

        ctx_total += 1
        ctx_right += int(bool(expected["refs_prior_turn"]) == bool(pred.refs_prior_turn))
        write_right += int(bool(expected["has_write"]) == bool(pred.has_write))

        want_route = expected.get("route")
        if want_route and pred.route:
            route_total += 1
            route_agree += int(_route_matches(want_route, pred.route))

        latencies.append(pred.latency_ms)
        if pred.llm_calls is not None:
            llm_calls.append(pred.llm_calls)

    per_intent = []
    for intent in sorted(support | predicted_count):
        tp = hits_per_intent[intent]
        fp = predicted_count[intent] - tp
        fn = support[intent] - tp
        precision, recall, f1 = prf(tp, fp, fn)
        per_intent.append(
            {
                "intent": intent,
                "support": support[intent],
                "predicted": predicted_count[intent],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = mean([r["f1"] for r in per_intent if r["support"]])
    svc_p, svc_r, svc_f1 = prf(svc_tp, svc_fp, svc_fn)
    ent_p, ent_r, ent_f1 = prf(ent_tp, ent_fp, ent_fn)
    amb_p, amb_r, amb_f1 = prf(amb_tp, amb_fp, amb_fn)

    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "confusions": [
            {"expected": e, "got": g, "count": c} for (e, g), c in confusion.most_common()
        ],
        "services": {
            "micro_precision": svc_p,
            "micro_recall": svc_r,
            "micro_f1": svc_f1,
            "exact_set": svc_exact / n if n else 0.0,
        },
        "entities": {
            "key_precision": ent_p,
            "key_recall": ent_r,
            "key_f1": ent_f1,
            "exact_set": ent_exact / n if n else 0.0,
            "value_accuracy": (ent_value_right / ent_value_total) if ent_value_total else None,
            "value_n": ent_value_total,
        },
        "ambiguity": {
            "should_flag": amb_tp + amb_fn,
            "flagged": amb_tp + amb_fp,
            "caught": amb_tp,
            "missed": amb_fn,
            "false_alarms": amb_fp,
            "correctly_silent": amb_tn,
            "precision": amb_p,
            "recall": amb_r,
            "f1": amb_f1,
            "missed_rows": missed_ambiguities,
            "false_alarm_rows": false_alarms,
        },
        "prior_turn_accuracy": ctx_right / ctx_total if ctx_total else 0.0,
        "has_write_accuracy": write_right / n if n else 0.0,
        "route_agreement": route_agree / route_total if route_total else None,
        "route_n": route_total,
        "slices": {
            name: {"n": total, "correct": right, "accuracy": right / total if total else 0.0}
            for name, (right, total) in sorted(slice_counts.items())
        },
        "latency_ms": summarise(latencies).to_dict(),
        "llm_calls": {
            "mean": mean(llm_calls) if llm_calls else None,
            "total": sum(llm_calls) if llm_calls else None,
            "n": len(llm_calls),
        },
        "errors": errors,
        "wrong": wrong,
    }


def _same_value(want: Any, got: Any) -> bool:
    return str(want).strip().lower() == str(got).strip().lower()


def _route_matches(want: str, got: str) -> bool:
    """``rule_router`` and ``rule_router:intent_carry`` are the same family."""
    return want.split(":")[0] == got.split(":")[0]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(result: Mapping[str, Any], *, dataset: str, backend: str, canned: bool,
           show_wrong: bool) -> str:
    out: list[str] = []
    banner = "  [CANNED — NOT A MEASUREMENT]" if canned else ""
    out.append(f"Intent classification — {result['n']} queries from {dataset}{banner}")
    out.append(f"classifier: {backend}")
    out.append("")
    out.append(f"  accuracy        {result['accuracy']:.3f}  ({result['correct']}/{result['n']})")
    out.append(f"  macro F1        {result['macro_f1']:.3f}")
    out.append(f"  service set F1  {result['services']['micro_f1']:.3f}  "
               f"(micro; exact-set {result['services']['exact_set']:.3f})")
    out.append(f"  entity key F1   {result['entities']['key_f1']:.3f}  "
               f"(exact-set {result['entities']['exact_set']:.3f})")
    value_accuracy = result["entities"]["value_accuracy"]
    out.append(f"  entity values   {fmt(value_accuracy)}  over {result['entities']['value_n']} labelled values")
    out.append(f"  prior-turn      {result['prior_turn_accuracy']:.3f}")
    out.append(f"  has_write       {result['has_write_accuracy']:.3f}")
    if result["route_agreement"] is not None:
        out.append(f"  route agreement {result['route_agreement']:.3f}  over {result['route_n']} rows (reported, not scored)")

    amb = result["ambiguity"]
    out.append("")
    out.append("Ambiguity — the direction that matters")
    out.append(f"  should have flagged   {amb['should_flag']}")
    out.append(f"  caught                {amb['caught']}")
    out.append(f"  MISSED                {amb['missed']}   <- each one is the system guessing")
    out.append(f"  false alarms          {amb['false_alarms']}   (asked when it did not need to)")
    out.append(f"  recall {amb['recall']:.3f}   precision {amb['precision']:.3f}")
    for row in amb["missed_rows"]:
        out.append(f"    missed  {row['id']}  {row['query']!r} -> {row['got']}")
    for row in amb["false_alarm_rows"]:
        out.append(f"    alarm   {row['id']}  {row['query']!r} -> {row['got']}")

    out.append("")
    out.append("Per intent")
    out.append(
        render_table(
            ["intent", "n", "pred", "precision", "recall", "f1"],
            [
                [r["intent"], r["support"], r["predicted"], f"{r['precision']:.3f}",
                 f"{r['recall']:.3f}", f"{r['f1']:.3f}"]
                for r in result["per_intent"]
            ],
            right=(1, 2, 3, 4, 5),
        )
    )

    if result["confusions"]:
        out.append("")
        out.append("Worst confusions")
        out.append(
            render_table(
                ["expected", "got", "n"],
                [[c["expected"], c["got"], c["count"]] for c in result["confusions"][:10]],
                right=(2,),
            )
        )

    if result["slices"]:
        out.append("")
        out.append("By slice")
        out.append(
            render_table(
                ["slice", "n", "intent acc"],
                [[name, s["n"], f"{s['accuracy']:.3f}"] for name, s in result["slices"].items()],
                right=(1, 2),
            )
        )

    latency = result["latency_ms"]
    out.append("")
    out.append(
        f"Classification latency (ms)  p50 {latency['p50']:.0f}  p95 {latency['p95']:.0f}  "
        f"p99 {latency['p99']:.0f}  n={latency['n']}"
    )
    calls = result["llm_calls"]
    if calls["mean"] is not None:
        out.append(f"LLM calls per query  mean {calls['mean']:.2f}  total {calls['total']}")

    if show_wrong and result["wrong"]:
        out.append("")
        out.append("Every miss")
        for row in result["wrong"]:
            out.append(f"  {row['id']}  {row['expected']} -> {row['got']}   {row['query']!r}")

    if result["errors"]:
        out.append("")
        out.append(f"Errors ({len(result['errors'])})")
        for row in result["errors"][:10]:
            out.append(f"  {row['id']}  {row['error']}")

    return "\n".join(out)


def markdown_section(result: Mapping[str, Any], *, canned: bool) -> str:
    """The block precision_at_k folds into RESULTS.md."""
    amb = result["ambiguity"]
    lines = []
    if canned:
        lines.append("> Produced with `--dry-run`. These are canned responses, **not measurements**.")
        lines.append("")
    lines.append(
        markdown_table(
            ["metric", "value", "what it means"],
            [
                ["Accuracy", f"{result['accuracy']:.3f}", f"{result['correct']}/{result['n']} exact intent-name matches"],
                ["Macro F1", f"{result['macro_f1']:.3f}", "unweighted across intents, so rare intents count"],
                ["Service-set F1", f"{result['services']['micro_f1']:.3f}", "micro over service labels"],
                ["Service-set exact", f"{result['services']['exact_set']:.3f}", "the whole set right"],
                ["Entity-key F1", f"{result['entities']['key_f1']:.3f}", "keys the planner produced vs labelled"],
                ["Prior-turn detection", f"{result['prior_turn_accuracy']:.3f}", "did it know the query leans on the conversation"],
                ["Ambiguities missed", str(amb["missed"]), f"of {amb['should_flag']} — the dangerous direction"],
                ["False alarms", str(amb["false_alarms"]), "asked when it did not need to — the cheap direction"],
            ],
        )
    )
    lines.append("")
    lines.append("Per intent:")
    lines.append("")
    lines.append(
        markdown_table(
            ["intent", "n", "precision", "recall", "F1"],
            [
                [r["intent"], r["support"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}"]
                for r in result["per_intent"] if r["support"]
            ],
        )
    )
    if result["confusions"]:
        lines.append("")
        lines.append("Worst confusions:")
        lines.append("")
        lines.append(
            markdown_table(
                ["expected", "got", "n"],
                [[c["expected"], c["got"], c["count"]] for c in result["confusions"][:6]],
            )
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def evaluate(rows: Sequence[Mapping[str, Any]], classifier, *, now, concurrency: int,
                   verbose: bool) -> list[Prediction]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    predictions: list[Prediction | None] = [None] * len(rows)

    async def one(index: int, row: Mapping[str, Any]) -> None:
        async with semaphore:
            prediction = await classifier.classify(row, now=now)
            predictions[index] = prediction
            if verbose:
                mark = "ok " if prediction.intent == row["expected"]["intent"] else "MISS"
                print(f"  {mark} {row['id']}  {row['query'][:56]!r:60} -> {prediction.intent}",
                      file=sys.stderr)

    await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
    return [p or Prediction(error="no prediction") for p in predictions]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="intents.jsonl")
    parser.add_argument("--backend", choices=("live", "api", "canned"), default="live",
                        help="live: front_door + route in process. api: POST /api/v1/query. canned: --dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="canned responses; no API key, no database")
    parser.add_argument("--base-url", default=None, help="API base for --backend api")
    parser.add_argument("--user-email", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--now", default=None, help=f"evaluation instant (default {parse_now(None).isoformat()})")
    parser.add_argument("--tz", default=None)
    parser.add_argument("--only", default=None, help="comma-separated row ids")
    parser.add_argument("--tag", default=None, help="restrict to one slice, e.g. multilingual")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 if accuracy is below this")
    parser.add_argument("--max-missed-ambiguities", type=int, default=None, help="exit 1 above this many")
    parser.add_argument("--json", dest="json_out", default="intent_accuracy.json",
                        help="metrics file under tests/eval/out/ ('' to skip)")
    parser.add_argument("--show-wrong", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--traceback", action="store_true", help="re-raise instead of explaining")
    args = parser.parse_args(argv)

    from tests.eval import EVAL_USER_EMAIL, FIXED_TZ

    kind = "canned" if args.dry_run else args.backend
    dataset = load_jsonl(args.dataset).filter(
        args.only.split(",") if args.only else None, args.tag
    )
    now = parse_now(args.now)
    kwargs: dict[str, Any] = {
        "user_email": args.user_email or EVAL_USER_EMAIL,
        "user_id": args.user_id,
        "tz": args.tz or FIXED_TZ,
    }
    if args.base_url:
        kwargs["base_url"] = args.base_url
    classifier = build_classifier(kind, **kwargs)

    async def go() -> dict[str, Any]:
        await classifier.setup()
        try:
            predictions = await evaluate(
                dataset.rows, classifier, now=now, concurrency=args.concurrency, verbose=args.verbose
            )
        finally:
            await classifier.close()
        return score(dataset.rows, predictions)

    try:
        result = run_async(go())
    except Exception as exc:
        return explain_failure(exc, what="intent eval", traceback_wanted=args.traceback)
    canned = kind == "canned"
    print(report(result, dataset=dataset.path.name, backend=classifier.describe(),
                 canned=canned, show_wrong=args.show_wrong))

    if args.json_out:
        payload = {
            "kind": "intent_accuracy",
            "canned": canned,
            "dataset": dataset.path.name,
            "dataset_size": len(dataset),
            "classifier": classifier.describe(),
            "now": now.isoformat().replace("+00:00", "Z"),
            "env": env_fingerprint(),
            "markdown": markdown_section(result, canned=canned),
            "result": result,
        }
        path = write_json(args.json_out, payload)
        print(f"\nwrote {path}")

    failed = False
    if args.fail_under is not None and result["accuracy"] < args.fail_under:
        print(f"\nFAIL: accuracy {result['accuracy']:.3f} < {args.fail_under}", file=sys.stderr)
        failed = True
    if args.max_missed_ambiguities is not None and result["ambiguity"]["missed"] > args.max_missed_ambiguities:
        print(
            f"FAIL: {result['ambiguity']['missed']} missed ambiguities > {args.max_missed_ambiguities}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
