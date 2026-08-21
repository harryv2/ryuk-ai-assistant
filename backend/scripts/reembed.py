"""Re-embed a mirror table after the embedding model changes.

    python -m scripts.reembed --table all --dry-run
    python -m scripts.reembed --table all
    python -m scripts.reembed --table gmail --model gemini:gemini-embedding-001

Why this exists
---------------

Chat models are swappable and embedding models are not. Every `VECTOR(1536)`
column in the mirror was filled by one specific model, and a vector from a
different model is not comparable with those — not because the widths differ
(Gemini's `gemini-embedding-001` will produce 1536 on request, same as OpenAI's
`text-embedding-3-small`) but because the two models put meaning in different
places. A cosine distance across a mix is not a weak signal; it is a made-up
one, and it looks exactly like a real one. No error, no warning, no obviously
wrong output. Just the wrong emails.

So a mirror may only ever hold vectors from **one** embedding model at a time.
That is what `sync_*.embed_model` records, what `app.llm.assert_same_embed_model`
checks on the search path, and what this script exists to restore after
`EMBED_MODEL` moves.

What it does
------------

Walks the rows that are not on the target model, in batches, embedding each
batch and writing back the vector **and** the model name in one statement. It
reports progress and a running dollar figure as it goes, and it can be stopped
and restarted: a row is off the work list the moment its update commits, so a
second run picks up exactly where the first one died.

The safety rule
---------------

**It refuses to start a run that would leave the mirror holding vectors from
two models**, and says which rows would be left behind. `--force` overrides,
because a deliberate two-stage run — one table tonight, the rest tomorrow — is
a real thing to want; but search is broken for everything in between and you
should be choosing that, not discovering it.

Rows whose text is empty cannot be embedded at all. Those have their stale
vector cleared and their `embed_model` blanked rather than left claiming a model
that never saw them: a row with no vector is invisible to vector search, which
is the honest outcome. They do not count as a mixed mirror.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

TABLES: tuple[str, ...] = ("gmail", "gcal", "gdrive")

#: Texts per embedding request. Under both vendors' per-request caps.
DEFAULT_BATCH = 96

#: Characters per token, for the estimate printed before the run starts. Rough
#: on purpose — the running figure below it comes from the usage ledger, which
#: is what the provider actually billed.
CHARS_PER_TOKEN = 4.0


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass
class Scope:
    """Which rows this run is allowed to touch."""

    tables: tuple[str, ...]
    model: str
    user_id: str | None = None
    limit: int | None = None

    def covers(self, table: str) -> bool:
        return table in self.tables


@dataclass
class Survey:
    """What the mirror holds right now, per table and per embedding model.

    Only rows that actually carry a vector are counted. A row with no vector is
    not a second opinion about anything; it is a row that has not been embedded.
    """

    by_table: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, table: str, model: str, count: int) -> None:
        self.by_table.setdefault(table, {})[model] = count

    def models_other_than(self, target: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for models in self.by_table.values():
            for model, count in models.items():
                if model != target and model:
                    out[model] = out.get(model, 0) + count
        return out

    def stale_rows(self, table: str, target: str) -> int:
        """Rows in one table that carry a vector from some other model."""
        return sum(
            count for model, count in self.by_table.get(table, {}).items() if model != target
        )


@dataclass
class Progress:
    """Counters for one run, and everything the progress line reads."""

    total: int = 0
    done: int = 0
    embedded: int = 0
    cleared: int = 0
    chars: int = 0
    usd: float = 0.0
    tokens: int = 0
    requests: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started)

    @property
    def rate(self) -> float:
        return self.done / self.elapsed

    @property
    def eta_s(self) -> float:
        left = max(0, self.total - self.done)
        return left / self.rate if self.rate > 0 else 0.0


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def estimate_tokens(chars: int) -> int:
    return math.ceil(max(0, chars) / CHARS_PER_TOKEN)


def estimate_usd(model: str, chars: int) -> float:
    """A rough dollar figure for `chars` worth of text on `model`.

    Embeddings bill input only, so only the prompt rate is used. An unpriced
    model estimates at zero, which reads as "we do not know" rather than as
    "free" once the run prints the real ledger figure beside it.
    """
    from app.llm.usage import rates

    prompt_rate, _ = rates(model)
    return (estimate_tokens(chars) / 1_000_000) * prompt_rate


def money(value: float) -> str:
    if value and value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def duration(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- #
# Reading the mirror
# --------------------------------------------------------------------------- #


async def survey(session: Any, scope: Scope) -> Survey:
    """Count rows per (table, embed_model) across all three mirror tables.

    Every table, not only the ones in scope — the question the safety check
    asks is about the whole mirror, since one search touches all three.
    """
    from app.db.repositories import mirror
    from sqlalchemy import func, select

    out = Survey()
    for table in TABLES:
        model: Any = mirror.spec_for(table).model
        stmt = (
            select(model.embed_model, func.count())
            .where(model.embedding.is_not(None))
            .group_by(model.embed_model)
        )
        if scope.user_id:
            stmt = stmt.where(model.user_id == scope.user_id)
        for embed_model, count in (await session.execute(stmt)).all():
            out.add(table, str(embed_model or ""), int(count))
    return out


async def count_work(session: Any, table: str, scope: Scope) -> int:
    """How many rows in this table still have to be re-embedded."""
    from app.db.repositories import mirror
    from sqlalchemy import func, select

    model: Any = mirror.spec_for(table).model
    stmt = select(func.count()).select_from(model).where(_needs_work(model, scope.model))
    if scope.user_id:
        stmt = stmt.where(model.user_id == scope.user_id)
    return int((await session.execute(stmt)).scalar() or 0)


def _needs_work(model: Any, target: str) -> Any:
    """Rows that carry a vector from some other model.

    Both halves matter. A row with **no** vector is not this script's business —
    the sync path's `embed_pending` will embed it and stamp it, and pulling it in
    here would mean a row whose text is empty can never leave the work list,
    because clearing it is the only correct thing to do with it.

    `IS DISTINCT FROM` rather than `!=` so a NULL `embed_model` — which the
    column forbids, but a hand-written backfill can still produce — counts as
    work instead of quietly matching nothing.
    """
    from sqlalchemy import and_

    return and_(model.embedding.is_not(None), model.embed_model.is_distinct_from(target))


async def fetch_batch(session: Any, table: str, scope: Scope, after: str, size: int) -> list[Any]:
    """The next `size` rows to do, ordered by id so the walk is resumable."""
    from app.db.repositories import mirror
    from sqlalchemy import select

    model: Any = mirror.spec_for(table).model
    stmt = select(model).where(_needs_work(model, scope.model)).order_by(model.id.asc()).limit(size)
    if after:
        stmt = stmt.where(model.id > after)
    if scope.user_id:
        stmt = stmt.where(model.user_id == scope.user_id)
    return list((await session.execute(stmt)).scalars().all())


async def write_back(
    session: Any,
    table: str,
    rows: Sequence[tuple[str, str, list[float] | None, str]],
) -> None:
    """Write vector and model name together, in one statement.

    Together on purpose. Two statements can be interrupted between them, and a
    vector whose `embed_model` still names the old model is exactly the silent
    wrongness this column exists to prevent.
    """
    if not rows:
        return
    from app.db.repositories import mirror
    from app.llm import embed_dimensions
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import bindparam, update

    model: Any = mirror.spec_for(table).model
    stmt = (
        update(model)
        # The owner is in the WHERE clause even though the id alone is unique.
        # Every mirror write in this codebase carries the tenant it belongs to,
        # and a script that writes across all of them is the last place to make
        # an exception.
        .where(model.id == bindparam("row_id"), model.user_id == bindparam("owner"))
        .values(
            embedding=bindparam("vec", type_=Vector(embed_dimensions())),
            embed_model=bindparam("embed_model_name"),
        )
    )
    await session.execute(
        stmt,
        [
            {"row_id": row_id, "owner": owner, "vec": vector, "embed_model_name": name}
            for row_id, owner, vector, name in rows
        ],
    )


# --------------------------------------------------------------------------- #
# The safety check
# --------------------------------------------------------------------------- #


def mixed_after(surveyed: Survey, scope: Scope, work: dict[str, int]) -> list[str]:
    """What would still be on another model when this run finishes.

    Three ways to end up mixed, and all three are easy to do by accident:
    a table left out of `--table`, a `--user` that is not every user, and a
    `--limit` smaller than the work.
    """
    reasons: list[str] = []

    for table in TABLES:
        stale = surveyed.stale_rows(table, scope.model)
        if stale and not scope.covers(table):
            reasons.append(
                f"{table}: {stale:,} rows stay on another model — this run does not include "
                f"that table (add --table all)"
            )

    if scope.user_id:
        reasons.append(
            "--user limits the run to one account; every other account's rows stay on "
            "whatever model embedded them"
        )

    if scope.limit is not None:
        total = sum(work.get(table, 0) for table in scope.tables)
        if scope.limit < total:
            reasons.append(
                f"--limit {scope.limit:,} is smaller than the {total:,} rows that need "
                f"doing, so {total - scope.limit:,} would be left behind"
            )

    return reasons


MIXING_EXPLAINED = """
  A mirror may hold vectors from exactly one embedding model.

  Two models can both produce 1536-dimension vectors and still disagree
  completely about what each dimension means. Cosine distance across a mix is
  still a number, and the number is nonsense: no error, no warning, nothing that
  looks wrong in the output. Just the wrong emails, quietly, until somebody
  notices the answers stopped making sense.

  That is why every row records the model that embedded it, and why the search
  path refuses to compare rows whose `embed_model` is not the one that embedded
  the question. A half-done re-embed does not degrade search; it stops it.

  Run the whole mirror in one go, or pass --force if you have decided to accept
  a broken search until the rest of it is done.
"""


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def text_of(table: str, row: Any) -> str:
    """The exact string the sync path would embed for this row.

    Reused from `app.search.chunking` rather than rebuilt here: a re-embed that
    embedded slightly different text would produce vectors that do not match
    the ones a later sync writes, which is the same failure this script is
    fixing, arriving by a different road.
    """
    from app.search import embedder

    return embedder.text_for_row(table, row).strip()


async def do_table(
    session: Any,
    table: str,
    scope: Scope,
    progress: Progress,
    *,
    batch_size: int,
    ledger: Any,
) -> None:
    """Walk one table to the end, or to `--limit`."""
    from app.llm import embed as embed_texts

    after = ""
    while True:
        if scope.limit is not None and progress.done >= scope.limit:
            return

        size = batch_size
        if scope.limit is not None:
            size = min(size, scope.limit - progress.done)
        if size <= 0:
            return

        rows = await fetch_batch(session, table, scope, after, size)
        if not rows:
            return
        after = str(rows[-1].id)

        pending: list[tuple[str, str, str]] = []  # (row id, owner, text)
        cleared: list[tuple[str, str, None, str]] = []
        for row in rows:
            text = text_of(table, row)
            if text:
                pending.append((str(row.id), str(row.user_id), text))
            else:
                # No text to embed. Drop whatever vector is there rather than
                # leave the old model's answer in a row this run has passed.
                cleared.append((str(row.id), str(row.user_id), None, ""))

        progress.chars += sum(len(text) for _, _, text in pending)

        vectors: list[list[float]] = []
        if pending:
            vectors = await embed_texts([text for _, _, text in pending], scope.model)
            progress.requests += 1
        await write_back(
            session,
            table,
            [
                *[
                    (row_id, owner, vector, scope.model)
                    for (row_id, owner, _), vector in zip(pending, vectors, strict=True)
                ],
                *cleared,
            ],
        )
        await session.commit()
        if ledger is not None:
            progress.usd = ledger.usd
            progress.tokens = sum(e.usage.total_tokens for e in ledger.entries)

        progress.done += len(rows)
        progress.embedded += len(pending)
        progress.cleared += len(cleared)
        report_progress(table, progress, scope)


def report_progress(table: str, progress: Progress, scope: Scope) -> None:
    percent = (progress.done / progress.total * 100) if progress.total else 100.0
    spent = progress.usd or estimate_usd(scope.model, progress.chars)
    marker = "~" if not progress.usd else " "
    print(
        f"  {table:<7} {progress.done:>7,}/{progress.total:<7,} {percent:5.1f}%  "
        f"{progress.rate:6.1f} rows/s  eta {duration(progress.eta_s):>8}  "
        f"{marker}{money(spent)}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reembed",
        description="Re-embed the Google mirror after the embedding model changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Vectors from two embedding models are not comparable, so the mirror may hold\n"
            "exactly one model's vectors at a time. This walks the rows that are not on the\n"
            "target model and rebuilds them, in batches, reporting cost as it goes.\n"
        ),
    )
    parser.add_argument(
        "--table",
        default="all",
        choices=[*TABLES, "all"],
        help="which mirror table to rebuild (default: all three)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="embedding model to rebuild with, e.g. openai:text-embedding-3-small "
        "(default: whatever EMBED_MODEL says)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"texts per embedding request (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count the work and estimate the cost, without calling the API or writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even though it would leave the mirror holding two models' vectors",
    )
    parser.add_argument(
        "--user",
        default="",
        help="one user id only. Leaves every other account on the old model, so it needs --force",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many rows. Leaves the rest on the old model, so it needs --force",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="do not ask before starting",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    from app.config import settings
    from app.db.session import session_scope, shutdown_engine
    from app.llm import embed_dimensions, embed_model_id, track_usage

    configured = embed_model_id()
    target = (args.model or configured).strip()
    tables = TABLES if args.table == "all" else (args.table,)
    scope = Scope(
        tables=tables,
        model=target,
        user_id=args.user or None,
        limit=args.limit,
    )

    print()
    print(f"  re-embed  ->  {target}  ({embed_dimensions()} dimensions)")
    print(f"  tables       {', '.join(tables)}")
    if scope.user_id:
        print(f"  user         {scope.user_id}")
    print(f"  database     {_safe_url(settings.DATABASE_URL)}")
    print()

    if target != configured:
        print(
            f"  NOTE  EMBED_MODEL is {configured}, and this run writes {target}.\n"
            f"        Search compares the query's model against each row's, so set\n"
            f"        EMBED_MODEL={target} and restart the app and the workers, or the\n"
            f"        rows this run writes will be the ones search refuses to read.\n"
        )

    try:
        async with session_scope() as session:
            surveyed = await survey(session, scope)
            work = {table: await count_work(session, table, scope) for table in tables}

            _print_survey(surveyed, target)

            total = sum(work.values())
            if scope.limit is not None:
                total = min(total, scope.limit)
            if total == 0:
                print("  Nothing to do: every row is already on this model.\n")
                return 0

            reasons = mixed_after(surveyed, scope, work)
            if reasons and not args.force:
                print("  REFUSING TO RUN — this would leave the mirror holding two models.\n")
                for reason in reasons:
                    print(f"    - {reason}")
                print(MIXING_EXPLAINED)
                return 1
            if reasons and args.force:
                print("  --force: running anyway. Search stays broken until this is finished.\n")
                for reason in reasons:
                    print(f"    - {reason}")
                print()

            chars = await _sample_chars(session, tables, scope, total)
            print(
                f"  {total:,} rows to rebuild, roughly {estimate_tokens(chars):,} tokens, "
                f"about {money(estimate_usd(target, chars))} at list price."
            )
            print("  Estimated from text length; the figure during the run is the ledger's.\n")

            if args.dry_run:
                print("  --dry-run: nothing was called and nothing was written.\n")
                return 0

            if not args.yes and not _confirm():
                print("  Stopped.\n")
                return 1

            progress = Progress(total=total)
            async with track_usage() as ledger:
                for table in tables:
                    if work.get(table):
                        await do_table(
                            session,
                            table,
                            scope,
                            progress,
                            batch_size=max(1, args.batch_size),
                            ledger=ledger,
                        )

            _print_summary(progress, scope)
            leftover = {table: await count_work(session, table, scope) for table in tables}
            remaining = sum(leftover.values())
            if remaining:
                print(
                    f"  {remaining:,} rows are still not on {target}. Run this again to "
                    f"finish them.\n"
                )
                return 1
    finally:
        await shutdown_engine()

    return 0


async def _sample_chars(session: Any, tables: Sequence[str], scope: Scope, total: int) -> int:
    """Total characters to embed, extrapolated from a sample.

    Reading every row to add up its length costs about as much as the run does.
    A few hundred rows per table gives a mean that is good enough for a figure
    printed with the word "about" in front of it.
    """
    sample_size = 200
    chars = 0
    counted = 0
    for table in tables:
        rows = await fetch_batch(session, table, scope, "", sample_size)
        for row in rows:
            chars += len(text_of(table, row))
            counted += 1
    if not counted:
        return 0
    return int((chars / counted) * total)


def _print_survey(surveyed: Survey, target: str) -> None:
    print("  the mirror today")
    for table in TABLES:
        models = surveyed.by_table.get(table, {})
        if not models:
            print(f"    {table:<8} no vectors")
            continue
        parts = [
            f"{count:,} {model or '(none)'}{' <-' if model == target else ''}"
            for model, count in sorted(models.items(), key=lambda kv: -kv[1])
        ]
        print(f"    {table:<8} {'   '.join(parts)}")
    others = surveyed.models_other_than(target)
    if others:
        listed = ", ".join(f"{model} ({count:,})" for model, count in others.items())
        print(f"\n  not on {target}: {listed}")
    print()


def _print_summary(progress: Progress, scope: Scope) -> None:
    spent = progress.usd or estimate_usd(scope.model, progress.chars)
    print()
    print(f"  done      {progress.done:,} rows in {duration(progress.elapsed)}")
    print(f"  embedded  {progress.embedded:,}   cleared (no text) {progress.cleared:,}")
    print(f"  requests  {progress.requests:,}   tokens {progress.tokens:,}")
    print(f"  cost      {money(spent)}{'' if progress.usd else ' (estimated)'}")
    print()


def _confirm() -> bool:
    if not sys.stdin.isatty():
        print("  Not a terminal, and --yes was not given.\n")
        return False
    answer = input("  Rebuild them now? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _safe_url(url: str) -> str:
    """The database URL without the password."""
    if "@" not in url:
        return url
    head, _, tail = url.rpartition("@")
    scheme, _, credentials = head.partition("://")
    user = credentials.partition(":")[0]
    return f"{scheme}://{user}:***@{tail}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n  Stopped. Rows already written keep their new model; run again to finish.\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
