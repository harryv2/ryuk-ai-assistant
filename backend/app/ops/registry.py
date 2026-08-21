"""The op registry, and the catalogue text the planner prompt embeds.

One dictionary is the single source of truth for what the system can do. The
planner is told about ops from here, `validate.py` rejects a step naming
anything not in here, and `dispatch.py` looks the instance up in here. There is
no second list anywhere that could drift.

**Aliases.** A few ops answer to more than one name — `gdrive.search_files` for
`drive.search_files`, `gmail.get_email` for `gmail.get_emails`, and so on. They
are in `REGISTRY` so a lookup never fails on a spelling, but the catalogue only
ever advertises the canonical name, so the planner learns one.
"""

from __future__ import annotations

from app.core.errors import AppError
from app.ops import drive_ops, gcal_ops, gmail_ops, meta_ops
from app.ops.base import ConfirmableOp, Op, SearchOp

#: Every op, in the order the catalogue lists them.
CANONICAL: list[Op] = [*gmail_ops.OPS, *gcal_ops.OPS, *drive_ops.OPS, *meta_ops.OPS]

#: Other spellings that mean the same op. Kept because the sample plans, the
#: fixtures and the API examples do not all use the same one, and a run must
#: not die over a synonym.
ALIASES: dict[str, str] = {
    "gdrive.search_files": "drive.search_files",
    "gdrive.get_file": "drive.get_files",
    "gdrive.get_files": "drive.get_files",
    "gdrive.create_folder": "drive.create_folder",
    "gdrive.move_file": "drive.move_file",
    "gdrive.share_file": "drive.share_file",
    "drive.get_file": "drive.get_files",
    "gmail.get_email": "gmail.get_emails",
    "gmail.resolve_person": "resolve.person",
    "gcal.get_event": "gcal.get_events",
    "gcal.free_slots": "gcal.find_free_slots",
    "meta.intersect": "gcal.find_conflicts",
    "meta.extract": "llm.extract",
    "meta.filter": "data.filter",
    "meta.map": "llm.map",
    "search.everything": "search.all",
    "ask.clarify": "ask.user",
}


def _build() -> dict[str, Op]:
    registry: dict[str, Op] = {}
    for op in CANONICAL:
        if not op.name:
            raise RuntimeError(f"{type(op).__name__} has no name")
        if op.name in registry:
            raise RuntimeError(f"two ops are called {op.name!r}")
        registry[op.name] = op
    for alias, target in ALIASES.items():
        if target not in registry:
            raise RuntimeError(f"alias {alias!r} points at {target!r}, which does not exist")
        if alias in registry:
            raise RuntimeError(f"alias {alias!r} collides with a real op")
        registry[alias] = registry[target]
    return registry


#: name -> op. Canonical names and aliases both resolve.
REGISTRY: dict[str, Op] = _build()


def get(name: str) -> Op | None:
    """The op, or ``None``. Aliases resolve; unknown names do not raise."""
    return REGISTRY.get((name or "").strip())


def require(name: str) -> Op:
    """The op, or a validation error naming what does exist."""
    op = get(name)
    if op is None:
        raise AppError(
            "VALIDATION_ERROR",
            f"There is no op called {name!r}.",
            http=422,
            details={"op": name, "known": names()},
        )
    return op


def names() -> list[str]:
    """Canonical names only, in catalogue order."""
    return [op.name for op in CANONICAL]


def by_service() -> dict[str, list[Op]]:
    grouped: dict[str, list[Op]] = {}
    for op in CANONICAL:
        grouped.setdefault(op.service, []).append(op)
    return grouped


def writes() -> list[Op]:
    return [op for op in CANONICAL if op.is_write]


def confirmables() -> list[ConfirmableOp]:
    return [op for op in CANONICAL if isinstance(op, ConfirmableOp)]


def local_ops() -> list[Op]:
    return [op for op in CANONICAL if op.is_local]


_HEADER = """Ops you may use. Anything not on this list does not exist.

Notation: name(arguments) -> fields you may reference.
  [local]   reads our own mirror of the user's data; no Google call, fast
  [write]   changes something, and runs as soon as the step runs
  [CONFIRM] changes something irreversible: the step only prepares it, and the
            person is asked before anything happens. Never plan around this.

Reference values by path instead of retyping them:
  {{step_id.field}} · {{step_id.hits[0].message_id}} · {{step_id.hits[*].id}}
  {{search.gmail[0].extracted.pnr}} · {{windows.next_week.start}}
A value you type yourself is a value you can type wrong."""

_FOOTER = """Search ops share a shape: query, filter, project, order_by,
order_dir, limit, offset, body, expect, freshness. `expect: "one"` makes the op
apply the ambiguity test and raise a choice card rather than pick for the user.
`freshness: "live"` buys one targeted Google call before the search; it costs
about 300 ms, so ask for it only when minutes matter.

Filter keys are checked against the list above. An unknown key is an error, not
a dropped filter."""


def catalogue(include_filters: bool = True) -> str:
    """The block of text the planner prompt embeds.

    Stable across runs on purpose: it is the longest fixed prefix in the
    prompt, so keeping it byte-identical is what makes prefix caching pay.
    """
    lines: list[str] = [_HEADER, ""]
    for service, ops in by_service().items():
        lines.append(f"## {service}")
        for op in ops:
            lines.append(f"  {op.catalogue_line()}")
            if include_filters and isinstance(op, SearchOp) and op.filter_spec:
                lines.append(
                    f"      filter keys: {', '.join(sorted(op.filter_spec))}"
                    f"  ·  order_by: {', '.join(sorted(op.order_spec))}"
                )
        lines.append("")
    lines.append(_FOOTER)
    return "\n".join(lines)


__all__ = [
    "ALIASES",
    "CANONICAL",
    "REGISTRY",
    "by_service",
    "catalogue",
    "confirmables",
    "get",
    "local_ops",
    "names",
    "require",
    "writes",
]
