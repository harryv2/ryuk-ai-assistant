"""Plan validation. Pure Python, no model, no I/O, before anything executes.

One LLM call produces a plan. Everything after that point is deterministic, and
this is the gate it goes through first: turn a plausible-looking blob of JSON
into something safe to run, or refuse it — cheaply, before a Google call has
been made and before a write has been staged.

Both halves matter. A validator that rejects too much makes the system look
broken; a validator that lets a write through under a read intent makes it
dangerous. So every rule below is narrow and every rejection names itself:

* the op exists in the registry;
* the args satisfy that op's own model, with ``{{references}}`` stood in for;
* every reference points at a step that is upstream — transitively, not merely
  present — and names a field that op declares, after one singular/plural
  correction;
* the graph is acyclic, at most twelve steps, at most three Google-touching
  steps per service;
* a ``needs_confirm`` step has something to confirm, and a write only exists
  under an intent that admits it is a write;
* a write below the confidence floor has a question standing in front of it;
* a speculative step's whole subtree is local and read-only;
* the services are the ones the intent named.
"""

from __future__ import annotations

from collections.abc import Sequence

import re
import typing
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

from app.ops.base import service_of

MAX_STEPS: Final[int] = 12
MAX_GOOGLE_STEPS_PER_SERVICE: Final[int] = 3
WRITE_CONFIDENCE_FLOOR: Final[float] = 0.75

# Roots that exist before any step runs, so they never need a dependency edge.
AMBIENT_ROOTS: Final[frozenset[str]] = frozenset(
    {"search", "windows", "probe", "user", "now", "tz", "run", "conversation", "intent"}
)

# Services that are ours rather than Google's, listed by namespace exactly as
# the registry spells them. `intent.services` is what the user was shown — the
# mailbox, the calendar, the drive — so requiring these to appear in it would
# mean every ambiguity card and every summarising step rewrote that list.
#
# Anything NOT here has to be named in the intent, which is the rule that stops
# a plan reaching a service nobody was told about.
LOCAL_SERVICES: Final[frozenset[str]] = frozenset(
    {
        "ask",       # ask.user, ask.clarify — a question for the person
        "meta",      # meta.map, meta.filter, meta.extract, meta.intersect
        "llm",       # llm.map, llm.extract — the same ops under their own name
        "data",      # data.filter
        "resolve",   # resolve.person, resolve.reference
        "search",    # search.all — our mirror, all three corpora at once
        "page",      # page.more — paging a result already on screen
        "action",    # action.revise — editing a write that is already staged
        "chat",      # chat.set_title — naming this conversation
        "local",
        "self",
    }
)

VALID_EXPECT: Final[frozenset[str]] = frozenset({"one", "many"})
VALID_FRESHNESS: Final[frozenset[str]] = frozenset({"cached", "live"})
VALID_TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "event_list",
        "email_list",
        "file_list",
        "free_slots",
        "conflict_list",
        "summary_list",
        "count_answer",
        "empty_result",
        "degraded_banner",
        "reconnect_card",
    }
)

_REFERENCE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Accepted or refused, with the reason spelled out either way.

    ``plan`` is a corrected copy — the singular/plural reference fix is applied
    here so the dispatcher never has to guess what the planner meant.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.ok

    @property
    def message(self) -> str:
        return "; ".join(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def _registry(registry: Any) -> dict[str, Any]:
    if registry is None:
        from app.ops.registry import REGISTRY  # imported late: another module owns it

        return REGISTRY
    if hasattr(registry, "get") and not isinstance(registry, (list, tuple)):
        return registry  # a dict, or anything dict-shaped
    return {getattr(op, "name", str(op)): op for op in registry}


def _op_field(op: Any, name: str, default: Any) -> Any:
    value = getattr(op, name, default)
    return default if value is None else value


def _service_of(op_name: str) -> str:
    """The service a step reaches, under the one name the intent uses.

    The intent only ever says ``gdrive``, so comparing raw namespaces rejects
    a perfectly good ``drive.search_files`` plan and buys a repair round for
    nothing.
    """
    return service_of(op_name)


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """One ``{{...}}`` occurrence: where it was, and what it points at."""

    raw: str  # the full "{{ ... }}" text
    path: str  # the path with filters stripped
    root: str
    rest: list[str]  # path segments after the root, indexes removed
    location: str  # a human-readable place, for the error message


def _split_path(path: str) -> list[str]:
    """``hits[0].extracted.pnr`` becomes ``[hits, extracted, pnr]``."""
    segments: list[str] = []
    for chunk in path.split("."):
        name = _SEGMENT.match(chunk.strip())
        if name:
            segments.append(name.group(1))
    return segments


def references_in(value: Any, location: str = "args") -> list[Reference]:
    """Every reference anywhere inside an argument tree."""
    out: list[Reference] = []

    def walk(node: Any, place: str) -> None:
        if isinstance(node, str):
            for match in _REFERENCE.finditer(node):
                body = match.group(1).strip()
                path = body.split("|", 1)[0].strip()
                segments = _split_path(path)
                if not segments:
                    continue
                out.append(
                    Reference(match.group(0), path, segments[0], segments[1:], place)
                )
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{place}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                walk(child, f"{place}[{index}]")

    walk(value, location)
    return out


def _plural_variants(name: str) -> list[str]:
    """Every spelling of a field name a planner reasonably reaches for."""
    variants = [name + "s", name + "es"]
    if name.endswith("s"):
        variants.append(name[:-1])
    if name.endswith("ies"):
        variants.append(name[:-3] + "y")
    if name.endswith("y"):
        variants.append(name[:-1] + "ies")
    if name.endswith("es"):
        variants.append(name[:-2])
    return [v for v in variants if v and v != name]


def _rewrite_reference(node: Any, old: str, new: str) -> Any:
    """Replace a corrected reference everywhere it appears in the tree."""
    if isinstance(node, str):
        return node.replace(old, new)
    if isinstance(node, dict):
        return {key: _rewrite_reference(child, old, new) for key, child in node.items()}
    if isinstance(node, list):
        return [_rewrite_reference(child, old, new) for child in node]
    return node


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _has_reference(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_REFERENCE.search(value))
    if isinstance(value, dict):
        return any(_has_reference(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_reference(v) for v in value)
    return False


def _limits(field: Any) -> dict[str, Any]:
    """The bounds a field declares — ``Ge(5)``, ``MinLen(1)`` and the rest.

    Pydantic keeps them as annotated-types markers on ``FieldInfo.metadata``
    rather than on the annotation, so a placeholder built from the annotation
    alone lands outside them and fails a check it was never meant to make.
    """
    found: dict[str, Any] = {}
    for marker in getattr(field, "metadata", ()) or ():
        for name in ("ge", "gt", "le", "lt", "min_length", "max_length", "multiple_of"):
            value = getattr(marker, name, None)
            if value is not None:
                found[name] = value
    return found


def _number_in(limits: dict[str, Any], *, integral: bool) -> Any:
    """The smallest sane number inside the declared bounds."""
    low = limits.get("ge")
    if low is None and limits.get("gt") is not None:
        low = limits["gt"] + (1 if integral else 1.0)
    value = low if low is not None else 1

    high = limits.get("le")
    if high is None and limits.get("lt") is not None:
        high = limits["lt"] - (1 if integral else 1.0)
    if high is not None and value > high:
        value = high

    step = limits.get("multiple_of")
    if step:
        multiple = -(-value // step) * step  # round up onto the step
        if high is None or multiple <= high:
            value = multiple
    return int(value) if integral else float(value)


def _placeholder(annotation: Any, limits: dict[str, Any] | None = None) -> Any:
    """A value of the right shape, standing in for one that binds at run time.

    A reference is a promise, not a value, so type-checking the promise proves
    nothing. Substituting something of the declared type — and inside the
    declared bounds — keeps every *other* check meaningful: required fields,
    unknown keys, literal types, the shape of a nested object.
    """
    limits = limits or {}
    if annotation is None or annotation is Any:
        return "x" * max(1, int(limits.get("min_length") or 1))

    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]

    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return _placeholder(args[0], limits) if args else "x"
    if origin in (list, set, tuple, frozenset):
        inner = _placeholder(args[0]) if args else "x"
        return [inner] * max(1, int(limits.get("min_length") or 1))
    if origin is dict:
        return {}
    if origin is typing.Literal:
        return args[0] if args else "x"

    if isinstance(annotation, type):
        if issubclass(annotation, bool):
            return True
        if issubclass(annotation, int):
            return _number_in(limits, integral=True)
        if issubclass(annotation, float):
            return _number_in(limits, integral=False)
        if issubclass(annotation, str):
            length = max(1, int(limits.get("min_length") or 1))
            maximum = limits.get("max_length")
            if maximum is not None:
                length = min(length, int(maximum))
            return "x" * length
        if issubclass(annotation, (list, tuple, set)):
            return ["x"]
        if issubclass(annotation, dict):
            return {}
    return "x"


def _blank_references(value: Any, annotation: Any = None, limits: dict[str, Any] | None = None) -> Any:
    """Swap every reference for a placeholder of the declared type."""
    if isinstance(value, str) and _REFERENCE.search(value):
        return _placeholder(annotation, limits)
    if isinstance(value, dict):
        return {key: _blank_references(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_blank_references(child) for child in value]
    return value


def _only_about(exc: Exception, keys: set[str]) -> bool:
    """True when every complaint is about a field whose value binds later.

    A pattern, a custom validator or a cross-field rule can reject a stand-in
    that no placeholder could have satisfied. Refusing the plan for that would
    be refusing it for a value nobody has seen yet — so a failure that names
    only reference-carrying fields, and is not a missing-field failure, is not
    a reason to reject. Anything the planner wrote out in full still is.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return False
    try:
        items = list(exc.errors())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False
    if not items:
        return False
    for item in items:
        location = item.get("loc") or ()
        if not location or str(location[0]) not in keys:
            return False
        if str(item.get("type", "")).startswith("missing"):
            return False
    return True


def _check_args(op: Any, args: Any) -> list[str]:
    model = _op_field(op, "args_model", None)
    if not isinstance(args, dict):
        return ["args must be an object"]
    if model is None:
        return []

    fields = getattr(model, "model_fields", {}) or {}
    prepared: dict[str, Any] = {}
    referenced: set[str] = set()
    for key, value in args.items():
        field = fields.get(key)
        if _has_reference(value):
            referenced.add(key)
            prepared[key] = _blank_references(
                value, getattr(field, "annotation", None), _limits(field)
            )
        else:
            prepared[key] = value

    try:
        model.model_validate(prepared)
    except Exception as exc:  # noqa: BLE001 - pydantic's message is the reason
        if referenced and _only_about(exc, referenced):
            return []
        return [_pydantic_reason(exc)]
    return []


def _pydantic_reason(exc: Exception) -> str:
    """Flatten a pydantic ValidationError into one readable clause."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = []
            for item in exc.errors():  # type: ignore[attr-defined]
                where = ".".join(str(p) for p in item.get("loc", ())) or "args"
                parts.append(f"{where}: {item.get('msg', 'invalid')}")
            if parts:
                return "; ".join(parts)
        except Exception:  # noqa: BLE001 - fall back to the string form
            pass
    return str(exc).replace("\n", " ")[:400]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _ancestors(step_id: str, edges: dict[str, list[str]]) -> set[str]:
    """Every step upstream of this one, however many edges away."""
    seen: set[str] = set()
    stack = list(edges.get(step_id, ()))
    while stack:
        current = stack.pop()
        if current in seen or current == step_id:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return seen


def _descendants(step_id: str, edges: dict[str, list[str]]) -> set[str]:
    children: dict[str, list[str]] = {}
    for node, parents in edges.items():
        for parent in parents:
            children.setdefault(parent, []).append(node)

    seen: set[str] = set()
    stack = list(children.get(step_id, ()))
    while stack:
        current = stack.pop()
        if current in seen or current == step_id:
            continue
        seen.add(current)
        stack.extend(children.get(current, ()))
    return seen


def _find_cycle(ids: Iterable[str], edges: dict[str, list[str]]) -> list[str] | None:
    """A cycle as the path around it, so the message can show the loop."""
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for parent in edges.get(node, ()):
            if parent not in edges:
                continue
            colour = state.get(parent, 0)
            if colour == 1:
                return path[path.index(parent):] + [parent]
            if colour == 0:
                found = visit(parent)
                if found:
                    return found
        state[node] = 2
        path.pop()
        return None

    for node in ids:
        if state.get(node, 0) == 0:
            found = visit(node)
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def validate_plan(
    plan: Any,
    *,
    registry: Any = None,
    intent: dict[str, Any] | None = None,
    confidence_floor: float = WRITE_CONFIDENCE_FLOOR,
    max_steps: int = MAX_STEPS,
    known_windows: Sequence[str] | None = None,
) -> ValidationResult:
    """Check a plan and hand back a corrected copy, or say why not."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(plan, dict):
        return ValidationResult(False, [f"a plan must be an object, got {type(plan).__name__}"])

    verb = str(plan.get("type") or "plan")
    if verb != "plan":
        return _validate_other_verb(verb, plan)

    ops = _registry(registry)
    checked = dict(plan)
    declared_intent = dict(intent or checked.get("intent") or {})
    checked["intent"] = declared_intent

    raw_steps = checked.get("steps")
    if raw_steps is None or not isinstance(raw_steps, list):
        return ValidationResult(False, ["a plan needs a `steps` list"], plan=checked)
    if not raw_steps:
        return ValidationResult(
            False,
            [
                "a plan with no steps is not an answer — return "
                '{"type":"answer","text":"..."} instead of an empty plan'
            ],
            plan=checked,
        )
    if len(raw_steps) > max_steps:
        errors.append(
            f"too many steps: {len(raw_steps)}, and at most {max_steps} may run in one plan"
        )

    steps: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            errors.append(f"step {index} is not an object")
            continue
        step = dict(raw)
        step_id = str(step.get("id") or step.get("node_id") or "").strip()
        if not step_id:
            errors.append(f"step {index} has no id")
            continue
        step["id"] = step_id
        if step_id in ids:
            errors.append(f"duplicate step id {step_id!r} — every id must be unique in a plan")
            continue
        ids.append(step_id)
        steps.append(step)

    if errors and not steps:
        return ValidationResult(False, errors, warnings, checked)

    by_id = {step["id"]: step for step in steps}
    edges = {
        step["id"]: [str(dep) for dep in (step.get("depends_on") or [])] for step in steps
    }

    # -- ops exist ---------------------------------------------------------
    for step in steps:
        name = str(step.get("op") or "")
        if not name:
            errors.append(f"step {step['id']!r} names no op")
        elif ops.get(name) is None:
            errors.append(
                f"step {step['id']!r} uses {name!r}, which is not a registered op — "
                "the registry is the whole list of what exists"
            )

    if errors:  # nothing below can be trusted once an op is unknown
        return ValidationResult(False, errors, warnings, checked)

    # -- edges point somewhere ---------------------------------------------
    for step in steps:
        for dep in edges[step["id"]]:
            if dep not in by_id:
                errors.append(
                    f"step {step['id']!r} depends on {dep!r}, which is not a step in this plan"
                )

    # -- acyclic ------------------------------------------------------------
    cycle = _find_cycle(ids, {k: [d for d in v if d in by_id] for k, v in edges.items()})
    if cycle:
        errors.append("the steps form a cycle: " + " -> ".join(cycle))

    if errors:  # a cycle makes "upstream" meaningless
        return ValidationResult(False, errors, warnings, checked)

    # -- args, references, flags -------------------------------------------
    services_used: dict[str, int] = {}
    write_steps: list[str] = []
    ask_steps: list[str] = []

    for step in steps:
        step_id = step["id"]
        op = ops[str(step["op"])]
        op_name = str(step["op"])

        for reason in _check_args(op, step.get("args")):
            errors.append(f"step {step_id!r}: args are invalid for {op_name} — {reason}")

        step["args"] = _check_references(
            step, op, by_id, edges, ops, errors, known_windows=known_windows
        )

        expect = str(step.get("expect") or "many")
        if expect not in VALID_EXPECT:
            warnings.append(f"step {step_id!r}: unknown expect {expect!r}, reading it as 'many'")
            step["expect"] = "many"
        freshness = str(step.get("freshness") or "cached")
        if freshness not in VALID_FRESHNESS:
            warnings.append(
                f"step {step_id!r}: unknown freshness {freshness!r}, reading it as 'cached'"
            )
            step["freshness"] = "cached"

        service = _service_of(op_name)
        if not _op_field(op, "is_local", False):
            services_used[service] = services_used.get(service, 0) + 1

        if _op_field(op, "is_write", False):
            write_steps.append(step_id)
        if op_name == "ask.user":
            ask_steps.append(step_id)

        # A confirm gate with nothing behind it would write a prompt gating no
        # action, or an action with no side effect. `actions.requires_input_id`
        # is NOT NULL; neither shape should reach the database.
        wants_confirm = bool(step.get("needs_confirm")) or bool(
            _op_field(op, "needs_confirm", False)
        )
        if wants_confirm and not _op_field(op, "is_write", False):
            errors.append(
                f"step {step_id!r} is marked needs_confirm, but {op_name} writes nothing — "
                "there would be no action to confirm and nothing to undo"
            )

    # -- Google budget ------------------------------------------------------
    for service, count in services_used.items():
        if count > MAX_GOOGLE_STEPS_PER_SERVICE:
            errors.append(
                f"{count} steps touch {service} directly, and at most "
                f"{MAX_GOOGLE_STEPS_PER_SERVICE} may — fan out inside one step instead"
            )

    # -- writes -------------------------------------------------------------
    has_write = bool(declared_intent.get("has_write"))
    if write_steps and not has_write:
        errors.append(
            f"step(s) {', '.join(repr(s) for s in write_steps)} write, but the intent says "
            "has_write is false — a plan that writes under a read intent is either a "
            "planner mistake or an injection, and from here they look the same"
        )

    confidence = declared_intent.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else 1.0
    except (TypeError, ValueError):
        confidence_value = 1.0
        warnings.append(f"intent confidence {confidence!r} is not a number; reading it as 1.0")

    if write_steps and confidence_value < confidence_floor:
        for step_id in write_steps:
            upstream = _ancestors(step_id, edges)
            if not any(gate in upstream for gate in ask_steps):
                errors.append(
                    f"step {step_id!r} writes at confidence {confidence_value:.2f}, under the "
                    f"{confidence_floor:.2f} floor, with no ask.user question in front of it — "
                    "below the floor a write needs a person to confirm what it is aimed at"
                )

    # -- services -----------------------------------------------------------
    declared_services = {str(s).lower() for s in (declared_intent.get("services") or [])}
    for step in steps:
        service = _service_of(str(step["op"]))
        if service in LOCAL_SERVICES:
            continue
        if declared_services and service not in declared_services:
            errors.append(
                f"step {step['id']!r} uses the {service} service, which the intent did not "
                f"name (it named {sorted(declared_services) or 'nothing'}) — the intent is "
                "what the user was shown"
            )

    # -- speculation --------------------------------------------------------
    errors.extend(_check_speculation(steps, by_id, edges, ops))

    # -- answer style -------------------------------------------------------
    style = str(checked.get("answer_style") or "prose")
    if style.startswith("template:"):
        template = style.split(":", 1)[1]
        if template not in VALID_TEMPLATES:
            warnings.append(
                f"unknown template {template!r}; the renderer will fall back to summary_list"
            )
    elif style not in ("card", "prose"):
        warnings.append(f"unknown answer_style {style!r}; the renderer will fall back to prose")

    checked["steps"] = steps
    return ValidationResult(not errors, errors, warnings, checked)


def _check_references(
    step: dict[str, Any],
    op: Any,
    by_id: dict[str, dict[str, Any]],
    edges: dict[str, list[str]],
    ops: dict[str, Any],
    errors: list[str],
    known_windows: Sequence[str] | None = None,
) -> Any:
    """Check every reference in one step's args, correcting what is correctable."""
    args = step.get("args")
    step_id = step["id"]
    upstream = _ancestors(step_id, edges)

    for reference in references_in(args, f"step {step_id!r} args"):
        root = reference.root

        if root in AMBIENT_ROOTS:
            # The probe, the pre-pass and the user exist already — but a
            # window reference still has to name a window this run computed.
            # `{{windows.tomorrow.start}}` over a query that never said
            # "tomorrow" passes every other check and then dies at bind time,
            # after the searches have already run. Caught here, it costs one
            # repair round instead of a degraded answer.
            if (
                root == "windows"
                and known_windows is not None
                and reference.rest
                and reference.rest[0] not in known_windows
            ):
                have = ", ".join(sorted(known_windows)) or "none"
                errors.append(
                    f"{reference.raw} in step {step_id!r} names window "
                    f"{reference.rest[0]!r}, but this run computed: {have}. Use one "
                    "of those, or write literal ISO datetimes instead"
                )
            continue

        target_id = root
        rest = reference.rest
        if root == "steps":  # {{steps.mail.hits[0].id}} is the same thing
            if not rest:
                errors.append(f"{reference.raw} in step {step_id!r} names no step")
                continue
            target_id, rest = rest[0], rest[1:]

        if target_id == step_id:
            errors.append(f"{reference.raw} in step {step_id!r} refers to itself")
            continue

        if target_id not in by_id:
            errors.append(
                f"{reference.raw} in step {step_id!r} refers to {target_id!r}, which is not "
                "a step in this plan — there is no such step to read from"
            )
            continue

        if target_id not in upstream:
            errors.append(
                f"{reference.raw} in step {step_id!r} reads from {target_id!r}, which is not "
                f"upstream of it — add {target_id!r} to depends_on, or the two run at the "
                "same moment and the reference binds to nothing"
            )
            continue

        if not rest:
            continue  # the whole result, which is always available

        target_op = ops.get(str(by_id[target_id]["op"]))
        declared = list(_op_field(target_op, "output_fields", []) or [])
        if not declared:
            continue  # an op that declares nothing constrains nothing

        wanted = rest[0]
        if wanted in declared:
            continue

        # The planner writes `{{mail.hit[0].id}}` often enough that rejecting it
        # would cost a replan for nothing. Correct it, then check again.
        corrected = next((v for v in _plural_variants(wanted) if v in declared), None)
        if corrected is not None:
            fixed = reference.raw.replace(f".{wanted}", f".{corrected}", 1)
            args = _rewrite_reference(args, reference.raw, fixed)
            continue

        errors.append(
            f"{reference.raw} in step {step_id!r} asks {target_id!r} for {wanted!r}, which "
            f"{by_id[target_id]['op']} does not declare as an output field "
            f"(it declares {', '.join(declared)})"
        )

    step["args"] = args
    return args


def _check_speculation(
    steps: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    edges: dict[str, list[str]],
    ops: dict[str, Any],
) -> list[str]:
    """A speculative step runs before we know it is needed.

    That is only safe when nothing it does can be seen from outside: no Google
    call, no quota spent, no write, nothing to undo. Our own mirror is fair
    game — and so the whole subtree hanging off it has to be too, because the
    subtree is what gets run.
    """
    errors: list[str] = []
    for step in steps:
        if not step.get("speculate"):
            continue
        step_id = step["id"]

        for node_id in [step_id, *sorted(_descendants(step_id, edges))]:
            node = by_id.get(node_id)
            if node is None:
                continue
            op = ops.get(str(node["op"]))
            where = (
                "it" if node_id == step_id else f"its subtree reaches {node_id!r}, which"
            )
            if _op_field(op, "is_write", False):
                errors.append(
                    f"step {step_id!r} is marked speculate but {where} writes "
                    f"({node['op']}) — a speculative write has a side effect that cannot "
                    "be taken back if the guess was wrong"
                )
                break
            if not _op_field(op, "is_local", False):
                errors.append(
                    f"step {step_id!r} is marked speculate but {where} calls Google "
                    f"({node['op']}) — speculation is only safe on a local read-only "
                    "subtree over our own mirror, because the quota is shared"
                )
                break
    return errors


def _validate_other_verb(verb: str, payload: dict[str, Any]) -> ValidationResult:
    """`answer`, `revise` and `answer_input` are small and checked in full."""
    if verb == "answer":
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return ValidationResult(False, ["an `answer` needs non-empty text"], plan=payload)
        return ValidationResult(True, plan=payload)

    if verb == "revise":
        if not payload.get("action_id"):
            return ValidationResult(False, ["a `revise` needs an action_id"], plan=payload)
        if not isinstance(payload.get("patch"), dict) or not payload["patch"]:
            return ValidationResult(
                False, ["a `revise` needs a non-empty patch object"], plan=payload
            )
        return ValidationResult(True, plan=payload)

    if verb == "answer_input":
        if not payload.get("input_id"):
            return ValidationResult(False, ["an `answer_input` needs an input_id"], plan=payload)
        if "value" not in payload:
            return ValidationResult(False, ["an `answer_input` needs a value"], plan=payload)
        return ValidationResult(True, plan=payload)

    return ValidationResult(
        False,
        [
            f"unknown verb {verb!r} — a response is one of plan, answer, revise, answer_input"
        ],
        plan=payload,
    )


# Names the rest of the system reaches for.
validate = validate_plan
check_plan = validate_plan


__all__ = [
    "AMBIENT_ROOTS",
    "MAX_GOOGLE_STEPS_PER_SERVICE",
    "MAX_STEPS",
    "WRITE_CONFIDENCE_FLOOR",
    "Reference",
    "ValidationResult",
    "check_plan",
    "references_in",
    "validate",
    "validate_plan",
]
