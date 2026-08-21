"""Our JSON Schema, rewritten into the subset Anthropic's structured output accepts.

Anthropic's ``output_config.format`` takes ordinary JSON Schema, but it is strict
about two things and quietly refuses a handful of others:

* every object must list **every** one of its properties in ``required``
* every object must carry ``"additionalProperties": false``

Get either wrong and the request comes back 400. That is the same deal OpenAI's
strict mode offers, so this translation is close to a copy — much less work than
the Gemini one, which has to rebuild the schema in a different vocabulary.

What is not accepted
--------------------

Anthropic understands ``object``, ``array``, ``string``, ``integer``, ``number``,
``boolean`` and ``null``, plus ``enum``, ``const``, ``anyOf``, ``allOf`` and the
string formats listed in `SUPPORTED_FORMATS`. It does not understand:

* number ranges — ``minimum``, ``maximum``, ``multipleOf``
* string lengths and patterns — ``minLength``, ``maxLength``, ``pattern``
* list sizes and per-position shapes — ``minItems``, ``uniqueItems``, ``prefixItems``
* ``additionalProperties`` set to anything other than ``false``
* schemas that refer back to themselves

Rather than raise on those — which would turn a slightly-too-fancy schema into a
dead call with no fallback — this module **strips them and says so in plain
words in the node's ``description``**. A rule the validator cannot enforce
becomes a sentence the model can read, which is the next best thing:

    {"type": "integer", "minimum": 1, "maximum": 5}
    -> {"type": "integer", "description": "Must be 1 or more. Must be 5 or less."}

Two rewrites worth knowing about
--------------------------------

**Optional fields become nullable.** Forcing every property into ``required``
turns "you may leave this out" into "you must fill this in", and a model told it
must fill something in will invent a value. So a property that was optional gets
``null`` added to what it may be, and the note says the model may answer null.
Pass ``nullable_optionals=False`` to get the blunt version instead.

**Self-referencing schemas get cut.** A definition that contains itself cannot be
expressed at all, so the property that closes the loop is dropped and a note
names it. The call then works and returns the top level, rather than 400-ing on
the whole thing.

Pure function, no I/O, no globals touched, input never mutated. Everything here
is meant to be read and tested one case at a time.
"""

from __future__ import annotations

import json
from typing import Any, Final

# String formats Anthropic recognises. Anything else is dropped and described.
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)

# The seven type names that mean anything here.
SUPPORTED_TYPES: Final[frozenset[str]] = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)

# Keywords that survive the trip. Everything else is either described in words
# (see `_CONSTRAINTS`) or dropped without comment because it carries no meaning
# for the model — `$schema`, `$id`, `default`, `examples`, `readOnly` and so on.
_KEEP: Final[frozenset[str]] = frozenset(
    {
        "type",
        "description",
        "title",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "allOf",
        "format",
    }
)

# How deep a chain of `$ref`s may go before we call it runaway. Real schemas are
# nowhere near this; a cycle would otherwise be the only thing that stops us.
_MAX_DEPTH: Final[int] = 24


def to_anthropic_schema(json_schema: Any, *, nullable_optionals: bool = True) -> dict[str, Any]:
    """One JSON Schema in, the Anthropic-safe version of it out.

    Every object comes back with all of its properties in ``required`` and with
    ``additionalProperties`` false. Anything Anthropic would reject is removed
    and explained in that node's ``description``.

    ``nullable_optionals`` controls what happens to a property that was optional.
    On (the default), it may also be null, so a model with nothing to say has a
    legal way to say nothing. Off, it is simply required, which is the letter of
    what Anthropic wants and a good way to get invented data.

    The input is never modified. Unusable input — not a dict, or a schema that
    is nothing but constructs we had to drop — comes back as an empty object
    with a note saying so, because a provider is not the right place to raise.
    """
    if not isinstance(json_schema, dict):
        return _empty(
            "This schema could not be read, so no shape is being enforced. "
            "Answer with a JSON object."
        )

    defs = _collect_defs(json_schema)
    converted = _convert(json_schema, defs, (), 0, nullable_optionals)
    if converted is None:
        return _empty(
            "This schema used only things that cannot be expressed here, so no "
            "shape is being enforced. Answer with a JSON object."
        )
    return converted


# ------------------------------------------------------------------ the walk


def _convert(
    node: Any,
    defs: dict[str, Any],
    open_refs: tuple[str, ...],
    depth: int,
    nullable_optionals: bool,
) -> dict[str, Any] | None:
    """One node, converted. ``None`` means "this cannot be expressed — drop me".

    ``open_refs`` is the chain of ``$ref`` pointers currently being followed. A
    pointer that shows up in it twice is a loop, and a loop is the one thing we
    cannot rewrite our way out of.
    """
    if node is True:  # the "anything goes" schema
        return _empty("Any value is allowed here.")
    if node is False or not isinstance(node, dict):
        return None
    if depth > _MAX_DEPTH:
        return None

    # A `$ref` is followed and inlined. Anthropic does accept `$ref`/`$defs`,
    # but inlining is what lets us see a loop before the API does, and our
    # schemas are small enough that the duplication costs nothing.
    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in open_refs:
            return None  # the loop closes here; the caller drops whatever holds us
        target = _resolve(ref, defs)
        if target is None:
            return _empty(f"Part of this schema pointed at {ref}, which is missing.")
        merged = {k: v for k, v in node.items() if k != "$ref"}
        combined = {**target, **merged} if merged else target
        return _convert(combined, defs, (*open_refs, ref), depth + 1, nullable_optionals)

    out: dict[str, Any] = {}
    notes: list[str] = []

    _carry_description(node, out)
    _carry_type(node, out, notes)
    _carry_enum_and_const(node, out)
    _carry_format(node, out, notes)
    _describe_dropped(node, notes)

    kind = _kind_of(out, node)

    if ("anyOf" in node or "oneOf" in node) and not _carry_branches(
        node, out, notes, defs, open_refs, depth, nullable_optionals
    ):
        return None
    if "allOf" in node and not _carry_all_of(
        node, out, notes, defs, open_refs, depth, nullable_optionals
    ):
        return None

    # A list of alternatives is the whole shape; a type or a property list
    # sitting beside it is either redundant or a contradiction.
    if "anyOf" not in out:
        if kind == "object" or "properties" in node:
            _build_object(node, out, notes, defs, open_refs, depth, nullable_optionals)
        if (kind == "array" or "items" in node or "prefixItems" in node) and not _build_array(
            node, out, notes, defs, open_refs, depth, nullable_optionals
        ):
            return None

    # A node with nothing left but a description says nothing at all.
    if not any(key in out for key in ("type", "enum", "const", "anyOf", "allOf", "properties")):
        return None

    _append_notes(out, notes)
    return out


# ------------------------------------------------------------- simple keywords


def _carry_description(node: dict[str, Any], out: dict[str, Any]) -> None:
    for key in ("title", "description"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()


def _carry_type(node: dict[str, Any], out: dict[str, Any], notes: list[str]) -> None:
    """``type``, with the OpenAPI-style ``nullable: true`` folded into it."""
    raw = node.get("type")
    names: list[str] = []
    if isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, list):
        names = [t for t in raw if isinstance(t, str)]

    kept = [t for t in names if t in SUPPORTED_TYPES]
    dropped = [t for t in names if t not in SUPPORTED_TYPES]
    if dropped:
        notes.append(f"The type {', '.join(sorted(set(dropped)))} is not available here.")

    if node.get("nullable") is True and "null" not in kept:
        kept.append("null")

    if len(kept) == 1:
        out["type"] = kept[0]
    elif kept:
        out["type"] = kept


def _carry_enum_and_const(node: dict[str, Any], out: dict[str, Any]) -> None:
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        out["enum"] = list(enum)
    if "const" in node:
        out["const"] = node["const"]


def _carry_format(node: dict[str, Any], out: dict[str, Any], notes: list[str]) -> None:
    fmt = node.get("format")
    if not isinstance(fmt, str) or not fmt:
        return
    if fmt in SUPPORTED_FORMATS:
        out["format"] = fmt
    else:
        notes.append(f"Write this in {fmt} format.")


def _kind_of(out: dict[str, Any], node: dict[str, Any]) -> str:
    """The single type name this node is about, ignoring a nullable partner."""
    raw = out.get("type", node.get("type"))
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        real = [t for t in raw if t != "null"]
        if len(real) == 1:
            return real[0]
    return ""


# ------------------------------------------------------------------- objects


def _build_object(
    node: dict[str, Any],
    out: dict[str, Any],
    notes: list[str],
    defs: dict[str, Any],
    open_refs: tuple[str, ...],
    depth: int,
    nullable_optionals: bool,
) -> None:
    """Properties, then the two rules Anthropic will not bend on."""
    raw_props = node.get("properties")
    props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}

    was_required = node.get("required")
    optional_before = {
        name for name in props if not isinstance(was_required, list) or name not in was_required
    }

    # Anything an `allOf` merge already put here comes first; the node's own
    # properties are more specific, so they win where the two overlap.
    kept: dict[str, Any] = dict(out.get("properties") or {})
    dropped: list[str] = []
    made_nullable: list[str] = []

    for name, child in props.items():
        converted = _convert(child, defs, open_refs, depth + 1, nullable_optionals)
        if converted is None:
            # Usually this is the property that closed a `$ref` loop.
            dropped.append(name)
            continue
        if nullable_optionals and name in optional_before:
            widened = _make_nullable(converted)
            if widened is not None:
                converted = widened
                made_nullable.append(name)
        kept[name] = converted

    out["properties"] = kept
    out["required"] = list(kept)
    out["additionalProperties"] = False
    out.setdefault("type", "object")

    if dropped:
        notes.append(
            f"{_and_list(dropped)} had to be left out because "
            f"{'they refer' if len(dropped) > 1 else 'it refers'} back to this same shape."
        )
    if made_nullable:
        notes.append(f"Answer null for {_and_list(made_nullable)} when there is nothing to say.")
    elif optional_before and not nullable_optionals:
        names = sorted(n for n in optional_before if n in kept)
        if names:
            notes.append(f"{_and_list(names)} used to be optional but must now be filled in.")

    extra = node.get("additionalProperties")
    if isinstance(extra, dict) and not kept:
        notes.append(
            "This was a free-form map of keys to values, which cannot be described "
            "here. Answer with an empty object, or ask for named fields instead."
        )
    elif extra is not None and extra is not False:
        notes.append("Only the fields listed above are allowed.")

    for key, phrase in (("minProperties", "at least"), ("maxProperties", "at most")):
        value = node.get(key)
        if isinstance(value, int):
            notes.append(f"Fill in {phrase} {value} of these fields.")


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any] | None:
    """The same schema, plus permission to be null. ``None`` when that is impossible."""
    out = dict(schema)

    raw = out.get("type")
    if isinstance(raw, str):
        if raw == "null":
            return out
        out["type"] = [raw, "null"]
        if isinstance(out.get("enum"), list) and None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
        return out
    if isinstance(raw, list):
        if "null" not in raw:
            out["type"] = [*raw, "null"]
        if isinstance(out.get("enum"), list) and None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
        return out
    if isinstance(out.get("enum"), list):
        if None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
        return out

    # A list of alternatives just gains one more alternative. Wrapping it in a
    # second anyOf would mean the same thing and read much worse.
    branches = out.get("anyOf")
    if isinstance(branches, list) and not any(
        key in out for key in ("allOf", "const", "properties", "items")
    ):
        if not any(b.get("type") == "null" for b in branches if isinstance(b, dict)):
            out["anyOf"] = [*branches, {"type": "null"}]
        return out

    # allOf and const: wrap the whole thing, keeping the words on the outside
    # where a reader will find them.
    if any(key in out for key in ("anyOf", "allOf", "const")):
        wrapper: dict[str, Any] = {}
        inner = dict(out)
        for key in ("title", "description"):
            if key in inner:
                wrapper[key] = inner.pop(key)
        wrapper["anyOf"] = [inner, {"type": "null"}]
        return wrapper
    return None


# -------------------------------------------------------------------- arrays


def _build_array(
    node: dict[str, Any],
    out: dict[str, Any],
    notes: list[str],
    defs: dict[str, Any],
    open_refs: tuple[str, ...],
    depth: int,
    nullable_optionals: bool,
) -> bool:
    """The item shape. Returns False when the list has no usable shape left."""
    items = node.get("items")
    prefix = node.get("prefixItems")

    if isinstance(prefix, list) and prefix:
        notes.append("Every item in this list has the same shape.")
        items = items if isinstance(items, dict) else prefix[0]
    elif isinstance(items, list):
        notes.append("Every item in this list has the same shape.")
        items = items[0] if items else None

    if items is None:
        # A list with no stated item shape is not something we can send.
        out.pop("type", None)
        notes.append("A list was asked for here but its contents were never described.")
        return bool(out.get("anyOf") or out.get("allOf") or out.get("enum") or out.get("const"))

    converted = _convert(items, defs, open_refs, depth + 1, nullable_optionals)
    if converted is None:
        return False

    out["items"] = converted
    out.setdefault("type", "array")

    if isinstance(node.get("contains"), dict):
        notes.append("At least one item must match what the description asks for.")
    return True


# ------------------------------------------------------------------ branches


def _carry_branches(
    node: dict[str, Any],
    out: dict[str, Any],
    notes: list[str],
    defs: dict[str, Any],
    open_refs: tuple[str, ...],
    depth: int,
    nullable_optionals: bool,
) -> bool:
    """``anyOf``, and ``oneOf`` folded into it. False when no branch survived."""
    raw = node.get("anyOf")
    if not isinstance(raw, list):
        raw = node.get("oneOf")
        if isinstance(raw, list):
            notes.append("Exactly one of these shapes should match.")
    if not isinstance(raw, list) or not raw:
        return True

    branches = [
        converted
        for branch in raw
        if (converted := _convert(branch, defs, open_refs, depth + 1, nullable_optionals))
        is not None
    ]
    if not branches:
        return False
    if len(branches) < len(raw):
        notes.append("Some of the shapes offered here had to be left out.")

    out["anyOf"] = branches
    # A type sitting next to anyOf is redundant at best and contradictory at worst.
    out.pop("type", None)
    return True


def _carry_all_of(
    node: dict[str, Any],
    out: dict[str, Any],
    notes: list[str],
    defs: dict[str, Any],
    open_refs: tuple[str, ...],
    depth: int,
    nullable_optionals: bool,
) -> bool:
    """``allOf``, merged into one object when every branch is an object.

    Merging matters: ``additionalProperties: false`` on each branch separately
    means no value can satisfy all of them at once, which is not what whoever
    wrote the schema had in mind.
    """
    raw = node.get("allOf")
    if not isinstance(raw, list) or not raw:
        return True

    branches = [
        converted
        for branch in raw
        if (converted := _convert(branch, defs, open_refs, depth + 1, nullable_optionals))
        is not None
    ]
    if not branches:
        return False

    if all(b.get("type") == "object" for b in branches):
        props: dict[str, Any] = dict(out.get("properties") or {})
        for branch in branches:
            props.update(branch.get("properties") or {})
            text = branch.get("description")
            if isinstance(text, str) and text.strip() and text not in notes:
                notes.append(text.strip())
        out["type"] = "object"
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
        return True

    out["allOf"] = branches
    return True


# ------------------------------------------------------- what we had to drop


# A rule Anthropic will not check, rewritten as a sentence a model can follow.
# `None` back from one of these means "nothing worth saying".
_CONSTRAINTS: Final[dict[str, Any]] = {
    "minimum": lambda v: f"Must be {v} or more.",
    "maximum": lambda v: f"Must be {v} or less.",
    "exclusiveMinimum": lambda v: f"Must be more than {v}.",
    "exclusiveMaximum": lambda v: f"Must be less than {v}.",
    "multipleOf": lambda v: f"Must be a multiple of {v}.",
    "minLength": lambda v: f"Must be at least {_count(v, 'character')} long.",
    "maxLength": lambda v: f"Must be no longer than {_count(v, 'character')}.",
    "pattern": lambda v: f"Must match this pattern: {v}",
    "minItems": lambda v: f"Include at least {_count(v, 'item')}.",
    "maxItems": lambda v: f"Include no more than {_count(v, 'item')}.",
    "uniqueItems": lambda v: "Every item must be different." if v else None,
    "not": lambda v: "Some shapes are not allowed here.",
    "if": lambda v: "Which fields apply depends on the values of the others.",
    "dependentRequired": lambda v: "Some of these fields have to be filled in together.",
    "propertyNames": lambda v: "The field names themselves are restricted.",
}


def _describe_dropped(node: dict[str, Any], notes: list[str]) -> None:
    """Turn every rule we cannot send into a sentence, in a fixed order.

    Fixed order, not dictionary order, so the same schema always produces the
    same bytes. Anthropic caches a compiled schema for a day and prompt caching
    is a prefix match, so a description that reshuffles itself between calls
    would quietly throw both away.
    """
    for key, phrase in _CONSTRAINTS.items():
        if key not in node:
            continue
        try:
            sentence = phrase(node[key])
        except Exception:  # a hostile value in a schema is not worth a crash
            sentence = None
        if sentence:
            notes.append(sentence)


def _append_notes(out: dict[str, Any], notes: list[str]) -> None:
    if not notes:
        return
    existing = out.get("description", "")
    parts = [existing.strip()] if isinstance(existing, str) and existing.strip() else []
    seen: set[str] = set()
    for note in notes:
        if note not in seen:
            seen.add(note)
            parts.append(note)
    out["description"] = " ".join(parts)


# ------------------------------------------------------------------- helpers


def _collect_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Everything a ``$ref`` might point at, keyed by pointer.

    Both spellings, because schemas in the wild use both, and both bare names so
    that a hand-written ``{"$ref": "Address"}`` still resolves.
    """
    found: dict[str, Any] = {}
    for holder in ("$defs", "definitions"):
        block = schema.get(holder)
        if not isinstance(block, dict):
            continue
        for name, value in block.items():
            if isinstance(value, dict):
                found[f"#/{holder}/{name}"] = value
                found.setdefault(name, value)
    return found


def _resolve(pointer: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    if pointer in defs:
        return defs[pointer]
    return defs.get(pointer.rsplit("/", 1)[-1])


def _empty(note: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
        "description": note,
    }


def _count(value: Any, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


def _and_list(names: list[str]) -> str:
    quoted = [f"'{n}'" for n in names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """A stable string for a translated schema, for logging and for cache checks.

    Sorted keys on purpose. Anthropic keeps a compiled copy of each schema for a
    day, and the same schema serialised two different ways is two schemas to it.
    """
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


__all__ = [
    "SUPPORTED_FORMATS",
    "SUPPORTED_TYPES",
    "schema_fingerprint",
    "to_anthropic_schema",
]
