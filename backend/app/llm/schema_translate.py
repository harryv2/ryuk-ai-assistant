"""JSON Schema in, the small thing Gemini actually accepts out.

OpenAI takes a whole JSON Schema in strict mode, `$defs` and all. Gemini takes a
cut-down version of OpenAPI 3.0's Schema object: a handful of types, `enum` on
strings, `items`, `properties`, `required`, `propertyOrdering`, and very little
else. Hand it a keyword it does not know and the answer is a 400 — which, in
`app.llm.router`, is `INVALID`, which is deliberately *not* in
`LLM_FALLBACK_ON`. So a schema this file gets wrong does not fail over to
another provider; it fails the run. That is the reason this is careful.

One function is the point::

    from app.llm.schema_translate import to_gemini_schema
    body["generationConfig"]["responseSchema"] = to_gemini_schema(plan_schema)

Pure, deterministic, no I/O, no network, no SDK. It never mutates what you give
it. `translate` is the same work with the paperwork kept — what was dropped and
why — for logging and for tests.

What survives the trip
----------------------

Emitted, because Gemini documents them: `type` (upper case, its enum spelling),
`description`, `nullable`, `enum` on strings, `format` for the handful of values
it knows, `items`, `properties`, `required`, `propertyOrdering`, `minItems`,
`maxItems`.

Everything else is **stripped and said in words** — appended as a sentence to
that field's `description`, where the model still reads it::

    {"type": "string", "pattern": "^[a-z_]+$", "maxLength": 40}

    {"type": "STRING",
     "description": 'Must match the pattern "^[a-z_]+$". At most 40 characters.'}

The constraint stops being enforced and becomes an instruction. That is a real
loss and it is the right trade: an unenforced rule the model usually follows
beats a 400 that ends the run. It applies to `pattern`, `minimum`/`maximum`,
`minLength`/`maxLength`, `multipleOf`, `uniqueItems`, `not`, `if`/`then`,
`patternProperties`, `propertyNames`, `default`, unknown `format` strings, and
enums whose values are not strings.

Three that are handled rather than merely dropped:

* **`$ref`** is resolved against `$defs` / `definitions` / `components.schemas`
  and inlined. A ref that points at itself cannot be inlined — it would not
  terminate — and is treated as inexpressible (below).
* **`allOf`** is merged into one node, because the merge is exact.
* **`anyOf` / `oneOf`** collapses to `nullable` when the only alternative is
  `null` — `["string", "null"]` too. Otherwise the first branch is kept and the
  rest are described in words. Gemini has no top-level union, and picking a
  branch keeps the *shape* the caller is going to parse.

`additionalProperties: false` is not dropped so much as already true: Gemini's
structured output only ever emits fields the schema declares. `false` is
therefore silent; a permissive `additionalProperties` becomes a sentence,
because that one really is being lost.

Field order is not cosmetic
---------------------------

`propertyOrdering` is emitted for every object, in the order the source schema
declares its properties. Gemini decides field order for itself otherwise, and
for a streamed plan that decides whether the UI can render anything early: our
plan grammar puts `intent` before `steps` so the intent object closes while the
steps are still being written, and the step trace can show what the model
thinks the question is before the plan is finished. Shuffle those two and the
first thing the user sees arrives at the end.

When the shape itself cannot be said
------------------------------------

Stripping a constraint keeps the answer's shape. Two things change it, and both
are shapes Gemini has no way to name:

* an object with no declared properties — `{"type": "object"}`, the free-form
  `args` bag in our plan grammar. Gemini rejects `OBJECT` with empty
  `properties`, and there is no "any object here" to fall back on.
* a node with no type at all and nothing to infer one from — `{}`, "anything".

Dropping such a field is worse than useless: Gemini emits exactly the declared
fields, so a dropped `args` means every step comes back without its arguments,
and nothing in the answer says so. Turning it into a string changes what the
caller parses, which for a streamed plan cannot even be undone afterwards.

So the fallback there is coarser and it is deliberate: **the whole schema is
dropped**, `to_gemini_schema` returns `{}`, and the provider sends
`responseMimeType: application/json` with no `responseSchema` and the shape
written out in the prompt instead (`Translation.prompt_hint`). Unconstrained
JSON of the right shape beats enforced JSON of the wrong one. `translate` says
so in `usable` and `reason` rather than raising, so the caller can log it once
and carry on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final

# JSON Schema's type names, in Gemini's spelling. Gemini wants the OpenAPI enum
# name in upper case; "string" is not the name of the enum member and can be
# rejected on the way in.
TYPES: Final[dict[str, str]] = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}

# The only `format` values Gemini knows, per type. Anything else — "email",
# "uri", "uuid", "date" — is a 400 waiting to happen, so it becomes a sentence.
SAFE_FORMATS: Final[dict[str, frozenset[str]]] = {
    "STRING": frozenset({"date-time", "enum"}),
    "INTEGER": frozenset({"int32", "int64"}),
    "NUMBER": frozenset({"float", "double"}),
}

# What may appear in the output. Widening this is a one-line change, but check
# the current Gemini docs first: the list has grown over time and every addition
# is a keyword that used to be a 400.
EMITTED: Final[frozenset[str]] = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "propertyOrdering",
        "minItems",
        "maxItems",
    }
)

# Keywords that describe the schema rather than the answer. Dropping them says
# nothing to the model, so they go quietly.
_METADATA: Final[frozenset[str]] = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$comment",
        "$defs",
        "definitions",
        "components",
        "title",
        "deprecated",
        "readOnly",
        "writeOnly",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)

# Words for the `format` values Gemini cannot enforce. Anything not here gets
# the generic sentence, which is still better than silence.
_FORMAT_WORDS: Final[dict[str, str]] = {
    "date": "A date, written as YYYY-MM-DD.",
    "time": "A time of day, written as HH:MM:SS.",
    "date-time": "A date and time in ISO 8601, with a timezone.",
    "duration": "A length of time in ISO 8601, like PT30M.",
    "email": "An email address.",
    "hostname": "A hostname.",
    "ipv4": "An IPv4 address.",
    "ipv6": "An IPv6 address.",
    "uri": "A URL.",
    "uri-reference": "A URL, which may be relative.",
    "url": "A URL.",
    "uuid": "A UUID.",
    "regex": "A regular expression.",
    "byte": "Base64 text.",
}

# How deep a schema may nest before we stop. Well past anything hand-written;
# it is here so a schema that refers to itself through a chain of refs cannot
# spin instead of failing.
MAX_DEPTH: Final[int] = 12

# How many times a union may contain a union before we stop trying to pick one
# branch. Two is already unusual.
_UNION_ROUNDS: Final[int] = 4

# Keywords consumed by the walk itself rather than emitted or described.
_HANDLED: Final[frozenset[str]] = frozenset(
    {"$ref", "allOf", "anyOf", "oneOf", "const", "additionalProperties", "nullable"}
)


# ------------------------------------------------------------------ the result


@dataclass
class Translation:
    """A translated schema plus what it cost to translate.

    `schema` is what goes on the wire. `notes` is one line per thing that was
    stripped, path first — ``"steps[].id: Must match the pattern ..."`` — for a
    single debug log line rather than a surprise in production. `usable` is
    False when the shape could not be expressed at all; `schema` is then `{}`
    and the provider should send no `responseSchema`.
    """

    schema: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    usable: bool = True
    reason: str = ""
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        """How many constraints did not survive. Cheap to log every call."""
        return len(self.notes)

    def prompt_hint(self) -> str:
        """The shape in words, for when `usable` is False.

        Gemini in JSON mode with no schema still returns JSON; it just returns
        whatever JSON it likes. This is what makes it the right JSON: the
        original schema, verbatim, in the system instruction. Models read JSON
        Schema well — it is the *enforcement* Gemini cannot do, not the
        reading.
        """
        if not self.source:
            return ""
        shape = json.dumps(self.source, indent=2, ensure_ascii=False, sort_keys=False)
        lines = [
            "Answer with a single JSON object and nothing else.",
            "It must match this shape exactly, with no extra fields:",
            shape,
        ]
        if self.reason:
            lines.append(f"Note: {self.reason}")
        return "\n".join(lines)


class _Inexpressible(Exception):
    """A shape Gemini has no way to name. Caught in `translate`, never escapes."""

    def __init__(self, where: str, reason: str) -> None:
        self.where = where
        self.reason = reason
        super().__init__(f"{where or 'the schema'}: {reason}")


@dataclass
class _Ctx:
    """Everything the walk carries: the definitions, the notes, the ref stack."""

    defs: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    active_refs: tuple[str, ...] = ()

    def note(self, path: str, sentence: str) -> None:
        self.notes.append(f"{path or '(root)'}: {sentence}")


# ------------------------------------------------------------------- the entry


def to_gemini_schema(json_schema: dict) -> dict:
    """A JSON Schema as the `responseSchema` Gemini accepts.

    Returns `{}` when the shape cannot be expressed — see the module docstring.
    An empty result means "send no schema", not "send an empty schema": Gemini
    rejects `{}` on `responseSchema`, and `app.llm.providers.gemini_provider`
    checks for it.
    """
    return translate(json_schema).schema


def translate(json_schema: dict | None) -> Translation:
    """`to_gemini_schema` with the reasons kept.

    Never raises. A schema this cannot express comes back as
    ``usable=False``, because the provider's job at that point is to log it and
    ask the question anyway, not to fail the user's run over a schema.
    """
    if not isinstance(json_schema, dict) or not json_schema:
        return Translation(schema={}, usable=False, reason="No schema was given.", source={})

    source = _unwrap(json_schema)
    ctx = _Ctx(defs=_collect_defs(source))

    try:
        schema = _node(source, ctx, path="", depth=0)
    except _Inexpressible as exc:
        reason = (
            f"Gemini cannot enforce this shape ({exc.where or 'the whole schema'}: {exc.reason}), "
            "so the answer is only checked for being JSON."
        )
        return Translation(
            schema={},
            notes=[*ctx.notes, f"{exc.where or '(root)'}: {exc.reason}"],
            usable=False,
            reason=reason,
            source=source,
        )

    return Translation(schema=schema, notes=ctx.notes, usable=True, source=source)


def _unwrap(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept OpenAI's wrapper as well as a bare schema.

    Call sites pass one `schema` argument to whichever provider answers, and
    ``{"name": ..., "schema": {...}, "strict": true}`` is how OpenAI wants it.
    Unwrapping here means a caller never has to know who is on the other end,
    which is the whole point of the package.
    """
    inner = schema.get("schema")
    if isinstance(inner, dict) and "type" not in schema and "properties" not in schema:
        return inner
    return schema


def _collect_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Every local definition, keyed by the pointer that reaches it."""
    defs: dict[str, Any] = {}
    for holder in ("$defs", "definitions"):
        block = schema.get(holder)
        if isinstance(block, dict):
            for name, value in block.items():
                if isinstance(value, dict):
                    defs[f"#/{holder}/{name}"] = value
    components = schema.get("components")
    if isinstance(components, dict):
        schemas = components.get("schemas")
        if isinstance(schemas, dict):
            for name, value in schemas.items():
                if isinstance(value, dict):
                    defs[f"#/components/schemas/{name}"] = value
    return defs


# -------------------------------------------------------------------- the walk


def _node(schema: Any, ctx: _Ctx, *, path: str, depth: int) -> dict[str, Any]:
    """One schema node, translated. Recurses through properties and items."""
    if depth > MAX_DEPTH:
        raise _Inexpressible(path, f"it nests more than {MAX_DEPTH} levels deep")
    if schema is True:  # JSON Schema's "anything goes"
        raise _Inexpressible(path, "it allows anything, which has no shape to describe")
    if not isinstance(schema, dict):
        raise _Inexpressible(path, "it is not a schema")

    sentences: list[str] = []

    schema, ctx = _resolve(schema, ctx, path=path)
    schema = _flatten_all_of(schema, ctx, path=path)

    # A branch of a union may be a union itself. Collapsing repeats until one
    # shape is left, and gives up rather than spinning: a schema that refers to
    # itself through its alternatives has no bottom to reach.
    nullable_from_union = False
    for _ in range(_UNION_ROUNDS):
        schema, nullable_here = _collapse_union(schema, ctx, path=path, sentences=sentences)
        nullable_from_union = nullable_from_union or nullable_here
        if "anyOf" not in schema and "oneOf" not in schema:
            break
    else:
        raise _Inexpressible(path, "its alternatives nest too deeply to reduce to one shape")

    kind, nullable_from_type = _type_of(schema, ctx, path=path, sentences=sentences)
    nullable = nullable_from_type or nullable_from_union or bool(schema.get("nullable"))

    out: dict[str, Any] = {"type": kind}

    if kind == "OBJECT":
        _fill_object(out, schema, ctx, path=path, depth=depth, sentences=sentences)
    elif kind == "ARRAY":
        _fill_array(out, schema, ctx, path=path, depth=depth, sentences=sentences)

    _fill_enum(out, schema, ctx, path=path, sentences=sentences)
    _fill_format(out, schema, ctx, path=path, sentences=sentences)

    if nullable:
        out["nullable"] = True

    # Everything left over: either said in words or quietly dropped.
    for key, value in schema.items():
        if key in EMITTED or key in _METADATA or key in _HANDLED:
            continue
        sentence = _in_words(key, value)
        if sentence:
            ctx.note(path, sentence)
            sentences.append(sentence)

    description = _description(schema, sentences)
    if description:
        out["description"] = description
    return out


def _resolve(schema: dict[str, Any], ctx: _Ctx, *, path: str) -> tuple[dict[str, Any], _Ctx]:
    """Inline a local `$ref`. A ref that leads back to itself cannot be inlined."""
    pointer = schema.get("$ref")
    if not isinstance(pointer, str):
        return schema, ctx

    if pointer in ctx.active_refs:
        raise _Inexpressible(path, f"{pointer} refers to itself, and Gemini has no way to say that")

    target = ctx.defs.get(pointer)
    if not isinstance(target, dict):
        raise _Inexpressible(path, f"{pointer} points at a definition that is not in this schema")

    merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
    return merged, _Ctx(defs=ctx.defs, notes=ctx.notes, active_refs=(*ctx.active_refs, pointer))


def _flatten_all_of(schema: dict[str, Any], ctx: _Ctx, *, path: str) -> dict[str, Any]:
    """Merge `allOf` into the node. The merge is exact, so nothing is lost."""
    branches = schema.get("allOf")
    if not isinstance(branches, list) or not branches:
        return schema

    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        resolved, _ = _resolve(branch, ctx, path=path)
        for key, value in resolved.items():
            if key == "properties" and isinstance(value, dict):
                merged["properties"] = {**merged.get("properties", {}), **value}
            elif key == "required" and isinstance(value, list):
                have = list(merged.get("required", []))
                merged["required"] = have + [n for n in value if n not in have]
            elif key not in merged:
                merged[key] = value
    return merged


def _collapse_union(
    schema: dict[str, Any], ctx: _Ctx, *, path: str, sentences: list[str]
) -> tuple[dict[str, Any], bool]:
    """`anyOf` / `oneOf` down to one branch, and whether null was one of them.

    The common case by far is a nullable field, which Gemini *can* say. A real
    union cannot be said, so the first branch wins and the others become words:
    the caller is about to parse this, and a wrong-but-consistent shape is
    something it can handle. Two shapes is not.
    """
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list) or not branches:
            continue

        real = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
        nullable = len(real) < len([b for b in branches if isinstance(b, dict)])
        if not real:
            raise _Inexpressible(path, f"every branch of {key} is null")

        chosen, _ = _resolve(real[0], ctx, path=path)
        rest = {**{k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}, **chosen}

        if len(real) > 1:
            others = ", ".join(_shape_word(b) for b in real[1:])
            sentence = f"May also be {others}."
            ctx.note(path, f"{key} has more than one shape. {sentence}")
            sentences.append(sentence)

        return rest, nullable
    return schema, False


def _type_of(
    schema: dict[str, Any], ctx: _Ctx, *, path: str, sentences: list[str]
) -> tuple[str, bool]:
    """The Gemini type name, and whether the source allowed null too."""
    declared = schema.get("type")
    nullable = False

    if isinstance(declared, list):
        names = [str(t) for t in declared]
        nullable = "null" in names
        real = [t for t in names if t != "null"]
        if not real:
            raise _Inexpressible(path, "it may only be null")
        if len(real) > 1:
            sentence = f"May be any of: {', '.join(real)}."
            ctx.note(path, f"more than one type. {sentence}")
            sentences.append(sentence)
        declared = real[0]

    if isinstance(declared, str):
        kind = TYPES.get(declared.strip().lower())
        if kind is None:
            raise _Inexpressible(path, f"{declared!r} is not a type Gemini knows")
        return kind, nullable

    # No type. Infer one, because a schema written by hand often leaves it out
    # when the rest of the node makes it obvious.
    if isinstance(schema.get("properties"), dict):
        return "OBJECT", nullable
    if "items" in schema:
        return "ARRAY", nullable
    values = schema.get("enum")
    if isinstance(values, list) and values and all(isinstance(v, str) for v in values):
        return "STRING", nullable
    if isinstance(schema.get("const"), str):
        return "STRING", nullable
    raise _Inexpressible(path, "it has no type and nothing to work one out from")


def _fill_object(
    out: dict[str, Any],
    schema: dict[str, Any],
    ctx: _Ctx,
    *,
    path: str,
    depth: int,
    sentences: list[str],
) -> None:
    """Properties, required, and the field order that has to hold."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise _Inexpressible(
            path,
            "it is an object with no fixed fields, and Gemini has no way to say 'any object here'",
        )

    order = _ordering(schema, properties)
    translated: dict[str, Any] = {}
    for name in order:
        child = f"{path}.{name}" if path else name
        translated[name] = _node(properties[name], ctx, path=child, depth=depth + 1)

    out["properties"] = translated
    out["propertyOrdering"] = order

    required = schema.get("required")
    if isinstance(required, list):
        kept = [str(n) for n in required if str(n) in translated]
        if kept:
            out["required"] = kept

    extra = schema.get("additionalProperties")
    # `false` is what Gemini does anyway — it only ever emits declared fields —
    # so only a permissive setting is worth a sentence.
    if extra not in (None, False):
        sentence = "Other fields may be added if the instructions ask for them."
        ctx.note(path, f"additionalProperties. {sentence}")
        sentences.append(sentence)


def _fill_array(
    out: dict[str, Any],
    schema: dict[str, Any],
    ctx: _Ctx,
    *,
    path: str,
    depth: int,
    sentences: list[str],
) -> None:
    """`items`, plus the two length limits Gemini can enforce."""
    items = schema.get("items")
    if isinstance(items, list):
        # A tuple-typed array: item 0 is a string, item 1 is a number. Gemini has
        # one `items` for the whole array, so the first entry has to stand in.
        sentence = f"A list of {len(items)} values, in the order described."
        ctx.note(path, f"per-position items. {sentence}")
        sentences.append(sentence)
        items = items[0] if items else None
    if items is None:
        raise _Inexpressible(path, "it is a list with no item type")

    out["items"] = _node(items, ctx, path=f"{path}[]", depth=depth + 1)
    for key in ("minItems", "maxItems"):
        value = schema.get(key)
        if isinstance(value, int):
            out[key] = value


def _fill_enum(
    out: dict[str, Any],
    schema: dict[str, Any],
    ctx: _Ctx,
    *,
    path: str,
    sentences: list[str],
) -> None:
    """`enum` and `const`, which Gemini only understands on strings."""
    values = schema.get("enum")
    const = schema.get("const")
    if values is None and const is not None:
        values = [const]
    if not isinstance(values, list) or not values:
        return

    real = [v for v in values if v is not None]
    if len(real) < len(values):
        out["nullable"] = True

    if out["type"] == "STRING" and real and all(isinstance(v, str) for v in real):
        out["enum"] = list(real)
        out["format"] = "enum"  # the spelling Gemini's own docs use
        return

    # Numbers, booleans, mixed types: the list has to be said rather than set.
    listed = ", ".join(json.dumps(v, ensure_ascii=False) for v in real)
    sentence = f"Always {listed}." if len(real) == 1 else f"One of: {listed}."
    ctx.note(path, sentence)
    sentences.append(sentence)


def _fill_format(
    out: dict[str, Any],
    schema: dict[str, Any],
    ctx: _Ctx,
    *,
    path: str,
    sentences: list[str],
) -> None:
    """Keep the four formats Gemini knows; say the rest."""
    raw = schema.get("format")
    if not isinstance(raw, str) or not raw:
        return
    if raw == "enum":
        # `format: enum` without a list of values is a 400. `_fill_enum` has
        # already set both when there was a list, so there is nothing to say.
        return
    if raw in SAFE_FORMATS.get(out["type"], frozenset()):
        out.setdefault("format", raw)
        return

    sentence = _FORMAT_WORDS.get(raw, f"In {raw} format.")
    ctx.note(path, f"format {raw!r}. {sentence}")
    sentences.append(sentence)


def _ordering(schema: dict[str, Any], properties: dict[str, Any]) -> list[str]:
    """The order fields must come back in.

    An explicit `propertyOrdering` wins, filtered to fields that exist; anything
    it leaves out follows in the order the schema declares it. With neither, the
    declaration order is the answer, which is what a person writing the schema
    meant by writing `intent` above `steps`.
    """
    declared = list(properties.keys())
    given = schema.get("propertyOrdering")
    if not isinstance(given, list):
        return declared
    order = [str(n) for n in given if str(n) in properties]
    return order + [n for n in declared if n not in order]


def _description(schema: dict[str, Any], sentences: list[str]) -> str:
    """The field's own words, then everything Gemini could not enforce."""
    text = schema.get("description") or schema.get("title") or ""
    parts = [str(text).strip(), *sentences]
    return " ".join(part for part in parts if part).strip()


# --------------------------------------------------------- constraints, in words


def _in_words(key: str, value: Any) -> str | None:
    """One stripped keyword as a sentence a model can follow.

    Plain sentences on purpose. The model reads this the same way it reads the
    rest of the prompt, and "At most 40 characters." lands better than
    "maxLength: 40" does.
    """
    if key == "pattern" and isinstance(value, str):
        return f'Must match the pattern "{value}".'
    if key == "minLength" and isinstance(value, int):
        return f"At least {value} character{'s' if value != 1 else ''}."
    if key == "maxLength" and isinstance(value, int):
        return f"At most {value} character{'s' if value != 1 else ''}."
    if key == "minimum" and isinstance(value, int | float):
        return f"At least {_num(value)}."
    if key == "maximum" and isinstance(value, int | float):
        return f"At most {_num(value)}."
    if key == "exclusiveMinimum" and isinstance(value, int | float):
        return f"Greater than {_num(value)}."
    if key == "exclusiveMaximum" and isinstance(value, int | float):
        return f"Less than {_num(value)}."
    if key == "multipleOf" and isinstance(value, int | float):
        return f"A multiple of {_num(value)}."
    if key == "uniqueItems" and value:
        return "Every item must be different."
    if key == "minProperties" and isinstance(value, int):
        return f"At least {value} field{'s' if value != 1 else ''}."
    if key == "maxProperties" and isinstance(value, int):
        return f"At most {value} field{'s' if value != 1 else ''}."
    if key == "default":
        return f"Defaults to {json.dumps(value, ensure_ascii=False)}."
    if key in ("example", "examples"):
        sample = value[0] if key == "examples" and isinstance(value, list) and value else value
        if isinstance(sample, str | int | float | bool):
            return f"For example: {json.dumps(sample, ensure_ascii=False)}."
        return None
    if key == "not":
        return f"Must not be {_shape_word(value)}."
    if key == "propertyNames":
        pattern = value.get("pattern") if isinstance(value, dict) else None
        if isinstance(pattern, str):
            return f'Every field name must match "{pattern}".'
        return "Field names are restricted; follow the instructions above."
    if key == "patternProperties" and isinstance(value, dict):
        names = ", ".join(f'"{k}"' for k in value)
        return f"Fields whose names match {names} follow their own rules."
    if key in ("dependentRequired", "dependencies") and isinstance(value, dict):
        clauses = [
            f"if {name} is given, {', '.join(str(n) for n in needs)} must be given too"
            for name, needs in value.items()
            if isinstance(needs, list) and needs
        ]
        return f"{'; '.join(clauses).capitalize()}." if clauses else None
    if key in ("if", "then", "else", "dependentSchemas"):
        return "Some fields only apply in certain cases; follow the instructions above."
    if key == "contains":
        return f"At least one item must be {_shape_word(value)}."
    # `minItems` / `maxItems` never reach here: they are in `EMITTED`, so an
    # array keeps them and any other type quietly loses a keyword that meant
    # nothing on it anyway.
    return None


def _num(value: float) -> str:
    """A number the way a person would write it: 3, not 3.0."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _shape_word(schema: Any) -> str:
    """A branch of a union in a few words, for a sentence rather than a schema."""
    if not isinstance(schema, dict):
        return "something else"
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((t for t in kind if t != "null"), None)
    if kind == "array":
        inner = schema.get("items")
        return f"a list of {_shape_word(inner)}" if isinstance(inner, dict) else "a list"
    if kind == "object":
        names = schema.get("properties")
        if isinstance(names, dict) and names:
            return f"an object with {', '.join(list(names)[:4])}"
        return "an object"
    if isinstance(kind, str):
        return {"string": "text", "integer": "a whole number", "number": "a number"}.get(
            kind, f"a {kind}"
        )
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    return "something else"


__all__ = [
    "EMITTED",
    "MAX_DEPTH",
    "SAFE_FORMATS",
    "TYPES",
    "Translation",
    "to_gemini_schema",
    "translate",
]
