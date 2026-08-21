"""Our JSON Schema, rewritten into the subset Anthropic will accept.

`to_anthropic_schema` has one job and two hard rules. The job is to hand
``output_config.format`` something it will not 400 on. The rules are Anthropic's:

* every object lists **every** one of its properties in ``required``
* every object carries ``"additionalProperties": false``

Both apply at every level, not just the top one — a nested object that forgets
either is the same 400 as the root forgetting it, and it is much easier to miss.
So most of this file walks the whole converted tree and checks every object it
finds, rather than checking the one at the top and hoping.

The other half is what happens to a schema Anthropic cannot express. There is no
good way to raise here: the caller is a provider, the request has not been sent
yet, and a schema that is slightly too fancy would become a dead call with no
fallback — `INVALID` does not fall back, so it costs the whole answer. So the
rule is strip and say so: the constraint is removed and rewritten as a sentence
in that node's ``description``, where the model can still read it. These tests
check that a whole catalogue of unrepresentable things degrades that way instead
of raising or vanishing silently.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from app.llm.anthropic_schema import (
    SUPPORTED_FORMATS,
    schema_fingerprint,
    to_anthropic_schema,
)

# ---------------------------------------------------------------------------
# Walking a converted schema
# ---------------------------------------------------------------------------


def objects(node: Any) -> list[dict[str, Any]]:
    """Every object node anywhere in a converted schema, top level included.

    An optional object comes back as ``"type": ["object", "null"]``, so the type
    is matched as a set rather than as a string.
    """
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        kind = node.get("type")
        names = {kind} if isinstance(kind, str) else set(kind or ())
        if "object" in names or "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(objects(item))
    return found


def assert_anthropic_safe(converted: dict[str, Any]) -> None:
    """The two rules, at every depth."""
    every = objects(converted)
    assert every, "expected at least one object node to check"
    for node in every:
        assert node.get("additionalProperties") is False, f"missing on {node.get('title', node)}"
        properties = node.get("properties")
        assert isinstance(properties, dict), "an object must carry a properties map"
        assert sorted(node.get("required", [])) == sorted(properties), (
            "required must name every property, not only the ones that started out required"
        )


def description_of(node: dict[str, Any]) -> str:
    return str(node.get("description", ""))


# ---------------------------------------------------------------------------
# The two rules Anthropic will not bend on
# ---------------------------------------------------------------------------


def test_a_flat_object_gets_both_rules():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
    )

    assert converted["additionalProperties"] is False
    assert sorted(converted["required"]) == ["a", "b"]
    assert_anthropic_safe(converted)


def test_additional_properties_is_false_at_every_nested_level():
    """Three objects deep. A rule applied only at the root is the easy bug."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "middle": {
                            "type": "object",
                            "properties": {"inner": {"type": "string"}},
                            "required": ["inner"],
                        }
                    },
                    "required": ["middle"],
                }
            },
            "required": ["outer"],
        }
    )

    assert len(objects(converted)) == 3
    assert_anthropic_safe(converted)
    assert converted["properties"]["outer"]["properties"]["middle"]["additionalProperties"] is False


def test_required_lists_every_property_even_the_ones_that_were_optional():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"kept": {"type": "string"}, "was_optional": {"type": "integer"}},
            "required": ["kept"],
        }
    )

    assert sorted(converted["required"]) == ["kept", "was_optional"]
    assert_anthropic_safe(converted)


def test_an_optional_property_may_answer_null():
    """Forcing every field into `required` turns "you may leave this out" into
    "you must fill this in", and a model told to fill something in invents it.
    In a legal-domain app an invented value is worse than a missing one."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"kept": {"type": "string"}, "maybe": {"type": "integer"}},
            "required": ["kept"],
        }
    )

    assert converted["properties"]["maybe"]["type"] == ["integer", "null"]
    assert converted["properties"]["kept"]["type"] == "string"
    assert "null" in description_of(converted).lower()


def test_the_blunt_version_says_so_instead():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"kept": {"type": "string"}, "maybe": {"type": "integer"}},
            "required": ["kept"],
        },
        nullable_optionals=False,
    )

    assert converted["properties"]["maybe"]["type"] == "integer"
    assert "optional" in description_of(converted).lower()
    assert_anthropic_safe(converted)


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def test_an_array_of_objects_is_handled():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string"},
                            "args": {"type": "object", "properties": {}},
                        },
                        "required": ["op", "args"],
                    },
                }
            },
            "required": ["steps"],
        }
    )

    items = converted["properties"]["steps"]["items"]
    assert items["type"] == "object"
    assert items["additionalProperties"] is False
    assert sorted(items["required"]) == ["args", "op"]
    assert_anthropic_safe(converted)


def test_arrays_nested_inside_arrays_are_handled_all_the_way_down():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cells": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                    "required": ["value"],
                                },
                            }
                        },
                        "required": ["cells"],
                    },
                }
            },
            "required": ["rows"],
        }
    )

    deepest = converted["properties"]["rows"]["items"]["properties"]["cells"]["items"]
    assert deepest["properties"]["value"]["type"] == "string"
    assert len(objects(converted)) == 3
    assert_anthropic_safe(converted)


def test_a_tuple_typed_array_becomes_one_repeated_shape_and_says_so():
    """prefixItems means "position 0 is a string, position 1 is a number".
    Anthropic has no way to say that, so it becomes one shape plus a sentence."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "pair": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}
            },
            "required": ["pair"],
        }
    )

    pair = converted["properties"]["pair"]
    assert "prefixItems" not in pair
    assert pair["items"]["type"] == "string"
    assert "same shape" in description_of(pair)


# ---------------------------------------------------------------------------
# Things Anthropic cannot express: describe, never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constraint", "expected_phrase"),
    [
        ({"minimum": 1}, "1 or more"),
        ({"maximum": 5}, "5 or less"),
        ({"exclusiveMinimum": 0}, "more than 0"),
        ({"exclusiveMaximum": 10}, "less than 10"),
        ({"multipleOf": 5}, "multiple of 5"),
    ],
)
def test_a_number_range_becomes_a_sentence(constraint, expected_phrase):
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"n": {"type": "integer", **constraint}},
            "required": ["n"],
        }
    )

    field = converted["properties"]["n"]
    assert field["type"] == "integer"  # the type survives
    for key in constraint:
        assert key not in field  # the unsendable keyword does not
    assert expected_phrase in description_of(field)


@pytest.mark.parametrize(
    ("constraint", "expected_phrase"),
    [
        ({"minLength": 2}, "at least 2 characters"),
        ({"maxLength": 40}, "no longer than 40 characters"),
        ({"pattern": "^[A-Z]{6}$"}, "^[A-Z]{6}$"),
    ],
)
def test_a_string_rule_becomes_a_sentence(constraint, expected_phrase):
    converted = to_anthropic_schema(
        {"type": "object", "properties": {"s": {"type": "string", **constraint}}, "required": ["s"]}
    )

    field = converted["properties"]["s"]
    assert field["type"] == "string"
    for key in constraint:
        assert key not in field
    assert expected_phrase in description_of(field)


@pytest.mark.parametrize(
    ("constraint", "expected_phrase"),
    [
        ({"minItems": 1}, "at least 1 item"),
        ({"maxItems": 3}, "no more than 3 items"),
        ({"uniqueItems": True}, "must be different"),
    ],
)
def test_a_list_size_rule_becomes_a_sentence(constraint, expected_phrase):
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"l": {"type": "array", "items": {"type": "string"}, **constraint}},
            "required": ["l"],
        }
    )

    field = converted["properties"]["l"]
    assert field["type"] == "array"
    assert field["items"]["type"] == "string"
    for key in constraint:
        assert key not in field
    assert expected_phrase in description_of(field)


def test_a_supported_format_survives_and_an_unsupported_one_is_described():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "when": {"type": "string", "format": "date-time"},
                "card": {"type": "string", "format": "credit-card"},
            },
            "required": ["when", "card"],
        }
    )

    assert "date-time" in SUPPORTED_FORMATS
    assert converted["properties"]["when"]["format"] == "date-time"

    card = converted["properties"]["card"]
    assert "format" not in card
    assert "credit-card" in description_of(card)


@pytest.mark.parametrize(
    "construct",
    [
        {"not": {"const": "no"}},
        {"if": {"const": "a"}, "then": {"const": "b"}},
        {"dependentRequired": {"a": ["b"]}},
        {"propertyNames": {"pattern": "^a"}},
        {"minProperties": 1, "maxProperties": 4},
    ],
)
def test_an_unrepresentable_construct_degrades_into_words_and_never_raises(construct):
    converted = to_anthropic_schema(
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], **construct}
    )

    for key in construct:
        assert key not in converted
    assert description_of(converted), "the constraint should survive as a sentence"
    assert_anthropic_safe(converted)


def test_contains_on_a_list_becomes_a_sentence():
    """`contains` is an array keyword — "at least one item matches this". There
    is no way to send it, so it becomes a sentence on the list itself."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "l": {
                    "type": "array",
                    "items": {"type": "string"},
                    "contains": {"const": "urgent"},
                }
            },
            "required": ["l"],
        }
    )

    field = converted["properties"]["l"]
    assert "contains" not in field
    assert "at least one item" in description_of(field).lower()


def test_a_free_form_map_says_what_to_do_instead_of_becoming_a_four_hundred():
    """`additionalProperties: {...}` cannot be sent, and an empty object with no
    explanation would just look broken."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {"bag": {"type": "object", "additionalProperties": {"type": "string"}}},
            "required": ["bag"],
        }
    )

    bag = converted["properties"]["bag"]
    assert bag["additionalProperties"] is False
    assert bag["properties"] == {}
    assert "free-form" in description_of(bag)


def test_a_schema_that_refers_back_to_itself_is_cut_and_the_cut_is_named():
    """A loop cannot be expressed at all. Dropping the property that closes it
    returns a usable top level; raising would have cost the whole call."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "child": {"$ref": "#/$defs/Node"}},
                    "required": ["name", "child"],
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "required": ["root"],
        }
    )

    root = converted["properties"]["root"]
    assert "name" in root["properties"]
    assert "child" not in root["properties"]
    assert "'child'" in description_of(root)
    assert_anthropic_safe(converted)


@pytest.mark.parametrize("junk", [None, [], "a string", 3, True, {"type": "array"}, {}])
def test_unusable_input_comes_back_as_an_empty_object_rather_than_an_exception(junk):
    converted = to_anthropic_schema(junk)

    assert converted["type"] == "object"
    assert converted["properties"] == {}
    assert converted["additionalProperties"] is False
    assert description_of(converted), "an empty schema must explain itself"


# ---------------------------------------------------------------------------
# References and combinators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("holder", ["$defs", "definitions"])
def test_a_reference_is_followed_and_inlined(holder):
    converted = to_anthropic_schema(
        {
            "type": "object",
            holder: {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
            "properties": {"home": {"$ref": f"#/{holder}/Address"}},
            "required": ["home"],
        }
    )

    assert converted["properties"]["home"]["properties"]["city"]["type"] == "string"
    assert "$ref" not in json.dumps(converted)
    assert_anthropic_safe(converted)


def test_a_reference_that_points_nowhere_is_described_not_raised():
    converted = to_anthropic_schema(
        {"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}, "required": ["x"]}
    )

    assert "missing" in description_of(converted["properties"]["x"]).lower()
    assert_anthropic_safe(converted)


def test_all_of_objects_are_merged_into_one():
    """Merging matters. `additionalProperties: false` on each branch separately
    means no value can satisfy all of them at once, which is never what the
    person who wrote the schema meant."""
    converted = to_anthropic_schema(
        {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
            ]
        }
    )

    assert "allOf" not in converted
    assert sorted(converted["properties"]) == ["a", "b"]
    assert sorted(converted["required"]) == ["a", "b"]
    assert converted["additionalProperties"] is False


def test_one_of_becomes_any_of_with_a_note():
    converted = to_anthropic_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})

    assert "oneOf" not in converted
    assert [b["type"] for b in converted["anyOf"]] == ["string", "integer"]
    assert "exactly one" in description_of(converted).lower()


def test_enum_and_const_survive():
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "colour": {"type": "string", "enum": ["red", "green"]},
                "kind": {"const": "plan"},
            },
            "required": ["colour", "kind"],
        }
    )

    assert converted["properties"]["colour"]["enum"] == ["red", "green"]
    assert converted["properties"]["kind"]["const"] == "plan"


# ---------------------------------------------------------------------------
# Being a pure function
# ---------------------------------------------------------------------------


def test_the_input_is_never_modified():
    original = {
        "type": "object",
        "properties": {"a": {"type": "integer", "minimum": 1}, "b": {"type": "string"}},
        "required": ["a"],
    }
    untouched = copy.deepcopy(original)

    to_anthropic_schema(original)

    assert original == untouched


def test_the_same_schema_always_produces_the_same_bytes():
    """Anthropic keeps a compiled copy of a schema for a day and prompt caching
    is a prefix match, so a description that reshuffles between calls quietly
    throws both away."""
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5, "multipleOf": 1},
            "s": {"type": "string", "pattern": "^A", "minLength": 2, "maxLength": 9},
        },
        "required": ["n"],
    }

    first = schema_fingerprint(to_anthropic_schema(schema))
    second = schema_fingerprint(to_anthropic_schema(copy.deepcopy(schema)))
    assert first == second


def test_the_result_is_always_json_serialisable():
    """It is about to be posted as JSON. Anything clever in there is a 500 in
    our own process rather than a 400 from theirs."""
    converted = to_anthropic_schema(
        {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"op": {"type": "string", "enum": ["read", "write"]}},
                        "required": ["op"],
                    },
                }
            },
            "required": ["steps"],
        }
    )

    assert json.loads(json.dumps(converted)) == converted
