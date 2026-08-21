"""JSON Schema in, the small thing Gemini accepts out.

Pure functions, no I/O, no SDK, no key. `to_gemini_schema` is what goes on the
wire as `responseSchema`; `translate` is the same work with the paperwork kept.

The four things that have to hold:

* **A nested plan schema survives the trip.** Our plan grammar is objects inside
  arrays inside objects, and every level has to come out the other side with its
  fields, its types and its `required` list intact.
* **Enums survive as enums.** They are what stops the planner inventing an op
  name or a `freshness` value nothing handles.
* **`additionalProperties` goes.** Gemini has no such keyword; sending it is a
  400, which is `INVALID`, which is deliberately not in `LLM_FALLBACK_ON` — a
  schema this file gets wrong does not fail over, it fails the run.
* **`propertyOrdering` keeps `intent` before `steps`.** Not cosmetic: the trace
  panel shows the intent while the steps are still streaming. Shuffle those two
  and the first thing the user sees arrives last.

And the fifth, which is the interesting one: a construct Gemini cannot express
**degrades** — into a sentence in the field's description where the shape is
still right, or, where the shape itself cannot be said, into no schema at all
plus the original written into the prompt. It never raises. A schema is not
worth failing a user's run over.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from app.llm.schema_translate import Translation, to_gemini_schema, translate

# --------------------------------------------------------------------------
# The plan grammar, near enough to the real one to be worth testing
# --------------------------------------------------------------------------


def plan_schema() -> dict[str, Any]:
    """`docs/contracts.md`'s plan object, as a strict JSON Schema.

    Built fresh each time. A shared dict that a translation quietly mutated
    would make the next test pass for the wrong reason.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["plan"]},
            "intent": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z_]+$", "maxLength": 40},
                    "services": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["gmail", "gcal", "gdrive"]},
                        "minItems": 1,
                    },
                    "has_write": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "services", "has_write", "confidence"],
            },
            "answer_style": {"type": "string", "description": "card, template:<name> or prose."},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                        "op": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "expect": {"type": "string", "enum": ["one", "many"]},
                        "optional": {"type": "boolean", "default": False},
                        "freshness": {"type": "string", "enum": ["cached", "live"]},
                        "gate": {
                            "type": ["object", "null"],
                            "properties": {
                                "left": {"type": "string"},
                                "test": {
                                    "type": "string",
                                    "enum": ["exists", "empty", "count_gt", "within"],
                                },
                                "right": {"type": "string"},
                            },
                            "required": ["left", "test"],
                        },
                    },
                    "required": ["id", "op", "expect"],
                },
            },
        },
        "required": ["type", "intent", "steps"],
    }


def walk(node: Any) -> list[dict[str, Any]]:
    """Every object node in a translated schema, root first."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for key in ("properties", "items"):
            child = node.get(key)
            if isinstance(child, dict):
                if key == "items":
                    found.extend(walk(child))
                else:
                    for value in child.values():
                        found.extend(walk(value))
    return found


def keys_everywhere(schema: dict[str, Any]) -> set[str]:
    return {key for node in walk(schema) for key in node}


@pytest.fixture
def plan() -> dict[str, Any]:
    return plan_schema()


@pytest.fixture
def translated(plan) -> dict[str, Any]:
    return to_gemini_schema(plan)


# --------------------------------------------------------------------------
# A nested plan schema round-trips
# --------------------------------------------------------------------------


class TestNestedPlanRoundTrips:
    def test_the_shape_survives_three_levels_down(self, translated):
        assert translated["type"] == "OBJECT"

        intent = translated["properties"]["intent"]
        assert intent["type"] == "OBJECT"
        assert set(intent["properties"]) == {"name", "services", "has_write", "confidence"}

        step = translated["properties"]["steps"]["items"]
        assert step["type"] == "OBJECT"
        assert step["properties"]["id"]["type"] == "STRING"
        assert step["properties"]["depends_on"]["items"]["type"] == "STRING"
        assert step["properties"]["gate"]["properties"]["left"]["type"] == "STRING"

    def test_types_come_out_in_geminis_spelling(self, translated):
        types = {node["type"] for node in walk(translated) if "type" in node}
        assert types <= {"OBJECT", "ARRAY", "STRING", "NUMBER", "INTEGER", "BOOLEAN"}
        assert "object" not in types  # lower case is a 400 on the way in

    def test_required_lists_are_kept_at_every_level(self, translated):
        assert translated["required"] == ["type", "intent", "steps"]
        assert translated["properties"]["steps"]["items"]["required"] == ["id", "op", "expect"]
        assert translated["properties"]["intent"]["required"] == [
            "name",
            "services",
            "has_write",
            "confidence",
        ]

    def test_the_limits_gemini_can_enforce_are_kept(self, translated):
        steps = translated["properties"]["steps"]
        assert steps["minItems"] == 1
        assert steps["maxItems"] == 8
        assert translated["properties"]["intent"]["properties"]["services"]["minItems"] == 1

    def test_only_keywords_gemini_knows_come_out(self, translated):
        allowed = {
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
        assert keys_everywhere(translated) <= allowed

    def test_a_nullable_branch_becomes_nullable(self, translated):
        """`["object", "null"]` is a union Gemini has one word for."""
        gate = translated["properties"]["steps"]["items"]["properties"]["gate"]
        assert gate["type"] == "OBJECT"
        assert gate["nullable"] is True

    def test_the_source_schema_is_never_mutated(self, plan):
        before = copy.deepcopy(plan)
        to_gemini_schema(plan)
        assert plan == before

    def test_translating_twice_gives_the_same_thing(self, plan):
        assert to_gemini_schema(plan) == to_gemini_schema(plan_schema())

    def test_an_openai_style_wrapper_is_unwrapped(self):
        wrapped = {
            "name": "plan",
            "strict": True,
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        }
        assert to_gemini_schema(wrapped)["properties"] == {"a": {"type": "STRING"}}

    def test_a_local_ref_is_inlined(self):
        schema = {
            "$defs": {"Step": {"type": "object", "properties": {"id": {"type": "string"}}}},
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/Step"}}},
        }
        out = to_gemini_schema(schema)

        assert out["properties"]["steps"]["items"]["properties"] == {"id": {"type": "STRING"}}
        assert "$defs" not in keys_everywhere(out)
        assert "$ref" not in keys_everywhere(out)


# --------------------------------------------------------------------------
# Enums survive
# --------------------------------------------------------------------------


class TestEnumsSurvive:
    def test_a_string_enum_stays_an_enum(self, translated):
        expect = translated["properties"]["steps"]["items"]["properties"]["expect"]
        assert expect["type"] == "STRING"
        assert expect["enum"] == ["one", "many"]
        assert expect["format"] == "enum"  # the spelling Gemini's own docs use

    def test_every_enum_in_the_plan_survives(self, translated):
        enums = {tuple(node["enum"]) for node in walk(translated) if "enum" in node}
        assert ("plan",) in enums
        assert ("gmail", "gcal", "gdrive") in enums
        assert ("one", "many") in enums
        assert ("cached", "live") in enums
        assert ("exists", "empty", "count_gt", "within") in enums

    def test_an_enum_inside_an_array_survives(self, translated):
        services = translated["properties"]["intent"]["properties"]["services"]
        assert services["items"]["enum"] == ["gmail", "gcal", "gdrive"]

    def test_a_const_becomes_a_one_value_enum(self):
        out = to_gemini_schema({"type": "object", "properties": {"t": {"const": "plan"}}})
        assert out["properties"]["t"]["enum"] == ["plan"]

    def test_a_nullable_enum_keeps_its_values_and_says_it_may_be_null(self):
        out = to_gemini_schema(
            {"type": "object", "properties": {"f": {"type": "string", "enum": ["a", "b", None]}}}
        )
        field = out["properties"]["f"]
        assert field["enum"] == ["a", "b"]
        assert field["nullable"] is True

    def test_a_numeric_enum_becomes_a_sentence_because_gemini_only_enums_strings(self):
        """Not silently dropped, and not sent as an `enum` Gemini would reject."""
        result = translate(
            {"type": "object", "properties": {"n": {"type": "integer", "enum": [1, 2, 3]}}}
        )

        field = result.schema["properties"]["n"]
        assert field["type"] == "INTEGER"
        assert "enum" not in field
        assert field["description"] == "One of: 1, 2, 3."
        assert result.notes == ["n: One of: 1, 2, 3."]


# --------------------------------------------------------------------------
# additionalProperties is stripped
# --------------------------------------------------------------------------


class TestAdditionalPropertiesIsStripped:
    def test_it_is_gone_from_every_level(self, plan, translated):
        assert "additionalProperties" in plan  # it was there to begin with
        assert "additionalProperties" not in keys_everywhere(translated)

    def test_false_leaves_no_trace_at_all(self):
        """Gemini only ever emits declared fields, so `false` is already true.
        Saying it in the description would be noise in every prompt."""
        result = translate(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"a": {"type": "string"}},
            }
        )
        assert result.schema == {
            "type": "OBJECT",
            "properties": {"a": {"type": "STRING"}},
            "propertyOrdering": ["a"],
        }
        assert result.notes == []

    def test_a_permissive_setting_is_the_one_that_gets_said_in_words(self):
        """That one really is being lost, so it is worth a sentence."""
        result = translate(
            {
                "type": "object",
                "additionalProperties": True,
                "properties": {"a": {"type": "string"}},
            }
        )
        assert "additionalProperties" not in result.schema
        assert "Other fields may be added" in result.schema["description"]
        assert result.dropped == 1

    def test_the_other_schema_keywords_go_too(self):
        out = to_gemini_schema(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://example.test/plan",
                "title": "Plan",
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        )
        assert keys_everywhere(out) <= {"type", "properties", "propertyOrdering", "description"}


# --------------------------------------------------------------------------
# Field order, which the streaming UI depends on
# --------------------------------------------------------------------------


class TestPropertyOrdering:
    def test_intent_closes_before_steps_begin(self, translated):
        order = translated["propertyOrdering"]
        assert order == ["type", "intent", "answer_style", "steps"]
        assert order.index("intent") < order.index("steps")

    def test_every_object_gets_an_ordering(self, translated):
        objects = [node for node in walk(translated) if node.get("type") == "OBJECT"]
        assert objects  # the plan, the intent, a step, a gate
        for node in objects:
            assert list(node["properties"]) == node["propertyOrdering"]

    def test_it_is_the_order_the_schema_declares(self):
        out = to_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "answer_style": {"type": "string"},
                    "steps": {"type": "string"},
                },
            }
        )
        assert out["propertyOrdering"] == ["intent", "answer_style", "steps"]

    def test_an_explicit_ordering_wins_and_the_rest_follow(self):
        out = to_gemini_schema(
            {
                "type": "object",
                "propertyOrdering": ["steps"],
                "properties": {"intent": {"type": "string"}, "steps": {"type": "string"}},
            }
        )
        assert out["propertyOrdering"] == ["steps", "intent"]

    def test_a_name_that_is_not_a_field_is_dropped_from_the_ordering(self):
        out = to_gemini_schema(
            {
                "type": "object",
                "propertyOrdering": ["ghost", "steps"],
                "properties": {"intent": {"type": "string"}, "steps": {"type": "string"}},
            }
        )
        assert out["propertyOrdering"] == ["steps", "intent"]


# --------------------------------------------------------------------------
# What cannot be expressed degrades. It never raises.
# --------------------------------------------------------------------------


class TestDegradesRatherThanRaises:
    def test_a_pattern_becomes_a_sentence_the_model_can_follow(self, translated):
        name = translated["properties"]["intent"]["properties"]["name"]
        assert name["type"] == "STRING"
        assert name["description"] == 'Must match the pattern "^[a-z_]+$". At most 40 characters.'

    def test_number_bounds_become_sentences(self, translated):
        confidence = translated["properties"]["intent"]["properties"]["confidence"]
        assert confidence["type"] == "NUMBER"
        assert confidence["description"] == "At least 0. At most 1."

    def test_a_default_becomes_a_sentence(self, translated):
        optional = translated["properties"]["steps"]["items"]["properties"]["optional"]
        assert optional["description"] == "Defaults to false."

    def test_the_fields_own_words_come_first(self):
        out = to_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The intent name.",
                        "maxLength": 40,
                    }
                },
            }
        )
        assert out["properties"]["name"]["description"] == "The intent name. At most 40 characters."

    def test_every_degradation_is_logged_once_with_its_path(self, plan):
        result = translate(plan)
        assert result.usable is True
        assert result.notes == [
            'intent.name: Must match the pattern "^[a-z_]+$".',
            "intent.name: At most 40 characters.",
            "intent.confidence: At least 0.",
            "intent.confidence: At most 1.",
            'steps[].id: Must match the pattern "^[a-z0-9_]+$".',
            "steps[].optional: Defaults to false.",
        ]
        assert result.dropped == 6

    def test_an_unknown_format_becomes_a_sentence_and_a_known_one_is_kept(self):
        out = to_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "who": {"type": "string", "format": "email"},
                    "when": {"type": "string", "format": "date-time"},
                },
            }
        )
        assert out["properties"]["who"] == {
            "type": "STRING",
            "description": "An email address.",
        }
        assert out["properties"]["when"]["format"] == "date-time"

    def test_a_union_keeps_the_first_shape_and_describes_the_rest(self):
        result = translate(
            {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            }
        )
        assert result.usable is True
        assert result.schema["properties"]["x"]["type"] == "STRING"
        assert "May also be a whole number." in result.schema["properties"]["x"]["description"]

    @pytest.mark.parametrize(
        "constraint",
        [
            {"pattern": "^x"},
            {"minLength": 2},
            {"maxLength": 3},
            {"not": {"type": "string"}},
            {"if": {"const": "a"}, "then": {"const": "b"}},
            {"patternProperties": {"^s_": {"type": "string"}}},
            {"propertyNames": {"pattern": "^[a-z]+$"}},
            {"uniqueItems": True},
            {"multipleOf": 2},
        ],
    )
    def test_no_constraint_anywhere_makes_it_raise(self, constraint):
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string", **constraint}},
        }
        result = translate(schema)

        assert isinstance(result, Translation)
        assert result.schema["properties"]["field"]["type"] == "STRING"
        assert result.schema["properties"]["field"]["description"]  # said, not dropped silently

    # -- and the coarser fallback, for a shape rather than a constraint ------

    def test_a_free_form_object_drops_the_whole_schema_instead_of_the_field(self):
        """`args` is the free-form bag in our plan grammar, and Gemini emits
        only declared fields — so dropping it would mean every step coming back
        without its arguments, with nothing in the answer saying so."""
        schema = {
            "type": "object",
            "properties": {
                "op": {"type": "string"},
                "args": {"type": "object", "description": "whatever the op takes"},
            },
            "required": ["op", "args"],
        }
        result = translate(schema)

        assert result.usable is False
        assert result.schema == {}
        assert to_gemini_schema(schema) == {}
        assert "no fixed fields" in result.reason

    def test_the_dropped_schema_comes_back_as_a_prompt_instead(self):
        schema = {"type": "object", "properties": {"args": {"type": "object"}}}
        hint = translate(schema).prompt_hint()

        assert "single JSON object" in hint
        assert '"args"' in hint  # the original schema, verbatim, for the model to read

    def test_a_self_referencing_ref_is_refused_rather_than_unrolled_forever(self):
        schema = {
            "$defs": {
                "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}
            },
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
        result = translate(schema)

        assert result.usable is False
        assert "refers to itself" in result.reason

    @pytest.mark.parametrize("empty", [None, {}, [], "not a schema", True])
    def test_nothing_useful_is_not_an_exception_either(self, empty):
        result = translate(empty)
        assert result.usable is False
        assert result.schema == {}
