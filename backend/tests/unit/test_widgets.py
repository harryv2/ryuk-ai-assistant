"""The widget validator, which is the boundary a model writes across.

A widget is data the browser turns into buttons and links inside a page that
holds somebody's mailbox. Two of the things that arrive here are written by a
language model — the shape it chose and the values it filled in — so this
module's job is not to be permissive. Everything below is a case where being
permissive would cost something real:

* a `javascript:` url rendered as a button is a script somebody clicks;
* a plausible-looking url the model invented is phishing wearing our chrome;
* an unbounded list is one message that costs a megabyte;
* a widget with no text fallback is an answer that vanishes in a client that
  does not know this kind, in a screen reader, and in a copy-paste.

The negative cases matter more than the positive ones here. A widget that
fails to render is a small loss — the prose is still on screen. A widget that
renders something it should not is the kind of bug you read about.
"""

from __future__ import annotations

import pytest

from app.orchestrator import widgets

TEXT = "Six events next week."


def block(kind: str, body: dict, **kw):
    return widgets.widget_block(kind, body, text=TEXT, **kw)


# --------------------------------------------------------------------------- #
# The fallback is not optional
# --------------------------------------------------------------------------- #


def test_a_widget_without_a_text_fallback_is_refused():
    """The text is the only copy that survives an unknown widget kind."""
    out = widgets.widget_block(
        "stat", {"value": "6"}, text="   "
    )
    assert out is None


def test_every_widget_carries_its_fallback():
    out = block("stat", {"value": "6", "label": "events"})
    assert out is not None
    assert out["text"] == TEXT
    assert out["v"] == widgets.WIDGET_VERSION


def test_replaces_text_is_off_unless_asked_for():
    """A template's widget stands in for its markdown; a model's decorates it.

    Defaulting the other way would delete prose the widget does not contain.
    """
    assert block("stat", {"value": "6"})["replaces_text"] is False
    assert block("stat", {"value": "6"}, replaces_text=True)["replaces_text"] is True


# --------------------------------------------------------------------------- #
# Unknown kinds are a non-event, never an error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["", "iframe", "html", "script", "Table ", "unknown"])
def test_an_unrecognised_kind_is_dropped(kind):
    assert block(kind, {"items": [{"title": "x"}]}) is None


def test_kind_matching_is_case_insensitive_and_trimmed():
    assert block("  TABLE ", {"columns": [{"key": "a"}], "rows": [{"a": "1"}]}) is not None


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///etc/passwd",
        "vbscript:msgbox",
        "//evil.example/x",
        "notaurl",
        "",
    ],
)
def test_only_http_urls_survive(url):
    out = block(
        "list",
        {"items": [{"title": "x", "actions": [{"kind": "open", "label": "Go", "url": url}]}]},
    )
    assert out["data"]["items"][0]["actions"] == []


def test_an_http_url_survives_when_nothing_constrains_it():
    """A backend-authored widget takes its urls from the row it just read."""
    out = block(
        "list",
        {
            "items": [
                {
                    "title": "x",
                    "actions": [
                        {"kind": "open", "label": "Go", "url": "https://drive.google.com/f/1"}
                    ],
                }
            ]
        },
    )
    assert out["data"]["items"][0]["actions"][0]["url"] == "https://drive.google.com/f/1"


def test_a_model_cannot_invent_a_link_the_run_never_saw():
    """The whole point of `allowed_urls`.

    The invented url here is well-formed, https, and on a real Google domain —
    which is exactly why matching on shape would let it through.
    """
    seen = {"https://drive.google.com/file/d/REAL/view"}
    out = block(
        "list",
        {
            "items": [
                {
                    "title": "x",
                    "actions": [
                        {
                            "kind": "open",
                            "label": "Open",
                            "url": "https://drive.google.com/file/d/INVENTED/view",
                        }
                    ],
                }
            ]
        },
        allowed_urls=seen,
    )
    assert out["data"]["items"][0]["actions"] == []


def test_a_grounded_link_is_kept():
    seen = {"https://mail.google.com/mail/u/0/#inbox/abc"}
    out = block(
        "list",
        {
            "items": [
                {
                    "title": "x",
                    "actions": [{"kind": "open", "label": "Open", "url": next(iter(seen))}],
                }
            ]
        },
        allowed_urls=seen,
    )
    assert out["data"]["items"][0]["actions"][0]["url"] == next(iter(seen))


def test_grounded_urls_walks_nested_results():
    found = widgets.grounded_urls(
        [
            {"hits": [{"url": "https://a.example/1", "nested": {"u": "http://b.example/2"}}]},
            ["https://c.example/3", {"skip": "not a url"}],
        ]
    )
    assert found == {"https://a.example/1", "http://b.example/2", "https://c.example/3"}


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def test_only_ask_and_open_are_actions():
    """A third verb is a second write path around the confirmation gate."""
    out = block(
        "list",
        {
            "items": [
                {
                    "title": "x",
                    "actions": [
                        {"kind": "run", "label": "Send it", "op": "gmail.send_email"},
                        {"kind": "delete", "label": "Delete", "id": "abc"},
                        {"kind": "ask", "label": "Move", "query": "Move it to Friday"},
                    ],
                }
            ]
        },
    )
    kept = out["data"]["items"][0]["actions"]
    assert [a["kind"] for a in kept] == ["ask"]
    assert kept[0]["query"] == "Move it to Friday"


def test_an_action_without_a_label_is_dropped():
    out = block(
        "list",
        {"items": [{"title": "x", "actions": [{"kind": "ask", "query": "do a thing"}]}]},
    )
    assert out["data"]["items"][0]["actions"] == []


def test_actions_are_capped():
    many = [{"kind": "ask", "label": f"a{i}", "query": f"q{i}"} for i in range(20)]
    out = block("list", {"items": [{"title": "x", "actions": many}]})
    assert len(out["data"]["items"][0]["actions"]) == widgets.MAX_ACTIONS


# --------------------------------------------------------------------------- #
# Size and shape
# --------------------------------------------------------------------------- #


def test_items_are_capped():
    rows = [{"title": f"row {i}"} for i in range(500)]
    out = block("list", {"items": rows})
    assert len(out["data"]["items"]) == widgets.MAX_ITEMS


def test_long_values_are_truncated():
    out = block("list", {"items": [{"title": "x" * 5000}]})
    assert len(out["data"]["items"][0]["title"]) <= 200


def test_a_widget_with_nothing_in_it_is_dropped():
    assert block("list", {"items": []}) is None
    assert block("table", {"columns": [], "rows": []}) is None
    assert block("key_values", {"pairs": []}) is None
    assert block("timeline", {"entries": []}) is None
    assert block("chips", {"items": []}) is None


def test_a_table_needs_both_columns_and_rows():
    assert block("table", {"columns": [{"key": "a", "label": "A"}]}) is None
    assert block("table", {"rows": [{"a": "1"}]}) is None


def test_a_table_keeps_only_declared_columns():
    """A row key nobody declared has nowhere to render, so it is not carried."""
    out = block(
        "table",
        {
            "columns": [{"key": "name", "label": "Name"}],
            "rows": [{"name": "Ada", "secret": "do not carry this"}],
        },
    )
    assert out["data"]["rows"] == [{"name": "Ada"}]


def test_a_comparison_needs_both_sides():
    one = {"label": "v2", "pairs": [{"label": "Price", "value": "55"}]}
    assert block("comparison", {"left": one}) is None
    assert block("comparison", {"left": one, "right": {"label": "v3", "pairs": []}}) is None
    assert block("comparison", {"left": one, "right": {**one, "label": "v3"}}) is not None


def test_nul_bytes_are_stripped():
    out = block("stat", {"value": "6\x00", "label": "ev\x00ents"})
    assert "\x00" not in out["data"]["value"]
    assert "\x00" not in out["data"]["label"]


def test_a_malformed_body_never_raises():
    """A bad widget costs the widget, never the answer or the run."""
    for body in ({"items": "not a list"}, {"items": [None, 3, "x"]}, {}, {"items": [{}]}):
        assert block("list", body) is None


# --------------------------------------------------------------------------- #
# from_model — the path a language model writes across
# --------------------------------------------------------------------------- #


def test_from_model_reads_either_key_for_the_kind():
    for payload in (
        {"widget": "stat", "data": {"value": "6"}},
        {"kind": "stat", "data": {"value": "6"}},
        {"type": "stat", "data": {"value": "6"}},
    ):
        assert widgets.from_model(payload, text=TEXT) is not None


def test_from_model_accepts_a_flat_body():
    out = widgets.from_model({"widget": "stat", "value": "6"}, text=TEXT)
    assert out is not None and out["data"]["value"] == "6"


def test_from_model_never_replaces_the_prose():
    """The words are the answer on this path; the widget only decorates them."""
    out = widgets.from_model({"widget": "stat", "data": {"value": "6"}}, text=TEXT)
    assert out["replaces_text"] is False


def test_from_model_grounds_links_against_the_run():
    payload = {
        "widget": "list",
        "data": {
            "items": [
                {
                    "title": "Report",
                    "actions": [
                        {"kind": "open", "label": "Open", "url": "https://evil.example/steal"}
                    ],
                }
            ]
        },
    }
    out = widgets.from_model(
        payload, text=TEXT, results=[{"hits": [{"url": "https://drive.google.com/x"}]}]
    )
    assert out["data"]["items"][0]["actions"] == []


@pytest.mark.parametrize("payload", [None, "a string", 42, [], {"widget": "nope"}])
def test_from_model_shrugs_at_rubbish(payload):
    assert widgets.from_model(payload, text=TEXT) is None
