"""Structured answers the UI can draw, instead of markdown it can only print.

An answer like "what is on my calendar next week" is a *set of things with
structure* — six events, each with a time, a title and some guests. Flattening
that to markdown and asking the browser to re-read it throws the structure away
at the last possible moment, and it means the answer can never do anything: no
tapping a free slot to book it, no opening the file, no "move this one".

So a message can carry widget blocks beside its text. Two rules make that safe.

**Never HTML.** A widget is data, and the UI owns every pixel of how it is
drawn. Model-authored markup injected into a page that holds somebody's mailbox
is an XSS hole, and sanitising it is a game you lose slowly. It is also worse
on its own terms: unstyleable, untestable, expensive in tokens, and different
every time.

**Always a text fallback.** Every widget carries the markdown it replaces.
Messages are durable — one written today has to still render in a year, in a
client that may not know this widget, in a screen reader, in a copy-paste, in
an email digest. The widget is an enhancement, never the only copy.

Two families live here:

* **Typed** — ``event_list``, ``email_list``, ``file_list``, ``free_slots``.
  Built by :mod:`app.orchestrator.render` from rows an op actually returned,
  which costs nothing extra because the template already knew the shape.
* **Generic** — ``table``, ``list``, ``stat``, ``key_values``, ``timeline``,
  ``comparison``, ``chips``. Small composable primitives the model can reach
  for when an answer has a shape nobody predicted.

Interaction is deliberately tiny. A widget button does one of two things:

* ``ask``  — puts a sentence in the composer as if the person typed it, which
  means it goes through the ordinary pipeline and a write still stops at its
  confirmation card;
* ``open`` — follows a link that came out of the mirror.

A widget cannot execute a write. Writes stay behind ``pending_inputs`` and
``actions``, where ``requires_input_id`` is NOT NULL and the database enforces
it. A button that could act directly would be a second write path around the
one property the whole design rests on.
"""

from __future__ import annotations

from typing import Any, Final, Iterable, Sequence
from urllib.parse import urlparse

from app.core.logging import get_logger

log = get_logger(__name__)

WIDGET_VERSION: Final[int] = 1

#: The one place a widget kind becomes real. A kind not listed here is dropped
#: and the text stands in for it, which is what makes an unknown widget a
#: non-event rather than a broken message.
TYPED_KINDS: Final[frozenset[str]] = frozenset(
    {"event_list", "email_list", "file_list", "free_slots"}
)
GENERIC_KINDS: Final[frozenset[str]] = frozenset(
    {"table", "list", "stat", "key_values", "timeline", "comparison", "chips"}
)
KINDS: Final[frozenset[str]] = TYPED_KINDS | GENERIC_KINDS

#: Only these schemes can appear behind an `open`. `javascript:` and `data:`
#: are the obvious ones to keep out; anything not http(s) has no business in a
#: link we rendered for somebody.
SAFE_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

#: Caps. A widget is a summary, not a data dump — and an unbounded list from a
#: model is a way to make one message cost a megabyte.
MAX_ITEMS: Final[int] = 50
MAX_COLUMNS: Final[int] = 8
MAX_TEXT: Final[int] = 400
MAX_ACTIONS: Final[int] = 4


class WidgetError(ValueError):
    """A widget that cannot be rendered safely. The text is used instead."""


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #


def _text(value: Any, *, limit: int = MAX_TEXT) -> str:
    """Plain text, truncated. Never interpreted as markup by anything."""
    if value is None:
        return ""
    out = str(value).replace("\x00", "").strip()
    return out[:limit]


def _safe_url(value: Any, *, allowed: set[str] | None = None) -> str | None:
    """A link, or nothing.

    ``allowed`` is the set of urls that actually appeared in this run's data.
    When it is given — which it is for anything a model wrote — a link has to
    be one of them. A model inventing a plausible-looking url and us rendering
    it as a button is phishing with our own chrome, and the fact that it would
    usually be right is exactly what makes it dangerous.
    """
    url = _text(value, limit=2048)
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in SAFE_SCHEMES or not parsed.netloc:
        return None
    if allowed is not None and url not in allowed:
        log.info("widget.url_not_grounded", host=parsed.netloc)
        return None
    return url


def _actions(raw: Any, *, allowed_urls: set[str] | None) -> list[dict[str, Any]]:
    """The buttons on a widget, reduced to the two verbs that exist."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw[:MAX_ACTIONS]:
        if not isinstance(entry, dict):
            continue
        label = _text(entry.get("label"), limit=48)
        if not label:
            continue
        kind = str(entry.get("kind") or entry.get("type") or "").lower()
        if kind == "ask":
            question = _text(entry.get("query") or entry.get("ask"), limit=280)
            if question:
                out.append({"kind": "ask", "label": label, "query": question})
        elif kind == "open":
            url = _safe_url(entry.get("url"), allowed=allowed_urls)
            if url:
                out.append({"kind": "open", "label": label, "url": url})
    return out


# --------------------------------------------------------------------------- #
# Per-kind normalisation
# --------------------------------------------------------------------------- #


def _items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw[:MAX_ITEMS] if isinstance(entry, dict)]


def _clean_list(body: dict[str, Any], urls: set[str] | None) -> dict[str, Any]:
    items = []
    for entry in _items(body.get("items")):
        title = _text(entry.get("title") or entry.get("label"), limit=200)
        if not title:
            continue
        items.append(
            {
                "title": title,
                "subtitle": _text(entry.get("subtitle"), limit=200) or None,
                "meta": _text(entry.get("meta"), limit=80) or None,
                "badge": _text(entry.get("badge"), limit=32) or None,
                "actions": _actions(entry.get("actions"), allowed_urls=urls),
            }
        )
    if not items:
        raise WidgetError("a list with nothing in it")
    return {"items": items}


def _clean_table(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    columns = []
    for column in (body.get("columns") or [])[:MAX_COLUMNS]:
        if not isinstance(column, dict):
            continue
        key = _text(column.get("key"), limit=40)
        if not key:
            continue
        columns.append(
            {
                "key": key,
                "label": _text(column.get("label"), limit=60) or key,
                "align": "right" if column.get("align") == "right" else "left",
            }
        )
    if not columns:
        raise WidgetError("a table with no columns")

    keys = [column["key"] for column in columns]
    rows = [
        {key: _text(entry.get(key), limit=160) for key in keys}
        for entry in _items(body.get("rows"))
    ]
    if not rows:
        raise WidgetError("a table with no rows")
    return {"columns": columns, "rows": rows}


def _clean_stat(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    value = _text(body.get("value"), limit=40)
    if not value:
        raise WidgetError("a stat with no value")
    tone = str(body.get("tone") or "").lower()
    return {
        "value": value,
        "label": _text(body.get("label"), limit=80) or None,
        "detail": _text(body.get("detail"), limit=160) or None,
        "tone": tone if tone in {"good", "bad", "warn"} else None,
    }


def _clean_key_values(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    pairs = []
    for entry in _items(body.get("pairs")):
        label = _text(entry.get("label"), limit=80)
        if not label:
            continue
        pairs.append({"label": label, "value": _text(entry.get("value"), limit=200)})
    if not pairs:
        raise WidgetError("no pairs")
    return {"pairs": pairs}


def _clean_timeline(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    entries = []
    for entry in _items(body.get("entries")):
        title = _text(entry.get("title"), limit=200)
        if not title:
            continue
        entries.append(
            {
                "at": _text(entry.get("at"), limit=60) or None,
                "title": title,
                "detail": _text(entry.get("detail"), limit=240) or None,
            }
        )
    if not entries:
        raise WidgetError("an empty timeline")
    return {"entries": entries}


def _side(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    pairs = []
    for entry in _items(raw.get("pairs")):
        label = _text(entry.get("label"), limit=80)
        if label:
            pairs.append({"label": label, "value": _text(entry.get("value"), limit=200)})
    if not pairs:
        return None
    return {"label": _text(raw.get("label"), limit=60) or "", "pairs": pairs}


def _clean_comparison(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    left, right = _side(body.get("left")), _side(body.get("right"))
    if left is None or right is None:
        raise WidgetError("a comparison needs both sides")
    return {"left": left, "right": right}


def _clean_chips(body: dict[str, Any], urls: set[str] | None) -> dict[str, Any]:
    chips = []
    for entry in _items(body.get("items")):
        label = _text(entry.get("label"), limit=60)
        if not label:
            continue
        actions = _actions([entry.get("action")] if entry.get("action") else [], allowed_urls=urls)
        chips.append({"label": label, "action": actions[0] if actions else None})
    if not chips:
        raise WidgetError("no chips")
    return {"items": chips}


def _clean_free_slots(body: dict[str, Any], _urls: set[str] | None) -> dict[str, Any]:
    days = []
    for day in _items(body.get("days")):
        slots = []
        for slot in _items(day.get("slots")):
            start, end = _text(slot.get("start"), limit=40), _text(slot.get("end"), limit=40)
            if start and end:
                slots.append({"start": start, "end": end, "label": _text(slot.get("label"), limit=40) or None})
        if slots:
            days.append({"date": _text(day.get("date"), limit=40), "slots": slots})
    if not days:
        raise WidgetError("no free time to show")
    return {"days": days, "duration_minutes": int(body.get("duration_minutes") or 0) or None}


def _clean_event_list(body: dict[str, Any], urls: set[str] | None) -> dict[str, Any]:
    events = []
    for entry in _items(body.get("items")):
        title = _text(entry.get("title"), limit=200)
        if not title:
            continue
        events.append(
            {
                "id": _text(entry.get("id"), limit=255) or None,
                "title": title,
                "starts_at": _text(entry.get("starts_at"), limit=40) or None,
                "ends_at": _text(entry.get("ends_at"), limit=40) or None,
                "all_day": bool(entry.get("all_day")),
                "location": _text(entry.get("location"), limit=120) or None,
                "guests": [_text(g, limit=160) for g in (entry.get("guests") or [])[:12]],
                "status": _text(entry.get("status"), limit=32) or None,
                "url": _safe_url(entry.get("url"), allowed=urls),
            }
        )
    if not events:
        raise WidgetError("no events")
    return {"items": events}


def _clean_email_list(body: dict[str, Any], urls: set[str] | None) -> dict[str, Any]:
    mails = []
    for entry in _items(body.get("items")):
        subject = _text(entry.get("subject"), limit=200)
        if not subject:
            continue
        mails.append(
            {
                "id": _text(entry.get("id"), limit=255) or None,
                "subject": subject,
                "from_name": _text(entry.get("from_name") or entry.get("from_email"), limit=120) or None,
                "from_email": _text(entry.get("from_email"), limit=160) or None,
                "received_at": _text(entry.get("received_at"), limit=40) or None,
                "excerpt": _text(entry.get("excerpt"), limit=240) or None,
                "unread": bool(entry.get("unread")),
                "url": _safe_url(entry.get("url"), allowed=urls),
            }
        )
    if not mails:
        raise WidgetError("no emails")
    return {"items": mails}


def _clean_file_list(body: dict[str, Any], urls: set[str] | None) -> dict[str, Any]:
    files = []
    for entry in _items(body.get("items")):
        name = _text(entry.get("name"), limit=200)
        if not name:
            continue
        files.append(
            {
                "id": _text(entry.get("id"), limit=255) or None,
                "name": name,
                "mime_type": _text(entry.get("mime_type"), limit=120) or None,
                "owner": _text(entry.get("owner"), limit=160) or None,
                "modified_at": _text(entry.get("modified_at"), limit=40) or None,
                "size_bytes": int(entry["size_bytes"]) if str(entry.get("size_bytes") or "").isdigit() else None,
                "url": _safe_url(entry.get("url"), allowed=urls),
            }
        )
    if not files:
        raise WidgetError("no files")
    return {"items": files}


_CLEANERS = {
    "event_list": _clean_event_list,
    "email_list": _clean_email_list,
    "file_list": _clean_file_list,
    "free_slots": _clean_free_slots,
    "table": _clean_table,
    "list": _clean_list,
    "stat": _clean_stat,
    "key_values": _clean_key_values,
    "timeline": _clean_timeline,
    "comparison": _clean_comparison,
    "chips": _clean_chips,
}


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #


def grounded_urls(results: Iterable[Any]) -> set[str]:
    """Every url that came out of this run's data.

    A model-authored `open` has to be one of these. It is cheap to collect and
    it is the difference between "a link to your file" and "a link the model
    thought sounded right".
    """
    found: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or len(found) > 500:
            return
        if isinstance(node, str):
            if node.startswith("http://") or node.startswith("https://"):
                found.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, depth + 1)

    for result in results:
        walk(result)
    return found


def widget_block(
    kind: str,
    body: dict[str, Any],
    *,
    text: str,
    allowed_urls: set[str] | None = None,
    replaces_text: bool = False,
) -> dict[str, Any] | None:
    """One validated widget block, or ``None`` when it cannot be drawn.

    ``None`` is not an error path — it is the ordinary outcome for a widget
    that arrived malformed, and the caller keeps the text block it already had.
    A message that renders as plain prose is a small loss; a message that
    renders as an exception is a broken product.
    """
    name = str(kind or "").strip().lower()
    if name not in KINDS:
        log.info("widget.unknown_kind", kind=name[:40])
        return None

    fallback = _text(text, limit=4000)
    if not fallback:
        # Without a fallback the widget would be the only copy of the answer,
        # which is exactly the situation this module exists to avoid.
        log.info("widget.no_fallback", kind=name)
        return None

    try:
        cleaned = _CLEANERS[name](dict(body or {}), allowed_urls)
    except WidgetError as exc:
        log.info("widget.rejected", kind=name, reason=str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 - a bad widget must never fail a run
        log.warning("widget.cleaner_failed", kind=name, error=str(exc))
        return None

    return {
        "type": "widget",
        "v": WIDGET_VERSION,
        "widget": name,
        "text": fallback,
        # Whether the widget *is* the answer or merely decorates it.
        #
        # A template widget is the same rows the markdown listed, so showing
        # both is the answer twice. A widget the synthesiser offered sits under
        # prose that says something the widget does not — the words are the
        # answer there, and dropping them would lose it.
        "replaces_text": bool(replaces_text),
        "data": cleaned,
    }


def from_model(
    payload: Any,
    *,
    text: str,
    results: Sequence[Any] = (),
) -> dict[str, Any] | None:
    """A widget the synthesiser asked for, checked before it is believed.

    The model chooses the *shape*; the data still has to survive this. Links
    are held to what the run actually saw, every string is plain text, and
    anything that does not fit is dropped in favour of the prose it wrote
    anyway.
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("widget") or payload.get("kind") or payload.get("type")
    body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    return widget_block(
        str(kind or ""),
        body or {},
        text=text,
        allowed_urls=grounded_urls(results),
        replaces_text=False,
    )


__all__ = [
    "GENERIC_KINDS",
    "KINDS",
    "TYPED_KINDS",
    "WIDGET_VERSION",
    "WidgetError",
    "from_model",
    "grounded_urls",
    "widget_block",
]
