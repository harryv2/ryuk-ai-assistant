"""What this conversation has referred to, and how a follow-up finds it again.

Every time a node surfaces something to the user — an email in a list, an event,
a file, a person — a row goes into ``conversation_entities``. Then "that email
about the proposal" resolves against about twenty rows instead of re-parsing
five runs' worth of result blobs, and it costs one indexed query rather than an
embedding and three searches.

Two halves:

* :func:`record` reads what the steps produced and writes the chips;
* :func:`resolve_reference` reads them back for a deixis phrase, and is honest
  about ties. Two candidates that both fit is an ambiguity to ask about, not a
  coin to flip — and recency inside one run is not evidence of anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Sequence

from app.core.logging import get_logger
from app.ops.base import service_of

log = get_logger(__name__)

EMAIL = "email"
EVENT = "event"
FILE = "file"
PERSON = "person"

# Where a list of results hides, whatever the op called it.
_LIST_KEYS: Final[tuple[str, ...]] = ("hits", "results", "items", "events", "messages", "files")

_ID_KEYS: Final[dict[str, tuple[str, ...]]] = {
    EMAIL: ("message_id", "id", "gmail_message_id", "thread_id"),
    EVENT: ("event_id", "id", "ical_uid"),
    FILE: ("file_id", "id"),
}

_LABEL_KEYS: Final[dict[str, tuple[str, ...]]] = {
    EMAIL: ("subject", "title", "snippet"),
    EVENT: ("title", "summary", "name"),
    FILE: ("name", "title", "filename"),
}

_TYPE_BY_SERVICE: Final[dict[str, str]] = {"gmail": EMAIL, "gcal": EVENT, "gdrive": FILE}

_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Deixis words carry no information of their own — they say "the one we were
# just talking about". Stripping them is what leaves the part that does.
_DEIXIS_WORDS: Final[frozenset[str]] = frozenset(
    {
        "that", "this", "those", "these", "the", "it", "them", "one", "ones",
        "same", "above", "previous", "earlier", "last", "first", "second",
        "you", "found", "mentioned", "showed", "sent", "my", "me", "a", "an",
        "about", "with", "from", "of", "in", "on", "for",
    }
)

_TYPE_WORDS: Final[dict[str, str]] = {
    "email": EMAIL, "emails": EMAIL, "mail": EMAIL, "message": EMAIL,
    "messages": EMAIL, "thread": EMAIL, "reply": EMAIL,
    "meeting": EVENT, "meetings": EVENT, "event": EVENT, "events": EVENT,
    "invite": EVENT, "call": EVENT, "appointment": EVENT,
    "file": FILE, "files": FILE, "doc": FILE, "docs": FILE, "document": FILE,
    "deck": FILE, "sheet": FILE, "folder": FILE, "attachment": FILE, "pdf": FILE,
    "person": PERSON, "people": PERSON, "guy": PERSON, "contact": PERSON,
}


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------


def _get(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict):
            if item.get(name) not in (None, ""):
                return item[name]
        else:
            value = getattr(item, name, None)
            if value not in (None, ""):
                return value
    return None


def _rows(result: Any) -> list[Any]:
    """Every item a step surfaced, whichever key the op used for the list."""
    if isinstance(result, list):
        return [r for r in result if isinstance(r, (dict, object))]
    if not isinstance(result, dict):
        return []
    for key in _LIST_KEYS:
        value = result.get(key)
        if isinstance(value, list) and value:
            return value
    # A single-item result — `gmail.get_email` returns the message itself.
    if any(result.get(key) for key in ("message_id", "event_id", "file_id")):
        return [result]
    return []


def _people_from(item: Any, entity_type: str) -> list[dict[str, Any]]:
    """Addresses worth remembering: a sender, an organiser, a guest list."""
    people: list[dict[str, Any]] = []

    sender = _get(item, "from_email", "sender", "organizer_email", "owner_email")
    if isinstance(sender, str) and _ADDRESS.fullmatch(sender.strip()):
        name = _get(item, "from_name", "sender_name", "organizer_name", "owner_name")
        people.append(
            {
                "entity_type": PERSON,
                "entity_ref": sender.strip().lower(),
                "label": str(name or sender).strip(),
                "meta": {"role": "sender" if entity_type == EMAIL else "organizer"},
            }
        )

    attendees = _get(item, "attendees", "attendee_emails", "participants", "to_emails")
    for attendee in list(attendees or [])[:10]:
        if isinstance(attendee, str):
            address, label = attendee, attendee
        elif isinstance(attendee, dict):
            address = attendee.get("email") or ""
            label = attendee.get("name") or address
        else:
            continue
        if isinstance(address, str) and _ADDRESS.fullmatch(address.strip()):
            people.append(
                {
                    "entity_type": PERSON,
                    "entity_ref": address.strip().lower(),
                    "label": str(label).strip(),
                    "meta": {"role": "attendee"},
                }
            )
    return people


def extract(op_name: str, result: Any, *, limit: int = 25) -> list[dict[str, Any]]:
    """Turn one step's result into chip rows. Nothing here touches the database."""
    entity_type = _TYPE_BY_SERVICE.get(service_of(op_name))
    if entity_type is None:
        return []

    out: list[dict[str, Any]] = []
    for item in _rows(result)[:limit]:
        ref = _get(item, *_ID_KEYS[entity_type])
        if not ref:
            continue
        label = _get(item, *_LABEL_KEYS[entity_type]) or str(ref)
        meta: dict[str, Any] = {}
        if entity_type == EMAIL:
            meta = {
                "from": _get(item, "from_email", "sender"),
                "from_name": _get(item, "from_name"),
                "date": _str_time(_get(item, "received_at", "date", "internal_date")),
                "thread_id": _get(item, "thread_id"),
                # A drafted email is the one entity a follow-up turn acts on
                # most — "send it" needs this id, and without it the planner
                # can see the draft happened but cannot reference it.
                "draft_id": _get(item, "draft_id"),
                "kind": "draft" if _get(item, "draft_id") else None,
                "to": _get(item, "to"),
            }
        elif entity_type == EVENT:
            meta = {
                "starts_at": _str_time(_get(item, "starts_at", "start")),
                "ends_at": _str_time(_get(item, "ends_at", "end")),
                "location": _get(item, "location"),
                "organizer": _get(item, "organizer_email"),
            }
        elif entity_type == FILE:
            meta = {
                "mime_type": _get(item, "mime_type"),
                "modified_at": _str_time(_get(item, "modified_at", "modified")),
                "link": _get(item, "web_view_link", "link", "url"),
                "owner": _get(item, "owner_email"),
            }
        out.append(
            {
                "entity_type": entity_type,
                "entity_ref": str(ref),
                "label": str(label)[:300],
                "meta": {k: v for k, v in meta.items() if v not in (None, "")},
            }
        )
        out.extend(_people_from(item, entity_type))

    # Collapse duplicates, keeping the first spelling of each.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in out:
        key = (row["entity_type"], row["entity_ref"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _str_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def from_results(results: dict[str, Any], plan_steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chips for every step of a finished run, in plan order."""
    op_by_node = {str(step.get("id")): str(step.get("op") or "") for step in plan_steps}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for node_id, result in results.items():
        for row in extract(op_by_node.get(node_id, ""), result):
            key = (row["entity_type"], row["entity_ref"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


async def record(
    session: Any,
    user_id: str,
    conversation_id: str,
    *,
    results: dict[str, Any],
    steps: Sequence[dict[str, Any]],
    run_id: str | None = None,
    trim_to: int = 60,
) -> list[dict[str, Any]]:
    """Write every surfaced item to ``conversation_entities``.

    Called once at the end of a run rather than per step: a search result set is
    one insert, not twenty. Never raises — losing a chip costs a follow-up a
    search, and that is not worth failing an answer over.
    """
    rows = from_results(results, steps)
    if not rows:
        return []
    try:
        from app.db.repositories import entities as repo

        await repo.upsert_entities(session, user_id, conversation_id, rows, run_id=run_id)
        if trim_to:
            await repo.trim_entities(session, user_id, conversation_id, keep=trim_to)
    except Exception as exc:  # noqa: BLE001 - a chip is a convenience, not the answer
        log.warning("entities.record_failed", conversation_id=conversation_id, error=str(exc))
        return []
    return rows


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    """What a deixis phrase resolved to, and whether it resolved cleanly."""

    phrase: str
    entity_type: str | None
    candidates: list[dict[str, Any]]
    confident: bool
    reason: str

    @property
    def entity(self) -> dict[str, Any] | None:
        return self.candidates[0] if self.confident and self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "entity_type": self.entity_type,
            "confident": self.confident,
            "reason": self.reason,
            "candidates": self.candidates,
        }


def _as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return {
        "entity_type": getattr(row, "entity_type", None),
        "entity_ref": getattr(row, "entity_ref", None),
        "label": getattr(row, "label", None),
        "meta": getattr(row, "meta", None) or {},
        "last_seen_at": _str_time(getattr(row, "last_seen_at", None)),
    }


def _terms(phrase: str) -> tuple[str | None, list[str]]:
    """Split "that email about the proposal" into a type and the words that matter."""
    words = re.findall(r"[a-z0-9@._-]+", (phrase or "").lower())
    entity_type = next((_TYPE_WORDS[w] for w in words if w in _TYPE_WORDS), None)
    terms = [w for w in words if w not in _DEIXIS_WORDS and w not in _TYPE_WORDS and len(w) > 2]
    return entity_type, terms


async def resolve_reference(
    session: Any,
    user_id: str,
    conversation_id: str,
    phrase: str,
    *,
    entity_type: str | None = None,
    limit: int = 5,
) -> Resolution:
    """Resolve "that email about the proposal" against what has been shown.

    Confident only when one candidate stands alone. Two matches is the honest
    answer "I do not know which", and the caller turns that into a question —
    picking the more recent one would be guessing dressed as an answer.
    """
    from app.db.repositories import entities as repo

    wanted_type, terms = _terms(phrase)
    wanted_type = entity_type or wanted_type

    rows: list[Any] = []
    try:
        if terms:
            for term in terms[:3]:
                rows.extend(
                    await repo.search_entities(
                        session,
                        user_id,
                        conversation_id,
                        term,
                        entity_type=wanted_type,
                        limit=limit,
                    )
                )
        if not rows:
            rows = await repo.list_entities(
                session, user_id, conversation_id, entity_type=wanted_type, limit=limit
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("entities.resolve_failed", conversation_id=conversation_id, error=str(exc))
        return Resolution(phrase, wanted_type, [], False, "the entity store could not be read")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item = _as_dict(row)
        ref = str(item.get("entity_ref") or "")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        candidates.append(item)

    if not candidates:
        return Resolution(
            phrase, wanted_type, [], False, "nothing in this conversation matches that"
        )

    if terms:
        # A term that actually appears in a label is evidence. Anything else is
        # only "we mentioned it recently", which is not.
        matched = [
            c
            for c in candidates
            if any(term in str(c.get("label", "")).lower() for term in terms)
        ]
        if len(matched) == 1:
            return Resolution(phrase, wanted_type, matched, True, "one label matches the words used")
        if len(matched) > 1:
            return Resolution(
                phrase,
                wanted_type,
                matched[:limit],
                False,
                f"{len(matched)} things in this conversation match — ask which one",
            )

    if len(candidates) == 1:
        return Resolution(
            phrase, wanted_type, candidates, True, "only one thing of that kind has come up"
        )

    return Resolution(
        phrase,
        wanted_type,
        candidates[:limit],
        False,
        "several things of that kind have come up; recency alone is not evidence",
    )


async def recent(
    session: Any,
    user_id: str,
    conversation_id: str,
    *,
    entity_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """The chip list the planner is shown, most recently referred to first."""
    try:
        from app.db.repositories import entities as repo

        rows = await repo.list_entities(
            session, user_id, conversation_id, entity_type=entity_type, limit=limit
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("entities.recent_failed", conversation_id=conversation_id, error=str(exc))
        return []
    return [_as_dict(row) for row in rows]


__all__ = [
    "EMAIL",
    "EVENT",
    "FILE",
    "PERSON",
    "Resolution",
    "extract",
    "from_results",
    "recent",
    "record",
    "resolve_reference",
]
