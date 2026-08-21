"""The connector mirror: one table, ``sync_items``, four shapes.

A cache. Drop it and a resync rebuilds everything — nothing here is a source of
truth. Two jobs live in this module:

* **upsert** by the natural key each connector gives us, keeping an embedding
  alive when the content did not change so re-embedding can be skipped;
* **hybrid_search**, the query behind ``app.search.hybrid``. One ANN pass and
  one full-text pass, both pre-filtered on ``user_id``, joined on id, collapsed
  to one row per message / event / file / task keeping the best chunk.

One table per *shape*, not per source: ``sync_messages``, ``sync_events``,
``sync_files``. Gmail, Outlook mail, Slack and Jira comments are all messages;
Google and Outlook calendars are both events; Drive, OneDrive and Notion are
all files. Each table names its columns for what that shape actually holds — a
message is ``sent_at``, an event ``starts_at``, a file ``modified_at`` — and
each carries its own HNSW, so a vector search for events walks an index of
events rather than filtering the whole mirror afterwards.

They share a spine — ``connector``, ``source_id``, ``scope_key``,
``chunk_index``, ``content_hash``, ``embedding``, ``tsv`` — which is what lets
one search layer read all three without knowing which it has in front of it.

A spec is a *shape*, optionally narrowed to one *connector*:

* ``spec_for("message")`` — mail, chat and ticket comments from every connector
* ``spec_for("gmail")``   — the same shape, Gmail only

Which is why adding Outlook is a mapper and a row in ``SPECS``: no DDL, no new
index, and no extra arm in the search fan-out, because Outlook mail is a
message and Outlook Calendar is an event.

The rule for what earns a column: **a field you filter on is a column, a field
you only display is an ``attributes`` key.** Burying a queryable dimension in
JSONB is how recipients ended up unindexed under the old single table.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, case, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.ids import new_id
from app.db.models import EMBEDDING_DIM, SyncEvent, SyncFile, SyncMessage
from app.db.repositories import ensure_utc, utcnow

# --------------------------------------------------------------------------- #
# Shape specs — everything the generic helpers need to know about a shape
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Spec:
    key: str
    #: message | event | file | task — what the thing *is*
    kind: str
    #: None means every connector of this shape; a string narrows to one
    connector: str | None
    #: what a chunked row collapses to in search, in projected names
    collapse: tuple[str, ...]
    #: columns a search returns, projected into the shape's own vocabulary
    fields: tuple[str, ...]
    #: whitelisted filter key -> SQL fragment
    filters: dict[str, str]
    #: the shape's word for its source id, as callers write it
    ref_alias: str = "message_id"
    #: chunked shapes key on chunk_index; a whole-row shape does not
    chunked: bool = True
    model: type = SyncMessage
    table: str = "sync_messages"
    #: the column a time window filters on, named for what it holds
    time_col: str = "sent_at"
    #: the column holding the source's own id
    ref_col: str = "source_id"
    #: the column a person would recognise as the grouping
    group_col: str = "thread_id"
    #: natural key of a row, the conflict target of an upsert
    conflict: tuple[str, ...] = (
        "user_id",
        "connector",
        "scope_key",
        "source_id",
        "chunk_index",
    )
    #: Columns an upsert overwrites — everything but the id, the natural key
    #: and the generated columns. Per shape, because the shapes have different
    #: columns now; `_SHAPES` supplies the real list.
    mutable: tuple[str, ...] = ()
    #: extra columns stamped onto every row this spec writes
    defaults: dict[str, Any] = field(default_factory=dict)



# Every shape shares an identity prefix, so callers always know which connector
# an answer came from even when the search spanned several.
#
# A projection carries its own ``{t}`` because it is not always a bare column:
# some entries reach into ``attributes`` and some call a function, and blindly
# prefixing those with the alias produces ``t.coalesce(...)``.
# The shape is the table now, so `kind` is a constant rather than a column.
# It stays in the projection because callers downstream still switch on it.
def _identity(kind: str) -> tuple[str, ...]:
    return ("{t}.id", "{t}.connector", f"'{kind}'::text AS kind")


#: Filters every shape understands. `{time_col}`, `{ref_col}` and
#: `{group_col}` are filled from the spec, so one definition covers three
#: tables whose columns are named for what they actually hold.
_COMMON_FILTERS = {
    "since": "{t}.{time_col} >= :f_since",
    "until": "{t}.{time_col} < :f_until",
    "connectors": "{t}.connector = ANY(CAST(:f_connectors AS text[]))",
    "container_id": "{t}.{group_col} = :f_container_id",
    "external_ids": "{t}.{ref_col} = ANY(CAST(:f_external_ids AS text[]))",
    "chunk_index": "{t}.chunk_index = :f_chunk_index",
}


# --------------------------------------------------------------------------- #
# message — Gmail, Outlook mail, Slack, Teams, Jira comments
# --------------------------------------------------------------------------- #

_MESSAGE_FIELDS = _identity("message") + (
    "{t}.source_id AS message_id",
    "{t}.thread_id",
    "{t}.chunk_index",
    "{t}.subject",
    "{t}.from_email",
    "{t}.from_name",
    "{t}.to_emails",
    "{t}.cc_emails",
    "{t}.participant_emails",
    "{t}.body AS body_clean",
    "{t}.labels",
    "{t}.has_attachments",
    "{t}.is_unread",
    "{t}.sent_at AS received_at",
    "{t}.url",
    "{t}.content_hash",
    "{t}.updated_at",
)

_MESSAGE_FILTERS = {
    **_COMMON_FILTERS,
    "from_email": "{t}.from_email = :f_from_email",
    "from_emails": "{t}.from_email = ANY(CAST(:f_from_emails AS citext[]))",
    # "Was X a recipient", not "was X involved" — the sender is involved in
    # their own mail, and folding the two together turns "mail to Sarah" into
    # "mail with Sarah", which quietly returns her sent items too. Both are
    # indexed columns now, so both questions are a GIN hit.
    "to_email": "CAST(:f_to_email AS citext) = ANY({t}.to_emails)",
    "cc_email": "CAST(:f_cc_email AS citext) = ANY({t}.cc_emails)",
    "participant_emails": (
        "{t}.participant_emails && CAST(:f_participant_emails AS citext[])"
    ),
    "thread_id": "{t}.thread_id = :f_thread_id",
    "message_ids": "{t}.source_id = ANY(CAST(:f_message_ids AS text[]))",
    "labels": "{t}.labels && CAST(:f_labels AS text[])",
    "has_attachments": "{t}.has_attachments = :f_has_attachments",
    "is_unread": "{t}.is_unread = :f_is_unread",
    "subject_contains": "{t}.subject ILIKE '%' || :f_subject_contains || '%'",
}


# --------------------------------------------------------------------------- #
# event — Google Calendar, Outlook Calendar
# --------------------------------------------------------------------------- #

_EVENT_FIELDS = _identity("event") + (
    "{t}.source_id AS event_id",
    "{t}.scope_key AS calendar_id",
    "{t}.recurring_event_id",
    "{t}.title",
    "{t}.description",
    "{t}.location",
    "{t}.organizer_email",
    "{t}.attendees",
    "{t}.attendee_emails",
    "{t}.attendee_emails AS participant_emails",
    "{t}.starts_at",
    "{t}.ends_at",
    "{t}.all_day",
    "{t}.attributes ->> 'event_timezone' AS event_timezone",
    "{t}.status",
    "{t}.attributes ->> 'etag' AS etag",
    "{t}.url",
    "{t}.content_hash",
    "{t}.updated_at",
)

_EVENT_FILTERS = {
    **_COMMON_FILTERS,
    "calendar_id": "{t}.scope_key = :f_calendar_id",
    "organizer_email": "{t}.organizer_email = :f_organizer_email",
    "attendee_emails": "{t}.attendee_emails && CAST(:f_attendee_emails AS citext[])",
    "participant_emails": (
        "{t}.attendee_emails && CAST(:f_participant_emails AS citext[])"
    ),
    "event_ids": "{t}.source_id = ANY(CAST(:f_event_ids AS text[]))",
    "recurring_event_id": "{t}.recurring_event_id = :f_recurring_event_id",
    "status": "{t}.status = :f_status",
    "not_cancelled": "({t}.status IS DISTINCT FROM 'cancelled')",
    "all_day": "{t}.all_day = :f_all_day",
    "ends_after": "coalesce({t}.ends_at, {t}.starts_at) > :f_ends_after",
}


# --------------------------------------------------------------------------- #
# file — Drive, OneDrive, Dropbox, Notion
# --------------------------------------------------------------------------- #

_FILE_FIELDS = _identity("file") + (
    "{t}.source_id AS file_id",
    "{t}.chunk_index",
    "{t}.name",
    "{t}.mime_type",
    "{t}.owner_email",
    "{t}.shared_with_emails",
    "{t}.shared_with_emails AS participant_emails",
    "{t}.is_shared",
    "{t}.url AS web_view_link",
    "{t}.folder_path",
    "{t}.size_bytes",
    "{t}.content AS content_excerpt",
    "{t}.modified_at",
    "{t}.content_hash",
    "{t}.updated_at",
)

_FILE_FILTERS = {
    **_COMMON_FILTERS,
    "mime_type": "{t}.mime_type = :f_mime_type",
    "mime_types": "{t}.mime_type = ANY(CAST(:f_mime_types AS text[]))",
    "owner_email": "{t}.owner_email = :f_owner_email",
    "is_shared": "{t}.is_shared = :f_is_shared",
    "participant_emails": (
        "{t}.shared_with_emails && CAST(:f_participant_emails AS citext[])"
    ),
    "file_ids": "{t}.source_id = ANY(CAST(:f_file_ids AS text[]))",
    "folder_prefix": "{t}.folder_path LIKE :f_folder_prefix || '%'",
    "name_contains": "{t}.name ILIKE '%' || :f_name_contains || '%'",
}


#: Everything that differs between the three tables, in one place.
#:
#: `time_col` is the column a date window filters on, `ref_col` the source id,
#: `group_col` the thing a person would recognise as the grouping. They have
#: different names per shape because they *are* different things — a message is
#: sent, an event starts, a file is modified — and naming them all
#: `occurred_at` was the compromise the single table forced.
_SHAPES: dict[str, dict[str, Any]] = {
    "message": {
        "model": SyncMessage,
        "table": "sync_messages",
        "fields": _MESSAGE_FIELDS,
        "filters": _MESSAGE_FILTERS,
        "time_col": "sent_at",
        "ref_col": "source_id",
        "group_col": "thread_id",
        "ref_alias": "message_id",
        "collapse": "message_id",
        "chunked": True,
        "mutable": (
            "subject", "body", "from_email", "from_name", "to_emails", "cc_emails",
            "sent_at", "thread_id", "is_unread", "has_attachments",
            "labels", "url", "attributes", "content_hash", "updated_at",
        ),
    },
    "event": {
        "model": SyncEvent,
        "table": "sync_events",
        "fields": _EVENT_FIELDS,
        "filters": _EVENT_FILTERS,
        "time_col": "starts_at",
        "ref_col": "source_id",
        "group_col": "recurring_event_id",
        "ref_alias": "event_id",
        "collapse": "event_id",
        "chunked": False,
        "mutable": (
            "title", "description", "organizer_email", "attendee_emails", "attendees",
            "starts_at", "ends_at", "all_day", "location", "status",
            "recurring_event_id",
            "labels", "url", "attributes", "content_hash", "updated_at",
        ),
    },
    "file": {
        "model": SyncFile,
        "table": "sync_files",
        "fields": _FILE_FIELDS,
        "filters": _FILE_FILTERS,
        "time_col": "modified_at",
        "ref_col": "source_id",
        "group_col": "folder_id",
        "ref_alias": "file_id",
        "collapse": "file_id",
        "chunked": True,
        "mutable": (
            "name", "content", "owner_email", "shared_with_emails",
            "mime_type", "size_bytes", "modified_at",
            "folder_id", "folder_path", "is_shared",
            "labels", "url", "attributes", "content_hash", "updated_at",
        ),
    },
}


def _shape(key: str, kind: str, connector: str | None, **kw: Any) -> _Spec:
    """A shape spec, optionally narrowed to one connector."""
    shape = _SHAPES[kind]
    return _Spec(
        key=key,
        kind=kind,
        connector=connector,
        # Two connectors can hand out the same id, so the source is part of
        # what makes a hit distinct once a search spans more than one.
        collapse=("connector", shape["collapse"]),
        fields=shape["fields"],
        filters=shape["filters"],
        ref_alias=shape["ref_alias"],
        chunked=shape["chunked"],
        model=shape["model"],
        table=shape["table"],
        time_col=shape["time_col"],
        ref_col=shape["ref_col"],
        group_col=shape["group_col"],
        mutable=shape["mutable"],
        conflict=("user_id", "connector", "scope_key", "source_id", "chunk_index"),
        defaults={"connector": connector} if connector else {},
        **kw,
    )


#: The shapes, and the connector-scoped views onto them. Adding Outlook is two
#: lines here plus a mapper — no DDL, no new index, no new search arm, because
#: Outlook mail is a message and Outlook Calendar is an event.
SPECS: dict[str, _Spec] = {
    # every connector of a shape
    "message": _shape("message", "message", None),
    "event": _shape("event", "event", None),
    "file": _shape("file", "file", None),
    # one connector each
    "gmail": _shape("gmail", "message", "gmail"),
    "gcal": _shape("gcal", "event", "gcal"),
    "gdrive": _shape("gdrive", "file", "gdrive"),
}

#: The connectors that actually hold data, for counting and purging. Keyed by
#: the name ``sync_state`` and ``/sync/status`` use.
CONNECTOR_SPECS: dict[str, _Spec] = {
    key: spec for key, spec in SPECS.items() if spec.connector is not None
}


def spec_for(table: str) -> _Spec:
    """The spec for a shape ('message'…) or a connector ('gmail'…)."""
    try:
        return SPECS[table]
    except KeyError:
        raise AppError(
            "VALIDATION_ERROR",
            f"Unknown mirror shape {table!r}.",
            http=422,
            details={"table": table, "known": sorted(SPECS)},
        ) from None


def _scope(spec: _Spec) -> list[Any]:
    """The predicates that narrow a shape's table to this spec's rows.

    Under one table this also had to filter on `kind`, and forgetting it would
    not raise — it would quietly read another shape's rows. That failure mode
    is gone: the table *is* the shape, so all that remains is narrowing a
    shape to one connector when the spec asks for it.
    """
    if spec.connector is None:
        return []
    return [spec.model.connector == spec.connector]


GMAIL = SPECS["gmail"]
GCAL = SPECS["gcal"]
GDRIVE = SPECS["gdrive"]


# --------------------------------------------------------------------------- #
# Writing: the shape's vocabulary back into the table's columns
# --------------------------------------------------------------------------- #
#
# The mirror image of the projections above. A sync job hands us the words its
# connector uses — a message has a `subject` and a `from_email`, a file has a
# `name` and an `owner_email` — and this turns them into the columns every
# connector shares. Keeping the translation here rather than in each job is
# what stops the storage schema leaking into thirty mappers, and it means
# adding Outlook is a job that says `subject` like Gmail does.
#
# Anything a shape does not have a column for lands in `attributes`, which is
# the long tail: a mime type, an attachment flag, a Jira priority.

#: Columns that pass through untouched whatever the shape.
_PASSTHROUGH = frozenset(
    {
        "id",
        "user_id",
        "connector",
        "chunk_index",
        "content_hash",
        "embedding",
        "embed_model",
        "updated_at",
        "labels",
        "url",
    }
)

#: shape -> {incoming word: storage column}
_COLUMNS: dict[str, dict[str, str]] = {
    "message": {
        "message_id": "source_id",
        "mailbox_id": "scope_key",
        "thread_id": "thread_id",
        "subject": "subject",
        "from_email": "from_email",
        "from_name": "from_name",
        "to_emails": "to_emails",
        "cc_emails": "cc_emails",
        "body_clean": "body",
        "body": "body",
        "received_at": "sent_at",
        "sent_at": "sent_at",
        "is_unread": "is_unread",
        "has_attachments": "has_attachments",
    },
    "event": {
        "event_id": "source_id",
        "calendar_id": "scope_key",
        "title": "title",
        "description": "description",
        "location": "location",
        "organizer_email": "organizer_email",
        "attendees": "attendees",
        "attendee_emails": "attendee_emails",
        "starts_at": "starts_at",
        "ends_at": "ends_at",
        "all_day": "all_day",
        "status": "status",
        "recurring_event_id": "recurring_event_id",
    },
    "file": {
        "file_id": "source_id",
        "drive_id": "scope_key",
        "name": "name",
        "mime_type": "mime_type",
        "owner_email": "owner_email",
        "shared_with_emails": "shared_with_emails",
        "is_shared": "is_shared",
        "size_bytes": "size_bytes",
        "content_excerpt": "content",
        "content": "content",
        "web_view_link": "url",
        "folder_id": "folder_id",
        "folder_path": "folder_path",
        "modified_at": "modified_at",
    },
}


#: shape -> the words that belong in `attributes` rather than a column
_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "message": ("from_name", "has_attachments"),
    "event": ("location", "event_timezone", "etag", "recurring_event_id"),
    "file": ("mime_type", "size_bytes", "is_shared"),
    "task": ("priority", "issue_type", "assignee_email"),
}


def _person(email: Any, role: str, **extra: Any) -> dict[str, Any] | None:
    """One participant. Emails are stored folded, because they are matched as
    literal JSONB text and citext's case-insensitivity does not reach inside a
    JSON document."""
    address = str(email or "").strip().lower()
    if not address:
        return None
    out: dict[str, Any] = {"email": address, "role": role}
    out.update({k: v for k, v in extra.items() if v})
    return out


def _participants(kind: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Everyone involved, each under the role they played.

    One shape for every "who was on this" question — sender and recipients,
    organiser and attendees, owner, reporter and assignee — which is why one
    GIN index answers all of them.
    """
    people: list[dict[str, Any]] = []

    if kind == "message":
        people.append(_person(row.get("from_email"), "from", name=row.get("from_name")))
        for address in row.get("to_emails") or []:
            people.append(_person(address, "to"))
        for address in row.get("cc_emails") or []:
            people.append(_person(address, "cc"))

    elif kind == "event":
        people.append(_person(row.get("organizer_email"), "organizer"))
        for guest in row.get("attendees") or []:
            if not isinstance(guest, dict):
                people.append(_person(guest, "attendee"))
                continue
            people.append(
                _person(
                    guest.get("email"),
                    "attendee",
                    name=guest.get("name"),
                    status=guest.get("response_status") or guest.get("status"),
                )
            )

    elif kind == "file":
        people.append(_person(row.get("owner_email"), "owner"))
        for address in row.get("shared_with") or []:
            people.append(_person(address, "editor"))

    elif kind == "task":
        people.append(_person(row.get("reporter_email"), "reporter"))
        people.append(_person(row.get("assignee_email"), "assignee"))
        for address in row.get("watcher_emails") or []:
            people.append(_person(address, "watcher"))

    # A caller that already speaks the storage vocabulary wins over the
    # derived list, so a connector with a richer idea of roles can say so.
    given = row.get("participants")
    if isinstance(given, list) and given:
        return [p for p in (_person(g.get("email"), g.get("role") or "participant",
                                    name=g.get("name"), status=g.get("status"))
                            for g in given if isinstance(g, dict)) if p]

    return [p for p in people if p]


def _to_storage(spec: _Spec, row: dict[str, Any]) -> dict[str, Any]:
    """One incoming row, in the words of its shape, as table columns."""
    kind = spec.kind
    columns = _COLUMNS[kind]
    out: dict[str, Any] = {}
    attributes: dict[str, Any] = dict(row.get("attributes") or {})

    for key, value in row.items():
        if key in columns:
            out[columns[key]] = value
        elif key in _PASSTHROUGH:
            out[key] = value
        elif key in _ATTRIBUTES[kind]:
            if value is not None:
                attributes[key] = value
        # Anything else is a word this shape does not know. Dropping it
        # silently would lose data on a mapper typo, so it goes to the long
        # tail where it is at least still there to be found.
        elif key not in ("attributes", "participants", "to_emails", "cc_emails",
                         "attendees", "shared_with", "watcher_emails"):
            attributes[key] = value

    out["attributes"] = attributes

    # Who was involved, as indexed columns rather than roles inside a blob.
    # An event's flat address list is derived from its attendee objects so the
    # two can never disagree; a message's is generated by the database from
    # from/to/cc for the same reason.
    if kind == "event":
        out.setdefault("attendees", _attendee_objects(row))
        out["attendee_emails"] = _emails_of(out.get("attendees"))
    elif kind == "file":
        shared = row.get("shared_with_emails") or row.get("shared_with") or []
        out["shared_with_emails"] = _emails_of(shared)
    elif kind == "message":
        out["to_emails"] = _emails_of(row.get("to_emails"))
        out["cc_emails"] = _emails_of(row.get("cc_emails"))

    # The text column is NOT NULL so the tsvector never has to think about it.
    # An event with no description and a file with no readable text are both
    # ordinary, and they mean "no words", not "unknown".
    body_col = {"message": "body", "event": "description", "file": "content"}[kind]
    if out.get(body_col) is None:
        out[body_col] = ""

    # The time column is NOT NULL because it is what every window filters and
    # every result sorts on. A file Drive gave us no modified time for is the
    # one case that reaches here; now is the best we know, and a row with no
    # time at all would simply never appear in an answer.
    if out.get(spec.time_col) is None:
        out[spec.time_col] = utcnow()

    # An event's calendar both makes its id unique and is how a person groups
    # it, so it is the one field that lands in two places.
    if kind == "event":
        out.setdefault("scope_key", row.get("calendar_id") or "primary")

    out.setdefault("scope_key", "")
    return out


def _emails_of(value: Any) -> list[str]:
    """A flat list of addresses from whatever a connector handed over.

    Folded to lower case on the way in. The columns are `citext`, so matching
    is case-insensitive either way — but storing them folded means the arrays
    compare equal too, and a resync does not rewrite rows that did not change.
    """
    out: list[str] = []
    for entry in value or []:
        address = (
            entry.get("email") if isinstance(entry, dict) else entry
        )
        address = str(address or "").strip().lower()
        if address and address not in out:
            out.append(address)
    return out


def _attendee_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    """`[{email, name, response_status}]`, however the connector phrased it."""
    out: list[dict[str, Any]] = []
    for guest in row.get("attendees") or []:
        if not isinstance(guest, dict):
            address = str(guest or "").strip().lower()
            if address:
                out.append({"email": address})
            continue
        address = str(guest.get("email") or "").strip().lower()
        if not address:
            continue
        entry: dict[str, Any] = {"email": address}
        if guest.get("name"):
            entry["name"] = guest["name"]
        status = guest.get("response_status") or guest.get("status")
        if status:
            entry["response_status"] = status
        if guest.get("optional"):
            entry["optional"] = True
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Upserts
# --------------------------------------------------------------------------- #


def _dedupe(spec: _Spec, rows: Sequence[dict[str, Any]], user_id: str) -> list[dict]:
    """Collapse a batch on its conflict target — Postgres will not let one
    statement hit the same target twice — and fill in the columns we own."""
    now = utcnow()
    out: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        values = _to_storage(spec, row)
        values["user_id"] = user_id
        values.setdefault("id", new_id())
        values["updated_at"] = now
        values.setdefault("chunk_index", 0)
        # The shape, and the connector when the spec names one. A caller that
        # writes through `spec_for("message")` has to say which connector each
        # row came from; one that writes through `spec_for("gmail")` cannot get
        # it wrong.
        for column, default in spec.defaults.items():
            values.setdefault(column, default)
        values.setdefault("scope_key", "")

        # Every datetime, whatever the shape calls it. Naming the columns here
        # meant `sent_at` silently stopped being coerced the moment the shapes
        # got their own time columns, so a naive datetime from a connector
        # would have gone in unconverted and every window filter on it would
        # have been off by the offset.
        for column, value in list(values.items()):
            if isinstance(value, dt.datetime):
                values[column] = ensure_utc(value)
        values.pop("tsv", None)
        values.pop("participant_emails", None)  # generated
        key = tuple(values[c] for c in spec.conflict)
        out[key] = values
    return list(out.values())


def _upsert_set(spec: _Spec, stmt: Any) -> dict[str, Any]:
    """The SET clause of an upsert.

    The embedding rule: same ``content_hash`` means the text did not change, so
    keep the vector we already paid for unless a fresh one came with the row.
    Different hash means the old vector is stale, so take whatever arrived —
    usually NULL, which is exactly what queues the row for re-embedding.
    """
    model = spec.model
    values: dict[str, Any] = {c: getattr(stmt.excluded, c) for c in spec.mutable}
    values["updated_at"] = stmt.excluded.updated_at
    values["embedding"] = case(
        (
            model.content_hash == stmt.excluded.content_hash,
            func.coalesce(stmt.excluded.embedding, model.embedding),
        ),
        else_=stmt.excluded.embedding,
    )
    return values


async def _upsert(
    session: AsyncSession, spec: _Spec, user_id: str, rows: Sequence[dict[str, Any]]
) -> list[str]:
    if not rows:
        return []
    values = _dedupe(spec, rows, user_id)
    stmt = pg_insert(spec.model).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=list(spec.conflict), set_=_upsert_set(spec, stmt)
    ).returning(spec.model.id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_gmail(
    session: AsyncSession, user_id: str, rows: Sequence[dict[str, Any]]
) -> list[str]:
    """Upsert message chunks on (user_id, message_id, chunk_index)."""
    return await _upsert(session, GMAIL, user_id, rows)


async def upsert_gcal(
    session: AsyncSession, user_id: str, rows: Sequence[dict[str, Any]]
) -> list[str]:
    """Upsert events on (user_id, calendar_id, event_id).

    ``attendee_emails`` is generated from ``attendees``; do not pass it.
    """
    return await _upsert(session, GCAL, user_id, rows)


async def upsert_gdrive(
    session: AsyncSession, user_id: str, rows: Sequence[dict[str, Any]]
) -> list[str]:
    """Upsert file chunks on (user_id, file_id, chunk_index)."""
    return await _upsert(session, GDRIVE, user_id, rows)


async def upsert(
    session: AsyncSession, user_id: str, table: str, rows: Sequence[dict[str, Any]]
) -> list[str]:
    """Table-agnostic upsert, for the sync tasks that loop over services."""
    return await _upsert(session, spec_for(table), user_id, rows)


# --------------------------------------------------------------------------- #
# Embedding bookkeeping
# --------------------------------------------------------------------------- #


async def existing_hashes(
    session: AsyncSession, user_id: str, table: str, refs: Sequence[str]
) -> dict[tuple[str, int], uuid.UUID]:
    """``{(provider_ref, chunk_index): content_hash}`` for the refs given.

    An unchanged hash means the body is byte-for-byte what we embedded, so the
    row can be skipped entirely. ``chunk_index`` is always 0 for calendar.
    """
    if not refs:
        return {}
    spec = spec_for(table)
    model = spec.model
    result = await session.execute(
        select(model.source_id, model.chunk_index, model.content_hash).where(
            model.user_id == user_id,
            model.source_id.in_(list(refs)),
            *_scope(spec),
        )
    )
    return {(row[0], int(row[1])): row[2] for row in result.all()}


async def rows_needing_embedding(
    session: AsyncSession, user_id: str | None, table: str, *, limit: int = 200
) -> list[Any]:
    """Rows the embed task still has to vectorise.

    ``user_id=None`` scans every user — for the embed queue's sweep only.
    """
    spec = spec_for(table)
    model = spec.model
    stmt = select(model).where(model.embedding.is_(None), *_scope(spec))
    if user_id is not None:
        stmt = stmt.where(model.user_id == user_id)
    stmt = stmt.order_by(model.updated_at.asc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def set_embeddings(
    session: AsyncSession,
    user_id: str,
    table: str,
    pairs: Sequence[tuple[str, Sequence[float]]],
) -> int:
    """Write vectors back. One round trip for the whole batch."""
    if not pairs:
        return 0
    spec = spec_for(table)
    model = spec.model
    # Against the Table, not the mapped class, and deliberately.
    #
    # An executemany UPDATE aimed at the ORM entity is taken as a bulk update
    # by primary key, which wants `id` in every parameter row and then tries to
    # reconcile the identity map — neither of which works alongside the WHERE
    # criteria this single-table design needs (the tenant guard and the shape
    # scope). Core emits exactly the statement written here, once, for the
    # whole batch. Nothing in this session reads these rows again, so there is
    # no ORM state to keep in step.
    table = model.__table__
    stmt = (
        update(table)
        .where(
            table.c.id == bindparam("row_id"),
            table.c.user_id == user_id,
            *_scope(spec),
        )
        .values(embedding=bindparam("vec", type_=Vector(EMBEDDING_DIM)))
    )
    await session.execute(
        stmt,
        [{"row_id": row_id, "vec": list(vector)} for row_id, vector in pairs],
    )
    return len(pairs)


# --------------------------------------------------------------------------- #
# Deletes and pruning
# --------------------------------------------------------------------------- #


async def delete_by_refs(
    session: AsyncSession, user_id: str, table: str, refs: Sequence[str]
) -> int:
    """Drop every chunk of the given messages / events / files.

    This is what a Gmail history delete or a Calendar cancellation maps to.
    """
    if not refs:
        return 0
    spec = spec_for(table)
    model = spec.model
    result = await session.execute(
        delete(model).where(
            model.user_id == user_id,
            model.source_id.in_(list(refs)),
            *_scope(spec),
        )
    )
    return int(result.rowcount or 0)


async def delete_extra_chunks(
    session: AsyncSession, user_id: str, table: str, ref: str, keep: int
) -> int:
    """Drop chunks left over when a re-chunked document got shorter."""
    spec = spec_for(table)
    if not spec.chunked:
        return 0
    model = spec.model
    result = await session.execute(
        delete(model).where(
            model.user_id == user_id,
            model.source_id == ref,
            model.chunk_index >= keep,
            *_scope(spec),
        )
    )
    return int(result.rowcount or 0)


async def prune(
    session: AsyncSession,
    user_id: str | None,
    table: str,
    older_than: dt.datetime,
    *,
    limit: int = 5000,
) -> int:
    """Drop mirror rows older than a cutoff. ``maintenance.prune_sync`` only;
    ``user_id=None`` prunes every user."""
    spec = spec_for(table)
    model = spec.model
    time_col = getattr(model, spec.time_col)
    doomed = (
        select(model.id)
        .where(time_col < ensure_utc(older_than), *_scope(spec))
        .limit(limit)
    )
    if user_id is not None:
        doomed = doomed.where(model.user_id == user_id)
    result = await session.execute(delete(model).where(model.id.in_(doomed)))
    return int(result.rowcount or 0)


async def purge_user(session: AsyncSession, user_id: str) -> dict[str, int]:
    """Wipe the whole mirror for one person. Account disconnect.

    Counted per connector before the delete, because the caller wants the
    breakdown and it is gone once the rows are.
    """
    before = await counts(session, user_id)
    for model in (SyncMessage, SyncEvent, SyncFile):
        await session.execute(delete(model).where(model.user_id == user_id))
    return before


async def counts(session: AsyncSession, user_id: str) -> dict[str, int]:
    """Rows mirrored per connector — the figure on /sync/status.

    Every known connector appears, including the ones with nothing in them:
    a missing key and a zero mean different things to the caller, and only one
    of them is true here.
    """
    out: dict[str, int] = {key: 0 for key in CONNECTOR_SPECS}
    for model in (SyncMessage, SyncEvent, SyncFile):
        result = await session.execute(
            select(model.connector, func.count())
            .where(model.user_id == user_id)
            .group_by(model.connector)
        )
        for connector, total in result.all():
            out[str(connector)] = out.get(str(connector), 0) + int(total)
    return out


# --------------------------------------------------------------------------- #
# Hybrid search
# --------------------------------------------------------------------------- #


def _vector_literal(embedding: Sequence[float]) -> str:
    """pgvector's text form. Bound as a string and cast in SQL, which keeps the
    driver out of the business of knowing the type."""
    return "[" + ",".join(f"{float(x):.7g}" for x in embedding) + "]"


def _build_where(
    spec: _Spec, alias: str, filters: dict[str, Any], params: dict[str, Any]
) -> str:
    """Turn the filter dict into SQL. Unknown keys are refused rather than
    silently dropped — a typo in a filter must not widen a search.

    The shape needs no predicate: it is the table. Under one table this had to
    prepend `kind = ...`, and a query that forgot it read events out of a mail
    search — returning rows rather than failing, which is worse. Only the
    connector narrowing remains, and only when the spec names one.
    """
    fragments: list[str] = []
    if spec.connector is not None:
        fragments.append(f"{alias}.connector = :spec_connector")
        params["spec_connector"] = spec.connector
    flags = {"not_cancelled"}
    for key, value in filters.items():
        if value is None:
            continue
        if key in flags and not value:
            continue
        template = spec.filters.get(key)
        if template is None:
            raise AppError(
                "VALIDATION_ERROR",
                f"{key!r} is not a filter on {spec.key}.",
                http=422,
                details={"filter": key, "known": sorted(spec.filters)},
            )
        fragments.append(template.format(t=alias, time_col=spec.time_col))
        if isinstance(value, dt.datetime):
            params[f"f_{key}"] = ensure_utc(value)
        elif isinstance(value, (list, tuple, set)):
            params[f"f_{key}"] = list(value)
        elif key in flags:
            pass  # a flag, its SQL fragment carries no parameter
        else:
            params[f"f_{key}"] = value
    return "".join(f" AND {f}" for f in fragments)


async def hybrid_search(
    session: AsyncSession,
    user_id: str,
    table: str,
    embedding: Sequence[float] | None,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
    *,
    candidates: int | None = None,
    w_vec: float = 0.6,
    w_lex: float = 0.4,
    ef_search: int | None = None,
) -> list[dict[str, Any]]:
    """One ANN pass and one full-text pass over a mirror table, fused.

    ``filters`` is the whitelist in each table's spec, plus ``text`` (or
    ``query``) which is the lexical string. Both passes are pre-filtered on
    ``user_id``; the HNSW index holds up under that prefilter, which is why it
    is HNSW and not ivfflat.

    Returns one dict per message / event / file — chunked tables collapse to
    their best-scoring chunk — carrying every field in the spec plus ``cos``,
    ``lex`` and the fused ``score``. Callers that want their own weighting have
    the two components; ``score`` is only the ordering this query used.
    """
    spec = spec_for(table)
    work = dict(filters or {})
    query_text = work.pop("text", None) or work.pop("query", None)
    query_text = (query_text or "").strip() or None

    has_vec = embedding is not None and len(embedding) > 0
    if not has_vec and query_text is None:
        return []

    cand = candidates or max(limit * 5, 50)
    params: dict[str, Any] = {
        "user_id": user_id,
        "limit": limit,
        "cand": cand,
        "w_vec": float(w_vec),
        "w_lex": float(w_lex),
    }
    where = _build_where(spec, spec.table, work, params)
    t = spec.table

    if has_vec:
        params["emb"] = _vector_literal(embedding or [])
        vec_cte = f"""
            SELECT {t}.id AS id,
                   (1 - ({t}.embedding <=> CAST(:emb AS vector)))::float8 AS cos
            FROM {t}
            WHERE {t}.user_id = :user_id AND {t}.embedding IS NOT NULL{where}
            ORDER BY {t}.embedding <=> CAST(:emb AS vector)
            LIMIT :cand
        """
    else:
        vec_cte = "SELECT CAST(NULL AS CHAR(21)) AS id, CAST(0 AS float8) AS cos WHERE false"

    if query_text is not None:
        params["qtext"] = query_text
        lex_cte = f"""
            SELECT {t}.id AS id,
                   ts_rank_cd({t}.tsv, websearch_to_tsquery('english', :qtext))::float8 AS lex
            FROM {t}
            WHERE {t}.user_id = :user_id
              AND {t}.tsv @@ websearch_to_tsquery('english', :qtext){where}
            ORDER BY 2 DESC
            LIMIT :cand
        """
    else:
        lex_cte = "SELECT CAST(NULL AS CHAR(21)) AS id, CAST(0 AS float8) AS lex WHERE false"

    fields = ", ".join(c.format(t="s") for c in spec.fields)
    collapse = ", ".join(spec.collapse)

    sql = f"""
        WITH vec AS ({vec_cte}),
             lex AS ({lex_cte}),
             merged AS (
                 SELECT coalesce(v.id, l.id) AS id,
                        coalesce(v.cos, 0)::float8 AS cos,
                        coalesce(l.lex, 0)::float8 AS lex
                 FROM vec v FULL OUTER JOIN lex l ON l.id = v.id
             ),
             hits AS (
                 SELECT {fields}, m.cos AS cos, m.lex AS lex,
                        (:w_vec * m.cos + :w_lex * least(m.lex, 1.0))::float8 AS score
                 FROM merged m JOIN {t} s ON s.id = m.id
             ),
             best AS (
                 SELECT DISTINCT ON ({collapse}) *
                 FROM hits
                 ORDER BY {collapse}, score DESC
             )
        SELECT * FROM best ORDER BY score DESC, {spec.collapse[0]} LIMIT :limit
    """

    if ef_search is not None:
        # SET LOCAL is transaction-scoped, so it cannot leak to another query.
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

    result = await session.execute(text(sql), params)
    return [dict(row) for row in result.mappings().all()]


async def get_by_ref(
    session: AsyncSession, user_id: str, table: str, ref: str
) -> list[Any]:
    """Every chunk of one message / event / file / task, in chunk order.

    Projected, not raw rows. A caller that fetched by id and one that searched
    must get the same keys back, or every op would need to know which of the
    two it was holding.
    """
    spec = spec_for(table)
    params: dict[str, Any] = {"user_id": user_id, "f_ref": ref}
    where = _build_where(spec, "t", {}, params)
    columns = ", ".join(c.format(t="t") for c in spec.fields)
    sql = text(
        f"SELECT {columns} FROM {spec.table} AS t "
        f"WHERE t.user_id = :user_id AND t.source_id = :f_ref{where} "
        f"ORDER BY t.chunk_index ASC"
    )
    rows = (await session.execute(sql, params)).mappings().all()
    return [dict(r) for r in rows]


async def list_filtered(
    session: AsyncSession,
    user_id: str,
    table: str,
    filters: dict[str, Any] | None = None,
    *,
    limit: int = 50,
    newest_first: bool | None = None,
) -> list[dict[str, Any]]:
    """Rows matching the filters, unscored, in the corpus's own time order.

    The read behind an enumeration — "what is on my calendar next week", "PDFs
    from last month". There is nothing to rank by, so nothing is ranked; the
    filter IS the question. Uses the same whitelist as
    :func:`hybrid_search`, so a filter that works for a search works here and a
    typo is refused in both.

    Calendars read forward (soonest first); mail and files read backward
    (newest first). Both are what a person means by "my next week" versus "my
    recent mail".
    """
    spec = spec_for(table)
    params: dict[str, Any] = {"user_id": user_id, "limit": int(limit)}
    where = _build_where(spec, "t", dict(filters or {}), params)
    if newest_first is None:
        # Calendars read forward — "my next week" means the soonest first.
        # Everything else reads backward, which is what "recent" means.
        newest_first = spec.kind != "event"
    direction = "DESC" if newest_first else "ASC"

    columns = ", ".join(c.format(t="t") for c in spec.fields)
    sql = text(
        f"SELECT {columns} FROM {spec.table} AS t "
        f"WHERE t.user_id = :user_id{where} "
        f"ORDER BY t.{spec.time_col} {direction} NULLS LAST "
        f"LIMIT :limit"
    )
    rows = (await session.execute(sql, params)).mappings().all()
    return [dict(r) for r in rows]


async def list_window(
    session: AsyncSession,
    user_id: str,
    table: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[Any]:
    """Plain time-window read, no scoring — "what is on my calendar next week".

    Windows are half-open, [start, end).

    A window is a filtered list with the window as two of the filters, so it
    goes through the same whitelist and the same projection. It used to reach
    for columns by name on the model, which under one shared table meant a
    calendar window could return mail, and a filter written in the shape's own
    words — ``calendar_id`` — no longer matched a column at all.
    """
    work = dict(filters or {})
    work["since"] = ensure_utc(start)
    work["until"] = ensure_utc(end)
    return await list_filtered(
        session, user_id, table, work, limit=limit, newest_first=False
    )


__all__ = [
    "CONNECTOR_SPECS",
    "GCAL",
    "GDRIVE",
    "GMAIL",
    "SPECS",
    "spec_for",
    "upsert_gmail",
    "upsert_gcal",
    "upsert_gdrive",
    "upsert",
    "existing_hashes",
    "rows_needing_embedding",
    "set_embeddings",
    "delete_by_refs",
    "delete_extra_chunks",
    "prune",
    "purge_user",
    "counts",
    "hybrid_search",
    "get_by_ref",
    "list_filtered",
    "list_window",
]
