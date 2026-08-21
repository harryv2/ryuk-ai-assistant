"""The ops that belong to no service.

Three of them are the load-bearing ones:

* `ask.user` is why pausing costs nothing. A question is a step, so a run that
  needs one goes to `awaiting_input` with its plan already on disk, and the
  answer re-enters the same DAG with one node filled in — **zero LLM calls**.
* `data.filter` is a deterministic predicate over a previous step's result. It
  exists so that "only the ones with attachments" never becomes a model call.
* `llm.extract` tries `app.search.extractors` first and only spends a call when
  the patterns miss. That branch is the one honest extra cost in the design and
  it is measured rather than hidden: the result says which path answered.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import AppError
from app.db.repositories import actions as actions_repo
from app.db.repositories import entities as entities_repo
from app.ops.base import (
    FLOOR_READ,
    InputRequest,
    Op,
    OpContext,
    OpResult,
    ambiguity_test,
    as_bool,
    as_list,
    choice_input,
    excerpt_of,
    extractors_module,
    hybrid_many,
    is_exact,
    iso,
    jsonable,
    parse_dt,
    run_extractors,
    shape_hit,
    trim_for_llm,
    window_bounds,
)

# --------------------------------------------------------------------------- #
# search.all
# --------------------------------------------------------------------------- #

_SERVICES = ("gmail", "gcal", "gdrive")
_SERVICE_ALIASES = {"drive": "gdrive", "calendar": "gcal", "mail": "gmail", "email": "gmail"}


class SearchAllArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    services: list[str] = Field(default_factory=lambda: list(_SERVICES))
    window: dict[str, Any] | list[Any] | None = None
    limit: int = Field(default=5, ge=1, le=25)
    body: bool = False

    @model_validator(mode="after")
    def _normalise(self) -> "SearchAllArgs":
        names = []
        for service in as_list(self.services) or list(_SERVICES):
            key = _SERVICE_ALIASES.get(str(service).lower(), str(service).lower())
            if key not in _SERVICES:
                raise ValueError(f"there is no service called {service!r}")
            if key not in names:
                names.append(key)
        self.services = names
        return self


class SearchAll(Op):
    """One query across all three mirrors, in parallel.

    The same fan-out the probe does, available as a step for when the planner
    wants it again with a different query — three Postgres round trips started
    in the same event-loop tick, not three sequential searches.
    """

    name = "search.all"
    args_model = SearchAllArgs
    output_fields = ["gmail", "gcal", "gdrive", "top", "counts", "total"]
    is_local = True
    timeout_s = 4.0
    summary = "search mail, calendar and Drive at once"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        filters: dict[str, Any] = {}
        if parsed.window is not None:
            lo, hi = window_bounds(parsed.window)
            filters = {"since": lo, "until": hi}

        gathered = await hybrid_many(
            ctx, parsed.services, query=parsed.query, filters=filters, limit=parsed.limit
        )

        data: dict[str, Any] = {c: [] for c in _SERVICES}
        for corpus, rows in gathered.items():
            data[corpus] = [
                shape_hit(
                    corpus,
                    row,
                    query=parsed.query,
                    filters=filters,
                    body=parsed.body,
                    extract=corpus == "gmail",
                )
                for row in rows
            ]

        everything = [
            {**hit, "service": corpus} for corpus in _SERVICES for hit in data[corpus]
        ]
        everything.sort(key=lambda h: float(h.get("cn") or 0.0), reverse=True)
        data["top"] = [h for h in everything if float(h.get("cn") or 0) >= FLOOR_READ or is_exact(h.get("evidence"))][:5]
        data["counts"] = {c: len(data[c]) for c in _SERVICES}
        data["total"] = sum(data["counts"].values())
        missed = [c for c in parsed.services if c not in gathered]
        if missed:
            data["degraded"] = missed
        return OpResult(data=jsonable(data))

    def progress_label(self, args: dict) -> str:
        return f"Searching everything for “{excerpt_of((args or {}).get('query'), 40)}”"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        return trim_for_llm(
            {"top": data.get("top", [])[:6], "counts": data.get("counts"), "total": data.get("total")},
            budget,
        )


# --------------------------------------------------------------------------- #
# resolve.person
# --------------------------------------------------------------------------- #

_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")

#: Local parts that mean a machine or a shared mailbox, not a person. "John at
#: Taskade <updates@taskade.com>" matches the name "John" perfectly and mails
#: often — and putting it on a calendar invitation or an attendee filter is
#: always wrong. Demoted, never excluded: a card can still offer it, it just
#: cannot *win* a "which person" question on its own.
_MACHINE_LOCALS = frozenset(
    {
        "alert", "alerts", "automated", "billing", "bot", "bounce", "bounces",
        "digest", "donotreply", "do-not-reply", "hello", "info", "mailer",
        "mailer-daemon", "marketing", "news", "newsletter", "no-reply",
        "no_reply", "noreply", "notification", "notifications", "notify",
        "promo", "promotions", "receipts", "robot", "support", "team",
        "update", "updates",
    }
)


def _is_machine_address(email: str) -> bool:
    local = email.split("@")[0].lower()
    return local in _MACHINE_LOCALS or local.startswith(("noreply", "no-reply", "no_reply", "donotreply"))


class ResolvePersonArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    limit: int = Field(default=5, ge=1, le=20)
    expect: str = "one"


class ResolvePerson(Op):
    """"John" as an email address, or a question about which John.

    Evidence comes from the mirror: who writes to this mailbox, who is on these
    invitations, and who the conversation has already named. Frequency breaks
    near-ties, because the John you email weekly is more probably the John you
    mean than the John you emailed once in March — but only frequency, never a
    guess: two candidates inside `MARGIN` produce a choice card.
    """

    name = "resolve.person"
    args_model = ResolvePersonArgs
    output_fields = ["email", "display_name", "candidates", "resolved"]
    is_local = True
    timeout_s = 3.0
    summary = "turn a first name into an email address, or ask which"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        tokens = {t.lower() for t in _NAME_TOKEN.findall(parsed.name)}
        if not tokens:
            raise AppError("VALIDATION_ERROR", "That is not a name.", http=422)

        people: dict[str, dict[str, Any]] = {}

        def note(email: str | None, display: str | None, source: str, weight: float) -> None:
            address = (email or "").strip().lower()
            if not address or "@" not in address:
                return
            entry = people.setdefault(
                address,
                {"email": address, "display_name": display or "", "seen": 0, "weight": 0.0, "sources": []},
            )
            entry["seen"] += 1
            entry["weight"] += weight
            if display and len(display) > len(entry["display_name"]):
                entry["display_name"] = display
            if source not in entry["sources"]:
                entry["sources"].append(source)

        gathered = await hybrid_many(ctx, ("gmail", "gcal"), query=parsed.name, filters={}, limit=40)
        for row in gathered.get("gmail", []):
            note(row.get("from_email"), row.get("from_name"), "mail_from", 1.0)
            for address in as_list(row.get("to_emails")):
                note(str(address), None, "mail_to", 0.4)
        for row in gathered.get("gcal", []):
            for attendee in row.get("attendees") or []:
                if isinstance(attendee, dict):
                    note(attendee.get("email"), attendee.get("name"), "calendar", 0.8)
            note(row.get("organizer_email"), None, "calendar", 0.6)

        for entity in await entities_repo.list_entities(
            ctx.session, ctx.user_id, ctx.conversation_id, entity_type="person", limit=20
        ):
            note(getattr(entity, "entity_ref", None), getattr(entity, "label", None), "conversation", 2.0)

        candidates: list[dict] = []
        for entry in people.values():
            match = _name_match(tokens, entry["email"], entry["display_name"])
            if match <= 0.0:
                continue
            machine = _is_machine_address(entry["email"])
            if machine:
                # A newsletter that signs itself "John" still matches the
                # letters; it does not answer the question.
                match *= 0.3
            frequency = min(entry["weight"], 10.0) / 10.0
            candidates.append(
                {
                    "email": entry["email"],
                    "display_name": entry["display_name"] or entry["email"].split("@")[0],
                    "seen": entry["seen"],
                    "sources": entry["sources"],
                    "cn": round(min(1.0, 0.75 * match + 0.25 * frequency), 4),
                    "evidence": [] if machine else (["EXACT_SENDER"] if match >= 0.99 else []),
                    "ref": entry["email"],
                    "label": f"{entry['display_name'] or entry['email']} ({entry['email']})",
                    "meta": {"seen_in": ", ".join(entry["sources"]), "mentions": entry["seen"]},
                }
            )
        candidates.sort(key=lambda c: (c["cn"], c["seen"]), reverse=True)
        candidates = candidates[: parsed.limit]

        reason = ambiguity_test(candidates) if parsed.expect == "one" else None
        if reason == "margin":
            return OpResult(
                data={"candidates": candidates, "resolved": False, "reason": reason},
                needs_input=choice_input(
                    f"Which {parsed.name.strip()} did you mean?",
                    candidates,
                    id_field="email",
                    help_text="More than one person here answers to that name.",
                ),
            )
        if reason == "absent" or not candidates:
            return OpResult(
                data={"candidates": candidates, "resolved": False, "reason": "absent", "name": parsed.name},
                needs_input=InputRequest(
                    kind="text",
                    question=f"What is {parsed.name.strip()}'s email address?",
                    value_schema={"type": "string", "format": "email", "minLength": 5},
                ),
            )

        top = candidates[0]
        return OpResult(
            data={
                "email": top["email"],
                "display_name": top["display_name"],
                "resolved": True,
                "cn": top["cn"],
                "candidates": candidates[1:],
            }
        )

    def progress_label(self, args: dict) -> str:
        return f"Working out who {(args or {}).get('name', 'that')} is"


def _name_match(tokens: set[str], email: str, display: str) -> float:
    """How well a name matches an address. 1.0 is every token accounted for."""
    local = email.split("@")[0]
    haystack = {t.lower() for t in _NAME_TOKEN.findall(f"{display} {local.replace('.', ' ').replace('_', ' ')}")}
    if not haystack:
        return 0.0
    hits = sum(1 for token in tokens if token in haystack)
    if hits == 0:
        # A prefix still counts: "jkowalski" for "John Kowalski".
        hits = sum(1 for token in tokens if any(h.startswith(token[:4]) for h in haystack if len(token) >= 4))
        if hits == 0:
            return 0.0
        return 0.5 * hits / len(tokens)
    return hits / len(tokens)


# --------------------------------------------------------------------------- #
# resolve.reference
# --------------------------------------------------------------------------- #


class ResolveReferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase: str
    entity_type: str | None = None
    limit: int = Field(default=5, ge=1, le=20)
    expect: str = "one"

    @model_validator(mode="after")
    def _kind(self) -> "ResolveReferenceArgs":
        if self.entity_type is not None and self.entity_type not in ("email", "event", "file", "person"):
            raise ValueError("entity_type is email, event, file or person")
        return self


class ResolveReference(Op):
    """"That email", "the proposal", "the meeting we just moved".

    Resolved against `conversation_entities` — the twenty-odd things this
    conversation has already put on screen — rather than by searching the
    mailbox again. What was shown a minute ago is what "that" means.
    """

    name = "resolve.reference"
    args_model = ResolveReferenceArgs
    output_fields = ["entity_type", "entity_ref", "label", "candidates", "resolved"]
    is_local = True
    timeout_s = 2.0
    summary = "resolve “that email” against what this chat has already shown"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        term = " ".join(w for w in parsed.phrase.split() if w.lower() not in _DEICTIC).strip()

        rows = []
        if term:
            rows = await entities_repo.search_entities(
                ctx.session,
                ctx.user_id,
                ctx.conversation_id,
                term,
                entity_type=parsed.entity_type,
                limit=parsed.limit,
            )
        if not rows:
            rows = await entities_repo.list_entities(
                ctx.session,
                ctx.user_id,
                ctx.conversation_id,
                entity_type=parsed.entity_type,
                limit=parsed.limit,
            )

        now = ctx.now
        candidates: list[dict] = []
        for index, row in enumerate(rows):
            last_seen = parse_dt(getattr(row, "last_seen_at", None)) or now
            age_minutes = max(0.0, (now - last_seen).total_seconds() / 60.0)
            # Recency is the whole signal here: position in the list decides,
            # and the score only has to order it. The most recent match is the
            # referent unless something else matched the words better.
            recency = 1.0 / (1.0 + age_minutes / 30.0)
            matched = bool(term) and term.lower() in str(getattr(row, "label", "")).lower()
            candidates.append(
                {
                    "entity_type": getattr(row, "entity_type", None),
                    "entity_ref": getattr(row, "entity_ref", None),
                    "ref": getattr(row, "entity_ref", None),
                    "label": getattr(row, "label", None),
                    "meta": jsonable(getattr(row, "meta", None) or {}),
                    "last_seen_at": iso(last_seen),
                    "cn": round(min(1.0, (0.55 if matched else 0.35) + 0.4 * recency - 0.02 * index), 4),
                    "evidence": ["EXACT_ID"] if matched and index == 0 else [],
                }
            )
        candidates.sort(key=lambda c: c["cn"], reverse=True)

        reason = ambiguity_test(candidates) if parsed.expect == "one" else None
        if reason == "margin":
            return OpResult(
                data={"candidates": candidates, "resolved": False, "reason": reason},
                needs_input=choice_input(
                    f"Which one do you mean by “{parsed.phrase.strip()}”?",
                    candidates,
                    id_field="entity_ref",
                ),
            )
        if reason == "absent" or not candidates:
            return OpResult(
                data={"candidates": [], "resolved": False, "reason": "absent", "phrase": parsed.phrase},
                needs_replan=True,
                replan_reason="nothing in this conversation matches that reference",
            )

        top = candidates[0]
        return OpResult(data={**top, "resolved": True, "candidates": candidates[1:]})

    def progress_label(self, args: dict) -> str:
        return f"Working out what “{excerpt_of((args or {}).get('phrase'), 30)}” refers to"


_DEICTIC = {"that", "the", "this", "those", "these", "it", "one", "my", "our", "a", "an"}


# --------------------------------------------------------------------------- #
# ask.user
# --------------------------------------------------------------------------- #

_KINDS = ("confirm", "choice", "multi_choice", "text", "form", "date_range")


#: Field names a person cannot TYPE.
#:
#: Nobody knows their Google event id, so a text box asking for one is the
#: planner handing its job to the reader — the one person in the loop with no
#: way to look it up.
#:
#: The same name as a `choice` is fine, and is in fact the shape we want: the
#: options carry real labels ("1:1 with John Okafor"), the person taps one, and
#: the id travels underneath. The identifier is not the problem; asking someone
#: to transcribe it is.
#:
#: Enforced here rather than asked for in the prompt, because a prompt is a
#: request and this is a rule: a plan that breaks it is rejected and repaired.
_UNASKABLE: Final[frozenset[str]] = frozenset(
    {
        "event_id", "message_id", "file_id", "thread_id", "draft_id",
        "calendar_id", "folder_id", "action_id", "run_id", "node_id",
        "id", "etag", "external_id", "history_id", "page_token",
    }
)


class AskUserArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    question: str
    value_schema: dict[str, Any] | None = None
    help_text: str | None = None
    fields: list[dict[str, Any]] | None = None
    options: list[dict[str, Any]] | None = None
    blocking: bool = True

    @model_validator(mode="after")
    def _check(self) -> "AskUserArgs":
        if self.kind not in _KINDS:
            raise ValueError(f"kind is one of {', '.join(_KINDS)}")
        if self.kind in ("choice", "multi_choice") and not self.options:
            raise ValueError(f"a {self.kind} needs options")
        if self.kind == "form" and not self.fields:
            raise ValueError("a form needs fields")

        typed_out = sorted(
            str(f.get("name") or "").strip().lower()
            for f in (self.fields or [])
            if str(f.get("name") or "").strip().lower() in _UNASKABLE
            and str(f.get("kind") or "text").lower() not in ("choice", "multi_choice")
        )
        if typed_out:
            raise ValueError(
                f"nobody can type {', '.join(typed_out)} — these are internal "
                "identifiers. Offer them as a `choice` with real labels, or search "
                "for the row first; ask by hand only for what is in their head alone"
            )
        return self


class AskUser(Op):
    """Raise a card and stop.

    This op does nothing else, and that is the point. The run pauses, the plan
    stays on disk, and answering resumes the same DAG at this node with the
    value bound — no replay, no second planner call.
    """

    name = "ask.user"
    args_model = AskUserArgs
    output_fields = ["value", "raised", "kind"]
    is_local = True
    timeout_s = 1.0
    max_attempts = 1
    summary = "ask the person a question and pause the run"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        schema = parsed.value_schema or _schema_for(parsed)
        if parsed.kind == "form" and parsed.value_schema and parsed.fields:
            # The planner may write its own schema and its own fields; when a
            # choice field's ids are missing from the schema, fold them in.
            # Without the enum the row would accept any string a client sends,
            # and the write behind this card would target it.
            schema = _merge_field_enums(schema, parsed.fields)
        return OpResult(
            data={"raised": True, "kind": parsed.kind, "question": parsed.question},
            needs_input=InputRequest(
                kind=parsed.kind,
                question=parsed.question,
                value_schema=jsonable({**schema, **({"description": parsed.help_text} if parsed.help_text else {})}),
                options=jsonable(parsed.options) if parsed.options else None,
                fields=jsonable(parsed.fields) if parsed.fields else None,
                blocking=parsed.blocking,
            ),
        )

    def progress_label(self, args: dict) -> str:
        return "Asking you something"


def _merge_field_enums(schema: dict[str, Any], fields: list[dict[str, Any]]) -> dict[str, Any]:
    """The supplied schema, with each choice field's ids as its enum."""
    out = {**schema, "properties": dict(schema.get("properties") or {})}
    for spec in fields:
        name = str(spec.get("name") or "")
        options = spec.get("options")
        if not name or str(spec.get("kind") or "") not in ("choice", "multi_choice"):
            continue
        if not isinstance(options, list) or not options:
            continue
        ids = [str(o.get("id")) for o in options if isinstance(o, dict) and o.get("id") is not None]
        if not ids:
            continue
        prop = dict(out["properties"].get(name) or {"type": "string"})
        if "enum" not in prop and "oneOf" not in prop:
            prop["enum"] = ids
        out["properties"][name] = prop
    return out


def _schema_for(parsed: AskUserArgs) -> dict[str, Any]:
    """A value schema when the planner did not write one.

    `pending_inputs.value_schema` is the validation authority, so a card
    without one would accept anything a client sent.
    """
    if parsed.kind == "confirm":
        # `approve`, no d — the one spelling every confirm in the product uses.
        # The staged-write cards (runner) and the client's buttons both say
        # `{"approve": bool}`; a second spelling here made planner-authored
        # confirms unanswerable from the UI.
        return {"type": "object", "properties": {"approve": {"type": "boolean"}}, "required": ["approve"]}
    if parsed.kind == "choice":
        return {"type": "string", "enum": [str(o.get("id")) for o in parsed.options or []]}
    if parsed.kind == "multi_choice":
        return {
            "type": "array",
            "items": {"type": "string", "enum": [str(o.get("id")) for o in parsed.options or []]},
            "minItems": 1,
        }
    if parsed.kind == "date_range":
        return {
            "type": "object",
            "properties": {"start": {"type": "string", "format": "date-time"},
                           "end": {"type": "string", "format": "date-time"}},
            "required": ["start", "end"],
        }
    if parsed.kind == "form":
        properties: dict[str, Any] = {}
        required: list[str] = []
        for spec in parsed.fields or []:
            field_name = str(spec.get("name") or "")
            if not field_name:
                continue
            kind = str(spec.get("kind") or "text")
            if kind == "choice":
                options = spec.get("options")
                enum = [str(o.get("id")) for o in options] if isinstance(options, list) else None
                properties[field_name] = {"type": "string", **({"enum": enum} if enum else {})}
            elif kind == "boolean":
                properties[field_name] = {"type": "boolean"}
            elif kind in ("datetime", "date_time", "timestamp"):
                # A moment in time is not free text. Typing it as one lets the
                # client offer a picker and lets the value be validated, rather
                # than hoping "next tues 3ish" parses on the far side.
                properties[field_name] = {"type": "string", "format": "date-time"}
            elif kind == "date":
                properties[field_name] = {"type": "string", "format": "date"}
            elif kind in ("number", "integer"):
                properties[field_name] = {"type": "number"}
            elif kind == "email":
                properties[field_name] = {"type": "string", "format": "email"}
            else:
                properties[field_name] = {"type": "string", "minLength": 1}
            if spec.get("required", True):
                required.append(field_name)
        return {"type": "object", "properties": properties, "required": required}
    return {"type": "string", "minLength": 1}


# --------------------------------------------------------------------------- #
# data.filter
# --------------------------------------------------------------------------- #

_PREDICATES = (
    "eq", "ne", "in", "not_in", "contains", "icontains", "starts_with",
    "exists", "empty", "gt", "gte", "lt", "lte", "before", "after",
    "within", "matches",
)


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    test: str = "eq"
    value: Any = None

    @model_validator(mode="after")
    def _known(self) -> "Condition":
        if self.test not in _PREDICATES:
            raise ValueError(f"{self.test!r} is not a test; use one of {', '.join(_PREDICATES)}")
        return self


class DataFilterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Any] = Field(default_factory=list)
    where: list[Condition] = Field(default_factory=list)
    mode: str = "all"
    order_by: str | None = None
    order_dir: str = "desc"
    limit: int | None = Field(default=None, ge=1, le=200)
    project: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, values: Any) -> Any:
        """A `{{step}}` reference resolves to the whole result, so accept the
        result and take its list."""
        if not isinstance(values, dict):
            return values
        out = dict(values)
        items = out.get("items")
        if isinstance(items, dict):
            for key in ("hits", "emails", "events", "files", "items", "slots", "conflicts"):
                if isinstance(items.get(key), list):
                    out["items"] = items[key]
                    break
        if isinstance(out.get("where"), dict):
            out["where"] = [out["where"]]
        return out

    @model_validator(mode="after")
    def _mode(self) -> "DataFilterArgs":
        if self.mode not in ("all", "any"):
            raise ValueError("mode is all or any")
        return self


class DataFilter(Op):
    """A predicate over a previous step's result. No model, no network.

    "Only the ones with attachments", "just next week's", "drop the cancelled
    ones" — all of it is a comparison, and a comparison should cost nothing.
    """

    name = "data.filter"
    args_model = DataFilterArgs
    output_fields = ["items", "count", "dropped", "kept_ratio"]
    is_local = True
    timeout_s = 2.0
    summary = "filter, sort and cut a previous step's items"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        rows = [r for r in parsed.items if isinstance(r, dict)]
        kept = [r for r in rows if _matches(r, parsed.where, parsed.mode, ctx)]

        if parsed.order_by:
            reverse = parsed.order_dir.lower() != "asc"
            kept.sort(key=lambda r: _sortable(_dig(r, parsed.order_by)), reverse=reverse)
        if parsed.limit is not None:
            kept = kept[: parsed.limit]
        if parsed.project:
            keep = set(parsed.project)
            kept = [{k: v for k, v in r.items() if k in keep} for r in kept]

        return OpResult(
            data=jsonable(
                {
                    "items": kept,
                    "count": len(kept),
                    "dropped": len(rows) - len(kept),
                    "kept_ratio": round(len(kept) / len(rows), 3) if rows else 0.0,
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        return "Narrowing that down"


def _dig(row: dict, path: str) -> Any:
    """``"extracted.pnr"`` out of a nested dict."""
    value: Any = row
    for part in str(path).split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            value = value[index] if index < len(value) else None
        else:
            return None
    return value


def _sortable(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    if isinstance(value, str):
        parsed = None
        if len(value) >= 10 and value[4:5] == "-" and value[7:8] == "-":
            try:
                parsed = parse_dt(value)
            except AppError:
                parsed = None
        return (0, parsed.timestamp() if parsed else value.lower())
    if isinstance(value, dt.datetime):
        return (0, value.timestamp())
    if isinstance(value, (int, float, bool)):
        return (0, float(value))
    return (0, str(value).lower())


def _matches(row: dict, conditions: list[Condition], mode: str, ctx: OpContext) -> bool:
    if not conditions:
        return True
    results = [_test_one(_dig(row, c.field), c) for c in conditions]
    return all(results) if mode == "all" else any(results)


def _test_one(left: Any, condition: Condition) -> bool:
    test, right = condition.test, condition.value

    if test == "exists":
        want = as_bool(right) if right is not None else True
        return (left not in (None, "", [], {})) is want
    if test == "empty":
        want = as_bool(right) if right is not None else True
        return (left in (None, "", [], {})) is want
    if test == "eq":
        return _same(left, right)
    if test == "ne":
        return not _same(left, right)
    if test == "in":
        return any(_same(left, option) for option in as_list(right))
    if test == "not_in":
        return not any(_same(left, option) for option in as_list(right))
    if test == "contains":
        if isinstance(left, (list, tuple, set)):
            return any(_same(item, right) for item in left)
        return str(right) in str(left or "")
    if test == "icontains":
        if isinstance(left, (list, tuple, set)):
            return any(str(right).lower() == str(item).lower() for item in left)
        return str(right).lower() in str(left or "").lower()
    if test == "starts_with":
        return str(left or "").lower().startswith(str(right).lower())
    if test == "matches":
        try:
            return re.search(str(right), str(left or ""), re.I) is not None
        except re.error as exc:
            raise AppError("VALIDATION_ERROR", f"{right!r} is not a usable pattern.", http=422) from exc
    if test == "within":
        moment = parse_dt(left)
        if moment is None:
            return False
        lo, hi = window_bounds(right)
        return lo <= moment < hi
    if test in ("before", "after"):
        moment, mark = parse_dt(left), parse_dt(right)
        if moment is None or mark is None:
            return False
        return moment < mark if test == "before" else moment > mark

    lo, hi = _numbers(left, right)
    if lo is None or hi is None:
        return False
    return {"gt": lo > hi, "gte": lo >= hi, "lt": lo < hi, "lte": lo <= hi}[test]


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        try:
            return as_bool(left) == as_bool(right)
        except AppError:
            return False
    if isinstance(left, str) or isinstance(right, str):
        return str(left).strip().lower() == str(right).strip().lower()
    return left == right


def _numbers(left: Any, right: Any) -> tuple[float | None, float | None]:
    def number(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, dt.datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    moment = parse_dt(value)
                except AppError:
                    return None
                return moment.timestamp() if moment else None
        return None

    return number(left), number(right)


# --------------------------------------------------------------------------- #
# llm.extract
# --------------------------------------------------------------------------- #


class ExtractArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    source: Any = None
    fields: list[str] = Field(default_factory=list)
    hint: str | None = None
    allow_llm: bool = True

    @model_validator(mode="after")
    def _something(self) -> "ExtractArgs":
        if not self.text and self.source is None:
            raise ValueError("llm.extract needs text or a source")
        return self


class LlmExtract(Op):
    """Pull named values out of text. Regex first, model only on a miss.

    The order is the whole design. Twenty compiled patterns answer in two
    milliseconds for the shapes that have shapes — booking references, flight
    numbers, amounts, links — and the model is spent only on the branch where
    they came up empty. `source` says which happened, so the cost table in
    `docs/DESIGN.md` stays a measurement.
    """

    name = "llm.extract"
    args_model = ExtractArgs
    output_fields = ["extracted", "found", "missing", "source", "llm_calls"]
    is_local = False
    timeout_s = 12.0
    summary = "extract named values from text (patterns first, model on a miss)"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        text = parsed.text or _text_from(parsed.source)
        if not text.strip():
            return OpResult(
                data={"extracted": {}, "found": [], "missing": parsed.fields, "source": "none", "llm_calls": 0}
            )

        found = run_extractors(text, parsed.fields or None)
        wanted = parsed.fields or list(found.keys())
        missing = [f for f in wanted if not str(found.get(f) or "").strip()]

        if not missing or not parsed.allow_llm:
            return OpResult(
                data=jsonable(
                    {
                        "extracted": found,
                        "found": sorted(found),
                        "missing": missing,
                        "source": "patterns" if found else "none",
                        "llm_calls": 0,
                        "patterns": "app.search.extractors" if extractors_module() else "builtin",
                    }
                )
            )

        from app.core import llm

        system = (
            "You read one document and pull out exactly the fields asked for.\n"
            "Return JSON with one key per field. Use null when the document does "
            "not contain it. Copy values verbatim — never reformat, never guess, "
            "never infer a value from a similar one."
        )
        user = (
            f"Fields: {', '.join(missing)}\n"
            + (f"Context: {parsed.hint}\n" if parsed.hint else "")
            + f"\nDocument:\n{text[:6000]}"
        )
        answer = await llm.complete_json(system, user, max_tokens=400)
        filled = {k: v for k, v in answer.items() if k in missing and v not in (None, "", [], {})}
        merged = {**found, **filled}

        return OpResult(
            data=jsonable(
                {
                    "extracted": merged,
                    "found": sorted(merged),
                    "missing": [f for f in wanted if f not in merged],
                    "source": "llm" if filled else "patterns",
                    "llm_calls": 1,
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        fields = as_list((args or {}).get("fields"))
        return f"Pulling out the {', '.join(str(f) for f in fields[:3])}" if fields else "Reading the details"


def _text_from(source: Any, limit: int = 8000) -> str:
    """Whatever a `{{step.x}}` reference resolved to, as readable text."""
    parts: list[str] = []

    def absorb(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            for sub in item[:5]:
                absorb(sub)
        elif isinstance(item, dict):
            for key in ("subject", "title", "name", "label"):
                if item.get(key):
                    parts.append(str(item[key]))
            for key in ("body", "body_clean", "description", "content_excerpt", "excerpt", "text"):
                if item.get(key):
                    parts.append(str(item[key]))
            for key in ("hits", "emails", "events", "files", "items"):
                if isinstance(item.get(key), list):
                    absorb(item[key][:3])

    absorb(source)
    return "\n\n".join(p for p in parts if p.strip())[:limit]


# --------------------------------------------------------------------------- #
# llm.map
# --------------------------------------------------------------------------- #


class MapArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Any] = Field(default_factory=list)
    instruction: str
    output_field: str = "value"
    keep_fields: list[str] = Field(default_factory=list)
    max_items: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, values: Any) -> Any:
        if isinstance(values, dict) and isinstance(values.get("items"), dict):
            holder = values["items"]
            for key in ("hits", "emails", "events", "files", "items"):
                if isinstance(holder.get(key), list):
                    return {**values, "items": holder[key]}
        return values


class LlmMap(Op):
    """One instruction applied to every item — in **one** call, not N.

    Classifying eight emails is one request with eight rows in it. A loop that
    called the model per item would be eight times the latency and eight times
    the bill for the same answer.
    """

    name = "llm.map"
    args_model = MapArgs
    output_fields = ["items", "count", "llm_calls"]
    timeout_s = 15.0
    summary = "apply one instruction to a list of items in a single call"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        rows = [r for r in parsed.items if r is not None][: parsed.max_items]
        if not rows:
            return OpResult(data={"items": [], "count": 0, "llm_calls": 0})

        from app.core import llm

        compact = []
        for index, row in enumerate(rows):
            if isinstance(row, dict):
                keep = parsed.keep_fields or ["ref", "label", "subject", "title", "name", "excerpt"]
                compact.append({"i": index, **{k: row.get(k) for k in keep if row.get(k) is not None}})
            else:
                compact.append({"i": index, "value": str(row)[:400]})

        system = (
            "You apply one instruction to every row you are given.\n"
            'Return JSON: {"results": [{"i": <row index>, "value": <answer>}]}\n'
            "One entry per row, same indexes, same order. No commentary."
        )
        answer = await llm.complete_json(
            system, f"Instruction: {parsed.instruction}\n\nRows:\n{jsonable(compact)}", max_tokens=900
        )
        by_index = {
            int(entry.get("i")): entry.get("value")
            for entry in (answer.get("results") or [])
            if isinstance(entry, dict) and str(entry.get("i", "")).lstrip("-").isdigit()
        }

        out = []
        for index, row in enumerate(rows):
            base = dict(row) if isinstance(row, dict) else {"value": row}
            base[parsed.output_field] = by_index.get(index)
            out.append(base)

        return OpResult(data=jsonable({"items": out, "count": len(out), "llm_calls": 1}))

    def progress_label(self, args: dict) -> str:
        return "Working through the list"


# --------------------------------------------------------------------------- #
# page.more
# --------------------------------------------------------------------------- #


class PageMoreArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    offset: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=1, le=100)


class PageMore(Op):
    """The next page of a search that had more.

    "Show me the rest" is a UI verb: the front door catches it, this op serves
    it, and the whole turn costs zero model calls. It re-runs the original op
    with a bigger offset rather than inventing a cursor of its own — the
    ordering is deterministic, so page two is page two.
    """

    name = "page.more"
    args_model = PageMoreArgs
    output_fields = ["hits", "count", "has_more", "next_offset", "op"]
    is_local = True
    timeout_s = 4.0
    summary = "the next page of a previous search"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        from app.ops.registry import get as get_op

        target = get_op(parsed.op)
        if target is None:
            raise AppError(
                "VALIDATION_ERROR",
                f"There is no op called {parsed.op!r} to page through.",
                http=422,
                details={"op": parsed.op},
            )
        if not target.is_local or target.is_write:
            raise AppError(
                "VALIDATION_ERROR",
                "Only a read can be paged.",
                http=422,
                details={"op": parsed.op},
            )

        forwarded = dict(parsed.args)
        previous_limit = int(forwarded.get("limit") or 10)
        forwarded["limit"] = parsed.limit or previous_limit
        forwarded["offset"] = (
            parsed.offset if parsed.offset is not None else int(forwarded.get("offset") or 0) + previous_limit
        )
        result = await target.run(ctx, forwarded)
        result.data = {**result.data, "op": parsed.op, "page_of": forwarded["offset"]}
        return result

    def progress_label(self, args: dict) -> str:
        return "Fetching more"


# --------------------------------------------------------------------------- #
# action.revise
# --------------------------------------------------------------------------- #


class ReviseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    patch: dict[str, Any] = Field(default_factory=dict)
    replace: bool = False

    @model_validator(mode="after")
    def _something(self) -> "ReviseArgs":
        if not self.patch:
            raise ValueError("action.revise needs a patch")
        return self


class ActionRevise(Op):
    """"Edit" on a confirm card.

    The patch is validated against the target op's own argument model before it
    is stored, so a revision cannot smuggle in a field the op would not have
    accepted in the first place. The old payload is kept in `revisions`, and
    only a `draft` action can be touched at all — once approved, the payload is
    what will execute.
    """

    name = "action.revise"
    args_model = ReviseArgs
    output_fields = ["action_id", "op", "payload", "revised", "status"]
    is_local = True
    is_write = True
    timeout_s = 4.0
    summary = "change a prepared action before it is approved"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        from app.ops.registry import get as get_op

        action = await actions_repo.require_action(ctx.session, ctx.user_id, parsed.action_id)
        merged = dict(parsed.patch) if parsed.replace else {**(action.payload or {}), **parsed.patch}

        target = get_op(action.op)
        if target is not None:
            # Validate the *shape* the op would accept. The payload carries
            # prepared extras — a draft id, a pinned etag — that are not op
            # arguments, so unknown keys are left alone and only the ones the
            # model declares are checked.
            declared = set(target.args_model.model_fields)
            overlap = {k: v for k, v in merged.items() if k in declared}
            if overlap:
                try:
                    target.args_model(**{**overlap, **_required_stubs(target, overlap)})
                except AppError:
                    raise
                except Exception as exc:
                    raise AppError(
                        "VALIDATION_ERROR",
                        "That edit does not fit what this action can do.",
                        http=422,
                        details={"op": action.op, "problem": str(exc)[:300]},
                    ) from exc

        updated = await actions_repo.revise_action(
            ctx.session, ctx.user_id, parsed.action_id, merged, replace=True
        )
        return OpResult(
            data=jsonable(
                {
                    "action_id": updated.id,
                    "op": updated.op,
                    "payload": updated.payload,
                    "revised": True,
                    "status": getattr(updated.status, "value", updated.status),
                    "revisions": len(updated.revisions or []),
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        return "Applying your edit"


def _required_stubs(target: Op, overlap: dict[str, Any]) -> dict[str, Any]:
    """Fill required fields the patch did not mention, so validating a partial
    edit does not fail for the wrong reason."""
    stubs: dict[str, Any] = {}
    for name, field in target.args_model.model_fields.items():
        if name in overlap or not field.is_required():
            continue
        annotation = str(field.annotation)
        if "list" in annotation:
            stubs[name] = []
        elif "int" in annotation or "float" in annotation:
            stubs[name] = 1
        elif "bool" in annotation:
            stubs[name] = False
        elif "dict" in annotation:
            stubs[name] = {}
        else:
            stubs[name] = "x"
    return stubs


class SetChatTitleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description=(
            "A short name for this conversation, in the person's own terms — "
            "'Turkish Airlines cancellation', not 'Flight Cancellation Request "
            "Workflow'. No trailing full stop."
        ),
    )


class SetChatTitle(Op):
    """Name this conversation.

    The sidebar falls back to the first thing somebody typed, which is a
    reasonable guess and often a bad name — "cancel my flight" reads the same
    whether it was Istanbul or Dubai, and a list of near-identical first lines
    is a list you cannot navigate.

    The model already knows what the conversation turned out to be about, so it
    can say so. Cheap on purpose: no model call of its own, no network, one
    UPDATE. A title somebody set by hand is left alone — they meant it, and a
    later turn quietly renaming their thread would be maddening.
    """

    name = "chat.set_title"
    args_model = SetChatTitleArgs
    output_fields = ["title", "applied"]
    is_local = True
    timeout_s = 2.0
    summary = "give this conversation a short name for the sidebar"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        from app.db.repositories import conversations as conv_repo

        parsed = self.parse(args)
        title = parsed.title.strip()

        row = await conv_repo.get_conversation(ctx.session, ctx.user_id, ctx.conversation_id)
        if row is None:
            return OpResult(data={"title": title, "applied": False})

        # `title IS NULL` means nobody has named it, so the sidebar is showing
        # a derived one. That is the only case worth overwriting.
        if getattr(row, "title", None) is not None:
            return OpResult(data={"title": getattr(row, "title"), "applied": False})

        await conv_repo.set_title(ctx.session, ctx.user_id, ctx.conversation_id, title)
        return OpResult(data={"title": title, "applied": True})

    def progress_label(self, args: dict) -> str:
        return "Naming this chat"


OPS: list[Op] = [
    SearchAll(),
    ResolvePerson(),
    ResolveReference(),
    AskUser(),
    DataFilter(),
    LlmExtract(),
    LlmMap(),
    PageMore(),
    ActionRevise(),
    SetChatTitle(),
]

__all__ = [
    "OPS",
    "ActionRevise",
    "SetChatTitle",
    "AskUser",
    "DataFilter",
    "LlmExtract",
    "LlmMap",
    "PageMore",
    "ResolvePerson",
    "ResolveReference",
    "SearchAll",
]
