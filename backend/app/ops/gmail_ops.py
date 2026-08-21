"""Gmail ops: two reads, three writes.

The reads go to our pgvector mirror, never to Google. `gmail.get_emails` is the
one that runs the regex extractors over the body, which is why the plans in
`docs/SAMPLE_QUERIES.md` reference `{{booking.extracted.pnr}}` off it.

The writes are where the money is:

* `gmail.draft_email` is a write but **not** a confirm. A draft is reversible,
  so it is made immediately — and making it up front is what lets the confirm
  card show the real message rather than a rendering of one.
* `gmail.send_email` is a `ConfirmableOp`. `run` creates the Gmail draft and
  stops. `execute` sends that draft, later, once a person has said yes.
* `gmail.update_labels` is a write with no confirm: archiving is undoable.

Both composing ops take a body three ways, in this order of preference:

1. **A literal** ``body``. Nothing is generated.
2. **A template** — ``template`` plus ``sources`` plus ``use_fields`` — where
   the fields come from what the extractors already pulled out of the sources.
   If every field the template needs is present, the body renders with **no
   LLM call**.
3. **A brief** plus ``sources``, or a template with fields missing. One CONTENT
   call, the only model call on the write path.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import AppError
from app.core.ids import fingerprint
from app.db.repositories import mirror as mirror_repo
from app.ops.base import (
    ConfirmableOp,
    Op,
    OpContext,
    OpResult,
    SearchOp,
    as_list,
    excerpt_of,
    google_call,
    has_google,
    iso,
    jsonable,
    row_to_dict,
    run_extractors,
    trim_for_llm,
)

# --------------------------------------------------------------------------- #
# Body templates
# --------------------------------------------------------------------------- #


class BodyTemplate:
    """A subject and a body with named holes, plus which holes are required.

    Rendering is `str.format` with a mapping that reports misses instead of
    raising, so "which fields are missing" is an answer, not an exception.
    """

    def __init__(
        self,
        name: str,
        subject: str,
        body: str,
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
        blurb: str = "",
    ) -> None:
        self.name = name
        self.subject = subject
        self.body = body
        self.required = required
        self.optional = optional
        self.blurb = blurb

    @property
    def fields(self) -> tuple[str, ...]:
        return self.required + self.optional

    def missing(self, values: dict[str, Any]) -> list[str]:
        values = normalise_vars(values)
        return [f for f in self.required if not str(values.get(f) or "").strip()]

    def render(self, values: dict[str, Any]) -> tuple[str, str]:
        """``(subject, body)``.

        A line whose every hole came out empty is dropped whole — that is what
        turns "Ticket   {ticket_no}" into nothing at all rather than into a
        dangling label. Runs of blank lines left behind are collapsed.
        """
        values = normalise_vars(values)
        filled = {f: str(values.get(f) or "").strip() for f in self.fields}
        lines: list[str] = []
        for line in self.body.splitlines():
            holes = _HOLE.findall(line)
            if holes and not any(filled.get(h, "").strip() for h in holes):
                continue
            lines.append(line.format_map(_Blanks(filled)).rstrip())
        body = _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()
        subject = " ".join(self.subject.format_map(_Blanks(filled)).split())
        return subject, body + "\n"


#: What the planner calls a hole, mapped to what the template calls it. The
#: same value arrives as `new_start` from a window reference and as `new_time`
#: from a phrase, and a template that silently rendered a blank because of the
#: spelling would send a person an email with the time missing from it.
VAR_ALIASES: dict[str, str] = {
    "new_start": "new_time",
    "starts_at": "new_time",
    "start": "new_time",
    "old_start": "old_time",
    "previous_start": "old_time",
    "was": "old_time",
    "summary": "title",
    "event_title": "title",
    "subject": "title",
    "from_name": "sender",
    "sender_name": "sender",
    "file": "file_name",
    "filename": "file_name",
    "url": "link",
    "web_link": "link",
}


def normalise_vars(values: dict[str, Any]) -> dict[str, Any]:
    """Template variables under the names the templates use.

    A name the template already carries always wins; an alias only fills a hole
    that would otherwise be blank.
    """
    if not values:
        return {}
    out = dict(values)
    for spelling, canonical in VAR_ALIASES.items():
        if str(out.get(canonical) or "").strip():
            continue
        supplied = out.get(spelling)
        if str(supplied or "").strip():
            out[canonical] = supplied
    return out


class _Blanks(dict):
    def __missing__(self, key: str) -> str:  # noqa: D105 - a blank, not a KeyError
        return ""


_HOLE = re.compile(r"\{(\w+)\}")
_BLANK_RUN = re.compile(r"\n{3,}")


CANCEL_FLIGHT = BodyTemplate(
    name="cancel_flight",
    subject="Cancellation request — booking {pnr}",
    body="""Hello,

Please cancel the booking below and confirm the cancellation by reply.

  Booking reference   {pnr}
  Ticket              {ticket_no}
  Flight              {flight_no}
  Route               {route}
  Departure           {depart_at}
  Passenger           {passenger}

If a refund is due, please apply it to the original form of payment.

Thank you,
{passenger}
""",
    required=("pnr", "flight_no", "passenger"),
    optional=("ticket_no", "route", "depart_at"),
    blurb="cancel a booking, quoting the reference",
)

RESCHEDULE_MEETING = BodyTemplate(
    name="reschedule_meeting",
    subject="Moving {title}",
    body="""Hi,

I need to move {title}.

  Currently   {old_time}
  Proposed    {new_time}

If that does not work, tell me what does and I will send an updated invite.

Thanks,
{sender}
""",
    required=("title", "new_time", "sender"),
    optional=("old_time",),
    blurb="propose a new time for an existing meeting",
)

DECLINE_INVITE = BodyTemplate(
    name="decline_invite",
    subject="Can't make {title}",
    body="""Hi,

Thanks for the invitation to {title}{when_clause}. I am not going to be able to
make it.

{reason}

Please carry on without me, and send me the notes if there are any.

Best,
{sender}
""",
    required=("title", "sender"),
    optional=("when_clause", "reason"),
    blurb="turn down an invitation",
)

SHARE_NOTICE = BodyTemplate(
    name="share_notice",
    subject="Shared with you: {file_name}",
    body="""Hi,

I have shared {file_name} with you.

  Link   {link}

{note}

Let me know if you cannot open it.

Best,
{sender}
""",
    required=("file_name", "sender"),
    optional=("link", "note"),
    blurb="tell someone a file has been shared with them",
)

TEMPLATES: dict[str, BodyTemplate] = {
    t.name: t for t in (CANCEL_FLIGHT, RESCHEDULE_MEETING, DECLINE_INVITE, SHARE_NOTICE)
}

#: Spellings the planner reaches for that mean one of the four.
TEMPLATE_ALIASES: dict[str, str] = {
    "flight_cancellation": "cancel_flight",
    "cancel_booking": "cancel_flight",
    "cancellation": "cancel_flight",
    "reschedule": "reschedule_meeting",
    "move_meeting": "reschedule_meeting",
    "meeting_moved": "reschedule_meeting",
    "moved_meeting": "reschedule_meeting",
    "reschedule_notice": "reschedule_meeting",
    "decline": "decline_invite",
    "decline_meeting": "decline_invite",
    "share": "share_notice",
    "file_shared": "share_notice",
}


def template_for(name: str) -> BodyTemplate:
    key = TEMPLATE_ALIASES.get(name.strip().lower(), name.strip().lower())
    template = TEMPLATES.get(key)
    if template is None:
        raise AppError(
            "VALIDATION_ERROR",
            f"There is no body template called {name!r}.",
            http=422,
            details={"template": name, "known": sorted(TEMPLATES)},
        )
    return template


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #

GMAIL_FILTERS: dict[str, str | None] = {
    "from_email": "from_email",
    "from_emails": "from_emails[]",
    "to_email": "to_email",
    "thread_id": "thread_id",
    "message_ids": "message_ids[]",
    "labels": "labels[]",
    "has_attachments": "bool:has_attachments",
    "subject_contains": "subject_contains",
    "since": "since",
    "until": "until",
    "window": "window",
    "received_window": "window",
    "participants": None,
    "is_unread": None,
    "has_link": None,
    "min_cn": None,
    "exclude_refs": None,
    "text_contains": None,
}

GMAIL_ORDER: dict[str, str] = {
    "relevance": "cn",
    "received_at": "received_at",
    "date": "received_at",
    "from_email": "from_email",
    "subject": "label",
}


class SearchEmails(SearchOp):
    """Mail out of the mirror. One hybrid pass, then Python for the rest."""

    name = "gmail.search_emails"
    corpus = "gmail"
    entity_type = "email"
    filter_spec = GMAIL_FILTERS
    order_spec = GMAIL_ORDER
    summary = "search your mail (mirror, not Google)"

    async def refresh_live(self, ctx: OpContext, args: Any, filters: Any) -> dict | None:
        """One targeted `messages.list`, upserted into the mirror.

        Narrow on purpose: the point is to close a gap of minutes on the exact
        thing this step is about, not to resync a mailbox.
        """
        if not has_google(ctx, "gmail", ("list_messages", "search_messages", "list")):
            return None
        query = (args.query or "").strip()
        sender = filters.sql.get("from_email")
        if sender:
            query = f"from:{sender} {query}".strip()
        raw = await google_call(
            ctx,
            "gmail",
            ("list_messages", "search_messages", "list"),
            query=query or None,
            max_results=min(args.limit + args.offset, 25),
        )
        rows = [_mirror_row_from_message(row_to_dict(m)) for m in (raw or [])]
        rows = [r for r in rows if r.get("message_id")]
        if rows:
            await mirror_repo.upsert_gmail(ctx.session, ctx.user_id, rows)
        return {"fetched": len(rows)}

    def progress_label(self, args: dict) -> str:
        query = (args or {}).get("query")
        sender = (args or {}).get("from_email") or ((args or {}).get("filter") or {}).get("from_email")
        if sender and query:
            return f"Searching mail from {sender} for “{excerpt_of(query, 40)}”"
        if sender:
            return f"Reading mail from {sender}"
        if query:
            return f"Searching your mail for “{excerpt_of(query, 40)}”"
        return "Reading your mail"

    def ambiguity_question(self, args: Any, hits: list[dict]) -> str:
        return "Which email did you mean?"


class GetEmailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message_ids: list[str] = Field(default_factory=list)
    message_id: str | None = None
    include_body: bool = True
    extract: bool = True
    project: list[str] | None = None
    expect: str = "many"
    freshness: str = "cached"

    @model_validator(mode="after")
    def _one_or_many(self) -> "GetEmailsArgs":
        if self.message_id and self.message_id not in self.message_ids:
            self.message_ids = [self.message_id, *self.message_ids]
        if not self.message_ids:
            raise ValueError("gmail.get_emails needs at least one message id")
        return self


class GetEmails(Op):
    """Whole messages by id, with the extractors run over the body.

    Chunks are stitched back together here — the mirror splits a long thread so
    retrieval works, but a cancellation email needs the whole text before the
    PNR regex has a chance.
    """

    name = "gmail.get_emails"
    args_model = GetEmailsArgs
    output_fields = [
        "emails",
        "count",
        "message_id",
        "thread_id",
        "subject",
        "from_email",
        "from_name",
        "to_emails",
        "received_at",
        "body",
        "extracted",
    ]
    is_local = True
    timeout_s = 4.0
    summary = "read whole emails by id, with PNRs and links pulled out"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        emails: list[dict] = []
        missing: list[str] = []

        for message_id in parsed.message_ids:
            row = await self._one(ctx, message_id, parsed)
            if row is None:
                missing.append(message_id)
            else:
                emails.append(row)

        data: dict[str, Any] = {
            "emails": emails,
            "count": len(emails),
            "missing": missing,
            "found": bool(emails),
        }
        if parsed.expect == "one" and emails:
            data = {**emails[0], **data}
        if not emails:
            return OpResult(
                data=data,
                needs_replan=parsed.expect == "one",
                replan_reason="none of those messages are in the mirror" if parsed.expect == "one" else None,
            )
        return OpResult(data=data)

    async def _one(self, ctx: OpContext, message_id: str, parsed: Any) -> dict | None:
        chunks = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gmail", message_id)
        rows = [row_to_dict(c) for c in chunks]

        if not rows and parsed.freshness == "live" and has_google(
            ctx, "gmail", ("get_message", "get", "message")
        ):
            fetched = await google_call(
                ctx, "gmail", ("get_message", "get", "message"), message_id=message_id, format="full"
            )
            if fetched:
                row = _mirror_row_from_message(row_to_dict(fetched))
                if row.get("message_id"):
                    await mirror_repo.upsert_gmail(ctx.session, ctx.user_id, [row])
                    rows = [row]
        if not rows:
            return None

        head = rows[0]
        body = "\n".join(str(r.get("body_clean") or "") for r in rows).strip()
        out: dict[str, Any] = {
            "message_id": head.get("message_id"),
            "thread_id": head.get("thread_id"),
            "subject": head.get("subject"),
            "from_email": head.get("from_email"),
            "from_name": head.get("from_name"),
            "to_emails": as_list(head.get("to_emails")),
            "labels": as_list(head.get("labels")),
            "has_attachments": bool(head.get("has_attachments")),
            "received_at": head.get("received_at"),
            "excerpt": excerpt_of(body),
            "chunks": len(rows),
        }
        if parsed.include_body:
            out["body"] = body
        if parsed.extract:
            out["extracted"] = run_extractors(f"{head.get('subject') or ''}\n{body}")
        if parsed.project:
            keep = set(parsed.project) | {"message_id", "extracted"}
            out = {k: v for k, v in out.items() if k in keep}
        return jsonable(out)

    def progress_label(self, args: dict) -> str:
        count = len((args or {}).get("message_ids") or []) or (1 if (args or {}).get("message_id") else 0)
        return "Opening that email" if count == 1 else f"Opening {count} emails"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        slim = dict(data)
        slim["emails"] = [
            {
                "message_id": e.get("message_id"),
                "subject": e.get("subject"),
                "from_email": e.get("from_email"),
                "received_at": e.get("received_at"),
                "excerpt": e.get("excerpt"),
                "extracted": e.get("extracted"),
            }
            for e in (data.get("emails") or [])[:5]
        ]
        slim.pop("body", None)
        return trim_for_llm(slim, budget)


def _mirror_row_from_message(message: dict) -> dict[str, Any]:
    """A Gmail API message as a `sync_gmail` row.

    Tolerant about field names because the thin wrapper may hand back either
    Google's spelling or ours; whichever it is, the mirror wants ours.
    """
    body = (
        message.get("body_clean")
        or message.get("body")
        or message.get("snippet")
        or message.get("text")
        or ""
    )
    received = message.get("received_at") or message.get("date") or message.get("internalDate")
    row = {
        "message_id": message.get("message_id") or message.get("id"),
        "thread_id": message.get("thread_id") or message.get("threadId"),
        "chunk_index": int(message.get("chunk_index") or 0),
        "subject": message.get("subject"),
        "from_email": message.get("from_email") or message.get("from"),
        "from_name": message.get("from_name"),
        "to_emails": [str(t) for t in as_list(message.get("to_emails") or message.get("to"))],
        "body_clean": str(body),
        "labels": [str(l) for l in as_list(message.get("labels") or message.get("labelIds"))],
        "has_attachments": bool(message.get("has_attachments") or message.get("hasAttachments")),
        "received_at": received,
    }
    row["content_hash"] = fingerprint(
        "sync_gmail.body", f"{row['message_id']}|{row['chunk_index']}|{row['body_clean']}"
    )
    return row


# --------------------------------------------------------------------------- #
# Composing
# --------------------------------------------------------------------------- #


class ComposeSpec(BaseModel):
    """How to make a body when there is no literal one."""

    model_config = ConfigDict(extra="allow")

    template: str | None = None
    brief: str | None = None
    sources: list[Any] = Field(default_factory=list)
    use_fields: dict[str, Any] = Field(default_factory=dict)
    tone: str = "plain, direct, no filler"


class ComposeArgs(BaseModel):
    """Everything both composing ops accept.

    Three spellings of the same thing are allowed — top-level ``body_template``
    and ``template_vars`` as the sample plans write them, a nested ``compose``
    block, or a plain literal body — because all three appear in the wild and
    normalising them here is cheaper than teaching the planner one.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    bcc: list[str] = Field(default_factory=list)
    subject: str | None = None
    body: str | None = None

    body_template: str | None = None
    template_vars: dict[str, Any] = Field(default_factory=dict)
    sources: list[Any] = Field(default_factory=list)
    compose: ComposeSpec | None = None

    in_reply_to: str | None = None
    thread_id: str | None = None
    draft_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_scalars(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = dict(values)
        for key in ("to", "cc", "bcc", "sources"):
            if key in out and not isinstance(out[key], list):
                out[key] = as_list(out[key])
        if "template" in out and "body_template" not in out:
            out["body_template"] = out.pop("template")
        if "use_fields" in out and "template_vars" not in out:
            out["template_vars"] = out.pop("use_fields")
        # A model writes `compose: "write a friendly email about X"` at least
        # as often as the object form. The string IS the brief; refusing it
        # fails the one step the whole plan existed to run.
        if isinstance(out.get("compose"), str):
            out["compose"] = {"brief": out["compose"]}
        for key in ("brief", "instructions", "prompt"):
            if isinstance(out.get(key), str) and "compose" not in out:
                out["compose"] = {"brief": out.pop(key)}
                break
        return out

    @model_validator(mode="after")
    def _needs_a_body(self) -> "ComposeArgs":
        if self.compose is not None:
            self.body_template = self.body_template or self.compose.template
            self.template_vars = {**(self.compose.use_fields or {}), **(self.template_vars or {})}
            self.sources = list(self.sources) + list(self.compose.sources or [])
        has_source = bool(
            self.body or self.body_template or self.draft_id or (self.compose and self.compose.brief)
        )
        if not has_source:
            raise ValueError(
                "give a body, or a template plus sources, or a brief plus sources"
            )
        if not self.to and not self.draft_id:
            raise ValueError("an email needs at least one recipient")
        return self


def _source_values(sources: list[Any]) -> dict[str, Any]:
    """Everything the extractors already found, flattened.

    A source is whatever a `{{step.x}}` reference resolved to: a hit, a list of
    hits, an email, or a bare dict of values. Earlier sources win, because the
    planner puts the one it means first.
    """
    values: dict[str, Any] = {}

    def absorb(item: Any) -> None:
        if isinstance(item, list):
            for sub in item:
                absorb(sub)
            return
        if not isinstance(item, dict):
            return
        for key in ("extracted", "value", "fields"):
            inner = item.get(key)
            if isinstance(inner, dict):
                for k, v in inner.items():
                    values.setdefault(k, v)
        for k, v in item.items():
            if k in ("extracted", "value", "fields", "hits", "emails"):
                continue
            if isinstance(v, (str, int, float, bool)) and v not in (None, ""):
                values.setdefault(k, v)
        for key in ("hits", "emails", "events", "files"):
            inner = item.get(key)
            if isinstance(inner, list):
                absorb(inner)

    for source in sources:
        absorb(source)
    return values


def _text_of(sources: list[Any], limit: int = 2400) -> str:
    """The source material a CONTENT call is allowed to see."""
    parts: list[str] = []

    def absorb(item: Any) -> None:
        if isinstance(item, list):
            for sub in item:
                absorb(sub)
        elif isinstance(item, dict):
            for key in ("subject", "title", "label", "name"):
                if item.get(key):
                    parts.append(str(item[key]))
            for key in ("body", "body_clean", "excerpt", "description", "content_excerpt"):
                if item.get(key):
                    parts.append(str(item[key]))
            for key in ("hits", "emails", "events", "files"):
                if isinstance(item.get(key), list):
                    absorb(item[key][:3])
        elif isinstance(item, str):
            parts.append(item)

    for source in sources:
        absorb(source)
    joined = "\n".join(p for p in parts if p.strip())
    return joined[:limit]


async def compose_body(ctx: OpContext, parsed: ComposeArgs, *, sender_name: str) -> dict[str, Any]:
    """The body, and an honest record of what it cost.

    Returns ``{subject, body, source, template, missing, llm_calls}``. ``source``
    is ``"literal"``, ``"template"`` or ``"llm"`` — the number the cost table in
    `docs/DESIGN.md` is counted from.
    """
    if parsed.body:
        return {
            "subject": parsed.subject or "",
            "body": parsed.body,
            "source": "literal",
            "template": None,
            "missing": [],
            "llm_calls": 0,
        }

    values = {**_source_values(parsed.sources), **(parsed.template_vars or {})}
    values.setdefault("sender", sender_name)
    values.setdefault("passenger", sender_name)
    if values.get("when") and not values.get("when_clause"):
        values["when_clause"] = f" on {values['when']}"

    if parsed.body_template:
        template = template_for(parsed.body_template)
        missing = template.missing(values)
        if not missing:
            subject, body = template.render(values)
            return {
                "subject": parsed.subject or subject,
                "body": body,
                "source": "template",
                "template": template.name,
                "missing": [],
                "llm_calls": 0,
            }
        drafted = await _llm_body(
            ctx,
            brief=f"Write the email this template describes: {template.blurb}.",
            template=template,
            values=values,
            sources=parsed.sources,
            subject=parsed.subject,
        )
        drafted["missing"] = missing
        return drafted

    brief = parsed.compose.brief if parsed.compose else None
    return await _llm_body(
        ctx,
        brief=brief or "Write the email the sources call for.",
        template=None,
        values=values,
        sources=parsed.sources,
        subject=parsed.subject,
    )


_COMPOSE_SYSTEM = """You write short work emails on someone's behalf.

Rules:
- Plain, direct, no filler and no flattery. Six sentences at most.
- Use only facts present in the material given. Never invent a reference
  number, a date, a price or a name. If a fact is missing, write around it.
- No placeholders, no square brackets, nothing for the sender to fill in.
- Sign off with the sender's name exactly as given.

Return JSON: {"subject": "...", "body": "..."}"""


async def _llm_body(
    ctx: OpContext,
    *,
    brief: str,
    template: BodyTemplate | None,
    values: dict[str, Any],
    sources: list[Any],
    subject: str | None,
) -> dict[str, Any]:
    """The one CONTENT call on the write path."""
    from app.core import llm  # imported here so the ops import without a key

    known = {k: v for k, v in values.items() if isinstance(v, (str, int, float)) and str(v).strip()}
    shape = ""
    if template is not None:
        shape = (
            f"\nShape it like the '{template.name}' template, which normally shows: "
            + ", ".join(template.fields)
            + ". Leave out anything you do not have."
        )
    user = (
        f"{brief}{shape}\n\n"
        f"Facts already known:\n{jsonable(known)}\n\n"
        f"Source material:\n{_text_of(sources) or '(none)'}\n\n"
        f"Sender's name: {values.get('sender') or 'the sender'}"
    )
    result = await llm.complete_json(_COMPOSE_SYSTEM, user, max_tokens=600)
    body = str(result.get("body") or "").strip()
    if not body:
        raise AppError("INTERNAL", "The model returned an empty email body.")
    return {
        "subject": subject or str(result.get("subject") or "").strip() or "(no subject)",
        "body": body if body.endswith("\n") else body + "\n",
        "source": "llm",
        "template": template.name if template else None,
        "missing": [],
        "llm_calls": 1,
    }


async def _sender_name(ctx: OpContext) -> str:
    from app.db.repositories import users as users_repo

    user = await users_repo.get_user(ctx.session, ctx.user_id)
    if user is None:
        return "me"
    return (getattr(user, "display_name", None) or str(getattr(user, "email", "")).split("@")[0]) or "me"


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


class DraftEmail(Op):
    """Create a real Gmail draft.

    A write, not a confirm: nothing has left the building, and the user can see
    and delete it in Gmail. `gmail.send_email` is where the gate is.
    """

    name = "gmail.draft_email"
    args_model = ComposeArgs
    output_fields = ["draft_id", "message_id", "thread_id", "to", "subject", "body", "body_source"]
    is_write = True
    needs_confirm = False
    timeout_s = 8.0
    summary = "save a Gmail draft (reversible, not sent)"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        composed = await compose_body(ctx, parsed, sender_name=await _sender_name(ctx))
        subject = parsed.subject or composed["subject"] or "(no subject)"

        created = row_to_dict(
            await google_call(
                ctx,
                "gmail",
                ("create_draft", "drafts_create", "create"),
                to=parsed.to,
                cc=parsed.cc,
                bcc=parsed.bcc,
                subject=subject,
                body=composed["body"],
                in_reply_to=parsed.in_reply_to,
                thread_id=parsed.thread_id,
            )
            or {}
        )
        draft_id = created.get("draft_id") or created.get("id")
        if not draft_id:
            raise AppError("GOOGLE_UNAVAILABLE", "Gmail did not return a draft id.")

        return OpResult(
            data=jsonable(
                {
                    "draft_id": draft_id,
                    "message_id": created.get("message_id") or (created.get("message") or {}).get("id"),
                    "thread_id": created.get("thread_id") or parsed.thread_id,
                    "to": parsed.to,
                    "cc": parsed.cc,
                    "bcc": parsed.bcc,
                    "subject": subject,
                    "body": composed["body"],
                    "body_source": composed["source"],
                    "template": composed["template"],
                    "llm_calls": composed["llm_calls"],
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        to = as_list((args or {}).get("to"))
        return f"Drafting an email to {to[0]}" if to else "Drafting an email"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        slim = {k: v for k, v in data.items() if k != "body"}
        slim["body_excerpt"] = excerpt_of(data.get("body"), 300)
        return trim_for_llm(slim, budget)


class SendEmail(ConfirmableOp):
    """Send an email — prepared here, sent later.

    `run` makes the Gmail draft if there is not one already and returns the
    payload the confirm card is built from. Nothing is sent. `execute` sends
    that exact draft, which is also why `actions.external_ref` is a draft id:
    a send that is retried sends the same message rather than a second copy.
    """

    name = "gmail.send_email"
    args_model = ComposeArgs
    output_fields = ["draft_id", "to", "subject", "body", "payload", "prepared"]
    timeout_s = 8.0
    summary = "send an email — prepares a draft and asks first"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)

        if parsed.draft_id and not parsed.body and not parsed.body_template:
            existing = row_to_dict(
                await google_call(
                    ctx, "gmail", ("get_draft", "drafts_get", "get"), draft_id=parsed.draft_id
                )
                or {}
            )
            payload = {
                "draft_id": parsed.draft_id,
                # `parse_message` spells these `to_emails`/`cc_emails`; the
                # short names cover a raw Gmail payload. Missing both is how
                # the confirm card ends up asking "Send this to them?".
                "to": parsed.to or as_list(existing.get("to") or existing.get("to_emails")),
                "cc": parsed.cc or as_list(existing.get("cc") or existing.get("cc_emails")),
                "bcc": parsed.bcc or as_list(existing.get("bcc") or existing.get("bcc_emails")),
                "subject": parsed.subject or existing.get("subject") or "(no subject)",
                "body": existing.get("body") or existing.get("body_clean") or "",
                "body_source": "existing_draft",
            }
        else:
            composed = await compose_body(ctx, parsed, sender_name=await _sender_name(ctx))
            subject = parsed.subject or composed["subject"] or "(no subject)"
            created = row_to_dict(
                await google_call(
                    ctx,
                    "gmail",
                    ("create_draft", "drafts_create", "create"),
                    to=parsed.to,
                    cc=parsed.cc,
                    bcc=parsed.bcc,
                    subject=subject,
                    body=composed["body"],
                    in_reply_to=parsed.in_reply_to,
                    thread_id=parsed.thread_id,
                )
                or {}
            )
            draft_id = created.get("draft_id") or created.get("id")
            if not draft_id:
                raise AppError("GOOGLE_UNAVAILABLE", "Gmail did not return a draft id.")
            payload = {
                "draft_id": draft_id,
                "to": parsed.to,
                "cc": parsed.cc,
                "bcc": parsed.bcc,
                "subject": subject,
                "body": composed["body"],
                "body_source": composed["source"],
                "template": composed["template"],
                "thread_id": created.get("thread_id") or parsed.thread_id,
                "llm_calls": composed["llm_calls"],
            }

        payload = jsonable({k: v for k, v in payload.items() if v not in (None, [], "")})
        return OpResult(
            data={
                **payload,
                "payload": payload,
                "prepared": True,
                "external_ref": payload["draft_id"],
                "preview": self.preview(payload),
                "confirm_question": self.confirm_question(payload),
            }
        )

    async def execute(self, ctx: OpContext, payload: dict) -> dict:
        """The irreversible half. Called from the actions worker, after a yes."""
        draft_id = payload.get("draft_id")
        if not draft_id:
            raise AppError("VALIDATION_ERROR", "There is no draft to send.", http=422)
        sent = row_to_dict(
            await google_call(
                ctx, "gmail", ("send_draft", "drafts_send", "send"), draft_id=draft_id
            )
            or {}
        )
        return jsonable(
            {
                "message_id": sent.get("message_id") or sent.get("id"),
                "thread_id": sent.get("thread_id") or sent.get("threadId") or payload.get("thread_id"),
                "to": payload.get("to"),
                "subject": payload.get("subject"),
                "sent_at": iso(dt.datetime.now(dt.timezone.utc)),
            }
        )

    def preview(self, payload: dict) -> dict:
        return {
            "to": payload.get("to"),
            "cc": payload.get("cc"),
            "subject": payload.get("subject"),
            "body": payload.get("body"),
            "note": "The draft is already saved in your Gmail drafts. Nothing has been sent.",
        }

    def confirm_question(self, payload: dict) -> str:
        to = as_list(payload.get("to"))
        who = to[0] if to else "them"
        if len(to) > 1:
            who = f"{to[0]} and {len(to) - 1} other" + ("s" if len(to) > 2 else "")
        return f"Send this to {who}?"

    def progress_label(self, args: dict) -> str:
        to = as_list((args or {}).get("to"))
        return f"Preparing an email to {to[0]}" if to else "Preparing an email"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        slim = {
            "draft_id": data.get("draft_id"),
            "to": data.get("to"),
            "subject": data.get("subject"),
            "body_excerpt": excerpt_of(data.get("body"), 300),
            "body_source": data.get("body_source"),
            "prepared": True,
        }
        return trim_for_llm(slim, budget)


class UpdateLabelsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_ids: list[str] = Field(default_factory=list)
    message_id: str | None = None
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = dict(values)
        for key in ("message_ids", "add", "remove"):
            if key in out:
                out[key] = as_list(out[key])
        for old, new in (("add_labels", "add"), ("remove_labels", "remove")):
            if old in out:
                out[new] = as_list(out.pop(old))
        return out

    @model_validator(mode="after")
    def _check(self) -> "UpdateLabelsArgs":
        if self.message_id and self.message_id not in self.message_ids:
            self.message_ids = [self.message_id, *self.message_ids]
        if not self.message_ids:
            raise ValueError("gmail.update_labels needs at least one message id")
        if not self.add and not self.remove:
            raise ValueError("give at least one label to add or remove")
        return self


class UpdateLabels(Op):
    """Add and remove labels. A write, but an undoable one, so no confirm.

    Archiving is `remove: ["INBOX"]`; marking read is `remove: ["UNREAD"]`.
    """

    name = "gmail.update_labels"
    args_model = UpdateLabelsArgs
    output_fields = ["message_ids", "added", "removed", "count"]
    is_write = True
    timeout_s = 8.0
    summary = "label, archive or mark mail read"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        add = [str(l).upper() for l in parsed.add]
        remove = [str(l).upper() for l in parsed.remove]

        await google_call(
            ctx,
            "gmail",
            ("modify_labels", "update_labels", "batch_modify", "modify"),
            message_ids=parsed.message_ids,
            add=add,
            remove=remove,
        )
        # Keep the mirror honest without waiting for the next sync cycle.
        for message_id in parsed.message_ids:
            chunks = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gmail", message_id)
            for chunk in chunks:
                labels = [str(l) for l in as_list(getattr(chunk, "labels", None))]
                labels = [l for l in labels if l.upper() not in set(remove)]
                labels += [l for l in add if l not in labels]
                chunk.labels = labels

        return OpResult(
            data={
                "message_ids": parsed.message_ids,
                "added": add,
                "removed": remove,
                "count": len(parsed.message_ids),
            }
        )

    def progress_label(self, args: dict) -> str:
        remove = {str(l).upper() for l in as_list((args or {}).get("remove"))}
        count = len(as_list((args or {}).get("message_ids"))) or 1
        noun = "email" if count == 1 else f"{count} emails"
        if "INBOX" in remove:
            return f"Archiving {noun}"
        if "UNREAD" in remove:
            return f"Marking {noun} read"
        return f"Relabelling {noun}"


OPS: list[Op] = [SearchEmails(), GetEmails(), DraftEmail(), SendEmail(), UpdateLabels()]

__all__ = [
    "CANCEL_FLIGHT",
    "DECLINE_INVITE",
    "OPS",
    "RESCHEDULE_MEETING",
    "SHARE_NOTICE",
    "TEMPLATES",
    "BodyTemplate",
    "ComposeArgs",
    "DraftEmail",
    "GetEmails",
    "SearchEmails",
    "SendEmail",
    "UpdateLabels",
    "compose_body",
    "template_for",
]
