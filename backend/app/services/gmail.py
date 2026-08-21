"""Gmail, as this system needs it.

Two jobs. The first is a thin, honest wrapper over the REST methods we call:
search, fetch, draft, send, relabel, and the two listing calls the sync uses.

The second is the part that matters more. A Gmail message off the wire is a
tree of MIME parts, base64url-encoded, usually carrying a quoted copy of every
earlier message in the thread, a signature, and a footer full of tracking
links. Embedding that is embedding noise: every message in a thread ends up
looking like every other one, and a search for "the budget email" matches all
five. So :func:`clean_body` cuts the quoted trail, the signature and the
footer, and what is left — ``body_clean`` — is what gets stored, embedded and
searched.

The idempotency header ``X-Orchestrator-Idem`` goes on every draft and every
send. Gmail has no idempotency key of its own, so if a send is retried after a
timeout the header is what proves the message in the Sent folder is the one we
already made, not a second copy.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import html as html_module
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from email.message import EmailMessage
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from typing import Any, Final

from app.core.errors import AppError
from app.core.logging import get_logger
from app.google.client import SAFE_TO_REPEAT, Transport
from app.google.retry import ErrorClass, GoogleAPIError

log = get_logger(__name__)

USER: Final[str] = "me"
IDEM_HEADER: Final[str] = "X-Orchestrator-Idem"

#: How many message fetches run at once. Gmail is happy with more; the quota
#: governor is the real limit, and this keeps a 200-message backfill from
#: opening 200 sockets.
FETCH_CONCURRENCY: Final[int] = 8

#: Anything longer than this is a newsletter or a legal footer, not a message
#: someone wrote. Chunking splits it afterwards; this only stops one absurd
#: mail from dominating a batch.
MAX_BODY_CHARS: Final[int] = 40_000


# --------------------------------------------------------------------------- #
# base64url
# --------------------------------------------------------------------------- #


def b64url_decode(data: str) -> bytes:
    """Gmail's encoding: base64url, padding usually stripped."""
    if not data:
        return b""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return b""


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# MIME -> text
# --------------------------------------------------------------------------- #


def _part_charset(part: dict[str, Any]) -> str:
    for header in part.get("headers") or []:
        if str(header.get("name", "")).lower() == "content-type":
            match = re.search(r"charset=\"?([\w\-]+)\"?", str(header.get("value", "")), re.I)
            if match:
                return match.group(1)
    return "utf-8"


def _decode_part(part: dict[str, Any]) -> str:
    body = part.get("body") or {}
    raw = b64url_decode(body.get("data") or "")
    if not raw:
        return ""
    try:
        return raw.decode(_part_charset(part), errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def html_to_text(html: str) -> str:
    """Enough HTML handling for an email. Not a browser, and not trying to be."""
    if not html:
        return ""
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</(div|tr|li|h[1-6]|table)\s*>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "- ", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html_module.unescape(text)
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def walk_parts(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Every part of the MIME tree, depth first."""
    stack = [payload]
    while stack:
        part = stack.pop(0)
        if not isinstance(part, dict):
            continue
        yield part
        children = part.get("parts")
        if isinstance(children, list):
            stack = list(children) + stack


def extract_bodies(payload: dict[str, Any]) -> tuple[str, str]:
    """``(plain_text, html)`` from a message payload.

    Prefers the ``text/plain`` alternative, because it is what the sender's
    client generated from the same content and it is free of markup.
    """
    plain: list[str] = []
    html: list[str] = []
    for part in walk_parts(payload or {}):
        mime = str(part.get("mimeType") or "")
        if part.get("filename"):
            continue  # an attachment, not the message
        if mime == "text/plain":
            plain.append(_decode_part(part))
        elif mime == "text/html":
            html.append(_decode_part(part))
    return "\n".join(p for p in plain if p).strip(), "\n".join(h for h in html if h).strip()


def attachments_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in walk_parts(payload or {}):
        filename = part.get("filename")
        if not filename:
            continue
        body = part.get("body") or {}
        out.append(
            {
                "attachment_id": body.get("attachmentId"),
                "filename": filename,
                "mime_type": part.get("mimeType"),
                "size": body.get("size"),
            }
        )
    return out


def headers_of(message: dict[str, Any]) -> dict[str, str]:
    """Headers as a lower-cased dict. Last one wins, which matches Gmail."""
    out: dict[str, str] = {}
    for item in (message.get("payload") or {}).get("headers") or []:
        name = str(item.get("name") or "").lower()
        if name:
            out[name] = str(item.get("value") or "")
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

# Where a quoted trail starts. Each is matched over the whole body, and the
# earliest hit wins — a reply can carry more than one of these.
_QUOTE_MARKERS: Final[tuple[re.Pattern[str], ...]] = (
    # "On Tue, 12 Aug 2026 at 10:04, Sarah Chen <sarah@company.com> wrote:"
    # The attribution wraps across lines often enough that this has to span them.
    re.compile(r"(?im)^[ \t>]*On\b[\s\S]{0,300}?\bwrote:[ \t]*$"),
    re.compile(r"(?im)^[ \t>]*-{2,}\s*Original Message\s*-{2,}[ \t]*$"),
    re.compile(r"(?im)^[ \t>]*-{2,}\s*Forwarded message\s*-{2,}[ \t]*$"),
    re.compile(r"(?im)^[ \t>]*_{10,}[ \t]*$"),
    # Outlook's block header
    re.compile(r"(?im)^[ \t>]*From:[ \t]*.+\n[ \t>]*(Sent|Date):[ \t]*.+$"),
    # "El 12 ago 2026, a las 10:04, ... escribió:" / "Am 12.08.2026 schrieb ..."
    re.compile(r"(?im)^[ \t>]*(El|Le|Am|Il)\b[\s\S]{0,300}?\b(escribió|a écrit|schrieb|ha scritto):[ \t]*$"),
    re.compile(r"(?im)^[ \t>]*\d{4}-\d{2}-\d{2}\b[^\n]{0,200}<[^>\n]+@[^>\n]+>[^\n]{0,40}:[ \t]*$"),
)

# Where a signature starts.
_SIGNATURE_MARKERS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?m)^--[ \t]?$"),
    re.compile(r"(?im)^[ \t]*Sent from my (iPhone|iPad|Android|Samsung|BlackBerry|mobile)[^\n]*$"),
    re.compile(r"(?im)^[ \t]*Get Outlook for (iOS|Android)[^\n]*$"),
)

# Footers a machine added: unsubscribe blocks, delivery notices, legalese.
_FOOTER_MARKERS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?im)^[^\n]{0,120}\bunsubscribe\b[^\n]{0,200}$"),
    re.compile(r"(?im)^[^\n]{0,120}\bview (this email|it) in your browser\b[^\n]{0,120}$"),
    re.compile(r"(?im)^[^\n]{0,120}\bthis (e-?mail|message) was sent to\b[^\n]{0,200}$"),
    re.compile(r"(?im)^[^\n]{0,120}\byou (are )?receiv(ed|ing) this (e-?mail|message)\b[^\n]{0,200}$"),
    re.compile(r"(?im)^[^\n]{0,120}\bmanage (your )?(email )?preferences\b[^\n]{0,120}$"),
    re.compile(r"(?im)^[^\n]{0,120}\ball rights reserved\b[^\n]{0,120}$"),
    re.compile(r"(?im)^[^\n]{0,120}\bconfidential(ity)? notice\b[^\n]{0,200}$"),
    re.compile(r"(?im)^[^\n]{0,80}\bprivacy policy\b[^\n]{0,80}\|[^\n]{0,120}$"),
)

# A footer is only a footer near the end. "Unsubscribe me from the Tuesday
# call" in the first paragraph is the message, not boilerplate — so a marker
# only counts once most of the mail is behind it, and never in the opening
# line of a short one.
_FOOTER_FLOOR_CHARS: Final[int] = 80
_FOOTER_FLOOR_RATIO: Final[float] = 0.55

_TRACKING_URL = re.compile(
    r"https?://\S*?(?:utm_[a-z]+=|/track/|/o/|/wf/click|list-manage\.com|sendgrid\.net|mailchimp)\S*",
    re.I,
)


def _earliest(text: str, patterns: Sequence[re.Pattern[str]]) -> int | None:
    best: int | None = None
    for pattern in patterns:
        match = pattern.search(text)
        if match and (best is None or match.start() < best):
            best = match.start()
    return best


def strip_quoted(text: str) -> str:
    """Cut the quoted copy of the thread, and any lines left prefixed with '>'."""
    cut = _earliest(text, _QUOTE_MARKERS)
    if cut is not None:
        text = text[:cut]
    lines = [line for line in text.splitlines() if not re.match(r"^[ \t]*>", line)]
    return "\n".join(lines)


def strip_signature(text: str) -> str:
    cut = _earliest(text, _SIGNATURE_MARKERS)
    return text[:cut] if cut is not None else text


def strip_footers(text: str) -> str:
    """Cut a marketing or legal footer, but only where one can plausibly be."""
    cut = _earliest(text, _FOOTER_MARKERS)
    if cut is None:
        return text
    floor = max(_FOOTER_FLOOR_CHARS, int(len(text) * _FOOTER_FLOOR_RATIO))
    if cut < floor:
        return text
    return text[:cut]


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("​", "").replace("­", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_body(text: str, *, html: str | None = None) -> str:
    """The message someone actually wrote.

    Quoted trail, signature and footer removed; tracking links dropped;
    whitespace tidied. This is the string that is embedded and full-text
    indexed, which is why the same input has to give the same output every
    time — nothing here depends on the clock or on anything outside the text.
    """
    body = text or (html_to_text(html or "") if html else "")
    body = normalise_whitespace(body)
    body = strip_quoted(body)
    body = strip_signature(body)
    body = strip_footers(body)
    body = _TRACKING_URL.sub("", body)
    body = normalise_whitespace(body)
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + " ..."
    return body


# --------------------------------------------------------------------------- #
# Parsing a message
# --------------------------------------------------------------------------- #


def _addresses(value: str) -> list[str]:
    return [addr.lower() for _, addr in getaddresses([value or ""]) if addr]


def _received_at(message: dict[str, Any], headers: dict[str, str]) -> datetime:
    internal = message.get("internalDate")
    if internal:
        try:
            return datetime.fromtimestamp(int(internal) / 1000, tz=UTC)
        except (TypeError, ValueError, OSError):
            pass
    raw = headers.get("date")
    if raw:
        try:
            when = parsedate_to_datetime(raw)
            if when is not None:
                return when.astimezone(UTC) if when.tzinfo else when.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC)


def parse_message(message: dict[str, Any]) -> dict[str, Any]:
    """One Gmail message, flattened into the shape the rest of the system uses.

    The keys line up with ``sync_gmail`` on purpose: the sync task can hand
    this almost straight to the mirror repository, and an op can read the same
    field names in a plan reference.
    """
    headers = headers_of(message)
    payload = message.get("payload") or {}
    plain, html = extract_bodies(payload)
    body_clean = clean_body(plain, html=html)
    from_name, from_email = parseaddr(headers.get("from", ""))
    attachments = attachments_of(payload)

    return {
        "id": message.get("id"),
        "message_id": message.get("id"),
        "thread_id": message.get("threadId"),
        "history_id": message.get("historyId"),
        "snippet": html_module.unescape(str(message.get("snippet") or "")),
        "subject": headers.get("subject") or None,
        "from_email": (from_email or "").lower() or None,
        "from_name": from_name or None,
        "to_emails": _addresses(headers.get("to", "")),
        "cc_emails": _addresses(headers.get("cc", "")),
        "reply_to": (parseaddr(headers.get("reply-to", ""))[1] or "").lower() or None,
        "rfc822_message_id": headers.get("message-id"),
        "references": headers.get("references"),
        "list_unsubscribe": headers.get("list-unsubscribe"),
        "idem_key": headers.get(IDEM_HEADER.lower()),
        "labels": list(message.get("labelIds") or []),
        "received_at": _received_at(message, headers),
        "body_text": plain,
        "body_html": html or None,
        "body_clean": body_clean,
        "attachments": attachments,
        "has_attachments": bool(attachments),
        "size_estimate": message.get("sizeEstimate"),
    }


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #


def _as_gmail_date(value: Any) -> str | None:
    """Gmail wants ``YYYY/MM/DD``. Dates are resolved in Python long before here."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y/%m/%d")
    if isinstance(value, date):
        return value.strftime("%Y/%m/%d")
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text.replace("-", "/")
    return parsed.strftime("%Y/%m/%d")


def _quote(value: str) -> str:
    text = str(value).strip()
    return f'"{text}"' if " " in text and not text.startswith('"') else text


#: filter name -> Gmail operator. Everything else is refused rather than
#: silently dropped, because a filter that vanishes widens a search.
_FILTER_OPS: Final[dict[str, str]] = {
    "from": "from",
    "from_email": "from",
    "to": "to",
    "cc": "cc",
    "bcc": "bcc",
    "subject": "subject",
    "label": "label",
    "in": "in",
    "category": "category",
    "filename": "filename",
    "larger": "larger",
    "smaller": "smaller",
    "rfc822msgid": "rfc822msgid",
    "list": "list",
}

_DATE_FILTERS: Final[dict[str, str]] = {
    "after": "after",
    "before": "before",
    "start": "after",
    "end": "before",
    "newer_than": "newer_than",
    "older_than": "older_than",
}


def build_query(q: str | None = None, filters: dict[str, Any] | None = None) -> str:
    """Turn free text plus a filter dict into one Gmail ``q`` string."""
    parts: list[str] = []
    if q:
        parts.append(str(q).strip())

    for name, value in (filters or {}).items():
        if value is None or value == "" or value == []:
            continue
        key = str(name).strip().lower()

        if key in _DATE_FILTERS:
            operator = _DATE_FILTERS[key]
            if operator in {"newer_than", "older_than"}:
                parts.append(f"{operator}:{value}")
            else:
                rendered = _as_gmail_date(value)
                if rendered:
                    parts.append(f"{operator}:{rendered}")
            continue

        if key in {"has_attachment", "has_attachments"}:
            if value:
                parts.append("has:attachment")
            continue
        if key == "is_unread":
            parts.append("is:unread" if value else "is:read")
            continue
        if key == "is_starred" and value:
            parts.append("is:starred")
            continue
        if key in {"labels", "label_ids"}:
            for label in value if isinstance(value, (list, tuple, set)) else [value]:
                parts.append(f"label:{_quote(label)}")
            continue
        if key in {"exclude", "not"}:
            for term in value if isinstance(value, (list, tuple, set)) else [value]:
                parts.append(f"-{_quote(term)}")
            continue
        if key in {"q", "text", "query"}:
            parts.append(str(value))
            continue

        operator = _FILTER_OPS.get(key)
        if operator is None:
            log.warning("gmail.unknown_filter", filter=key)
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        rendered = " OR ".join(f"{operator}:{_quote(v)}" for v in values if v)
        if len(values) > 1:
            rendered = f"({rendered})"
        parts.append(rendered)

    return " ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Building an outgoing message
# --------------------------------------------------------------------------- #


def build_mime(
    *,
    to: Sequence[str] | str,
    subject: str,
    body: str,
    cc: Sequence[str] | str | None = None,
    bcc: Sequence[str] | str | None = None,
    from_email: str | None = None,
    reply_to: str | None = None,
    in_reply_to: str | None = None,
    references: str | None = None,
    idem_key: str | None = None,
    html: str | None = None,
) -> str:
    """An RFC 5322 message, base64url encoded the way Gmail wants it.

    ``idem_key`` is written as ``X-Orchestrator-Idem``. It survives into the
    sent copy, so a retry can look for it before sending a second time.
    """

    def _join(value: Sequence[str] | str | None) -> str | None:
        if not value:
            return None
        if isinstance(value, str):
            return value
        return ", ".join(v for v in value if v)

    message = EmailMessage()
    message["To"] = _join(to) or ""
    message["Subject"] = subject or ""
    if from_email:
        message["From"] = from_email
    if _join(cc):
        message["Cc"] = _join(cc)
    if _join(bcc):
        message["Bcc"] = _join(bcc)
    if reply_to:
        message["Reply-To"] = reply_to
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
        message["References"] = references or in_reply_to
    elif references:
        message["References"] = references
    if idem_key:
        message[IDEM_HEADER] = idem_key

    message.set_content(body or "")
    if html:
        message.add_alternative(html, subtype="html")
    return b64url_encode(message.as_bytes())


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class GmailService:
    """Gmail bound to one user. Every method is one or a few REST calls."""

    service = "gmail"

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    @property
    def user_id(self) -> str:
        return self.transport.user_id

    # -- reading ------------------------------------------------------------ #

    async def messages_list(
        self,
        *,
        q: str | None = None,
        label_ids: Sequence[str] | None = None,
        page_token: str | None = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
    ) -> dict[str, Any]:
        """One page of message ids. What the backfill walks."""
        page = await self.transport.get(
            f"users/{USER}/messages",
            api_method="gmail.messages.list",
            params={
                "q": q or None,
                "labelIds": list(label_ids) if label_ids else None,
                "pageToken": page_token,
                "maxResults": max(1, min(int(max_results), 500)),
                "includeSpamTrash": include_spam_trash,
            },
        )
        messages = page.get("messages") or []
        return {
            "ids": [m["id"] for m in messages if m.get("id")],
            "threads": [m.get("threadId") for m in messages],
            "next_page_token": page.get("nextPageToken"),
            "estimate": page.get("resultSizeEstimate"),
        }

    async def get_message(
        self, message_id: str, *, format: str = "full"
    ) -> dict[str, Any] | None:
        """One message, parsed. ``None`` when Gmail says it is gone."""
        try:
            raw = await self.transport.get(
                f"users/{USER}/messages/{message_id}",
                api_method="gmail.messages.get",
                params={"format": format},
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return None
            raise
        return parse_message(raw)

    async def get_messages(
        self,
        ids: Sequence[str],
        *,
        format: str = "full",
        concurrency: int = FETCH_CONCURRENCY,
    ) -> list[dict[str, Any]]:
        """Several messages at once, in the order asked for.

        Deleted messages drop out silently — a search hit that has since been
        deleted is not an error, it is one fewer result. Anything else is
        raised, because a 401 halfway through a batch is not "fewer results".
        """
        wanted = [i for i in dict.fromkeys(ids) if i]
        if not wanted:
            return []

        gate = asyncio.Semaphore(max(1, concurrency))

        async def one(message_id: str) -> dict[str, Any] | None:
            async with gate:
                return await self.get_message(message_id, format=format)

        results = await asyncio.gather(
            *(one(i) for i in wanted), return_exceptions=True
        )
        out: list[dict[str, Any]] = []
        for message_id, result in zip(wanted, results, strict=True):
            if isinstance(result, BaseException):
                log.warning(
                    "gmail.fetch_failed", message_id=message_id, error=str(result)[:200]
                )
                raise result
            if result is not None:
                out.append(result)
        return out

    async def search(
        self,
        q: str | None = None,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 25,
        hydrate: bool = True,
        page_token: str | None = None,
        include_spam_trash: bool = False,
        format: str = "full",
    ) -> dict[str, Any]:
        """Search, then fetch the hits.

        Gmail's list call returns ids only, so a real search is two round
        trips. ``hydrate=False`` skips the second when the caller only needs
        to know how many there are.
        """
        query = build_query(q, filters)
        page = await self.messages_list(
            q=query or None,
            page_token=page_token,
            max_results=limit,
            include_spam_trash=include_spam_trash,
        )
        ids = page["ids"][:limit]
        messages = await self.get_messages(ids, format=format) if hydrate and ids else []
        return {
            "query": query,
            "ids": ids,
            "messages": messages,
            "next_page_token": page["next_page_token"],
            "estimate": page["estimate"],
        }

    async def list_messages(
        self,
        *,
        query: str | None = None,
        max_results: int = 25,
        include_spam_trash: bool = False,
    ) -> list[dict[str, Any]]:
        """Messages matching a Gmail query, hydrated, as a plain list.

        The convenience shape the ops call, matching `GDriveService.list_files`:
        one page, already fetched. `messages_list` is the raw id-only paging
        call underneath, and is what the sync task walks.
        """
        found = await self.search(
            query or None,
            limit=max_results,
            hydrate=True,
            include_spam_trash=include_spam_trash,
        )
        return list(found.get("messages") or [])

    async def history_list(
        self,
        start_history_id: str | int,
        *,
        page_token: str | None = None,
        history_types: Sequence[str] | None = None,
        label_id: str | None = None,
        max_results: int = 500,
    ) -> dict[str, Any]:
        """What changed since ``start_history_id``.

        Returns the ids that changed and the ids that went away, plus the new
        cursor. A 404 or 410 here means the history id is older than Gmail
        keeps; the caller has to fall back to a bounded full sync, so it is
        raised as PRECONDITION rather than swallowed.
        """
        page = await self.transport.get(
            f"users/{USER}/history",
            api_method="gmail.history.list",
            params={
                "startHistoryId": str(start_history_id),
                "pageToken": page_token,
                "historyTypes": list(history_types)
                if history_types
                else ["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"],
                "labelId": label_id,
                "maxResults": max(1, min(int(max_results), 500)),
            },
        )

        changed: list[str] = []
        deleted: list[str] = []
        for record in page.get("history") or []:
            for key in ("messagesAdded", "labelsAdded", "labelsRemoved"):
                for item in record.get(key) or []:
                    message = item.get("message") or {}
                    if message.get("id"):
                        changed.append(message["id"])
            for item in record.get("messagesDeleted") or []:
                message = item.get("message") or {}
                if message.get("id"):
                    deleted.append(message["id"])

        gone = set(deleted)
        return {
            "changed": [i for i in dict.fromkeys(changed) if i not in gone],
            "deleted": list(dict.fromkeys(deleted)),
            "history_id": page.get("historyId"),
            "next_page_token": page.get("nextPageToken"),
            "raw": page.get("history") or [],
        }

    async def get_profile(self) -> dict[str, Any]:
        """Address, message count, and the historyId a first sync starts from."""
        profile = await self.transport.get(
            f"users/{USER}/profile", api_method="gmail.users.getProfile"
        )
        return {
            "email": profile.get("emailAddress"),
            "messages_total": profile.get("messagesTotal"),
            "threads_total": profile.get("threadsTotal"),
            "history_id": profile.get("historyId"),
        }

    async def labels_list(self) -> list[dict[str, Any]]:
        page = await self.transport.get(
            f"users/{USER}/labels", api_method="gmail.labels.list"
        )
        return page.get("labels") or []

    # -- writing ------------------------------------------------------------ #

    async def create_draft(
        self,
        *,
        to: Sequence[str] | str,
        subject: str,
        body: str,
        cc: Sequence[str] | str | None = None,
        bcc: Sequence[str] | str | None = None,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        references: str | None = None,
        idem_key: str | None = None,
        html: str | None = None,
    ) -> dict[str, Any]:
        """Create a draft. This is the *prepare* half of every send.

        A confirm card is backed by a real Gmail draft, so the person can open
        it in Gmail, and so approving it later is a send rather than a compose.
        """
        raw = build_mime(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            in_reply_to=in_reply_to,
            references=references,
            idem_key=idem_key,
            html=html,
        )
        message: dict[str, Any] = {"raw": raw}
        if thread_id:
            message["threadId"] = thread_id
        created = await self.transport.post(
            f"users/{USER}/drafts",
            api_method="gmail.drafts.create",
            json={"message": message},
            headers={IDEM_HEADER: idem_key} if idem_key else None,
        )
        inner = created.get("message") or {}
        return {
            "draft_id": created.get("id"),
            "message_id": inner.get("id"),
            "thread_id": inner.get("threadId"),
            "labels": inner.get("labelIds") or [],
        }

    async def send_draft(self, draft_id: str, *, idem_key: str | None = None) -> dict[str, Any]:
        """Send a draft that already exists. The irreversible half."""
        sent = await self.transport.post(
            f"users/{USER}/drafts/send",
            api_method="gmail.drafts.send",
            json={"id": draft_id},
            headers={IDEM_HEADER: idem_key} if idem_key else None,
            # A send that timed out may still have landed. Retrying blind is how
            # you send twice, so only the failures that prove Gmail refused the
            # request are retried; anything else goes back to the caller, which
            # checks for the idempotency header before trying again.
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )
        message = sent.get("message") or sent
        return {
            "draft_id": draft_id,
            "message_id": message.get("id"),
            "thread_id": message.get("threadId"),
            "labels": message.get("labelIds") or [],
        }

    async def send_message(
        self,
        *,
        to: Sequence[str] | str,
        subject: str,
        body: str,
        cc: Sequence[str] | str | None = None,
        bcc: Sequence[str] | str | None = None,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        references: str | None = None,
        idem_key: str | None = None,
        html: str | None = None,
    ) -> dict[str, Any]:
        """Compose and send in one call, stamped with ``X-Orchestrator-Idem``."""
        raw = build_mime(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            in_reply_to=in_reply_to,
            references=references,
            idem_key=idem_key,
            html=html,
        )
        payload: dict[str, Any] = {"raw": raw}
        if thread_id:
            payload["threadId"] = thread_id
        sent = await self.transport.post(
            f"users/{USER}/messages/send",
            api_method="gmail.messages.send",
            json=payload,
            headers={IDEM_HEADER: idem_key} if idem_key else None,
            retry_on=SAFE_TO_REPEAT,  # see send_draft
            retry_on_network=False,
        )
        message = sent.get("message") or sent
        return {
            "message_id": message.get("id"),
            "thread_id": message.get("threadId"),
            "labels": message.get("labelIds") or [],
            "idem_key": idem_key,
        }

    async def find_sent_with_idem(self, idem_key: str) -> dict[str, Any] | None:
        """Has this exact send already happened?

        Gmail indexes custom headers, so the key we stamped is searchable. This
        is what makes a retry after a timeout safe: look before sending again.
        """
        page = await self.messages_list(
            q=f'in:sent "{idem_key}"', max_results=5, include_spam_trash=True
        )
        for message_id in page["ids"]:
            message = await self.get_message(message_id)
            if message and message.get("idem_key") == idem_key:
                return message
        return None

    async def update_labels(
        self,
        message_id: str,
        *,
        add: Sequence[str] | None = None,
        remove: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Add and remove labels in one call. Archiving is removing INBOX."""
        updated = await self.transport.post(
            f"users/{USER}/messages/{message_id}/modify",
            api_method="gmail.messages.modify",
            json={
                "addLabelIds": list(add or []),
                "removeLabelIds": list(remove or []),
            },
        )
        return {
            "message_id": updated.get("id", message_id),
            "thread_id": updated.get("threadId"),
            "labels": updated.get("labelIds") or [],
        }

    async def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        """A draft as a flat dict: recipients, subject, body.

        The confirm card is rebuilt from this when a person comes back to an
        action later, so it has to carry the whole message, not just an id.
        """
        try:
            raw = await self.transport.get(
                f"users/{USER}/drafts/{draft_id}",
                api_method="gmail.drafts.get",
                params={"format": "full"},
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return None
            raise
        message = parse_message(raw.get("message") or {})
        headers = headers_of(raw.get("message") or {})
        return {
            "draft_id": raw.get("id"),
            "id": raw.get("id"),
            "message_id": message["message_id"],
            "thread_id": message["thread_id"],
            "to": message["to_emails"],
            "cc": message["cc_emails"],
            "bcc": _addresses(headers.get("bcc", "")),
            "subject": message["subject"],
            "body": message["body_text"] or message["body_clean"],
            "body_clean": message["body_clean"],
            "labels": message["labels"],
        }

    async def delete_draft(self, draft_id: str) -> bool:
        """Discard a draft. Already gone counts as done."""
        try:
            await self.transport.delete(
                f"users/{USER}/drafts/{draft_id}",
                api_method="gmail.drafts.delete",
                expect="none",
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return False
            raise
        return True

    async def modify_labels(
        self,
        message_ids: Sequence[str] | str,
        *,
        add: Sequence[str] | None = None,
        remove: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Relabel several messages. One call each, run together.

        Gmail has ``batchModify``, but it returns nothing — no labels, no
        confirmation of which ids it touched — and this system has to tell a
        person what actually changed.
        """
        ids = [message_ids] if isinstance(message_ids, str) else [i for i in message_ids if i]
        if not ids:
            return {"updated": [], "failed": []}

        gate = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def one(message_id: str) -> dict[str, Any]:
            async with gate:
                return await self.update_labels(message_id, add=add, remove=remove)

        results = await asyncio.gather(*(one(i) for i in ids), return_exceptions=True)
        updated: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for message_id, result in zip(ids, results, strict=True):
            if isinstance(result, BaseException):
                failed.append({"message_id": message_id, "error": str(result)[:200]})
            else:
                updated.append(result)
        if failed and not updated:
            raise GoogleAPIError(
                502,
                reason="backendError",
                message=f"None of the {len(ids)} messages could be relabelled.",
                service="gmail",
                method="gmail.messages.modify",
            )
        return {"updated": updated, "failed": failed, "add": list(add or []), "remove": list(remove or [])}

    async def trash_message(self, message_id: str) -> dict[str, Any]:
        """Move to trash. Never a hard delete — that is not ours to do."""
        trashed = await self.transport.post(
            f"users/{USER}/messages/{message_id}/trash",
            api_method="gmail.messages.trash",
        )
        return {
            "message_id": trashed.get("id", message_id),
            "labels": trashed.get("labelIds") or [],
        }


# --------------------------------------------------------------------------- #
# Raw-payload functions, for the sync and action workers
# --------------------------------------------------------------------------- #
#
# The workers do their own chunking, hashing and upserting, so they want what
# Google actually said — ``{"messages": [...], "nextPageToken": ...}`` — and
# they call ``normalise_message`` themselves when they are ready. Every one of
# these takes the clients container first, the way a repository takes user_id.


def _transport(clients: Any) -> Transport:
    """The Gmail transport out of whatever the caller is holding."""
    if isinstance(clients, Transport):
        return clients
    service = getattr(clients, "gmail", clients)
    transport = getattr(service, "transport", None)
    if isinstance(transport, Transport):
        return transport
    raise AppError.internal(
        "Expected Google clients with a gmail transport.",
        got=type(clients).__name__,
    )


#: ``normalise_message`` is the name the sync tasks use; it is the same
#: function as :func:`parse_message`.
normalise_message = parse_message


async def messages_list(
    clients: Any,
    *,
    query: str | None = None,
    page_token: str | None = None,
    max_results: int = 100,
    label_ids: Sequence[str] | None = None,
    include_spam_trash: bool = False,
) -> dict[str, Any]:
    """Raw ``users.messages.list`` page."""
    return await _transport(clients).get(
        f"users/{USER}/messages",
        api_method="gmail.messages.list",
        params={
            "q": query or None,
            "labelIds": list(label_ids) if label_ids else None,
            "pageToken": page_token,
            "maxResults": max(1, min(int(max_results), 500)),
            "includeSpamTrash": include_spam_trash,
        },
    )


async def messages_get(
    clients: Any, message_id: str, *, format: str = "full"
) -> dict[str, Any] | None:
    """Raw ``users.messages.get``. ``None`` when the message has gone."""
    try:
        return await _transport(clients).get(
            f"users/{USER}/messages/{message_id}",
            api_method="gmail.messages.get",
            params={"format": format},
        )
    except GoogleAPIError as exc:
        if exc.error_class is ErrorClass.NOT_FOUND:
            return None
        raise


async def history_list(
    clients: Any,
    *,
    start_history_id: str | int,
    page_token: str | None = None,
    max_results: int = 500,
    history_types: Sequence[str] | None = None,
    label_id: str | None = None,
) -> dict[str, Any]:
    """Raw ``users.history.list`` page.

    A 404 or 410 is left to reach the caller: the history id is older than
    Gmail keeps, and only the sync task can decide to fall back to a bounded
    full walk.
    """
    return await _transport(clients).get(
        f"users/{USER}/history",
        api_method="gmail.history.list",
        params={
            "startHistoryId": str(start_history_id),
            "pageToken": page_token,
            "historyTypes": list(history_types)
            if history_types
            else ["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"],
            "labelId": label_id,
            "maxResults": max(1, min(int(max_results), 500)),
        },
    )


async def get_profile(clients: Any) -> dict[str, Any]:
    """Raw ``users.getProfile`` — the first historyId a sync starts from."""
    return await _transport(clients).get(
        f"users/{USER}/profile", api_method="gmail.users.getProfile"
    )


async def get_draft(clients: Any, draft_id: str) -> dict[str, Any] | None:
    """Raw draft, or ``None`` when it is not there any more."""
    try:
        return await _transport(clients).get(
            f"users/{USER}/drafts/{draft_id}",
            api_method="gmail.drafts.get",
            params={"format": "full"},
        )
    except GoogleAPIError as exc:
        if exc.error_class is ErrorClass.NOT_FOUND:
            return None
        raise


async def create_draft(clients: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a draft from an action payload.

    The payload is what the person approved — ``to``, ``subject``, ``body`` and
    the rest — so this is also how a draft someone deleted from their mailbox
    gets recreated before a send.
    """
    message: dict[str, Any]
    if payload.get("raw"):
        message = {"raw": payload["raw"]}
    elif isinstance(payload.get("message"), dict):
        message = payload["message"]
    else:
        message = {
            "raw": build_mime(
                to=payload.get("to") or [],
                subject=payload.get("subject") or "",
                body=payload.get("body") or "",
                cc=payload.get("cc"),
                bcc=payload.get("bcc"),
                in_reply_to=payload.get("in_reply_to"),
                references=payload.get("references"),
                idem_key=payload.get("idem_key") or payload.get("dedupe_key"),
                html=payload.get("html"),
            )
        }
    thread_id = payload.get("thread_id") or payload.get("threadId")
    if thread_id:
        message["threadId"] = thread_id
    return await _transport(clients).post(
        f"users/{USER}/drafts",
        api_method="gmail.drafts.create",
        json={"message": message},
    )


async def delete_draft(clients: Any, draft_id: str) -> bool:
    """Discard a draft. Already gone counts as done."""
    try:
        await _transport(clients).delete(
            f"users/{USER}/drafts/{draft_id}",
            api_method="gmail.drafts.delete",
            expect="none",
        )
    except GoogleAPIError as exc:
        if exc.error_class is ErrorClass.NOT_FOUND:
            return False
        raise
    return True


async def search_sent(
    clients: Any, query: str, *, max_results: int = 5
) -> list[dict[str, Any]]:
    """Message stubs for a Sent-folder search.

    This is the "did this already go out?" question the action worker asks
    before repeating a send, so it looks in Sent whatever the query says.
    """
    text = query if "in:sent" in query else f"in:sent {query}"
    page = await messages_list(
        clients, query=text, max_results=max_results, include_spam_trash=True
    )
    return list(page.get("messages") or [])


__all__ = [
    "GmailService",
    "IDEM_HEADER",
    "MAX_BODY_CHARS",
    "create_draft",
    "delete_draft",
    "get_draft",
    "get_profile",
    "history_list",
    "messages_get",
    "messages_list",
    "normalise_message",
    "search_sent",
    "b64url_decode",
    "b64url_encode",
    "build_mime",
    "build_query",
    "clean_body",
    "extract_bodies",
    "headers_of",
    "html_to_text",
    "parse_message",
    "attachments_of",
    "strip_footers",
    "strip_quoted",
    "strip_signature",
    "walk_parts",
]
