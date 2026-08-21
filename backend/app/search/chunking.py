"""Turning a message, an event or a file into the text we actually embed.

Three jobs, in the order the sync tasks call them:

1. :func:`clean_email` — throw away the parts of an email nobody searches for.
   A quoted reply chain embeds the *previous* conversation into this message's
   vector; a legal footer embeds the company's lawyers into every vector in the
   mailbox. Both make every email look slightly like every other email, which
   is exactly the failure mode a vector index cannot recover from.
2. :func:`chunk` — split long text on boundaries a human would recognise, with
   an overlap so a sentence cut in half still appears whole somewhere.
3. ``embed_text_for_*`` — assemble what goes to the embedding model. The
   subject is repeated, because in a 40-word email the subject is the part that
   says what it is, and one copy of it is 2% of the vector.

Everything here is pure and synchronous. No database, no network, no clock.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

#: Anything past one of these is a quotation of an earlier message.
_QUOTE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}\s*$", re.I | re.M),
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}\s*$", re.I | re.M),
    re.compile(r"^_{10,}\s*$", re.M),
    # "On Tue, 12 Aug 2026 at 09:14, Priya Raman <priya@x> wrote:" — the line
    # may wrap, so the "wrote:" is allowed up to 200 characters later.
    re.compile(r"^\s*On\b.{0,200}?\bwrote:\s*$", re.I | re.M | re.S),
    re.compile(r"^\s*El\b.{0,200}?\bescribió:\s*$", re.I | re.M | re.S),
    re.compile(r"^\s*Le\b.{0,200}?\ba écrit\s*:\s*$", re.I | re.M | re.S),
    re.compile(r"^\s*Am\b.{0,200}?\bschrieb\b.{0,80}?:\s*$", re.I | re.M | re.S),
    re.compile(r"^\s*\d{1,2}[.\s].{0,120}?\btarihinde .{0,80}?yazdı:\s*$", re.I | re.M | re.S),  # noqa: RUF001
    # Outlook's header block, which starts with a bare From: line.
    re.compile(r"^\s*From:\s.+?\n\s*(Sent|Date):\s.+?$", re.I | re.M | re.S),
    re.compile(r"^\s*(Kimden|De|Von|Van):\s.+?\n\s*(Gönderilen|Enviado|Gesendet|Verzonden):\s.+?$",
               re.I | re.M | re.S),
)

#: Anything past one of these is a signature.
_SIGNATURE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^--\s*$", re.M),                      # RFC 3676 §4.3
    re.compile(r"^__+\s*$", re.M),
    re.compile(r"^\s*Sent from my (iPhone|iPad|Android|mobile|Galaxy|BlackBerry).*$", re.I | re.M),
    re.compile(r"^\s*Get Outlook for (iOS|Android).*$", re.I | re.M),
    re.compile(r"^\s*Sent via .{0,40}$", re.I | re.M),
)

#: Anything past one of these is a footer — legal, marketing or list plumbing.
_FOOTER_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(This (e-?mail|message)( and any attachments)? (is|are|may be)"
               r"\s+(confidential|intended)).*$", re.I | re.M),
    re.compile(r"^\s*(Bu (e-?posta|mesaj)).{0,60}(gizli|yalnızca).*$", re.I | re.M),  # noqa: RUF001
    re.compile(r"^\s*(If you (are not|received this) .{0,80}(in error|intended recipient)).*$",
               re.I | re.M),
    re.compile(r"^\s*(You (are )?receiv(e|ed|ing) this (e-?mail|message|notification)"
               r"\s+because).*$", re.I | re.M),
    re.compile(r"^\s*(To )?[Uu]nsubscribe\b.*$", re.M),
    re.compile(r"^\s*(Manage|Update) your (e-?mail )?(preferences|subscription).*$", re.I | re.M),
    re.compile(r"^\s*View (this (e-?mail|message)|it) in your browser.*$", re.I | re.M),
    re.compile(r"^\s*(Privacy Policy|Terms of Service|Do not reply to this)\b.*$", re.I | re.M),
    re.compile(r"^\s*(©|\(c\)|Copyright)\s*\d{4}\b.*$", re.I | re.M),
    re.compile(r"^\s*Please consider the environment before printing.*$", re.I | re.M),
)

_HTML_HINT = re.compile(r"<\s*(html|body|div|p|br|table|td|a|span|img)\b", re.I)
_SCRIPT_STYLE = re.compile(r"<\s*(script|style|head)[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_BLOCK_END = re.compile(r"<\s*/?\s*(p|div|tr|table|ul|ol|li|h[1-6]|blockquote|br)\s*/?\s*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_QUOTED_LINE = re.compile(r"^\s*>+\s?.*$", re.M)
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.M)
_MANY_BLANKS = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-Z0-9İÇĞÖŞÜ])")


def strip_html(raw: str) -> str:
    """Plain text out of an HTML body, block structure kept as newlines."""
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = _BLOCK_END.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return text


def normalise_whitespace(text: str) -> str:
    """Tidy without destroying paragraphs: they are the chunk boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text).replace("\xa0", " ")
    text = _MANY_SPACES.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def _cut_at_first(text: str, patterns: Iterable[re.Pattern[str]], *, keep_min: int = 0) -> str:
    """Truncate at the earliest marker, unless that would leave nothing.

    ``keep_min`` protects a short message whose first line already looks like a
    marker — a one-line reply under a quoted chain still has to keep its line.
    """
    cut = len(text)
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None and match.start() < cut and match.start() >= keep_min:
            cut = match.start()
    return text[:cut]


def clean_email(raw: str | None) -> str:
    """The part of an email that is this message and nothing else.

    Drops HTML markup, quoted reply chains, signature blocks and footers, then
    normalises whitespace. Idempotent: cleaning an already-clean body returns
    it unchanged.
    """
    if not raw:
        return ""
    text = str(raw)
    if _HTML_HINT.search(text):
        text = strip_html(text)
    text = normalise_whitespace(text)
    if not text:
        return ""

    # Quoted chains first: a signature inside a quotation should not be the cut
    # point for the whole message.
    text = _cut_at_first(text, _QUOTE_MARKERS, keep_min=1)
    text = _QUOTED_LINE.sub("", text)
    text = _cut_at_first(text, _FOOTER_MARKERS, keep_min=1)
    # A signature marker in the first 40 characters is almost always a stray
    # "--" bullet rather than a real sig block.
    text = _cut_at_first(text, _SIGNATURE_MARKERS, keep_min=40)

    return normalise_whitespace(text)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _split_long(piece: str, size: int) -> list[str]:
    """Break one over-long paragraph on sentence ends, then on words."""
    if len(piece) <= size:
        return [piece]
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(piece):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                out.append(current)
                current = ""
            buffer = ""
            for word in sentence.split(" "):
                while len(word) > size:  # a URL or a base64 blob: cut it dead
                    if buffer:
                        out.append(buffer)
                        buffer = ""
                    out.append(word[:size])
                    word = word[size:]
                if not word:
                    continue
                if buffer and len(buffer) + 1 + len(word) > size:
                    out.append(buffer)
                    buffer = word
                else:
                    buffer = f"{buffer} {word}".strip()
            if buffer:
                current = buffer
            continue
        if current and len(current) + 1 + len(sentence) > size:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def _tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters, cut at a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if space > 0:
        tail = tail[space + 1 :]
    return tail.strip()


def _join(prefix: str, content: str, size: int) -> str:
    """One finished chunk: as much of the previous tail as still fits."""
    if not prefix:
        return content
    room = size - len(content) - 1
    if room <= 0:
        return content
    return f"{_tail(prefix, room)} {content}".strip()


def chunk(
    text: str | None,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping pieces of at most ``size`` characters.

    Paragraphs are kept whole where they fit, then sentences, then words. Each
    chunk after the first is prefixed with the tail of the one before it, so a
    fact that straddles a boundary is complete in at least one chunk.

    An empty input gives an empty list — the caller decides whether a row with
    no body is worth storing.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(int(overlap), size // 2))

    body = normalise_whitespace(text or "")
    if not body:
        return []
    if len(body) <= size:
        return [body]

    # New content gets `size - overlap`; the rest of the budget is reserved for
    # the tail of the chunk before, so the overlap never costs a chunk its own
    # content and no chunk ever exceeds `size`.
    budget = size - overlap
    pieces: list[str] = []
    for paragraph in _split_paragraphs(body):
        pieces.extend(_split_long(paragraph, budget))

    chunks: list[str] = []
    content = ""
    prefix = ""
    for piece in pieces:
        candidate = f"{content}\n\n{piece}".strip() if content else piece
        if content and len(candidate) > budget:
            chunks.append(_join(prefix, content, size))
            prefix = _tail(content, overlap)
            content = piece
        else:
            content = candidate
    if content:
        chunks.append(_join(prefix, content, size))
    return chunks


# --------------------------------------------------------------------------- #
# What we hand the embedding model
# --------------------------------------------------------------------------- #


def _clip(text: str | None, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def embed_text_for_email(subject: str | None, body: str | None) -> str:
    """Subject, subject again, then the body.

    The repetition is deliberate and it is the cheapest ranking lever we have.
    "Your Turkish Airlines booking is confirmed" is the sentence that decides
    what this email *is*; the body is four paragraphs of seat numbers and legal
    text. One copy of the subject in 400 words of body barely moves the vector.
    """
    head = (subject or "").strip()
    text = (body or "").strip()
    parts = [head, head, text] if head else [text]
    return "\n".join(p for p in parts if p).strip()


def _attendee_names(attendees: Any) -> list[str]:
    out: list[str] = []
    if not attendees:
        return out
    if isinstance(attendees, Mapping):
        attendees = [attendees]
    if isinstance(attendees, (str, bytes)):
        return [str(attendees)]
    for person in attendees:
        if isinstance(person, Mapping):
            label = (person.get("name") or "").strip() or (person.get("email") or "").strip()
            if label:
                out.append(str(label))
        elif person:
            out.append(str(person))
    return out


def embed_text_for_event(
    title: str | None,
    description: str | None = None,
    location: str | None = None,
    attendees: Any = None,
    *,
    organizer_email: str | None = None,
) -> str:
    """Title twice, then the things people search an event by.

    Attendees are in the text on purpose: "the meeting with John" is a query
    about the guest list, not about the description.
    """
    head = (title or "").strip()
    lines: list[str] = []
    if head:
        lines.extend([head, head])
    if location:
        lines.append(f"Location: {str(location).strip()}")
    people = _attendee_names(attendees)
    if organizer_email:
        organiser = str(organizer_email).strip()
        if organiser and organiser not in people:
            people.insert(0, organiser)
    if people:
        lines.append("With: " + ", ".join(people[:20]))
    if description:
        lines.append(_clip(description, 2000))
    return "\n".join(line for line in lines if line).strip()


_MIME_LABELS = {
    "application/pdf": "PDF document",
    "application/vnd.google-apps.document": "Google Doc",
    "application/vnd.google-apps.spreadsheet": "Google Sheet",
    "application/vnd.google-apps.presentation": "Google Slides",
    "application/vnd.google-apps.folder": "Folder",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel spreadsheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint deck",
    "text/plain": "Text file",
    "text/csv": "CSV file",
    "image/png": "Image",
    "image/jpeg": "Image",
}


def filename_words(name: str | None) -> str:
    """``Acme_MSA_countersigned.pdf`` → ``Acme MSA countersigned``.

    Filenames are written in snake case, kebab case and camel case, and none of
    those tokenise the way a sentence does. Splitting them is what lets a
    filename match a query typed as words.
    """
    if not name:
        return ""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", str(name))
    spaced = re.sub(r"[_\-.]+", " ", stem)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return re.sub(r"\s{2,}", " ", spaced).strip()


def embed_text_for_file(
    name: str | None,
    content_excerpt: str | None = None,
    mime_type: str | None = None,
    folder_path: str | None = None,
) -> str:
    """Name twice — once verbatim, once split into words — then the content."""
    raw_name = (name or "").strip()
    lines: list[str] = []
    if raw_name:
        lines.append(raw_name)
        words = filename_words(raw_name)
        lines.append(words or raw_name)
    label = _MIME_LABELS.get((mime_type or "").strip())
    if label:
        lines.append(label)
    if folder_path:
        lines.append(f"In: {str(folder_path).strip()}")
    if content_excerpt:
        lines.append(_clip(content_excerpt, 4000))
    return "\n".join(line for line in lines if line).strip()


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _first(row: Any, *names: str, default: Any = None) -> Any:
    """The first of these fields the row actually has.

    Rows reach this function in two shapes. A stored row uses the unified
    column names — ``title``, ``body``, ``participants`` — while a row still in
    flight from a Google response carries that API's own names. Reading only
    one set is how a whole connector ends up with no embeddings and no error:
    the text comes back empty, the row is "skipped_empty", and the search
    quietly cannot see it.
    """
    for name in names:
        value = _get(row, name)
        if value not in (None, "", [], {}):
            return value
    return default


def embed_text_for_row(table: str, row: Any) -> str:
    """Dispatch to the right builder for a mirror row (ORM object or dict)."""
    key = table.lower()
    if key in {"gmail", "sync_messages", "mail", "message"}:
        return embed_text_for_email(
            _first(row, "subject", "title"),
            _first(row, "body", "body_clean"),
        )
    if key in {"gcal", "sync_events", "calendar", "event"}:
        return embed_text_for_event(
            _first(row, "title", "summary"),
            _first(row, "description", "body"),
            _first(row, "location") or _get_attr(row, "location"),
            _first(row, "attendees", "attendee_emails", "participants"),
            organizer_email=_first(row, "organizer_email", "author_email"),
        )
    if key in {"gdrive", "sync_files", "drive", "file"}:
        return embed_text_for_file(
            _first(row, "name", "title"),
            _first(row, "content", "content_excerpt", "body"),
            _first(row, "mime_type") or _get_attr(row, "mime_type"),
            _first(row, "folder_path") or _get_attr(row, "folder_path"),
        )
    raise ValueError(f"unknown mirror table {table!r}")


def _get_attr(row: Any, name: str) -> Any:
    """A value that moved into the unified ``attributes`` blob."""
    attrs = _get(row, "attributes")
    return attrs.get(name) if isinstance(attrs, Mapping) else None


def email_chunks(
    subject: str | None,
    raw_body: str | None,
    *,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[int, str, str]]:
    """``[(chunk_index, body_clean, embed_text), ...]`` for one message.

    The subject rides on every chunk, so chunk 7 of a long thread is still
    recognisable as part of "Your Turkish Airlines booking".
    """
    body = clean_email(raw_body)
    pieces: Sequence[str] = chunk(body, size, overlap) or [""]
    return [(i, piece, embed_text_for_email(subject, piece)) for i, piece in enumerate(pieces)]


def file_chunks(
    name: str | None,
    content: str | None,
    *,
    mime_type: str | None = None,
    folder_path: str | None = None,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[int, str, str]]:
    """``[(chunk_index, content_excerpt, embed_text), ...]`` for one file."""
    body = normalise_whitespace(content or "")
    pieces: Sequence[str] = chunk(body, size, overlap) or [""]
    return [
        (i, piece, embed_text_for_file(name, piece, mime_type, folder_path))
        for i, piece in enumerate(pieces)
    ]


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "chunk",
    "clean_email",
    "email_chunks",
    "embed_text_for_email",
    "embed_text_for_event",
    "embed_text_for_file",
    "embed_text_for_row",
    "file_chunks",
    "filename_words",
    "normalise_whitespace",
    "strip_html",
]
