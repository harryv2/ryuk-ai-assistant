"""The pre-pass: everything about a query that Python can settle on its own.

Four jobs, all pure functions, no database, no network, no model:

1. **Time.** ``temporal.scan`` pulls every phrase out and resolves it, plus the
   class default (180 days back for a read, 12 months for a write) so a query
   with no date still has a bounded window.
2. **Literals.** Email addresses, URLs, phone numbers, booking references,
   flight numbers, amounts, order ids, quoted phrases. A value the user typed is
   a value nobody has to guess, and a value nobody guesses is a value nobody
   hallucinates.
3. **Aliases.** "Turkish Airlines" becomes a token set, two sender domains and a
   flight-code pattern. Hand-maintained data, not inference: ``thy.com`` is in
   the table because Turkish Airlines uses it, and that fact beats any amount of
   embedding cleverness on a Turkish-language booking email.
4. **Mime words.** "PDF" becomes ``application/pdf``, "spreadsheet" becomes the
   two mime types a spreadsheet can be.

What comes out is a briefing the probe searches with and the planner reads. It
costs about 10 ms and no tokens.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Iterable

from app.core.logging import get_logger
from app.orchestrator import temporal
from app.orchestrator.temporal import Window

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Vendor aliases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasGroup:
    """One brand, and every hard signal that identifies it in a mailbox."""

    name: str
    tokens: tuple[str, ...]
    sender_domains: tuple[str, ...] = ()
    code_patterns: tuple[str, ...] = ()
    support_email: str | None = None
    kind: str = "vendor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_group": self.name,
            "tokens": list(self.tokens),
            "sender_domains": list(self.sender_domains),
            "code_patterns": list(self.code_patterns),
            "support_email": self.support_email,
        }


# The support address a cancellation is written to when the booking email does
# not carry one. Kept here rather than in the shared alias table because it is a
# fact about *writing to* a brand, and that table is about recognising one.
SUPPORT_EMAILS: Final[dict[str, str]] = {
    "turkish_airlines": "cancel@turkishairlines.com",
    "lufthansa": "service@lufthansa.com",
    "emirates": "support@emirates.com",
    "british_airways": "customer.relations@britishairways.com",
    "united_airlines": "customercare@united.com",
    "delta": "charter@delta.com",
    "air_india": "contactus@airindia.com",
    "indigo": "customer.relations@goindigo.in",
    "uber": "support@uber.com",
}

# The fallback table. `app.search.aliases` owns the real one — 700 lines of
# hand-checked surface forms in a YAML file — and this module defers to it
# whenever it can be imported. Two copies of hand-maintained data is the worst
# kind of duplication, because they drift and nobody notices. What is left here
# is enough to keep the pre-pass working, and testable, on its own.
FALLBACK_GROUPS: Final[tuple[AliasGroup, ...]] = (
    AliasGroup(
        "turkish_airlines",
        ("turkish airlines", "turkish air", "thy", "türk hava yolları", "turk hava yollari"),
        ("turkishairlines.com", "thy.com"),
        (r"\bTK\s?\d{1,4}\b",),
        "cancel@turkishairlines.com",
        "airline",
    ),
    AliasGroup(
        "lufthansa",
        ("lufthansa", "lh"),
        ("lufthansa.com",),
        (r"\bLH\s?\d{1,4}\b",),
        "service@lufthansa.com",
        "airline",
    ),
    AliasGroup(
        "emirates",
        ("emirates", "ek"),
        ("emirates.com",),
        (r"\bEK\s?\d{1,4}\b",),
        "support@emirates.com",
        "airline",
    ),
    AliasGroup(
        "british_airways",
        ("british airways", "ba"),
        ("britishairways.com", "ba.com"),
        (r"\bBA\s?\d{1,4}\b",),
        "customer.relations@britishairways.com",
        "airline",
    ),
    AliasGroup(
        "united_airlines",
        ("united airlines", "united"),
        ("united.com",),
        (r"\bUA\s?\d{1,4}\b",),
        "customercare@united.com",
        "airline",
    ),
    AliasGroup(
        "delta",
        ("delta air lines", "delta"),
        ("delta.com",),
        (r"\bDL\s?\d{1,4}\b",),
        "charter@delta.com",
        "airline",
    ),
    AliasGroup(
        "air_india",
        ("air india", "airindia"),
        ("airindia.com", "airindia.in"),
        (r"\bAI\s?\d{1,4}\b",),
        "contactus@airindia.com",
        "airline",
    ),
    AliasGroup(
        "indigo",
        ("indigo", "6e"),
        ("goindigo.in",),
        (r"\b6E\s?\d{1,4}\b",),
        "customer.relations@goindigo.in",
        "airline",
    ),
    AliasGroup(
        "acme",
        ("acme", "acme corp", "acme corporation"),
        ("acmecorp.com", "acme.com"),
        (),
        None,
        "company",
    ),
    AliasGroup("northwind", ("northwind", "northwind traders"), ("northwind.io",), (), None, "company"),
    AliasGroup("amazon", ("amazon", "amzn"), ("amazon.com", "amazon.in"), (r"\b\d{3}-\d{7}-\d{7}\b",), None, "retail"),
    AliasGroup("uber", ("uber",), ("uber.com",), (), "support@uber.com", "travel"),
    AliasGroup("airbnb", ("airbnb",), ("airbnb.com",), (r"\bHM[A-Z0-9]{8}\b",), None, "travel"),
    AliasGroup("booking_com", ("booking.com", "booking com"), ("booking.com",), (), None, "travel"),
    AliasGroup("marriott", ("marriott", "bonvoy"), ("marriott.com",), (), None, "hotel"),
    AliasGroup("stripe", ("stripe",), ("stripe.com",), (r"\b(?:pi|ch|in)_[A-Za-z0-9]{16,}\b",), None, "saas"),
    AliasGroup("github", ("github",), ("github.com",), (), None, "saas"),
    AliasGroup("slack", ("slack",), ("slack.com",), (), None, "saas"),
    AliasGroup("zoom", ("zoom",), ("zoom.us",), (), None, "saas"),
    AliasGroup("linkedin", ("linkedin",), ("linkedin.com",), (), None, "saas"),
    AliasGroup("dropbox", ("dropbox",), ("dropbox.com",), (), None, "saas"),
    AliasGroup("google", ("google workspace", "google"), ("google.com",), (), None, "saas"),
)

_FALLBACK_BY_NAME: Final[dict[str, AliasGroup]] = {g.name: g for g in FALLBACK_GROUPS}

# Kept as a name for callers that only want to read the table.
ALIAS_GROUPS = FALLBACK_GROUPS

# Tokens that are also ordinary English words. They only count as a brand when
# something else in the group also fires, or when the token is multi-word.
_WEAK_TOKENS: Final[frozenset[str]] = frozenset(
    {"united", "delta", "indigo", "google", "zoom", "slack", "ba", "lh", "ek", "ai", "6e", "thy"}
)


# ---------------------------------------------------------------------------
# Mime words
# ---------------------------------------------------------------------------

MIME_MAP: Final[dict[str, tuple[str, ...]]] = {
    "pdf": ("application/pdf",),
    "pdfs": ("application/pdf",),
    "doc": ("application/vnd.google-apps.document", "application/msword"),
    "docs": ("application/vnd.google-apps.document",),
    "document": ("application/vnd.google-apps.document",),
    "documents": ("application/vnd.google-apps.document",),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    "word": (
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "sheet": ("application/vnd.google-apps.spreadsheet",),
    "sheets": ("application/vnd.google-apps.spreadsheet",),
    "spreadsheet": (
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "spreadsheets": ("application/vnd.google-apps.spreadsheet",),
    "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
    "csv": ("text/csv",),
    "slide": ("application/vnd.google-apps.presentation",),
    "slides": ("application/vnd.google-apps.presentation",),
    "deck": ("application/vnd.google-apps.presentation",),
    "decks": ("application/vnd.google-apps.presentation",),
    "presentation": ("application/vnd.google-apps.presentation",),
    "powerpoint": ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
    "pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
    "image": ("image/png", "image/jpeg"),
    "images": ("image/png", "image/jpeg"),
    "photo": ("image/jpeg",),
    "photos": ("image/jpeg",),
    "screenshot": ("image/png",),
    "screenshots": ("image/png",),
    "video": ("video/mp4",),
    "videos": ("video/mp4",),
    "folder": ("application/vnd.google-apps.folder",),
    "folders": ("application/vnd.google-apps.folder",),
    "form": ("application/vnd.google-apps.form",),
    "zip": ("application/zip",),
}


# ---------------------------------------------------------------------------
# Literal extractors
# ---------------------------------------------------------------------------
#
# Structure, not vocabulary. A booking reference is a shape, and shapes survive
# translation — which is why a Turkish confirmation email still gives up its PNR.

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DOMAIN_RE = re.compile(r"(?<![\w@.])@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"')]+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,17}\d(?![\w.])")
_FLIGHT_RE = re.compile(r"\b([A-Z]{2}|[A-Z]\d|\d[A-Z])\s?(\d{1,4})\b")
_ROUTE_RE = re.compile(r"\b([A-Z]{3})\s*(?:→|->|—|–|-|to)\s*([A-Z]{3})\b")
_AMOUNT_RE = re.compile(
    r"(?:(?P<cur1>USD|EUR|GBP|INR|TRY|CAD|AUD|JPY|\$|€|£|₹|₺)\s?(?P<n1>[\d][\d,.\s]*\d|\d)"
    r"|(?P<n2>[\d][\d,.\s]*\d|\d)\s?(?P<cur2>USD|EUR|GBP|INR|TRY|CAD|AUD|JPY))",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(r"\b(?:order|invoice|receipt|ref|reference|booking|confirmation)\s*#?\s*([A-Z0-9][A-Z0-9-]{4,})\b", re.IGNORECASE)
_PNR_RE = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b")
_QUOTED_RE = re.compile(r"[\"“']([^\"”']{2,80})[\"”']")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_GDRIVE_ID_RE = re.compile(r"\b[A-Za-z0-9_-]{25,44}\b")
_LABEL_WORDS: Final[dict[str, str]] = {
    "unread": "UNREAD",
    "starred": "STARRED",
    "important": "IMPORTANT",
    "sent": "SENT",
    "draft": "DRAFT",
    "drafts": "DRAFT",
    "spam": "SPAM",
    "archived": "ARCHIVE",
    "inbox": "INBOX",
}

# Verbs that mean the turn intends to change something. Only a hint: the
# planner decides `has_write`, and the validator enforces it.
#
# Split in two because half of these are also nouns. "That email about the
# proposal" is a lookup; "email Bob" is a send. A weak verb only counts when it
# stands where a verb stands.
_WRITE_VERBS_STRONG: Final[frozenset[str]] = frozenset(
    {
        "send", "reply", "forward", "compose", "cancel", "unsubscribe",
        "decline", "reschedule", "postpone", "rsvp", "delete",
    }
)

_WRITE_VERBS_WEAK: Final[frozenset[str]] = frozenset(
    {
        "email", "draft", "write", "remove", "move", "push", "shift", "book",
        "schedule", "create", "add", "invite", "share", "rename", "archive",
        "star", "accept", "confirm", "update", "change", "set",
    }
)

# What a weak verb may follow and still be a verb.
_VERB_LEAD: Final[frozenset[str]] = frozenset(
    {"please", "pls", "can", "could", "would", "you", "and", "then", "also",
     "to", "i", "want", "need", "let", "lets", "go", "ahead", "now", "just"}
)

_DEIXIS_RE = re.compile(
    r"\b(?P<marker>that|those|this|these|the same|the one|the first|the second|the last|"
    r"the above|the previous|the earlier|it|them)\b"
    r"(?:\s+(?P<noun>email|message|thread|meeting|event|invite|file|doc|document|"
    r"deck|sheet|person|guy|one|proposal))?",
    re.IGNORECASE,
)

_CAPABILITY_NOUNS: Final[frozenset[str]] = frozenset(
    {"email", "emails", "mail", "inbox", "message", "messages", "calendar", "event",
     "events", "meeting", "meetings", "file", "files", "doc", "docs", "drive"}
)


def _norm(text: str) -> str:
    """Casefold, collapse whitespace, strip accents for token comparison."""
    folded = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _clean_amount(raw: str) -> str:
    """``812,40`` and ``1 234.50`` both become a plain decimal string."""
    body = raw.replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", body):  # 1.234,50
        body = body.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d{2}", body):  # 812,40
        body = body.replace(",", ".")
    else:
        body = body.replace(",", "")
    return body


def extract_literals(text: str) -> dict[str, list[str]]:
    """Every value the text states outright, keyed by what kind of value it is.

    Used on the query here, and on candidate excerpts by the probe. Same
    regexes either way — a PNR in a subject line and a PNR the user typed are
    the same shape.

    Keys are plural, and booking references, order numbers and flight codes all
    land under ``refs``: those are the names `app.search.probe` reads when it
    turns the pre-pass into search terms.
    """
    found: dict[str, list[str]] = {}

    def add(kind: str, value: str | None) -> None:
        if not value:
            return
        value = value.strip()
        bucket = found.setdefault(kind, [])
        if value and value not in bucket:
            bucket.append(value)

    if not text:
        return found

    for match in _EMAIL_RE.finditer(text):
        add("emails", match.group(0).lower())
    for match in _DOMAIN_RE.finditer(text):
        add("domains", match.group(1).lower())
    for match in _URL_RE.finditer(text):
        add("urls", match.group(0))
    for match in _ISO_DATE_RE.finditer(text):
        add("iso_dates", match.group(0))
    for match in _QUOTED_RE.finditer(text):
        add("quoted", match.group(1))
    for match in _ORDER_RE.finditer(text):
        add("refs", match.group(1).upper())
    for match in _ROUTE_RE.finditer(text):
        add("routes", f"{match.group(1)}→{match.group(2)}")

    # A phone number is only a phone number when it is not part of a URL or an
    # ISO timestamp we already claimed.
    claimed = [m.span() for m in _URL_RE.finditer(text)] + [
        m.span() for m in _ISO_DATE_RE.finditer(text)
    ]
    for match in _PHONE_RE.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in claimed):
            continue
        digits = re.sub(r"\D", "", match.group(0))
        if 8 <= len(digits) <= 15:
            add("phones", match.group(0).strip())

    upper = text.upper()
    for match in _FLIGHT_RE.finditer(upper):
        add("flight_nos", f"{match.group(1)}{match.group(2)}")
    for match in _PNR_RE.finditer(upper):
        value = match.group(0)
        if value in found.get("flight_nos", []):
            continue
        add("refs", value)

    for match in _AMOUNT_RE.finditer(text):
        currency = (match.group("cur1") or match.group("cur2") or "").upper()
        number = match.group("n1") or match.group("n2") or ""
        symbols = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "₺": "TRY"}
        currency = symbols.get(currency, currency)
        cleaned = _clean_amount(number)
        if cleaned and re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            add("amounts", f"{currency} {cleaned}".strip())

    return found


def _shared_table() -> Any | None:
    """`app.search.aliases`, when it is importable. It owns the real table."""
    try:
        from app.search import aliases as table

        return table
    except Exception:  # noqa: BLE001 - the pre-pass works without it
        return None


def _adopt(group: Any) -> AliasGroup:
    """One of the shared table's groups, in the shape this module returns."""
    key = str(getattr(group, "key", "") or getattr(group, "canonical", ""))
    return AliasGroup(
        name=key,
        tokens=tuple(getattr(group, "tokens", ()) or ()),
        sender_domains=tuple(getattr(group, "sender_domains", ()) or ()),
        code_patterns=tuple(getattr(group, "code_patterns", ()) or ()),
        support_email=SUPPORT_EMAILS.get(key),
        kind=str(getattr(group, "kind", "vendor") or "vendor"),
    )


def match_aliases(text: str) -> list[AliasGroup]:
    """Which brands the text names, best match first.

    Delegates to the shared alias table, which is the one somebody maintains.
    The local table below only runs when that import is unavailable.
    """
    if not (text or "").strip():
        return []

    found: list[AliasGroup] = []
    table = _shared_table()
    if table is not None:
        try:
            found = [_adopt(group) for group in table.detect(text)]
        except Exception as exc:  # noqa: BLE001 - a bad table is not a bad turn
            log.warning("prepass.alias_table_failed", error=str(exc))

    # Then anything the local table knows and the shared one does not. Merging
    # rather than replacing means a group added here still works, and a group
    # in both is read from the file somebody maintains.
    known = {group.name for group in found}
    for group in _match_fallback(text):
        if group.name not in known:
            found.append(group)
    return found


def _match_fallback(text: str) -> list[AliasGroup]:
    """The built-in table. Weak tokens need a second signal to count."""
    normalized = _norm(text)
    if not normalized:
        return []

    hits: list[AliasGroup] = []
    for group in FALLBACK_GROUPS:
        strong = False
        weak = False
        for token in group.tokens:
            token_norm = _norm(token)
            if not token_norm:
                continue
            if not re.search(rf"(?<!\w){re.escape(token_norm)}(?!\w)", normalized):
                continue
            if token_norm in _WEAK_TOKENS and " " not in token_norm:
                weak = True
            else:
                strong = True
        if not strong and weak:
            # An ordinary English word only counts as a brand when the query
            # also carries the group's code shape or one of its domains.
            strong = any(re.search(p, text, re.IGNORECASE) for p in group.code_patterns) or any(
                domain in normalized for domain in group.sender_domains
            )
        if strong:
            hits.append(group)
    return hits


def mime_types(text: str) -> list[str]:
    """Mime types named by the words in the query, in first-seen order."""
    out: list[str] = []
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        for mime in MIME_MAP.get(word, ()):
            if mime not in out:
                out.append(mime)
    return out


def labels(text: str) -> list[str]:
    """Gmail labels a query names outright: unread, starred, in my inbox."""
    out: list[str] = []
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        label = _LABEL_WORDS.get(word)
        if label and label not in out:
            out.append(label)
    return out


def deixis(text: str) -> list[dict[str, str]]:
    """References to something already on screen: "that email", "the one you found"."""
    out: list[dict[str, str]] = []
    for match in _DEIXIS_RE.finditer(text or ""):
        marker = match.group("marker").lower()
        noun = (match.group("noun") or "").lower()
        if marker in ("it", "them") and not noun:
            # Too weak on its own — "send it" is a UI verb, not a lookup.
            continue
        out.append({"marker": marker, "noun": noun, "phrase": match.group(0).strip()})
    return out


def looks_like_write(text: str) -> bool:
    """A cheap hint used only to pick the default window class."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    if set(words) & _WRITE_VERBS_STRONG:
        return True
    for index, word in enumerate(words):
        if word not in _WRITE_VERBS_WEAK:
            continue
        if index == 0 or words[index - 1] in _VERB_LEAD:
            return True
    return False


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class PrePass:
    """Everything Python worked out before anything was searched or called."""

    query: str
    normalized: str
    tz: str
    week_start: int
    now: datetime
    windows: dict[str, Window] = field(default_factory=dict)
    literals: dict[str, list[str]] = field(default_factory=dict)
    alias_groups: list[AliasGroup] = field(default_factory=list)
    mime_types: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    deixis: list[dict[str, str]] = field(default_factory=list)
    kind: str = "read"
    took_ms: float = 0.0

    # -- what the probe reads ----------------------------------------------
    #
    # `app.search.probe.Prepass.of` adapts this object rather than importing
    # it, and reads plain dicts and strings. These three properties are that
    # handover, named the way it looks for them.

    @property
    def aliases(self) -> list[dict[str, Any]]:
        """The matched brands as plain dicts — tokens, domains, code patterns."""
        return [group.to_dict() for group in self.alias_groups]

    @property
    def search_text(self) -> str:
        """The query with the parts already turned into filters taken out.

        Time phrases are windows now, and brand names are token lists and
        sender domains. What is left is the topic — the only part a vector
        search can actually help with.
        """
        text = self.query
        for phrase in temporal.find_phrases(
            self.query, self.tz, self.week_start, self.now
        ):
            text = text.replace(phrase.text, " ")
        for group in self.alias_groups:
            for token in sorted(group.tokens, key=len, reverse=True):
                text = re.sub(re.escape(token), " ", text, flags=re.IGNORECASE)
        for address in self.literals.get("emails", ()):
            text = text.replace(address, " ")
        return re.sub(r"\s+", " ", text).strip()

    # -- convenience --------------------------------------------------------

    @property
    def emails(self) -> list[str]:
        return list(self.literals.get("emails", ()))

    @property
    def default_window(self) -> Window | None:
        return self.windows.get("default_read") or self.windows.get("write_default")

    @property
    def named_windows(self) -> dict[str, Window]:
        """Windows the user actually named, without the class default."""
        return {
            name: window
            for name, window in self.windows.items()
            if name not in ("default_read", "write_default")
        }

    def alias_tokens(self) -> list[str]:
        out: list[str] = []
        for group in self.alias_groups:
            for token in group.tokens:
                if token not in out:
                    out.append(token)
        return out

    def sender_domains(self) -> list[str]:
        out: list[str] = []
        for group in self.alias_groups:
            for domain in group.sender_domains:
                if domain not in out:
                    out.append(domain)
        for domain in self.literals.get("domains", ()):
            if domain not in out:
                out.append(domain)
        for address in self.literals.get("emails", ()):
            domain = address.split("@", 1)[-1]
            if domain and domain not in out:
                out.append(domain)
        return out

    def support_emails(self) -> list[str]:
        return [g.support_email for g in self.alias_groups if g.support_email]

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "kind": self.kind,
            "windows": {name: window.to_dict() for name, window in self.windows.items()},
            "literals": {k: list(v) for k, v in self.literals.items()},
            "aliases": self.aliases,
            "search_text": self.search_text,
            "mime_types": list(self.mime_types),
            "labels": list(self.labels),
            "deixis": list(self.deixis),
            "took_ms": round(self.took_ms, 2),
        }

    def to_prompt(self) -> dict[str, Any]:
        """The trimmed shape that goes into the planner's volatile half."""
        body: dict[str, Any] = {}
        if self.windows:
            body["windows"] = {
                name: {
                    "start": window.to_dict()["start"],
                    "end": window.to_dict()["end"],
                    "interpretation": window.interpretation,
                }
                for name, window in self.windows.items()
            }
        if self.literals:
            body["literals"] = {k: v for k, v in self.literals.items() if v}
        if self.alias_groups:
            body["aliases"] = self.aliases
        if self.mime_types:
            body["mime_types"] = self.mime_types
        if self.labels:
            body["labels"] = self.labels
        if self.deixis:
            body["refers_to_something_earlier"] = [d["phrase"] for d in self.deixis]
        return body


def run(
    query: str,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
    *,
    kind: str | None = None,
) -> PrePass:
    """The whole pre-pass. Pure Python, no I/O, about ten milliseconds."""
    started = time.perf_counter()
    text = query or ""
    moment = now or datetime.now(UTC)

    windows = temporal.scan(text, tz, week_start, moment)
    write_ish = looks_like_write(text) if kind is None else kind == "write"
    resolved_kind = "write" if write_ish else "read"

    # A query that names no window still gets one, so a search is always
    # bounded and the answer can say what it covered.
    if not windows:
        name = "write_default" if write_ish else "default_read"
        windows = {name: temporal.default_window(resolved_kind, tz, moment)}

    result = PrePass(
        query=text,
        normalized=_norm(text),
        tz=tz,
        week_start=int(week_start or 1),
        now=moment,
        windows=windows,
        literals=extract_literals(text),
        alias_groups=match_aliases(text),
        mime_types=mime_types(text),
        labels=labels(text),
        deixis=deixis(text),
        kind=resolved_kind,
    )
    result.took_ms = (time.perf_counter() - started) * 1000.0
    return result


def alias_group(name: str) -> AliasGroup | None:
    """One group by name — the ladder's rung 1 reads its domains and codes.

    The shared table wins when it has the group, for the same reason
    :func:`match_aliases` prefers it: it is the copy somebody maintains.
    """
    if not name:
        return None
    table = _shared_table()
    if table is not None:
        try:
            group = table.groups().get(name) or table.find(name)
        except Exception as exc:  # noqa: BLE001 - a bad table is not a bad turn
            log.warning("prepass.alias_table_failed", error=str(exc))
        else:
            if group is not None:
                return _adopt(group)
    return _FALLBACK_BY_NAME.get(name)


def expand(terms: Iterable[str]) -> list[str]:
    """Every alias token for every group any of ``terms`` names."""
    out: list[str] = []
    for term in terms:
        for group in match_aliases(term):
            for token in group.tokens:
                if token not in out:
                    out.append(token)
    return out


__all__ = [
    "ALIAS_GROUPS",
    "MIME_MAP",
    "AliasGroup",
    "PrePass",
    "alias_group",
    "deixis",
    "expand",
    "extract_literals",
    "labels",
    "looks_like_write",
    "match_aliases",
    "mime_types",
    "run",
]
