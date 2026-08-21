"""Deterministic extraction over candidate excerpts.

This module is the reason the planner never types a booking reference. The
probe finds the email; regex pulls `6F2QK9` out of it in about two
milliseconds; the plan says ``{{search.gmail[0].extracted.pnr}}``. A value the
model never held is a value the model cannot invent.

Every rule is anchored. A bare six-character token is not a PNR — half the
words in a mailbox are six characters. It is a PNR when it sits within 60
characters of a word that means *booking reference*, in English or in Turkish,
which is what ``rezervasyon kodu`` and ``bilet no`` are doing in the cue list.

Each match carries the span it came from, so a card can quote the sentence that
produced a value, and a wrong extraction can be traced to the text that fooled
it rather than argued about.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta  # noqa: F401  (dateutil dependency check)

# --------------------------------------------------------------------------- #
# Cues
# --------------------------------------------------------------------------- #

#: Within this many characters of a cue word, a token may be a reference.
CUE_WINDOW = 60

_PNR_CUES = re.compile(
    r"\b("
    r"booking(?:\s+(?:reference|ref|code|number|no))?"
    r"|reservation(?:\s+(?:code|number|no))?"
    r"|reference|ref|pnr|record\s+locator|confirmation(?:\s+(?:code|number|no))?"
    r"|conf(?:irmation)?\s*#"
    # Turkish glues its suffixes on: "rezervasyonunuz" is "your reservation".
    # Requiring a word boundary right after the stem misses every real email.
    r"|rezervasyon\w*|bilet\s*(?:no|numaras[ıi])?|pnr\s*kodu"  # noqa: RUF001
    # "6F2QK9 numaralı" — the cue TRAILS the code here, which _near_cue  # noqa: RUF003
    # already handles because it looks both ways.
    r"|numaral[ıi]\w*|numaras[ıi]\w*|onay\s*kodu"  # noqa: RUF001
    r"|buchungs(?:nummer|code)|r[ée]servation|localizador"
    r")\b",
    re.I,
)

_ORDER_CUES = re.compile(
    r"\b(order|order\s*(?:number|no|id|#)|sipari[şs]|invoice|invoice\s*(?:number|no|#)"
    r"|fatura|receipt|receipt\s*(?:number|no|#)|transaction\s*(?:id|number|no)"
    r"|payment\s*(?:id|reference))\b",
    re.I,
)

_TICKET_CUES = re.compile(
    r"\b(ticket(?:\s*(?:number|no|#))?|e-?ticket|bilet\s*(?:no|numaras[ıi])?|"  # noqa: RUF001
    r"document\s*number)\b",
    re.I,
)

_SUPPORT_LOCALS = (
    "support", "help", "helpdesk", "cancel", "cancellation", "refunds", "refund",
    "reservations", "booking", "bookings", "customercare", "customer.care",
    "customerservice", "customer.service", "care", "service", "contact",
    "destek", "iletisim", "musteri", "rezervasyon",
)

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

# Six alphanumerics, upper case, at least one digit and at least one letter.
# Airline record locators are exactly six and never contain punctuation.
_PNR_TOKEN = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")

# Two letters and one to four digits: TK1984, TK 1984, LH 400.
_FLIGHT_NO = re.compile(r"\b([A-Z]{2})\s?(\d{1,4})(?![A-Za-z0-9])")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}\b")

_URL = re.compile(r"\bhttps?://[^\s<>\"')\]]+", re.I)

_GOOGLE_DOC = re.compile(
    r"\bhttps?://docs\.google\.com/(document|spreadsheets|presentation|forms)/d/"
    r"(?:e/)?([A-Za-z0-9_\-]{10,})[^\s<>\"')\]]*",
    re.I,
)
_GOOGLE_DRIVE_FILE = re.compile(
    r"\bhttps?://drive\.google\.com/(?:file/d/|open\?id=|drive/folders/)"
    r"([A-Za-z0-9_\-]{10,})[^\s<>\"')\]]*",
    re.I,
)

_CURRENCY_SYMBOLS = {
    "$": "USD", "€": "EUR", "£": "GBP", "₺": "TRY", "₹": "INR", "¥": "JPY",
}
_CURRENCY_CODES = (
    "USD", "EUR", "GBP", "TRY", "INR", "JPY", "CHF", "CAD", "AUD", "AED", "SGD",
    "SEK", "NOK", "DKK", "PLN", "QAR", "SAR",
)
_AMOUNT = re.compile(
    r"(?P<sym>[$€£₺₹¥])\s?(?P<n1>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)"
    r"|(?P<code>" + "|".join(_CURRENCY_CODES) + r")\s?(?P<n2>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)"
    r"|(?P<n3>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?)\s?(?P<code2>"
    + "|".join(_CURRENCY_CODES) + r")\b",
    re.I,
)

_ORDER_TOKEN = re.compile(r"\b(?=[A-Z0-9][A-Z0-9\-]{3,29}\b)(?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{3,29}\b")

_TICKET_13 = re.compile(r"\b\d{3}[\-\s]?\d{10}\b")

# Route, two shapes: "IST → JFK" and "Istanbul (IST) to New York (JFK)".
_ROUTE = re.compile(r"\b([A-Z]{3})\s?(?:→|->|—|–|-|to|/)\s?([A-Z]{3})\b")  # noqa: RUF001
_ROUTE_PARENS = re.compile(r"\(([A-Z]{3})\)[^()]{0,40}?\(([A-Z]{3})\)")

_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{7,17}\d(?![\w])")

_ISO_DATETIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r"(?:Z|[+\-]\d{2}:?\d{2})?)?\b"
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    "|ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|ağustos|agustos"  # noqa: RUF001
    "|eylül|eylul|ekim|kasım|kasim|aralık|aralik"  # noqa: RUF001
)
_TEXT_DATE = re.compile(
    r"\b(?:\d{1,2}\s+(?:" + _MONTHS + r")|(?:" + _MONTHS + r")\s+\d{1,2})"
    r"(?:,?\s+\d{4})?(?:\s+(?:at\s+)?\d{1,2}[:.]\d{2}\s*(?:am|pm)?)?\b",
    re.I,
)
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}(?:\s+\d{1,2}:\d{2})?\b")

_TIMEZONE_HINT = re.compile(r"\b(UTC|GMT|EST|EDT|PST|PDT|CET|CEST|IST|TRT)\b")

#: dateutil speaks English. A booking confirmation from Istanbul does not, and
#: the date in it is the whole point of reading the email.
_MONTH_TRANSLATIONS: dict[str, str] = {
    "ocak": "January", "şubat": "February", "subat": "February", "mart": "March",
    "nisan": "April", "mayıs": "May", "mayis": "May", "haziran": "June",  # noqa: RUF001
    "temmuz": "July", "ağustos": "August", "agustos": "August",
    "eylül": "September", "eylul": "September", "ekim": "October",
    "kasım": "November", "kasim": "November", "aralık": "December", "aralik": "December",  # noqa: RUF001
}
_TRANSLATABLE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_TRANSLATIONS, key=len, reverse=True)) + r")\b", re.I
)


def _to_english_months(raw: str) -> str:
    """Swap a non-English month name for the English one, or return as-is."""
    return _TRANSLATABLE.sub(
        lambda m: _MONTH_TRANSLATIONS[m.group(0).lower()], raw
    )

#: Fields whose value is a list rather than a single best match.
LIST_FIELDS: frozenset[str] = frozenset(
    {"emails", "dates", "amounts", "links", "doc_links", "drive_file_ids",
     "flight_nos", "phones", "codes"}
)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Match:
    """One extracted value and the text it came from.

    ``span`` indexes the excerpt that was passed in, so a caller can highlight
    it. ``context`` is the surrounding sentence fragment, which is what a
    confirm card shows when it says "found in: …".
    """

    field: str
    value: Any
    start: int
    end: int
    text: str
    rule: str
    context: str = ""

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "span": [self.start, self.end],
            "text": self.text,
            "rule": self.rule,
            "context": self.context,
        }


@dataclass
class Extraction:
    """Everything the rules found in one excerpt."""

    values: dict[str, Any] = field(default_factory=dict)
    matches: dict[str, list[Match]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.values)

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def span(self, name: str) -> tuple[int, int] | None:
        found = self.matches.get(name)
        return found[0].span if found else None

    def provenance(self) -> dict[str, dict[str, Any]]:
        """``field -> {value, span, context}``, for the step trace."""
        return {
            name: found[0].as_dict()
            for name, found in self.matches.items()
            if found
        }

    def as_dict(self) -> dict[str, Any]:
        """Plain values, which is what hangs off ``candidate.extracted``."""
        return dict(self.values)


def _context(text: str, start: int, end: int, width: int = 60) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = text[left:right].replace("\n", " ").strip()
    return re.sub(r"\s{2,}", " ", snippet)


def _near_cue(text: str, start: int, end: int, cues: re.Pattern[str],
              window: int = CUE_WINDOW) -> re.Match[str] | None:
    """The cue word nearest a candidate token, or None if there is not one."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    best: re.Match[str] | None = None
    best_distance = window + 1
    for cue in cues.finditer(text, left, right):
        if cue.start() >= end:
            distance = cue.start() - end
        elif cue.end() <= start:
            distance = start - cue.end()
        else:
            distance = 0
        if distance <= window and distance < best_distance:
            best, best_distance = cue, distance
    return best


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #

#: Six-letter-ish tokens that are words, not record locators.
_PNR_STOPWORDS = frozenset({
    "TICKET", "FLIGHT", "AIRPORT", "PLEASE", "NUMBER", "BOOKED", "TRAVEL", "AMOUNT",
    "DEPART", "ARRIVE", "CANCEL", "REFUND", "STATUS", "ONLINE", "MOBILE", "SEATED",
})


#: Cues that mean *this exact thing is the record locator*, as opposed to
#: "booking", which merely means the email is about one.
_STRONG_PNR_CUE = re.compile(
    r"\b(pnr|record\s+locator|booking\s+(?:reference|ref|code)|reservation\s+code"
    r"|confirmation\s+(?:code|number|no)|rezervasyon\s+kodu|localizador|buchungscode)\b",
    re.I,
)
_LOOKS_LIKE_FLIGHT = re.compile(r"^[A-Z]{2}\d{1,4}$")


def _pnr_score(value: str, cue: re.Match[str], distance: int) -> int:
    """How much this six-character token looks like a record locator.

    A booking email says "booking" three times and carries a flight number, a
    ticket number and a locator, all of which are six characters of letters and
    digits. The cue that is nearest and most specific wins, and anything shaped
    like a flight number loses.
    """
    score = 3 if _STRONG_PNR_CUE.fullmatch(cue.group(0)) else 1
    if distance <= 12:
        score += 2
    if _LOOKS_LIKE_FLIGHT.match(value):
        score -= 3
    if value.isdigit():
        score -= 1
    return score


def find_pnr(text: str) -> list[Match]:
    """Six alphanumerics near a booking cue, in English or Turkish.

    Best candidate first, not first occurrence: the subject line's flight code
    sits next to the word "booking" too, and it is not the locator.
    """
    scored: list[tuple[int, int, Match]] = []
    for token in _PNR_TOKEN.finditer(text):
        value = token.group(0)
        if value in _PNR_STOPWORDS:
            continue
        cue = _near_cue(text, token.start(), token.end(), _PNR_CUES)
        if cue is None:
            continue
        distance = max(0, token.start() - cue.end(), cue.start() - token.end())
        scored.append(
            (
                _pnr_score(value, cue, distance),
                -token.start(),
                Match(
                    field="pnr",
                    value=value,
                    start=token.start(),
                    end=token.end(),
                    text=value,
                    rule=f"six alphanumerics within {CUE_WINDOW} chars of {cue.group(0)!r}",
                    context=_context(text, token.start(), token.end()),
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [match for _, _, match in scored]


#: A flight number and a ticket number are the same shape. Only the words
#: around them tell you which is which, so the cue decides, not the position.
_FLIGHT_CUES = re.compile(
    r"\b(flight|flt|u[çc]u[şs]\w*|sefer\w*|vuelo|vol|flug)\b",
    re.I,
)


def _flight_confidence(text: str, start: int, end: int) -> int:
    """Higher is more likely to be the flight, not the ticket.

    The English confirmation carries TK1984 in the subject next to "booking"
    and TK1 in the body right after "Flight". Reading in document order picks
    the subject, which puts the ticket number in the draft email. The cue is
    the only thing that separates them.
    """
    score = 0
    if _near_cue(text, start, end, _FLIGHT_CUES):
        score += 3
    if _near_cue(text, start, end, _TICKET_CUES):
        score -= 4
    if _near_cue(text, start, end, _PNR_CUES):
        score -= 1
    return score


def find_flight_no(text: str) -> list[Match]:
    """Airline code plus number. ``TK1984``, ``LH 400``.

    Best candidate first, not first occurrence — see :func:`_flight_confidence`.
    """
    out: list[Match] = []
    seen: set[str] = set()
    for hit in _FLIGHT_NO.finditer(text):
        carrier, number = hit.group(1), hit.group(2)
        value = f"{carrier}{int(number)}" if number.isdigit() else f"{carrier}{number}"
        # A three-letter airport code followed by a number is not a flight.
        if carrier in {"PM", "AM"} and len(number) <= 2:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(
            Match(
                field="flight_no",
                value=value,
                start=hit.start(),
                end=hit.end(),
                text=hit.group(0),
                rule="two letters plus one to four digits",
                context=_context(text, hit.start(), hit.end()),
            )
        )
    out.sort(key=lambda m: -_flight_confidence(text, m.start, m.end))
    return out


def find_emails(text: str) -> list[Match]:
    out: list[Match] = []
    seen: set[str] = set()
    for hit in _EMAIL.finditer(text):
        value = hit.group(0).lower().rstrip(".")
        if value in seen:
            continue
        seen.add(value)
        out.append(
            Match(
                field="emails",
                value=value,
                start=hit.start(),
                end=hit.end(),
                text=hit.group(0),
                rule="rfc-shaped address",
                context=_context(text, hit.start(), hit.end()),
            )
        )
    return out


def find_links(text: str) -> tuple[list[Match], list[Match], list[Match]]:
    """``(all urls, docs.google.com documents, drive file ids)``."""
    urls: list[Match] = []
    docs: list[Match] = []
    drive: list[Match] = []
    seen: set[str] = set()
    for hit in _URL.finditer(text):
        value = hit.group(0).rstrip(".,);]")
        if value in seen:
            continue
        seen.add(value)
        urls.append(
            Match(field="links", value=value, start=hit.start(), end=hit.end(),
                  text=value, rule="http(s) url",
                  context=_context(text, hit.start(), hit.end()))
        )
    for hit in _GOOGLE_DOC.finditer(text):
        value = hit.group(0).rstrip(".,);]")
        docs.append(
            Match(
                field="doc_links",
                value={"kind": hit.group(1), "id": hit.group(2), "url": value},
                start=hit.start(),
                end=hit.end(),
                text=value,
                rule="docs.google.com document link",
                context=_context(text, hit.start(), hit.end()),
            )
        )
    for hit in _GOOGLE_DRIVE_FILE.finditer(text):
        drive.append(
            Match(
                field="drive_file_ids",
                value=hit.group(1),
                start=hit.start(),
                end=hit.end(),
                text=hit.group(0).rstrip(".,);]"),
                rule="drive.google.com file link",
                context=_context(text, hit.start(), hit.end()),
            )
        )
    return urls, docs, drive


def _clean_number(raw: str) -> float | None:
    """``1.234,56`` and ``1,234.56`` both mean 1234.56."""
    text = raw.strip().replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        decimals = len(text) - text.rfind(",") - 1
        text = text.replace(",", "." if decimals in (1, 2) else "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def find_amounts(text: str) -> list[Match]:
    """Money, normalised to ``{"currency": "USD", "value": 812.4}``."""
    out: list[Match] = []
    for hit in _AMOUNT.finditer(text):
        groups = hit.groupdict()
        if groups.get("sym"):
            currency = _CURRENCY_SYMBOLS[groups["sym"]]
            number = groups["n1"]
        elif groups.get("code"):
            currency = groups["code"].upper()
            number = groups["n2"]
        elif groups.get("code2"):
            currency = groups["code2"].upper()
            number = groups["n3"]
        else:
            continue
        value = _clean_number(number or "")
        if value is None:
            continue
        out.append(
            Match(
                field="amounts",
                value={"currency": currency, "value": value,
                       "display": f"{currency} {value:,.2f}"},
                start=hit.start(),
                end=hit.end(),
                text=hit.group(0),
                rule="currency symbol or ISO code beside a number",
                context=_context(text, hit.start(), hit.end()),
            )
        )
    return out


def find_order_no(text: str) -> list[Match]:
    """An order, invoice or receipt number, anchored on its cue word."""
    out: list[Match] = []
    seen: set[str] = set()
    for cue in _ORDER_CUES.finditer(text):
        window = text[cue.end() : min(len(text), cue.end() + CUE_WINDOW)]
        for token in _ORDER_TOKEN.finditer(window):
            value = token.group(0)
            if value.upper() in {"NUMBER", "NUMBERS"} or value in seen:
                continue
            seen.add(value)
            start = cue.end() + token.start()
            end = cue.end() + token.end()
            label = cue.group(0).lower()
            name = "invoice_no" if label.startswith(("invoice", "fatura")) else "order_no"
            out.append(
                Match(
                    field=name,
                    value=value,
                    start=start,
                    end=end,
                    text=value,
                    rule=f"token within {CUE_WINDOW} chars after {cue.group(0)!r}",
                    context=_context(text, start, end),
                )
            )
            break
    return out


def find_ticket_no(text: str) -> list[Match]:
    """An airline ticket number: the 13-digit form, or a code near a cue."""
    out: list[Match] = []
    for hit in _TICKET_13.finditer(text):
        out.append(
            Match(field="ticket_no", value=re.sub(r"[\s\-]", "", hit.group(0)),
                  start=hit.start(), end=hit.end(), text=hit.group(0),
                  rule="13-digit IATA ticket number",
                  context=_context(text, hit.start(), hit.end()))
        )
    if out:
        return out
    for cue in _TICKET_CUES.finditer(text):
        window_end = min(len(text), cue.end() + CUE_WINDOW)
        window = text[cue.end() : window_end]
        token = re.search(r"\b(?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{3,19}\b", window)
        if token is None:
            continue
        start = cue.end() + token.start()
        end = cue.end() + token.end()
        out.append(
            Match(field="ticket_no", value=token.group(0), start=start, end=end,
                  text=token.group(0),
                  rule=f"token within {CUE_WINDOW} chars after {cue.group(0)!r}",
                  context=_context(text, start, end))
        )
        break
    return out


def find_route(text: str) -> list[Match]:
    """``IST → JFK``. Both halves must be plausible airport codes."""
    out: list[Match] = []
    for pattern, rule in (
        (_ROUTE, "two three-letter codes joined by an arrow or dash"),
        (_ROUTE_PARENS, "two parenthesised three-letter codes in one sentence"),
    ):
        for hit in pattern.finditer(text):
            origin, destination = hit.group(1), hit.group(2)
            if origin == destination:
                continue
            out.append(
                Match(
                    field="route",
                    value=f"{origin}→{destination}",
                    start=hit.start(),
                    end=hit.end(),
                    text=hit.group(0),
                    rule=rule,
                    context=_context(text, hit.start(), hit.end()),
                )
            )
    out.sort(key=lambda m: m.start)
    return out


def find_phones(text: str) -> list[Match]:
    out: list[Match] = []
    seen: set[str] = set()
    for hit in _PHONE.finditer(text):
        raw = hit.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if not 9 <= len(digits) <= 15:
            continue
        # An unbroken run of digits is a ticket or an order number. A phone
        # number in an email is written with separators or a country code.
        if raw.isdigit():
            continue
        value = ("+" if raw.startswith("+") else "") + digits
        if value in seen:
            continue
        seen.add(value)
        out.append(
            Match(field="phones", value=value, start=hit.start(), end=hit.end(),
                  text=hit.group(0).strip(), rule="9 to 15 digits with phone punctuation",
                  context=_context(text, hit.start(), hit.end()))
        )
    return out


def find_dates(text: str, *, now: dt.datetime | None = None,
               tz: dt.tzinfo | None = None) -> list[Match]:
    """Dates and datetimes, parsed by dateutil, returned as ISO strings.

    The default for missing parts is midnight on the first of the current
    month at ``now``, so "5 November" in a booking read in August 2026 comes
    back as 2026-11-05 rather than today's day-of-month. A date with no
    timezone stays naive-in-ISO — this module never guesses an offset, because
    guessing one is how a flight ends up three hours out.
    """
    anchor = now or dt.datetime.now(dt.UTC)
    default = dt.datetime(anchor.year, anchor.month, 1, 0, 0, tzinfo=tz)
    out: list[Match] = []
    seen: set[tuple[int, int]] = set()

    for pattern, rule in (
        (_ISO_DATETIME, "iso 8601"),
        (_TEXT_DATE, "written date"),
        (_NUMERIC_DATE, "numeric date"),
    ):
        for hit in pattern.finditer(text):
            key = (hit.start(), hit.end())
            if any(s <= hit.start() < e for s, e in seen):
                continue
            raw = hit.group(0).strip()
            try:
                parsed = date_parser.parse(
                    _to_english_months(raw), default=default, fuzzy=False
                )
            except (ValueError, OverflowError, TypeError):
                continue
            tail = text[hit.end() : hit.end() + 12]
            zone = _TIMEZONE_HINT.search(tail)
            seen.add(key)
            out.append(
                Match(
                    field="dates",
                    value=parsed.isoformat(),
                    start=hit.start(),
                    end=hit.end(),
                    text=raw + (f" {zone.group(0)}" if zone else ""),
                    rule=rule,
                    context=_context(text, hit.start(), hit.end()),
                )
            )
    out.sort(key=lambda m: m.start)
    return out


def find_support_email(emails: Sequence[Match], text: str) -> Match | None:
    """The address a cancellation should go to, if one is in the excerpt."""
    for match in emails:
        local = str(match.value).split("@", 1)[0].lower()
        if any(local == name or local.startswith(name) for name in _SUPPORT_LOCALS):
            return Match(
                field="support_email",
                value=match.value,
                start=match.start,
                end=match.end,
                text=match.text,
                rule="address whose local part means support",
                context=match.context,
            )
    for match in emails:
        window = text[max(0, match.start - CUE_WINDOW) : match.end + CUE_WINDOW].lower()
        if any(word in window for word in ("cancel", "support", "assist", "iptal", "destek")):
            return Match(
                field="support_email",
                value=match.value,
                start=match.start,
                end=match.end,
                text=match.text,
                rule="address beside a cancellation or support cue",
                context=match.context,
            )
    return None


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def extract_matches(
    text: str | None,
    *,
    subject: str | None = None,
    now: dt.datetime | None = None,
    codes: Iterable[str] = (),
) -> Extraction:
    """Run every rule over one excerpt.

    ``subject`` is scanned first and its offsets are negative-free by being run
    as a separate pass, because a booking reference lives in the subject line
    at least as often as in the body. ``codes`` are extra regexes from the
    matched vendor alias group — ``\\bTK\\s?\\d{1,4}\\b`` for Turkish Airlines.
    """
    body = text or ""
    head = (subject or "").strip()
    # One string, so every span is an index into the same text. The subject is
    # first and separated by a newline, which no rule spans across.
    full = f"{head}\n{body}" if head else body
    if not full.strip():
        return Extraction()

    matches: dict[str, list[Match]] = {}

    def add(found: Iterable[Match]) -> None:
        for match in found:
            matches.setdefault(match.field, []).append(match)

    add(find_pnr(full))
    add(find_flight_no(full))
    emails = find_emails(full)
    add(emails)
    urls, docs, drive = find_links(full)
    add(urls)
    add(docs)
    add(drive)
    add(find_amounts(full))
    reference_numbers = find_order_no(full) + find_ticket_no(full)
    add(reference_numbers)
    add(find_route(full))
    # A 13-digit ticket number reads as a phone number to any digit-counting
    # rule, so a reference already claimed by a cue word wins the span.
    claimed = [(m.start, m.end) for m in reference_numbers]
    add(
        phone
        for phone in find_phones(full)
        if not any(start < phone.end and phone.start < end for start, end in claimed)
    )
    add(find_dates(full, now=now))

    support = find_support_email(emails, full)
    if support is not None:
        matches.setdefault("support_email", []).append(support)

    for pattern in codes:
        try:
            compiled = re.compile(pattern, re.I)
        except re.error:
            continue
        for hit in compiled.finditer(full):
            matches.setdefault("codes", []).append(
                Match(field="codes", value=hit.group(0).strip(), start=hit.start(),
                      end=hit.end(), text=hit.group(0), rule=f"alias code pattern {pattern}",
                      context=_context(full, hit.start(), hit.end()))
            )

    values: dict[str, Any] = {}
    for name, found in matches.items():
        if not found:
            continue
        if name in LIST_FIELDS:
            seen: list[Any] = []
            for match in found:
                if match.value not in seen:
                    seen.append(match.value)
            values[name] = seen
        else:
            values[name] = found[0].value

    # Two conveniences the planner references by name in the worked examples.
    if values.get("dates"):
        values["date"] = values["dates"][0]
        future = [d for d in values["dates"] if _is_future(d, now)]
        if future:
            values["depart_at"] = future[0]
    if "flight_nos" not in values and "flight_no" in matches:
        values["flight_nos"] = [m.value for m in matches["flight_no"]]
    if values.get("amounts"):
        values["amount"] = values["amounts"][0]["display"]

    return Extraction(values=values, matches=matches)


def _is_future(iso: str, now: dt.datetime | None) -> bool:
    anchor = now or dt.datetime.now(dt.UTC)
    try:
        parsed = dt.datetime.fromisoformat(iso)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=anchor.tzinfo or dt.UTC)
    return parsed >= anchor


def extract(
    text: str | None,
    *,
    subject: str | None = None,
    now: dt.datetime | None = None,
    codes: Iterable[str] = (),
) -> dict[str, Any]:
    """``{field: value}`` for one excerpt. What hangs off ``candidate.extracted``."""
    return extract_matches(text, subject=subject, now=now, codes=codes).as_dict()


def _read(row: Any, *names: str) -> Any:
    for name in names:
        value = row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)
        if value:
            return value
    return None


def extract_from_candidate(
    candidate: Any,
    *,
    now: dt.datetime | None = None,
    codes: Iterable[str] = (),
) -> dict[str, Any]:
    """Run the rules over a search hit, whatever corpus it came from.

    Reads the subject/title/name and the body/description/excerpt off a mirror
    row, a ``Hit`` or a plain dict, so the probe does not have to branch per
    service.
    """
    subject = _read(candidate, "subject", "title", "name", "label")
    body = _read(candidate, "body_clean", "body", "description", "content_excerpt",
                 "snippet", "excerpt", "text")
    extra = _read(candidate, "location")
    if extra and body:
        body = f"{body}\n{extra}"
    elif extra:
        body = str(extra)
    return extract(body, subject=subject, now=now, codes=codes)


__all__ = [
    "CUE_WINDOW",
    "LIST_FIELDS",
    "Extraction",
    "Match",
    "extract",
    "extract_from_candidate",
    "extract_matches",
    "find_amounts",
    "find_dates",
    "find_emails",
    "find_flight_no",
    "find_links",
    "find_order_no",
    "find_phones",
    "find_pnr",
    "find_route",
    "find_support_email",
    "find_ticket_no",
]
