"""Plant the demo scenarios in a real, throwaway Google account.

`scripts/seed_local.py` writes our own mirror tables directly, which is enough
to click around offline. This script is the other half: it writes the same
scenarios into a real Gmail, Calendar and Drive through the same service
clients the app uses, so the demo video and the eval harnesses run against
data that a sync task genuinely fetched from Google.

Everything it plants carries a marker, and the marker is how it stays
idempotent. Re-running updates in place instead of duplicating:

    Gmail       a label, `alphalaw-demo`, plus three headers on every message:
                X-Alphalaw-Demo, X-Alphalaw-Slug, X-Alphalaw-Hash
    Calendar    extendedProperties.private.alphalaw_demo / _slug / _hash
    Drive       appProperties.alphalaw_demo / _slug / _hash

The slug says which planted item this is; the hash says which version of it.
Same hash means nothing to do. Different hash means update — a patch for
Calendar and Drive, and for Gmail, where a message is immutable, a fresh
insert and the old one to the bin.

    python -m scripts.seed_demo_account --email demo@example.com --dry-run
    python -m scripts.seed_demo_account --email demo@example.com
    python -m scripts.seed_demo_account --email demo@example.com --reseed
    python -m scripts.seed_demo_account --email demo@example.com --clean

Dates are computed from an anchor, which defaults to today, so "next week"
and "tomorrow" mean what they say on the day you run it. Pass `--anchor` to
pin them. Re-seed before recording; a Tuesday's data is wrong by Thursday.

The account has to be connected through the app's OAuth flow first, so the
encrypted refresh token is in `oauth_tokens`. Read `scripts/README.md` before
running this — in particular the seven-day refresh-token expiry that applies
to every OAuth app still in testing mode.
"""

# The Turkish booking, the em dashes and the arrows in the event titles are
# the data, not typos. RUF001-003 flag exactly those characters as ambiguous,
# which here would mean flagging the point of the fixture.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import datetime as dt
import hashlib
import json
import os
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:  # pragma: no cover - import only for the type checker
    from app.google.client import GoogleClients

# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

MARKER = "alphalaw-demo"

GMAIL_LABEL = "alphalaw-demo"
HDR_DEMO = "X-Alphalaw-Demo"
HDR_SLUG = "X-Alphalaw-Slug"
HDR_HASH = "X-Alphalaw-Hash"

PROP_DEMO = "alphalaw_demo"
PROP_SLUG = "alphalaw_slug"
PROP_HASH = "alphalaw_hash"

#: Message-IDs we mint. `.invalid` is reserved by RFC 2606 and can never be a
#: real domain, so nothing we plant can collide with real mail.
MSGID_DOMAIN = "alphalaw-demo.invalid"

UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

MIME_PDF = "application/pdf"
MIME_GDOC = "application/vnd.google-apps.document"
MIME_GSHEET = "application/vnd.google-apps.spreadsheet"
MIME_FOLDER = "application/vnd.google-apps.folder"

# --------------------------------------------------------------------------- #
# The scenarios these items exist to serve
# --------------------------------------------------------------------------- #

SCENARIOS: dict[str, str] = {
    "#1": "What's on my calendar next week?",
    "#2": "Find emails from sarah@company.com about the budget",
    "#3": "Show me PDFs in Drive from last month",
    "#4": "Cancel my Turkish Airlines flight",
    "#5": "Prepare for tomorrow's meeting with Acme Corp",
    "#6": "Find events next week that conflict with my out-of-office doc",
    "#7": "Move the meeting with John",
    "#8": "That email about the proposal",
    "#9": "Next Tuesday",
    "#10": "Cancel my flight to Istanbul",
    "#11": "Prepare for tomorrow's meeting with Acme Corp (Calendar 503s mid-run)",
    "#12": "Cancel my Turkish Airlines flight (booking email is in Turkish)",
    "#13": "Push my Acme review next Thursday to Friday 3pm and tell the attendees",
    "#14": "Summarise everything Acme sent me this month",
    "#15": "thanks, that's perfect",
    "#16": "What's on my calendar next week where john@company.com is invited?",
    "#S": "Share the Acme Q3 proposal with Dana (not in SAMPLE_QUERIES.md)",
    "noise": "background, so search has to discriminate",
}

#: Where this seed knowingly differs from docs/SAMPLE_QUERIES.md, and why.
#: Printed at the end of every run, because a silent divergence between the
#: graded document and the account it is demonstrated on is the worst kind.
DEVIATIONS: tuple[tuple[str, str], ...] = (
    (
        "second booking is Pegasus (PC1996), not a second Turkish Airlines "
        "flight (TK1996)",
        "The doc gives the user two TK bookings, which makes #4 ambiguous as "
        "well as #10. A different carrier to the same city keeps #10's tie "
        "(two Istanbul bookings, near-equal scores) while leaving #4 clean — "
        "only one booking is Turkish Airlines. SAW, Pegasus's Istanbul hub, "
        "is already in the pre-pass's airport codes for #10.",
    ),
    (
        "only the Turkish booking email is planted, not the English one",
        "#4 and #12 are the same query over two versions of one mailbox. "
        "Planting both would mean the English mail always wins and the "
        "escalation ladder never runs. The Turkish one exercises #4's plan "
        "and #12's ladder in a single demo.",
    ),
    (
        "John Okafor is john@company.com, not john.okafor@company.com",
        "#16 filters on the literal address john@company.com and #7 shows "
        "john.okafor@company.com on the same person. One human, one address; "
        "the brief's own sample query picks the winner.",
    ),
    (
        "the out-of-office file is a Google Doc, not a .docx",
        "Drive can only export text from its own formats. An uploaded .docx "
        "comes back as bytes we cannot read, the date range never extracts, "
        "and #6 demonstrates its fallback instead of its happy path.",
    ),
)

# --------------------------------------------------------------------------- #
# Cast
# --------------------------------------------------------------------------- #

SARAH = ("Sarah Chen", "sarah@company.com")
JOHN_OKAFOR = ("John Okafor", "john@company.com")
PRIYA = ("Priya Raman", "priya@company.com")
TOM = ("Tom Alvarez", "tom@company.com")
DANA = ("Dana Whitfield", "dana@acmecorp.com")
MARCUS = ("Marcus Iyer", "marcus@acmecorp.com")
JOHN_REYES = ("John Reyes", "john.reyes@northwind.io")
THY = ("Türk Hava Yolları", "bilet@thy.com")
PEGASUS = ("Pegasus Airlines", "noreply@flypgs.com")

TR_MONTHS = (
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
)

IST_TZ = ZoneInfo("Europe/Istanbul")


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Clock:
    """Every date in the dataset, derived from one anchor day.

    The anchor is a local date, not an instant, because every phrase the
    orchestrator resolves — "tomorrow", "next week", "last month" — is
    computed on the user's local calendar. Pinning the anchor pins the whole
    dataset, which is what makes a re-run and an eval run comparable.
    """

    anchor: dt.date
    tz: ZoneInfo

    def at(self, day: dt.date, hour: int, minute: int = 0, *, tz: ZoneInfo | None = None) -> dt.datetime:
        return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz or self.tz)

    @property
    def now(self) -> dt.datetime:
        # 09:12 local, the instant docs/SAMPLE_QUERIES.md evaluates against.
        return self.at(self.anchor, 9, 12)

    def day(self, offset: int) -> dt.date:
        return self.anchor + dt.timedelta(days=offset)

    def moment(self, offset: int, hour: int, minute: int = 0) -> dt.datetime:
        return self.at(self.day(offset), hour, minute)

    @property
    def tomorrow(self) -> dt.date:
        return self.day(1)

    @property
    def next_monday(self) -> dt.date:
        """Monday of the ISO week after the anchor's."""
        this_monday = self.anchor - dt.timedelta(days=self.anchor.weekday())
        return this_monday + dt.timedelta(days=7)

    def next_week(self, weekday: int) -> dt.date:
        """A day of next week. 0 is Monday, 6 is Sunday."""
        return self.next_monday + dt.timedelta(days=weekday)

    @property
    def this_month_first(self) -> dt.date:
        return self.anchor.replace(day=1)

    @property
    def last_month_first(self) -> dt.date:
        return (self.this_month_first - dt.timedelta(days=1)).replace(day=1)

    def last_month_day(self, day: int) -> dt.date:
        """The nth day of the previous calendar month, clamped to its length."""
        first = self.last_month_first
        length = (self.this_month_first - first).days
        return first + dt.timedelta(days=min(max(day, 1), length) - 1)

    def this_month_day(self, day: int) -> dt.date:
        """The nth day of this month, never later than the anchor itself."""
        return self.anchor.replace(day=min(max(day, 1), self.anchor.day))


def tr_date(moment: dt.datetime) -> str:
    """A date the way a Turkish airline writes it: 5 Eylül 2026."""
    return f"{moment.day} {TR_MONTHS[moment.month - 1]} {moment.year}"


def human(moment: dt.datetime | dt.date) -> str:
    if isinstance(moment, dt.datetime):
        return moment.strftime("%a %d %b %H:%M")
    return moment.strftime("%a %d %b")


# --------------------------------------------------------------------------- #
# The items
# --------------------------------------------------------------------------- #

@dataclass
class Mail:
    slug: str
    subject: str
    sender: tuple[str, str]
    body: str
    received_at: dt.datetime
    supports: tuple[str, ...]
    thread: str | None = None          # thread key; first item in it starts it
    reply_to: str | None = None        # slug of the message this answers
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)

    @property
    def when(self) -> str:
        return human(self.received_at)

    @property
    def label(self) -> str:
        return self.subject

    def fingerprint(self) -> str:
        return content_hash(
            self.subject,
            self.sender[1],
            self.body,
            self.received_at.isoformat(),
            ",".join(self.to),
            ",".join(self.cc),
            self.reply_to or "",
        )


@dataclass
class Event:
    slug: str
    title: str
    start: dt.datetime | dt.date
    end: dt.datetime | dt.date
    supports: tuple[str, ...]
    description: str | None = None
    location: str | None = None
    attendees: list[dict[str, Any]] = field(default_factory=list)
    all_day: bool = False
    start_tz: str | None = None
    end_tz: str | None = None

    @property
    def when(self) -> str:
        return human(self.start)

    @property
    def label(self) -> str:
        return self.title

    def fingerprint(self) -> str:
        return content_hash(
            self.title,
            self.description or "",
            self.location or "",
            json.dumps(self.attendees, sort_keys=True),
            self.start.isoformat(),
            self.end.isoformat(),
            str(self.all_day),
            self.start_tz or "",
            self.end_tz or "",
        )


@dataclass
class Doc:
    slug: str
    name: str
    mime_type: str
    folder: str
    modified_at: dt.datetime
    lines: list[str]
    supports: tuple[str, ...]
    title: str | None = None           # first line of a PDF; defaults to name

    @property
    def when(self) -> str:
        return human(self.modified_at)

    @property
    def label(self) -> str:
        return self.name

    @property
    def upload_mime(self) -> str:
        """What we send. Google converts text/plain into a Doc, csv into a Sheet."""
        if self.mime_type == MIME_GDOC:
            return "text/plain"
        if self.mime_type == MIME_GSHEET:
            return "text/csv"
        return self.mime_type

    def content(self) -> bytes:
        if self.mime_type == MIME_PDF:
            return make_pdf(self.title or self.name, self.lines)
        return ("\n".join(self.lines) + "\n").encode("utf-8")

    def fingerprint(self) -> str:
        return content_hash(
            self.name,
            self.mime_type,
            self.folder,
            self.modified_at.isoformat(),
            "\n".join(self.lines),
        )


@dataclass
class Dataset:
    mails: list[Mail]
    events: list[Event]
    docs: list[Doc]

    def all_items(self) -> list[Any]:
        return [*self.mails, *self.events, *self.docs]


def content_hash(*parts: str) -> str:
    """Twelve hex characters over the fields a person would call the content.

    Short on purpose. It rides in a mail header and in Drive's appProperties,
    both of which are size-limited, and it only has to separate versions of
    one slug — not be a global identifier.
    """
    digest = hashlib.blake2s("\x1f".join(parts).encode("utf-8"), digest_size=6)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# A very small PDF writer
# --------------------------------------------------------------------------- #

def _pdf_escape(line: str) -> bytes:
    """A PDF literal string, in the one encoding Helvetica is declared with."""
    text = line.encode("cp1252", "replace").decode("cp1252")
    for bad, good in (("\\", "\\\\"), ("(", "\\("), (")", "\\)")):
        text = text.replace(bad, good)
    return text.encode("cp1252", "replace")


def make_pdf(title: str, lines: Sequence[str]) -> bytes:
    """One page of Helvetica, enough for `pypdf` to read the text back.

    Real PDFs matter here: `gdrive.text_for` downloads a PDF and runs
    `pdf_text` over it, so a fake one would leave every planted PDF with an
    empty excerpt and no embedding worth having. This is the smallest
    structurally valid file that survives that round trip, and it needs no
    third-party library to write.
    """
    wrapped: list[str] = [title, ""]
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=88) or [""])

    stream_lines = [b"BT", b"/F1 11 Tf", b"72 720 Td", b"15 TL"]
    for line in wrapped[:46]:
        stream_lines.append(b"(" + _pdf_escape(line) + b") Tj T*")
    stream_lines.append(b"ET")
    stream = b"\n".join(stream_lines)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
        b" /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + obj + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_at).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #

def build_mail(clock: Clock, me: str, name: str) -> list[Mail]:
    dep_ist = clock.at(clock.day(16), 10, 30, tz=IST_TZ)
    dep_saw = clock.at(clock.day(53), 9, 15)

    mails: list[Mail] = []

    def add(**kwargs: Any) -> None:
        kwargs.setdefault("to", [me])
        mails.append(Mail(**kwargs))

    # -- the Turkish booking. #4's orchestration, #12's escalation ladder. --- #
    # No support address in the body on purpose: #12 turns on the planner
    # falling back to the alias table's cancel@ address and saying that it did.
    add(
        slug="mail_thy_booking",
        subject="Uçuş rezervasyonunuz onaylandı — TK1984",
        sender=THY,
        received_at=clock.moment(-29, 8, 41),
        supports=("#4", "#12", "#10"),
        body=(
            "Sayın Yolcumuz,\n\n"
            "6F2QK9 numaralı rezervasyonunuz onaylanmıştır.\n\n"
            f"Uçuş: TK1, İstanbul (IST) → New York (JFK), {tr_date(dep_ist)}, 10:30.\n"
            "Bilet numarası: TK1984\n"
            f"Yolcu: {name}\n"
            "Kabin: Economy · Koltuk: 24A\n"
            "Toplam: 812,40 USD\n\n"
            "Uçuştan 24 saat önce çevrimiçi kontrol yapabilirsiniz.\n"
            "Bagaj hakkınız: 2 x 23 kg.\n\n"
            "İyi yolculuklar dileriz.\n"
            "Türk Hava Yolları"
        ),
    )

    # -- the second carrier, same city. #10's tie. -------------------------- #
    add(
        slug="mail_pgs_booking",
        subject="Booking confirmed — PC1996, New York to Istanbul",
        sender=PEGASUS,
        received_at=clock.moment(-12, 15, 2),
        supports=("#10",),
        body=(
            f"Dear {name},\n\n"
            "Your booking is confirmed.\n\n"
            "Booking reference: RT8HW2\n"
            "Ticket number: PC1996\n"
            f"Flight PC1996, New York (JFK) → Istanbul (SAW), "
            f"{dep_saw.strftime('%d %B %Y')}, 09:15.\n"
            "Passenger: " + name + "\n"
            "Cabin: Economy · Seat 11C\n"
            "Total paid: USD 640.00\n\n"
            "To change or cancel this booking, write to support@flypgs.com "
            "quoting RT8HW2.\n\n"
            "Pegasus Airlines"
        ),
    )

    add(
        slug="mail_istanbul_hotel",
        subject="Istanbul hotel — reservation held",
        sender=("Bosphorus Hotel", "reservations@bosphorushotel.com"),
        received_at=clock.moment(-10, 11, 20),
        supports=("#10",),
        body=(
            "Your room is held for two nights in Beyoğlu, Istanbul.\n"
            "Confirmation HB-449120. Free cancellation up to 48 hours before "
            "arrival.\n\n"
            "This is a near miss on purpose: it mentions Istanbul and is not a "
            "flight. A search for a booking to cancel should rank it below both "
            "airline confirmations."
        ),
    )

    # -- the proposal thread. #8's pinned conversation entity. -------------- #
    add(
        slug="mail_proposal_1",
        subject="Acme Q3 proposal — pricing",
        sender=DANA,
        received_at=clock.moment(-2, 9, 40),
        thread="acme_proposal",
        supports=("#8", "#5", "#14"),
        to=[me],
        cc=[SARAH[1]],
        body=(
            f"Hi {name.split()[0]},\n\n"
            "Revised proposal attached in Drive — Acme Q3 renewal proposal v4.\n\n"
            "We have moved the term to 14 months at the 12-month rate, holding "
            "the per-seat price at $42. Two conditions come with that: the "
            "liability cap moves to 12 months of fees, and data residency has "
            "to be EU-only.\n\n"
            "We need the signed security questionnaire back before Friday's "
            "call.\n\n"
            "Dana"
        ),
    )
    add(
        slug="mail_proposal_2",
        subject="Re: Acme Q3 proposal — pricing",
        sender=SARAH,
        received_at=clock.moment(-2, 19, 5),
        thread="acme_proposal",
        reply_to="mail_proposal_1",
        supports=("#8",),
        to=[me, DANA[1]],
        body=(
            "The questionnaire went out on the 17th — I have the send receipt. "
            "Nobody at Acme has confirmed they got it, so I would open with "
            "that.\n\nSarah"
        ),
    )
    add(
        slug="mail_proposal_3",
        subject="Re: Acme Q3 proposal — pricing",
        sender=MARCUS,
        received_at=clock.moment(-1, 11, 20),
        thread="acme_proposal",
        reply_to="mail_proposal_2",
        supports=("#8", "#5", "#14"),
        to=[me],
        cc=[DANA[1], SARAH[1]],
        body=(
            "Adding my two MSA redlines so they are in one place:\n\n"
            "1. Liability cap — 12 months of fees, not 6.\n"
            "2. Data residency — EU-only, no US failover.\n\n"
            "Neither has been answered yet.\n\nMarcus"
        ),
    )
    add(
        slug="mail_proposal_4",
        subject="Re: Acme Q3 proposal — pricing",
        sender=SARAH,
        received_at=clock.moment(-1, 16, 2),
        thread="acme_proposal",
        reply_to="mail_proposal_3",
        supports=("#8",),
        to=[me],
        body=(
            "Noted. I do not have an answer on the liability cap — that is a "
            "legal call and it is still open.\n\nSarah"
        ),
    )

    # -- the budget thread. #2's sender-plus-topic sample. ------------------ #
    add(
        slug="mail_budget_1",
        subject="Re: FY27 budget — headcount lines",
        sender=SARAH,
        received_at=clock.moment(-6, 10, 15),
        supports=("#2",),
        body=(
            "The three open reqs move to Q1, so the FY27 budget headcount lines "
            "drop by about 240k in the second half. I have not repointed the "
            "contractor line yet.\n\nSarah"
        ),
    )
    add(
        slug="mail_budget_2",
        subject="Budget review deck (v4)",
        sender=SARAH,
        received_at=clock.moment(-14, 17, 48),
        supports=("#2",),
        body=(
            "Attached the version we walked through on Tuesday. Slide 6 is the "
            "one that changed — cloud spend is now split by environment rather "
            "than by team.\n\nSarah"
        ),
    )
    add(
        slug="mail_budget_3",
        subject="Q3 budget variance",
        sender=SARAH,
        received_at=clock.moment(-22, 9, 3),
        supports=("#2",),
        body=(
            "We are 8% under on cloud spend, mostly because the migration "
            "slipped a month. Everything else is inside 2%.\n\nSarah"
        ),
    )
    add(
        slug="mail_budget_4",
        subject="Budget kickoff — dates",
        sender=SARAH,
        received_at=clock.moment(-70, 14, 30),
        supports=("#2",),
        body=(
            "First pass due the 25th, final by the middle of the following "
            "month. I will send the template once finance signs off on the "
            "categories.\n\nSarah"
        ),
    )

    # -- meeting prep. #5 and #11. ------------------------------------------ #
    add(
        slug="mail_acme_pricing",
        subject="Re: renewal pricing — revised",
        sender=DANA,
        received_at=clock.moment(-2, 8, 12),
        supports=("#5", "#11", "#14"),
        body=(
            "Revised pricing, as discussed: 14-month term at the 12-month rate. "
            "Before the call I need the security questionnaire back — we cannot "
            "put it through procurement without it.\n\nDana"
        ),
    )
    add(
        slug="mail_acme_questionnaire",
        subject="Security questionnaire — sent Friday",
        sender=SARAH,
        received_at=clock.moment(-3, 16, 40),
        supports=("#5", "#11"),
        body=(
            "The security questionnaire went to Acme on Friday. No "
            "acknowledgement from them yet, which is worth chasing before the "
            "renewal review.\n\nSarah"
        ),
    )
    add(
        slug="mail_acme_msa",
        subject="MSA redlines — liability cap and data residency",
        sender=MARCUS,
        received_at=clock.moment(-9, 11, 55),
        supports=("#5", "#11", "#14"),
        body=(
            "Two redlines on the MSA. The liability cap should be 12 months of "
            "fees. Data residency has to be EU-only. Both are unanswered on our "
            "side.\n\nMarcus"
        ),
    )

    # -- Acme, spread over this calendar month. #14's fan-out. -------------- #
    digest = (
        ("Weekly status — Acme integration", DANA,
         "Sandbox is up. Two of the four webhooks are live."),
        ("Invoice 4412 — August", MARCUS,
         "August invoice attached. Net 30, due at the end of next month."),
        ("Acme support: ticket 8891 resolved", DANA,
         "The SSO redirect loop is fixed. Root cause was a stale metadata file."),
        ("Renewal timeline", DANA,
         "Procurement needs ten working days once the questionnaire clears."),
        ("Q3 usage report", MARCUS,
         "Seat usage is at 84% of the contracted number, up six points."),
        ("Acme — new security contact", MARCUS,
         "Priya Raman is off the account; route security questions to me."),
        ("Data residency — follow up", MARCUS,
         "Repeating the EU-only requirement so it is not lost in the thread."),
        ("Acme roadmap briefing — slides", DANA,
         "Slides from the briefing. The audit-log work lands in Q4, not Q3."),
        ("Re: sandbox credentials", DANA,
         "New sandbox credentials sent separately. The old ones are revoked."),
        ("Acme — holiday coverage", DANA,
         "Our team is thin over the last week of the month. Plan around it."),
    )
    for index, (subject, sender, line) in enumerate(digest):
        # Spread across day 1 to the anchor's own day, so "this month" has
        # something in it no matter which day of the month the seed runs.
        day = clock.this_month_day(1 + round(index * (clock.anchor.day - 1) / max(len(digest) - 1, 1)))
        add(
            slug=f"mail_acme_digest_{index + 1:02d}",
            subject=subject,
            sender=sender,
            received_at=clock.at(day, 8 + (index % 9), (index * 7) % 60),
            supports=("#14",),
            body=line + "\n\n" + sender[0].split()[0],
        )

    # -- background. Near misses on purpose. -------------------------------- #
    noise: tuple[tuple[str, tuple[str, str], int, str], ...] = (
        ("Your weekly digest: airline deals and fare drops",
         ("Flightwatch", "digest@flightwatch.example"), -4,
         "Fares to twelve cities dropped this week. Istanbul is not one of them."),
        ("Receipt from Northwind Cloud — August",
         ("Northwind Billing", "billing@northwind.io"), -5,
         "USD 1,204.60 charged to the card ending 4417. Invoice NW-20418."),
        ("Design system: weekly release notes",
         PRIYA, -6,
         "Twelve components shipped. The date picker keyboard trap is fixed."),
        ("Q3 planning notes — my rough version",
         TOM, -7,
         "Rough notes before the planning session. Not a proposal, just notes."),
        ("Standup notes, Tuesday",
         PRIYA, -8,
         "Three in progress, one blocked on the vendor sync."),
        ("Proposal template — internal wiki",
         TOM, -11,
         "The template lives in the wiki now. Do not copy the old deck."),
        ("Your subscription renews soon",
         ("Ledgerly", "no-reply@ledgerly.example"), -13,
         "Annual plan renews in fourteen days. USD 96.00."),
        ("Package delivered",
         ("Shipfast", "notifications@shipfast.example"), -15,
         "Left with the building manager. Tracking SF-771203."),
        ("John's team offsite — notes",
         TOM, -16,
         "Notes from the offsite. Nothing needs a decision from you."),
        ("Payroll: August payslip available",
         ("People Ops", "payroll@company.com"), -17,
         "Your payslip is in the portal."),
        ("Security alert: new sign-in",
         ("Account Security", "security@company.com"), -18,
         "A new device signed in from a browser we have not seen before."),
        ("Re: office move — floor plan",
         PRIYA, -20,
         "The desks by the window are taken. Second draft attached."),
        ("Conference CFP closes Friday",
         ("DevWeek", "cfp@devweek.example"), -23,
         "Last call for talks. Twenty minutes, no slides required."),
        ("Northwind maintenance window",
         ("Northwind Status", "status@northwind.io"), -25,
         "Two hours of downtime on the vendor API, Sunday morning."),
        ("Team lunch — Thursday",
         TOM, -27,
         "Booked for eight at one o'clock. Say if you cannot make it."),
        ("Your invoice from Bright Legal",
         ("Bright Legal", "accounts@brightlegal.example"), -30,
         "Fees for July. Payable within thirty days."),
        ("Reminder: expense report due",
         ("Finance", "finance@company.com"), -33,
         "Anything from last month has to be filed by the end of the week."),
        ("Newsletter: what shipped in engineering",
         ("Engineering", "eng-news@company.com"), -35,
         "Search latency is down forty percent after the index change."),
        ("Re: hiring loop for the platform role",
         SARAH, -38,
         "Two onsites next week. I have the debrief slot held."),
        ("Vendor review — Northwind",
         TOM, -41,
         "Northwind renewal is in December. Starting the review early."),
        ("Password expiry notice",
         ("IT Helpdesk", "helpdesk@company.com"), -44,
         "Your password expires in fourteen days."),
        ("Flight delay: unrelated trip",
         ("Skyline Air", "alerts@skylineair.example"), -47,
         "A trip you searched for once and never booked. Ignore."),
        ("Re: contract template questions",
         SARAH, -50,
         "Legal signed off on the shortened template. Use v3 from now on."),
        ("Quarterly all-hands: recording",
         ("Comms", "comms@company.com"), -53,
         "Recording and slides from the all-hands."),
        ("Your order has shipped",
         ("Deskworks", "orders@deskworks.example"), -57,
         "The monitor arm shipped. Two working days."),
        ("Re: on-call rotation swap",
         PRIYA, -61,
         "I can take your Thursday if you take my Monday."),
        ("Library: book due back",
         ("City Library", "no-reply@citylibrary.example"), -65,
         "One item due back this week."),
        ("Survey: how are we doing?",
         ("People Ops", "peopleops@company.com"), -70,
         "Six questions, two minutes, anonymous."),
        ("Re: laptop refresh",
         ("IT Helpdesk", "helpdesk@company.com"), -76,
         "You are in the next refresh batch. No action needed yet."),
        ("Annual insurance documents",
         ("Assure Group", "documents@assuregroup.example"), -81,
         "Your renewal documents for the year, attached."),
        ("Welcome to Signalbox",
         ("Signalbox", "hello@signalbox.example"), -86,
         "Thanks for signing up. Here is how to get started."),
        ("Re: notes from the customer call",
         TOM, -89,
         "Cleaned up my notes from the call. Nothing urgent."),
    )
    for index, (subject, sender, offset, line) in enumerate(noise):
        add(
            slug=f"mail_noise_{index + 1:02d}",
            subject=subject,
            sender=sender,
            received_at=clock.moment(offset, 7 + (index % 11), (index * 13) % 60),
            supports=("noise",),
            body=line + "\n",
        )

    return mails


def build_events(clock: Clock, me: str, name: str) -> list[Event]:
    def guest(person: tuple[str, str], status: str = "needsAction") -> dict[str, Any]:
        return {"email": person[1], "displayName": person[0], "responseStatus": status}

    def myself() -> dict[str, Any]:
        return {"email": me, "self": True, "responseStatus": "accepted", "organizer": True}

    dep_ist = clock.at(clock.day(16), 10, 30, tz=IST_TZ)
    arr_jfk = dep_ist + dt.timedelta(hours=11)
    dep_jfk = clock.at(clock.day(53), 9, 15)
    arr_saw = dep_jfk + dt.timedelta(hours=9, minutes=45)

    return [
        # -- next week. #1, #9, #16, and the OOO overlap for #6. ------------- #
        Event(
            slug="ev_standup",
            title="Standup",
            start=clock.at(clock.next_week(0), 9, 30),
            end=clock.at(clock.next_week(0), 10, 0),
            supports=("#1", "#6"),
            description=None,   # the no-agenda meeting, deliberately empty
        ),
        Event(
            slug="ev_acme_mon",
            title="Acme Corp — Q3 renewal review",
            start=clock.at(clock.next_week(0), 14, 0),
            end=clock.at(clock.next_week(0), 15, 0),
            supports=("#1", "#6", "#13"),
            description="Standing renewal check-in. Agenda in the thread.",
            attendees=[guest(DANA, "accepted"), guest(SARAH, "accepted"), myself()],
        ),
        Event(
            slug="ev_design_review",
            title="Design review",
            start=clock.at(clock.next_week(1), 11, 0),
            end=clock.at(clock.next_week(1), 12, 0),
            supports=("#1", "#6", "#7", "#9", "#16"),
            description="Walk the new date picker and the empty states.",
            location="Room 4",
            attendees=[
                guest(JOHN_OKAFOR, "accepted"),
                guest(SARAH, "accepted"),
                guest(PRIYA, "accepted"),
                guest(TOM, "tentative"),
                myself(),
            ],
        ),
        Event(
            slug="ev_john_okafor_1_1",
            title="1:1 with John Okafor",
            start=clock.at(clock.next_week(1), 16, 0),
            end=clock.at(clock.next_week(1), 16, 30),
            supports=("#1", "#7", "#9", "#16"),
            description="Fortnightly.",
            attendees=[guest(JOHN_OKAFOR, "needsAction"), myself()],
        ),
        Event(
            slug="ev_vendor_sync_reyes",
            title="Vendor sync — John Reyes (Northwind)",
            start=clock.at(clock.next_week(2), 9, 0),
            end=clock.at(clock.next_week(2), 9, 45),
            supports=("#1", "#7"),
            description="Quarterly vendor sync. Renewal is in December.",
            attendees=[guest(JOHN_REYES, "accepted"), myself()],
        ),
        Event(
            slug="ev_acme_review_thu",
            title="Acme review",
            start=clock.at(clock.next_week(3), 13, 0),
            end=clock.at(clock.next_week(3), 14, 0),
            supports=("#1", "#6", "#13"),
            description="Contract review with Acme.",
            attendees=[
                guest(DANA, "accepted"),
                guest(MARCUS, "accepted"),
                guest(SARAH, "accepted"),
                myself(),
            ],
        ),
        Event(
            slug="ev_company_offsite",
            title="Company offsite",
            start=clock.next_week(4),
            end=clock.next_week(5),      # all-day end is exclusive in Calendar
            supports=("#1", "#6"),
            all_day=True,
            description="All hands, offsite. No laptops.",
        ),
        # -- tomorrow. #5 and #11, and the Drive link in a description. ------ #
        Event(
            slug="ev_acme_renewal_tomorrow",
            title="Acme Corp — Q3 renewal review",
            start=clock.at(clock.tomorrow, 10, 0),
            end=clock.at(clock.tomorrow, 11, 0),
            supports=("#5", "#11"),
            location="Google Meet",
            description=(
                "Renewal review with Acme.\n\n"
                "Proposal: https://drive.google.com/drive/search?q=Acme%20Q3%20renewal%20proposal%20v4\n"
                "Open items: liability cap, data residency, security "
                "questionnaire acknowledgement."
            ),
            attendees=[
                guest(DANA, "accepted"),
                guest(MARCUS, "accepted"),
                guest(SARAH, "accepted"),
                myself(),
            ],
        ),
        # -- the flights. #4, #10, #12. ------------------------------------- #
        Event(
            slug="ev_flight_ist_jfk",
            title="Istanbul → NYC Flight (TK1984)",
            start=dep_ist,
            end=arr_jfk,
            start_tz="Europe/Istanbul",
            end_tz=str(clock.tz),
            supports=("#4", "#10", "#12"),
            location="Istanbul Airport (IST)",
            description="IST → JFK · TK1 · ticket TK1984 · booking 6F2QK9",
        ),
        Event(
            slug="ev_flight_jfk_saw",
            title="NYC → Istanbul (PC1996)",
            start=dep_jfk,
            end=arr_saw,
            start_tz=str(clock.tz),
            end_tz="Europe/Istanbul",
            supports=("#10",),
            location="John F. Kennedy International Airport (JFK)",
            description="JFK → SAW · PC1996 · booking RT8HW2",
        ),
    ]


def build_docs(clock: Clock, me: str, name: str) -> list[Doc]:
    ooo_start = clock.next_week(1)
    ooo_end = clock.next_week(4)

    return [
        # -- the out-of-office doc. #6's whole premise. ---------------------- #
        Doc(
            slug="doc_ooo",
            name=f"{name} — OOO and travel, Q3",
            mime_type=MIME_GDOC,
            folder="/Personal",
            modified_at=clock.moment(-5, 17, 30),
            supports=("#6",),
            lines=[
                f"{name} — out of office and travel, Q3",
                "",
                "Out of office: "
                f"{ooo_start.isoformat()} to {ooo_end.isoformat()} (Lisbon).",
                "",
                "I am away for the offsite and two days either side. Sarah Chen "
                "covers anything on the Acme renewal. Do not book me into "
                "anything in that window; if something is already there, it "
                "needs moving.",
                "",
                "Other travel this quarter:",
                f"  - Istanbul, {clock.day(16).isoformat()} to {clock.day(21).isoformat()}",
                f"  - Istanbul again, {clock.day(53).isoformat()} to {clock.day(58).isoformat()}",
                "",
                "Contact: only for something that cannot wait until I am back.",
            ],
        ),
        Doc(
            slug="doc_pto_calendar",
            name="Team PTO calendar 2026",
            mime_type=MIME_GSHEET,
            folder="/Personal",
            modified_at=clock.moment(-19, 10, 5),
            supports=("#6",),
            lines=[
                "Person,Start,End,Type,Notes",
                f"{SARAH[0]},{clock.day(30).isoformat()},{clock.day(34).isoformat()},PTO,",
                f"{PRIYA[0]},{clock.day(9).isoformat()},{clock.day(11).isoformat()},PTO,",
                f"{TOM[0]},{clock.day(45).isoformat()},{clock.day(52).isoformat()},PTO,",
                f"{name},{ooo_start.isoformat()},{ooo_end.isoformat()},Offsite + travel,Lisbon",
            ],
        ),
        # -- the live proposal, as a Doc. #5's brief. ------------------------ #
        Doc(
            slug="doc_acme_proposal_v4",
            name="Acme — Q3 renewal proposal v4",
            mime_type=MIME_GDOC,
            folder="/Sales/Acme",
            modified_at=clock.moment(-2, 12, 14),
            supports=("#5", "#11"),
            lines=[
                "Acme Corp — Q3 renewal proposal, version 4",
                "",
                "Term: 14 months at the 12-month rate. Per-seat price held at "
                "$42.",
                "Seats: 340 committed, true-up quarterly.",
                "",
                "Open conditions from Acme:",
                "  - Liability cap at 12 months of fees (currently 6).",
                "  - Data residency EU-only, no US failover.",
                "  - Security questionnaire acknowledged before signature.",
                "",
                "Prepared by Sarah Chen. Supersedes v3, which quoted a "
                "12-month term at list.",
            ],
        ),
        # -- the ambiguous share pair. -------------------------------------- #
        Doc(
            slug="doc_acme_proposal_v3",
            name="Acme Q3 Proposal v3.pdf",
            mime_type=MIME_PDF,
            folder="/Sales/Acme",
            modified_at=clock.moment(-8, 15, 40),
            supports=("#S",),
            title="Acme Corp — Q3 renewal proposal (v3)",
            lines=[
                "Term: 12 months at list. Per-seat price $46.",
                "Seats: 300 committed.",
                "",
                "Superseded by v4. Kept because procurement has the PDF.",
                "",
                "Liability cap: 6 months of fees.",
                "Data residency: US primary, EU failover.",
            ],
        ),
        Doc(
            slug="doc_acme_proposal_v3_final",
            name="Acme Q3 Proposal v3 (final).pdf",
            mime_type=MIME_PDF,
            folder="/Sales/Acme",
            modified_at=clock.moment(-7, 9, 25),
            supports=("#S",),
            title="Acme Corp — Q3 renewal proposal (v3, final)",
            lines=[
                "Same commercial terms as v3, with the signature block filled "
                "in and the appendix renumbered.",
                "",
                "This is the near-identical twin. A share anchored on a name "
                "match alone cannot tell it from the other v3, which is the "
                "point: the margin between the two sits under MARGIN, so a "
                "write has to ask rather than guess.",
            ],
        ),
        # -- last month's PDFs. #3. ------------------------------------------ #
        Doc(
            slug="doc_acme_msa",
            name="Acme_MSA_countersigned.pdf",
            mime_type=MIME_PDF,
            folder="/Contracts/Acme",
            modified_at=clock.at(clock.last_month_day(30), 16, 20),
            supports=("#3", "#5"),
            title="Master Services Agreement — Acme Corp (countersigned)",
            lines=[
                "Countersigned copy of the MSA on the previous term.",
                "",
                "Liability cap: 6 months of fees.",
                "Data residency: US primary.",
                "Term: 12 months from the effective date.",
                "",
                "This is the old term. The renewal proposal changes both the "
                "cap and the residency clause.",
            ],
        ),
        Doc(
            slug="doc_board_pack",
            name="Q3_board_pack.pdf",
            mime_type=MIME_PDF,
            folder="/Board/2026",
            modified_at=clock.at(clock.last_month_day(24), 11, 5),
            supports=("#3",),
            title="Q3 board pack",
            lines=[
                "Revenue, retention and pipeline for the quarter.",
                "",
                "Net revenue retention 112%. Two renewals at risk, one of them "
                "Acme, which is on a 14-month proposal.",
                "Cash runway 19 months at the current burn.",
            ],
        ),
        Doc(
            slug="doc_invoice_tk",
            name="Invoice_TK_1984.pdf",
            mime_type=MIME_PDF,
            folder="/Travel",
            modified_at=clock.at(clock.last_month_day(22), 9, 45),
            supports=("#3", "#4", "#12"),
            title="Invoice — Turkish Airlines",
            lines=[
                "Booking reference: 6F2QK9",
                "Ticket number: TK1984",
                "Flight: TK1, IST to JFK",
                f"Passenger: {name}",
                "Total: USD 812.40",
                "",
                "This file carries the record locator in its name and its text, "
                "which is what rung 3 of the escalation ladder looks for when "
                "the booking email itself does not surface.",
            ],
        ),
        Doc(
            slug="doc_security_questionnaire",
            name="Security_questionnaire_response.pdf",
            mime_type=MIME_PDF,
            folder="/Sales/Acme",
            modified_at=clock.at(clock.last_month_day(17), 14, 0),
            supports=("#3", "#5", "#11"),
            title="Security questionnaire — response",
            lines=[
                "Completed questionnaire returned to Acme.",
                "",
                "Covers access control, encryption at rest and in transit, "
                "subprocessors, incident response and data deletion.",
                "Two answers marked 'roadmap': customer-managed keys and "
                "region pinning.",
            ],
        ),
        Doc(
            slug="doc_offsite_quote",
            name="Offsite_venue_quote.pdf",
            mime_type=MIME_PDF,
            folder="/Ops",
            modified_at=clock.at(clock.last_month_day(9), 10, 30),
            supports=("#3",),
            title="Offsite venue quote",
            lines=[
                "Quote for the company offsite: room hire, catering for 40, "
                "and audio-visual.",
                "Total GBP 6,400 including tax. Valid for thirty days.",
            ],
        ),
        Doc(
            slug="doc_insurance",
            name="Insurance_renewal_2026.pdf",
            mime_type=MIME_PDF,
            folder="/Admin",
            modified_at=clock.at(clock.last_month_day(3), 8, 55),
            supports=("#3",),
            title="Insurance renewal 2026",
            lines=[
                "Renewal schedule for professional indemnity and cyber cover.",
                "Premium up 9% on last year. No change to the limits.",
            ],
        ),
        Doc(
            slug="doc_fy27_budget",
            name="FY27_budget_v2.pdf",
            mime_type=MIME_PDF,
            folder="/Finance",
            modified_at=clock.at(clock.last_month_day(1), 17, 10),
            supports=("#3", "#2"),
            title="FY27 budget, version 2",
            lines=[
                "Second pass at the FY27 budget.",
                "",
                "Headcount: three open reqs moved to Q1.",
                "Cloud spend split by environment rather than by team.",
                "Contractor line unchanged pending a decision.",
            ],
        ),
    ]


def build_dataset(clock: Clock, me: str, name: str) -> Dataset:
    return Dataset(
        mails=build_mail(clock, me, name),
        events=build_events(clock, me, name),
        docs=build_docs(clock, me, name),
    )


# --------------------------------------------------------------------------- #
# What happened to each item
# --------------------------------------------------------------------------- #

CREATED = "created"
UPDATED = "updated"
UNCHANGED = "unchanged"
REMOVED = "removed"
PLANNED = "would create"
FAILED = "failed"

GLYPH = {
    CREATED: "+",
    UPDATED: "~",
    UNCHANGED: "=",
    REMOVED: "-",
    PLANNED: ".",
    FAILED: "!",
}


@dataclass
class Outcome:
    service: str
    slug: str
    label: str
    when: str
    supports: tuple[str, ...]
    action: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Gmail
# --------------------------------------------------------------------------- #

class GmailPlanter:
    """Insert, not send.

    `users.messages.insert` puts a message straight into the mailbox with no
    delivery, no spam classification and no outbound mail to addresses that do
    not exist. With `internalDateSource=dateHeader` the Date: header sets the
    received time, which is the only way to plant ninety days of history.
    """

    def __init__(self, clients: GoogleClients, me: str) -> None:
        self.transport = clients.gmail.transport
        self.me = me
        self.label_id: str | None = None

    # -- the marker label --------------------------------------------------- #

    async def find_label(self) -> str | None:
        page = await self.transport.get("users/me/labels", api_method="gmail.labels.list")
        for label in page.get("labels") or []:
            if label.get("name") == GMAIL_LABEL:
                return str(label["id"])
        return None

    async def ensure_label(self) -> str:
        if self.label_id:
            return self.label_id
        found = await self.find_label()
        if found is None:
            created = await self.transport.post(
                "users/me/labels",
                api_method="gmail.labels.create",
                json={
                    "name": GMAIL_LABEL,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            found = str(created["id"])
        self.label_id = found
        return found

    # -- what is already there ---------------------------------------------- #

    async def existing(self) -> dict[str, dict[str, str]]:
        """`{slug: {"id": ..., "hash": ..., "thread": ...}}` for what is planted."""
        label_id = await self.find_label()
        if label_id is None:
            return {}
        found: dict[str, dict[str, str]] = {}
        token: str | None = None
        while True:
            page = await self.transport.get(
                "users/me/messages",
                api_method="gmail.messages.list",
                params={
                    "labelIds": [label_id],
                    "maxResults": 200,
                    "pageToken": token,
                    "includeSpamTrash": False,
                },
            )
            for stub in page.get("messages") or []:
                meta = await self.transport.get(
                    f"users/me/messages/{stub['id']}",
                    api_method="gmail.messages.get",
                    params={
                        "format": "metadata",
                        "metadataHeaders": [HDR_SLUG, HDR_HASH],
                    },
                )
                headers = {
                    str(h.get("name", "")).lower(): str(h.get("value", ""))
                    for h in (meta.get("payload") or {}).get("headers") or []
                }
                slug = headers.get(HDR_SLUG.lower())
                if not slug:
                    continue
                found[slug] = {
                    "id": str(stub["id"]),
                    "hash": headers.get(HDR_HASH.lower(), ""),
                    "thread": str(meta.get("threadId") or ""),
                }
            token = page.get("nextPageToken")
            if not token:
                break
        return found

    # -- writes -------------------------------------------------------------- #

    #: CRLF line endings and a strictly 7-bit body. `cte_type="7bit"` is what
    #: pushes the Turkish message into base64 rather than leaving raw UTF-8
    #: bytes under an `8bit` encoding, which not every parser between here and
    #: the mailbox is obliged to accept.
    MAIL_POLICY = policy.SMTP.clone(cte_type="7bit")

    def raw(self, mail: Mail, refs: Sequence[str]) -> str:
        message = EmailMessage(policy=self.MAIL_POLICY)
        message["From"] = f"{mail.sender[0]} <{mail.sender[1]}>"
        message["To"] = ", ".join(mail.to)
        if mail.cc:
            message["Cc"] = ", ".join(mail.cc)
        message["Subject"] = mail.subject
        message["Date"] = format_datetime(mail.received_at)
        message["Message-ID"] = f"<{mail.slug}@{MSGID_DOMAIN}>"
        if refs:
            message["In-Reply-To"] = refs[-1]
            message["References"] = " ".join(refs)
        message[HDR_DEMO] = MARKER
        message[HDR_SLUG] = mail.slug
        message[HDR_HASH] = mail.fingerprint()
        message.set_content(mail.body)
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    async def insert(self, mail: Mail, refs: Sequence[str], thread_id: str | None) -> dict[str, Any]:
        label_id = await self.ensure_label()
        body: dict[str, Any] = {
            "raw": self.raw(mail, refs),
            "labelIds": ["INBOX", label_id],
        }
        if thread_id:
            body["threadId"] = thread_id
        from app.google.client import SAFE_TO_REPEAT

        return await self.transport.post(
            "users/me/messages",
            api_method="gmail.messages.insert",
            params={"internalDateSource": "dateHeader", "deleted": False},
            json=body,
            # An insert that may have landed is not repeated blind. A duplicate
            # here is not an error Google reports; it is two copies in the
            # mailbox and a demo that reads wrong.
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )

    async def trash(self, message_id: str) -> None:
        await self.transport.post(
            f"users/me/messages/{message_id}/trash",
            api_method="gmail.messages.trash",
            expect="json",
        )


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #

class CalendarPlanter:
    """Events carry their marker in `extendedProperties.private`.

    Calendar can filter a listing on a private extended property server-side,
    so finding what we planted is one call and never a scan of the user's
    whole calendar.
    """

    def __init__(self, clients: GoogleClients) -> None:
        self.service = clients.gcal
        self.transport = clients.gcal.transport

    async def existing(self) -> dict[str, dict[str, str]]:
        found: dict[str, dict[str, str]] = {}
        token: str | None = None
        while True:
            page = await self.transport.get(
                "calendars/primary/events",
                api_method="gcal.events.list",
                params={
                    "privateExtendedProperty": [f"{PROP_DEMO}={MARKER}"],
                    "maxResults": 250,
                    "singleEvents": False,
                    "showDeleted": False,
                    "pageToken": token,
                },
            )
            for item in page.get("items") or []:
                private = (item.get("extendedProperties") or {}).get("private") or {}
                slug = private.get(PROP_SLUG)
                if not slug:
                    continue
                found[slug] = {
                    "id": str(item.get("id")),
                    "hash": str(private.get(PROP_HASH, "")),
                    "etag": str(item.get("etag") or ""),
                }
            token = page.get("nextPageToken")
            if not token:
                break
        return found

    def body(self, event: Event) -> dict[str, Any]:
        from app.services.gcal import time_node

        body: dict[str, Any] = {
            "summary": event.title,
            "description": event.description or "",
            "location": event.location or "",
            "start": time_node(event.start, timezone=event.start_tz, all_day=event.all_day),
            "end": time_node(event.end, timezone=event.end_tz, all_day=event.all_day),
            "attendees": event.attendees,
            "reminders": {"useDefault": True},
            "extendedProperties": {
                "private": {
                    PROP_DEMO: MARKER,
                    PROP_SLUG: event.slug,
                    PROP_HASH: event.fingerprint(),
                }
            },
        }
        return body

    async def create(self, event: Event) -> dict[str, Any]:
        # sendUpdates stays "none": these guests are fictional addresses on
        # domains that do not accept mail, and a throwaway account should not
        # be firing invitations at them.
        return await self.service.events_insert(
            self.body(event),
            request_id=f"{MARKER}:{event.slug}:{event.fingerprint()}",
            send_updates="none",
        )

    async def update(self, event_id: str, event: Event, etag: str | None) -> dict[str, Any]:
        return await self.service.events_patch(
            event_id, self.body(event), etag=etag or None, send_updates="none"
        )

    async def remove(self, event_id: str) -> None:
        await self.service.events_delete(event_id, send_updates="none")


# --------------------------------------------------------------------------- #
# Drive
# --------------------------------------------------------------------------- #

class DrivePlanter:
    """Files carry their marker in `appProperties`, which only this app sees.

    Content goes up as a multipart upload, because `files.create` on the
    normal endpoint takes metadata only. `modifiedTime` is writable, which is
    what lets "PDFs from last month" mean anything; `createdTime` is not,
    which is the honest limit `sync_gdrive` already documents.
    """

    def __init__(self, clients: GoogleClients) -> None:
        self.service = clients.gdrive
        self.transport = clients.gdrive.transport
        self._folders: dict[str, str] = {}

    async def existing(self) -> dict[str, dict[str, str]]:
        found: dict[str, dict[str, str]] = {}
        token: str | None = None
        query = (
            f"appProperties has {{ key='{PROP_DEMO}' and value='{MARKER}' }}"
            " and trashed = false"
        )
        while True:
            page = await self.transport.get(
                "files",
                api_method="gdrive.files.list",
                params={
                    "q": query,
                    "pageSize": 200,
                    "pageToken": token,
                    "fields": "nextPageToken,files(id,name,mimeType,appProperties)",
                    "spaces": "drive",
                },
            )
            for item in page.get("files") or []:
                props = item.get("appProperties") or {}
                slug = props.get(PROP_SLUG)
                if not slug:
                    continue
                found[slug] = {
                    "id": str(item["id"]),
                    "hash": str(props.get(PROP_HASH, "")),
                    "name": str(item.get("name") or ""),
                }
            token = page.get("nextPageToken")
            if not token:
                break
        return found

    # -- folders ------------------------------------------------------------- #

    async def folder(self, path: str) -> str | None:
        """The id of a folder path such as `/Sales/Acme`, creating as it goes."""
        path = path.strip("/")
        if not path:
            return None
        if path in self._folders:
            return self._folders[path]
        parent: str | None = None
        walked: list[str] = []
        for segment in path.split("/"):
            walked.append(segment)
            key = "/".join(walked)
            if key in self._folders:
                parent = self._folders[key]
                continue
            found = await self._find_folder(segment, parent)
            if found is None:
                created = await self.transport.post(
                    "files",
                    api_method="gdrive.files.create",
                    params={"fields": "id"},
                    json={
                        "name": segment,
                        "mimeType": MIME_FOLDER,
                        "parents": [parent] if parent else ["root"],
                        "appProperties": {PROP_DEMO: MARKER, PROP_SLUG: f"folder:{key}"},
                    },
                )
                found = str(created["id"])
            self._folders[key] = found
            parent = found
        return parent

    async def _find_folder(self, name: str, parent: str | None) -> str | None:
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        clauses = [
            f"name = '{safe}'",
            f"mimeType = '{MIME_FOLDER}'",
            "trashed = false",
            f"'{parent or 'root'}' in parents",
        ]
        page = await self.transport.get(
            "files",
            api_method="gdrive.files.list",
            params={"q": " and ".join(clauses), "pageSize": 5, "fields": "files(id)"},
        )
        files = page.get("files") or []
        return str(files[0]["id"]) if files else None

    # -- writes --------------------------------------------------------------- #

    def metadata(self, doc: Doc, parent: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": doc.name,
            "mimeType": doc.mime_type,
            "modifiedTime": doc.modified_at.astimezone(dt.UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "appProperties": {
                PROP_DEMO: MARKER,
                PROP_SLUG: doc.slug,
                PROP_HASH: doc.fingerprint(),
            },
        }
        if parent:
            body["parents"] = [parent]
        return body

    @staticmethod
    def multipart(metadata: dict[str, Any], content: bytes, mime: str) -> tuple[bytes, str]:
        boundary = "alphalaw" + hashlib.blake2s(content, digest_size=8).hexdigest()
        sep = f"--{boundary}\r\n".encode()
        body = bytearray()
        body += sep
        body += b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        body += json.dumps(metadata).encode("utf-8")
        body += b"\r\n"
        body += sep
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += content
        body += f"\r\n--{boundary}--\r\n".encode()
        return bytes(body), f"multipart/related; boundary={boundary}"

    async def create(self, doc: Doc) -> dict[str, Any]:
        from app.google.client import SAFE_TO_REPEAT

        parent = await self.folder(doc.folder)
        metadata = self.metadata(doc, parent)
        body, content_type = self.multipart(metadata, doc.content(), doc.upload_mime)
        return await self.transport.post(
            f"{UPLOAD_BASE}/files",
            api_method="gdrive.files.create",
            params={"uploadType": "multipart", "fields": "id,name,modifiedTime"},
            content=body,
            headers={"Content-Type": content_type},
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )

    async def update(self, file_id: str, doc: Doc) -> dict[str, Any]:
        from app.google.client import SAFE_TO_REPEAT

        metadata = self.metadata(doc, None)   # parents move with addParents, not here
        body, content_type = self.multipart(metadata, doc.content(), doc.upload_mime)
        return await self.transport.patch(
            f"{UPLOAD_BASE}/files/{file_id}",
            api_method="gdrive.files.update",
            params={"uploadType": "multipart", "fields": "id,name,modifiedTime"},
            content=body,
            headers={"Content-Type": content_type},
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )

    async def touch_modified(self, file_id: str, doc: Doc) -> None:
        """Put `modifiedTime` back after an upload, which always bumps it."""
        await self.transport.patch(
            f"files/{file_id}",
            api_method="gdrive.files.update",
            params={"fields": "id,modifiedTime"},
            json={
                "modifiedTime": doc.modified_at.astimezone(dt.UTC)
                .isoformat()
                .replace("+00:00", "Z")
            },
        )

    async def trash(self, file_id: str) -> None:
        await self.transport.patch(
            f"files/{file_id}",
            api_method="gdrive.files.update",
            params={"fields": "id,trashed"},
            json={"trashed": True},
        )


# --------------------------------------------------------------------------- #
# Planting
# --------------------------------------------------------------------------- #

async def plant_mail(
    planter: GmailPlanter, mails: Sequence[Mail], out: list[Outcome]
) -> None:
    existing = await planter.existing()
    # slug -> (message-id header chain, gmail thread id), built as we go so a
    # reply can be threaded onto the message we just inserted.
    threads: dict[str, str] = {}
    refs_of: dict[str, list[str]] = {}

    for mail in mails:
        refs: list[str] = []
        if mail.reply_to:
            refs = [*refs_of.get(mail.reply_to, []), f"<{mail.reply_to}@{MSGID_DOMAIN}>"]
        refs_of[mail.slug] = refs

        thread_id = threads.get(mail.thread or "") or None
        current = existing.get(mail.slug)
        want = mail.fingerprint()

        if current and current["hash"] == want:
            if mail.thread and current.get("thread"):
                threads.setdefault(mail.thread, current["thread"])
            out.append(outcome("gmail", mail, UNCHANGED))
            continue

        try:
            inserted = await planter.insert(mail, refs, thread_id)
        except Exception as exc:  # one bad message must not lose the rest
            out.append(outcome("gmail", mail, FAILED, short(exc)))
            continue

        if mail.thread:
            threads.setdefault(mail.thread, str(inserted.get("threadId") or ""))

        if current:
            try:
                await planter.trash(current["id"])
            except Exception as exc:
                out.append(outcome("gmail", mail, UPDATED, f"old copy left: {short(exc)}"))
                continue
            out.append(outcome("gmail", mail, UPDATED))
        else:
            out.append(outcome("gmail", mail, CREATED))


async def plant_events(
    planter: CalendarPlanter, events: Sequence[Event], out: list[Outcome]
) -> None:
    existing = await planter.existing()
    for event in events:
        current = existing.get(event.slug)
        want = event.fingerprint()
        if current and current["hash"] == want:
            out.append(outcome("gcal", event, UNCHANGED))
            continue
        try:
            if current:
                await planter.update(current["id"], event, current.get("etag"))
                out.append(outcome("gcal", event, UPDATED))
            else:
                await planter.create(event)
                out.append(outcome("gcal", event, CREATED))
        except Exception as exc:
            out.append(outcome("gcal", event, FAILED, short(exc)))


async def plant_docs(
    planter: DrivePlanter, docs: Sequence[Doc], out: list[Outcome]
) -> None:
    existing = await planter.existing()
    for doc in docs:
        current = existing.get(doc.slug)
        want = doc.fingerprint()
        if current and current["hash"] == want:
            out.append(outcome("gdrive", doc, UNCHANGED))
            continue
        try:
            if current:
                await planter.update(current["id"], doc)
                await planter.touch_modified(current["id"], doc)
                out.append(outcome("gdrive", doc, UPDATED))
            else:
                created = await planter.create(doc)
                await planter.touch_modified(str(created["id"]), doc)
                out.append(outcome("gdrive", doc, CREATED))
        except Exception as exc:
            out.append(outcome("gdrive", doc, FAILED, short(exc)))


async def clean(clients: GoogleClients, services: set[str], out: list[Outcome]) -> None:
    if "gmail" in services:
        gmail = GmailPlanter(clients, clients.user_id)
        for slug, row in (await gmail.existing()).items():
            try:
                await gmail.trash(row["id"])
                out.append(Outcome("gmail", slug, slug, "", (), REMOVED, "to bin"))
            except Exception as exc:
                out.append(Outcome("gmail", slug, slug, "", (), FAILED, short(exc)))

    if "gcal" in services:
        cal = CalendarPlanter(clients)
        for slug, row in (await cal.existing()).items():
            try:
                await cal.remove(row["id"])
                out.append(Outcome("gcal", slug, slug, "", (), REMOVED, "deleted"))
            except Exception as exc:
                out.append(Outcome("gcal", slug, slug, "", (), FAILED, short(exc)))

    if "gdrive" in services:
        drive = DrivePlanter(clients)
        for slug, row in (await drive.existing()).items():
            try:
                await drive.trash(row["id"])
                out.append(
                    Outcome("gdrive", slug, row.get("name") or slug, "", (), REMOVED, "to bin")
                )
            except Exception as exc:
                out.append(Outcome("gdrive", slug, slug, "", (), FAILED, short(exc)))


def outcome(service: str, item: Any, action: str, detail: str = "") -> Outcome:
    return Outcome(
        service=service,
        slug=item.slug,
        label=item.label,
        when=item.when,
        supports=item.supports,
        action=action,
        detail=detail,
    )


def short(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {text[:120]}" if text else type(exc).__name__


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

SERVICE_TITLE = {"gmail": "Gmail", "gcal": "Calendar", "gdrive": "Drive"}


def print_table(outcomes: Sequence[Outcome]) -> None:
    if not outcomes:
        print("  nothing to report")
        return

    label_width = min(max((len(o.label) for o in outcomes), default=10), 46)
    when_width = max((len(o.when) for o in outcomes), default=0)

    for service in ("gmail", "gcal", "gdrive"):
        rows = [o for o in outcomes if o.service == service]
        if not rows:
            continue
        print(f"\n  {SERVICE_TITLE[service]}")
        print(
            f"    {'':1}  {'item':{label_width}}  {'when':{when_width}}  "
            f"{'supports':22}  what happened"
        )
        print(f"    {'-' * (label_width + when_width + 48)}")
        for row in rows:
            label = row.label if len(row.label) <= label_width else row.label[: label_width - 1] + "…"
            supports = ", ".join(row.supports) if row.supports else "—"
            if len(supports) > 22:
                supports = supports[:21] + "…"
            detail = f"  {row.detail}" if row.detail else ""
            print(
                f"    {GLYPH.get(row.action, '?')}  {label:{label_width}}  "
                f"{row.when:{when_width}}  {supports:22}  {row.action}{detail}"
            )


def print_scenarios(outcomes: Sequence[Outcome]) -> None:
    by_scenario: dict[str, list[str]] = {}
    for row in outcomes:
        for key in row.supports:
            by_scenario.setdefault(key, []).append(row.label)

    print("\n  scenario coverage — docs/SAMPLE_QUERIES.md")
    print(f"    {'id':6} {'query':62}  planted")
    print(f"    {'-' * 96}")
    for key, query in SCENARIOS.items():
        items = by_scenario.get(key, [])
        text = query if len(query) <= 62 else query[:61] + "…"
        if not items:
            note = "— needs no data" if key == "#15" else "— NOTHING PLANTED"
            print(f"    {key:6} {text:62}  {note}")
            continue
        print(f"    {key:6} {text:62}  {len(items)} item(s)")
        for label in items[:6]:
            print(f"    {'':6} {'':62}    {label[:60]}")
        if len(items) > 6:
            print(f"    {'':6} {'':62}    … and {len(items) - 6} more")

    print()
    print("    #15 is chit-chat the front door answers without touching a")
    print("    service, so it has nothing to plant. #9 has no data of its own")
    print("    either — it reads Tuesday out of the same events as #1, through")
    print("    an intent carried from the previous run.")
    print("    #S is not in SAMPLE_QUERIES.md. It is the near-identical file")
    print("    pair that makes a Drive share ambiguous, which is the write-side")
    print("    half of what #7 and #10 show on the read side.")


def print_deviations() -> None:
    print("\n  where this differs from docs/SAMPLE_QUERIES.md")
    for what, why in DEVIATIONS:
        print(f"\n    · {what}")
        for line in textwrap.wrap(why, width=72):
            print(f"      {line}")


def print_counts(outcomes: Sequence[Outcome]) -> None:
    counts: dict[str, int] = {}
    for row in outcomes:
        counts[row.action] = counts.get(row.action, 0) + 1
    summary = "  ".join(f"{GLYPH.get(k, '?')} {v} {k}" for k, v in sorted(counts.items()))
    print(f"\n  {summary or 'nothing done'}")
    failures = [row for row in outcomes if row.action == FAILED]
    if failures:
        print(f"\n  {len(failures)} item(s) failed:")
        for row in failures:
            print(f"    {row.service:7} {row.slug:28} {row.detail}")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

async def resolve_identity(email: str | None, user_id: str | None) -> tuple[str, str, str, int]:
    """`(user_id, email, display_name, week_start)` out of the database."""
    from app.db.models import User
    from app.db.session import session_scope
    from sqlalchemy import select

    async with session_scope() as session:
        statement = select(User)
        if user_id:
            statement = statement.where(User.id == user_id)
        elif email:
            statement = statement.where(User.email == email)
        else:
            raise SystemExit("give me --email or --user-id")
        user = (await session.execute(statement)).scalar_one_or_none()

    if user is None:
        raise SystemExit(
            f"no user for {user_id or email}. Connect the account through "
            f"GET /api/v1/auth/google first — see scripts/README.md."
        )
    name = user.display_name or user.email.split("@", 1)[0].replace(".", " ").title()
    return user.id, user.email, name, user.work_week_start


async def build_clients(user_id: str, access_token: str | None) -> GoogleClients:
    from app.db.session import session_scope
    from app.google.client import GoogleClients, clients_for

    if access_token:
        return GoogleClients.from_token(user_id, access_token)
    async with session_scope() as session:
        return await clients_for(session, user_id, share="background")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed_demo_account",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--email", help="the demo account's address")
    parser.add_argument("--user-id", help="the demo account's users.id, instead of --email")
    parser.add_argument(
        "--access-token",
        default=os.getenv("DEMO_ACCESS_TOKEN") or None,
        help="use this token and skip the database entirely (env DEMO_ACCESS_TOKEN)",
    )
    parser.add_argument(
        "--anchor",
        help="the local date every other date is computed from (default: today)",
    )
    parser.add_argument("--tz", help="IANA zone, if the user row's is wrong or absent")
    parser.add_argument(
        "--only",
        default="gmail,gcal,gdrive",
        help="comma-separated subset of gmail,gcal,gdrive",
    )
    parser.add_argument("--dry-run", action="store_true", help="print what it would do")
    parser.add_argument("--clean", action="store_true", help="remove what it planted, then stop")
    parser.add_argument("--reseed", action="store_true", help="clean, then plant")
    parser.add_argument(
        "--yes", action="store_true", help="do not ask before removing anything"
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    services = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = services - {"gmail", "gcal", "gdrive"}
    if unknown:
        raise SystemExit(f"--only does not know: {', '.join(sorted(unknown))}")

    anchor = (
        dt.date.fromisoformat(args.anchor)
        if args.anchor
        else dt.datetime.now().date()
    )

    # --dry-run must work on a bare checkout: no database, no network, no keys.
    if args.dry_run:
        email = args.email or "demo@example.com"
        name = email.split("@", 1)[0].replace(".", " ").title()
        tz = ZoneInfo(args.tz or "America/New_York")
        clock = Clock(anchor=anchor, tz=tz)
        data = build_dataset(clock, email, name)
        planned: list[Outcome] = [
            outcome(service, item, PLANNED)
            for service, items in (
                ("gmail", data.mails),
                ("gcal", data.events),
                ("gdrive", data.docs),
            )
            if service in services
            for item in items
        ]
        header(email, "(dry run — nothing was contacted)", clock)
        print_table(planned)
        print_scenarios(planned)
        print_deviations()
        print_counts(planned)
        print("\n  Drop --dry-run to write this to the account.\n")
        return 0

    if args.access_token:
        # With a raw token there is no users row, and the id is only ever used
        # as a quota and circuit-breaker key. A fixed 21-character string —
        # the width `app.core.ids.new_id` produces — keeps both honest.
        user_id = args.user_id or "seed_demo_account_raw"
        email = args.email or ""
        name = (email.split("@", 1)[0] or "Demo User").replace(".", " ").title()
        tz_name = args.tz or "America/New_York"
    else:
        user_id, email, name, _week_start = await resolve_identity(args.email, args.user_id)
        tz_name = args.tz or await user_timezone(user_id)

    clock = Clock(anchor=anchor, tz=ZoneInfo(tz_name))
    clients = await build_clients(user_id, args.access_token)

    try:
        if args.access_token and not email:
            profile = await clients.gmail.get_profile()
            email = str(profile.get("emailAddress") or "")
            name = email.split("@", 1)[0].replace(".", " ").title()

        header(email, user_id, clock)

        outcomes: list[Outcome] = []

        if args.clean or args.reseed:
            if not args.yes and not confirm(email, args.clean and not args.reseed):
                print("  nothing done")
                return 1
            await clean(clients, services, outcomes)
            if args.clean and not args.reseed:
                print_table(outcomes)
                print_counts(outcomes)
                print(
                    "\n  Gmail and Drive items went to the bin, not to nothing —"
                    "\n  empty it by hand if you want them gone for good."
                    "\n  Calendar has no bin, so those events are deleted.\n"
                )
                return 0

        data = build_dataset(clock, email, name)
        if "gmail" in services:
            await plant_mail(GmailPlanter(clients, email), data.mails, outcomes)
        if "gcal" in services:
            await plant_events(CalendarPlanter(clients), data.events, outcomes)
        if "gdrive" in services:
            await plant_docs(DrivePlanter(clients), data.docs, outcomes)

        planted = [o for o in outcomes if o.action != REMOVED]
        print_table(outcomes)
        print_scenarios(planted)
        print_deviations()
        print_counts(outcomes)
        print_next_steps(email)
        return 1 if any(o.action == FAILED for o in outcomes) else 0
    finally:
        await clients.aclose()
        await close_engine()


async def user_timezone(user_id: str) -> str:
    from app.db.repositories import users as users_repo
    from app.db.session import session_scope

    async with session_scope() as session:
        user = await users_repo.get_user(session, user_id)
    return (user.timezone if user and user.timezone else None) or "America/New_York"


async def close_engine() -> None:
    """Let the pool go. Nothing here is worth failing a finished run over."""
    try:
        from app.db.session import shutdown_engine
    except ImportError:
        return
    with contextlib.suppress(Exception):
        await shutdown_engine()


def confirm(email: str, removing_only: bool) -> bool:
    what = "remove everything this script planted" if removing_only else "remove and replant"
    print(f"\n  About to {what} in {email}.")
    print("  This writes to a real Google account. It should be a throwaway.")
    try:
        answer = input("  Type the account address to continue: ").strip()
    except EOFError:
        return False
    return answer == email


def header(email: str, user_id: str, clock: Clock) -> None:
    print()
    print("  seed_demo_account")
    print(f"    account      {email}")
    print(f"    user         {user_id}")
    print(f"    anchor       {clock.anchor.isoformat()} ({clock.anchor.strftime('%A')})")
    print(f"    timezone     {clock.tz}")
    print(f"    tomorrow     {clock.tomorrow.isoformat()}")
    print(
        f"    next week    {clock.next_week(0).isoformat()} .. "
        f"{clock.next_week(6).isoformat()}"
    )
    print(
        f"    last month   {clock.last_month_first.isoformat()} .. "
        f"{(clock.this_month_first - dt.timedelta(days=1)).isoformat()}"
    )
    print(f"    marker       {MARKER}")


def print_next_steps(email: str) -> None:
    print()
    print("  next")
    print("    1. POST /api/v1/sync/trigger {\"mode\": \"full\"} — pull it all")
    print("       into the mirror, then GET /api/v1/sync/status until the")
    print("       three cursors have moved. scripts/README.md has the curl,")
    print("       and the direct enqueue to fall back on.")
    print("    2. python -m tests.eval.precision_at_k --write-results")
    print("       python -m tests.eval.intent_accuracy")
    print("       python -m tests.eval.latency")
    print("    3. Try: \"Cancel my Turkish Airlines flight\" — the booking is in")
    print("       Turkish, so this exercises the escalation ladder as well.")
    print("       \"Cancel my flight to Istanbul\" — two carriers, it should ask.")
    print("       \"Move the meeting with John\" — two Johns, it should ask.")
    print()
    print("  Dates are anchored to the day you ran this. Re-run before recording")
    print("  if the calendar has slid into a different week.")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not (args.email or args.user_id or args.access_token):
        raise SystemExit("give me --email, --user-id or --access-token")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
