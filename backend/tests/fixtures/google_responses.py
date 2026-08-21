"""Recorded Google payloads, and the router that serves them.

Everything here is the real wire shape: Gmail's base64url message parts and
``internalDate`` in milliseconds, Calendar's ``etag`` and ``nextSyncToken``,
Drive's ``files`` array, and Google's error envelope with its ``errors[].reason``
— which is what ``app.google.retry.classify()`` reads to tell RATE_LIMITED from
QUOTA_EXHAUSTED.

The corpus is the one ``docs/SAMPLE_QUERIES.md`` uses, so a failing assertion can
be read against the document. Same user, same instant:

    now = 2026-08-20T13:12:04Z  (Thursday, ISO week 34)
    tz  = America/New_York      (EDT, UTC-4 in August 2026)

Three groups of content, deliberately overlapping so tenant-isolation tests have
something to confuse:

* a **Turkish Airlines** booking, in English *and* in Turkish (`bilet@thy.com`,
  subject ``Uçuş rezervasyonunuz onaylandı — TK1984``) — the Turkish one is the
  zero-hit case that the escalation ladder recovers, and it is here to prove the
  regex extractors are language-independent;
* an **Acme Corp** renewal thread across Gmail, Calendar and Drive;
* two people called **John**, which is what makes "move the meeting with John"
  ambiguous.

:func:`resolve` maps one HTTP request onto one of these payloads. It is pure —
no failure injection, no counters. ``tests/integration/conftest.py`` owns both.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------- #
# When everything is
# --------------------------------------------------------------------------- #
#
# `docs/SAMPLE_QUERIES.md` works every example against one fixed instant —
# Thursday 20 August 2026, 13:12 UTC — so the date arithmetic can be checked by
# hand. This corpus keeps every *relationship* from that document (a meeting
# tomorrow, six events next week, three PDFs last month, a flight a fortnight
# out) but hangs them off the real clock.
#
# A fixture pinned to a calendar date passes on the day it was written and rots
# quietly afterwards, which is a worse failure than not having the test: it
# starts reporting that "next week" is broken when all that happened is that
# next week arrived.

USER_EMAIL = "demo@alphalaw.test"
USER_NAME = "Demo Ozturk"
USER_TZ = "America/New_York"
TZ = ZoneInfo(USER_TZ)

NOW = datetime.now(UTC)
TODAY = NOW.astimezone(TZ).date()


def at_local(day: date, hour: int = 0, minute: int = 0, tz: ZoneInfo = TZ) -> datetime:
    """A local wall time, as UTC. Never build a naive datetime in a fixture."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).astimezone(UTC)


def ago(days: int, hour: int = 9, minute: int = 0) -> datetime:
    return at_local(TODAY - timedelta(days=days), hour, minute)


#: Monday of the ISO week after this one, honouring work_week_start = 1.
NEXT_MONDAY = TODAY - timedelta(days=TODAY.weekday()) + timedelta(days=7)
NEXT_WEEK_START = at_local(NEXT_MONDAY)
NEXT_WEEK_END = at_local(NEXT_MONDAY + timedelta(days=7))

TOMORROW = TODAY + timedelta(days=1)
TOMORROW_START = at_local(TOMORROW)
TOMORROW_END = at_local(TOMORROW + timedelta(days=1))

#: True only when today is a Sunday, when "tomorrow" is also the first day of
#: "next week". Two windows the rest of the time; one test has to know.
TOMORROW_IS_NEXT_WEEK = NEXT_WEEK_START <= TOMORROW_START < NEXT_WEEK_END

_first_of_this_month = TODAY.replace(day=1)
_last_month_end = _first_of_this_month
_last_month_start = (_first_of_this_month - timedelta(days=1)).replace(day=1)
LAST_MONTH_START = at_local(_last_month_start)
LAST_MONTH_END = at_local(_last_month_end)


def last_month(day_of_month: int, hour: int = 10) -> datetime:
    """A day inside the previous calendar month. Capped at 26 so February works."""
    return at_local(_last_month_start + timedelta(days=min(day_of_month, 26) - 1), hour)


#: The flight is a fortnight past next week, so it is comfortably in the future
#: whatever day the suite runs, and the booking email is a month old.
FLIGHT_DEPARTS_LOCAL = NEXT_MONDAY + timedelta(days=12)
FLIGHT_DEPARTS = at_local(FLIGHT_DEPARTS_LOCAL, 10, 30, ZoneInfo("Europe/Istanbul"))
FLIGHT_ARRIVES = FLIGHT_DEPARTS + timedelta(hours=10, minutes=45)
SECOND_FLIGHT_DEPARTS = at_local(
    NEXT_MONDAY + timedelta(days=40), 9, 0, ZoneInfo("Europe/Istanbul")
)

BOOKED_AT = ago(28, 9, 41)

#: The out-of-office covers the middle of next week, so there is a real conflict
#: to find rather than a contrived one.
OOO_FROM = NEXT_MONDAY + timedelta(days=2)
OOO_TO = NEXT_MONDAY + timedelta(days=3)

_TURKISH_MONTHS = (
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
)


def english_date(day: date) -> str:
    return f"{day.day} {day.strftime('%B')} {day.year}"


def turkish_date(day: date) -> str:
    return f"{day.day} {_TURKISH_MONTHS[day.month - 1]} {day.year}"


GMAIL_HOSTS = ("gmail.googleapis.com", "www.googleapis.com")
CALENDAR_HOSTS = ("www.googleapis.com", "calendar.googleapis.com")
DRIVE_HOSTS = ("www.googleapis.com", "drive.googleapis.com")


# --------------------------------------------------------------------------- #
# Small builders
# --------------------------------------------------------------------------- #


def b64url(text: str | bytes) -> str:
    """Gmail's body encoding: base64url, padding stripped."""
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def unb64url(data: str) -> str:
    """The inverse, for a test that wants to read a draft back."""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", "replace")


def epoch_ms(moment: datetime) -> str:
    return str(int(moment.timestamp() * 1000))


def content_hash(text: str) -> uuid.UUID:
    """A stable fingerprint for a mirror row.

    Not `app.core.ids.fingerprint` — these fixtures deliberately import nothing
    from `app`. What matters for the tests that use it is only that the same
    text gives the same uuid twice running.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, "alpha-law/test/content\x1f" + text)


def gmail_message(
    *,
    message_id: str,
    thread_id: str | None = None,
    subject: str,
    from_name: str,
    from_email: str,
    to: tuple[str, ...] = (USER_EMAIL,),
    body: str,
    received_at: datetime,
    labels: tuple[str, ...] = ("INBOX",),
    history_id: str = "9912841",
    has_attachments: bool = False,
) -> dict[str, Any]:
    """One message in Gmail's ``format=full`` shape."""
    thread = thread_id or message_id
    headers = [
        {"name": "Delivered-To", "value": USER_EMAIL},
        {"name": "Date", "value": received_at.strftime("%a, %d %b %Y %H:%M:%S %z")},
        {"name": "From", "value": f"{from_name} <{from_email}>"},
        {"name": "To", "value": ", ".join(to)},
        {"name": "Subject", "value": subject},
        {"name": "Message-ID", "value": f"<{message_id}@mail.gmail.com>"},
        {"name": "MIME-Version", "value": "1.0"},
        {"name": "Content-Type", "value": 'text/plain; charset="UTF-8"'},
    ]
    parts: list[dict[str, Any]] = [
        {
            "partId": "0",
            "mimeType": "text/plain",
            "filename": "",
            "headers": [{"name": "Content-Type", "value": 'text/plain; charset="UTF-8"'}],
            "body": {"size": len(body.encode()), "data": b64url(body)},
        }
    ]
    if has_attachments:
        parts.append(
            {
                "partId": "1",
                "mimeType": "application/pdf",
                "filename": "itinerary.pdf",
                "headers": [{"name": "Content-Type", "value": "application/pdf"}],
                "body": {"attachmentId": f"att_{message_id}", "size": 88213},
            }
        )
    return {
        "id": message_id,
        "threadId": thread,
        "labelIds": list(labels),
        "snippet": body.strip().splitlines()[0][:180],
        "historyId": history_id,
        "internalDate": epoch_ms(received_at),
        "sizeEstimate": 2048 + len(body),
        "payload": {
            "partId": "",
            "mimeType": "multipart/mixed" if has_attachments else "text/plain",
            "filename": "",
            "headers": headers,
            "body": {"size": 0},
            "parts": parts,
        },
    }


def plain_body(message: dict[str, Any]) -> str:
    """Pull the text/plain part back out of a message payload."""
    payload = message.get("payload") or {}
    for part in payload.get("parts") or [payload]:
        if part.get("mimeType") == "text/plain":
            data = (part.get("body") or {}).get("data")
            if data:
                return unb64url(data)
    return ""


def header(message: dict[str, Any], name: str) -> str:
    for item in (message.get("payload") or {}).get("headers") or []:
        if item["name"].lower() == name.lower():
            return item["value"]
    return ""


def google_error(
    status: int,
    reason: str,
    message: str,
    *,
    domain: str = "usageLimits",
    status_name: str | None = None,
) -> dict[str, Any]:
    """Google's error envelope, exactly as it comes off the wire."""
    return {
        "error": {
            "code": status,
            "message": message,
            "status": status_name or _STATUS_NAMES.get(status, "UNKNOWN"),
            "errors": [{"domain": domain, "reason": reason, "message": message}],
        }
    }


_STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    410: "NOT_FOUND",
    412: "FAILED_PRECONDITION",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}


# --------------------------------------------------------------------------- #
# The error payloads the reliability tests need
# --------------------------------------------------------------------------- #

ERROR_429 = google_error(
    429,
    "userRateLimitExceeded",
    "User-rate limit exceeded. Retry after 2026-08-20T13:12:34.000Z",
)
ERROR_429_HEADERS = {"Retry-After": "2"}

ERROR_QUOTA = google_error(
    429, "dailyLimitExceeded", "Daily Limit Exceeded", domain="usageLimits"
)

ERROR_503 = google_error(
    503,
    "backendError",
    "The service is currently unavailable.",
    domain="global",
    status_name="UNAVAILABLE",
)

ERROR_401 = google_error(
    401,
    "authError",
    "Invalid Credentials",
    domain="global",
    status_name="UNAUTHENTICATED",
)

ERROR_412 = google_error(
    412,
    "conditionNotMet",
    "Precondition Failed",
    domain="global",
    status_name="FAILED_PRECONDITION",
)

#: Gmail's historyId is too old / Calendar's syncToken has expired. The only
#: honest answer is a bounded full resync.
ERROR_410 = google_error(
    410,
    "failedPrecondition",
    "Requested entity was not found.",
    domain="global",
    status_name="NOT_FOUND",
)

#: The refresh grant is dead. This is what turns AUTH_EXPIRED into AUTH_REVOKED.
ERROR_INVALID_GRANT = {
    "error": "invalid_grant",
    "error_description": "Token has been expired or revoked.",
}


# --------------------------------------------------------------------------- #
# Gmail corpus
# --------------------------------------------------------------------------- #

PNR = "6F2QK9"
TICKET_NO = "TK1984"
FLIGHT_NO = "TK1"
SUPPORT_EMAIL = "cancel@turkishairlines.com"

TK_BOOKING_EN_BODY = f"""Dear Mr Ozturk,

Your booking is confirmed.

Booking reference (PNR): {PNR}
Ticket number: {TICKET_NO}
Flight: {FLIGHT_NO}, Istanbul (IST) -> New York (JFK)
Departure: {english_date(FLIGHT_DEPARTS_LOCAL)}, 10:30 (+03:00)
Passenger: {USER_NAME}
Total paid: USD 812.40

To change or cancel this booking, reply to this message or write to
{SUPPORT_EMAIL} quoting your booking reference.

Turkish Airlines
"""

# The same confirmation, in Turkish, and this one carries no cancellation
# address. Two things it is here to prove: the vector leg scores it near zero
# against an English query, and the regex extractors do not care — a PNR is a
# shape, and shapes survive translation.
TK_BOOKING_TR_BODY = f"""Sayın Yolcumuz,

{PNR} numaralı rezervasyonunuz onaylanmıştır.

Bilet numarası: {TICKET_NO}
Uçuş: {FLIGHT_NO}, İstanbul (IST) -> New York (JFK)
Kalkış: {turkish_date(FLIGHT_DEPARTS_LOCAL)}, 10:30 (+03:00)
Yolcu: {USER_NAME}
Toplam: 812,40 USD

İyi yolculuklar dileriz.
Türk Hava Yolları
"""

TK_PROMO_BODY = """Autumn is calling.

25% off selected fares to Istanbul, Ankara and Izmir when you book this month.
Terms apply.

Turkish Airlines
"""

SARAH_BUDGET_BODY = """Hi,

Draft numbers for the Q3 budget are attached below in the summary. Headline:
we are 4% under on headcount and 11% over on cloud. The cloud overrun is the
Postgres replica fleet, which we knew about.

I need your comments on the marketing line before Friday.

Sarah
"""

SARAH_BUDGET_FOLLOWUP_BODY = """Quick one - can we move the budget review to
next week? Thursday afternoon works for me, and it gives us time to fold in
your comments on the marketing line.

Sarah
"""

ACME_PRICING_BODY = """Hi both,

Revised pricing attached. We can do a 14-month term at the 12-month rate,
which is the best I can get signed off this quarter.

Before the call on Friday, could you send the security questionnaire back?
Legal will not counter-sign without it.

Dana
Acme Corp
"""

ACME_MSA_BODY = """Two redlines on the MSA that are still open from our side:

1. The liability cap at 12 months of fees - we need 24.
2. Data residency: our customers require EU-only storage.

Neither is a deal breaker but both need an answer before we sign.

Marcus
Acme Corp
"""

JOHN_OKAFOR_BODY = """Confirming our 1:1 for next Tuesday at 4pm. I would like
to talk through the renewal pipeline and the two open reqs on my team.

John Okafor
"""

JOHN_REYES_BODY = """Following up on the Northwind vendor sync. Wednesday
morning still suits us. I will bring the updated SOC 2 report.

John Reyes
Northwind
"""

MSG_TK_BOOKING_EN = "18f2c9a4b7e10d33"
MSG_TK_BOOKING_TR = "18f2c9a4b7e10d44"
MSG_TK_PROMO = "18f4d1b7c2e30a91"
MSG_SARAH_BUDGET = "18f6a2c9d4e51b02"
MSG_SARAH_FOLLOWUP = "18f6a2c9d4e51b13"
MSG_ACME_PRICING = "18f81b3e5a7c2d64"
MSG_ACME_MSA = "18f70c2d4b6e1a55"
MSG_JOHN_OKAFOR = "18f9c4e6b8a02f77"
MSG_JOHN_REYES = "18f9c4e6b8a02f88"

MESSAGES: dict[str, dict[str, Any]] = {
    MSG_TK_BOOKING_EN: gmail_message(
        message_id=MSG_TK_BOOKING_EN,
        subject=f"Your Turkish Airlines booking is confirmed - {TICKET_NO}",
        from_name="Turkish Airlines",
        from_email="noreply@turkishairlines.com",
        body=TK_BOOKING_EN_BODY,
        received_at=BOOKED_AT,
        labels=("INBOX", "CATEGORY_UPDATES"),
        has_attachments=True,
    ),
    MSG_TK_BOOKING_TR: gmail_message(
        message_id=MSG_TK_BOOKING_TR,
        subject=f"Uçuş rezervasyonunuz onaylandı - {TICKET_NO}",
        from_name="Türk Hava Yolları",
        from_email="bilet@thy.com",
        body=TK_BOOKING_TR_BODY,
        received_at=BOOKED_AT + timedelta(minutes=3),
        labels=("INBOX", "CATEGORY_UPDATES"),
    ),
    MSG_TK_PROMO: gmail_message(
        message_id=MSG_TK_PROMO,
        subject="Turkish Airlines - 25% off autumn fares",
        from_name="Turkish Airlines",
        from_email="offers@turkishairlines.com",
        body=TK_PROMO_BODY,
        received_at=ago(18, 6, 15),
        labels=("INBOX", "CATEGORY_PROMOTIONS"),
    ),
    MSG_SARAH_BUDGET: gmail_message(
        message_id=MSG_SARAH_BUDGET,
        subject="Q3 budget - draft numbers",
        from_name="Sarah Chen",
        from_email="sarah@company.com",
        body=SARAH_BUDGET_BODY,
        received_at=ago(8, 11, 3),
    ),
    MSG_SARAH_FOLLOWUP: gmail_message(
        message_id=MSG_SARAH_FOLLOWUP,
        thread_id=MSG_SARAH_BUDGET,
        subject="Re: Q3 budget - draft numbers",
        from_name="Sarah Chen",
        from_email="sarah@company.com",
        body=SARAH_BUDGET_FOLLOWUP_BODY,
        received_at=ago(3, 7, 22),
    ),
    MSG_ACME_PRICING: gmail_message(
        message_id=MSG_ACME_PRICING,
        subject="Re: renewal pricing - revised",
        from_name="Dana Whitfield",
        from_email="dana@acmecorp.com",
        to=(USER_EMAIL, "sarah@company.com"),
        body=ACME_PRICING_BODY,
        received_at=ago(2, 9, 5),
    ),
    MSG_ACME_MSA: gmail_message(
        message_id=MSG_ACME_MSA,
        subject="Acme MSA - two open redlines",
        from_name="Marcus Iyer",
        from_email="marcus@acmecorp.com",
        to=(USER_EMAIL, "sarah@company.com"),
        body=ACME_MSA_BODY,
        received_at=ago(9, 12, 48),
    ),
    MSG_JOHN_OKAFOR: gmail_message(
        message_id=MSG_JOHN_OKAFOR,
        subject="Next week's 1:1",
        from_name="John Okafor",
        from_email="john.okafor@company.com",
        body=JOHN_OKAFOR_BODY,
        received_at=ago(1, 10, 2),
    ),
    MSG_JOHN_REYES: gmail_message(
        message_id=MSG_JOHN_REYES,
        subject="Northwind vendor sync",
        from_name="John Reyes",
        from_email="john.reyes@northwind.io",
        body=JOHN_REYES_BODY,
        received_at=ago(1, 14, 26),
    ),
}

#: Newest first, the order Gmail returns them in.
MESSAGE_ORDER = [
    MSG_ACME_PRICING,
    MSG_SARAH_FOLLOWUP,
    MSG_JOHN_REYES,
    MSG_JOHN_OKAFOR,
    MSG_SARAH_BUDGET,
    MSG_ACME_MSA,
    MSG_TK_PROMO,
    MSG_TK_BOOKING_TR,
    MSG_TK_BOOKING_EN,
]

GMAIL_PROFILE = {
    "emailAddress": USER_EMAIL,
    "messagesTotal": 18422,
    "threadsTotal": 9110,
    "historyId": "9912841",
}


def gmail_messages_list(ids: list[str] | None = None, page_token: str | None = None) -> dict:
    """``users.messages.list`` — ids and thread ids only, which is why a sync
    always costs a second round trip per message."""
    chosen = ids if ids is not None else MESSAGE_ORDER
    return {
        "messages": [
            {"id": mid, "threadId": MESSAGES[mid]["threadId"]} for mid in chosen
        ],
        "resultSizeEstimate": len(chosen),
        **({"nextPageToken": page_token} if page_token else {}),
    }


GMAIL_MESSAGES_LIST = gmail_messages_list()

#: ``users.history.list`` from a live cursor: two new messages, one label
#: change, and the historyId to store next.
GMAIL_HISTORY = {
    "history": [
        {
            "id": "9912855",
            "messages": [{"id": MSG_ACME_PRICING, "threadId": MSG_ACME_PRICING}],
            "messagesAdded": [
                {
                    "message": {
                        "id": MSG_ACME_PRICING,
                        "threadId": MSG_ACME_PRICING,
                        "labelIds": ["INBOX"],
                    }
                }
            ],
        },
        {
            "id": "9912861",
            "messages": [{"id": MSG_SARAH_FOLLOWUP, "threadId": MSG_SARAH_BUDGET}],
            "messagesAdded": [
                {
                    "message": {
                        "id": MSG_SARAH_FOLLOWUP,
                        "threadId": MSG_SARAH_BUDGET,
                        "labelIds": ["INBOX", "UNREAD"],
                    }
                }
            ],
        },
    ],
    "historyId": "9912903",
}

#: What the mirror gets back after a 410, when it walks the mailbox again.
GMAIL_HISTORY_IDS = {"added": [MSG_ACME_PRICING, MSG_SARAH_FOLLOWUP]}

GMAIL_DRAFT_ID = "r-8827441290034"
GMAIL_DRAFT_MESSAGE_ID = "18fa0d1e2c3b4a56"
GMAIL_SENT_MESSAGE_ID = "18fa0d1e2c3b4b67"

GMAIL_DRAFT_CREATED = {
    "id": GMAIL_DRAFT_ID,
    "message": {
        "id": GMAIL_DRAFT_MESSAGE_ID,
        "threadId": MSG_TK_BOOKING_EN,
        "labelIds": ["DRAFT"],
    },
}

GMAIL_DRAFT_SENT = {
    "id": GMAIL_SENT_MESSAGE_ID,
    "threadId": MSG_TK_BOOKING_EN,
    "labelIds": ["SENT"],
}


# --------------------------------------------------------------------------- #
# Calendar corpus
# --------------------------------------------------------------------------- #


def calendar_event(
    *,
    event_id: str,
    summary: str,
    starts_at: datetime,
    ends_at: datetime,
    attendees: tuple[tuple[str, str], ...] = (),
    description: str = "",
    location: str = "",
    etag: str,
    organizer: str = USER_EMAIL,
    timezone: str = USER_TZ,
    status: str = "confirmed",
    all_day: bool = False,
    recurring_event_id: str | None = None,
) -> dict[str, Any]:
    """One event in Calendar's v3 shape, ``etag`` included — that is what an
    ``If-Match`` update is checked against."""
    if all_day:
        start = {"date": starts_at.date().isoformat()}
        end = {"date": ends_at.date().isoformat()}
    else:
        start = {"dateTime": starts_at.isoformat().replace("+00:00", "Z"), "timeZone": timezone}
        end = {"dateTime": ends_at.isoformat().replace("+00:00", "Z"), "timeZone": timezone}
    event: dict[str, Any] = {
        "kind": "calendar#event",
        "etag": etag,
        "id": event_id,
        "status": status,
        "htmlLink": f"https://www.google.com/calendar/event?eid={event_id}",
        "created": (starts_at - timedelta(days=21)).isoformat().replace("+00:00", "Z"),
        "updated": (starts_at - timedelta(days=3)).isoformat().replace("+00:00", "Z"),
        "summary": summary,
        "description": description,
        "location": location,
        "creator": {"email": organizer, "self": organizer == USER_EMAIL},
        "organizer": {"email": organizer, "self": organizer == USER_EMAIL},
        "start": start,
        "end": end,
        "iCalUID": f"{event_id}@google.com",
        "sequence": 0,
        "reminders": {"useDefault": True},
        "eventType": "default",
    }
    if attendees:
        event["attendees"] = [
            {
                "email": email,
                "displayName": name,
                "responseStatus": "accepted" if email == USER_EMAIL else "needsAction",
                "self": email == USER_EMAIL,
                "optional": False,
            }
            for email, name in attendees
        ]
    if recurring_event_id:
        event["recurringEventId"] = recurring_event_id
    return event


EVT_FLIGHT_TK1984 = "3k9m2p4q8r1s3t5u7v9w"
EVT_FLIGHT_TK1979 = "4l0n3q5r9s2t4u6v8w0x"
EVT_ACME_RENEWAL = "5m1o4r6s0t3u5v7w9x1y"
EVT_ACME_REVIEW = "9r2k4m_20260827T170000Z"
EVT_JOHN_OKAFOR = "3k9m2p_20260825T200000Z"
EVT_JOHN_REYES = "7t4v8q_20260826T130000Z"
EVT_DESIGN_REVIEW = "8u5w9r_20260825T150000Z"
EVT_STANDUP = "2j8l1o_20260824T133000Z"
EVT_OFFSITE = "6s3u7p_20260828"

def _next_week(offset: int, hour: int, minute: int = 0) -> datetime:
    """A local time on a given weekday of next week. 0 = Monday."""
    return at_local(NEXT_MONDAY + timedelta(days=offset), hour, minute)


EVENTS: dict[str, dict[str, Any]] = {
    # Two flights. Both Turkish Airlines, which is exactly what makes
    # "cancel my flight to Istanbul" ambiguous and "cancel my Turkish Airlines
    # flight" not.
    EVT_FLIGHT_TK1984: calendar_event(
        event_id=EVT_FLIGHT_TK1984,
        summary=f"Istanbul -> NYC Flight ({TICKET_NO})",
        description=f"{FLIGHT_NO}  IST -> JFK\nBooking reference {PNR}\nSeat 14A",
        location="Istanbul Airport (IST)",
        starts_at=FLIGHT_DEPARTS,
        ends_at=FLIGHT_ARRIVES,
        timezone="Europe/Istanbul",
        etag='"3401882940110000"',
    ),
    EVT_FLIGHT_TK1979: calendar_event(
        event_id=EVT_FLIGHT_TK1979,
        summary="Istanbul -> London Flight (TK1979)",
        description="TK1979  IST -> LHR\nBooking reference 9J4XP2",
        location="Istanbul Airport (IST)",
        starts_at=SECOND_FLIGHT_DEPARTS,
        ends_at=SECOND_FLIGHT_DEPARTS + timedelta(hours=4, minutes=5),
        timezone="Europe/Istanbul",
        etag='"3401882940220000"',
    ),
    # Tomorrow. The Acme meeting the prep query is about.
    EVT_ACME_RENEWAL: calendar_event(
        event_id=EVT_ACME_RENEWAL,
        summary="Acme Corp - Q3 renewal review",
        description="Agenda: pricing, security questionnaire, MSA redlines.",
        location="Google Meet",
        starts_at=at_local(TOMORROW, 10, 0),
        ends_at=at_local(TOMORROW, 11, 0),
        attendees=(
            (USER_EMAIL, USER_NAME),
            ("dana@acmecorp.com", "Dana Whitfield"),
            ("marcus@acmecorp.com", "Marcus Iyer"),
            ("sarah@company.com", "Sarah Chen"),
        ),
        etag='"3401882940330000"',
    ),
    # Next week: Mon, Tue, Tue, Wed, Thu, Fri. Six events, which is what the
    # worked example in docs/SAMPLE_QUERIES.md lists.
    EVT_STANDUP: calendar_event(
        event_id=EVT_STANDUP,
        summary="Standup",
        starts_at=_next_week(0, 9, 30),
        ends_at=_next_week(0, 10, 0),
        attendees=((USER_EMAIL, USER_NAME), ("sarah@company.com", "Sarah Chen")),
        etag='"3401882940440000"',
        recurring_event_id="2j8l1o",
    ),
    EVT_DESIGN_REVIEW: calendar_event(
        event_id=EVT_DESIGN_REVIEW,
        summary="Design review",
        description="Walkthrough of the new trace panel.",
        starts_at=_next_week(1, 11, 0),
        ends_at=_next_week(1, 12, 0),
        attendees=(
            (USER_EMAIL, USER_NAME),
            ("john@company.com", "John Adeyemi"),
            ("sarah@company.com", "Sarah Chen"),
        ),
        etag='"3401882940550000"',
    ),
    EVT_JOHN_OKAFOR: calendar_event(
        event_id=EVT_JOHN_OKAFOR,
        summary="1:1 with John Okafor",
        starts_at=_next_week(1, 16, 0),
        ends_at=_next_week(1, 16, 30),
        attendees=((USER_EMAIL, USER_NAME), ("john.okafor@company.com", "John Okafor")),
        etag='"3401882940660000"',
    ),
    EVT_JOHN_REYES: calendar_event(
        event_id=EVT_JOHN_REYES,
        summary="Vendor sync - John Reyes (Northwind)",
        starts_at=_next_week(2, 9, 0),
        ends_at=_next_week(2, 9, 45),
        attendees=((USER_EMAIL, USER_NAME), ("john.reyes@northwind.io", "John Reyes")),
        etag='"3401882940770000"',
    ),
    EVT_ACME_REVIEW: calendar_event(
        event_id=EVT_ACME_REVIEW,
        summary="Acme review",
        description="Quarterly review with Acme.",
        starts_at=_next_week(3, 13, 0),
        ends_at=_next_week(3, 14, 0),
        attendees=(
            (USER_EMAIL, USER_NAME),
            ("dana@acmecorp.com", "Dana Whitfield"),
            ("marcus@acmecorp.com", "Marcus Iyer"),
            ("sarah@company.com", "Sarah Chen"),
        ),
        etag='"3401882940880000"',
    ),
    EVT_OFFSITE: calendar_event(
        event_id=EVT_OFFSITE,
        summary="Company offsite",
        starts_at=_next_week(4, 0),
        ends_at=_next_week(5, 0),
        all_day=True,
        etag='"3401882940990000"',
    ),
}

CALENDAR_SYNC_TOKEN = "CPjqvbLh1_YCEPjqvbLh1_YCGAUgg8vGpAI="

#: The two flights and the Acme meeting, which is the list `events.list` returns
#: for an unbounded query.
CALENDAR_EVENTS_LIST = {
    "kind": "calendar#events",
    "etag": '"p32ofplf5o8fe20g"',
    "summary": USER_EMAIL,
    "updated": NOW.isoformat().replace("+00:00", "Z"),
    "timeZone": USER_TZ,
    "accessRole": "owner",
    "defaultReminders": [{"method": "popup", "minutes": 10}],
    "nextSyncToken": CALENDAR_SYNC_TOKEN,
    "items": [
        EVENTS[EVT_ACME_RENEWAL],
        EVENTS[EVT_FLIGHT_TK1984],
        EVENTS[EVT_FLIGHT_TK1979],
    ],
}


def calendar_events_list(
    event_ids: list[str] | None = None, *, sync_token: str | None = CALENDAR_SYNC_TOKEN
) -> dict[str, Any]:
    chosen = event_ids if event_ids is not None else list(EVENTS)
    body = dict(CALENDAR_EVENTS_LIST)
    body["items"] = [EVENTS[e] for e in chosen]
    if sync_token:
        body["nextSyncToken"] = sync_token
    else:
        body.pop("nextSyncToken", None)
    return body


def events_in_window(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Half-open [start, end), which is how every window in this system works."""
    out = []
    for event in EVENTS.values():
        when = event_start(event)
        if start <= when < end:
            out.append(event)
    return sorted(out, key=event_start)


def _edge(field: dict[str, Any]) -> datetime:
    """A Calendar start/end, as UTC.

    An all-day event has a ``date`` and no zone: midnight *local*, which is a
    different instant in June and December. Reading it as UTC is the bug that
    makes an all-day event fall out of a window at the boundary.
    """
    if "dateTime" in field:
        return datetime.fromisoformat(field["dateTime"].replace("Z", "+00:00"))
    return at_local(date.fromisoformat(field["date"]))


def event_start(event: dict[str, Any]) -> datetime:
    return _edge(event["start"])


def event_end(event: dict[str, Any]) -> datetime:
    return _edge(event["end"])


# --------------------------------------------------------------------------- #
# Drive corpus
# --------------------------------------------------------------------------- #

OOO_DOC_TEXT = f"""Out of office

I am away from {OOO_FROM.strftime('%A')} {english_date(OOO_FROM)} to
{OOO_TO.strftime('%A')} {english_date(OOO_TO)} inclusive.

During that week please route anything urgent to Sarah Chen. I will not be
taking meetings; anything already in the calendar for those days needs moving.
"""

ACME_PROPOSAL_TEXT = """Acme Corp - Q3 renewal proposal (v4)

Term: 14 months at the 12-month rate.
Seats: 240, up from 180.
Security questionnaire: returned 17 July, awaiting acknowledgement.
Open MSA points: liability cap, data residency.
"""

ACME_MSA_TEXT = """Master Services Agreement - countersigned 30 July 2026.
Liability cap: 12 months of fees. Data residency: US-East.
"""

INVOICE_TEXT = f"""Invoice
Turkish Airlines
Booking reference {PNR}  Ticket {TICKET_NO}
IST -> JFK, {english_date(FLIGHT_DEPARTS_LOCAL)}
Total USD 812.40
"""

SECURITY_Q_TEXT = f"""Security questionnaire response - {english_date(_last_month_start)}.
SOC 2 Type II, penetration test summary, sub-processor list.
"""

FIL_OOO = "1aBcD3fGhIjKlMnOpQrStUvWxYz001"
FIL_ACME_PROPOSAL = "1aBcD3fGhIjKlMnOpQrStUvWxYz002"
FIL_ACME_MSA = "1aBcD3fGhIjKlMnOpQrStUvWxYz003"
FIL_INVOICE_TK = "1aBcD3fGhIjKlMnOpQrStUvWxYz004"
FIL_SECURITY_Q = "1aBcD3fGhIjKlMnOpQrStUvWxYz005"

MIME_DOC = "application/vnd.google-apps.document"
MIME_PDF = "application/pdf"


def drive_file(
    *,
    file_id: str,
    name: str,
    mime_type: str,
    modified_at: datetime,
    owner: str = USER_EMAIL,
    shared: bool = False,
    size: int | None = None,
    parents: tuple[str, ...] = ("root",),
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "drive#file",
        "id": file_id,
        "name": name,
        "mimeType": mime_type,
        "modifiedTime": modified_at.isoformat().replace("+00:00", "Z"),
        "createdTime": (modified_at - timedelta(days=9)).isoformat().replace("+00:00", "Z"),
        "owners": [{"emailAddress": owner, "displayName": owner.split("@")[0]}],
        "shared": shared,
        "trashed": False,
        "parents": list(parents),
        "webViewLink": f"https://docs.google.com/document/d/{file_id}/edit",
    }
    if size is not None:
        body["size"] = str(size)
    return body


# Three PDFs modified last month, two Google Docs modified this one. "Show me
# PDFs in Drive from last month" has to return the first three and none of the
# rest — and `sync_gdrive` has no created date, so "from last month" can only
# ever mean *modified* last month. That limitation is stated, not hidden.
FILES: dict[str, dict[str, Any]] = {
    FIL_OOO: drive_file(
        file_id=FIL_OOO,
        name=f"Out of office - {OOO_FROM.strftime('%b')} {OOO_FROM.day}-{OOO_TO.day}.gdoc",
        mime_type=MIME_DOC,
        modified_at=ago(6, 8, 30),
    ),
    FIL_ACME_PROPOSAL: drive_file(
        file_id=FIL_ACME_PROPOSAL,
        name="Acme - Q3 renewal proposal v4.gdoc",
        mime_type=MIME_DOC,
        modified_at=ago(2, 16, 11),
        owner="sarah@company.com",
        shared=True,
    ),
    FIL_ACME_MSA: drive_file(
        file_id=FIL_ACME_MSA,
        name="Acme_MSA_countersigned.pdf",
        mime_type=MIME_PDF,
        modified_at=last_month(26, 9),
        shared=True,
        size=418_233,
    ),
    FIL_INVOICE_TK: drive_file(
        file_id=FIL_INVOICE_TK,
        name="Invoice_TK_1984.pdf",
        mime_type=MIME_PDF,
        modified_at=last_month(20, 10),
        size=88_213,
    ),
    FIL_SECURITY_Q: drive_file(
        file_id=FIL_SECURITY_Q,
        name="Security_questionnaire_response.pdf",
        mime_type=MIME_PDF,
        modified_at=last_month(16, 16),
        size=221_004,
    ),
}

FILE_TEXT: dict[str, str] = {
    FIL_OOO: OOO_DOC_TEXT,
    FIL_ACME_PROPOSAL: ACME_PROPOSAL_TEXT,
    FIL_ACME_MSA: ACME_MSA_TEXT,
    FIL_INVOICE_TK: INVOICE_TEXT,
    FIL_SECURITY_Q: SECURITY_Q_TEXT,
}

DRIVE_START_PAGE_TOKEN = {"startPageToken": "84021", "kind": "drive#startPageToken"}

DRIVE_FILES_LIST = {
    "kind": "drive#fileList",
    "incompleteSearch": False,
    "files": list(FILES.values()),
}


def drive_files_list(file_ids: list[str] | None = None, page_token: str | None = None) -> dict:
    chosen = file_ids if file_ids is not None else list(FILES)
    body: dict[str, Any] = {
        "kind": "drive#fileList",
        "incompleteSearch": False,
        "files": [FILES[f] for f in chosen],
    }
    if page_token:
        body["nextPageToken"] = page_token
    return body


DRIVE_CHANGES = {
    "kind": "drive#changeList",
    "newStartPageToken": "84029",
    "changes": [
        {
            "kind": "drive#change",
            "changeType": "file",
            "time": datetime(2026, 8, 18, 20, 11, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
            "removed": False,
            "fileId": FIL_ACME_PROPOSAL,
            "file": FILES[FIL_ACME_PROPOSAL],
        }
    ],
}


# --------------------------------------------------------------------------- #
# OAuth
# --------------------------------------------------------------------------- #

TOKEN_REFRESHED = {
    "access_token": "ya29.refreshed-access-token",
    "expires_in": 3599,
    "scope": (
        "https://www.googleapis.com/auth/gmail.modify "
        "https://www.googleapis.com/auth/calendar "
        "https://www.googleapis.com/auth/drive"
    ),
    "token_type": "Bearer",
}

USERINFO = {
    "sub": "104928374650192837465",
    "email": USER_EMAIL,
    "email_verified": True,
    "name": USER_NAME,
    "picture": "https://lh3.googleusercontent.com/a/demo",
}


# --------------------------------------------------------------------------- #
# Mirror rows — the same corpus, shaped for sync_gmail / sync_gcal / sync_gdrive
# --------------------------------------------------------------------------- #


def gmail_body_clean(message: dict[str, Any]) -> str:
    """What the chunker would have stored: subject then body, quoted trail cut.

    Kept deliberately simple. It is the text the embedding is taken over, and the
    text a full-text search matches, so a test that seeds the mirror this way is
    searching the same words a real sync would have indexed.
    """
    return f"{header(message, 'Subject')}\n\n{plain_body(message).strip()}"


def gmail_mirror_rows(embed, message_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """`sync_gmail` rows for the corpus. ``embed`` is called once per row."""
    rows = []
    for mid in message_ids if message_ids is not None else MESSAGE_ORDER:
        message = MESSAGES[mid]
        body = gmail_body_clean(message)
        from_value = header(message, "From")
        from_name, _, addr = from_value.rpartition(" <")
        rows.append(
            {
                "message_id": mid,
                "thread_id": message["threadId"],
                "chunk_index": 0,
                "subject": header(message, "Subject"),
                "from_email": addr.rstrip(">") or from_value,
                "from_name": from_name or None,
                "to_emails": [
                    a.strip() for a in header(message, "To").split(",") if a.strip()
                ],
                "body_clean": body,
                "content_hash": content_hash(body),
                "embedding": embed(body),
                "labels": list(message["labelIds"]),
                "has_attachments": any(
                    p.get("filename") for p in message["payload"].get("parts", [])
                ),
                "received_at": datetime.fromtimestamp(
                    int(message["internalDate"]) / 1000, tz=UTC
                ),
            }
        )
    return rows


def gcal_text(event: dict[str, Any]) -> str:
    return f"{event.get('summary', '')}\n\n{event.get('description', '')}".strip()


def gcal_mirror_rows(embed, event_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """`sync_gcal` rows. ``attendee_emails`` is generated by the database from
    ``attendees`` — never pass it."""
    rows = []
    for eid in event_ids if event_ids is not None else list(EVENTS):
        event = EVENTS[eid]
        text = gcal_text(event)
        rows.append(
            {
                "event_id": eid,
                "calendar_id": "primary",
                "recurring_event_id": event.get("recurringEventId"),
                "title": event.get("summary"),
                "description": event.get("description") or None,
                "location": event.get("location") or None,
                "organizer_email": (event.get("organizer") or {}).get("email"),
                "attendees": [
                    {
                        "email": a["email"],
                        "name": a.get("displayName"),
                        "response_status": a.get("responseStatus"),
                        "optional": a.get("optional", False),
                    }
                    for a in event.get("attendees", [])
                ],
                "content_hash": content_hash(text),
                "embedding": embed(text),
                "starts_at": event_start(event),
                "ends_at": event_end(event),
                "all_day": "date" in event["start"],
                "event_timezone": event["start"].get("timeZone", USER_TZ),
                "status": event.get("status", "confirmed"),
                "etag": event.get("etag"),
            }
        )
    return rows


def gdrive_mirror_rows(embed, file_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """`sync_gdrive` rows. No created date — the schema has none, which is why
    "from last month" can only ever mean *modified* last month."""
    rows = []
    for fid in file_ids if file_ids is not None else list(FILES):
        item = FILES[fid]
        excerpt = FILE_TEXT.get(fid, "")
        text = f"{item['name']}\n\n{excerpt}".strip()
        rows.append(
            {
                "file_id": fid,
                "chunk_index": 0,
                "name": item["name"],
                "mime_type": item["mimeType"],
                "owner_email": item["owners"][0]["emailAddress"],
                "is_shared": item.get("shared", False),
                "web_view_link": item.get("webViewLink"),
                "folder_path": "/",
                "size_bytes": int(item["size"]) if item.get("size") else None,
                "content_excerpt": excerpt,
                "content_hash": content_hash(text),
                "embedding": embed(text),
                "modified_at": datetime.fromisoformat(
                    item["modifiedTime"].replace("Z", "+00:00")
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #


def _q(params: dict[str, str], name: str, default: str | None = None) -> str | None:
    return params.get(name, default)


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve(
    method: str,
    path: str,
    params: dict[str, str],
    body: bytes | None = None,
) -> tuple[int, Any] | None:
    """Map one request onto a recorded payload.

    Returns ``(status, json_body)``, or ``None`` when nothing here answers that
    URL — the caller turns that into a loud failure naming the path, which is
    far more useful than a silent 404.

    Pure: no counters, no failure injection, no state. Everything that varies
    between tests lives in the fixture that wraps this.
    """
    method = method.upper()
    parts = [p for p in path.split("/") if p]
    payload = json.loads(body) if body else {}

    # -- OAuth ------------------------------------------------------------- #
    if path.endswith("/token") and method == "POST":
        return 200, TOKEN_REFRESHED
    if path.endswith("/userinfo") or path.endswith("/oauth2/v3/userinfo"):
        return 200, USERINFO
    if path.endswith("/revoke"):
        return 200, {}

    # -- Gmail ------------------------------------------------------------- #
    if "gmail" in parts and "v1" in parts:
        tail = parts[parts.index("v1") + 1 :]  # users/<id>/...
        rest = tail[2:] if len(tail) >= 2 and tail[0] == "users" else tail

        if rest == ["profile"]:
            return 200, GMAIL_PROFILE

        if rest == ["messages"] and method == "GET":
            wanted = _filter_messages(params)
            limit = _int(_q(params, "maxResults"), 100)
            return 200, gmail_messages_list(wanted[:limit])

        if len(rest) == 2 and rest[0] == "messages" and method == "GET":
            message = MESSAGES.get(rest[1])
            if message is None:
                return 404, google_error(404, "notFound", "Requested entity was not found.")
            if _q(params, "format") == "metadata":
                trimmed = dict(message)
                trimmed["payload"] = {
                    k: v for k, v in message["payload"].items() if k != "parts"
                }
                return 200, trimmed
            return 200, message

        if len(rest) == 3 and rest[0] == "messages" and rest[2] == "modify":
            message = dict(MESSAGES.get(rest[1], {}))
            labels = set(message.get("labelIds", []))
            labels |= set(payload.get("addLabelIds", []))
            labels -= set(payload.get("removeLabelIds", []))
            message["labelIds"] = sorted(labels)
            return 200, message

        if rest == ["messages", "send"] and method == "POST":
            return 200, GMAIL_DRAFT_SENT

        if rest == ["history"] and method == "GET":
            return 200, GMAIL_HISTORY

        if rest == ["drafts"] and method == "POST":
            return 200, GMAIL_DRAFT_CREATED
        if rest == ["drafts"] and method == "GET":
            return 200, {"drafts": [GMAIL_DRAFT_CREATED], "resultSizeEstimate": 1}
        if rest == ["drafts", "send"] and method == "POST":
            return 200, GMAIL_DRAFT_SENT
        if len(rest) == 2 and rest[0] == "drafts":
            if method == "DELETE":
                return 204, None
            if method in {"PUT", "PATCH"}:
                return 200, GMAIL_DRAFT_CREATED
            return 200, GMAIL_DRAFT_CREATED

    # -- Calendar ---------------------------------------------------------- #
    if "calendar" in parts and "v3" in parts:
        tail = parts[parts.index("v3") + 1 :]  # calendars/<id>/events[/<eid>]
        if len(tail) >= 3 and tail[0] == "calendars" and tail[2] == "events":
            event_id = tail[3] if len(tail) > 3 else None

            if event_id is None and method == "GET":
                start = _parse_iso(_q(params, "timeMin"))
                end = _parse_iso(_q(params, "timeMax"))
                query = (_q(params, "q") or "").strip().lower()
                items = (
                    events_in_window(start, end)
                    if start and end
                    else sorted(EVENTS.values(), key=event_start)
                )
                if query:
                    items = [e for e in items if query in gcal_text(e).lower()]
                body_out = dict(CALENDAR_EVENTS_LIST)
                body_out["items"] = items
                return 200, body_out

            if event_id is None and method == "POST":
                return 200, EVENTS[EVT_ACME_REVIEW]

            if event_id is not None:
                event = EVENTS.get(event_id)
                if event is None:
                    return 404, google_error(404, "notFound", "Not Found")
                if method == "DELETE":
                    return 204, None
                if method in {"PATCH", "PUT"}:
                    moved = dict(event)
                    moved.update({k: v for k, v in payload.items() if k in event or k in
                                  {"start", "end", "summary", "description", "location"}})
                    moved["etag"] = '"3401882940110001"'
                    moved["sequence"] = event.get("sequence", 0) + 1
                    return 200, moved
                return 200, event

    # -- Drive ------------------------------------------------------------- #
    if "drive" in parts and "v3" in parts:
        tail = parts[parts.index("v3") + 1 :]

        if tail == ["files"] and method == "GET":
            return 200, drive_files_list(_filter_files(params))
        if tail == ["files"] and method == "POST":
            return 200, FILES[FIL_ACME_PROPOSAL]
        if tail == ["changes", "startPageToken"]:
            return 200, DRIVE_START_PAGE_TOKEN
        if tail == ["changes"]:
            return 200, DRIVE_CHANGES
        if len(tail) == 2 and tail[0] == "files":
            item = FILES.get(tail[1])
            if item is None:
                return 404, google_error(404, "notFound", "File not found.")
            return 200, item
        if len(tail) == 3 and tail[0] == "files" and tail[2] == "export":
            return 200, {"_text": FILE_TEXT.get(tail[1], "")}
        if len(tail) == 3 and tail[0] == "files" and tail[2] == "permissions":
            return 200, {"kind": "drive#permission", "id": "anyoneWithLink", "type": "user"}

    return None


#: ``newer_than:30d`` / ``older_than:2m``. The bounded backfill sends one of
#: these on every full resync, so a fixture that treated it as free text would
#: match nothing and report an empty mailbox.
_AGE_OPERATOR = re.compile(r"\b(newer_than|older_than):(\d+)([dmy])\b")

_AGE_DAYS = {"d": 1, "m": 30, "y": 365}


def _received_at(message: dict) -> datetime:
    return datetime.fromtimestamp(int(message["internalDate"]) / 1000, tz=UTC)


def _filter_messages(params: dict[str, str]) -> list[str]:
    """Honour the bits of Gmail's ``q`` the tests actually use."""
    query = (_q(params, "q") or "").strip().lower()
    if not query:
        return list(MESSAGE_ORDER)

    # Age operators are date arithmetic, not words to look for in a body.
    bounds: list[tuple[str, datetime]] = []
    for operator, count, unit in _AGE_OPERATOR.findall(query):
        edge = NOW - timedelta(days=int(count) * _AGE_DAYS[unit])
        bounds.append((operator, edge))
    query = _AGE_OPERATOR.sub(" ", query)

    out = []
    for mid in MESSAGE_ORDER:
        message = MESSAGES[mid]
        received = _received_at(message)
        if any(
            received < edge if operator == "newer_than" else received > edge
            for operator, edge in bounds
        ):
            continue
        haystack = (
            header(message, "Subject") + " " + header(message, "From") + " " + plain_body(message)
        ).lower()
        terms = [t for t in query.replace(":", " ").split() if t not in {"from", "subject", "in"}]
        if all(term.strip('"') in haystack for term in terms):
            out.append(mid)
    return out


def _filter_files(params: dict[str, str]) -> list[str]:
    query = (_q(params, "q") or "").lower()
    if not query:
        return list(FILES)
    out = []
    for fid, item in FILES.items():
        if "mimetype" in query and item["mimeType"].lower() not in query:
            continue
        name_terms = [
            part.strip("'\" ")
            for part in query.split("name contains")[1:]
        ]
        if name_terms and not any(t.split()[0] in item["name"].lower() for t in name_terms if t):
            continue
        out.append(fid)
    return out


__all__ = [
    "NOW",
    "TODAY",
    "TZ",
    "USER_EMAIL",
    "USER_NAME",
    "USER_TZ",
    "NEXT_MONDAY",
    "NEXT_WEEK_START",
    "NEXT_WEEK_END",
    "TOMORROW",
    "TOMORROW_START",
    "TOMORROW_END",
    "TOMORROW_IS_NEXT_WEEK",
    "LAST_MONTH_START",
    "LAST_MONTH_END",
    "FLIGHT_DEPARTS",
    "FLIGHT_DEPARTS_LOCAL",
    "FLIGHT_ARRIVES",
    "SECOND_FLIGHT_DEPARTS",
    "BOOKED_AT",
    "OOO_FROM",
    "OOO_TO",
    "PNR",
    "TICKET_NO",
    "FLIGHT_NO",
    "SUPPORT_EMAIL",
    "at_local",
    "ago",
    "last_month",
    "english_date",
    "turkish_date",
    "MESSAGES",
    "MESSAGE_ORDER",
    "MSG_TK_BOOKING_EN",
    "MSG_TK_BOOKING_TR",
    "MSG_TK_PROMO",
    "MSG_SARAH_BUDGET",
    "MSG_SARAH_FOLLOWUP",
    "MSG_ACME_PRICING",
    "MSG_ACME_MSA",
    "MSG_JOHN_OKAFOR",
    "MSG_JOHN_REYES",
    "GMAIL_PROFILE",
    "GMAIL_MESSAGES_LIST",
    "GMAIL_HISTORY",
    "GMAIL_DRAFT_ID",
    "GMAIL_DRAFT_CREATED",
    "GMAIL_DRAFT_SENT",
    "EVENTS",
    "EVT_FLIGHT_TK1984",
    "EVT_FLIGHT_TK1979",
    "EVT_ACME_RENEWAL",
    "EVT_ACME_REVIEW",
    "EVT_JOHN_OKAFOR",
    "EVT_JOHN_REYES",
    "EVT_DESIGN_REVIEW",
    "EVT_STANDUP",
    "EVT_OFFSITE",
    "CALENDAR_EVENTS_LIST",
    "CALENDAR_SYNC_TOKEN",
    "FILES",
    "FIL_OOO",
    "FIL_ACME_PROPOSAL",
    "FIL_ACME_MSA",
    "FIL_INVOICE_TK",
    "FIL_SECURITY_Q",
    "MIME_DOC",
    "MIME_PDF",
    "DRIVE_FILES_LIST",
    "DRIVE_CHANGES",
    "DRIVE_START_PAGE_TOKEN",
    "TOKEN_REFRESHED",
    "USERINFO",
    "ERROR_401",
    "ERROR_410",
    "ERROR_412",
    "ERROR_429",
    "ERROR_429_HEADERS",
    "ERROR_503",
    "ERROR_QUOTA",
    "ERROR_INVALID_GRANT",
    "b64url",
    "unb64url",
    "content_hash",
    "google_error",
    "gmail_message",
    "gmail_messages_list",
    "gmail_mirror_rows",
    "gcal_mirror_rows",
    "gdrive_mirror_rows",
    "calendar_events_list",
    "drive_files_list",
    "events_in_window",
    "event_start",
    "event_end",
    "header",
    "plain_body",
    "resolve",
]
