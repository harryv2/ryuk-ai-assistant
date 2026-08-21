"""What each Google call costs, and the gate that charges it.

Google does not rate its APIs in calls per second. It rates them in *units*:
against Gmail a ``messages.list`` costs 5 and a ``messages.send`` costs 100, and
the ceiling is 250 units per second per user. Counting calls would let a single
send-heavy run look twenty times cheaper than it is.

So every method this codebase calls has a price here, and
:func:`acquire` spends it against the user's token bucket in
``app.core.ratelimit`` before the request goes out. The bucket is split: the
background sync gets 70 percent of the budget, interactive work 30, in separate
buckets, so a backfill can never starve the person typing.

Calendar and Drive publish their limits in requests rather than units, so their
prices here are modelled — relative weights that keep the one bucket honest
about how expensive a write is next to a read.

This table is the authority for the methods we call. ``acquire`` always passes
the cost explicitly, so nothing depends on the fallback table inside
``ratelimit``.
"""

from __future__ import annotations

from typing import Final

from app.core import ratelimit
from app.core.ratelimit import Share

#: Anything not listed. Deliberately the price of a cheap read, so a new method
#: that slips through is charged something rather than nothing.
DEFAULT_UNITS: Final[int] = 5

#: The three services, as they appear in an op name (``gmail.search_emails``)
#: and in ``sync_service``.
SERVICES: Final[tuple[str, ...]] = ("gmail", "gcal", "gdrive")

UNITS: Final[dict[str, int]] = {
    # ---------------------------------------------------------------- gmail
    # Published quota units, per Google's Gmail usage limits page.
    "gmail.messages.list": 5,
    "gmail.messages.get": 5,
    "gmail.messages.send": 100,
    "gmail.messages.modify": 5,
    "gmail.messages.trash": 5,
    "gmail.messages.untrash": 5,
    "gmail.messages.attachments.get": 5,
    "gmail.drafts.create": 10,
    "gmail.drafts.update": 15,
    "gmail.drafts.send": 100,
    "gmail.drafts.get": 5,
    "gmail.drafts.list": 5,
    "gmail.drafts.delete": 10,
    "gmail.threads.get": 10,
    "gmail.threads.list": 10,
    "gmail.threads.modify": 10,
    "gmail.history.list": 2,
    "gmail.labels.list": 1,
    "gmail.labels.get": 1,
    "gmail.labels.create": 5,
    "gmail.users.getProfile": 1,
    # ---------------------------------------------------------------- calendar
    "gcal.events.list": 5,
    "gcal.events.get": 3,
    "gcal.events.insert": 20,
    "gcal.events.patch": 20,
    "gcal.events.update": 20,
    "gcal.events.delete": 20,
    "gcal.events.move": 20,
    "gcal.events.instances": 5,
    "gcal.freebusy.query": 10,
    "gcal.calendarList.list": 3,
    "gcal.calendars.get": 3,
    "gcal.settings.get": 1,
    "gcal.settings.list": 1,
    # ---------------------------------------------------------------- drive
    "gdrive.files.list": 5,
    "gdrive.files.get": 3,
    # An export renders the whole document server-side. It is not a cheap read.
    "gdrive.files.export": 20,
    "gdrive.files.download": 10,
    "gdrive.files.create": 20,
    "gdrive.files.update": 20,
    "gdrive.files.copy": 20,
    "gdrive.files.delete": 20,
    "gdrive.permissions.create": 25,
    "gdrive.permissions.list": 5,
    "gdrive.permissions.delete": 20,
    "gdrive.changes.list": 5,
    "gdrive.changes.getStartPageToken": 1,
    "gdrive.about.get": 1,
}


def units_for(method: str) -> int:
    """Price of a method, e.g. ``gmail.messages.send`` -> 100.

    Tolerates the fully qualified REST name (``gmail.users.messages.send``) by
    walking in from the left until something matches, so a caller that copies a
    name out of Google's docs is still charged correctly.
    """
    if method in UNITS:
        return UNITS[method]
    parts = method.split(".")
    for start in range(1, len(parts)):
        tail = ".".join(parts[start:])
        if tail in UNITS:
            return UNITS[tail]
        prefixed = f"{parts[0]}.{tail}"
        if prefixed in UNITS:
            return UNITS[prefixed]
    return DEFAULT_UNITS


def service_of(method: str) -> str:
    """The service a method belongs to: ``gmail`` | ``gcal`` | ``gdrive``."""
    head = method.split(".", 1)[0]
    if head in SERVICES:
        return head
    if head == "calendar":
        return "gcal"
    if head == "drive":
        return "gdrive"
    return head


def is_write(method: str) -> bool:
    """True for the methods that change something on Google's side."""
    verb = method.rsplit(".", 1)[-1]
    return verb in {
        "send",
        "insert",
        "create",
        "update",
        "patch",
        "delete",
        "modify",
        "trash",
        "untrash",
        "move",
        "copy",
        "import",
    }


async def acquire(
    user_id: str,
    method: str,
    *,
    units: int | None = None,
    share: Share = "interactive",
) -> int:
    """Wait for room in the user's Google budget, then charge this call.

    Returns the units charged. Raises ``AppError('RATE_LIMITED')`` if the wait
    would run past the governor's ceiling — a call that has queued that long
    should fail loudly rather than hold a request open.
    """
    cost = int(units) if units is not None else units_for(method)
    return await ratelimit.acquire_google(user_id, method, units=cost, share=share)


async def try_acquire(
    user_id: str,
    method: str,
    *,
    units: int | None = None,
    share: Share = "interactive",
) -> tuple[bool, float, int]:
    """One non-blocking attempt: ``(granted, wait_s, units_left)``."""
    cost = int(units) if units is not None else units_for(method)
    return await ratelimit.try_acquire_google(user_id, method, units=cost, share=share)


async def state(user_id: str) -> dict[str, dict[str, float]]:
    """Both buckets, as they stand. For /metrics."""
    return await ratelimit.google_quota_state(user_id)


def estimate(methods: dict[str, int]) -> int:
    """Total units for a planned batch: ``{"gmail.messages.get": 40}`` -> 200.

    The sync tasks use this to size a page against what is left in the bucket.
    """
    return sum(units_for(name) * max(0, count) for name, count in methods.items())


__all__ = [
    "DEFAULT_UNITS",
    "SERVICES",
    "UNITS",
    "Share",
    "units_for",
    "service_of",
    "is_write",
    "acquire",
    "try_acquire",
    "state",
    "estimate",
]
