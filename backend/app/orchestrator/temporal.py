"""Dates, resolved in Python. The model never does arithmetic on a calendar.

Two entry points, both pure functions with no I/O:

* :func:`resolve` turns one phrase — ``"next Tuesday"``, ``"last month"``,
  ``"5 September 2026"``, ``"Friday 3pm"`` — into a :class:`Window`, or into
  ``None`` when the phrase has no time in it at all.
* :func:`scan` finds every time phrase in a sentence and resolves each one,
  keyed by a plain identifier so a plan can write ``{{windows.next_week.start}}``.

Three rules the whole file is built around.

**"next <weekday>" is the following ISO week.** Not "the next occurrence".
Standing on Monday, "next Tuesday" is eight days out, not tomorrow — nobody
standing on a Monday says "next Tuesday" and means tomorrow. The two readings
agree everywhere except that case, which is exactly why the rule is written
down rather than left to a library.

**Windows are half-open, [start, end).** An event at exactly midnight belongs to
one day and not to two, and "is the last day included" stops being a question.

**All arithmetic happens on local wall time.** Build the local naive datetime,
attach the zone, then convert to UTC — never the other way round. A week across
a clock change is still seven calendar days, and it is 167 or 169 real hours.
Adding ``timedelta(days=7)`` in UTC gets the length right and the days wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, Final, Iterator
from zoneinfo import ZoneInfo

# 1 = Monday .. 7 = Sunday, matching `users.work_week_start`.
MONDAY: Final[int] = 1
SUNDAY: Final[int] = 7

# The read and write defaults the planner applies when a query names no window.
DEFAULT_READ_DAYS: Final[int] = 180
DEFAULT_WRITE_DAYS: Final[int] = 365

#: Files outlive mail. A deck written eighteen months ago is still *the* deck,
#: and somebody asking for it by name is not asking "if it is recent" — but a
#: 180-day default answers "no such file" about a document sitting in Drive.
#: Mail ages out of relevance far faster, so the two do not share a number.
DEFAULT_FILE_DAYS: Final[int] = 1095

#: A calendar's interesting rows are ahead of you, not behind.
#:
#: Mail and files are a record of what happened, so a default window that looks
#: backwards finds them. An event you want to move, cancel or prepare for has
#: not happened yet — and a backward-only default cannot see it at all, which
#: reads as "no such meeting" about a meeting sitting in next week.
#:
#: Some history still counts ("the meeting we had yesterday"), so it spans.
DEFAULT_CALENDAR_BACK_DAYS: Final[int] = 30
DEFAULT_CALENDAR_AHEAD_DAYS: Final[int] = 180


@dataclass(frozen=True)
class Window:
    """A half-open interval of real time, plus how we read the phrase.

    ``interpretation`` is not decoration. "Next Tuesday read as Aug 25 — the
    Tuesday of next week" in the answer turns a silent wrong week into a
    one-sentence disagreement.
    """

    start: datetime
    end: datetime
    tz: str
    interpretation: str

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("a Window is always tz-aware")
        # Stored in UTC, always. Two aware datetimes that share a `tzinfo`
        # object subtract as *naive* wall times — Python ignores the common
        # zone — so a week across a clock change would measure 168 hours when
        # it really lasted 167. Keeping the instants in UTC makes elapsed time
        # elapsed time; `tz` and `local_start()` carry the wall-clock reading.
        object.__setattr__(self, "start", self.start.astimezone(UTC))
        object.__setattr__(self, "end", self.end.astimezone(UTC))

    # -- membership ---------------------------------------------------------

    def contains(self, moment: datetime) -> bool:
        """Half-open membership. This is the only test callers should use."""
        return self.start <= moment < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    # -- rendering ----------------------------------------------------------

    def local_start(self) -> datetime:
        return self.start.astimezone(ZoneInfo(self.tz))

    def local_end(self) -> datetime:
        return self.end.astimezone(ZoneInfo(self.tz))

    def to_dict(self) -> dict[str, Any]:
        """The JSON shape carried on ``runs.intent.resolved_window``."""
        return {
            "start": self.start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": self.end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "tz": self.tz,
            "interpretation": self.interpretation,
        }


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or "UTC")
    except Exception:  # noqa: BLE001 - an unknown zone must not take a run down
        return ZoneInfo("UTC")


def _localize(naive: datetime, zone: ZoneInfo) -> datetime:
    """Attach ``zone`` to a wall time, stepping over a spring-forward gap.

    A wall time inside the gap does not exist. Attaching a zone to it silently
    produces an instant whose local reading is a different wall time, so we
    detect that by round-tripping and move forward past the gap instead.
    """
    aware = naive.replace(tzinfo=zone)
    back = aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
    if back != naive:
        aware = (naive + timedelta(hours=1)).replace(tzinfo=zone)
    return aware


def _day_start(day: date, zone: ZoneInfo) -> datetime:
    return _localize(datetime(day.year, day.month, day.day), zone)


def _at(day: date, hour: int, minute: int, zone: ZoneInfo) -> datetime:
    return _localize(datetime(day.year, day.month, day.day, hour, minute), zone)


def _week_start(day: date, week_start: int) -> date:
    """First day of the week containing ``day`` for a user's week start."""
    start_index = (int(week_start) - 1) % 7  # 0 = Monday
    return day - timedelta(days=(day.weekday() - start_index) % 7)


def _iso_monday(day: date) -> date:
    """Monday of ``day``'s ISO week. Independent of ``work_week_start``."""
    return day - timedelta(days=day.weekday())


def _month_start(day: date, months: int = 0) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    return date(year, month % 12 + 1, 1)


def _quarter_start(day: date, quarters: int = 0) -> date:
    index = (day.month - 1) // 3 + quarters
    year = day.year + index // 4
    return date(year, (index % 4) * 3 + 1, 1)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

_WEEKDAY_FULL: Final[dict[str, int]] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WEEKDAY_ABBR: Final[dict[str, int]] = {
    "mon": 0,
    "tue": 1,
    "tues": 1,
    "wed": 2,
    "weds": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

_WEEKDAY_ANY: Final[dict[str, int]] = {**_WEEKDAY_FULL, **_WEEKDAY_ABBR}

_MONTHS: Final[dict[str, int]] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_WORD_NUMBERS: Final[dict[str, int]] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fourteen": 14,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "sixty": 60,
    "ninety": 90,
}

_ORDINAL_SUFFIX = r"(?:st|nd|rd|th)?"
_WD_FULL = "|".join(sorted(_WEEKDAY_FULL, key=len, reverse=True))
_WD_ANY = "|".join(sorted(_WEEKDAY_ANY, key=len, reverse=True))
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NUM = r"\d{1,3}|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True))
_TIME = (
    r"(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|"
    r"(?:[01]?\d|2[0-3]):[0-5]\d|"
    r"noon|midday|midnight)"
)


def _number(text: str) -> int:
    text = (text or "").strip().lower()
    if text.isdigit():
        return int(text)
    return _WORD_NUMBERS.get(text, 1)


def _time_of_day(text: str | None) -> tuple[int, int] | None:
    """``"3pm"`` and ``"15:00"`` and ``"noon"`` all become (hour, minute)."""
    if not text:
        return None
    raw = text.strip().lower().replace(".", "")
    if raw in ("noon", "midday"):
        return (12, 0)
    if raw == "midnight":
        return (0, 0)
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", raw)
    if match:
        hour = int(match.group(1)) % 12
        if match.group(3) == "pm":
            hour += 12
        return (hour, int(match.group(2) or 0))
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return (hour, minute)
    return None


# ---------------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------------
#
# One ordered list of rules, used by both `resolve` and `scan`. Order is
# significance, not preference: "day after tomorrow" has to be tried before
# "tomorrow" or it resolves to the wrong day and still looks right.


@dataclass(frozen=True)
class _Ctx:
    """Everything a rule needs besides its own match."""

    now: datetime
    zone: ZoneInfo
    tz: str
    week_start: int
    anchor: date | None  # an earlier resolved day in the same sentence

    @property
    def today(self) -> date:
        return self.now.astimezone(self.zone).date()


_Handler = Callable[[re.Match[str], _Ctx], "tuple[datetime, datetime, str] | None"]
_RULES: list[tuple[str, re.Pattern[str], _Handler]] = []


def _rule(name: str, pattern: str) -> Callable[[_Handler], _Handler]:
    compiled = re.compile(pattern, re.IGNORECASE)

    def register(fn: _Handler) -> _Handler:
        _RULES.append((name, compiled, fn))
        return fn

    return register


def _whole_day(day: date, ctx: _Ctx, said: str) -> tuple[datetime, datetime, str]:
    return _day_start(day, ctx.zone), _day_start(day + timedelta(days=1), ctx.zone), said


def _hour_at(day: date, clock: tuple[int, int], ctx: _Ctx, said: str):
    start = _at(day, clock[0], clock[1], ctx.zone)
    return start, start + timedelta(hours=1), said


def _day_or_hour(day: date, clock: tuple[int, int] | None, ctx: _Ctx, said: str):
    if clock is None:
        return _whole_day(day, ctx, said)
    return _hour_at(day, clock, ctx, f"{said}, {clock[0]:02d}:{clock[1]:02d} local")


# -- explicit dates ---------------------------------------------------------


@_rule("iso_date", rf"\b(?P<y>\d{{4}})-(?P<m>\d{{2}})-(?P<d>\d{{2}})\b(?:\s+(?:at\s+)?(?P<time>{_TIME}))?")
def _iso_date(match: re.Match[str], ctx: _Ctx):
    try:
        day = date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
    except ValueError:
        return None
    return _day_or_hour(day, _time_of_day(match.group("time")), ctx, f"{day.isoformat()}, local day")


@_rule(
    "month_day",
    rf"\b(?P<mon>{_MON})\.?\s+(?P<d>\d{{1,2}}){_ORDINAL_SUFFIX}"
    rf"(?:,?\s+(?P<y>\d{{4}}))?(?:\s+(?:at\s+)?(?P<time>{_TIME}))?\b",
)
def _month_day(match: re.Match[str], ctx: _Ctx):
    return _named_date(match, ctx)


@_rule(
    "day_month",
    rf"\b(?P<d>\d{{1,2}}){_ORDINAL_SUFFIX}\s+(?:of\s+)?(?P<mon>{_MON})\.?"
    rf"(?:,?\s+(?P<y>\d{{4}}))?(?:\s+(?:at\s+)?(?P<time>{_TIME}))?\b",
)
def _day_month(match: re.Match[str], ctx: _Ctx):
    return _named_date(match, ctx)


def _named_date(match: re.Match[str], ctx: _Ctx):
    month = _MONTHS[match.group("mon").lower().rstrip(".")]
    day_number = int(match.group("d"))
    year = int(match.group("y")) if match.group("y") else ctx.today.year
    try:
        day = date(year, month, day_number)
    except ValueError:
        return None
    if not match.group("y") and day < ctx.today - timedelta(days=180):
        # No year given and it is well behind us: they meant the coming one.
        day = date(year + 1, month, day_number)
    return _day_or_hour(day, _time_of_day(match.group("time")), ctx, f"{day.isoformat()}, local day")


@_rule("month_year", rf"\b(?P<mon>{_MON})\s+(?P<y>\d{{4}})\b")
def _month_year(match: re.Match[str], ctx: _Ctx):
    first = date(int(match.group("y")), _MONTHS[match.group("mon").lower()], 1)
    return (
        _day_start(first, ctx.zone),
        _day_start(_month_start(first, 1), ctx.zone),
        f"the whole of {first.strftime('%B %Y')}; half-open",
    )


# -- named days -------------------------------------------------------------


@_rule("day_after_tomorrow", r"\bday\s+after\s+tomorrow\b")
def _day_after_tomorrow(match: re.Match[str], ctx: _Ctx):
    return _whole_day(ctx.today + timedelta(days=2), ctx, "the day after tomorrow, local day")


@_rule("day_before_yesterday", r"\bday\s+before\s+yesterday\b")
def _day_before_yesterday(match: re.Match[str], ctx: _Ctx):
    return _whole_day(ctx.today - timedelta(days=2), ctx, "the day before yesterday, local day")


@_rule("tonight", r"\btonight\b")
def _tonight(match: re.Match[str], ctx: _Ctx):
    start = _at(ctx.today, 18, 0, ctx.zone)
    return start, _day_start(ctx.today + timedelta(days=1), ctx.zone), "tonight, 18:00 to local midnight"


@_rule("this_part_of_day", r"\bthis\s+(?P<part>morning|afternoon|evening)\b")
def _this_part(match: re.Match[str], ctx: _Ctx):
    part = match.group("part").lower()
    span = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 23)}[part]
    return (
        _at(ctx.today, span[0], 0, ctx.zone),
        _at(ctx.today, span[1], 0, ctx.zone),
        f"this {part}, local",
    )


@_rule("today", rf"\btoday\b(?:\s+(?:at\s+)?(?P<time>{_TIME}))?")
def _today(match: re.Match[str], ctx: _Ctx):
    return _day_or_hour(ctx.today, _time_of_day(match.group("time")), ctx, "today, local day")


@_rule("tomorrow", rf"\b(?:tomorrow|tomorrows|tmrw)\b(?:\s+(?:at\s+)?(?P<time>{_TIME}))?")
def _tomorrow(match: re.Match[str], ctx: _Ctx):
    return _day_or_hour(
        ctx.today + timedelta(days=1), _time_of_day(match.group("time")), ctx, "tomorrow, local day"
    )


@_rule("yesterday", r"\byesterday\b")
def _yesterday(match: re.Match[str], ctx: _Ctx):
    return _whole_day(ctx.today - timedelta(days=1), ctx, "yesterday, local day")


# -- weekdays ---------------------------------------------------------------


@_rule(
    "rel_weekday",
    rf"\b(?P<rel>next|this\s+coming|this|last|past|coming|upcoming)\s+(?P<wd>{_WD_ANY})\b"
    rf"(?:\s+(?:at\s+)?(?P<time>{_TIME}))?",
)
def _rel_weekday(match: re.Match[str], ctx: _Ctx):
    index = _WEEKDAY_ANY[match.group("wd").lower()]
    rel = re.sub(r"\s+", " ", match.group("rel").lower())
    monday = _iso_monday(ctx.today)

    if rel in ("next", "coming", "upcoming", "this coming"):
        # THE RULE: the following ISO week, never "the next occurrence".
        day = monday + timedelta(days=7 + index)
        said = (
            f"{day.strftime('%A')} of ISO week {day.isocalendar().week} "
            f"({day.isoformat()}); the weekday of next week, not the next occurrence; "
            "local day boundaries; half-open"
        )
    elif rel in ("last", "past"):
        day = monday - timedelta(days=7 - index)
        said = f"{day.strftime('%A')} of last ISO week ({day.isoformat()}); local day; half-open"
    else:  # "this <weekday>" — inside the current ISO week
        day = monday + timedelta(days=index)
        said = f"{day.strftime('%A')} of this ISO week ({day.isoformat()}); local day; half-open"

    return _day_or_hour(day, _time_of_day(match.group("time")), ctx, said)


@_rule(
    "bare_weekday",
    rf"\b(?:on\s+)?(?P<wd>{_WD_FULL})\b(?:\s+(?:at\s+)?(?P<time>{_TIME}))?",
)
def _bare_weekday(match: re.Match[str], ctx: _Ctx):
    """A weekday on its own: the coming one, or the one in the anchor's week.

    The anchored reading is what stops "push my review next Thursday to Friday
    3pm" from moving the meeting six days backwards. When an earlier phrase in
    the same sentence already fixed a week, a bare weekday belongs to that week.
    """
    index = _WEEKDAY_FULL[match.group("wd").lower()]
    if ctx.anchor is not None:
        day = _iso_monday(ctx.anchor) + timedelta(days=index)
        said = (
            f"{day.strftime('%A')} of the week already named in this sentence "
            f"({day.isoformat()}); local day"
        )
    else:
        ahead = (index - ctx.today.weekday()) % 7
        day = ctx.today + timedelta(days=ahead or 7)
        said = f"the coming {day.strftime('%A')} ({day.isoformat()}); local day"
    return _day_or_hour(day, _time_of_day(match.group("time")), ctx, said)


# -- weeks, months, quarters, years -----------------------------------------


@_rule("rel_weekend", r"\b(?P<rel>next|this|last|coming)\s+weekend\b")
def _rel_weekend(match: re.Match[str], ctx: _Ctx):
    rel = match.group("rel").lower()
    shift = {"next": 7, "coming": 7, "this": 0, "last": -7}[rel]
    saturday = _iso_monday(ctx.today) + timedelta(days=5 + shift)
    return (
        _day_start(saturday, ctx.zone),
        _day_start(saturday + timedelta(days=2), ctx.zone),
        f"{rel} weekend: Sat {saturday.isoformat()} to Mon; half-open",
    )


@_rule("rel_week", r"\b(?P<rel>next|this|last|current|past|coming|upcoming)\s+week\b")
def _rel_week(match: re.Match[str], ctx: _Ctx):
    rel = match.group("rel").lower()
    shift = {"next": 7, "coming": 7, "upcoming": 7, "this": 0, "current": 0, "last": -7, "past": -7}[
        rel
    ]
    start = _week_start(ctx.today, ctx.week_start) + timedelta(days=shift)
    end = start + timedelta(days=7)
    return (
        _day_start(start, ctx.zone),
        _day_start(end, ctx.zone),
        f"{rel} week: {start.isoformat()} to {end.isoformat()}, "
        f"week_start={ctx.week_start}; half-open",
    )


@_rule("rel_month", r"\b(?P<rel>next|this|last|current|past)\s+month\b")
def _rel_month(match: re.Match[str], ctx: _Ctx):
    rel = match.group("rel").lower()
    shift = {"next": 1, "this": 0, "current": 0, "last": -1, "past": -1}[rel]
    start = _month_start(ctx.today, shift)
    end = _month_start(start, 1)
    return (
        _day_start(start, ctx.zone),
        _day_start(end, ctx.zone),
        f"the whole of {start.strftime('%B %Y')}; local days; half-open",
    )


@_rule("rel_quarter", r"\b(?P<rel>next|this|last|current)\s+quarter\b")
def _rel_quarter(match: re.Match[str], ctx: _Ctx):
    rel = match.group("rel").lower()
    shift = {"next": 1, "this": 0, "current": 0, "last": -1}[rel]
    start = _quarter_start(ctx.today, shift)
    end = _quarter_start(start, 1)
    return (
        _day_start(start, ctx.zone),
        _day_start(end, ctx.zone),
        f"the quarter beginning {start.isoformat()}; half-open",
    )


@_rule("rel_year", r"\b(?P<rel>next|this|last|current)\s+year\b")
def _rel_year(match: re.Match[str], ctx: _Ctx):
    rel = match.group("rel").lower()
    shift = {"next": 1, "this": 0, "current": 0, "last": -1}[rel]
    start = date(ctx.today.year + shift, 1, 1)
    return (
        _day_start(start, ctx.zone),
        _day_start(date(start.year + 1, 1, 1), ctx.zone),
        f"calendar year {start.year}; half-open",
    )


# -- counted spans ----------------------------------------------------------

_UNITS = r"(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)"


def _delta(count: int, unit: str) -> timedelta | None:
    unit = unit.rstrip(".").lower()
    if unit.startswith(("minute", "min")):
        return timedelta(minutes=count)
    if unit.startswith(("hour", "hr")):
        return timedelta(hours=count)
    if unit.startswith("day"):
        return timedelta(days=count)
    if unit.startswith("week"):
        return timedelta(weeks=count)
    return None  # months and years are calendar maths, handled by the caller


def _shift_months(day: date, months: int) -> date:
    total = day.year * 12 + (day.month - 1) + months
    year, month = divmod(total, 12)
    last = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]
    return date(year, month + 1, min(day.day, last))


@_rule("last_n", rf"\b(?:last|past|previous|prior)\s+(?P<n>{_NUM})\s+{_UNITS}\b")
def _last_n(match: re.Match[str], ctx: _Ctx):
    count = _number(match.group("n"))
    unit = match.group("unit").lower()
    span = _delta(count, unit)
    if span is not None:
        start = ctx.now - span
    elif unit.startswith("month"):
        start = _localize(
            datetime.combine(_shift_months(ctx.today, -count), ctx.now.astimezone(ctx.zone).time()),
            ctx.zone,
        )
    else:
        start = _localize(
            datetime.combine(
                date(ctx.today.year - count, ctx.today.month, ctx.today.day),
                ctx.now.astimezone(ctx.zone).time(),
            ),
            ctx.zone,
        )
    return start, ctx.now, f"rolling {count} {unit} back from now; half-open"


@_rule("n_ago", rf"\b(?P<n>{_NUM})\s+{_UNITS}\s+ago\b")
def _n_ago(match: re.Match[str], ctx: _Ctx):
    count = _number(match.group("n"))
    unit = match.group("unit").lower()
    if unit.startswith("month"):
        day = _shift_months(ctx.today, -count)
    elif unit.startswith("year"):
        day = date(ctx.today.year - count, ctx.today.month, ctx.today.day)
    else:
        span = _delta(count, unit) or timedelta(days=count)
        day = (ctx.now - span).astimezone(ctx.zone).date()
    return _whole_day(day, ctx, f"{count} {unit} ago ({day.isoformat()}); local day")


@_rule("next_n", rf"\b(?:next|coming|following|upcoming)\s+(?P<n>{_NUM})\s+{_UNITS}\b")
def _next_n(match: re.Match[str], ctx: _Ctx):
    count = _number(match.group("n"))
    unit = match.group("unit").lower()
    span = _delta(count, unit)
    if span is not None:
        end = ctx.now + span
    elif unit.startswith("month"):
        end = _localize(
            datetime.combine(_shift_months(ctx.today, count), ctx.now.astimezone(ctx.zone).time()),
            ctx.zone,
        )
    else:
        end = _localize(
            datetime.combine(
                date(ctx.today.year + count, ctx.today.month, ctx.today.day),
                ctx.now.astimezone(ctx.zone).time(),
            ),
            ctx.zone,
        )
    return ctx.now, end, f"the next {count} {unit} from now; half-open"


@_rule("in_n", rf"\bin\s+(?P<n>{_NUM})\s+{_UNITS}\b")
def _in_n(match: re.Match[str], ctx: _Ctx):
    count = _number(match.group("n"))
    unit = match.group("unit").lower()
    if unit.startswith(("minute", "min", "hour", "hr")):
        start = ctx.now + (_delta(count, unit) or timedelta())
        return start, start + timedelta(hours=1), f"in {count} {unit}"
    if unit.startswith("month"):
        day = _shift_months(ctx.today, count)
    elif unit.startswith("year"):
        day = date(ctx.today.year + count, ctx.today.month, ctx.today.day)
    else:
        day = (ctx.now + (_delta(count, unit) or timedelta(days=count))).astimezone(ctx.zone).date()
    return _whole_day(day, ctx, f"in {count} {unit} ({day.isoformat()}); local day")


@_rule("bare_time", rf"\bat\s+(?P<time>{_TIME})\b")
def _bare_time(match: re.Match[str], ctx: _Ctx):
    clock = _time_of_day(match.group("time"))
    if clock is None:
        return None
    day = ctx.anchor or ctx.today
    return _hour_at(day, clock, ctx, f"{day.isoformat()} at {clock[0]:02d}:{clock[1]:02d} local")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phrase:
    """One time phrase, where it sat in the text, and what it resolved to.

    The span is what lets a caller check that a phrase was consumed *whole* —
    the rule router will not route "next week where john@company.com is
    invited" as a bare window, and it needs the offsets to know that.
    """

    name: str
    text: str
    start: int
    end: int
    window: Window


_Hit = Phrase


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in taken)


def _find(text: str, ctx: _Ctx) -> Iterator[Phrase]:
    """Every time phrase in ``text``, most specific rule first, no overlaps.

    Rules are tried in declaration order, which is significance order, and a
    match that overlaps something already claimed is dropped. That is what
    stops "next Tuesday" from also being read as a bare "Tuesday".
    """
    if not text or not text.strip():
        return

    taken: list[tuple[int, int]] = []
    found: list[Phrase] = []
    anchor = ctx.anchor

    for name, pattern, handler in _RULES:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, taken):
                continue
            resolved = handler(match, _Ctx(ctx.now, ctx.zone, ctx.tz, ctx.week_start, anchor))
            if resolved is None:
                continue
            start, end, said = resolved
            if end <= start:
                continue
            taken.append(span)
            phrase = match.group(0)
            lead = len(phrase) - len(phrase.lstrip())
            trail = len(phrase) - len(phrase.rstrip())
            found.append(
                Phrase(
                    name,
                    phrase.strip(),
                    span[0] + lead,
                    span[1] - trail,
                    Window(start, end, ctx.tz, said),
                )
            )

    for hit in sorted(found, key=lambda h: h.start):
        yield hit


def find_phrases(
    text: str,
    tz: str = "UTC",
    week_start: int = MONDAY,
    now: datetime | None = None,
    *,
    anchor: datetime | date | None = None,
) -> list[Phrase]:
    """Every time phrase with its offsets, left to right, no overlaps."""
    if not text or not text.strip():
        return []
    zone = _zone(tz)
    moment = now or datetime.now(UTC)
    anchor_day = anchor.astimezone(zone).date() if isinstance(anchor, datetime) else anchor
    ctx = _Ctx(moment, zone, tz, int(week_start or MONDAY), anchor_day)
    return list(_find(text, ctx))


def _slug(text: str, rule: str) -> str:
    """A reference name: ``"next Tuesday"`` becomes ``next_tuesday``.

    Plans write ``{{windows.<name>.start}}``, so the key has to be a plain
    identifier rather than the raw phrase.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = rule
    if slug[0].isdigit():
        slug = f"on_{slug}"
    return slug


# ---------------------------------------------------------------------------
# The public two
# ---------------------------------------------------------------------------


def resolve(
    phrase: str,
    tz: str = "UTC",
    week_start: int = MONDAY,
    now: datetime | None = None,
    *,
    anchor: datetime | date | None = None,
) -> Window | None:
    """One phrase to one window, or ``None`` when there is no time in it.

    ``anchor`` is an earlier resolved day. A bare weekday resolves inside the
    anchor's ISO week, which is what "…next Thursday to Friday 3pm" needs.
    """
    if not phrase or not phrase.strip():
        return None

    zone = _zone(tz)
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must be tz-aware")

    anchor_day: date | None
    if isinstance(anchor, datetime):
        anchor_day = anchor.astimezone(zone).date()
    else:
        anchor_day = anchor

    ctx = _Ctx(moment, zone, tz, int(week_start or MONDAY), anchor_day)
    for hit in _find(phrase, ctx):
        return hit.window
    return None


def scan(
    text: str,
    tz: str = "UTC",
    week_start: int = MONDAY,
    now: datetime | None = None,
) -> dict[str, Window]:
    """Every time phrase in a sentence, keyed by a usable reference name.

    Phrases are resolved left to right, and each resolved day becomes the
    anchor for the ones after it, so the second phrase in "move it from
    Thursday to Friday" lands in the right week.
    """
    if not text or not text.strip():
        return {}

    zone = _zone(tz)
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must be tz-aware")

    week = int(week_start or MONDAY)
    windows: dict[str, Window] = {}
    anchor: date | None = None

    # Two passes: find the spans with no anchor, then re-resolve each phrase in
    # order carrying the anchor forward. A single pass would anchor a phrase on
    # something that appears after it.
    for hit in _find(text, _Ctx(moment, zone, tz, week, None)):
        window = (
            resolve(hit.text, tz, week, moment, anchor=anchor) if anchor is not None else hit.window
        ) or hit.window
        name = _slug(hit.text, hit.name)
        if name in windows:
            suffix = 2
            while f"{name}_{suffix}" in windows:
                suffix += 1
            name = f"{name}_{suffix}"
        windows[name] = window
        anchor = window.local_start().date()

    _name_the_destination(text, windows)
    return windows


#: "move X **to** Y", "push it **to** Y", "reschedule **for** Y". The word that
#: separates where something is now from where it is going.
_MOVE_VERB = re.compile(r"\b(move|push|shift|reschedul\w*|bump|change|postpone|bring)\b", re.I)
_MOVE_PREPOSITION = re.compile(r"\b(to|until|till|for|onto)\b", re.I)


def _name_the_destination(text: str, windows: dict[str, Window]) -> None:
    """Alias the destination window of a move as ``target_slot``.

    "Push my Acme review next Thursday to Friday 3pm" has two windows: where the
    meeting is, and where it is going. Both keep their descriptive names —
    ``next_thursday`` and ``friday_3pm`` — because those are what a reader
    recognises. But a plan that moves an event needs to say *which one is the
    destination*, and the phrase it was written from is not stable enough to
    reference: the same intent arrives as "to Friday 3pm", "to next Tuesday" or
    "to the 4th". ``target_slot`` is the name the design gives that slot (see
    `docs/SAMPLE_QUERIES.md`), so it is added as a second key onto the same
    window rather than replacing anything.

    Only for a sentence that actually moves something, and only when there are
    two windows to tell apart — one window is the subject, not a destination.
    """
    if len(windows) < 2 or "target_slot" in windows:
        return
    if not (_MOVE_VERB.search(text) and _MOVE_PREPOSITION.search(text)):
        return
    windows["target_slot"] = windows[list(windows)[-1]]


def default_window(
    kind: str = "read",
    tz: str = "UTC",
    now: datetime | None = None,
    *,
    days: int | None = None,
    corpus: str | None = None,
) -> Window:
    """The window a query gets when it names none.

    Mail and files look backwards — they are a record, and a booking for a
    September flight can have been made last autumn. The calendar looks mostly
    forwards, because the event somebody wants to move is ahead of them; a
    backward-only default simply cannot see it, and then reports that it does
    not exist.
    """
    moment = now or datetime.now(UTC)

    if corpus in {"gdrive", "drive", "file"} and days is None:
        return Window(
            moment - timedelta(days=DEFAULT_FILE_DAYS),
            moment,
            tz,
            f"{DEFAULT_FILE_DAYS} days back from now; file default; half-open",
        )

    if corpus in {"gcal", "calendar", "event"} and days is None:
        return Window(
            moment - timedelta(days=DEFAULT_CALENDAR_BACK_DAYS),
            moment + timedelta(days=DEFAULT_CALENDAR_AHEAD_DAYS),
            tz,
            f"{DEFAULT_CALENDAR_BACK_DAYS} days back to "
            f"{DEFAULT_CALENDAR_AHEAD_DAYS} days ahead; calendar default; half-open",
        )

    back = days or (DEFAULT_WRITE_DAYS if kind == "write" else DEFAULT_READ_DAYS)
    return Window(
        moment - timedelta(days=back),
        moment,
        tz,
        f"{back} days back from now; {kind}-class default; half-open",
    )


def window_from(value: Any, tz: str = "UTC") -> Window | None:
    """Rebuild a Window from the dict shape stored on ``runs.intent``."""
    if isinstance(value, Window):
        return value
    if not isinstance(value, dict):
        return None
    start, end = value.get("start"), value.get("end")
    if not start or not end:
        return None
    try:
        start_dt = start if isinstance(start, datetime) else datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end_dt = end if isinstance(end, datetime) else datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    if start_dt.tzinfo is None or end_dt.tzinfo is None:
        return None
    return Window(start_dt, end_dt, value.get("tz") or tz, value.get("interpretation") or "")


__all__ = [
    "DEFAULT_CALENDAR_AHEAD_DAYS",
    "DEFAULT_FILE_DAYS",
    "DEFAULT_CALENDAR_BACK_DAYS",
    "DEFAULT_READ_DAYS",
    "DEFAULT_WRITE_DAYS",
    "MONDAY",
    "SUNDAY",
    "Phrase",
    "Window",
    "default_window",
    "find_phrases",
    "resolve",
    "scan",
    "window_from",
]
