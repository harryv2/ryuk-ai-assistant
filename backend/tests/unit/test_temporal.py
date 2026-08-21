"""Temporal resolution.

The most important file in the unit suite. "Next Tuesday" is one of the brief's
three hard cases, and every one of the ways it goes wrong is silent: the answer
still renders, it is just about the wrong week. So this file pins the behaviour
down instead of trusting it.

What is checked:

* "next <weekday>" means that weekday in the FOLLOWING ISO week — so asked on a
  Tuesday it is seven days out, not today, and not tomorrow when asked on a
  Monday;
* "next week" honours `users.work_week_start`, for a Monday start and a Sunday
  start, on the same instant;
* today, tomorrow, this week, last month and explicit dates;
* three timezones, including Asia/Kolkata's half-hour offset, which is the one
  that catches code that thinks in whole hours;
* both daylight-saving transitions in America/New_York and Europe/London — a
  week window there is still seven calendar days, even though it is 167 or 169
  real hours;
* every window is half-open, so an event at exactly midnight belongs to one day
  and not to two.

Expected values are taken from `docs/SAMPLE_QUERIES.md` §9 wherever that
document states them, so the two can be read side by side.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import (
    FROZEN_NOW,
    MONDAY,
    SUNDAY,
    TZ_KOLKATA,
    TZ_LONDON,
    TZ_NY,
    TZ_UTC,
    at,
    call,
    load_any,
    local,
)

resolve = load_any("app.orchestrator.temporal", "resolve")
scan = load_any("app.orchestrator.temporal", "scan")


def w(phrase: str, tz: str = TZ_NY, week_start: int = MONDAY, now: datetime = FROZEN_NOW):
    """Resolve a phrase, and insist it resolved to something."""
    window = call(resolve, phrase, tz, week_start, now)
    assert window is not None, f"{phrase!r} did not resolve in {tz}"
    return window


def utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC)


def span_hours(window) -> float:
    return (window.end - window.start).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_window_is_tz_aware_and_forward_going():
    window = w("tomorrow")
    assert window.start.tzinfo is not None
    assert window.end.tzinfo is not None
    assert window.start < window.end
    assert window.tz == TZ_NY
    assert isinstance(window.interpretation, str) and window.interpretation.strip()


def test_a_phrase_with_no_time_in_it_resolves_to_nothing():
    # The pre-pass asks about every noun phrase it sees. Returning a window for
    # "the proposal" would silently narrow a search to a week nobody asked for.
    for phrase in ("the proposal", "acme corp", "john", ""):
        assert call(resolve, phrase, TZ_NY, MONDAY, FROZEN_NOW) is None, phrase


# ---------------------------------------------------------------------------
# "next <weekday>" — the following ISO week, never the next occurrence
# ---------------------------------------------------------------------------


def test_next_tuesday_asked_on_a_tuesday_is_seven_days_out():
    # Standing on Tuesday 25 August 2026, "next Tuesday" is 1 September, not today.
    tuesday = at(2026, 8, 25, 9, 0, tz=TZ_NY)
    window = w("next Tuesday", TZ_NY, MONDAY, tuesday)

    start_local = local(window.start, TZ_NY)
    assert (start_local.year, start_local.month, start_local.day) == (2026, 9, 1)
    assert start_local.hour == 0 and start_local.minute == 0
    assert start_local.date() - tuesday.date() == timedelta(days=7)
    assert span_hours(window) == pytest.approx(24.0)

    # And it says so, rather than leaving the user to guess which Tuesday it meant.
    said = window.interpretation.lower()
    assert "2026-09-01" in said or "sep" in said or "week" in said, window.interpretation


@pytest.mark.parametrize(
    ("asked_on", "expected"),
    [
        # From docs/SAMPLE_QUERIES.md §9. The Monday row is the one that matters:
        # nobody standing on a Monday says "next Tuesday" and means tomorrow.
        ((2026, 8, 20), (2026, 8, 25)),  # Thu, week 34 — both readings agree
        ((2026, 8, 23), (2026, 8, 25)),  # Sun, week 34 — both readings agree
        ((2026, 8, 24), (2026, 9, 1)),   # Mon, week 35 — the readings diverge
        ((2026, 8, 25), (2026, 9, 1)),   # Tue, week 35 — both readings agree
    ],
)
def test_next_weekday_is_the_following_iso_week(asked_on, expected):
    now = at(*asked_on, 11, 0, tz=TZ_NY)
    window = w("next Tuesday", TZ_NY, MONDAY, now)
    assert local(window.start, TZ_NY).date() == datetime(*expected).date()


def test_next_weekday_ignores_work_week_start():
    # A weekday is an ISO weekday. Where a user's week begins changes "next
    # week"; it does not change which day Tuesday is.
    monday_start = w("next Tuesday", TZ_NY, MONDAY, FROZEN_NOW)
    sunday_start = w("next Tuesday", TZ_NY, SUNDAY, FROZEN_NOW)
    assert utc(monday_start.start) == utc(sunday_start.start)
    assert utc(monday_start.start) == datetime(2026, 8, 25, 4, 0, tzinfo=UTC)
    assert utc(monday_start.end) == datetime(2026, 8, 26, 4, 0, tzinfo=UTC)


def test_a_bare_weekday_is_not_the_same_as_next_weekday():
    # "Tuesday" on a Thursday is the coming Tuesday; "next Tuesday" is a rule.
    # They agree this week, which is exactly why the rule has to be tested on a
    # day where they do not.
    monday = at(2026, 8, 24, 10, 0, tz=TZ_NY)
    bare = call(resolve, "Tuesday", TZ_NY, MONDAY, monday)
    if bare is not None:  # a bare weekday is optional; the rule for "next" is not
        assert local(bare.start, TZ_NY).date() == datetime(2026, 8, 25).date()
    explicit = w("next Tuesday", TZ_NY, MONDAY, monday)
    assert local(explicit.start, TZ_NY).date() == datetime(2026, 9, 1).date()


# ---------------------------------------------------------------------------
# "next week" and work_week_start
# ---------------------------------------------------------------------------


def test_next_week_with_a_monday_week_start():
    # ISO week 34 + 1 = week 35: Mon 24 Aug 00:00 EDT to Mon 31 Aug 00:00 EDT.
    window = w("next week", TZ_NY, MONDAY, FROZEN_NOW)
    assert utc(window.start) == datetime(2026, 8, 24, 4, 0, tzinfo=UTC)
    assert utc(window.end) == datetime(2026, 8, 31, 4, 0, tzinfo=UTC)
    assert local(window.start, TZ_NY).weekday() == 0  # Monday


def test_next_week_with_a_sunday_week_start():
    # Same instant, same timezone, one setting different: the week slides back a day.
    window = w("next week", TZ_NY, SUNDAY, FROZEN_NOW)
    assert utc(window.start) == datetime(2026, 8, 23, 4, 0, tzinfo=UTC)
    assert utc(window.end) == datetime(2026, 8, 30, 4, 0, tzinfo=UTC)
    assert local(window.start, TZ_NY).weekday() == 6  # Sunday


def test_next_week_is_seven_days_either_way():
    for week_start in (MONDAY, SUNDAY):
        window = w("next week", TZ_NY, week_start, FROZEN_NOW)
        assert local(window.end, TZ_NY).date() - local(window.start, TZ_NY).date() == timedelta(
            days=7
        )


def test_this_week_and_next_week_meet_exactly():
    # Half-open windows: no gap to fall into, no overlap to be counted twice.
    this_week = w("this week", TZ_NY, MONDAY, FROZEN_NOW)
    next_week = w("next week", TZ_NY, MONDAY, FROZEN_NOW)
    assert this_week.end == next_week.start
    assert utc(this_week.start) == datetime(2026, 8, 17, 4, 0, tzinfo=UTC)
    assert utc(this_week.end) == datetime(2026, 8, 24, 4, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# The everyday phrases
# ---------------------------------------------------------------------------


def test_today_and_tomorrow_in_new_york():
    # 13:12 UTC is 09:12 in New York, so "today" is still 20 August there.
    today = w("today", TZ_NY, MONDAY, FROZEN_NOW)
    tomorrow = w("tomorrow", TZ_NY, MONDAY, FROZEN_NOW)

    assert utc(today.start) == datetime(2026, 8, 20, 4, 0, tzinfo=UTC)
    assert utc(today.end) == datetime(2026, 8, 21, 4, 0, tzinfo=UTC)
    assert utc(tomorrow.start) == datetime(2026, 8, 21, 4, 0, tzinfo=UTC)
    assert utc(tomorrow.end) == datetime(2026, 8, 22, 4, 0, tzinfo=UTC)
    assert today.end == tomorrow.start


def test_today_and_tomorrow_in_kolkata():
    # 13:12 UTC is 18:42 in Kolkata — same calendar day, different day boundary.
    today = w("today", TZ_KOLKATA, MONDAY, FROZEN_NOW)
    tomorrow = w("tomorrow", TZ_KOLKATA, MONDAY, FROZEN_NOW)

    assert utc(today.start) == datetime(2026, 8, 19, 18, 30, tzinfo=UTC)
    assert utc(today.end) == datetime(2026, 8, 20, 18, 30, tzinfo=UTC)
    assert utc(tomorrow.start) == datetime(2026, 8, 20, 18, 30, tzinfo=UTC)
    assert local(today.start, TZ_KOLKATA).hour == 0


def test_yesterday_is_the_day_before_today():
    yesterday = w("yesterday", TZ_NY, MONDAY, FROZEN_NOW)
    today = w("today", TZ_NY, MONDAY, FROZEN_NOW)
    assert yesterday.end == today.start
    assert utc(yesterday.start) == datetime(2026, 8, 19, 4, 0, tzinfo=UTC)


def test_last_month_is_the_whole_previous_calendar_month():
    # "Show me PDFs in Drive from last month", asked on 20 August, means July.
    window = w("last month", TZ_NY, MONDAY, FROZEN_NOW)
    assert utc(window.start) == datetime(2026, 7, 1, 4, 0, tzinfo=UTC)
    assert utc(window.end) == datetime(2026, 8, 1, 4, 0, tzinfo=UTC)
    # Not "the last thirty days" — the two differ by nine days here.
    assert span_hours(window) == pytest.approx(31 * 24.0)


def test_this_month_runs_to_the_first_of_the_next_one():
    window = w("this month", TZ_NY, MONDAY, FROZEN_NOW)
    assert utc(window.start) == datetime(2026, 8, 1, 4, 0, tzinfo=UTC)
    assert utc(window.end) == datetime(2026, 9, 1, 4, 0, tzinfo=UTC)


@pytest.mark.parametrize("phrase", ["2026-09-05", "September 5, 2026", "5 September 2026"])
def test_an_explicit_date_is_that_local_day(phrase):
    window = w(phrase, TZ_NY, MONDAY, FROZEN_NOW)
    start_local = local(window.start, TZ_NY)
    end_local = local(window.end, TZ_NY)
    assert (start_local.year, start_local.month, start_local.day) == (2026, 9, 5)
    assert start_local.hour == 0 and start_local.minute == 0
    assert (end_local.year, end_local.month, end_local.day) == (2026, 9, 6)


# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tz", "expected_start"),
    [
        (TZ_UTC, datetime(2026, 8, 21, 0, 0, tzinfo=UTC)),
        (TZ_NY, datetime(2026, 8, 21, 4, 0, tzinfo=UTC)),
        (TZ_KOLKATA, datetime(2026, 8, 20, 18, 30, tzinfo=UTC)),
    ],
)
def test_tomorrow_starts_at_local_midnight_in_every_timezone(tz, expected_start):
    window = w("tomorrow", tz, MONDAY, FROZEN_NOW)
    assert utc(window.start) == expected_start
    assert local(window.start, tz).hour == 0
    assert local(window.start, tz).minute == 0


def test_kolkata_keeps_its_half_hour_offset():
    # The offset is +05:30, so every window boundary lands on :30 in UTC. Code
    # that rounds to the hour passes every other timezone and fails here.
    window = w("next week", TZ_KOLKATA, MONDAY, FROZEN_NOW)
    assert utc(window.start) == datetime(2026, 8, 23, 18, 30, tzinfo=UTC)
    assert utc(window.end) == datetime(2026, 8, 30, 18, 30, tzinfo=UTC)
    assert utc(window.start).minute == 30


def test_the_same_instant_gives_three_different_day_boundaries():
    # 2026-08-20T13:12:04Z, three users, one question. All three are on the same
    # calendar day, and all three days start at a different instant. This is why
    # `tz` is carried on the window rather than assumed by the caller.
    days = {
        tz: local(w("today", tz, MONDAY, FROZEN_NOW).start, tz).date()
        for tz in (TZ_UTC, TZ_NY, TZ_KOLKATA)
    }
    assert days[TZ_UTC] == days[TZ_NY] == days[TZ_KOLKATA] == datetime(2026, 8, 20).date()

    starts = {tz: w("today", tz, MONDAY, FROZEN_NOW).start for tz in (TZ_UTC, TZ_NY, TZ_KOLKATA)}
    assert len({s for s in starts.values()}) == 3
    assert starts[TZ_NY] - starts[TZ_KOLKATA] == timedelta(hours=9, minutes=30)
    assert starts[TZ_NY] - starts[TZ_UTC] == timedelta(hours=4)


# ---------------------------------------------------------------------------
# Daylight saving
# ---------------------------------------------------------------------------
#
# A week is seven calendar days. Across a transition it is not 168 real hours,
# and both of those facts have to hold at once: the window still covers Monday
# to Monday, and its length in real time is 167 or 169 hours. Code that builds a
# week as `start + timedelta(days=7)` in UTC gets the length right and the days
# wrong — the second Monday lands at 23:00 or 01:00 local, and every "is this
# event in the window" answer near the edge flips.


@pytest.mark.parametrize(
    ("tz", "asked_on", "first_monday", "last_monday", "hours"),
    [
        # New York: clocks go forward Sun 8 March 2026, back Sun 1 November 2026.
        (TZ_NY, (2026, 3, 4), (2026, 3, 2), (2026, 3, 9), 167.0),
        (TZ_NY, (2026, 10, 28), (2026, 10, 26), (2026, 11, 2), 169.0),
        # London: BST starts Sun 29 March 2026, ends Sun 25 October 2026.
        (TZ_LONDON, (2026, 3, 25), (2026, 3, 23), (2026, 3, 30), 167.0),
        (TZ_LONDON, (2026, 10, 21), (2026, 10, 19), (2026, 10, 26), 169.0),
    ],
)
def test_a_week_across_a_clock_change_is_still_seven_days(
    tz, asked_on, first_monday, last_monday, hours
):
    now = at(*asked_on, 12, 0, tz=tz)
    window = w("this week", tz, MONDAY, now)

    start_local = local(window.start, tz)
    end_local = local(window.end, tz)

    # Seven calendar days, both ends on local midnight.
    assert start_local.date() == datetime(*first_monday).date()
    assert end_local.date() == datetime(*last_monday).date()
    assert end_local.date() - start_local.date() == timedelta(days=7)
    assert (start_local.hour, start_local.minute) == (0, 0)
    assert (end_local.hour, end_local.minute) == (0, 0)

    # And the real elapsed time is an hour short or an hour long, which is the
    # whole point: the arithmetic happened on local wall time, not on UTC.
    assert span_hours(window) == pytest.approx(hours)


@pytest.mark.parametrize(
    ("tz", "asked_on", "hours"),
    [
        (TZ_NY, (2026, 3, 7), 23.0),   # Saturday, so "tomorrow" is the short day
        (TZ_NY, (2026, 10, 31), 25.0),  # Saturday, so "tomorrow" is the long day
        (TZ_LONDON, (2026, 3, 28), 23.0),
        (TZ_LONDON, (2026, 10, 24), 25.0),
    ],
)
def test_the_day_a_clock_changes_is_not_twenty_four_hours(tz, asked_on, hours):
    now = at(*asked_on, 12, 0, tz=tz)
    window = w("tomorrow", tz, MONDAY, now)
    assert local(window.start, tz).hour == 0
    assert local(window.end, tz).hour == 0
    assert span_hours(window) == pytest.approx(hours)


def test_next_week_across_the_spring_change_still_starts_on_a_monday_midnight():
    # Asked the week before the change, so the transition falls inside the
    # window rather than at its edge.
    now = at(2026, 2, 25, 9, 0, tz=TZ_NY)
    window = w("next week", TZ_NY, MONDAY, now)
    start_local = local(window.start, TZ_NY)
    assert start_local.date() == datetime(2026, 3, 2).date()
    assert start_local.weekday() == 0
    assert local(window.end, TZ_NY).date() == datetime(2026, 3, 9).date()
    assert span_hours(window) == pytest.approx(167.0)


# ---------------------------------------------------------------------------
# Half-open, and what that buys
# ---------------------------------------------------------------------------


def contains(window, moment: datetime) -> bool:
    """The membership test every caller of a window is expected to use."""
    return window.start <= moment < window.end


def test_midnight_belongs_to_exactly_one_day():
    today = w("today", TZ_NY, MONDAY, FROZEN_NOW)
    tomorrow = w("tomorrow", TZ_NY, MONDAY, FROZEN_NOW)
    midnight = at(2026, 8, 21, 0, 0, tz=TZ_NY)  # the boundary itself

    assert not contains(today, midnight)
    assert contains(tomorrow, midnight)
    # One second earlier is still today.
    assert contains(today, midnight - timedelta(seconds=1))


def test_midnight_belongs_to_exactly_one_week():
    this_week = w("this week", TZ_NY, MONDAY, FROZEN_NOW)
    next_week = w("next week", TZ_NY, MONDAY, FROZEN_NOW)
    boundary = at(2026, 8, 24, 0, 0, tz=TZ_NY)  # Monday 00:00, the seam

    assert not contains(this_week, boundary)
    assert contains(next_week, boundary)
    assert contains(this_week, boundary - timedelta(microseconds=1))


def test_sunday_is_in_next_week_and_the_following_monday_is_not():
    # Straight out of docs/SAMPLE_QUERIES.md §1.
    window = w("next week", TZ_NY, MONDAY, FROZEN_NOW)
    assert contains(window, at(2026, 8, 30, 23, 59, tz=TZ_NY))  # Sunday night
    assert not contains(window, at(2026, 8, 31, 0, 0, tz=TZ_NY))  # Monday morning


def test_a_kolkata_midnight_event_is_not_a_new_york_midnight_event():
    # The half-hour offset again: 2026-08-21T00:00 IST is 18:30Z on the 20th,
    # which is still today for a New York user.
    kolkata_midnight = at(2026, 8, 21, 0, 0, tz=TZ_KOLKATA)
    ny_today = w("today", TZ_NY, MONDAY, FROZEN_NOW)
    kolkata_tomorrow = w("tomorrow", TZ_KOLKATA, MONDAY, FROZEN_NOW)

    assert contains(ny_today, kolkata_midnight)
    assert contains(kolkata_tomorrow, kolkata_midnight)


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------


def test_scan_finds_every_phrase_in_a_sentence():
    text = "Move my meeting from tomorrow to next Tuesday"
    windows = call(scan, text, TZ_NY, MONDAY, FROZEN_NOW)

    assert isinstance(windows, dict)
    assert len(windows) >= 2, windows

    found = {(local(win.start, TZ_NY).date()) for win in windows.values()}
    assert datetime(2026, 8, 21).date() in found  # tomorrow
    assert datetime(2026, 8, 25).date() in found  # next Tuesday


def test_scan_agrees_with_resolve():
    windows = call(scan, "what is on my calendar next week", TZ_NY, MONDAY, FROZEN_NOW)
    assert windows, "scan found no time phrase in a sentence that has one"
    direct = w("next week", TZ_NY, MONDAY, FROZEN_NOW)
    assert any(
        win.start == direct.start and win.end == direct.end for win in windows.values()
    ), windows


def test_scan_finds_nothing_when_there_is_nothing():
    windows = call(scan, "find the Acme proposal", TZ_NY, MONDAY, FROZEN_NOW)
    assert windows == {}


def test_scan_keys_are_usable_as_reference_names():
    # The planner writes `{{windows.<name>.start}}`, so the keys have to be
    # plain identifiers rather than the raw phrase.
    windows = call(scan, "anything on my calendar next week?", TZ_NY, MONDAY, FROZEN_NOW)
    for name in windows:
        assert name.replace("_", "").isalnum(), name
        assert name == name.strip()
