"""Calendar ops: four reads, one matcher, three gated writes.

The interesting ones are the two that do arithmetic instead of retrieval.
`gcal.find_free_slots` and `gcal.find_conflicts` are pure Python over tz-aware
datetimes — no network, no model, about two milliseconds — because interval
maths is the kind of thing a language model gets subtly wrong and a computer
gets exactly right.

The three writes are all `ConfirmableOp`. `run` prepares:

* `gcal.create_event` resolves attendees and works out the end time;
* `gcal.update_event` and `gcal.delete_event` read the event **live** and pin
  its `etag`, so `execute` can send `If-Match` and get a 412 rather than
  overwriting a change somebody else made in between.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    ambiguity_test,
    as_list,
    admits,
    choice_input,
    duration_minutes,
    excerpt_of,
    google_call,
    has_google,
    hybrid,
    is_exact,
    iso,
    jsonable,
    label_for,
    parse_dt,
    row_to_dict,
    shape_hit,
    window_bounds,
)

# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #

GCAL_FILTERS: dict[str, str | None] = {
    "calendar_id": "calendar_id",
    "organizer_email": "organizer_email",
    "attendee_emails": "attendee_emails[]",
    "event_ids": "event_ids[]",
    "recurring_event_id": "recurring_event_id",
    "status": "status",
    "status_in": None,
    "not_cancelled": "not_cancelled",
    "all_day": "bool:all_day",
    "ends_after": "ends_after",
    "since": "since",
    "until": "until",
    "window": "window",
    "participants": None,
    "min_cn": None,
    "exclude_refs": None,
    "text_contains": None,
}

GCAL_ORDER: dict[str, str] = {
    "relevance": "cn",
    "starts_at": "starts_at",
    "start": "starts_at",
    "ends_at": "ends_at",
    "title": "label",
}


class SearchEvents(SearchOp):
    """Events out of the mirror.

    "What is on my calendar next week" is a window read with no query string,
    which is why `order_by` defaults to relevance but almost every plan sets
    `starts_at`: a week reads as a week, not as a ranking.
    """

    name = "gcal.search_events"
    corpus = "gcal"
    entity_type = "event"
    filter_spec = GCAL_FILTERS
    order_spec = GCAL_ORDER
    summary = "search or list your calendar (mirror, not Google)"
    # A guest filter starving a *named* search gets dropped over ending the
    # run: "meeting with John" often names a title, not an invitee, and the
    # address the filter holds is only as good as the resolution that made it.
    rescuable_filters = frozenset({"attendee_emails", "organizer_email"})

    async def refresh_live(self, ctx: OpContext, args: Any, filters: Any) -> dict | None:
        if not has_google(ctx, "gcal", ("list_events", "search_events", "list")):
            return None
        raw = await google_call(
            ctx,
            "gcal",
            ("list_events", "search_events", "list"),
            calendar_id=filters.sql.get("calendar_id") or "primary",
            time_min=filters.sql.get("since"),
            time_max=filters.sql.get("until"),
            query=(args.query or "").strip() or None,
            max_results=min(args.limit + args.offset + 10, 100),
        )
        rows = [_mirror_row_from_event(row_to_dict(e)) for e in (raw or [])]
        rows = [r for r in rows if r.get("event_id") and r.get("starts_at")]
        if rows:
            await mirror_repo.upsert_gcal(ctx.session, ctx.user_id, rows)
        return {"fetched": len(rows)}

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        # `status_in` is the one filter the mirror's whitelist cannot take as a
        # list, so it is folded into the single-value form when it can be and
        # left to the Python pass when it cannot.
        raw = dict(args or {})
        merged = {**raw.get("filter", {}), **{k: v for k, v in raw.items() if k in GCAL_FILTERS}}
        statuses = [str(s).lower() for s in as_list(merged.get("status_in"))]
        if statuses:
            raw = {k: v for k, v in raw.items() if k != "status_in"}
            filt = {k: v for k, v in (raw.get("filter") or {}).items() if k != "status_in"}
            if len(statuses) == 1:
                filt["status"] = statuses[0]
            elif "cancelled" not in statuses:
                filt["not_cancelled"] = True
            raw["filter"] = filt
        result = await super().run(ctx, raw)
        if statuses and len(statuses) > 1:
            keep = set(statuses)
            hits = [h for h in result.data.get("hits", []) if str(h.get("status") or "confirmed").lower() in keep]
            result.data["hits"] = hits
            result.data["count"] = len(hits)
        return result

    def progress_label(self, args: dict) -> str:
        args = args or {}
        window = args.get("window") or (args.get("filter") or {}).get("window")
        if window:
            try:
                lo, hi = window_bounds(window)
                return f"Reading your calendar for {_span_label(lo, hi, args.get('tz'))}"
            except AppError:
                pass
        query = args.get("query")
        return f"Looking for “{excerpt_of(query, 40)}” on your calendar" if query else "Reading your calendar"

    def ambiguity_question(self, args: Any, hits: list[dict]) -> str:
        return "Which meeting did you mean?"


def _span_label(lo: dt.datetime, hi: dt.datetime, tz: str | None = None) -> str:
    """The window in words, in the reader's own timezone.

    Rendering in UTC is off by a day for anyone east of Greenwich: a window
    starting midnight in Asia/Kolkata is 18:30 the previous day in UTC, so the
    label said "23-30 Aug" for a week that begins on the 24th.
    """
    if tz:
        try:
            zone = ZoneInfo(tz)
            lo, hi = lo.astimezone(zone), hi.astimezone(zone)
        except Exception:
            pass
    last = hi - dt.timedelta(seconds=1)
    if lo.date() == last.date():
        return lo.strftime("%a %-d %b")
    if lo.year == last.year and lo.month == last.month:
        return f"{lo.day}–{last.day} {lo.strftime('%b')}"
    return f"{lo.strftime('%-d %b')} – {last.strftime('%-d %b')}"


class GetEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_ids: list[str] = Field(default_factory=list)
    event_id: str | None = None
    calendar_id: str = "primary"
    expect: str = "many"
    freshness: str = "cached"
    project: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if isinstance(values, dict) and "event_ids" in values:
            values = {**values, "event_ids": as_list(values["event_ids"])}
        return values

    @model_validator(mode="after")
    def _check(self) -> "GetEventsArgs":
        if self.event_id and self.event_id not in self.event_ids:
            self.event_ids = [self.event_id, *self.event_ids]
        if not self.event_ids:
            raise ValueError("gcal.get_events needs at least one event id")
        return self


class GetEvents(Op):
    """Whole events by id.

    `freshness: "live"` matters more here than anywhere else: an `etag` read
    from the mirror is an `etag` that may already be stale, and the update ops
    refuse to prepare against one.
    """

    name = "gcal.get_events"
    args_model = GetEventsArgs
    output_fields = [
        "events",
        "count",
        "event_id",
        "calendar_id",
        "etag",
        "title",
        "description",
        "location",
        "starts_at",
        "ends_at",
        "duration_minutes",
        "attendees",
        "attendee_emails",
        "organizer_email",
        "recurring_event_id",
        "status",
    ]
    is_local = True
    timeout_s = 4.0
    summary = "read whole calendar events by id"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        events: list[dict] = []
        missing: list[str] = []
        for event_id in parsed.event_ids:
            row = await self._one(ctx, event_id, parsed)
            if row is None:
                missing.append(event_id)
            else:
                events.append(row)

        data: dict[str, Any] = {
            "events": events,
            "count": len(events),
            "missing": missing,
            "found": bool(events),
        }
        if parsed.expect == "one" and events:
            data = {**events[0], **data}
        return OpResult(
            data=data,
            needs_replan=not events and parsed.expect == "one",
            replan_reason="that event is not in the mirror" if not events and parsed.expect == "one" else None,
        )

    async def _one(self, ctx: OpContext, event_id: str, parsed: Any) -> dict | None:
        live = parsed.freshness == "live" and has_google(ctx, "gcal", ("get_event", "get"))
        if live:
            fetched = await google_call(
                ctx,
                "gcal",
                ("get_event", "get"),
                event_id=event_id,
                calendar_id=parsed.calendar_id,
            )
            if fetched:
                row = _mirror_row_from_event(row_to_dict(fetched))
                if row.get("event_id") and row.get("starts_at"):
                    await mirror_repo.upsert_gcal(ctx.session, ctx.user_id, [row])
                return _shape_event(row)

        rows = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gcal", event_id)
        if not rows:
            return None
        return _shape_event(row_to_dict(rows[0]), project=parsed.project)

    def progress_label(self, args: dict) -> str:
        count = len(as_list((args or {}).get("event_ids"))) or 1
        return "Opening that event" if count == 1 else f"Opening {count} events"


def _shape_event(row: dict, project: list[str] | None = None) -> dict:
    out = {
        "event_id": row.get("event_id"),
        "calendar_id": row.get("calendar_id") or "primary",
        "recurring_event_id": row.get("recurring_event_id"),
        "etag": row.get("etag"),
        "title": row.get("title"),
        "description": row.get("description"),
        "location": row.get("location"),
        "organizer_email": row.get("organizer_email"),
        "attendees": row.get("attendees") or [],
        "attendee_emails": as_list(row.get("attendee_emails"))
        or [str(a.get("email")) for a in (row.get("attendees") or []) if isinstance(a, dict) and a.get("email")],
        "starts_at": row.get("starts_at"),
        "ends_at": row.get("ends_at"),
        "all_day": bool(row.get("all_day")),
        "event_timezone": row.get("event_timezone"),
        "status": row.get("status"),
        "duration_minutes": duration_minutes(row.get("starts_at"), row.get("ends_at")),
        "label": label_for("gcal", row),
    }
    if project:
        keep = set(project) | {"event_id", "etag"}
        out = {k: v for k, v in out.items() if k in keep}
    return jsonable(out)


def _mirror_row_from_event(event: dict) -> dict[str, Any]:
    """A Calendar API event as a `sync_gcal` row."""
    start_block = event.get("start")
    end_block = event.get("end")
    organizer = event.get("organizer")
    row = {
        "event_id": event.get("event_id") or event.get("id"),
        "calendar_id": event.get("calendar_id") or event.get("calendarId") or "primary",
        "recurring_event_id": event.get("recurring_event_id") or event.get("recurringEventId"),
        "title": event.get("title") or event.get("summary"),
        "description": event.get("description"),
        "location": event.get("location"),
        "organizer_email": event.get("organizer_email")
        or (organizer.get("email") if isinstance(organizer, dict) else organizer),
        "attendees": jsonable(event.get("attendees") or []),
        "starts_at": event.get("starts_at") or _google_time(start_block),
        "ends_at": event.get("ends_at") or _google_time(end_block),
        "all_day": bool(
            event.get("all_day") or (isinstance(start_block, dict) and start_block.get("date"))
        ),
        "event_timezone": event.get("event_timezone")
        or (start_block.get("timeZone") if isinstance(start_block, dict) else None),
        "status": event.get("status"),
        "etag": event.get("etag"),
    }
    row["content_hash"] = fingerprint(
        "sync_gcal.event",
        f"{row['event_id']}|{row['title']}|{iso(row['starts_at'])}|{iso(row['ends_at'])}|{row['status']}",
    )
    return row


def _google_time(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("dateTime") or value.get("date")
    return value


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo("UTC")


def _merge(intervals: list[tuple[dt.datetime, dt.datetime]]) -> list[tuple[dt.datetime, dt.datetime]]:
    """Overlapping ranges collapsed into disjoint ones, in order."""
    ordered = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    out: list[tuple[dt.datetime, dt.datetime]] = []
    for lo, hi in ordered:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _loose_dt(value: Any, *, tz: str | None = None) -> dt.datetime | None:
    """A date from anything, or ``None`` — never an exception.

    The values reaching here often come from a model reading a document, so
    they arrive as somebody wrote them: "26 August 2026", "29 August". Strict
    parsing raises on those and takes the whole step with it, which turns a
    date we could not read into "Calendar is not responding". Dropping the one
    range we cannot understand and checking the rest is the honest behaviour.
    """
    try:
        parsed = parse_dt(value)
        if parsed is not None:
            return parsed
    except Exception:  # noqa: BLE001 - fall through to the prose reader
        pass

    text = str(value or "").strip()
    if not text:
        return None
    try:
        from app.orchestrator import temporal

        window = temporal.resolve(text, tz=tz or "UTC")
        return window.start if window else None
    except Exception:  # noqa: BLE001 - an unreadable date is not a failure
        return None


def _ranges_from(
    items: Iterable[Any], start_key: str, end_key: str, *, tz: str | None = None
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Whatever the plan pointed at, as intervals.

    Takes events, `{"start","end"}` dicts, and two-element lists, because a
    reference like `{{ooo_doc.extracted.ranges}}` can resolve to any of them —
    and prose dates, because that is what a document extraction produces.
    """
    out: list[tuple[dt.datetime, dt.datetime]] = []
    for item in items or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lo, hi = _loose_dt(item[0], tz=tz), _loose_dt(item[1], tz=tz)
        elif isinstance(item, dict):
            lo = _loose_dt(
                item.get(start_key) or item.get("start") or item.get("starts_at")
                or item.get("from"), tz=tz
            )
            hi = _loose_dt(
                item.get(end_key) or item.get("end") or item.get("ends_at")
                or item.get("to"), tz=tz
            )
            if lo is not None and hi is None:
                minutes = item.get("duration_minutes")
                hi = lo + dt.timedelta(minutes=int(minutes)) if minutes else lo + dt.timedelta(hours=1)
            elif lo is not None and hi is not None and hi <= lo:
                # "26 August to 29 August" resolves both ends to midnight, so
                # the 29th would contribute nothing. An out-of-office note that
                # says "through the 29th" means the whole of it.
                hi = hi + dt.timedelta(days=1)
        else:
            continue
        if lo is not None and hi is not None and hi > lo:
            out.append((lo, hi))
    return out


class FreeSlotsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: dict[str, Any] | list[Any]
    duration_minutes: int = Field(default=30, ge=5, le=8 * 60)
    working_hours: dict[str, str] = Field(default_factory=lambda: {"start": "09:00", "end": "17:00"})
    days: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])  # 1=Mon .. 7=Sun
    buffer_minutes: int = Field(default=0, ge=0, le=120)
    calendar_id: str = "primary"
    busy: list[Any] = Field(default_factory=list)
    tz: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class FindFreeSlots(Op):
    """Gaps big enough to put a meeting in.

    Working hours are wall-clock local, so a slot never lands at 3 a.m. because
    of a UTC offset. Everything is computed on tz-aware datetimes and returned
    in UTC, with the local time alongside for the answer to read out.
    """

    name = "gcal.find_free_slots"
    args_model = FreeSlotsArgs
    output_fields = ["slots", "count", "window", "duration_minutes"]
    is_local = True
    timeout_s = 3.0
    summary = "find open slots of a given length in a window"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        lo, hi = window_bounds(parsed.window)
        zone = _zone(parsed.tz or ctx.tz)

        busy = _ranges_from(parsed.busy, "starts_at", "ends_at")
        if not busy:
            rows = await mirror_repo.list_window(
                ctx.session,
                ctx.user_id,
                "gcal",
                lo - dt.timedelta(days=1),
                hi + dt.timedelta(days=1),
                filters={"calendar_id": parsed.calendar_id},
                limit=500,
            )
            busy = [
                (parse_dt(r["starts_at"]), parse_dt(r["ends_at"]) or parse_dt(r["starts_at"]) + dt.timedelta(hours=1))
                for r in (row_to_dict(row) for row in rows)
                if r.get("starts_at") and str(r.get("status") or "").lower() != "cancelled"
            ]
        pad = dt.timedelta(minutes=parsed.buffer_minutes)
        busy = _merge([(b0 - pad, b1 + pad) for b0, b1 in busy])

        need = dt.timedelta(minutes=parsed.duration_minutes)
        slots: list[dict] = []
        for day_lo, day_hi in _working_spans(lo, hi, zone, parsed.working_hours, set(parsed.days)):
            cursor = day_lo
            for b0, b1 in busy:
                if b1 <= cursor or b0 >= day_hi:
                    continue
                if b0 - cursor >= need:
                    slots.append(_slot(cursor, min(b0, day_hi), zone, need))
                cursor = max(cursor, b1)
                if cursor >= day_hi:
                    break
            if day_hi - cursor >= need:
                slots.append(_slot(cursor, day_hi, zone, need))
            if len(slots) >= parsed.limit:
                break

        slots = slots[: parsed.limit]
        return OpResult(
            data=jsonable(
                {
                    "slots": slots,
                    "count": len(slots),
                    "window": {"start": lo, "end": hi},
                    "duration_minutes": parsed.duration_minutes,
                    "tz": str(zone),
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        minutes = (args or {}).get("duration_minutes") or 30
        return f"Looking for a free {minutes} minutes"


def _slot(start: dt.datetime, end: dt.datetime, zone: ZoneInfo, need: dt.timedelta) -> dict:
    local = start.astimezone(zone)
    return {
        "start": start,
        "end": end,
        "minutes": int((end - start).total_seconds() // 60),
        "local": local.strftime("%a %-d %b %H:%M"),
        "fits": (end - start) >= need,
    }


def _working_spans(
    lo: dt.datetime,
    hi: dt.datetime,
    zone: ZoneInfo,
    hours: dict[str, str],
    days: set[int],
) -> list[tuple[dt.datetime, dt.datetime]]:
    """The window cut into one span per working day, in wall-clock local time."""
    start_h, start_m = _hhmm(hours.get("start", "09:00"))
    end_h, end_m = _hhmm(hours.get("end", "17:00"))
    spans: list[tuple[dt.datetime, dt.datetime]] = []
    day = lo.astimezone(zone).date()
    last = (hi - dt.timedelta(seconds=1)).astimezone(zone).date()
    while day <= last:
        if day.isoweekday() in days:
            begin = dt.datetime.combine(day, dt.time(start_h, start_m), zone).astimezone(dt.timezone.utc)
            close = dt.datetime.combine(day, dt.time(end_h, end_m), zone).astimezone(dt.timezone.utc)
            begin, close = max(begin, lo), min(close, hi)
            if close > begin:
                spans.append((begin, close))
        day += dt.timedelta(days=1)
    return spans


def _hhmm(value: str) -> tuple[int, int]:
    try:
        hour, _, minute = str(value).partition(":")
        return int(hour), int(minute or 0)
    except ValueError:
        raise AppError("VALIDATION_ERROR", f"{value!r} is not a time of day.", http=422) from None


class ConflictsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Any] = Field(default_factory=list)
    item_start: str = "starts_at"
    item_end: str = "ends_at"
    against: list[Any] = Field(default_factory=list)
    window: dict[str, Any] | list[Any] | None = None
    calendar_id: str = "primary"
    tz: str | None = None
    min_overlap_minutes: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _something_to_compare(self) -> "ConflictsArgs":
        if not self.items and self.window is None:
            raise ValueError("give items to check, or a window to read them from")
        return self


class FindConflicts(Op):
    """Which of these events overlap something they should not.

    Two modes, one implementation. With ``against`` it is an intersection —
    meetings against the dates in an out-of-office note. Without it, the items
    are checked against each other: double bookings.
    """

    name = "gcal.find_conflicts"
    args_model = ConflictsArgs
    output_fields = ["conflicts", "count", "clear", "checked"]
    is_local = True
    timeout_s = 3.0
    summary = "overlap check between events, or events and date ranges"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        items = list(parsed.items)
        if not items and parsed.window is not None:
            lo, hi = window_bounds(parsed.window)
            rows = await mirror_repo.list_window(
                ctx.session,
                ctx.user_id,
                "gcal",
                lo,
                hi,
                filters={"calendar_id": parsed.calendar_id},
                limit=200,
            )
            items = [_shape_event(row_to_dict(r)) for r in rows]
            items = [i for i in items if str(i.get("status") or "").lower() != "cancelled"]

        blocks = _ranges_from(parsed.against, "start", "end", tz=parsed.tz or ctx.tz)
        floor = dt.timedelta(minutes=parsed.min_overlap_minutes)
        conflicts: list[dict] = []
        clear: list[dict] = []

        # Pair each item with its own interval rather than zipping two lists:
        # an item with no usable times drops out of both, and a silent
        # off-by-one here would report the wrong meeting as the clash.
        pairs: list[tuple[dict, tuple[dt.datetime, dt.datetime]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            span = _ranges_from(
                [item], parsed.item_start, parsed.item_end, tz=parsed.tz or ctx.tz
            )
            if span:
                pairs.append((item, span[0]))
        spans = [span for _, span in pairs]

        for item, (lo, hi) in pairs:
            hits: list[dict] = []
            if blocks:
                for b0, b1 in blocks:
                    overlap = min(hi, b1) - max(lo, b0)
                    if overlap >= floor and overlap > dt.timedelta(0):
                        hits.append(
                            {
                                "against": {"start": b0, "end": b1},
                                "overlap_minutes": int(overlap.total_seconds() // 60),
                            }
                        )
            else:
                for other, (o0, o1) in pairs:
                    if other is item:
                        continue
                    overlap = min(hi, o1) - max(lo, o0)
                    if overlap >= floor and overlap > dt.timedelta(0):
                        hits.append(
                            {
                                "against": {
                                    "event_id": other.get("event_id"),
                                    "title": other.get("title") or other.get("label"),
                                    "start": o0,
                                    "end": o1,
                                },
                                "overlap_minutes": int(overlap.total_seconds() // 60),
                            }
                        )
            record = {
                "event_id": item.get("event_id"),
                "title": item.get("title") or item.get("label"),
                "starts_at": lo,
                "ends_at": hi,
                "attendee_emails": as_list(item.get("attendee_emails")),
            }
            if hits:
                conflicts.append({**record, "overlaps": hits})
            else:
                clear.append(record)

        return OpResult(
            data=jsonable(
                {
                    "conflicts": conflicts,
                    "count": len(conflicts),
                    "clear": clear,
                    "checked": len(spans),
                    "mode": "against_ranges" if blocks else "pairwise",
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        return "Checking for clashes"


class MatchEventArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    query: str | None = None
    window: dict[str, Any] | list[Any] | None = None
    participants: list[str] = Field(default_factory=list)
    starts_at: Any = None
    calendar_id: str = "primary"
    expect: str = "one"
    limit: int = Field(default=5, ge=1, le=25)

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if isinstance(values, dict) and "participants" in values:
            values = {**values, "participants": as_list(values["participants"])}
        return values


class MatchEvent(Op):
    """Turn "the meeting with John" into one event, or into a question.

    Retrieval ranks; this decides. A candidate is admitted on `cn` **or** an
    exact flag, the top two are compared against `MARGIN`, and anything that
    fails the test comes back as a choice card carrying the real events rather
    than as a guess with a nice label on it.
    """

    name = "gcal.match_event"
    args_model = MatchEventArgs
    output_fields = ["event_id", "etag", "title", "starts_at", "ends_at", "candidates", "matched"]
    is_local = True
    timeout_s = 3.0
    summary = "pick the one event a phrase refers to, or ask"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        text = (parsed.description or parsed.query or "").strip()

        filters: dict[str, Any] = {}
        if parsed.window is not None:
            lo, hi = window_bounds(parsed.window)
            filters["since"], filters["until"] = lo, hi
        if parsed.calendar_id:
            filters["calendar_id"] = parsed.calendar_id
        filters["not_cancelled"] = True

        rows = await hybrid(ctx, "gcal", query=text or None, filters=filters, limit=max(parsed.limit * 3, 15))
        hits = [
            shape_hit("gcal", row, query=text, filters=filters, body=False, extract=False)
            for row in rows
        ]
        if parsed.participants:
            wanted = {p.lower() for p in parsed.participants}
            hits = [
                h
                for h in hits
                if wanted & {str(e).lower() for e in as_list(h.get("attendee_emails")) + as_list(h.get("organizer_email"))}
            ]
        pinned = parse_dt(parsed.starts_at) if parsed.starts_at else None
        if pinned is not None:
            hits.sort(key=lambda h: abs((parse_dt(h.get("starts_at")) or pinned) - pinned))
            near = [h for h in hits if abs((parse_dt(h.get("starts_at")) or pinned) - pinned) <= dt.timedelta(hours=2)]
            if near:
                hits = near
        else:
            hits.sort(key=lambda h: float(h.get("cn") or 0.0), reverse=True)
        hits = hits[: parsed.limit]

        reason = ambiguity_test(hits) if parsed.expect == "one" else None
        if reason == "margin":
            # Only candidates the search actually rates go on the card. Padding
            # a two-way pick with a row that scored 0.17 makes the real options
            # look like guesses and gives the reader a wrong answer to click.
            plausible = [h for h in hits if admits(h)] or hits[:1]
            return OpResult(
                data={"candidates": plausible, "matched": False, "reason": reason},
                needs_input=choice_input(
                    "Which meeting did you mean?",
                    plausible,
                    id_field="event_id",
                    help_text="These scored too close to each other to choose between.",
                ),
            )
        if reason == "absent" or not hits:
            return OpResult(
                data={"candidates": hits, "matched": False, "reason": "absent"},
                needs_replan=True,
                replan_reason="no event matches that description",
            )

        top = hits[0]
        return OpResult(
            data=jsonable(
                {
                    **{k: v for k, v in top.items() if k not in ("meta",)},
                    "candidates": hits[1:],
                    "matched": True,
                    "confident": float(top.get("cn") or 0) >= 0.8 or is_exact(top.get("evidence")),
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        text = (args or {}).get("description") or (args or {}).get("query") or ""
        return f"Finding “{excerpt_of(text, 40)}”" if text else "Finding that meeting"


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


class CreateEventArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    starts_at: Any
    ends_at: Any = None
    duration_minutes: int | None = Field(default=None, ge=5, le=24 * 60)
    description: str | None = None
    location: str | None = None
    attendees: list[str] = Field(default_factory=list)
    calendar_id: str = "primary"
    all_day: bool = False
    send_updates: str = "all"

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if isinstance(values, dict) and "attendees" in values:
            people = []
            for a in as_list(values["attendees"]):
                people.append(a.get("email") if isinstance(a, dict) else str(a))
            values = {**values, "attendees": [p for p in people if p]}
        return values

    @model_validator(mode="after")
    def _end(self) -> "CreateEventArgs":
        if self.ends_at is None and self.duration_minutes is None:
            self.duration_minutes = 30
        if str(self.send_updates) not in ("all", "externalOnly", "none"):
            raise ValueError("send_updates is all, externalOnly or none")
        return self


class CreateEvent(ConfirmableOp):
    """Put a meeting in the calendar. Prepared here, created on approval."""

    name = "gcal.create_event"
    args_model = CreateEventArgs
    output_fields = ["payload", "prepared", "starts_at", "ends_at", "title", "attendees", "clashes"]
    timeout_s = 8.0
    summary = "create a calendar event — asks first"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        start = parse_dt(parsed.starts_at)
        if start is None:
            raise AppError("VALIDATION_ERROR", "An event needs a start time.", http=422)
        end = parse_dt(parsed.ends_at) if parsed.ends_at else start + dt.timedelta(
            minutes=parsed.duration_minutes or 30
        )
        if end <= start:
            raise AppError("VALIDATION_ERROR", "That event ends before it starts.", http=422)

        # Worth knowing before the card is shown, not after the invite has gone.
        rows = await mirror_repo.list_window(ctx.session, ctx.user_id, "gcal", start, end, limit=20)
        clashes = [
            {"event_id": r.get("event_id"), "title": r.get("title"), "starts_at": r.get("starts_at")}
            for r in (row_to_dict(row) for row in rows)
            if str(r.get("status") or "").lower() != "cancelled"
        ]

        payload = jsonable(
            {
                "calendar_id": parsed.calendar_id,
                "title": parsed.title,
                "description": parsed.description,
                "location": parsed.location,
                "starts_at": start,
                "ends_at": end,
                "all_day": parsed.all_day,
                "attendees": parsed.attendees,
                "send_updates": parsed.send_updates,
            }
        )
        return OpResult(
            data={
                **payload,
                "payload": payload,
                "prepared": True,
                "clashes": clashes,
                "duration_minutes": duration_minutes(start, end),
                "preview": self.preview(payload),
                "confirm_question": self.confirm_question(payload),
            }
        )

    async def execute(self, ctx: OpContext, payload: dict) -> dict:
        created = row_to_dict(
            await google_call(
                ctx,
                "gcal",
                ("create_event", "insert_event", "create"),
                calendar_id=payload.get("calendar_id") or "primary",
                title=payload.get("title"),
                description=payload.get("description"),
                location=payload.get("location"),
                starts_at=parse_dt(payload.get("starts_at")),
                ends_at=parse_dt(payload.get("ends_at")),
                all_day=bool(payload.get("all_day")),
                attendees=as_list(payload.get("attendees")),
                send_updates=payload.get("send_updates") or "all",
            )
            or {}
        )
        row = _mirror_row_from_event(created)
        if row.get("event_id") and row.get("starts_at"):
            await mirror_repo.upsert_gcal(ctx.session, ctx.user_id, [row])
        return jsonable(
            {
                "event_id": row.get("event_id"),
                "calendar_id": row.get("calendar_id"),
                "etag": row.get("etag"),
                "html_link": created.get("htmlLink") or created.get("html_link"),
                "starts_at": row.get("starts_at"),
            }
        )

    def preview(self, payload: dict) -> dict:
        return {
            "title": payload.get("title"),
            "when": f"{payload.get('starts_at')} → {payload.get('ends_at')}",
            "where": payload.get("location"),
            "guests": as_list(payload.get("attendees")),
            "note": "Invitations go out when you say yes, not before.",
        }

    def confirm_question(self, payload: dict) -> str:
        guests = as_list(payload.get("attendees"))
        who = f" and invite {len(guests)} " + ("guest" if len(guests) == 1 else "guests") if guests else ""
        return f"Create “{payload.get('title')}”{who}?"

    def progress_label(self, args: dict) -> str:
        return f"Preparing “{excerpt_of((args or {}).get('title'), 40)}”"


class UpdateEventArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    calendar_id: str = "primary"
    etag: str | None = None
    title: str | None = None
    description: str | None = None
    location: str | None = None
    starts_at: Any = None
    ends_at: Any = None
    duration_minutes: int | None = Field(default=None, ge=5, le=24 * 60)
    attendees: list[str] | None = None
    send_updates: str = "all"
    scope: str = "single"  # single | series

    @model_validator(mode="after")
    def _something_to_change(self) -> "UpdateEventArgs":
        changed = any(
            v is not None
            for v in (self.title, self.description, self.location, self.starts_at, self.ends_at, self.attendees)
        )
        if not changed:
            raise ValueError("gcal.update_event was given nothing to change")
        if self.scope not in ("single", "series"):
            raise ValueError("scope is single or series")
        return self


class UpdateEvent(ConfirmableOp):
    """Move or edit an event.

    `run` reads the event live and pins the `etag` into the payload. That is
    the whole point: between preparing and approving, somebody else may move
    the same meeting, and `If-Match` is what turns that into a 412 we can
    re-ask about instead of a silent overwrite.
    """

    name = "gcal.update_event"
    args_model = UpdateEventArgs
    output_fields = ["payload", "prepared", "event_id", "etag", "starts_at", "ends_at", "changes"]
    timeout_s = 8.0
    summary = "move or edit an event — asks first, pins the etag"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        current = await _read_live(ctx, parsed.event_id, parsed.calendar_id)
        etag = parsed.etag or current.get("etag")
        target_id = current.get("event_id") or parsed.event_id
        if parsed.scope == "series" and current.get("recurring_event_id"):
            target_id = current["recurring_event_id"]

        start = parse_dt(parsed.starts_at) if parsed.starts_at else None
        end = parse_dt(parsed.ends_at) if parsed.ends_at else None
        if start is not None and end is None:
            minutes = parsed.duration_minutes or current.get("duration_minutes") or 30
            end = start + dt.timedelta(minutes=int(minutes))
        if start is not None and end is not None and end <= start:
            raise AppError("VALIDATION_ERROR", "That event would end before it starts.", http=422)

        changes = {
            k: v
            for k, v in {
                "title": parsed.title,
                "description": parsed.description,
                "location": parsed.location,
                "starts_at": start,
                "ends_at": end,
                "attendees": parsed.attendees,
            }.items()
            if v is not None
        }
        payload = jsonable(
            {
                "event_id": target_id,
                "calendar_id": parsed.calendar_id,
                "etag": etag,
                "scope": parsed.scope,
                "changes": changes,
                "send_updates": parsed.send_updates,
                "was": {
                    "title": current.get("title"),
                    "starts_at": current.get("starts_at"),
                    "ends_at": current.get("ends_at"),
                },
            }
        )
        return OpResult(
            data={
                **payload,
                "payload": payload,
                "prepared": True,
                "preview": self.preview(payload),
                "confirm_question": self.confirm_question(payload),
            }
        )

    async def execute(self, ctx: OpContext, payload: dict) -> dict:
        changes = dict(payload.get("changes") or {})
        updated = row_to_dict(
            await google_call(
                ctx,
                "gcal",
                ("update_event", "patch_event", "update"),
                event_id=payload.get("event_id"),
                calendar_id=payload.get("calendar_id") or "primary",
                etag=payload.get("etag"),
                patch=changes,
                title=changes.get("title"),
                description=changes.get("description"),
                location=changes.get("location"),
                starts_at=parse_dt(changes.get("starts_at")),
                ends_at=parse_dt(changes.get("ends_at")),
                attendees=as_list(changes.get("attendees")) or None,
                send_updates=payload.get("send_updates") or "all",
            )
            or {}
        )
        row = _mirror_row_from_event(updated)
        if row.get("event_id") and row.get("starts_at"):
            await mirror_repo.upsert_gcal(ctx.session, ctx.user_id, [row])
        return jsonable(
            {
                "event_id": row.get("event_id") or payload.get("event_id"),
                "etag": row.get("etag"),
                "starts_at": row.get("starts_at") or changes.get("starts_at"),
                "ends_at": row.get("ends_at") or changes.get("ends_at"),
            }
        )

    def preview(self, payload: dict) -> dict:
        changes = payload.get("changes") or {}
        was = payload.get("was") or {}
        return {
            "event": was.get("title"),
            "from": was.get("starts_at"),
            "to": changes.get("starts_at") or was.get("starts_at"),
            "changes": {k: v for k, v in changes.items() if k != "attendees"},
            "guests_notified": payload.get("send_updates") != "none",
        }

    def confirm_question(self, payload: dict) -> str:
        was = (payload.get("was") or {}).get("title") or "that event"
        changes = payload.get("changes") or {}
        if changes.get("starts_at"):
            return f"Move “{was}” to {changes['starts_at']}?"
        return f"Update “{was}”?"

    def progress_label(self, args: dict) -> str:
        return "Preparing the change to that event"


class DeleteEventArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    calendar_id: str = "primary"
    etag: str | None = None
    scope: str = "single"
    send_updates: str = "all"

    @model_validator(mode="after")
    def _scope(self) -> "DeleteEventArgs":
        if self.scope not in ("single", "series"):
            raise ValueError("scope is single or series")
        return self


class DeleteEvent(ConfirmableOp):
    """Remove an event. Prepared with a pinned etag, deleted on approval."""

    name = "gcal.delete_event"
    args_model = DeleteEventArgs
    output_fields = ["payload", "prepared", "event_id", "etag", "title", "starts_at"]
    timeout_s = 8.0
    summary = "delete an event — asks first, pins the etag"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        current = await _read_live(ctx, parsed.event_id, parsed.calendar_id)
        target_id = current.get("event_id") or parsed.event_id
        if parsed.scope == "series" and current.get("recurring_event_id"):
            target_id = current["recurring_event_id"]

        payload = jsonable(
            {
                "event_id": target_id,
                "calendar_id": parsed.calendar_id,
                "etag": parsed.etag or current.get("etag"),
                "scope": parsed.scope,
                "send_updates": parsed.send_updates,
                "title": current.get("title"),
                "starts_at": current.get("starts_at"),
                "attendee_emails": as_list(current.get("attendee_emails")),
            }
        )
        return OpResult(
            data={
                **payload,
                "payload": payload,
                "prepared": True,
                "preview": self.preview(payload),
                "confirm_question": self.confirm_question(payload),
            }
        )

    async def execute(self, ctx: OpContext, payload: dict) -> dict:
        await google_call(
            ctx,
            "gcal",
            ("delete_event", "delete"),
            event_id=payload.get("event_id"),
            calendar_id=payload.get("calendar_id") or "primary",
            etag=payload.get("etag"),
            send_updates=payload.get("send_updates") or "all",
        )
        await mirror_repo.delete_by_refs(ctx.session, ctx.user_id, "gcal", [payload.get("event_id")])
        return jsonable(
            {
                "event_id": payload.get("event_id"),
                "deleted": True,
                "title": payload.get("title"),
                "starts_at": payload.get("starts_at"),
            }
        )

    def preview(self, payload: dict) -> dict:
        guests = as_list(payload.get("attendee_emails"))
        return {
            "event": payload.get("title"),
            "when": payload.get("starts_at"),
            "guests": len(guests),
            "note": "Guests are told it was cancelled." if guests and payload.get("send_updates") != "none" else None,
        }

    def confirm_question(self, payload: dict) -> str:
        return f"Delete “{payload.get('title') or 'that event'}”?"

    def progress_label(self, args: dict) -> str:
        return "Preparing to remove that event"


async def _read_live(ctx: OpContext, event_id: str, calendar_id: str) -> dict:
    """The event as Google has it right now, falling back to the mirror.

    A pinned `etag` only means anything if it came from a live read, so this
    tries Google first and says which one answered.
    """
    if has_google(ctx, "gcal", ("get_event", "get")):
        fetched = await google_call(
            ctx, "gcal", ("get_event", "get"), event_id=event_id, calendar_id=calendar_id
        )
        if fetched:
            row = _mirror_row_from_event(row_to_dict(fetched))
            if row.get("event_id") and row.get("starts_at"):
                await mirror_repo.upsert_gcal(ctx.session, ctx.user_id, [row])
            return {**_shape_event(row), "source": "live"}

    rows = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gcal", event_id)
    if not rows:
        raise AppError(
            "NOT_FOUND",
            "That event is not on your calendar.",
            http=404,
            details={"event_id": event_id},
        )
    return {**_shape_event(row_to_dict(rows[0])), "source": "mirror"}


OPS: list[Op] = [
    SearchEvents(),
    GetEvents(),
    FindFreeSlots(),
    FindConflicts(),
    MatchEvent(),
    CreateEvent(),
    UpdateEvent(),
    DeleteEvent(),
]

__all__ = [
    "OPS",
    "CreateEvent",
    "DeleteEvent",
    "FindConflicts",
    "FindFreeSlots",
    "GetEvents",
    "MatchEvent",
    "SearchEvents",
    "UpdateEvent",
]
