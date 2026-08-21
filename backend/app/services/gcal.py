"""Google Calendar, as this system needs it.

Calendar is the service where being careful pays. Three things drive the shape
of this module:

**Sync tokens.** ``events_list`` takes either a sync token or a time range,
never both — Google rejects the combination. A token that has gone stale comes
back as a 410, which classifies as PRECONDITION, and the only honest answer is
to drop the token and resync the window.

**Etags.** Every update and delete sends ``If-Match``. Without it, "move the
meeting to Thursday" would happily overwrite a change someone else made in the
seconds since we read the event. With it, Google refuses, we refetch once, and
the retry writes against the version that actually exists.

**Recurrence.** ``recurring_event_id`` is carried through everything, because
moving one instance and moving a series are different requests and the caller
has to be able to tell which it is holding.

Times in and out are tz-aware UTC. Any date arithmetic happened in
``orchestrator/temporal.py`` long before a call reaches here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.errors import AppError
from app.core.logging import get_logger
from app.google.client import SAFE_TO_REPEAT, Transport
from app.google.retry import ErrorClass, GoogleAPIError

log = get_logger(__name__)

PRIMARY: Final[str] = "primary"

#: Calendar caps a page at 2500; this is the page the sync actually asks for.
DEFAULT_PAGE_SIZE: Final[int] = 250

#: ``sendUpdates`` values Google accepts.
SEND_UPDATES: Final[frozenset[str]] = frozenset({"all", "externalOnly", "none"})


# --------------------------------------------------------------------------- #
# Times
# --------------------------------------------------------------------------- #


def _zone(name: str | None) -> ZoneInfo:
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("gcal.unknown_timezone", timezone=name)
        return ZoneInfo("UTC")


def parse_dt(node: dict[str, Any] | None, *, fallback_tz: str | None = None) -> datetime | None:
    """One end of an event, as tz-aware UTC.

    An all-day event carries ``date`` rather than ``dateTime``, and a bare date
    is midnight *somewhere* — in the event's own zone, which is why the zone
    has to come along.
    """
    if not node:
        return None
    if node.get("dateTime"):
        text = str(node["dateTime"])
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_zone(node.get("timeZone") or fallback_tz))
        return moment.astimezone(UTC)
    if node.get("date"):
        try:
            day = date.fromisoformat(str(node["date"]))
        except ValueError:
            return None
        zone = _zone(node.get("timeZone") or fallback_tz)
        return datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)
    return None


def to_iso(moment: datetime | str | None) -> str | None:
    """RFC 3339, which is what every Calendar parameter wants."""
    if moment is None:
        return None
    if isinstance(moment, str):
        return moment
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def time_node(
    moment: datetime | date, *, timezone: str | None = None, all_day: bool = False
) -> dict[str, str]:
    """The ``start``/``end`` shape Calendar expects."""
    if all_day or (isinstance(moment, date) and not isinstance(moment, datetime)):
        day = moment.date() if isinstance(moment, datetime) else moment
        return {"date": day.isoformat()}
    node: dict[str, str] = {"dateTime": to_iso(moment) or ""}
    if timezone:
        node["timeZone"] = timezone
    return node


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_event(
    event: dict[str, Any], *, calendar_id: str = PRIMARY, fallback_tz: str | None = None
) -> dict[str, Any]:
    """One event, flattened into the shape ``sync_gcal`` stores.

    ``attendee_emails`` is deliberately absent: it is a generated column, and
    the database owns it.
    """
    start_node = event.get("start") or {}
    end_node = event.get("end") or {}
    attendees = [
        {
            "email": (a.get("email") or "").lower(),
            "name": a.get("displayName"),
            "response_status": a.get("responseStatus"),
            "optional": bool(a.get("optional", False)),
        }
        for a in event.get("attendees") or []
        if a.get("email")
    ]
    return {
        "event_id": event.get("id"),
        "calendar_id": calendar_id,
        "recurring_event_id": event.get("recurringEventId"),
        "title": event.get("summary"),
        "description": event.get("description") or None,
        "location": event.get("location") or None,
        "organizer_email": ((event.get("organizer") or {}).get("email") or "").lower() or None,
        "organizer_self": bool((event.get("organizer") or {}).get("self", False)),
        "attendees": attendees,
        "starts_at": parse_dt(start_node, fallback_tz=fallback_tz),
        "ends_at": parse_dt(end_node, fallback_tz=fallback_tz),
        "all_day": "date" in start_node,
        "event_timezone": start_node.get("timeZone") or fallback_tz,
        "status": event.get("status") or "confirmed",
        "etag": event.get("etag"),
        "html_link": event.get("htmlLink"),
        "hangout_link": event.get("hangoutLink"),
        "conference": (event.get("conferenceData") or {}).get("entryPoints"),
        "sequence": event.get("sequence"),
        "updated_at": parse_dt({"dateTime": event.get("updated")}) if event.get("updated") else None,
        "creator_email": ((event.get("creator") or {}).get("email") or "").lower() or None,
        "guests_can_modify": bool(event.get("guestsCanModify", False)),
        "recurrence": event.get("recurrence"),
    }


def build_event_body(
    *,
    title: str | None = None,
    start: datetime | date | None = None,
    end: datetime | date | None = None,
    timezone: str | None = None,
    all_day: bool = False,
    description: str | None = None,
    location: str | None = None,
    attendees: Sequence[str | dict[str, Any]] | None = None,
    recurrence: Sequence[str] | None = None,
    reminders: dict[str, Any] | None = None,
    conference_request_id: str | None = None,
) -> dict[str, Any]:
    """A Calendar event body from plain arguments.

    Only the fields given are included, so the same builder serves ``insert``
    (everything) and ``patch`` (one field), and a patch never blanks something
    it was not asked about.
    """
    body: dict[str, Any] = {}
    if title is not None:
        body["summary"] = title
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    if start is not None:
        body["start"] = time_node(start, timezone=timezone, all_day=all_day)
    if end is not None:
        body["end"] = time_node(end, timezone=timezone, all_day=all_day)
    if attendees is not None:
        body["attendees"] = [
            {"email": a.lower()} if isinstance(a, str) else a for a in attendees if a
        ]
    if recurrence is not None:
        body["recurrence"] = list(recurrence)
    if reminders is not None:
        body["reminders"] = reminders
    if conference_request_id:
        body["conferenceData"] = {
            "createRequest": {
                "requestId": conference_request_id,
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }
    return body


class SyncTokenExpired(Exception):
    """The sync token is older than Calendar keeps. Resync the window."""

    def __init__(self, calendar_id: str) -> None:
        self.calendar_id = calendar_id
        super().__init__(f"Calendar sync token for {calendar_id} is no longer valid.")


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class CalendarService:
    """Calendar bound to one user."""

    service = "gcal"

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    @property
    def user_id(self) -> str:
        return self.transport.user_id

    # -- reading ------------------------------------------------------------ #

    async def events_list(
        self,
        *,
        calendar_id: str = PRIMARY,
        sync_token: str | None = None,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        q: str | None = None,
        page_token: str | None = None,
        max_results: int = DEFAULT_PAGE_SIZE,
        single_events: bool = True,
        show_deleted: bool | None = None,
        order_by: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Events, either incrementally by sync token or over a time range.

        With a sync token, the range parameters are dropped — Calendar returns
        400 if both are sent — and deleted events come back as ``cancelled``,
        which is the only way an incremental sync learns something is gone.
        """
        params: dict[str, Any] = {
            "pageToken": page_token,
            "maxResults": max(1, min(int(max_results), 2500)),
            "singleEvents": single_events,
        }
        if sync_token:
            params["syncToken"] = sync_token
            params["showDeleted"] = True if show_deleted is None else show_deleted
        else:
            params["timeMin"] = to_iso(time_min)
            params["timeMax"] = to_iso(time_max)
            params["q"] = q or None
            params["showDeleted"] = False if show_deleted is None else show_deleted
            params["orderBy"] = order_by or ("startTime" if single_events else None)
            params["timeZone"] = timezone

        try:
            page = await self.transport.get(
                f"calendars/{_esc(calendar_id)}/events",
                api_method="gcal.events.list",
                params=params,
            )
        except GoogleAPIError as exc:
            if sync_token and exc.error_class is ErrorClass.PRECONDITION:
                log.info("gcal.sync_token_expired", calendar_id=calendar_id)
                raise SyncTokenExpired(calendar_id) from exc
            raise

        fallback_tz = page.get("timeZone") or timezone
        items = page.get("items") or []
        events = [
            parse_event(item, calendar_id=calendar_id, fallback_tz=fallback_tz)
            for item in items
        ]
        return {
            "events": [e for e in events if e["status"] != "cancelled"],
            "cancelled": [e["event_id"] for e in events if e["status"] == "cancelled"],
            "next_page_token": page.get("nextPageToken"),
            "next_sync_token": page.get("nextSyncToken"),
            "calendar_timezone": fallback_tz,
            "calendar_id": calendar_id,
        }

    async def events_get(
        self, event_id: str, *, calendar_id: str = PRIMARY
    ) -> dict[str, Any] | None:
        """One event. ``None`` when it is not there any more."""
        try:
            raw = await self.transport.get(
                f"calendars/{_esc(calendar_id)}/events/{_esc(event_id)}",
                api_method="gcal.events.get",
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return None
            raise
        return parse_event(raw, calendar_id=calendar_id)

    async def freebusy_query(
        self,
        time_min: datetime,
        time_max: datetime,
        *,
        calendar_ids: Sequence[str] = (PRIMARY,),
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Busy blocks per calendar, plus the gaps between them.

        Free/busy is one call for many calendars, which is what makes "when are
        we all free on Thursday" cheap. The free windows are computed here, in
        Python, from the busy ones.
        """
        result = await self.transport.post(
            "freeBusy",
            api_method="gcal.freebusy.query",
            json={
                "timeMin": to_iso(time_min),
                "timeMax": to_iso(time_max),
                "timeZone": timezone or "UTC",
                "items": [{"id": c} for c in calendar_ids],
            },
        )

        calendars: dict[str, list[dict[str, datetime]]] = {}
        errors: dict[str, Any] = {}
        merged: list[tuple[datetime, datetime]] = []
        for name, node in (result.get("calendars") or {}).items():
            if node.get("errors"):
                errors[name] = node["errors"]
            blocks = []
            for block in node.get("busy") or []:
                start = parse_dt({"dateTime": block.get("start")})
                end = parse_dt({"dateTime": block.get("end")})
                if start and end:
                    blocks.append({"start": start, "end": end})
                    merged.append((start, end))
            calendars[name] = blocks

        return {
            "time_min": time_min,
            "time_max": time_max,
            "calendars": calendars,
            "busy": [{"start": s, "end": e} for s, e in _merge(merged)],
            "free": [
                {"start": s, "end": e}
                for s, e in _gaps(_merge(merged), time_min, time_max)
            ],
            "errors": errors,
        }

    async def settings_get(self, setting: str = "timezone") -> str | None:
        """One Calendar setting. ``timezone`` is the one that matters: it is the
        zone every "tomorrow at 3" is resolved against."""
        try:
            result = await self.transport.get(
                f"users/me/settings/{_esc(setting)}",
                api_method="gcal.settings.get",
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return None
            raise
        value = result.get("value")
        return str(value) if value is not None else None

    async def calendar_list(self) -> list[dict[str, Any]]:
        """The user's calendars, primary first."""
        page = await self.transport.get(
            "users/me/calendarList", api_method="gcal.calendarList.list"
        )
        items = [
            {
                "id": item.get("id"),
                "summary": item.get("summary"),
                "primary": bool(item.get("primary", False)),
                "access_role": item.get("accessRole"),
                "timezone": item.get("timeZone"),
                "selected": bool(item.get("selected", False)),
            }
            for item in page.get("items") or []
        ]
        return sorted(items, key=lambda c: (not c["primary"], c["summary"] or ""))

    # -- writing ------------------------------------------------------------ #

    async def events_insert(
        self,
        body: dict[str, Any],
        *,
        calendar_id: str = PRIMARY,
        request_id: str | None = None,
        send_updates: str = "none",
        conference: bool = False,
    ) -> dict[str, Any]:
        """Create an event.

        ``request_id`` is our idempotency key. It is sent as a query parameter
        and, when a Meet link is asked for, reused as the conference
        ``requestId`` — Calendar's own de-duplication field. Same key on a
        retry, same conference, not a second one.
        """
        payload = dict(body)
        if conference and "conferenceData" not in payload and request_id:
            payload["conferenceData"] = {
                "createRequest": {
                    "requestId": request_id,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        created = await self.transport.post(
            f"calendars/{_esc(calendar_id)}/events",
            api_method="gcal.events.insert",
            params={
                "requestId": request_id,
                "sendUpdates": send_updates if send_updates in SEND_UPDATES else "none",
                "conferenceDataVersion": 1 if payload.get("conferenceData") else None,
            },
            json=payload,
            # A create that may have landed is not repeated blind; requestId
            # covers the rest.
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )
        return parse_event(created, calendar_id=calendar_id)

    async def events_patch(
        self,
        event_id: str,
        patch: dict[str, Any],
        *,
        calendar_id: str = PRIMARY,
        etag: str | None = None,
        send_updates: str = "none",
    ) -> dict[str, Any]:
        """Change some fields of an event, guarded by its etag.

        If Google says the etag is stale, the event is read once more and the
        same patch is applied to the version that exists now. That is safe
        precisely because this is a PATCH: it touches the fields we were asked
        to change and nothing else.
        """

        async def refetch() -> dict[str, Any]:
            current = await self.events_get(event_id, calendar_id=calendar_id)
            if current is None or not current.get("etag"):
                # Nothing to match against any more; let the write through.
                return {"headers": {"If-Match": "*"}}
            log.info(
                "gcal.etag_refetched",
                event_id=event_id,
                calendar_id=calendar_id,
                etag=current["etag"],
            )
            return {"headers": {"If-Match": current["etag"]}}

        updated = await self.transport.patch(
            f"calendars/{_esc(calendar_id)}/events/{_esc(event_id)}",
            api_method="gcal.events.patch",
            params={
                "sendUpdates": send_updates if send_updates in SEND_UPDATES else "none",
                "conferenceDataVersion": 1 if patch.get("conferenceData") else None,
            },
            json=patch,
            headers={"If-Match": etag} if etag else None,
            refetch=refetch if etag else None,
        )
        return parse_event(updated, calendar_id=calendar_id)

    async def events_delete(
        self,
        event_id: str,
        *,
        calendar_id: str = PRIMARY,
        etag: str | None = None,
        send_updates: str = "none",
    ) -> dict[str, Any]:
        """Delete an event, guarded by its etag.

        Already gone is success. Deleting a deleted event is the outcome the
        caller asked for, and reporting it as a failure would only produce a
        confusing card.
        """

        async def refetch() -> dict[str, Any]:
            current = await self.events_get(event_id, calendar_id=calendar_id)
            if current is None or not current.get("etag"):
                return {"headers": {"If-Match": "*"}}
            return {"headers": {"If-Match": current["etag"]}}

        try:
            await self.transport.delete(
                f"calendars/{_esc(calendar_id)}/events/{_esc(event_id)}",
                api_method="gcal.events.delete",
                params={
                    "sendUpdates": send_updates if send_updates in SEND_UPDATES else "none"
                },
                headers={"If-Match": etag} if etag else None,
                refetch=refetch if etag else None,
                expect="none",
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return {
                    "event_id": event_id,
                    "calendar_id": calendar_id,
                    "deleted": True,
                    "already_gone": True,
                }
            raise
        return {
            "event_id": event_id,
            "calendar_id": calendar_id,
            "deleted": True,
            "already_gone": False,
        }

    # -- the shape the ops layer asks for ----------------------------------- #
    #
    # Ops speak in plain arguments — title, starts_at, attendees — because that
    # is what a plan step carries. These three turn that into the bodies the
    # methods above want, and hand back a parsed event either way.

    async def list_events(
        self,
        *,
        query: str | None = None,
        calendar_id: str = PRIMARY,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int = 25,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Events matching free text over a window, as a plain list.

        The convenience shape the ops call, matching `GDriveService.list_files`:
        one page, already parsed, cancelled events dropped. `events_list` is the
        raw paging call underneath, and is what the sync task walks.
        """
        page = await self.events_list(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
            q=(query or None),
            max_results=max_results,
            order_by=order_by,
        )
        return list(page.get("events") or [])

    async def get_event(
        self, event_id: str, *, calendar_id: str = PRIMARY
    ) -> dict[str, Any] | None:
        return await self.events_get(event_id, calendar_id=calendar_id)

    async def create_event(
        self,
        *,
        calendar_id: str = PRIMARY,
        title: str | None = None,
        description: str | None = None,
        location: str | None = None,
        starts_at: datetime | date | None = None,
        ends_at: datetime | date | None = None,
        all_day: bool = False,
        attendees: Sequence[str | dict[str, Any]] | None = None,
        timezone: str | None = None,
        recurrence: Sequence[str] | None = None,
        send_updates: str = "none",
        request_id: str | None = None,
        conference: bool = False,
    ) -> dict[str, Any]:
        """Create an event from plain arguments."""
        body = build_event_body(
            title=title,
            description=description,
            location=location,
            start=starts_at,
            end=ends_at,
            all_day=all_day,
            attendees=attendees,
            timezone=timezone,
            recurrence=recurrence,
        )
        return await self.events_insert(
            body,
            calendar_id=calendar_id,
            request_id=request_id,
            send_updates=send_updates,
            conference=conference,
        )

    async def update_event(
        self,
        event_id: str,
        *,
        calendar_id: str = PRIMARY,
        etag: str | None = None,
        patch: dict[str, Any] | None = None,
        title: str | None = None,
        description: str | None = None,
        location: str | None = None,
        starts_at: datetime | date | None = None,
        ends_at: datetime | date | None = None,
        all_day: bool = False,
        attendees: Sequence[str | dict[str, Any]] | None = None,
        timezone: str | None = None,
        send_updates: str = "none",
    ) -> dict[str, Any]:
        """Change an event, from either a ready-made patch or plain arguments.

        A ``patch`` that already speaks Calendar's language is passed through;
        anything given as an argument wins over the same key inside it, because
        the argument is the more specific thing the caller said.
        """
        body = dict(patch or {})
        # A patch written in our own vocabulary, as a plan step would send it.
        for ours, theirs in (
            ("title", "summary"),
            ("description", "description"),
            ("location", "location"),
        ):
            if ours in body and theirs not in body:
                body[theirs] = body.pop(ours)
        for key in ("starts_at", "ends_at", "attendees", "all_day", "timezone"):
            body.pop(key, None)

        body.update(
            build_event_body(
                title=title,
                description=description,
                location=location,
                start=starts_at,
                end=ends_at,
                all_day=all_day,
                attendees=attendees,
                timezone=timezone,
            )
        )
        return await self.events_patch(
            event_id,
            body,
            calendar_id=calendar_id,
            etag=etag,
            send_updates=send_updates,
        )

    async def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str = PRIMARY,
        etag: str | None = None,
        send_updates: str = "none",
    ) -> dict[str, Any]:
        return await self.events_delete(
            event_id, calendar_id=calendar_id, etag=etag, send_updates=send_updates
        )

    async def events_move(
        self,
        event_id: str,
        *,
        destination: str,
        calendar_id: str = PRIMARY,
        send_updates: str = "none",
    ) -> dict[str, Any]:
        """Move an event to another calendar."""
        moved = await self.transport.post(
            f"calendars/{_esc(calendar_id)}/events/{_esc(event_id)}/move",
            api_method="gcal.events.move",
            params={
                "destination": destination,
                "sendUpdates": send_updates if send_updates in SEND_UPDATES else "none",
            },
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )
        return parse_event(moved, calendar_id=destination)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _esc(value: str) -> str:
    """Path-escape a calendar or event id. Addresses contain '@' and '+'."""
    return quote(str(value), safe="")


def _merge(blocks: Sequence[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Overlapping busy blocks collapsed into one list."""
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: b[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _gaps(
    busy: Sequence[tuple[datetime, datetime]],
    time_min: datetime,
    time_max: datetime,
    *,
    minimum: timedelta = timedelta(minutes=15),
) -> list[tuple[datetime, datetime]]:
    """The free windows between busy blocks, ignoring slivers."""
    free: list[tuple[datetime, datetime]] = []
    cursor = time_min
    for start, end in busy:
        if start > cursor and start - cursor >= minimum:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if time_max > cursor and time_max - cursor >= minimum:
        free.append((cursor, time_max))
    return free


# --------------------------------------------------------------------------- #
# Raw-payload functions, for the sync and action workers
# --------------------------------------------------------------------------- #
#
# The sync task pages through events itself, hashes each one and decides what
# to upsert, so it wants Google's own payload — ``items``, ``nextPageToken``,
# ``nextSyncToken`` — and calls ``normalise_event`` when it is ready.


def _transport(clients: Any) -> Transport:
    """The Calendar transport out of whatever the caller is holding."""
    if isinstance(clients, Transport):
        return clients
    service = getattr(clients, "gcal", None) or getattr(clients, "calendar", clients)
    transport = getattr(service, "transport", None)
    if isinstance(transport, Transport):
        return transport
    raise AppError.internal(
        "Expected Google clients with a calendar transport.",
        got=type(clients).__name__,
    )


#: The name the sync task uses for :func:`parse_event`.
normalise_event = parse_event


async def events_list(
    clients: Any,
    *,
    calendar_id: str = PRIMARY,
    sync_token: str | None = None,
    page_token: str | None = None,
    time_min: datetime | None = None,
    time_max: datetime | None = None,
    q: str | None = None,
    max_results: int = DEFAULT_PAGE_SIZE,
    single_events: bool = True,
    show_deleted: bool = False,
    order_by: str | None = None,
) -> dict[str, Any]:
    """Raw ``events.list`` page.

    A sync token and a time range cannot travel together, so when a token is
    given the range is dropped rather than sent and rejected.
    """
    params: dict[str, Any] = {
        "pageToken": page_token,
        "maxResults": max(1, min(int(max_results), 2500)),
        "singleEvents": single_events,
        "showDeleted": show_deleted,
    }
    if sync_token:
        params["syncToken"] = sync_token
    else:
        params["timeMin"] = to_iso(time_min)
        params["timeMax"] = to_iso(time_max)
        params["q"] = q or None
        params["orderBy"] = order_by or ("startTime" if single_events else None)
    return await _transport(clients).get(
        f"calendars/{_esc(calendar_id)}/events",
        api_method="gcal.events.list",
        params=params,
    )


async def events_get(
    clients: Any, event_id: str, *, calendar_id: str = PRIMARY
) -> dict[str, Any] | None:
    """Raw event, or ``None`` when it has gone.

    The action worker compares ``etag`` against what the person approved, so
    this deliberately returns Google's own fields rather than our flattened
    ones.
    """
    try:
        return await _transport(clients).get(
            f"calendars/{_esc(calendar_id)}/events/{_esc(event_id)}",
            api_method="gcal.events.get",
        )
    except GoogleAPIError as exc:
        if exc.error_class is ErrorClass.NOT_FOUND:
            return None
        raise


__all__ = [
    "CalendarService",
    "PRIMARY",
    "events_get",
    "events_list",
    "normalise_event",
    "SyncTokenExpired",
    "build_event_body",
    "parse_dt",
    "parse_event",
    "time_node",
    "to_iso",
]
