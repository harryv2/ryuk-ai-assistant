"""The last stage: turn what happened into what the person reads.

Three routes, and two of them cost nothing:

* ``card`` — a write is staged. A templated preamble plus the confirm card,
  built from ``Op.preview(payload)``. **0 LLM calls.**
* ``template:<name>`` — a list is the answer, so a renderer writes it. Ten of
  them, all pure functions. **0 LLM calls.**
* ``prose`` — the value is in the synthesis: a briefing, a digest, a comparison.
  One streamed call, and the only one this module ever makes.

After a prose answer there is a deterministic post-check. If a service failed
and the model did not name it, the banner is **prepended** — not appended, and
not left to the model's judgement. A degraded answer that does not admit it is
degraded is worse than an error, and that property should not depend on a
sampling temperature.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Final, Iterable, Sequence
from zoneinfo import ZoneInfo

from app.core.logging import get_logger
from app.orchestrator import widgets
from app.ops.base import service_of

log = get_logger(__name__)

TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "event_list",
        "email_list",
        "file_list",
        "free_slots",
        "conflict_list",
        "summary_list",
        "count_answer",
        "empty_result",
        "degraded_banner",
        "reconnect_card",
    }
)

SERVICE_NAMES: Final[dict[str, str]] = {
    "gmail": "Gmail",
    "gcal": "Calendar",
    "gdrive": "Drive",
}

# What "nothing matched" is about, per renderer, so the sentence is specific.
_EMPTY_NOUN: Final[dict[str, str]] = {
    "event_list": "events",
    "email_list": "emails",
    "file_list": "files",
    "conflict_list": "clashes",
    "summary_list": "results",
}

WORKDAY_START: Final[int] = 9
WORKDAY_END: Final[int] = 18
MIN_SLOT_MINUTES: Final[int] = 30


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Rendered:
    """The assistant message, ready to store and to stream."""

    style: str
    text: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    degraded_prepended: bool = False

    def with_refs(
        self, actions: Sequence[str] = (), inputs: Sequence[str] = ()
    ) -> "Rendered":
        """Append the action and input blocks, in the order they are shown."""
        for action_id in actions:
            self.blocks.append({"type": "action", "ref": action_id})
        for input_id in inputs:
            self.blocks.append({"type": "input", "ref": input_id})
        return self


def _text_block(markdown: str) -> dict[str, Any]:
    return {"type": "text", "data": {"markdown": markdown}}


# ---------------------------------------------------------------------------
# Reading results without knowing exactly which op wrote them
# ---------------------------------------------------------------------------

_LIST_KEYS: Final[tuple[str, ...]] = (
    "hits", "results", "items", "events", "messages", "files", "slots", "conflicts"
)


def _get(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict):
            if item.get(name) not in (None, ""):
                return item[name]
        else:
            value = getattr(item, name, None)
            if value not in (None, ""):
                return value
    return default


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in _LIST_KEYS:
            found = value.get(key)
            if isinstance(found, list):
                return found
        if any(value.get(k) for k in ("message_id", "event_id", "file_id")):
            return [value]
    return []


def collect(results: dict[str, Any], *, node_ids: Iterable[str] | None = None) -> list[Any]:
    """Every item every step produced, in step order."""
    out: list[Any] = []
    for node_id, result in results.items():
        if node_ids is not None and node_id not in node_ids:
            continue
        out.extend(_rows(result))
    return out


def _moment(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _zone(tz: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz or "UTC")
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def _day(moment: datetime, tz: str) -> str:
    local = moment.astimezone(_zone(tz))
    return f"{local:%a} {local:%b} {local.day}"


def _clock(moment: datetime, tz: str) -> str:
    local = moment.astimezone(_zone(tz))
    return f"{local:%H:%M}"


def _friendly(moment: datetime, tz: str) -> str:
    local = moment.astimezone(_zone(tz))
    hour = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    return f"{local:%a} {local:%b} {local.day}, {hour}:{local:%M} {suffix}"


def _short_date(value: Any, tz: str) -> str:
    moment = _moment(value)
    if moment is None:
        return ""
    local = moment.astimezone(_zone(tz))
    return f"{local:%b} {local.day:>2}"


def _excerpt(text: Any, width: int = 60) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return ""
    return body if len(body) <= width else body[: width - 1] + "…"


def _plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else (many or one + 's')}"


# ---------------------------------------------------------------------------
# Templates — every one of these is a pure function and costs nothing
# ---------------------------------------------------------------------------


def event_list(
    items: Sequence[Any],
    *,
    tz: str = "UTC",
    window: dict[str, Any] | None = None,
    footer: str = "",
    attendee: str | None = None,
) -> str:
    if not items:
        return empty_result("events", window=window, tz=tz)

    events = []
    for item in items:
        start = _moment(_get(item, "starts_at", "start", "start_time"))
        if start is None:
            continue
        events.append((start, item))
    events.sort(key=lambda pair: pair[0])
    if not events:
        return empty_result("events", window=window, tz=tz)

    lines: list[str] = []
    header = _window_header(window, tz)
    if header:
        suffix = f" — {_plural(len(events), 'event')}"
        if attendee:
            suffix += f" with {attendee}"
        lines.append(f"**{header}**{suffix}")
        lines.append("")

    current_day = None
    for start, item in events:
        day = _day(start, tz)
        if day != current_day:
            if current_day is not None:
                lines.append("")
            lines.append(f"**{day}**")
            current_day = day

        end = _moment(_get(item, "ends_at", "end", "end_time"))
        if _get(item, "all_day"):
            when = "All day"
        elif end is not None:
            when = f"{_clock(start, tz)}–{_clock(end, tz)}"
        else:
            when = _clock(start, tz)

        title = _get(item, "title", "summary", "name", default="(no title)")
        # Padded outside the code span: trailing spaces inside one are visible.
        row = f"- `{when}`{' ' * max(1, 14 - len(when))}{title}"

        guests = _get(item, "attendees", "attendee_emails", default=[])
        if guests:
            row += f"  · {_plural(len(guests), 'guest')}"
        location = _get(item, "location")
        if location:
            row += f"  · {_excerpt(location, 30)}"
        status = _get(item, "response_status")
        if status:
            row += f"  · you: {status.replace('_', ' ')}"
        lines.append(row)

    # The header already carries the count. Repeating it as a footer is the
    # same sentence twice, which reads like the answer lost its place.
    tail = footer or ("" if header else _plural(len(events), "event") + ".")
    if tail:
        lines += ["", tail]
    return "\n".join(lines).strip()


def email_list(
    items: Sequence[Any],
    *,
    tz: str = "UTC",
    window: dict[str, Any] | None = None,
    footer: str = "",
) -> str:
    if not items:
        return empty_result("emails", window=window, tz=tz)

    lines: list[str] = [f"**{_plural(len(items), 'email')}**", ""]
    for item in items:
        when = _short_date(_get(item, "received_at", "date", "internal_date"), tz)
        subject = _get(item, "subject", "title", default="(no subject)")
        sender = _get(item, "from_name", "from_email", "sender", default="")
        snippet = _excerpt(_get(item, "snippet", "body_excerpt", "body_clean", default=""), 58)
        row = f"- `{when:>6}`  **{_excerpt(subject, 60)}**"
        if sender:
            row += f" — {sender}"
        if snippet:
            row += f"\n   \"{snippet}\""
        lines.append(row)

    lines.append("")
    lines.append(footer or _searched_line(window, tz))
    return "\n".join(lines).strip()


def file_list(
    items: Sequence[Any],
    *,
    tz: str = "UTC",
    window: dict[str, Any] | None = None,
    footer: str = "",
) -> str:
    if not items:
        return empty_result("files", window=window, tz=tz)

    lines: list[str] = [f"**{_plural(len(items), 'file')}**", ""]
    for item in items:
        name = _get(item, "name", "title", "filename", default="(untitled)")
        link = _get(item, "web_view_link", "link", "url")
        modified = _short_date(_get(item, "modified_at", "modified"), tz)
        owner = _get(item, "owner_email", "owner", default="")
        label = f"[{name}]({link})" if link else f"**{name}**"
        row = f"- {label}"
        if modified:
            row += f"  · changed {modified}"
        if owner:
            row += f" by {owner}"
        size = _get(item, "size_bytes")
        if size:
            row += f"  · {_size(size)}"
        lines.append(row)

    lines.append("")
    lines.append(footer or _searched_line(window, tz))
    return "\n".join(lines).strip()


def _size(size_bytes: Any) -> str:
    try:
        value = float(size_bytes)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


def free_slots(
    items: Sequence[Any],
    *,
    tz: str = "UTC",
    window: dict[str, Any] | None = None,
    workday: tuple[int, int] = (WORKDAY_START, WORKDAY_END),
    footer: str = "",
) -> str:
    """The gaps between meetings, inside working hours, in local time."""
    start = _moment((window or {}).get("start"))
    end = _moment((window or {}).get("end"))
    if start is None or end is None:
        return "I need a date range to work out when you are free."

    zone = _zone(tz)
    busy: list[tuple[datetime, datetime]] = []
    for item in items:
        begins = _moment(_get(item, "starts_at", "start"))
        finishes = _moment(_get(item, "ends_at", "end"))
        if begins is None:
            continue
        if finishes is None:
            finishes = begins + timedelta(minutes=30)
        if _get(item, "all_day"):
            continue
        busy.append((begins, finishes))
    busy.sort()

    lines: list[str] = []
    header = _window_header(window, tz)
    if header:
        lines.append(f"**Free time — {header}**")
        lines.append("")

    day = start.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    total = 0
    while day < end.astimezone(zone):
        open_at = day.replace(hour=workday[0])
        close_at = day.replace(hour=workday[1])
        cursor = max(open_at, start.astimezone(zone))
        gaps: list[tuple[datetime, datetime]] = []
        for begins, finishes in busy:
            begins_local, finishes_local = begins.astimezone(zone), finishes.astimezone(zone)
            if finishes_local <= cursor or begins_local >= close_at:
                continue
            if begins_local - cursor >= timedelta(minutes=MIN_SLOT_MINUTES):
                gaps.append((cursor, begins_local))
            cursor = max(cursor, finishes_local)
        if close_at - cursor >= timedelta(minutes=MIN_SLOT_MINUTES) and cursor < close_at:
            gaps.append((cursor, close_at))

        if gaps and day.weekday() < 5:
            lines.append(f"**{_day(day, tz)}**")
            for gap_start, gap_end in gaps:
                minutes = int((gap_end - gap_start).total_seconds() // 60)
                lines.append(
                    f"- `{_clock(gap_start, tz)}–{_clock(gap_end, tz)}`  "
                    f"{minutes // 60}h {minutes % 60:02d}m"
                )
                total += minutes
            lines.append("")
        day += timedelta(days=1)

    if total == 0:
        return "\n".join(lines + ["Nothing free in working hours in that stretch."]).strip()
    lines.append(footer or f"{total // 60}h {total % 60:02d}m free in working hours.")
    return "\n".join(lines).strip()


def conflict_list(
    conflicts: Sequence[Any],
    *,
    tz: str = "UTC",
    window: dict[str, Any] | None = None,
    footer: str = "",
) -> str:
    """Overlaps, stated as the pair that overlaps and by how much."""
    if not conflicts:
        return "Nothing clashes in that stretch."

    lines: list[str] = [f"**{_plural(len(conflicts), 'clash', 'clashes')}**", ""]
    for conflict in conflicts:
        left = _get(conflict, "a", "left", "event", default=conflict)
        right = _get(conflict, "b", "right", "against", default=None)
        left_start = _moment(_get(left, "starts_at", "start"))
        line = f"- **{_get(left, 'title', 'summary', 'name', default='(no title)')}**"
        if left_start:
            line += f" — {_friendly(left_start, tz)}"
        lines.append(line)
        if right is not None:
            right_start = _moment(_get(right, "starts_at", "start"))
            against = f"   clashes with **{_get(right, 'title', 'summary', 'name', default='(no title)')}**"
            if right_start:
                against += f" at {_clock(right_start, tz)}"
            lines.append(against)
        reason = _get(conflict, "reason", "why")
        if reason:
            lines.append(f"   {reason}")

    tail = footer or ""
    if tail:
        lines += ["", tail]
    return "\n".join(lines).strip()


def summary_list(
    items: Sequence[Any],
    *,
    tz: str = "UTC",
    title: str = "",
    footer: str = "",
) -> str:
    """The fallback shape: whatever we have, as a readable list."""
    if not items:
        return empty_result("results", tz=tz)

    lines: list[str] = []
    if title:
        lines += [f"**{title}**", ""]
    for item in items:
        label = _get(
            item, "title", "subject", "name", "summary", "label", "text", default=None
        )
        if label is None:
            label = _excerpt(item, 90)
        line = f"- {_excerpt(label, 90)}"
        when = _moment(_get(item, "starts_at", "received_at", "modified_at", "date"))
        if when:
            line += f"  · {_short_date(when, tz)}"
        lines.append(line)
    if footer:
        lines += ["", footer]
    return "\n".join(lines).strip()


def count_answer(
    count: int,
    *,
    noun: str = "result",
    window: dict[str, Any] | None = None,
    tz: str = "UTC",
    detail: str = "",
) -> str:
    header = _window_header(window, tz)
    sentence = f"**{_plural(count, noun)}**"
    if header:
        # Not lowercased: the header is always a date label — "Mon Aug 24", or
        # "Mon Aug 24 to Mon Aug 31" — and "mon aug 24" reads like a typo.
        sentence += f" — {header}"
    sentence += "."
    if detail:
        sentence += f"\n\n{detail}"
    # The count sentence already carries the date label, so restating the
    # window under it would be the same fact twice.
    return sentence


def empty_result(
    noun: str = "results",
    *,
    window: dict[str, Any] | None = None,
    tz: str = "UTC",
    searched: str = "",
    suggestion: str = "",
) -> str:
    """Absence, reported as absence — with what was looked at, and a way forward."""
    lines = [f"No {noun} matched."]
    detail = searched or _searched_line(window, tz)
    if detail:
        lines.append("")
        lines.append(detail)
    if suggestion:
        lines.append("")
        lines.append(suggestion)
    return "\n".join(lines)


def _ago(seconds: Any) -> str:
    """A lag in words. Used for staleness, which is not a failure."""
    try:
        n = int(seconds or 0)
    except (TypeError, ValueError):
        return "a while ago"
    if n < 90:
        return "moments ago"
    if n < 3600:
        return f"{n // 60} minutes ago"
    if n < 86400:
        hours = n // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = n // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def degraded_banner(failures: Sequence[dict[str, Any]], *, skipped: Sequence[dict[str, Any]] = ()) -> str:
    """One sentence naming what went wrong and what it cost. Goes first, always.

    Three different things end up in ``degraded`` and they must not be said the
    same way:

    * a **service failure** — Google returned 503, the step really did not run
    * a **plan fault** — we built a step we could not run. Ours, not Google's
    * **staleness** — every step succeeded, the copy is simply a little old

    Calling the third one "Gmail is not responding" is the worst of the three,
    because it is said on an answer that is complete and correct. It sends the
    reader to a status page to investigate a service that is fine, and it
    teaches them to distrust answers that are actually good.
    """
    if not failures:
        return ""

    STALE = {"stale", "staleness", "lagging"}

    outages: list[str] = []
    ours = False
    stale: list[tuple[str, Any]] = []

    for failure in failures:
        reason = str(failure.get("reason") or "")
        service_key = str(failure.get("service") or "")
        name = SERVICE_NAMES.get(service_key, service_key or "a service")

        # Every step worked; the mirror is just behind. Not a failure.
        if reason in STALE:
            stale.append((name, failure.get("lag_seconds") or failure.get("lag")))
            continue

        # Ours. Own it rather than pointing at Google.
        if failure.get("fault") == "plan" or not service_key:
            ours = True
            continue

        # A step that never ran because the one before it failed is not an
        # outage. The "anything that depended on it" line below already says
        # what it cost, and naming the service here would blame Google for a
        # call it was never asked to make.
        if failure.get("fault") == "dependency":
            continue

        # Nothing below this point may name a service unless the failure was
        # genuinely the service's. Reaching the outage wording by default —
        # because `service` happened to be set — is how "Gmail is not
        # responding" ends up over an answer Gmail supplied correctly.
        if failure.get("fault") not in (None, "service"):
            ours = True
            continue

        if reason == "never_synced":
            outages.append(f"{name} has not been mirrored yet")
            continue

        # A revoked or expired grant is not an outage. Google is answering
        # perfectly well — it is declining us, which is a different problem
        # with a different fix, and "not responding" sends the reader to a
        # status page instead of to the reconnect button.
        if str(failure.get("class") or "").upper() in {"AUTH_EXPIRED", "AUTH_REVOKED"}:
            outages.append(
                f"{name} is not connected — reconnect Google and ask again"
            )
            continue

        detail = f"{name} is not responding"
        if reason == "circuit_open":
            detail = f"{name} is failing repeatedly, so it was not called"
        code = failure.get("code")
        if code:
            detail += f" — Google returned {code}"
        attempts = failure.get("attempts")
        if attempts and int(attempts) > 1:
            detail += f" on {attempts} attempts"
        outages.append(detail)

    parts: list[str] = []
    # Dedupe: three steps failing on one service is one outage, not three.
    for detail in dict.fromkeys(outages):
        parts.append(detail)
    if ours:
        parts.append(
            "I built a step I could not run — that is a fault on our side, "
            "not with Google. Anything it would have found is missing here"
        )

    banner = ""
    if parts:
        banner = "; ".join(parts) + "."
        if skipped:
            # Named the blocked steps here, which meant printing plan node ids
            # — "send_cancellation" — at the person whose mail this is. The id
            # tells them nothing; the consequence is the whole message.
            banner += " Anything that depended on it is missing here."

    if stale:
        worst = max(stale, key=lambda item: int(item[1] or 0))
        note = f"_Read from a copy last synced {_ago(worst[1])}._"
        banner = f"{banner}\n\n{note}" if banner else note

    return banner


def reconnect_card(service: str = "google") -> str:
    name = SERVICE_NAMES.get(service, "Google")
    return (
        f"**Your {name} connection needs renewing.**\n\n"
        "The access token was revoked or has expired, so I cannot read or change "
        "anything until it is reconnected. Nothing was lost — reconnect and ask again."
    )



# ---------------------------------------------------------------------------
# The same answer, as something the UI can draw
# ---------------------------------------------------------------------------
#
# The rows are right here and already structured — a template flattens them to
# markdown and the browser re-reads the markdown, which throws the structure
# away at the last possible moment. Emitting a widget beside the text keeps it,
# and costs nothing: the template already decided what shape the answer is.
#
# The text block stays. It is what a screen reader reads, what a copy-paste
# copies, and what renders if a client meets a widget it does not know.


def _widget_items(name: str, items: Sequence[Any], tz: str) -> dict[str, Any] | None:
    """Widget data for a template, built from the rows it just rendered."""
    if name == "event_list":
        out = []
        # Chronological, like the text version. The rows arrive in whatever
        # order the query returned them, and a day list that jumps around is
        # worse than no widget at all.
        ordered = sorted(
            (i for i in items if _moment(_get(i, "starts_at", "start")) is not None),
            key=lambda i: _moment(_get(i, "starts_at", "start")),
        )
        for item in ordered:
            starts = _moment(_get(item, "starts_at", "start"))
            if starts is None:
                continue
            ends = _moment(_get(item, "ends_at", "end"))
            guests = _get(item, "attendee_emails", "attendees", default=[]) or []
            out.append(
                {
                    "id": _get(item, "event_id", "id"),
                    "title": _get(item, "title", "summary", "name", default="(no title)"),
                    "starts_at": _iso_local(starts, tz),
                    "ends_at": _iso_local(ends, tz) if ends else None,
                    "all_day": bool(_get(item, "all_day")),
                    "location": _get(item, "location"),
                    "guests": [
                        g.get("email") if isinstance(g, dict) else g for g in guests
                    ][:12],
                    "status": _get(item, "status"),
                    "url": _get(item, "url", "html_link"),
                }
            )
        return {"items": out} if out else None

    if name == "conflict_list":
        entries = []
        for conflict in items:
            left = _get(conflict, "left", "event", default=conflict)
            right = _get(conflict, "right", "against")
            title = _get(left, "title", "summary", "name", default="(no title)")
            when = _moment(_get(left, "starts_at", "start"))
            detail = None
            if isinstance(right, dict):
                other = _get(right, "title", "summary", "name", default="(no title)")
                clash = _moment(_get(right, "starts_at", "start"))
                detail = f"clashes with {other}" + (
                    f" at {_clock(clash, tz)}" if clash else ""
                )
            entries.append(
                {
                    "at": _clock(when, tz) if when else None,
                    "title": title,
                    "detail": detail or _get(conflict, "reason", "why"),
                }
            )
        return {"entries": entries} if entries else None

    if name == "email_list":
        out = []
        for item in items:
            out.append(
                {
                    "id": _get(item, "message_id", "id"),
                    "subject": _get(item, "subject", default="(no subject)"),
                    "from_name": _get(item, "from_name"),
                    "from_email": _get(item, "from_email"),
                    "received_at": _iso_local(_moment(_get(item, "received_at")), tz),
                    "excerpt": _excerpt(_get(item, "body_clean", "snippet", default=""), 160),
                    "url": _get(item, "url"),
                }
            )
        return {"items": out} if out else None

    if name == "file_list":
        out = []
        for item in items:
            out.append(
                {
                    "id": _get(item, "file_id", "id"),
                    "name": _get(item, "name", "title", default="(no name)"),
                    "mime_type": _get(item, "mime_type"),
                    "owner": _get(item, "owner_email"),
                    "modified_at": _iso_local(_moment(_get(item, "modified_at")), tz),
                    "size_bytes": _get(item, "size_bytes"),
                    "url": _get(item, "web_view_link", "url"),
                }
            )
        return {"items": out} if out else None

    return None


def _iso_local(moment: datetime | None, tz: str) -> str | None:
    """An instant in the reader's timezone. The UI formats it; we place it."""
    if moment is None:
        return None
    try:
        return moment.astimezone(_zone(tz)).isoformat()
    except Exception:  # noqa: BLE001 - a bad zone must not cost the answer
        return moment.isoformat()


def _free_slots_widget(
    items: Sequence[Any], *, tz: str, window: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The gaps, as something you can tap.

    This is the one answer markdown genuinely cannot express: "you are free
    3:00–4:30" is a fact, and a button that books it is the thing somebody
    actually wanted.
    """
    start, end = _moment((window or {}).get("start")), _moment((window or {}).get("end"))
    if start is None or end is None:
        return None
    zone = _zone(tz)

    busy: list[tuple[datetime, datetime]] = []
    for item in items:
        begins = _moment(_get(item, "starts_at", "start"))
        if begins is None:
            continue
        finishes = _moment(_get(item, "ends_at", "end")) or begins + timedelta(minutes=30)
        if str(_get(item, "status") or "").lower() != "cancelled":
            busy.append((begins, finishes))
    busy.sort()

    days: list[dict[str, Any]] = []
    cursor = start.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    limit = end.astimezone(zone)
    while cursor < limit and len(days) < 14:
        day_open = cursor.replace(hour=WORKDAY_START)
        day_close = cursor.replace(hour=WORKDAY_END)
        free: list[dict[str, str]] = []
        at = max(day_open, start.astimezone(zone))
        for b0, b1 in busy:
            b0, b1 = b0.astimezone(zone), b1.astimezone(zone)
            if b1 <= at or b0 >= day_close:
                continue
            if b0 > at:
                free.append({"start": at.isoformat(), "end": min(b0, day_close).isoformat()})
            at = max(at, b1)
        if at < day_close:
            free.append({"start": at.isoformat(), "end": day_close.isoformat()})
        # A sliver between two meetings is not free time anybody can use.
        free = [
            slot
            for slot in free
            if (datetime.fromisoformat(slot["end"]) - datetime.fromisoformat(slot["start"]))
            >= timedelta(minutes=15)
        ]
        if free:
            days.append({"date": cursor.date().isoformat(), "slots": free})
        cursor += timedelta(days=1)

    return {"days": days} if days else None


def _count_widget(
    items: Sequence[Any],
    *,
    tz: str,
    window: dict[str, Any] | None,
    count: int | None,
) -> dict[str, Any] | None:
    """One number, which is the whole answer. A card, not a sentence."""
    if count is None:
        return None
    header = _window_header(window, tz)
    return {"value": str(count), "label": header or None}


def _attach_widget(
    rendered: Rendered,
    name: str,
    items: Sequence[Any],
    *,
    tz: str,
    window: dict[str, Any] | None,
    count: int | None = None,
) -> Rendered:
    """Add the widget for a template, when there is one and it is valid."""
    if name == "free_slots":
        body = _free_slots_widget(items, tz=tz, window=window)
    elif name == "count_answer":
        body = _count_widget(items, tz=tz, window=window, count=count)
    else:
        body = _widget_items(name, items, tz)
    if body is None:
        return rendered
    # The widget kind and the template name are the same word for most of
    # these. A count is a `stat` and a clash list is a `timeline`, because what
    # they are is not what produced them.
    kind = {"count_answer": "stat", "conflict_list": "timeline"}.get(name, name)
    block = widgets.widget_block(kind, body, text=rendered.text, replaces_text=True)
    if block is not None:
        rendered.blocks.append(block)
    return rendered


def _window_header(window: dict[str, Any] | None, tz: str) -> str:
    if not window:
        return ""
    start, end = _moment(window.get("start")), _moment(window.get("end"))
    if start is None or end is None:
        return ""
    last = end - timedelta(seconds=1)
    if _day(start, tz) == _day(last, tz):
        return _day(start, tz)
    return f"{_day(start, tz)} to {_day(last, tz)}"


def _plain_window(window: dict[str, Any] | None, tz: str) -> str:
    """The dates we read a phrase as, in words a person would use.

    ``interpretation`` on the window is precise and deliberately technical —
    "week_start=7; half-open" is exactly what you want in a trace when a date
    is disputed. It is not what you want under an answer. The dates themselves
    say the same thing, so the restatement is built from those instead.
    """
    header = _window_header(window, tz)
    return f"_Showing {header}._" if header else ""


def _searched_line(window: dict[str, Any] | None, tz: str) -> str:
    if not window:
        return ""
    start, end = _moment(window.get("start")), _moment(window.get("end"))
    if start is None or end is None:
        return ""
    days = max(1, round((end - start).total_seconds() / 86400))
    if days > 45:
        return f"_Searched the last {days} days._"
    return f"_Searched {_window_header(window, tz)}._"


# ---------------------------------------------------------------------------
# The card path
# ---------------------------------------------------------------------------


def card_preamble(
    staged: Sequence[dict[str, Any]],
    *,
    found: str = "",
    tz: str = "UTC",
) -> str:
    """The sentence above a confirm card. Templated, never a model call."""
    if not staged:
        return found or "Ready when you are."

    lines: list[str] = []
    if found:
        lines += [found, ""]

    verbs = {
        "gmail.send_email": "an email to send",
        "gmail.draft_email": "a draft",
        "gmail.update_labels": "a label change",
        "gcal.update_event": "a calendar change",
        "gcal.create_event": "a new event",
        "gcal.delete_event": "an event to delete",
        "gdrive.share_file": "a file to share",
        "gdrive.move_file": "a file to move",
    }
    if len(staged) == 1:
        what = verbs.get(str(staged[0].get("op")), "a change")
        lines.append(f"I have prepared {what}. **Nothing has happened yet.**")
    else:
        listed = ", then ".join(
            verbs.get(str(action.get("op")), "a change") for action in staged
        )
        lines.append(
            f"I have prepared {listed}. They run in that order, and **nothing has "
            "happened yet** — if the first one fails the second does not run."
        )
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# The prose path
# ---------------------------------------------------------------------------


def _names_the_failure(text: str, failures: Sequence[dict[str, Any]]) -> bool:
    """Did the answer actually say what broke? Checked, not assumed."""
    lowered = (text or "").lower()
    for failure in failures:
        service = str(failure.get("service", ""))
        friendly = SERVICE_NAMES.get(service, service).lower()
        if friendly and friendly in lowered:
            continue
        if service and service in lowered:
            continue
        return False
    return True



#: How a synthesised answer offers a widget: prose first, then a fenced block.
#: A separate call to ask "how should this be shown" would double the cost of
#: every answer to buy presentation, so it rides along on the one call already
#: being made and is simply absent when the model has nothing to add.
WIDGET_FENCE: Final[str] = "```widget"


def _split_widget_fence(text: str) -> tuple[str, dict[str, Any] | None]:
    """The prose, and the widget the model appended to it if any."""
    at = text.find(WIDGET_FENCE)
    if at == -1:
        return text, None
    body = text[at + len(WIDGET_FENCE) :]
    end = body.find("```")
    if end != -1:
        body = body[:end]
    try:
        parsed = json.loads(body.strip())
    except (ValueError, TypeError):
        log.info("render.widget_fence_unparsed")
        return text[:at].rstrip(), None
    return text[:at].rstrip(), parsed if isinstance(parsed, dict) else None


class _FenceGate:
    """Forwards streamed text until the widget fence starts, then stops.

    The fence is JSON meant for this module, not for the person reading. It
    arrives last, so the whole answer still streams — but a chunk can split the
    marker across two pieces, which is why the tail is held back rather than
    matched piece by piece.
    """

    def __init__(self) -> None:
        self._held = ""
        self._closed = False

    def feed(self, piece: str) -> str:
        if self._closed:
            return ""
        self._held += piece
        at = self._held.find(WIDGET_FENCE)
        if at != -1:
            self._closed = True
            out, self._held = self._held[:at], ""
            return out
        # Hold back anything that could still turn into the marker.
        keep = 0
        for size in range(min(len(WIDGET_FENCE) - 1, len(self._held)), 0, -1):
            if WIDGET_FENCE.startswith(self._held[-size:]):
                keep = size
                break
        out = self._held[: len(self._held) - keep] if keep else self._held
        self._held = self._held[len(out) :]
        return out

    def drain(self) -> str:
        if self._closed:
            return ""
        out, self._held = self._held, ""
        return out


async def prose(
    *,
    query: str,
    results: dict[str, Any],
    tz: str,
    now: datetime,
    intent: dict[str, Any] | None = None,
    windows: dict[str, Any] | None = None,
    degraded: dict[str, Any] | None = None,
    history: Sequence[dict[str, Any]] | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> Rendered:
    """One streamed call, then a post-check that does not trust its output."""
    from app.core import llm
    from app.orchestrator import prompts

    system = prompts.SYNTH_SYSTEM
    user = prompts.synth_user(
        query=query,
        intent=intent,
        results=results,
        timezone=tz,
        now_iso=now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        windows=windows,
        degraded=degraded,
        history=history,
    )

    chunks: list[str] = []
    gate = _FenceGate()
    try:
        async for piece in llm.stream_text(system, user):
            chunks.append(piece)
            if on_delta is not None:
                visible = gate.feed(piece)
                if visible:
                    await on_delta(visible)
        if on_delta is not None:
            tail = gate.drain()
            if tail:
                await on_delta(tail)
        calls = 1
    except Exception as exc:  # noqa: BLE001 - the template is always available
        log.warning("render.prose_failed", error=str(exc))
        text = summary_list(collect(results), tz=tz)
        rendered = Rendered("template:summary_list", text, [_text_block(text)], 0)
        return _prepend_banner(rendered, degraded, on_delta is not None)

    text, offered = _split_widget_fence("".join(chunks).strip())
    text = text.strip()
    if not text:
        text = summary_list(collect(results), tz=tz)
        calls = 1

    rendered = Rendered("prose", text, [_text_block(text)], calls)
    if offered is not None:
        # The model chose the shape; the data still has to survive validation,
        # and any link in it has to be one this run actually saw. A widget that
        # does not pass is dropped and the prose stands on its own.
        block = widgets.from_model(
            offered, text=text, results=list((results or {}).values())
        )
        if block is not None:
            rendered.blocks.append(block)
    return _prepend_banner(rendered, degraded, streamed=True)


def _prepend_banner(
    rendered: Rendered, degraded: dict[str, Any] | None, streamed: bool = False
) -> Rendered:
    """Put the failure first if the answer did not.

    Deterministic on purpose. Whether a partial answer admits it is partial is
    not something to leave to a sampler.
    """
    failures = list((degraded or {}).get("failed") or [])
    if not failures or _names_the_failure(rendered.text, failures):
        return rendered

    banner = degraded_banner(failures, skipped=(degraded or {}).get("skipped") or ())
    if not banner:
        return rendered

    rendered.text = f"{banner}\n\n{rendered.text}".strip()
    rendered.blocks = [_text_block(rendered.text)]
    rendered.degraded_prepended = True
    log.info("render.degraded_banner_prepended", services=[f.get("service") for f in failures])
    return rendered


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _primary_window(windows: dict[str, Any] | None, intent: dict[str, Any] | None) -> dict[str, Any] | None:
    if intent and isinstance(intent.get("resolved_window"), dict):
        return intent["resolved_window"]
    if not windows:
        return None
    named = {k: v for k, v in windows.items() if k not in ("default_read", "write_default")}
    chosen = next(iter(named.values()), None) or next(iter(windows.values()), None)
    if chosen is None:
        return None
    return chosen if isinstance(chosen, dict) else getattr(chosen, "to_dict", lambda: None)()


async def render(
    style: str,
    *,
    query: str = "",
    results: dict[str, Any] | None = None,
    steps: Sequence[dict[str, Any]] = (),
    intent: dict[str, Any] | None = None,
    windows: dict[str, Any] | None = None,
    degraded: dict[str, Any] | None = None,
    staged: Sequence[dict[str, Any]] = (),
    action_ids: Sequence[str] = (),
    input_ids: Sequence[str] = (),
    tz: str = "UTC",
    now: datetime | None = None,
    history: Sequence[dict[str, Any]] | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    freshness: str = "",
) -> Rendered:
    """Render the answer in whichever of the three ways the plan asked for."""
    results = results or {}
    moment = now or datetime.now(UTC)
    window = _primary_window(windows, intent)
    items = collect(results)

    if style == "card":
        found = _card_found_line(results, tz)
        text = card_preamble(staged, found=found, tz=tz)
        if not staged:
            # Nothing waiting on approval — but the run may have already done
            # the reversible half. Saying "Ready when you are." over a draft
            # that now sits in their Gmail is how a working compose reads as
            # a dead end.
            done = _done_lines(results)
            if done:
                text = "\n\n".join(filter(None, [found, *done]))
        rendered = Rendered("card", text, [_text_block(text)], 0)
        return _prepend_banner(rendered, degraded).with_refs(action_ids, input_ids)

    if style == "prose":
        rendered = await prose(
            query=query,
            results=results,
            tz=tz,
            now=moment,
            intent=intent,
            windows=windows,
            degraded=degraded,
            history=history,
            on_delta=on_delta,
        )
        return rendered.with_refs(action_ids, input_ids)

    name = style.split(":", 1)[1] if style.startswith("template:") else style
    if name not in TEMPLATES:
        name = "summary_list"

    footer = _freshness_line(freshness)
    if name == "event_list":
        text = event_list(items, tz=tz, window=window, footer=footer)
    elif name == "email_list":
        text = email_list(items, tz=tz, window=window, footer=footer)
    elif name == "file_list":
        text = file_list(items, tz=tz, window=window, footer=footer)
    elif name == "free_slots":
        text = free_slots(items, tz=tz, window=window, footer=footer)
    elif name == "conflict_list":
        text = conflict_list(items, tz=tz, window=window, footer=footer)
    elif name == "count_answer":
        text = count_answer(
            _count_of(results, items), noun=_noun_of(intent, steps), window=window, tz=tz
        )
    elif name == "empty_result":
        text = empty_result(_noun_of(intent, steps) + "s", window=window, tz=tz)
    elif name == "degraded_banner":
        text = degraded_banner((degraded or {}).get("failed") or [])
    elif name == "reconnect_card":
        text = reconnect_card()
    else:
        text = summary_list(items, tz=tz, footer=footer)

    if not items and name not in ("empty_result", "degraded_banner", "reconnect_card", "count_answer", "free_slots"):
        text = empty_result(_EMPTY_NOUN.get(name, _noun_of(intent, steps) + "s"), window=window, tz=tz)

    # The dates are restated so a disagreement is one sentence away — but as
    # dates, not as the internal reading of the phrase.
    if name in ("event_list", "count_answer", "free_slots"):
        plain = _plain_window(window, tz)
        if plain and _window_header(window, tz) not in text:
            text = f"{text}\n\n{plain}"

    rendered = Rendered(f"template:{name}", text, [_text_block(text)], 0)
    rendered = _attach_widget(
        rendered,
        name,
        items,
        tz=tz,
        window=window,
        count=_count_of(results, items) if name == "count_answer" else None,
    )
    return _prepend_banner(rendered, degraded).with_refs(action_ids, input_ids)


def _count_of(results: dict[str, Any], items: Sequence[Any]) -> int:
    for result in results.values():
        if isinstance(result, dict) and isinstance(result.get("count"), int):
            return int(result["count"])
    return len(items)


def _noun_of(intent: dict[str, Any] | None, steps: Sequence[dict[str, Any]]) -> str:
    services = [service_of(str(step.get("op", ""))) for step in steps]
    if not services and intent:
        services = [str(s) for s in (intent.get("services") or [])]
    if "gcal" in services:
        return "event"
    if "gmail" in services:
        return "email"
    if "gdrive" in services:
        return "file"
    return "result"


def _freshness_line(freshness: str) -> str:
    return freshness or ""


def _done_lines(results: dict[str, Any]) -> list[str]:
    """The writes this run already performed, said plainly.

    Reversible halves execute without a card — a draft lands in Gmail drafts,
    a label is applied — and the card-style answer used to skip them entirely.
    Recognised by result shape, never guessed.
    """
    lines: list[str] = []
    for result in results.values():
        if not isinstance(result, dict):
            continue
        # A draft: `draft_id` set by gmail.draft_email, which executes in-run.
        # `prepared` marks gmail.send_email's staged payload — that one is
        # described by its confirm card, not here. An actual send never
        # appears in run results at all (the actions worker does it after
        # approval), so nothing in this function may ever say "sent".
        if result.get("draft_id") and not result.get("prepared"):
            to = ", ".join(str(t) for t in (result.get("to") or [])) or "the recipient"
            subject = result.get("subject")
            line = f"I have drafted an email to {to}"
            if subject:
                line += f" — subject **{subject}**"
            line += ". It is in your Gmail drafts, and nothing has been sent."
            body = str(result.get("body") or "").strip()
            if body:
                preview = body if len(body) <= 400 else body[:400] + "…"
                line += f"\n\n> {preview}"
            lines.append(line)
    return lines


def _card_found_line(results: dict[str, Any], tz: str) -> str:
    """What the write is anchored on, stated before the card shows it.

    Only speaks when a search surfaced a real reference — a booking code, an
    order number — because that is the one case where "I found X" earns its
    place above the card. Matching on any subject in the results made this
    line quote whatever the probe happened to rank first ("I found *hello*"),
    which reads as a non sequitur over a compose and as a wrong anchor over
    everything else.
    """
    for result in results.values():
        if not isinstance(result, dict):
            continue
        extracted = result.get("extracted") or {}
        reference = extracted.get("pnr") or extracted.get("order_id") or extracted.get("ticket_no")
        if not reference:
            continue
        subject = result.get("subject") or result.get("title")
        if subject:
            return f"I found **{subject}** ({reference})."
    return ""


__all__ = [
    "SERVICE_NAMES",
    "TEMPLATES",
    "Rendered",
    "card_preamble",
    "collect",
    "conflict_list",
    "count_answer",
    "degraded_banner",
    "email_list",
    "empty_result",
    "event_list",
    "file_list",
    "free_slots",
    "prose",
    "reconnect_card",
    "render",
    "summary_list",
]
