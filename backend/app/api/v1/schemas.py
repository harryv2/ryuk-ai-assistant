"""Request bodies, and the serialisers every router shares.

Two halves.

**In** — the Pydantic models FastAPI validates a request against. They are
deliberately strict: an unknown field is a typo, and a typo that is silently
ignored is a feature the caller thinks they used. A failure here becomes
``422 VALIDATION_ERROR`` through the handler in ``app.main``.

**Out** — plain functions from a database row to the dict ``docs/API.md``
promises. Not Pydantic models, because these shapes are assembled from several
tables and half their content is JSONB that no schema of ours describes; a
response model would only re-validate what the database already holds and drop
anything it did not expect.

Three rules hold across every serialiser here:

* **Every timestamp is RFC 3339 with a ``Z``.** :func:`iso` is the only place
  that decision is made.
* **An action's ``payload`` never leaves the process.** What a person sees is
  ``Op.preview()``, which the op wrote for exactly that purpose. The response
  carries ``payload_fields`` so a client knows what an edit may touch.
* **A ``ref`` in a content block resolves against the arrays in the same
  response.** A ref with nothing behind it is a row that is gone; the client
  drops the block rather than rendering an empty box.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.logging import get_logger

log = get_logger(__name__)

#: Characters of a body excerpt on a confirm card. Long enough to recognise the
#: message, short enough that a preview is not a copy.
EXCERPT = 240

#: Values a preview may carry verbatim. Anything else is trimmed or dropped.
_SCALARS = (str, int, float, bool)

SERVICES: tuple[str, ...] = ("gmail", "gcal", "gdrive")

Service = Literal["gmail", "gcal", "gdrive"]
PromptStatusFilter = Literal[
    "pending", "answered", "cancelled", "expired", "superseded", "all"
]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """``POST /api/v1/query``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, min_length=21, max_length=21)

    #: Overrides ``users.timezone`` for this query only — the browser knows
    #: where the person is right now, the stored value knows where they live.
    timezone: str | None = None
    #: Overrides ``users.work_week_start``. 1 = Monday. Changes what "next
    #: week" means, which is why it is per query rather than a constant.
    week_start: int | None = Field(default=None, ge=1, le=7)

    freshness: Literal["auto", "cached", "live"] = "auto"

    #: The caller's own idempotency handle. Re-posting it inside
    #: ``IDEMPOTENCY_TTL_S`` returns the original run instead of starting a
    #: second one, which is what a double-tapped send button needs.
    client_request_id: str | None = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a query needs some words in it")
        return value

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"{value!r} is not an IANA time zone") from exc
        return value


class PromptResponseRequest(BaseModel):
    """``POST /api/v1/prompts/{id}/respond``.

    ``value`` is validated against the prompt's own ``value_schema`` in the
    handler, not here. That schema is the validation authority; a second
    opinion in this file would be a second contract to keep in step.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any = Field(default=..., description="Must satisfy the prompt's value_schema.")


class SyncTriggerRequest(BaseModel):
    """``POST /api/v1/sync/trigger``."""

    model_config = ConfigDict(extra="forbid")

    services: list[Service] | None = None
    mode: Literal["incremental", "backfill", "full"] = "incremental"

    def wanted(self) -> list[str]:
        """The services to act on, deduplicated, in a stable order."""
        if not self.services:
            return list(SERVICES)
        return [s for s in SERVICES if s in set(self.services)]


# --------------------------------------------------------------------------- #
# Confirm-card values
# --------------------------------------------------------------------------- #

#: Keys a confirm card's schema may use for the yes/no. An op writes the
#: schema, so the API does not get to assume one name.
APPROVE_KEYS: tuple[str, ...] = ("approve", "approved", "confirm", "confirmed", "ok", "yes", "send")

_YES_WORDS = {"yes", "y", "yeah", "yep", "send", "send it", "do it", "go", "go ahead", "ok", "okay"}
_NO_WORDS = {"no", "n", "nope", "cancel", "stop", "not now", "don't", "dont", "never mind"}


def reads_as_approval(value: Any, *, depth: int = 0) -> bool | None:
    """Did the person say yes? ``True``, ``False``, or ``None`` for neither.

    Deliberately strict. A value nobody can read as a decision must not send an
    email, so the caller turns ``None`` into an error rather than into a send.
    """
    if depth > 3:
        return None
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        word = value.strip().lower().rstrip("!.")
        if word in _YES_WORDS:
            return True
        if word in _NO_WORDS:
            return False
        return None
    if isinstance(value, dict):
        for name in APPROVE_KEYS:
            if name in value:
                return reads_as_approval(value[name], depth=depth + 1)
        return None
    return None


def patch_in(value: Any) -> dict[str, Any] | None:
    """The ``patch`` an Edit click carries, or ``None`` for a plain yes/no."""
    if not isinstance(value, dict):
        return None
    patch = value.get("patch")
    if isinstance(patch, dict) and patch:
        return dict(patch)
    return None


# --------------------------------------------------------------------------- #
# Small conversions
# --------------------------------------------------------------------------- #


def iso(value: Any) -> str | None:
    """RFC 3339 in UTC, with a ``Z``. The only place that decision is made."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=dt.UTC)
        return moment.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def status_of(value: Any) -> str | None:
    """A status column's value, whether it came back as an enum or a string."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def duration_ms(started: Any, finished: Any) -> int | None:
    """Wall time between two stamps, or ``None`` while one of them is missing."""
    if not isinstance(started, dt.datetime) or not isinstance(finished, dt.datetime):
        return None
    return max(0, int((finished - started).total_seconds() * 1000))


def get(row: Any, name: str, default: Any = None) -> Any:
    """Read a field off an ORM row or a dict, whichever this is."""
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def flatten_text(content: Any) -> str:
    """The text blocks of a message, flattened to markdown.

    Derived, never stored: a client that only wants prose reads this, and a
    client that renders cards reads ``content`` instead.
    """
    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict) or block.get("type") not in (None, "text"):
            continue
        data = block.get("data") or {}
        for key in ("markdown", "text", "body"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
                break
    return "\n\n".join(parts)


def refs_in(content: Any, kind: str) -> list[str]:
    """Every ``{"type": kind, "ref": X}`` in a content block list, in order."""
    out: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == kind:
            ref = block.get("ref")
            if isinstance(ref, str) and ref not in out:
                out.append(ref)
    return out


# --------------------------------------------------------------------------- #
# Cursors
# --------------------------------------------------------------------------- #


def encode_cursor(moment: dt.datetime | None, ident: str) -> str:
    """An opaque keyset cursor over ``(last_message_at, id)``."""
    payload = json.dumps({"l": iso(moment), "i": ident}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str | None) -> tuple[dt.datetime | None, str | None]:
    """The other half of :func:`encode_cursor`.

    A cursor we cannot read is a validation error, not a silent first page —
    silently restarting a paging loop is how a client spins forever.
    """
    if not cursor:
        return None, None
    from app.core.errors import AppError

    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw)
        moment = payload.get("l")
        parsed = (
            dt.datetime.fromisoformat(str(moment).replace("Z", "+00:00"))
            if moment
            else None
        )
    except (binascii.Error, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise AppError.validation("That cursor is not one of ours.", cursor=cursor) from exc
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    ident = payload.get("i")
    return parsed, str(ident) if ident else None


# --------------------------------------------------------------------------- #
# Previews
# --------------------------------------------------------------------------- #


def _trim(value: Any, width: int = EXCERPT) -> Any:
    """One value, small enough to put on a card."""
    if isinstance(value, str):
        return value if len(value) <= width else value[: width - 1].rstrip() + "…"
    if isinstance(value, _SCALARS) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_trim(item, width) for item in list(value)[:10]]
    if isinstance(value, dict):
        return {str(k): _trim(v, width) for k, v in list(value.items())[:10]}
    if isinstance(value, dt.datetime):
        return iso(value)
    return _trim(str(value), width)


def _fallback_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """What to show when the op cannot show it itself.

    Used for an op that is not in the registry any more (a plan from before a
    rename) and for one whose ``preview`` raised. Both are rare and neither is
    a reason to leave a person staring at a card with nothing on it.
    """
    out: dict[str, Any] = {}
    for name, value in (payload or {}).items():
        if name in {"body", "html", "content", "text"} and isinstance(value, str):
            out["body_excerpt"] = _trim(value)
            continue
        out[str(name)] = _trim(value)
    return out


def preview_for(op_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """The card's view of a staged write.

    ``Op.preview()`` decides what a person needs in order to say yes, which is
    never the whole payload — a 20 kB body is not a preview and a raw payload
    is not a summary.
    """
    payload = dict(payload or {})
    try:
        from app.ops import registry

        op = registry.get(op_name)
        preview = getattr(op, "preview", None)
        if callable(preview):
            shown = preview(payload)
            if isinstance(shown, dict):
                return {str(k): _trim(v, 2000) for k, v in shown.items()}
    except Exception as exc:
        log.warning("api.preview_failed", op=op_name, error=str(exc))
    return _fallback_preview(payload)


# --------------------------------------------------------------------------- #
# Rows -> responses
# --------------------------------------------------------------------------- #


def action_dto(row: Any, *, expires_at: Any = None) -> dict[str, Any]:
    """One ``actions`` row.

    ``status`` is the field to read before telling a person anything happened:
    ``draft`` means prepared and *not* done. ``external_ref`` on a
    ``gmail.send_email`` is the Gmail draft already sitting in their account.
    """
    payload = get(row, "payload") or {}
    op = str(get(row, "op", ""))
    revisions = get(row, "revisions") or []
    return {
        "id": get(row, "id"),
        "op": op,
        "status": status_of(get(row, "status")),
        "requires_input_id": get(row, "requires_input_id"),
        "message_id": get(row, "message_id"),
        "external_ref": get(row, "external_ref"),
        "preview": preview_for(op, payload),
        "payload_fields": sorted(str(k) for k in payload),
        "revision": len(revisions),
        "result": get(row, "result"),
        "error": get(row, "error"),
        "attempts": int(get(row, "attempts", 0) or 0),
        "job_id": get(row, "job_id"),
        "executed_at": iso(get(row, "executed_at")),
        "expires_at": iso(expires_at),
        "created_at": iso(get(row, "created_at")),
    }


def prompt_dto(
    row: Any,
    *,
    conversation_id: str | None = None,
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``pending_inputs`` row — a card.

    ``value_schema`` travels with it because that schema *is* the contract: a
    client posts a value the schema accepts, and there is no per-kind branch
    anywhere on the server. ``kind`` only says which of six renderers to use.
    """
    out: dict[str, Any] = {
        "id": get(row, "id"),
        "run_id": get(row, "run_id"),
        "conversation_id": conversation_id,
        "message_id": get(row, "message_id"),
        "node_execution_id": get(row, "node_execution_id"),
        "kind": status_of(get(row, "kind")),
        "blocking": bool(get(row, "blocking", False)),
        "prompt": get(row, "prompt") or {},
        "value_schema": get(row, "value_schema") or {},
        "options": get(row, "options"),
        "status": status_of(get(row, "status")),
        "response": get(row, "response"),
        "expires_at": iso(get(row, "expires_at")),
        "answered_at": iso(get(row, "answered_at")),
        "created_at": iso(get(row, "created_at")),
    }
    if action is not None:
        out["action"] = action
    return out


def summarise(result: Any) -> dict[str, Any] | None:
    """A step's ``result``, trimmed to what a trace row shows.

    The whole blob is on the stream and in ``GET /runs/{id}/steps``. A list of
    forty emails inside a hundred-step response helps nobody.
    """
    if result is None:
        return None
    if not isinstance(result, dict):
        return {"value": _trim(result, 120)}

    out: dict[str, Any] = {}
    for name, value in result.items():
        if isinstance(value, list):
            out.setdefault("count", len(value))
            first = value[0] if value else None
            if isinstance(first, dict):
                for label in ("title", "subject", "name", "label", "summary"):
                    if isinstance(first.get(label), str):
                        out.setdefault("top", _trim(first[label], 120))
                        break
            continue
        if isinstance(value, _SCALARS) or value is None:
            out[str(name)] = _trim(value, 120)
    return out or {"ok": True}


def step_dto(row: Any) -> dict[str, Any]:
    """One ``node_executions`` row, or a step the runner already shaped.

    Takes both because the runner hands its own trace back on the result and
    the trace endpoints read the table; they have to render identically.
    """
    started, finished = get(row, "started_at"), get(row, "finished_at")
    retries = get(row, "retries") or []
    explicit = get(row, "duration_ms")
    status = status_of(get(row, "status"))

    out: dict[str, Any] = {
        "node_id": get(row, "node_id") or get(row, "id"),
        "op": str(get(row, "op", "")),
        "seq": get(row, "seq"),
        "round": int(get(row, "round", 0) or 0),
        "status": status,
        "depends_on": list(get(row, "depends_on") or []),
        "args": get(row, "args") or {},
        "result_summary": (
            get(row, "result_summary")
            if "result_summary" in (row if isinstance(row, dict) else {})
            else summarise(get(row, "result"))
        ),
        "started_at": iso(started),
        "finished_at": iso(finished),
        "duration_ms": explicit if explicit is not None else duration_ms(started, finished),
        "attempts": (
            get(row, "attempts")
            if get(row, "attempts") is not None
            else (len(retries) + 1 if started is not None else 0)
        ),
    }
    outcome = get(row, "outcome")
    if outcome:
        out["outcome"] = outcome
    if retries:
        out["retries"] = retries
    return out


def run_dto(run: Any) -> dict[str, Any]:
    """One ``runs`` row."""
    started, finished = get(run, "started_at"), get(run, "finished_at")
    return {
        "id": get(run, "id"),
        "conversation_id": get(run, "conversation_id"),
        "trigger_message_id": get(run, "trigger_message_id"),
        "status": status_of(get(run, "status")),
        "planner_tier": get(run, "planner_tier"),
        "intent": get(run, "intent"),
        "token_usage": get(run, "token_usage"),
        "error": get(run, "error"),
        "started_at": iso(started),
        "finished_at": iso(finished),
        "duration_ms": duration_ms(started, finished),
    }


def message_dto(row: Any, *, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One ``messages`` row. ``content`` is returned exactly as stored."""
    out: dict[str, Any] = {
        "id": get(row, "id"),
        "seq": get(row, "seq"),
        "role": status_of(get(row, "role")),
        "content": get(row, "content") or [],
        "run_id": get(row, "run_id"),
        "created_at": iso(get(row, "created_at")),
    }
    if trace is not None:
        out["trace"] = trace
    return out


def entity_dto(row: Any) -> dict[str, Any]:
    """One ``conversation_entities`` row — something this thread referred to."""
    return {
        "id": get(row, "id"),
        "entity_type": get(row, "entity_type"),
        "entity_ref": get(row, "entity_ref"),
        "label": get(row, "label"),
        "meta": get(row, "meta") or {},
        "run_id": get(row, "run_id"),
        "last_seen_at": iso(get(row, "last_seen_at")),
    }


# --------------------------------------------------------------------------- #
# The run's own bookkeeping
# --------------------------------------------------------------------------- #


def normalise_usage(usage: Any) -> dict[str, Any]:
    """What the run spent, under both spellings.

    The runner counts ``{prompt, completion, calls}``; the API document calls
    them ``{prompt_tokens, completion_tokens, llm_calls}``. Rather than pick a
    winner and make one of the two wrong, both names carry the same number.
    """
    source = dict(usage or {})
    calls = source.get("calls", source.get("llm_calls", 0)) or 0
    prompt = source.get("prompt", source.get("prompt_tokens", 0)) or 0
    completion = source.get("completion", source.get("completion_tokens", 0)) or 0
    out = {
        "llm_calls": int(calls),
        "calls": int(calls),
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "prompt": int(prompt),
        "completion": int(completion),
        "model": source.get("model"),
        "provider": source.get("provider"),
        "usd": round(float(source.get("usd") or 0.0), 6),
    }
    for name, value in source.items():
        out.setdefault(str(name), value)
    return out


def normalise_timings(timings: Any) -> dict[str, Any]:
    """Milliseconds per phase. ``plan_ms`` and ``route_ms`` are the same clock."""
    # The runner hands back a Timings dataclass; the trace endpoints hand back a
    # plain dict. Both have to render identically, so coerce here rather than
    # making every caller remember which one it is holding.
    if timings is not None and not isinstance(timings, dict):
        if is_dataclass(timings) and not isinstance(timings, type):
            timings = asdict(timings)
        elif hasattr(timings, "__dict__"):
            timings = {k: v for k, v in vars(timings).items() if not k.startswith("_")}
        else:
            timings = {}
    out = {str(k): v for k, v in (timings or {}).items()}
    if "plan_ms" not in out and "route_ms" in out:
        out["plan_ms"] = out["route_ms"]
    if "route_ms" not in out and "plan_ms" in out:
        out["route_ms"] = out["plan_ms"]
    out.setdefault("total_ms", 0)
    return out


def normalise_degraded(degraded: Any) -> list[dict[str, Any]]:
    """Services that fell over, as objects.

    The runner may hand back bare service names; a client renders "Calendar is
    not answering" from a name and a reason, so a name on its own is widened
    rather than dropped.
    """
    out: list[dict[str, Any]] = []
    for item in degraded or []:
        if isinstance(item, str):
            out.append({"service": item, "reason": "failed"})
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


# --------------------------------------------------------------------------- #
# The whole response body for one turn
# --------------------------------------------------------------------------- #


def run_body(
    outcome: Any,
    *,
    pending_inputs: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """The body ``POST /api/v1/query`` returns, exactly as ``docs/API.md`` has it.

    ``content`` is the stored block list. A block that points at a row which is
    gone is dropped here rather than rendered as an empty box on screen — that
    is the read half of the "one transaction" rule.
    """
    content = [b for b in (get(outcome, "content") or []) if isinstance(b, dict)]
    live_inputs = {p["id"] for p in pending_inputs if p.get("id")}
    live_actions = {a["id"] for a in actions if a.get("id")}
    kept: list[dict[str, Any]] = []
    for block in content:
        kind, ref = block.get("type"), block.get("ref")
        if kind == "input" and ref is not None and ref not in live_inputs:
            log.info("api.dropped_block", block="input", ref=ref)
            continue
        if kind == "action" and ref is not None and ref not in live_actions:
            log.info("api.dropped_block", block="action", ref=ref)
            continue
        kept.append(block)

    text = get(outcome, "answer") or get(outcome, "text") or flatten_text(kept)
    degraded = normalise_degraded(get(outcome, "degraded_detail") or get(outcome, "degraded") or [])

    return {
        "run_id": get(outcome, "run_id"),
        "conversation_id": get(outcome, "conversation_id"),
        "message_id": get(outcome, "message_id"),
        "status": status_of(get(outcome, "status")),
        "planner_tier": get(outcome, "planner_tier"),
        "route": get(outcome, "route"),
        "intent": get(outcome, "intent"),
        "answer_style": get(outcome, "answer_style") or "prose",
        # `text` is the documented name; `answer` is what the orchestrator calls
        # it. Both carry the same string so neither side has to translate.
        "text": text,
        "answer": text,
        "content": kept,
        "actions": actions,
        "pending_inputs": pending_inputs,
        "steps": steps,
        "entities": entities,
        "degraded": degraded,
        "degraded_services": sorted({d["service"] for d in degraded if d.get("service")}),
        "usage": normalise_usage(get(outcome, "usage")),
        "timings": normalise_timings(get(outcome, "timings")),
    }


# --------------------------------------------------------------------------- #
# Conversations and sync
# --------------------------------------------------------------------------- #


def conversation_summary_dto(
    row: Any,
    *,
    fallback_title: str | None = None,
    message_count: int = 0,
    pending_input_count: int = 0,
    last_run: Any = None,
) -> dict[str, Any]:
    """One row of ``GET /api/v1/conversations``.

    ``title IS NULL`` is not a missing title — it means the person never set
    one, so the display falls back to their first message. Nothing is stored
    for that, which is why the flag travels with it.
    """
    title = get(row, "title")
    derived = title is None
    summary: dict[str, Any] | None = None
    if last_run is not None:
        intent = get(last_run, "intent") or {}
        summary = {
            "id": get(last_run, "id"),
            "status": status_of(get(last_run, "status")),
            "intent": intent.get("name") if isinstance(intent, dict) else None,
        }
    return {
        "id": get(row, "id"),
        "title": title if not derived else (fallback_title or "New conversation"),
        "title_is_derived": derived,
        "last_message_at": iso(get(row, "last_message_at")),
        "created_at": iso(get(row, "created_at")),
        "archived_at": iso(get(row, "archived_at")),
        "message_count": int(message_count),
        "pending_input_count": int(pending_input_count),
        "last_run": summary,
    }


def sync_service_dto(
    row: Any, *, now: dt.datetime, target_seconds: int
) -> dict[str, Any]:
    """One service on ``GET /api/v1/sync/status``.

    ``lag_seconds`` is measured from ``last_success_at``, never from
    ``last_synced_at``: a failing sync updates the attempt but not the success,
    and the lag figure has to describe the data rather than the effort.
    """
    success = get(row, "last_success_at")
    lag = int((now - success).total_seconds()) if isinstance(success, dt.datetime) else None
    open_until = get(row, "circuit_open_until")
    circuit_open = isinstance(open_until, dt.datetime) and open_until > now
    failures = int(get(row, "consecutive_failures", 0) or 0)
    return {
        "last_synced_at": iso(get(row, "last_synced_at")),
        "last_success_at": iso(success),
        "lag_seconds": lag,
        "items_indexed": int(get(row, "items_indexed", 0) or 0),
        "backfill_complete": bool(get(row, "backfill_complete", False)),
        "cursor_present": get(row, "cursor") is not None,
        "consecutive_failures": failures,
        "circuit_open_until": iso(open_until) if circuit_open else None,
        "last_error": get(row, "last_error"),
        "healthy": bool(
            not circuit_open
            and failures == 0
            and lag is not None
            and lag <= target_seconds
        ),
    }


__all__ = [
    "APPROVE_KEYS",
    "EXCERPT",
    "SERVICES",
    "PromptResponseRequest",
    "PromptStatusFilter",
    "QueryRequest",
    "Service",
    "SyncTriggerRequest",
    "action_dto",
    "conversation_summary_dto",
    "decode_cursor",
    "duration_ms",
    "encode_cursor",
    "entity_dto",
    "flatten_text",
    "get",
    "iso",
    "message_dto",
    "normalise_degraded",
    "normalise_timings",
    "normalise_usage",
    "patch_in",
    "preview_for",
    "prompt_dto",
    "reads_as_approval",
    "refs_in",
    "run_body",
    "run_dto",
    "status_of",
    "step_dto",
    "summarise",
    "sync_service_dto",
]
