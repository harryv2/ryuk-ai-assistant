"""One turn, start to finish.

    front door → pre-pass → probe → ONE planning call → validate → dispatch
              → pause, or render

Four things this file exists to guarantee.

**A turn the front door owns costs nothing.** Chit-chat, a UI verb, an answer to
a card already on screen, a capability question and the rule-router shapes all
return before the pre-pass runs. There is still a `runs` row, because "answered
without calling anything" is a true fact about the system and worth counting.

**One planning call.** The intent and the DAG stream back together. The intent is
written to `runs.intent` as its object closes; every step becomes a
`node_executions` row the moment it closes, so the step trace draws the graph
while the rest of the plan is still arriving. A step that cannot be wrong to have
run starts right there — a few hundred milliseconds before the last step exists.
A rejected plan gets exactly one repair round; a second rejection answers from
what the probe already found rather than dying.

**Writes are prepared, never performed.** Nothing in here sends, moves or
deletes. A confirmable step stages a `pending_inputs` row and an `actions` row,
and the assistant message that points at them is written in the *same*
transaction — so a card can never reference an action that does not exist, and an
action can never exist without a card gating it.

**Resuming costs nothing.** The plan is already in `node_executions`. Answering a
blocking question rebuilds it from those rows, drops the answer into the scope
where the paused node's result belongs, and re-enters dispatch with zero model
calls. That is the whole reason ambiguity is a step and not a special exit.

Everything is written as it happens rather than at the end. The step trace reads
`node_executions` live, and a worker that dies mid-run has to be resumable from
the database alone.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Sequence

from sqlalchemy.exc import IntegrityError

from app.core.errors import AppError
from app.core.ids import fingerprint_parts, new_id
from app.core.logging import get_logger
from app.ops.base import service_of
from app.orchestrator import dispatch as dispatcher
from app.orchestrator import entities as entity_store
from app.orchestrator import events, front_door, prepass, render, route, validate
from app.orchestrator.temporal import Window, window_from

log = get_logger(__name__)

# planner_tier, as `runs.planner_tier` records it.
TIER_TEMPLATE = 1
TIER_COMPOSED = 2
TIER_REPLAN = 3
TIER_STEP_LOOP = 4

MAX_DELTA_ROUNDS = 1  # one repair round on a rejected plan
MAX_REPLAN_ROUNDS = dispatcher.MAX_REPLAN_ROUNDS

#: The three corpora the probe searches, in the order the planner is shown them.
CORPORA: tuple[str, ...] = ("gmail", "gcal", "gdrive")

#: The card that gates every staged write.
#:
#: Three buttons, not two: "Send it" is ``{"approve": true}``, "Not now" is
#: ``{"approve": false}``, and "Edit" is ``{"approve": false, "patch": {...}}``
#: — a revision to the staged payload rather than a refusal of it. `patch` has
#: to be declared here or `additionalProperties: false` turns every edit into a
#: 422, and a client that dropped it to get past the schema would be sending a
#: plain rejection and cancelling the action it meant to revise.
CONFIRM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["approve"],
    "properties": {
        "approve": {"type": "boolean"},
        "patch": {"type": "object"},
    },
}

#: The run this task is inside. The NDJSON relay in `api/v1/query.py` needs the
#: id before `handle_query` returns, and there is no other way to hand it over.
_current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "orchestrator_run_id", default=None
)


def current_run_id() -> str | None:
    """The run this task is inside, or None."""
    return _current_run.get()


def max_llm_calls() -> int:
    """The per-run ceiling, read at call time so a test can move it."""
    try:
        from app.config import settings

        return int(getattr(settings, "MAX_LLM_CALLS_PER_RUN", 5) or 5)
    except Exception:  # noqa: BLE001 - a missing setting is not a reason to fail
        return 5


#: The old module-level name. `max_llm_calls()` is the live value.
MAX_LLM_CALLS = 5


# ---------------------------------------------------------------------------
# Who the turn belongs to
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if value is None else value


@dataclass
class Actor:
    """The person, and the two settings that change what an answer means."""

    id: str
    email: str = ""
    display_name: str = ""
    timezone: str = "UTC"
    week_start: int = 1

    @classmethod
    def of(cls, user: Any) -> "Actor":
        if isinstance(user, Actor):
            return user
        return cls(
            id=str(_field(user, "id", "")),
            email=str(_field(user, "email", "")),
            display_name=str(_field(user, "display_name", "") or ""),
            timezone=str(_field(user, "timezone", "UTC") or "UTC"),
            week_start=int(_field(user, "work_week_start", 1) or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "timezone": self.timezone,
        }


async def _load_actor(session: Any, user_id: str) -> Actor:
    from app.db.repositories import users as user_repo

    user = await user_repo.get_user(session, user_id)
    if user is None:
        raise AppError("NOT_FOUND", "No such user.", http=404, details={"user_id": user_id})
    return Actor.of(user)


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """What one turn produced. The API layer renders this and nothing else."""

    conversation_id: str
    message_id: str
    run_id: str
    status: str  # complete | awaiting_input | failed | timeout
    answer: str = ""
    content: list[dict[str, Any]] = field(default_factory=list)
    intent: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    timings: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    # Detail the wire format also carries. `degraded` is a list of service names
    # because that is the contract; why each one failed is here, so a client can
    # say *why* Calendar is missing rather than only that it is.
    degraded_detail: list[dict[str, Any]] = field(default_factory=list)
    answer_style: str = "prose"
    planner_tier: int = TIER_TEMPLATE
    action_ids: list[str] = field(default_factory=list)
    input_ids: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    route: str = front_door.ROUTE_MISS

    # `api/v1/_shared.hydrate` reads the answer under these two names.
    @property
    def text(self) -> str:
        return self.answer

    @property
    def blocks(self) -> list[dict[str, Any]]:
        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "run_id": self.run_id,
            "status": self.status,
            "answer": self.answer,
            "content": self.content,
            "intent": self.intent,
            "planner_tier": self.planner_tier,
            "answer_style": self.answer_style,
            "steps": self.steps,
            "actions": self.actions,
            "prompts": self.prompts,
            "degraded": self.degraded,
            "degraded_detail": self.degraded_detail,
            "entities": self.entities,
            "timings": self.timings,
            "usage": self.usage,
        }


#: `RunOutcome` was this class before the interface settled.
RunOutcome = QueryResult


class Timings:
    """Milliseconds per phase, in the order they happen."""

    __slots__ = (
        "front_door_ms",
        "prepass_ms",
        "probe_ms",
        "route_ms",
        "dispatch_ms",
        "render_ms",
        "total_ms",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    #: Always reported, even at zero. A phase that did not happen is a fact
    #: worth reading off the response rather than a missing key to guess at.
    ALWAYS = ("total_ms", "probe_ms", "route_ms", "dispatch_ms", "render_ms")

    def to_dict(self) -> dict[str, int]:
        out = {name: int(getattr(self, name)) for name in self.__slots__}
        # Two names for the same clock, so neither caller has to know which
        # spelling this module picked.
        out["plan_ms"] = out["route_ms"]
        return {k: v for k, v in out.items() if v or k in self.ALWAYS}


# ---------------------------------------------------------------------------
# Google, or a clear reason there is no Google
# ---------------------------------------------------------------------------


class _NoGoogle:
    """Stands in when the user's Google connection cannot be built.

    Reaching for a client raises the real reason rather than an AttributeError,
    so the step fails with "reconnect Google" and the answer can say so.
    """

    def __init__(self, error: AppError) -> None:
        self._error = error

    def __getattr__(self, name: str) -> Any:
        raise self._error


async def _google_for(session: Any, user_id: str) -> tuple[Any, AppError | None]:
    """The user's Google clients, or a stand-in and the reason."""
    try:
        from app.google.client import clients_for

        return await clients_for(session, user_id), None
    except AppError as exc:
        log.info("runner.google_unavailable", user_id=user_id, code=exc.code)
        return _NoGoogle(exc), exc
    except Exception as exc:  # noqa: BLE001
        error = AppError(
            "GOOGLE_UNAVAILABLE",
            "I could not reach Google for this account.",
            http=503,
            details={"detail": str(exc)[:300]},
        )
        return _NoGoogle(error), error


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _json(value: Any) -> Any:
    return dispatcher.json_safe(value) if value is not None else None


def _text_block(markdown: str) -> dict[str, Any]:
    return {"type": "text", "data": {"markdown": markdown}}


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = [
        str((block.get("data") or {}).get("markdown", ""))
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def _summarise(result: Any) -> dict[str, Any]:
    """The trimmed result a `step.finished` event carries."""
    if not isinstance(result, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("count", "action_id", "input_id", "draft_id", "message_id", "event_id"):
        if result.get(key) is not None:
            summary[key] = result[key]
    for key in ("hits", "results", "items", "events", "messages", "files"):
        rows = result.get(key)
        if isinstance(rows, list):
            summary.setdefault("count", len(rows))
            if rows and isinstance(rows[0], dict):
                label = (
                    rows[0].get("subject")
                    or rows[0].get("title")
                    or rows[0].get("name")
                    or rows[0].get("summary")
                )
                if label:
                    summary["top"] = str(label)[:120]
            break
    return summary


def _usage_dict(usage: Any) -> dict[str, Any]:
    """What the run spent, counting an LLM call the way everyone else counts it.

    The ledger files one entry per attempt, embeddings included, so its `calls`
    is "requests we made to a model vendor". Every budget in this system — the
    per-run cap, the numbers in `docs/SAMPLE_QUERIES.md` — means *chat* calls,
    and the probe's one embedding is not one of those. So `calls` is the chat
    count and the embeddings are reported beside it rather than folded in.
    """
    out = dict(usage.to_dict())
    entries = list(getattr(getattr(usage, "ledger", None), "entries", ()) or ())
    embeds = sum(1 for entry in entries if getattr(entry, "kind", "chat") == "embed")
    chat = max(0, len(entries) - embeds)
    out["calls"] = chat
    out["llm_calls"] = chat
    out["embedding_calls"] = embeds
    return out


def _windows_from(intent: dict[str, Any] | None) -> dict[str, Window]:
    """Rebuild the windows a stored intent carries, for a resumed run."""
    out: dict[str, Window] = {}
    for name, raw in ((intent or {}).get("windows") or {}).items():
        window = window_from(raw)
        if window is not None:
            out[str(name)] = window
    resolved = (intent or {}).get("resolved_window")
    if isinstance(resolved, dict):
        window = window_from(resolved)
        if window is not None:
            out.setdefault(str(resolved.get("name") or "resolved"), window)
    return out


def _progress_label(op: Any, args: dict[str, Any]) -> str:
    if op is None or not hasattr(op, "progress_label"):
        return ""
    try:
        return str(op.progress_label(args) or "")
    except Exception:  # noqa: BLE001 - a label must never fail a step
        return ""


_YES_WORDS = frozenset(
    {"yes", "y", "yeah", "yep", "ok", "okay", "send", "send it", "do it", "go",
     "go ahead", "confirm", "approve", "approved", "sure", "please do"}
)
_NO_WORDS = frozenset(
    {"no", "n", "nope", "cancel", "stop", "not now", "don't", "dont", "never mind",
     "nevermind", "decline", "declined", "leave it"}
)
_APPROVE_KEYS = ("approve", "approved", "confirm", "confirmed", "ok", "yes", "value")


def _reads_as_approval(value: Any, *, depth: int = 0) -> bool | None:
    """Did the person say yes? True, False, or None when it is neither.

    Deliberately strict. A value nobody can read as a decision must not send an
    email, so the caller turns None into an error rather than into a send.
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
        for key in _APPROVE_KEYS:
            if key in value:
                return _reads_as_approval(value[key], depth=depth + 1)
        return None
    return None


def _default_question(op_name: str, preview: Any) -> str:
    """The sentence on a confirm card when the op did not write one."""
    if isinstance(preview, dict) and op_name.startswith("gmail"):
        recipients = preview.get("to")
        if isinstance(recipients, list) and recipients:
            if len(recipients) == 1:
                return f"Send this to {recipients[0]}?"
            return f"Send this to {len(recipients)} people?"
    return {
        "gmail.send_email": "Send this email?",
        "gmail.draft_email": "Save this draft?",
        "gmail.update_labels": "Change these labels?",
        "gcal.create_event": "Add this to your calendar?",
        "gcal.update_event": "Move this meeting?",
        "gcal.delete_event": "Delete this event?",
        "drive.share_file": "Share this file?",
        "gdrive.share_file": "Share this file?",
        "drive.move_file": "Move this file?",
        "gdrive.move_file": "Move this file?",
    }.get(op_name, "Go ahead with this?")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


#: Staleness is not a failure. Every step succeeded; the mirror is simply a few
#: minutes behind, which is the normal steady state of a 15-minute sync. Passing
#: it to the synthesizer in a list called "degraded" makes the model announce
#: "Gmail is not responding" on top of a complete and correct answer — the one
#: kind of wrong that teaches people to distrust the right answers too.
#: Probe reasons that describe how much of a service we hold rather than a
#: service that fell over. `never_synced` is the one that bites: a mailbox we
#: have not mirrored yet reports it on every corpus, and treating that as an
#: outage puts "Gmail is not responding" above an answer Gmail just supplied.
STALE_REASONS = frozenset(
    {"stale", "staleness", "lagging", "never_synced", "not_synced", "no_sync"}
)


def _is_outage(entry: dict[str, Any]) -> bool:
    """True when a service genuinely did not answer."""
    return str(entry.get("reason") or "") not in STALE_REASONS



class Run:
    """The state of a single turn. Lives no longer than the request."""

    def __init__(
        self,
        session: Any,
        actor: Actor,
        query: str,
        *,
        conversation_id: str | None = None,
        google: Any = None,
        now: datetime | None = None,
        registry: dict[str, Any] | None = None,
        request_id: str | None = None,
        freshness: str | None = None,
    ) -> None:
        self.session = session
        self.actor = actor
        self.query = query or ""
        self.conversation_id = conversation_id or ""
        self.request_id = request_id
        # A run-wide override of the planner's per-step choice. "Refresh and
        # ask again" is a thing people do, and it has to reach the ops.
        self.freshness = freshness if freshness in ("cached", "live") else None
        self.now = now or datetime.now(UTC)
        self._google = google
        self._google_error: AppError | None = None
        self._registry = registry

        self.run_id = ""
        self.trigger_message_id = ""
        # Allocated now so a streamed `content.delta` can name the message it
        # belongs to before that message has been written.
        self.assistant_message_id = new_id()

        self.timings = Timings()
        self.started = time.perf_counter()
        self.llm_calls = 0
        self.max_calls = max_llm_calls()

        self.windows: dict[str, Window] = {}
        self.prepass: prepass.PrePass | None = None
        self.probe: Any = None
        self.probe_bindings: dict[str, Any] = {}
        self.probe_degraded: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.intent: dict[str, Any] | None = None
        self.answer_style = "prose"
        self.planner_tier = TIER_TEMPLATE

        # Buffered until the closing transaction. The ids are allocated when a
        # write is staged so the events can fire straight away, while the rows
        # land together with the message that references them.
        self.pending_inputs: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []

        self.node_rows: dict[str, str] = {}  # node_id -> node_executions.id
        self._seq = 0
        self._written: set[tuple[str, int]] = set()
        self._raised_for_node: dict[str, str] = {}  # node_id -> input_id
        # Nodes parked behind a blocking card: the one that asked, and anything
        # that was waiting on it. They are not skipped — they have not run yet.
        self._parked_nodes: set[str] = set()
        self._early_args: dict[str, dict[str, Any]] = {}
        self._early_times: dict[str, tuple[datetime, datetime]] = {}
        self._budget_note = ""
        self._planner_error = ""
        # The dispatcher runs one task per step, and they all report through the
        # hooks below into this one session. An AsyncSession is a single
        # connection: two steps finishing together would interleave on it and
        # raise. The writes are short, so serialising them costs nothing and is
        # the difference between a step trace that updates live and one that
        # loses rows.
        self._db_lock = asyncio.Lock()
        self._google_lock = asyncio.Lock()

        self._history: list[dict[str, Any]] = []
        self._entity_chips: list[dict[str, Any]] = []
        self._open_prompts: list[dict[str, Any]] = []
        self._staged_summaries: list[dict[str, Any]] = []

    # -- wiring ------------------------------------------------------------

    @property
    def registry(self) -> dict[str, Any]:
        if self._registry is None:
            from app.ops.registry import REGISTRY

            self._registry = REGISTRY
        return self._registry

    async def google(self) -> Any:
        """The Google clients, built once and only when something needs them.

        Guarded because several steps ask for them at the same moment, and
        building them reads the token table on the run's session — two of those
        at once is the same connection twice.
        """
        if self._google is not None:
            return self._google
        async with self._google_lock:
            if self._google is None:
                self._google, self._google_error = await _google_for(
                    self.session, self.actor.id
                )
        return self._google

    def _probe_query(self) -> str:
        """The turn's query text, as the probe searched it."""
        return str(getattr(self.probe, "query", "") or self.query or "")

    def _probe_embedding(self) -> list[float] | None:
        """The query vector the probe already paid for, if it made one.

        A turn buys one embedding. Handing it to every op that searches the
        mirror is what keeps that true when a plan has three search steps.
        """
        vector = getattr(self.probe, "embedding", None)
        return list(vector) if vector else None

    async def op_context(self) -> Any:
        from app.ops.base import OpContext

        return OpContext(
            user_id=self.actor.id,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            session=self.session,
            google=await self.google(),
            now=self.now,
            tz=self.actor.timezone,
            probe_embedding=self._probe_embedding(),
            probe_query=self._probe_query(),
        )

    # -- events ------------------------------------------------------------

    async def emit(self, type: str, data: dict[str, Any] | None = None) -> None:
        if self.run_id:
            await events.publish(self.run_id, type, data or {})

    async def progress(self, phase: str, label: str, pct: int | None = None) -> None:
        data: dict[str, Any] = {"phase": phase, "label": label}
        if pct is not None:
            data["pct"] = pct
        await self.emit("progress", data)

    # -- opening -----------------------------------------------------------

    async def open(self) -> None:
        """Conversation, user message, run row. The insert order matters.

        `messages.run_id` points at `runs.id` and `runs.trigger_message_id`
        points back at `messages.id`, so it goes user message, run, then link —
        nothing ever needs a row that does not exist yet.
        """
        from app.db.repositories import conversations as conv_repo
        from app.db.repositories import runs as run_repo

        if not self.conversation_id:
            conversation = await conv_repo.create_conversation(
                self.session, self.actor.id, now=self.now
            )
            self.conversation_id = conversation.id
        else:
            await conv_repo.require_conversation(
                self.session, self.actor.id, self.conversation_id
            )

        message = await conv_repo.add_message(
            self.session,
            self.actor.id,
            self.conversation_id,
            role="user",
            content=[_text_block(self.query)],
            now=self.now,
        )
        self.trigger_message_id = message.id

        run = await run_repo.create_run(
            self.session,
            self.actor.id,
            self.conversation_id,
            self.trigger_message_id,
            now=self.now,
        )
        self.run_id = run.id
        _current_run.set(self.run_id)

        await conv_repo.set_message_run_id(
            self.session, self.actor.id, self.trigger_message_id, self.run_id
        )
        await self.session.commit()

        await self.emit(
            "run.started",
            {
                "conversation_id": self.conversation_id,
                "message_id": self.trigger_message_id,
                "query": self.query,
                "timezone": self.actor.timezone,
                "request_id": self.request_id,
            },
        )

    # -- the whole turn ----------------------------------------------------

    async def go(self) -> QueryResult:
        from app.core import llm

        with llm.track_usage() as usage:
            try:
                result = await self._pipeline()
            except AppError as exc:
                result = await self._fail(exc)
            except Exception as exc:  # noqa: BLE001 - a crash still owes an answer
                log.exception("runner.crashed", run_id=self.run_id)
                result = await self._fail(
                    AppError(
                        "INTERNAL",
                        "Something broke while I was working on that.",
                        http=500,
                        details={"detail": str(exc)[:300]},
                    )
                )
            result.usage = _usage_dict(usage)

        await self._record_usage(result.usage)

        self.timings.total_ms = int((time.perf_counter() - self.started) * 1000)
        result.timings = self.timings.to_dict()

        if result.status == "failed":
            await self.emit(
                "error",
                {
                    "code": "INTERNAL",
                    "message": result.answer,
                    "partial": bool(result.message_id),
                },
            )
        else:
            await self.emit(
                "run.complete",
                {
                    "status": result.status,
                    "message_id": result.message_id,
                    "answer_style": result.answer_style,
                    "planner_tier": result.planner_tier,
                    "usage": result.usage,
                    "timings": result.timings,
                    "degraded": result.degraded,
                },
            )
        return result

    async def _record_usage(self, usage: dict[str, Any]) -> None:
        try:
            from app.db.repositories import runs as run_repo

            await run_repo.add_token_usage(self.session, self.actor.id, self.run_id, usage)
            await self.session.commit()
        except Exception:  # noqa: BLE001 - accounting must never fail a turn
            log.warning("runner.usage_not_recorded", run_id=self.run_id)
            await self._safe_rollback()

    async def _safe_rollback(self) -> None:
        try:
            await self.session.rollback()
        except Exception:  # noqa: BLE001
            pass

    async def _pipeline(self) -> QueryResult:
        # 1. FRONT DOOR. Five matchers, no model, no embedding.
        mark = time.perf_counter()
        decision = front_door.decide(
            self.query,
            tz=self.actor.timezone,
            week_start=self.actor.week_start,
            now=self.now,
            open_prompts=await self._pending_prompts(),
            last_intent=await self._last_intent(),
        )
        self.timings.front_door_ms = int((time.perf_counter() - mark) * 1000)
        log.info(
            "runner.front_door",
            run_id=self.run_id,
            route=decision.route,
            shape=decision.shape,
        )

        handled = await self._front_door(decision)
        if handled is not None:
            return handled

        # 2. PRE-PASS. Dates through temporal, literals, aliases, mime words.
        await self.progress("prepass", "Working out what the question refers to", 5)
        mark = time.perf_counter()
        self.prepass = prepass.run(
            self.query, self.actor.timezone, self.actor.week_start, self.now
        )
        self.windows = dict(self.prepass.windows)
        self.timings.prepass_ms = int((time.perf_counter() - mark) * 1000)

        # 3. PROBE. One embedding, three hybrid searches, extractors, wave 2.
        await self.progress("probe", "Looking through mail, calendar and files", 10)
        mark = time.perf_counter()
        await self._run_probe()
        self.timings.probe_ms = int((time.perf_counter() - mark) * 1000)

        # 4. ROUTE. The one planning call, streamed.
        await self.progress("plan", "Working out what to do", 30)
        mark = time.perf_counter()
        payload, early = await self._route()
        self.timings.route_ms = int((time.perf_counter() - mark) * 1000)

        if self._planner_error:
            # The planning call itself fell over. What the probe found is still
            # real, so it is what the answer is built from — and the answer says
            # the planner is the part that is missing.
            return await self._probe_only_answer(
                [self._planner_error],
                lead=(
                    "I could not reach the part of me that works out what to do, so "
                    "here is what turned up in your own mail, calendar and files. "
                    "Ask again in a moment and I will do the whole thing."
                ),
                service="planner",
                reason="planner_unavailable",
            )

        verb = route.verb_of(payload)
        if verb == "answer":
            return await self._answered(payload)
        if verb == "revise":
            return await self._revise(payload)
        if verb == "answer_input":
            return await self._answer_input(payload)

        # 5. VALIDATE. Pure Python, one repair round, then degrade.
        plan, degraded_answer = await self._validated(payload)
        if degraded_answer is not None:
            return degraded_answer

        plan = self._apply_freshness(plan)
        self.plan = plan
        self.intent = dict(plan.get("intent") or {})
        self.answer_style = str(plan.get("answer_style") or "prose")
        await self._record_intent()
        await self._sync_step_rows(plan["steps"])

        # 6. DISPATCH, plus up to two replan rounds if a step asks for one.
        await self.progress("dispatch", "Running the plan", 55)
        mark = time.perf_counter()
        result = await self._dispatch(plan, completed=early)

        rounds = 0
        while result.status == "replan" and rounds < MAX_REPLAN_ROUNDS and self._can_call():
            rounds += 1
            self.planner_tier = TIER_REPLAN
            revised = await self._replan(plan, result, rounds)
            if revised is None:
                break
            plan = self._apply_freshness(revised)
            self.plan = plan
            await self._write_steps(plan["steps"], round_number=rounds)
            for step in plan["steps"]:
                await self.emit("plan.step", self._step_event(step, round_number=rounds))
            result = await self._dispatch(plan, completed=result.results)
        self.timings.dispatch_ms = int((time.perf_counter() - mark) * 1000)

        if result.status == "replan" and not self._can_call():
            # Past the ceiling we stop and say so rather than spending more.
            self._budget_note = (
                "That needed more re-planning than one turn allows, so this is what "
                "I have. Tell me which part matters most and I will do that one."
            )

        # 7. RENDER, then one transaction for the message and everything it
        #    points at.
        return await self._finish(plan, result)

    # -- front-door outcomes ----------------------------------------------

    async def _front_door(self, decision: front_door.Decision) -> QueryResult | None:
        """Honour a front-door hit. Every branch here costs zero model calls."""
        if decision.route == front_door.ROUTE_OPEN_CARD and decision.answer is not None:
            return await self._answer_open_card(decision)

        if decision.route == front_door.ROUTE_UI_VERB:
            return await self._ui_verb(decision)

        if decision.route in (front_door.ROUTE_CHIT_CHAT, front_door.ROUTE_CAPABILITY):
            return await self._say(decision.text or "", route=decision.route)

        if decision.route == front_door.ROUTE_RULE_ROUTER and decision.plan:
            routed = await self._rule_routed(decision)
            if routed is not None:
                return routed
            # The shape did not survive validation against the real registry.
            # Falling through costs one model call and is still the right answer.
            log.info("runner.router_fell_through", shape=decision.shape)
        return None

    async def _say(
        self,
        text: str,
        *,
        route: str = front_door.ROUTE_CHIT_CHAT,
        tier: int = TIER_TEMPLATE,
        style: str = "template:summary_list",
        status: str = "complete",
    ) -> QueryResult:
        """An answer with no plan behind it."""
        body = (text or "").strip() or "I did not have anything to add there."
        blocks = [_text_block(body)]
        message_id = await self._write_message(blocks)
        await self._settle(status, tier)
        return await self._result(
            message_id=message_id,
            status=status,
            answer=body,
            blocks=blocks,
            style=style,
            route=route,
            tier=tier,
        )

    async def _answer_open_card(self, decision: front_door.Decision) -> QueryResult:
        """The message was an answer to a card already on screen.

        The cheapest path in the system: the plan is on disk, the value is in
        hand, and nothing needs a model.
        """
        answer = decision.answer
        assert answer is not None

        from app.db.repositories import prompts as prompt_repo

        prompt = await prompt_repo.get_prompt(self.session, self.actor.id, answer.prompt_id)
        if prompt is None or str(prompt.status) != "pending":
            return await self._say(
                "That card is not open any more.", route=front_door.ROUTE_OPEN_CARD
            )

        await self._settle("complete", TIER_TEMPLATE)
        result = await _respond(
            self.session,
            self.actor,
            prompt,
            answer.value,
            google=self._google,
            now=self.now,
            registry=self._registry,
        )
        result.route = front_door.ROUTE_OPEN_CARD
        return result

    async def _ui_verb(self, decision: front_door.Decision) -> QueryResult:
        """Show more, retry Calendar, undo. None of these plan anything."""
        verb = decision.verb or ""

        if verb == "retry":
            retried = await retry_run(
                self.session,
                self.actor,
                conversation_id=self.conversation_id,
                service=decision.target,
                google=self._google,
                now=self.now,
                registry=self._registry,
            )
            if retried is not None:
                await self._settle("complete", TIER_TEMPLATE)
                retried.route = front_door.ROUTE_UI_VERB
                return retried
            return await self._say(
                "There is nothing to retry in this conversation yet.",
                route=front_door.ROUTE_UI_VERB,
            )

        if verb in ("cancel", "undo", "stop"):
            cancelled = await self._cancel_open_cards()
            text = (
                f"Cancelled. {cancelled} prepared change"
                f"{'' if cancelled == 1 else 's'} will not happen."
                if cancelled
                else "Nothing was prepared, so there is nothing to undo."
            )
            return await self._say(text, route=front_door.ROUTE_UI_VERB)

        if verb == "sync":
            queued = self._queue_sync()
            text = (
                "A sync is running. Ask again in a moment for fresher results."
                if queued
                else "I could not start a sync just now. Try again shortly."
            )
            return await self._say(text, route=front_door.ROUTE_UI_VERB)

        text = {
            "show_more": "Ask again with a wider range and I will list more.",
            "show_less": "Noted.",
            "open": "Use the link on the item you mean and it opens in Google.",
            "edit": "Use **Edit** on the card and change the text there.",
            "reconnect": render.reconnect_card(),
        }.get(verb, "Done.")
        return await self._say(text, route=front_door.ROUTE_UI_VERB)

    async def _rule_routed(self, decision: front_door.Decision) -> QueryResult | None:
        """A shape the router owns: no probe, no planner, no embedding."""
        self.windows = dict(decision.windows)
        checked = validate.validate_plan(decision.plan or {}, registry=self.registry)
        if not checked.ok:
            log.info(
                "runner.router_plan_rejected", shape=decision.shape, errors=checked.errors
            )
            return None

        self.plan = self._apply_freshness(checked.plan or dict(decision.plan or {}))
        self.intent = dict(decision.intent or self.plan.get("intent") or {})
        self.answer_style = str(self.plan.get("answer_style") or "prose")
        self.planner_tier = TIER_TEMPLATE

        await self._record_intent()
        await self.emit(
            "intent",
            {
                **self.intent,
                "answer_style": self.answer_style,
                "planner_tier": TIER_TEMPLATE,
            },
        )
        await self._write_steps(self.plan["steps"])
        for step in self.plan["steps"]:
            await self.emit("plan.step", self._step_event(step))

        mark = time.perf_counter()
        result = await self._dispatch(self.plan)
        self.timings.dispatch_ms = int((time.perf_counter() - mark) * 1000)

        out = await self._finish(self.plan, result)
        out.route = front_door.ROUTE_RULE_ROUTER
        return out

    # -- probe -------------------------------------------------------------

    async def _run_probe(self) -> None:
        """One embedding plus three hybrid searches. Never fatal.

        A probe that falls over leaves the planner blind, which is worse than a
        probe that works and much better than a dead turn.
        """
        try:
            from app.search.probe import probe as run_probe

            self.probe = await run_probe(
                self.session, self.actor.id, self.prepass, now=self.now
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("runner.probe_failed", run_id=self.run_id, error=str(exc))
            await self._safe_rollback()
            self.probe = None
            self.probe_degraded = [
                {
                    "service": "search",
                    "reason": "probe_failed",
                    "detail": "I could not search this account's own copy of the data",
                }
            ]
            await self.emit(
                "probe.done",
                {
                    "took_ms": 0,
                    "candidates": {},
                    "top": [],
                    "degraded": self.probe_degraded,
                },
            )
            return

        self.probe_bindings = self.probe.to_bindings()
        self.probe_degraded = list(self.probe.degraded or [])
        await self.emit("probe.done", self.probe.to_event())
        await self._write_probe_step()

    async def _write_probe_step(self) -> None:
        """Record the grounding pass as step 0, so the trace shows it too.

        One embedding and three searches is work, and work that never appears in
        the trace is work nobody can check.
        """
        if self.probe is None:
            return
        from app.db.repositories import steps as step_repo

        row = self.probe.to_step()
        try:
            written = await step_repo.insert_steps(
                self.session,
                self.actor.id,
                self.run_id,
                self.conversation_id,
                [row],
                start_seq=0,
            )
            self._seq = 1
            self._written.add((str(row.get("node_id")), 0))
            for record in written:
                self.node_rows[record.node_id] = record.id
                await step_repo.mark_finished(
                    self.session,
                    self.actor.id,
                    record.id,
                    row.get("status", "succeeded"),
                    result=_json(row.get("result")),
                )
            await self.session.commit()
        except Exception as exc:  # noqa: BLE001 - a trace row is not the answer
            log.warning("runner.probe_step_not_written", error=str(exc))
            await self._safe_rollback()

    def _candidates(self) -> dict[str, list[dict[str, Any]]]:
        return {corpus: list(self.probe_bindings.get(corpus) or []) for corpus in CORPORA}

    # -- planning ----------------------------------------------------------

    def _can_call(self) -> bool:
        return self.llm_calls < self.max_calls

    def _route_context(self) -> route.RouteContext:
        from app.ops import registry as ops_registry

        return route.RouteContext(
            query=self.query,
            catalogue=ops_registry.catalogue(),
            now=self.now,
            tz=self.actor.timezone,
            week_start=self.actor.week_start,
            windows={name: w.to_dict() for name, w in self.windows.items()},
            candidates=self._candidates(),
            extracted=dict(self.probe_bindings.get("extracted") or {}),
            ambiguity=self.probe_bindings.get("ambiguity") or None,
            literals=(self.prepass.literals if self.prepass else {}),
            aliases=(self.prepass.aliases if self.prepass else []),
            entities=self._entity_chips,
            history=self._history,
            open_prompts=self._open_prompts,
            staged_actions=self._staged_summaries,
            degraded=[e for e in self.probe_degraded if _is_outage(e)],
            user=self.actor.to_dict(),
        )

    async def _route(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The one planning call, consumed as it streams.

        Three things happen before the closing brace arrives: the intent is
        written and published, each step becomes a pending `node_executions` row
        and a `plan.step` event, and a step that cannot be wrong to have run is
        started.
        """
        await self._gather_context()

        if not self._can_call():
            return {"type": "answer", "text": self._out_of_budget_text()}, {}

        self.planner_tier = TIER_COMPOSED
        self.llm_calls += 1
        ctx = self._route_context()

        payload: dict[str, Any] = {}
        emitted_intent = False
        early: dict[str, asyncio.Task[Any]] = {}
        specs: dict[str, dict[str, Any]] = {}
        rows: list[asyncio.Task[None]] = []

        try:
            async for event in route.route(ctx):
                if event.type == "intent":
                    emitted_intent = True
                    self.intent = dict(event.data)
                    self.answer_style = str(
                        event.data.get("answer_style") or self.answer_style
                    )
                    await self._record_intent()
                    await self.emit(
                        "intent", {**event.data, "planner_tier": self.planner_tier}
                    )
                elif event.type == "step":
                    step = event.data
                    if not isinstance(step, dict):
                        continue
                    # Start the work first. Writing the row is a commit, and a
                    # commit between two independent steps is enough to stagger
                    # them into running one after the other — which is exactly
                    # the serialisation early dispatch exists to avoid.
                    task = await self._start_early(step)
                    if task is not None:
                        early[str(step.get("id"))] = task
                        specs[str(step.get("id"))] = step
                    # The row still goes in as the step arrives — but waiting
                    # for the commit would hold up the next step by the length
                    # of a round trip, and on a five-step plan that is most of a
                    # second of the head start this whole path exists to buy.
                    seq = self._claim_step_row(step)
                    if seq is not None:
                        rows.append(
                            asyncio.create_task(
                                self._write_step_row(step, seq=seq),
                                name=f"row:{step.get('id')}",
                            )
                        )
                    await self.emit("plan.step", self._step_event(step))
                elif event.type == "done":
                    payload = event.data if isinstance(event.data, dict) else {}
        except Exception as exc:  # noqa: BLE001 - the probe still found real things
            # The planning call died. That is one service being down, not the turn
            # being over: the caller degrades to what the probe already has.
            log.warning("runner.planner_failed", run_id=self.run_id, error=str(exc))
            self._planner_error = f"{type(exc).__name__}: {exc}"[:200]
            await asyncio.gather(*rows, return_exceptions=True)
            await self._abandon_steps(early)
            return {}, {}

        # The step-level fields have to be on the args before the early results
        # are matched against the finished plan, or the two sides disagree about
        # what was asked and every early result is thrown away.
        # Every row is on disk before anything looks up a node's row id.
        await asyncio.gather(*rows, return_exceptions=True)

        plan = self._apply_freshness(route.normalise(payload))
        if not emitted_intent and plan.get("intent"):
            self.intent = dict(plan["intent"])
            self.answer_style = str(plan.get("answer_style") or self.answer_style)
            await self._record_intent()
            await self.emit(
                "intent", {**plan["intent"], "answer_style": plan.get("answer_style")}
            )
        return plan, await self._collect_early(early, specs, plan)

    async def _abandon_steps(self, early: dict[str, "asyncio.Task[Any]"]) -> None:
        """Close out a half-written plan, so no row is left saying "pending".

        The probe row is already terminal, so this only touches the steps the
        stream had time to write before it died.
        """
        for task in early.values():
            task.cancel()
        if early:
            await asyncio.gather(*early.values(), return_exceptions=True)
        try:
            from app.db.repositories import steps as step_repo

            await step_repo.cancel_pending(
                self.session, self.actor.id, self.run_id, reason="planner_failed"
            )
            await self.session.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("runner.steps_not_cancelled", error=str(exc))
            await self._safe_rollback()

    async def _gather_context(self) -> None:
        self._entity_chips = await entity_store.recent(
            self.session, self.actor.id, self.conversation_id
        )
        self._history = await self._recent_messages()
        self._open_prompts = [
            {
                "input_id": row.id,
                "kind": str(row.kind),
                "question": (row.prompt or {}).get("question"),
            }
            for row in await self._pending_prompts()
        ]
        self._staged_summaries = await self._staged_actions()

    async def _start_early(self, step: dict[str, Any]) -> "asyncio.Task[Any] | None":
        """Start a step it cannot be wrong to have run.

        Local, no write, no gate, no dependencies, and no reference to anything
        but ambient scope. Those five together mean running it early spends no
        quota, changes nothing, and cannot bind to a value that does not exist
        yet. The worst case is a wasted read of our own database.

        `ask.user` is included on purpose: its whole job is to produce a card,
        and the card should be on screen while the rest of the plan is still
        being written. The pause itself still happens in dispatch, which is what
        keeps the run resumable.
        """
        step_id = str(step.get("id") or "")
        op_name = str(step.get("op") or "")
        if not step_id or not op_name or step.get("depends_on") or step.get("gate"):
            return None

        # `freshness: live` means the planner decided the mirror is not good
        # enough and this has to ask Google. A local op with a live flag is
        # still a call over the network, against a shared quota, that a
        # half-arrived plan might turn out not to want. Not speculative work.
        if self._freshness_of(step) == "live":
            return None

        op = self.registry.get(op_name)
        if op is None or not getattr(op, "is_local", False) or getattr(op, "is_write", False):
            return None

        for reference in validate.references_in(step.get("args")):
            if reference.root not in validate.AMBIENT_ROOTS:
                return None

        # The same bridging dispatch will do later, so the early run and the
        # real one are given identical arguments.
        self._apply_freshness({"steps": [step]})

        try:
            bound = dispatcher.bind(
                step.get("args") or {},
                self._scope(),
                now=self.now,
                tz=self.actor.timezone,
                week_start=self.actor.week_start,
            )
        except dispatcher.BindError:
            return None

        log.info("runner.early_step", run_id=self.run_id, node_id=step_id, op=op_name)
        self._early_args[step_id] = dispatcher.json_safe(bound)
        return asyncio.create_task(
            self._run_early(step_id, op, bound), name=f"early:{step_id}"
        )

    async def _run_early(self, step_id: str, op: Any, bound: dict[str, Any]) -> Any:
        """Run one early step on a session of its own.

        Two things are already using the run's session — the streaming loop
        writing step rows, and any other early step — and one AsyncSession is
        one connection, so statements on it have to be serialised. Serialising
        these would be the wrong fix: two services starting at once is the point
        of running early at all. A short-lived session each keeps them parallel
        and keeps them off the connection the run is writing on.
        """
        from app.db.session import session_scope
        from app.ops.base import OpContext

        # Stamped before the connection is acquired, the way the dispatcher
        # stamps before it waits for a service slot: the step started when the
        # runner admitted it, and waiting for a connection is part of its time.
        started = datetime.now(UTC)
        async with session_scope() as session:
            async with self._google_lock:
                if self._google is None:
                    self._google, self._google_error = await _google_for(
                        session, self.actor.id
                    )
            ctx = OpContext(
                user_id=self.actor.id,
                conversation_id=self.conversation_id,
                run_id=self.run_id,
                session=session,
                google=self._google,
                now=self.now,
                tz=self.actor.timezone,
                probe_embedding=self._probe_embedding(),
                probe_query=self._probe_query(),
            )
            try:
                return await op.run(ctx, bound)
            finally:
                # When it really ran, not when the row was written afterwards.
                # The trace is read to answer "did these two go at once?", and
                # recording time would say no however parallel they were.
                self._early_times[step_id] = (started, datetime.now(UTC))

    async def _collect_early(
        self,
        tasks: dict[str, "asyncio.Task[Any]"],
        specs: dict[str, dict[str, Any]],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep an early result only if the finished plan still asked for it."""
        if not tasks:
            return {}

        final = {str(step.get("id")): step for step in (plan.get("steps") or [])}
        results: dict[str, Any] = {}

        for step_id, task in tasks.items():
            kept = final.get(step_id)
            spec = specs[step_id]
            if (
                kept is None
                or kept.get("op") != spec.get("op")
                or kept.get("args") != spec.get("args")
            ):
                task.cancel()  # the plan changed under it; that work is wasted
                continue
            try:
                op_result = await task
            except asyncio.CancelledError:
                continue
            except Exception as exc:  # noqa: BLE001 - let dispatch run it properly
                log.info("runner.early_step_failed", node_id=step_id, error=str(exc))
                continue

            request = getattr(op_result, "needs_input", None)
            if request is not None:
                # The card goes on the wire now. Its result is deliberately not
                # carried into dispatch: the node has to run there so the run
                # actually pauses on it.
                input_id = await self._raise_input(spec, request)
                self._raised_for_node[step_id] = input_id
                continue
            if getattr(op_result, "needs_replan", False):
                continue
            results[step_id] = dict(getattr(op_result, "data", None) or {})

        await asyncio.gather(*tasks.values(), return_exceptions=True)
        await self._record_early(results, specs)
        return results

    async def _record_early(
        self, results: dict[str, Any], specs: dict[str, dict[str, Any]]
    ) -> None:
        """Write the trace for a step that ran before dispatch started.

        A result handed to the dispatcher as already-completed short-circuits
        inside it and never reaches the step hooks — which is right, it is not
        running anything — so the row and the events are this module's to write.
        Without this an early step sits at `pending` forever and the trace lies
        about work that was actually done.
        """
        for step_id, data in results.items():
            step = specs.get(step_id) or {}
            op_name = str(step.get("op") or "")
            args = self._early_args.get(step_id, {})
            await self.emit(
                "step.started",
                {
                    "node_id": step_id,
                    "op": op_name,
                    "label": _progress_label(self.registry.get(op_name), args),
                    "args": args,
                    "early": True,
                },
            )
            started, finished = self._early_times.get(step_id, (self.now, self.now))
            await self.emit(
                "step.finished",
                {
                    "node_id": step_id,
                    "op": op_name,
                    "status": "succeeded",
                    "duration_ms": int((finished - started).total_seconds() * 1000),
                    "summary": _summarise(data),
                    "early": True,
                },
            )
            await self._write_early_row(step_id, args, data, started, finished)


    async def _write_early_row(
        self,
        node_id: str,
        args: dict[str, Any],
        data: dict[str, Any],
        started: datetime,
        finished: datetime,
    ) -> None:
        """One write for a step that finished before dispatch began."""
        node_row = self.node_rows.get(node_id)
        if not node_row:
            return
        from app.db.repositories import steps as step_repo

        async with self._db_lock:
            try:
                await step_repo.update_step(
                    self.session,
                    self.actor.id,
                    node_row,
                    status="succeeded",
                    args=args,
                    result=_json(data),
                    started_at=started,
                    finished_at=finished,
                )
                await self.session.commit()
            except Exception as exc:  # noqa: BLE001 - a trace row is not the answer
                log.warning("runner.early_row_not_written", node_id=node_id, error=str(exc))
                await self._safe_rollback()

    async def _validated(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], QueryResult | None]:
        """Check the plan. One repair round, then answer from the probe."""
        checked = validate.validate_plan(
            payload, registry=self.registry, known_windows=list(self.windows)
        )
        if checked.ok:
            return checked.plan or payload, None

        log.info("runner.plan_rejected", run_id=self.run_id, errors=checked.errors)
        if not self._can_call():
            return {}, await self._probe_only_answer(checked.errors)

        self.llm_calls += 1
        self.planner_tier = TIER_REPLAN
        try:
            repaired = route.normalise(
                await route.repair(
                    self._route_context(),
                    payload,
                    reason="the validator refused this plan",
                    errors=checked.errors,
                    round_number=1,
                )
            )
        except AppError as exc:
            log.warning("runner.repair_failed", run_id=self.run_id, error=exc.message)
            return {}, await self._probe_only_answer(checked.errors)

        if route.verb_of(repaired) == "answer":
            return {}, await self._answered(repaired)

        second = validate.validate_plan(
            repaired, registry=self.registry, known_windows=list(self.windows)
        )
        if not second.ok:
            log.info("runner.repair_rejected", run_id=self.run_id, errors=second.errors)
            return {}, await self._probe_only_answer([*checked.errors, *second.errors])

        return second.plan or repaired, None

    async def _replan(
        self,
        plan: dict[str, Any],
        result: dispatcher.DispatchResult,
        round_number: int,
    ) -> dict[str, Any] | None:
        self.llm_calls += 1
        try:
            payload = await route.repair(
                self._route_context(),
                plan,
                reason=result.replan_reason or "a step asked to replan",
                completed=_json(result.results),
                round_number=round_number,
            )
        except AppError as exc:
            log.warning("runner.replan_failed", run_id=self.run_id, error=exc.message)
            return None

        candidate = route.normalise(payload)
        if route.verb_of(candidate) != "plan":
            return None
        checked = validate.validate_plan(candidate, registry=self.registry)
        if not checked.ok:
            log.info("runner.replan_rejected", errors=checked.errors)
            return None
        return checked.plan or candidate

    # -- the other three verbs --------------------------------------------

    async def _answered(self, payload: dict[str, Any]) -> QueryResult:
        """`{"type":"answer","text":"..."}` — nothing to run."""
        text = str(payload.get("text") or "").strip()
        if not text:
            text = "I do not have enough to answer that. Tell me a bit more?"
        self.planner_tier = TIER_COMPOSED
        self.intent = {
            **(self.intent or {}),
            "name": (self.intent or {}).get("name") or "direct_answer",
            "answer_style": "prose",
        }
        await self._record_intent()
        return await self._say(text, route="planner_answer", tier=TIER_COMPOSED, style="prose")

    async def _revise(self, payload: dict[str, Any]) -> QueryResult:
        """`{"type":"revise","action_id":"..","patch":{..}}` — edit a draft."""
        from app.db.repositories import actions as action_repo

        action_id = str(payload.get("action_id") or "")
        patch = payload.get("patch")
        if not action_id or not isinstance(patch, dict) or not patch:
            return await self._say(
                "I could not tell which prepared change you meant. Use **Edit** on "
                "the card and I will follow it.",
                route="revise",
                tier=TIER_COMPOSED,
            )
        try:
            action = await action_repo.revise_action(
                self.session, self.actor.id, action_id, dispatcher.json_safe(patch)
            )
            await self.session.commit()
        except AppError as exc:
            await self._safe_rollback()
            return await self._say(exc.message, route="revise", tier=TIER_COMPOSED)

        await self.emit(
            "action.prepared",
            {
                "action_id": action.id,
                "op": action.op,
                "status": str(action.status),
                "requires_input_id": action.requires_input_id,
                "revised": True,
                "preview": dispatcher.json_safe(action.payload),
            },
        )
        result = await self._say(
            "Updated. It still has not been sent — approve the card when it looks right.",
            route="revise",
            tier=TIER_COMPOSED,
            style="card",
        )
        result.action_ids = [action.id]
        result.input_ids = [action.requires_input_id]
        await self._attach_rows(result)
        return result

    async def _answer_input(self, payload: dict[str, Any]) -> QueryResult:
        """`{"type":"answer_input","input_id":"..","value":..}`.

        The planner read the message as an answer to a card the front door did
        not catch. Route it the way the front door would have.
        """
        from app.db.repositories import prompts as prompt_repo

        input_id = str(payload.get("input_id") or "")
        prompt = (
            await prompt_repo.get_prompt(self.session, self.actor.id, input_id)
            if input_id
            else None
        )
        if prompt is None or str(prompt.status) != "pending":
            return await self._say("That card is not open any more.", route="answer_input")

        await self._settle("complete", TIER_COMPOSED)
        result = await _respond(
            self.session,
            self.actor,
            prompt,
            payload.get("value"),
            google=self._google,
            now=self.now,
            registry=self._registry,
        )
        result.route = "answer_input"
        return result

    # -- dispatch ----------------------------------------------------------

    def _freshness_of(self, step: dict[str, Any]) -> str:
        """Cached or live for one step, with the run-wide override on top."""
        if self.freshness:
            return self.freshness
        return "live" if str(step.get("freshness") or "cached") == "live" else "cached"

    def _apply_freshness(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Push the step-level fields down into the args the op actually reads.

        The plan writes `expect` and `freshness` beside the step, because they
        are properties of the step. The ops take both as arguments, because that
        is where an op's inputs live. Nothing in between copies them across, so
        this does. Without it they are words in the plan that change nothing:
        `freshness: "live"` would state tomorrow's meeting time from a mirror
        that is fifteen minutes stale, and `expect: "one"` would leave a single
        result wrapped in a list, so `{{booking.extracted.pnr}}` — the whole
        reason the planner asked for one — has nothing to resolve against.
        """
        for step in plan.get("steps") or []:
            fresh = self._freshness_of(step)
            step["freshness"] = fresh
            expect = "one" if str(step.get("expect") or "many") == "one" else "many"
            step["expect"] = expect

            op = self.registry.get(str(step.get("op") or ""))
            fields = getattr(getattr(op, "args_model", None), "model_fields", {}) or {}
            args = step.setdefault("args", {})
            if not isinstance(args, dict):
                continue
            # A value the planner wrote into args wins, except where the person
            # asked for fresh data — that is not the planner's call to make.
            if "freshness" in fields and (self.freshness or "freshness" not in args):
                args["freshness"] = fresh
            if "expect" in fields and "expect" not in args:
                args["expect"] = expect
        return plan

    def _scope(self) -> dict[str, Any]:
        """The bag every `{{...}}` reference resolves against."""
        return {
            "windows": {name: w.to_dict() for name, w in self.windows.items()},
            "search": self.probe_bindings,
            "probe": self.probe_bindings,
            "user": self.actor.to_dict(),
            "now": self.now,
            "tz": self.actor.timezone,
            "run": {"id": self.run_id, "query": self.query},
            # The chips the planner was shown, resolvable the same way. A plan
            # that says {{conversation.entities[0].meta.draft_id}} is reading
            # a value it was literally given — refusing to bind it turned
            # "send it" into a fault.
            "conversation": {
                "id": self.conversation_id,
                "entities": list(self._entity_chips or []),
            },
            "intent": dict(self.intent or {}),
            "steps": {},
        }

    async def _dispatch(
        self, plan: dict[str, Any], *, completed: dict[str, Any] | None = None
    ) -> dispatcher.DispatchResult:
        ctx = await self.op_context()
        return await dispatcher.dispatch(
            plan,
            ctx,
            scope=self._scope(),
            registry=self.registry,
            hooks=self._hooks(),
            completed=completed,
            week_start=self.actor.week_start,
        )

    def _hooks(self) -> dispatcher.Hooks:
        return dispatcher.Hooks(
            on_event=self.emit,
            on_step_started=self._on_step_started,
            on_step_finished=self._on_step_finished,
            on_retry=self._on_retry,
            raise_input=self._raise_input,
            stage_write=self._stage_write,
        )

    async def _on_step_started(self, outcome: dispatcher.StepOutcome) -> None:
        await self.emit(
            "step.started",
            {
                "node_id": outcome.node_id,
                "op": outcome.op,
                "label": _progress_label(self.registry.get(outcome.op), outcome.args),
                "args": outcome.args,
            },
        )
        await self._persist_step(outcome.node_id, started=True, args=outcome.args)

    def _is_parked(self, outcome: dispatcher.StepOutcome) -> bool:
        """Is this step waiting on a card rather than skipped?

        The dispatcher marks everything behind a blocking question `skipped`,
        which is the right in-memory answer — it did not run — and the wrong
        thing to write down. A resume rebuilds the plan from these rows, and a
        row saying `skipped` reads as "we tried and gave up" when the truth is
        "nobody has answered yet". Only the two pause-shaped reasons qualify; a
        gate that was not met, or a dependency that genuinely failed, stays
        skipped.
        """
        if outcome.status != "skipped":
            return False
        detail = outcome.outcome or {}
        reason = str(detail.get("reason") or "")
        if reason == "run_paused":
            return True
        return reason == "dependency_failed" and str(
            detail.get("depends_on") or ""
        ) in self._parked_nodes

    async def _on_step_finished(self, outcome: dispatcher.StepOutcome) -> None:
        parked = self._is_parked(outcome)
        if parked:
            self._parked_nodes.add(outcome.node_id)

        status = "pending" if parked else outcome.status
        await self.emit(
            "step.finished",
            {
                "node_id": outcome.node_id,
                "op": outcome.op,
                "status": status,
                "duration_ms": outcome.duration_ms,
                "attempts": outcome.attempts,
                "summary": _summarise(outcome.result),
                "outcome": None if parked else outcome.outcome,
                "waiting": parked,
            },
        )
        if parked:
            # Left exactly as it was written: pending, with no result and no
            # outcome, which is what resuming needs to find.
            return
        await self._persist_step(
            outcome.node_id,
            status=outcome.status,
            result=outcome.result,
            outcome=outcome.outcome,
        )

    async def _on_retry(
        self, outcome: dispatcher.StepOutcome, detail: dict[str, Any]
    ) -> None:
        service = service_of(outcome.op)
        await self.emit(
            "step.retrying",
            {
                "node_id": outcome.node_id,
                "op": outcome.op,
                "attempt": detail.get("attempt"),
                "of": detail.get("of"),
                "error_class": detail.get("error_class"),
                "google_status": detail.get("google_status"),
                "backoff_ms": detail.get("backoff_ms"),
                "label": (
                    f"{render.SERVICE_NAMES.get(service, 'That service')} hiccuped, "
                    "trying once more"
                ),
            },
        )
        node_row = self.node_rows.get(outcome.node_id)
        if not node_row:
            return
        async with self._db_lock:
            try:
                from app.db.repositories import steps as step_repo

                await step_repo.record_retry(
                    self.session,
                    self.actor.id,
                    node_row,
                    error_class=str(detail.get("error_class") or "UNKNOWN"),
                    google_status=detail.get("google_status"),
                    backoff_ms=detail.get("backoff_ms"),
                )
                await self.session.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("runner.retry_not_recorded", error=str(exc))
                await self._safe_rollback()

    async def _persist_step(
        self,
        node_id: str,
        *,
        started: bool = False,
        status: str | None = None,
        args: dict[str, Any] | None = None,
        result: Any = None,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        """Write the step row now, not at the end of the run.

        The step trace reads this table live, and a worker that dies has to
        leave behind enough for another one to pick the run up.
        """
        node_row = self.node_rows.get(node_id)
        if not node_row:
            return
        from app.db.repositories import steps as step_repo

        async with self._db_lock:
            try:
                if started:
                    await step_repo.mark_started(
                        self.session, self.actor.id, node_row, args=_json(args)
                    )
                elif status in ("succeeded", "failed", "skipped", "timeout", "cancelled"):
                    await step_repo.mark_finished(
                        self.session,
                        self.actor.id,
                        node_row,
                        status,
                        result=_json(result),
                        outcome=outcome,
                    )
                elif status is not None:
                    # The paused node: still running, waiting on a person.
                    await step_repo.update_step(
                        self.session,
                        self.actor.id,
                        node_row,
                        status=status,
                        result=_json(result),
                        outcome=outcome,
                    )
                await self.session.commit()
            except Exception as exc:  # noqa: BLE001 - a trace row is not the answer
                log.warning("runner.step_not_persisted", node_id=node_id, error=str(exc))
                await self._safe_rollback()

    # -- questions and staged writes ---------------------------------------

    async def _raise_input(self, step: dict[str, Any], request: Any) -> str:
        """Buffer a question and put it on the wire straight away.

        The id is allocated here rather than by the database so the card can be
        on screen while the rows are still being written — and the row, when it
        lands, carries the id the browser already has.
        """
        node_id = str(step.get("id") or "")
        existing = self._raised_for_node.get(node_id)
        if existing:
            return existing  # raised early; do not ask the same thing twice

        input_id = new_id()
        kind = str(_field(request, "kind", "confirm"))
        blocking = bool(_field(request, "blocking", True))
        entry = {
            "id": input_id,
            "kind": kind,
            "blocking": blocking,
            "node_id": node_id,
            "op": str(step.get("op") or ""),
            "prompt": {
                "question": _field(request, "question", ""),
                "help_text": _field(request, "help_text", None),
                "fields": _field(request, "fields", None),
            },
            "value_schema": _field(request, "value_schema", {}) or {},
            "options": _field(request, "options", None),
        }
        self.pending_inputs.append(entry)
        self._raised_for_node[node_id] = input_id
        if blocking:
            # Everything waiting on this node is parked, not skipped.
            self._parked_nodes.add(node_id)

        await self.emit(
            "input.raised",
            {
                "input_id": input_id,
                "kind": kind,
                "blocking": blocking,
                "node_id": node_id,
                "prompt": entry["prompt"],
                "value_schema": entry["value_schema"],
                "options": entry["options"],
            },
        )
        return input_id

    async def _stage_write(self, step: dict[str, Any], staged: dict[str, Any]) -> str:
        """Prepare a write. Nothing is sent, moved or deleted here.

        Every staged action gets a confirm card: `actions.requires_input_id` is
        NOT NULL, so the database itself refuses a write with no gate. It is not
        unique, so one card can gate two writes that run in order.
        """
        from app.db.repositories import actions as action_repo

        op_name = str(staged.get("op") or step.get("op") or "")
        payload = dispatcher.json_safe(staged.get("payload") or {})
        question = staged.get("question") or _default_question(op_name, staged.get("preview"))
        dedupe = fingerprint_parts(
            "action.dedupe", self.actor.id, op_name, payload, self.conversation_id
        )

        # An identical write already waiting for a yes is the same write. Point
        # the card at the row that exists rather than making a second one — and
        # find that out now, so the id in the event is the id that lands.
        async with self._db_lock:
            existing = await action_repo.find_in_flight(self.session, self.actor.id, dedupe)
        if existing is not None:
            self.actions.append(
                {
                    "id": existing.id,
                    "node_id": str(step.get("id") or ""),
                    "op": op_name,
                    "payload": payload,
                    "preview": staged.get("preview"),
                    "question": question,
                    "external_ref": existing.external_ref,
                    "requires_input_id": existing.requires_input_id,
                    "dedupe": dedupe,
                    "exists": True,
                }
            )
            await self.emit(
                "action.prepared",
                {
                    "action_id": existing.id,
                    "op": existing.op,
                    "status": str(existing.status),
                    "requires_input_id": existing.requires_input_id,
                    "already_staged": True,
                    "preview": staged.get("preview"),
                },
            )
            return existing.id

        gate = self._confirm_gate(step, question)
        if gate.pop("new", False):
            await self.emit(
                "input.raised",
                {
                    "input_id": gate["id"],
                    "kind": "confirm",
                    "blocking": False,
                    "node_id": gate["node_id"],
                    "prompt": gate["prompt"],
                    "value_schema": gate["value_schema"],
                },
            )

        action_id = new_id()
        self.actions.append(
            {
                "id": action_id,
                "node_id": str(step.get("id") or ""),
                "op": op_name,
                "payload": payload,
                "preview": staged.get("preview"),
                "question": question,
                "external_ref": staged.get("external_ref"),
                "requires_input_id": gate["id"],
                "dedupe": dedupe,
                "exists": False,
            }
        )
        await self.emit(
            "action.prepared",
            {
                "action_id": action_id,
                "op": op_name,
                "status": "draft",
                "requires_input_id": gate["id"],
                "external_ref": staged.get("external_ref"),
                "preview": staged.get("preview"),
            },
        )
        return action_id

    def _confirm_gate(self, step: dict[str, Any], question: str) -> dict[str, Any]:
        """The card gating a write. Reused when one plan stages two of them."""
        for entry in self.pending_inputs:
            if entry["kind"] == "confirm" and not entry["blocking"]:
                return entry

        gate = {
            "id": new_id(),
            "kind": "confirm",
            "blocking": False,  # the run finishes; this waits on a yes
            "node_id": str(step.get("id") or ""),
            "op": str(step.get("op") or ""),
            "prompt": {
                "question": question,
                "help_text": "Nothing has been sent. Approving is what makes it happen.",
            },
            "value_schema": CONFIRM_SCHEMA,
            "options": None,
            "new": True,
        }
        self.pending_inputs.append(gate)
        return gate

    # -- finishing ---------------------------------------------------------

    def _degraded_context(self, result: dispatcher.DispatchResult) -> dict[str, Any]:
        """What failed, from the dispatcher and from the probe together.

        A corpus that died during the probe cost the answer just as much as a
        step that failed, so the banner has to name both.
        """
        context = result.degraded_context()
        failed = list(context.get("failed") or [])
        for entry in self.probe_degraded:
            # A stale corpus still answered. It belongs in the freshness note,
            # not in the list of things that failed.
            if not _is_outage(entry):
                continue
            failed.append(
                {
                    "node": "search",
                    "service": entry.get("service"),
                    "class": entry.get("class"),
                    "code": entry.get("code"),
                    "attempts": 1,
                    "message": entry.get("detail") or entry.get("reason"),
                }
            )
        if self._google_error is not None and any(
            o.status in ("failed", "timeout") for o in result.outcomes.values()
        ):
            context["google"] = self._google_error.code
        context["failed"] = failed
        context["degraded"] = bool(failed or context.get("skipped"))
        return context

    def _degraded_services(self, result: dispatcher.DispatchResult) -> list[str]:
        names = set(result.failed_services)
        for entry in self.probe_degraded:
            if not _is_outage(entry):
                continue
            service = str(entry.get("service") or "").strip()
            if service:
                names.add(service)
        return sorted(names)

    async def _finish(
        self, plan: dict[str, Any], result: dispatcher.DispatchResult
    ) -> QueryResult:
        mark = time.perf_counter()
        degraded = self._degraded_context(result)
        paused = result.status == "paused" and result.paused_on is not None
        if not paused and await self._park_on_reconnect(degraded):
            paused = True

        style = self.answer_style
        if paused or self.actions:
            # A card on screen is the answer. Prose about a card is noise.
            style = "card"
        if style == "prose" and not self._can_call():
            style = "template:summary_list"

        rendered = await render.render(
            style,
            query=self.query,
            results=result.results,
            steps=plan.get("steps") or [],
            intent=self.intent,
            windows={name: w.to_dict() for name, w in self.windows.items()},
            degraded=degraded,
            staged=result.staged,
            action_ids=[a["id"] for a in self.actions],
            input_ids=[p["id"] for p in self.pending_inputs],
            tz=self.actor.timezone,
            now=self.now,
            history=self._history,
            on_delta=self._on_delta if style == "prose" else None,
        )
        self.llm_calls += rendered.llm_calls
        self.timings.render_ms = int((time.perf_counter() - mark) * 1000)

        text = rendered.text.strip()
        blocks = list(rendered.blocks)
        if self._budget_note:
            text = f"{text}\n\n{self._budget_note}".strip()
            blocks = [_text_block(text)] + [b for b in blocks if b.get("type") != "text"]
        if not text and not blocks:
            # Never an empty body. A run that timed out still says what it got.
            text = self._nothing_to_show(result)
            blocks = [_text_block(text)]

        input_ids = [p["id"] for p in self.pending_inputs]
        action_ids = [a["id"] for a in self.actions]
        message_id = await self._write_message(blocks)

        chips = await entity_store.record(
            self.session,
            self.actor.id,
            self.conversation_id,
            results=result.results,
            steps=plan.get("steps") or [],
            run_id=self.run_id,
        )

        status = self._status_of(result, paused)
        await self._settle(status, self.planner_tier)

        if paused and result.paused_on is not None:
            await self.emit(
                "run.paused",
                {
                    "reason": "awaiting_input",
                    "input_id": result.paused_on.input_id,
                    "node_id": result.paused_on.node_id,
                    "completed_nodes": sum(1 for o in result.outcomes.values() if o.ok),
                    "remaining_nodes": sum(
                        1
                        for o in result.outcomes.values()
                        if o.status in ("pending", "running")
                    ),
                },
            )

        return await self._result(
            message_id=message_id,
            status=status,
            answer=text,
            blocks=blocks,
            style=rendered.style,
            route=front_door.ROUTE_MISS,
            tier=self.planner_tier,
            steps=[o.to_dict() for o in result.outcomes.values()],
            degraded=self._degraded_services(result),
            degraded_detail=[
                *(e for e in self.probe_degraded if _is_outage(e)),
                *result.degraded,
            ],
            input_ids=input_ids,
            action_ids=action_ids,
            entities=chips,
        )

    #: Failure classes and error codes that mean the grant itself is gone.
    _REAUTH_MARKERS = frozenset(
        {"AUTH_REVOKED", "AUTH_EXPIRED", "GOOGLE_REAUTH_REQUIRED"}
    )

    async def _park_on_reconnect(self, degraded: dict[str, Any]) -> bool:
        """Park on a reconnect card when the Google grant is dead.

        A revoked or unrefreshable token is not a degraded answer. Every other
        outage is something the person cannot do anything about, so the honest
        response is a partial answer that says what is missing — but this one
        they can fix in about ten seconds, and the plan is still good. So the
        run parks the way it parks on any other question, and reconnecting then
        answering the card resumes it for nothing.

        Returns whether a card was raised.
        """
        if any(p.get("blocking") for p in self.pending_inputs):
            return False  # a question is already on screen; do not stack another
        culprit = next(
            (
                entry
                for entry in (degraded.get("failed") or [])
                if str(entry.get("class") or "").upper() in self._REAUTH_MARKERS
                or str(entry.get("code") or "").upper() in self._REAUTH_MARKERS
            ),
            None,
        )
        if culprit is None:
            return False

        await self._raise_input(
            {"id": "reconnect", "op": "auth.reconnect"},
            {
                "kind": "confirm",
                # Parked, not asking for approval. The run is held so no work
                # is lost, but "Send it / Not now" is the wrong thing to put
                # under "your connection expired" — the only useful action is
                # to go and reconnect.
                "variant": "reconnect",
                "blocking": True,
                "question": render.reconnect_card(),
                "value_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["approve"],
                    "properties": {"approve": {"type": "boolean"}},
                },
            },
        )
        log.info(
            "runner.parked_on_reconnect",
            run_id=self.run_id,
            service=culprit.get("service"),
        )
        return True

    def _status_of(self, result: dispatcher.DispatchResult, paused: bool) -> str:
        """What this run is, once there is an answer written for it.

        A step that failed does not fail the run. Calendar going down costs the
        calendar half of an answer, and the other half is still worth having —
        so the run is `complete` and `degraded` names what is missing. The only
        two exceptions are a run waiting on a person, and a run that ran out of
        time with nothing to show; a crash never reaches here, because a crash
        goes through `_fail`.
        """
        if paused:
            return "awaiting_input"
        outcomes = list(result.outcomes.values())
        ran_out = [
            o
            for o in outcomes
            if o.status == "timeout" and (o.outcome or {}).get("reason") == "hard_deadline"
        ]
        if ran_out and not any(o.ok for o in outcomes):
            return "timeout"
        return "complete"

    def _nothing_to_show(self, result: dispatcher.DispatchResult) -> str:
        """The last-resort sentence. It still has to be true."""
        failed = self._degraded_services(result)
        if failed:
            names = ", ".join(render.SERVICE_NAMES.get(s, s) for s in failed)
            return (
                f"{names} did not answer, so I have nothing to show for that one. "
                "Say **retry** and I will try again."
            )
        return "I did not find anything for that."

    async def _probe_only_answer(
        self,
        errors: Sequence[str],
        *,
        lead: str = "",
        service: str = "planner",
        reason: str = "plan_rejected",
    ) -> QueryResult:
        """The planner could not give us a runnable plan.

        Either it was refused twice by the validator or the call itself fell
        over. Whichever it was, what the probe already found is real, so that is
        what the answer is built from. No further model calls.
        """
        log.info("runner.degraded_to_probe", run_id=self.run_id, errors=list(errors)[:4])
        self.planner_tier = TIER_REPLAN
        results = {
            f"search_{corpus}": {"hits": hits}
            for corpus, hits in self._candidates().items()
            if hits
        }

        mark = time.perf_counter()
        rendered = await render.render(
            "template:summary_list",
            query=self.query,
            results=results,
            intent=self.intent,
            windows={name: w.to_dict() for name, w in self.windows.items()},
            tz=self.actor.timezone,
            now=self.now,
        )
        self.timings.render_ms = int((time.perf_counter() - mark) * 1000)

        opener = lead or (
            "I could not work out a safe set of steps for that, so here is what turned "
            "up in your own mail, calendar and files. Ask again more specifically and "
            "I will do better."
        )
        text = f"{opener}\n\n{rendered.text}".strip() if rendered.text else opener
        blocks = [_text_block(text)]

        message_id = await self._write_message(blocks)
        await self._settle("complete", self.planner_tier)
        return await self._result(
            message_id=message_id,
            status="complete",
            answer=text,
            blocks=blocks,
            style="template:summary_list",
            route="degraded",
            tier=self.planner_tier,
            degraded=sorted({service, *self._degraded_names()}),
            degraded_detail=[
                *(e for e in self.probe_degraded if _is_outage(e)),
                {
                    "service": service,
                    "reason": reason,
                    "detail": "; ".join(list(errors)[:3]),
                },
            ],
        )

    def _degraded_names(self) -> list[str]:
        return [
            str(entry.get("service") or "").strip()
            for entry in self.probe_degraded
            if entry.get("service")
        ]

    def _out_of_budget_text(self) -> str:
        return (
            "This turn has used the thinking budget it is allowed. Tell me which part "
            "you want first and I will do that one."
        )

    async def _on_delta(self, piece: str) -> None:
        await self.emit(
            "content.delta",
            {"message_id": self.assistant_message_id, "block_index": 0, "text": piece},
        )

    async def _fail(self, exc: AppError) -> QueryResult:
        """The failure path. It still writes a message and still tells the truth."""
        await self._safe_rollback()
        text = exc.message or "Something went wrong on that one."
        blocks = [_text_block(text)]
        message_id = ""
        try:
            message_id = await self._write_message(blocks)
        except Exception:  # noqa: BLE001 - the failure path must not fail
            log.warning("runner.fail_message_not_written", run_id=self.run_id)
            await self._safe_rollback()
        try:
            from app.db.repositories import runs as run_repo

            await run_repo.mark_failed(
                self.session,
                self.actor.id,
                self.run_id,
                {"code": exc.code, "message": exc.message, "details": exc.details},
            )
            await self.session.commit()
        except Exception:  # noqa: BLE001
            log.warning("runner.fail_not_recorded", run_id=self.run_id)
            await self._safe_rollback()

        return QueryResult(
            conversation_id=self.conversation_id,
            message_id=message_id,
            run_id=self.run_id,
            status="failed",
            answer=text,
            content=blocks,
            intent=self.intent,
            planner_tier=self.planner_tier,
            timings=self.timings.to_dict(),
            usage={"llm_calls": self.llm_calls},
            route="failed",
        )

    async def _result(
        self,
        *,
        message_id: str,
        status: str,
        answer: str,
        blocks: list[dict[str, Any]],
        style: str,
        route: str,
        tier: int,
        steps: list[dict[str, Any]] | None = None,
        degraded: list[str] | None = None,
        degraded_detail: list[dict[str, Any]] | None = None,
        input_ids: list[str] | None = None,
        action_ids: list[str] | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> QueryResult:
        result = QueryResult(
            conversation_id=self.conversation_id,
            message_id=message_id,
            run_id=self.run_id,
            status=status,
            answer=answer,
            content=blocks,
            intent=self.intent,
            steps=steps or [],
            degraded=degraded or [],
            degraded_detail=degraded_detail or [],
            usage={"llm_calls": self.llm_calls},
            timings=self.timings.to_dict(),
            answer_style=style,
            planner_tier=tier,
            input_ids=input_ids or [],
            action_ids=action_ids or [],
            entities=entities or [],
            route=route,
        )
        await self._attach_rows(result)
        return result

    async def _attach_rows(self, result: QueryResult) -> None:
        """Read the cards and actions back, so the response has their real state."""
        result.prompts, result.actions = await _read_back(
            self.session, self.actor.id, result.input_ids, result.action_ids
        )

    # -- persistence -------------------------------------------------------

    async def _write_message(self, blocks: list[dict[str, Any]]) -> str:
        """The assistant message and every row it points at. One transaction.

        The message goes first because `pending_inputs.message_id` and
        `actions.message_id` are both NOT NULL. Their ids were allocated when the
        write was staged, so the blocks already reference rows that are about to
        exist.
        """
        from app.db.repositories import actions as action_repo
        from app.db.repositories import conversations as conv_repo
        from app.db.repositories import prompts as prompt_repo
        from app.db.repositories import steps as step_repo

        message = await conv_repo.add_message(
            self.session,
            self.actor.id,
            self.conversation_id,
            role="assistant",
            content=blocks,
            run_id=self.run_id,
            message_id=self.assistant_message_id,
        )

        for entry in self.pending_inputs:
            await prompt_repo.create_prompt(
                self.session,
                self.actor.id,
                self.run_id,
                message.id,
                kind=entry["kind"],
                prompt=entry["prompt"],
                value_schema=entry["value_schema"],
                options=entry.get("options"),
                blocking=entry["blocking"],
                node_execution_id=self.node_rows.get(entry.get("node_id") or ""),
                conversation_id=self.conversation_id,
                op=entry.get("op"),
                prompt_id=entry["id"],
                expires_at=self.now + timedelta(hours=24),
            )

        for entry in self.actions:
            if entry.get("exists"):
                continue  # an identical write is already staged and already gated
            await action_repo.create_action(
                self.session,
                self.actor.id,
                message.id,
                requires_input_id=entry["requires_input_id"],
                op=entry["op"],
                payload=entry["payload"],
                dedupe_key=entry["dedupe"],
                node_execution_id=self.node_rows.get(entry.get("node_id") or ""),
                external_ref=entry.get("external_ref"),
                action_id=entry["id"],
            )

        # Which message reports which steps. Passing no list claims every step of
        # the run no message has claimed yet, which is exactly what a paused run
        # and then a resumed one need.
        await step_repo.attach_message(self.session, self.actor.id, self.run_id, message.id)

        await self.session.commit()

        self.pending_inputs = []
        self.actions = []
        # A run can produce two messages: the card, then the answer.
        self.assistant_message_id = new_id()
        return message.id

    def _claim_step_row(self, step: dict[str, Any], round_number: int = 0) -> int | None:
        """Take this step's place in the trace, without touching the database.

        Allocating the sequence number here rather than inside the write keeps
        the rows in plan order even though the writes themselves run loose.
        """
        node_id = str(step.get("id") or step.get("node_id") or "")
        if not node_id or not str(step.get("op") or ""):
            return None
        if (node_id, round_number) in self._written:
            return None  # the planner repeated an id; the validator will say so
        self._written.add((node_id, round_number))
        seq = self._seq
        self._seq += 1
        return seq

    async def _write_step_row(
        self, step: dict[str, Any], round_number: int = 0, seq: int | None = None
    ) -> None:
        """One pending `node_executions` row, the moment its object closes."""
        if seq is None:
            seq = self._claim_step_row(step, round_number)
            if seq is None:
                return
        node_id = str(step.get("id") or step.get("node_id") or "")

        from app.db.repositories import steps as step_repo

        async with self._db_lock:
            rows: list[Any] = []
            try:
                async with self.session.begin_nested():
                    rows = await step_repo.insert_steps(
                        self.session,
                        self.actor.id,
                        self.run_id,
                        self.conversation_id,
                        [step],
                        round=round_number,
                        start_seq=seq,
                    )
            except (IntegrityError, AppError) as exc:
                # The savepoint has already undone the insert; nothing else to
                # clean up here. Reaching into `session.new` to detach things
                # by hand — which an earlier version of this did — takes out
                # unrelated pending rows and leaves the connection confused.
                log.warning("runner.step_row_rejected", node_id=node_id, error=str(exc))
                return

            try:
                for row in rows:
                    self.node_rows[row.node_id] = row.id
                await self.session.commit()
            except (IntegrityError, AppError) as exc:
                # The commit is the other half of the same risk and used to sit
                # outside the guard. A step row is bookkeeping for the trace —
                # losing one costs a line in a panel almost nobody opens. It
                # must never cost somebody their answer.
                log.warning(
                    "runner.step_row_commit_failed", node_id=node_id, error=str(exc)
                )

    async def _write_steps(
        self, steps: Sequence[dict[str, Any]], *, round_number: int = 0
    ) -> None:
        for step in steps:
            await self._write_step_row(step, round_number)

    async def _sync_step_rows(self, steps: Sequence[dict[str, Any]]) -> None:
        """Write any step the stream missed, and store the validator's fixes.

        The validator hands back a corrected copy — the singular/plural reference
        fix lives there — so the row has to hold what will actually run, not what
        first arrived.
        """
        from app.db.repositories import steps as step_repo

        for step in steps:
            node_id = str(step.get("id") or "")
            if not node_id:
                continue
            if (node_id, 0) not in self._written:
                await self._write_step_row(step)
                continue
            node_row = self.node_rows.get(node_id)
            if not node_row:
                continue
            try:
                await step_repo.update_step(
                    self.session,
                    self.actor.id,
                    node_row,
                    args=dispatcher.json_safe(step.get("args") or {}),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("runner.step_args_not_updated", node_id=node_id, error=str(exc))
                await self._safe_rollback()
        await self.session.commit()

    async def _record_intent(self) -> None:
        from app.db.repositories import runs as run_repo

        intent = dict(self.intent or {})
        intent["windows"] = {name: w.to_dict() for name, w in self.windows.items()}
        if self.windows and not intent.get("resolved_window"):
            name, window = next(iter(self.windows.items()))
            intent["resolved_window"] = {"name": name, **window.to_dict()}
        # Carried so a resumed run can rebuild without re-reading the thread.
        intent["query"] = self.query
        intent["answer_style"] = self.answer_style
        self.intent = intent
        try:
            await run_repo.set_intent(
                self.session,
                self.actor.id,
                self.run_id,
                dispatcher.json_safe(intent),
                planner_tier=self.planner_tier,
            )
            await self.session.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("runner.intent_not_recorded", error=str(exc))
            await self._safe_rollback()

    async def _settle(self, status: str, tier: int) -> None:
        from app.db.repositories import runs as run_repo

        self.planner_tier = tier
        try:
            await run_repo.set_planner_tier(self.session, self.actor.id, self.run_id, tier)
            await run_repo.set_status(self.session, self.actor.id, self.run_id, status)
            await self.session.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("runner.status_not_recorded", status=status, error=str(exc))
            await self._safe_rollback()

    # -- reads -------------------------------------------------------------

    async def _pending_prompts(self) -> list[Any]:
        if not self.conversation_id:
            return []
        from app.db.repositories import prompts as prompt_repo

        return await prompt_repo.list_prompts(
            self.session,
            self.actor.id,
            status="pending",
            conversation_id=self.conversation_id,
            limit=5,
        )

    async def _staged_actions(self) -> list[dict[str, Any]]:
        from app.db.repositories import actions as action_repo

        rows = await action_repo.list_actions(
            self.session,
            self.actor.id,
            status="draft",
            conversation_id=self.conversation_id or None,
            limit=5,
        )
        return [{"action_id": row.id, "op": row.op, "status": str(row.status)} for row in rows]

    async def _last_intent(self) -> dict[str, Any] | None:
        """The previous turn's intent — what a bare "next Tuesday" hangs off."""
        if not self.conversation_id:
            return None
        from app.db.repositories import runs as run_repo

        previous = await run_repo.latest_run(self.session, self.actor.id, self.conversation_id)
        if previous is None or previous.id == self.run_id or not previous.intent:
            return None
        return {**previous.intent, "run_id": previous.id}

    async def _recent_messages(self, limit: int = 8) -> list[dict[str, Any]]:
        if not self.conversation_id:
            return []
        from app.db.repositories import conversations as conv_repo

        rows = await conv_repo.list_recent_messages(
            self.session, self.actor.id, self.conversation_id, limit=limit
        )
        return [
            {"role": str(row.role), "text": _flatten(row.content)}
            for row in rows
            if row.id != self.trigger_message_id
        ]

    async def _cancel_open_cards(self) -> int:
        from app.db.repositories import actions as action_repo
        from app.db.repositories import prompts as prompt_repo

        cancelled = 0
        for prompt in await self._pending_prompts():
            cancelled += await action_repo.cancel_for_prompt(
                self.session, self.actor.id, prompt.id, reason="user_cancelled"
            )
            try:
                await prompt_repo.cancel_prompt(self.session, self.actor.id, prompt.id)
            except AppError:
                continue
        await self.session.commit()
        return cancelled

    def _queue_sync(self) -> bool:
        try:
            from app.tasks.celery_app import celery_app

            celery_app.send_task("sync.dispatch_user", args=[self.actor.id], queue="sync")
            return True
        except Exception as exc:  # noqa: BLE001 - the answer says so either way
            log.info("runner.sync_not_queued", error=str(exc))
            return False

    def _step_event(self, step: dict[str, Any], *, round_number: int = 0) -> dict[str, Any]:
        op = self.registry.get(str(step.get("op") or ""))
        return {
            "node_id": step.get("id"),
            "op": step.get("op"),
            "round": round_number,
            "depends_on": step.get("depends_on") or [],
            "expect": step.get("expect", "many"),
            "optional": bool(step.get("optional", False)),
            "freshness": step.get("freshness", "cached"),
            "gate": step.get("gate"),
            "is_write": bool(getattr(op, "is_write", False)),
            "needs_confirm": bool(getattr(op, "needs_confirm", False)),
            "label": _progress_label(op, step.get("args") or {}),
        }


# ---------------------------------------------------------------------------
# Reading cards and actions back
# ---------------------------------------------------------------------------


async def _read_back(
    session: Any,
    user_id: str,
    input_ids: Iterable[str],
    action_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The cards and actions this turn produced, in their current state."""
    inputs = [i for i in input_ids if i]
    actions = [a for a in action_ids if a]
    if not inputs and not actions:
        return [], []

    from app.db.repositories import actions as action_repo
    from app.db.repositories import prompts as prompt_repo

    prompt_rows = []
    for input_id in inputs:
        row = await prompt_repo.get_prompt(session, user_id, input_id)
        if row is not None:
            prompt_rows.append(row)

    action_rows = []
    for action_id in actions:
        row = await action_repo.get_action(session, user_id, action_id)
        if row is not None:
            action_rows.append(row)

    return (
        [
            {
                "id": row.id,
                "kind": str(row.kind),
                "status": str(row.status),
                "blocking": bool(row.blocking),
                "prompt": row.prompt,
                "value_schema": row.value_schema,
                "options": row.options,
                "run_id": row.run_id,
                "message_id": row.message_id,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            }
            for row in prompt_rows
        ],
        [
            {
                "id": row.id,
                "op": row.op,
                "status": str(row.status),
                "payload": row.payload,
                "requires_input_id": row.requires_input_id,
                "external_ref": row.external_ref,
                "message_id": row.message_id,
            }
            for row in action_rows
        ],
    )


# ---------------------------------------------------------------------------
# 1. A new query
# ---------------------------------------------------------------------------


async def handle_query(
    session: Any,
    *,
    user_id: str,
    text: str,
    conversation_id: str | None = None,
    request_id: str | None = None,
    freshness: str | None = None,
) -> QueryResult:
    """Answer one message. This is what `POST /api/v1/query` calls.

    `freshness` overrides what the planner chose per step: "live" makes the
    reads go to Google rather than to our mirror, which is what "refresh and ask
    again" has to mean.
    """
    actor = await _load_actor(session, user_id)
    run = Run(
        session,
        actor,
        text,
        conversation_id=conversation_id,
        request_id=request_id,
        freshness=freshness,
    )
    await run.open()
    return await run.go()


# ---------------------------------------------------------------------------
# 2. Answering a card
# ---------------------------------------------------------------------------


async def respond_to_prompt(
    session: Any,
    *,
    user_id: str,
    prompt_id: str,
    value: Any,
) -> QueryResult:
    """Answer a card.

    A blocking card resumes the run it paused — the plan is already in
    `node_executions`, so this costs **zero model calls**. A confirm card
    approves the writes it gates and hands them to the worker.
    """
    from app.db.repositories import prompts as prompt_repo

    actor = await _load_actor(session, user_id)
    prompt = await prompt_repo.require_prompt(session, user_id, prompt_id)
    if str(prompt.status) != "pending":
        raise AppError(
            "PROMPT_NOT_PENDING",
            f"That card is already {prompt.status}.",
            http=409,
            details={"input_id": prompt_id, "status": str(prompt.status)},
        )
    return await _respond(session, actor, prompt, value)


async def _respond(
    session: Any,
    actor: Actor,
    prompt: Any,
    value: Any,
    *,
    google: Any = None,
    now: datetime | None = None,
    registry: dict[str, Any] | None = None,
) -> QueryResult:
    if bool(getattr(prompt, "blocking", True)):
        return await _resume(
            session, actor, prompt, value, google=google, now=now, registry=registry
        )
    return await _confirm(session, actor, prompt, value, now=now)


async def _resume(
    session: Any,
    actor: Actor,
    prompt: Any,
    value: Any,
    *,
    google: Any = None,
    now: datetime | None = None,
    registry: dict[str, Any] | None = None,
) -> QueryResult:
    """Carry on from a blocking question. **Zero model calls.**

    The plan is already on disk. Rebuild it from `node_executions`, put the
    answer where the paused node's result belongs, and re-enter dispatch. Steps
    that already succeeded are reused rather than run again.
    """
    from app.db.repositories import prompts as prompt_repo
    from app.db.repositories import runs as run_repo
    from app.db.repositories import steps as step_repo

    run_id = str(getattr(prompt, "run_id", "") or "")
    run_row = await run_repo.require_run(session, actor.id, run_id)
    moment = now or datetime.now(UTC)

    answered = await prompt_repo.answer_prompt(session, actor.id, prompt.id, value)
    await session.commit()

    rows = await step_repo.list_steps(session, actor.id, run_id)
    if not rows:
        raise AppError(
            "NOT_FOUND",
            "That run has no plan left to pick up.",
            http=404,
            details={"run_id": run_id},
        )

    latest: dict[str, Any] = {}
    for row in rows:  # ordered by round then seq, so a later round wins
        latest[row.node_id] = row

    steps: list[dict[str, Any]] = []
    completed: dict[str, Any] = {}
    for node_id, row in latest.items():
        if row.op == "search.probe":
            continue  # the grounding pass, not a plan step
        steps.append(
            {
                "id": node_id,
                "op": row.op,
                "args": dict(row.args or {}),
                "depends_on": list(row.depends_on or []),
                "expect": "many",
                "optional": False,
                "freshness": "cached",
                "speculate": False,
            }
        )
        if str(row.status) == "succeeded" and row.result is not None:
            completed[node_id] = row.result

    # The answer stands where the paused node's result would have been, which is
    # exactly what `{{disambiguate.value.event_id}}` reads.
    paused_row_id = str(getattr(answered, "node_execution_id", "") or "")
    answered_node = ""
    for node_id, row in latest.items():
        if row.id == paused_row_id or (
            row.op in ("ask.user", "ask.clarify") and node_id not in completed
        ):
            completed[node_id] = {"value": value, "input_id": prompt.id, "answered": True}
            answered_node = node_id
            break

    intent = dict(run_row.intent or {})
    run = Run(
        session,
        actor,
        str(intent.get("query") or ""),
        conversation_id=run_row.conversation_id,
        google=google,
        now=moment,
        registry=registry,
    )
    run.run_id = run_id
    _current_run.set(run_id)
    run.intent = intent
    run.planner_tier = int(run_row.planner_tier or TIER_COMPOSED)
    run.answer_style = str(intent.get("answer_style") or "card")
    run.windows = _windows_from(intent)
    run.node_rows = {node_id: row.id for node_id, row in latest.items()}
    run._seq = max((int(row.seq) for row in rows), default=0) + 1
    run._written = {(row.node_id, int(row.round or 0)) for row in rows}
    if answered_node:
        # The card that was just answered must not be raised a second time.
        run._raised_for_node[answered_node] = prompt.id

    await run_repo.mark_resumed(session, actor.id, run_id)
    await session.commit()

    if answered_node:
        # The node that asked has its answer. Dispatch treats a result handed to
        # it as already done and never calls the step hooks for it, so closing
        # the row is this function's job — otherwise the question sits at
        # `running` for good and the trace never shows it was answered.
        await run._persist_step(
            answered_node, status="succeeded", result=completed[answered_node]
        )

    await run.progress("dispatch", "Picking up where we left off", 60)

    plan = {
        "type": "plan",
        "intent": intent,
        "answer_style": run.answer_style,
        "steps": steps,
    }

    from app.core import llm

    with llm.track_usage() as usage:
        mark = time.perf_counter()
        try:
            result = await run._dispatch(plan, completed=completed)
            run.timings.dispatch_ms = int((time.perf_counter() - mark) * 1000)
            out = await run._finish(plan, result)
        except AppError as exc:
            out = await run._fail(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("runner.resume_crashed", run_id=run_id)
            out = await run._fail(
                AppError(
                    "INTERNAL",
                    "Something broke while picking that back up.",
                    http=500,
                    details={"detail": str(exc)[:300]},
                )
            )
        out.usage = _usage_dict(usage)

    await run._record_usage(out.usage)
    run.timings.total_ms = int((time.perf_counter() - run.started) * 1000)
    out.timings = run.timings.to_dict()
    out.route = "resume"

    await run.emit(
        "run.complete" if out.status != "failed" else "error",
        {
            "status": out.status,
            "message_id": out.message_id,
            "answer_style": out.answer_style,
            "planner_tier": out.planner_tier,
            "usage": out.usage,
            "timings": out.timings,
            "degraded": out.degraded,
            "resumed": True,
        },
    )
    return out


async def _confirm(
    session: Any,
    actor: Actor,
    prompt: Any,
    value: Any,
    *,
    now: datetime | None = None,
) -> QueryResult:
    """Approve — or decline — the writes one card gates. Zero model calls.

    Approving flips each draft to `approved` and hands it to the actions queue.
    Nothing is sent from here; the worker owns that, and it re-checks before it
    acts.
    """
    from app.db.repositories import actions as action_repo
    from app.db.repositories import conversations as conv_repo
    from app.db.repositories import prompts as prompt_repo
    from app.db.repositories import runs as run_repo

    decision = _reads_as_approval(value)
    if decision is None:
        raise AppError(
            "PROMPT_VALUE_INVALID",
            "I could not read that as a yes or a no.",
            http=422,
            details={"input_id": prompt.id, "value": _json(value)},
        )

    run_id = str(getattr(prompt, "run_id", "") or "")
    run_row = await run_repo.get_run(session, actor.id, run_id) if run_id else None
    conversation_id = str(getattr(run_row, "conversation_id", "") or "")

    await prompt_repo.answer_prompt(session, actor.id, prompt.id, value)
    gated = await action_repo.list_for_prompt(session, actor.id, prompt.id)

    approved: list[str] = []
    if decision:
        for action in gated:
            if str(action.status) != "draft":
                continue
            await action_repo.approve_action(session, actor.id, action.id)
            approved.append(action.id)
        await session.commit()
        for action_id in approved:
            _enqueue_action(action_id, actor.id)
        text = _approved_text(gated, approved)
    else:
        cancelled = await action_repo.cancel_for_prompt(
            session, actor.id, prompt.id, reason="declined"
        )
        await session.commit()
        text = (
            f"Left alone. {cancelled} prepared change"
            f"{'' if cancelled == 1 else 's'} will not happen."
            if cancelled
            else "Left alone. Nothing was sent."
        )

    blocks = [_text_block(text)]
    message_id = ""
    if conversation_id:
        message = await conv_repo.add_message(
            session,
            actor.id,
            conversation_id,
            role="assistant",
            content=blocks,
            run_id=run_id or None,
            now=now,
        )
        message_id = message.id
        await session.commit()

    action_ids = [a.id for a in gated]
    prompts_out, actions_out = await _read_back(session, actor.id, [prompt.id], action_ids)

    if run_id:
        await events.publish(
            run_id,
            "progress",
            {
                "phase": "confirm",
                "input_id": prompt.id,
                "approved": approved,
                "declined": [] if decision else action_ids,
            },
        )
    for action_id in approved:
        if conversation_id:
            await events.publish_conversation(
                conversation_id,
                "progress",
                {"phase": "queued", "action_id": action_id},
                run_id=run_id or None,
            )

    return QueryResult(
        conversation_id=conversation_id,
        message_id=message_id,
        run_id=run_id,
        status="complete",
        answer=text,
        content=blocks,
        intent=dict(getattr(run_row, "intent", None) or {}) or None,
        actions=actions_out,
        prompts=prompts_out,
        usage={"llm_calls": 0, "calls": 0, "prompt": 0, "completion": 0},
        timings={"total_ms": 0},
        answer_style="card",
        planner_tier=int(getattr(run_row, "planner_tier", TIER_TEMPLATE) or TIER_TEMPLATE),
        action_ids=action_ids,
        input_ids=[prompt.id],
        route="confirm",
    )


def _approved_text(gated: Sequence[Any], approved: Sequence[str]) -> str:
    if not approved:
        return "Nothing was left to approve on that card."
    if len(approved) == 1:
        op = next((a.op for a in gated if a.id == approved[0]), "")
        what = {
            "gmail.send_email": "Sending it now.",
            "gmail.draft_email": "Saving the draft now.",
            "gmail.update_labels": "Changing the labels now.",
            "gcal.create_event": "Adding it to your calendar now.",
            "gcal.update_event": "Moving it now.",
            "gcal.delete_event": "Deleting it now.",
            "drive.share_file": "Sharing it now.",
            "drive.move_file": "Moving it now.",
        }.get(str(op), "On it now.")
        return f"{what} I will say here when it is done."
    return (
        f"On it — {len(approved)} changes, in order. If the first one fails the rest do "
        "not run. I will say here when they are done."
    )


def _enqueue_action(action_id: str, user_id: str) -> None:
    """Hand an approved action to the worker. Never raises into the response."""
    try:
        from app.tasks.actions import execute_action

        execute_action.delay(action_id, user_id)
    except Exception as exc:  # noqa: BLE001 - the row is approved; a sweep retries it
        log.warning("runner.action_not_enqueued", action_id=action_id, error=str(exc))


# ---------------------------------------------------------------------------
# 3. Declining a card
# ---------------------------------------------------------------------------


async def cancel_prompt(session: Any, *, user_id: str, prompt_id: str) -> dict[str, Any]:
    """"Not now". The card closes and everything it gated is cancelled."""
    from app.db.repositories import actions as action_repo
    from app.db.repositories import conversations as conv_repo
    from app.db.repositories import prompts as prompt_repo
    from app.db.repositories import runs as run_repo
    from app.db.repositories import steps as step_repo

    prompt = await prompt_repo.require_prompt(session, user_id, prompt_id)
    if str(prompt.status) != "pending":
        raise AppError(
            "PROMPT_NOT_PENDING",
            f"That card is already {prompt.status}.",
            http=409,
            details={"input_id": prompt_id, "status": str(prompt.status)},
        )

    blocking = bool(getattr(prompt, "blocking", True))
    run_id = str(getattr(prompt, "run_id", "") or "")

    await prompt_repo.cancel_prompt(session, user_id, prompt_id)
    cancelled = await action_repo.cancel_for_prompt(
        session, user_id, prompt_id, reason="user_cancelled"
    )

    conversation_id = ""
    message_id = ""
    if run_id:
        run_row = await run_repo.get_run(session, user_id, run_id)
        conversation_id = str(getattr(run_row, "conversation_id", "") or "")

    if blocking and run_id:
        # The run was waiting on this and now never will be. Close it out so
        # nothing is left sitting in `running` for a sweep to worry about.
        await step_repo.cancel_pending(session, user_id, run_id, reason="prompt_cancelled")
        await run_repo.set_status(session, user_id, run_id, "cancelled")

    text = (
        f"Dropped it. {cancelled} prepared change"
        f"{'' if cancelled == 1 else 's'} will not happen."
        if cancelled
        else "Dropped it. Nothing was sent."
    )
    if conversation_id:
        message = await conv_repo.add_message(
            session,
            user_id,
            conversation_id,
            role="assistant",
            content=[_text_block(text)],
            run_id=run_id or None,
        )
        message_id = message.id

    await session.commit()

    if run_id:
        await events.publish(
            run_id,
            "run.complete",
            {
                "status": "cancelled",
                "message_id": message_id,
                "input_id": prompt_id,
                "cancelled_actions": cancelled,
            },
        )

    return {
        "id": prompt_id,
        "status": "cancelled",
        "blocking": blocking,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "cancelled_actions": cancelled,
        "answer": text,
    }


# ---------------------------------------------------------------------------
# Retry: run what broke again, at round + 1
# ---------------------------------------------------------------------------


async def retry_run(
    session: Any,
    user: Any,
    *,
    conversation_id: str,
    service: str | None = None,
    google: Any = None,
    now: datetime | None = None,
    registry: dict[str, Any] | None = None,
) -> QueryResult | None:
    """Run the failed part of the last turn again. **Zero planning calls.**

    The plan is on disk. The failed node and everything skipped because of it go
    again; nodes that succeeded are reused. The unique key on
    `(run_id, node_id, round)` is what makes this a new row rather than an
    overwrite, so the trace keeps both attempts.
    """
    from app.db.repositories import runs as run_repo
    from app.db.repositories import steps as step_repo

    actor = Actor.of(user)
    previous = await run_repo.latest_run(session, actor.id, conversation_id)
    if previous is None:
        return None

    rows = await step_repo.list_steps(session, actor.id, previous.id)
    if not rows:
        return None

    latest: dict[str, Any] = {}
    for row in rows:
        latest[row.node_id] = row

    broken = {
        node_id
        for node_id, row in latest.items()
        if str(row.status) in ("failed", "timeout", "skipped", "cancelled")
        and row.op != "search.probe"
        and (service is None or row.op.split(".", 1)[0] == service)
    }
    if not broken:
        return None

    # Anything downstream has to run again too: its references could not bind the
    # first time round.
    changed = True
    while changed:
        changed = False
        for node_id, row in latest.items():
            if node_id in broken:
                continue
            if set(row.depends_on or ()) & broken:
                broken.add(node_id)
                changed = True

    steps = [
        {
            "id": node_id,
            "op": row.op,
            "args": dict(row.args or {}),
            "depends_on": list(row.depends_on or []),
            "expect": "many",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        }
        for node_id, row in latest.items()
        if row.op != "search.probe"
    ]
    completed = {
        node_id: row.result
        for node_id, row in latest.items()
        if node_id not in broken
        and str(row.status) == "succeeded"
        and row.result is not None
    }

    intent = dict(previous.intent or {})
    run = Run(
        session,
        actor,
        str(intent.get("query") or ""),
        conversation_id=conversation_id,
        google=google,
        now=now,
        registry=registry,
    )
    run.run_id = previous.id
    _current_run.set(previous.id)
    run.intent = intent
    run.windows = _windows_from(intent)
    run.answer_style = str(intent.get("answer_style") or "prose")
    run.planner_tier = int(previous.planner_tier or TIER_COMPOSED)
    run.node_rows = {node_id: row.id for node_id, row in latest.items()}
    run._seq = max((int(row.seq) for row in rows), default=0) + 1

    round_number = await step_repo.next_round(session, actor.id, previous.id)
    await run._write_steps(
        [step for step in steps if step["id"] in broken], round_number=round_number
    )

    await run_repo.set_status(session, actor.id, previous.id, "running")
    await session.commit()

    plan = {
        "type": "plan",
        "intent": intent,
        "answer_style": run.answer_style,
        "steps": steps,
    }

    from app.core import llm

    with llm.track_usage() as usage:
        mark = time.perf_counter()
        result = await run._dispatch(plan, completed=completed)
        run.timings.dispatch_ms = int((time.perf_counter() - mark) * 1000)
        out = await run._finish(plan, result)
        out.usage = _usage_dict(usage)

    await run._record_usage(out.usage)
    run.timings.total_ms = int((time.perf_counter() - run.started) * 1000)
    out.timings = run.timings.to_dict()
    out.route = "retry"
    return out


# ---------------------------------------------------------------------------
# Staging a write from outside a run
# ---------------------------------------------------------------------------


async def stage_write(
    session: Any,
    user_id: str,
    *,
    run_id: str,
    message_id: str,
    op: str,
    payload: dict[str, Any],
    conversation_id: str,
    question: str,
    preview: dict[str, Any] | None = None,
    help_text: str | None = None,
    external_ref: str | None = None,
    node_execution_id: str | None = None,
    ttl_hours: int = 24,
) -> tuple[str, str]:
    """Write the gate and the action together, in one transaction.

    `actions.requires_input_id` is NOT NULL and references `pending_inputs`, so
    the card has to exist first — and because both rows are written here, a write
    that needs a yes physically cannot exist without one gating it. Returns
    `(input_id, action_id)`.
    """
    from app.db.repositories import actions as action_repo
    from app.db.repositories import prompts as prompt_repo

    prompt = await prompt_repo.create_prompt(
        session,
        user_id,
        run_id,
        message_id,
        kind="confirm",
        prompt={
            "question": question,
            "help_text": help_text
            or "Nothing has been sent. Approving is what makes it happen.",
        },
        value_schema=CONFIRM_SCHEMA,
        blocking=False,  # the run finishes; this waits on a yes
        node_execution_id=node_execution_id,
        conversation_id=conversation_id,
        op=op,
        expires_at=datetime.now(UTC) + timedelta(hours=ttl_hours),
    )

    body = dispatcher.json_safe(payload)
    action = await action_repo.create_action(
        session,
        user_id,
        message_id,
        requires_input_id=prompt.id,
        op=op,
        payload=body,
        dedupe_key=fingerprint_parts("action.dedupe", user_id, op, body, conversation_id),
        node_execution_id=node_execution_id,
        external_ref=external_ref,
    )
    await session.commit()

    await events.publish(
        run_id,
        "action.prepared",
        {
            "action_id": action.id,
            "op": op,
            "status": "draft",
            "requires_input_id": prompt.id,
            "external_ref": external_ref,
            "preview": preview,
        },
    )
    return prompt.id, action.id


# ---------------------------------------------------------------------------
# The older spellings, so the API layer needs no edit
# ---------------------------------------------------------------------------


async def run_query(
    session: Any,
    user: Any,
    query: str,
    *,
    conversation_id: str | None = None,
    google: Any = None,
    now: datetime | None = None,
    registry: dict[str, Any] | None = None,
    request_id: str | None = None,
    freshness: str | None = None,
) -> QueryResult:
    """`handle_query`, with the user row already in hand."""
    actor = Actor.of(user)
    if not actor.id:
        raise AppError("NOT_AUTHENTICATED", "No user on this request.", http=401)
    run = Run(
        session,
        actor,
        query,
        conversation_id=conversation_id,
        google=google,
        now=now,
        registry=registry,
        request_id=request_id,
        freshness=freshness,
    )
    await run.open()
    return await run.go()


async def resume_run(
    session: Any,
    user: Any,
    prompt: Any,
    value: Any,
    *,
    google: Any = None,
    now: datetime | None = None,
    registry: dict[str, Any] | None = None,
) -> QueryResult:
    """`respond_to_prompt`, with the prompt row already loaded."""
    actor = Actor.of(user)
    return await _respond(
        session, actor, prompt, value, google=google, now=now, registry=registry
    )


__all__ = [
    "CONFIRM_SCHEMA",
    "CORPORA",
    "MAX_DELTA_ROUNDS",
    "MAX_LLM_CALLS",
    "MAX_REPLAN_ROUNDS",
    "TIER_COMPOSED",
    "TIER_REPLAN",
    "TIER_STEP_LOOP",
    "TIER_TEMPLATE",
    "Actor",
    "QueryResult",
    "Run",
    "RunOutcome",
    "Timings",
    "cancel_prompt",
    "current_run_id",
    "handle_query",
    "max_llm_calls",
    "respond_to_prompt",
    "resume_run",
    "retry_run",
    "run_query",
    "stage_write",
]
