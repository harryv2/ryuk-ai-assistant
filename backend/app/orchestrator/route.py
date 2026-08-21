"""The one planning call, streamed so the work can start before it finishes.

A plan arrives left to right: the intent closes after a couple of hundred
tokens, then one step object at a time. Waiting for the closing brace before
doing anything throws that away — so this module scans the partial JSON as it
streams and yields each piece the moment it is syntactically complete.

What that buys, in order of arrival:

* the **intent** goes on the wire at about 590 ms, so the UI can say what it
  understood while the rest is still being written;
* each **step** is emitted as its object closes, so the step trace draws the
  DAG progressively — and so the runner can start a local read-only step that
  has no dependencies before the last step exists;
* an **ask.user** first step means the card can be rendered before the plan has
  finished arriving.

All four verbs come back through here: `plan`, `answer`, `revise` and
`answer_input`. The verb itself is emitted as soon as the `"type"` field closes,
which is usually inside the first chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Final, Sequence

from app.core.errors import AppError
from app.core.logging import get_logger
from app.orchestrator import prompts

log = get_logger(__name__)

PLAN_MAX_TOKENS: Final[int] = 1800
REPAIR_MAX_TOKENS: Final[int] = 1600

VERBS: Final[frozenset[str]] = frozenset({"plan", "answer", "revise", "answer_input"})


@dataclass
class RouteEvent:
    """One piece of the plan, the moment it became complete."""

    type: str  # verb | intent | step | text | done
    data: Any


# ---------------------------------------------------------------------------
# Incremental JSON scanning
# ---------------------------------------------------------------------------


def _match_object(text: str, start: int) -> int:
    """Index just past the object that begins at ``start``, or -1 if unfinished.

    A hand-rolled brace counter rather than a parser because the input is a
    prefix of a JSON document, and every parser worth using refuses those.
    """
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _find_value(text: str, key: str, *, after: int = 0) -> int:
    """Index of the first character of ``key``'s value, or -1."""
    needle = f'"{key}"'
    at = text.find(needle, after)
    if at == -1:
        return -1
    colon = text.find(":", at + len(needle))
    if colon == -1:
        return -1
    index = colon + 1
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index if index < len(text) else -1


def _string_at(text: str, start: int) -> str | None:
    """A complete JSON string starting at ``start``, or None if still arriving."""
    if start < 0 or start >= len(text) or text[start] != '"':
        return None
    index = start + 1
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            try:
                return json.loads(text[start : index + 1])
            except json.JSONDecodeError:
                return None
        index += 1
    return None


class PlanScanner:
    """Feed it chunks; it hands back whole pieces of the plan as they close."""

    def __init__(self) -> None:
        self.buffer = ""
        self._verb: str | None = None
        self._intent: dict[str, Any] | None = None
        self._answer_style: str | None = None
        self._steps: list[dict[str, Any]] = []
        self._steps_cursor: int = -1
        self._text_emitted = False

    # -- state -------------------------------------------------------------

    @property
    def verb(self) -> str | None:
        return self._verb

    @property
    def intent(self) -> dict[str, Any] | None:
        return self._intent

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._steps)

    def feed(self, chunk: str) -> list[RouteEvent]:
        self.buffer += chunk or ""
        events: list[RouteEvent] = []

        if self._verb is None:
            verb = _string_at(self.buffer, _find_value(self.buffer, "type"))
            if verb in VERBS:
                self._verb = verb
                events.append(RouteEvent("verb", verb))

        if self._answer_style is None:
            style = _string_at(self.buffer, _find_value(self.buffer, "answer_style"))
            if style:
                self._answer_style = style

        if self._intent is None:
            at = _find_value(self.buffer, "intent")
            if at != -1:
                end = _match_object(self.buffer, at)
                if end != -1:
                    try:
                        self._intent = json.loads(self.buffer[at:end])
                    except json.JSONDecodeError:
                        self._intent = None
                    else:
                        events.append(
                            RouteEvent(
                                "intent",
                                {**self._intent, "answer_style": self._answer_style},
                            )
                        )

        if self._steps_cursor == -1:
            at = _find_value(self.buffer, "steps")
            if at != -1 and self.buffer[at] == "[":
                self._steps_cursor = at + 1

        if self._steps_cursor != -1:
            while True:
                cursor = self._steps_cursor
                while cursor < len(self.buffer) and self.buffer[cursor] in " \t\r\n,":
                    cursor += 1
                if cursor >= len(self.buffer) or self.buffer[cursor] != "{":
                    self._steps_cursor = cursor
                    break
                end = _match_object(self.buffer, cursor)
                if end == -1:
                    self._steps_cursor = cursor
                    break
                try:
                    step = json.loads(self.buffer[cursor:end])
                except json.JSONDecodeError:
                    self._steps_cursor = end
                    continue
                self._steps.append(step)
                self._steps_cursor = end
                events.append(RouteEvent("step", step))

        if self._verb == "answer" and not self._text_emitted:
            text = _string_at(self.buffer, _find_value(self.buffer, "text"))
            if text:
                self._text_emitted = True
                events.append(RouteEvent("text", text))

        return events

    def finish(self) -> dict[str, Any]:
        """The whole object, repaired if the model fenced or trailed it."""
        from app.core import llm

        try:
            return llm.parse_json(self.buffer, "route")
        except AppError:
            # A truncated stream is still worth something when the intent and
            # some steps closed — better a short plan than a dead turn.
            if self._intent is not None or self._steps:
                log.warning("route.partial_plan", steps=len(self._steps))
                return {
                    "type": self._verb or "plan",
                    "intent": self._intent or {},
                    "answer_style": self._answer_style or "prose",
                    "steps": self._steps,
                    "truncated": True,
                }
            raise


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


@dataclass
class RouteContext:
    """Everything the volatile half of the planning prompt is built from."""

    query: str
    catalogue: str
    now: datetime
    tz: str = "UTC"
    week_start: int = 1
    windows: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, Any] = field(default_factory=dict)
    extracted: dict[str, Any] = field(default_factory=dict)
    ambiguity: dict[str, Any] | None = None
    literals: dict[str, Any] = field(default_factory=dict)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    open_prompts: list[dict[str, Any]] = field(default_factory=list)
    staged_actions: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[dict[str, Any]] = field(default_factory=list)
    user: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> tuple[str, str]:
        """The cached prefix and the volatile half, in that order."""
        system = prompts.route_system(self.catalogue)
        user = prompts.route_user(
            query=self.query,
            now_iso=self.now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            timezone=self.tz,
            week_start=self.week_start,
            windows=self.windows or None,
            candidates=self.candidates or None,
            extracted=self.extracted or None,
            ambiguity=self.ambiguity,
            literals=self.literals or None,
            aliases=self.aliases or None,
            entities=self.entities or None,
            history=self.history or None,
            open_prompts=self.open_prompts or None,
            staged_actions=self.staged_actions or None,
            degraded=self.degraded or None,
            user=self.user or None,
        )
        return system, user


async def route(
    ctx: RouteContext,
    *,
    model: str | None = None,
    max_tokens: int = PLAN_MAX_TOKENS,
) -> AsyncIterator[RouteEvent]:
    """Make the planning call and yield each piece as it becomes complete.

    The last event is always ``done``, carrying the whole parsed object. A
    caller that only wants the finished plan can ignore everything before it.
    """
    from app.core import llm

    system, user = ctx.messages()
    scanner = PlanScanner()

    log.info(
        "route.call",
        query_chars=len(ctx.query),
        candidates={k: len(v or []) for k, v in (ctx.candidates or {}).items()},
        windows=list(ctx.windows or {}),
    )

    async for chunk in llm.stream_json(system, user, max_tokens, model=model):
        for event in scanner.feed(chunk):
            yield event

    yield RouteEvent("done", scanner.finish())


async def plan(
    query: str,
    *,
    now: datetime | None = None,
    tz: str = "UTC",
    week_start: int = 1,
    history: Sequence[dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Classify and plan one query standalone — no run, no probe, no database.

    The full pipeline grounds the planner with the probe's candidates; this
    entry point deliberately runs without them, which is exactly what an
    intent-accuracy measurement wants to measure: what the classifier does
    with the words alone. `tests/eval/intent_accuracy.py` binds here.
    """
    from app.ops import registry as ops_registry

    ctx = RouteContext(
        query=query or "",
        catalogue=ops_registry.catalogue(),
        now=now or datetime.now(UTC),
        tz=tz,
        week_start=week_start,
        history=[
            h if isinstance(h, dict) else {"role": "user", "text": str(h)}
            for h in (history or [])
        ],
    )
    return await route_once(ctx, model=model)


async def route_once(
    ctx: RouteContext,
    *,
    model: str | None = None,
    max_tokens: int = PLAN_MAX_TOKENS,
) -> dict[str, Any]:
    """The same call without the streaming, for callers that cannot use it."""
    from app.core import llm

    system, user = ctx.messages()
    return await llm.complete_json(system, user, max_tokens, model=model)


async def repair(
    ctx: RouteContext,
    plan: dict[str, Any],
    *,
    reason: str,
    errors: Sequence[str] | None = None,
    completed: dict[str, Any] | None = None,
    round_number: int = 1,
    model: str | None = None,
) -> dict[str, Any]:
    """The delta round: one attempt to fix a rejected plan, or to replan.

    Not streamed. A repair is usually one edited step, it arrives quickly, and
    starting execution from a half-arrived correction to something that was
    already wrong is not a trade worth making.
    """
    from app.core import llm

    system = prompts.route_system(ctx.catalogue) + "\n\n" + prompts.DELTA_SYSTEM
    user = prompts.delta_user(
        reason=reason,
        errors=list(errors or ()),
        plan=plan,
        completed=completed,
        query=ctx.query,
        round_number=round_number,
    )
    log.info("route.repair", reason=reason, round=round_number, errors=len(errors or ()))
    return await llm.complete_json(system, user, REPAIR_MAX_TOKENS, model=model)


# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------


def verb_of(payload: Any) -> str:
    """Which of the four verbs came back. Unrecognised reads as a plan."""
    if not isinstance(payload, dict):
        return "plan"
    verb = str(payload.get("type") or "").strip()
    if verb in VERBS:
        return verb
    # A response with steps is a plan whatever it called itself.
    return "plan" if payload.get("steps") is not None else "answer"


def normalise(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill in the fields a step may omit, so nothing downstream guesses.

    The defaults are the conservative readings: not optional, not speculative,
    read the mirror rather than Google, expect a list.
    """
    plan = dict(payload or {})
    plan["type"] = verb_of(plan)
    if plan["type"] != "plan":
        return plan

    intent = dict(plan.get("intent") or {})
    intent.setdefault("name", "unknown")
    intent.setdefault("services", [])
    intent.setdefault("has_write", False)
    intent.setdefault("confidence", 0.5)
    plan["intent"] = intent
    plan.setdefault("answer_style", "prose")

    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(plan.get("steps") or []):
        if not isinstance(raw, dict):
            continue
        step = dict(raw)
        step["id"] = str(step.get("id") or step.get("node_id") or f"step_{index}")
        step["op"] = str(step.get("op") or "")
        step["args"] = dict(step.get("args") or {})
        step["depends_on"] = [str(d) for d in (step.get("depends_on") or [])]
        step["expect"] = str(step.get("expect") or "many")
        step["optional"] = bool(step.get("optional", False))
        step["freshness"] = str(step.get("freshness") or "cached")
        step["speculate"] = bool(step.get("speculate", False))
        steps.append(step)
    plan["steps"] = steps

    # The services the steps actually use, when the intent forgot to say. The
    # validator checks the two agree; filling a blank is not the same as
    # widening one the planner wrote.
    if not intent["services"]:
        intent["services"] = sorted(
            {
                step["op"].split(".", 1)[0]
                for step in steps
                if step["op"] and step["op"].split(".", 1)[0] not in ("ask", "meta")
            }
        )
    return plan


__all__ = [
    "PLAN_MAX_TOKENS",
    "VERBS",
    "PlanScanner",
    "RouteContext",
    "RouteEvent",
    "normalise",
    "repair",
    "route",
    "route_once",
    "verb_of",
]
