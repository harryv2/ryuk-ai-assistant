"""The front door. Five matchers, in order, and none of them costs a token.

Every turn this absorbs is a turn that never touches a provider rate limit,
which is the thing that breaks first at scale. So the order is fixed and each
matcher is deliberately narrow:

1. **An answer to an open card.** A per-kind deterministic grammar, run against
   every prompt still pending. ``parse_answer`` never *partly* understands a
   message — an unparsed answer returns ``None`` and falls through, because a
   half-read "yes" that sends the wrong email is worse than one more question.
2. **A UI verb.** Show more, retry Calendar, cancel that, undo. These act on
   something already on screen and never plan anything.
3. **Chit-chat**, and only when nothing is open. Whole-message match against a
   closed set, never a prefix and never a substring: silently dropping the real
   request appended to a thank-you is a bad failure and an invisible one.
4. **A capability question.** "What can you do?" is answered from a constant.
5. **The rule router.** About fifteen parameterised shapes, each of which fires
   only when the grammar consumes the **whole** message. "What's on my calendar
   next week where john@company.com is invited?" routes because an `@` needs no
   interpretation; "…where John is invited?" does not, because there may be two.

Anything left over is a miss, and a miss is the only thing that reaches a model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Final

from app.orchestrator import prepass, temporal
from app.orchestrator.temporal import Window

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

ROUTE_OPEN_CARD: Final = "open_card"
ROUTE_UI_VERB: Final = "ui_verb"
ROUTE_CHIT_CHAT: Final = "chit_chat"
ROUTE_CAPABILITY: Final = "capability"
ROUTE_RULE_ROUTER: Final = "rule_router"
ROUTE_MISS: Final = "miss"


@dataclass(frozen=True)
class Answer:
    """A message read as the answer to one open card."""

    prompt_id: str
    kind: str
    value: Any
    said: str

    def to_dict(self) -> dict[str, Any]:
        return {"input_id": self.prompt_id, "kind": self.kind, "value": self.value}


@dataclass
class Decision:
    """What the front door concluded, and everything the caller needs to act."""

    route: str
    shape: str | None = None
    answer: Answer | None = None
    verb: str | None = None
    target: str | None = None
    text: str | None = None
    plan: dict[str, Any] | None = None
    intent: dict[str, Any] | None = None
    windows: dict[str, Window] = field(default_factory=dict)
    reason: str = ""

    @property
    def handled(self) -> bool:
        """True when the turn is settled without a model."""
        return self.route != ROUTE_MISS

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "shape": self.shape,
            "verb": self.verb,
            "target": self.target,
            "answer": self.answer.to_dict() if self.answer else None,
            # What the rule router concluded. Dropping these made every
            # serialized rule-router decision look like it had no intent,
            # which any consumer — the eval harness included — misread.
            "intent": self.intent,
            "steps": (self.plan or {}).get("steps"),
            "reason": self.reason,
        }


def miss(reason: str) -> Decision:
    return Decision(ROUTE_MISS, reason=reason)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT_EDGE = re.compile(r"^[\s\"'“”‘’(\[]+|[\s\"'“”‘’)\].!?…,;:]+$")
_POLITE_LEAD = re.compile(
    r"^(?:hey|hi|hello|ok|okay|so|um|please|pls|can you|could you|would you|will you|"
    r"i want to|i'd like to|i would like to|id like to|let's|lets|just)\s+",
    re.IGNORECASE,
)
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff⬀-⯿️]"
)


def normalise(message: str) -> str:
    """Lower case, no edge punctuation, no emoji, single spaces."""
    text = _EMOJI.sub(" ", message or "")
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = _PUNCT_EDGE.sub("", text)
    return text.strip().lower()


def _strip_lead(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _POLITE_LEAD.sub("", text).strip()
    return text


# ---------------------------------------------------------------------------
# 1. An answer to an open card
# ---------------------------------------------------------------------------

_YES: Final[frozenset[str]] = frozenset(
    {
        "y", "yes", "yes please", "yep", "yup", "yeah", "yeh", "ya", "sure",
        "ok", "okay", "k", "fine", "do it", "do that", "do both", "send it",
        "send", "send them", "ship it", "go", "go ahead", "go for it",
        "approve", "approved", "confirm", "confirmed", "please do", "yes do it",
        "sounds good", "looks good", "lgtm", "perfect send it", "move it",
        "make it so", "affirmative", "correct", "that's right", "thats right",
        "right", "accept", "yes send it", "yes please do", "proceed",
    }
)

_NO: Final[frozenset[str]] = frozenset(
    {
        "n", "no", "nope", "nah", "no thanks", "no thank you", "not now",
        "not yet", "don't", "dont", "do not", "don't send", "dont send",
        "cancel", "cancel that", "cancel it", "never mind", "nevermind",
        "stop", "hold off", "hold on", "wait", "leave it", "skip", "skip it",
        "discard", "delete it", "forget it", "abort", "decline", "reject",
        "no don't", "no dont", "negative",
    }
)

_ORDINALS: Final[dict[str, int]] = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "four": 3,
    "fifth": 4, "5th": 4, "five": 4,
    "sixth": 5, "6th": 5, "six": 5,
    "seventh": 6, "7th": 6, "seven": 6,
    "eighth": 7, "8th": 7, "eight": 7,
    "ninth": 8, "9th": 8, "nine": 8,
    "tenth": 9, "10th": 9, "ten": 9,
}

_ALL_WORDS: Final[frozenset[str]] = frozenset({"all", "all of them", "everything", "every one", "both", "all three"})
_NONE_WORDS: Final[frozenset[str]] = frozenset({"none", "none of them", "neither", "no one", "nothing"})


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off a dict or an ORM row, whichever the caller had."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    return default if value is None else value


def _options_of(prompt: Any) -> list[dict[str, Any]]:
    options = _get(prompt, "options") or []
    out: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        if isinstance(option, dict):
            out.append(
                {
                    "id": str(option.get("id", option.get("value", index))),
                    "label": str(option.get("label", option.get("id", index))),
                }
            )
        else:
            out.append({"id": str(option), "label": str(option)})
    if out:
        return out
    # No option list: a string enum in the schema is the same thing.
    schema = _get(prompt, "value_schema") or {}
    for value in schema.get("enum", []) or []:
        out.append({"id": str(value), "label": str(value)})
    return out


def _confirm_value(prompt: Any, decision: bool) -> Any:
    """Shape a yes/no to whatever the card's schema says an answer looks like."""
    schema = _get(prompt, "value_schema") or {}
    if schema.get("type") == "object":
        properties = schema.get("properties") or {}
        for name in ("approved", "approve", "confirmed", "confirm", "ok", "value"):
            if name in properties:
                return {name: decision}
        required = schema.get("required") or []
        if len(required) == 1:
            return {required[0]: decision}
    return decision


def _pick_option(text: str, options: list[dict[str, Any]]) -> str | None:
    """One option, or nothing. Never a best guess between two."""
    if not options:
        return None

    cleaned = re.sub(r"^(?:the\s+|option\s+|number\s+|#)+", "", text).strip()
    cleaned = re.sub(r"\s+one$", "", cleaned).strip()

    for option in options:  # an exact id or label, which is what a card sends
        if text == option["id"].lower() or cleaned == option["id"].lower():
            return option["id"]
        if text == option["label"].lower() or cleaned == option["label"].lower():
            return option["id"]

    if cleaned == "last":
        return options[-1]["id"]
    if cleaned.isdigit():
        index = int(cleaned) - 1
        return options[index]["id"] if 0 <= index < len(options) else None
    if cleaned in _ORDINALS:
        index = _ORDINALS[cleaned]
        return options[index]["id"] if index < len(options) else None

    if len(cleaned) >= 4:  # a distinctive fragment, but only if it is unique
        matches = [o for o in options if cleaned in o["label"].lower() or cleaned in o["id"].lower()]
        if len(matches) == 1:
            return matches[0]["id"]
    return None


def _text_ok(value: str, schema: dict[str, Any]) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if schema.get("enum") and value not in schema["enum"]:
        return False
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if isinstance(minimum, int) and len(value) < minimum:
        return False
    if isinstance(maximum, int) and len(value) > maximum:
        return False
    pattern = schema.get("pattern")
    if pattern:
        try:
            if not re.search(pattern, value):
                return False
        except re.error:
            return False
    return True


def parse_answer(
    message: str,
    prompt: Any,
    *,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
) -> Answer | None:
    """Read a message as the answer to one open card, or return ``None``.

    There is no partial success here on purpose. Half-understanding "yes, but
    send it tomorrow" and executing the "yes" is the one failure this whole
    design exists to prevent, so anything the grammar does not fully consume
    falls through to the rest of the pipeline.
    """
    if not message or not message.strip() or prompt is None:
        return None
    if str(_get(prompt, "status", "pending")) not in ("pending", "InputStatus.PENDING"):
        return None

    kind = str(_get(prompt, "kind", "")).split(".")[-1].lower()
    prompt_id = str(_get(prompt, "id", ""))
    schema = _get(prompt, "value_schema") or {}
    text = normalise(message)
    if not text:
        return None

    if kind == "confirm":
        if text in _YES:
            return Answer(prompt_id, kind, _confirm_value(prompt, True), "approved")
        if text in _NO:
            return Answer(prompt_id, kind, _confirm_value(prompt, False), "declined")
        return None

    if kind == "choice":
        chosen = _pick_option(text, _options_of(prompt))
        if chosen is not None:
            return Answer(prompt_id, kind, chosen, f"chose {chosen}")
        if text in _NO:  # declining a choice is still an answer
            return Answer(prompt_id, kind, None, "declined")
        return None

    if kind == "multi_choice":
        options = _options_of(prompt)
        if text in _ALL_WORDS:
            return Answer(prompt_id, kind, [o["id"] for o in options], "all of them")
        if text in _NONE_WORDS:
            return Answer(prompt_id, kind, [], "none of them")
        parts = [p.strip() for p in re.split(r",| and | plus |\+|&", text) if p.strip()]
        if not parts:
            return None
        chosen: list[str] = []
        for part in parts:
            picked = _pick_option(part, options)
            if picked is None:  # one unreadable part means the whole thing is
                return None
            if picked not in chosen:
                chosen.append(picked)
        return Answer(prompt_id, kind, chosen, f"chose {len(chosen)}")

    if kind == "date_range":
        moment = now or datetime.now(UTC)
        halves = re.split(r"\s+(?:to|until|through|thru|-|–|—)\s+", message.strip(), maxsplit=1)
        if len(halves) == 2:
            first = temporal.resolve(halves[0], tz, week_start, moment)
            second = temporal.resolve(halves[1], tz, week_start, moment, anchor=first.start if first else None)
            if first and second and second.end > first.start:
                return Answer(
                    prompt_id,
                    kind,
                    {
                        "start": first.to_dict()["start"],
                        "end": second.to_dict()["end"],
                        "interpretation": f"{first.interpretation}; to {second.interpretation}",
                    },
                    "a date range",
                )
            return None
        window = temporal.resolve(message, tz, week_start, moment)
        if window is None:
            return None
        return Answer(prompt_id, kind, window.to_dict(), window.interpretation)

    if kind == "text":
        raw = message.strip()
        # A reply that asks its own question is a new turn, not an answer.
        if raw.endswith("?") or len(raw) > 400 or text in _UI_VERBS or text in _CHIT_CHAT:
            return None
        if not _text_ok(raw, schema):
            return None
        return Answer(prompt_id, kind, raw, "typed an answer")

    # `form` is deliberately absent. A form has several fields, and deciding
    # which part of one sentence fills which field is exactly the partial guess
    # this function refuses to make. The card sends a structured value instead.
    return None


def match_open_prompts(
    message: str,
    prompts: Any,
    *,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
) -> Answer | None:
    """First open card this message answers, newest first."""
    for prompt in list(prompts or []):
        answer = parse_answer(message, prompt, tz=tz, week_start=week_start, now=now)
        if answer is not None:
            return answer
    return None


# ---------------------------------------------------------------------------
# 2. UI verbs
# ---------------------------------------------------------------------------

_SERVICE_WORDS: Final[dict[str, str]] = {
    "calendar": "gcal",
    "gcal": "gcal",
    "cal": "gcal",
    "gmail": "gmail",
    "mail": "gmail",
    "email": "gmail",
    "inbox": "gmail",
    "drive": "gdrive",
    "gdrive": "gdrive",
    "files": "gdrive",
}

_UI_VERBS: Final[dict[str, str]] = {
    "show more": "show_more",
    "more": "show_more",
    "show the rest": "show_more",
    "see more": "show_more",
    "expand": "show_more",
    "show all": "show_more",
    "show less": "show_less",
    "collapse": "show_less",
    "retry": "retry",
    "try again": "retry",
    "retry it": "retry",
    "run it again": "retry",
    "again": "retry",
    "undo": "undo",
    "undo that": "undo",
    "cancel that": "cancel",
    "cancel it": "cancel",
    "never mind": "cancel",
    "nevermind": "cancel",
    "stop": "stop",
    "stop it": "stop",
    "open it": "open",
    "open that": "open",
    "edit": "edit",
    "edit it": "edit",
    "edit that": "edit",
    "let me edit it": "edit",
    "refresh": "sync",
    "resync": "sync",
    "sync": "sync",
    "sync now": "sync",
    "reconnect": "reconnect",
    "reconnect google": "reconnect",
}

_RETRY_SERVICE = re.compile(
    r"^(?:retry|try)\s+(?:the\s+)?(?P<svc>calendar|gcal|cal|gmail|mail|email|inbox|drive|gdrive|files)"
    r"(?:\s+again)?$"
)


def match_ui_verb(message: str) -> Decision | None:
    """A verb that acts on what is already on screen. Whole message only."""
    text = normalise(message)
    if not text:
        return None

    verb = _UI_VERBS.get(text)
    if verb:
        return Decision(ROUTE_UI_VERB, shape=verb, verb=verb, reason=f"ui verb {verb!r}")

    match = _RETRY_SERVICE.fullmatch(text)
    if match:
        service = _SERVICE_WORDS[match.group("svc")]
        return Decision(
            ROUTE_UI_VERB,
            shape="retry_service",
            verb="retry",
            target=service,
            reason=f"retry {service}",
        )
    return None


# ---------------------------------------------------------------------------
# 3. Chit-chat
# ---------------------------------------------------------------------------

_CHIT_CHAT: Final[frozenset[str]] = frozenset(
    {
        "thanks", "thank you", "thanks a lot", "thanks so much", "many thanks",
        "thanks again", "thx", "ty", "tysm", "cheers", "much appreciated",
        "appreciate it", "thanks for that", "thanks for the help",
        "perfect", "that's perfect", "thats perfect", "great", "great stuff",
        "brilliant", "excellent", "nice", "nice one", "lovely", "awesome",
        "amazing", "cool", "sweet", "good stuff", "well done", "beautiful",
        "ok", "okay", "k", "kk", "got it", "gotcha", "understood", "noted",
        "sounds good", "makes sense", "fair enough", "right", "yep", "yeah",
        "hi", "hey", "hello", "hiya", "yo", "good morning", "good afternoon",
        "good evening", "morning", "howdy",
        "bye", "goodbye", "see you", "see ya", "later", "goodnight", "night",
        "sorry", "my bad", "oops", "lol", "haha", "hah", "hmm", "wow",
        "no worries", "np", "you're welcome", "youre welcome",
        "you rock", "love it", "legend", "top", "spot on",
    }
)

_CHIT_CHAT_REPLIES: Final[tuple[str, ...]] = (
    "Anytime.",
    "Happy to help.",
    "Any time.",
    "Glad that worked.",
    "You're welcome.",
)

_GREETING_REPLIES: Final[tuple[str, ...]] = (
    "Hello. What do you need?",
    "Hi. What can I look up?",
    "Hey. What are we doing?",
)

_GREETINGS: Final[frozenset[str]] = frozenset(
    {"hi", "hey", "hello", "hiya", "yo", "good morning", "good afternoon",
     "good evening", "morning", "howdy"}
)

_FAREWELLS: Final[frozenset[str]] = frozenset(
    {"bye", "goodbye", "see you", "see ya", "later", "goodnight", "night"}
)

def _chit_chat_reply(message: str, clauses: list[str]) -> str:
    """Short, and varied by hash so it is not identical every time."""
    if any(c in _GREETINGS for c in clauses):
        pool = _GREETING_REPLIES
    elif any(c in _FAREWELLS for c in clauses):
        return "Bye."
    else:
        pool = _CHIT_CHAT_REPLIES
    digest = hashlib.sha256(message.strip().lower().encode()).digest()
    return pool[digest[0] % len(pool)]


def match_chit_chat(
    message: str,
    *,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
) -> Decision | None:
    """Gratitude and greetings, and nothing that is hiding a request inside one.

    Every clause has to be chit-chat. "thanks — also what's on Friday?" splits
    into two clauses and the second one is a real query, so this declines and
    the whole message goes down the pipeline.
    """
    text = normalise(message)
    if not text or len(text) > 80:
        return None
    if "@" in text or "?" in message:
        return None
    if temporal.find_phrases(message, tz, week_start, now or datetime.now(UTC)):
        return None

    clauses = [c.strip() for c in re.split(r"[,.;!—–-]+| but | also | and also ", text) if c.strip()]
    if not clauses or len(clauses) > 3:
        return None
    if not all(clause in _CHIT_CHAT for clause in clauses):
        return None

    return Decision(
        ROUTE_CHIT_CHAT,
        shape="chit_chat",
        text=_chit_chat_reply(message, clauses),
        reason="whole message is chit-chat",
    )


# ---------------------------------------------------------------------------
# 4. Capability question
# ---------------------------------------------------------------------------

_CAPABILITY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p)
    for p in (
        r"^(?:what|which things?)\s+can\s+you\s+do(?:\s+for\s+me)?$",
        r"^what\s+(?:else\s+)?can\s+you\s+(?:do|help(?:\s+me)?\s+with)$",
        r"^what\s+do\s+you\s+do$",
        r"^who\s+are\s+you$",
        r"^what\s+are\s+you$",
        r"^help$",
        r"^help\s+me$",
        r"^how\s+(?:does\s+this|do\s+you)\s+work$",
        r"^what\s+can\s+i\s+ask(?:\s+you)?$",
        r"^(?:do\s+you|can\s+you)\s+(?:have\s+access\s+to|read|see|search)\s+my\s+"
        r"(?:email|emails|mail|inbox|calendar|drive|files|documents)$",
        r"^what\s+(?:can\s+you\s+)?(?:see|access)$",
    )
)

CAPABILITY_ANSWER: Final[str] = (
    "I work across your Gmail, Google Calendar and Google Drive.\n\n"
    "- **Find things.** \"Emails from sarah@company.com about the budget\", "
    "\"PDFs in Drive from last month\", \"what's on my calendar next week\".\n"
    "- **Pull threads together.** \"Prepare for tomorrow's meeting with Acme\" "
    "reads the invite, the mail around it and the files that go with it.\n"
    "- **Draft changes.** \"Cancel my Turkish Airlines flight\", "
    "\"move the meeting with John to Friday\". I prepare the write and show you "
    "exactly what will happen — nothing is sent, moved or deleted until you say yes.\n\n"
    "I answer from a local copy of your data that syncs every 15 minutes, so most "
    "questions come back in well under a second, and I say so when something is stale "
    "or when a service is not responding."
)


def match_capability(message: str) -> Decision | None:
    text = normalise(message)
    if not text:
        return None
    if any(pattern.fullmatch(text) for pattern in _CAPABILITY_PATTERNS):
        return Decision(
            ROUTE_CAPABILITY,
            shape="capability",
            text=CAPABILITY_ANSWER,
            reason="capability question",
        )
    return None


# ---------------------------------------------------------------------------
# 5. The rule router
# ---------------------------------------------------------------------------
#
# A shape fires only when its grammar consumes the whole message. That is the
# difference between "…where john@company.com is invited" — an exact value, no
# ranking, route it — and "…where John is invited", which needs resolution and
# may have two answers, so the model gets it.

# Words that carry no meaning once the time phrase has been lifted out. Kept
# deliberately short: a leftover word that is not on this list means the router
# did not understand the whole message, and not understanding it is the signal
# to hand it to the model.
_FILLER = re.compile(
    r"^(?:\s|for|on|in|at|of|the|my|this|up|around|during|please|pls|"
    r"\?|,|\.|!|-)*$",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_DOMAIN_RE = re.compile(r"@?([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")


@dataclass(frozen=True)
class _When:
    name: str
    window: Window


def _when(
    text: str,
    tz: str,
    week_start: int,
    now: datetime,
    *,
    default: _When | None = None,
) -> _When | None:
    """Read a captured fragment as exactly one time phrase, or refuse.

    Whole-consumption is the whole point. Anything left over after the phrase
    is lifted out — a name, a topic, an attendee — means the router does not
    understand this message well enough to own it.
    """
    fragment = (text or "").strip()
    if not fragment or _FILLER.fullmatch(fragment):
        return default

    phrases = temporal.find_phrases(fragment, tz, week_start, now)
    if len(phrases) != 1:
        return None
    phrase = phrases[0]
    leftover = fragment[: phrase.start] + " " + fragment[phrase.end :]
    if not _FILLER.fullmatch(leftover.strip()):
        return None

    name = re.sub(r"[^a-z0-9]+", "_", phrase.text.lower()).strip("_") or phrase.name
    if name[0].isdigit():
        name = f"on_{name}"
    return _When(name, phrase.window)


def _plan(
    *,
    name: str,
    services: list[str],
    steps: list[dict[str, Any]],
    answer_style: str,
    entities: dict[str, Any] | None = None,
    window: _When | None = None,
    source: str = "rule_router",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The intent object and the executable plan, in the locked shapes."""
    intent: dict[str, Any] = {
        "name": name,
        "services": services,
        "entities": entities or {},
        "steps": [step["id"] for step in steps],
        "has_write": False,
        "confidence": 1.0,
        "source": source,
        "resolved_window": None,
    }
    if window is not None:
        intent["resolved_window"] = {"name": window.name, **window.window.to_dict()}

    plan = {
        "type": "plan",
        "intent": {
            "name": name,
            "services": services,
            "has_write": False,
            "confidence": 1.0,
        },
        "answer_style": answer_style,
        "steps": steps,
    }
    return intent, plan


def _step(step_id: str, op: str, args: dict[str, Any], *, expect: str = "many") -> dict[str, Any]:
    return {
        "id": step_id,
        "op": op,
        "args": args,
        "depends_on": [],
        "expect": expect,
        "optional": False,
        "freshness": "cached",
        "speculate": False,
    }


def _window_args(when: _When) -> dict[str, Any]:
    return {
        "window": {
            "start": f"{{{{windows.{when.name}.start}}}}",
            "end": f"{{{{windows.{when.name}.end}}}}",
        }
    }


@dataclass(frozen=True)
class _Ctx:
    query: str
    text: str  # normalised, politeness stripped
    tz: str
    week_start: int
    now: datetime
    last_intent: dict[str, Any] | None


_Shape = Callable[[_Ctx], "Decision | None"]
_SHAPES: list[tuple[str, _Shape]] = []


def _shape(name: str) -> Callable[[_Shape], _Shape]:
    def register(fn: _Shape) -> _Shape:
        _SHAPES.append((name, fn))
        return fn

    return register


_CAL = r"(?:calendar|schedule|agenda|diary)"
_LEAD = (
    r"(?:what'?s|whats|what is|what do i have|what have i got|what|which|show me|show|list|"
    r"display|anything|is there anything|do i have anything|give me|tell me|find|find me)"
)
_MAIL = r"(?:emails?|mail|messages?|e-mails?)"


# -- calendar ---------------------------------------------------------------


@_shape("calendar_window_attendee")
def _calendar_window_attendee(ctx: _Ctx) -> Decision | None:
    """#16. The `@` is the trigger: an address needs no interpretation."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:on\s+|in\s+|for\s+)?(?:my\s+)?{_CAL}\s*(?P<when>.*?)\s*"
        r"(?:where|with|involving|including)\s+(?P<email>\S+@\S+)\s*"
        r"(?:is\s+|are\s+)?(?:invited|attending|going|on it|a guest|there)?"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    address = _EMAIL_RE.search(match.group("email"))
    if not address:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    steps = [
        _step(
            "events",
            "gcal.search_events",
            {
                **_window_args(when),
                # The filter is called attendee_emails on gcal.search_events —
                # "attendee_emails_any" is refused, which turned the brief's
                # "next week where john@company.com is invited" into an error.
                "attendee_emails": [address.group(0).lower()],
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
        )
    ]
    intent, plan = _plan(
        name="calendar_list",
        services=["gcal"],
        steps=steps,
        answer_style="template:event_list",
        entities={"attendee_email": address.group(0).lower()},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_window_attendee",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="calendar window plus a literal attendee address",
    )


@_shape("calendar_attendee_window")
def _calendar_attendee_window(ctx: _Ctx) -> Decision | None:
    """"meetings with john@company.com next week" — the same facts, reordered."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:my\s+)?(?:meetings?|events?|calls?|{_CAL})\s+"
        r"(?:do i have\s+|have i got\s+|are there\s+|is there\s+)?"
        r"(?:with|involving|including)\s+(?P<email>\S+@\S+)\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    address = _EMAIL_RE.search(match.group("email"))
    if not address:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    steps = [
        _step(
            "events",
            "gcal.search_events",
            {
                **_window_args(when),
                "attendee_emails_any": [address.group(0).lower()],
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
        )
    ]
    intent, plan = _plan(
        name="calendar_list",
        services=["gcal"],
        steps=steps,
        answer_style="template:event_list",
        entities={"attendee_email": address.group(0).lower()},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_attendee_window",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="meetings with a literal address in a window",
    )


@_shape("calendar_count")
def _calendar_count(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        r"how many (?:meetings?|events?|calls?)\s*(?:do i have|are there|have i got)?\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    steps = [
        _step(
            "events",
            "gcal.search_events",
            {
                **_window_args(when),
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 250,
            },
        )
    ]
    intent, plan = _plan(
        name="calendar_count",
        services=["gcal"],
        steps=steps,
        answer_style="template:count_answer",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_count",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="counting events in a window",
    )


@_shape("calendar_free")
def _calendar_free(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        r"(?:am i (?:free|busy|available)|do i have (?:anything|any time|time free)|"
        r"when am i free|what'?s free|free time)\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    steps = [
        _step(
            "events",
            "gcal.search_events",
            {
                **_window_args(when),
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 250,
            },
        )
    ]
    intent, plan = _plan(
        name="calendar_free",
        services=["gcal"],
        steps=steps,
        answer_style="template:free_slots",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_free",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="free time in a window",
    )


@_shape("calendar_next")
def _calendar_next(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:my\s+)?next\s+(?:meeting|event|call|thing)"
        rf"(?:\s+on\s+(?:my\s+)?{_CAL})?"
    )
    if not pattern.fullmatch(ctx.text):
        return None

    ahead = temporal.Window(
        ctx.now,
        ctx.now + timedelta(days=14),
        ctx.tz,
        "from now to a fortnight out; the next thing in the diary",
    )
    when = _When("next_fortnight", ahead)
    steps = [
        _step(
            "event",
            "gcal.search_events",
            {
                **_window_args(when),
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 1,
            },
            expect="one",
        )
    ]
    intent, plan = _plan(
        name="calendar_next",
        services=["gcal"],
        steps=steps,
        answer_style="template:event_list",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_next",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="the next event in the diary",
    )


@_shape("calendar_window")
def _calendar_window(ctx: _Ctx) -> Decision | None:
    """#1. One op, one window, one ordering — nothing to disambiguate."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:on\s+|in\s+|for\s+)?(?:my\s+)?{_CAL}\s*(?:for\s+|on\s+|in\s+)?(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None

    today = _When("today", temporal.resolve("today", ctx.tz, ctx.week_start, ctx.now))  # type: ignore[arg-type]
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now, default=today)
    if when is None:
        return None

    steps = [
        _step(
            "events",
            "gcal.search_events",
            {
                **_window_args(when),
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
        )
    ]
    intent, plan = _plan(
        name="calendar_list",
        services=["gcal"],
        steps=steps,
        answer_style="template:event_list",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="calendar_window",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="a calendar window and nothing else",
    )


@_shape("intent_carry_window")
def _intent_carry(ctx: _Ctx) -> Decision | None:
    """#9. A bare time phrase is not a query — unless the last one was.

    "Next Tuesday" on its own carries the previous calendar intent with a new
    window. With no calendar-shaped run behind it the router declines, and the
    planner gets a query that only a calendar intent can serve anyway.
    """
    if not ctx.last_intent:
        return None
    name = str(ctx.last_intent.get("name") or "")
    if not name.startswith("calendar"):
        return None
    if "gcal" not in (ctx.last_intent.get("services") or []):
        return None
    if ctx.last_intent.get("has_write"):
        return None

    when = _when(ctx.text, ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    carried = dict(ctx.last_intent.get("entities") or {})
    args: dict[str, Any] = {
        **_window_args(when),
        "status_in": ["confirmed", "tentative"],
        "order_by": "starts_at",
        "limit": 50,
    }
    if carried.get("attendee_email"):
        args["attendee_emails_any"] = [carried["attendee_email"]]

    steps = [_step("events", "gcal.search_events", args)]
    intent, plan = _plan(
        name="calendar_list",
        services=["gcal"],
        steps=steps,
        answer_style="template:event_list",
        entities={**carried, "carried_from_run": ctx.last_intent.get("run_id")},
        window=when,
        source="rule_router:intent_carry",
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="intent_carry_window",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="a bare time phrase carrying the previous calendar intent",
    )


# -- mail -------------------------------------------------------------------


@_shape("email_from_address")
def _email_from_address(ctx: _Ctx) -> Decision | None:
    """A sender is an exact value. A topic is not — that one goes to the model."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:me\s+)?(?:all\s+|any\s+|recent\s+)?{_MAIL}\s+"
        r"(?:from|by|sent by)\s+(?P<email>\S+@\S+)\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    address = _EMAIL_RE.search(match.group("email"))
    if not address:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now, default=_default_read(ctx))
    if when is None:
        return None

    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {
                "from_email": address.group(0).lower(),
                **_window_args(when),
                "order_by": "received_at",
                "limit": 25,
            },
        )
    ]
    intent, plan = _plan(
        name="email_search",
        services=["gmail"],
        steps=steps,
        answer_style="template:email_list",
        entities={"from_email": address.group(0).lower()},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_from_address",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="mail from a literal address",
    )


@_shape("email_from_domain")
def _email_from_domain(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:me\s+)?(?:all\s+|any\s+)?{_MAIL}\s+"
        r"(?:from|by)\s+(?:anyone at\s+|anybody at\s+|people at\s+)?@?(?P<domain>[a-z0-9-]+(?:\.[a-z0-9-]+)+)"
        r"\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    domain = match.group("domain").lower()
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now, default=_default_read(ctx))
    if when is None:
        return None

    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {
                "from_domain": domain,
                **_window_args(when),
                "order_by": "received_at",
                "limit": 25,
            },
        )
    ]
    intent, plan = _plan(
        name="email_search",
        services=["gmail"],
        steps=steps,
        answer_style="template:email_list",
        entities={"sender_domain": domain},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_from_domain",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="mail from a literal domain",
    )


@_shape("email_from_alias")
def _email_from_alias(ctx: _Ctx) -> Decision | None:
    """"emails from Acme" — the alias table turns a brand into a domain."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:me\s+)?(?:all\s+|any\s+)?{_MAIL}\s+from\s+(?P<rest>[a-z][a-z0-9 .&'-]*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None

    # Try the shortest brand first and grow. A regex with two greedy groups
    # would settle on one split and never revisit it, which is how "emails from
    # Acme last week" ends up looking for a sender called "a".
    words = match.group("rest").split()
    for take in range(1, min(4, len(words)) + 1):
        who = " ".join(words[:take])
        groups = prepass.match_aliases(who)
        if len(groups) != 1 or not groups[0].sender_domains:
            continue
        when = _when(
            " ".join(words[take:]), ctx.tz, ctx.week_start, ctx.now, default=_default_read(ctx)
        )
        if when is not None:
            break
    else:
        return None
    if when is None:
        return None

    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {
                "from_domain": groups[0].sender_domains[0],
                **_window_args(when),
                "order_by": "received_at",
                "limit": 25,
            },
        )
    ]
    intent, plan = _plan(
        name="email_search",
        services=["gmail"],
        steps=steps,
        answer_style="template:email_list",
        entities={"alias_group": groups[0].name, "sender_domain": groups[0].sender_domains[0]},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_from_alias",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason=f"mail from the {groups[0].name} alias group",
    )


@_shape("email_unread")
def _email_unread(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:me\s+)?(?:any\s+)?unread\s+{_MAIL}\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now, default=_default_read(ctx))
    if when is None:
        return None

    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {
                "labels": ["UNREAD"],
                **_window_args(when),
                "order_by": "received_at",
                "limit": 25,
            },
        )
    ]
    intent, plan = _plan(
        name="email_search",
        services=["gmail"],
        steps=steps,
        answer_style="template:email_list",
        entities={"labels": ["UNREAD"]},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_unread",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="unread mail",
    )


@_shape("email_count")
def _email_count(ctx: _Ctx) -> Decision | None:
    pattern = re.compile(
        rf"how many {_MAIL}\s*(?:did i (?:get|receive)|do i have|are there|have i got)?\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now)
    if when is None:
        return None

    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {**_window_args(when), "order_by": "received_at", "limit": 250},
        )
    ]
    intent, plan = _plan(
        name="email_count",
        services=["gmail"],
        steps=steps,
        answer_style="template:count_answer",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_count",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="counting mail in a window",
    )


@_shape("email_by_literal")
def _email_by_literal(ctx: _Ctx) -> Decision | None:
    """"find the email with booking 6F2QK9" — a literal is not a topic."""
    pattern = re.compile(
        rf"(?:find|show|get|pull up|look up|search for)\s+(?:me\s+)?(?:the\s+|an?\s+)?{_MAIL}?\s*"
        r"(?:about|with|for|containing|mentioning|re)?\s*"
        r"(?:booking|order|invoice|reference|ref|confirmation|pnr|ticket)?\s*"
        r"#?(?P<literal>[A-Za-z0-9][A-Za-z0-9-]{4,})",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(re.sub(r"\s+", " ", ctx.query).strip().rstrip("?."))
    if not match:
        return None
    literal = match.group("literal")
    # A word is not a literal. Only a code with a digit in it qualifies.
    if not re.search(r"\d", literal) or literal.lower() in _SERVICE_WORDS:
        return None

    when = _default_read(ctx)
    steps = [
        _step(
            "mail",
            "gmail.search_emails",
            {"query": literal, **_window_args(when), "order_by": "received_at", "limit": 10},
        )
    ]
    intent, plan = _plan(
        name="email_search",
        services=["gmail"],
        steps=steps,
        answer_style="template:email_list",
        entities={"literal": literal},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="email_by_literal",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="mail carrying a literal reference",
    )


# -- drive ------------------------------------------------------------------


@_shape("drive_recent")
def _drive_recent(ctx: _Ctx) -> Decision | None:
    """Recently changed files. A *type* filter is not here on purpose — a
    mime word plus a topic is what the planner is for."""
    pattern = re.compile(
        rf"(?:{_LEAD}\s+)?(?:my\s+)?(?:recent|recently (?:changed|modified|edited|updated))?\s*"
        r"(?:files|docs|documents)\s*"
        r"(?:in (?:my )?(?:drive|google drive))?\s*"
        r"(?:that (?:changed|were modified|were updated)|changed|modified|updated)?\s*(?P<when>.*)"
    )
    match = pattern.fullmatch(ctx.text)
    if not match:
        return None
    if prepass.mime_types(ctx.text):
        return None  # a type filter means there is a judgement to make
    when = _when(match.group("when"), ctx.tz, ctx.week_start, ctx.now, default=_default_read(ctx))
    if when is None:
        return None

    steps = [
        _step(
            "files",
            "gdrive.search_files",
            {**_window_args(when), "order_by": "modified_at", "limit": 25},
        )
    ]
    intent, plan = _plan(
        name="file_search",
        services=["gdrive"],
        steps=steps,
        answer_style="template:file_list",
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="drive_recent",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="recently changed files",
    )


@_shape("drive_named")
def _drive_named(ctx: _Ctx) -> Decision | None:
    """A file named in quotes is an exact string, not a topic to rank."""
    match = re.fullmatch(
        r"(?:find|open|show|get|pull up|look up)\s+(?:me\s+)?(?:the\s+)?"
        r"(?:file|doc|document|sheet|deck|folder)\s+(?:called|named|titled)\s+"
        r"[\"“'](?P<name>[^\"”']{2,80})[\"”']",
        re.sub(r"\s+", " ", ctx.query).strip().rstrip("?."),
        re.IGNORECASE,
    )
    if not match:
        return None

    when = _default_read(ctx)
    steps = [
        _step(
            "files",
            "gdrive.search_files",
            {"query": match.group("name"), **_window_args(when), "limit": 10},
        )
    ]
    intent, plan = _plan(
        name="file_search",
        services=["gdrive"],
        steps=steps,
        answer_style="template:file_list",
        entities={"file_name": match.group("name")},
        window=when,
    )
    return Decision(
        ROUTE_RULE_ROUTER,
        shape="drive_named",
        plan=plan,
        intent=intent,
        windows={when.name: when.window},
        reason="a file named exactly",
    )


def _default_read(ctx: _Ctx) -> _When:
    return _When("default_read", temporal.default_window("read", ctx.tz, ctx.now))


def match_rule_router(
    message: str,
    *,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
    last_intent: dict[str, Any] | None = None,
) -> Decision | None:
    """The first shape whose grammar consumes the whole message."""
    text = _strip_lead(normalise(message))
    if not text:
        return None

    ctx = _Ctx(
        query=message,
        text=text,
        tz=tz,
        week_start=int(week_start or 1),
        now=now or datetime.now(UTC),
        last_intent=last_intent,
    )
    for _name, shape in _SHAPES:
        try:
            decision = shape(ctx)
        except Exception:  # noqa: BLE001 - a broken shape must not break the turn
            continue
        if decision is not None:
            return decision
    return None


# ---------------------------------------------------------------------------
# The door itself
# ---------------------------------------------------------------------------


def decide(
    message: str,
    *,
    tz: str = "UTC",
    week_start: int = 1,
    now: datetime | None = None,
    open_prompts: Any = (),
    last_intent: dict[str, Any] | None = None,
) -> Decision:
    """Run the five matchers in order and return the first hit, or a miss.

    The order is the contract. An answer to an open card outranks everything,
    because "cancel that" with a confirm card on screen means *do not send it*
    and not *start a new turn about cancelling something*.
    """
    moment = now or datetime.now(UTC)
    prompts = list(open_prompts or [])

    answer = match_open_prompts(message, prompts, tz=tz, week_start=week_start, now=moment)
    if answer is not None:
        return Decision(
            ROUTE_OPEN_CARD,
            shape=f"answer:{answer.kind}",
            answer=answer,
            reason=f"answers the open {answer.kind} card ({answer.said})",
        )

    verb = match_ui_verb(message)
    if verb is not None:
        return verb

    if not prompts:
        chat = match_chit_chat(message, tz=tz, week_start=week_start, now=moment)
        if chat is not None:
            return chat

    capability = match_capability(message)
    if capability is not None:
        return capability

    routed = match_rule_router(
        message, tz=tz, week_start=week_start, now=moment, last_intent=last_intent
    )
    if routed is not None:
        return routed

    return miss("no matcher owns this message; it needs the model")


__all__ = [
    "CAPABILITY_ANSWER",
    "ROUTE_CAPABILITY",
    "ROUTE_CHIT_CHAT",
    "ROUTE_MISS",
    "ROUTE_OPEN_CARD",
    "ROUTE_RULE_ROUTER",
    "ROUTE_UI_VERB",
    "Answer",
    "Decision",
    "decide",
    "match_capability",
    "match_chit_chat",
    "match_open_prompts",
    "match_rule_router",
    "match_ui_verb",
    "normalise",
    "parse_answer",
]
