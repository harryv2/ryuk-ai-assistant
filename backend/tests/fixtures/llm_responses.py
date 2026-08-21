"""Canned model output, one entry per scenario in ``docs/SAMPLE_QUERIES.md``.

The model is never called in these tests. What it *would* have returned is
recorded here, so a plan is a fixed input and every assertion is about our own
code — the validator, the binder, the dispatcher, the renderer — rather than
about a sampling temperature.

Three things live here:

* :data:`PLANS` — the plan JSON, copied from ``docs/SAMPLE_QUERIES.md`` so a
  failing test can be read against the document;
* :func:`chat_completion` / :func:`sse_chunks` / :func:`embeddings_response` —
  the OpenAI wire envelopes, buffered and streamed;
* :func:`fake_embedding` — a deterministic stand-in for
  ``text-embedding-3-small``.

**The embedding is not random noise.** It is a bag of hashed tokens, L2
normalised, so two texts that share words land near each other and two that do
not, do not. That makes the hybrid search in the mirror behave the way it will
in production: the English booking confirmation scores well against "Turkish
Airlines booking", and the Turkish one scores badly — which is precisely the
zero-hit case scenario 12 exists to recover from.

**The prose answer reads the prompt it was given.** :func:`prose_for` only names
a failed service when the *caller* put the failure in the synthesis context. If
the orchestrator forgets to pass `degraded` through, the canned answer does not
mention Calendar and ``test_degradation`` fails — which is the point. A fixture
that always said the right thing would be testing itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Iterable
from typing import Any

EMBED_DIM = 1536
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# --------------------------------------------------------------------------- #
# Deterministic embeddings
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_DIMS_PER_TOKEN = 24
_cache: dict[tuple[str, int], list[float]] = {}


def tokens_of(text: str) -> list[str]:
    """Words, lower-cased, singles dropped. Unicode aware, so Turkish counts."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def fake_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """A stable vector for a string: hashed token bag, L2 normalised.

    Same text, same vector, on any machine, in any process. Shared vocabulary
    means a high cosine; no shared vocabulary means roughly zero. Empty text
    gets the zero vector, which scores zero against everything — the honest
    answer, and the same thing ``app.core.llm.embed`` does.
    """
    key = (text or "", dim)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    vector = [0.0] * dim
    for token in set(tokens_of(text)):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        for _ in range(_DIMS_PER_TOKEN):
            index = rng.randrange(dim)
            vector[index] += 1.0 if rng.random() < 0.5 else -1.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    _cache[key] = vector
    return vector


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Handy in a test that wants to explain why a candidate ranked where it did."""
    a, b = list(a), list(b)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embeddings_response(
    inputs: list[str] | str,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    dim: int = EMBED_DIM,
) -> dict[str, Any]:
    """The `/v1/embeddings` body."""
    items = [inputs] if isinstance(inputs, str) else list(inputs)
    total = sum(max(1, len(tokens_of(t))) for t in items)
    return {
        "object": "list",
        "model": model,
        "data": [
            {"object": "embedding", "index": i, "embedding": fake_embedding(text, dim)}
            for i, text in enumerate(items)
        ],
        "usage": {"prompt_tokens": total, "total_tokens": total},
    }


# --------------------------------------------------------------------------- #
# Chat completion envelopes
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chat_completion(
    content: str | dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    prompt_tokens: int = 1842,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """The buffered `/v1/chat/completions` body."""
    body = content if isinstance(content, str) else json.dumps(content)
    completion_tokens = _tokens(body)
    return {
        "id": "chatcmpl-alphalaw-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": body, "refusal": None},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_chunks(
    content: str | dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    prompt_tokens: int = 1842,
    piece: int = 48,
) -> bytes:
    """The same answer as a `text/event-stream` body.

    Ends the way OpenAI ends one: a chunk carrying ``finish_reason``, then a
    choice-less chunk carrying ``usage`` (that is what ``include_usage`` buys),
    then ``[DONE]``.
    """
    body = content if isinstance(content, str) else json.dumps(content)
    created = int(time.time())
    head = {
        "id": "chatcmpl-alphalaw-test",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }

    lines: list[str] = []
    for start in range(0, len(body), piece):
        chunk = dict(head)
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"content": body[start : start + piece]},
                "finish_reason": None,
            }
        ]
        lines.append(f"data: {json.dumps(chunk)}")

    stop = dict(head)
    stop["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    lines.append(f"data: {json.dumps(stop)}")

    usage = dict(head)
    usage["choices"] = []
    completion_tokens = _tokens(body)
    usage["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    lines.append(f"data: {json.dumps(usage)}")
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# The plans
# --------------------------------------------------------------------------- #

# 1 — "What's on my calendar next week?"
# The rule router should take this at zero LLM calls. It is here so that a
# planner call which should not have happened still produces a sane answer, and
# the LLM-call assertion is the thing that fails rather than the whole run.
PLAN_CALENDAR_NEXT_WEEK = {
    "type": "plan",
    "intent": {
        "name": "calendar_list",
        "services": ["gcal"],
        "has_write": False,
        "confidence": 1.0,
    },
    "answer_style": "template:event_list",
    "steps": [
        {
            "id": "events",
            "op": "gcal.search_events",
            "args": {
                "window": {
                    "start": "{{windows.next_week.start}}",
                    "end": "{{windows.next_week.end}}",
                },
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
            "depends_on": [],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        }
    ],
}

# 16 — the same, filtered on a literal attendee address.
PLAN_CALENDAR_NEXT_WEEK_JOHN = {
    "type": "plan",
    "intent": {
        "name": "calendar_list",
        "services": ["gcal"],
        "has_write": False,
        "confidence": 1.0,
    },
    "answer_style": "template:event_list",
    "steps": [
        {
            "id": "events",
            "op": "gcal.search_events",
            "args": {
                "window": {
                    "start": "{{windows.next_week.start}}",
                    "end": "{{windows.next_week.end}}",
                },
                "attendee_emails_any": ["john@company.com"],
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
            "depends_on": [],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        }
    ],
}

# 2 — "Find emails from sarah@company.com about the budget"
PLAN_SARAH_BUDGET = {
    "type": "plan",
    "intent": {
        "name": "email_search",
        "services": ["gmail"],
        "has_write": False,
        "confidence": 0.96,
    },
    "answer_style": "template:email_list",
    "steps": [
        {
            "id": "mail",
            "op": "gmail.search_emails",
            "args": {
                "from_email": "sarah@company.com",
                "query": "budget",
                "order_by": "relevance",
                "limit": 10,
            },
            "depends_on": [],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        }
    ],
}

# 3 — "Show me PDFs in Drive from last month"
PLAN_DRIVE_PDFS = {
    "type": "plan",
    "intent": {
        "name": "drive_filter",
        "services": ["gdrive"],
        "has_write": False,
        "confidence": 0.93,
    },
    "answer_style": "template:file_list",
    "steps": [
        {
            "id": "files",
            "op": "gdrive.search_files",
            "args": {
                "mime_type": "application/pdf",
                "modified_window": {
                    "start": "{{windows.last_month.start}}",
                    "end": "{{windows.last_month.end}}",
                },
                "order_by": "modified_at",
                "order_dir": "desc",
                "limit": 25,
            },
            "depends_on": [],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        }
    ],
}

# 4 — "Cancel my Turkish Airlines flight".
# `booking` and `flight_event` both depend on nothing, so they start in the same
# event-loop tick; `draft` waits on `booking` only. That shape is the whole
# parallelism assertion in test_query_flows.
PLAN_CANCEL_TURKISH_FLIGHT = {
    "type": "plan",
    "intent": {
        "name": "cancel_flight",
        "services": ["gmail", "gcal"],
        "has_write": True,
        "confidence": 0.91,
    },
    "answer_style": "card",
    "steps": [
        {
            "id": "booking",
            "op": "gmail.get_email",
            "args": {
                "message_id": "{{search.gmail[0].message_id}}",
                "include_body": True,
            },
            "depends_on": [],
            "expect": "one",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "flight_event",
            "op": "gcal.search_events",
            "args": {"query": "TK1984", "limit": 5},
            "depends_on": [],
            "expect": "one",
            "optional": True,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "draft",
            "op": "gmail.draft_email",
            "args": {
                "to": ["cancel@turkishairlines.com"],
                "subject": "Cancellation request - booking {{booking.extracted.pnr}}",
                "body_template": "flight_cancellation",
                "template_vars": {
                    "pnr": "{{booking.extracted.pnr}}",
                    "ticket_no": "{{booking.extracted.ticket_no}}",
                    "flight_no": "{{booking.extracted.flight_no}}",
                    "route": "{{booking.extracted.route}}",
                    "depart_at": "{{booking.extracted.depart_at}}",
                    "passenger": "{{user.display_name}}",
                },
                "in_reply_to": "{{booking.message_id}}",
            },
            "depends_on": ["booking"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
        {
            "id": "send",
            "op": "gmail.send_email",
            "args": {"draft_id": "{{draft.draft_id}}"},
            "depends_on": ["draft"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
    ],
}

# 12 — same query, Turkish booking email. The plan is identical except that the
# body carried no cancellation address, so the alias group's published one is
# used and the answer has to say so.
PLAN_CANCEL_TURKISH_FLIGHT_TR = json.loads(json.dumps(PLAN_CANCEL_TURKISH_FLIGHT))
PLAN_CANCEL_TURKISH_FLIGHT_TR["intent"]["confidence"] = 0.87
PLAN_CANCEL_TURKISH_FLIGHT_TR["intent"]["entities"] = {
    "airline": "Turkish Airlines",
    "alias_group": "turkish_airlines",
    "recovered_by": "escalation_rung_1",
    "source_language": "tr",
}

# 5 — "Prepare for tomorrow's meeting with Acme Corp"
PLAN_ACME_PREP = {
    "type": "plan",
    "intent": {
        "name": "meeting_prep",
        "services": ["gcal", "gmail", "gdrive"],
        "has_write": False,
        "confidence": 0.94,
    },
    "answer_style": "prose",
    "steps": [
        {
            "id": "meeting",
            "op": "gcal.search_events",
            "args": {
                "query": "Acme",
                "window": {
                    "start": "{{windows.tomorrow.start}}",
                    "end": "{{windows.tomorrow.end}}",
                },
                "limit": 3,
            },
            "depends_on": [],
            "expect": "one",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "docs",
            "op": "gdrive.search_files",
            "args": {"query": "Acme Corp Q3 renewal proposal MSA", "limit": 5},
            "depends_on": [],
            "expect": "many",
            "optional": True,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "mail",
            "op": "gmail.search_emails",
            "args": {
                "participants": "{{meeting.hits[0].attendee_emails[*]}}",
                "query": "renewal pricing proposal",
                "limit": 8,
            },
            "depends_on": ["meeting"],
            "expect": "many",
            "optional": True,
            "freshness": "cached",
            "speculate": False,
        },
    ],
}

# 11 — the same query while Calendar is down. The only difference the planner
# makes is `freshness: live` on the meeting: "tomorrow" is not a question a
# 15-minute-stale mirror gets to answer.
PLAN_ACME_PREP_LIVE = json.loads(json.dumps(PLAN_ACME_PREP))
PLAN_ACME_PREP_LIVE["steps"][0]["freshness"] = "live"

# 6 — "Find events next week that conflict with my out-of-office doc"
PLAN_CONFLICT_OOO = {
    "type": "plan",
    "intent": {
        "name": "conflict_check",
        "services": ["gcal", "gdrive"],
        "has_write": False,
        "confidence": 0.9,
    },
    "answer_style": "template:conflict_list",
    "steps": [
        {
            "id": "ooo_doc",
            "op": "gdrive.get_file",
            "args": {
                "file_id": "{{search.gdrive[0].file_id}}",
                "include_excerpt": True,
            },
            "depends_on": [],
            "expect": "one",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "events",
            "op": "gcal.search_events",
            "args": {
                "window": {
                    "start": "{{windows.next_week.start}}",
                    "end": "{{windows.next_week.end}}",
                },
                "status_in": ["confirmed", "tentative"],
                "order_by": "starts_at",
                "limit": 50,
            },
            "depends_on": [],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "conflicts",
            "op": "meta.intersect",
            "args": {
                "items": "{{events.hits[*]}}",
                "item_start": "starts_at",
                "item_end": "ends_at",
                "against": "{{ooo_doc.extracted.ranges}}",
                "tz": "America/New_York",
            },
            "depends_on": ["ooo_doc", "events"],
            "expect": "many",
            "optional": False,
            "freshness": "cached",
            "gate": {"left": "{{ooo_doc.extracted.ranges}}", "test": "exists", "right": None},
            "speculate": False,
        },
    ],
}

# 7 — "Move the meeting with John". Two unknowns, one form prompt, and the
# ambiguity is a step rather than a subsystem.
PLAN_MOVE_MEETING_JOHN = {
    "type": "plan",
    "intent": {
        "name": "move_event",
        "services": ["gcal"],
        "has_write": True,
        "confidence": 0.62,
    },
    "answer_style": "card",
    "steps": [
        {
            "id": "disambiguate",
            "op": "ask.user",
            "args": {
                "kind": "form",
                "blocking": True,
                "question": "Which meeting, and when should it move to?",
                "help_text": "Two of your meetings next week involve a John.",
                "fields": [
                    {
                        "name": "event_id",
                        "kind": "choice",
                        "label": "Meeting",
                        "options": [
                            {
                                "id": "3k9m2p_20260825T200000Z",
                                "label": "1:1 with John Okafor",
                                "meta": {"when": "Tue Aug 25, 4:00 PM"},
                            },
                            {
                                "id": "7t4v8q_20260826T130000Z",
                                "label": "Vendor sync - John Reyes (Northwind)",
                                "meta": {"when": "Wed Aug 26, 9:00 AM"},
                            },
                        ],
                    },
                    {
                        "name": "new_time",
                        "kind": "text",
                        "label": "New time",
                        "placeholder": "e.g. Friday 3pm, or Sep 2 at 10:00",
                    },
                ],
                "value_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "enum": [
                                "3k9m2p_20260825T200000Z",
                                "7t4v8q_20260826T130000Z",
                            ],
                        },
                        "new_time": {"type": "string", "minLength": 3},
                    },
                    "required": ["event_id", "new_time"],
                },
            },
            "depends_on": [],
            "expect": "one",
            "optional": False,
            "freshness": "cached",
            "speculate": False,
        },
        {
            "id": "event",
            "op": "gcal.get_event",
            "args": {"event_id": "{{disambiguate.value.event_id}}"},
            "depends_on": ["disambiguate"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
        {
            "id": "move",
            "op": "gcal.update_event",
            "args": {
                "event_id": "{{event.event_id}}",
                "etag": "{{event.etag}}",
                "starts_at": "{{disambiguate.value.new_time|resolve_time}}",
                "duration_minutes": "{{event.duration_minutes}}",
                "send_updates": "all",
            },
            "depends_on": ["event"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
    ],
}

# 13 — two writes, one card, an ordering edge that exists purely so the email
# announcing a move cannot go out before the move.
PLAN_PUSH_ACME_REVIEW = {
    "type": "plan",
    "intent": {
        "name": "reschedule_and_notify",
        "services": ["gcal", "gmail"],
        "has_write": True,
        "confidence": 0.9,
    },
    "answer_style": "card",
    "steps": [
        {
            "id": "event",
            "op": "gcal.get_event",
            "args": {"event_id": "{{search.gcal[0].event_id}}"},
            "depends_on": [],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
        {
            "id": "move",
            "op": "gcal.update_event",
            "args": {
                "event_id": "{{event.event_id}}",
                "etag": "{{event.etag}}",
                "starts_at": "{{windows.target_slot.start}}",
                "ends_at": "{{windows.target_slot.end}}",
                "send_updates": "none",
            },
            "depends_on": ["event"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
        {
            "id": "notify",
            "op": "gmail.send_email",
            "args": {
                "to": "{{event.attendee_emails[*]|exclude(user.email)}}",
                "subject": "Moved: {{event.title}} -> Fri Aug 28, 3:00 PM",
                "body_template": "meeting_moved",
                "template_vars": {
                    "title": "{{event.title}}",
                    "old_start": "{{event.starts_at}}",
                    "new_start": "{{windows.target_slot.start}}",
                    "tz": "America/New_York",
                },
            },
            "depends_on": ["move"],
            "expect": "one",
            "optional": False,
            "freshness": "live",
            "speculate": False,
        },
    ],
}

PLANS: dict[str, dict[str, Any]] = {
    "calendar_next_week": PLAN_CALENDAR_NEXT_WEEK,
    "calendar_next_week_john": PLAN_CALENDAR_NEXT_WEEK_JOHN,
    "sarah_budget": PLAN_SARAH_BUDGET,
    "drive_pdfs_last_month": PLAN_DRIVE_PDFS,
    "cancel_turkish_flight": PLAN_CANCEL_TURKISH_FLIGHT,
    "cancel_turkish_flight_tr": PLAN_CANCEL_TURKISH_FLIGHT_TR,
    "acme_meeting_prep": PLAN_ACME_PREP,
    "acme_meeting_prep_degraded": PLAN_ACME_PREP_LIVE,
    "conflict_ooo": PLAN_CONFLICT_OOO,
    "move_meeting_john": PLAN_MOVE_MEETING_JOHN,
    "push_acme_review": PLAN_PUSH_ACME_REVIEW,
}

#: Ordered, most specific first. Every substring must appear in the prompt.
TRIGGERS: list[tuple[str, tuple[str, ...]]] = [
    ("move_meeting_john", ("move the meeting with john",)),
    ("push_acme_review", ("push my acme review",)),
    ("cancel_turkish_flight", ("cancel my turkish airlines flight",)),
    ("acme_meeting_prep", ("prepare for tomorrow", "acme")),
    ("conflict_ooo", ("conflict", "out-of-office")),
    ("conflict_ooo", ("conflict", "out of office")),
    ("calendar_next_week_john", ("calendar next week", "john@company.com")),
    ("sarah_budget", ("sarah@company.com", "budget")),
    ("drive_pdfs_last_month", ("pdfs", "drive")),
    ("calendar_next_week", ("calendar next week",)),
]

#: Returned when nothing matches: a plain answer verb, which is a legal reply
#: and keeps an unrecognised query from blowing up as a parse error.
FALLBACK_ANSWER = {
    "type": "answer",
    "text": "I do not have a fixture for that query.",
}


def scenario_for(prompt: str) -> str | None:
    """Which scenario a prompt is asking about, or None."""
    low = (prompt or "").lower()
    for slug, needles in TRIGGERS:
        if all(needle in low for needle in needles):
            return slug
    return None


def plan_for(prompt: str) -> dict[str, Any]:
    """The plan JSON for a prompt. Falls back to the `answer` verb."""
    slug = scenario_for(prompt)
    return PLANS.get(slug, FALLBACK_ANSWER) if slug else FALLBACK_ANSWER


# --------------------------------------------------------------------------- #
# Prose
# --------------------------------------------------------------------------- #

PROSE: dict[str, str] = {
    "acme_meeting_prep": (
        "Tomorrow at 10:00 you have Acme Corp - Q3 renewal review, one hour, in "
        "Meet. Four guests: Dana Whitfield and Marcus Iyer from Acme, plus Sarah "
        "Chen and you.\n\n"
        "Where things stand. Dana sent revised pricing on Tuesday - she has moved "
        "to a 14-month term at the 12-month rate and asked for the security "
        "questionnaire back before the call. Marcus raised two MSA redlines on "
        "the 11th that are still unanswered: the liability cap and data "
        "residency.\n\n"
        "Documents that matter:\n"
        "  - Acme - Q3 renewal proposal v4.gdoc (modified Aug 18, by Sarah Chen)\n"
        "  - Acme_MSA_countersigned.pdf (Jul 30 - this is the old term)\n\n"
        "Three things to decide before 10:00: whether the 14-month term is "
        "acceptable, who answers the liability cap, and whether to send the "
        "questionnaire again."
    ),
    "cancel_turkish_flight": (
        "I found your Turkish Airlines booking (6F2QK9) in an email from July 22.\n\n"
        "  - Calendar event \"Istanbul -> NYC Flight (TK1984)\" on Sat Sep 5, 10:30\n"
        "  - Drafted a cancellation email to cancel@turkishairlines.com\n\n"
        "I have not touched the calendar event."
    ),
    "sarah_budget": (
        "Two emails from Sarah Chen about the budget: the draft numbers on Aug 12 "
        "and her follow-up on Aug 17 asking to move the review."
    ),
}

#: Same query, same brief — the only difference in scenario 11 is that Calendar
#: is down, and that sentence is added by `prose_for` from the prompt itself.
PROSE["acme_meeting_prep_degraded"] = PROSE["acme_meeting_prep"]
PROSE["cancel_turkish_flight_tr"] = (
    PROSE["cancel_turkish_flight"]
    + "\n\nThe booking email is in Turkish; I read the reference, flight number "
    "and date out of it and drafted the cancellation in English."
)

DEFAULT_PROSE = "Here is what I found."

#: Words that only appear in a synthesis prompt when the caller passed failure
#: context in. Matched near a service name — see `degraded_services_in`.
_FAILURE_MARKERS = (
    "degraded",
    "failed",
    "skipped",
    "unavailable",
    "503",
    "circuit_open",
    "google_unavailable",
)

_SERVICE_WORDS = {
    "gcal": ("gcal", "calendar"),
    "gmail": ("gmail",),
    "gdrive": ("gdrive", "drive"),
}

SERVICE_LABEL = {"gcal": "Calendar", "gmail": "Gmail", "gdrive": "Drive"}

_NEAR = 240


def degraded_services_in(prompt: str) -> list[str]:
    """Which services the prompt says fell over.

    A marker on its own is not enough — "failed" turns up in plenty of innocent
    text. It counts only when a service name sits within a couple of hundred
    characters of it, which is what a `{"service": "gcal", "class": ...}` blob
    looks like.
    """
    low = (prompt or "").lower()
    hits: list[str] = []
    for marker in _FAILURE_MARKERS:
        start = 0
        while True:
            at = low.find(marker, start)
            if at < 0:
                break
            start = at + len(marker)
            window = low[max(0, at - _NEAR) : at + _NEAR]
            for service, words in _SERVICE_WORDS.items():
                if service in hits:
                    continue
                if any(word in window for word in words):
                    hits.append(service)
    return hits


def prose_for(prompt: str, scenario: str | None = None) -> str:
    """The streamed answer for a synthesis call.

    ``scenario`` is the plan that was served earlier in the same run, so the
    prose follows the plan even when the synthesis prompt carries only results
    and no longer quotes the question.

    Names a failed service **only** when the prompt carried the failure. If the
    orchestrator forgets to hand the synthesizer its ``degraded`` block, this
    answer will not mention Calendar and the degradation test will say so.
    """
    slug = scenario or scenario_for(prompt)
    body = PROSE.get(slug, DEFAULT_PROSE) if slug else DEFAULT_PROSE
    degraded = degraded_services_in(prompt)
    if not degraded:
        return body
    names = " and ".join(SERVICE_LABEL.get(s, s) for s in degraded)
    verb = "is" if len(degraded) == 1 else "are"
    return (
        f"{names} {verb} not responding - Google returned an error, so that part "
        f"of the answer is missing.\n\n{body}"
    )


__all__ = [
    "EMBED_DIM",
    "DEFAULT_MODEL",
    "DEFAULT_EMBED_MODEL",
    "PLANS",
    "PROSE",
    "TRIGGERS",
    "FALLBACK_ANSWER",
    "SERVICE_LABEL",
    "PLAN_CALENDAR_NEXT_WEEK",
    "PLAN_CALENDAR_NEXT_WEEK_JOHN",
    "PLAN_SARAH_BUDGET",
    "PLAN_DRIVE_PDFS",
    "PLAN_CANCEL_TURKISH_FLIGHT",
    "PLAN_CANCEL_TURKISH_FLIGHT_TR",
    "PLAN_ACME_PREP",
    "PLAN_ACME_PREP_LIVE",
    "PLAN_CONFLICT_OOO",
    "PLAN_MOVE_MEETING_JOHN",
    "PLAN_PUSH_ACME_REVIEW",
    "chat_completion",
    "cosine",
    "degraded_services_in",
    "embeddings_response",
    "fake_embedding",
    "plan_for",
    "prose_for",
    "scenario_for",
    "sse_chunks",
    "tokens_of",
]
