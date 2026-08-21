"""Executing an approved write. The only place a side effect reaches Google.

The order of operations is the whole design, and it is deliberate:

1. **Sequencing.** ``payload._sequence_after`` names actions that must finish
   first. While one is still in flight this task re-queues itself; if one
   *failed*, the successor is **held**, never run. "Reply to Dave, then delete
   the thread" must not delete the thread when the reply never went out.
2. **Idempotency.** ``attempts > 0`` means a previous attempt reached Google and
   we did not hear back. Before sending anything we look in Sent for
   ``rfc822msgid:{dedupe_key}@send.alpha-law.app`` — the deterministic
   ``Message-ID`` the op stamps on the outgoing mail, the searchable twin of the
   ``X-Orchestrator-Idem`` header. A hit means it already went; we adopt the
   message id and stop.
3. **Revalidation.** Refetch the thing being changed. A calendar event whose
   etag has moved is a 412: somebody edited it since the card was drawn, so we
   re-render rather than execute. A Gmail draft that has gone missing on a first
   attempt is recreated from the payload — the draft is a cache, the payload is
   the truth.
4. **Claim, execute, record.** The move to ``running`` is a conditional update,
   so only one worker wins. Then the op runs, an ``audit_log`` row is written,
   a real assistant message goes into the conversation, and the outcome is
   published on the conversation's SSE channel.

Nothing here decides *whether* to write. That was decided when a person
approved the card.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core import audit as audit_core
from app.core.ids import canonical_json
from app.core.logging import get_logger
from app.db.models import Action, ActionStatus
from app.db.repositories import actions as actions_repo
from app.db.repositories import audit as audit_repo
from app.db.repositories import conversations as conversations_repo
from app.db.repositories import steps as steps_repo
from app.db.repositories import users as users_repo
from app.db.session import session_scope
from app.tasks import classify_error, error_payload, http_status, utcnow
from app.tasks.celery_app import AppTask, NonRetryable, celery_app, run_async

log = get_logger(__name__)

#: Payload key naming the actions that must land first.
SEQUENCE_KEY = "_sequence_after"
# `app/orchestrator/dispatch.py` writes this key when it stages a write whose
# step depends on another staged write. Spelled in both places rather than
# imported (dispatch must not import the Celery package), so the two are
# checked against each other here — a rename in one file alone would silently
# stop enforcing write ordering, which is the failure this key exists to
# prevent.
_dispatch_key = getattr(
    __import__("app.orchestrator.dispatch", fromlist=["SEQUENCE_KEY"]),
    "SEQUENCE_KEY",
    SEQUENCE_KEY,
)
if _dispatch_key != SEQUENCE_KEY:  # pragma: no cover - a guard, not a path
    raise RuntimeError(
        f"sequencing key mismatch: dispatch writes {_dispatch_key!r}, "
        f"the actions worker reads {SEQUENCE_KEY!r}"
    )

#: The domain half of the deterministic Message-ID.
IDEM_DOMAIN = "send.alpha-law.app"

#: Gmail's search index lags a send by a second or two. Waiting is cheaper than
#: sending twice.
SENT_SEARCH_DELAY_S = 3.0

#: How long to wait between checks on a predecessor that is still running.
SEQUENCE_RETRY_S = 15


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


async def _load(session: Any, action_id: str, user_id: str | None) -> Action | None:
    """The action, scoped to its owner when the caller knew it.

    ``user_id`` is optional here and nowhere else: the queue message may carry
    only the action id, and an action id is a 21-character nanoid nobody can
    guess. Every call after this one is scoped by the owner it resolves.
    """
    if user_id:
        return await actions_repo.get_action(session, user_id, action_id)
    return await session.get(Action, action_id)


async def _clients(user_id: str) -> Any:
    from app.google.client import clients_for

    async with session_scope() as session:
        return await clients_for(session, user_id)


# --------------------------------------------------------------------------- #
# Talking back to the conversation
# --------------------------------------------------------------------------- #


async def _announce(
    session: Any,
    user_id: str,
    action: Action,
    *,
    markdown: str,
) -> tuple[str | None, str | None]:
    """Append a real assistant message reporting the outcome.

    Not a toast, not an SSE-only event: a message in the transcript, so
    reopening the conversation tomorrow still shows what happened.
    """
    origin = await conversations_repo.get_message(session, user_id, action.message_id)
    if origin is None:
        return None, None
    message = await conversations_repo.add_message(
        session,
        user_id,
        origin.conversation_id,
        role="assistant",
        content=[
            {"type": "text", "data": {"markdown": markdown}},
            {"type": "action", "ref": action.id},
        ],
        run_id=origin.run_id,
    )
    return origin.conversation_id, message.id


async def _publish(
    conversation_id: str | None, event: str, data: dict[str, Any]
) -> None:
    """Push the outcome onto the conversation's channel.

    A failure here is logged and swallowed. The action happened; a browser that
    missed the event will see the message when it reloads.
    """
    if not conversation_id:
        return
    try:
        from app.orchestrator.events import publish

        await publish(conversation_id, event, data)
    except Exception as exc:  # noqa: BLE001 - never fail a write over a fan-out
        log.warning(
            "actions.publish_failed",
            conversation_id=conversation_id,
            event=event,
            error=str(exc)[:200],
        )


def _summary(op: str, payload: dict[str, Any]) -> str:
    """A short, plain description of what an action was going to do."""
    if op.startswith("gmail.send") or op.startswith("gmail.reply"):
        to = payload.get("to") or payload.get("recipients") or []
        if isinstance(to, str):
            to = [to]
        who = ", ".join(str(t) for t in to[:3]) or "the recipient"
        return f"the email to {who}"
    if op.startswith("gcal."):
        return f"the event \"{payload.get('title') or payload.get('summary') or ''}\"".strip()
    if op.startswith("gdrive."):
        return f"the file \"{payload.get('name') or payload.get('file_id') or ''}\"".strip()
    return "that change"


# --------------------------------------------------------------------------- #
# Sequencing
# --------------------------------------------------------------------------- #


def _sequence_refs(payload: dict[str, Any]) -> list[str]:
    value = payload.get(SEQUENCE_KEY)
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


async def _sequence_verdict(user_id: str, refs: list[str]) -> tuple[str, str | None]:
    """``("clear"|"waiting"|"blocked", blocking_action_id)``."""
    async with session_scope() as session:
        for ref in refs:
            predecessor = await actions_repo.get_action(session, user_id, ref)
            if predecessor is None:
                # The step it was waiting on no longer exists. Running blind is
                # worse than stopping.
                return "blocked", ref
            if predecessor.status not in actions_repo.TERMINAL_STATUSES:
                return "waiting", ref
            if predecessor.status != ActionStatus.DONE:
                return "blocked", ref
    return "clear", None


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def _idem_message_id(dedupe_key: Any) -> str:
    return f"{dedupe_key}@{IDEM_DOMAIN}"


async def _already_sent(
    user_id: str,
    op: str,
    external_ref: str | None,
    dedupe_key: Any,
) -> dict[str, Any] | None:
    """Did a previous attempt already get through?

    Two signals. The cheap one: we execute ``drafts.send``, so if the draft id
    has gone, it was sent. The certain one: Gmail indexes the ``Message-ID`` we
    stamped, and ``in:sent rfc822msgid:...`` finds exactly our message and
    nobody else's.
    """
    if not op.startswith("gmail."):
        return None

    from app.services import gmail as gmail_api

    await asyncio.sleep(SENT_SEARCH_DELAY_S)
    clients = await _clients(user_id)

    draft_gone = False
    if external_ref:
        try:
            draft_gone = (await gmail_api.get_draft(clients, external_ref)) is None
        except Exception as exc:  # noqa: BLE001
            if classify_error(exc) == "NOT_FOUND" or http_status(exc) == 404:
                draft_gone = True
            else:
                raise

    hits = await gmail_api.search_sent(
        clients, f"in:sent rfc822msgid:{_idem_message_id(dedupe_key)}", max_results=1
    )
    if hits:
        hit = hits[0]
        return {
            "message_id": hit.get("id"),
            "thread_id": hit.get("threadId"),
            "adopted": True,
            "signal": "rfc822msgid",
        }
    if draft_gone:
        return {"message_id": None, "adopted": True, "signal": "draft_gone"}
    return None


# --------------------------------------------------------------------------- #
# Revalidation
# --------------------------------------------------------------------------- #


async def _revalidate(
    user_id: str, op: str, payload: dict[str, Any], external_ref: str | None
) -> dict[str, Any]:
    """Refetch what is about to change.

    Returns ``{"outcome": "fresh"|"stale", ...}``. A stale result never
    executes: the card is re-rendered against what the object looks like now.
    """
    if op.startswith("gcal.") and payload.get("event_id") and payload.get("etag"):
        from app.services import gcal as gcal_api

        clients = await _clients(user_id)
        try:
            current = await gcal_api.events_get(
                clients,
                payload["event_id"],
                calendar_id=payload.get("calendar_id") or "primary",
            )
        except Exception as exc:  # noqa: BLE001
            if classify_error(exc) == "NOT_FOUND" or http_status(exc) == 404:
                return {"outcome": "stale", "reason": "gone", "code": 404}
            raise
        if current is None:
            return {"outcome": "stale", "reason": "gone", "code": 404}
        etag = current.get("etag")
        if etag and etag != payload["etag"]:
            return {
                "outcome": "stale",
                "reason": "etag_mismatch",
                "code": 412,
                "current": {"etag": etag, "updated": current.get("updated")},
            }
        return {"outcome": "fresh"}

    if op.startswith("gmail.") and external_ref:
        from app.services import gmail as gmail_api

        clients = await _clients(user_id)
        try:
            draft = await gmail_api.get_draft(clients, external_ref)
        except Exception as exc:  # noqa: BLE001
            if classify_error(exc) == "NOT_FOUND" or http_status(exc) == 404:
                draft = None
            else:
                raise
        if draft is None:
            # First attempt and the draft has gone: somebody deleted it from
            # their mailbox. The payload is what the person approved, so make
            # the draft again rather than failing.
            created = await gmail_api.create_draft(clients, payload)
            new_ref = (created or {}).get("id")
            if not new_ref:
                raise NonRetryable(
                    "could not recreate the Gmail draft",
                    error_class="INVALID",
                    details={"op": op},
                )
            return {"outcome": "fresh", "external_ref": str(new_ref), "recreated": True}
        return {"outcome": "fresh"}

    return {"outcome": "fresh"}


# --------------------------------------------------------------------------- #
# Terminal writes
# --------------------------------------------------------------------------- #


async def _settle(
    user_id: str,
    action_id: str,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    markdown: str,
    event: str,
    audit_status: str,
    external_ref: str | None = None,
) -> dict[str, Any]:
    """Move the action, write the audit row, say so in the conversation.

    One transaction, so the transcript can never disagree with the row.
    """
    async with session_scope() as session:
        action = await actions_repo.require_action(session, user_id, action_id)
        if status in ("done", "failed") and action.status == ActionStatus.APPROVED:
            # The adopt-a-previous-send path never claimed the row. Take it now,
            # so the state machine still only ever goes approved -> running -> done.
            action = await actions_repo.claim_action(session, user_id, action_id)
        if status == "done":
            action = await actions_repo.complete_action(
                session, user_id, action_id, result or {}, external_ref=external_ref
            )
        elif status == "failed":
            action = await actions_repo.fail_action(
                session, user_id, action_id, error or {}
            )
        else:  # held or stale: the write never ran
            action = await actions_repo.cancel_action(
                session, user_id, action_id, reason=(error or {}).get("reason", status)
            )

        origin = await conversations_repo.get_message(
            session, user_id, action.message_id
        )
        conversation_id = origin.conversation_id if origin else None

        await audit_repo.write_audit(
            session,
            user_id,
            actor="worker",
            action=action.op,
            status=audit_status,
            conversation_id=conversation_id,
            resource_id=(result or {}).get("message_id")
            or (result or {}).get("id")
            or action.external_ref,
            payload=action.payload,
            payload_visible=audit_core.visible_fields(action.payload),
            error=error,
        )
        await _announce(session, user_id, action, markdown=markdown)

    await _publish(
        conversation_id,
        event,
        {
            "action_id": action_id,
            "op": action.op,
            "status": action.status.value,
            "result": result,
            "error": error,
        },
    )
    return {
        "status": action.status.value,
        "action_id": action_id,
        "op": action.op,
    }


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #


async def execute_async(action_id: str, user_id: str | None = None) -> dict[str, Any]:
    """Execute one approved action, in the caller's event loop.

    The public async half of the ``actions.execute`` task. The Celery task is a
    sync wrapper around this, and anything already running a loop — the API
    path, a test, another worker — calls this instead of paying for a nested
    ``asyncio.run``.
    """
    return await _execute(action_id, user_id)


async def _execute(action_id: str, user_id: str | None) -> dict[str, Any]:
    async with session_scope() as session:
        action = await _load(session, action_id, user_id)
        if action is None:
            log.warning("actions.missing", action_id=action_id)
            return {"status": "missing", "action_id": action_id}
        owner = action.user_id
        op_name = action.op
        payload = dict(action.payload or {})
        attempts = int(action.attempts or 0)
        external_ref = action.external_ref
        dedupe_key = action.dedupe_key
        node_execution_id = action.node_execution_id
        current = action.status

    if current != ActionStatus.APPROVED:
        # Somebody already took it, or the person changed their mind. Both are
        # fine; neither is a failure.
        log.info(
            "actions.not_approved",
            action_id=action_id,
            status=current.value,
        )
        return {"status": "skipped", "action_status": current.value}

    # 1 — sequencing ---------------------------------------------------------
    refs = _sequence_refs(payload)
    if refs:
        verdict, blocking = await _sequence_verdict(owner, refs)
        if verdict == "waiting":
            return {
                "status": "waiting",
                "action_id": action_id,
                "waiting_on": blocking,
                "retry_in": SEQUENCE_RETRY_S,
            }
        if verdict == "blocked":
            return await _settle(
                owner,
                action_id,
                status="held",
                error={"reason": "predecessor_failed", "predecessor": blocking},
                markdown=(
                    f"I did not send {_summary(op_name, payload)}: the step it depended "
                    "on did not go through. Nothing was changed. Ask me again and I "
                    "will start over."
                ),
                event="action.failed",
                audit_status="held",
            )

    # 2 — idempotency --------------------------------------------------------
    if attempts > 0:
        found = await _already_sent(owner, op_name, external_ref, dedupe_key)
        if found is not None:
            log.info(
                "actions.already_sent",
                action_id=action_id,
                signal=found.get("signal"),
            )
            return await _settle(
                owner,
                action_id,
                status="done",
                result=found,
                markdown=(
                    f"{_summary(op_name, payload).capitalize()} had already gone out on "
                    "the earlier attempt, so I did not send it again."
                ),
                event="action.done",
                audit_status="ok",
                external_ref=external_ref,
            )

    # 3 — revalidation -------------------------------------------------------
    try:
        check = await _revalidate(owner, op_name, payload, external_ref)
    except NonRetryable:
        raise
    except Exception as exc:
        log.warning(
            "actions.revalidate_failed", action_id=action_id, error=str(exc)[:300]
        )
        raise

    if check["outcome"] == "stale":
        return await _settle(
            owner,
            action_id,
            status="stale",
            error={
                "reason": check.get("reason", "stale"),
                "class": "PRECONDITION",
                "code": check.get("code", 412),
                "current": check.get("current"),
            },
            markdown=(
                f"I did not change {_summary(op_name, payload)}: it has moved since I "
                "drew that card"
                + (" — someone else edited it." if check.get("reason") == "etag_mismatch"
                   else " — it is no longer there.")
                + " Ask me again and I will read the current version first."
            ),
            event="action.failed",
            audit_status="stale",
        )

    if check.get("external_ref"):
        external_ref = str(check["external_ref"])
        async with session_scope() as session:
            await actions_repo.set_external_ref(
                session, owner, action_id, external_ref
            )

    # 4 — claim, then execute -----------------------------------------------
    async with session_scope() as session:
        try:
            await actions_repo.claim_action(session, owner, action_id)
        except Exception as exc:  # noqa: BLE001 - another worker won the race
            log.info("actions.claim_lost", action_id=action_id, error=str(exc)[:200])
            return {"status": "skipped", "action_id": action_id, "reason": "claimed"}

    try:
        result = await _run_op(
            owner, action_id, op_name, payload, node_execution_id, external_ref
        )
    except Exception as exc:
        error_class = classify_error(exc)
        try:
            await _settle(
                owner,
                action_id,
                status="failed",
                error={**error_payload(exc), "class": error_class},
                markdown=(
                    f"I could not send {_summary(op_name, payload)}. "
                    f"Google said: {str(exc)[:200]}. Nothing was changed."
                ),
                event="action.failed",
                audit_status="error",
            )
        except Exception as settle_error:  # noqa: BLE001 - keep the real failure
            log.error(
                "actions.settle_failed",
                action_id=action_id,
                error=str(settle_error)[:300],
            )
        if error_class in {"AUTH_REVOKED", "INVALID", "NOT_FOUND", "PRECONDITION"}:
            raise NonRetryable(
                f"{op_name} cannot succeed: {exc}",
                error_class=error_class,
                cause=exc,
                details={"action_id": action_id},
            ) from exc
        raise

    return await _settle(
        owner,
        action_id,
        status="done",
        result=result,
        markdown=_done_text(op_name, payload, result),
        event="action.done",
        audit_status="ok",
        external_ref=str(result.get("id") or external_ref or "") or None,
    )


def _done_text(op: str, payload: dict[str, Any], result: dict[str, Any]) -> str:
    if op.startswith("gmail.send") or op.startswith("gmail.reply"):
        return f"Sent {_summary(op, payload)}."
    if op.startswith("gcal.delete") or op.endswith(".delete_event"):
        return f"Deleted {_summary(op, payload)}."
    if op.startswith("gcal."):
        return f"Updated {_summary(op, payload)}."
    if op.startswith("gdrive."):
        return f"Done — {_summary(op, payload)}."
    return "Done."


async def _run_op(
    user_id: str,
    action_id: str,
    op_name: str,
    payload: dict[str, Any],
    node_execution_id: str | None,
    external_ref: str | None,
) -> dict[str, Any]:
    """Hand the payload to the op that owns it.

    The op is the only thing that knows how to talk to Google for this verb —
    including stamping the deterministic ``Message-ID`` and the
    ``X-Orchestrator-Idem`` header that step 2 searches for.
    """
    from app.ops.base import OpContext
    from app.ops.registry import get as get_op

    op = get_op(op_name)
    execute = getattr(op, "execute", None)
    if execute is None:
        raise NonRetryable(
            f"{op_name} has no execute(); only a confirmable op can be an action",
            error_class="INVALID",
            details={"op": op_name, "action_id": action_id},
        )

    clients = await _clients(user_id)
    async with session_scope() as session:
        user = await users_repo.get_user(session, user_id)
        action = await actions_repo.require_action(session, user_id, action_id)
        origin = await conversations_repo.get_message(
            session, user_id, action.message_id
        )
        run_id = origin.run_id if origin else None
        if run_id is None and node_execution_id:
            step = await steps_repo.get_step(session, user_id, node_execution_id)
            run_id = step.run_id if step else None

        ctx = OpContext(
            user_id=user_id,
            conversation_id=origin.conversation_id if origin else "",
            run_id=run_id or "",
            session=session,
            google=clients,
            now=utcnow(),
            tz=(user.timezone if user else "UTC"),
        )
        body = dict(payload)
        if external_ref:
            body.setdefault("draft_id", external_ref)
        result = await execute(ctx, body)

    if not isinstance(result, dict):
        return {"result": canonical_json(result)}
    return result


@celery_app.task(
    base=AppTask,
    bind=True,
    name="actions.execute",
    queue="actions",
    user_arg=1,
    max_retries=3,
)
def execute_action(
    self: AppTask, action_id: str, user_id: str | None = None
) -> dict[str, Any]:
    """Execute one approved action.

    Safe to deliver twice: an action that is not ``approved`` is left alone, the
    claim is a conditional update only one worker wins, and a retry after a lost
    response looks in Sent before it sends anything.
    """
    outcome = run_async(_execute(action_id, user_id))
    if outcome.get("status") == "waiting":
        # Retrying is how we wait: holding a worker on a predecessor that may
        # take minutes would occupy the queue a person is waiting on.
        raise self.retry(
            countdown=int(outcome.get("retry_in", SEQUENCE_RETRY_S)),
            max_retries=8,
        )
    return outcome


__all__ = [
    "SEQUENCE_KEY",
    "IDEM_DOMAIN",
    "SENT_SEARCH_DELAY_S",
    "execute_action",
    "execute_async",
]
