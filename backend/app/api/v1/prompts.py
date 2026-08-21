"""Cards: what the system is waiting on, and answering it.

One endpoint answers all six kinds, blocking and non-blocking alike, because
the validation authority is the row's own ``value_schema``. There is no
``if kind == "choice"`` anywhere in this file, and that is the point: adding a
seventh kind needs a new schema in an op, not a new endpoint, a client release
or a branch here.

Three things can happen when a card is answered.

* **Blocking** — the run paused on it. Answering resumes the run from the row
  it stopped at. The plan is already in ``node_executions``, so this costs
  **zero model calls**, and that is the whole reason ambiguity is a step rather
  than a special exit.
* **Yes on a confirm** — the writes the card gates go from ``draft`` to
  ``approved`` and are handed to the actions worker. The response is 202,
  because a Gmail send takes a second and nobody should watch a spinner for it.
* **Edit** — a patch on a confirm. The payload is revised, the previous one is
  kept on ``revisions``, the Gmail draft is rebuilt so what is in the person's
  own Drafts folder matches what will be sent, and a **new** card asks again.
  Still zero model calls: the change came from the person, so there is nothing
  to plan.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jsonschema
from fastapi import APIRouter, Query, Response
from jsonschema import Draft202012Validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._shared import hydrate, hydrate_fresh
from app.api.v1.schemas import (
    PromptResponseRequest,
    PromptStatusFilter,
    action_dto,
    iso,
    patch_in,
    preview_for,
    prompt_dto,
    reads_as_approval,
    status_of,
)
from app.auth.deps import CurrentUser, SessionDep
from app.core.errors import AppError
from app.core.ids import fingerprint_parts
from app.core.logging import get_logger
from app.db.models import Action, ActionStatus, PendingInput, Run
from app.db.repositories import actions as action_repo
from app.db.repositories import prompts as prompt_repo
from app.db.repositories import runs as run_repo
from app.orchestrator import runner

log = get_logger(__name__)
router = APIRouter(tags=["prompts"])

#: Ops whose prepared write is backed by a Gmail draft in the person's own
#: mailbox. Editing or cancelling one has to touch that draft too, or the
#: mailbox and the card stop agreeing.
DRAFT_BACKED = frozenset({"gmail.send_email", "gmail.draft_email"})


# --------------------------------------------------------------------------- #
# Loading and validating
# --------------------------------------------------------------------------- #


async def _load(session: AsyncSession, user_id: str, prompt_id: str) -> PendingInput:
    """A card belonging to somebody else is a 404, never a 403.

    A 403 confirms the id exists. That is a small leak and a free one to close.
    """
    row = await prompt_repo.get_prompt(session, user_id, prompt_id)
    if row is None:
        raise AppError("NOT_FOUND", "No card with that id.", details={"input_id": prompt_id})
    return row


def _require_pending(prompt: PendingInput) -> None:
    status = status_of(prompt.status)
    if status != "pending":
        raise AppError(
            "PROMPT_NOT_PENDING",
            f"That card is already {status}.",
            details={"input_id": prompt.id, "status": status},
        )


def _validate(prompt: PendingInput, value: Any) -> None:
    """The posted value against the row's schema, and nothing else.

    Format assertion is on, so ``"format": "email"`` and ``"format":
    "date-time"`` are enforced rather than advisory. Unknown keywords are
    ignored by the validator, as the spec says, so an op can annotate a schema
    with UI hints without breaking validation.
    """
    schema = prompt.value_schema or {}
    validator = Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    raise AppError(
        "PROMPT_VALUE_INVALID",
        "That answer does not fit what was asked.",
        details={
            "errors": [
                {"path": _json_path(error), "msg": error.message} for error in errors[:10]
            ],
            "schema_ref": prompt.id,
        },
    )


def _json_path(error: jsonschema.ValidationError) -> str:
    """``$.patch.to[0]`` — the failing place, in a form a client can show."""
    out = "$"
    for part in error.absolute_path:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


# --------------------------------------------------------------------------- #
# Gmail drafts that back a prepared write
# --------------------------------------------------------------------------- #


async def _google(session: AsyncSession, user_id: str) -> Any:
    from app.google.client import clients_for

    return await clients_for(session, user_id)


async def _discard_drafts(session: AsyncSession, user_id: str, rows: list[Action]) -> int:
    """Delete the Gmail drafts behind cancelled writes.

    Declining is not only a status change: the draft was created in the
    person's own account, so leaving it there means a half-written cancellation
    sitting in their Drafts folder forever. Failing to reach Google here does
    not fail the request — the card is already closed, and a stray draft is a
    smaller problem than a cancel that appears not to have worked.
    """
    targets = [r for r in rows if r.op in DRAFT_BACKED and r.external_ref]
    if not targets:
        return 0
    try:
        from app.services import gmail as gmail_service

        clients = await _google(session, user_id)
    except Exception as exc:
        log.warning("prompts.draft_cleanup_skipped", user_id=user_id, error=str(exc))
        return 0

    removed = 0
    for row in targets:
        try:
            if await gmail_service.delete_draft(clients, str(row.external_ref)):
                removed += 1
        except Exception as exc:
            log.warning(
                "prompts.draft_not_deleted",
                action_id=row.id,
                draft=row.external_ref,
                error=str(exc),
            )
    return removed


async def _rebuild_draft(session: AsyncSession, user_id: str, action: Action) -> None:
    """Replace the Gmail draft behind an edited send with the revised text.

    The send path sends *the draft*, not the payload, so an edit that only
    changed our row would send the old wording. If Google cannot be reached the
    edit is refused rather than left in that state — a card that says one thing
    and sends another is the worst outcome available here.
    """
    if action.op not in DRAFT_BACKED:
        return
    old_draft = str(action.external_ref or "")
    from app.services import gmail as gmail_service

    try:
        clients = await _google(session, user_id)
        created = await gmail_service.create_draft(clients, dict(action.payload or {}))
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "I could not update the draft in Gmail, so the edit was not saved.",
            details={"action_id": action.id, "cause": type(exc).__name__},
        ) from exc

    draft_id = str(created.get("id") or created.get("draft_id") or "")
    if not draft_id:
        raise AppError(
            "GOOGLE_UNAVAILABLE",
            "Gmail did not return a draft for the edited message.",
            details={"action_id": action.id},
        )

    payload = dict(action.payload or {})
    payload["draft_id"] = draft_id
    action.payload = payload
    await action_repo.set_external_ref(session, user_id, action.id, draft_id)
    await session.flush()

    if old_draft and old_draft != draft_id:
        try:
            await gmail_service.delete_draft(clients, old_draft)
        except Exception as exc:
            log.warning("prompts.old_draft_left", draft=old_draft, error=str(exc))


# --------------------------------------------------------------------------- #
# GET /prompts
# --------------------------------------------------------------------------- #


@router.get("/prompts")
async def list_prompts(
    session: SessionDep,
    user: CurrentUser,
    status: PromptStatusFilter = Query("pending"),
    conversation_id: str | None = Query(None),
    run_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
) -> dict[str, Any]:
    """Everything the system is waiting on, or has waited on.

    ``action`` is filled in only when a row in ``actions`` names this card as
    its ``requires_input_id``. A choice that merely disambiguates has no action,
    and says so with ``null`` rather than an empty object.
    """
    rows = await prompt_repo.list_prompts(
        session,
        user.id,
        status=None if status == "all" else status,
        run_id=run_id,
        conversation_id=conversation_id,
        limit=limit,
    )

    # Two queries for the whole page, not two per row. A hundred cards would
    # otherwise be two hundred round trips for a list nobody scrolls.
    card_ids = [row.id for row in rows]
    run_ids = {row.run_id for row in rows if row.run_id}

    gated: dict[str, dict[str, Any]] = {}
    if card_ids:
        found = await session.execute(
            select(Action)
            .where(Action.user_id == user.id, Action.requires_input_id.in_(card_ids))
            .order_by(Action.created_at.asc())
        )
        for action in found.scalars().all():
            gated.setdefault(
                action.requires_input_id,
                {
                    "id": action.id,
                    "op": action.op,
                    "status": status_of(action.status),
                    "preview": preview_for(action.op, action.payload),
                },
            )

    conversations: dict[str, str] = {}
    if run_ids:
        found = await session.execute(
            select(Run.id, Run.conversation_id).where(
                Run.user_id == user.id, Run.id.in_(run_ids)
            )
        )
        conversations = dict(found.all())

    items = [
        {
            **prompt_dto(row, conversation_id=conversations.get(row.run_id)),
            "action": gated.get(row.id),
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


# --------------------------------------------------------------------------- #
# POST /prompts/{id}/respond
# --------------------------------------------------------------------------- #


@router.post("/prompts/{prompt_id}/respond")
async def respond(
    prompt_id: str,
    body: PromptResponseRequest,
    response: Response,
    session: SessionDep,
    user: CurrentUser,
) -> Any:
    """Answer a card. Zero LLM calls, whichever branch it takes."""
    prompt = await _load(session, user.id, prompt_id)
    _require_pending(prompt)
    _validate(prompt, body.value)

    # Read what this branch needs off the row now. The runner commits, which
    # expires every loaded object, and a lazy refresh on an async session is an
    # error rather than a second query.
    blocking = bool(prompt.blocking)
    run_id = prompt.run_id

    patch = patch_in(body.value)
    approved = reads_as_approval(body.value)
    # "Edit" is a patch without a yes, and it is a confirm card that gets
    # edited — those are blocking by definition, so gating this branch on
    # `not blocking` made the Edit button unreachable on the only cards that
    # have one, and every edit fell through to the rejection path and
    # cancelled the very action it was revising.
    if patch is not None and approved is not True:
        return await _edit(session, user.id, prompt, body.value, patch)

    outcome = await runner.respond_to_prompt(
        session, user_id=user.id, prompt_id=prompt_id, value=body.value
    )

    if blocking:
        # The run picks up where it paused; the client reattaches to the stream
        # at its last seq and watches the rest of the plan execute.
        #
        # Everything from here down is presentation. `respond_to_prompt` has
        # already committed: the prompt is answered and the plan resumed. If
        # building the response body fails — an expired ORM attribute, a pool
        # checkout outside the greenlet bridge — the honest report is still
        # "answered", with a flag telling the client to refetch the thread.
        # Returning 500 for work that succeeded is the one wrong answer here:
        # the user sees a failure, retries, and answers the same card twice.
        resumed = outcome.run_id or run_id
        try:
            payload = await hydrate_fresh(user.id, outcome)
        except BaseException as exc:  # noqa: BLE001 - see the comment above
            log.warning(
                "prompts.respond_body_failed",
                cause=exc.__class__.__name__,
                prompt_id=prompt_id,
                run_id=resumed,
            )
            payload = {
                "conversation_id": getattr(outcome, "conversation_id", None),
                "run_id": resumed,
                "needs_refetch": True,
            }
        payload.update(
            {
                "input_id": prompt_id,
                "status": "answered",
                "answered_at": _now(),
                "value": body.value,
                "resumed_run_id": resumed,
                "action": None,
                "llm_calls": int((outcome.usage or {}).get("llm_calls", 0) or 0),
                "watch": f"/api/v1/runs/{resumed}/events",
            }
        )
        return payload

    return await _settled_confirm(
        session,
        user.id,
        prompt_id,
        run_id,
        body.value,
        outcome,
        response,
        declined=approved is False,
    )


def _now() -> str:
    """Now, RFC 3339 with a Z. Every timestamp on the wire looks like this."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def _settled_confirm(
    session: AsyncSession,
    user_id: str,
    prompt_id: str,
    run_id: str,
    value: Any,
    outcome: Any,
    response: Response,
    *,
    declined: bool,
) -> dict[str, Any]:
    """The body for a yes or a plain no on a confirm card."""
    gated = await action_repo.list_for_prompt(session, user_id, prompt_id)
    if declined:
        await _discard_drafts(session, user_id, gated)
        await session.commit()
        gated = await action_repo.list_for_prompt(session, user_id, prompt_id)

    fresh = await _load(session, user_id, prompt_id)
    first = gated[0] if gated else None
    shaped: dict[str, Any] | None = None
    if first is not None:
        shaped = {
            "id": first.id,
            "op": first.op,
            "status": status_of(first.status),
            "job_id": first.job_id,
            "queued_at": iso(first.updated_at),
            "watch": f"/api/v1/runs/{run_id}/events",
        }

    body = {
        "input_id": prompt_id,
        "status": status_of(fresh.status),
        "answered_at": iso(fresh.answered_at) or _now(),
        "value": value,
        "resumed_run_id": None,
        "action": shaped,
        "actions": [action_dto(a, expires_at=fresh.expires_at) for a in gated],
        "llm_calls": 0,
        "message_id": getattr(outcome, "message_id", None),
        "text": getattr(outcome, "answer", "") or "",
    }
    # 202: the worker owns the irreversible half from here, and the HTTP call
    # must not wait on Google to answer.
    response.status_code = 200 if declined else 202
    return body


async def _edit(
    session: AsyncSession,
    user_id: str,
    prompt: PendingInput,
    value: Any,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """"Edit" — patch the staged payload and ask again.

    The old payload goes onto ``actions.revisions`` so the trail of what was
    proposed survives, the old card is spent, and a fresh one gates the revised
    write. Nothing is sent, and nothing is planned.
    """
    # Everything the new card copies, read before the first write: an UPDATE
    # expires the rows it matched, and a lazy reload on an async session raises.
    card = {
        "id": prompt.id,
        "run_id": prompt.run_id,
        "message_id": prompt.message_id,
        "kind": prompt.kind,
        "value_schema": dict(prompt.value_schema or {}),
        "options": list(prompt.options) if prompt.options else None,
        "node_execution_id": prompt.node_execution_id,
        "expires_at": prompt.expires_at,
    }

    gated = await action_repo.list_for_prompt(session, user_id, prompt.id)
    editable = [a for a in gated if a.status == ActionStatus.DRAFT]
    if not editable:
        raise AppError(
            "PROMPT_NOT_PENDING",
            "There is nothing left to edit on that card.",
            details={
                "input_id": prompt.id,
                "action_status": status_of(gated[0].status) if gated else None,
            },
        )
    if len(editable) > 1:
        raise AppError(
            "VALIDATION_ERROR",
            "That card gates more than one change, so a single patch is ambiguous.",
            details={"input_id": prompt.id, "actions": [a.id for a in editable]},
        )

    run = await run_repo.get_run(session, user_id, card["run_id"])
    conversation_id = getattr(run, "conversation_id", "") or ""

    action = await action_repo.revise_action(session, user_id, editable[0].id, patch)
    await _rebuild_draft(session, user_id, action)

    await prompt_repo.answer_prompt(session, user_id, card["id"], value)

    from app.ops import registry

    op = registry.get(action.op)
    preview = preview_for(action.op, action.payload)
    question = "Send the revised message?"
    ask = getattr(op, "confirm_question", None)
    if callable(ask):
        try:
            question = str(ask(dict(action.payload or {})))
        except Exception as exc:
            log.warning("prompts.question_failed", op=action.op, error=str(exc))

    next_prompt = await prompt_repo.create_prompt(
        session,
        user_id,
        card["run_id"],
        card["message_id"],
        kind=card["kind"],
        prompt={"question": question, "help_text": "Edited just now. Still a draft."},
        value_schema=card["value_schema"],
        options=card["options"],
        blocking=False,
        node_execution_id=card["node_execution_id"],
        op=action.op,
        expires_at=card["expires_at"],
        supersede=False,
    )
    # The new card is what gates the revised write now, and the dedupe key is a
    # function of the payload, so it has to move with it: an edited message is a
    # different message, and the partial unique index has to see that.
    await session.execute(
        update(Action)
        .where(Action.id == action.id, Action.user_id == user_id)
        .values(
            requires_input_id=next_prompt.id,
            dedupe_key=fingerprint_parts(
                "action.dedupe", user_id, action.op, action.payload, conversation_id
            ),
        )
    )
    await session.commit()

    log.info("prompts.edited", action_id=action.id, input_id=card["id"], next_input=next_prompt.id)
    return {
        "input_id": card["id"],
        "status": "answered",
        "answered_at": _now(),
        "value": value,
        "resumed_run_id": None,
        "action": {
            "id": action.id,
            "op": action.op,
            "status": status_of(action.status),
            "revision": len(action.revisions or []),
            "external_ref": action.external_ref,
            "preview": preview,
        },
        "next_input": prompt_dto(next_prompt),
        "llm_calls": 0,
    }


# --------------------------------------------------------------------------- #
# POST /prompts/{id}/cancel
# --------------------------------------------------------------------------- #


@router.post("/prompts/{prompt_id}/cancel")
async def cancel(
    prompt_id: str, session: SessionDep, user: CurrentUser
) -> dict[str, Any]:
    """Dismiss a card without answering it.

    Cancelling a card that gates a write cancels the write and deletes the
    Gmail draft it created. Cancelling a **blocking** card cancels the run — the
    steps that already ran keep their rows, so the trace is intact. Nothing is
    deleted; the status moves, which is what lets a reopened chat show the card
    in its cancelled state rather than a frozen snapshot.
    """
    prompt = await _load(session, user.id, prompt_id)
    _require_pending(prompt)

    gated = await action_repo.list_for_prompt(session, user.id, prompt_id)
    result = await runner.cancel_prompt(session, user_id=user.id, prompt_id=prompt_id)
    await _discard_drafts(session, user.id, gated)
    await session.commit()

    run = await run_repo.get_run(session, user.id, prompt.run_id) if prompt.run_id else None
    first = gated[0] if gated else None
    return {
        "input_id": prompt_id,
        "status": "cancelled",
        "cancelled_at": _now(),
        "action": (
            {
                "id": first.id,
                "op": first.op,
                "status": status_of(
                    (await action_repo.get_action(session, user.id, first.id) or first).status
                ),
            }
            if first is not None
            else None
        ),
        "cancelled_actions": int(result.get("cancelled_actions", 0) or 0),
        "run_status": status_of(getattr(run, "status", None)),
        "text": result.get("answer", ""),
    }


__all__ = ["router"]
