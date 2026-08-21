"""Multi-tenant isolation, proved three ways.

The rule is one line of ``docs/contracts.md``: *every repository function takes
``user_id`` as its first argument. No exceptions.* That is not a style
preference, it is the mechanism — a query that cannot be written without a
tenant id is a query that cannot silently omit one.

Three tests, and they check different things:

1. **structural** — walk every repository module and assert the signature. This
   catches the bug before it is written;
2. **behavioural** — seed two users with deliberately identical content (same
   sender, same subject, same vendor alias, same file names) and assert that
   every read as user A returns nothing of user B's. Identical content matters:
   with different data a broken filter still looks right;
3. **the sharp one** — call each read as user A with an id that belongs to user
   B, and assert nothing comes back. Isolation that only works when you ask
   politely is not isolation.

Then the same thing at the API boundary, where a cross-tenant read has to be
indistinguishable from a missing row: 404, never 403. A 403 confirms the row
exists, which is itself a leak.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import uuid
from datetime import timedelta

import pytest

from tests.fixtures import google_responses as gr
from tests.integration.conftest import (
    load_prompts,
    make_user,
    post_query,
    require,
    seed_mirror,
    status_of,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# 1. Structural
# --------------------------------------------------------------------------- #

#: The two genuine exceptions, both from the OAuth callback, which runs before
#: a `user_id` exists. Anything else added to this list needs an argument.
TENANT_FREE = {
    "users.get_user_by_email",
    "users.create_user",
    "users.upsert_user_by_email",
    # Runs before an account exists — there is no tenant yet to scope it to.
    "users.stage_password_signup",
    "audit.record_many",
}


def repository_modules():
    package = require("app.db.repositories")
    found = []
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        found.append(importlib.import_module(f"app.db.repositories.{info.name}"))
    return found


async def test_every_repository_function_takes_user_id_first():
    """``session`` first, ``user_id`` first *real* argument, everywhere.

    A function that gets this wrong is one autocomplete away from reading
    another tenant's rows, and no amount of care at the call site fixes it.
    """
    modules = repository_modules()
    assert modules, "no repository modules found"

    offenders: list[str] = []
    checked = 0
    for module in modules:
        short = module.__name__.rsplit(".", 1)[-1]
        for name, fn in inspect.getmembers(module, inspect.iscoroutinefunction):
            if name.startswith("_") or fn.__module__ != module.__name__:
                continue
            if getattr(fn, "_no_tenant", False) or f"{short}.{name}" in TENANT_FREE:
                continue
            params = list(inspect.signature(fn).parameters)
            checked += 1
            if params[:2] != ["session", "user_id"] and params[:1] != ["user_id"]:
                offenders.append(f"{short}.{name}({', '.join(params[:3])})")

    assert checked > 20, f"only {checked} repository functions were inspected"
    assert not offenders, (
        "these take something other than user_id first:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# Seeding two tenants with the same content
# --------------------------------------------------------------------------- #


async def seed_app_rows(db, user, label: str) -> dict[str, str]:
    """One of everything in the app tables, for one user."""
    conversations = require("app.db.repositories.conversations")
    runs_repo = require("app.db.repositories.runs")
    steps_repo = require("app.db.repositories.steps")
    prompts_repo = require("app.db.repositories.prompts")
    actions_repo = require("app.db.repositories.actions")
    entities_repo = require("app.db.repositories.entities")
    audit_repo = require("app.db.repositories.audit")
    ids = require("app.core.ids")

    conversation = await conversations.create_conversation(db, user.id, title=label)
    message = await conversations.add_message(
        db,
        user.id,
        conversation.id,
        role="user",
        content=[{"type": "text", "data": {"markdown": "Cancel my Turkish Airlines flight"}}],
    )
    run = await runs_repo.create_run(db, user.id, conversation.id, message.id)
    steps = await steps_repo.insert_steps(
        db,
        user.id,
        run.id,
        conversation.id,
        [{"id": "booking", "op": "gmail.get_email", "args": {"message_id": gr.MSG_TK_BOOKING_EN}}],
    )
    assistant = await conversations.add_message(
        db,
        user.id,
        conversation.id,
        role="assistant",
        content=[{"type": "text", "data": {"markdown": "Found it."}}],
        run_id=run.id,
    )
    prompt = await prompts_repo.create_prompt(
        db,
        user.id,
        run.id,
        assistant.id,
        kind="confirm",
        prompt={"question": "Send the cancellation?"},
        value_schema={"type": "object", "properties": {"approve": {"type": "boolean"}}},
        blocking=False,
        node_execution_id=steps[0].id,
        conversation_id=conversation.id,
        op="gmail.send_email",
    )
    action = await actions_repo.create_action(
        db,
        user.id,
        assistant.id,
        requires_input_id=prompt.id,
        op="gmail.send_email",
        payload={"to": ["cancel@turkishairlines.com"], "subject": "Cancellation request"},
        dedupe_key=ids.fingerprint("test.dedupe", f"{user.id}:{label}"),
        node_execution_id=steps[0].id,
    )
    entity = await entities_repo.upsert_entity(
        db,
        user.id,
        conversation.id,
        entity_type="email",
        entity_ref=gr.MSG_TK_BOOKING_EN,
        label="Your Turkish Airlines booking is confirmed - TK1984",
        run_id=run.id,
    )
    audit_row = await audit_repo.write_audit(
        db,
        user.id,
        actor="user",
        action="gmail.send_email",
        status="ok",
        conversation_id=conversation.id,
        resource_id=action.id,
        payload={"to": ["cancel@turkishairlines.com"]},
    )
    await db.commit()

    return {
        "conversation": conversation.id,
        "message": assistant.id,
        "user_message": message.id,
        "run": run.id,
        "step": steps[0].id,
        "prompt": prompt.id,
        "action": action.id,
        "entity": entity.id,
        "audit": str(audit_row.id),
    }


@pytest.fixture
async def two_tenants(db, user, embed):
    """User A and user B, holding the same corpus and the same app rows.

    Same sender, same subject, same booking reference, same file names — so a
    filter that has quietly stopped working returns something that *looks*
    right, and only the ownership check catches it.
    """
    other = await make_user(db, email="rival@alphalaw.test", display_name="Rival Ozturk")
    a_rows = await seed_mirror(db, user.id, embed)
    b_rows = await seed_mirror(db, other.id, embed)
    assert a_rows == b_rows, "both tenants should hold identical corpora"

    a_ids = await seed_app_rows(db, user, "tenant-a")
    b_ids = await seed_app_rows(db, other, "tenant-b")
    return {"a": user, "b": other, "a_ids": a_ids, "b_ids": b_ids}


async def mirror_ids(db, user_id: str) -> dict[str, set[str]]:
    """Every mirror row id belonging to one user, per table."""
    models = require("app.db.models")
    from sqlalchemy import select

    # One table per shape, each holding every connector of that shape. The
    # tenant guard has to hold in all three, so all three are read.
    out: dict[str, set[str]] = {"gmail": set(), "gcal": set(), "gdrive": set()}
    for model in (models.SyncMessage, models.SyncEvent, models.SyncFile):
        rows = await db.execute(
            select(model.connector, model.id).where(model.user_id == user_id)
        )
        for connector, row_id in rows.all():
            out.setdefault(str(connector), set()).add(row_id)
    return out


# --------------------------------------------------------------------------- #
# 2. Behavioural — the mirror
# --------------------------------------------------------------------------- #


async def test_no_search_of_the_mirror_crosses_a_tenant(db, two_tenants, embed):
    """Search, list, get, count — as user A, over three tables of twin rows."""
    mirror = require("app.db.repositories.mirror")
    a, b = two_tenants["a"], two_tenants["b"]
    a_ids = await mirror_ids(db, a.id)
    b_ids = await mirror_ids(db, b.id)
    assert all(b_ids[t] for t in b_ids), "user B has no rows to leak"

    query = embed("Turkish Airlines booking confirmation TK1984")

    for table in ("gmail", "gcal", "gdrive"):
        hits = await mirror.hybrid_search(
            db, a.id, table, query, {"text": "Turkish Airlines booking"}, limit=25
        )
        returned = {h["id"] for h in hits}
        assert returned <= a_ids[table], (
            f"hybrid_search on {table} returned rows belonging to another user: "
            f"{sorted(returned - a_ids[table])}"
        )
        assert not returned & b_ids[table]

        window = await mirror.list_window(
            db,
            a.id,
            table,
            gr.NOW - timedelta(days=4000),
            gr.NOW + timedelta(days=4000),
            limit=100,
        )
        assert {row["id"] for row in window} <= a_ids[table], f"list_window leaked on {table}"

        counts = await mirror.counts(db, a.id)
        assert counts[table] == len(a_ids[table])

    booking = await mirror.get_by_ref(db, a.id, "gmail", gr.MSG_TK_BOOKING_EN)
    assert booking, "user A has that message too — the test is worthless if not"
    assert {row["id"] for row in booking} <= a_ids["gmail"]

    hashes = await mirror.existing_hashes(
        db, a.id, "gmail", [gr.MSG_TK_BOOKING_EN, gr.MSG_ACME_PRICING]
    )
    assert hashes, "the same refs exist for both users"
    from sqlalchemy import select

    models = require("app.db.models")
    owners = await db.execute(
        select(models.SyncMessage.user_id).where(
            models.SyncMessage.source_id.in_([gr.MSG_TK_BOOKING_EN])
        )
    )
    assert set(owners.scalars().all()) == {a.id, b.id}, (
        "both users must genuinely hold the same message id for this to prove "
        "anything"
    )


async def test_no_app_table_read_crosses_a_tenant(db, two_tenants):
    """The same sweep over the ten app tables."""
    a, b = two_tenants["a"], two_tenants["b"]
    b_ids = two_tenants["b_ids"]

    conversations = require("app.db.repositories.conversations")
    runs_repo = require("app.db.repositories.runs")
    steps_repo = require("app.db.repositories.steps")
    prompts_repo = require("app.db.repositories.prompts")
    actions_repo = require("app.db.repositories.actions")
    entities_repo = require("app.db.repositories.entities")
    audit_repo = require("app.db.repositories.audit")

    mine = await conversations.list_conversations(db, a.id)
    assert mine, "user A has a conversation"
    assert b_ids["conversation"] not in {c.id for c in mine}

    assert await conversations.get_conversation(db, a.id, b_ids["conversation"]) is None
    assert await conversations.get_message(db, a.id, b_ids["message"]) is None
    assert not await conversations.list_messages(db, a.id, b_ids["conversation"])

    assert await runs_repo.get_run(db, a.id, b_ids["run"]) is None
    assert not await runs_repo.list_runs(db, a.id, conversation_id=b_ids["conversation"])
    assert b_ids["run"] not in {r.id for r in await runs_repo.list_runs(db, a.id)}
    assert await runs_repo.latest_run(db, a.id, b_ids["conversation"]) is None
    assert b_ids["run"] not in {r.id for r in await runs_repo.list_resumable(db, a.id)}

    assert await steps_repo.get_step(db, a.id, b_ids["step"]) is None
    assert not await steps_repo.list_steps(db, a.id, b_ids["run"])
    assert not await steps_repo.list_steps_for_message(db, a.id, b_ids["message"])
    assert not await steps_repo.list_steps_for_conversation(db, a.id, b_ids["conversation"])
    assert await steps_repo.get_step_by_node(db, a.id, b_ids["run"], "booking") is None

    assert await prompts_repo.get_prompt(db, a.id, b_ids["prompt"]) is None
    assert b_ids["prompt"] not in {p.id for p in await prompts_repo.list_prompts(db, a.id)}
    assert not await prompts_repo.list_for_message(db, a.id, b_ids["message"])
    assert await prompts_repo.get_blocking_for_run(db, a.id, b_ids["run"]) is None

    assert await actions_repo.get_action(db, a.id, b_ids["action"]) is None
    assert b_ids["action"] not in {x.id for x in await actions_repo.list_actions(db, a.id)}
    assert not await actions_repo.list_for_prompt(db, a.id, b_ids["prompt"])
    assert not await actions_repo.list_for_message(db, a.id, b_ids["message"])
    assert await actions_repo.expiry_of(db, a.id, b_ids["action"]) is None

    assert not await entities_repo.list_entities(db, a.id, b_ids["conversation"])
    assert (
        await entities_repo.get_entity(
            db, a.id, b_ids["conversation"], "email", gr.MSG_TK_BOOKING_EN
        )
        is None
    )
    assert not await entities_repo.search_entities(db, a.id, b_ids["conversation"], "Turkish")

    b_audit = await audit_repo.list_audit(db, b.id)
    a_audit = await audit_repo.list_audit(db, a.id)
    assert b_audit and a_audit, "both users have audit rows"
    assert {row.id for row in a_audit}.isdisjoint({row.id for row in b_audit})
    assert not await audit_repo.list_audit(db, a.id, conversation_id=b_ids["conversation"])


async def test_the_dedupe_key_does_not_collide_across_tenants(db, two_tenants):
    """Two tenants may prepare the same write at the same moment.

    ``dedupe_key`` is uuid5 over user, op, payload *and* conversation, so the
    partial unique index cannot make one tenant's pending send block another's.
    """
    actions_repo = require("app.db.repositories.actions")
    a, b = two_tenants["a"], two_tenants["b"]

    a_action = await actions_repo.get_action(db, a.id, two_tenants["a_ids"]["action"])
    b_action = await actions_repo.get_action(db, b.id, two_tenants["b_ids"]["action"])
    assert a_action and b_action
    assert a_action.dedupe_key != b_action.dedupe_key, (
        "identical payloads from two tenants produced the same fingerprint"
    )

    assert await actions_repo.find_in_flight(db, a.id, b_action.dedupe_key) is None


async def test_a_shared_content_hash_does_not_share_a_row(db, two_tenants):
    """The same email in two mailboxes is two rows, and two vectors.

    ``content_hash`` is a fingerprint of the text, so it is equal across
    tenants on purpose. What must never be shared is the row.
    """
    from sqlalchemy import select

    models = require("app.db.models")
    a, b = two_tenants["a"], two_tenants["b"]

    rows = await db.execute(
        select(
            models.SyncMessage.user_id, models.SyncMessage.id, models.SyncMessage.content_hash
        )
        .where(models.SyncMessage.source_id == gr.MSG_TK_BOOKING_EN)
        .order_by(models.SyncMessage.user_id)
    )
    found = rows.all()
    assert len(found) == 2, f"expected one row per tenant, got {len(found)}"
    assert {r[0] for r in found} == {a.id, b.id}
    assert found[0][1] != found[1][1], "two tenants, two rows"
    assert isinstance(found[0][2], uuid.UUID)
    assert found[0][2] == found[1][2], (
        "the same text should fingerprint the same either way — that is what "
        "makes the embedding cache worth having"
    )


# --------------------------------------------------------------------------- #
# 3. The API boundary
# --------------------------------------------------------------------------- #


async def test_responding_to_another_users_prompt_is_refused(
    client, db, two_tenants
):
    """The card belongs to user B. User A cannot answer it, and cannot even
    learn that it exists — a cross-tenant read is a 404, not a 403."""
    b_ids = two_tenants["b_ids"]

    response = await client.post(
        f"/api/v1/prompts/{b_ids['prompt']}/respond", json={"value": {"approve": True}}
    )
    assert response.status_code == 404, (
        f"expected 404 (indistinguishable from missing), got {response.status_code}: "
        f"{response.text[:400]}"
    )
    assert response.json()["error"]["code"] == "NOT_FOUND"

    cancelled = await client.post(f"/api/v1/prompts/{b_ids['prompt']}/cancel")
    assert cancelled.status_code == 404, cancelled.text

    still = await load_prompts(db, two_tenants["b"].id)
    untouched = [p for p in still if p.id == b_ids["prompt"]]
    assert untouched and status_of(untouched[0]) == "pending", (
        "user B's card was touched by a request from user A"
    )


async def test_listing_endpoints_only_show_your_own_rows(client, two_tenants):
    """Conversations and prompts, as user A."""
    a_ids, b_ids = two_tenants["a_ids"], two_tenants["b_ids"]

    listed = await client.get("/api/v1/conversations")
    assert listed.status_code == 200, listed.text
    body = listed.json()
    shown = {c["id"] for c in (body.get("items") or body.get("conversations") or [])}
    assert a_ids["conversation"] in shown, f"user A's own thread is missing: {body}"
    assert b_ids["conversation"] not in shown

    detail = await client.get(f"/api/v1/conversations/{b_ids['conversation']}")
    assert detail.status_code == 404, detail.text

    prompts = await client.get("/api/v1/prompts?status=pending")
    assert prompts.status_code == 200, prompts.text
    listed_ids = {p["id"] for p in (prompts.json().get("items") or [])}
    assert b_ids["prompt"] not in listed_ids
    assert a_ids["prompt"] in listed_ids, "user A's own card should be there"


async def test_a_query_into_another_users_conversation_is_not_found(
    client, two_tenants
):
    """Continuing somebody else's thread is a 404 too."""
    response = await client.post(
        "/api/v1/query",
        json={
            "query": "What's on my calendar next week?",
            "conversation_id": two_tenants["b_ids"]["conversation"],
        },
    )
    assert response.status_code == 404, (
        f"expected 404 for another tenant's conversation_id: {response.status_code} "
        f"{response.text[:400]}"
    )


async def test_two_tenants_asking_the_same_question_get_their_own_answers(
    db, client, client_for, two_tenants
):
    """The end-to-end version: same query, same corpus, two users, one process.

    Each run has to belong to the user that asked for it. Reading either run as
    the other user gets nothing, which is the same property as above but now
    with the whole request path in between.
    """
    runs_repo = require("app.db.repositories.runs")
    a, b = two_tenants["a"], two_tenants["b"]
    b_client = await client_for(b)

    first = await post_query(client, "What's on my calendar next week?")
    second = await post_query(b_client, "What's on my calendar next week?")

    assert first["conversation_id"] != second["conversation_id"]
    assert first["run_id"] != second["run_id"]

    await db.rollback()
    assert await runs_repo.get_run(db, a.id, first["run_id"]) is not None
    assert await runs_repo.get_run(db, a.id, second["run_id"]) is None
    assert await runs_repo.get_run(db, b.id, first["run_id"]) is None
    assert await runs_repo.get_run(db, b.id, second["run_id"]) is not None
