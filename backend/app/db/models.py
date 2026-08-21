"""The 15 tables from docs/schema.md, one to one. Locked.

10 app tables, 4 sync tables, 1 job table. Ids are CHAR(21) nanoids generated in
the service (no database defaults) except ``audit_log.id``, which is a
BIGSERIAL on purpose. The only UUIDs are content fingerprints.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, ENUM, INET, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from app.db.base import Base

EMBEDDING_DIM = 1536

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class NodeStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class InputKind(enum.StrEnum):
    CONFIRM = "confirm"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    TEXT = "text"
    FORM = "form"
    DATE_RANGE = "date_range"


class InputStatus(enum.StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ActionStatus(enum.StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SyncService(enum.StrEnum):
    GMAIL = "gmail"
    GCAL = "gcal"
    GDRIVE = "gdrive"


def _pg_enum(python_enum: type[enum.StrEnum], name: str) -> ENUM:
    """Postgres ENUM bound to a StrEnum, stored by value."""
    return ENUM(
        python_enum,
        name=name,
        values_callable=lambda e: [m.value for m in e],
        native_enum=True,
    )


MESSAGE_ROLE = _pg_enum(MessageRole, "message_role")
RUN_STATUS = _pg_enum(RunStatus, "run_status")
NODE_STATUS = _pg_enum(NodeStatus, "node_status")
INPUT_KIND = _pg_enum(InputKind, "input_kind")
INPUT_STATUS = _pg_enum(InputStatus, "input_status")
ACTION_STATUS = _pg_enum(ActionStatus, "action_status")
SYNC_SERVICE = _pg_enum(SyncService, "sync_service")

_NOW = text("now()")

# --------------------------------------------------------------------------- #
# App tables
# --------------------------------------------------------------------------- #


class User(Base):
    """Identity, plus the two settings that change how a query is read."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    email: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    display_name: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=text("'UTC'")
    )
    work_week_start: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )

    #: NULL for an account that only ever signed in with Google. A password can
    #: be added to such an account later — the emailed code proves the mailbox,
    #: which is the same thing Google proved.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    #: True once a code sent to this address has been entered correctly.
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )


class OAuthToken(Base):
    """One row per user per provider. Another provider is a row, not a migration."""

    __tablename__ = "oauth_tokens"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'google'")
    )
    provider_account_id: Mapped[str | None] = mapped_column(String(128))
    #: The address of the connected workspace, which need not be the address
    #: on the account — you can sign in with one and connect another.
    account_email: Mapped[str | None] = mapped_column(String(320))

    access_token_enc: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    refresh_token_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    key_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text()), nullable=False)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    refresh_failures: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="oauth_tokens_user_id_provider_key"),
        Index(
            "oauth_tokens_expires_at_idx",
            "expires_at",
            postgresql_where=text("revoked_at IS NULL"),
        ),
        Index(
            "oauth_tokens_provider_provider_account_id_idx",
            "provider",
            "provider_account_id",
            unique=True,
            postgresql_where=text("provider_account_id IS NOT NULL"),
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text)
    archived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_message_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        Index(
            "conversations_user_id_last_message_at_idx",
            "user_id",
            text("last_message_at DESC"),
            postgresql_where=text("archived_at IS NULL"),
        ),
    )


class Message(Base):
    """Who said what, in what order. Nothing about execution."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    # Circular with runs.trigger_message_id — the FK is added after both tables
    # exist, hence use_alter.
    run_id: Mapped[str | None] = mapped_column(
        CHAR(21),
        ForeignKey("runs.id", ondelete="SET NULL", use_alter=True, name="messages_run_id_fkey"),
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)

    role: Mapped[MessageRole] = mapped_column(MESSAGE_ROLE, nullable=False)
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    hidden: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="messages_conversation_id_seq_key"),
        Index("messages_conversation_id_seq_idx", "conversation_id", "seq"),
        Index("messages_run_id_idx", "run_id", postgresql_where=text("run_id IS NOT NULL")),
    )


class Run(Base):
    """One execution of classify -> plan -> execute -> synthesize."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    trigger_message_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )

    intent: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    planner_tier: Mapped[int | None] = mapped_column(SmallInteger)

    status: Mapped[RunStatus] = mapped_column(
        RUN_STATUS, nullable=False, server_default=text("'running'")
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("runs_conversation_id_started_at_idx", "conversation_id", text("started_at DESC")),
        Index("runs_user_id_started_at_idx", "user_id", text("started_at DESC")),
        Index(
            "runs_status_idx",
            "status",
            postgresql_where=text("status IN ('running', 'awaiting_input')"),
        ),
    )


class NodeExecution(Base):
    """One row per step. Progress feed, step trace, and the record afterwards."""

    __tablename__ = "node_executions"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    message_id: Mapped[str | None] = mapped_column(
        CHAR(21), ForeignKey("messages.id", ondelete="SET NULL")
    )

    seq: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    node_id: Mapped[str] = mapped_column(String(64), nullable=False)
    op: Mapped[str] = mapped_column(String(64), nullable=False)
    round: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    args: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    depends_on: Mapped[list[str]] = mapped_column(
        ARRAY(Text()), nullable=False, server_default=text("'{}'::text[]")
    )

    status: Mapped[NodeStatus] = mapped_column(
        NODE_STATUS, nullable=False, server_default=text("'pending'")
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    outcome: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    retries: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)

    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "run_id", "node_id", "round", name="node_executions_run_id_node_id_round_key"
        ),
        Index("node_executions_run_id_seq_idx", "run_id", "seq"),
        Index(
            "node_executions_message_id_seq_idx",
            "message_id",
            "seq",
            postgresql_where=text("message_id IS NOT NULL"),
        ),
        Index(
            "node_executions_conversation_id_started_at_idx",
            "conversation_id",
            text("started_at DESC"),
        ),
        Index("node_executions_user_id_started_at_idx", "user_id", text("started_at DESC")),
        Index(
            "node_executions_split_part_status_idx",
            text("(split_part(op, '.', 1))"),
            text("status"),
        ),
    )


class PendingInput(Base):
    """Anything the system needs from the person. One mechanism for all of it."""

    __tablename__ = "pending_inputs"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    node_execution_id: Mapped[str | None] = mapped_column(
        CHAR(21), ForeignKey("node_executions.id", ondelete="SET NULL")
    )

    kind: Mapped[InputKind] = mapped_column(INPUT_KIND, nullable=False)
    blocking: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    prompt: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    value_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    options: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)

    status: Mapped[InputStatus] = mapped_column(
        INPUT_STATUS, nullable=False, server_default=text("'pending'")
    )
    response: Mapped[Any | None] = mapped_column(JSONB)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    answered_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        Index("pending_inputs_user_id_status_idx", "user_id", "status"),
        Index("pending_inputs_message_id_idx", "message_id"),
        Index(
            "pending_inputs_expires_at_idx",
            "expires_at",
            postgresql_where=text("status = 'pending'"),
        ),
    )


class Action(Base):
    """Every side effect the system intends to perform.

    ``requires_input_id`` is NOT NULL, so the database guarantees no
    confirm-requiring write exists without a prompt gating it.
    """

    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    node_execution_id: Mapped[str | None] = mapped_column(
        CHAR(21), ForeignKey("node_executions.id", ondelete="SET NULL")
    )
    requires_input_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("pending_inputs.id"), nullable=False
    )

    op: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    revisions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    dedupe_key: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)

    status: Mapped[ActionStatus] = mapped_column(
        ACTION_STATUS, nullable=False, server_default=text("'draft'")
    )
    external_ref: Mapped[str | None] = mapped_column(String(255))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    attempts: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    job_id: Mapped[str | None] = mapped_column(String(64))

    executed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        # Partial on purpose: dedup applies to in-flight actions only.
        Index(
            "actions_dedupe_key_idx",
            "dedupe_key",
            unique=True,
            postgresql_where=text("status IN ('draft', 'approved', 'running')"),
        ),
        Index("actions_user_id_status_idx", "user_id", "status"),
        Index("actions_message_id_idx", "message_id"),
        Index("actions_requires_input_id_idx", "requires_input_id"),
    )


class ConversationEntity(Base):
    """What this conversation has referred to."""

    __tablename__ = "conversation_entities"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    run_id: Mapped[str | None] = mapped_column(
        CHAR(21), ForeignKey("runs.id", ondelete="SET NULL")
    )

    entity_type: Mapped[str] = mapped_column(String(16), nullable=False)
    entity_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    last_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        # Named explicitly and kept short. The Postgres-default spelling for
        # these three columns is 64 characters; Postgres truncates it to 63
        # without complaint, so the name in the database never matched the name
        # in the metadata, and `alembic revision --autogenerate` refused to run
        # at all rather than emit an identifier it knew was too long.
        UniqueConstraint(
            "conversation_id",
            "entity_type",
            "entity_ref",
            name="conversation_entities_ref_key",
        ),
        Index(
            "conversation_entities_conversation_id_last_seen_at_idx",
            "conversation_id",
            text("last_seen_at DESC"),
        ),
    )


class AuditLog(Base):
    """What changed outside our system. Append-only, never cascades."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(CHAR(21), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(CHAR(21))
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(255))
    payload_hash: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    payload_visible: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    ip: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        Index("audit_log_user_id_created_at_idx", "user_id", text("created_at DESC")),
        Index("audit_log_action_created_at_idx", "action", text("created_at DESC")),
    )


# --------------------------------------------------------------------------- #
# sync_ — the Google mirror. A cache. Drop it and a resync rebuilds everything.
# --------------------------------------------------------------------------- #
#
# Every sync_ table carries `embed_model`: the exact model reference that
# produced that row's vector, e.g. "openai:text-embedding-3-small". Fill it from
# `app.llm.embed_model_id()`, which is the same string EMBED_MODEL uses.
#
# It exists because chat models are swappable and embedding models are not. Two
# models can both produce 1536 dimensions and still disagree completely about
# what each dimension means, so a cosine distance across a mix is not a weak
# signal — it is a made-up one, and it looks exactly like a real one. No error,
# no warning, just the wrong emails. Recording the model is what lets the search
# path call `app.llm.assert_same_embed_model` and fail loudly instead.
#
# An empty string means no vector has been written for that row yet. Changing
# EMBED_MODEL means re-embedding every row: `backend/scripts/reembed.py`.

def _tsv(title_col: str, body_col: str) -> str:
    """Title weighted above body, so a subject match beats a body mention."""
    return (
        f"setweight(to_tsvector('english', coalesce({title_col}, '')), 'A') || "
        f"setweight(to_tsvector('english', coalesce({body_col}, '')), 'B')"
    )


class _MirrorSpine:
    """What every mirrored thing carries, whatever shape it is.

    Deliberately identical across the three tables so one search layer can read
    all of them without knowing which it is looking at. Anything that differs
    between shapes belongs on the shape, not here.
    """

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)

    @declared_attr
    def user_id(cls) -> Mapped[str]:
        return mapped_column(
            CHAR(21), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        )

    #: Where the row came from — gmail, outlook, slack. The *table* is the
    #: shape; this is the source, and the two are independent.
    connector: Mapped[str] = mapped_column(String(32), nullable=False)
    #: The id that source gave it.
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    #: The namespace that makes `source_id` unique inside a connector — a
    #: mailbox, a calendar, a Slack workspace. Part of the natural key.
    scope_key: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=""
    )
    chunk_index: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )

    labels: Mapped[list[str] | None] = mapped_column(ARRAY(Text()))
    url: Mapped[str | None] = mapped_column(Text)
    #: The connector-specific long tail. The rule for what belongs here: a
    #: field you filter on is a column, a field you only display is a key.
    attributes: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    content_hash: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embed_model: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=""
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


def _spine_indexes(table: str) -> tuple[Any, ...]:
    """The indexes every shape needs.

    A separate HNSW per table is the point of the split: a vector search for
    events walks an index holding only events, so top-k means top-k *events*
    rather than top-k of everything followed by a filter that thins it.
    """
    return (
        Index(
            f"{table}_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(f"{table}_tsv_idx", "tsv", postgresql_using="gin"),
        Index(f"{table}_labels_idx", "labels", postgresql_using="gin"),
        Index(
            f"{table}_natural_key_idx",
            "user_id",
            "connector",
            "scope_key",
            "source_id",
            "chunk_index",
            unique=True,
        ),
    )


class SyncMessage(_MirrorSpine, Base):
    """Anything somebody said: Gmail, Outlook mail, Slack, Jira comments.

    Recipients are real columns rather than roles inside a blob, because
    "was this addressed to Sarah" and "was Sarah involved at all" are
    different questions and both have to be indexable. Folding them together
    turns "mail to Sarah" into "mail with Sarah" and quietly returns her sent
    items too.
    """

    __tablename__ = "sync_messages"

    subject: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    from_email: Mapped[str | None] = mapped_column(CITEXT())
    from_name: Mapped[str | None] = mapped_column(Text)
    to_emails: Mapped[list[str]] = mapped_column(
        ARRAY(CITEXT()), nullable=False, server_default=text("'{}'")
    )
    cc_emails: Mapped[list[str]] = mapped_column(
        ARRAY(CITEXT()), nullable=False, server_default=text("'{}'")
    )
    #: Everyone touched by the message, for the "involving X" question. Derived
    #: so it can never disagree with the three columns above.
    participant_emails: Mapped[list[str] | None] = mapped_column(
        ARRAY(CITEXT()),
        Computed(
            "array_remove(array_cat(array_cat(to_emails, cc_emails), "
            "ARRAY[from_email]::CITEXT[]), NULL)",
            persisted=True,
        ),
    )

    sent_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(255))

    is_unread: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    has_attachments: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(_tsv("subject", "body"), persisted=True)
    )

    __table_args__ = _spine_indexes("sync_messages") + (
        Index("sync_messages_time_idx", "user_id", "connector", text("sent_at DESC")),
        Index(
            "sync_messages_from_idx",
            "user_id",
            "from_email",
            postgresql_where=text("from_email IS NOT NULL"),
        ),
        Index("sync_messages_to_idx", "to_emails", postgresql_using="gin"),
        Index("sync_messages_cc_idx", "cc_emails", postgresql_using="gin"),
        Index(
            "sync_messages_participants_idx", "participant_emails", postgresql_using="gin"
        ),
        Index(
            "sync_messages_thread_idx",
            "user_id",
            "thread_id",
            postgresql_where=text("thread_id IS NOT NULL"),
        ),
    )


class SyncEvent(_MirrorSpine, Base):
    """Anything scheduled: Google Calendar, Outlook Calendar.

    `scope_key` here is the calendar id, and it is load-bearing rather than
    cosmetic — Google event ids are unique only *within* a calendar, so two
    calendars can hand out the same id for different meetings.
    """

    __tablename__ = "sync_events"

    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    organizer_email: Mapped[str | None] = mapped_column(CITEXT())
    attendee_emails: Mapped[list[str]] = mapped_column(
        ARRAY(CITEXT()), nullable=False, server_default=text("'{}'")
    )
    #: `[{email, name, response_status, optional}]` — the detail behind the
    #: flat array above, for showing who actually accepted.
    attendees: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    starts_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ends_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    all_day: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    location: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))
    recurring_event_id: Mapped[str | None] = mapped_column(String(255))

    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(_tsv("title", "description"), persisted=True)
    )

    __table_args__ = _spine_indexes("sync_events") + (
        Index("sync_events_time_idx", "user_id", "connector", text("starts_at DESC")),
        Index("sync_events_attendees_idx", "attendee_emails", postgresql_using="gin"),
        Index(
            "sync_events_organizer_idx",
            "user_id",
            "organizer_email",
            postgresql_where=text("organizer_email IS NOT NULL"),
        ),
        # Cancelled events stay — "what did I cancel" is a real question — but
        # almost every read wants them gone, so the common case gets a smaller
        # index of its own.
        Index(
            "sync_events_live_idx",
            "user_id",
            "starts_at",
            postgresql_where=text("status IS DISTINCT FROM 'cancelled'"),
        ),
    )


class SyncFile(_MirrorSpine, Base):
    """Anything stored: Drive, OneDrive, Dropbox, Notion."""

    __tablename__ = "sync_files"

    name: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    owner_email: Mapped[str | None] = mapped_column(CITEXT())
    shared_with_emails: Mapped[list[str]] = mapped_column(
        ARRAY(CITEXT()), nullable=False, server_default=text("'{}'")
    )

    mime_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    #: Drive's *modified* date. The created date is not mirrored, which is why
    #: "files from last month" honestly means modified last month.
    modified_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    folder_id: Mapped[str | None] = mapped_column(String(255))
    folder_path: Mapped[str | None] = mapped_column(Text)
    is_shared: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(_tsv("name", "content"), persisted=True)
    )

    __table_args__ = _spine_indexes("sync_files") + (
        Index("sync_files_time_idx", "user_id", "connector", text("modified_at DESC")),
        Index(
            "sync_files_owner_idx",
            "user_id",
            "owner_email",
            postgresql_where=text("owner_email IS NOT NULL"),
        ),
        Index("sync_files_shared_idx", "shared_with_emails", postgresql_using="gin"),
        Index(
            "sync_files_folder_idx",
            "user_id",
            "folder_id",
            postgresql_where=text("folder_id IS NOT NULL"),
        ),
        # "PDFs from last month" is a mime + time question, and a common one.
        Index(
            "sync_files_mime_idx",
            "user_id",
            "mime_type",
            text("modified_at DESC"),
            postgresql_where=text("mime_type IS NOT NULL"),
        ),
    )


class SyncState(Base):
    """Where each sync got to. Cursor advances only after the upsert commits."""

    __tablename__ = "sync_state"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        CHAR(21), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    service: Mapped[SyncService] = mapped_column(SYNC_SERVICE, nullable=False)

    cursor: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    backfill_complete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    backfill_cursor: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    last_synced_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    items_indexed: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    consecutive_failures: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    circuit_open_until: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        UniqueConstraint("user_id", "service", name="sync_state_user_id_service_key"),
    )


# --------------------------------------------------------------------------- #
# job_ — queue bookkeeping
# --------------------------------------------------------------------------- #


class JobFailedTask(Base):
    """Jobs that exhausted their retries."""

    __tablename__ = "job_failed_tasks"

    id: Mapped[str] = mapped_column(CHAR(21), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(CHAR(21))
    task_name: Mapped[str] = mapped_column(String(128), nullable=False)
    queue: Mapped[str] = mapped_column(String(32), nullable=False)
    task_input: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_class: Mapped[str] = mapped_column(String(32), nullable=False)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    traceback: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'open'")
    )
    first_failed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    last_failed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )

    __table_args__ = (
        Index("job_failed_tasks_status_last_failed_at_idx", "status", "last_failed_at"),
        Index("job_failed_tasks_user_id_status_idx", "user_id", "status"),
    )


__all__ = [
    "EMBEDDING_DIM",
    "MessageRole",
    "RunStatus",
    "NodeStatus",
    "InputKind",
    "InputStatus",
    "ActionStatus",
    "SyncService",
    "User",
    "OAuthToken",
    "Conversation",
    "Message",
    "Run",
    "NodeExecution",
    "PendingInput",
    "Action",
    "ConversationEntity",
    "AuditLog",
    "SyncEvent",
    "SyncFile",
    "SyncMessage",
    "ITEM_KINDS",
    "SyncState",
    "JobFailedTask",
]
