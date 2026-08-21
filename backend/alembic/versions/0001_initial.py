"""initial schema — 15 tables

Everything is raw SQL on purpose. Generated columns, a CITEXT[] built by a
custom function, HNSW indexes and a partial unique index are all clearer written
out than assembled through op.create_table, and this way the migration is
readable next to docs/schema.md line for line.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-20
"""

from __future__ import annotations

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ----------------------------------------------------------------- #
    # Extensions
    # ----------------------------------------------------------------- #
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    # ----------------------------------------------------------------- #
    # Functions
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE OR REPLACE FUNCTION attendee_email_list(a JSONB) RETURNS CITEXT[] AS $$
            SELECT array_agg(lower(x ->> 'email'))::CITEXT[]
            FROM jsonb_array_elements(coalesce(a, '[]'::jsonb)) x;
        $$ LANGUAGE sql IMMUTABLE;
        """
    )

    # ----------------------------------------------------------------- #
    # Enums
    # ----------------------------------------------------------------- #
    op.execute("CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system')")
    op.execute(
        """
        CREATE TYPE run_status AS ENUM (
            'running', 'awaiting_input', 'complete', 'failed', 'timeout', 'cancelled'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE node_status AS ENUM (
            'pending', 'running', 'succeeded', 'failed', 'skipped', 'timeout', 'cancelled'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE input_kind AS ENUM (
            'confirm', 'choice', 'multi_choice', 'text', 'form', 'date_range'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE input_status AS ENUM (
            'pending', 'answered', 'cancelled', 'expired', 'superseded'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE action_status AS ENUM (
            'draft', 'approved', 'running', 'done', 'failed', 'cancelled', 'expired'
        )
        """
    )
    op.execute("CREATE TYPE sync_service AS ENUM ('gmail', 'gcal', 'gdrive')")

    # ----------------------------------------------------------------- #
    # users
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE users (
            id               CHAR(21) PRIMARY KEY,
            email            CITEXT NOT NULL UNIQUE,
            display_name     TEXT,
            timezone         VARCHAR(64) NOT NULL DEFAULT 'UTC',
            work_week_start  SMALLINT NOT NULL DEFAULT 1,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # ----------------------------------------------------------------- #
    # oauth_tokens
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE oauth_tokens (
            id                   CHAR(21) PRIMARY KEY,
            user_id              CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider             VARCHAR(32) NOT NULL DEFAULT 'google',
            provider_account_id  VARCHAR(128),

            access_token_enc     BYTEA NOT NULL,
            refresh_token_enc    BYTEA,
            key_version          SMALLINT NOT NULL DEFAULT 1,
            scopes               TEXT[] NOT NULL,
            expires_at           TIMESTAMPTZ,
            revoked_at           TIMESTAMPTZ,
            refresh_failures     SMALLINT NOT NULL DEFAULT 0,

            created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT oauth_tokens_user_id_provider_key UNIQUE (user_id, provider)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX oauth_tokens_expires_at_idx ON oauth_tokens (expires_at)
            WHERE revoked_at IS NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX oauth_tokens_provider_provider_account_id_idx
            ON oauth_tokens (provider, provider_account_id)
            WHERE provider_account_id IS NOT NULL
        """
    )

    # ----------------------------------------------------------------- #
    # conversations
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE conversations (
            id               CHAR(21) PRIMARY KEY,
            user_id          CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title            TEXT,
            archived_at      TIMESTAMPTZ,
            last_message_at  TIMESTAMPTZ NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX conversations_user_id_last_message_at_idx
            ON conversations (user_id, last_message_at DESC)
            WHERE archived_at IS NULL
        """
    )

    # ----------------------------------------------------------------- #
    # messages — run_id's FK is added after runs exists (circular)
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE messages (
            id               CHAR(21) PRIMARY KEY,
            conversation_id  CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id          CHAR(21) NOT NULL,
            run_id           CHAR(21),
            seq              INTEGER NOT NULL,

            role             message_role NOT NULL,
            content          JSONB NOT NULL,
            hidden           BOOLEAN NOT NULL DEFAULT false,

            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT messages_conversation_id_seq_key UNIQUE (conversation_id, seq)
        )
        """
    )
    op.execute("CREATE INDEX messages_conversation_id_seq_idx ON messages (conversation_id, seq)")
    op.execute("CREATE INDEX messages_run_id_idx ON messages (run_id) WHERE run_id IS NOT NULL")

    # ----------------------------------------------------------------- #
    # runs
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE runs (
            id                  CHAR(21) PRIMARY KEY,
            conversation_id     CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id             CHAR(21) NOT NULL,
            trigger_message_id  CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,

            intent              JSONB,
            planner_tier        SMALLINT,

            status              run_status NOT NULL DEFAULT 'running',
            error               JSONB,
            token_usage         JSONB,

            started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at         TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX runs_conversation_id_started_at_idx ON runs (conversation_id, started_at DESC)"
    )
    op.execute("CREATE INDEX runs_user_id_started_at_idx ON runs (user_id, started_at DESC)")
    op.execute(
        """
        CREATE INDEX runs_status_idx ON runs (status)
            WHERE status IN ('running', 'awaiting_input')
        """
    )

    # The other half of the circular FK. Insert order is user message, run,
    # assistant message, so no row ever needs a value that does not exist.
    op.execute(
        """
        ALTER TABLE messages
            ADD CONSTRAINT messages_run_id_fkey
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
        """
    )

    # ----------------------------------------------------------------- #
    # node_executions
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE node_executions (
            id               CHAR(21) PRIMARY KEY,
            run_id           CHAR(21) NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            conversation_id  CHAR(21) NOT NULL,
            user_id          CHAR(21) NOT NULL,
            message_id       CHAR(21) REFERENCES messages(id) ON DELETE SET NULL,

            seq              SMALLINT NOT NULL,
            node_id          VARCHAR(64) NOT NULL,
            op               VARCHAR(64) NOT NULL,
            round            SMALLINT NOT NULL DEFAULT 0,
            args             JSONB NOT NULL,
            depends_on       TEXT[] NOT NULL DEFAULT '{}',

            status           node_status NOT NULL DEFAULT 'pending',
            result           JSONB,
            outcome          JSONB,
            retries          JSONB,

            started_at       TIMESTAMPTZ,
            finished_at      TIMESTAMPTZ,
            CONSTRAINT node_executions_run_id_node_id_round_key
                UNIQUE (run_id, node_id, round)
        )
        """
    )
    op.execute("CREATE INDEX node_executions_run_id_seq_idx ON node_executions (run_id, seq)")
    op.execute(
        """
        CREATE INDEX node_executions_message_id_seq_idx ON node_executions (message_id, seq)
            WHERE message_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX node_executions_conversation_id_started_at_idx
            ON node_executions (conversation_id, started_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX node_executions_user_id_started_at_idx
            ON node_executions (user_id, started_at DESC)
        """
    )
    # "which services failed this run" without a scan
    op.execute(
        """
        CREATE INDEX node_executions_split_part_status_idx
            ON node_executions ((split_part(op, '.', 1)), status)
        """
    )

    # ----------------------------------------------------------------- #
    # pending_inputs
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE pending_inputs (
            id                CHAR(21) PRIMARY KEY,
            run_id            CHAR(21) NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            message_id        CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            user_id           CHAR(21) NOT NULL,
            node_execution_id CHAR(21) REFERENCES node_executions(id) ON DELETE SET NULL,

            kind              input_kind NOT NULL,
            blocking          BOOLEAN NOT NULL DEFAULT false,

            prompt            JSONB NOT NULL,
            value_schema      JSONB NOT NULL,
            options           JSONB,

            status            input_status NOT NULL DEFAULT 'pending',
            response          JSONB,
            expires_at        TIMESTAMPTZ NOT NULL,
            answered_at       TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX pending_inputs_user_id_status_idx ON pending_inputs (user_id, status)")
    op.execute("CREATE INDEX pending_inputs_message_id_idx ON pending_inputs (message_id)")
    op.execute(
        """
        CREATE INDEX pending_inputs_expires_at_idx ON pending_inputs (expires_at)
            WHERE status = 'pending'
        """
    )

    # ----------------------------------------------------------------- #
    # actions
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE actions (
            id                CHAR(21) PRIMARY KEY,
            message_id        CHAR(21) NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            user_id           CHAR(21) NOT NULL,
            node_execution_id CHAR(21) REFERENCES node_executions(id) ON DELETE SET NULL,
            requires_input_id CHAR(21) NOT NULL REFERENCES pending_inputs(id),

            op                VARCHAR(64) NOT NULL,
            payload           JSONB NOT NULL,
            revisions         JSONB NOT NULL DEFAULT '[]',
            dedupe_key        UUID NOT NULL,

            status            action_status NOT NULL DEFAULT 'draft',
            external_ref      VARCHAR(255),
            result            JSONB,
            error             JSONB,
            attempts          SMALLINT NOT NULL DEFAULT 0,
            job_id            VARCHAR(64),

            executed_at       TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # Partial on purpose: dedup applies to in-flight actions only. Once
    # something is done, cancelled, expired or failed, an identical one is a
    # legitimate new request.
    op.execute(
        """
        CREATE UNIQUE INDEX actions_dedupe_key_idx ON actions (dedupe_key)
            WHERE status IN ('draft', 'approved', 'running')
        """
    )
    op.execute("CREATE INDEX actions_user_id_status_idx ON actions (user_id, status)")
    op.execute("CREATE INDEX actions_message_id_idx ON actions (message_id)")
    op.execute("CREATE INDEX actions_requires_input_id_idx ON actions (requires_input_id)")

    # ----------------------------------------------------------------- #
    # conversation_entities
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE conversation_entities (
            id               CHAR(21) PRIMARY KEY,
            conversation_id  CHAR(21) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id          CHAR(21) NOT NULL,
            run_id           CHAR(21) REFERENCES runs(id) ON DELETE SET NULL,

            entity_type      VARCHAR(16) NOT NULL,
            entity_ref       VARCHAR(255) NOT NULL,
            label            TEXT NOT NULL,
            meta             JSONB,

            last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT conversation_entities_ref_key
                UNIQUE (conversation_id, entity_type, entity_ref)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX conversation_entities_conversation_id_last_seen_at_idx
            ON conversation_entities (conversation_id, last_seen_at DESC)
        """
    )

    # ----------------------------------------------------------------- #
    # audit_log — BIGSERIAL: append-only, highest volume
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE audit_log (
            id               BIGSERIAL PRIMARY KEY,
            user_id          CHAR(21) NOT NULL,
            conversation_id  CHAR(21),
            actor            VARCHAR(16) NOT NULL,
            action           VARCHAR(64) NOT NULL,
            resource_id      VARCHAR(255),
            payload_hash     UUID,
            payload_visible  JSONB,
            status           VARCHAR(16) NOT NULL,
            error            JSONB,
            ip               INET,
            user_agent       TEXT,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX audit_log_user_id_created_at_idx ON audit_log (user_id, created_at DESC)")
    op.execute("CREATE INDEX audit_log_action_created_at_idx ON audit_log (action, created_at DESC)")

    # ----------------------------------------------------------------- #
    # sync_gmail
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE sync_gmail (
            id               CHAR(21) PRIMARY KEY,
            user_id          CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            message_id       VARCHAR(255) NOT NULL,
            thread_id        VARCHAR(255),
            chunk_index      SMALLINT NOT NULL DEFAULT 0,

            subject          TEXT,
            from_email       CITEXT,
            from_name        TEXT,
            to_emails        CITEXT[],
            body_clean       TEXT NOT NULL,
            content_hash     UUID NOT NULL,

            embedding        VECTOR(1536),
            -- which model produced `embedding`. '' means nothing has yet.
            embed_model      VARCHAR(64) NOT NULL DEFAULT '',
            labels           TEXT[],
            has_attachments  BOOLEAN NOT NULL DEFAULT false,
            received_at      TIMESTAMPTZ NOT NULL,

            tsv              TSVECTOR GENERATED ALWAYS AS (
                                 setweight(to_tsvector('english', coalesce(subject, '')),    'A') ||
                                 setweight(to_tsvector('english', coalesce(body_clean, '')), 'B')
                             ) STORED,

            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT sync_gmail_user_id_message_id_chunk_index_key
                UNIQUE (user_id, message_id, chunk_index)
        )
        """
    )
    # HNSW rather than the brief's ivfflat: it holds up better under a hard
    # user_id prefilter, which is every query we run.
    op.execute(
        """
        CREATE INDEX sync_gmail_embedding_idx ON sync_gmail
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
        """
    )
    # ivfflat alternative:
    # CREATE INDEX sync_gmail_embedding_idx ON sync_gmail
    #     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
    # (user_id, embed_model) is what makes the model check cheap: search already
    # prefilters on user_id, so this turns "were these rows embedded by the model
    # asking the question?" into an index-only scan.
    op.execute("CREATE INDEX sync_gmail_user_id_embed_model_idx ON sync_gmail (user_id, embed_model)")
    op.execute("CREATE INDEX sync_gmail_user_id_received_at_idx ON sync_gmail (user_id, received_at DESC)")
    op.execute("CREATE INDEX sync_gmail_user_id_from_email_idx ON sync_gmail (user_id, from_email)")
    op.execute("CREATE INDEX sync_gmail_user_id_thread_id_idx ON sync_gmail (user_id, thread_id)")
    op.execute("CREATE INDEX sync_gmail_tsv_idx ON sync_gmail USING GIN (tsv)")
    op.execute("CREATE INDEX sync_gmail_labels_idx ON sync_gmail USING GIN (labels)")

    # ----------------------------------------------------------------- #
    # sync_gcal
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE sync_gcal (
            id                  CHAR(21) PRIMARY KEY,
            user_id             CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            event_id            VARCHAR(255) NOT NULL,
            calendar_id         VARCHAR(255) NOT NULL DEFAULT 'primary',
            recurring_event_id  VARCHAR(255),

            title               TEXT,
            description         TEXT,
            location            TEXT,
            organizer_email     CITEXT,
            attendees           JSONB,
            attendee_emails     CITEXT[] GENERATED ALWAYS AS
                                    (attendee_email_list(attendees)) STORED,
            content_hash        UUID NOT NULL,
            embedding           VECTOR(1536),
            embed_model         VARCHAR(64) NOT NULL DEFAULT '',

            starts_at           TIMESTAMPTZ NOT NULL,
            ends_at             TIMESTAMPTZ,
            all_day             BOOLEAN NOT NULL DEFAULT false,
            event_timezone      VARCHAR(64),
            status              VARCHAR(16),
            etag                VARCHAR(128),

            tsv                 TSVECTOR GENERATED ALWAYS AS (
                                    setweight(to_tsvector('english', coalesce(title, '')),       'A') ||
                                    setweight(to_tsvector('english', coalesce(description, '')), 'B')
                                ) STORED,

            updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT sync_gcal_user_id_calendar_id_event_id_key
                UNIQUE (user_id, calendar_id, event_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX sync_gcal_embedding_idx ON sync_gcal
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
        """
    )
    # ivfflat alternative:
    # CREATE INDEX sync_gcal_embedding_idx ON sync_gcal
    #     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
    op.execute("CREATE INDEX sync_gcal_user_id_embed_model_idx ON sync_gcal (user_id, embed_model)")
    op.execute("CREATE INDEX sync_gcal_user_id_starts_at_idx ON sync_gcal (user_id, starts_at)")
    # this is what makes "next week where john@company.com is invited" an index hit
    op.execute("CREATE INDEX sync_gcal_attendee_emails_idx ON sync_gcal USING GIN (attendee_emails)")
    op.execute("CREATE INDEX sync_gcal_tsv_idx ON sync_gcal USING GIN (tsv)")

    # ----------------------------------------------------------------- #
    # sync_gdrive
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE sync_gdrive (
            id               CHAR(21) PRIMARY KEY,
            user_id          CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            file_id          VARCHAR(255) NOT NULL,
            chunk_index      SMALLINT NOT NULL DEFAULT 0,

            name             TEXT NOT NULL,
            mime_type        VARCHAR(128),
            owner_email      CITEXT,
            is_shared        BOOLEAN NOT NULL DEFAULT false,
            web_view_link    TEXT,
            folder_path      TEXT,
            size_bytes       BIGINT,

            content_excerpt  TEXT,
            content_hash     UUID NOT NULL,
            embedding        VECTOR(1536),
            embed_model      VARCHAR(64) NOT NULL DEFAULT '',
            modified_at      TIMESTAMPTZ,

            tsv              TSVECTOR GENERATED ALWAYS AS (
                                 setweight(to_tsvector('english', coalesce(name, '')),            'A') ||
                                 setweight(to_tsvector('english', coalesce(content_excerpt, '')), 'B')
                             ) STORED,

            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT sync_gdrive_user_id_file_id_chunk_index_key
                UNIQUE (user_id, file_id, chunk_index)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX sync_gdrive_embedding_idx ON sync_gdrive
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
        """
    )
    # ivfflat alternative:
    # CREATE INDEX sync_gdrive_embedding_idx ON sync_gdrive
    #     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
    op.execute("CREATE INDEX sync_gdrive_user_id_embed_model_idx ON sync_gdrive (user_id, embed_model)")
    op.execute("CREATE INDEX sync_gdrive_user_id_modified_at_idx ON sync_gdrive (user_id, modified_at DESC)")
    op.execute("CREATE INDEX sync_gdrive_user_id_mime_type_idx ON sync_gdrive (user_id, mime_type)")
    op.execute("CREATE INDEX sync_gdrive_tsv_idx ON sync_gdrive USING GIN (tsv)")

    # ----------------------------------------------------------------- #
    # sync_state
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE sync_state (
            id                    CHAR(21) PRIMARY KEY,
            user_id               CHAR(21) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            service               sync_service NOT NULL,

            cursor                JSONB,
            backfill_complete     BOOLEAN NOT NULL DEFAULT false,
            backfill_cursor       JSONB,

            last_synced_at        TIMESTAMPTZ,
            last_success_at       TIMESTAMPTZ,
            last_error            JSONB,
            items_indexed         INT NOT NULL DEFAULT 0,

            consecutive_failures  SMALLINT NOT NULL DEFAULT 0,
            circuit_open_until    TIMESTAMPTZ,

            updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT sync_state_user_id_service_key UNIQUE (user_id, service)
        )
        """
    )

    # ----------------------------------------------------------------- #
    # job_failed_tasks
    # ----------------------------------------------------------------- #
    op.execute(
        """
        CREATE TABLE job_failed_tasks (
            id              CHAR(21) PRIMARY KEY,
            user_id         CHAR(21),
            task_name       VARCHAR(128) NOT NULL,
            queue           VARCHAR(32) NOT NULL,
            task_input      JSONB,
            error_class     VARCHAR(32) NOT NULL,
            error           JSONB,
            traceback       TEXT,
            attempts        SMALLINT NOT NULL,
            celery_task_id  VARCHAR(64),
            status          VARCHAR(16) NOT NULL DEFAULT 'open',
            first_failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_failed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX job_failed_tasks_status_last_failed_at_idx"
        " ON job_failed_tasks (status, last_failed_at)"
    )
    op.execute("CREATE INDEX job_failed_tasks_user_id_status_idx ON job_failed_tasks (user_id, status)")


def downgrade() -> None:
    # Drop the circular FK first so messages and runs can go in either order.
    op.execute("ALTER TABLE IF EXISTS messages DROP CONSTRAINT IF EXISTS messages_run_id_fkey")

    for table in (
        "job_failed_tasks",
        "sync_state",
        "sync_gdrive",
        "sync_gcal",
        "sync_gmail",
        "audit_log",
        "conversation_entities",
        "actions",
        "pending_inputs",
        "node_executions",
        "runs",
        "messages",
        "conversations",
        "oauth_tokens",
        "users",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    for enum_name in (
        "sync_service",
        "action_status",
        "input_status",
        "input_kind",
        "node_status",
        "run_status",
        "message_role",
    ):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")

    op.execute("DROP FUNCTION IF EXISTS attendee_email_list(JSONB)")
    # Extensions are left in place — other schemas in the database may use them.
