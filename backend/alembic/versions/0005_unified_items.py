"""One mirror table for every connector.

``sync_gmail``, ``sync_gcal`` and ``sync_gdrive`` were three tables with the
same skeleton: an id from the source, some text, some people, a time, a vector
and a tsvector. The names differed — ``subject``/``title``/``name``,
``received_at``/``starts_at``/``modified_at`` — but the shape did not, and
every new connector under that design meant another table, another set of
indexes, and another arm in the search fan-out.

They collapse into ``sync_items``, discriminated by ``kind``:

* ``message`` — Gmail, Outlook mail, Slack, Teams, Jira comments
* ``event``   — Google Calendar, Outlook Calendar
* ``file``    — Drive, OneDrive, Dropbox, Notion
* ``task``    — Jira issues, Linear, Asana, GitHub issues

A Jira issue is a ``task``; the comments underneath it are ``message`` rows
whose ``container_id`` is the issue key. That is what lets one search cover
"what did anyone say about this" across mail, chat and tickets at once.

Two keys, because they answer different questions. ``scope_id`` is the
namespace that makes ``external_id`` unique inside a connector — a calendar id,
a Slack workspace, a Jira project — and is part of the natural key.
``container_id`` is the grouping a person would recognise — a mail thread, a
folder, a channel, the issue a comment hangs off — and is free to change.
Google event ids are only unique per calendar, which is why the two cannot be
the same column.

Connector-specific facets (mime type, attachment flag, location, priority)
live in ``attributes``. Anything that generalises across connectors is a real
column, because that is what indexes can be built on.
"""

from __future__ import annotations

from alembic import op

revision = "0005_unified_items"
down_revision = "0004_account_email"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A generalised `attendee_email_list`. Skipping blank addresses matters
    # more here: participants now include senders, owners and assignees, and a
    # NULL in the array would make `&&` behave in ways nobody expects.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION participant_email_list(p JSONB) RETURNS CITEXT[] AS $$
            SELECT array_agg(DISTINCT lower(x ->> 'email'))::CITEXT[]
            FROM jsonb_array_elements(coalesce(p, '[]'::jsonb)) x
            WHERE coalesce(x ->> 'email', '') <> '';
        $$ LANGUAGE sql IMMUTABLE;
        """
    )

    # "Was X a recipient" is a different question from "was X involved", and
    # the difference is the whole reason participants carry a role. Folding
    # them together turns "mail to Sarah" into "mail with Sarah", which
    # quietly returns her sent items too.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION role_emails(p JSONB, r TEXT) RETURNS CITEXT[] AS $$
            SELECT array_agg(DISTINCT lower(x ->> 'email'))::CITEXT[]
            FROM jsonb_array_elements(coalesce(p, '[]'::jsonb)) x
            WHERE x ->> 'role' = r AND coalesce(x ->> 'email', '') <> '';
        $$ LANGUAGE sql IMMUTABLE;
        """
    )

    op.execute(
        """
        CREATE TABLE sync_items (
            id                  CHAR(21) PRIMARY KEY,
            user_id             CHAR(21) NOT NULL
                                    REFERENCES users(id) ON DELETE CASCADE,

            -- where it came from
            connector           VARCHAR(32)  NOT NULL,
            kind                VARCHAR(16)  NOT NULL,
            external_id         VARCHAR(255) NOT NULL,
            scope_id            VARCHAR(255) NOT NULL DEFAULT '',
            chunk_index         SMALLINT     NOT NULL DEFAULT 0,

            -- what it says
            title               TEXT,
            body                TEXT NOT NULL DEFAULT '',

            -- who is involved: [{email, name, role, status}]
            participants        JSONB NOT NULL DEFAULT '[]'::jsonb,
            participant_emails  CITEXT[] GENERATED ALWAYS AS
                                    (participant_email_list(participants)) STORED,
            -- the one who sent / owns / organises / reported it
            author_email        CITEXT,

            -- when
            occurred_at         TIMESTAMPTZ NOT NULL,
            ends_at             TIMESTAMPTZ,
            all_day             BOOLEAN NOT NULL DEFAULT false,

            -- how it groups
            container_id        VARCHAR(255),

            -- facets
            labels              TEXT[],
            status              VARCHAR(32),
            url                 TEXT,
            attributes          JSONB NOT NULL DEFAULT '{}'::jsonb,

            -- retrieval
            content_hash        UUID NOT NULL,
            embedding           VECTOR(1536),
            embed_model         VARCHAR(64) NOT NULL DEFAULT '',
            tsv                 TSVECTOR GENERATED ALWAYS AS (
                                    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                                    setweight(to_tsvector('english', coalesce(body, '')), 'B')
                                ) STORED,

            updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT sync_items_kind_ck
                CHECK (kind IN ('message', 'event', 'file', 'task'))
        )
        """
    )

    # The natural key. `scope_id` defaults to '' so a connector that does not
    # namespace its ids does not have to think about it.
    op.execute(
        """
        CREATE UNIQUE INDEX sync_items_natural_key_idx
            ON sync_items (user_id, connector, scope_id, external_id, chunk_index)
        """
    )

    # One ANN index for everything. Cross-connector search was three index
    # scans and a merge; now it is one.
    op.execute(
        """
        CREATE INDEX sync_items_embedding_idx ON sync_items
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute("CREATE INDEX sync_items_tsv_idx ON sync_items USING GIN (tsv)")
    op.execute(
        "CREATE INDEX sync_items_participants_idx "
        "ON sync_items USING GIN (participant_emails)"
    )
    op.execute("CREATE INDEX sync_items_labels_idx ON sync_items USING GIN (labels)")

    # The three ways a window gets filtered: by shape, by connector, or both.
    op.execute(
        "CREATE INDEX sync_items_kind_time_idx "
        "ON sync_items (user_id, kind, occurred_at DESC)"
    )
    op.execute(
        "CREATE INDEX sync_items_connector_time_idx "
        "ON sync_items (user_id, connector, occurred_at DESC)"
    )
    op.execute(
        "CREATE INDEX sync_items_container_idx "
        "ON sync_items (user_id, container_id) WHERE container_id IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX sync_items_author_idx "
        "ON sync_items (user_id, author_email) WHERE author_email IS NOT NULL"
    )

    _migrate_gmail()
    _migrate_gcal()
    _migrate_gdrive()

    op.execute("DROP TABLE IF EXISTS sync_gmail")
    op.execute("DROP TABLE IF EXISTS sync_gcal")
    op.execute("DROP TABLE IF EXISTS sync_gdrive")
    op.execute("DROP FUNCTION IF EXISTS attendee_email_list(JSONB)")


def _migrate_gmail() -> None:
    """Mail: sender and recipients become participants with roles."""
    op.execute(
        """
        INSERT INTO sync_items (
            id, user_id, connector, kind, external_id, scope_id, chunk_index,
            title, body, participants, author_email,
            occurred_at, container_id, labels, attributes,
            content_hash, embedding, embed_model, updated_at
        )
        SELECT
            g.id, g.user_id, 'gmail', 'message', g.message_id, '', g.chunk_index,
            g.subject,
            coalesce(g.body_clean, ''),
            CASE WHEN g.from_email IS NULL THEN '[]'::jsonb ELSE
                jsonb_build_array(jsonb_build_object(
                    'email', lower(g.from_email), 'name', g.from_name, 'role', 'from'))
            END
            || coalesce((
                SELECT jsonb_agg(jsonb_build_object('email', lower(e), 'role', 'to'))
                FROM unnest(coalesce(g.to_emails, ARRAY[]::citext[])) AS e
            ), '[]'::jsonb),
            g.from_email,
            g.received_at,
            g.thread_id,
            g.labels,
            jsonb_strip_nulls(jsonb_build_object(
                'has_attachments', g.has_attachments,
                'from_name', g.from_name
            )),
            g.content_hash, g.embedding, g.embed_model, g.updated_at
        FROM sync_gmail g
        """
    )


def _migrate_gcal() -> None:
    """Events: the organiser joins the attendee list under its own role."""
    op.execute(
        """
        INSERT INTO sync_items (
            id, user_id, connector, kind, external_id, scope_id, chunk_index,
            title, body, participants, author_email,
            occurred_at, ends_at, all_day, container_id, status, attributes,
            content_hash, embedding, embed_model, updated_at
        )
        SELECT
            c.id, c.user_id, 'gcal', 'event', c.event_id, c.calendar_id, 0,
            c.title,
            coalesce(c.description, ''),
            CASE WHEN c.organizer_email IS NULL THEN '[]'::jsonb ELSE
                jsonb_build_array(jsonb_build_object(
                    'email', lower(c.organizer_email), 'role', 'organizer'))
            END
            || coalesce((
                SELECT jsonb_agg(jsonb_build_object(
                    'email', lower(x ->> 'email'),
                    'name', x ->> 'name',
                    'role', 'attendee',
                    'status', x ->> 'response_status'))
                FROM jsonb_array_elements(coalesce(c.attendees, '[]'::jsonb)) AS x
                WHERE coalesce(x ->> 'email', '') <> ''
            ), '[]'::jsonb),
            c.organizer_email,
            c.starts_at, c.ends_at, c.all_day, c.calendar_id, c.status,
            jsonb_strip_nulls(jsonb_build_object(
                'location', c.location,
                'event_timezone', c.event_timezone,
                'etag', c.etag,
                'recurring_event_id', c.recurring_event_id
            )),
            c.content_hash, c.embedding, c.embed_model, c.updated_at
        FROM sync_gcal c
        """
    )


def _migrate_gdrive() -> None:
    """Files: the owner is the sole participant until sharing is synced."""
    op.execute(
        """
        INSERT INTO sync_items (
            id, user_id, connector, kind, external_id, scope_id, chunk_index,
            title, body, participants, author_email,
            occurred_at, container_id, url, attributes,
            content_hash, embedding, embed_model, updated_at
        )
        SELECT
            d.id, d.user_id, 'gdrive', 'file', d.file_id, '', d.chunk_index,
            d.name,
            coalesce(d.content_excerpt, ''),
            CASE WHEN d.owner_email IS NULL THEN '[]'::jsonb ELSE
                jsonb_build_array(jsonb_build_object(
                    'email', lower(d.owner_email), 'role', 'owner'))
            END,
            d.owner_email,
            -- occurred_at is NOT NULL; a file with no modified time falls back
            -- to when we last wrote the row, which is the best we know.
            coalesce(d.modified_at, d.updated_at),
            d.folder_path,
            d.web_view_link,
            jsonb_strip_nulls(jsonb_build_object(
                'mime_type', d.mime_type,
                'size_bytes', d.size_bytes,
                'is_shared', d.is_shared
            )),
            d.content_hash, d.embedding, d.embed_model, d.updated_at
        FROM sync_gdrive d
        """
    )


def downgrade() -> None:
    # The three tables were a cache with no source of truth behind them, so a
    # downgrade drops the unified table and lets a resync rebuild. Recreating
    # three schemas here to hold data that a resync replaces anyway would be
    # a lot of SQL nobody ever runs.
    raise NotImplementedError(
        "Downgrading past the unified mirror is not supported. "
        "Drop sync_items and resync from the connectors instead."
    )
