"""Three mirror tables, one per shape.

``sync_items`` put messages, events and files in one table discriminated by
``kind``. That collapsed three near-identical tables into one, which was right,
but it charged for it in two places.

**Columns could not be honest.** A partition or a discriminator forces one
column list on every shape, so ``ends_at`` sat NULL on every email and the time
a thing happened was called ``occurred_at`` whether it was sent, scheduled or
modified. Worse, "who was this addressed to" had nowhere to live: recipients
went into a JSONB ``participants`` blob and the only way to ask *"mail to
Sarah"* was ``role_emails(participants, 'to')`` — a function evaluated per row,
which no index can help. "Mail involving Sarah" was fast; "mail **to** Sarah"
was a scan. Those are different questions and the schema indexed one of them.

**One ANN index served every shape.** A vector search for calendar events
walked an index full of mail and files, took the top-k and filtered afterwards.
That is ANN-then-filter: as mail volume grows it dominates the neighbourhood
and event searches quietly under-return.

So the shapes get their own tables:

* ``sync_messages`` — Gmail, Outlook mail, Slack, Teams, Jira comments
* ``sync_events``   — Google Calendar, Outlook Calendar
* ``sync_files``    — Drive, OneDrive, Dropbox, Notion

Each keeps the same spine — ids, text, retrieval columns — so the search layer
stays generic over them, and each adds the columns its shape actually has,
under the name that shape actually uses. ``to_emails`` and ``attendee_emails``
are real ``CITEXT[]`` columns with real GIN indexes.

**Where a new connector goes.** The tables are shapes, not sources, so a
connector picks the shape it fits and needs no DDL: Slack is a message,
OneDrive is a file, Outlook Calendar is an event. The rule for deciding what
becomes a column:

    A field you filter on is a column. A field you only display is an
    ``attributes`` key.

Burying a queryable dimension in JSONB is exactly how ``to`` ended up
unindexed. By that rule Jira issues want a fourth table when they arrive —
``assignee`` and ``due_at`` are things people filter on — while Jira comments
are simply messages.
"""

from __future__ import annotations

from alembic import op

revision = "0006_split_by_shape"
down_revision = "0005_unified_items"
branch_labels = None
depends_on = None


#: Columns every shape carries. Identity, text, and everything retrieval needs.
#: Kept character-for-character identical across the three so one search layer
#: can read all of them without knowing which it is looking at.
SPINE = """
            id                  CHAR(21) PRIMARY KEY,
            user_id             CHAR(21) NOT NULL
                                    REFERENCES users(id) ON DELETE CASCADE,

            -- where it came from. `connector` is the source; the table is the
            -- shape. `source_id` is the id that source gave it.
            connector           VARCHAR(32)  NOT NULL,
            source_id           VARCHAR(255) NOT NULL,
            chunk_index         SMALLINT     NOT NULL DEFAULT 0,

            -- facets every shape has
            labels              TEXT[],
            url                 TEXT,
            attributes          JSONB NOT NULL DEFAULT '{}'::jsonb,

            -- retrieval
            content_hash        UUID NOT NULL,
            embedding           VECTOR(1536),
            embed_model         VARCHAR(64) NOT NULL DEFAULT '',
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
"""


def _tsv(title_column: str, body_column: str) -> str:
    """Title weighted above body, so a subject match beats a body mention.

    This is what makes "find the mail with subject X" behave differently from
    "find mail about X" — without it a long body discussing the words would
    outrank the message actually titled with them.
    """
    return f"""
            tsv                 TSVECTOR GENERATED ALWAYS AS (
                                    setweight(to_tsvector('english', coalesce({title_column}, '')), 'A') ||
                                    setweight(to_tsvector('english', coalesce({body_column}, '')), 'B')
                                ) STORED
    """


def _retrieval_indexes(table: str, tsv_only: bool = False) -> None:
    """The indexes every shape needs: one ANN, one full-text.

    Separate HNSW per table is the point of the split. A search for events now
    walks an index containing only events, so top-k means top-k events.
    """
    op.execute(
        f"""
        CREATE INDEX {table}_embedding_idx ON {table}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(f"CREATE INDEX {table}_tsv_idx ON {table} USING GIN (tsv)")
    op.execute(f"CREATE INDEX {table}_labels_idx ON {table} USING GIN (labels)")
    op.execute(
        f"CREATE UNIQUE INDEX {table}_natural_key_idx "
        f"ON {table} (user_id, connector, scope_key, source_id, chunk_index)"
    )


def upgrade() -> None:
    # ---------------------------------------------------------------- messages
    #
    # `scope_key` is the namespace that makes `source_id` unique inside a
    # connector — a mailbox, a Slack workspace. It defaults to '' because most
    # mail connectors do not namespace their ids and should not have to think
    # about it.
    op.execute(
        f"""
        CREATE TABLE sync_messages (
            {SPINE},
            scope_key           VARCHAR(255) NOT NULL DEFAULT '',

            -- what it says
            subject             TEXT,
            body                TEXT NOT NULL DEFAULT '',

            -- who. Separate columns because "to Sarah" and "involving Sarah"
            -- are different questions and both need to be indexable.
            from_email          CITEXT,
            from_name           TEXT,
            to_emails           CITEXT[] NOT NULL DEFAULT '{{}}',
            cc_emails           CITEXT[] NOT NULL DEFAULT '{{}}',
            participant_emails  CITEXT[] GENERATED ALWAYS AS (
                                    array_remove(
                                        array_cat(
                                            array_cat(to_emails, cc_emails),
                                            ARRAY[from_email]::CITEXT[]
                                        ),
                                        NULL
                                    )
                                ) STORED,

            -- when, and how it groups
            sent_at             TIMESTAMPTZ NOT NULL,
            thread_id           VARCHAR(255),

            is_unread           BOOLEAN NOT NULL DEFAULT false,
            has_attachments     BOOLEAN NOT NULL DEFAULT false,

            {_tsv('subject', 'body')}
        )
        """
    )
    _retrieval_indexes("sync_messages")
    op.execute(
        "CREATE INDEX sync_messages_time_idx "
        "ON sync_messages (user_id, connector, sent_at DESC)"
    )
    op.execute(
        "CREATE INDEX sync_messages_from_idx ON sync_messages (user_id, from_email) "
        "WHERE from_email IS NOT NULL"
    )
    # The whole reason for the split: a recipient lookup that an index serves.
    op.execute("CREATE INDEX sync_messages_to_idx ON sync_messages USING GIN (to_emails)")
    op.execute("CREATE INDEX sync_messages_cc_idx ON sync_messages USING GIN (cc_emails)")
    op.execute(
        "CREATE INDEX sync_messages_participants_idx "
        "ON sync_messages USING GIN (participant_emails)"
    )
    op.execute(
        "CREATE INDEX sync_messages_thread_idx ON sync_messages (user_id, thread_id) "
        "WHERE thread_id IS NOT NULL"
    )

    # ------------------------------------------------------------------ events
    #
    # `scope_key` here is the calendar id, and it is load-bearing rather than
    # cosmetic: Google event ids are unique only *within* a calendar, so two
    # calendars can hand out the same id for different meetings.
    op.execute(
        f"""
        CREATE TABLE sync_events (
            {SPINE},
            scope_key           VARCHAR(255) NOT NULL DEFAULT '',

            title               TEXT,
            description         TEXT NOT NULL DEFAULT '',

            organizer_email     CITEXT,
            attendee_emails     CITEXT[] NOT NULL DEFAULT '{{}}',
            -- [{{email, name, response_status, optional}}] — the detail behind
            -- the flat array above, for showing who accepted.
            attendees           JSONB NOT NULL DEFAULT '[]'::jsonb,

            starts_at           TIMESTAMPTZ NOT NULL,
            ends_at             TIMESTAMPTZ,
            all_day             BOOLEAN NOT NULL DEFAULT false,

            location            TEXT,
            status              VARCHAR(32),
            recurring_event_id  VARCHAR(255),

            {_tsv('title', 'description')}
        )
        """
    )
    _retrieval_indexes("sync_events")
    op.execute(
        "CREATE INDEX sync_events_time_idx "
        "ON sync_events (user_id, connector, starts_at DESC)"
    )
    op.execute(
        "CREATE INDEX sync_events_attendees_idx "
        "ON sync_events USING GIN (attendee_emails)"
    )
    op.execute(
        "CREATE INDEX sync_events_organizer_idx ON sync_events (user_id, organizer_email) "
        "WHERE organizer_email IS NOT NULL"
    )
    # Cancelled events stay in the mirror — "what did I cancel" is a real
    # question — but almost every read wants them gone, so the common case
    # gets its own smaller index.
    op.execute(
        "CREATE INDEX sync_events_live_idx ON sync_events (user_id, starts_at) "
        "WHERE status IS DISTINCT FROM 'cancelled'"
    )

    # ------------------------------------------------------------------- files
    op.execute(
        f"""
        CREATE TABLE sync_files (
            {SPINE},
            scope_key           VARCHAR(255) NOT NULL DEFAULT '',

            name                TEXT,
            content             TEXT NOT NULL DEFAULT '',

            owner_email         CITEXT,
            shared_with_emails  CITEXT[] NOT NULL DEFAULT '{{}}',

            mime_type           VARCHAR(255),
            size_bytes          BIGINT,
            modified_at         TIMESTAMPTZ NOT NULL,

            folder_id           VARCHAR(255),
            folder_path         TEXT,
            is_shared           BOOLEAN NOT NULL DEFAULT false,

            {_tsv('name', 'content')}
        )
        """
    )
    _retrieval_indexes("sync_files")
    op.execute(
        "CREATE INDEX sync_files_time_idx "
        "ON sync_files (user_id, connector, modified_at DESC)"
    )
    op.execute(
        "CREATE INDEX sync_files_owner_idx ON sync_files (user_id, owner_email) "
        "WHERE owner_email IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX sync_files_shared_idx "
        "ON sync_files USING GIN (shared_with_emails)"
    )
    op.execute(
        "CREATE INDEX sync_files_folder_idx ON sync_files (user_id, folder_id) "
        "WHERE folder_id IS NOT NULL"
    )
    # "PDFs from last month" is a mime + time question and a common one.
    op.execute(
        "CREATE INDEX sync_files_mime_idx ON sync_files (user_id, mime_type, modified_at DESC) "
        "WHERE mime_type IS NOT NULL"
    )

    # The mirror is disposable by design, so the old table simply goes. A
    # resync rebuilds every row it held.
    op.execute("DROP TABLE IF EXISTS sync_items")
    op.execute("DROP FUNCTION IF EXISTS role_emails(JSONB, TEXT)")
    op.execute("DROP FUNCTION IF EXISTS participant_email_list(JSONB)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS sync_files")
    op.execute("DROP TABLE IF EXISTS sync_events")
    op.execute("DROP TABLE IF EXISTS sync_messages")
