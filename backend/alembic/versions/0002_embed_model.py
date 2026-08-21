"""Bring an older database up to the `embed_model` column 0001 now creates.

Chat models are swappable; embeddings are not. Vectors from two different models
are not comparable even at the same dimensionality, so a cosine search across a
mixed table returns confident nonsense. Stamping the model on the row is what
lets the search path refuse rather than mislead.

`0001_initial` creates `embed_model` and its `(user_id, embed_model)` index as
part of each `sync_` table, which is where they belong: they are not a later
addition to the design, they are the design. **A database built from today's
0001 therefore already has everything in here, and this migration is a no-op on
it.** It exists for the databases created from the earlier 0001 that had neither
— every statement is guarded so both paths end at the same schema.

That is also why the column is added here with a *non-empty* default. A database
that predates the column has rows whose vectors were produced by whatever
`OPENAI_EMBED_MODEL` was at the time, which in this codebase has only ever been
`text-embedding-3-small`; stamping them `''` would mark every existing row as
never embedded and buy a full, pointless re-embed. The default is then set back
to `''` for the same reason 0001 uses it: which model produced a vector is
something the embedder writes when it writes the vector, never something an
INSERT assumes on its behalf.

Revision ID: 0002_embed_model
Revises: 0001_initial
"""

from __future__ import annotations

from alembic import op

revision = "0002_embed_model"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

TABLES = ("sync_gmail", "sync_gcal", "sync_gdrive")

# What the vectors in a database older than this column were embedded with. An
# assumption, and a safe one: it is the only embedding model this codebase has
# ever had a default for. Written as a literal rather than read from settings so
# that running this migration twice, or a year from now, produces the same rows.
ASSUMED = "openai:text-embedding-3-small"


def upgrade() -> None:
    for table in TABLES:
        op.execute(
            f"ALTER TABLE {table} "
            f"ADD COLUMN IF NOT EXISTS embed_model VARCHAR(64) NOT NULL DEFAULT '{ASSUMED}'"
        )
        op.execute(f"ALTER TABLE {table} ALTER COLUMN embed_model SET DEFAULT ''")
        # The name 0001 and app/db/models.py both use. Search already prefilters
        # on user_id, so putting embed_model beside it makes "were these rows
        # embedded by the model asking the question?" an index-only scan.
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_user_id_embed_model_idx "
            f"ON {table} (user_id, embed_model)"
        )
        # An earlier cut of this migration created the same index under a
        # shorter name. One index on one pair of columns, under the name the
        # model metadata expects, or the next autogenerate proposes to add it
        # back.
        op.execute(f"DROP INDEX IF EXISTS {table}_embed_model_idx")

    _rename_long_constraint()


def downgrade() -> None:
    """Nothing to take back.

    The column and its index belong to `0001_initial` on any database built
    since it started creating them, and dropping them here would leave a
    downgrade to 0001 with a schema 0001 does not describe. A database old
    enough for this migration to have really added the column is old enough that
    dropping vectors' provenance on the way back down is the worse of the two
    outcomes.
    """


def _rename_long_constraint() -> None:
    """Give the conversation_entities unique constraint its short name.

    An earlier 0001 named it after its three columns, which is 64 characters.
    Postgres truncated that to 63 silently, so the database and
    `app/db/models.py` disagreed and every `alembic revision --autogenerate`
    died on the length before it could diff anything.

    Guarded on both sides so it is a no-op on a database built from today's
    0001 (already short) and on one that has run this migration before.
    Renaming is safe: nothing references this constraint by name.
    """
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'conversation_entities'::regclass
                  AND conname = 'conversation_entities_conversation_id_entity_type_entity_ref_ke'
            ) AND NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'conversation_entities'::regclass
                  AND conname = 'conversation_entities_ref_key'
            ) THEN
                ALTER TABLE conversation_entities
                RENAME CONSTRAINT conversation_entities_conversation_id_entity_type_entity_ref_ke
                TO conversation_entities_ref_key;
            END IF;
        END $$;
        """
    )
