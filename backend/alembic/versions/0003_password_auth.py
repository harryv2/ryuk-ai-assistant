"""Email + password sign-in, alongside Google.

Every column is nullable or defaulted, because the table already holds accounts
created through Google OAuth. Those have no password and must keep working
exactly as they do — signing in with Google is not being replaced, it is being
joined by a second door.

Revision ID: 0003_password_auth
Revises: 0002_embed_model
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_password_auth"
down_revision = "0002_embed_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(255), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))

    # Sign-in looks a user up by email on every attempt, including the failures.
    # `email` is already unique, so this is really about making the lookup for a
    # non-existent address as cheap as one for a real one — a slow miss is a way
    # to find out which addresses are registered.
    op.create_index(
        "users_password_lookup_idx",
        "users",
        ["email"],
        postgresql_where=sa.text("password_hash IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("users_password_lookup_idx", table_name="users")
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "email_verified")
    op.drop_column("users", "password_hash")
