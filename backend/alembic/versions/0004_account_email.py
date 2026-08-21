"""Remember which workspace a connection actually belongs to.

Signing in and connecting a workspace used to be the same act, so the address
on the account was necessarily the address of the workspace. They are separate
now — you can sign in with a password and connect any Google account — and the
moment that is true, "which workspace am I connected to?" stops being
answerable from the user row.

``provider_account_id`` already identifies the account, but it holds Google's
opaque subject id. Nobody recognises their own workspace from that.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_account_email"
down_revision = "0003_password_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "oauth_tokens",
        sa.Column("account_email", sa.String(320), nullable=True),
    )
    # Existing rows were created when the two addresses had to match, so the
    # user's own address is the right answer for them.
    op.execute(
        """
        UPDATE oauth_tokens AS t
           SET account_email = u.email
          FROM users AS u
         WHERE u.id = t.user_id
           AND t.account_email IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("oauth_tokens", "account_email")
