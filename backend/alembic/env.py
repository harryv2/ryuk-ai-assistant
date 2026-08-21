"""Alembic environment.

Migrations run synchronously over psycopg while the app runs over asyncpg, so
the one DATABASE_URL in the environment is translated here rather than being
configured twice.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# The backend package root, so `import app...` works however alembic was invoked.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.db import models as _models  # noqa: E402,F401  imported for its side effect

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.sync_database_url.replace("%", "%%"))


def _metadata():
    """The declarative Base lives in db/base.py; db/models.py registers the tables."""
    for module_name in ("app.db.base", "app.db.models"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        base = getattr(module, "Base", None)
        if base is not None and getattr(base, "metadata", None) is not None:
            return base.metadata
    try:
        from app.db.base import Base
    except ImportError:
        from app.db.models import Base  # type: ignore[no-redef]

    return Base.metadata


target_metadata = _metadata()

# Postgres builds these for us. Autogenerate must not offer to drop them.
EXCLUDED_TABLES = {"alembic_version", "spatial_ref_sys"}


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    if type_ == "table" and name in EXCLUDED_TABLES:
        return False
    # Anything reflected that the models do not know about is not ours to drop —
    # extension tables, and anything a DBA added by hand.
    if type_ == "table" and reflected and compare_to is None:
        return False
    return True


def render_item(type_, obj, autogen_context):
    """Emit `pgvector.sqlalchemy.Vector(1536)` with its import, not a bare repr."""
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector"):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        dim = getattr(obj, "dim", None)
        return f"pgvector.sqlalchemy.Vector(dim={dim})" if dim else "pgvector.sqlalchemy.Vector()"
    return False


def process_revision_directives(context_, revision, directives) -> None:
    """Do not write a migration that changes nothing."""
    if getattr(config.cmd_opts, "autogenerate", False) and directives:
        script = directives[0]
        if script.upgrade_ops is not None and script.upgrade_ops.is_empty():
            directives[:] = []
            print("no schema changes detected - nothing written")


COMMON = dict(
    target_metadata=target_metadata,
    compare_type=True,
    compare_server_default=True,
    include_object=include_object,
    render_item=render_item,
    render_as_batch=False,
    process_revision_directives=process_revision_directives,
)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **COMMON,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, **COMMON)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
