"""Database layer: declarative base, session handling, models, repositories.

Nothing outside this package writes SQL. Every repository function takes
``user_id`` as its first real argument so a query cannot cross tenants by
accident.
"""

from app.db.base import Base, metadata
from app.db.session import (
    get_engine,
    get_session,
    get_sessionmaker,
    session_scope,
    shutdown_engine,
)

__all__ = [
    "Base",
    "metadata",
    "get_engine",
    "get_sessionmaker",
    "get_session",
    "session_scope",
    "shutdown_engine",
]
