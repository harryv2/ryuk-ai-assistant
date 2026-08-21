"""Declarative base and the metadata every model hangs off."""

from __future__ import annotations

from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Explicit, predictable constraint names. Index names are given by hand on each
# model so the hand-written migration and the models agree character for
# character; this convention only covers the constraints we do not name.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "%(table_name)s_%(column_0_N_name)s_idx",
    "uq": "%(table_name)s_%(column_0_N_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_N_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """Column values as a plain dict. Handy for SSE payloads and tests."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk!r}>"
