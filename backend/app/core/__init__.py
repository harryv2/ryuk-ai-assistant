"""Core utilities shared by every other module.

Nothing in here imports from ``app.db``, ``app.orchestrator`` or ``app.google``.
The dependency arrow points one way: everything may import ``app.core``,
``app.core`` imports only ``app.config``.
"""

from app.core.errors import AppError

__all__ = ["AppError"]
