"""The op registry: every action the system can take, and nothing else.

Import what you need from here rather than from the service modules — the
registry is what guarantees there is one instance of each op and one list of
what exists::

    from app.ops import REGISTRY, catalogue, get

    op = get("gmail.send_email")
    prompt = catalogue()
"""

from app.ops.base import (
    ConfirmableOp,
    InputRequest,
    Op,
    OpContext,
    OpResult,
    SearchOp,
)
from app.ops.registry import (
    ALIASES,
    CANONICAL,
    REGISTRY,
    by_service,
    catalogue,
    confirmables,
    get,
    local_ops,
    names,
    require,
    writes,
)

__all__ = [
    "ALIASES",
    "CANONICAL",
    "REGISTRY",
    "ConfirmableOp",
    "InputRequest",
    "Op",
    "OpContext",
    "OpResult",
    "SearchOp",
    "by_service",
    "catalogue",
    "confirmables",
    "get",
    "local_ops",
    "names",
    "require",
    "writes",
]
