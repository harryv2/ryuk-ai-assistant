"""Retrieval: the mirror, the arms, the fusion and the probe.

Five modules, in the order a query meets them:

* :mod:`app.search.aliases` — "Turkish Airlines" is also THY, TK and thy.com.
* :mod:`app.search.chunking` — what actually gets embedded, and where a long
  document is cut.
* :mod:`app.search.embedder` — mirror rows to vectors, skipping anything whose
  content has not changed.
* :mod:`app.search.hybrid` — the two arms, RRF for ordering, ``cn`` and
  ``evidence`` for deciding, temporal shaping on top.
* :mod:`app.search.probe` — one embedding and three hybrid searches, run before
  the planner, so the model plans against facts instead of guesses.

Submodules are imported lazily: importing this package must not drag in the
database layer for something that only wanted the alias table.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from app.search.hybrid import Hit as Hit
    from app.search.hybrid import ServiceResult as ServiceResult
    from app.search.hybrid import Terms as Terms
    from app.search.hybrid import search as search
    from app.search.probe import ProbeResult as ProbeResult
    from app.search.probe import probe as probe

_SUBMODULES = {"aliases", "chunking", "embedder", "extractors", "hybrid", "probe"}

_SYMBOLS = {
    "Hit": "hybrid",
    "ServiceResult": "hybrid",
    "Terms": "hybrid",
    "search": "hybrid",
    "ProbeResult": "probe",
    "probe": "probe",
    "Extraction": "extractors",
    "extract": "extractors",
    "expand": "aliases",
    "chunk": "chunking",
    "clean_email": "chunking",
}

__all__ = sorted(_SUBMODULES | set(_SYMBOLS))


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"app.search.{name}")
    module = _SYMBOLS.get(name)
    if module is not None:
        return getattr(importlib.import_module(f"app.search.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
