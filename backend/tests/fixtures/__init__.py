"""Recorded payloads for the integration suite.

Two modules, one job each:

* :mod:`tests.fixtures.google_responses` — what Gmail, Calendar and Drive
  actually put on the wire, including the error bodies (429, 503, 401, 412,
  410) that the retry classifier reads. It also knows how to turn those same
  payloads into ``sync_*`` mirror rows, so a test can seed the mirror without
  running a sync first.
* :mod:`tests.fixtures.llm_responses` — canned plan JSON, one per scenario in
  ``docs/SAMPLE_QUERIES.md``, plus the OpenAI response envelopes and a
  deterministic stand-in for the embedding model.

Nothing in here reaches the network, and nothing in here imports ``app``. The
payloads are data; the interception lives in ``tests/integration/conftest.py``.
"""

from __future__ import annotations

__all__ = ["google_responses", "llm_responses"]
