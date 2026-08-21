"""Integration tests: a real Postgres, a real FastAPI app, no network at all.

Everything Gmail, Calendar, Drive and OpenAI would have said is served from
``tests/fixtures``; ``respx`` intercepts httpx so nothing leaves the process.
What is *not* faked is the database — these run against a live Postgres with
pgvector, on the schema the Alembic migration builds, so the generated columns
(``tsv``, ``attendee_emails``), the HNSW indexes and the partial unique index on
``actions.dedupe_key`` are genuinely exercised.

Set ``DATABASE_URL_TEST`` to run them. Without it every test in here skips with
a message saying so, rather than pretending to pass.
"""

from __future__ import annotations
