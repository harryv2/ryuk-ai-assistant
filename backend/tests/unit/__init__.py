"""Unit tests.

Nothing in here touches Postgres, Redis, Google or OpenAI. Every test is pure
Python against a frozen clock, so the suite runs in well under a second and
gives the same answer on any machine on any day.

Anything that needs a live store belongs in `tests/integration` and carries the
`integration` marker declared in `pyproject.toml`.
"""
