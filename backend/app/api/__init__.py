"""The HTTP surface.

Versioned under :mod:`app.api.v1`. Nothing in here holds business logic — a
router validates a request, calls one thing, and shapes what comes back. The
orchestrator, the ops and the repositories own the rest.
"""
