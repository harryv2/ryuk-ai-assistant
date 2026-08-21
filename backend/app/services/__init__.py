"""Thin wrappers over the three Google APIs.

One class per service, each holding a :class:`~app.google.client.Transport`
bound to one user. They are thin on purpose: they translate arguments into a
request, and a response into the flat dict shape the mirror and the ops use.
No policy, no database, no LLM. Retries, quota and the circuit breaker all live
one layer down in ``app.google``.

They are reached as ``ctx.google.gmail`` / ``.gcal`` / ``.gdrive``.
"""

from app.services.gcal import CalendarService
from app.services.gdrive import DriveService
from app.services.gmail import GmailService

__all__ = ["GmailService", "CalendarService", "DriveService"]
