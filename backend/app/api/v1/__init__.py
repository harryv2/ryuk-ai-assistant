"""Version 1 of the HTTP API.

Every module here exposes a module-level ``router``. ``app.main._mount`` looks
it up by name, so a module that forgets one fails loudly at import rather than
quietly serving nothing.

The whole surface:

===========================================  ================================
``POST   /api/v1/query``                     the one entry point
``GET    /api/v1/runs/{id}/events``          SSE for one run
``GET    /api/v1/runs/{id}/steps``           trace detail, loaded lazily
``GET    /api/v1/auth/google``               start the OAuth handshake
``GET    /api/v1/auth/google/callback``      finish it
``GET    /api/v1/auth/me``                   who is signed in
``POST   /api/v1/auth/logout``               drop the session cookie
``DELETE /api/v1/auth/google``               disconnect, and optionally purge
``POST   /api/v1/sync/trigger``              force a sync pass
``GET    /api/v1/sync/status``               where each sync got to
``GET    /api/v1/prompts``                   what the system is waiting on
``POST   /api/v1/prompts/{id}/respond``      answer a card
``POST   /api/v1/prompts/{id}/cancel``       dismiss one
``GET    /api/v1/conversations``             the thread list
``GET    /api/v1/conversations/{id}``        one thread, fully resolved
``GET    /api/v1/conversations/{id}/events`` SSE for action outcomes
``GET    /api/v1/search``                    retrieval, with every score shown
``GET    /healthz  /readyz  /metrics``       probes, at the root
===========================================  ================================
"""

from app.api.v1 import (
    auth,
    conversations,
    events,
    health,
    prompts,
    query,
    schemas,
    search,
    sync,
)

__all__ = [
    "auth",
    "conversations",
    "events",
    "health",
    "prompts",
    "query",
    "schemas",
    "search",
    "sync",
]
