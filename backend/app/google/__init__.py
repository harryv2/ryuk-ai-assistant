"""The Google API layer: transport, retry policy, and quota prices.

Three modules, and the split is deliberate:

* :mod:`app.google.quota` — what each method costs, in Google's own units.
* :mod:`app.google.retry` — what a failure means, how long to wait, and the per
  user-and-service circuit breaker.
* :mod:`app.google.client` — async httpx transport that puts the two together,
  plus :class:`~app.google.client.GoogleClients`, the container an op sees as
  ``ctx.google``.

Nothing above this layer talks to Google. The thin service wrappers in
``app.services`` are the only callers.
"""

from app.google.client import (
    GoogleClients,
    TokenHolder,
    Transport,
    close_http,
    get_http,
)
from app.google.quota import units_for
from app.google.retry import (
    CircuitOpen,
    ErrorClass,
    GoogleAPIError,
    backoff,
    classify,
    retryable,
)

__all__ = [
    "GoogleClients",
    "Transport",
    "TokenHolder",
    "get_http",
    "close_http",
    "units_for",
    "ErrorClass",
    "GoogleAPIError",
    "CircuitOpen",
    "classify",
    "backoff",
    "retryable",
]
