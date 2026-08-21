"""The HTTP layer under every Google call.

One `httpx.AsyncClient` per process, shared. Not `googleapiclient`: that library
is synchronous, builds its surface by downloading a discovery document, and
would block the event loop on every call. Three parallel searches at ~110ms
only work if the calls are actually parallel, so the request path talks to the
REST endpoints directly.

A :class:`Transport` binds one service to one user's access token and owns
everything that has to happen around a call:

1. ask the circuit breaker whether this service is worth calling at all,
2. charge the call's quota units to the user's bucket,
3. send it,
4. put any failure in a class, and act on the class — refresh the token once
   for AUTH_EXPIRED, refetch once for PRECONDITION, sleep with full jitter for
   TRANSIENT and RATE_LIMITED, stop for the rest,
5. tell the breaker what happened.

:class:`GoogleClients` is what an op sees: ``ctx.google.gmail.search(...)``.
It holds the three service objects, all sharing one token holder, so a refresh
triggered by Gmail is immediately in use by Calendar and Drive.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx

from app.core.errors import AppError
from app.core.logging import get_logger
from app.google import quota
from app.google.retry import (
    CircuitOpen,
    ErrorClass,
    GoogleAPIError,
    backoff,
    breaker_allow,
    breaker_record_failure,
    breaker_record_success,
    classify,
    max_attempts,
    retryable,
)

if TYPE_CHECKING:  # imported lazily at run time to keep the layers acyclic
    from app.services.gcal import CalendarService
    from app.services.gdrive import DriveService
    from app.services.gmail import GmailService

log = get_logger(__name__)

GMAIL_BASE: Final[str] = "https://gmail.googleapis.com/gmail/v1"
CALENDAR_BASE: Final[str] = "https://www.googleapis.com/calendar/v3"
DRIVE_BASE: Final[str] = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE: Final[str] = "https://www.googleapis.com/upload/drive/v3"

SERVICE_BASE: Final[dict[str, str]] = {
    "gmail": GMAIL_BASE,
    "gcal": CALENDAR_BASE,
    "gdrive": DRIVE_BASE,
}

#: Per-request ceiling. An op's own timeout is usually tighter; this is the
#: backstop that stops a socket hanging for ever.
DEFAULT_TIMEOUT_S: Final[float] = 10.0
CONNECT_TIMEOUT_S: Final[float] = 3.0

#: How long one call may spend inside the retry loop, sleeps included.
DEFAULT_MAX_ELAPSED_S: Final[float] = 45.0

#: Retries a single interactive call may make on its own.
#:
#: There are two retry tiers and they must not multiply. `app.google.retry`
#: sizes its policy for a Celery worker, where five attempts over a minute is
#: the right call. On a request a person is watching, `dispatch.py` owns the
#: budget — two attempts, capped at 1.5 s of added latency — so the transport
#: underneath retries at most once and lets the dispatcher decide whether to
#: try again. Without this cap, two dispatcher attempts times five transport
#: attempts is ten calls to a service that is already down.
INTERACTIVE_MAX_RETRIES: Final[int] = 1
#: And the whole interactive call, sleeps included, stays inside the
#: dispatcher's per-step timeout rather than its own 45 s backstop.
INTERACTIVE_MAX_ELAPSED_S: Final[float] = 10.0

#: The longest an interactive retry may sleep before trying again.
#:
#: `app.google.retry.backoff` is the worker's policy: a 429 carrying
#: ``Retry-After: 1`` comes back as roughly four and a half seconds, which is
#: the right answer for a Celery task and far past a step timeout on a request
#: somebody is watching. Sleeping it does not produce a slow success, it
#: produces a cancelled step — so an interactive wait is capped here, and a
#: wait that cannot fit is not taken at all: the error is raised and the
#: dispatcher decides, which is the tier that owns the user-facing budget.
INTERACTIVE_MAX_BACKOFF_S: Final[float] = 1.2

Expect = Literal["json", "text", "bytes", "none"]

#: The failures that prove Google did not do the thing: it turned the request
#: away at the door. A send or a share may only be repeated on one of these —
#: a 500 is ambiguous, and repeating it is how a person gets two emails.
SAFE_TO_REPEAT: Final[frozenset[ErrorClass]] = frozenset(
    {ErrorClass.AUTH_EXPIRED, ErrorClass.RATE_LIMITED, ErrorClass.PRECONDITION}
)

_http: httpx.AsyncClient | None = None
_http_lock = asyncio.Lock()


# --------------------------------------------------------------------------- #
# The shared connection pool
# --------------------------------------------------------------------------- #


async def get_http() -> httpx.AsyncClient:
    """The process-wide client. Keep-alive to Google is most of the latency win."""
    global _http
    if _http is not None:
        return _http
    async with _http_lock:
        if _http is None:
            _http = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
                http2=False,
                follow_redirects=True,
                headers={"User-Agent": "alpha-law-orchestrator/1.0"},
            )
    return _http


async def close_http() -> None:
    """Shut the pool down. Called on app shutdown."""
    global _http
    if _http is not None:
        try:
            await _http.aclose()
        except Exception:  # pragma: no cover - shutdown is best effort
            pass
        _http = None


def set_http(client: httpx.AsyncClient | None) -> None:
    """Replace the shared client. Tests mount a transport this way."""
    global _http
    _http = client


# --------------------------------------------------------------------------- #
# Access tokens
# --------------------------------------------------------------------------- #

TokenGetter = Callable[..., Awaitable[str]]


class TokenHolder:
    """One user's access token, shared by their three service objects.

    A refresh is done under a lock, so three parallel calls that all see a 401
    cause one refresh, not three.
    """

    def __init__(
        self,
        user_id: str,
        access_token: str | None = None,
        getter: TokenGetter | None = None,
    ) -> None:
        self.user_id = user_id
        self._token = access_token
        self._getter = getter
        self._lock = asyncio.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        """Bumped on every refresh, so a caller can tell whether the token it
        used has already been replaced by someone else."""
        return self._version

    async def get(self, *, force_refresh: bool = False) -> str:
        if self._token and not force_refresh:
            return self._token
        async with self._lock:
            if self._token and not force_refresh:
                return self._token
            if self._getter is None:
                if self._token:
                    return self._token
                raise AppError(
                    "GOOGLE_REAUTH_REQUIRED",
                    details={"user_id": self.user_id, "reason": "no_access_token"},
                )
            token = await _call_getter(self._getter, force_refresh)
            if not token:
                raise AppError(
                    "GOOGLE_REAUTH_REQUIRED",
                    details={"user_id": self.user_id, "reason": "refresh_returned_nothing"},
                )
            self._token = token
            self._version += 1
            return token

    async def refreshed(self, stale_version: int) -> str:
        """A token newer than ``stale_version``, refreshing only if needed."""
        if self._version > stale_version:
            return await self.get()
        return await self.get(force_refresh=True)


_getter_takes_force: dict[int, bool] = {}


async def _call_getter(getter: TokenGetter, force_refresh: bool) -> str:
    """Call the token source, passing ``force_refresh`` only if it takes one.

    ``auth/token_store.py`` is written by another module against the same
    contract; this keeps us honest about the one parameter whose name is not
    fixed by it.
    """
    key = id(getter)
    takes = _getter_takes_force.get(key)
    if takes is None:
        try:
            params = inspect.signature(getter).parameters
            takes = "force_refresh" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):  # builtins, partials without signatures
            takes = False
        _getter_takes_force[key] = takes
    result = getter(force_refresh=force_refresh) if takes else getter()
    return await result


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


class Transport:
    """One Google service, bound to one user's token.

    Every call goes through :meth:`request`, which names the method it is
    making — ``"gmail.messages.send"`` — because that name is the quota price,
    the log line and the breaker's service key all at once.
    """

    def __init__(
        self,
        service: str,
        user_id: str,
        tokens: TokenHolder,
        *,
        base_url: str | None = None,
        share: quota.Share = "interactive",
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_elapsed_s: float = DEFAULT_MAX_ELAPSED_S,
    ) -> None:
        self.service = service
        self.user_id = user_id
        self.tokens = tokens
        self.base_url = (base_url or SERVICE_BASE[service]).rstrip("/")
        self.share = share
        self.timeout = timeout
        # An interactive call answers to the dispatcher's budget, not to the
        # worker-sized policy in app.google.retry. See INTERACTIVE_MAX_RETRIES.
        if share == "interactive" and max_elapsed_s == DEFAULT_MAX_ELAPSED_S:
            max_elapsed_s = INTERACTIVE_MAX_ELAPSED_S
        self.max_elapsed_s = max_elapsed_s
        self._http = http
        #: Populated as calls are made. The dispatcher copies it onto the node.
        self.attempts: list[dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------ #

    async def _client(self) -> httpx.AsyncClient:
        return self._http or await get_http()

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
        """Drop Nones and render booleans and lists the way Google reads them."""
        if not params:
            return None
        out: dict[str, Any] = {}
        for name, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                out[name] = "true" if value else "false"
            elif isinstance(value, (list, tuple, set)):
                items = [str(v) for v in value if v is not None]
                if items:
                    out[name] = items
            else:
                out[name] = value
        return out or None

    def _wait(self, cls: ErrorClass, attempt: int, retry_after: float | None) -> float | None:
        """How long to sleep before retrying, or None to not retry at all.

        Background keeps the worker policy. Interactive gets it capped, and a
        wait too long to fit inside a step is refused rather than truncated:
        retrying before `Retry-After` has elapsed just collects a second 429.
        """
        delay = backoff(cls, attempt, retry_after)
        if self.share != "interactive":
            return delay
        if retry_after is not None and retry_after > INTERACTIVE_MAX_BACKOFF_S:
            return None
        return min(delay, INTERACTIVE_MAX_BACKOFF_S)

    def _attempt_cap(self, cls: ErrorClass) -> int:
        """Retries this transport allows for one error class.

        The policy in `app.google.retry` is sized for a worker. An interactive
        call is one attempt inside a dispatcher that will itself try again, so
        it is capped here rather than compounding into ten calls at a service
        that is already down.
        """
        allowed = max_attempts(cls)
        if self.share == "interactive":
            return min(allowed, INTERACTIVE_MAX_RETRIES)
        return allowed

    def _record(self, cls: ErrorClass, status: int | None, delay: float) -> None:
        self.attempts.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "error_class": str(cls),
                "google_status": status,
                "backoff_ms": int(delay * 1000),
            }
        )

    # -- the one call ------------------------------------------------------- #

    async def request(
        self,
        http_method: str,
        path: str,
        *,
        api_method: str,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        expect: Expect = "json",
        timeout: float | None = None,
        units: int | None = None,
        retry: bool = True,
        retry_on_network: bool = True,
        retry_on: frozenset[ErrorClass] | None = None,
        refetch: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    ) -> Any:
        """Make one Google call, with the whole reliability story around it.

        ``api_method`` is the quota name, such as ``"gmail.messages.send"``.

        ``refetch`` is called when Google says our etag is stale: it returns
        overrides — ``{"headers": {...}, "json": {...}}`` — for exactly one more
        attempt. Without it a PRECONDITION is final, because nothing about
        repeating the same stale request would help.

        ``retry_on_network=False`` is for the calls that are not safe to repeat
        blind: a send, an insert, a share. An error *response* is still retried,
        because a status code is Google telling us it did not do the thing. A
        timeout or a dropped connection is not — the send may well have landed,
        and the only safe move is to hand the decision up to the op, which can
        look for the idempotency key before trying again.

        ``retry_on`` narrows retries to a set of classes. Writes pass
        :data:`SAFE_TO_REPEAT`, which is the set that proves the request was
        turned away rather than performed.
        """
        client = await self._client()
        url = self._url(path)
        query = self._clean_params(params)
        body = json
        extra_headers = dict(headers or {})
        used: dict[ErrorClass, int] = {}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_elapsed_s
        probing = False
        failed_once = False

        while True:
            decision = await breaker_allow(self.user_id, self.service)
            if not decision.allowed:
                raise CircuitOpen(self.user_id, self.service, decision.retry_after_s)
            probing = probing or decision.is_probe

            token = await self.tokens.get()
            token_version = self.tokens.version
            await quota.acquire(
                self.user_id, api_method, units=units, share=self.share
            )

            request_headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                **extra_headers,
            }

            try:
                response = await client.request(
                    http_method.upper(),
                    url,
                    params=query,
                    json=body if content is None else None,
                    content=content,
                    headers=request_headers,
                    timeout=timeout or self.timeout,
                )
            except Exception as exc:  # network, timeout, protocol
                cls = classify(exc)
                failed_once = True
                await breaker_record_failure(self.user_id, self.service, cls)
                attempt = used.get(cls, 0)
                if (
                    not retry
                    or not retry_on_network
                    or not retryable(cls)
                    or (retry_on is not None and cls not in retry_on)
                    or attempt >= self._attempt_cap(cls)
                ):
                    raise self._wrap(exc, api_method) from exc
                delay = self._wait(cls, attempt, None)
                if delay is None or loop.time() + delay > deadline:
                    raise self._wrap(exc, api_method) from exc
                used[cls] = attempt + 1
                self._record(cls, None, delay)
                log.info(
                    "google.retry",
                    service=self.service,
                    method=api_method,
                    error_class=str(cls),
                    attempt=attempt + 1,
                    backoff_ms=int(delay * 1000),
                    error=str(exc)[:200],
                )
                if delay:
                    await asyncio.sleep(delay)
                continue

            if response.status_code < 400:
                if failed_once or probing:
                    await breaker_record_success(self.user_id, self.service)
                return _read(response, expect)

            error = GoogleAPIError.from_response(
                response, service=self.service, method=api_method
            )
            cls = error.error_class
            failed_once = True
            await breaker_record_failure(self.user_id, self.service, cls)
            attempt = used.get(cls, 0)

            if (
                not retry
                or not retryable(cls)
                or (retry_on is not None and cls not in retry_on)
                or attempt >= self._attempt_cap(cls)
            ):
                raise error

            # A stale access token: get a new one and go again, no sleep.
            if cls is ErrorClass.AUTH_EXPIRED:
                try:
                    await self.tokens.refreshed(token_version)
                except AppError:
                    raise error from None
                used[cls] = attempt + 1
                self._record(cls, error.status, 0.0)
                log.info(
                    "google.token_refreshed",
                    service=self.service,
                    method=api_method,
                    user_id=self.user_id,
                )
                continue

            # A stale etag: read the current version once, then retry with it.
            if cls is ErrorClass.PRECONDITION:
                if refetch is None:
                    raise error
                overrides = await refetch()
                if overrides.get("headers"):
                    extra_headers.update(overrides["headers"])
                if "json" in overrides:
                    body = overrides["json"]
                if overrides.get("params"):
                    query = self._clean_params({**(query or {}), **overrides["params"]})
                used[cls] = attempt + 1
                self._record(cls, error.status, 0.0)
                log.info(
                    "google.refetched",
                    service=self.service,
                    method=api_method,
                    user_id=self.user_id,
                )
                continue

            delay = self._wait(cls, attempt, error.retry_after)
            if delay is None or loop.time() + delay > deadline:
                raise error
            used[cls] = attempt + 1
            self._record(cls, error.status, delay)
            log.info(
                "google.retry",
                service=self.service,
                method=api_method,
                error_class=str(cls),
                google_status=error.status,
                google_reason=error.reason,
                attempt=attempt + 1,
                backoff_ms=int(delay * 1000),
            )
            if delay:
                await asyncio.sleep(delay)

    def _wrap(self, exc: Exception, api_method: str) -> Exception:
        """Turn a transport failure into something the dispatcher can read."""
        if isinstance(exc, (GoogleAPIError, AppError)):
            return exc
        return GoogleAPIError(
            503,
            reason="backendError",
            message=f"{type(exc).__name__}: {exc}"[:400],
            service=self.service,
            method=api_method,
        )

    # -- verbs -------------------------------------------------------------- #

    async def get(self, path: str, *, api_method: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, api_method=api_method, **kwargs)

    async def post(self, path: str, *, api_method: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, api_method=api_method, **kwargs)

    async def patch(self, path: str, *, api_method: str, **kwargs: Any) -> Any:
        return await self.request("PATCH", path, api_method=api_method, **kwargs)

    async def put(self, path: str, *, api_method: str, **kwargs: Any) -> Any:
        return await self.request("PUT", path, api_method=api_method, **kwargs)

    async def delete(self, path: str, *, api_method: str, **kwargs: Any) -> Any:
        return await self.request("DELETE", path, api_method=api_method, **kwargs)

    async def paginate(
        self,
        path: str,
        *,
        api_method: str,
        items_key: str,
        params: dict[str, Any] | None = None,
        page_token: str | None = None,
        max_items: int | None = None,
        max_pages: int = 20,
        token_param: str = "pageToken",
    ) -> tuple[list[dict[str, Any]], str | None, str | None]:
        """Follow ``nextPageToken`` until the caller has enough.

        Returns ``(items, next_page_token, next_sync_token)``. Stopping early
        still returns the token, so the caller can pick the walk back up.
        """
        items: list[dict[str, Any]] = []
        token = page_token
        sync_token: str | None = None
        for _ in range(max(1, max_pages)):
            page = await self.get(
                path,
                api_method=api_method,
                params={**(params or {}), token_param: token},
            )
            batch = page.get(items_key) or []
            items.extend(batch)
            sync_token = page.get("nextSyncToken") or page.get("newStartPageToken") or sync_token
            token = page.get("nextPageToken")
            if not token:
                break
            if max_items is not None and len(items) >= max_items:
                break
        if max_items is not None:
            items = items[:max_items]
        return items, token, sync_token


def _read(response: httpx.Response, expect: Expect) -> Any:
    """The body, in the shape the caller asked for."""
    if expect == "none" or response.status_code == 204 or not response.content:
        return {} if expect == "json" else b"" if expect == "bytes" else ""
    if expect == "bytes":
        return response.content
    if expect == "text":
        return response.text
    try:
        return response.json()
    except Exception:
        # 200 with a body that is not JSON. The recorded Drive export does
        # this, and so does a real export of a Google Doc.
        return {"_text": response.text}


# --------------------------------------------------------------------------- #
# The container an op sees
# --------------------------------------------------------------------------- #


@dataclass
class GoogleClients:
    """``ctx.google`` — the three services, one token, one user.

    Build it with :meth:`from_token` when the caller already holds a decrypted
    access token, or :meth:`for_user`, which reads one through
    ``auth/token_store.py`` and refreshes it when Google says it is stale.
    """

    user_id: str
    gmail: "GmailService"
    gcal: "CalendarService"
    gdrive: "DriveService"
    tokens: TokenHolder
    share: quota.Share = "interactive"
    _owns_http: bool = field(default=False, repr=False)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_holder(
        cls,
        user_id: str,
        tokens: TokenHolder,
        *,
        share: quota.Share = "interactive",
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> "GoogleClients":
        from app.services.gcal import CalendarService
        from app.services.gdrive import DriveService
        from app.services.gmail import GmailService

        def transport(service: str) -> Transport:
            return Transport(
                service, user_id, tokens, share=share, http=http, timeout=timeout
            )

        return cls(
            user_id=user_id,
            gmail=GmailService(transport("gmail")),
            gcal=CalendarService(transport("gcal")),
            gdrive=DriveService(transport("gdrive")),
            tokens=tokens,
            share=share,
        )

    @classmethod
    def from_token(
        cls,
        user_id: str,
        access_token: str,
        *,
        refresh: TokenGetter | None = None,
        share: quota.Share = "interactive",
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> "GoogleClients":
        holder = TokenHolder(user_id, access_token, refresh)
        return cls.from_holder(user_id, holder, share=share, http=http, timeout=timeout)

    @classmethod
    def from_provider(
        cls,
        user_id: str,
        getter: TokenGetter,
        *,
        share: quota.Share = "interactive",
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> "GoogleClients":
        holder = TokenHolder(user_id, None, getter)
        return cls.from_holder(user_id, holder, share=share, http=http, timeout=timeout)

    @classmethod
    def for_user(
        cls,
        user_id: str,
        session: Any = None,
        *,
        share: quota.Share = "interactive",
        http: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> "GoogleClients":
        """Clients whose tokens come from ``auth/token_store.py``.

        Nothing is read here — the store is asked for a token on the first call
        that needs one, and again when Google says the one we have is stale.
        """
        return cls.from_provider(
            user_id,
            _token_store_getter(user_id, session),
            share=share,
            http=http,
            timeout=timeout,
        )

    # -- use ---------------------------------------------------------------- #

    @property
    def transports(self) -> dict[str, Transport]:
        return {
            "gmail": self.gmail.transport,
            "gcal": self.gcal.transport,
            "gdrive": self.gdrive.transport,
        }

    # The ops layer reaches for ``ctx.google.drive`` and ``ctx.google.calendar``
    # — the names a person would use. The mirror tables are called gdrive and
    # gcal, so both spellings have to work, and one alias is cheaper than
    # asking every caller to remember which layer it is in.
    @property
    def drive(self) -> "DriveService":
        return self.gdrive

    @property
    def calendar(self) -> "CalendarService":
        return self.gcal

    def service(self, name: str) -> Any:
        """The service object for gmail | gcal | gdrive, or their aliases."""
        table = {
            "gmail": self.gmail,
            "gcal": self.gcal,
            "calendar": self.gcal,
            "gdrive": self.gdrive,
            "drive": self.gdrive,
        }
        try:
            return table[name]
        except KeyError:
            raise AppError.internal(f"No Google service named {name!r}.") from None

    def retries(self) -> list[dict[str, Any]]:
        """Every retry any of the three has made. Goes on the node row."""
        out: list[dict[str, Any]] = []
        for transport in self.transports.values():
            out.extend(transport.attempts)
        return sorted(out, key=lambda a: a["at"])

    async def aclose(self) -> None:
        """Only closes a client this container created for itself."""
        if self._owns_http:
            await close_http()


async def _require_google_scopes(session: Any, user_id: str) -> None:
    """Refuse to build clients for a token that cannot call Google.

    The local seed stores a placeholder grant so the app can be browsed with
    data but without a Google account. Treating that as a live connection buys
    a guaranteed 401 on the first write — and the 401 revokes the token, so the
    app then reports "no workspace connected" about the account somebody was
    happily using a moment earlier.

    Failing here instead lands on `_NoGoogle`, which is the same path a genuine
    outage takes: reads answer from the mirror, and writes say plainly that
    Google is not connected.
    """
    from app.db.repositories import users as users_repo

    token = await users_repo.get_token(session, user_id)
    scopes = list(getattr(token, "scopes", None) or [])
    if not any(str(scope).startswith("https://www.googleapis.com/") for scope in scopes):
        raise AppError(
            "GOOGLE_REAUTH_REQUIRED",
            "This account has no Google connection yet.",
            http=428,
            details={"scopes": scopes[:4]},
        )


async def clients_for(
    session: Any,
    user_id: str,
    *,
    share: quota.Share = "interactive",
    http: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> GoogleClients:
    """Authorised clients for one user, using the session for the first read.

    The token is fetched now, with the session the caller already has open. A
    later refresh opens its own session on purpose: the sync tasks build their
    clients inside a ``session_scope`` block and then use them outside it, so
    holding on to that session would mean refreshing against a closed one.
    """
    await _require_google_scopes(session, user_id)
    token = await _access_token(session, user_id)

    async def refresh(*, force_refresh: bool = True) -> str:
        from app.db.session import session_scope

        async with session_scope() as fresh:
            return await _access_token(fresh, user_id, force=force_refresh)

    return GoogleClients.from_token(
        user_id, token, refresh=refresh, share=share, http=http, timeout=timeout
    )


async def _access_token(session: Any, user_id: str, *, force: bool = False) -> str:
    """One access token out of ``auth.token_store``.

    ``force`` asks for a refresh even when the stored token still looks valid,
    which is what a 401 from Google means: our clock and Google's disagree, and
    Google's is the one that counts.
    """
    from app.auth import token_store

    if force:
        refresher = getattr(token_store, "refresh_access_token", None)
        if refresher is not None:
            try:
                return await refresher(session, user_id)
            except AppError:
                raise
            except Exception as exc:  # a refresh that fails is still a reauth
                raise AppError(
                    "GOOGLE_REAUTH_REQUIRED",
                    details={"user_id": user_id, "reason": type(exc).__name__},
                ) from exc
    return await token_store.get_access_token(session, user_id)


def _token_store_getter(user_id: str, session: Any = None) -> TokenGetter:
    """A callable that asks ``auth.token_store`` for a live access token.

    The import is deferred so this module does not depend on the auth package
    being loaded, and the function is looked up by name so a small difference
    in the store's spelling is a clear error here rather than an import crash
    at start-up.
    """

    async def getter(*, force_refresh: bool = False) -> str:
        from app.auth import token_store  # local import: see docstring

        candidates = (
            "get_access_token",
            "access_token",
            "access_token_for",
            "ensure_access_token",
            "valid_access_token",
        )
        fn = next(
            (getattr(token_store, name) for name in candidates if hasattr(token_store, name)),
            None,
        )
        if fn is None:
            raise AppError.internal(
                "token_store exposes none of the expected accessors.",
                tried=list(candidates),
            )
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "force_refresh" in params:
            kwargs["force_refresh"] = force_refresh
        if session is not None and "session" in params:
            kwargs["session"] = session
        args: list[Any] = [user_id]
        if session is not None and "session" not in kwargs and len(params) >= 2:
            names = list(params)
            if names[0] in {"session", "db"}:
                args = [session, user_id]
        result = fn(*args, **kwargs)
        token = await result if inspect.isawaitable(result) else result
        if isinstance(token, str):
            return token
        # Some stores hand back the row or a small record.
        for attribute in ("access_token", "token"):
            value = getattr(token, attribute, None)
            if isinstance(value, str):
                return value
        raise AppError(
            "GOOGLE_REAUTH_REQUIRED",
            details={"user_id": user_id, "reason": "no_access_token"},
        )

    return getter


__all__ = [
    "GMAIL_BASE",
    "CALENDAR_BASE",
    "DRIVE_BASE",
    "DRIVE_UPLOAD_BASE",
    "SERVICE_BASE",
    "DEFAULT_TIMEOUT_S",
    "Expect",
    "Transport",
    "TokenHolder",
    "TokenGetter",
    "GoogleClients",
    "SAFE_TO_REPEAT",
    "clients_for",
    "get_http",
    "close_http",
    "set_http",
]
