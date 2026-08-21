"""The FastAPI application.

Owns four things and nothing else:

  1. the lifespan — build the database engine and the redis pool, tear them down
  2. CORS for the vite dev server
  3. one error shape for every failure, always carrying the request id
  4. mounting the v1 routers

Business logic lives in the routers and the orchestrator.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from types import ModuleType

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from app import __version__
from app.api.v1 import auth, conversations, events, health, prompts, query, search, sync
from app.config import settings
from app.core import cache
from app.core import logging as applog
from app.core.errors import CODES, AppError
from app.core.ids import new_id
from app.db.session import get_engine, shutdown_engine

REQUEST_ID_HEADER = "X-Request-ID"

log = applog.get_logger(__name__)

# HTTP status -> the code we report when a bare HTTPException reaches the
# boundary. Anything unlisted is INTERNAL.
_STATUS_CODES: dict[int, str] = {
    400: "VALIDATION_ERROR",
    401: "NOT_AUTHENTICATED",
    403: "NOT_AUTHENTICATED",
    404: "NOT_FOUND",
    409: "PROMPT_NOT_PENDING",
    422: "VALIDATION_ERROR",
    428: "GOOGLE_REAUTH_REQUIRED",
    429: "RATE_LIMITED",
    503: "GOOGLE_UNAVAILABLE",
    504: "ORCHESTRATION_TIMEOUT",
}


# --------------------------------------------------------------------------- helpers


def _error_body(code: str, message: str, details: dict | None, request_id: str) -> dict:
    if code not in CODES:
        code = "INTERNAL"
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid:
        return rid
    return applog.get_request_id() or request.headers.get(REQUEST_ID_HEADER) or new_id()


def _mount(app: FastAPI, module: ModuleType, prefix: str = "/api/v1") -> None:
    """Include a router, without doubling the prefix if it declares its own."""
    router = getattr(module, "router", None)
    if router is None:
        raise RuntimeError(f"{module.__name__} defines no `router`")
    declared = getattr(router, "prefix", "") or ""
    app.include_router(router, prefix="" if declared.startswith(prefix) else prefix)


# --------------------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    applog.configure()
    logging.getLogger("uvicorn.access").setLevel(settings.LOG_LEVEL)
    settings.require_production_secrets()

    # Both are lazy singletons. Touching them here means a bad DSN or an
    # unreachable redis shows up at boot, not on the first query.
    get_engine()
    await cache.get_redis()

    log.info(
        "api.started",
        env=settings.ENV,
        version=__version__,
        model=settings.OPENAI_MODEL,
        embed_model=settings.OPENAI_EMBED_MODEL,
    )
    try:
        yield
    finally:
        await cache.close()
        await shutdown_engine()
        log.info("api.stopped")


# --------------------------------------------------------------------------- app


app = FastAPI(
    title="Alpha Law orchestrator",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_dev else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.is_dev else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER, "X-Elapsed-Ms"],
    max_age=600,
)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """One id per request: on the state, in every log line, back in the header.

    An id supplied by the caller is honoured, so the browser can tie a query to
    the SSE stream it opens straight afterwards.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER)
    started = time.perf_counter()

    with applog.request_context(request_id=incoming) as request_id:
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request.unhandled",
                method=request.method,
                path=request.url.path,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers["X-Elapsed-Ms"] = str(elapsed_ms)
    return response


# --------------------------------------------------------------------------- errors


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = _request_id(request)
    log.warning(
        "request.app_error",
        code=exc.code,
        status=exc.http,
        message=exc.message,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.http,
        content=exc.to_response(request_id),
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=422,
        content=_error_body(
            "VALIDATION_ERROR",
            "That request did not make sense.",
            {"errors": _jsonable_errors(exc)},
            request_id,
        ),
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = _request_id(request)
    code = _STATUS_CODES.get(exc.status_code, "INTERNAL")
    headers = {REQUEST_ID_HEADER: request_id}
    headers.update(getattr(exc, "headers", None) or {})
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(code, str(exc.detail), None, request_id),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id(request)
    log.exception("request.failed", error=type(exc).__name__, path=request.url.path)
    details = {"cause": type(exc).__name__, "detail": str(exc)[:500]} if settings.is_dev else None
    return JSONResponse(
        status_code=500,
        content=_error_body("INTERNAL", "Something went wrong on our side.", details, request_id),
        headers={REQUEST_ID_HEADER: request_id},
    )


def _jsonable_errors(exc: RequestValidationError) -> list[dict]:
    """Pydantic errors can carry exception objects; keep only what JSON holds."""
    return [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]


# --------------------------------------------------------------------------- routes

_mount(app, health, prefix="")  # /healthz /readyz /metrics sit at the root
_mount(app, query)
_mount(app, events)
_mount(app, auth)
_mount(app, sync)
_mount(app, prompts)
_mount(app, conversations)
_mount(app, search)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"name": settings.APP_NAME, "version": __version__, "docs": "/docs"}
