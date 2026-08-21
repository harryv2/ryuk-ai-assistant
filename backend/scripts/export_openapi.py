"""Dump the OpenAPI spec to `docs/openapi.json` without touching a database.

Run it from the `backend` directory:

    python -m scripts.export_openapi

Or anywhere, with flags:

    python -m scripts.export_openapi --out /tmp/openapi.json
    python -m scripts.export_openapi --check        # CI: fail if the file is stale

Why this exists rather than `curl localhost:8000/openapi.json`:

  * CI has no Postgres and no Redis, and should still be able to prove the spec
    on disk matches the code.
  * `app.openapi()` is pure schema generation. It reads the route table and the
    Pydantic models and nothing else. The only thing standing between an import
    and a spec is the lifespan, which opens the engine and the Redis pool — so
    we replace it with a no-op before anything can run it.

The lifespan is stubbed in two places on purpose. `app.main.lifespan` is swapped
so a later import gets the harmless one, and `app.router.lifespan_context` is
swapped because the application object was already built with the real one bound
to it. Belt and braces: `app.openapi()` never enters the lifespan, but a future
edit that reaches for `TestClient` in here would, and that should stay safe.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# scripts/ -> backend/ -> repo root
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "openapi.json"

# Enough environment to import `app.config` cleanly on a machine with no .env.
# None of these are used — nothing connects — but pydantic-settings validates on
# construction and a bad LOG_LEVEL or a missing DSN would fail the import for no
# good reason. `ENV=development` also keeps `openapi_url` mounted, which is what
# the served spec looks like in dev.
SAFE_ENV: dict[str, str] = {
    "ENV": "development",
    "LOG_LEVEL": "WARNING",
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@localhost:5432/orchestrator",
    "REDIS_URL": "redis://localhost:6379/0",
    "OPENAI_API_KEY": "sk-not-used-during-spec-export",
    "GOOGLE_CLIENT_ID": "spec-export",
    "GOOGLE_CLIENT_SECRET": "spec-export",
    # 32 bytes, urlsafe-base64. A throwaway; it never encrypts anything here.
    "TOKEN_ENCRYPTION_KEY": "c3BlYy1leHBvcnQtb25seS1kby1ub3QtdXNlLTMyYnk=",
    "SESSION_SECRET": "spec-export-only",
}


def _prime_environment() -> None:
    """Fill in only what is missing. A real .env or a real env var always wins."""
    for key, value in SAFE_ENV.items():
        os.environ.setdefault(key, value)


def _ensure_importable() -> None:
    """Make `import app` work regardless of the working directory."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))


@contextlib.asynccontextmanager
async def _null_lifespan(_app: Any) -> AsyncIterator[None]:
    """Start and stop nothing."""
    yield


def load_app() -> Any:
    """Import `app.main` with the lifespan neutered, and hand back the app."""
    _prime_environment()
    _ensure_importable()

    try:
        from app import main as main_module
    except ImportError as exc:  # a router that is still being written
        raise SystemExit(
            f"Could not import app.main: {exc}\n"
            "The backend is incomplete or a dependency is missing. "
            f"Try `pip install -e {BACKEND_DIR}`."
        ) from exc

    main_module.lifespan = _null_lifespan  # type: ignore[assignment]
    application = main_module.app
    application.router.lifespan_context = _null_lifespan
    return application


def build_spec(application: Any) -> dict[str, Any]:
    """`app.openapi()`, plus the metadata FastAPI cannot know on its own."""
    # FastAPI memoises the spec on the app. Clear it so repeated calls in one
    # process (tests, a --check followed by a write) regenerate honestly.
    application.openapi_schema = None
    spec: dict[str, Any] = application.openapi()

    info = spec.setdefault("info", {})
    info["title"] = "Alpha Law orchestrator"
    info["description"] = (
        "Natural-language orchestration over Gmail, Calendar and Drive.\n\n"
        "Writes are two-phase: a query only ever *prepares* an action and the "
        "prompt that gates it. Nothing reaches Google until a person approves "
        "it through `POST /api/v1/prompts/{id}/respond`.\n\n"
        "Prose documentation, including the SSE event union and the seq "
        "gap-detection contract, is in `docs/API.md`."
    )
    info.setdefault("contact", {"name": "Alpha Law orchestrator", "url": "https://example.invalid"})
    info.setdefault("license", {"name": "Proprietary"})

    spec["servers"] = [
        {"url": "http://localhost:8000", "description": "local development"},
        {"url": "{scheme}://{host}", "description": "anywhere else",
         "variables": {
             "scheme": {"default": "https", "enum": ["https", "http"]},
             "host": {"default": "localhost:8000"},
         }},
    ]

    spec.setdefault("tags", [
        {"name": "query", "description": "Ask a question. The one entry point."},
        {"name": "runs", "description": "Server-sent events for a single run."},
        {"name": "prompts", "description": "Answer or dismiss what the system asked for."},
        {"name": "conversations", "description": "The durable record of a thread."},
        {"name": "auth", "description": "Google OAuth, session, disconnect."},
        {"name": "sync", "description": "The pgvector mirror of the three services."},
        {"name": "search", "description": "Retrieval with the lid off, for debugging and eval."},
        {"name": "health", "description": "Liveness, readiness, metrics."},
    ])

    _add_session_security(spec)
    _document_error_envelope(spec)
    return spec


def _add_session_security(spec: dict[str, Any]) -> None:
    """The session cookie, declared once and applied to everything but auth start."""
    components = spec.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    schemes.setdefault(
        "sessionCookie",
        {
            "type": "apiKey",
            "in": "cookie",
            "name": os.environ.get("SESSION_COOKIE_NAME", "alpha_session"),
            "description": (
                "Signed session cookie, set by GET /api/v1/auth/google/callback. "
                "Google tokens never leave the server."
            ),
        },
    )
    spec.setdefault("security", [{"sessionCookie": []}])

    # The two endpoints that must work without a session.
    for path in ("/api/v1/auth/google", "/api/v1/auth/google/callback"):
        operations = spec.get("paths", {}).get(path, {})
        for method, operation in operations.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                operation["security"] = []


def _document_error_envelope(spec: dict[str, Any]) -> None:
    """One error shape for every failure, named so operations can point at it."""
    schemas = spec.setdefault("components", {}).setdefault("schemas", {})
    schemas.setdefault(
        "ErrorEnvelope",
        {
            "type": "object",
            "required": ["error"],
            "properties": {
                "error": {
                    "type": "object",
                    "required": ["code", "message", "details", "request_id"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Closed set. See docs/API.md.",
                            "enum": _error_codes(),
                        },
                        "message": {
                            "type": "string",
                            "description": "Safe to show a person.",
                        },
                        "details": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": "Developer detail. Shape varies by code.",
                        },
                        "request_id": {
                            "type": "string",
                            "description": "Echoes X-Request-ID, or the one the server generated.",
                        },
                    },
                }
            },
            "example": {
                "error": {
                    "code": "PROMPT_VALUE_INVALID",
                    "message": "That answer does not fit what was asked.",
                    "details": {"errors": [{"path": "$.approve", "msg": "'yes' is not of type 'boolean'"}]},
                    "request_id": "NKgjLgOI3xAxLtdLLB0pX",
                }
            },
        },
    )


def _error_codes() -> list[str]:
    """The code table, read from the source of truth rather than retyped."""
    try:
        from app.core.errors import CODES

        return sorted(CODES)
    except Exception:  # pragma: no cover - errors.py is owned elsewhere
        return [
            "GOOGLE_REAUTH_REQUIRED",
            "GOOGLE_UNAVAILABLE",
            "INTERNAL",
            "NOT_AUTHENTICATED",
            "NOT_FOUND",
            "ORCHESTRATION_TIMEOUT",
            "PROMPT_NOT_PENDING",
            "PROMPT_VALUE_INVALID",
            "RATE_LIMITED",
            "VALIDATION_ERROR",
        ]


def render(spec: dict[str, Any], indent: int) -> str:
    """Deterministic bytes, so `--check` compares content and not formatting."""
    return json.dumps(spec, indent=indent, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_openapi",
        description="Write the OpenAPI spec to docs/openapi.json. No database needed.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit 1 if the file on disk differs from the generated spec.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = build_spec(load_app())
    text = render(spec, args.indent)

    paths = spec.get("paths", {})
    operations = sum(
        1
        for methods in paths.values()
        for method in methods
        if method in {"get", "post", "put", "patch", "delete"}
    )

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist. Run without --check.", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(
                f"{args.out} is out of date. Run `python -m scripts.export_openapi`.",
                file=sys.stderr,
            )
            return 1
        if not args.quiet:
            print(f"{args.out} is up to date ({len(paths)} paths, {operations} operations).")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    if not args.quiet:
        print(
            f"Wrote {args.out} — OpenAPI {spec.get('openapi', '?')}, "
            f"{len(paths)} paths, {operations} operations, {len(text)} bytes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
