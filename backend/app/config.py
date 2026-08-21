"""Every knob the system has, read once from the environment.

Import the singleton:

    from app.config import settings
    settings.DATABASE_URL

Names are upper case and match the environment variable exactly, so there is
never a question of what a value is called in .env versus in code.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any, Literal

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The dimension of text-embedding-3-small, and of every VECTOR column in the
# schema. Changing the embed model means a migration, so it lives here as a
# constant rather than a setting anyone can nudge.
EMBED_DIM = 1536

# How hard Claude is allowed to think, cheapest first. Spelled out here rather
# than imported from app.llm.providers.anthropic_provider on purpose: importing
# that module pulls in the Anthropic SDK, and config is imported by everything.
ANTHROPIC_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Every prefix that means Anthropic in a "<provider>:<model>" string. Kept in
# step with PROVIDER_ALIASES in app/llm/base.py, and duplicated for the same
# reason as above — config must not import the llm package to validate itself.
ANTHROPIC_PREFIXES: frozenset[str] = frozenset({"anthropic", "claude"})

# The scopes the consent screen actually asks for.
#
# `app/auth/google_oauth.py` builds the authorisation URL and is the authority
# on this list; it is repeated here, rather than imported, for the same reason
# as the two tuples above — config must not import the packages that import it.
# The two lists are asserted equal at the bottom of this module, so a scope
# added in one place cannot quietly disagree with the other and leave whoever
# is configuring Google Cloud reading the wrong one.
#
# Narrow on purpose: `drive.readonly` + `drive.file` instead of full `drive`,
# so the app can read what it is shown and touch only the files it made.
GOOGLE_SCOPES: tuple[str, ...] = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    ENV: Literal["development", "test", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    APP_NAME: str = "alpha-law"
    APP_URL: str = "http://localhost:5173"

    # ------------------------------------------------------------------ stores
    # These names match the environment variables app/db/session.py reads when
    # it builds the engine. Keep the two in step.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/orchestrator"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 10
    DB_POOL_RECYCLE: int = 1800
    SQL_ECHO: bool = False

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50

    # ------------------------------------------------------------------ openai
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.6-terra"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    # Sent to the reasoning families (o-series, gpt-5*) only. "low" keeps the
    # planner's structured-output call fast; blank sends nothing and lets the
    # model use its own default depth.
    OPENAI_REASONING_EFFORT: str = "low"
    OPENAI_TIMEOUT_S: float = 20.0
    OPENAI_MAX_RETRIES: int = 2

    # ------------------------------------------------------------------ anthropic
    # Chat only. Anthropic has no embeddings endpoint, so EMBED_MODEL can never
    # point here — see the check on EMBED_MODEL below.
    ANTHROPIC_API_KEY: str = ""

    # How hard Claude thinks before it answers: low, medium, high, xhigh, max.
    # Anthropic's own default is high; ours is medium, because the planner call
    # sits inside SOFT_DEADLINE_MS and effort is the cheapest way to buy latency
    # back. There is no temperature to turn down on these models — it was
    # removed and now returns a 400.
    ANTHROPIC_EFFORT: str = "medium"

    # Extended thinking. False sends {"type": "disabled"} rather than leaving
    # the parameter out, because on claude-opus-5 leaving it out means adaptive
    # thinking — "not set" is on, not off. True sends {"type": "adaptive"},
    # which is the same thing the API does when the parameter is absent.
    # Never budget_tokens: that knob was removed and is now a 400.
    ANTHROPIC_THINKING: bool = False

    # ------------------------------------------------------------------ llm
    # A model string is "<provider>:<name>" — "openai:gpt-5.6-terra",
    # "gemini:gemini-2.5-flash", "anthropic:claude-opus-5". The prefix is the
    # only thing that picks a provider, and a bare name belongs to
    # LLM_DEFAULT_PROVIDER. Moving the app from one vendor to another is these
    # lines and a restart: no call site anywhere names a provider. See app/llm/
    # and docs/MODELS.md.
    LLM_MODEL: str = "openai:gpt-5.6-terra"
    LLM_FALLBACK_MODELS: str = "gemini:gemini-3.5-flash"
    LLM_PROSE_MODEL: str = ""
    LLM_DEFAULT_PROVIDER: str = "openai"
    LLM_TIMEOUT_S: float = 20.0
    # Failure classes where another provider is worth trying. Everything else —
    # INVALID, CONTENT_FILTERED, AUTH_* — fails on the spot, because a request
    # the first model could not read is a request the second one cannot either.
    LLM_FALLBACK_ON: str = "RATE_LIMITED,TRANSIENT,QUOTA_EXHAUSTED,TIMEOUT"

    # The Gemini API key. Not the OAuth client below — that is for the user's
    # Gmail and Calendar, this is for the model.
    GOOGLE_AI_API_KEY: str = ""

    #: Every emailed code is 123456. There is no mailer wired up, so a real
    #: random code would be a door nobody can open. Turn this off the day a
    #: mailer exists — app.auth.otp.send_code is the only place to change.
    OTP_DEV_MODE: bool = True

    # ------------------------------------------------------------------ provider knobs
    # Optional, and every one of them has a working default inside its adapter.
    # Declared here so `.env` lines with these names actually reach the code:
    # Settings has extra="ignore", and the adapters fall back to reading the raw
    # environment, so a name that is not a field here works but only by that
    # second route. One place to look beats two.

    # For an OpenAI-compatible gateway or a proxy. Empty means api.openai.com.
    OPENAI_BASE_URL: str = ""

    # How the Gemini adapter talks: "auto" uses the google-genai SDK when it is
    # importable and REST over httpx when it is not; "sdk" and "rest" pin it.
    GEMINI_TRANSPORT: str = "auto"

    # Tokens Gemini may spend thinking before it answers. On 2.5 models thoughts
    # and answer share one output budget, so a non-zero value here eats into
    # max_tokens. 0 turns thinking off, which gemini-2.5-pro does not allow —
    # the adapter leaves the parameter off entirely for -pro.
    GEMINI_THINKING_BUDGET: int = 0

    # What Gemini is told the embedding is for. Part of what makes two vectors
    # comparable, exactly like the model name: change it and the stored vectors
    # stop matching new ones, with no error. Treat it as fixed once the mirror
    # has rows, and re-embed if it has to move.
    GEMINI_EMBED_TASK_TYPE: str = "SEMANTIC_SIMILARITY"

    # ------------------------------------------------------------------ embeddings
    # Chat models are swappable. Embedding models are NOT. Every VECTOR column
    # was filled by one specific model, and a vector from another model is not
    # comparable with them even at the same width — the search still returns
    # rows, they are just the wrong rows. So each mirror row records the model
    # that embedded it in `embed_model`, and changing EMBED_MODEL means
    # re-embedding the whole mirror. See app/llm/__init__.py.
    EMBED_MODEL: str = "openai:text-embedding-3-small"
    EMBED_DIMENSIONS: int = EMBED_DIM

    # ------------------------------------------------------------------ google
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    # ------------------------------------------------------------------ secrets
    # 32 raw bytes, urlsafe-base64 encoded. AES-256-GCM for the oauth token blobs.
    TOKEN_ENCRYPTION_KEY: str = ""
    TOKEN_KEY_VERSION: int = 1
    SESSION_SECRET: str = ""
    SESSION_COOKIE_NAME: str = "alpha_session"
    SESSION_MAX_AGE_S: int = 60 * 60 * 24 * 30

    # ------------------------------------------------------------------ budgets
    RATE_LIMIT_PER_HOUR: int = 100
    SOFT_DEADLINE_MS: int = 2500
    HARD_DEADLINE_MS: int = 12000
    MAX_LLM_CALLS_PER_RUN: int = 5

    # ------------------------------------------------------------------ retrieval
    # Below FLOOR_READ a candidate is not worth showing the planner. A top hit
    # that beats the runner-up by less than MARGIN is ambiguous, so we ask.
    # A write may only ever target something above FLOOR_WRITE.
    FLOOR_READ: float = 0.55
    MARGIN: float = 0.15
    FLOOR_WRITE: float = 0.80

    # ------------------------------------------------------------------ sync
    SYNC_INTERVAL_MIN: int = 15
    SYNC_BACKFILL_DAYS: int = 180
    SYNC_PAGE_SIZE: int = 100

    # ------------------------------------------------------------------ misc
    PROMPT_TTL_MIN: int = 60 * 24
    CORS_ORIGINS: str = ""

    # ------------------------------------------------------------------ checks
    @field_validator("FLOOR_READ", "MARGIN", "FLOOR_WRITE")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v

    @field_validator("EMBED_DIMENSIONS")
    @classmethod
    def _embed_dimensions(cls, v: int) -> int:
        """Refuse to start rather than write vectors pgvector will reject.

        EMBED_DIM is the width of every VECTOR column in the schema. Changing
        this without a migration produces rows the database cannot store, and
        changing it *with* one still means re-embedding the whole mirror.
        """
        if v != EMBED_DIM:
            raise ValueError(
                f"EMBED_DIMENSIONS is {v} but every VECTOR column is {EMBED_DIM}. "
                f"Changing it needs a migration and a full re-embed of the mirror."
            )
        return v

    @field_validator("EMBED_MODEL")
    @classmethod
    def _embed_model_can_embed(cls, v: str, info: ValidationInfo) -> str:
        """Refuse to start rather than discover at the first search that
        Anthropic cannot embed.

        Anthropic has no embeddings endpoint. Not a missing feature in our
        adapter, not something a newer SDK adds — the API does not have one. So
        `EMBED_MODEL=anthropic:...` is a configuration that can never work, and
        the only question is whether it fails now, at startup, with this
        message, or in twenty minutes when the first user searches and the
        sync worker has already written nothing.

        A bare name with no prefix belongs to LLM_DEFAULT_PROVIDER, so that
        spelling is checked too. LLM_DEFAULT_PROVIDER is declared above this
        field, which is why `info.data` already has it.
        """
        raw = (v or "").strip()
        if not raw:
            return v

        if ":" in raw:
            prefix = raw.partition(":")[0].strip().lower()
            where = f"EMBED_MODEL={raw}"
        else:
            prefix = str(info.data.get("LLM_DEFAULT_PROVIDER", "openai")).strip().lower()
            where = f"EMBED_MODEL={raw} with LLM_DEFAULT_PROVIDER={prefix}"

        if prefix in ANTHROPIC_PREFIXES:
            raise ValueError(
                f"{where} asks Anthropic for embeddings, and Anthropic has no embeddings "
                f"API at all. Point EMBED_MODEL at openai:text-embedding-3-small or "
                f"gemini:gemini-embedding-001. Chat can still run on Anthropic — set "
                f"LLM_MODEL=anthropic:claude-opus-5 and leave EMBED_MODEL where it is. "
                f"Note that changing EMBED_MODEL on a mirror that already has vectors "
                f"means re-embedding all of it: vectors from two models are not comparable."
            )
        return v

    @field_validator("ANTHROPIC_EFFORT")
    @classmethod
    def _anthropic_effort(cls, v: str) -> str:
        """One of five words. A typo here would otherwise be silently ignored
        and paid for on every call at whatever effort the provider defaults to."""
        effort = (v or "").strip().lower()
        if effort not in ANTHROPIC_EFFORTS:
            raise ValueError(
                f"ANTHROPIC_EFFORT is {v!r}; it must be one of {', '.join(ANTHROPIC_EFFORTS)}."
            )
        return effort

    @field_validator("ANTHROPIC_THINKING", mode="before")
    @classmethod
    def _anthropic_thinking(cls, v: Any) -> Any:
        """Accept the words the Anthropic adapter documents, not just true/false.

        `ANTHROPIC_THINKING=adaptive` and `ANTHROPIC_THINKING=default` both mean
        thinking is on: omitting the parameter on claude-opus-5 runs adaptive,
        so "leave it at the default" and "adaptive" are the same request. Anything
        pydantic already understands — true/false, on/off, yes/no, 1/0 — is left
        for it to parse.
        """
        if not isinstance(v, str):
            return v
        word = v.strip().lower()
        if word in ("adaptive", "default", "auto", "unset"):
            return True
        if word == "disabled":
            return False
        return v

    @field_validator("GEMINI_TRANSPORT")
    @classmethod
    def _gemini_transport(cls, v: str) -> str:
        """One of three words. A typo would otherwise be read as "auto" and the
        pin the operator asked for would silently not happen."""
        word = (v or "").strip().lower() or "auto"
        if word not in ("auto", "sdk", "rest"):
            raise ValueError(f"GEMINI_TRANSPORT is {v!r}; it must be auto, sdk or rest.")
        return word

    @field_validator("LOG_LEVEL")
    @classmethod
    def _log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level: {v}")
        return level

    # ------------------------------------------------------------------ derived
    @property
    def is_dev(self) -> bool:
        return self.ENV in ("development", "test")

    @property
    def sync_database_url(self) -> str:
        """The same database, over psycopg. Alembic and any sync tooling use this."""
        url = self.DATABASE_URL
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "+psycopg", 1)
        if url.startswith("postgresql+"):
            return url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

    @property
    def async_database_url(self) -> str:
        """The same database, over asyncpg. What the app runs on."""
        url = self.DATABASE_URL
        if "+asyncpg" in url:
            return url
        for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix) :]
        return url

    @property
    def celery_broker_url(self) -> str:
        return self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        return self.REDIS_URL

    @property
    def cors_origins(self) -> list[str]:
        """The vite dev server, plus anything the operator adds, comma separated."""
        origins = [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        for default in (self.APP_URL, "http://localhost:5173", "http://127.0.0.1:5173"):
            if default and default not in origins:
                origins.append(default)
        return origins

    @property
    def google_scopes(self) -> list[str]:
        return list(GOOGLE_SCOPES)

    @property
    def token_encryption_key(self) -> bytes:
        """The 32 raw bytes behind TOKEN_ENCRYPTION_KEY.

        Accepts urlsafe-base64 (what .env.example tells you to generate),
        standard base64, or 64 hex characters.
        """
        raw = self.TOKEN_ENCRYPTION_KEY.strip()
        if not raw:
            raise ValueError("TOKEN_ENCRYPTION_KEY is not set")

        padded = raw + "=" * (-len(raw) % 4)
        for decode in (base64.urlsafe_b64decode, base64.b64decode):
            try:
                key = decode(padded)
            except Exception:
                continue  # not this encoding, try the next one
            if len(key) == 32:
                return key

        try:
            key = bytes.fromhex(raw)
        except ValueError:
            key = b""
        if len(key) == 32:
            return key

        raise ValueError(
            "TOKEN_ENCRYPTION_KEY must decode to 32 bytes (urlsafe-base64, base64, or 64 hex chars)"
        )

    @property
    def uses_anthropic(self) -> bool:
        """True when any configured chat model is served by Anthropic.

        Reads the prefixes only. This is deliberately not `ModelRef.parse`:
        config is imported by everything and must not drag the llm package —
        and through it an SDK — onto the import path just to answer this.
        """
        specs = f"{self.LLM_MODEL},{self.LLM_FALLBACK_MODELS},{self.LLM_PROSE_MODEL}"
        for spec in specs.split(","):
            name = spec.strip()
            if not name:
                continue
            prefix = name.partition(":")[0].strip().lower() if ":" in name else ""
            if not prefix:
                prefix = self.LLM_DEFAULT_PROVIDER.strip().lower()
            if prefix in ANTHROPIC_PREFIXES:
                return True
        return False

    def require_production_secrets(self) -> None:
        """Refuse to serve real traffic with placeholder secrets."""
        if self.is_dev:
            return
        required = [
            "OPENAI_API_KEY",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
            "TOKEN_ENCRYPTION_KEY",
            "SESSION_SECRET",
        ]
        # Only when something is actually pointed at Anthropic. A deployment
        # that never names it has no reason to hold a key for it.
        if self.uses_anthropic:
            required.append("ANTHROPIC_API_KEY")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise RuntimeError(f"missing required settings: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor, for use as a FastAPI dependency."""
    return Settings()


settings: Settings = get_settings()

__all__ = [
    "ANTHROPIC_EFFORTS",
    "ANTHROPIC_PREFIXES",
    "EMBED_DIM",
    "GOOGLE_SCOPES",
    "Settings",
    "get_settings",
    "settings",
]


def _assert_scopes_agree() -> None:
    """Fail loudly if the two copies of the scope list drift apart.

    Imported lazily and swallowed on ImportError: `app.auth` imports config, so
    this cannot run at config import time, and a partially-installed checkout
    should not be unable to read its own settings.
    """
    try:
        from app.auth.google_oauth import SCOPES
    except Exception:  # pragma: no cover - the auth package is not importable yet
        return
    if sorted(SCOPES) != sorted(GOOGLE_SCOPES):
        raise RuntimeError(
            "app.config.GOOGLE_SCOPES and app.auth.google_oauth.SCOPES disagree. "
            "The consent screen is built from the latter; keep them identical.\n"
            f"  config only: {sorted(set(GOOGLE_SCOPES) - set(SCOPES))}\n"
            f"  oauth only:  {sorted(set(SCOPES) - set(GOOGLE_SCOPES))}"
        )
