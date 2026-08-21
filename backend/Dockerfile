FROM python:3.12-slim-bookworm

# Runtime bits: libpq for psycopg, curl for poking at things from inside the
# container, tini so celery and uvicorn get their signals.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        libpq-dev \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# The source is bind-mounted in development, so an editable install keeps the
# container and the working tree pointing at the same files.
COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY alembic ./alembic

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev]"

# Non-root. Owns /app because the editable install writes there.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
