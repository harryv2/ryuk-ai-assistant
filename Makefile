# Everything you need to drive the stack. `make help` lists it.
# Recipes run against docker compose; the db targets shell into the api container
# so alembic sees the same DATABASE_URL the app does.

SHELL := /bin/bash
COMPOSE ?= docker compose
API := $(COMPOSE) exec -T api
API_TTY := $(COMPOSE) exec api

.DEFAULT_GOAL := help
.PHONY: help env up up-build down stop restart ps logs logs-api logs-worker \
        build pull sh shell psql redis-cli \
        migrate revision downgrade history current stamp reset-db \
        test test-watch lint fmt typecheck check \
        worker beat ui-install clean nuke

## ---------------------------------------------------------------- meta

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if it is missing
	@test -f .env && echo ".env already exists, leaving it alone" \
	  || (cp .env.example .env && echo "wrote .env - fill in the secrets")

## ---------------------------------------------------------------- lifecycle

up: env ## Start the whole stack in the background
	$(COMPOSE) up -d

up-build: env ## Rebuild images, then start
	$(COMPOSE) up -d --build

build: ## Build the backend image
	$(COMPOSE) build

pull: ## Pull the third-party images
	$(COMPOSE) pull postgres redis ui

down: ## Stop and remove containers (volumes survive)
	$(COMPOSE) down --remove-orphans

stop: ## Stop containers without removing them
	$(COMPOSE) stop

restart: ## Restart api, worker and beat
	$(COMPOSE) restart api worker beat

ps: ## Show container status
	$(COMPOSE) ps

## ---------------------------------------------------------------- logs and shells

logs: ## Tail every service
	$(COMPOSE) logs -f --tail=100

logs-api: ## Tail the api
	$(COMPOSE) logs -f --tail=200 api

jobs: ## Are worker, beat, queues and sync all healthy?
	cd backend && .venv/bin/python scripts/jobs_status.py

logs-worker: ## Tail the celery worker
	$(COMPOSE) logs -f --tail=200 worker

sh: shell
shell: ## Bash inside the api container
	$(API_TTY) bash

psql: ## psql on the app database
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-postgres} -d $${POSTGRES_DB:-orchestrator}

redis-cli: ## redis-cli on the broker
	$(COMPOSE) exec redis redis-cli

## ---------------------------------------------------------------- migrations

migrate: ## Apply every migration
	$(API) alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add foo"
	@test -n "$(m)" || (echo 'usage: make revision m="what changed"'; exit 1)
	$(API) alembic revision --autogenerate -m "$(m)"

downgrade: ## Step back one migration
	$(API) alembic downgrade -1

history: ## Show the migration history
	$(API) alembic history --indicate-current

current: ## Show the revision the database is on
	$(API) alembic current

stamp: ## Mark the database as being at head without running anything
	$(API) alembic stamp head

reset-db: ## Drop the database volume and migrate from scratch (destructive)
	$(COMPOSE) down -v
	$(COMPOSE) up -d postgres redis
	@echo "waiting for postgres..."
	@until $(COMPOSE) exec -T postgres pg_isready -U $${POSTGRES_USER:-postgres} >/dev/null 2>&1; do sleep 1; done
	$(COMPOSE) up -d api
	$(API) alembic upgrade head

## ------------------------------------------------------------------- seed

eval: ## Run the evaluation harness and regenerate tests/eval/RESULTS.md
	docker compose exec -T api python -m tests.eval.intent_accuracy
	docker compose exec -T api python -m tests.eval.precision_at_k
	docker compose exec -T api python -m tests.eval.latency
	docker compose exec -T api python -m tests.eval.precision_at_k --write-results

seed: ## Load the demo account and its mail, calendar and files
	$(API) python scripts/seed_local.py

seed-clear: ## Remove the demo data (the account itself survives)
	$(API) python scripts/seed_local.py --clear

seed-reset: ## Clear the demo data and load it again
	$(API) python scripts/seed_local.py --clean

seed-preview: ## Print what would be seeded, without writing anything
	$(API) python scripts/seed_local.py --dry-run

## ---------------------------------------------------------------- quality

test: ## Run the backend test suite
	$(API) pytest -q

test-watch: ## Run one test file: make test-watch k=test_temporal
	$(API) pytest -q -k "$(k)"

lint: ## Ruff check
	$(API) ruff check app alembic

fmt: ## Ruff format and fix imports
	$(API) ruff check --fix app alembic
	$(API) ruff format app alembic

typecheck: ## mypy over the app package
	$(API) mypy app

check: lint typecheck test ## Lint, typecheck, test

## ---------------------------------------------------------------- misc

worker: ## Run a foreground worker (handy for pdb)
	$(COMPOSE) run --rm --service-ports worker celery -A app.tasks.celery_app worker -l DEBUG \
	  -Q sync,embed,actions,orchestration,maintenance --concurrency=1

beat: ## Run beat in the foreground
	$(COMPOSE) run --rm beat celery -A app.tasks.celery_app beat -l DEBUG

ui-install: ## Reinstall the frontend dependencies
	$(COMPOSE) run --rm ui npm install

clean: ## Remove python and build junk from the working tree
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache frontend/dist

nuke: ## Containers, volumes and images, all of it (destructive)
	$(COMPOSE) down -v --rmi local --remove-orphans
