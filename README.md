# Ryuk — a chat orchestrator over Gmail, Calendar and Drive

You type "cancel my Turkish Airlines flight". It finds the booking email, reads
the PNR out of it, finds the matching calendar event, drafts the cancellation
with the real reference in it, and shows you the draft with a Send button. It
does not send it. Nothing that leaves your account happens without you saying
yes.

The point of the design is that it does all of that on **one model call**.

## Architecture

A turn goes through six stages, and most of them cost nothing. The **front
door** is pure Python: is this answering an open card, a UI verb, chit-chat, or
a shape the rule router already knows? Any of those and the turn finishes with
zero model calls. Otherwise a **pre-pass** resolves dates in Python — never in
the model — pulls out literals, and expands vendor aliases (`Turkish Airlines`
→ `Türk Hava Yolları`, `THY`, `TK`, `thy.com`). Then the **probe** spends one
embedding and fires three hybrid searches in parallel across our own pgvector
mirror of Gmail, Calendar and Drive, roughly 110 ms, still no model call;
regexes run over the candidate excerpts to pull out PNRs, flight numbers, dates
and links. Only now does the **planner** run: one call, which sees those
candidates and returns an intent plus a DAG of steps. It never retypes a value
— it references facts by path, `{{search.gmail[0].extracted.pnr}}`, so a
booking reference in a draft email is one the model read rather than one it
invented. The plan is **validated in pure Python** and then **dispatched**: one
asyncio task per step, each awaiting only its own dependencies, with
per-service semaphores so one fan-out cannot take the whole Google quota.
Finally it **renders** — a template or a card for free, or one streamed prose
call.

Two consequences fall out of that shape. Ambiguity is a *step* (`ask.user`), so
pausing on a question and resuming later costs zero model calls. And writes are
only ever **prepared**: a `pending_inputs` row and an `actions` row gate them,
approving is a database transition rather than a thinking task, and the actions
worker performs the irreversible half afterwards.

```
frontend/          Vite + React + TypeScript. Chat, with the steps inline.
backend/app/
  core/            ids, crypto, errors, cache, rate limiting, audit, logging
  db/              SQLAlchemy models + repositories over 15 tables
  auth/            Google OAuth, encrypted token store, request dependencies
  google/          the httpx transport: quota, retries, circuit breaker
  services/        thin Gmail / Calendar / Drive wrappers
  search/          chunking, embeddings, hybrid search, the probe, extractors
  ops/             the step catalogue the planner picks from
  orchestrator/    front door, pre-pass, planner, validate, dispatch, render
  tasks/           Celery: sync, embed, actions, maintenance
  api/v1/          the HTTP surface
```

`docs/schema.md` and `docs/contracts.md` are the frozen contract — the tables,
the module layout, the interfaces and the event names. Read those before
changing anything structural.

## Prerequisites

- Docker and Docker Compose — everything runs through `docker compose up`
- A Google Cloud project with OAuth credentials (below)
- An OpenAI API key (or Anthropic or Gemini — see `docs/MODELS.md`; the model
  string is `provider:name` and no call site names a provider)

Working on the backend outside Docker also needs Python 3.12 and a Postgres
with the `pgvector` extension.

## Google Cloud setup

1. Create a project at <https://console.cloud.google.com/>.
2. **Enable three APIs** under *APIs & Services → Library*: Gmail API, Google
   Calendar API, Google Drive API.
3. **Configure the OAuth consent screen**. Choose **External** and leave the
   publishing status as **Testing**. Under *Test users*, add every Google
   account you intend to sign in with — in test mode Google refuses anyone not
   on that list, and the error it gives is not obvious.
4. **Add the scopes** below to the consent screen. They must match what the app
   asks for, or the callback rejects the grant as incomplete.
5. **Create credentials → OAuth client ID → Web application.** Add
   `http://localhost:5173/api/v1/auth/google/callback` as an authorised
   redirect URI — port **5173**, not 8000. The browser only ever talks to Vite,
   which proxies `/api` through to the API container; the API port is not
   published at all. Copy the client ID and secret into `.env` as
   `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.

### Scopes

| Scope | Why |
|---|---|
| `openid`, `email`, `profile` | who you are, and your calendar timezone |
| `https://www.googleapis.com/auth/gmail.readonly` | mirror your mail for search |
| `https://www.googleapis.com/auth/gmail.compose` | create drafts |
| `https://www.googleapis.com/auth/gmail.send` | send an approved draft |
| `https://www.googleapis.com/auth/gmail.modify` | archive, label |
| `https://www.googleapis.com/auth/calendar` | read and change events |
| `https://www.googleapis.com/auth/drive.readonly` | mirror file text for search |
| `https://www.googleapis.com/auth/drive.file` | touch only files this app made |

Drive is deliberately split: `drive.readonly` to read what you already have,
`drive.file` to write, so the app can never modify a document it did not
create. The authoritative list is `SCOPES` in
`backend/app/auth/google_oauth.py`; `app/config.py` keeps a copy and asserts
the two are identical.

> ### ⚠ Test-mode refresh tokens expire after 7 days
>
> While the consent screen is in **Testing**, Google expires every refresh
> token **7 days** after it is issued. The symptom is that everything works,
> you come back the following week, and every Google call fails with
> `GOOGLE_REAUTH_REQUIRED` — the app will ask you to reconnect, and signing in
> again fixes it for another 7 days. Nothing is wrong with your setup.
>
> The only real fix is to publish the consent screen to **In production**,
> which for these scopes means going through Google's verification. For a demo
> or a take-home, reconnecting weekly is the expected cost.

## Running it

```bash
cp .env.example .env      # or: make env
```

Fill in `.env`:

- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` from the step above
- `OPENAI_API_KEY`
- `SESSION_SECRET` — any long random string
- `TOKEN_ENCRYPTION_KEY` — 32 bytes, base64. Generate one with:

  ```bash
  python3 -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
  ```

  This is the AES-256-GCM key for the OAuth tokens at rest. Lose it and every
  stored token becomes unreadable; everyone has to sign in again.

Then:

```bash
make up            # docker compose up -d, builds on first run
make migrate       # apply the migrations
make logs          # tail everything
```

- UI: <http://localhost:5173> — **the only port the browser needs**
- API: reachable through the UI origin, e.g. <http://localhost:5173/api/v1/auth/me>.
  The container exposes 8000 to the compose network but does not publish it.
- OpenAPI: <http://localhost:5173/docs>; health at `/healthz`, `/readyz`, `/metrics`

Sign in at the UI, two ways:

- **Continue with Google** — signs you in *and* connects the workspace in one
  step. This is the one to use first.
- **Email and password**, with a 6-digit code. In development the code is
  always `123456` and no mail is sent. An account made this way starts with no
  workspace; connect one from **Integrations** in the account menu, and it can
  be any Google account — it does not have to match the address you signed up
  with.

The first connection queues a backfill of Gmail, Calendar and Drive into the
mirror. **The Celery worker does that**, so it has to be running — `make up`
starts it. Progress is under **Your information** in the account menu.
Searching before the backfill finishes works, it just has less to find.

Two Celery processes matter, and they are not the same thing:

- the **worker** executes tasks (`celery -A app.tasks.celery_app worker -Q
  sync,embed,actions,orchestration,maintenance`);
- **beat** puts the periodic ones on the queue (`celery -A
  app.tasks.celery_app beat`) — the every-15-minutes sync, token refresh, and
  the maintenance sweeps.

`make up` runs both. If you start processes by hand and skip beat, everything
appears to work: the first backfill runs (the connect flow queues it
directly), and then the mirror quietly ages until answers say the copy is
stale. Each user's periodic sync is also smeared across the 15-minute window
on purpose, so a freshly started beat can take up to two windows to touch
every service.

> If a migration changes the schema, restart the worker as well as the API.
> A worker holding stale code fails every sync, and the only symptom in the UI
> is that answers quietly have nothing to find.

### Running the app outside Docker

`make up` containerizes everything, but the everyday dev loop on this repo
runs only the **infrastructure** in Docker and the app as local processes —
uvicorn and Vite both hot-reload on save that way. Docker Desktop showing just
`postgres-1` and `redis-1` under `alpha-law` is this mode working as intended.

```bash
docker compose up -d postgres redis
```

The same `.env` from the top of this section is read in this mode too — fill
it in first.

Backend — Python 3.12, from `backend/`:

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/alembic upgrade head
.venv/bin/python -m uvicorn app.main:app --port 8000 --reload
```

The two Celery processes, each in its own terminal (also from `backend/`):

```bash
.venv/bin/celery -A app.tasks.celery_app worker -Q sync,embed,actions,orchestration,maintenance -l info
```

```bash
.venv/bin/celery -A app.tasks.celery_app beat -l info
```

Frontend — from `frontend/`:

```bash
cd frontend
npm install
npm run dev
```

The UI is still <http://localhost:5173> and the Google redirect URI does not
change: outside compose, Vite's `/api` proxy targets `http://localhost:8000`
instead of the `api` container (see `frontend/vite.config.ts`), so the browser
never needs to know which mode is running.

One check answers "is all of it actually up?" — processes, queues, mirror
freshness and the dead-letter table:

```bash
make jobs
```

The worker-restart rule from above applies double here: after a migration or
any backend code change, restart uvicorn **and** the worker by hand — nothing
restarts them for you in this mode.

### Demo data, without a Google account

`make seed` loads a dataset into the mirror directly — no Google needed — built
so the brief's own sample queries return real results, including the literal
ones (`sarah@company.com`, PDFs in last month, an event next Tuesday). It
prints the sign-in it created.

```bash
make seed          # load it            make seed-preview   # print without writing
make seed-clear    # remove it          make seed-reset     # clear then load
```

The seeded account's Google grant is a placeholder, so **reads** answer from
the mirror while **writes** say plainly that Google is not connected. Connect a
real account to exercise the write path.

Useful targets — `make help` lists them all:

```
make down          stop (volumes survive)      make ps        container status
make shell         bash in the api container   make psql      psql on the app db
make reset-db      drop the volume and migrate from scratch (destructive)
make revision m="add foo"                      autogenerate a migration
```

Alembic reads the DSN from `DATABASE_URL`; `alembic -x sqlalchemy.url=…` is
ignored. To migrate a database other than the configured one, set the variable:

```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/other alembic upgrade head
```

Note the driver: migrations run over `psycopg`, the app runs over `asyncpg`.

## Demo

<!-- To embed an inline player: edit this README on github.com and drag
     docs/demo/demo.mp4 into the editor right below this comment. GitHub
     uploads it and inserts a URL that renders as a playable video. Until
     then, the poster below links to the committed file, which GitHub's
     file viewer plays natively. -->

[![Demo video — 5½ minutes, every flow](docs/demo/poster.png)](docs/demo/demo.mp4)

**▶ [Watch the demo](docs/demo/demo.mp4)** — 5½ minutes, rendered with
[HyperFrames](https://github.com/heygen-com/hyperframes) from live
screenshots. GitHub plays it directly when the link is opened.

The demo data is honest about itself: the account is a real Google account, and
the video opens by showing the seed — events, inbox mail and Drive files
created through the app's own stored OAuth grant, then mirrored and searched
like anything else.

The walkthrough runs the brief's nine sample queries against a real Google
account — the three single-service reads, the three multi-service
orchestrations (Turkish Airlines cancellation with PNR extraction, Acme
meeting prep across all three services, out-of-office conflict detection), and
the three hard cases (the ambiguous "meeting with John" resolved as choice
chips, "that email" resolved from conversation context, "next Tuesday"
resolved in the user's timezone). It ends with the two-phase write flow:
compose → draft in Gmail → "Send it" → approval card → send.

[`docs/demo/`](docs/demo/) holds a full-resolution screenshot of every scene
and [`docs/demo/VIDEO_SCRIPT.md`](docs/demo/VIDEO_SCRIPT.md) the scene-by-scene
narration. The screenshots are reproducible: with the stack running,
`scripts/` seeding done and a connected account, the capture script drives
every flow headless through the real UI.

## Running the tests

```bash
make test          # the whole backend suite, inside the api container
```

Or directly, from `backend/`:

```bash
pytest tests/unit -q          # 619 tests, no services needed
```

The **unit** suite is pure: no database, no network, no Redis.

The **integration** suite runs the real FastAPI app, the real repositories and
the real Alembic migration against a real Postgres. Only the network is faked —
`respx` serves recorded Google and OpenAI payloads from `tests/fixtures/`, and
an unrecognised URL fails loudly by name instead of opening a socket. It needs
a database, and it drops and rebuilds that database's schema, so it refuses to
run against one whose name does not contain `test`:

```bash
docker compose up -d postgres
docker exec -it alpha-law-postgres-1 psql -U postgres -c "CREATE DATABASE orchestrator_test;"

cd backend
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5432/orchestrator_test \
  pytest tests/integration -q
```

Without `DATABASE_URL_TEST` the integration tests **skip** rather than fail.
A skip there means "not exercised", never "passed" — check the count.

Run one file or one test with `-k`:

```bash
pytest -q -k test_temporal
make test-watch k=test_temporal
```

Quality gates:

```bash
make lint          # ruff
make typecheck     # mypy over app/
make check         # lint + typecheck + test
```

The frontend typechecks and builds with:

```bash
cd frontend && npm install && npm run build     # tsc --noEmit && vite build
```

## What the UI does

- **The steps run inline.** Each turn draws its plan as it arrives — dimmed
  while pending, pulsing while running — then collapses to a one-line summary
  you can expand. There is no separate trace panel.
- **Structured answers are drawn, not printed.** A calendar answer is a list of
  events with times and guest counts; a file answer has working links. Every
  widget carries the markdown it replaced, so an older client, a screen reader
  or a copy-paste still gets the whole answer.
- **Ambiguity is a card, not a blank box.** "Move the meeting with John" offers
  the Johns it found, by name and address. Picking one resumes the same run at
  no extra model call.
- **Nothing leaves your account without a confirmation card**, and that is
  enforced by the database rather than by the planner remembering to ask.
- **Account menu** holds Integrations (what Ryuk can read) and Your information
  (what is imported and how fresh it is).

## Notes for anyone changing this

- Every repository function takes `user_id` as its **first** argument. It is
  the tenant key and there are no exceptions.
- Ids are `nanoid(21)` from `app.core.ids.new_id()`. Fingerprints —
  `content_hash`, `dedupe_key`, `payload_hash` — are uuid5 from
  `app.core.ids.fingerprint()`.
- Datetimes are timezone-aware UTC, everywhere, always. Date arithmetic happens
  in `app/orchestrator/temporal.py`, never in the model.
- A `needs_confirm` op must never perform its side effect in `run()`. It
  prepares; `execute()` runs later, after a person has approved it.
- One turn buys one embedding. A plan step that searches the mirror reuses the
  probe's vector rather than paying for a near-duplicate of it.
