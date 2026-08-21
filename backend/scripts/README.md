# scripts

Things a person runs by hand. Nothing in `app/` imports from here, and none of
it is on the request path.

| script | what it does | needs |
|---|---|---|
| `seed_local.py` | Fills our own mirror tables directly. No Google, no OAuth, no network. | Postgres |
| `seed_demo_account.py` | Plants the demo scenarios in a **real** throwaway Google account, through the same clients the app uses. | Postgres + a connected Google account |
| `reembed.py` | Refills the vector columns after the embedding model changes. | Postgres + an embedding key |
| `export_openapi.py` | Writes `docs/openapi.json` from the route table. | nothing |

Most of this file is about `seed_demo_account.py`, which is the one with real
prerequisites. The others explain themselves at the top of their own source.

Run them from `backend/`:

```bash
python -m scripts.seed_local
python -m scripts.seed_demo_account --email demo@example.com --dry-run
python -m scripts.export_openapi
```

Or inside the stack, which is what the Makefile does everywhere else:

```bash
docker compose exec api python -m scripts.seed_demo_account --email demo@example.com --dry-run
```

---

## Which seeder do I want?

**`seed_local`** writes `sync_gmail`, `sync_gcal` and `sync_gdrive` straight from
Python. Search reads our copy of Google and never Google itself, so the entire
read path — hybrid search, ranking, the probe, planning, ambiguity, confirm
cards — works with nothing connected. This is the one for clicking around, for
tests, and for the graded precision numbers.

**`seed_demo_account`** writes the same scenarios into a real Gmail, Calendar
and Drive. A real sync task then pulls them back into the mirror. Slower, needs
an account, and the only way to demonstrate that the sync path, the OAuth path
and the live write path work at all. This is the one for the video.

They are not alternatives so much as two halves. Seed local to develop; seed the
account before recording.

---

# `seed_demo_account.py`

## What it plants

About 78 items, all carrying a marker, anchored to the day you run it.

**Gmail** — a Turkish-language booking confirmation from `bilet@thy.com` with a
PNR and flight TK1984; a second booking on a different carrier so "cancel my
flight to Istanbul" is genuinely ambiguous; a four-message Acme proposal thread
for the conversation-context case; four separate budget messages from
`sarah@company.com` for the sender-plus-topic query; and about thirty filler
messages spread over the last ninety days, so search has to discriminate rather
than return everything.

**Calendar** — the Istanbul flight matching the Turkish booking, a second flight
for the ambiguity case, an Acme meeting tomorrow, next week's events including
two different Johns and two with `john@company.com` invited, one meeting with no
agenda and one with a Drive link in its description.

**Drive** — an out-of-office doc naming a date range, `Acme Q3 Proposal v3.pdf`
plus a near-identical decoy so a share is genuinely ambiguous, and seven PDFs
modified last month.

Every run ends with a table mapping each item to the scenario it serves in
`docs/SAMPLE_QUERIES.md`, plus a short list of the places where this data
knowingly differs from that document and why. Read that list. A silent
divergence between the graded document and the account it is demonstrated on is
the worst kind.

## Nothing is sent

Mail goes in through `users.messages.insert`, which puts a message in the
mailbox with no delivery and no spam classification. Calendar writes pass
`sendUpdates=none`. No mail leaves the account, and none of the fictional people
in the data — `sarah@company.com`, `dana@acmecorp.com` — is ever contacted.

Even so: **point this at a throwaway account.** It creates, updates and trashes
things without asking twice.

---

## Before you run it

### 1. A throwaway Google account

A fresh personal Google account. Not your mail, not a work account, not anything
with a real calendar in it.

### 2. A Google Cloud project

- Enable the **Gmail API**, **Google Calendar API** and **Google Drive API**.
- Create an **OAuth client ID**, type *Web application*.
- Add the redirect URI, exactly matching `GOOGLE_REDIRECT_URI` in `.env`:
  `http://localhost:8000/api/v1/auth/google/callback`
- Put the client id and secret in `.env` as `GOOGLE_CLIENT_ID` and
  `GOOGLE_CLIENT_SECRET`.

### 3. The consent screen, in testing mode

User type **External**, publishing status **Testing**, and the throwaway account
added under **Test users**. Leave it in testing — see the seven-day note below.
Publishing an app that asks for Gmail and Drive scopes means Google's
verification review, which takes weeks and is not what this is for.

### 4. Scopes

`app/auth/google_oauth.py` asks for all of these:

| scope | why the seeder needs it |
|---|---|
| `gmail.readonly` | reading back what is already planted |
| `gmail.modify` | **`messages.insert`** and the marker label; also trashing on `--clean` |
| `gmail.compose`, `gmail.send` | the app's drafts and sends, not the seeder's |
| `calendar` | creating, patching and deleting events |
| `drive.readonly` | the sync task's file listing |
| `drive.file` | creating the folders and files the seeder plants, and trashing them |
| `openid`, `email`, `profile` | who is signing in |

Google's consent screen lets a person untick individual scopes. Don't. The app
refuses a grant missing `gmail.readonly`, `gmail.compose`, `calendar` or
`drive.readonly` at the door, and the seeder needs `gmail.modify`, `calendar`
and `drive.file` on top of that. If in doubt, disconnect and grant again with
everything ticked.

`drive.file` only ever sees files this OAuth client created, which is exactly
the set the seeder plants and exactly the set `--clean` removes. It cannot see
the rest of the account's Drive.

### 5. Connect the account once, through the app

The seeder reads the encrypted refresh token out of `oauth_tokens`. It does not
run an OAuth flow of its own, so the account has to be connected first:

```bash
make up
open http://localhost:8000/api/v1/auth/google     # consent as the throwaway account
```

The callback writes the `users` row and the encrypted `oauth_tokens` row, and
sets a session cookie in that browser. The seeder needs only the database rows,
so it can now find the account by address.

The curl commands further down need that cookie. Copy its value out of the
browser once — it is called `alpha_session` unless `SESSION_COOKIE_NAME` says
otherwise — and keep it in the shell:

```bash
export EVAL_SESSION_COOKIE='alpha_session=<value from the browser>'
curl -s localhost:8000/api/v1/auth/me -H "Cookie: $EVAL_SESSION_COOKIE"
```

### 6. Environment

`DATABASE_URL`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`
and `TOKEN_ENCRYPTION_KEY` must all be set — the last one is the AES key the
refresh token is stored under, so a changed key means every stored token is
unreadable and every account has to reconnect.

---

## Running it

Always look first. `--dry-run` needs no database, no network and no keys:

```bash
python -m scripts.seed_demo_account --dry-run
python -m scripts.seed_demo_account --dry-run --anchor 2026-08-20 --tz America/New_York
```

Then plant:

```bash
python -m scripts.seed_demo_account --email demo@example.com
```

| flag | |
|---|---|
| `--email` / `--user-id` | which connected account. One of them is required. |
| `--dry-run` | print the whole plan and stop. Touches nothing. |
| `--clean` | trash everything it previously planted, then stop. |
| `--reseed` | clean, then plant. The one to run before recording. |
| `--only gmail,gcal,gdrive` | a subset of the services. |
| `--anchor 2026-08-20` | the local date every other date is computed from. Default: today. |
| `--tz America/New_York` | override the timezone on the `users` row. |
| `--yes` | skip the "type the account address" confirmation on a removing run. |
| `--access-token` | use a raw token and skip the database entirely. Also `DEMO_ACCESS_TOKEN`. |

It takes a couple of minutes. Items are planted one at a time so a single
failure loses one item and not the run; anything that fails is listed at the
bottom, and the process exits non-zero.

## Re-running is safe

Everything planted carries a marker, and the marker is how it stays idempotent:

| | marker | version |
|---|---|---|
| Gmail | label `alphalaw-demo` + header `X-Alphalaw-Demo` | `X-Alphalaw-Hash` |
| Calendar | `extendedProperties.private.alphalaw_demo` | `…alphalaw_hash` |
| Drive | `appProperties.alphalaw_demo` | `…alphalaw_hash` |

The slug says which item this is; the hash says which version. Same hash, nothing
to do. Different hash means update — a patch for Calendar and Drive, and for
Gmail, where a message is immutable, a fresh insert with the old one to the bin.

So a second run reports `unchanged` for everything it recognises, and only
touches what actually changed.

## The anchor, and why to re-seed before recording

Every date is derived from one anchor day, defaulting to today. "Tomorrow" and
"next week" mean what they say **on the day you run it**. A dataset seeded on a
Tuesday has the wrong week in it by Thursday, and "Prepare for tomorrow's
meeting with Acme Corp" quietly finds nothing.

Re-seed the morning of the demo. `--reseed` is one command and takes two minutes.

Pass `--anchor` when you want the dataset pinned instead — comparing two runs,
or matching the fixed instant `docs/SAMPLE_QUERIES.md` evaluates against.

## Cleaning up

```bash
python -m scripts.seed_demo_account --email demo@example.com --clean
```

It asks you to type the account address before removing anything, unless you
pass `--yes`.

Gmail messages and Drive files go to the **bin**, not to nothing — empty it by
hand if you want them gone for good. Calendar has no bin, so those events are
deleted outright. Anything the seeder did not plant is untouched: it only ever
acts on rows carrying its own marker.

---

## The seven-day refresh token

**While your OAuth consent screen is in Testing, Google expires refresh tokens
after seven days.** This is Google's rule for unverified apps, not something the
code can work around.

What it looks like when it bites:

- `GET /api/v1/sync/status` goes red, `last_error` says `invalid_grant`
- `oauth_tokens.refresh_failures` climbs
- API calls come back `428 GOOGLE_REAUTH_REQUIRED`
- the seeder fails on its first Google call with an auth error

The fix is thirty seconds: reconnect.

```bash
open http://localhost:8000/api/v1/auth/google
```

Consent again as the throwaway account. The flow sends `prompt=consent` and
`access_type=offline`, so Google issues a fresh refresh token and
`auth/token_store.py` re-encrypts it in place. **The planted data survives** —
only the token died. You do not need to re-seed, though re-seeding is what you
want anyway if the anchor has gone stale.

Do this the morning of any demo, whether or not it has broken yet. Seven days is
a floor, not a promise, and finding out on camera is not the moment.

The other way out is publishing the app so the tokens stop expiring, but the
scopes here are restricted, publishing means Google's verification review, and
the review takes weeks. For a throwaway account, reconnecting is the answer.

---

## After seeding: pull it into the mirror

Nothing the orchestrator does reads Google live on the read path — it reads our
mirror. So the account is not usable until a sync has run.

```bash
curl -X POST localhost:8000/api/v1/sync/trigger \
     -H 'Content-Type: application/json' -H "Cookie: $EVAL_SESSION_COOKIE" \
     -d '{"mode": "full"}'

curl -s localhost:8000/api/v1/sync/status -H "Cookie: $EVAL_SESSION_COOKIE" | jq
```

Watch `items_indexed` climb and `last_success_at` move on all three services.
Give it a minute; embeddings are a second queue behind the fetch.

If the worker logs an *unregistered task* (`docker compose logs worker`), the
trigger endpoint and the task names have drifted apart. Enqueue directly and
carry on:

```bash
docker compose exec api python -c "
from app.tasks.celery_app import celery_app
for s in ('gmail', 'gcal', 'gdrive'):
    celery_app.send_task(f'sync.{s}.backfill', args=['<users.id>'], queue='sync')
"
```

The registered names are `sync.gmail`, `sync.gcal`, `sync.gdrive` for the
incremental pass and `sync.<service>.backfill` for the first full pull.

Set the embedding key before syncing — `OPENAI_API_KEY` for the default
`EMBED_MODEL=openai:text-embedding-3-small`, or whatever key the model you have
configured needs. Without one the embedder falls back to deterministic hashed
vectors: the app still runs end to end and the keyword arm stays honest, but the
vector arm is only self-consistent and any precision number measured against it
is meaningless.

Changing `EMBED_MODEL` after a sync means every vector in the mirror came from
two different models and is no longer comparable. `scripts/reembed.py` is the
way back.

---

## Regenerating the eval numbers

```bash
python -m tests.eval.intent_accuracy                  # -> out/intent_accuracy.json
python -m tests.eval.precision_at_k                   # -> out/precision_at_k.json
python -m tests.eval.latency                          # -> out/latency.json
python -m tests.eval.precision_at_k --write-results   # -> tests/eval/RESULTS.md
```

Those four are not all measuring the same thing, and it matters which corpus
each one is pointed at.

**Precision, recall, MRR, nDCG — run these against `seed_local`, not the demo
account.** `tests/eval/datasets/relevance.jsonl` is graded against the offline
fixture corpus: it names documents by the ids that seeder assigns, and its
judgements describe those exact subjects. The demo account is a different
corpus with different subjects — a Turkish booking rather than an English one,
Pegasus rather than a second Turkish flight, John Okafor rather than John Park —
and Google mints its own ids besides. Scoring the demo account against those
judgements measures nothing. `--match title` and the default `--match auto` join
on normalised titles rather than ids, which handles the id problem but not the
different-corpus problem. Judging the demo account is writing a new dataset, not
re-running an old one, and no shortcut makes that untrue.

**Latency is the number worth re-taking on the demo account**, because it is the
only place the timings include real Google calls:

```bash
python -m tests.eval.latency --path query --backend http \
       --base-url http://localhost:8000 --cookie "$EVAL_SESSION_COOKIE" \
       --user-email demo@example.com
```

`--path search` measures the mirror and is the figure the brief's *under 500 ms*
line refers to; `--path query` measures the whole request. They are never
averaged together.

CI variants, which need neither a database nor a key:

```bash
python -m tests.eval.intent_accuracy --dry-run    # canned classifier
python -m tests.eval.precision_at_k --self-test   # checks the metric maths
```

`tests/eval/README.md` has the rest — what is in each dataset, how a hit is
matched to a judgement, and how the ablation is run.

---

## When it goes wrong

| what you see | what it is |
|---|---|
| `no user for demo@example.com` | the account was never connected. Do step 5. |
| auth error on the first Google call | the refresh token expired. Reconnect — see above. |
| `403 insufficientPermissions` on insert | a scope was unticked on the consent screen. Disconnect and grant again. |
| `403 accessNotConfigured` | that API is not enabled on the Cloud project. |
| every item says `unchanged` | working as intended. Nothing about the data changed. Use `--reseed` to force. |
| duplicate messages in the mailbox | an interrupted run left an old copy untrashed. `--clean` then plant again. |
| a synced mailbox that looks empty | the sync ran but the embeddings queue has not. Check the `embed` queue and `OPENAI_API_KEY`. |
| "tomorrow's meeting" finds nothing | the anchor has gone stale. `--reseed`. |
