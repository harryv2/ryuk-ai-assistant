"""Seed the local mirror directly. No Google account, no OAuth, no network.

Search reads our own copy of Gmail/Calendar/Drive, never Google live. So if we
write that copy ourselves, the whole read path works with nothing connected:
search, ranking, the probe, planning, ambiguity, confirm cards.

Embeddings come from OpenAI when OPENAI_API_KEY is set. Without a key it falls
back to deterministic hashed vectors so the app still runs end to end — the
keyword arm stays honest, the vector arm is only self-consistent. Fine for
clicking around, useless for measuring precision. It says so when it does it.

    python -m scripts.seed_local                 # seed
    python -m scripts.seed_local --clean         # remove and reseed
    python -m scripts.seed_local --dry-run       # show what it would write

Prints a signed session cookie at the end so you can use the UI immediately.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imported lazily inside the functions that need them, so --dry-run works on a
# bare checkout with nothing installed.
if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession

EMBEDDING_DIM = 1536

TZ = ZoneInfo("Asia/Kolkata")
ME = "demo@example.com"
PASSWORD = "demo1234"
MARKER = "[seed-local]"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def days(n: int) -> dt.datetime:
    return now() + dt.timedelta(days=n)


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------

def hashed_vector(text: str) -> list[float]:
    """A deterministic unit vector. Same text always gives the same direction,
    different text gives a different one. Enough to exercise the plumbing."""
    out: list[float] = []
    counter = 0
    while len(out) < EMBEDDING_DIM:
        digest = hashlib.sha256(f"{text}|{counter}".encode()).digest()
        out.extend((b - 127.5) / 127.5 for b in digest)
        counter += 1
    vec = out[:EMBEDDING_DIM]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


async def embed_all(texts: list[str], real: bool) -> list[list[float]]:
    if not real:
        return [hashed_vector(t) for t in texts]
    from app.llm import embed  # imported late so no key is needed to --dry-run
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 128):
        vectors.extend(await embed(texts[i : i + 128]))
    return vectors


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------

@dataclass
class Mail:
    message_id: str
    thread_id: str
    subject: str
    from_email: str
    from_name: str
    body: str
    received_at: dt.datetime
    to_emails: list[str] = field(default_factory=lambda: [ME])
    labels: list[str] = field(default_factory=lambda: ["INBOX"])
    has_attachments: bool = False
    supports: str = ""


@dataclass
class Event:
    event_id: str
    title: str
    description: str
    starts_at: dt.datetime
    duration_min: int
    attendees: list[dict] = field(default_factory=list)
    location: str = ""
    supports: str = ""


@dataclass
class File:
    file_id: str
    name: str
    mime_type: str
    excerpt: str
    modified_at: dt.datetime
    folder_path: str = "/My Drive"
    size_bytes: int = 100_000
    supports: str = ""


def att(email: str, name: str, status: str = "accepted") -> dict:
    return {"email": email, "name": name, "response_status": status, "optional": False}


def build_mail() -> list[Mail]:
    mail = [
        # The brief's own sample query names this address verbatim —
        # "Find emails from sarah@company.com about the budget" — so the
        # dataset answers the question as asked rather than as paraphrased.
        Mail(
            "m_budget_q4", "t_budget_q4",
            "Q4 budget — headcount and tooling",
            "sarah@company.com", "Sarah Mitchell",
            "Hi,\n\nQ4 budget draft is ready. Headcount is up two engineers, "
            "tooling holds flat at 18,000, and the travel line drops by a third "
            "now that the offsite moved. Marketing is the only overspend at 12%."
            "\n\nCan you review before Thursday?\n\nSarah",
            days(-4),
            supports="brief sample 2 — emails from sarah@company.com about the budget",
        ),
        Mail(
            "m_budget_q4_re", "t_budget_q4",
            "Re: Q4 budget — headcount and tooling",
            "sarah@company.com", "Sarah Mitchell",
            "One correction — tooling is 19,400, not 18,000. The rest stands.",
            days(-3),
            supports="brief sample 2 — a second hit so the search has to rank",
        ),
        Mail(
            "m_thy_booking", "t_thy",
            "Türk Hava Yolları — Rezervasyon Onayı TK1234",
            "noreply@thy.com", "Türk Hava Yolları",
            "Sayın Yolcumuz,\n\nTK1234 numaralı rezervasyonunuz onaylandı.\n"
            "JFK 09:15 → IST 03:40, 3 Eylül 2026.\n"
            "Rezervasyon kodu: X7QM2P\n\n"
            "Değişiklik için: reservations@thy.com\n\nİyi yolculuklar dileriz.",
            days(-6),
            supports="S04 cancel flight — cross-lingual alias + PNR extraction",
        ),
        Mail(
            "m_ek_booking", "t_ek",
            "Emirates booking confirmed — EK507",
            "noreply@emirates.com", "Emirates",
            "Your booking is confirmed.\n\nBooking reference: KP93MB\n"
            "Flight EK507, BOM 04:30 → DXB 06:05, 12 September 2026.\n"
            "To change or cancel, reply to support@emirates.com.",
            days(-4),
            supports="S15 two flights — makes 'cancel my flight' genuinely ambiguous",
        ),
        Mail(
            "m_acme_prop_1", "t_acme_prop",
            "Acme Q3 proposal — revised pricing",
            "sarah@acme.com", "Sarah Chen",
            "Hi,\n\nAttaching the revised Q3 proposal. Headline changes: the "
            "platform tier drops to 48k annually, and we've pulled the "
            "onboarding fee entirely.\n\nDelivery is still October as discussed. "
            "Let me know if the numbers work.\n\nSarah",
            days(-6), has_attachments=True,
            supports="S08 'that email about the proposal' — conversation context",
        ),
        Mail(
            "m_acme_prop_2", "t_acme_prop",
            "Re: Acme Q3 proposal — revised pricing",
            ME, "Demo User",
            "Thanks Sarah — reviewing with the team, back to you Thursday.",
            days(-5), to_emails=["sarah@acme.com"], labels=["SENT"],
            supports="S21 thread with a reply — multi-hop discovery",
        ),
        Mail(
            "m_acme_prop_3", "t_acme_prop",
            "Re: Acme Q3 proposal — revised pricing",
            "sarah@acme.com", "Sarah Chen",
            "One more thing — the signed contract is here if legal needs it:\n"
            "https://docs.google.com/document/d/1AcmeContract2026/edit\n\n"
            "No rush.",
            days(-3),
            supports="S22 'the contract Sarah mentioned' — Drive link extraction",
        ),
        Mail(
            "m_budget_1", "t_budget",
            "Q3 budget review — numbers attached",
            "sarah@acme.com", "Sarah Chen",
            "Here's the Q3 budget breakdown. Marketing is 12% over, everything "
            "else is on plan. Can we talk Thursday?",
            days(-9), has_attachments=True,
            supports="S02 'emails from sarah about the budget'",
        ),
        Mail(
            "m_budget_2", "t_budget2",
            "Budget freeze until Q4",
            "finance@company.com", "Finance",
            "All discretionary spend is frozen until the Q4 planning cycle closes.",
            days(-14),
            supports="S02 decoy — same topic, different sender",
        ),
        Mail(
            "m_ooo", "t_ooo",
            "Out of office 26–28 August",
            ME, "Demo User",
            "I'm away 26 to 28 August. See the OOO doc for cover arrangements.",
            days(-2), to_emails=["team@company.com"], labels=["SENT"],
            supports="S06 conflict detection",
        ),
    ]
    fillers = [
        ("Standup notes", "eng@company.com", "Engineering"),
        ("Invoice #4471", "billing@vendor.io", "Vendor Billing"),
        ("Your weekly digest", "digest@news.com", "News Digest"),
        ("Security alert: new sign-in", "no-reply@accounts.google.com", "Google"),
        ("Lunch?", "priya@company.com", "Priya N"),
        ("Design review moved", "design@company.com", "Design"),
        ("Contract renewal reminder", "legal@vendor.io", "Vendor Legal"),
        ("Re: onboarding checklist", "hr@company.com", "People Ops"),
        ("Server maintenance window", "ops@company.com", "Ops"),
        ("Quarterly all-hands deck", "comms@company.com", "Comms"),
        ("Expense report approved", "finance@company.com", "Finance"),
        ("New comment on your doc", "no-reply@docs.google.com", "Google Docs"),
    ]
    for i, (subject, sender, name) in enumerate(fillers):
        mail.append(Mail(
            f"m_fill_{i}", f"t_fill_{i}", subject, sender, name,
            f"{subject}. Routine message, nothing actionable.",
            days(-(11 + i * 5)),
            supports="filler — search has to discriminate, not just return everything",
        ))
    return mail


def build_events() -> list[Event]:
    def at(day_offset: int, hour: int, minute: int = 0) -> dt.datetime:
        local = (now().astimezone(TZ) + dt.timedelta(days=day_offset)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        return local.astimezone(dt.timezone.utc)

    # "Next Tuesday" is one of the brief's hard cases, so the week ahead has
    # to contain a Tuesday worth finding. Computed rather than hard-coded, or
    # the fixture rots the moment the seed is run on a different weekday.
    days_to_tuesday = (1 - now().astimezone(TZ).weekday()) % 7 or 7

    return [
        Event("e_next_tuesday_review", "Design review — mobile onboarding",
              "Walk through the new onboarding screens with the design team.",
              at(days_to_tuesday, 11), 60,
              attendees=[att("priya@company.com", "Priya Nair"),
                         att("sarah@company.com", "Sarah Mitchell")],
              supports="brief hard case — 'next Tuesday' temporal reasoning"),
        Event("e_thy_flight", "JFK → IST TK1234",
              "Türk Hava Yolları TK1234. Booking X7QM2P.",
              at(14, 9, 15), 600, location="JFK Terminal 1",
              supports="S04 matches the Turkish booking email"),
        Event("e_ek_flight", "BOM → DXB EK507",
              "Emirates EK507. Booking KP93MB.",
              at(23, 4, 30), 215, location="BOM Terminal 2",
              supports="S15 second flight — the ambiguity"),
        Event("e_acme_tomorrow", "Acme Corp — quarterly review",
              "Agenda: https://docs.google.com/document/d/1AcmeAgenda/edit",
              at(1, 15), 60,
              attendees=[att(ME, "Demo User"), att("sarah@acme.com", "Sarah Chen"),
                         att("john.smith@acme.com", "John Smith")],
              supports="S05 prepare for tomorrow's Acme meeting"),
        Event("e_john_smith_sync", "Acme sync — John S", "Weekly catch-up.",
              at(2, 11), 60,
              attendees=[att(ME, "Demo User"), att("john.smith@acme.com", "John Smith")],
              supports="S07 'the meeting with John' — candidate 1"),
        Event("e_john_doe_intro", "Vendor intro call", "First call with the vendor.",
              at(7, 15), 30,
              attendees=[att(ME, "Demo User"), att("john.doe@vendor.io", "John Doe")],
              supports="S07 'the meeting with John' — candidate 2, forces the ask"),
        Event("e_next_week_1", "Platform planning",
              "Roadmap for the next quarter.", at(5, 10), 90,
              attendees=[att(ME, "Demo User"), att("john@company.com", "John Patel"),
                         att("priya@company.com", "Priya N")],
              supports="S01 / attendee filter — john@company.com invited"),
        Event("e_next_week_2", "Design critique", "", at(6, 14), 60,
              attendees=[att(ME, "Demo User"), att("john@company.com", "John Patel")],
              supports="attendee filter — second hit"),
        Event("e_next_week_3", "1:1 with Priya", "Career chat.", at(6, 16, 30), 30,
              attendees=[att(ME, "Demo User"), att("priya@company.com", "Priya N")],
              supports="attendee filter — must NOT match john@company.com"),
        Event("e_no_agenda", "Partner sync", "", at(5, 17), 45,
              attendees=[att(ME, "Demo User"), att("sarah@acme.com", "Sarah Chen")],
              supports="S19 'meetings with no agenda' — empty description, no link"),
        Event("e_ooo_clash", "Vendor QBR", "Quarterly business review.",
              at(6, 11), 120,
              attendees=[att(ME, "Demo User"), att("john.doe@vendor.io", "John Doe")],
              supports="S06 clashes with the out-of-office window"),
        Event("e_standup", "Daily standup", "", at(1, 9, 30), 15,
              attendees=[att(ME, "Demo User")],
              supports="filler"),
    ]


def build_files() -> list[File]:
    return [
        # "Show me PDFs in Drive from last month" is a mime + window query, so
        # the set needs a PDF that actually falls in the previous month.
        File("f_vendor_msa", "Vendor MSA 2026.pdf", "application/pdf",
             "Master services agreement, 2026 term.\n\nPayment net 30. "
             "Termination for convenience with 60 days notice. Liability "
             "capped at fees paid in the preceding twelve months.",
             days(-38), size_bytes=1_180_000,
             supports="brief sample 3 — PDFs in Drive from last month"),
        File("f_q3_report", "Q3 performance report.pdf", "application/pdf",
             "Q3 2026 performance.\n\nRevenue up 14% quarter on quarter. "
             "Churn steady at 1.9%. Support backlog cleared in September.",
             days(-45), size_bytes=860_000,
             supports="brief sample 3 — a second PDF in the same window"),
        File("f_ooo_doc", "Out of office — August.gdoc",
             "application/vnd.google-apps.document",
             "Out of office 26 to 28 August 2026.\n\nCover: Priya handles "
             "escalations, Sarah covers the Acme account. No meetings should be "
             "booked in this window. Back at desk 29 August.",
             days(-3), supports="S06 conflict detection — the OOO window"),
        File("f_acme_prop", "Acme Q3 Proposal v3.pdf", "application/pdf",
             "Acme Corporation — Q3 2026 Proposal, version 3.\n\nScope: platform "
             "licence, onboarding, support.\nPricing: 48,000 annually, onboarding "
             "fee waived.\nTimeline: delivery October 2026.",
             days(-6), size_bytes=2_400_000,
             supports="S16 share the proposal — candidate 1"),
        File("f_acme_prop_old", "Acme Q3 Proposal v2.pdf", "application/pdf",
             "Acme Corporation — Q3 2026 Proposal, version 2. Superseded. "
             "Pricing: 55,000 annually plus a 5,000 onboarding fee.",
             days(-20), size_bytes=2_100_000,
             supports="S16 decoy — makes the share genuinely ambiguous"),
        File("f_contract", "Acme Contract 2026.gdoc",
             "application/vnd.google-apps.document",
             "Master services agreement between Acme Corporation and Company. "
             "Term: 12 months from execution. Signed by both parties.",
             days(-3), supports="S22 the contract Sarah linked"),
        File("f_budget", "Q3 Budget.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "Q3 budget by department. Marketing 12 percent over plan. "
             "Engineering on plan. Total spend 1.24M.",
             days(-9), size_bytes=88_000, supports="S02 budget context"),
        File("f_pdf_1", "Vendor invoice 4471.pdf", "application/pdf",
             "Invoice 4471. Amount due 12,400. Net 30.",
             days(-25), size_bytes=140_000,
             supports="S03 PDFs from last month"),
        File("f_pdf_2", "Security review 2026.pdf", "application/pdf",
             "Annual security review. No critical findings. Two medium findings "
             "on access review cadence.",
             days(-31), size_bytes=910_000,
             supports="S03 PDFs from last month"),
        File("f_pdf_3", "Onboarding handbook.pdf", "application/pdf",
             "New joiner handbook. Equipment, accounts, first-week checklist.",
             days(-40), size_bytes=3_100_000,
             supports="S03 boundary — outside 'last month'"),
        File("f_notes", "Meeting notes.gdoc",
             "application/vnd.google-apps.document",
             "Assorted meeting notes.", days(-12), supports="filler"),
    ]


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def mail_embed_text(m: Mail) -> str:
    return f"{m.subject}\n{m.subject}\n{m.body}"


def event_embed_text(e: Event) -> str:
    who = " ".join(f"{a['name']} {a['email']}" for a in e.attendees)
    return f"{e.title}\n{e.description}\n{e.location}\n{who}"


def file_embed_text(f: File) -> str:
    return f"{f.name}\n{f.mime_type}\n{f.excerpt}"


async def wipe(session: "AsyncSession", user_id: str) -> None:
    """Everything this script created for one person. The account stays.

    Conversations go too. They are not seed data in the strict sense, but a
    reset that leaves twenty old threads in the sidebar is not a reset — and
    this only ever runs against the demo address.
    """
    from sqlalchemy import delete
    from app.db.models import (
        Conversation, OAuthToken, SyncEvent, SyncFile, SyncMessage, SyncState,
    )

    # Conversations first: runs, messages, actions and pending inputs hang off
    # them by foreign key and go with the cascade.
    for model in (Conversation, SyncMessage, SyncEvent, SyncFile, SyncState, OAuthToken):
        await session.execute(delete(model).where(model.user_id == user_id))


async def seed(session: "AsyncSession", *, real_embeddings: bool) -> tuple[str, dict]:
    from sqlalchemy import select
    from app.core.ids import fingerprint, new_id
    from app.db.models import OAuthToken, SyncState, User
    from app.db.repositories import mirror

    from app.auth.passwords import hash_password

    existing = (await session.execute(select(User).where(User.email == ME))).scalar_one_or_none()
    if existing:
        user = existing
        user.timezone = "Asia/Kolkata"
        user.work_week_start = 1
        await wipe(session, user.id)
    else:
        user = User(id=new_id(), email=ME, display_name="Demo User",
                    timezone="Asia/Kolkata", work_week_start=1)
        session.add(user)
    # Seeded so the account can be signed into from the form like any other,
    # rather than only through a pasted cookie.
    user.password_hash = hash_password(PASSWORD)
    user.email_verified = True
    await session.flush()

    # A placeholder token so anything that checks "is Google connected" is happy.
    # It is not a real credential and cannot call Google — reads come from the
    # mirror we are about to write.
    from app.core.crypto import encrypt
    session.add(OAuthToken(
        id=new_id(), user_id=user.id, provider="google",
        provider_account_id="seed-local",
        account_email=ME,
        access_token_enc=encrypt("seed-local-not-a-real-token"),
        refresh_token_enc=encrypt("seed-local-not-a-real-token"),
        scopes=["seed.local"], expires_at=days(3650),
    ))

    mails, events, files = build_mail(), build_events(), build_files()
    texts = ([mail_embed_text(m) for m in mails]
             + [event_embed_text(e) for e in events]
             + [file_embed_text(f) for f in files])
    vectors = await embed_all(texts, real_embeddings)
    vi = iter(vectors)

    # Written through the repository, not with raw model objects, so the seed
    # goes down the same path a real sync does — participants, attributes and
    # the natural key all get built by the code that has to get it right in
    # production. A seed that writes rows its own way stops being evidence.
    from app.llm.router import embed_model_id

    model_id = embed_model_id() if real_embeddings else "hashed-local"

    await mirror.upsert_gmail(session, user.id, [
        dict(
            message_id=m.message_id, thread_id=m.thread_id, chunk_index=0,
            subject=m.subject, from_email=m.from_email, from_name=m.from_name,
            to_emails=m.to_emails, body_clean=m.body, labels=m.labels,
            has_attachments=m.has_attachments, received_at=m.received_at,
            content_hash=fingerprint("gmail", mail_embed_text(m)),
            embedding=next(vi), embed_model=model_id,
        )
        for m in mails
    ])

    await mirror.upsert_gcal(session, user.id, [
        dict(
            event_id=e.event_id, calendar_id="primary", title=e.title,
            description=e.description, location=e.location or None,
            organizer_email=ME, attendees=e.attendees or [att(ME, "Demo User")],
            starts_at=e.starts_at,
            ends_at=e.starts_at + dt.timedelta(minutes=e.duration_min),
            all_day=False, event_timezone="Asia/Kolkata", status="confirmed",
            etag=f'W/"{fingerprint("etag", e.event_id).hex[:12]}"',
            content_hash=fingerprint("gcal", event_embed_text(e)),
            embedding=next(vi), embed_model=model_id,
        )
        for e in events
    ])

    await mirror.upsert_gdrive(session, user.id, [
        dict(
            file_id=f.file_id, chunk_index=0, name=f.name,
            mime_type=f.mime_type, owner_email=ME, is_shared=False,
            web_view_link=f"https://drive.google.com/file/d/{f.file_id}/view",
            folder_path=f.folder_path, size_bytes=f.size_bytes,
            content_excerpt=f.excerpt, modified_at=f.modified_at,
            content_hash=fingerprint("gdrive", file_embed_text(f)),
            embedding=next(vi), embed_model=model_id,
        )
        for f in files
    ])

    from app.db.models import SyncService
    counts = {"gmail": len(mails), "gcal": len(events), "gdrive": len(files)}
    for service, count in (("gmail", counts["gmail"]), ("gcal", counts["gcal"]),
                           ("gdrive", counts["gdrive"])):
        session.add(SyncState(
            id=new_id(), user_id=user.id, service=SyncService(service),
            backfill_complete=True, last_synced_at=now(), last_success_at=now(),
            items_indexed=count,
        ))

    await session.commit()
    return user.id, counts


def report(user_id: str, counts: dict, real: bool) -> None:
    from app.auth.deps import issue_session, session_cookie_name
    cookie = issue_session(user_id)
    name = session_cookie_name()

    print()
    print("  seeded")
    print(f"    user            {ME}  ({user_id})")
    print(f"    timezone        Asia/Kolkata, week starts Monday")
    print(f"    gmail           {counts['gmail']} messages")
    print(f"    calendar        {counts['gcal']} events")
    print(f"    drive           {counts['gdrive']} files")
    print(f"    embeddings      {'OpenAI (real)' if real else 'hashed (offline)'}")
    if not real:
        print()
        print("    Hashed vectors are self-consistent but carry no meaning.")
        print("    The keyword arm still works. Do not read precision numbers")
        print("    off this — set OPENAI_API_KEY and reseed for that.")
    print()
    print("  sign in at http://localhost:5173")
    print()
    print(f"    email     {ME}")
    print(f"    password  {PASSWORD}")
    print(f"    code      123456")
    print()
    print("  or skip the form — paste into the browser console on the app origin:")
    print()
    print(f'    document.cookie = "{name}={cookie}; path=/"')
    print()
    print("  or with curl:")
    print()
    print(f'    curl -s localhost:8000/api/v1/auth/me -H "Cookie: {name}={cookie}"')
    print()
    print("  reads work offline — the mirror is already filled.")
    print("  writes need a real Google connection: this token is a placeholder,")
    print("  so a send or a cancel gets a 401 and marks the connection stale.")
    print("  run `make seed` again to put it back.")
    print()
    print("  try:  What's on my calendar next week?")
    print("        Cancel my Turkish Airlines flight")
    print("        Cancel my flight            (two match — it should ask)")
    print("        Move the meeting with John  (two Johns — it should ask)")
    print()


async def clear() -> int:
    """Remove the demo data. The account survives so a sign-in still works.

    Deliberately narrow: this drops what the seeder created for one known
    address, not every row in the database. A "clear the seed" command that
    quietly emptied a real user's mirror would be a bad surprise exactly once.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from app.config import settings
    from app.db.models import User

    engine = create_async_engine(settings.DATABASE_URL, future=True)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            user = (
                await session.execute(select(User).where(User.email == ME))
            ).scalar_one_or_none()
            if user is None:
                print(f"\n  nothing to clear — no account for {ME}\n")
                return 0
            await wipe(session, user.id)
            await session.commit()
    finally:
        await engine.dispose()

    print()
    print("  cleared")
    print(f"    {ME} still exists and can still sign in")
    print("    mail, calendar, files, chat history and the connection are gone")
    print()
    print("  put it back with:  make seed")
    print()
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true",
                    help="delete the demo data, then seed it again")
    ap.add_argument("--clear", action="store_true",
                    help="delete the demo data and stop — does not seed")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="force hashed vectors even with a key present")
    args = ap.parse_args()

    real = bool(os.getenv("OPENAI_API_KEY")) and not args.no_embeddings

    if args.clear:
        return await clear()

    if args.dry_run:
        mails, events, files = build_mail(), build_events(), build_files()
        print(f"\n  would seed {ME}\n")
        for group, items in (("gmail", mails), ("calendar", events), ("drive", files)):
            print(f"  {group}")
            for it in items:
                label = getattr(it, "subject", None) or getattr(it, "title", None) or it.name
                print(f"    {label[:52]:54} {it.supports}")
            print()
        print(f"  embeddings: {'OpenAI' if real else 'hashed (no OPENAI_API_KEY)'}\n")
        return 0

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from app.config import settings
    from app.db.models import User

    engine = create_async_engine(settings.DATABASE_URL, future=True)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            if args.clean:
                user = (await session.execute(
                    select(User).where(User.email == ME))).scalar_one_or_none()
                if user:
                    await wipe(session, user.id)
                    await session.commit()
                    print("  cleaned")
            user_id, counts = await seed(session, real_embeddings=real)
    finally:
        await engine.dispose()

    report(user_id, counts, real)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
