"""Regex extractors over the probe's candidate excerpts.

These run in about two milliseconds, with no model involved, and they are the
reason the planner never has to retype a booking reference — it writes
`{{search.gmail[0].extracted.pnr}}` and the real value is bound at dispatch
time. A value that was never typed cannot be hallucinated.

They are also the strongest argument for matching structure rather than
vocabulary. A PNR is a shape, and shapes survive translation: the Turkish
booking confirmation in `docs/SAMPLE_QUERIES.md` §12 gives up its record
locator, flight number, route, date and fare to exactly the same expressions
that read the English one, at a point where both the vector arm and the
`'english'` full-text arm have already failed.

The negative cases matter as much as the positive ones. Six upper-case
characters in a row is a common shape — a room number, a coupon code, a git
ref — and an extractor that grabs every one of them fills the planner's
briefing with rubbish it will confidently reference.
"""

from __future__ import annotations

import pytest

from tests.conftest import FROZEN_NOW, call, load_any

extract = load_any(
    ["app.search.extractors", "app.orchestrator.entities", "app.search.probe", "app.orchestrator.prepass"],
    ["extract", "extract_all", "run_extractors", "extract_entities"],
)


# ---------------------------------------------------------------------------
# Reading the result, whatever it decided to call things
# ---------------------------------------------------------------------------


def run(text: str, subject: str | None = None) -> dict:
    result = call(extract, text, subject=subject, now=FROZEN_NOW)
    assert isinstance(result, dict), f"expected a dict of extracted values, got {result!r}"
    return result


def values(result: dict, *keys: str) -> list[str]:
    """Every value stored under any of `keys`, flattened to strings."""
    out: list[str] = []
    for key, value in result.items():
        if key not in keys:
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out.extend(str(v) for v in value)
        else:
            out.append(str(value))
    return out


PNR_KEYS = ("pnr", "booking_ref", "booking_reference", "record_locator", "pnrs")
FLIGHT_KEYS = ("flight_no", "flight_nos", "flight_number", "flight_numbers", "flights")
TICKET_KEYS = ("ticket_no", "ticket_number", "ticket", "tickets")
ROUTE_KEYS = ("route", "routes", "itinerary")
DATE_KEYS = ("dates", "date", "depart_at", "departs_at", "iso_dates", "datetimes")
AMOUNT_KEYS = ("amount", "amounts", "total", "fare", "price")
EMAIL_KEYS = ("emails", "email", "support_email", "email_addresses", "addresses")
DRIVE_KEYS = ("doc_links", "drive_file_ids", "drive_links", "drive_files", "drive_urls")


def has(result: dict, keys: tuple[str, ...], wanted: str) -> bool:
    return any(wanted in v for v in values(result, *keys))


# ---------------------------------------------------------------------------
# The English booking confirmation, docs/SAMPLE_QUERIES.md §4
# ---------------------------------------------------------------------------

ENGLISH_SUBJECT = "Your Turkish Airlines booking is confirmed — TK1984"
ENGLISH_BODY = """\
Dear Passenger,

Your booking reference 6F2QK9 is confirmed.
Flight TK1, Istanbul (IST) -> New York (JFK), 5 September 2026, 10:30.
Ticket number TK1984. Total: USD 812.40.

To change or cancel, write to cancel@turkishairlines.com or call +90 212 444 0849.
"""


@pytest.fixture
def english():
    return run(ENGLISH_BODY, subject=ENGLISH_SUBJECT)


def test_the_pnr_comes_out_of_an_english_booking(english):
    assert has(english, PNR_KEYS, "6F2QK9")


def test_the_flight_number_comes_out(english):
    assert has(english, FLIGHT_KEYS, "TK1")


def test_the_ticket_number_is_not_confused_with_the_flight_number(english):
    # Both codes match `\b[A-Z]{2}\s?\d{1,4}\b`, and docs/SAMPLE_QUERIES.md §4
    # keeps them apart: flight TK1, ticket TK1984. The ticket number is the one
    # a cancellation quotes, and the flight number is the one that identifies
    # the leg — swapping them puts the wrong code in a draft email.
    ticket = " ".join(values(english, *TICKET_KEYS))
    flight = " ".join(values(english, "flight_no", "flight_number", "flight"))
    assert "TK1984" in ticket, english
    assert flight == "TK1", f"flight_no should be TK1, not the ticket number: {flight!r}"


def test_the_route_comes_out(english):
    route = " ".join(values(english, *ROUTE_KEYS))
    assert "IST" in route and "JFK" in route, english


def test_the_support_address_comes_out(english):
    assert has(english, EMAIL_KEYS, "cancel@turkishairlines.com")


def test_the_fare_comes_out(english):
    assert has(english, AMOUNT_KEYS, "812.40")


def test_the_departure_date_comes_out(english):
    assert has(english, DATE_KEYS, "2026-09-05")


# ---------------------------------------------------------------------------
# The Turkish booking confirmation, docs/SAMPLE_QUERIES.md §12
# ---------------------------------------------------------------------------
#
# Same booking, same shapes, no English anywhere. Neither the embedding nor the
# 'english' text search arm can see this email; the extractors read it fine.

TURKISH_SUBJECT = "Uçuş rezervasyonunuz onaylandı — TK1984"
TURKISH_BODY = """\
Sayın Yolcumuz, 6F2QK9 numaralı rezervasyonunuz onaylanmıştır.
Uçuş: TK1, İstanbul (IST) → New York (JFK), 5 Eylül 2026, 10:30.
Toplam: 812,40 USD.
"""


@pytest.fixture
def turkish():
    return run(TURKISH_BODY, subject=TURKISH_SUBJECT)


def test_the_pnr_survives_translation(turkish):
    # "numaralı" is the booking keyword here instead of "reference". The shape
    # is unchanged, which is the entire argument for doing this with regexes.
    assert has(turkish, PNR_KEYS, "6F2QK9")


def test_the_flight_number_survives_translation(turkish):
    assert has(turkish, FLIGHT_KEYS, "TK1")


def test_the_route_survives_translation(turkish):
    # An arrow, not a hyphen, and a Turkish city name in front of the code.
    route = " ".join(values(turkish, *ROUTE_KEYS))
    assert "IST" in route and "JFK" in route, turkish


def test_a_turkish_month_name_still_gives_an_iso_date(turkish):
    # "5 Eylül 2026" is 5 September 2026. The month lexicon is hand-maintained
    # alongside the vendor alias table.
    assert has(turkish, DATE_KEYS, "2026-09-05")


def test_a_comma_decimal_fare_is_normalised(turkish):
    # "812,40 USD" is eight hundred and twelve, not eighty-one thousand.
    amounts = " ".join(values(turkish, *AMOUNT_KEYS))
    assert "812.40" in amounts, amounts
    # Left as written it would be read as a thousands separator, and the fare on
    # the confirm card would say eighty-one thousand.
    assert "81,240" not in amounts and "81240" not in amounts, amounts


def test_both_languages_give_the_same_booking(english, turkish):
    assert values(english, *PNR_KEYS) == values(turkish, *PNR_KEYS)


# ---------------------------------------------------------------------------
# Negative cases — the six-character token that is not a booking reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "See you in room A1B2C3 at noon.",
        "The build is green at commit 9F3A21, ready when you are.",
        "Use coupon SAVE20 at checkout for the conference ticket.",
        "URGENT: the Q3 numbers are wrong in the deck.",
        "Meeting ID 8H2K4M — dial in five minutes early.",
    ],
)
def test_a_six_character_token_with_no_booking_word_near_it_is_not_a_pnr(text):
    # `\b[A-Z0-9]{6}\b` on its own matches all of these. The proximity
    # requirement is what makes the expression useful rather than noisy.
    result = run(text)
    assert not values(result, *PNR_KEYS), (text, result)


def test_a_booking_word_far_away_does_not_rescue_a_token():
    # Same words, a paragraph apart. Proximity means proximity.
    text = (
        "Your booking is confirmed and we look forward to seeing you.\n\n"
        + ("Filler sentence about baggage allowance. " * 20)
        + "\nSee you in room A1B2C3 at noon."
    )
    result = run(text)
    assert not has(result, PNR_KEYS, "A1B2C3")


@pytest.mark.parametrize(
    "text",
    [
        "Our Q3 revenue target is up 12 percent.",
        "The TK office will be closed on Monday.",
        "Invoice number 2026-114 is attached.",
    ],
)
def test_two_letters_and_a_number_is_not_always_a_flight(text):
    result = run(text)
    assert not values(result, *FLIGHT_KEYS), (text, result)


def test_a_room_number_is_not_an_amount():
    result = run("We are in room 812 on the fourth floor.")
    assert not values(result, *AMOUNT_KEYS), result


def test_an_at_mention_is_not_an_email_address():
    result = run("cc @sarah and @finance on the reply, please.")
    assert not values(result, *EMAIL_KEYS), result


def test_an_address_with_nothing_after_the_at_is_not_extracted():
    result = run("Write to priya@ and I will forward it.")
    assert not any(v.startswith("priya@") for v in values(result, *EMAIL_KEYS))


# ---------------------------------------------------------------------------
# Email addresses
# ---------------------------------------------------------------------------


def test_ordinary_addresses_are_extracted():
    result = run(
        "Loop in sarah@company.com and finance.team@acme.co.uk; "
        "the vendor is billing+eu@northwind.io."
    )
    found = " ".join(values(result, *EMAIL_KEYS))
    assert "sarah@company.com" in found
    assert "finance.team@acme.co.uk" in found
    assert "billing+eu@northwind.io" in found


def test_trailing_punctuation_is_not_part_of_the_address():
    result = run("Send it to sarah@company.com.")
    assert "sarah@company.com." not in values(result, *EMAIL_KEYS)
    assert has(result, EMAIL_KEYS, "sarah@company.com")


# ---------------------------------------------------------------------------
# Drive links
# ---------------------------------------------------------------------------

DOC_ID = "1A2b3C4d5E6f7G8h9I0jKlMnOpQrStUvWxYz"
SHEET_ID = "1ZyXwVuTsRqPoNmLkJiHgFeDcBa9876543210"


def test_a_drive_document_link_is_extracted():
    text = f"Agenda is here: https://docs.google.com/document/d/{DOC_ID}/edit?usp=sharing"
    result = run(text)
    found = " ".join(values(result, *DRIVE_KEYS))
    assert DOC_ID in found, result


def test_a_drive_spreadsheet_link_is_extracted():
    text = f"Numbers: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=0"
    result = run(text)
    assert SHEET_ID in " ".join(values(result, *DRIVE_KEYS)), result


def test_a_drive_file_open_link_is_extracted():
    text = f"The signed PDF: https://drive.google.com/file/d/{DOC_ID}/view"
    result = run(text)
    assert DOC_ID in " ".join(values(result, *DRIVE_KEYS)), result


def test_two_drive_links_both_come_out():
    text = (
        f"https://docs.google.com/document/d/{DOC_ID}/edit and "
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
    )
    found = " ".join(values(run(text), *DRIVE_KEYS))
    assert DOC_ID in found and SHEET_ID in found


def test_an_ordinary_link_is_not_a_drive_link():
    # `sync_gcal` has no attachments column, so a Drive link in the description
    # is how "does this meeting have an agenda" gets answered. A blog post is
    # not an agenda.
    result = run("Background reading: https://example.com/blog/quarterly-planning")
    assert not any("example.com" in v for v in values(result, *DRIVE_KEYS))


def test_a_google_search_link_is_not_a_drive_link():
    result = run("https://www.google.com/search?q=turkish+airlines+refund+policy")
    assert not values(result, *DRIVE_KEYS)


# ---------------------------------------------------------------------------
# Dates and amounts, on their own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The deadline is 2026-09-05.",
        "The deadline is 5 September 2026.",
        "The deadline is September 5, 2026.",
    ],
)
def test_a_date_comes_out_as_iso(text):
    assert has(run(text), DATE_KEYS, "2026-09-05")


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Total: USD 812.40", "812.40"),
        ("Total: $1,240.00", "1,240.00"),
        ("Total: 812,40 USD", "812.40"),
        ("Refund of EUR 99.99 processed", "99.99"),
    ],
)
def test_amounts_are_normalised(text, wanted):
    amounts = " ".join(values(run(text), *AMOUNT_KEYS))
    assert wanted in amounts, (text, amounts)


def test_an_empty_excerpt_gives_nothing_rather_than_failing():
    # Excerpts come from the mirror, and `content_excerpt` is nullable.
    result = run("")
    assert isinstance(result, dict)
    assert not any(values(result, *PNR_KEYS, *FLIGHT_KEYS, *AMOUNT_KEYS, *EMAIL_KEYS))
