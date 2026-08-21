"""Operator scripts. Run them, do not import them from the app.

Everything here is a `python -m scripts.<name>` entry point: it exists so a
person can do a thing once — export the OpenAPI document, fill a local
database, plant the demo data in a real Google account. None of it is on the
request path, and nothing in `app/` imports from here.

The package exists for two reasons. `python -m scripts.seed_demo_account`
needs it, and it makes `from scripts.x import y` work inside a test without a
path hack.
"""

from __future__ import annotations

__all__: list[str] = []
