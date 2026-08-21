"""The keys inside `sync_state.cursor`, in one place.

`docs/schema.md` fixes the spelling: the cursor "holds whatever the service uses
— Gmail's ``historyId``, Calendar's ``syncToken``, Drive's ``pageToken``". So
what goes in the column is Google's own name for the token, not a translated
one, and a row can be read against Google's documentation without a decoder.

Reading is forgiving and writing is not. A database that predates this module
has snake_case keys in it, and a sync that silently decided those rows had no
cursor would walk the whole mailbox again and call it an incremental pass —
which is exactly the bug this module was written to fix. So :func:`get` accepts
either spelling, and the writers use the constants below.
"""

from __future__ import annotations

from typing import Any, Final

#: Gmail: the mailbox history id an incremental pass resumes from.
HISTORY_ID: Final[str] = "historyId"
#: Calendar: the opaque token `events.list` hands back.
SYNC_TOKEN: Final[str] = "syncToken"
#: Drive, and any half-finished page walk: where to carry on.
PAGE_TOKEN: Final[str] = "pageToken"

#: The spelling each key used before `docs/schema.md` settled the question.
_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    HISTORY_ID: ("history_id",),
    SYNC_TOKEN: ("sync_token",),
    PAGE_TOKEN: ("page_token",),
}


def get(cursor: dict[str, Any] | None, key: str) -> Any:
    """One cursor value, under the contract's name or a legacy one.

    Returns ``None`` for a missing key and for an empty string, so a caller can
    write ``if not cursors.get(...)`` and mean "there is nothing to resume
    from" — an empty token is not a resumable position.
    """
    if not cursor:
        return None
    for name in (key, *_ALIASES.get(key, ())):
        value = cursor.get(name)
        if value not in (None, ""):
            return value
    return None


def with_value(key: str, value: Any) -> dict[str, Any]:
    """A one-key cursor, or an empty dict when there is nothing to store."""
    return {key: value} if value not in (None, "") else {}


__all__ = ["HISTORY_ID", "SYNC_TOKEN", "PAGE_TOKEN", "get", "with_value"]
