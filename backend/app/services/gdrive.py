"""Google Drive, as this system needs it.

Drive's problem is not the API, it is the content. A file is a pointer; what
makes "the Acme proposal from last month" findable is the text inside it, and
every format gets there a different way. A Google Doc has no bytes to download
— it has to be *exported*. A PDF has bytes and no text until something reads
it. A plain text file just downloads.

So this module has a metadata half and a content half. :meth:`DriveService.text_for`
is the content half: it picks the right route for the mime type and gives back
a string, capped at :data:`EXCERPT_LIMIT` characters, because that string is
what gets embedded and no embedding is improved by the ninetieth page of an
appendix.

``folder_path`` is built by walking ``parents`` upward, with a per-instance
cache — the same three folders are the parents of everything a person owns, so
a backfill of two thousand files makes a handful of extra calls, not two
thousand.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import quote

from app.core.errors import AppError
from app.core.logging import get_logger
from app.google.client import SAFE_TO_REPEAT, Transport
from app.google.retry import ErrorClass, GoogleAPIError

log = get_logger(__name__)

#: The text we keep per file. Enough to answer "what is in it"; the chunker
#: splits it, and nothing downstream wants more.
EXCERPT_LIMIT: Final[int] = 8000

#: Google's own formats, and the plain text each exports as.
EXPORT_AS: Final[dict[str, str]] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.script": "application/vnd.google-apps.script+json",
}

MIME_FOLDER: Final[str] = "application/vnd.google-apps.folder"
MIME_PDF: Final[str] = "application/pdf"

#: What we ask for on every file. Asking for `*` doubles the payload and slows
#: a listing down for fields nothing reads.
FILE_FIELDS: Final[str] = (
    "id,name,mimeType,owners(emailAddress,displayName),shared,webViewLink,"
    "parents,size,modifiedTime,createdTime,trashed,driveId,starred"
)
LIST_FIELDS: Final[str] = f"nextPageToken,incompleteSearch,files({FILE_FIELDS})"
CHANGE_FIELDS: Final[str] = (
    f"nextPageToken,newStartPageToken,changes(changeType,removed,fileId,time,file({FILE_FIELDS}))"
)

DEFAULT_PAGE_SIZE: Final[int] = 100


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)


def parse_file(item: dict[str, Any]) -> dict[str, Any]:
    """One Drive file, flattened into the shape ``sync_gdrive`` stores."""
    owners = item.get("owners") or []
    owner = owners[0] if owners else {}
    size = item.get("size")
    return {
        "file_id": item.get("id"),
        "name": item.get("name") or "",
        "mime_type": item.get("mimeType"),
        "owner_email": (owner.get("emailAddress") or "").lower() or None,
        "owner_name": owner.get("displayName"),
        "is_shared": bool(item.get("shared", False)),
        "web_view_link": item.get("webViewLink"),
        "parents": list(item.get("parents") or []),
        "size_bytes": int(size) if size not in (None, "") else None,
        "modified_at": _parse_time(item.get("modifiedTime")),
        "created_at": _parse_time(item.get("createdTime")),
        "trashed": bool(item.get("trashed", False)),
        "is_folder": item.get("mimeType") == MIME_FOLDER,
        "drive_id": item.get("driveId"),
    }


def build_query(
    *,
    name_contains: str | None = None,
    full_text: str | None = None,
    mime_type: str | Sequence[str] | None = None,
    owner: str | None = None,
    shared_with_me: bool | None = None,
    parent: str | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    trashed: bool | None = False,
    starred: bool | None = None,
    extra: str | None = None,
) -> str:
    """Drive's query language, assembled from arguments rather than by hand.

    Single quotes inside a term are escaped, which is the whole of Drive's
    injection surface.
    """
    clauses: list[str] = []

    def esc(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    if name_contains:
        clauses.append(f"name contains '{esc(name_contains)}'")
    if full_text:
        clauses.append(f"fullText contains '{esc(full_text)}'")
    if mime_type:
        kinds = [mime_type] if isinstance(mime_type, str) else list(mime_type)
        joined = " or ".join(f"mimeType = '{esc(k)}'" for k in kinds)
        clauses.append(f"({joined})" if len(kinds) > 1 else joined)
    if owner:
        clauses.append(f"'{esc(owner)}' in owners")
    if shared_with_me is not None:
        clauses.append("sharedWithMe" if shared_with_me else "not sharedWithMe")
    if parent:
        clauses.append(f"'{esc(parent)}' in parents")
    if modified_after:
        clauses.append(f"modifiedTime > '{_rfc3339(modified_after)}'")
    if modified_before:
        clauses.append(f"modifiedTime < '{_rfc3339(modified_before)}'")
    if trashed is not None:
        clauses.append("trashed = true" if trashed else "trashed = false")
    if starred is not None:
        clauses.append("starred = true" if starred else "starred = false")
    if extra:
        clauses.append(f"({extra})")
    return " and ".join(clauses)


def _rfc3339(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #


def clean_text(text: str, limit: int = EXCERPT_LIMIT) -> str:
    """Tidy extracted text and cap it at ``limit`` characters, on a word."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut[-120:] else cut).rstrip() + " ..."


def pdf_text(data: bytes, limit: int = EXCERPT_LIMIT) -> str:
    """Text out of a PDF, stopping as soon as there is enough of it.

    Pages are read one at a time and the loop breaks at the cap, so a
    four-hundred-page contract costs the same as a four-page one.
    """
    if not data:
        return ""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is a hard dependency
        log.warning("gdrive.pypdf_missing")
        return ""

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # encrypted, truncated, not really a PDF
        log.warning("gdrive.pdf_unreadable", error=str(exc)[:200])
        return ""

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # many PDFs are "encrypted" with an empty password
        except Exception:
            log.info("gdrive.pdf_encrypted")
            return ""

    chunks: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            piece = page.extract_text() or ""
        except Exception:
            continue
        if not piece:
            continue
        chunks.append(piece)
        total += len(piece)
        if total >= limit:
            break
    return clean_text("\n\n".join(chunks), limit)


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class DriveService:
    """Drive bound to one user."""

    service = "gdrive"

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        #: file_id -> (name, parent_id). Folders barely change inside one run.
        self._folders: dict[str, tuple[str, str | None]] = {}

    @property
    def user_id(self) -> str:
        return self.transport.user_id

    # -- metadata ----------------------------------------------------------- #

    async def files_list(
        self,
        *,
        q: str | None = None,
        page_token: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        order_by: str | None = None,
        fields: str = LIST_FIELDS,
        spaces: str = "drive",
        include_shared_drives: bool = True,
        corpora: str | None = None,
    ) -> dict[str, Any]:
        """One page of files."""
        page = await self.transport.get(
            "files",
            api_method="gdrive.files.list",
            params={
                "q": q or None,
                "pageToken": page_token,
                "pageSize": max(1, min(int(page_size), 1000)),
                "orderBy": order_by,
                "fields": fields,
                "spaces": spaces,
                "includeItemsFromAllDrives": include_shared_drives,
                "supportsAllDrives": include_shared_drives,
                "corpora": corpora or ("allDrives" if include_shared_drives else None),
            },
        )
        return {
            "files": [parse_file(f) for f in page.get("files") or []],
            "next_page_token": page.get("nextPageToken"),
            "incomplete": bool(page.get("incompleteSearch", False)),
        }

    async def files_get(
        self, file_id: str, *, fields: str = FILE_FIELDS
    ) -> dict[str, Any] | None:
        """One file's metadata. ``None`` when it is gone or not shared with us."""
        try:
            raw = await self.transport.get(
                f"files/{_esc(file_id)}",
                api_method="gdrive.files.get",
                params={"fields": fields, "supportsAllDrives": True},
            )
        except GoogleAPIError as exc:
            if exc.error_class is ErrorClass.NOT_FOUND:
                return None
            raise
        return parse_file(raw)

    async def folder_path(self, file: dict[str, Any], *, max_depth: int = 8) -> str:
        """``/Clients/Acme/2026`` for a file, by walking its parents.

        Drive has no path — only a parent list — so this is the only way to
        show someone where a file lives. Cached per instance: the same folders
        are the parents of nearly everything.
        """
        parents = file.get("parents") or []
        if not parents:
            return "/"
        names: list[str] = []
        current: str | None = parents[0]
        for _ in range(max_depth):
            if not current:
                break
            cached = self._folders.get(current)
            if cached is None:
                folder = await self.files_get(current, fields="id,name,parents")
                if folder is None:
                    break
                cached = (folder["name"], (folder.get("parents") or [None])[0])
                self._folders[current] = cached
            name, parent = cached
            if name in {"My Drive", "Drive"} and parent is None:
                break
            names.append(name)
            current = parent
        return "/" + "/".join(reversed(names)) if names else "/"

    # -- content ------------------------------------------------------------ #

    async def files_export(
        self, file_id: str, *, mime_type: str = "text/plain", limit: int = EXCERPT_LIMIT
    ) -> str:
        """Export a Google-native file as text.

        Docs, Sheets and Slides have no bytes of their own — Google renders
        them on request, which is why this costs more quota than a read.
        """
        result = await self.transport.get(
            f"files/{_esc(file_id)}/export",
            api_method="gdrive.files.export",
            params={"mimeType": mime_type},
        )
        return clean_text(_as_text(result), limit)

    async def files_download(self, file_id: str, *, max_bytes: int = 10_000_000) -> bytes:
        """The file's own bytes. Used for PDFs and plain text."""
        data = await self.transport.get(
            f"files/{_esc(file_id)}",
            api_method="gdrive.files.download",
            params={"alt": "media", "supportsAllDrives": True},
            expect="bytes",
        )
        if isinstance(data, bytes) and len(data) > max_bytes:
            return data[:max_bytes]
        return data if isinstance(data, bytes) else str(data).encode("utf-8", "replace")

    async def text_for(
        self, file: dict[str, Any] | str, *, limit: int = EXCERPT_LIMIT
    ) -> str:
        """The searchable text of a file, whatever kind of file it is.

        Unknown binary formats return an empty string rather than an error: a
        file we cannot read is still a file worth listing by name.
        """
        if isinstance(file, str):
            fetched = await self.files_get(file)
            if fetched is None:
                return ""
            file = fetched

        file_id = file.get("file_id") or file.get("id")
        mime = str(file.get("mime_type") or file.get("mimeType") or "")
        if not file_id or mime == MIME_FOLDER:
            return ""

        try:
            if mime in EXPORT_AS:
                return await self.files_export(
                    file_id, mime_type=EXPORT_AS[mime], limit=limit
                )
            if mime == MIME_PDF:
                return pdf_text(await self.files_download(file_id), limit)
            if mime.startswith("text/") or mime in {
                "application/json",
                "application/xml",
                "application/rtf",
            }:
                raw = await self.files_download(file_id, max_bytes=limit * 8)
                return clean_text(raw.decode("utf-8", errors="replace"), limit)
        except GoogleAPIError as exc:
            if exc.error_class in {ErrorClass.NOT_FOUND, ErrorClass.AUTH_REVOKED}:
                log.info(
                    "gdrive.text_unavailable",
                    file_id=file_id,
                    error_class=str(exc.error_class),
                )
                return ""
            raise
        return ""

    # -- changes ------------------------------------------------------------ #

    async def changes_start_page_token(self, *, drive_id: str | None = None) -> str:
        """Where an incremental sync starts. Taken before the backfill, so
        nothing that changes during it is missed."""
        result = await self.transport.get(
            "changes/startPageToken",
            api_method="gdrive.changes.getStartPageToken",
            params={"supportsAllDrives": True, "driveId": drive_id},
        )
        return str(result.get("startPageToken") or "")

    async def changes_list(
        self,
        page_token: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        include_removed: bool = True,
        restrict_to_my_drive: bool = False,
    ) -> dict[str, Any]:
        """What changed since ``page_token``.

        Removals come back as ids with no file, so they are separated here:
        the caller deletes those from the mirror and upserts the rest.
        """
        page = await self.transport.get(
            "changes",
            api_method="gdrive.changes.list",
            params={
                "pageToken": page_token,
                "pageSize": max(1, min(int(page_size), 1000)),
                "includeRemoved": include_removed,
                "restrictToMyDrive": restrict_to_my_drive,
                "fields": CHANGE_FIELDS,
                "spaces": "drive",
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            },
        )

        changed: list[dict[str, Any]] = []
        removed: list[str] = []
        for change in page.get("changes") or []:
            file_id = change.get("fileId")
            item = change.get("file")
            if change.get("removed") or not item or item.get("trashed"):
                if file_id:
                    removed.append(file_id)
                continue
            changed.append(parse_file(item))

        return {
            "changed": changed,
            "removed": list(dict.fromkeys(removed)),
            "next_page_token": page.get("nextPageToken"),
            "new_start_page_token": page.get("newStartPageToken"),
        }

    # -- writing ------------------------------------------------------------ #

    async def permissions_create(
        self,
        file_id: str,
        *,
        email: str | None = None,
        role: str = "reader",
        type: str = "user",
        domain: str | None = None,
        send_notification: bool = True,
        message: str | None = None,
        transfer_ownership: bool = False,
    ) -> dict[str, Any]:
        """Share a file. One of the irreversible ones, so it is never speculative."""
        body: dict[str, Any] = {"role": role, "type": type}
        if email:
            body["emailAddress"] = email
        if domain:
            body["domain"] = domain
        created = await self.transport.post(
            f"files/{_esc(file_id)}/permissions",
            api_method="gdrive.permissions.create",
            params={
                "sendNotificationEmail": send_notification if email else False,
                "emailMessage": message if send_notification else None,
                "transferOwnership": transfer_ownership or None,
                "supportsAllDrives": True,
                "fields": "id,type,role,emailAddress,domain",
            },
            json=body,
            # A share that may have landed is never repeated blind: only the
            # failures that prove Google turned it away are retried.
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )
        return {
            "permission_id": created.get("id"),
            "file_id": file_id,
            "role": created.get("role", role),
            "type": created.get("type", type),
            "email": created.get("emailAddress", email),
        }

    async def permissions_list(self, file_id: str) -> list[dict[str, Any]]:
        page = await self.transport.get(
            f"files/{_esc(file_id)}/permissions",
            api_method="gdrive.permissions.list",
            params={
                "supportsAllDrives": True,
                "fields": "permissions(id,type,role,emailAddress,domain,displayName)",
            },
        )
        return page.get("permissions") or []

    async def files_update(
        self,
        file_id: str,
        *,
        body: dict[str, Any] | None = None,
        add_parents: Sequence[str] | str | None = None,
        remove_parents: Sequence[str] | str | None = None,
        fields: str = FILE_FIELDS,
    ) -> dict[str, Any]:
        """Rename a file, or move it by changing its parents.

        Drive has no move: a move is a metadata update that adds one parent and
        removes another, which is why both lists are here.
        """

        def _join(value: Sequence[str] | str | None) -> str | None:
            if not value:
                return None
            return value if isinstance(value, str) else ",".join(value)

        updated = await self.transport.patch(
            f"files/{_esc(file_id)}",
            api_method="gdrive.files.update",
            params={
                "addParents": _join(add_parents),
                "removeParents": _join(remove_parents),
                "supportsAllDrives": True,
                "fields": fields,
            },
            json=body or {},
        )
        return parse_file(updated)


    async def files_create(
        self,
        *,
        name: str,
        mime_type: str = MIME_FOLDER,
        parent_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a file or a folder. A folder is a file with a folder mime type."""
        payload: dict[str, Any] = {"name": name, "mimeType": mime_type, **(body or {})}
        if parent_id:
            payload["parents"] = [parent_id]
        created = await self.transport.post(
            "files",
            api_method="gdrive.files.create",
            params={"supportsAllDrives": True, "fields": FILE_FIELDS},
            json=payload,
            retry_on=SAFE_TO_REPEAT,
            retry_on_network=False,
        )
        return parse_file(created)

    # -- the shape the ops layer asks for ----------------------------------- #

    async def list_files(
        self,
        *,
        query: str | None = None,
        mime_type: str | Sequence[str] | None = None,
        max_results: int = 25,
        order_by: str | None = "modifiedTime desc",
    ) -> list[dict[str, Any]]:
        """Files matching free text and an optional mime type.

        The text goes to ``fullText contains``, which is Drive's own index over
        names and content — the fallback for when our mirror has not caught up.
        """
        drive_query = build_query(full_text=query or None, mime_type=mime_type)
        page = await self.files_list(
            q=drive_query or None, page_size=max_results, order_by=order_by
        )
        return page["files"]

    async def get_file(self, file_id: str) -> dict[str, Any] | None:
        return await self.files_get(file_id)

    async def create_folder(
        self,
        *,
        name: str,
        parent_id: str | None = None,
        mime_type: str = MIME_FOLDER,
    ) -> dict[str, Any]:
        return await self.files_create(name=name, mime_type=mime_type, parent_id=parent_id)

    async def create_file(
        self,
        *,
        name: str,
        mime_type: str = MIME_FOLDER,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.files_create(name=name, mime_type=mime_type, parent_id=parent_id)

    async def move_file(
        self,
        *,
        file_id: str,
        add_parents: Sequence[str] | str | None = None,
        remove_parents: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Move a file between folders.

        When the caller does not say what to remove, the file's current parents
        are removed — otherwise "move it to Clients" would leave it in both
        places, which is not what anybody means by move.
        """
        if not remove_parents:
            current = await self.files_get(file_id, fields="id,parents")
            remove_parents = (current or {}).get("parents") or None
        return await self.files_update(
            file_id, add_parents=add_parents, remove_parents=remove_parents
        )

    async def share_file(
        self,
        *,
        file_id: str,
        email: str | None = None,
        role: str = "reader",
        notify: bool = True,
        message: str | None = None,
        expires_at: Any = None,
        type: str = "user",
    ) -> dict[str, Any]:
        """Give someone access. ``expires_at`` is ignored on personal accounts,
        which is where Google, not us, draws the line."""
        return await self.permissions_create(
            file_id,
            email=email,
            role=role,
            type=type,
            send_notification=notify,
            message=message,
        )


# --------------------------------------------------------------------------- #
# Raw-payload functions, for the sync worker
# --------------------------------------------------------------------------- #


def _transport(clients: Any) -> Transport:
    """The Drive transport out of whatever the caller is holding."""
    if isinstance(clients, Transport):
        return clients
    service = getattr(clients, "gdrive", None) or getattr(clients, "drive", clients)
    transport = getattr(service, "transport", None)
    if isinstance(transport, Transport):
        return transport
    raise AppError.internal(
        "Expected Google clients with a drive transport.",
        got=type(clients).__name__,
    )


#: The name the sync task uses for :func:`parse_file`.
normalise_file = parse_file


async def files_list(
    clients: Any,
    *,
    query: str | None = None,
    page_token: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    order_by: str | None = None,
    fields: str = LIST_FIELDS,
) -> dict[str, Any]:
    """Raw ``files.list`` page."""
    return await _transport(clients).get(
        "files",
        api_method="gdrive.files.list",
        params={
            "q": query or None,
            "pageToken": page_token,
            "pageSize": max(1, min(int(page_size), 1000)),
            "orderBy": order_by,
            "fields": fields,
            "spaces": "drive",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "corpora": "allDrives",
        },
    )


async def changes_list(
    clients: Any,
    *,
    page_token: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    include_removed: bool = True,
) -> dict[str, Any]:
    """Raw ``changes.list`` page."""
    return await _transport(clients).get(
        "changes",
        api_method="gdrive.changes.list",
        params={
            "pageToken": page_token,
            "pageSize": max(1, min(int(page_size), 1000)),
            "includeRemoved": include_removed,
            "fields": CHANGE_FIELDS,
            "spaces": "drive",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        },
    )


async def start_page_token(clients: Any) -> str:
    """The change cursor to start from. Taken before a backfill, not after."""
    result = await _transport(clients).get(
        "changes/startPageToken",
        api_method="gdrive.changes.getStartPageToken",
        params={"supportsAllDrives": True},
    )
    return str(result.get("startPageToken") or "")


async def extract_text(clients: Any, file: dict[str, Any], *, limit: int = EXCERPT_LIMIT) -> str:
    """The searchable text of a raw Drive file. Empty when there is none."""
    service = getattr(clients, "gdrive", None) or getattr(clients, "drive", None)
    if service is not None and hasattr(service, "text_for"):
        return await service.text_for(file, limit=limit)
    return await DriveService(_transport(clients)).text_for(file, limit=limit)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _esc(value: str) -> str:
    return quote(str(value), safe="")


def _as_text(result: Any) -> str:
    """An export body as text, whatever shape the transport handed back."""
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, dict):
        if "_text" in result:
            return str(result["_text"])
        from app.core.ids import canonical_json

        return canonical_json(result)
    return str(result or "")


__all__ = [
    "DriveService",
    "EXCERPT_LIMIT",
    "changes_list",
    "extract_text",
    "files_list",
    "normalise_file",
    "start_page_token",
    "EXPORT_AS",
    "FILE_FIELDS",
    "LIST_FIELDS",
    "MIME_FOLDER",
    "MIME_PDF",
    "build_query",
    "clean_text",
    "parse_file",
    "pdf_text",
]
