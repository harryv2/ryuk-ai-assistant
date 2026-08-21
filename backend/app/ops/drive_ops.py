"""Drive ops: two reads, two plain writes, one gated one.

`drive.create_folder` and `drive.move_file` are writes with no confirm — a
folder can be deleted and a file can be moved back, and asking about every
tidy-up is how a person learns to click yes without reading.

`drive.share_file` is a `ConfirmableOp` and it is the only one here, because
granting access is the one thing on this service you cannot take back: whoever
you shared it with may already have opened it. `run` resolves the file and the
grantee so the card can name both; `execute` creates the permission.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import AppError
from app.core.ids import fingerprint
from app.db.repositories import mirror as mirror_repo
from app.ops.base import (
    ConfirmableOp,
    Op,
    OpContext,
    OpResult,
    SearchOp,
    as_list,
    excerpt_of,
    google_call,
    has_google,
    iso,
    jsonable,
    row_to_dict,
    run_extractors,
    trim_for_llm,
)

FOLDER_MIME = "application/vnd.google-apps.folder"

#: Words people use for a file type, and the mime they mean by them.
MIME_WORDS: dict[str, str] = {
    "pdf": "application/pdf",
    "doc": "application/vnd.google-apps.document",
    "docs": "application/vnd.google-apps.document",
    "document": "application/vnd.google-apps.document",
    "sheet": "application/vnd.google-apps.spreadsheet",
    "spreadsheet": "application/vnd.google-apps.spreadsheet",
    "slides": "application/vnd.google-apps.presentation",
    "deck": "application/vnd.google-apps.presentation",
    "folder": FOLDER_MIME,
}

DRIVE_FILTERS: dict[str, str | None] = {
    "mime_type": "mime_type",
    "mime_types": "mime_types[]",
    "owner_email": "owner_email",
    "is_shared": "bool:is_shared",
    "file_ids": "file_ids[]",
    "folder_prefix": "folder_prefix",
    "name_contains": "name_contains",
    "since": "since",
    "until": "until",
    "window": "window",
    "modified_window": "window",
    "min_cn": None,
    "exclude_refs": None,
    "text_contains": None,
}

DRIVE_ORDER: dict[str, str] = {
    "relevance": "cn",
    "modified_at": "modified_at",
    "modified": "modified_at",
    "name": "label",
    "size_bytes": "size_bytes",
}


class SearchFiles(SearchOp):
    """Files out of the mirror, matched on name and on the indexed excerpt."""

    name = "drive.search_files"
    corpus = "gdrive"
    entity_type = "file"
    filter_spec = DRIVE_FILTERS
    order_spec = DRIVE_ORDER
    summary = "search your Drive (mirror, not Google)"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        # "the PDFs from last month" arrives as a word, not a mime type.
        raw = dict(args or {})
        consumed: set[str] = set()
        for holder in (raw, raw.get("filter") or {}):
            word = holder.get("mime_type")
            if isinstance(word, str) and "/" not in word:
                key = word.strip().lower()
                mime = MIME_WORDS.get(key)
                if mime:
                    consumed.add(key)
                holder["mime_type"] = mime or word

        # A type word that became a filter must leave the query text.
        #
        # No PDF's *content* says "PDF", so searching for the word finds
        # nothing and the whole result is empty before the mime filter gets a
        # say — "show me PDFs from last month" returns none of the five PDFs
        # sitting in last month. The word was a description of the file, not
        # of what is written in it, and it has already been honoured.
        has_mime = any(
            (holder or {}).get("mime_type") for holder in (raw, raw.get("filter") or {})
        )
        query = str(raw.get("query") or "")
        if has_mime and query:
            # Strip against the whole vocabulary, not just words this call
            # happened to translate: the planner usually resolves "PDF" to
            # `application/pdf` itself and still leaves "PDF" in the query.
            drop = {w.rstrip("s") for w in MIME_WORDS} | {m.rstrip("s") for m in consumed}
            kept = [
                word
                for word in query.split()
                if word.strip(".,").lower().rstrip("s") not in drop
            ]
            raw["query"] = " ".join(kept)

        return await super().run(ctx, raw)

    async def refresh_live(self, ctx: OpContext, args: Any, filters: Any) -> dict | None:
        if not has_google(ctx, "drive", ("list_files", "search_files", "list")):
            return None
        raw = await google_call(
            ctx,
            "drive",
            ("list_files", "search_files", "list"),
            query=(args.query or "").strip() or None,
            mime_type=filters.sql.get("mime_type"),
            max_results=min(args.limit + args.offset + 5, 50),
        )
        rows = [_mirror_row_from_file(row_to_dict(f)) for f in (raw or [])]
        rows = [r for r in rows if r.get("file_id")]
        if rows:
            await mirror_repo.upsert_gdrive(ctx.session, ctx.user_id, rows)
        return {"fetched": len(rows)}

    def progress_label(self, args: dict) -> str:
        args = args or {}
        query = args.get("query")
        mime = args.get("mime_type") or (args.get("filter") or {}).get("mime_type")
        if query:
            return f"Searching Drive for “{excerpt_of(query, 40)}”"
        if mime:
            return f"Listing your {str(mime).rsplit('.', 1)[-1].rsplit('/', 1)[-1]} files"
        return "Searching your Drive"

    def ambiguity_question(self, args: Any, hits: list[dict]) -> str:
        return "Which file did you mean?"


class GetFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_ids: list[str] = Field(default_factory=list)
    file_id: str | None = None
    include_excerpt: bool = True
    extract: bool = False
    expect: str = "many"
    freshness: str = "cached"
    project: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if isinstance(values, dict) and "file_ids" in values:
            values = {**values, "file_ids": as_list(values["file_ids"])}
        return values

    @model_validator(mode="after")
    def _check(self) -> "GetFilesArgs":
        if self.file_id and self.file_id not in self.file_ids:
            self.file_ids = [self.file_id, *self.file_ids]
        if not self.file_ids:
            raise ValueError("drive.get_files needs at least one file id")
        return self


class GetFiles(Op):
    """File metadata and the indexed excerpt, by id.

    The excerpt is what we embedded, not the file — Drive documents are not
    stored here and a `get` never downloads one.
    """

    name = "drive.get_files"
    args_model = GetFilesArgs
    output_fields = [
        "files",
        "count",
        "file_id",
        "name",
        "mime_type",
        "owner_email",
        "web_view_link",
        "folder_path",
        "size_bytes",
        "modified_at",
        "excerpt",
        "extracted",
    ]
    is_local = True
    timeout_s = 4.0
    summary = "read file metadata and its indexed excerpt"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        files: list[dict] = []
        missing: list[str] = []

        for file_id in parsed.file_ids:
            chunks = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gdrive", file_id)
            rows = [row_to_dict(c) for c in chunks]
            if not rows and parsed.freshness == "live" and has_google(ctx, "drive", ("get_file", "get")):
                fetched = await google_call(ctx, "drive", ("get_file", "get"), file_id=file_id)
                if fetched:
                    row = _mirror_row_from_file(row_to_dict(fetched))
                    if row.get("file_id"):
                        await mirror_repo.upsert_gdrive(ctx.session, ctx.user_id, [row])
                        rows = [row]
            if not rows:
                missing.append(file_id)
                continue

            head = rows[0]
            text = "\n".join(str(r.get("content_excerpt") or "") for r in rows).strip()
            out: dict[str, Any] = {
                "file_id": head.get("file_id"),
                "name": head.get("name"),
                "mime_type": head.get("mime_type"),
                "owner_email": head.get("owner_email"),
                "is_shared": bool(head.get("is_shared")),
                "web_view_link": head.get("web_view_link"),
                "folder_path": head.get("folder_path"),
                "size_bytes": head.get("size_bytes"),
                "modified_at": head.get("modified_at"),
                "chunks": len(rows),
            }
            if parsed.include_excerpt:
                out["excerpt"] = text
            if parsed.extract:
                out["extracted"] = run_extractors(f"{head.get('name') or ''}\n{text}")
            if parsed.project:
                keep = set(parsed.project) | {"file_id"}
                out = {k: v for k, v in out.items() if k in keep}
            files.append(jsonable(out))

        data: dict[str, Any] = {
            "files": files,
            "count": len(files),
            "missing": missing,
            "found": bool(files),
        }
        if parsed.expect == "one" and files:
            data = {**files[0], **data}
        return OpResult(data=data)

    def progress_label(self, args: dict) -> str:
        count = len(as_list((args or {}).get("file_ids"))) or 1
        return "Opening that file" if count == 1 else f"Opening {count} files"

    def to_llm(self, data: dict, budget: int = 900) -> dict:
        slim = dict(data)
        slim["files"] = [
            {
                "file_id": f.get("file_id"),
                "name": f.get("name"),
                "mime_type": f.get("mime_type"),
                "modified_at": f.get("modified_at"),
                "excerpt": excerpt_of(f.get("excerpt"), 200),
            }
            for f in (data.get("files") or [])[:5]
        ]
        return trim_for_llm(slim, budget)


def _mirror_row_from_file(file: dict) -> dict[str, Any]:
    """A Drive API file as a `sync_gdrive` row."""
    owner = file.get("owner_email")
    if not owner:
        owners = file.get("owners")
        if isinstance(owners, list) and owners and isinstance(owners[0], dict):
            owner = owners[0].get("emailAddress")
    row = {
        "file_id": file.get("file_id") or file.get("id"),
        "chunk_index": int(file.get("chunk_index") or 0),
        "name": file.get("name") or "(unnamed)",
        "mime_type": file.get("mime_type") or file.get("mimeType"),
        "owner_email": owner,
        "is_shared": bool(file.get("is_shared") or file.get("shared")),
        "web_view_link": file.get("web_view_link") or file.get("webViewLink"),
        "folder_path": file.get("folder_path"),
        "size_bytes": int(file.get("size_bytes") or file.get("size") or 0) or None,
        "content_excerpt": file.get("content_excerpt") or file.get("excerpt") or "",
        "modified_at": file.get("modified_at") or file.get("modifiedTime"),
    }
    row["content_hash"] = fingerprint(
        "sync_gdrive.file",
        f"{row['file_id']}|{row['chunk_index']}|{row['name']}|{row['content_excerpt']}",
    )
    return row


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


class CreateFolderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    parent_id: str | None = None
    parent_path: str | None = None


class CreateFolder(Op):
    """Make a folder. A write, but an undoable one, so no confirm."""

    name = "drive.create_folder"
    args_model = CreateFolderArgs
    output_fields = ["file_id", "name", "web_view_link", "parent_id"]
    is_write = True
    timeout_s = 8.0
    summary = "create a Drive folder"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        created = row_to_dict(
            await google_call(
                ctx,
                "drive",
                ("create_folder", "create_file", "create"),
                name=parsed.name,
                parent_id=parsed.parent_id,
                mime_type=FOLDER_MIME,
            )
            or {}
        )
        file_id = created.get("file_id") or created.get("id")
        if not file_id:
            raise AppError("GOOGLE_UNAVAILABLE", "Drive did not return a folder id.")

        row = _mirror_row_from_file({**created, "file_id": file_id, "mime_type": FOLDER_MIME, "name": parsed.name})
        row["folder_path"] = parsed.parent_path
        await mirror_repo.upsert_gdrive(ctx.session, ctx.user_id, [row])

        return OpResult(
            data=jsonable(
                {
                    "file_id": file_id,
                    "name": parsed.name,
                    "web_view_link": created.get("web_view_link") or created.get("webViewLink"),
                    "parent_id": parsed.parent_id,
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        return f"Creating the folder “{(args or {}).get('name', '')}”"


class MoveFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_ids: list[str] = Field(default_factory=list)
    file_id: str | None = None
    target_folder_id: str
    remove_parents: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = dict(values)
        for key in ("file_ids", "remove_parents"):
            if key in out:
                out[key] = as_list(out[key])
        if "folder_id" in out and "target_folder_id" not in out:
            out["target_folder_id"] = out.pop("folder_id")
        return out

    @model_validator(mode="after")
    def _check(self) -> "MoveFileArgs":
        if self.file_id and self.file_id not in self.file_ids:
            self.file_ids = [self.file_id, *self.file_ids]
        if not self.file_ids:
            raise ValueError("drive.move_file needs at least one file id")
        return self


class MoveFile(Op):
    """Move files into a folder. Reversible, so no confirm."""

    name = "drive.move_file"
    args_model = MoveFileArgs
    output_fields = ["moved", "count", "target_folder_id", "failed"]
    is_write = True
    timeout_s = 10.0
    summary = "move files into a folder"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        moved: list[dict] = []
        failed: list[dict] = []

        for file_id in parsed.file_ids:
            try:
                result = row_to_dict(
                    await google_call(
                        ctx,
                        "drive",
                        ("move_file", "update_parents", "move"),
                        file_id=file_id,
                        add_parents=[parsed.target_folder_id],
                        remove_parents=parsed.remove_parents or None,
                    )
                    or {}
                )
            except AppError as exc:
                failed.append({"file_id": file_id, "code": exc.code, "message": exc.message})
                continue
            moved.append(
                {
                    "file_id": result.get("file_id") or result.get("id") or file_id,
                    "name": result.get("name"),
                    "folder_path": result.get("folder_path"),
                }
            )

        return OpResult(
            data=jsonable(
                {
                    "moved": moved,
                    "count": len(moved),
                    "failed": failed,
                    "target_folder_id": parsed.target_folder_id,
                }
            )
        )

    def progress_label(self, args: dict) -> str:
        count = len(as_list((args or {}).get("file_ids"))) or 1
        return "Moving that file" if count == 1 else f"Moving {count} files"


_EMAIL = re.compile(r"^[\w.\-+]+@[\w.\-]+\.\w{2,}$")


class ShareFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    email: str | None = None
    emails: list[str] = Field(default_factory=list)
    role: str = "reader"
    notify: bool = True
    message: str | None = None
    expires_at: Any = None

    @model_validator(mode="before")
    @classmethod
    def _listify(cls, values: Any) -> Any:
        if isinstance(values, dict) and "emails" in values:
            values = {**values, "emails": as_list(values["emails"])}
        return values

    @model_validator(mode="after")
    def _check(self) -> "ShareFileArgs":
        if self.email and self.email not in self.emails:
            self.emails = [self.email, *self.emails]
        if not self.emails:
            raise ValueError("drive.share_file needs someone to share with")
        for address in self.emails:
            if not _EMAIL.match(str(address)):
                raise ValueError(f"{address!r} is not an email address")
        if self.role not in ("reader", "commenter", "writer"):
            raise ValueError("role is reader, commenter or writer")
        return self


class ShareFile(ConfirmableOp):
    """Give someone access to a file.

    The one Drive op behind a confirm gate: access, once granted, has already
    been used by the time you think better of it. `run` resolves the file so
    the card can name it and its owner; nothing is granted until `execute`.
    """

    name = "drive.share_file"
    args_model = ShareFileArgs
    output_fields = ["payload", "prepared", "file_id", "name", "emails", "role", "web_view_link"]
    timeout_s = 8.0
    summary = "share a file with someone — asks first"

    async def run(self, ctx: OpContext, args: dict) -> OpResult:
        parsed = self.parse(args)
        target = await self._resolve(ctx, parsed.file_id)

        payload = jsonable(
            {
                "file_id": target["file_id"],
                "name": target.get("name"),
                "mime_type": target.get("mime_type"),
                "web_view_link": target.get("web_view_link"),
                "owner_email": target.get("owner_email"),
                "emails": parsed.emails,
                "role": parsed.role,
                "notify": parsed.notify,
                "message": parsed.message,
                "expires_at": iso(parsed.expires_at) if parsed.expires_at else None,
            }
        )
        return OpResult(
            data={
                **payload,
                "payload": payload,
                "prepared": True,
                "preview": self.preview(payload),
                "confirm_question": self.confirm_question(payload),
            }
        )

    async def _resolve(self, ctx: OpContext, file_id: str) -> dict:
        rows = await mirror_repo.get_by_ref(ctx.session, ctx.user_id, "gdrive", file_id)
        if rows:
            row = row_to_dict(rows[0])
            return {
                "file_id": row.get("file_id"),
                "name": row.get("name"),
                "mime_type": row.get("mime_type"),
                "web_view_link": row.get("web_view_link"),
                "owner_email": row.get("owner_email"),
            }
        if has_google(ctx, "drive", ("get_file", "get")):
            fetched = row_to_dict(
                await google_call(ctx, "drive", ("get_file", "get"), file_id=file_id) or {}
            )
            if fetched:
                row = _mirror_row_from_file(fetched)
                if row.get("file_id"):
                    await mirror_repo.upsert_gdrive(ctx.session, ctx.user_id, [row])
                    return row
        raise AppError(
            "NOT_FOUND",
            "That file is not in your Drive.",
            http=404,
            details={"file_id": file_id},
        )

    async def execute(self, ctx: OpContext, payload: dict) -> dict:
        granted: list[str] = []
        failed: list[dict] = []
        for address in as_list(payload.get("emails")):
            try:
                await google_call(
                    ctx,
                    "drive",
                    ("share_file", "create_permission", "add_permission", "share"),
                    file_id=payload.get("file_id"),
                    email=address,
                    role=payload.get("role") or "reader",
                    notify=bool(payload.get("notify", True)),
                    message=payload.get("message"),
                    expires_at=payload.get("expires_at"),
                )
            except AppError as exc:
                failed.append({"email": address, "code": exc.code, "message": exc.message})
                continue
            granted.append(address)
        if not granted:
            raise AppError(
                "GOOGLE_UNAVAILABLE",
                "Drive would not share that file.",
                details={"failed": failed},
            )
        return jsonable(
            {
                "file_id": payload.get("file_id"),
                "shared_with": granted,
                "failed": failed,
                "role": payload.get("role"),
                "web_view_link": payload.get("web_view_link"),
                "shared_at": iso(dt.datetime.now(dt.timezone.utc)),
            }
        )

    def preview(self, payload: dict) -> dict:
        return {
            "file": payload.get("name"),
            "with": as_list(payload.get("emails")),
            "access": payload.get("role"),
            "link": payload.get("web_view_link"),
            "note": "They get an email about it." if payload.get("notify") else "No notification is sent.",
        }

    def confirm_question(self, payload: dict) -> str:
        people = as_list(payload.get("emails"))
        who = people[0] if len(people) == 1 else f"{len(people)} people"
        return f"Give {who} {payload.get('role')} access to “{payload.get('name')}”?"

    def progress_label(self, args: dict) -> str:
        return "Preparing to share that file"


OPS: list[Op] = [SearchFiles(), GetFiles(), CreateFolder(), MoveFile(), ShareFile()]

__all__ = [
    "FOLDER_MIME",
    "MIME_WORDS",
    "OPS",
    "CreateFolder",
    "GetFiles",
    "MoveFile",
    "SearchFiles",
    "ShareFile",
]
