"""Read-only local filesystem implementation of the workspace contract."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from .base import ResourceNotFoundError, Workspace, WorkspaceError
from .models import (
    ResourceKind,
    WorkspaceEntry,
    WorkspacePage,
    WorkspacePath,
    WorkspaceQuery,
    WorkspaceRef,
)

_MEDIA_TYPE_OVERRIDES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
}


class LocalWorkspace(Workspace):
    """A sandboxed, read-only workspace rooted at one local directory."""

    def __init__(self, root: Path | str, *, workspace_id: str | None = None) -> None:
        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {resolved_root}")
        self._root = resolved_root
        root_digest = hashlib.sha256(str(resolved_root).encode()).hexdigest()[:12]
        self._workspace_id = workspace_id or f"local-{resolved_root.name}-{root_digest}"

    @property
    def root(self) -> Path:
        """Return the backend root for diagnostics, not for tool data access."""

        return self._root

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    async def list_entries(
        self,
        query: WorkspaceQuery,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> WorkspacePage:
        if limit < 1:
            raise WorkspaceError("limit must be at least 1")

        base = self._resolve_path(query.path)
        if not base.exists():
            raise ResourceNotFoundError(f"workspace path does not exist: {query.path}")

        if base.is_file():
            paths = [base]
        elif query.recursive:
            paths = list(base.rglob("*"))
        else:
            paths = list(base.iterdir())

        entries = [self._entry_for_path(path) for path in paths]
        entries = [entry for entry in entries if self._matches(entry, query)]
        entries.sort(key=lambda entry: str(entry.path))

        start = self._parse_cursor(cursor)
        page_entries = tuple(entries[start : start + limit])
        next_offset = start + len(page_entries)
        next_cursor = str(next_offset) if next_offset < len(entries) else None
        return WorkspacePage(entries=page_entries, next_cursor=next_cursor)

    async def get_entry(self, ref: WorkspaceRef) -> WorkspaceEntry:
        if ref.path is not None:
            path = self._resolve_path(ref.path)
            if not path.exists():
                raise ResourceNotFoundError(f"workspace resource does not exist: {ref.path}")
            return self._entry_for_path(path)

        for path in self._root.rglob("*"):
            entry = self._entry_for_path(path)
            if entry.resource_id == ref.resource_id:
                return entry
        raise ResourceNotFoundError(f"workspace resource id does not exist: {ref.resource_id}")

    async def read_bytes(
        self,
        ref: WorkspaceRef,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        if offset < 0:
            raise WorkspaceError("offset cannot be negative")
        if length is not None and length < 1:
            raise WorkspaceError("length must be at least 1")

        entry = await self.get_entry(ref)
        if entry.kind is not ResourceKind.FILE:
            raise WorkspaceError(f"cannot read directory content: {entry.path}")
        path = self._resolve_path(entry.path)
        with path.open("rb") as file:
            file.seek(offset)
            return file.read() if length is None else file.read(length)

    def _resolve_path(self, path: WorkspacePath) -> Path:
        candidate = (self._root / str(path)).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace root: {path}") from exc
        return candidate

    def _entry_for_path(self, path: Path) -> WorkspaceEntry:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._root)
        except ValueError as exc:
            raise WorkspaceError(f"resource escapes workspace root: {path}") from exc
        if not resolved.exists():
            raise ResourceNotFoundError(f"workspace resource does not exist: {relative}")

        stat = resolved.stat()
        workspace_path = WorkspacePath(str(relative) if relative.parts else ".")
        kind = ResourceKind.DIRECTORY if resolved.is_dir() else ResourceKind.FILE
        media_type = None if kind is ResourceKind.DIRECTORY else self._media_type(resolved)
        resource_identity = f"{self._workspace_id}:{workspace_path}"
        resource_digest = hashlib.sha256(resource_identity.encode()).hexdigest()[:24]
        return WorkspaceEntry(
            resource_id=f"local-{resource_digest}",
            path=workspace_path,
            kind=kind,
            media_type=media_type,
            size_bytes=None if kind is ResourceKind.DIRECTORY else stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )

    @staticmethod
    def _media_type(path: Path) -> str:
        return _MEDIA_TYPE_OVERRIDES.get(
            path.suffix.lower(),
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    @staticmethod
    def _parse_cursor(cursor: str | None) -> int:
        if cursor is None:
            return 0
        try:
            offset = int(cursor)
        except ValueError as exc:
            raise WorkspaceError("cursor must be a non-negative integer") from exc
        if offset < 0:
            raise WorkspaceError("cursor must be a non-negative integer")
        return offset

    @staticmethod
    def _matches(entry: WorkspaceEntry, query: WorkspaceQuery) -> bool:
        if query.kinds and entry.kind not in query.kinds:
            return False
        if query.extensions:
            suffix = Path(str(entry.path)).suffix.lower().lstrip(".")
            extensions = {extension.lower().lstrip(".") for extension in query.extensions}
            if suffix not in extensions:
                return False
        if query.media_types and entry.media_type not in query.media_types:
            return False
        if query.metadata and any(
            entry.metadata.get(key) != value for key, value in query.metadata.items()
        ):
            return False
        if entry.modified_at is not None:
            after = _as_aware(query.modified_after)
            before = _as_aware(query.modified_before)
            if after is not None and entry.modified_at < after:
                return False
            if before is not None and entry.modified_at > before:
                return False
        return True


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)
