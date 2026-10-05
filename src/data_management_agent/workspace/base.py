"""Abstract storage boundary for a data-management workspace."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from pydantic import JsonValue

from .models import WorkspaceEntry, WorkspacePage, WorkspacePath, WorkspaceQuery, WorkspaceRef


class WorkspaceError(Exception):
    """Base exception raised by workspace backends."""


class ResourceNotFoundError(WorkspaceError):
    """A referenced workspace resource does not exist."""


class ResourceConflictError(WorkspaceError):
    """An operation conflicts with the current workspace state."""


class UnsupportedWorkspaceOperationError(WorkspaceError):
    """The selected backend does not support an optional operation."""


class Workspace(ABC):
    """Read-only workspace capability.

    This boundary intentionally exposes bytes and metadata, not parsing or search.
    Parsing, indexing, retrieval, and data querying remain separate components.
    """

    @property
    @abstractmethod
    def workspace_id(self) -> str:
        """Return the stable identity of this workspace."""

    @abstractmethod
    async def list_entries(
        self,
        query: WorkspaceQuery,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> WorkspacePage:
        """Return one deterministic page of resources matching ``query``."""

    @abstractmethod
    async def get_entry(self, ref: WorkspaceRef) -> WorkspaceEntry:
        """Resolve a resource reference to its current metadata."""

    @abstractmethod
    async def read_bytes(
        self,
        ref: WorkspaceRef,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read all or part of a resource without interpreting its format."""


class MutableWorkspace(Workspace):
    """Workspace capability for recoverable create and mutation operations.

    Permanent deletion is intentionally absent from the v0 contract.
    """

    @abstractmethod
    async def create_directory(
        self,
        path: WorkspacePath,
        *,
        parents: bool = True,
        exist_ok: bool = False,
    ) -> WorkspaceEntry:
        """Create a directory and return its metadata."""

    @abstractmethod
    async def write_bytes(
        self,
        path: WorkspacePath,
        content: bytes,
        *,
        media_type: str | None = None,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        """Create a file, or replace one only when explicitly allowed."""

    @abstractmethod
    async def move(
        self,
        source: WorkspaceRef,
        destination: WorkspacePath,
        *,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        """Move or rename a resource."""

    @abstractmethod
    async def copy(
        self,
        source: WorkspaceRef,
        destination: WorkspacePath,
        *,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        """Copy a resource."""

    @abstractmethod
    async def update_tags(
        self,
        ref: WorkspaceRef,
        *,
        add: frozenset[str] = frozenset(),
        remove: frozenset[str] = frozenset(),
    ) -> WorkspaceEntry:
        """Atomically add and remove resource tags."""

    @abstractmethod
    async def update_metadata(
        self,
        ref: WorkspaceRef,
        values: Mapping[str, JsonValue],
    ) -> WorkspaceEntry:
        """Merge backend-neutral metadata into a resource."""
