"""Value objects shared by workspace backends and agent tools."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    field_validator,
    model_validator,
)


class WorkspacePath(RootModel[str]):
    """A normalized, workspace-relative POSIX path.

    Workspace paths are deliberately independent from the host operating system.
    Backends are responsible for mapping them to their own storage representation.
    """

    model_config = ConfigDict(frozen=True)

    @field_validator("root", mode="before")
    @classmethod
    def validate_path(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("workspace path must be a string")
        if "\x00" in value:
            raise ValueError("workspace path cannot contain a null byte")
        if "\\" in value:
            raise ValueError("workspace path must use POSIX '/' separators")

        path = PurePosixPath(value or ".")
        if path.is_absolute():
            raise ValueError("workspace path must be relative")
        if ".." in path.parts:
            raise ValueError("workspace path cannot escape the workspace root")
        return str(path)

    def __str__(self) -> str:
        return self.root


class WorkspaceRef(BaseModel):
    """Stable reference to a workspace resource by id or by path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str | None = Field(default=None, min_length=1)
    path: WorkspacePath | None = None

    @model_validator(mode="after")
    def require_exactly_one_locator(self) -> WorkspaceRef:
        if (self.resource_id is None) == (self.path is None):
            raise ValueError("exactly one of resource_id or path must be provided")
        return self


class ResourceKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


class WorkspaceEntry(BaseModel):
    """Backend-neutral metadata describing one workspace resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: str = Field(min_length=1)
    path: WorkspacePath
    kind: ResourceKind
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None
    tags: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class WorkspaceQuery(BaseModel):
    """Storage-level filters used while enumerating a workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: WorkspacePath = Field(default_factory=lambda: WorkspacePath(root="."))
    recursive: bool = False
    kinds: frozenset[ResourceKind] = Field(default_factory=frozenset)
    extensions: frozenset[str] = Field(default_factory=frozenset)
    media_types: frozenset[str] = Field(default_factory=frozenset)
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_range(self) -> WorkspaceQuery:
        if (
            self.modified_after is not None
            and self.modified_before is not None
            and self.modified_after > self.modified_before
        ):
            raise ValueError("modified_after cannot be later than modified_before")
        return self


class WorkspacePage(BaseModel):
    """One page of workspace entries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[WorkspaceEntry, ...] = ()
    next_cursor: str | None = None
