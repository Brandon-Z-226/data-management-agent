"""Pydantic request and response models for Tool Surface v0."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from data_management_agent.workspace import WorkspaceEntry, WorkspacePath, WorkspaceRef


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListWorkspaceInput(ToolArguments):
    path: WorkspacePath = Field(
        default_factory=lambda: WorkspacePath(root="."),
        description="Workspace-relative path. Root is '.', never '/'.",
    )
    recursive: bool = False
    file_types: frozenset[str] = Field(default_factory=frozenset)
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    cursor: str | None = Field(
        default=None, description="Copy next_cursor from the previous listing."
    )
    limit: int = Field(default=100, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_time_range(self) -> ListWorkspaceInput:
        if (
            self.modified_after is not None
            and self.modified_before is not None
            and self.modified_after > self.modified_before
        ):
            raise ValueError("modified_after cannot be later than modified_before")
        return self


class ListWorkspaceOutput(ToolOutput):
    entries: tuple[WorkspaceEntry, ...] = ()
    next_cursor: str | None = None


class SearchWorkspaceInput(ToolArguments):
    query: str = Field(min_length=1)
    scope: WorkspacePath = Field(
        default_factory=lambda: WorkspacePath(root="."),
        description="Workspace-relative search scope; '.' searches the root.",
    )
    file_types: frozenset[str] = Field(default_factory=frozenset)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=100)


class ContentLocation(BaseModel):
    """Format-neutral location of content inside a resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str | None = Field(
        default=None, description="ID from search_workspace; use alone."
    )
    page: int | None = Field(
        default=None, ge=1, description="Reserved; unsupported by current tools."
    )
    section: str | None = Field(
        default=None, description="Markdown heading text; do not mix with char ranges."
    )
    sheet: str | None = Field(default=None, description="Reserved; unsupported by current tools.")
    row_start: int | None = Field(
        default=None,
        ge=0,
        description="CSV/JSON array record start, inclusive, 0-based; CSV header excluded.",
    )
    row_end: int | None = Field(
        default=None, ge=0, description="Record end, exclusive. Omit for open-ended paging."
    )
    char_start: int | None = Field(
        default=None, ge=0, description="Markdown character start, inclusive, 0-based; not bytes."
    )
    char_end: int | None = Field(
        default=None,
        ge=0,
        description="Markdown character end, exclusive. Omit to request through EOF.",
    )

    @model_validator(mode="after")
    def validate_location(self) -> ContentLocation:
        if not any(
            value is not None
            for value in (
                self.evidence_id,
                self.page,
                self.section,
                self.sheet,
                self.row_start,
                self.row_end,
                self.char_start,
                self.char_end,
            )
        ):
            raise ValueError("at least one content locator must be provided")
        if self.row_start is not None and self.row_end is not None:
            if self.row_start > self.row_end:
                raise ValueError("row_start cannot be greater than row_end")
        if self.char_start is not None and self.char_end is not None:
            if self.char_start > self.char_end:
                raise ValueError("char_start cannot be greater than char_end")
        return self


class SearchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    path: WorkspacePath
    snippet: str
    location: ContentLocation
    score: float | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SearchWorkspaceOutput(ToolOutput):
    evidence: tuple[SearchEvidence, ...] = ()


class InspectFileInput(ToolArguments):
    file: WorkspaceRef = Field(description="Reference with exactly one of path or resource_id.")


class InspectFileOutput(ToolOutput):
    entry: WorkspaceEntry
    format: str = Field(min_length=1)
    summary: str | None = None
    structure: dict[str, JsonValue] = Field(default_factory=dict)


class ReadFileInput(ToolArguments):
    file: WorkspaceRef = Field(
        description="Only path OR resource_id. Do not put locator/max_chars inside file."
    )
    locator: ContentLocation | None = Field(
        default=None,
        description="Top-level selection, sibling of file. Copy next_locator to continue; omit for default first page.",
    )
    max_chars: int = Field(
        default=20_000,
        ge=1,
        le=200_000,
        description="Top-level content character budget, including CSV header/JSON framing; too-small records error instead of being split.",
    )


class ReadFileOutput(ToolOutput):
    entry: WorkspaceEntry
    content: str
    media_type: str = "text/plain"
    location: ContentLocation | None = Field(
        default=None, description="Actual returned range; zero-based, end-exclusive."
    )
    truncated: bool = Field(
        default=False, description="True when some of the requested scope remains unread."
    )
    requested_location: ContentLocation | None = Field(
        default=None,
        description="Resolved requested scope, clipped to EOF; not the page size. "
        "Null for a whole non-array JSON value.",
    )
    next_locator: ContentLocation | None = Field(
        default=None,
        description="Pass as read_file.locator with the same file to continue within the "
        "requested scope. Null when that scope is complete.",
    )


class QueryDataInput(ToolArguments):
    sources: tuple[WorkspaceRef, ...] = Field(
        min_length=1,
        description="All CSV/JSON array sources referenced by the query, each with path OR resource_id.",
    )
    query: str = Field(
        min_length=1,
        description="Single DuckDB SELECT/WITH. Use inspect_file's table_name and columns, not file paths.",
    )
    dialect: Literal["duckdb"] = "duckdb"
    max_rows: int = Field(
        default=1_000,
        ge=1,
        le=100_000,
        description="Returned row limit only; aggregates still use the full source data.",
    )


class QueryDataOutput(ToolOutput):
    columns: tuple[str, ...]
    rows: tuple[dict[str, JsonValue], ...]
    row_count: int = Field(ge=0)
    truncated: bool = False


class RunPythonInput(ToolArguments):
    code: str = Field(min_length=1)
    input_files: tuple[WorkspaceRef, ...] = ()
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_output_chars: int = Field(default=20_000, ge=1, le=200_000)


class RunPythonOutput(ToolOutput):
    status: Literal["succeeded", "failed", "timed_out"]
    stdout: str = ""
    stderr: str = ""
    result: JsonValue | None = None
    artifacts: tuple[WorkspaceRef, ...] = ()
    exit_code: int | None = None


class CreateDirectoryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["create_directory"]
    path: WorkspacePath
    parents: bool = True
    exist_ok: bool = False


class MoveOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["move"]
    source: WorkspaceRef
    destination: WorkspacePath
    overwrite: bool = False


class RenameOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["rename"]
    source: WorkspaceRef
    destination: WorkspacePath
    overwrite: bool = False


class CopyOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["copy"]
    source: WorkspaceRef
    destination: WorkspacePath
    overwrite: bool = False


class UpdateTagsOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["update_tags"]
    target: WorkspaceRef
    add: frozenset[str] = Field(default_factory=frozenset)
    remove: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def reject_overlapping_tags(self) -> UpdateTagsOperation:
        overlap = self.add & self.remove
        if overlap:
            raise ValueError(f"tags cannot be both added and removed: {sorted(overlap)}")
        return self


WorkspaceOperation = Annotated[
    CreateDirectoryOperation
    | MoveOperation
    | RenameOperation
    | CopyOperation
    | UpdateTagsOperation,
    Field(discriminator="operation"),
]


class WorkspaceOpsInput(ToolArguments):
    operations: tuple[WorkspaceOperation, ...] = Field(min_length=1)
    dry_run: bool = True


class WorkspaceOperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    operation: Literal["create_directory", "move", "rename", "copy", "update_tags"]
    status: Literal["planned", "succeeded", "failed", "skipped"]
    message: str | None = None
    entry: WorkspaceEntry | None = None


class WorkspaceOpsOutput(ToolOutput):
    dry_run: bool
    results: tuple[WorkspaceOperationResult, ...]
