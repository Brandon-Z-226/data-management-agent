"""Named contracts for the seven tools in Tool Surface v0."""

from __future__ import annotations

from typing import Final

from .base import ToolContract
from .schemas import (
    InspectFileInput,
    InspectFileOutput,
    ListWorkspaceInput,
    ListWorkspaceOutput,
    QueryDataInput,
    QueryDataOutput,
    ReadFileInput,
    ReadFileOutput,
    RunPythonInput,
    RunPythonOutput,
    SearchWorkspaceInput,
    SearchWorkspaceOutput,
    WorkspaceOpsInput,
    WorkspaceOpsOutput,
)

LIST_WORKSPACE: Final = ToolContract(
    name="list_workspace",
    description="Browse workspace directories and file metadata using explicit filters.",
    input_type=ListWorkspaceInput,
    output_type=ListWorkspaceOutput,
)

SEARCH_WORKSPACE: Final = ToolContract(
    name="search_workspace",
    description="Search workspace content and metadata and return ranked evidence locations.",
    input_type=SearchWorkspaceInput,
    output_type=SearchWorkspaceOutput,
)

INSPECT_FILE: Final = ToolContract(
    name="inspect_file",
    description="Inspect a file's format and structure without reading all of its content.",
    input_type=InspectFileInput,
    output_type=InspectFileOutput,
)

READ_FILE: Final = ToolContract(
    name="read_file",
    description="Read precise content from a file using an optional format-neutral locator.",
    input_type=ReadFileInput,
    output_type=ReadFileOutput,
)

QUERY_DATA: Final = ToolContract(
    name="query_data",
    description="Run a bounded query across one or more structured workspace data sources.",
    input_type=QueryDataInput,
    output_type=QueryDataOutput,
)

RUN_PYTHON: Final = ToolContract(
    name="run_python",
    description="Run bounded Python analysis in a controlled sandbox over workspace inputs.",
    input_type=RunPythonInput,
    output_type=RunPythonOutput,
)

WORKSPACE_OPS: Final = ToolContract(
    name="workspace_ops",
    description="Plan or execute recoverable workspace mutations; dry-run is the default.",
    input_type=WorkspaceOpsInput,
    output_type=WorkspaceOpsOutput,
)

TOOL_CONTRACTS: Final = {
    contract.name: contract
    for contract in (
        LIST_WORKSPACE,
        SEARCH_WORKSPACE,
        INSPECT_FILE,
        READ_FILE,
        QUERY_DATA,
        RUN_PYTHON,
        WORKSPACE_OPS,
    )
}
