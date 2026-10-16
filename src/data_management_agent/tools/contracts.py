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
    description="Browse workspace directories and file metadata. Root is '.', not '/'. "
    "Paths are workspace-relative; use next_cursor to continue a listing.",
    input_type=ListWorkspaceInput,
    output_type=ListWorkspaceOutput,
    examples=({"path": ".", "recursive": False},),
)

SEARCH_WORKSPACE: Final = ToolContract(
    name="search_workspace",
    description="Lexical search over workspace content. Returns snippets and evidence locations, "
    "not full documents. Use read_file for full evidence; do not treat no hits as proof of absence.",
    input_type=SearchWorkspaceInput,
    output_type=SearchWorkspaceOutput,
    examples=({"query": "fee calculation", "scope": ".", "file_types": ["md"], "top_k": 3},),
)

INSPECT_FILE: Final = ToolContract(
    name="inspect_file",
    description="Inspect file structure. For queryable CSV/JSON, structure.table_name and "
    "structure.columns give exact DuckDB names/types; quote SQL identifiers when needed. "
    "Sample rows are not the full dataset.",
    input_type=InspectFileInput,
    output_type=InspectFileOutput,
    examples=({"file": {"path": "records.csv"}},),
)

READ_FILE: Final = ToolContract(
    name="read_file",
    description="Read file content. file contains only path OR resource_id; locator and max_chars "
    "are separate top-level arguments, never inside file. Markdown supports section OR char range; "
    "CSV/JSON arrays support row range; evidence_id is used alone. page/sheet are unsupported. "
    "Returns requested and actual ranges. If truncated, pass next_locator as locator with "
    "the same file to continue within the requested scope.",
    input_type=ReadFileInput,
    output_type=ReadFileOutput,
    examples=(
        {
            "file": {"path": "notes.md"},
            "locator": {"char_start": 0, "char_end": 5000},
            "max_chars": 5000,
        },
        {"file": {"path": "records.csv"}, "locator": {"row_start": 0, "row_end": 10}},
    ),
)

QUERY_DATA: Final = ToolContract(
    name="query_data",
    description="Run one read-only DuckDB SELECT/WITH over CSV or JSON arrays. Declare every "
    "source in sources; use inspect_file's exact table/column names, not filenames as SQL tables. "
    "max_rows caps output rows, not rows used for aggregation. No writes or multiple statements.",
    input_type=QueryDataInput,
    output_type=QueryDataOutput,
    examples=(
        {
            "sources": [{"path": "records.csv"}],
            "query": "SELECT COUNT(*) AS row_count FROM records",
            "max_rows": 10,
        },
    ),
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
