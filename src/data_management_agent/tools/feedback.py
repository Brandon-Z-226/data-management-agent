"""Bounded, model-facing failure feedback; raw exceptions stay in trace artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from data_management_agent.workspace import (
    ResourceNotFoundError,
    UnsupportedWorkspaceOperationError,
    WorkspaceError,
)

from .base import ToolContract, ToolExecutionError

ErrorCode = Literal[
    "invalid_arguments",
    "unknown_tool",
    "resource_not_found",
    "unsupported_operation",
    "sql_error",
    "data_error",
    "execution_error",
    "internal_error",
    "runtime_limit",
]


class ArgumentIssue(BaseModel):
    field: str
    reason: str


class ToolErrorFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str
    suggested_action: str
    fields: tuple[ArgumentIssue, ...] = ()
    example_arguments: dict[str, Any] | None = None
    # Advice is not an authorization to auto-retry, and no retry policy is implemented here.
    retry_same_arguments: Literal[False] = False

    def payload(self) -> dict[str, Any]:
        return {"error": self.message, **self.model_dump(mode="json", exclude_none=True)}


def _short(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + "… [details in local trace]"


def unknown_tool_feedback(name: str, available: list[str]) -> ToolErrorFeedback:
    return ToolErrorFeedback(
        code="unknown_tool",
        message=_short(f"unknown tool: {name}"),
        suggested_action="Choose a tool from: " + ", ".join(available),
    )


def error_feedback(exc: Exception, contract: ToolContract) -> ToolErrorFeedback:
    if isinstance(exc, ValidationError):
        issues = exc.errors(include_input=False, include_url=False, include_context=False)
        fields = tuple(
            ArgumentIssue(
                field=_short(".".join(map(str, issue["loc"])) or "$", 100),
                reason=_short(issue["msg"], 180),
            )
            for issue in issues[:5]
        )
        action = "Correct the named fields against the input schema. Examples show structure only; use actual workspace paths."
        misplaced = [
            issue["loc"]
            for issue in issues
            if issue["type"] == "extra_forbidden"
            and issue["loc"] in {("file", "locator"), ("file", "max_chars")}
        ]
        if contract.name == "read_file" and misplaced:
            action = "Move locator/max_chars out of file to the top level. file contains only path OR resource_id. Do not repeat the same nesting. Example paths are illustrative."
        elif any("path" in issue["loc"] or "scope" in issue["loc"] for issue in issues):
            action = "Use workspace-relative paths (root is '.', not '/'); use exactly one of path/resource_id per reference. Correct the listed fields; do not escape the workspace."
        return ToolErrorFeedback(
            code="invalid_arguments",
            message=f"Invalid arguments for {contract.name} ({len(issues)} issue(s)).",
            suggested_action=action,
            fields=fields,
            example_arguments=contract.examples[0] if contract.examples else None,
        )
    if isinstance(exc, ResourceNotFoundError):
        return ToolErrorFeedback(
            code="resource_not_found",
            message=_short(str(exc)),
            suggested_action="Use list_workspace/search_workspace to discover an existing relative path or resource_id, then correct the reference.",
        )
    if isinstance(exc, UnsupportedWorkspaceOperationError):
        return ToolErrorFeedback(
            code="unsupported_operation",
            message=_short(str(exc)),
            suggested_action="Use a supported read-only operation and inspect the file format first.",
        )
    if isinstance(exc, ToolExecutionError):
        actions = {
            "invalid_arguments": "Correct the stated argument constraint; do not repeat unchanged arguments.",
            "unsupported_operation": "Check the tool's supported formats/locators and choose a supported operation.",
            "sql_error": "Inspect source table/column names and DuckDB types, then correct the SQL syntax, references or casts. Do not repeat unchanged SQL.",
            "data_error": "The source could not be parsed. Inspect the source format/content; unchanged SQL will not repair malformed data.",
            "execution_error": "Execution failed. Check the available evidence before changing arguments; if unresolved, report the limitation.",
        }
        if exc.code in actions:
            message = str(exc)
            if exc.code == "sql_error":
                message = message.split("Candidate functions:", 1)[0].strip()
            return ToolErrorFeedback(
                code=exc.code,
                message=_short(message),
                suggested_action=_short(exc.suggested_action or actions[exc.code], 400),
            )
    elif isinstance(exc, WorkspaceError):
        return ToolErrorFeedback(
            code="execution_error",
            message=_short(str(exc)),
            suggested_action="Check that the resource is accessible within the workspace boundary. Do not bypass that boundary; report unresolved access limitations.",
        )
    return ToolErrorFeedback(
        code="internal_error",
        message="The tool failed internally; no verified result is available.",
        suggested_action="Do not assume the arguments are wrong or retry identically. Report the tool failure if no other evidence can answer the task; full diagnostics are in the local trace.",
    )
