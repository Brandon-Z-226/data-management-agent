import asyncio
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from data_management_agent.tools import TOOL_CONTRACTS, Tool, ToolContext, ToolContract
from data_management_agent.tools.schemas import (
    ContentLocation,
    UpdateTagsOperation,
    WorkspaceOpsInput,
)
from data_management_agent.workspace import Workspace


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class EchoOutput(BaseModel):
    value: int


class EchoTool(Tool[EchoInput, EchoOutput]):
    contract = ToolContract(
        name="echo",
        description="Echo an integer.",
        input_type=EchoInput,
        output_type=EchoOutput,
    )

    async def execute(self, arguments: EchoInput, *, context: ToolContext) -> EchoOutput:
        return EchoOutput(value=arguments.value)


def test_tool_surface_contains_exactly_the_seven_architecture_tools() -> None:
    assert set(TOOL_CONTRACTS) == {
        "list_workspace",
        "search_workspace",
        "inspect_file",
        "read_file",
        "query_data",
        "run_python",
        "workspace_ops",
    }


def test_tool_contract_produces_serializable_schemas() -> None:
    definition = TOOL_CONTRACTS["search_workspace"].definition()

    assert definition.name == "search_workspace"
    assert definition.input_schema["type"] == "object"
    assert "query" in definition.input_schema["properties"]
    assert definition.output_schema["type"] == "object"


def test_tool_invoke_validates_raw_arguments() -> None:
    context = ToolContext(workspace=cast(Workspace, object()))
    result = asyncio.run(EchoTool().invoke({"value": 42}, context=context))

    assert result == EchoOutput(value=42)

    with pytest.raises(ValidationError):
        asyncio.run(EchoTool().invoke({"value": 42, "unexpected": True}, context=context))


def test_workspace_operations_are_dry_run_by_default_and_have_no_delete_variant() -> None:
    request = WorkspaceOpsInput(
        operations=[{"operation": "move", "source": {"path": "a.csv"}, "destination": "b.csv"}]
    )

    assert request.dry_run is True
    operation_schema = WorkspaceOpsInput.model_json_schema()
    assert "delete" not in str(operation_schema)


def test_content_locations_and_tag_updates_validate_ranges_and_overlap() -> None:
    with pytest.raises(ValidationError):
        ContentLocation(row_start=10, row_end=2)
    with pytest.raises(ValidationError):
        UpdateTagsOperation(operation="update_tags", target={"path": "a.csv"}, add={"final"}, remove={"final"})
