"""Framework-neutral interface implemented by every agent tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from data_management_agent.workspace import Workspace

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolExecutionError(RuntimeError):
    """A tool call failed after its arguments were successfully validated."""


class ToolDefinition(BaseModel):
    """Serializable definition suitable for an LLM tool adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolContract(Generic[InputT, OutputT]):
    """Static name, documentation, and typed I/O models for one tool."""

    name: str
    description: str
    input_type: type[InputT]
    output_type: type[OutputT]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_type.model_json_schema(),
            output_schema=self.output_type.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Dependencies and tracing data available during a tool call."""

    workspace: Workspace
    call_id: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


class Tool(ABC, Generic[InputT, OutputT]):
    """Validated asynchronous tool boundary.

    Concrete tools only implement :meth:`execute`. ``invoke`` keeps model
    validation consistent across direct calls and future LangGraph adapters.
    """

    contract: ToolContract[InputT, OutputT]

    @property
    def definition(self) -> ToolDefinition:
        return self.contract.definition()

    async def invoke(
        self,
        arguments: InputT | Mapping[str, Any],
        *,
        context: ToolContext,
    ) -> OutputT:
        validated_input = (
            arguments
            if isinstance(arguments, self.contract.input_type)
            else self.contract.input_type.model_validate(arguments)
        )
        result = await self.execute(cast(InputT, validated_input), context=context)
        if isinstance(result, self.contract.output_type):
            return result
        return self.contract.output_type.model_validate(result)

    @abstractmethod
    async def execute(self, arguments: InputT, *, context: ToolContext) -> OutputT:
        """Perform the tool action using already validated arguments."""
