"""Framework-neutral interface implemented by every agent tool."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from data_management_agent.workspace import Workspace

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolExecutionError(RuntimeError):
    """A tool call failed after its arguments were successfully validated."""

    def __init__(
        self, message: str, *, code: str = "execution_error", suggested_action: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.suggested_action = suggested_action


class ToolOutputError(ToolExecutionError):
    """Execution/output validation failure, not invalid model arguments."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="internal_error")


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
    examples: tuple[dict[str, Any], ...] = ()

    def definition(self) -> ToolDefinition:
        schema = self.input_type.model_json_schema()
        description = self.description
        if self.examples:
            schema["examples"] = deepcopy(list(self.examples))
            description += " Example arguments (use actual workspace paths): " + json.dumps(
                self.examples, ensure_ascii=False, separators=(",", ":")
            )
        return ToolDefinition(
            name=self.name,
            description=description,
            input_schema=schema,
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
        try:
            result = await self.execute(cast(InputT, validated_input), context=context)
            if isinstance(result, self.contract.output_type):
                return result
            return self.contract.output_type.model_validate(result)
        except ValidationError as exc:
            raise ToolOutputError("tool execution/output failed validation") from exc

    @abstractmethod
    async def execute(self, arguments: InputT, *, context: ToolContext) -> OutputT:
        """Perform the tool action using already validated arguments."""
