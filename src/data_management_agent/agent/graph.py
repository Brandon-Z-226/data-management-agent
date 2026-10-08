"""Minimal LangGraph loop over the project's framework-neutral Tool interface."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from data_management_agent.tools import Tool, ToolContext
from data_management_agent.workspace import Workspace

DEFAULT_SYSTEM_PROMPT = """You are a data workspace agent. Answer the user's task using evidence from
the workspace; do not guess facts or calculate large datasets mentally.

Choose tools and arguments yourself. Inspect structured files before querying them. For query_data,
each source is exposed as the sanitized file stem shown as table_name by inspect_file. Use DuckDB
SELECT/WITH SQL. Prefer query_data for aggregation and joins, search_workspace for discovery,
and read_file for precise document evidence. Tool errors are recoverable: inspect the error and try a
corrected call. When enough evidence is available, stop calling tools and return only the requested
answer, respecting any formatting guidelines from the user.
"""


@dataclass(frozen=True, slots=True)
class ToolTrace:
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["success", "error"]
    result: str
    error: str | None = None
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    answer: str
    messages: tuple[AnyMessage, ...]
    trajectory: tuple[ToolTrace, ...]


class WorkspaceAgent:
    """A model-driven agent → tools → agent loop compiled with LangGraph."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[Tool],
        workspace: Workspace,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_rounds: int = 12,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        tool_map = {tool.contract.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("tool names must be unique")

        self._workspace = workspace
        self._tool_map = tool_map
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._model_with_tools = model.bind_tools([_model_tool_definition(tool) for tool in tools])
        self._graph = self._build_graph()

    async def arun(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        user_content = task if guidelines is None else f"{task}\n\nAnswer guidelines: {guidelines}"
        state = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=user_content)]},
            config={"recursion_limit": self._max_tool_rounds * 2 + 2},
        )
        messages = tuple(cast(list[AnyMessage], state["messages"]))
        return AgentRunResult(
            answer=_final_answer(messages),
            messages=messages,
            trajectory=_extract_trajectory(messages),
        )

    def run(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        return asyncio.run(self.arun(task, guidelines=guidelines))

    def _build_graph(self) -> Any:
        async def call_agent(state: MessagesState) -> dict[str, list[AIMessage]]:
            response = await self._model_with_tools.ainvoke(
                [SystemMessage(content=self._system_prompt), *state["messages"]]
            )
            return {"messages": [cast(AIMessage, response)]}

        async def execute_tools(state: MessagesState) -> dict[str, list[ToolMessage]]:
            message = state["messages"][-1]
            if not isinstance(message, AIMessage):
                raise TypeError("tool node requires the last message to be an AIMessage")

            results: list[ToolMessage] = []
            for call in message.tool_calls:
                started_at = time.perf_counter()
                call_id = call.get("id") or f"call-{len(results)}"
                name = call["name"]
                arguments = call.get("args", {})
                tool = self._tool_map.get(name)
                if tool is None:
                    results.append(
                        _error_message(
                            call_id,
                            name,
                            f"unknown tool: {name}",
                            latency_ms=_elapsed_ms(started_at),
                        )
                    )
                    continue
                try:
                    output = await tool.invoke(
                        arguments,
                        context=replace(
                            ToolContext(workspace=self._workspace),
                            call_id=call_id,
                        ),
                    )
                    results.append(
                        ToolMessage(
                            content=output.model_dump_json(exclude_none=True),
                            tool_call_id=call_id,
                            name=name,
                            status="success",
                            artifact={"latency_ms": _elapsed_ms(started_at)},
                        )
                    )
                except Exception as exc:
                    results.append(
                        _error_message(
                            call_id,
                            name,
                            str(exc),
                            latency_ms=_elapsed_ms(started_at),
                        )
                    )
            return {"messages": results}

        def route_after_agent(state: MessagesState) -> str:
            message = state["messages"][-1]
            return "tools" if isinstance(message, AIMessage) and message.tool_calls else END

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_agent)
        builder.add_node("tools", execute_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        return builder.compile()


def _model_tool_definition(tool: Tool) -> dict[str, Any]:
    definition = tool.definition
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.input_schema,
        },
    }


def _error_message(
    call_id: str,
    name: str,
    message: str,
    *,
    latency_ms: float | None = None,
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"error": message}),
        tool_call_id=call_id,
        name=name,
        status="error",
        artifact={"error": message, "latency_ms": latency_ms},
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _final_answer(messages: tuple[AnyMessage, ...]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            if isinstance(message.content, str):
                return message.content
            return message.text
    raise RuntimeError("agent stopped without a final answer")


def _extract_trajectory(messages: tuple[AnyMessage, ...]) -> tuple[ToolTrace, ...]:
    calls: dict[str, tuple[str, dict[str, Any], dict[str, Any] | None]] = {}
    traces: list[ToolTrace] = []
    for message in messages:
        if isinstance(message, AIMessage):
            token_usage = _token_usage(message)
            for call in message.tool_calls:
                call_id = call.get("id") or ""
                calls[call_id] = (call["name"], call.get("args", {}), token_usage)
        elif isinstance(message, ToolMessage):
            name, arguments, token_usage = calls.get(
                message.tool_call_id,
                (message.name or "unknown", {}, None),
            )
            content = message.content if isinstance(message.content, str) else message.text
            artifact = message.artifact if isinstance(message.artifact, dict) else {}
            traces.append(
                ToolTrace(
                    call_id=message.tool_call_id,
                    name=name,
                    arguments=arguments,
                    status=message.status,
                    result=content,
                    error=_trace_error(message, content, artifact),
                    latency_ms=_trace_latency(artifact),
                    token_usage=token_usage,
                )
            )
    return tuple(traces)


def _token_usage(message: AIMessage) -> dict[str, Any] | None:
    if message.usage_metadata:
        return dict(message.usage_metadata)
    for key in ("token_usage", "usage"):
        usage = message.response_metadata.get(key)
        if isinstance(usage, dict):
            return dict(usage)
    return None


def _trace_error(message: ToolMessage, content: str, artifact: dict[str, Any]) -> str | None:
    error = artifact.get("error")
    if isinstance(error, str):
        return error
    if message.status != "error":
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    parsed_error = parsed.get("error") if isinstance(parsed, dict) else None
    return parsed_error if isinstance(parsed_error, str) else content


def _trace_latency(artifact: dict[str, Any]) -> float | None:
    latency = artifact.get("latency_ms")
    if isinstance(latency, int | float):
        return float(latency)
    return None
