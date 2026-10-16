"""Minimal LangGraph loop over the project's framework-neutral Tool interface."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from data_management_agent.observability import RunJournal, emit
from data_management_agent.tools import Tool, ToolContext
from data_management_agent.tools.feedback import (
    ToolErrorFeedback,
    error_feedback,
    unknown_tool_feedback,
)
from data_management_agent.workspace import Workspace

from .runtime import AgentState, call_history, signature

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
    attempted: bool = True


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    answer: str
    messages: tuple[AnyMessage, ...]
    trajectory: tuple[ToolTrace, ...]
    status: Literal["succeeded", "limited"] = "succeeded"
    termination_reason: str = "completed"
    finalization_used: bool = False


class AgentRunError(RuntimeError):
    """Agent failure carrying the observable state produced before the error."""

    def __init__(
        self,
        message: str,
        *,
        messages: tuple[AnyMessage, ...],
        error_type: str = "AgentRunError",
        termination_reason: str = "run_error",
        limit_reason: str | None = None,
        cause_type: str | None = None,
        finalization_used: bool = False,
    ) -> None:
        super().__init__(message)
        self.messages = messages
        self.trajectory = _extract_trajectory(messages)
        self.error_type = error_type
        self.termination_reason = termination_reason
        self.limit_reason = limit_reason
        self.cause_type = cause_type
        self.finalization_used = finalization_used


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
        max_tool_calls: int = 24,
        max_repeated_failures: int = 2,
        journal: RunJournal | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if max_tool_calls < 1 or max_repeated_failures < 1:
            raise ValueError("max_tool_calls and max_repeated_failures must be at least 1")
        tool_map = {tool.contract.name: tool for tool in tools}
        if len(tool_map) != len(tools):
            raise ValueError("tool names must be unique")

        self._workspace = workspace
        self._tool_map = tool_map
        self._system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._max_tool_calls = max_tool_calls
        self._max_repeated_failures = max_repeated_failures
        self._final_model = model  # Unbound model: the finalization request advertises no tools.
        self._journal = journal
        self._log_config = {
            "model": getattr(model, "model_name", type(model).__name__),
            "base_url": str(getattr(model, "openai_api_base", "") or ""),
            "use_responses_api": getattr(model, "use_responses_api", None),
            "temperature": getattr(model, "temperature", None),
            "max_retries": getattr(model, "max_retries", None),
            "system_prompt": system_prompt,
            "tools": [_model_tool_definition(tool) for tool in tools],
            "workspace_id": workspace.workspace_id,
            "max_tool_rounds": max_tool_rounds,
            "max_tool_calls": max_tool_calls,
            "max_repeated_failures": max_repeated_failures,
            "runtime_policy": "bounded-loop-v1",
            "recursion_limit": max_tool_rounds * 2 + 4,
        }
        self._model_with_tools = model.bind_tools([_model_tool_definition(tool) for tool in tools])
        self._graph = self._build_graph()

    async def arun(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        if self._journal is None:
            return await self._arun(task, guidelines=guidelines)
        with self._journal.start(self._log_config, task=task, guidelines=guidelines) as log:
            try:
                result = await self._arun(task, guidelines=guidelines)
            except BaseException as exc:
                messages = exc.messages if isinstance(exc, AgentRunError) else ()
                log.emit(
                    "run_finished",
                    status="failed" if isinstance(exc, Exception) else "interrupted",
                    error_type=getattr(exc, "error_type", type(exc).__name__),
                    error=str(exc),
                    termination_reason=getattr(exc, "termination_reason", "interrupted"),
                    limit_reason=getattr(exc, "limit_reason", None),
                    cause_type=getattr(exc, "cause_type", None),
                    finalization_used=getattr(exc, "finalization_used", False),
                    messages=[m.model_dump(mode="json") for m in messages],
                )
                raise
            log.emit(
                "run_finished",
                status=result.status,
                final_answer=result.answer,
                termination_reason=result.termination_reason,
                finalization_used=result.finalization_used,
            )
            return result

    async def _arun(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        user_content = task if guidelines is None else f"{task}\n\nAnswer guidelines: {guidelines}"
        initial_message = HumanMessage(content=user_content)
        state: dict[str, Any] | None = None
        try:
            async for snapshot in self._graph.astream(
                {
                    "messages": [initial_message],
                    "tool_rounds": 0,
                    "tool_attempts": 0,
                    "stop_reason": None,
                    "error_type": None,
                    "error_message": None,
                    "cause_type": None,
                    "finalization_used": False,
                },
                config={"recursion_limit": self._max_tool_rounds * 2 + 4},
                stream_mode="values",
            ):
                state = cast(dict[str, Any], snapshot)
        except Exception as exc:
            partial_messages = (
                tuple(cast(list[AnyMessage], state["messages"]))
                if state is not None
                else (initial_message,)
            )
            raise AgentRunError(
                str(exc),
                messages=partial_messages,
                error_type=type(exc).__name__,
                termination_reason="recursion_limit"
                if type(exc).__name__ == "GraphRecursionError"
                else "run_error",
                limit_reason=state.get("stop_reason") if state else None,
                finalization_used=bool(state and state.get("finalization_used")),
            ) from exc
        if state is None:
            raise AgentRunError("agent graph produced no state", messages=(initial_message,))
        messages = tuple(cast(list[AnyMessage], state["messages"]))
        if state.get("error_type"):
            raise AgentRunError(
                state["error_message"],
                messages=messages,
                error_type=state["error_type"],
                termination_reason="finalization_failed"
                if state.get("stop_reason")
                else "model_error",
                limit_reason=state.get("stop_reason"),
                cause_type=state.get("cause_type"),
                finalization_used=state["finalization_used"],
            )
        try:
            answer = _final_answer(messages)
        except RuntimeError as exc:
            raise AgentRunError(
                str(exc),
                messages=messages,
                error_type="MissingFinalAnswer",
                termination_reason="missing_final_answer",
            ) from exc
        return AgentRunResult(
            answer=answer,
            messages=messages,
            trajectory=_extract_trajectory(messages),
            status="limited" if state.get("stop_reason") else "succeeded",
            termination_reason=state.get("stop_reason") or "completed",
            finalization_used=state["finalization_used"],
        )

    def run(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        return asyncio.run(self.arun(task, guidelines=guidelines))

    def _build_graph(self) -> Any:
        async def call_agent(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            finalizing = bool(state["stop_reason"])
            failures, successes, _ = call_history(state["messages"])
            instruction = (
                f"Remaining tool rounds: {self._max_tool_rounds - state['tool_rounds']}; "
                f"remaining tool dispatch attempts: {self._max_tool_calls - state['tool_attempts']}. "
                "Use structured error feedback; internal failures are not necessarily argument errors. "
                "Correct failed arguments before another attempt; do not repeat identical failures. "
                "When only one round remains, it is the last opportunity to gather tool evidence."
            )
            if any(count for count in failures.values()):
                instruction += (
                    " Some calls failed: consult their ToolMessage before choosing the next action."
                )
            if any(count >= 2 for count in successes.values()):
                instruction += " Some successful calls were repeated; reuse existing evidence unless another call has a specific purpose."
            if finalizing:
                instruction = (
                    f"Tool execution has stopped: {state['stop_reason']}. No more tools are available. "
                    "Give one final response using only evidence already obtained. Follow the requested "
                    "answer format if the evidence supports an answer; otherwise state that you could not "
                    "complete the task and identify missing evidence. Do not guess or claim completion."
                )
                emit("finalization_started", reason=state["stop_reason"])
            emit(
                "llm_started",
                message_count=len(state["messages"]) + 1,
                phase="finalization" if finalizing else "agent",
                runtime_instruction=instruction,
            )
            try:
                model = self._final_model if finalizing else self._model_with_tools
                response = await model.ainvoke(
                    [
                        SystemMessage(
                            content=self._system_prompt + "\n\nRuntime constraints:\n" + instruction
                        ),
                        *state["messages"],
                    ]
                )
            except Exception as exc:
                emit(
                    "llm_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    latency_ms=_elapsed_ms(started),
                )
                return {
                    "error_type": "FinalizationError" if finalizing else "ModelRequestError",
                    "error_message": str(exc),
                    "cause_type": type(exc).__name__,
                    "finalization_used": finalizing,
                }
            emit(
                "llm_finished",
                message=response.model_dump(mode="json"),
                latency_ms=_elapsed_ms(started),
                token_usage=_token_usage(cast(AIMessage, response)),
            )
            if finalizing and (
                response.tool_calls or response.invalid_tool_calls or not response.text.strip()
            ):
                return {
                    "messages": [response],
                    "error_type": "FinalizationError",
                    "error_message": "finalization returned tool calls or no non-empty answer",
                    "cause_type": "InvalidFinalResponse",
                    "finalization_used": True,
                }
            return {"messages": [cast(AIMessage, response)], "finalization_used": finalizing}

        async def execute_tools(state: AgentState) -> dict[str, Any]:
            message = state["messages"][-1]
            if not isinstance(message, AIMessage):
                raise TypeError("tool node requires the last message to be an AIMessage")

            results: list[ToolMessage] = []
            attempts = state["tool_attempts"]
            stop_reason = state["stop_reason"]
            for call in message.tool_calls:
                started_at = time.perf_counter()
                call_id = call.get("id") or f"call-{len(results)}"
                name = call["name"]
                arguments = call.get("args", {})
                failures, _, internal = call_history([*state["messages"], *results])
                key = signature(name, arguments)
                if not stop_reason:
                    if attempts >= self._max_tool_calls:
                        stop_reason = "tool_call_budget"
                    elif failures[key] >= self._max_repeated_failures or key in internal:
                        stop_reason = "repeated_tool_failure"
                if stop_reason:
                    feedback = ToolErrorFeedback(
                        code="runtime_limit",
                        message=f"Tool call was not attempted: {stop_reason}.",
                        suggested_action="No further tools will execute. Give a final response using existing evidence, or state the missing evidence without guessing.",
                    )
                    blocked = _error_message(
                        call_id, name, feedback.message, feedback=feedback, latency_ms=0
                    )
                    blocked.artifact["attempted"] = False
                    results.append(blocked)
                    emit(
                        "tool_blocked",
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        reason=stop_reason,
                        message=blocked.model_dump(mode="json"),
                    )
                    continue
                attempts += 1
                emit("tool_started", call_id=call_id, name=name, arguments=arguments)
                tool = self._tool_map.get(name)
                if tool is None:
                    results.append(
                        _error_message(
                            call_id,
                            name,
                            f"unknown tool: {name}",
                            feedback=unknown_tool_feedback(name, list(self._tool_map)),
                            latency_ms=_elapsed_ms(started_at),
                        )
                    )
                    emit("tool_finished", message=results[-1].model_dump(mode="json"))
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
                            artifact={"latency_ms": _elapsed_ms(started_at), "attempted": True},
                        )
                    )
                except Exception as exc:
                    results.append(
                        _error_message(
                            call_id,
                            name,
                            str(exc),
                            feedback=error_feedback(exc, tool.contract),
                            diagnostics="".join(traceback.format_exception(exc)),
                            latency_ms=_elapsed_ms(started_at),
                        )
                    )
                emit("tool_finished", message=results[-1].model_dump(mode="json"))
            rounds = state["tool_rounds"] + 1
            if not stop_reason:
                if attempts >= self._max_tool_calls:
                    stop_reason = "tool_call_budget"
                elif rounds >= self._max_tool_rounds:
                    stop_reason = "tool_round_budget"
            if stop_reason:
                emit(
                    "runtime_stopped",
                    reason=stop_reason,
                    tool_rounds=rounds,
                    tool_attempts=attempts,
                )
            return {
                "messages": results,
                "tool_rounds": rounds,
                "tool_attempts": attempts,
                "stop_reason": stop_reason,
            }

        def route_after_agent(state: AgentState) -> str:
            if state.get("error_type") or state["stop_reason"]:
                return END
            message = state["messages"][-1]
            return "tools" if isinstance(message, AIMessage) and message.tool_calls else END

        builder = StateGraph(AgentState)
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
    feedback: ToolErrorFeedback,
    diagnostics: str | None = None,
    latency_ms: float | None = None,
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(feedback.payload(), ensure_ascii=False),
        tool_call_id=call_id,
        name=name,
        status="error",
        artifact={
            "error": message,
            "latency_ms": latency_ms,
            "error_code": feedback.code,
            "diagnostics": diagnostics,
            "attempted": True,
        },
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 3)


def _final_answer(messages: tuple[AnyMessage, ...]) -> str:
    message = messages[-1] if messages else None
    if isinstance(message, AIMessage) and not message.tool_calls and not message.invalid_tool_calls:
        if message.text.strip():
            return message.text
    raise RuntimeError("agent stopped without a non-empty final answer")


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
                    attempted=artifact.get("attempted", True),
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
