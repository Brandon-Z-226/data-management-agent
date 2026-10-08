"""Small, task-agnostic DABstep benchmark runner primitives."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel, ConfigDict, Field

from data_management_agent.agent import AgentRunError, ToolTrace, WorkspaceAgent

SCHEMA_VERSION = "dabstep-benchmark-v0"
CORRECTNESS_METHOD = "normalized_exact_match_v0"


class DABstepTask(BaseModel):
    """One task from the official DABstep task manifest."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = ""
    guidelines: str = ""
    level: str | None = None


class CorrectnessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["normalized_exact_match_v0"] = CORRECTNESS_METHOD
    score: float | None = None
    is_correct: bool | None = None
    exact_match: bool | None = None
    normalized_match: bool | None = None


class ToolCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["success", "error"]
    result: str
    error: str | None = None
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None


class ToolErrorResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    call_id: str
    name: str
    error: str


class BenchmarkTaskResult(BaseModel):
    """One JSONL record for one independent, single-turn task run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["dabstep-benchmark-v0"] = SCHEMA_VERSION
    run_id: str
    conversation_id: str
    thread_id: str
    turn_id: str
    turn_index: int = 0
    task_id: str
    dataset_repo: str = "adyen/DABstep"
    dataset_revision: str = "main"
    dataset_split: str = "dev"
    model: str
    level: str | None = None
    question: str
    guidelines: str
    reference_answer: str
    final_answer: str | None = None
    status: Literal["succeeded", "failed"]
    error_type: str | None = None
    error: str | None = None
    correctness: CorrectnessResult
    trajectory: tuple[dict[str, Any], ...] = ()
    tool_calls: tuple[ToolCallResult, ...] = ()
    tool_errors: tuple[ToolErrorResult, ...] = ()
    latency_ms: float = Field(ge=0)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    failure_types: tuple[str, ...] = ()
    started_at: datetime
    completed_at: datetime


class BenchmarkRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    output_path: Path
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    scored: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float | None = None
    completion_rate: float | None = None
    overall_correct_rate: float | None = None
    failure_types: dict[str, int] = Field(default_factory=dict)


TaskStarted = Callable[[int, int, DABstepTask], None]
TaskCompleted = Callable[[BenchmarkTaskResult], None]


def load_tasks(path: Path) -> tuple[DABstepTask, ...]:
    """Load and validate a JSONL task manifest without dataset processing."""

    tasks: list[DABstepTask] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read task file {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            task = DABstepTask.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"invalid task at {path}:{line_number}: {exc}") from exc
        if task.task_id in seen_ids:
            raise ValueError(f"duplicate task_id {task.task_id!r} in {path}")
        seen_ids.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"task file contains no tasks: {path}")
    return tuple(tasks)


def select_tasks(
    tasks: Sequence[DABstepTask],
    *,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
) -> tuple[DABstepTask, ...]:
    if limit is not None and task_ids:
        raise ValueError("limit and task_ids are mutually exclusive")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return tuple(tasks[:limit])
    if not task_ids:
        return tuple(tasks)

    by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"task IDs not found: {', '.join(missing)}")
    return tuple(by_id[task_id] for task_id in task_ids)


def evaluate_answer(final_answer: str | None, reference_answer: str) -> CorrectnessResult:
    """Apply a transparent baseline metric without task-specific answer logic."""

    if final_answer is None or not reference_answer.strip():
        return CorrectnessResult(
            score=None,
            is_correct=None,
            exact_match=None,
            normalized_match=None,
        )
    exact_match = final_answer.strip() == reference_answer.strip()
    normalized_match = _normalize_answer(final_answer) == _normalize_answer(reference_answer)
    return CorrectnessResult(
        score=1.0 if normalized_match else 0.0,
        is_correct=normalized_match,
        exact_match=exact_match,
        normalized_match=normalized_match,
    )


def run_benchmark(
    *,
    agent: WorkspaceAgent,
    tasks: Sequence[DABstepTask],
    output_path: Path,
    model_name: str,
    run_id: str | None = None,
    on_task_started: TaskStarted | None = None,
    on_task_completed: TaskCompleted | None = None,
) -> BenchmarkRunSummary:
    """Run independent single-turn tasks and flush one structured result per line."""

    return asyncio.run(
        _run_benchmark_async(
            agent=agent,
            tasks=tasks,
            output_path=output_path,
            model_name=model_name,
            run_id=run_id,
            on_task_started=on_task_started,
            on_task_completed=on_task_completed,
        )
    )


async def _run_benchmark_async(
    *,
    agent: WorkspaceAgent,
    tasks: Sequence[DABstepTask],
    output_path: Path,
    model_name: str,
    run_id: str | None,
    on_task_started: TaskStarted | None,
    on_task_completed: TaskCompleted | None,
) -> BenchmarkRunSummary:
    """Keep all model calls on one event loop while running tasks sequentially."""

    if not tasks:
        raise ValueError("at least one task is required")
    if output_path.exists():
        raise FileExistsError(f"benchmark output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or str(uuid4())
    results: list[BenchmarkTaskResult] = []

    with output_path.open("x", encoding="utf-8") as output:
        for index, task in enumerate(tasks, start=1):
            if on_task_started:
                on_task_started(index, len(tasks), task)
            result = await _run_task(
                agent=agent,
                task=task,
                model_name=model_name,
                run_id=resolved_run_id,
            )
            output.write(result.model_dump_json(exclude_none=True) + "\n")
            output.flush()
            results.append(result)
            if on_task_completed:
                on_task_completed(result)

    return _summarize(resolved_run_id, output_path, results)


def load_results(path: Path) -> tuple[BenchmarkTaskResult, ...]:
    results: list[BenchmarkTaskResult] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read benchmark results {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            results.append(BenchmarkTaskResult.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid result at {path}:{line_number}: {exc}") from exc
    return tuple(results)


async def _run_task(
    *,
    agent: WorkspaceAgent,
    task: DABstepTask,
    model_name: str,
    run_id: str,
) -> BenchmarkTaskResult:
    conversation_id = str(uuid4())
    thread_id = str(uuid4())
    turn_id = str(uuid4())
    started_at = datetime.now(UTC)
    started_clock = time.perf_counter()

    try:
        agent_result = await agent.arun(task.question, guidelines=task.guidelines or None)
    except Exception as exc:
        completed_at = datetime.now(UTC)
        latency_ms = (time.perf_counter() - started_clock) * 1_000
        correctness = evaluate_answer(None, task.answer)
        messages = exc.messages if isinstance(exc, AgentRunError) else ()
        traces = exc.trajectory if isinstance(exc, AgentRunError) else ()
        tool_calls = _tool_calls(traces)
        tool_errors = _tool_errors(tool_calls)
        failure_types = ["run_error", "missing_final_answer"]
        error_type = exc.error_type if isinstance(exc, AgentRunError) else type(exc).__name__
        if error_type == "GraphRecursionError" or "Recursion limit" in str(exc):
            failure_types.append("recursion_limit")
        if tool_errors:
            failure_types.append("tool_error")
        if not tool_calls:
            failure_types.append("no_tool_calls")
        return BenchmarkTaskResult(
            run_id=run_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            turn_id=turn_id,
            task_id=task.task_id,
            model=model_name,
            level=task.level,
            question=task.question,
            guidelines=task.guidelines,
            reference_answer=task.answer,
            status="failed",
            error_type=error_type,
            error=str(exc),
            correctness=correctness,
            trajectory=_message_trajectory(messages),
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            latency_ms=latency_ms,
            token_usage=_aggregate_token_usage(messages),
            failure_types=tuple(failure_types),
            started_at=started_at,
            completed_at=completed_at,
        )

    completed_at = datetime.now(UTC)
    latency_ms = (time.perf_counter() - started_clock) * 1_000
    correctness = evaluate_answer(agent_result.answer, task.answer)
    tool_calls = _tool_calls(agent_result.trajectory)
    tool_errors = _tool_errors(tool_calls)
    failure_types = _failure_types(
        correctness=correctness,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
    )
    return BenchmarkTaskResult(
        run_id=run_id,
        conversation_id=conversation_id,
        thread_id=thread_id,
        turn_id=turn_id,
        task_id=task.task_id,
        model=model_name,
        level=task.level,
        question=task.question,
        guidelines=task.guidelines,
        reference_answer=task.answer,
        final_answer=agent_result.answer,
        status="succeeded",
        correctness=correctness,
        trajectory=_message_trajectory(agent_result.messages),
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        latency_ms=latency_ms,
        token_usage=_aggregate_token_usage(agent_result.messages),
        failure_types=failure_types,
        started_at=started_at,
        completed_at=completed_at,
    )


def _tool_calls(traces: Sequence[ToolTrace]) -> tuple[ToolCallResult, ...]:
    return tuple(
        ToolCallResult(
            sequence=index,
            call_id=trace.call_id,
            name=trace.name,
            arguments=trace.arguments,
            status=trace.status,
            result=trace.result,
            error=trace.error,
            latency_ms=trace.latency_ms,
            token_usage=trace.token_usage,
        )
        for index, trace in enumerate(traces)
    )


def _tool_errors(tool_calls: Sequence[ToolCallResult]) -> tuple[ToolErrorResult, ...]:
    return tuple(
        ToolErrorResult(
            sequence=call.sequence,
            call_id=call.call_id,
            name=call.name,
            error=call.error or call.result,
        )
        for call in tool_calls
        if call.status == "error"
    )


def _message_trajectory(messages: Iterable[AnyMessage]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "sequence": index,
            **message.model_dump(mode="json", exclude_none=True),
        }
        for index, message in enumerate(messages)
    )


def _aggregate_token_usage(messages: Iterable[AnyMessage]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata
        if not usage:
            candidate = message.response_metadata.get(
                "token_usage"
            ) or message.response_metadata.get("usage")
            usage = candidate if isinstance(candidate, dict) else None
        if usage:
            _merge_numeric_mapping(total, dict(usage))
    return total


def _merge_numeric_mapping(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_numeric_mapping(nested, value)


def _normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return re.sub(r"\s*,\s*", ",", normalized)


def _failure_types(
    *,
    correctness: CorrectnessResult,
    tool_calls: Sequence[ToolCallResult],
    tool_errors: Sequence[ToolErrorResult],
) -> tuple[str, ...]:
    failures: list[str] = []
    if correctness.is_correct is False:
        failures.append("incorrect_answer")
    if tool_errors:
        failures.append("tool_error")
    if not tool_calls:
        failures.append("no_tool_calls")
    return tuple(failures)


def _summarize(
    run_id: str,
    output_path: Path,
    results: Sequence[BenchmarkTaskResult],
) -> BenchmarkRunSummary:
    scored = [result for result in results if result.correctness.score is not None]
    correct = sum(result.correctness.is_correct is True for result in scored)
    failure_counts = Counter(
        failure_type for result in results for failure_type in result.failure_types
    )
    return BenchmarkRunSummary(
        run_id=run_id,
        output_path=output_path,
        total=len(results),
        succeeded=sum(result.status == "succeeded" for result in results),
        failed=sum(result.status == "failed" for result in results),
        scored=len(scored),
        correct=correct,
        accuracy=correct / len(scored) if scored else None,
        completion_rate=(
            sum(result.status == "succeeded" for result in results) / len(results)
            if results
            else None
        ),
        overall_correct_rate=correct / len(results) if results else None,
        failure_types=dict(sorted(failure_counts.items())),
    )
