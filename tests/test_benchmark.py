from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from data_management_agent.agent import AgentRunError, AgentRunResult, ToolTrace, WorkspaceAgent
from data_management_agent.benchmark import (
    DABstepTask,
    evaluate_answer,
    load_results,
    load_tasks,
    run_benchmark,
    select_tasks,
)


class StubAgent:
    def __init__(self, result: AgentRunResult | Exception) -> None:
        self.result = result
        self.loop_ids: list[int] = []

    async def arun(self, task: str, *, guidelines: str | None = None) -> AgentRunResult:
        self.loop_ids.append(id(asyncio.get_running_loop()))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_load_and_select_tasks(tmp_path: Path) -> None:
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text(
        '{"task_id":"5","question":"Question 5?","answer":"NL"}\n'
        '{"task_id":"49","question":"Question 49?","answer":"FR"}\n'
    )

    tasks = load_tasks(manifest)

    assert [task.task_id for task in select_tasks(tasks, limit=1)] == ["5"]
    assert [task.task_id for task in select_tasks(tasks, task_ids=("49", "5"))] == [
        "49",
        "5",
    ]


def test_normalized_exact_match_is_transparent_and_conservative() -> None:
    equivalent = evaluate_answer(" A,  B ", "a,b")
    different = evaluate_answer("NL because it has the most transactions", "NL")
    hidden_reference = evaluate_answer("anything", "")

    assert equivalent.is_correct is True
    assert equivalent.exact_match is False
    assert different.is_correct is False
    assert hidden_reference.score is None


def test_run_benchmark_writes_complete_single_turn_record(tmp_path: Path) -> None:
    tool_call = {
        "name": "query_data",
        "args": {"sources": [{"path": "payments.csv"}], "query": "SELECT 'NL'"},
        "id": "call-query",
        "type": "tool_call",
    }
    messages = (
        HumanMessage(content="Which country?"),
        AIMessage(
            content="",
            tool_calls=[tool_call],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
        ToolMessage(
            content='{"rows":[{"country":"NL"}]}',
            tool_call_id="call-query",
            name="query_data",
            status="success",
        ),
        AIMessage(
            content="NL",
            usage_metadata={"input_tokens": 20, "output_tokens": 1, "total_tokens": 21},
        ),
    )
    agent_result = AgentRunResult(
        answer="NL",
        messages=messages,
        trajectory=(
            ToolTrace(
                call_id="call-query",
                name="query_data",
                arguments=tool_call["args"],
                status="success",
                result='{"rows":[{"country":"NL"}]}',
                latency_ms=2.5,
                token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
        ),
    )
    output_path = tmp_path / "run.jsonl"

    summary = run_benchmark(
        agent=cast(WorkspaceAgent, StubAgent(agent_result)),
        tasks=(DABstepTask(task_id="5", question="Which country?", answer="NL"),),
        output_path=output_path,
        model_name="test-model",
        run_id="run-test",
    )
    result = load_results(output_path)[0]

    assert summary.accuracy == 1.0
    assert summary.completion_rate == 1.0
    assert summary.overall_correct_rate == 1.0
    assert result.run_id == "run-test"
    assert result.task_id == "5"
    assert result.conversation_id != result.thread_id
    assert result.turn_index == 0
    assert result.correctness.is_correct is True
    assert len(result.trajectory) == 4
    assert result.tool_calls[0].name == "query_data"
    assert result.tool_errors == ()
    assert result.token_usage == {"input_tokens": 30, "output_tokens": 6, "total_tokens": 36}


def test_run_benchmark_persists_task_failure(tmp_path: Path) -> None:
    output_path = tmp_path / "failed.jsonl"

    summary = run_benchmark(
        agent=cast(WorkspaceAgent, StubAgent(RuntimeError("model unavailable"))),
        tasks=(DABstepTask(task_id="5", question="Which country?", answer="NL"),),
        output_path=output_path,
        model_name="test-model",
    )
    result = load_results(output_path)[0]

    assert summary.failed == 1
    assert result.status == "failed"
    assert result.error == "model unavailable"
    assert result.failure_types == ("run_error", "missing_final_answer", "no_tool_calls")


def test_run_benchmark_uses_one_event_loop_for_all_tasks(tmp_path: Path) -> None:
    result = AgentRunResult(
        answer="NL",
        messages=(HumanMessage(content="Question?"), AIMessage(content="NL")),
        trajectory=(),
    )
    stub = StubAgent(result)

    run_benchmark(
        agent=cast(WorkspaceAgent, stub),
        tasks=(
            DABstepTask(task_id="1", question="Question 1?", answer="NL"),
            DABstepTask(task_id="2", question="Question 2?", answer="NL"),
        ),
        output_path=tmp_path / "loop.jsonl",
        model_name="test-model",
    )

    assert len(stub.loop_ids) == 2
    assert len(set(stub.loop_ids)) == 1


def test_run_benchmark_persists_partial_trajectory_on_agent_error(tmp_path: Path) -> None:
    messages = (
        HumanMessage(content="Question?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "missing_tool",
                    "args": {},
                    "id": "call-missing",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"error":"unknown tool"}',
            tool_call_id="call-missing",
            name="missing_tool",
            status="error",
            artifact={"error": "unknown tool", "latency_ms": 1.0},
        ),
    )
    error = AgentRunError(
        "recursion stopped",
        messages=messages,
        error_type="GraphRecursionError",
    )

    run_benchmark(
        agent=cast(WorkspaceAgent, StubAgent(error)),
        tasks=(DABstepTask(task_id="1", question="Question?", answer="NL"),),
        output_path=tmp_path / "partial.jsonl",
        model_name="test-model",
    )
    result = load_results(tmp_path / "partial.jsonl")[0]

    assert result.error_type == "GraphRecursionError"
    assert len(result.trajectory) == 3
    assert result.tool_calls[0].name == "missing_tool"
    assert result.tool_errors[0].error == "unknown tool"
    assert "recursion_limit" in result.failure_types
    assert "tool_error" in result.failure_types
