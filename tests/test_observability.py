from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from data_management_agent.agent import AgentRunError, WorkspaceAgent
from data_management_agent.benchmark import DABstepTask, load_results, run_benchmark
from data_management_agent.observability import RunContext, RunJournal, capture_version, run_scope
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFakeModel:
        return self


def agent_at(root: Path, responses: list[AIMessage], **kwargs: Any) -> WorkspaceAgent:
    return WorkspaceAgent(
        model=ToolCallingFakeModel(responses=responses),
        tools=create_read_only_tools(),
        workspace=LocalWorkspace(root),
        journal=RunJournal(root),
        **kwargs,
    )


def events_at(root: Path, context: RunContext) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "logs" / "runs" / f"{context.execution_id}.jsonl")
        .read_text()
        .splitlines()
    ]


def call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call-1", "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def test_success_journal_and_secret_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-secret-not-a-real-key")
    agent = agent_at(tmp_path, [call("list_workspace", {"path": "."}), AIMessage(content="done")])
    context = RunContext(source="test", conversation_id="conversation-one")
    with run_scope(context):
        result = agent.run("task fake-secret-not-a-real-key", guidelines="short")
    assert result.answer == "done"
    events = events_at(tmp_path, context)
    assert [e["event"] for e in events] == [
        "run_started",
        "llm_started",
        "llm_finished",
        "tool_started",
        "tool_finished",
        "llm_started",
        "llm_finished",
        "run_finished",
    ]
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert {e["conversation_id"] for e in events} == {"conversation-one"}
    assert events[0]["data"]["task"] == "task [REDACTED]"
    assert events[3]["data"]["arguments"] == {"path": "."}
    tool = events[4]["data"]["message"]
    assert tool["status"] == "success"
    assert tool["artifact"]["latency_ms"] >= 0
    final = events[-1]["data"]
    assert final["status"] == "succeeded"
    assert final["final_answer"] == "done"
    assert final["tool_calls"] == 1
    assert final["tool_errors"] == 0
    assert final["token_usage"]["total_tokens"] == 15
    assert final["usage_reports"] == 1  # Do not invent usage for unreported responses.
    version_path = tmp_path / "logs" / "versions" / f"{events[0]['version_id']}.json"
    snapshot = json.loads(version_path.read_text())
    assert snapshot["config"]["system_prompt"]
    assert len(snapshot["config"]["tools"]) == 5
    assert len((tmp_path / "logs" / "index.jsonl").read_text().splitlines()) == 2
    # A subsequent unscoped run must not inherit the previous conversation IDs.
    agent.run("another task")
    other = next(
        p for p in (tmp_path / "logs" / "runs").glob("*.jsonl") if p.stem != context.execution_id
    )
    assert json.loads(other.read_text().splitlines()[0])["conversation_id"] != "conversation-one"


def test_finalization_failure_preserves_validation_error(tmp_path: Path):
    agent = agent_at(
        tmp_path,
        [call("read_file", {"file": {"path": "manual.md", "max_chars": 20}})],
        max_tool_rounds=1,
    )
    context = RunContext(source="test")
    with run_scope(context), pytest.raises(AgentRunError):
        agent.run("repeat invalid call")
    events = events_at(tmp_path, context)
    tool_events = [e for e in events if e["event"] == "tool_finished"]
    assert tool_events[0]["data"]["message"]["status"] == "error"
    assert "max_chars" in tool_events[0]["data"]["message"]["content"]
    final = events[-1]["data"]
    assert final["status"] == "failed"
    assert final["error_type"] == "FinalizationError"
    assert final["limit_reason"] == "tool_round_budget"
    assert final["tool_errors"] >= 1


def test_benchmark_execution_links_to_event_log(tmp_path: Path):
    agent = agent_at(tmp_path, [AIMessage(content="NL")])
    path = tmp_path / "results.jsonl"
    run_benchmark(
        agent=agent,
        tasks=[DABstepTask(task_id="5", question="Country?", answer="NL")],
        output_path=path,
        model_name="fake",
    )
    result = load_results(path)[0]
    event_path = tmp_path / "logs" / "runs" / f"{result.execution_id}.jsonl"
    start = json.loads(event_path.read_text().splitlines()[0])
    assert start["run_id"] == result.run_id
    assert start["task_id"] == "5"
    assert start["thread_id"] == result.thread_id
    assert start["source"] == "benchmark"


def test_limited_run_logs_blocked_calls_and_finalization_separately(tmp_path: Path):
    calls = [
        {"name": "list_workspace", "args": {}, "id": str(i), "type": "tool_call"} for i in range(3)
    ]
    agent = agent_at(
        tmp_path,
        [AIMessage(content="", tool_calls=calls), AIMessage(content="Partial answer")],
        max_tool_calls=1,
    )
    context = RunContext(source="test")
    with run_scope(context):
        result = agent.run("test")
    assert result.status == "limited"
    events = events_at(tmp_path, context)
    final = events[-1]["data"]
    assert final["status"] == "limited"
    assert final["termination_reason"] == "tool_call_budget"
    assert final["finalization_used"]
    assert final["tool_calls"] == 1
    assert final["blocked_tool_calls"] == 2
    assert final["tool_errors"] == 0
    assert final["llm_calls"] == 2
    assert len([e for e in events if e["event"] == "finalization_started"]) == 1
    starts = [e["data"] for e in events if e["event"] == "llm_started"]
    assert starts[-1]["phase"] == "finalization"
    assert "No more tools" in starts[-1]["runtime_instruction"]


def test_fingerprint_detects_uncommitted_source_changes(tmp_path: Path):
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("version = 1\n")
    first = capture_version(tmp_path)
    source.write_text("version = 2\n")
    second = capture_version(tmp_path)
    assert first["source_sha256"] != second["source_sha256"]


def test_logging_failure_does_not_fail_agent(tmp_path: Path):
    (tmp_path / "logs").write_text("not a directory")
    agent = agent_at(tmp_path, [AIMessage(content="still works")])
    with pytest.warns(UserWarning, match="Run journal unavailable"):
        assert agent.run("hello").answer == "still works"


def test_model_failure_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def fail(*args: Any, **kwargs: Any):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(ToolCallingFakeModel, "_agenerate", fail)
    agent = agent_at(tmp_path, [AIMessage(content="unused")])
    context = RunContext(source="test")
    with run_scope(context), pytest.raises(AgentRunError):
        agent.run("hello")
    events = events_at(tmp_path, context)
    assert [e["event"] for e in events] == [
        "run_started",
        "llm_started",
        "llm_failed",
        "run_finished",
    ]
    assert events[-1]["data"]["error_type"] == "ModelRequestError"
    assert events[-1]["data"]["cause_type"] == "RuntimeError"
    assert events[-1]["data"]["token_usage"] is None


def test_concurrent_runs_keep_separate_contexts(tmp_path: Path):
    agent = agent_at(tmp_path, [AIMessage(content="done")])
    contexts = [RunContext(source="test"), RunContext(source="test")]

    async def one(context: RunContext):
        with run_scope(context):
            await agent.arun(context.task_id)

    async def both():
        await asyncio.gather(*(one(context) for context in contexts))

    asyncio.run(both())
    for context in contexts:
        events = events_at(tmp_path, context)
        assert {e["execution_id"] for e in events} == {context.execution_id}
        assert events[0]["data"]["task"] == context.task_id
        assert events[-1]["data"]["llm_calls"] == 1
