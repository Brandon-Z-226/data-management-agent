"""Scripted models test runtime control, not model planning or benchmark accuracy."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from data_management_agent.agent import AgentRunError, WorkspaceAgent
from data_management_agent.agent.runtime import call_history, signature
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace


class SplitModel(FakeMessagesListChatModel):
    """Unbound responses can only be reached via the finalization request."""

    bound: FakeMessagesListChatModel

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeMessagesListChatModel:
        return self.bound


def call(name: str = "list_workspace", args: dict | None = None, id: str = "c"):
    return {"name": name, "args": args or {}, "id": id, "type": "tool_call"}


def batch(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def make_agent(root, responses, final=None, **limits):
    model = SplitModel(
        responses=[final if final is not None else AIMessage(content="Insufficient evidence.")],
        bound=FakeMessagesListChatModel(responses=responses),
    )
    return WorkspaceAgent(
        model=model,
        workspace=LocalWorkspace(root),
        tools=create_read_only_tools(),
        **limits,
    )


def test_round_limit_uses_unbound_finalization_and_resets_each_run(tmp_path: Path):
    agent = make_agent(tmp_path, [batch(call())], max_tool_rounds=1)
    for _ in range(2):
        result = agent.run("test")
        assert result.answer == "Insufficient evidence."
        assert result.status == "limited"
        assert result.termination_reason == "tool_round_budget"
        assert result.finalization_used
        assert len(result.trajectory) == 1
        assert result.trajectory[0].attempted


def test_batch_limit_pairs_all_messages_but_only_dispatches_permitted_calls(tmp_path: Path):
    result = make_agent(
        tmp_path,
        [batch(call(id="a"), call(id="b"), call("nonexistent", id="c"))],
        max_tool_calls=1,
    ).run("test")
    assert result.termination_reason == "tool_call_budget"
    assert [trace.call_id for trace in result.trajectory] == ["a", "b", "c"]
    assert [trace.attempted for trace in result.trajectory] == [True, False, False]
    assert result.trajectory[0].status == "success"
    assert all('"code": "runtime_limit"' in t.result for t in result.trajectory[1:])
    assert [m.tool_call_id for m in result.messages if isinstance(m, ToolMessage)] == [
        "a",
        "b",
        "c",
    ]


@pytest.mark.parametrize("single_batch", [False, True])
def test_identical_failures_stop_before_third_dispatch(tmp_path: Path, single_batch: bool):
    calls = [call("read_file", {"file": {"path": "missing.md"}}, str(i)) for i in range(3)]
    responses = [batch(*calls)] if single_batch else [batch(c) for c in calls]
    result = make_agent(tmp_path, responses).run("test")
    assert result.termination_reason == "repeated_tool_failure"
    assert [trace.attempted for trace in result.trajectory] == [True, True, False]


def test_corrected_call_and_repeated_success_are_not_blocked(tmp_path: Path):
    responses = [batch(call("list_workspace", {"nonsense": 1}, "bad"))]
    responses += [batch(call(id=str(i))) for i in range(3)]
    responses += [AIMessage(content="done")]
    result = make_agent(tmp_path, responses).run("test")
    assert result.status == "succeeded"
    assert result.termination_reason == "completed"
    assert not result.finalization_used
    assert [t.status for t in result.trajectory] == ["error", "success", "success", "success"]
    assert all(t.attempted for t in result.trajectory)


def test_internal_error_is_not_retried_identically(tmp_path: Path, monkeypatch):
    tool = create_read_only_tools()[0]
    invocations = []

    async def fail(*args, **kwargs):
        invocations.append(1)
        raise RuntimeError("simulated internal defect")

    monkeypatch.setattr(type(tool), "invoke", fail)
    result = make_agent(
        tmp_path, [batch(call(tool.contract.name, id=str(i))) for i in range(2)]
    ).run("test")
    assert result.termination_reason == "repeated_tool_failure"
    assert [t.attempted for t in result.trajectory] == [True, False]
    assert len(invocations) == 1


def test_history_normalizes_key_order_and_success_resets_failures():
    args = {"a": 1, "b": 2}
    assert signature("t", args) == signature("t", {"b": 2, "a": 1})
    messages = []
    for i, status in enumerate(["error", "success", "error"]):
        messages += [
            batch(call("t", args, str(i))),
            ToolMessage(content="result", tool_call_id=str(i), status=status),
        ]
    failures, successes, internal = call_history(messages)
    assert failures[signature("t", args)] == 1
    assert successes[signature("t", args)] == 1
    assert not internal


@pytest.mark.parametrize("final", [AIMessage(content=""), batch(call())])
def test_invalid_finalization_is_failure_not_success_or_recursion(tmp_path: Path, final):
    agent = make_agent(tmp_path, [batch(call())], final=final, max_tool_calls=1)
    with pytest.raises(AgentRunError) as caught:
        agent.run("test")
    error = caught.value
    assert error.error_type == "FinalizationError"
    assert error.termination_reason == "finalization_failed"
    assert error.limit_reason == "tool_call_budget"
    assert error.cause_type == "InvalidFinalResponse"
    assert len(error.trajectory) == 1


def test_finalization_provider_failure_preserves_cause_and_evidence(tmp_path: Path, monkeypatch):
    async def fail(*args, **kwargs):
        raise TimeoutError("simulated finalization outage")

    monkeypatch.setattr(SplitModel, "_agenerate", fail)
    agent = make_agent(tmp_path, [batch(call())], max_tool_rounds=1)
    with pytest.raises(AgentRunError) as caught:
        agent.run("test")
    assert caught.value.error_type == "FinalizationError"
    assert caught.value.cause_type == "TimeoutError"
    assert caught.value.limit_reason == "tool_round_budget"
    assert caught.value.finalization_used
    assert len(caught.value.trajectory) == 1


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content=" "),
        AIMessage(
            content="bad",
            invalid_tool_calls=[
                {
                    "name": "t",
                    "args": "{",
                    "id": "broken",
                    "error": "invalid JSON",
                    "type": "invalid_tool_call",
                }
            ],
        ),
    ],
)
def test_missing_final_answer_is_not_mislabeled_recursion(tmp_path: Path, response):
    with pytest.raises(AgentRunError) as caught:
        make_agent(tmp_path, [response]).run("test")
    assert caught.value.error_type == "MissingFinalAnswer"
    assert caught.value.termination_reason == "missing_final_answer"


def test_actual_graph_recursion_is_still_classified(tmp_path: Path):
    class BrokenGraph:
        async def astream(self, *args, **kwargs):
            yield {"messages": [HumanMessage(content="partial")], "stop_reason": None}
            raise GraphRecursionError("safety guard")

    agent = make_agent(tmp_path, [AIMessage(content="unused")])
    agent._graph = BrokenGraph()
    with pytest.raises(AgentRunError) as caught:
        agent.run("test")
    assert caught.value.error_type == "GraphRecursionError"
    assert caught.value.termination_reason == "recursion_limit"
    assert caught.value.messages[0].content == "partial"
