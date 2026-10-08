from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from data_management_agent.agent import WorkspaceAgent
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFakeModel:
        return self


def test_langgraph_loops_through_project_tool_interface(tmp_path: Path) -> None:
    (tmp_path / "values.csv").write_text("country\nNL\nBE\nNL\n")
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                },
                tool_calls=[
                    {
                        "name": "query_data",
                        "args": {
                            "sources": [{"path": "values.csv"}],
                            "query": (
                                "SELECT country, COUNT(*) AS count FROM values "
                                "GROUP BY country ORDER BY count DESC LIMIT 1"
                            ),
                        },
                        "id": "call-query",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="NL"),
        ]
    )
    workspace = LocalWorkspace(tmp_path)
    agent = WorkspaceAgent(
        model=model,
        tools=create_read_only_tools(),
        workspace=workspace,
    )

    result = agent.run("Which country occurs most often?")

    assert result.answer == "NL"
    assert [step.name for step in result.trajectory] == ["query_data"]
    assert '"country":"NL"' in result.trajectory[0].result
    assert result.trajectory[0].error is None
    assert result.trajectory[0].latency_ms is not None
    assert result.trajectory[0].latency_ms >= 0
    assert result.trajectory[0].token_usage == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }


def test_trajectory_records_tool_error(tmp_path: Path) -> None:
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "missing_tool",
                        "args": {"value": "bad call"},
                        "id": "call-missing",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Could not complete the request."),
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=create_read_only_tools(),
        workspace=LocalWorkspace(tmp_path),
    )

    result = agent.run("Call a missing tool")

    trace = result.trajectory[0]
    assert trace.status == "error"
    assert trace.error == "unknown tool: missing_tool"
    assert trace.latency_ms is not None
    assert trace.latency_ms >= 0
