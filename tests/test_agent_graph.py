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
