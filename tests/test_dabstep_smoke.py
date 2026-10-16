from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from data_management_agent.agent import AgentRunResult, WorkspaceAgent
from data_management_agent.tools import ToolContext
from data_management_agent.tools.implementations import (
    QueryDataTool,
    ReadFileTool,
    create_read_only_tools,
)
from data_management_agent.workspace import LocalWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DABSTEP_ROOT = PROJECT_ROOT / "data" / "external" / "dabstep"
SMOKE_TASKS = json.loads(
    (PROJECT_ROOT / "tests" / "fixtures" / "dabstep_dev_smoke.json").read_text()
)

pytestmark = pytest.mark.skipif(
    not (DABSTEP_ROOT / "payments.csv").is_file(),
    reason="DABstep context is not downloaded",
)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFakeModel:
        return self


def _run(task_index: int, responses: list[AIMessage]) -> AgentRunResult:
    task = SMOKE_TASKS[task_index]
    agent = WorkspaceAgent(
        model=ToolCallingFakeModel(responses=responses),
        tools=create_read_only_tools(),
        workspace=LocalWorkspace(DABSTEP_ROOT, workspace_id="dabstep"),
    )
    return agent.run(task["question"], guidelines=task["guidelines"])


def test_dev_task_5_structured_data_pipeline() -> None:
    result = _run(
        0,
        [
            AIMessage(
                content="",
                tool_calls=[_call("inspect_file", {"file": {"path": "payments.csv"}}, "inspect")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "query_data",
                        {
                            "sources": [{"path": "payments.csv"}],
                            "query": (
                                "SELECT issuing_country, COUNT(*) AS transactions "
                                "FROM payments GROUP BY issuing_country "
                                "ORDER BY transactions DESC LIMIT 1"
                            ),
                        },
                        "query",
                    )
                ],
            ),
            AIMessage(content="NL"),
        ],
    )

    assert result.answer == SMOKE_TASKS[0]["answer"]
    assert [step.name for step in result.trajectory] == ["inspect_file", "query_data"]
    assert '"issuing_country":"NL"' in result.trajectory[-1].result


def test_dev_task_1305_document_and_data_pipeline() -> None:
    result = run_dev_task_1305_smoke()

    assert result.answer == SMOKE_TASKS[1]["answer"]
    assert [step.name for step in result.trajectory] == [
        "list_workspace",
        "search_workspace",
        "read_file",
        "inspect_file",
        "inspect_file",
        "query_data",
    ]
    assert '"average_fee":0.123217' in result.trajectory[-1].result


def test_real_fees_repeated_scans_are_stable() -> None:
    query = """
        SELECT 1 AS part, COUNT(*) AS n, AVG(fixed_amount + rate * 10.0 / 10000.0) AS fee
        FROM fees WHERE card_scheme = 'GlobalCard' AND is_credit = true
        UNION ALL
        SELECT 2, COUNT(*), AVG(fixed_amount + rate * 10.0 / 10000.0)
        FROM fees WHERE card_scheme = 'GlobalCard' AND (is_credit = true OR is_credit IS NULL)
        UNION ALL
        SELECT 3, COUNT(*), AVG(fixed_amount + rate * 10.0 / 10000.0)
        FROM fees WHERE card_scheme = 'GlobalCard' ORDER BY part
    """

    async def check():
        tool = QueryDataTool()
        context = ToolContext(workspace=LocalWorkspace(DABSTEP_ROOT))
        for _ in range(20):
            result = await tool.invoke(
                {"sources": [{"path": "fees.json"}], "query": query}, context=context
            )
            assert [row["n"] for row in result.rows] == [123, 144, 257]
            assert result.rows[1]["fee"] == pytest.approx(0.12013194444444449)

    asyncio.run(check())


def test_real_manual_can_be_read_completely_by_following_locator() -> None:
    async def check():
        tool = ReadFileTool()
        context = ToolContext(workspace=LocalWorkspace(DABSTEP_ROOT))
        first = await tool.invoke({"file": {"path": "manual.md"}}, context=context)
        assert first.truncated and len(first.content) == 20000
        second = await tool.invoke(
            {"file": {"path": "manual.md"}, "locator": first.next_locator}, context=context
        )
        assert not second.truncated and second.next_locator is None
        # Compare decoded characters, not bytes or newline-normalized text.
        expected = (DABSTEP_ROOT / "manual.md").read_bytes().decode("utf-8-sig")
        assert first.content + second.content == expected

    asyncio.run(check())


def run_dev_task_1305_smoke() -> AgentRunResult:
    sql = """
        WITH target_mcc AS (
            SELECT CAST(mcc AS INTEGER) AS mcc
            FROM merchant_category_codes
            WHERE description = 'Eating Places and Restaurants'
        ), applicable AS (
            SELECT fixed_amount + rate * 10 / 10000.0 AS fee
            FROM fees, target_mcc
            WHERE card_scheme = 'GlobalCard'
              AND (account_type IS NULL OR length(account_type) = 0
                   OR list_contains(account_type, 'H'))
              AND (merchant_category_code IS NULL OR length(merchant_category_code) = 0
                   OR list_contains(merchant_category_code, target_mcc.mcc))
        )
        SELECT ROUND(AVG(fee), 6) AS average_fee FROM applicable
    """
    return _run(
        1,
        [
            AIMessage(
                content="",
                tool_calls=[_call("list_workspace", {"path": ".", "recursive": False}, "list")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "search_workspace",
                        {
                            "query": "fee fixed_amount rate transaction_value formula",
                            "file_types": ["md"],
                            "top_k": 3,
                        },
                        "search",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "read_file",
                        {
                            "file": {"path": "manual.md"},
                            "locator": {"section": "Understanding Payment Processing Fees"},
                        },
                        "read",
                    ),
                    _call(
                        "inspect_file",
                        {"file": {"path": "merchant_category_codes.csv"}},
                        "inspect-mcc",
                    ),
                    _call(
                        "inspect_file",
                        {"file": {"path": "fees.json"}},
                        "inspect-fees",
                    ),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _call(
                        "query_data",
                        {
                            "sources": [
                                {"path": "merchant_category_codes.csv"},
                                {"path": "fees.json"},
                            ],
                            "query": sql,
                        },
                        "query",
                    )
                ],
            ),
            AIMessage(content="0.123217"),
        ],
    )


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "args": arguments,
        "id": call_id,
        "type": "tool_call",
    }
