import asyncio
import json
from pathlib import Path

import pytest

from data_management_agent.tools import ToolContext, ToolExecutionError
from data_management_agent.tools.implementations import (
    InspectFileTool,
    ListWorkspaceTool,
    QueryDataTool,
    ReadFileTool,
    SearchWorkspaceTool,
)
from data_management_agent.tools.implementations.evidence import EvidenceStore
from data_management_agent.workspace import LocalWorkspace


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspace:
    (tmp_path / "manual.md").write_text(
        "# Merchant manual\n\n## Fee formula\n"
        "The fee is fixed_amount + rate * transaction_value / 10000.\n"
    )
    (tmp_path / "payments.csv").write_text(
        "merchant,amount,country\nCoffee,10,NL\nBooks,20,BE\nCoffee,30,NL\n"
    )
    (tmp_path / "fees.json").write_text(
        json.dumps(
            [
                {"merchant": "Coffee", "fixed_amount": 0.1, "rate": 20},
                {"merchant": "Books", "fixed_amount": 0.2, "rate": 30},
            ]
        )
    )
    return LocalWorkspace(tmp_path)


def test_discovery_inspection_search_and_evidence_read(workspace: LocalWorkspace) -> None:
    context = ToolContext(workspace=workspace)
    evidence_store = EvidenceStore()

    listed = asyncio.run(
        ListWorkspaceTool().invoke(
            {"path": ".", "recursive": True, "file_types": ["md"]},
            context=context,
        )
    )
    assert [str(entry.path) for entry in listed.entries] == ["manual.md"]

    inspected = asyncio.run(
        InspectFileTool().invoke({"file": {"path": "payments.csv"}}, context=context)
    )
    assert inspected.structure["row_count"] == 3
    assert inspected.structure["table_name"] == "payments"

    searched = asyncio.run(
        SearchWorkspaceTool(evidence_store).invoke(
            {"query": "fixed amount transaction value", "file_types": ["md"]},
            context=context,
        )
    )
    assert searched.evidence[0].path.root == "manual.md"

    read = asyncio.run(
        ReadFileTool(evidence_store).invoke(
            {
                "file": {"path": "manual.md"},
                "locator": {"evidence_id": searched.evidence[0].evidence_id},
            },
            context=context,
        )
    )
    assert "fixed_amount" in read.content


def test_query_data_joins_csv_and_json_from_workspace_bytes(workspace: LocalWorkspace) -> None:
    result = asyncio.run(
        QueryDataTool().invoke(
            {
                "sources": [{"path": "payments.csv"}, {"path": "fees.json"}],
                "query": """
                    SELECT p.merchant, COUNT(*) AS payments, AVG(f.fixed_amount) AS fixed_fee
                    FROM payments p
                    JOIN fees f USING (merchant)
                    GROUP BY p.merchant
                    ORDER BY payments DESC
                """,
            },
            context=ToolContext(workspace=workspace),
        )
    )

    assert result.rows[0] == {"merchant": "Coffee", "payments": 2, "fixed_fee": 0.1}


def test_query_data_rejects_mutating_sql(workspace: LocalWorkspace) -> None:
    with pytest.raises(ToolExecutionError, match="only accepts SELECT or WITH"):
        asyncio.run(
            QueryDataTool().invoke(
                {
                    "sources": [{"path": "payments.csv"}],
                    "query": "DELETE FROM payments",
                },
                context=ToolContext(workspace=workspace),
            )
        )
