import asyncio
import csv
import json
from pathlib import Path

import pytest

from data_management_agent.tools import ToolContext, ToolExecutionError
from data_management_agent.tools.implementations import QueryDataTool
from data_management_agent.workspace import LocalWorkspace


@pytest.fixture(params=["csv", "json"])
def query_source(tmp_path: Path, request: pytest.FixtureRequest):
    rows = [
        {
            "id": i,
            "category": "A" if i % 2 == 0 else "B",
            "credit": (True, False, None)[i % 3],
            "value": i + 1,
        }
        for i in range(1000)
    ]
    path = tmp_path / f"records.{request.param}"
    if request.param == "json":
        path.write_text(json.dumps(rows))
    else:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path, rows


def invoke(path: Path, query: str, **kwargs):
    return asyncio.run(
        QueryDataTool().invoke(
            {"sources": [{"path": path.name}], "query": query, **kwargs},
            context=ToolContext(workspace=LocalWorkspace(path.parent)),
        )
    )


def test_repeated_filtered_union_scans_are_stable(query_source):
    path, rows = query_source
    groups = [
        [r for r in rows if r["category"] == "A" and r["credit"] is True],
        [r for r in rows if r["category"] == "A" and r["credit"] is not False],
        [r for r in rows if r["category"] == "A"],
    ]
    query = """
        SELECT 1 AS part, COUNT(*) AS n, AVG(value) AS mean FROM records
        WHERE category = 'A' AND credit = true
        UNION ALL
        SELECT 2, COUNT(*), AVG(value) FROM records
        WHERE category = 'A' AND (credit = true OR credit IS NULL)
        UNION ALL
        SELECT 3, COUNT(*), AVG(value) FROM records WHERE category = 'A'
        ORDER BY part
    """
    for _ in range(20):
        result = invoke(path, query)
        assert len(result.rows) == 3
        for actual, expected in zip(result.rows, groups, strict=True):
            assert actual["n"] == len(expected)
            assert actual["mean"] == pytest.approx(
                sum(r["value"] for r in expected) / len(expected)
            )


@pytest.mark.parametrize(
    "query",
    [
        "SELECT (SELECT COUNT(*) FROM records WHERE credit = true) AS n, "
        "(SELECT SUM(value) FROM records) AS total",
        "WITH r AS NOT MATERIALIZED (SELECT * FROM records) "
        "SELECT (SELECT COUNT(*) FROM r WHERE credit = true) AS n, "
        "(SELECT SUM(value) FROM r) AS total",
        "SELECT COUNT(*) FILTER (WHERE a.credit = true) AS n, SUM(a.value) AS total "
        "FROM records a JOIN records b ON a.id = b.id",
    ],
)
def test_subqueries_ctes_and_self_joins(query_source, query):
    path, rows = query_source
    for _ in range(5):
        result = invoke(path, query)
        assert result.rows == (
            {"n": sum(r["credit"] is True for r in rows), "total": sum(r["value"] for r in rows)},
        )


def test_limits_readonly_and_no_workspace_mutation(query_source):
    path, _ = query_source
    before = path.read_bytes()
    result = invoke(path, "SELECT * FROM records ORDER BY id", max_rows=2)
    assert result.row_count == 2 and result.truncated
    assert [r["id"] for r in result.rows] == [0, 1]
    with pytest.raises(ToolExecutionError):
        invoke(path, "CREATE TABLE wrong AS SELECT * FROM records")
    with pytest.raises(ToolExecutionError, match="DuckDB query failed"):
        invoke(path, "SELECT missing_column FROM records")
    assert invoke(path, "SELECT COUNT(*) AS n FROM records").rows == ({"n": 1000},)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_query_consumes_workspace_bytes_not_local_paths(tmp_path: Path, monkeypatch):
    path = tmp_path / "records.json"
    path.write_text('[{"value": -999}]')
    workspace = LocalWorkspace(tmp_path)
    calls = []

    async def read_bytes(ref, **kwargs):
        calls.append(ref)
        return b'[{"value": 42}]'

    monkeypatch.setattr(workspace, "read_bytes", read_bytes)
    result = asyncio.run(
        QueryDataTool().invoke(
            {"sources": [{"path": path.name}], "query": "SELECT value FROM records"},
            context=ToolContext(workspace=workspace),
        )
    )
    assert result.rows == ({"value": 42},)
    assert len(calls) == 1
