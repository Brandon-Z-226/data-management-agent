import asyncio
import json
from pathlib import Path

import pytest

from data_management_agent.tools import ToolContext
from data_management_agent.tools.implementations import InspectFileTool, QueryDataTool
from data_management_agent.workspace import LocalWorkspace


def inspect_and_query(path: Path):
    async def run():
        context = ToolContext(workspace=LocalWorkspace(path.parent))
        inspected = await InspectFileTool().invoke({"file": {"path": path.name}}, context=context)
        table = inspected.structure["table_name"]
        queried = await QueryDataTool().invoke(
            {"sources": [{"path": path.name}], "query": f'SELECT * FROM "{table}"'}, context=context
        )
        return inspected, queried

    return asyncio.run(run())


def test_csv_inspection_uses_sql_column_names_and_types(tmp_path):
    path = tmp_path / "2026 Data.csv"
    path.write_text(",name,name,day,flag\n1,A,B,2026-01-01,true\n2,C,D,2026-01-02,false\n")
    inspected, queried = inspect_and_query(path)
    structure = inspected.structure
    assert structure["table_name"] == "t_2026_data"
    names = [column["name"] for column in structure["columns"]]
    assert names == list(queried.columns)
    assert names[0] == "column0" and len(set(names)) == len(names)
    types = {column["name"]: column["type"] for column in structure["columns"]}
    assert types["day"] == "DATE" and types["flag"] == "BOOLEAN"
    assert structure["sample_rows"] == list(queried.rows)


def test_json_inspection_exposes_actual_sql_schema(tmp_path):
    path = tmp_path / "values.json"
    path.write_text(json.dumps([{"value": 1, "tags": ["a"]}, {"value": 2, "tags": []}]))
    inspected, queried = inspect_and_query(path)
    assert inspected.structure["queryable"]
    assert [c["name"] for c in inspected.structure["columns"]] == list(queried.columns)
    assert inspected.structure["columns"][1]["type"] == "VARCHAR[]"


def test_json_object_inspection_does_not_advertise_a_query_table(tmp_path):
    path = tmp_path / "object.json"
    path.write_text('{"key": 1}')
    result = asyncio.run(
        InspectFileTool().invoke(
            {"file": {"path": path.name}}, context=ToolContext(workspace=LocalWorkspace(tmp_path))
        )
    )
    assert result.structure["queryable"] is False
    assert "table_name" not in result.structure


@pytest.mark.parametrize("filename", ["acquirer_countries.csv", "merchant_category_codes.csv"])
def test_real_dabstep_headers_match_duckdb(filename):
    path = Path(__file__).resolve().parents[1] / "data" / "external" / "dabstep" / filename
    if not path.exists():
        pytest.skip("DABstep context not downloaded")
    inspected, queried = inspect_and_query(path)
    assert [c["name"] for c in inspected.structure["columns"]] == list(queried.columns)
    assert queried.columns[0] == "column0"
