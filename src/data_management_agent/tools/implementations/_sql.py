"""One SQL loading convention shared by inspection and querying."""

from __future__ import annotations

import io
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb
from pydantic import JsonValue

from data_management_agent.tools import ToolExecutionError


def materialize_source(
    connection: duckdb.DuckDBPyConnection, raw: bytes, extension: str, table_name: str
) -> None:
    try:
        with io.BytesIO(raw) as buffer:
            relation = (
                connection.read_csv(buffer, header=True, sample_size=-1)
                if extension == "csv"
                else connection.read_json(buffer, format="array")
            )
            # Consume a file-like source once; repeat SQL scans use a real in-memory table.
            relation.create(table_name)
    except (
        duckdb.InvalidInputException,
        duckdb.ParserException,
        duckdb.ConversionException,
        duckdb.BinderException,
    ) as exc:
        raise ToolExecutionError(f"DuckDB source parsing failed: {exc}", code="data_error") from exc


def inspect_sql_source(raw: bytes, extension: str, table_name: str) -> dict[str, Any]:
    connection = duckdb.connect(database=":memory:")
    try:
        materialize_source(connection, raw, extension, table_name)
        table = connection.table(table_name)
        names = table.columns
        return {
            "table_name": table_name,
            "columns": [
                {"name": name, "type": str(kind)}
                for name, kind in zip(names, table.types, strict=True)
            ],
            "row_count": table.count("*").fetchone()[0],
            "sample_rows": [
                {name: json_value(value) for name, value in zip(names, row, strict=True)}
                for row in table.limit(5).fetchall()
            ],
            "queryable": True,
        }
    finally:
        connection.close()


def json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)
