"""DuckDB-backed implementation of ``query_data``."""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb
from pydantic import JsonValue

from data_management_agent.tools import Tool, ToolContext, ToolExecutionError
from data_management_agent.tools.contracts import QUERY_DATA
from data_management_agent.tools.schemas import QueryDataInput, QueryDataOutput

from ._files import ensure_supported, table_name_for

_FORBIDDEN_SQL = re.compile(
    r"\b(ATTACH|CALL|COPY|CREATE|DELETE|DROP|EXPORT|IMPORT|INSERT|INSTALL|LOAD|PRAGMA|SET|UPDATE)\b",
    re.IGNORECASE,
)


class QueryDataTool(Tool[QueryDataInput, QueryDataOutput]):
    contract = QUERY_DATA

    async def execute(
        self,
        arguments: QueryDataInput,
        *,
        context: ToolContext,
    ) -> QueryDataOutput:
        query = _validate_query(arguments.query)
        connection = duckdb.connect(database=":memory:")
        buffers: list[io.BytesIO] = []
        relations: list[duckdb.DuckDBPyRelation] = []
        table_names: set[str] = set()
        try:
            for source in arguments.sources:
                entry = await context.workspace.get_entry(source)
                extension = ensure_supported(entry)
                if extension not in {"csv", "json"}:
                    raise ToolExecutionError(f"query_data does not support {entry.path}")
                table_name = table_name_for(entry)
                if table_name in table_names:
                    raise ToolExecutionError(
                        f"multiple sources resolve to SQL table name {table_name!r}"
                    )
                table_names.add(table_name)
                buffer = io.BytesIO(await context.workspace.read_bytes(source))
                buffers.append(buffer)
                relation = (
                    connection.read_csv(buffer, header=True, sample_size=-1)
                    if extension == "csv"
                    else connection.read_json(buffer, format="array")
                )
                relations.append(relation)
                relation.create_view(table_name)

            limited_query = (
                f"SELECT * FROM ({query}) AS __agent_result LIMIT {arguments.max_rows + 1}"
            )
            cursor = connection.execute(limited_query)
            columns = tuple(description[0] for description in cursor.description)
            raw_rows = cursor.fetchall()
        except ToolExecutionError:
            raise
        except duckdb.Error as exc:
            raise ToolExecutionError(f"DuckDB query failed: {exc}") from exc
        finally:
            connection.close()

        truncated = len(raw_rows) > arguments.max_rows
        visible_rows = raw_rows[: arguments.max_rows]
        rows = tuple(
            {column: _json_value(value) for column, value in zip(columns, raw_row, strict=True)}
            for raw_row in visible_rows
        )
        return QueryDataOutput(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )


def _validate_query(query: str) -> str:
    stripped = query.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    if ";" in stripped:
        raise ToolExecutionError("query_data accepts exactly one SQL statement")
    if not re.match(r"^(SELECT|WITH)\b", stripped, flags=re.IGNORECASE):
        raise ToolExecutionError("query_data only accepts SELECT or WITH queries")
    forbidden = _FORBIDDEN_SQL.search(stripped)
    if forbidden:
        raise ToolExecutionError(f"read-only query cannot contain {forbidden.group(1).upper()}")
    return stripped


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)
