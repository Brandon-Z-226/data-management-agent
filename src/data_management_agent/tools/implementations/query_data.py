"""DuckDB-backed implementation of ``query_data``."""

from __future__ import annotations

import re

import duckdb

from data_management_agent.tools import Tool, ToolContext, ToolExecutionError
from data_management_agent.tools.contracts import QUERY_DATA
from data_management_agent.tools.schemas import QueryDataInput, QueryDataOutput

from ._files import ensure_supported, table_name_for
from ._sql import json_value, materialize_source

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
        table_names: set[str] = set()
        try:
            for source in arguments.sources:
                entry = await context.workspace.get_entry(source)
                extension = ensure_supported(entry)
                if extension not in {"csv", "json"}:
                    raise ToolExecutionError(
                        f"query_data does not support {entry.path}", code="unsupported_operation"
                    )
                table_name = table_name_for(entry)
                if table_name in table_names:
                    raise ToolExecutionError(
                        f"multiple sources resolve to SQL table name {table_name!r}",
                        code="invalid_arguments",
                    )
                table_names.add(table_name)
                materialize_source(
                    connection, await context.workspace.read_bytes(source), extension, table_name
                )

            limited_query = (
                f"SELECT * FROM ({query}) AS __agent_result LIMIT {arguments.max_rows + 1}"
            )
            cursor = connection.execute(limited_query)
            columns = tuple(description[0] for description in cursor.description)
            raw_rows = cursor.fetchall()
        except ToolExecutionError:
            raise
        except (
            duckdb.ParserException,
            duckdb.BinderException,
            duckdb.CatalogException,
            duckdb.ConversionException,
            duckdb.OutOfRangeException,
        ) as exc:
            raise ToolExecutionError(f"DuckDB query failed: {exc}", code="sql_error") from exc
        except duckdb.Error as exc:
            raise ToolExecutionError(
                f"DuckDB execution failed: {exc}", code="internal_error"
            ) from exc
        finally:
            connection.close()

        truncated = len(raw_rows) > arguments.max_rows
        visible_rows = raw_rows[: arguments.max_rows]
        rows = tuple(
            {column: json_value(value) for column, value in zip(columns, raw_row, strict=True)}
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
        raise ToolExecutionError(
            "query_data accepts exactly one SQL statement", code="unsupported_operation"
        )
    if not re.match(r"^(SELECT|WITH)\b", stripped, flags=re.IGNORECASE):
        raise ToolExecutionError(
            "query_data only accepts SELECT or WITH queries", code="unsupported_operation"
        )
    forbidden = _FORBIDDEN_SQL.search(stripped)
    if forbidden:
        raise ToolExecutionError(
            f"read-only query cannot contain {forbidden.group(1).upper()}",
            code="unsupported_operation",
        )
    return stripped
