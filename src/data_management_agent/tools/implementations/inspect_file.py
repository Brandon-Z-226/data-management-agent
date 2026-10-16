"""Implementation of ``inspect_file`` for CSV, JSON, and Markdown."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from data_management_agent.tools import Tool, ToolContext, ToolExecutionError
from data_management_agent.tools.contracts import INSPECT_FILE
from data_management_agent.tools.schemas import InspectFileInput, InspectFileOutput

from ._files import decode_text, ensure_supported, table_name_for
from ._sql import inspect_sql_source

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class InspectFileTool(Tool[InspectFileInput, InspectFileOutput]):
    contract = INSPECT_FILE

    async def execute(
        self,
        arguments: InspectFileInput,
        *,
        context: ToolContext,
    ) -> InspectFileOutput:
        entry = await context.workspace.get_entry(arguments.file)
        extension = ensure_supported(entry)
        content = await context.workspace.read_bytes(arguments.file)

        if extension == "csv":
            structure = inspect_sql_source(content, extension, table_name_for(entry))
            summary = f"CSV table with {structure['row_count']} rows and {len(structure['columns'])} SQL columns."
            file_format = "csv"
        elif extension == "json":
            structure, summary = _inspect_json(decode_text(content, path=str(entry.path)))
            if structure["root_type"] == "array":
                structure.update(inspect_sql_source(content, extension, table_name_for(entry)))
            else:
                structure["queryable"] = False
            file_format = "json"
        else:
            structure, summary = _inspect_markdown(decode_text(content, path=str(entry.path)))
            file_format = "markdown"

        return InspectFileOutput(
            entry=entry,
            format=file_format,
            summary=summary,
            structure=structure,
        )


def _inspect_json(text: str) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"invalid JSON: {exc}", code="data_error") from exc

    if isinstance(value, list):
        keys = _ordered_keys(item for item in value[:100] if isinstance(item, dict))
        structure: dict[str, Any] = {
            "root_type": "array",
            "row_count": len(value),
            "keys": keys,
            "sample_rows": value[:5],
        }
        return structure, f"JSON array with {len(value)} items and {len(keys)} observed keys."
    if isinstance(value, dict):
        keys = list(value)
        return {
            "root_type": "object",
            "keys": keys,
            "sample": value,
        }, f"JSON object with {len(keys)} top-level keys."
    return {"root_type": type(value).__name__, "sample": value}, "JSON scalar value."


def _inspect_markdown(text: str) -> tuple[dict[str, Any], str]:
    sections = [
        {"level": len(match.group(1)), "title": match.group(2), "char_start": match.start()}
        for match in _HEADING_PATTERN.finditer(text)
    ]
    structure: dict[str, Any] = {
        "characters": len(text),
        "lines": text.count("\n") + 1,
        "sections": sections,
    }
    return structure, f"Markdown document with {len(sections)} headings."


def _ordered_keys(items: Iterable[dict[str, Any]]) -> list[str]:
    keys: dict[str, None] = {}
    for item in items:
        for key in item:
            keys.setdefault(key, None)
    return list(keys)
