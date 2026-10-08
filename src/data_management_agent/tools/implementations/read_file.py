"""Implementation of ``read_file`` for targeted DABstep content access."""

from __future__ import annotations

import csv
import io
import json
import re

from data_management_agent.tools import Tool, ToolContext, ToolExecutionError
from data_management_agent.tools.contracts import READ_FILE
from data_management_agent.tools.schemas import (
    ContentLocation,
    ReadFileInput,
    ReadFileOutput,
)
from data_management_agent.workspace import WorkspaceRef

from ._files import decode_text, ensure_supported
from .evidence import EvidenceStore

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class ReadFileTool(Tool[ReadFileInput, ReadFileOutput]):
    contract = READ_FILE

    def __init__(self, evidence_store: EvidenceStore | None = None) -> None:
        self._evidence_store = evidence_store or EvidenceStore()

    async def execute(
        self,
        arguments: ReadFileInput,
        *,
        context: ToolContext,
    ) -> ReadFileOutput:
        entry = await context.workspace.get_entry(arguments.file)
        extension = ensure_supported(entry)
        locator = self._resolve_evidence(arguments.locator, entry.resource_id, str(entry.path))
        if locator is not None and (locator.page is not None or locator.sheet is not None):
            raise ToolExecutionError("page and sheet locators are not supported for this file type")

        raw_content = await context.workspace.read_bytes(
            WorkspaceRef(resource_id=entry.resource_id)
        )
        text = decode_text(raw_content, path=str(entry.path))

        if extension == "csv":
            content, actual_location = _read_csv(text, locator)
        elif extension == "json":
            content, actual_location = _read_json(text, locator)
        else:
            content, actual_location = _read_markdown(text, locator, arguments.max_chars)

        truncated = len(content) > arguments.max_chars
        if truncated:
            content = content[: arguments.max_chars]
        return ReadFileOutput(
            entry=entry,
            content=content,
            media_type=entry.media_type or "text/plain",
            location=actual_location,
            truncated=truncated,
        )

    def _resolve_evidence(
        self,
        locator: ContentLocation | None,
        resource_id: str,
        path: str,
    ) -> ContentLocation | None:
        if locator is None or locator.evidence_id is None:
            return locator
        record = self._evidence_store.get(locator.evidence_id)
        if record is None:
            raise ToolExecutionError(f"unknown or expired evidence id: {locator.evidence_id}")
        if record.resource_id != resource_id:
            raise ToolExecutionError(
                f"evidence {locator.evidence_id} belongs to {record.path}, not {path}"
            )
        return record.location


def _read_csv(text: str, locator: ContentLocation | None) -> tuple[str, ContentLocation]:
    start = locator.row_start if locator and locator.row_start is not None else 0
    end = locator.row_end if locator and locator.row_end is not None else start + 20
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ToolExecutionError("CSV file is empty") from exc
    rows = [row for index, row in enumerate(reader) if start <= index < end]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue(), ContentLocation(row_start=start, row_end=start + len(rows))


def _read_json(text: str, locator: ContentLocation | None) -> tuple[str, ContentLocation | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        return json.dumps(value, indent=2, ensure_ascii=False), None

    start = locator.row_start if locator and locator.row_start is not None else 0
    end = locator.row_end if locator and locator.row_end is not None else start + 20
    selected = value[start:end]
    return (
        json.dumps(selected, indent=2, ensure_ascii=False),
        ContentLocation(row_start=start, row_end=start + len(selected)),
    )


def _read_markdown(
    text: str,
    locator: ContentLocation | None,
    max_chars: int,
) -> tuple[str, ContentLocation]:
    if locator is not None and locator.section is not None:
        start, end = _section_range(text, locator.section)
    else:
        start = locator.char_start if locator and locator.char_start is not None else 0
        end = locator.char_end if locator and locator.char_end is not None else start + max_chars
    start = min(start, len(text))
    end = min(max(end, start), len(text))
    return text[start:end], ContentLocation(char_start=start, char_end=end)


def _section_range(text: str, section: str) -> tuple[int, int]:
    headings = list(_HEADING_PATTERN.finditer(text))
    query = section.casefold()
    for index, heading in enumerate(headings):
        if query not in heading.group(2).casefold():
            continue
        level = len(heading.group(1))
        end = len(text)
        for next_heading in headings[index + 1 :]:
            if len(next_heading.group(1)) <= level:
                end = next_heading.start()
                break
        return heading.start(), end
    raise ToolExecutionError(f"Markdown section not found: {section}")
