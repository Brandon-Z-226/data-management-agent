"""Implementation of ``read_file`` for targeted DABstep content access."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class _ReadPage:
    content: str
    location: ContentLocation | None = None
    requested_location: ContentLocation | None = None
    next_locator: ContentLocation | None = None


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
        _validate_locator(locator, extension)

        raw_content = await context.workspace.read_bytes(
            WorkspaceRef(resource_id=entry.resource_id)
        )
        text = decode_text(raw_content, path=str(entry.path))

        if extension == "csv":
            page = _read_csv(text, locator, arguments.max_chars)
        elif extension == "json":
            page = _read_json(text, locator, arguments.max_chars)
        else:
            page = _read_markdown(text, locator, arguments.max_chars)

        return ReadFileOutput(
            entry=entry,
            content=page.content,
            media_type=entry.media_type or "text/plain",
            location=page.location,
            requested_location=page.requested_location,
            next_locator=page.next_locator,
            truncated=page.next_locator is not None,
        )

    def _resolve_evidence(
        self,
        locator: ContentLocation | None,
        resource_id: str,
        path: str,
    ) -> ContentLocation | None:
        if locator is None or locator.evidence_id is None:
            return locator
        if len(locator.model_dump(exclude_none=True)) != 1:
            raise ToolExecutionError(
                "evidence_id cannot be combined with other locator fields", code="invalid_arguments"
            )
        record = self._evidence_store.get(locator.evidence_id)
        if record is None:
            raise ToolExecutionError(
                f"unknown or expired evidence id: {locator.evidence_id}",
                code="invalid_arguments",
                suggested_action="Search the workspace again and use an evidence_id from the current search results.",
            )
        if record.resource_id != resource_id:
            raise ToolExecutionError(
                f"evidence {locator.evidence_id} belongs to {record.path}, not {path}",
                code="invalid_arguments",
            )
        return record.location


def _validate_locator(locator: ContentLocation | None, extension: str) -> None:
    if locator is None:
        return
    fields = set(locator.model_dump(exclude_none=True))
    allowed = (
        {"row_start", "row_end"}
        if extension in {"csv", "json"}
        else {"section", "char_start", "char_end"}
    )
    if fields - allowed:
        raise ToolExecutionError(
            f"unsupported {extension} locator fields: {sorted(fields - allowed)}",
            code="unsupported_operation",
            suggested_action="Use section OR character ranges for Markdown, row ranges for CSV/JSON arrays, or an evidence_id alone. page/sheet are unsupported.",
        )
    if "section" in fields and fields != {"section"}:
        raise ToolExecutionError(
            "section cannot be combined with character locator fields", code="invalid_arguments"
        )


def _csv_record(row: list[str]) -> str:
    output = io.StringIO(newline="")
    # CRLF also quotes embedded carriage returns; every returned record stays parseable.
    csv.writer(output, lineterminator="\r\n").writerow(row)
    return output.getvalue()


def _read_csv(text: str, locator: ContentLocation | None, max_chars: int) -> _ReadPage:
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ToolExecutionError("CSV file is empty", code="data_error") from exc
    return _read_rows(reader, locator, max_chars, _csv_record, prefix=_csv_record(header))


def _read_json(text: str, locator: ContentLocation | None, max_chars: int) -> _ReadPage:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"invalid JSON: {exc}", code="data_error") from exc
    if not isinstance(value, list):
        if locator is not None:
            raise ToolExecutionError(
                "row locators require a JSON array", code="unsupported_operation"
            )
        content = json.dumps(value, indent=2, ensure_ascii=False)
        if len(content) > max_chars:
            raise ToolExecutionError(
                f"complete JSON value requires {len(content)} characters, exceeding "
                f"max_chars={max_chars}; non-array JSON pagination is not supported",
                code="invalid_arguments",
                suggested_action="Increase max_chars within its 200000 upper bound to fit the whole value. If it cannot fit, report this reading limitation.",
            )
        return _ReadPage(content)
    return _read_rows(
        value,
        locator,
        max_chars,
        lambda row: json.dumps(row, ensure_ascii=False),
        prefix="[",
        suffix="]",
        separator=",\n",
    )


def _read_rows(
    rows: Iterable[Any],
    locator: ContentLocation | None,
    max_chars: int,
    serialize: Callable[[Any], str],
    *,
    prefix: str,
    suffix: str = "",
    separator: str = "",
) -> _ReadPage:
    start = locator.row_start if locator and locator.row_start is not None else 0
    requested_end = locator.row_end if locator else None
    # Keep the original default of 20 rows; an explicit end requests that entire range.
    page_limit = 20 if requested_end is None else None
    size = len(prefix) + len(suffix)
    if size > max_chars:
        raise ToolExecutionError(
            f"max_chars={max_chars} cannot fit the {size}-character header/container",
            code="invalid_arguments",
            suggested_action="Increase max_chars to fit the header/container and at least one complete record, within its 200000 upper bound.",
        )
    parts: list[str] = []
    scope_end = 0
    full = False
    for index, row in enumerate(rows):
        if requested_end is not None and index >= requested_end:
            break
        scope_end = index + 1
        if index < start or full:
            continue
        part = serialize(row)
        required = size + (len(separator) if parts else 0) + len(part)
        if required > max_chars:
            if not parts:
                raise ToolExecutionError(
                    f"row {index} requires at least {required} characters including framing; "
                    f"max_chars={max_chars} is too small for a complete record",
                    code="invalid_arguments",
                    suggested_action="Increase max_chars within its 200000 upper bound, or use query_data to select fewer fields. Do not retry the same budget.",
                )
            full = True
            continue
        parts.append(part)
        size = required
        full = page_limit is not None and len(parts) >= page_limit
    start = min(start, scope_end)
    actual_end = start + len(parts)
    next_locator = None
    if actual_end < scope_end:
        # Preserve open-ended paging; preserve explicit ends to avoid reading past a selection.
        next_locator = ContentLocation(
            row_start=actual_end, row_end=scope_end if requested_end is not None else None
        )
    return _ReadPage(
        content=prefix + separator.join(parts) + suffix,
        location=ContentLocation(row_start=start, row_end=actual_end),
        requested_location=ContentLocation(row_start=start, row_end=scope_end),
        next_locator=next_locator,
    )


def _read_markdown(
    text: str,
    locator: ContentLocation | None,
    max_chars: int,
) -> _ReadPage:
    if locator is not None and locator.section is not None:
        start, end = _section_range(text, locator.section)
    else:
        start = locator.char_start if locator and locator.char_start is not None else 0
        end = locator.char_end if locator and locator.char_end is not None else len(text)
    start = min(start, len(text))
    end = min(max(end, start), len(text))
    actual_end = min(end, start + max_chars)
    return _ReadPage(
        content=text[start:actual_end],
        location=ContentLocation(char_start=start, char_end=actual_end),
        requested_location=ContentLocation(char_start=start, char_end=end),
        next_locator=ContentLocation(char_start=actual_end, char_end=end)
        if actual_end < end
        else None,
    )


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
    raise ToolExecutionError(
        f"Markdown section not found: {section}",
        code="invalid_arguments",
        suggested_action="Use inspect_file to discover actual Markdown headings, then select an existing section.",
    )
