import asyncio
import csv
import io
import json
from pathlib import Path

import pytest

from data_management_agent.tools import ToolContext, ToolExecutionError
from data_management_agent.tools.implementations import ReadFileTool
from data_management_agent.tools.implementations.evidence import EvidenceRecord, EvidenceStore
from data_management_agent.tools.schemas import ContentLocation
from data_management_agent.workspace import LocalWorkspace, WorkspaceRef


def read(path: Path, **kwargs):
    return asyncio.run(
        ReadFileTool().invoke(
            {"file": {"path": path.name}, **kwargs},
            context=ToolContext(workspace=LocalWorkspace(path.parent)),
        )
    )


def test_markdown_default_budget_reports_remaining_content(tmp_path: Path):
    path = tmp_path / "manual.md"
    text = "中" * 20001
    path.write_text(text)
    first = read(path)
    assert first.truncated
    assert first.location.char_end == 20000
    assert first.requested_location.char_end == len(text)
    second = read(path, locator=first.next_locator.model_dump(exclude_none=True))
    assert first.content + second.content == text
    assert not second.truncated and second.next_locator is None


@pytest.mark.parametrize(
    "locator,start,end",
    [
        (None, 0, 35),
        ({"section": "One"}, 8, 25),
        ({"char_start": 10, "char_end": 19}, 10, 19),
    ],
)
def test_text_pages_stay_inside_requested_scope(tmp_path: Path, locator, start, end):
    path = tmp_path / "manual.md"
    text = "# Title\n## One\nabcdefghi\n## Two\nzz\n"
    assert len(text) == 35
    path.write_text(text)
    chunks = []
    position = start
    for _ in range(20):
        result = read(path, locator=locator, max_chars=5)
        assert result.location.char_start == position
        position = result.location.char_end
        chunks.append(result.content)
        assert result.requested_location.char_end == end
        if not result.truncated:
            assert result.next_locator is None
            break
        locator = result.next_locator.model_dump(exclude_none=True)
    else:
        pytest.fail("pagination failed to terminate")
    assert "".join(chunks) == text[start:end]
    assert position == end


def test_complete_section_and_eof_are_not_truncated(tmp_path: Path):
    path = tmp_path / "manual.md"
    path.write_text("## One\nsmall\n## Two\nother\n")
    section = read(path, locator={"section": "One"})
    assert section.content == "## One\nsmall\n"
    assert not section.truncated and section.next_locator is None
    eof = read(path, locator={"char_start": 999})
    assert eof.content == "" and not eof.truncated
    assert eof.location.char_start == eof.location.char_end == len(path.read_text())


@pytest.fixture(params=["csv", "json"])
def row_source(tmp_path: Path, request: pytest.FixtureRequest):
    rows = [
        {"id": i, "text": '中,quoted "value"\nand newline\rand carriage return'} for i in range(45)
    ]
    path = tmp_path / f"rows.{request.param}"
    if request.param == "json":
        path.write_text(json.dumps(rows, ensure_ascii=False))
    else:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerows(rows)
    return path, rows


@pytest.mark.parametrize("max_chars", [160, 20000])
@pytest.mark.parametrize("bounds", [None, (3, 38)])
def test_row_pages_are_complete_and_contiguous(row_source, max_chars, bounds):
    path, expected = row_source
    start, end = bounds or (0, len(expected))
    locator = {"row_start": start, "row_end": end} if bounds else None
    actual = []
    position = start
    for _ in range(50):
        result = read(path, locator=locator, max_chars=max_chars)
        assert len(result.content) <= max_chars
        if path.suffix == ".json":
            batch = json.loads(result.content)
        else:
            batch = [
                {"id": int(r["id"]), "text": r["text"]}
                for r in csv.DictReader(io.StringIO(result.content))
            ]
        assert result.location.row_start == position
        assert result.location.row_end == position + len(batch)
        assert result.requested_location.row_end == end
        actual.extend(batch)
        position += len(batch)
        if not result.truncated:
            assert result.next_locator is None
            break
        assert batch  # Every successful non-final page makes progress.
        locator = result.next_locator.model_dump(exclude_none=True)
    else:
        pytest.fail("pagination failed to terminate")
    assert actual == expected[start:end]


def test_row_too_large_is_an_error_not_a_partial_success(row_source):
    path, _ = row_source
    with pytest.raises(ToolExecutionError, match="max_chars"):
        read(path, max_chars=15)


@pytest.mark.parametrize("content", ['{"note":"large value"}', '"a long scalar value"'])
def test_non_array_json_is_complete_or_errors(tmp_path: Path, content):
    path = tmp_path / "object.json"
    path.write_text(content)
    assert json.loads(read(path).content) == json.loads(content)
    with pytest.raises(ToolExecutionError, match="max_chars"):
        read(path, max_chars=4)
    with pytest.raises(ToolExecutionError, match="locator"):
        read(path, locator={"row_start": 1})


@pytest.mark.parametrize("content", ["", "a", "abcd", "abcde"])
def test_text_budget_boundary(tmp_path: Path, content):
    path = tmp_path / "text.md"
    path.write_text(content)
    result = read(path, max_chars=4)
    assert result.content == content[:4]
    assert result.truncated == (len(content) > 4)


@pytest.mark.parametrize("extension", ["csv", "json"])
@pytest.mark.parametrize("count", [0, 20, 21])
def test_row_default_page_boundary_and_eof(tmp_path: Path, extension, count):
    path = tmp_path / f"rows.{extension}"
    path.write_text(
        "id\n" + "".join(f"{i}\n" for i in range(count))
        if extension == "csv"
        else json.dumps(list(range(count)))
    )
    result = read(path)
    assert result.truncated == (count > 20)
    assert result.location.row_end == min(count, 20)
    assert result.requested_location.row_end == count
    for locator in ({"row_start": 999}, {"row_start": 0, "row_end": 0}):
        empty = read(path, locator=locator)
        assert not empty.truncated and empty.next_locator is None
        assert empty.location.row_start == empty.location.row_end


@pytest.mark.parametrize(
    "extension,content,locator",
    [
        ("md", "abc", {"row_start": 0}),
        ("md", "## One\nabc", {"section": "One", "char_start": 0}),
        ("csv", "id\n1\n", {"char_start": 0}),
        ("json", "[1,2]", {"char_start": 0}),
        ("md", "abc", {"page": 1}),
    ],
)
def test_inapplicable_or_conflicting_locators_are_not_silently_ignored(
    tmp_path: Path, extension, content, locator
):
    path = tmp_path / f"file.{extension}"
    path.write_text(content)
    with pytest.raises(ToolExecutionError, match="locator"):
        read(path, locator=locator)


def test_csv_header_and_json_container_must_fit(tmp_path: Path):
    csv_path = tmp_path / "header.csv"
    csv_path.write_text("long_header\n")
    json_path = tmp_path / "empty.json"
    json_path.write_text("[]")
    for path in (csv_path, json_path):
        with pytest.raises(ToolExecutionError, match="max_chars"):
            read(path, max_chars=1)


def test_evidence_read_returns_bounded_continuation(tmp_path: Path):
    path = tmp_path / "manual.md"
    path.write_text("0123456789 extra content")
    workspace = LocalWorkspace(tmp_path)
    entry = asyncio.run(workspace.get_entry(WorkspaceRef(path=path.name)))
    store = EvidenceStore()
    store.put(
        EvidenceRecord(
            "ev-1", entry.resource_id, entry.path, ContentLocation(char_start=2, char_end=8)
        )
    )
    result = asyncio.run(
        ReadFileTool(store).invoke(
            {"file": {"path": path.name}, "locator": {"evidence_id": "ev-1"}, "max_chars": 3},
            context=ToolContext(workspace=workspace),
        )
    )
    assert result.content == "234" and result.truncated
    remainder = read(path, locator=result.next_locator.model_dump(exclude_none=True))
    assert remainder.content == "567" and not remainder.truncated
