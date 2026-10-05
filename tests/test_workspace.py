from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from data_management_agent.workspace import WorkspacePath, WorkspaceQuery, WorkspaceRef


def test_workspace_path_is_normalized_and_relative() -> None:
    assert str(WorkspacePath("./reports//weekly.md")) == "reports/weekly.md"
    assert str(WorkspacePath("")) == "."


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "reports/../../secret", r"a\b"])
def test_workspace_path_rejects_escaping_or_platform_specific_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        WorkspacePath(path)


def test_workspace_ref_requires_exactly_one_locator() -> None:
    assert WorkspaceRef(path="report.md").path == WorkspacePath("report.md")
    assert WorkspaceRef(resource_id="file-1").resource_id == "file-1"

    with pytest.raises(ValidationError):
        WorkspaceRef()
    with pytest.raises(ValidationError):
        WorkspaceRef(resource_id="file-1", path="report.md")


def test_workspace_query_rejects_reversed_time_range() -> None:
    with pytest.raises(ValidationError):
        WorkspaceQuery(
            modified_after=datetime(2026, 2, 1, tzinfo=UTC),
            modified_before=datetime(2026, 1, 1, tzinfo=UTC),
        )
