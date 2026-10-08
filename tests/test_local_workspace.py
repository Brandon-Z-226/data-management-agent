import asyncio
from pathlib import Path

import pytest

from data_management_agent.workspace import (
    LocalWorkspace,
    ResourceKind,
    WorkspaceError,
    WorkspacePath,
    WorkspaceQuery,
    WorkspaceRef,
)


def test_local_workspace_lists_filters_and_reads_files(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("# Report\nEvidence")
    (tmp_path / "data.csv").write_text("value\n42\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.json").write_text("[]")
    workspace = LocalWorkspace(tmp_path, workspace_id="test")

    page = asyncio.run(
        workspace.list_entries(
            WorkspaceQuery(
                recursive=True,
                kinds={ResourceKind.FILE},
                extensions={"md"},
            )
        )
    )

    assert [str(entry.path) for entry in page.entries] == ["report.md"]
    content = asyncio.run(workspace.read_bytes(WorkspaceRef(path="report.md")))
    assert content == b"# Report\nEvidence"
    assert page.entries[0].resource_id.startswith("local-")


def test_local_workspace_rejects_symlinks_that_escape_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace_root / "escape.txt").symlink_to(outside)
    workspace = LocalWorkspace(workspace_root)

    with pytest.raises(WorkspaceError, match="escapes workspace root"):
        asyncio.run(workspace.get_entry(WorkspaceRef(path=WorkspacePath("escape.txt"))))
