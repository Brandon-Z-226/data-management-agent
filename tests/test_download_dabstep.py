from pathlib import Path
from typing import Any

import pytest

from scripts.download_dabstep import (
    CONTEXT_FILES,
    CONTEXT_PREFIX,
    DEV_TASK_PATH,
    DownloadError,
    download_context,
    download_dev_tasks,
)


def test_existing_files_are_skipped_without_contacting_hub(tmp_path: Path) -> None:
    destination = tmp_path / "dabstep"
    destination.mkdir()
    for name in CONTEXT_FILES:
        (destination / name).write_text(name)

    def fail_if_called(**_: Any) -> str:
        raise AssertionError("snapshot_download should not be called")

    results = download_context(destination, snapshot_download_fn=fail_if_called)

    assert all(result.status == "skipped" for result in results)


def test_force_download_requests_only_context_files_and_replaces_targets(tmp_path: Path) -> None:
    destination = tmp_path / "dabstep"
    destination.mkdir()
    (destination / CONTEXT_FILES[0]).write_text("old")
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        snapshot_root = Path(kwargs["local_dir"])
        context_dir = snapshot_root / CONTEXT_PREFIX
        context_dir.mkdir(parents=True)
        for name in CONTEXT_FILES:
            (context_dir / name).write_text(f"new:{name}")
        return str(snapshot_root)

    results = download_context(
        destination,
        force=True,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert captured["repo_id"] == "adyen/DABstep"
    assert captured["repo_type"] == "dataset"
    assert captured["allow_patterns"] == [f"data/context/{name}" for name in CONTEXT_FILES]
    assert all("task_scores" not in pattern for pattern in captured["allow_patterns"])
    assert (destination / CONTEXT_FILES[0]).read_text() == f"new:{CONTEXT_FILES[0]}"
    assert all(result.status == "downloaded" for result in results)


def test_missing_file_in_snapshot_reports_clear_error(tmp_path: Path) -> None:
    def incomplete_snapshot(**kwargs: Any) -> str:
        return str(kwargs["local_dir"])

    with pytest.raises(DownloadError, match="did not contain required files"):
        download_context(tmp_path / "dabstep", snapshot_download_fn=incomplete_snapshot)


def test_download_dev_tasks_requests_only_dev_manifest(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        snapshot_root = Path(kwargs["local_dir"])
        source = snapshot_root / DEV_TASK_PATH
        source.parent.mkdir(parents=True)
        source.write_text('{"task_id":"5","question":"Question?"}\n')
        return str(snapshot_root)

    destination = tmp_path / "dabstep" / "tasks" / "dev.jsonl"
    result = download_dev_tasks(destination, snapshot_download_fn=fake_snapshot_download)

    assert captured["allow_patterns"] == ["data/tasks/dev.jsonl"]
    assert all("task_scores" not in pattern for pattern in captured["allow_patterns"])
    assert all("submissions" not in pattern for pattern in captured["allow_patterns"])
    assert destination.is_file()
    assert result.status == "downloaded"
