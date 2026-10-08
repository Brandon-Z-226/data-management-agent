"""Run a small DABstep baseline through the existing WorkspaceAgent API."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_management_agent.agent import WorkspaceAgent
from data_management_agent.benchmark import (
    BenchmarkTaskResult,
    DABstepTask,
    load_tasks,
    run_benchmark,
    select_tasks,
)
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "external" / "dabstep"
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "data" / "external" / "dabstep_benchmark"
DEFAULT_TASKS_FILE = DEFAULT_BENCHMARK_ROOT / "tasks" / "dev.jsonl"
DEFAULT_RESULTS_DIR = DEFAULT_BENCHMARK_ROOT / "results"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DABstep dev benchmark sequentially.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="run the first N tasks")
    selection.add_argument(
        "--task-ids",
        nargs="+",
        help="run specific task IDs; space-separated and/or comma-separated",
    )
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    tasks_file = _project_path(args.tasks_file)
    if not tasks_file.is_file():
        print(
            "error: DABstep dev tasks are missing; run "
            "`uv run python scripts/download_dabstep.py --include-dev-tasks`",
            file=sys.stderr,
        )
        return 2

    try:
        tasks = load_tasks(tasks_file)
        task_ids = _task_ids(args.task_ids or ())
        selected_tasks = select_tasks(tasks, limit=args.limit, task_ids=task_ids)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workspace_path = _workspace_path(args.workspace)
    model_name = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    output_path = _project_path(args.output) if args.output else _default_output(model_name)

    try:
        agent = WorkspaceAgent(
            model=ChatOpenAI(
                model=model_name,
                base_url=base_url,
                use_responses_api=True,
                timeout=120,
                max_retries=2,
            ),
            tools=create_read_only_tools(),
            workspace=LocalWorkspace(workspace_path),
        )
        summary = run_benchmark(
            agent=agent,
            tasks=selected_tasks,
            output_path=output_path,
            model_name=model_name,
            on_task_started=_print_task_started,
            on_task_completed=_print_task_completed,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    accuracy = "n/a" if summary.accuracy is None else f"{summary.accuracy:.1%}"
    overall = (
        "n/a" if summary.overall_correct_rate is None else f"{summary.overall_correct_rate:.1%}"
    )
    completion = "n/a" if summary.completion_rate is None else f"{summary.completion_rate:.1%}"
    print(f"Results: {summary.output_path}")
    print(
        f"Summary: total={summary.total} succeeded={summary.succeeded} failed={summary.failed} "
        f"correct={summary.correct}/{summary.scored} answered_accuracy={accuracy} "
        f"overall_correct={overall} completion={completion}"
    )
    print(f"Failure types: {summary.failure_types or '{}'}")
    return 0


def _workspace_path(argument: Path | None) -> Path:
    configured = os.getenv("DATA_WORKSPACE_PATH") or str(DEFAULT_WORKSPACE)
    return _project_path(argument or Path(configured))


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _task_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(item for value in values for item in value.split(",") if item)


def _default_output(model_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(character if character.isalnum() else "-" for character in model_name)
    return DEFAULT_RESULTS_DIR / f"{timestamp}_{safe_model}.jsonl"


def _print_task_started(index: int, total: int, task: DABstepTask) -> None:
    print(f"[{index}/{total}] task {task.task_id}: running")


def _print_task_completed(result: BenchmarkTaskResult) -> None:
    correctness = result.correctness.is_correct
    outcome = "unscored" if correctness is None else ("correct" if correctness else "incorrect")
    print(
        f"  {result.status}, {outcome}, {result.latency_ms / 1_000:.2f}s, "
        f"tools={len(result.tool_calls)}, errors={len(result.tool_errors)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
