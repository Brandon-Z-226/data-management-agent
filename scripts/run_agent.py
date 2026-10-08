"""Run the minimal LangGraph workspace agent against a local workspace."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_management_agent.agent import WorkspaceAgent
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "external" / "dabstep"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DABstep workspace agent.")
    parser.add_argument("task", help="the question or data task for the agent")
    parser.add_argument("--guidelines", help="optional answer-format requirements")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="workspace directory (default: DATA_WORKSPACE_PATH or downloaded DABstep context)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name (default: OPENAI_MODEL or deepseek-flash)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set; add your DeepSeek API key to .env",
            file=sys.stderr,
        )
        return 2

    configured_workspace = os.getenv("DATA_WORKSPACE_PATH") or str(DEFAULT_WORKSPACE)
    workspace_path = args.workspace or Path(configured_workspace)
    if not workspace_path.is_absolute():
        workspace_path = PROJECT_ROOT / workspace_path
    model_name = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    workspace = LocalWorkspace(workspace_path)
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        use_responses_api=True,
        timeout=120,
        max_retries=2,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=create_read_only_tools(),
        workspace=workspace,
    )
    result = agent.run(args.task, guidelines=args.guidelines)

    print("Trajectory:")
    for index, step in enumerate(result.trajectory, start=1):
        latency = "n/a" if step.latency_ms is None else f"{step.latency_ms:.3f} ms"
        print(f"{index}. {step.name}({step.arguments}) -> {step.status} [{latency}]")
        if step.error:
            print(f"   error: {step.error}")
        if step.token_usage:
            print(f"   token usage: {step.token_usage}")
        print(f"   {step.result[:2_000]}")
    print(f"Final answer: {result.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
