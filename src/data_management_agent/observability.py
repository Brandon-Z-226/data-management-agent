"""Local JSONL event journal. Identifiers correlate runs; they carry no memory."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4


def _id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class RunContext:
    source: str = "api"
    run_id: str = field(default_factory=_id)
    execution_id: str = field(default_factory=_id)
    conversation_id: str = field(default_factory=_id)
    thread_id: str = field(default_factory=_id)
    task_id: str = field(default_factory=_id)
    turn_id: str = field(default_factory=_id)
    turn_index: int = 0


_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)
_active: ContextVar[RunLog | None] = ContextVar("run_log", default=None)
_write_lock = threading.Lock()


@contextmanager
def run_scope(context: RunContext):
    token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(token)


def emit(event: str, **data: Any) -> None:
    log = _active.get()
    if log is not None:
        log.emit(event, **data)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capture_version(root: Path) -> dict[str, Any]:
    """Fingerprint executable source and dependency lock, including uncommitted edits."""
    hashes = {}
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for directory in ("src", "scripts"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    for path in paths:
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def git(*args: str) -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=5
                )
                .decode()
                .strip()
            )
        except (OSError, subprocess.SubprocessError):
            return None

    dependencies = {}
    for name in ("langgraph", "langchain-openai", "langchain-core", "duckdb", "pydantic"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": (
            bool(status)
            if (status := git("status", "--porcelain", "--untracked-files=normal")) is not None
            else None
        ),
        "source_sha256": _digest(hashes),
        "file_hashes": hashes,
        "python": sys.version.split()[0],
        "dependencies": dependencies,
    }


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if str(key).casefold() in {"api_key", "authorization", "password", "access_token"}
            else _redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
    return value


class RunJournal:
    def __init__(self, project_root: Path, directory: Path | None = None) -> None:
        self.project_root = project_root
        self.directory = directory or project_root / "logs"

    @contextmanager
    def start(self, config: dict[str, Any], *, task: str, guidelines: str | None):
        context = _context.get() or RunContext()
        log = RunLog(self, context, config)
        token = _active.set(log)
        try:
            log.emit("run_started", task=task, guidelines=guidelines, config=config)
            yield log
        finally:
            _active.reset(token)


class RunLog:
    def __init__(self, journal: RunJournal, context: RunContext, config: dict[str, Any]) -> None:
        self.context = context
        self.sequence = 0
        self.started = time.perf_counter()
        # Never serialize environment or the model client (which contains credentials).
        self.secrets = tuple(
            value
            for key, value in os.environ.items()
            if value and (key.endswith("API_KEY") or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})
        )
        self.directory = journal.directory
        self.path = self.directory / "runs" / f"{context.execution_id}.jsonl"
        self.failed = False
        self.llm_calls = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.usage: dict[str, int] = {}
        self.usage_reports = 0
        snapshot = {**capture_version(journal.project_root), "config": config}
        self.version_id = _digest(snapshot)
        snapshot["version_id"] = self.version_id
        self.version = snapshot
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            versions = self.directory / "versions"
            versions.mkdir(parents=True, exist_ok=True)
            with _write_lock:
                version_file = versions / f"{self.version_id}.json"
                if not version_file.exists():
                    with version_file.open("x", encoding="utf-8") as output:
                        json.dump(
                            _redact(snapshot, self.secrets), output, ensure_ascii=False, indent=2
                        )
        except OSError as exc:
            self._warn(exc)

    def _warn(self, exc: OSError) -> None:
        self.failed = True
        warnings.warn(
            f"Run journal unavailable ({type(exc).__name__}); run continues", stacklevel=2
        )

    def emit(self, event: str, **data: Any) -> None:
        if self.failed:
            return
        if event == "llm_started":
            self.llm_calls += 1
        if event == "llm_finished" and (usage := data.get("token_usage")):
            self.usage_reports += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    self.usage[key] = self.usage.get(key, 0) + usage[key]
        if event == "tool_started":
            self.tool_calls += 1
        if event == "tool_finished" and data["message"].get("status") == "error":
            self.tool_errors += 1
        if event == "run_finished":
            data.update(
                latency_ms=(time.perf_counter() - self.started) * 1000,
                llm_calls=self.llm_calls,
                tool_calls=self.tool_calls,
                tool_errors=self.tool_errors,
                token_usage=self.usage or None,
                usage_reports=self.usage_reports,
            )
        record = {
            "schema_version": "agent-events-v1",
            **asdict(self.context),
            "version_id": self.version_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_ms": (time.perf_counter() - self.started) * 1000,
            "event": event,
            "data": data,
        }
        self.sequence += 1
        try:
            line = json.dumps(_redact(record, self.secrets), ensure_ascii=False, default=str) + "\n"
            with _write_lock:
                # Close/flush every event: partial traces survive a later graph/model failure.
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(line)
                if event in {"run_started", "run_finished"}:
                    index = {key: value for key, value in record.items() if key != "data"}
                    index["log_file"] = str(self.path)
                    index["status"] = data.get("status", "running")
                    with (self.directory / "index.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(index, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._warn(exc)
