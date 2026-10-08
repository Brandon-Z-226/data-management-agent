"""Shared helpers for DABstep's supported file formats."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from data_management_agent.tools import ToolExecutionError
from data_management_agent.workspace import WorkspaceEntry

SUPPORTED_EXTENSIONS = frozenset({"csv", "json", "md"})


def extension_for(entry: WorkspaceEntry) -> str:
    return PurePosixPath(str(entry.path)).suffix.lower().lstrip(".")


def ensure_supported(entry: WorkspaceEntry) -> str:
    extension = extension_for(entry)
    if extension not in SUPPORTED_EXTENSIONS:
        raise ToolExecutionError(
            f"unsupported file type for {entry.path}: {extension or 'no extension'}"
        )
    return extension


def decode_text(content: bytes, *, path: str) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(f"{path} is not valid UTF-8 text") from exc


def table_name_for(entry: WorkspaceEntry) -> str:
    stem = PurePosixPath(str(entry.path)).stem.lower()
    name = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "data"
    return f"t_{name}" if name[0].isdigit() else name


def normalize_extensions(file_types: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    aliases = {
        "markdown": "md",
        "text/markdown": "md",
        "text/csv": "csv",
        "application/json": "json",
    }
    extensions: set[str] = set()
    media_types: set[str] = set()
    for file_type in file_types:
        normalized = file_type.lower().strip()
        if "/" in normalized and normalized not in aliases:
            media_types.add(normalized)
        else:
            extensions.add(aliases.get(normalized, normalized.lstrip(".")))
    return frozenset(extensions), frozenset(media_types)
