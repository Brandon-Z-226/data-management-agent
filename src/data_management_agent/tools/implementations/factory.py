"""Construct the read-only v0 tool set with shared dependencies."""

from __future__ import annotations

from data_management_agent.tools import Tool

from .evidence import EvidenceStore
from .inspect_file import InspectFileTool
from .list_workspace import ListWorkspaceTool
from .query_data import QueryDataTool
from .read_file import ReadFileTool
from .search_workspace import SearchWorkspaceTool


def create_read_only_tools() -> tuple[Tool, ...]:
    evidence_store = EvidenceStore()
    return (
        ListWorkspaceTool(),
        InspectFileTool(),
        ReadFileTool(evidence_store),
        SearchWorkspaceTool(evidence_store),
        QueryDataTool(),
    )
