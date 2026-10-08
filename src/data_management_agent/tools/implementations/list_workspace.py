"""Implementation of ``list_workspace``."""

from __future__ import annotations

from data_management_agent.tools import Tool, ToolContext
from data_management_agent.tools.contracts import LIST_WORKSPACE
from data_management_agent.tools.schemas import ListWorkspaceInput, ListWorkspaceOutput
from data_management_agent.workspace import WorkspaceQuery

from ._files import normalize_extensions


class ListWorkspaceTool(Tool[ListWorkspaceInput, ListWorkspaceOutput]):
    contract = LIST_WORKSPACE

    async def execute(
        self,
        arguments: ListWorkspaceInput,
        *,
        context: ToolContext,
    ) -> ListWorkspaceOutput:
        extensions, media_types = normalize_extensions(arguments.file_types)
        query = WorkspaceQuery(
            path=arguments.path,
            recursive=arguments.recursive,
            extensions=extensions,
            media_types=media_types,
            modified_after=arguments.modified_after,
            modified_before=arguments.modified_before,
            metadata=arguments.metadata,
        )
        page = await context.workspace.list_entries(
            query,
            cursor=arguments.cursor,
            limit=arguments.limit,
        )
        return ListWorkspaceOutput(entries=page.entries, next_cursor=page.next_cursor)
