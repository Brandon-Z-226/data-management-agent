"""Per-invocation budget/history helpers; no persisted memory or task-specific policy."""

import json
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    tool_rounds: int
    tool_attempts: int
    stop_reason: str | None
    error_type: str | None
    error_message: str | None
    cause_type: str | None
    finalization_used: bool


def signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps([name, arguments], sort_keys=True, separators=(",", ":"), default=str)


def call_history(messages: list[AnyMessage]):
    """Count failures since the same call last succeeded; key order is irrelevant."""
    pending = {}
    failures: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    internal_failures: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                pending[call.get("id")] = signature(call["name"], call.get("args", {}))
        elif isinstance(message, ToolMessage) and message.tool_call_id in pending:
            artifact = message.artifact if isinstance(message.artifact, dict) else {}
            if not artifact.get("attempted", True):
                continue
            key = pending[message.tool_call_id]
            if message.status == "error":
                failures[key] += 1
                if artifact.get("error_code") == "internal_error":
                    internal_failures.add(key)
            else:
                failures[key] = 0
                internal_failures.discard(key)
                successes[key] += 1
    return failures, successes, internal_failures
