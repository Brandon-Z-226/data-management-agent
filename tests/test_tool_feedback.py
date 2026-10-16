from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _convert_message_to_dict
from pydantic import ValidationError

from data_management_agent.agent import WorkspaceAgent
from data_management_agent.agent.graph import _model_tool_definition
from data_management_agent.observability import RunContext, RunJournal, run_scope
from data_management_agent.tools import ToolContext, ToolExecutionError
from data_management_agent.tools.base import ToolOutputError
from data_management_agent.tools.contracts import READ_FILE
from data_management_agent.tools.feedback import error_feedback
from data_management_agent.tools.implementations import (
    ReadFileTool,
    create_read_only_tools,
)
from data_management_agent.tools.schemas import ReadFileInput
from data_management_agent.workspace import LocalWorkspace


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        # These are the actual definitions passed to the model adapter, not just docs.
        for definition in tools:
            assert definition["function"]["parameters"]["examples"]
            assert "Example arguments" in definition["function"]["description"]
        return self


def call(name, args, call_id="call-error"):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def run_error(tmp_path, name, args):
    return (
        WorkspaceAgent(
            model=ScriptedModel(
                responses=[call(name, args), AIMessage(content="Cannot answer yet.")]
            ),
            tools=create_read_only_tools(),
            workspace=LocalWorkspace(tmp_path),
        )
        .run("Exercise feedback")
        .messages[2]
    )


def test_all_active_examples_validate_and_are_visible():
    for tool in create_read_only_tools():
        definition = tool.definition
        assert definition.input_schema["examples"]
        for example in definition.input_schema["examples"]:
            tool.contract.input_type.model_validate(example)
        assert json.dumps(tool.contract.examples, separators=(",", ":")) in definition.description
    props = ReadFileInput.model_json_schema()["properties"]
    assert "top-level" in props["locator"]["description"].lower()
    assert "Do not put" in props["file"]["description"]


def test_actual_adapter_binding_preserves_descriptions_and_examples():
    # Construction/binding only: no network request and no user credentials loaded.
    model = ChatOpenAI(model="offline-test", api_key="not-a-real-key", use_responses_api=True)
    definitions = [_model_tool_definition(tool) for tool in create_read_only_tools()]
    bound = model.bind_tools(definitions)
    assert bound.kwargs["tools"] == definitions


def test_error_then_corrected_read_preserves_raw_log_but_not_model_payload(tmp_path):
    (tmp_path / "notes.md").write_text("read me")
    wrong = {"file": {"path": "notes.md", "max_chars": 10, "locator": {"char_start": 0}}}
    corrected = {"file": {"path": "notes.md"}, "max_chars": 10, "locator": {"char_start": 0}}
    agent = WorkspaceAgent(
        model=ScriptedModel(
            responses=[
                call("read_file", wrong),
                call("read_file", corrected, "corrected"),
                AIMessage(content="read me"),
            ]
        ),
        tools=create_read_only_tools(),
        workspace=LocalWorkspace(tmp_path),
        journal=RunJournal(tmp_path),
    )
    context = RunContext(source="offline_validation")
    with run_scope(context):
        result = agent.run("Read notes")
    assert [trace.status for trace in result.trajectory] == ["error", "success"]
    assert result.trajectory[0].arguments == wrong  # No hidden argument rewriting.
    error = next(message for message in result.messages if isinstance(message, ToolMessage))
    payload = json.loads(error.content)
    assert payload["code"] == "invalid_arguments"
    assert {issue["field"] for issue in payload["fields"]} == {"file.max_chars", "file.locator"}
    assert "top level" in payload["suggested_action"]
    ReadFileInput.model_validate(payload["example_arguments"])
    assert payload["retry_same_arguments"] is False
    assert "ValidationError" in error.artifact["diagnostics"]
    wire_message = _convert_message_to_dict(error)
    assert "artifact" not in wire_message
    assert "Traceback" not in json.dumps(wire_message)
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "runs" / f"{context.execution_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    logged = next(e["data"]["message"] for e in events if e["event"] == "tool_finished")
    assert logged["artifact"]["error_code"] == "invalid_arguments"
    assert "ValidationError" in logged["artifact"]["diagnostics"]


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("list_workspace", {"path": "/"}, "invalid_arguments"),
        ("list_workspace", {"path": 7}, "invalid_arguments"),
        ("read_file", {"file": {"path": "missing.md"}}, "resource_not_found"),
        ("invented", {}, "unknown_tool"),
        (
            "read_file",
            {"file": {"path": "notes.md"}, "locator": {"page": 1}},
            "unsupported_operation",
        ),
        (
            "query_data",
            {"sources": [{"path": "records.csv"}], "query": "SELECT wrong FROM records"},
            "sql_error",
        ),
        (
            "query_data",
            {"sources": [{"path": "records.csv"}], "query": "DELETE FROM records"},
            "unsupported_operation",
        ),
        (
            "query_data",
            {"sources": [{"path": "bad.json"}], "query": "SELECT * FROM bad"},
            "data_error",
        ),
        ("search_workspace", {"query": "!!!"}, "invalid_arguments"),
    ],
)
def test_error_categories(tmp_path, name, args, code):
    (tmp_path / "notes.md").write_text("notes")
    (tmp_path / "records.csv").write_text("n\n1\n")
    (tmp_path / "bad.json").write_text("{invalid")
    message = run_error(tmp_path, name, args)
    assert isinstance(message, ToolMessage) and message.status == "error"
    payload = json.loads(message.content)
    assert payload["code"] == code
    assert payload["error"] == payload["message"]  # Existing consumers retain a string error.
    assert payload["suggested_action"]
    assert len(message.content) < 2500


def test_long_validation_and_sql_errors_are_bounded():
    with pytest.raises(ValidationError) as caught:
        ReadFileInput.model_validate({"file": {"path": "notes.md", "unexpected": "x" * 10000}})
    payload = error_feedback(caught.value, READ_FILE).payload()
    assert "x" * 100 not in json.dumps(payload)
    exc = ToolExecutionError(
        "Binder Error: bad types\nCandidate functions:\n" + "candidate\n" * 1000, code="sql_error"
    )
    feedback = error_feedback(exc, READ_FILE)
    assert feedback.message == "Binder Error: bad types"
    assert len(json.dumps(feedback.payload())) < 1000


@pytest.mark.parametrize("failure", [RuntimeError("internal implementation bug"), "bad_output"])
def test_internal_failures_are_not_blame_on_arguments(tmp_path, monkeypatch, failure):
    async def broken(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return {"not_a_read_file_output": True}

    monkeypatch.setattr(ReadFileTool, "execute", broken)
    message = run_error(tmp_path, "read_file", {"file": {"path": "anything.md"}})
    payload = json.loads(message.content)
    assert payload["code"] == "internal_error"
    assert "Do not assume the arguments are wrong" in payload["suggested_action"]
    assert "Traceback" in message.artifact["diagnostics"]
    assert "internal implementation bug" not in message.content


def test_direct_input_validation_remains_validation_error(tmp_path):
    with pytest.raises(ValidationError):
        asyncio.run(
            ReadFileTool().invoke(
                {"file": {"path": "x", "max_chars": 5}},
                context=ToolContext(workspace=LocalWorkspace(tmp_path)),
            )
        )


def test_internal_validation_is_wrapped_not_input_validation(tmp_path, monkeypatch):
    async def broken(*args, **kwargs):
        ReadFileInput.model_validate({})

    monkeypatch.setattr(ReadFileTool, "execute", broken)
    with pytest.raises(ToolOutputError):
        asyncio.run(
            ReadFileTool().invoke(
                {"file": {"path": "x"}}, context=ToolContext(workspace=LocalWorkspace(tmp_path))
            )
        )
