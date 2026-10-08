"""Local-only Streamlit chat UI for the workspace agent."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_management_agent.agent import ToolTrace, WorkspaceAgent
from data_management_agent.tools.implementations import create_read_only_tools
from data_management_agent.workspace import LocalWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "external" / "dabstep"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def _workspace_path() -> Path:
    configured = os.getenv("DATA_WORKSPACE_PATH")
    path = Path(configured) if configured else DEFAULT_WORKSPACE
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_resource(show_spinner=False)
def _create_agent(
    workspace_path: str,
    model_name: str,
    base_url: str,
    api_key: str,
) -> WorkspaceAgent:
    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        use_responses_api=True,
        timeout=120,
        max_retries=2,
    )
    return WorkspaceAgent(
        model=model,
        tools=create_read_only_tools(),
        workspace=LocalWorkspace(workspace_path),
    )


def _render_result(result: str) -> None:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        st.code(result or "(empty)", language=None)
    else:
        st.json(parsed)


def _render_trajectory(trajectory: list[dict[str, Any]]) -> None:
    if not trajectory:
        st.caption("No tool calls were made.")
        return

    for index, trace in enumerate(trajectory, start=1):
        latency = trace.get("latency_ms")
        latency_label = "latency unavailable" if latency is None else f"{latency:.3f} ms"
        label = f"{index}. {trace['name']} · {trace['status']} · {latency_label}"
        with st.expander(label):
            st.caption(f"Call ID: {trace['call_id']}")
            st.markdown("**Arguments**")
            st.json(trace["arguments"])

            st.markdown("**Token usage for the model turn that requested this tool**")
            if trace.get("token_usage"):
                st.json(trace["token_usage"])
            else:
                st.caption("Unavailable from the model response.")

            st.markdown("**Error**")
            if trace.get("error"):
                st.error(trace["error"])
            else:
                st.caption("None")

            st.markdown("**Result**")
            _render_result(trace["result"])


def _stored_trace(trace: ToolTrace) -> dict[str, Any]:
    return asdict(trace)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(page_title="Data Management Agent Debug UI", layout="centered")
    st.title("Data Management Agent")
    st.caption("Local development UI for the existing WorkspaceAgent API")

    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    workspace_path = _workspace_path()

    with st.sidebar:
        st.subheader("Runtime")
        st.text(f"Model: {model_name}")
        st.text(f"Workspace: {workspace_path}")
        guidelines = st.text_area("Answer guidelines", placeholder="Optional")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add your DeepSeek API key to .env.")
        st.stop()
    if not workspace_path.is_dir():
        st.error(f"Workspace directory does not exist: {workspace_path}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(f"Agent latency: {message['latency_ms']:.3f} ms")
                _render_trajectory(message["trajectory"])

    prompt = st.chat_input("Ask a question about the DABstep workspace")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = _create_agent(str(workspace_path), model_name, base_url, api_key)
    with st.chat_message("assistant"):
        try:
            started_at = time.perf_counter()
            with st.spinner("Running agent..."):
                result = agent.run(prompt, guidelines=guidelines or None)
            latency_ms = (time.perf_counter() - started_at) * 1_000
            trajectory = [_stored_trace(trace) for trace in result.trajectory]
            st.markdown(result.answer)
            st.caption(f"Agent latency: {latency_ms:.3f} ms")
            _render_trajectory(trajectory)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "latency_ms": latency_ms,
                    "trajectory": trajectory,
                }
            )
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")


if __name__ == "__main__":
    main()
