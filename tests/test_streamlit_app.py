from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_debug_ui_renders_without_calling_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_WORKSPACE_PATH", str(tmp_path))

    app = AppTest.from_file(PROJECT_ROOT / "scripts" / "streamlit_app.py").run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Data Management Agent"
    assert app.chat_input[0].placeholder == "Ask a question about the DABstep workspace"
