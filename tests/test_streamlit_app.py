from pathlib import Path

from streamlit.testing.v1 import AppTest

from data_management_agent.benchmark import BenchmarkTaskResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_debug_ui_renders_without_calling_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("DABSTEP_BENCHMARK_RESULTS_PATH", str(tmp_path / "results"))

    app = AppTest.from_file(PROJECT_ROOT / "scripts" / "streamlit_app.py").run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Data Management Agent"
    assert app.chat_input[0].placeholder == "Ask a question about the DABstep workspace"


def test_streamlit_benchmark_page_handles_missing_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("DABSTEP_BENCHMARK_RESULTS_PATH", str(tmp_path / "results"))
    app = AppTest.from_file(PROJECT_ROOT / "scripts" / "streamlit_app.py").run(timeout=10)

    app.radio[0].set_value("Benchmark Results").run(timeout=30)

    assert not app.exception
    assert app.header[0].value == "Benchmark Results"
    assert "No benchmark result files found" in app.info[0].value


def test_streamlit_benchmark_page_loads_structured_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    result = BenchmarkTaskResult.model_validate(
        {
            "run_id": "run-test",
            "conversation_id": "conversation-test",
            "thread_id": "thread-test",
            "turn_id": "turn-test",
            "task_id": "5",
            "model": "test-model",
            "level": "easy",
            "question": "Which country?",
            "guidelines": "Country code only",
            "reference_answer": "NL",
            "final_answer": "NL",
            "status": "succeeded",
            "correctness": {
                "score": 1.0,
                "is_correct": True,
                "exact_match": True,
                "normalized_match": True,
            },
            "latency_ms": 100.0,
            "started_at": "2026-10-08T00:00:00Z",
            "completed_at": "2026-10-08T00:00:00Z",
        }
    )
    (results_dir / "run.jsonl").write_text(result.model_dump_json() + "\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_WORKSPACE_PATH", str(tmp_path))
    monkeypatch.setenv("DABSTEP_BENCHMARK_RESULTS_PATH", str(results_dir))
    app = AppTest.from_file(PROJECT_ROOT / "scripts" / "streamlit_app.py").run(timeout=10)

    app.radio[0].set_value("Benchmark Results").run(timeout=30)

    assert not app.exception
    assert [metric.value for metric in app.metric] == [
        "1",
        "100.0%",
        "1",
        "1",
        "100.0%",
        "100.0%",
    ]
    assert app.selectbox[1].value == "5"
