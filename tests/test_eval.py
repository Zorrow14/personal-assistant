"""The eval harness: --fake runs offline and scores; failures are detected, the real vault untouched."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

EVAL_RUN = Path(__file__).resolve().parents[1] / "eval" / "run.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("jarvis_eval_run", EVAL_RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def keep_global_logging(harness: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    # main() configures structlog to print to the current sys.stderr, which under
    # capsys is a capture stream closed after the test: later tests would crash logging.
    monkeypatch.setattr(harness, "configure_logging", lambda level: None)


def test_fake_mode_runs_every_case_and_reports_a_score(
    harness: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = harness.load_cases(harness.DEFAULT_CASES)

    assert harness.main(["--fake"]) == 0

    out = capsys.readouterr().out
    assert f"Score: {len(cases)}/{len(cases)} passed (100%)" in out
    assert "write_task_note" in out and "get_datetime" in out


def test_failures_are_detected_and_explained(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Case, Result = harness.Case, harness.Result
    wrong_tool = Result(
        case=Case("log", "log it", "write_task_note"),
        calls=[("get_datetime", "completed")],
        reply="It's noon.",
        error=None,
        seconds=1.0,
        llm_requests=2,
        input_tokens=10,
        output_tokens=2,
    )
    tool_errored = Result(
        case=Case("remind", "remind me", "set_reminder"),
        calls=[("set_reminder", "error")],
        reply="Couldn't.",
        error=None,
        seconds=1.0,
        llm_requests=2,
        input_tokens=10,
        output_tokens=2,
    )
    unneeded_tool = Result(
        case=Case("hi", "say hi", None, expect_contains="hi"),
        calls=[("write_task_note", "completed")],
        reply="Hi!",
        error=None,
        seconds=1.0,
        llm_requests=2,
        input_tokens=10,
        output_tokens=2,
    )
    assert not wrong_tool.passed and not tool_errored.passed and not unneeded_tool.passed

    text = harness.report([wrong_tool, tool_errored, unneeded_tool], "model-x")
    assert "Score: 0/3 passed (0%)" in text
    assert "expected write_task_note, called get_datetime" in text
    assert "set_reminder (error)" in text

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps(
            [
                {
                    "input": "Log that the eval can fail.",
                    "expect_tool": "write_task_note",
                    "expect_contains": "never in the reply",
                    "fake_args": {"command": "x", "body": "y"},
                    "fake_reply": "Logged.",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert harness.main(["--fake", "--cases", str(cases_file)]) == 1  # below --min-score
    assert "reply lacks 'never in the reply'" in capsys.readouterr().out


def test_eval_never_touches_the_real_vault(harness: ModuleType, tmp_path: Path) -> None:
    settings, registry = harness.build_environment(tmp_path, fake=True)
    assert settings.vault_path == tmp_path / "vault"
    assert settings.reminders_path.is_relative_to(tmp_path)
    assert settings.file_sandbox_root == tmp_path / "vault"
    assert "web_search" in registry  # present, but canned in --fake mode
