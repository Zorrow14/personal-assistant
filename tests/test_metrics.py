"""Per-turn metrics: timers, the JSONL file, summary aggregates, and usage from the pipeline."""

import asyncio
import json
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.agent import Agent
from jarvis.core.events import METRICS, EventBus
from jarvis.core.interfaces import LLMResponse, TokenUsage, ToolCall
from jarvis.core.voice_loop import TurnResult, WakeWordLoop
from jarvis.obs.metrics import (
    STAGE_LLM,
    STAGE_STT,
    STAGE_TTS,
    MetricsRecorder,
    current_turn,
    format_summary,
    llm_call,
    percentile,
    read_records,
    summarize,
    summarize_file,
    timer,
    tool_call,
)
from jarvis.tools.base import ToolRegistry
from tests.fakes import (
    ExplodingTool,
    FakeFrameSource,
    FakeLLMClient,
    FakeSTTEngine,
    FakeTTSEngine,
    FakeVAD,
    FakeWakeWordDetector,
    RecordingTool,
)


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_timers_are_no_ops_outside_a_turn() -> None:
    with timer(STAGE_STT) as measured:
        pass
    with llm_call() as call:
        call.usage = TokenUsage(1, 1)
    assert measured.elapsed_ms >= 0
    assert current_turn() is None


def test_a_timed_turn_appends_one_json_line_and_publishes_it(tmp_path: Path) -> None:
    path = tmp_path / "state" / "metrics.jsonl"
    bus = EventBus()
    queue = bus.subscribe()
    recorder = MetricsRecorder(path, bus=bus, model="gemini-test")

    async def turn() -> None:
        with recorder.turn("voice"):
            with timer(STAGE_STT):
                await asyncio.sleep(0.01)
            with llm_call() as call:
                call.usage = TokenUsage(input_tokens=100, output_tokens=20, thinking_tokens=5)
            with llm_call() as call:
                call.usage = TokenUsage(input_tokens=150, output_tokens=10)
            with tool_call() as measured:
                measured.error = True
            async with timer(STAGE_TTS):
                await asyncio.sleep(0.01)

    asyncio.run(turn())

    [record] = _lines(path)
    assert record["source"] == "voice"
    assert record["model"] == "gemini-test"
    assert record["outcome"] == "replied"
    assert (record["llm_requests"], record["input_tokens"], record["output_tokens"]) == (2, 250, 30)
    assert record["thinking_tokens"] == 5
    assert (record["tool_calls"], record["tool_errors"]) == (1, 1)
    stages = record["stages_ms"]
    assert isinstance(stages, dict)
    assert list(stages) == ["stt", "llm_total", "tool_total", "tts", "total"]  # pipeline order
    assert stages["stt"] >= 5 and stages["tts"] >= 5
    assert stages["total"] >= stages["stt"] + stages["tts"]

    event = queue.get_nowait()
    assert event.type == METRICS
    assert event.data["turn"] == record
    assert event.data["session"]["turns"] == 1
    assert event.data["session"]["input_tokens"] == 250


def test_a_failed_turn_is_recorded_and_the_error_still_propagates(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path)

    with pytest.raises(RuntimeError, match="boom"), recorder.turn("text"):
        raise RuntimeError("boom")

    [record] = _lines(path)
    assert record["outcome"] == "failed"
    assert record["error"] == "RuntimeError: boom"
    assert recorder.session["failed"] == 1


def test_disabled_metrics_write_nothing_but_still_measure(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path, enabled=False)
    with recorder.turn("text"):
        pass
    assert not path.exists()
    assert recorder.session["turns"] == 1
    assert recorder.last_turn is not None


def test_an_unwritable_metrics_file_never_breaks_a_turn(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path)  # a directory: appending to it fails
    with recorder.turn("text"):
        result = "the turn itself completes"
    assert result
    assert recorder.session["turns"] == 1


def test_agent_records_llm_requests_tokens_and_tool_time(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("c1", "echo", {"text": "hi"}),
                    ToolCall("c2", "explode", {"text": "x"}),
                ],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=300, output_tokens=12),
            ),
            LLMResponse(text="Done.", usage=TokenUsage(input_tokens=340, output_tokens=4)),
        ]
    )
    registry = ToolRegistry()
    registry.register(RecordingTool())
    registry.register(ExplodingTool())
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")

    async def turn() -> str:
        with recorder.turn("text"):
            return await Agent(llm, registry).run("go")

    assert asyncio.run(turn()) == "Done."
    [record] = _lines(tmp_path / "metrics.jsonl")
    assert (record["llm_requests"], record["input_tokens"], record["output_tokens"]) == (2, 640, 16)
    assert (record["tool_calls"], record["tool_errors"]) == (2, 1)
    stages = record["stages_ms"]
    assert isinstance(stages, dict) and STAGE_LLM in stages and "tool_total" in stages


def test_voice_loop_turn_records_every_stage(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl", model="gemini-test")
    agent = Agent(
        FakeLLMClient(
            [LLMResponse(text="Hi.", usage=TokenUsage(input_tokens=90, output_tokens=2))]
        ),
        ToolRegistry(),
    )
    loop = WakeWordLoop(
        agent,
        FakeSTTEngine(["hello"]),
        FakeTTSEngine(),
        FakeWakeWordDetector([0.9]),
        FakeVAD(),
        FakeFrameSource(),
        display=lambda _line: None,
        metrics=recorder,
    )

    assert asyncio.run(loop.run_once()) is TurnResult.REPLIED

    [record] = _lines(tmp_path / "metrics.jsonl")
    assert record["source"] == "voice"
    stages = record["stages_ms"]
    assert isinstance(stages, dict)
    assert list(stages) == ["wake_to_stt", "stt", "llm_total", "tts", "total"]
    assert record["llm_requests"] == 1 and record["input_tokens"] == 90


def test_empty_voice_turn_is_labelled(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")
    loop = WakeWordLoop(
        Agent(FakeLLMClient([]), ToolRegistry()),
        FakeSTTEngine([""]),
        FakeTTSEngine(),
        FakeWakeWordDetector([0.9]),
        FakeVAD(),
        FakeFrameSource(),
        display=lambda _line: None,
        metrics=recorder,
    )
    asyncio.run(loop.run_once())
    assert _lines(tmp_path / "metrics.jsonl")[0]["outcome"] == "empty"


def _record(llm_ms: float, *, outcome: str = "replied", model: str = "m1") -> dict[str, object]:
    return {
        "ts": "2026-09-26T12:00:00+08:00",
        "model": model,
        "outcome": outcome,
        "stages_ms": {"stt": 400.0, "llm_total": llm_ms, "tts": 1000.0, "total": llm_ms + 1500},
        "llm_requests": 2,
        "input_tokens": 100,
        "output_tokens": 10,
        "thinking_tokens": 0,
        "tool_calls": 1,
        "tool_errors": 0,
    }


def test_summary_aggregates_percentiles_and_totals() -> None:
    records = [_record(ms) for ms in range(100, 1001, 100)]  # llm_total 100..1000 ms
    records[-1]["outcome"] = "failed"

    summary = summarize(records)

    assert summary["turns"] == 10
    assert summary["outcomes"] == {"replied": 9, "failed": 1}
    assert summary["llm_requests"] == 20 and summary["llm_requests_per_turn"] == 2.0
    assert (summary["input_tokens"], summary["output_tokens"]) == (1000, 100)
    llm = summary["stages_ms"]["llm_total"]
    assert (llm["count"], llm["p50"], llm["p95"], llm["max"]) == (10, 550.0, 955.0, 1000.0)
    assert summary["stages_ms"]["stt"]["p50"] == 400.0
    assert list(summary["stages_ms"]) == ["stt", "llm_total", "tts", "total"]
    # stt 4000 + llm 5500 + tts 10000 => tts dominates Jarvis's own time here
    assert summary["dominant_stage"] == {"stage": "tts", "share": round(10000 / 19500, 3)}


def test_summary_of_a_file_skips_bad_lines_and_handles_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    assert summarize_file(path)["turns"] == 0
    path.write_text(
        json.dumps(_record(200)) + "\nnot json\n[1, 2]\n\n" + json.dumps(_record(400)) + "\n",
        encoding="utf-8",
    )

    summary = summarize_file(path)
    assert summary["turns"] == 2
    assert summary["skipped_lines"] == 2
    assert summary["path"] == str(path)
    records, _ = read_records(path, last=1)
    assert [r["stages_ms"]["llm_total"] for r in records] == [400]  # type: ignore[index]


def test_percentile_interpolates() -> None:
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_format_summary_reads_well() -> None:
    text = format_summary(summarize([_record(8000), _record(9000)]) | {"path": "m.jsonl"})
    assert "2 turns (2 replied)" in text
    assert "llm_total" in text and "8.50" in text  # p50 in seconds
    assert "dominated by llm_total" in text
    assert "No turns recorded yet" in format_summary(summarize([]) | {"path": "m.jsonl"})


def test_metrics_cli_prints_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jarvis import cli

    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(_record(1200)) + "\n", encoding="utf-8")
    settings = Settings(vault_path=tmp_path, metrics_path=path, _env_file=None)  # type: ignore[call-arg]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    assert cli.main(["--metrics"]) == 0
    out = capsys.readouterr().out
    assert "1 turns (1 replied)" in out and "llm_total" in out
