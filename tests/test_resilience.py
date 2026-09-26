"""Resilience: one bad turn never takes the loop down, and Jarvis apologises aloud."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from jarvis.core import voice_loop as voice_loop_module
from jarvis.core.agent import DECLINED_RESULT, Agent
from jarvis.core.events import ERROR, REPLY, STATE, TOOL, Event, EventBus
from jarvis.core.interfaces import (
    AudioSamples,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    VoiceError,
)
from jarvis.core.voice_loop import (
    APOLOGY,
    RATE_LIMITED_APOLOGY,
    UNAVAILABLE_APOLOGY,
    TurnResult,
    WakeWordLoop,
)
from jarvis.obs.metrics import MetricsRecorder
from jarvis.tools.base import ToolRegistry
from tests.fakes import (
    DangerousTool,
    ExplodingTool,
    FakeFrameSource,
    FakeLLMClient,
    FakeSTTEngine,
    FakeTTSEngine,
    FakeVAD,
    FakeWakeWordDetector,
)


def _drain(queue: asyncio.Queue[Event]) -> list[Event]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


class BrokenSTT(FakeSTTEngine):
    """Fails `failures` times, then transcribes normally."""

    def __init__(self, transcripts: Sequence[str], failures: int = 1) -> None:
        super().__init__(transcripts)
        self.failures = failures

    def transcribe(self, audio: AudioSamples) -> str:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("whisper crashed")
        return super().transcribe(audio)


class DownLLM(FakeLLMClient):
    def __init__(self, status_code: int | None) -> None:
        super().__init__([])
        self.status_code = status_code

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMError(f"({self.status_code}) nope", status_code=self.status_code)


def _loop(
    *,
    llm: FakeLLMClient,
    stt: FakeSTTEngine,
    registry: ToolRegistry | None = None,
    scores: Sequence[float] = (0.9,),
    tts: FakeTTSEngine | None = None,
    bus: EventBus | None = None,
    metrics: MetricsRecorder | None = None,
) -> WakeWordLoop:
    return WakeWordLoop(
        Agent(llm, registry or ToolRegistry(), events=bus),
        stt,
        tts or FakeTTSEngine(),
        FakeWakeWordDetector(list(scores)),
        FakeVAD(),
        FakeFrameSource(),
        display=lambda _line: None,
        events=bus,
        metrics=metrics,
    )


def test_a_tool_exception_goes_back_to_the_model_and_the_turn_completes() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    registry = ToolRegistry()
    registry.register(ExplodingTool())
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("x1", "explode", {"text": "go"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Sorry, that tool failed."),
        ]
    )
    loop = _loop(llm=llm, stt=FakeSTTEngine(["blow it up"]), registry=registry, bus=bus)

    assert asyncio.run(loop.run_once()) is TurnResult.REPLIED

    events = _drain(queue)
    tool = [e.data for e in events if e.type == TOOL]
    assert [t["status"] for t in tool] == ["started", "error"]  # shown as failed in the panel
    assert "RuntimeError: kaboom" in tool[-1]["result"]
    result_msg = llm.requests[1][0][-1]
    assert result_msg.is_error and "kaboom" in result_msg.content  # the model was told
    assert [e.data["text"] for e in events if e.type == REPLY] == ["Sorry, that tool failed."]
    assert loop.state is not None and loop.state.value == "idle"


def test_a_forced_exception_is_caught_apologised_for_and_back_to_idle(tmp_path: Path) -> None:
    bus = EventBus()
    queue = bus.subscribe()
    tts = FakeTTSEngine()
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")
    loop = _loop(llm=FakeLLMClient([]), stt=BrokenSTT([]), tts=tts, bus=bus, metrics=recorder)

    result = asyncio.run(loop.run_once())  # must not raise

    assert result is TurnResult.FAILED
    events = _drain(queue)
    assert [e.data["message"] for e in events if e.type == ERROR] == [
        "RuntimeError: whisper crashed"
    ]
    assert tts.spoken == [APOLOGY]
    states = [e.data["state"] for e in events if e.type == STATE]
    assert states[-2:] == ["speaking", "idle"]
    [record] = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert record["outcome"] == "failed" and "whisper crashed" in record["error"]


@pytest.mark.parametrize(
    ("status", "apology"),
    [(429, RATE_LIMITED_APOLOGY), (503, UNAVAILABLE_APOLOGY), (400, APOLOGY), (None, APOLOGY)],
)
def test_llm_failures_get_a_fitting_apology(status: int | None, apology: str) -> None:
    tts = FakeTTSEngine()
    loop = _loop(llm=DownLLM(status), stt=FakeSTTEngine(["hello"]), tts=tts)

    assert asyncio.run(loop.run_once()) is TurnResult.FAILED
    assert tts.spoken == [apology]


def test_the_loop_keeps_listening_after_a_failed_turn() -> None:
    tts = FakeTTSEngine()
    loop = _loop(
        llm=FakeLLMClient([LLMResponse(text="Hi there.")]),
        stt=BrokenSTT(["hello", "exit"], failures=1),
        scores=[0.9, 0.9, 0.9],
        tts=tts,
    )

    asyncio.run(loop.run())  # ends on the spoken "exit", not on the error

    assert tts.spoken == [APOLOGY, "Hi there."]


def test_a_mic_failure_while_waiting_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyMic(FakeFrameSource):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def read(self, n_samples: int) -> AudioSamples:
            if not self.failed:
                self.failed = True
                raise VoiceError("no audio from the microphone for 3s")
            return super().read(n_samples)

    monkeypatch.setattr(voice_loop_module, "RETRY_DELAY_SECONDS", 0.0)
    bus = EventBus()
    queue = bus.subscribe()
    loop = WakeWordLoop(
        Agent(FakeLLMClient([]), ToolRegistry()),
        FakeSTTEngine(["exit"]),
        FakeTTSEngine(),
        FakeWakeWordDetector([0.9]),
        FakeVAD(),
        FlakyMic(),
        display=lambda _line: None,
        events=bus,
    )

    asyncio.run(loop.run())

    errors = [e.data["message"] for e in _drain(queue) if e.type == ERROR]
    assert errors == ["Voice error: no audio from the microphone for 3s"]


def test_a_broken_speaker_does_not_break_the_apology_path() -> None:
    class DeadTTS(FakeTTSEngine):
        def speak(self, text: str) -> None:
            raise RuntimeError("audio device gone")

    loop = _loop(llm=FakeLLMClient([]), stt=BrokenSTT([]), tts=DeadTTS())
    assert asyncio.run(loop.run_once()) is TurnResult.FAILED


def test_a_crashing_confirmation_prompt_declines_the_action() -> None:
    def broken_prompt(call: ToolCall) -> bool:
        raise OSError("stdin is closed")

    registry = ToolRegistry()
    dangerous = DangerousTool()
    registry.register(dangerous)
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("d1", "delete_everything", {"text": "all"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="I didn't do it."),
        ]
    )

    reply = asyncio.run(Agent(llm, registry, confirm=broken_prompt).run("delete everything"))

    assert reply == "I didn't do it."
    assert dangerous.runs == []  # fail closed: the action never ran
    assert llm.requests[1][0][-1].content == DECLINED_RESULT
