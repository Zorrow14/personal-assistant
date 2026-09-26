"""Events published by the agent, the wake-word loop and the audio hooks. Fakes only."""

import asyncio
import wave
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio import io as audio_io
from jarvis.config import Settings
from jarvis.core.agent import Agent
from jarvis.core.events import (
    ERROR,
    LEVEL,
    LEVEL_SPEAKER,
    REPLY,
    STATE,
    TOOL,
    TRANSCRIPT,
    Event,
    EventBus,
    LevelMeter,
)
from jarvis.core.interfaces import LLMError, LLMResponse, Message, ToolCall, ToolSpec
from jarvis.core.voice_loop import TurnResult, WakeWordLoop
from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool
from jarvis.tts.factory import create_tts_engine
from tests.fakes import (
    DangerousTool,
    ExplodingTool,
    FakeFrameSource,
    FakeLLMClient,
    FakeSTTEngine,
    FakeTTSEngine,
    FakeVAD,
    FakeWakeWordDetector,
    RecordingTool,
)


def _drain(queue: asyncio.Queue[Event]) -> list[Event]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _summary(events: Sequence[Event]) -> list[tuple[str, Any]]:
    """(type, the one field that matters) for compact sequence assertions."""
    key = {STATE: "state", TRANSCRIPT: "text", REPLY: "text", TOOL: "status", ERROR: "message"}
    return [(e.type, e.data.get(key.get(e.type, ""))) for e in events]


def _call(name: str, call_id: str = "c1", **args: object) -> LLMResponse:
    return LLMResponse(
        text=None, tool_calls=[ToolCall(call_id, name, args)], stop_reason="tool_use"
    )


def _loop(
    bus: EventBus | None,
    *,
    transcripts: list[str],
    responses: list[LLMResponse],
    registry: ToolRegistry | None = None,
    shown: list[str] | None = None,
) -> WakeWordLoop:
    agent = Agent(FakeLLMClient(responses), registry or ToolRegistry(), events=bus)
    return WakeWordLoop(
        agent,
        FakeSTTEngine(transcripts),
        FakeTTSEngine(),
        FakeWakeWordDetector([0.9]),
        FakeVAD(),
        FakeFrameSource(),
        display=(shown if shown is not None else []).append,
        events=bus,
    )


def test_wake_loop_publishes_state_transcript_tool_reply_sequence(tmp_path: Path) -> None:
    vault = Vault(tmp_path, clock=lambda: datetime(2026, 9, 26, 12, 0))
    registry = ToolRegistry()
    registry.register(WriteTaskNoteTool(vault))
    bus = EventBus()
    queue = bus.subscribe()
    loop = _loop(
        bus,
        transcripts=["Log that the panel works."],
        responses=[
            _call("write_task_note", command="Panel works", body="Done."),
            LLMResponse(text="Logged it."),
        ],
        registry=registry,
    )

    assert asyncio.run(loop.run_once()) is TurnResult.REPLIED

    events = _drain(queue)
    assert _summary(events) == [
        (STATE, "idle"),
        (STATE, "listening"),
        (STATE, "thinking"),
        (TRANSCRIPT, "Log that the panel works."),
        (TOOL, "started"),
        (TOOL, "completed"),
        (REPLY, "Logged it."),
        (STATE, "speaking"),
        (STATE, "idle"),
    ]
    started = events[4].data
    assert started["name"] == "write_task_note"
    assert started["category"] == "vault"
    assert started["args"] == {"command": "Panel works", "body": "Done."}
    assert events[3].data["source"] == "voice"
    assert events[5].data["result"].endswith("2026-09-26-1200-panel-works.md")


def test_empty_transcript_is_published_then_back_to_idle() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    loop = _loop(bus, transcripts=[""], responses=[])

    assert asyncio.run(loop.run_once()) is TurnResult.EMPTY

    assert _summary(_drain(queue))[-2:] == [(TRANSCRIPT, ""), (STATE, "idle")]


def test_loop_publishes_llm_errors() -> None:
    class DownLLM(FakeLLMClient):
        async def complete(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
        ) -> LLMResponse:
            raise LLMError("429 RESOURCE_EXHAUSTED")

    bus = EventBus()
    queue = bus.subscribe()
    loop = _loop(bus, transcripts=["hello", "exit"], responses=[])
    loop.detector = FakeWakeWordDetector([0.9, 0.9])
    loop.agent.llm = DownLLM([])

    asyncio.run(loop.run())

    errors = [e.data["message"] for e in _drain(queue) if e.type == ERROR]
    assert errors == ["LLM error: 429 RESOURCE_EXHAUSTED"]


def test_without_a_bus_the_loop_behaves_exactly_as_before() -> None:
    def run(bus: EventBus | None) -> tuple[TurnResult, list[str], list[str]]:
        shown: list[str] = []
        loop = _loop(bus, transcripts=["hello"], responses=[LLMResponse(text="Hi.")], shown=shown)
        result = asyncio.run(loop.run_once())
        assert isinstance(loop.tts, FakeTTSEngine)
        return result, shown, loop.tts.spoken

    assert run(None) == run(EventBus())


def test_agent_publishes_declined_error_and_unknown_tool_events() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    registry = ToolRegistry()
    for tool in (DangerousTool(), ExplodingTool(), RecordingTool()):
        registry.register(tool)
    llm = FakeLLMClient(
        [
            _call("delete_everything", "d1", text="all"),
            _call("explode", "e1", text="x"),
            _call("no_such_tool", "u1"),
            _call("echo", "b1", wrong="field"),
            LLMResponse(text="Done."),
        ]
    )
    agent = Agent(llm, registry, confirm=lambda call: False, events=bus)

    asyncio.run(agent.run("go"))

    tools = [(e.data["id"], e.data["status"]) for e in _drain(queue) if e.type == TOOL]
    assert tools == [
        ("d1", "declined"),
        ("e1", "started"),
        ("e1", "error"),
        ("u1", "error"),
        ("b1", "started"),
        ("b1", "error"),  # bad arguments
    ]


def test_agent_serialises_concurrent_turns() -> None:
    class SlowLLM(FakeLLMClient):
        async def complete(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
        ) -> LLMResponse:
            self.requests.append((list(messages), None))
            await asyncio.sleep(0.01)
            return LLMResponse(text=f"re: {messages[-1].content}")

    llm = SlowLLM([])
    agent = Agent(llm, ToolRegistry())

    async def both() -> list[str]:
        return list(await asyncio.gather(agent.run("first"), agent.run("second")))

    assert asyncio.run(both()) == ["re: first", "re: second"]
    # The second turn only started after the first finished: it saw the whole first turn.
    assert [len(messages) for messages, _ in llm.requests] == [1, 3]
    assert [m.content for m in agent.history] == ["first", "re: first", "second", "re: second"]


# --- audio hooks ---------------------------------------------------------------


class FakeRawStream:
    def __init__(self, *, callback: Any, **kwargs: Any) -> None:
        self.callback = callback

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_mic_stream_hands_every_block_to_the_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    streams: list[FakeRawStream] = []
    monkeypatch.setattr(
        audio_io.sd, "InputStream", lambda **kw: streams.append(FakeRawStream(**kw)) or streams[-1]
    )
    heard: list[list[float]] = []

    with audio_io.MicStream(audio_listener=lambda block: heard.append(block.tolist())) as mic:
        streams[0].callback(np.array([[0.25], [0.5]], dtype=np.float32), 2, None, None)
        assert mic.read(2).tolist() == [0.25, 0.5]  # reading is unaffected

    assert heard == [[0.25, 0.5]]


def test_play_reports_blocks_paced_to_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(audio_io.sd, "play", lambda *a, **k: calls.append("play"))
    monkeypatch.setattr(audio_io.sd, "wait", lambda: calls.append("wait"))
    slept: list[float] = []
    monkeypatch.setattr(audio_io.time, "sleep", slept.append)
    blocks: list[int] = []

    audio_io.play(
        np.zeros(1000, dtype=np.float32), 1000, level_listener=lambda b: blocks.append(len(b))
    )

    assert blocks == [50] * 20  # 50 ms blocks covering the whole clip
    assert calls == ["play", "wait"]
    # Each block is handed over when playback reaches it (sleep is a no-op here, so
    # the waits grow toward the clip's 1 s length).
    assert slept == sorted(slept)
    assert 0.9 < slept[-1] <= 1.0


def test_play_without_listener_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(audio_io.sd, "play", lambda *a, **k: calls.append("play"))
    monkeypatch.setattr(audio_io.sd, "wait", lambda: calls.append("wait"))
    monkeypatch.setattr(audio_io.time, "sleep", lambda s: calls.append("sleep"))

    audio_io.play(np.zeros(1000, dtype=np.float32), 1000)

    assert calls == ["play", "wait"]


def test_piper_output_is_metered_for_the_orb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = tmp_path / "en_US-test-medium.onnx"
    voice.write_bytes(b"fake")
    voice.with_name(voice.name + ".json").write_text("{}", encoding="utf-8")

    class FakeVoice:
        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes((np.full(16000, 8000, dtype="<i2")).tobytes())  # 1 s tone-ish

    monkeypatch.setattr(audio_io.sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(audio_io.sd, "wait", lambda: None)
    monkeypatch.setattr(audio_io.time, "sleep", lambda s: None)
    bus = EventBus()
    queue = bus.subscribe()
    settings = Settings(vault_path=tmp_path, tts_voice=voice, _env_file=None)  # type: ignore[call-arg]
    engine = create_tts_engine(settings, level_listener=LevelMeter(bus, LEVEL_SPEAKER))
    engine._voice = FakeVoice()  # type: ignore[attr-defined]

    engine.speak("Hello there.")

    levels = [e.data for e in _drain(queue) if e.type == LEVEL]
    assert levels, "speaking produced no level events"
    assert {d["source"] for d in levels} == {LEVEL_SPEAKER}
    assert levels[0]["rms"] == pytest.approx(8000 / 32768, abs=1e-3)
