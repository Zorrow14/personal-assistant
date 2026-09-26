"""Wake-word loop tests: fake detector/VAD/mic/STT/TTS/LLM, tmp vault, no hardware."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.core.agent import Agent
from jarvis.core.interfaces import AudioSamples, LLMError, LLMResponse, ToolCall
from jarvis.core.voice_loop import LoopState, TurnResult, WakeWordLoop
from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool
from tests.fakes import (
    FakeFrameSource,
    FakeLLMClient,
    FakeSTTEngine,
    FakeTTSEngine,
    FakeVAD,
    FakeWakeWordDetector,
)


class Harness:
    """Builds a WakeWordLoop from fakes and records everything that happens."""

    def __init__(
        self,
        *,
        scores: list[float],
        transcripts: list[str],
        responses: list[LLMResponse],
        vault: Vault | None = None,
    ) -> None:
        self.detector = FakeWakeWordDetector(scores)
        self.vad = FakeVAD()
        self.mic = FakeFrameSource()
        self.stt = FakeSTTEngine(transcripts)
        self.tts = FakeTTSEngine()
        self.llm = FakeLLMClient(responses)
        self.shown: list[str] = []
        self.states: list[LoopState] = []
        self.chimes = 0
        registry = ToolRegistry()
        if vault is not None:
            registry.register(WriteTaskNoteTool(vault))
        self.loop = WakeWordLoop(
            Agent(self.llm, registry),
            self.stt,
            self.tts,
            self.detector,
            self.vad,
            self.mic,
            threshold=0.5,
            chime=self._chime,
            display=self.shown.append,
            on_state=self.states.append,
        )

    def _chime(self) -> None:
        # The chime fires before this turn's detector reset.
        assert self.chimes == self.detector.resets
        self.chimes += 1


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault(tmp_path, clock=lambda: datetime(2026, 9, 26, 12, 0))


def test_full_cycle_wake_record_transcribe_agent_speak(vault: Vault) -> None:
    h = Harness(
        scores=[0.01, 0.2, 0.93],
        transcripts=["Log that I finished the wake-word module."],
        responses=[
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        "c1",
                        "write_task_note",
                        {"command": "Finish the wake-word module", "body": "Done."},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Logged it."),
        ],
        vault=vault,
    )

    result = asyncio.run(h.loop.run_once())

    assert result is TurnResult.REPLIED
    assert h.detector.frames_seen == 3  # stopped reading at the first score >= 0.5
    assert h.chimes == 1
    assert h.vad.calls == 1
    assert h.stt.received == [h.vad.buffer]
    assert h.llm.requests[0][0][0].content == "Log that I finished the wake-word module."
    assert h.tts.spoken == ["Logged it."]
    assert h.states == [
        LoopState.IDLE,
        LoopState.LISTENING,
        LoopState.PROCESSING,
        LoopState.SPEAKING,
        LoopState.IDLE,
    ]
    assert h.loop.state is LoopState.IDLE
    assert "you> Log that I finished the wake-word module." in h.shown
    assert "jarvis> Logged it." in h.shown
    assert (
        vault.jarvis_root / "Tasks" / "2026-09-26-1200-finish-the-wake-word-module.md"
    ).is_file()


def test_detector_reset_after_trigger() -> None:
    h = Harness(scores=[0.1, 0.7], transcripts=["hello"], responses=[LLMResponse(text="Hi.")])

    asyncio.run(h.loop.run_once())

    assert h.detector.resets == 1
    assert h.chimes == 1


def test_score_exactly_at_threshold_triggers() -> None:
    h = Harness(scores=[0.5], transcripts=["hello"], responses=[LLMResponse(text="Hi.")])
    assert asyncio.run(h.loop.run_once()) is TurnResult.REPLIED


@pytest.mark.parametrize("transcript", ["", "  \n "])
def test_empty_transcript_returns_to_idle_without_agent(transcript: str) -> None:
    h = Harness(scores=[0.9], transcripts=[transcript], responses=[])

    result = asyncio.run(h.loop.run_once())

    assert result is TurnResult.EMPTY
    assert h.llm.requests == []
    assert h.tts.spoken == []
    assert h.states[-1] is LoopState.IDLE
    assert LoopState.SPEAKING not in h.states
    assert any("Didn't catch that" in line for line in h.shown)


def test_stale_audio_is_cleared_before_listening_and_after_chime() -> None:
    h = Harness(scores=[0.9], transcripts=["hello"], responses=[LLMResponse(text="Hi.")])

    asyncio.run(h.loop.run_once())

    # Once entering IDLE (drops audio buffered while speaking) and once after the chime.
    assert h.mic.clears == 2


def test_no_chime_when_disabled() -> None:
    h = Harness(scores=[0.9], transcripts=["hello"], responses=[LLMResponse(text="Hi.")])
    h.loop._chime = None

    asyncio.run(h.loop.run_once())

    assert h.chimes == 0
    assert h.detector.resets == 1


def test_run_keeps_listening_until_spoken_exit() -> None:
    h = Harness(
        scores=[0.9, 0.9, 0.9],
        transcripts=["hello", "", "exit"],
        responses=[LLMResponse(text="Hi.")],
    )

    asyncio.run(h.loop.run())

    assert len(h.llm.requests) == 1  # "" skipped the agent; "exit" ended the loop
    assert h.tts.spoken == ["Hi."]
    assert h.detector.resets == 3


def test_llm_error_is_reported_and_loop_continues() -> None:
    class FlakyLLM(FakeLLMClient):
        async def complete(self, messages, tools=None):  # type: ignore[no-untyped-def]
            if not self.requests:
                self.requests.append((list(messages), None))
                raise LLMError("429 RESOURCE_EXHAUSTED")
            return await super().complete(messages, tools)

    h = Harness(scores=[0.9, 0.9], transcripts=["hello", "quit"], responses=[])
    h.loop.agent.llm = FlakyLLM([])

    asyncio.run(h.loop.run())

    assert any("[LLM error] 429" in line for line in h.shown)
    # Phase 6B: a failed turn is apologised for aloud instead of going silent.
    assert h.tts.spoken == ["Sorry, something went wrong."]


def test_stop_ends_wait_for_wake_word() -> None:
    class StoppingSource(FakeFrameSource):
        def __init__(self, loop_ref: list[WakeWordLoop]) -> None:
            super().__init__()
            self._loop_ref = loop_ref

        def read(self, n_samples: int) -> AudioSamples:
            if self.samples_read >= 1280 * 3:
                self._loop_ref[0].stop()
            return super().read(n_samples)

    h = Harness(scores=[0.0] * 10, transcripts=[], responses=[])
    ref: list[WakeWordLoop] = []
    source = StoppingSource(ref)
    h.loop = WakeWordLoop(
        h.loop.agent, h.stt, h.tts, h.detector, h.vad, source, display=h.shown.append
    )
    ref.append(h.loop)

    asyncio.run(h.loop.run())

    assert h.detector.frames_seen == 4
    assert h.vad.calls == 0


def test_persistent_failure_stops_loop_instead_of_spinning() -> None:
    class BrokenTTS(FakeTTSEngine):
        def speak(self, text: str) -> None:
            raise RuntimeError("audio device gone")

    h = Harness(
        scores=[0.9] * 5,
        transcripts=["a", "b", "c", "d", "e"],
        responses=[LLMResponse(text="ok")] * 5,
    )
    h.loop.tts = BrokenTTS()

    with pytest.raises(RuntimeError, match="audio device gone"):
        asyncio.run(h.loop.run())

    assert len(h.llm.requests) == 3  # MAX_CONSECUTIVE_FAILURES turns, then gave up
