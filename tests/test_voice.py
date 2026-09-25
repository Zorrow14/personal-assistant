"""Voice-loop orchestration tests: fake STT/TTS/LLM, tmp vault, no hardware."""

import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from jarvis.core.agent import Agent
from jarvis.core.interfaces import LLMResponse, ToolCall
from jarvis.core.voice_session import (
    NOT_HEARD_MESSAGE,
    TurnOutcome,
    VoiceSession,
    is_exit_command,
    speakable,
)
from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool
from tests.fakes import FakeLLMClient, FakeSTTEngine, FakeTTSEngine

AUDIO = np.zeros(16000, dtype=np.float32)


def _session(
    llm: FakeLLMClient, stt: FakeSTTEngine, tts: FakeTTSEngine, vault: Vault | None = None
) -> tuple[VoiceSession, list[str]]:
    registry = ToolRegistry()
    if vault is not None:
        registry.register(WriteTaskNoteTool(vault))
    shown: list[str] = []
    return VoiceSession(Agent(llm, registry), stt, tts, display=shown.append), shown


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault(tmp_path, clock=lambda: datetime(2026, 9, 26, 10, 0))


def test_transcript_goes_to_agent_and_reply_is_spoken(vault: Vault) -> None:
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        "c1",
                        "write_task_note",
                        {"command": "Finish the voice module", "body": "Done."},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Logged it."),
        ]
    )
    stt = FakeSTTEngine(["Log that I finished the voice module."])
    tts = FakeTTSEngine()
    session, shown = _session(llm, stt, tts, vault)

    outcome = asyncio.run(session.run_turn(AUDIO))

    assert outcome is TurnOutcome.REPLIED
    assert stt.received[0] is AUDIO
    assert llm.requests[0][0][0].content == "Log that I finished the voice module."
    assert tts.spoken == ["Logged it."]
    assert shown == ["you> Log that I finished the voice module.", "jarvis> Logged it."]
    assert (vault.jarvis_root / "Tasks" / "2026-09-26-1000-finish-the-voice-module.md").is_file()


@pytest.mark.parametrize("transcript", ["", "   "])
def test_empty_transcript_skips_agent(transcript: str) -> None:
    llm = FakeLLMClient([])
    tts = FakeTTSEngine()
    session, shown = _session(llm, FakeSTTEngine([transcript]), tts)

    outcome = asyncio.run(session.run_turn(AUDIO))

    assert outcome is TurnOutcome.EMPTY
    assert llm.requests == []
    assert tts.spoken == []
    assert shown == [NOT_HEARD_MESSAGE]


def test_spoken_exit_ends_session_without_agent() -> None:
    llm = FakeLLMClient([])
    tts = FakeTTSEngine()
    session, _ = _session(llm, FakeSTTEngine(["Exit."]), tts)

    assert asyncio.run(session.run_turn(AUDIO)) is TurnOutcome.EXIT
    assert llm.requests == []
    assert tts.spoken == []


def test_typed_text_uses_same_agent_path() -> None:
    llm = FakeLLMClient([LLMResponse(text="Hello!")])
    tts = FakeTTSEngine()
    stt = FakeSTTEngine([])
    session, _ = _session(llm, stt, tts)

    assert asyncio.run(session.respond("hi")) is TurnOutcome.REPLIED
    assert stt.received == []
    assert tts.spoken == ["Hello!"]


def test_markdown_is_not_read_aloud() -> None:
    llm = FakeLLMClient([LLMResponse(text="**Done** - saved `note.md`")])
    tts = FakeTTSEngine()
    session, shown = _session(llm, FakeSTTEngine(["do it"]), tts)

    asyncio.run(session.run_turn(AUDIO))

    assert shown[-1] == "jarvis> **Done** - saved `note.md`"  # screen keeps formatting
    assert tts.spoken == ["Done - saved note.md"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("exit", True), ("Quit.", True), (" EXIT! ", True), ("exit the app", False), ("", False)],
)
def test_is_exit_command(text: str, expected: bool) -> None:
    assert is_exit_command(text) is expected


def test_speakable_collapses_whitespace() -> None:
    assert speakable("# Title\n\n> quote _x_") == "Title quote x"
