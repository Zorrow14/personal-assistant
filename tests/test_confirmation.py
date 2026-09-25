"""The confirmation gate with real side-effect tools: deny, approve, prompt text, master switch."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.core import agent as agent_module
from jarvis.core.agent import DECLINED_RESULT, Agent, confirm_on_cli
from jarvis.core.interfaces import LLMResponse, ToolCall
from jarvis.memory.auto_index import AutoIndexingAgent
from jarvis.tools._reminders import ReminderStore
from jarvis.tools.base import ToolRegistry
from jarvis.tools.time_tools import SetReminderTool
from tests.fakes import FakeLLMClient

NOW = datetime(2026, 9, 26, 17, 45, tzinfo=timezone(timedelta(hours=6, minutes=30)))
CALL = ToolCall("r1", "set_reminder", {"text": "test the gate", "when": "tomorrow"})


def _setup(tmp_path: Path) -> tuple[ToolRegistry, FakeLLMClient, Path]:
    path = tmp_path / "reminders.json"
    registry = ToolRegistry()
    registry.register(SetReminderTool(ReminderStore(path), lambda: NOW))
    llm = FakeLLMClient(
        [LLMResponse(text=None, tool_calls=[CALL], stop_reason="tool_use"), LLMResponse(text="Done.")]
    )
    return registry, llm, path


def test_denied_reminder_is_not_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def deny(call: ToolCall, *, category: str = "general") -> bool:
        seen.append((call.name, category))
        return False

    monkeypatch.setattr(agent_module, "confirm_on_cli", deny)
    registry, llm, path = _setup(tmp_path)

    asyncio.run(Agent(llm, registry).run("set a reminder to test the gate tomorrow"))

    assert seen == [("set_reminder", "time")]  # the prompt learns the tool's category
    assert llm.requests[1][0][-1].content == DECLINED_RESULT
    assert not path.exists()


def test_approved_reminder_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "confirm_on_cli", lambda call, **_: True)
    registry, llm, path = _setup(tmp_path)

    asyncio.run(Agent(llm, registry).run("set a reminder"))

    assert "Reminder saved" in llm.requests[1][0][-1].content
    assert [r.text for r in ReminderStore(path).all()] == ["test the gate"]


@pytest.mark.parametrize("agent_cls", [Agent, AutoIndexingAgent])
def test_master_switch_off_bypasses_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, agent_cls: type
) -> None:
    def must_not_ask(call: ToolCall, **_: object) -> bool:
        raise AssertionError("gate should be bypassed")

    monkeypatch.setattr(agent_module, "confirm_on_cli", must_not_ask)
    registry, llm, path = _setup(tmp_path)
    kwargs = {"confirm_side_effects": False}
    if agent_cls is AutoIndexingAgent:
        from jarvis.memory.indexer import VaultIndexer
        from tests.fakes import FakeEmbedder, FakeVectorStore

        (tmp_path / "vault" / "Jarvis").mkdir(parents=True)
        kwargs["indexer"] = VaultIndexer(
            tmp_path / "vault" / "Jarvis", FakeEmbedder(), FakeVectorStore(), tmp_path / "m.json"
        )

    asyncio.run(agent_cls(llm, registry, **kwargs).run("set a reminder"))

    assert ReminderStore(path).all()[0].text == "test the gate"


def test_gate_on_is_the_default(tmp_path: Path) -> None:
    registry, llm, _ = _setup(tmp_path)
    assert Agent(llm, registry).confirm_side_effects is True


def test_prompt_shows_tool_category_and_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["", "y", "YES", "nope"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    decisions = [confirm_on_cli(CALL, category="time") for _ in range(4)]

    assert decisions == [False, True, True, False]  # only explicit yes approves
    out = capsys.readouterr().out
    assert "CONFIRM: Jarvis wants to run an action with side effects" in out
    assert "tool:      set_reminder" in out
    assert "category:  time" in out
    assert '"text": "test the gate"' in out and '"when": "tomorrow"' in out


def test_eof_at_prompt_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert confirm_on_cli(CALL) is False
