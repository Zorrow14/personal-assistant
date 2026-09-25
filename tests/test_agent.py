"""Agent loop tests. Uses FakeLLMClient: no Gemini, no network."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.core import agent as agent_module
from jarvis.core.agent import DECLINED_RESULT, MAX_ITERATIONS_REPLY, Agent
from jarvis.core.interfaces import LLMError, LLMResponse, Message, ToolCall
from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool
from tests.fakes import DangerousTool, ExplodingTool, FakeLLMClient, RecordingTool


def _call(name: str, call_id: str = "call-1", **input: object) -> LLMResponse:
    return LLMResponse(
        text=None, tool_calls=[ToolCall(id=call_id, name=name, input=input)], stop_reason="tool_use"
    )


def _text(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end_turn")


def _agent(llm: FakeLLMClient, *tools: object, **kwargs: object) -> Agent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)  # type: ignore[arg-type]
    return Agent(llm, registry, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    root = tmp_path / "vault"
    root.mkdir()
    return Vault(root, clock=lambda: datetime(2026, 9, 25, 9, 5))


def test_agent_calls_tool_writes_note_and_returns_text(vault: Vault) -> None:
    llm = FakeLLMClient(
        [
            _call(
                "write_task_note",
                command="Finish the vault module",
                body="Finished the vault module.",
                tags=["coding"],
            ),
            _text("Logged it."),
        ]
    )
    agent = _agent(llm, WriteTaskNoteTool(vault))

    reply = asyncio.run(agent.run("log that I finished the vault module"))

    assert reply == "Logged it."
    note = vault.jarvis_root / "Tasks" / "2026-09-25-0905-finish-the-vault-module.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert 'command: "Finish the vault module"' in text
    assert "tags: [coding]" in text

    # Tools were offered; second request carried the linked tool result.
    first_messages, first_tools = llm.requests[0]
    assert [t.name for t in first_tools or []] == ["write_task_note"]
    assert [m.role for m in first_messages] == ["user"]
    second_messages, _ = llm.requests[1]
    tool_msg = second_messages[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call-1"
    assert tool_msg.tool_name == "write_task_note"
    assert not tool_msg.is_error
    assert str(note) in tool_msg.content
    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]


def test_confirmation_gate_skips_tool_when_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[ToolCall] = []

    def deny(call: ToolCall) -> bool:
        asked.append(call)
        return False

    monkeypatch.setattr(agent_module, "confirm_on_cli", deny)
    tool = DangerousTool()
    llm = FakeLLMClient([_call("delete_everything", text="all"), _text("Okay, I won't.")])

    reply = asyncio.run(_agent(llm, tool).run("delete everything"))

    assert reply == "Okay, I won't."
    assert tool.runs == []
    assert [c.name for c in asked] == ["delete_everything"]
    tool_msg = llm.requests[1][0][-1]
    assert tool_msg.content == DECLINED_RESULT


def test_confirmation_gate_runs_tool_when_approved() -> None:
    tool = DangerousTool()
    llm = FakeLLMClient([_call("delete_everything", text="all"), _text("Done.")])

    asyncio.run(_agent(llm, tool, confirm=lambda call: True).run("delete everything"))

    assert tool.runs == [{"text": "all"}]


def test_tools_without_confirmation_never_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(call: ToolCall) -> bool:
        raise AssertionError("should not ask")

    monkeypatch.setattr(agent_module, "confirm_on_cli", fail)
    tool = RecordingTool()
    llm = FakeLLMClient([_call("echo", text="hi"), _text("ok")])

    asyncio.run(_agent(llm, tool).run("echo hi"))

    assert tool.runs == [{"text": "hi"}]


def test_unknown_tool_is_reported_not_raised() -> None:
    llm = FakeLLMClient([_call("launch_rockets"), _text("I can't do that.")])

    reply = asyncio.run(_agent(llm, RecordingTool()).run("launch the rockets"))

    assert reply == "I can't do that."
    tool_msg = llm.requests[1][0][-1]
    assert tool_msg.is_error
    assert "unknown tool 'launch_rockets'" in tool_msg.content
    assert "echo" in tool_msg.content


def test_invalid_arguments_are_reported(vault: Vault) -> None:
    llm = FakeLLMClient([_call("write_task_note", command="x"), _text("Sorry.")])

    asyncio.run(_agent(llm, WriteTaskNoteTool(vault)).run("log something"))

    tool_msg = llm.requests[1][0][-1]
    assert tool_msg.is_error
    assert "invalid arguments" in tool_msg.content
    assert "body" in tool_msg.content
    assert not (vault.jarvis_root / "Tasks").exists()


def test_tool_exception_is_reported() -> None:
    llm = FakeLLMClient([_call("explode", text="now"), _text("It failed.")])

    reply = asyncio.run(_agent(llm, ExplodingTool()).run("explode"))

    assert reply == "It failed."
    tool_msg = llm.requests[1][0][-1]
    assert tool_msg.is_error
    assert "RuntimeError: kaboom" in tool_msg.content


def test_stops_at_max_iterations() -> None:
    llm = FakeLLMClient([_call("echo", call_id=f"c{i}", text="again") for i in range(3)])
    tool = RecordingTool()

    reply = asyncio.run(_agent(llm, tool, max_iterations=3).run("loop forever"))

    assert reply == MAX_ITERATIONS_REPLY
    assert len(llm.requests) == 3
    assert len(tool.runs) == 3


def test_history_carries_across_turns() -> None:
    llm = FakeLLMClient([_text("Hi!"), _text("You said hello.")])
    agent = _agent(llm)

    asyncio.run(agent.run("hello"))
    asyncio.run(agent.run("what did I say?"))

    second_messages, second_tools = llm.requests[1]
    assert [(m.role, m.content) for m in second_messages] == [
        ("user", "hello"),
        ("assistant", "Hi!"),
        ("user", "what did I say?"),
    ]
    assert second_tools is None  # empty registry -> tools disabled


def test_failed_turn_is_rolled_back_from_history() -> None:
    class BrokenLLM(FakeLLMClient):
        async def complete(self, messages, tools=None):  # type: ignore[no-untyped-def]
            raise LLMError("quota exhausted")

    agent = _agent(BrokenLLM([]))
    agent.history.append(Message(role="user", content="earlier"))

    with pytest.raises(LLMError):
        asyncio.run(agent.run("this will fail"))

    assert [m.content for m in agent.history] == ["earlier"]


def test_empty_model_reply_gets_placeholder() -> None:
    llm = FakeLLMClient([LLMResponse(text=None, stop_reason="max_tokens")])

    reply = asyncio.run(_agent(llm).run("hi"))

    assert "max_tokens" in reply
