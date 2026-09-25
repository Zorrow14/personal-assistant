"""search_memory tool tests plus end-to-end agent use, with fakes only."""

import asyncio
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.core.agent import Agent
from jarvis.core.interfaces import LLMResponse, SearchHit, ToolCall
from jarvis.memory.auto_index import AutoIndexingAgent
from jarvis.memory.indexer import VaultIndexer
from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.memory_tools import EMPTY_INDEX_MESSAGE, SearchMemoryTool, format_hits
from jarvis.tools.vault_tools import WriteTaskNoteTool
from tests.fakes import FakeEmbedder, FakeLLMClient, FakeVectorStore

TODAY = date(2026, 9, 26)


class Memory:
    def __init__(self, tmp_path: Path, minute: list[int] | None = None) -> None:
        root = tmp_path / "vault"
        root.mkdir()
        clock_minute = minute or [0]

        def clock() -> datetime:
            clock_minute[0] += 1
            return datetime(2026, 9, 25, 10, clock_minute[0])

        self.vault = Vault(root, clock=clock)
        self.embedder = FakeEmbedder()
        self.store = FakeVectorStore()
        self.indexer = VaultIndexer(
            self.vault.jarvis_root, self.embedder, self.store, tmp_path / "manifest.json"
        )
        self.tool = SearchMemoryTool(self.embedder, self.store, default_top_k=5, today=lambda: TODAY)


@pytest.fixture
def mem(tmp_path: Path) -> Memory:
    return Memory(tmp_path)


def test_search_returns_formatted_citable_hits(mem: Memory) -> None:
    wake = mem.vault.write_task_note(
        "Finish the wake-word module", "Tuned the wake word threshold to 0.5.", tags=["voice"]
    )
    mem.vault.write_task_note("Buy groceries", "Milk, eggs and bread.")
    mem.indexer.reindex_all()

    result = asyncio.run(mem.tool.run(query="wake word module"))

    lines = result.splitlines()
    assert lines[0].startswith("Today is 2026-09-26.")
    assert lines[1] == f"1. [[{wake.stem}]] (2026-09-25, tags: voice, relevance {lines[1].split()[-1]}"
    assert "Tuned the wake word threshold" in lines[2]
    assert result.index(wake.stem) < result.index("buy-groceries")


def test_one_entry_per_note_even_with_several_chunks(mem: Memory) -> None:
    hits = [
        SearchHit("a.md::0", "first part", {"note_name": "a", "date": "2026-09-01"}, 0.6),
        SearchHit("a.md::1", "best part", {"note_name": "a", "date": "2026-09-01"}, 0.9),
        SearchHit("b.md::0", "other", {"note_name": "b"}, 0.4),
    ]
    text = format_hits("q", hits, today=TODAY)
    assert text.count("[[a]]") == 1
    assert "best part" in text and "first part" not in text
    assert text.index("[[a]]") < text.index("[[b]]")


def test_top_k_is_respected(mem: Memory) -> None:
    for i in range(6):
        mem.vault.write_task_note(f"Task number {i}", "same words here")
    mem.indexer.reindex_all()

    result = asyncio.run(mem.tool.run(query="task", top_k=2))

    assert result.count("[[") == 2


def test_empty_index_says_so_plainly(mem: Memory) -> None:
    assert asyncio.run(mem.tool.run(query="anything")) == EMPTY_INDEX_MESSAGE


def test_no_hits_says_nothing_matched() -> None:
    assert format_hits("unicorns", [], today=TODAY) == "No notes in memory matched 'unicorns'."


def test_args_are_validated(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.tool.parse_args({"query": "x", "top_k": 0})
    with pytest.raises(ValidationError):
        mem.tool.parse_args({})


def test_agent_calls_search_memory_then_answers_from_it(mem: Memory) -> None:
    note = mem.vault.write_task_note("Finish the wake-word module", "Threshold tuned.")
    mem.indexer.reindex_all()
    registry = ToolRegistry()
    registry.register(mem.tool)
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("s1", "search_memory", {"query": "wake-word module"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text=f"You asked me to finish the wake-word module ([[{note.stem}]])."),
        ]
    )

    reply = asyncio.run(Agent(llm, registry).run("What did I ask about the wake-word module?"))

    assert f"[[{note.stem}]]" in reply
    assert "search_memory" in [t.name for t in llm.requests[0][1] or []]
    tool_msg = llm.requests[1][0][-1]
    assert tool_msg.tool_name == "search_memory" and not tool_msg.is_error
    assert f"[[{note.stem}]]" in tool_msg.content  # the answer came from a real retrieved note


def test_note_logged_this_session_is_searchable_next_turn(mem: Memory) -> None:
    registry = ToolRegistry()
    registry.register(WriteTaskNoteTool(mem.vault))
    registry.register(mem.tool)
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("w1", "write_task_note", {"command": "Finish the memory module", "body": "RAG works."})
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Logged."),
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("s1", "search_memory", {"query": "memory module"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Found it."),
        ]
    )
    agent = AutoIndexingAgent(llm, registry, indexer=mem.indexer)

    asyncio.run(agent.run("log that I finished the memory module"))
    assert mem.store.count() == 1  # indexed right after the turn, no --reindex
    asyncio.run(agent.run("what did I just log?"))

    search_result = llm.requests[3][0][-1].content
    assert "finish-the-memory-module" in search_result


def test_auto_index_failure_never_fails_the_turn(mem: Memory) -> None:
    class BrokenIndexer(VaultIndexer):
        def index_paths(self, paths):  # type: ignore[no-untyped-def]
            raise RuntimeError("disk full")

    registry = ToolRegistry()
    registry.register(WriteTaskNoteTool(mem.vault))
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("w1", "write_task_note", {"command": "x", "body": "y"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Logged."),
        ]
    )
    indexer = BrokenIndexer(mem.vault.jarvis_root, mem.embedder, mem.store, mem.indexer._manifest_path)

    assert asyncio.run(AutoIndexingAgent(llm, registry, indexer=indexer).run("log x")) == "Logged."
