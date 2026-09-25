"""Tool contract and registry tests."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.memory.vault import Vault
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool
from tests.fakes import RecordingTool


@pytest.fixture
def note_tool(tmp_path: Path) -> WriteTaskNoteTool:
    return WriteTaskNoteTool(Vault(tmp_path))


def test_registry_emits_schema_from_args_model(note_tool: WriteTaskNoteTool) -> None:
    registry = ToolRegistry()
    registry.register(note_tool)

    [spec] = registry.schemas()

    assert spec.name == "write_task_note"
    assert "Obsidian vault" in spec.description
    schema = spec.input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"command", "body", "tags", "tools_used"}
    assert sorted(schema["required"]) == ["body", "command"]
    assert schema["properties"]["command"]["type"] == "string"
    assert "description" in schema["properties"]["body"]


def test_registry_lookup_and_duplicates() -> None:
    registry = ToolRegistry()
    tool = RecordingTool()
    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.get("missing") is None
    assert "echo" in registry
    assert registry.names() == ["echo"]
    with pytest.raises(ValueError):
        registry.register(RecordingTool())


def test_parse_args_validates_against_args_model(note_tool: WriteTaskNoteTool) -> None:
    assert note_tool.parse_args({"command": "c", "body": "b"}) == {
        "command": "c",
        "body": "b",
        "tags": None,
        "tools_used": None,
    }
    with pytest.raises(ValidationError):
        note_tool.parse_args({"command": "c"})
    with pytest.raises(ValidationError):
        note_tool.parse_args({"command": "c", "body": "b", "tags": "not-a-list"})


def test_write_task_note_tool_writes_note(note_tool: WriteTaskNoteTool, tmp_path: Path) -> None:
    result = asyncio.run(
        note_tool.run(command="Water the plants", body="Done.", tools_used=["write_task_note"])
    )

    [note] = (tmp_path / "Jarvis" / "Tasks").glob("*-water-the-plants.md")
    assert result == f"Saved task note: {note.resolve()}"
    assert "tools_used: [write_task_note]" in note.read_text(encoding="utf-8")
