"""Tool auto-discovery: real package, temporary plugin packages, whitelist."""

import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from jarvis.tools.base import DuplicateToolError, ToolRegistry, discover_tool_classes
from tests.fakes import make_tool_context

EXPECTED = {
    "write_task_note": ("vault", False),
    "search_memory": ("memory", False),
    "web_search": ("web", False),
    "get_datetime": ("time", False),
    "set_reminder": ("time", True),
    "list_reminders": ("time", False),
    "read_local_file": ("files", False),
}


def test_discovers_every_real_tool_without_manual_registration(tmp_path: Path) -> None:
    registry = ToolRegistry()

    added = registry.discover(context=make_tool_context(tmp_path))

    assert set(added) == set(EXPECTED)
    for name, (category, confirm) in EXPECTED.items():
        tool = registry.get(name)
        assert tool is not None
        assert tool.category == category
        assert tool.requires_confirmation is confirm
        assert tool.safe is not confirm
    assert registry.discovery_problems == []
    assert {spec.name for spec in registry.schemas()} == set(EXPECTED)


def test_discovery_is_idempotent(tmp_path: Path) -> None:
    registry = ToolRegistry()
    context = make_tool_context(tmp_path)
    registry.discover(context=context)

    assert registry.discover(context=context) == []
    assert len(registry) == len(EXPECTED)


def test_enabled_tools_whitelist_filters(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.discover(context=make_tool_context(tmp_path))

    unknown = registry.restrict(["web_search", "get_datetime", "launch_rockets"])

    assert registry.names() == ["get_datetime", "web_search"] or set(registry.names()) == {
        "get_datetime",
        "web_search",
    }
    assert unknown == ["launch_rockets"]


def _plugin_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> str:
    """Create an importable throwaway package with the given modules."""
    name = f"jarvis_plugins_{uuid.uuid4().hex[:8]}"
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for filename, source in files.items():
        (pkg / filename).write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


TOOL_SOURCE = """
    from typing import Any
    from pydantic import BaseModel
    from jarvis.tools.base import Tool

    class _Args(BaseModel):
        pass

    class {cls}(Tool):
        name = "{name}"
        description = "test tool"
        args_model = _Args
        category = "test"

        async def run(self, **kwargs: Any) -> str:
            return "ok"
"""


def test_dropping_one_new_file_is_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _plugin_package(
        tmp_path,
        monkeypatch,
        {
            "dice_tools.py": TOOL_SOURCE.format(cls="RollDiceTool", name="roll_dice"),
            "_private.py": TOOL_SOURCE.format(cls="HiddenTool", name="hidden"),
            "abstract_tools.py": """
                from abc import abstractmethod
                from jarvis.tools.base import Tool

                class BaseFancyTool(Tool):
                    @abstractmethod
                    def fancy(self) -> None: ...

                class _InternalTool(Tool):
                    name = "internal"
                    description = "x"
            """,
        },
    )
    registry = ToolRegistry()

    added = registry.discover(package, context=make_tool_context(tmp_path))

    assert added == ["roll_dice"]  # _module, _Class and abstract bases are skipped
    assert registry.get("roll_dice").category == "test"  # type: ignore[union-attr]


def test_duplicate_tool_names_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _plugin_package(
        tmp_path,
        monkeypatch,
        {
            "a_tools.py": TOOL_SOURCE.format(cls="FirstTool", name="same_name"),
            "b_tools.py": TOOL_SOURCE.format(cls="SecondTool", name="same_name"),
        },
    )
    with pytest.raises(DuplicateToolError, match="same_name"):
        discover_tool_classes(package)
    with pytest.raises(DuplicateToolError):
        ToolRegistry().discover(package, context=make_tool_context(tmp_path))


def test_clash_with_manually_registered_tool_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.fakes import RecordingTool  # name = "echo"

    package = _plugin_package(
        tmp_path, monkeypatch, {"echo_tools.py": TOOL_SOURCE.format(cls="EchoTool", name="echo")}
    )
    registry = ToolRegistry()
    registry.register(RecordingTool())

    with pytest.raises(DuplicateToolError, match="echo"):
        registry.discover(package, context=make_tool_context(tmp_path))


def test_broken_plugins_are_reported_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _plugin_package(
        tmp_path,
        monkeypatch,
        {
            "good_tools.py": TOOL_SOURCE.format(cls="GoodTool", name="good"),
            "broken_import.py": "import this_module_does_not_exist\n",
            "needs_args.py": """
                from typing import Any
                from pydantic import BaseModel
                from jarvis.tools.base import Tool

                class NeedsArgsTool(Tool):
                    name = "needs_args"
                    description = "x"
                    args_model = BaseModel

                    def __init__(self, required: str) -> None:
                        self.required = required

                    async def run(self, **kwargs: Any) -> str:
                        return self.required
            """,
        },
    )
    registry = ToolRegistry()

    added = registry.discover(package, context=make_tool_context(tmp_path))

    assert added == ["good"]
    problems = {p.where.rsplit(".", 1)[-1]: p.error for p in registry.discovery_problems}
    assert "import failed" in problems["broken_import"]
    assert "override from_context" in problems["NeedsArgsTool"]
    sys.modules.pop(f"{package}.broken_import", None)
