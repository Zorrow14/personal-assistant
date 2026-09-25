"""ReadLocalFileTool sandbox tests: the critical security checks."""

import asyncio
import os
from pathlib import Path

import pytest

from jarvis.tools.file_tools import ReadLocalFileTool, SandboxViolation, resolve_in_sandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "Projects").mkdir(parents=True)
    (root / "Projects" / "plan.md").write_text("# Plan\nShip phase 5.", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    (tmp_path / "vault-evil").mkdir()
    (tmp_path / "vault-evil" / "x.md").write_text("sibling", encoding="utf-8")
    return root


def _read(root: Path, path: str, **kwargs: object) -> str:
    return asyncio.run(ReadLocalFileTool(root, **kwargs).run(path=path))  # type: ignore[arg-type]


def test_reads_an_allowed_file(sandbox: Path) -> None:
    assert _read(sandbox, "Projects/plan.md") == "[Projects/plan.md]\n# Plan\nShip phase 5."
    assert _read(sandbox, str(sandbox / "Projects" / "plan.md")).endswith("Ship phase 5.")


@pytest.mark.parametrize(
    "escape",
    [
        "../secret.txt",
        "Projects/../../secret.txt",
        "../vault-evil/x.md",  # sibling whose name merely starts with the root's name
        "..",
    ],
)
def test_rejects_relative_escapes(sandbox: Path, escape: str) -> None:
    with pytest.raises(SandboxViolation):
        _read(sandbox, escape)


def test_rejects_absolute_paths_outside(sandbox: Path, tmp_path: Path) -> None:
    with pytest.raises(SandboxViolation):
        _read(sandbox, str(tmp_path / "secret.txt"))
    system_file = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
    with pytest.raises(SandboxViolation):
        _read(sandbox, system_file)


def test_rejects_symlink_pointing_outside(sandbox: Path, tmp_path: Path) -> None:
    link = sandbox / "innocent.md"
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except OSError:
        pytest.skip("creating symlinks needs Developer Mode / admin on Windows")
    with pytest.raises(SandboxViolation):
        _read(sandbox, "innocent.md")


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_rejects_junction_pointing_outside(sandbox: Path, tmp_path: Path) -> None:
    import _winapi  # CreateJunction needs no admin rights, unlike symlinks

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "loot.txt").write_text("TOP SECRET", encoding="utf-8")
    _winapi.CreateJunction(str(outside), str(sandbox / "shortcut"))

    with pytest.raises(SandboxViolation):
        _read(sandbox, "shortcut/loot.txt")


def test_escape_is_refused_through_the_agent(sandbox: Path, tmp_path: Path) -> None:
    from jarvis.core.agent import Agent
    from jarvis.core.interfaces import LLMResponse, ToolCall
    from jarvis.tools.base import ToolRegistry
    from tests.fakes import FakeLLMClient

    registry = ToolRegistry()
    registry.register(ReadLocalFileTool(sandbox))
    llm = FakeLLMClient(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("r1", "read_local_file", {"path": "../secret.txt"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="I can't read that."),
        ]
    )

    asyncio.run(Agent(llm, registry).run("read ../secret.txt"))

    result = llm.requests[1][0][-1]
    assert result.is_error and "SandboxViolation" in result.content
    assert "TOP SECRET" not in result.content


def test_missing_directory_and_binary_files(sandbox: Path) -> None:
    (sandbox / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    with pytest.raises(FileNotFoundError):
        _read(sandbox, "nope.md")
    with pytest.raises(IsADirectoryError):
        _read(sandbox, "Projects")
    with pytest.raises(ValueError, match="binary"):
        _read(sandbox, "image.png")


def test_large_files_are_truncated(sandbox: Path) -> None:
    (sandbox / "big.md").write_text("a" * 5000, encoding="utf-8")
    text = _read(sandbox, "big.md", max_chars=100)
    header, body = text.split("\n", 1)
    assert header == "[big.md, truncated to the first 100 characters]"
    assert body == "a" * 100


def test_resolve_in_sandbox_returns_resolved_inside_path(sandbox: Path) -> None:
    assert resolve_in_sandbox(sandbox, "Projects/./x/../plan.md") == (sandbox / "Projects" / "plan.md").resolve()
