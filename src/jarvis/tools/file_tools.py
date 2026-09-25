"""Read-only access to local text files, confined to a sandbox folder.

Uses the same guard pattern as `memory/vault.py`'s `_safe_path`: resolve the
requested path (following `..` and symlinks), then refuse anything that doesn't
land inside the sandbox root. The sandbox defaults to the vault folder.
"""

import asyncio
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from jarvis.logging import get_logger
from jarvis.tools.base import Tool
from jarvis.tools.context import ToolContext

MAX_CHARS = 20_000

log = get_logger(__name__)


class SandboxViolation(PermissionError):
    """A path resolves outside the sandbox root."""


def resolve_in_sandbox(root: Path, requested: str) -> Path:
    """Resolve `requested` (relative to `root`, or absolute) and require it stays inside `root`.

    Raises:
        SandboxViolation: If the resolved path is outside `root`.
    """
    root = Path(root).resolve()
    target = (root / Path(requested).expanduser()).resolve()
    if not target.is_relative_to(root):
        log.warning("read_file.path_rejected", requested=requested, resolved=str(target))
        raise SandboxViolation(f"refusing to read outside the allowed folder {root}: {requested}")
    return target


class ReadFileArgs(BaseModel):
    """Input for `read_local_file`."""

    path: str = Field(
        description="File path relative to the user's vault, e.g. 'Projects/plan.md'."
    )


class ReadLocalFileTool(Tool):
    """Read a text file from the user's vault (read-only, sandboxed)."""

    name = "read_local_file"
    description = (
        "Read a text file from the user's notes folder (their Obsidian vault) by path, e.g. "
        "'Jarvis/Tasks/2026-09-25-1827-finish-the-vault-module.md'. Only files inside that "
        "folder can be read; anything outside is refused. Large files are truncated."
    )
    args_model = ReadFileArgs
    requires_confirmation = False
    category = "files"

    def __init__(self, sandbox_root: Path, *, max_chars: int = MAX_CHARS) -> None:
        """
        Args:
            sandbox_root: The only folder reads may come from.
            max_chars: Content beyond this is cut off (with a note).
        """
        self._root = Path(sandbox_root).resolve()
        self._max_chars = max_chars

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with `file_sandbox_root` (default: the vault)."""
        return cls(context.file_sandbox_root)

    async def run(self, **kwargs: Any) -> str:
        """Return the file's text (capped).

        Raises:
            SandboxViolation: If the path escapes the sandbox.
            FileNotFoundError / IsADirectoryError / ValueError: For missing,
                directory or binary targets.
        """
        args = ReadFileArgs.model_validate(kwargs)
        target = resolve_in_sandbox(self._root, args.path)
        return await asyncio.to_thread(self._read, target, args.path)

    def _read(self, target: Path, requested: str) -> str:
        if not target.exists():
            raise FileNotFoundError(f"no such file in the vault: {requested}")
        if target.is_dir():
            raise IsADirectoryError(f"{requested} is a folder, not a file")
        with target.open("rb") as fh:
            raw = fh.read(self._max_chars * 4 + 1)  # UTF-8 is at most 4 bytes per char
        if b"\x00" in raw[:8192]:
            raise ValueError(f"{requested} looks like a binary file, not text")
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        rel = target.relative_to(self._root).as_posix()
        if len(text) > self._max_chars:
            return f"[{rel}, truncated to the first {self._max_chars} characters]\n{text[: self._max_chars]}"
        return f"[{rel}]\n{text}"
