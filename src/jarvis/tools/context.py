"""Runtime dependencies handed to tools at discovery time.

How tools get their dependencies: the CLI builds one `ToolContext` with
everything a tool might need, then `ToolRegistry.discover(context=...)` calls
each tool class's `from_context(context)`. A tool takes only what it uses
(e.g. `cls(context.vault)`), so adding a tool never means editing the CLI.
When a new kind of dependency appears, add a field here.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jarvis.config import Settings
from jarvis.memory.factory import MemoryStack
from jarvis.memory.vault import Vault


def local_now() -> datetime:
    """Current local time, timezone-aware."""
    return datetime.now().astimezone()


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool might need to construct itself."""

    settings: Settings
    vault: Vault
    memory: MemoryStack
    clock: Callable[[], datetime] = field(default=local_now)

    @property
    def file_sandbox_root(self) -> Path:
        """The only folder file tools may read under: `file_sandbox_root`, else the vault."""
        root = self.settings.file_sandbox_root
        return root.expanduser() if root is not None else self.vault.vault_root

    @property
    def reminders_path(self) -> Path:
        """Where reminders are stored (local JSON)."""
        return self.settings.reminders_path.expanduser()
