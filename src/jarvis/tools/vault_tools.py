"""Tools that write to the Obsidian vault."""

import asyncio
from typing import Any, Self

from pydantic import BaseModel, Field

from jarvis.memory.vault import Vault
from jarvis.tools.base import Tool
from jarvis.tools.context import ToolContext


class WriteTaskNoteArgs(BaseModel):
    """Input for `write_task_note`."""

    command: str = Field(
        description="The task as a short imperative phrase, e.g. 'Finish the vault module'."
    )
    body: str = Field(description="Markdown body: what was done, with any useful details.")
    tags: list[str] | None = Field(
        default=None, description="Optional short lowercase tags, e.g. ['coding']."
    )
    tools_used: list[str] | None = Field(
        default=None, description="Optional names of tools used to complete the task."
    )


class WriteTaskNoteTool(Tool):
    """Journal a task as a Markdown note in `<vault>/Jarvis/Tasks/`."""

    name = "write_task_note"
    description = (
        "Record a task or event as a note in the user's Obsidian vault. Use it whenever "
        "the user asks you to log, record or note something, or after you complete a task "
        "for them."
    )
    args_model = WriteTaskNoteArgs
    requires_confirmation = False
    category = "vault"

    def __init__(self, vault: Vault) -> None:
        self._vault = vault

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with the configured vault."""
        return cls(context.vault)

    async def run(self, **kwargs: Any) -> str:
        """Write the note and return a confirmation including its path."""
        args = WriteTaskNoteArgs.model_validate(kwargs)
        path = await asyncio.to_thread(
            self._vault.write_task_note,
            args.command,
            args.body,
            tags=args.tags,
            tools_used=args.tools_used,
        )
        return f"Saved task note: {path}"
