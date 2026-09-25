"""Keep the memory index current: index whatever notes a turn wrote, right after it.

Chosen approach: before each turn, fingerprint the vault's `Jarvis/` notes
(mtime + size, stat only); after the turn, index anything that changed. This
catches notes from any tool plus Daily-log appends, without touching the vault
writer, the tools, or the agent loop. It's an `Agent` subclass, so the text,
`--voice` and `--wake` modes all get it without any change of their own.
"""

import asyncio

from jarvis.core.agent import Agent, ConfirmFn
from jarvis.core.interfaces import LLMClient
from jarvis.logging import get_logger
from jarvis.memory.indexer import Snapshot, VaultIndexer
from jarvis.tools.base import ToolRegistry

log = get_logger(__name__)


class AutoIndexingAgent(Agent):
    """An `Agent` that indexes notes written during each turn, so they're searchable at once."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        indexer: VaultIndexer,
        max_iterations: int = 8,
        confirm: ConfirmFn | None = None,
    ) -> None:
        """Same arguments as `Agent`, plus the `indexer` to update after each turn."""
        super().__init__(llm, registry, max_iterations=max_iterations, confirm=confirm)
        self.indexer = indexer

    async def run(self, user_input: str) -> str:
        """Run the turn exactly as `Agent.run` does, then index any notes it changed.

        Indexing failures are logged and never fail the turn. Notes written
        before an LLM error are still indexed.
        """
        before = await asyncio.to_thread(self.indexer.snapshot)
        try:
            reply = await super().run(user_input)
        except Exception:
            await self._index_changes(before)
            raise
        await self._index_changes(before)
        return reply

    async def _index_changes(self, before: Snapshot) -> None:
        try:
            changed = await asyncio.to_thread(self.indexer.changed_since, before)
            if changed:
                await asyncio.to_thread(self.indexer.index_paths, changed)
        except Exception as exc:
            log.warning("memory.auto_index_failed", error=f"{type(exc).__name__}: {exc}")
