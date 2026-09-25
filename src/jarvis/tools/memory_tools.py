"""Tools that read Jarvis's memory (the indexed vault)."""

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from jarvis.core.interfaces import Embedder, SearchHit, VectorStore
from jarvis.tools.base import Tool

SNIPPET_CHARS = 220
EMPTY_INDEX_MESSAGE = (
    "Memory is empty: no notes have been indexed yet. "
    "(The user can build the index with `python -m jarvis.cli --reindex`.)"
)


class SearchMemoryArgs(BaseModel):
    """Input for `search_memory`."""

    query: str = Field(
        description="What to look for, in natural language, e.g. 'wake-word module' or 'groceries'."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Optional number of results (default 5)."
    )


class SearchMemoryTool(Tool):
    """Semantic search over the user's past task notes and daily logs."""

    name = "search_memory"
    description = (
        "Search the user's past task notes and daily logs (their Obsidian vault) by meaning. "
        "Use it whenever the user asks what they asked you, did, logged or planned before. "
        "Answer only from the returned notes, cite them as [[note_name]], and say so plainly "
        "if nothing relevant comes back. Never invent notes."
    )
    args_model = SearchMemoryArgs
    requires_confirmation = False

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        default_top_k: int = 5,
        today: Callable[[], date] = date.today,
    ) -> None:
        """
        Args:
            embedder: Embeds the query (must match the one used for indexing).
            store: The indexed vault.
            default_top_k: Results returned when the LLM doesn't specify.
            today: Current date (injectable), shown so relative dates like
                "last week" can be resolved against the notes' dates.
        """
        self._embedder = embedder
        self._store = store
        self._default_top_k = default_top_k
        self._today = today

    async def run(self, **kwargs: Any) -> str:
        """Embed the query, search, and format the hits for the LLM."""
        args = SearchMemoryArgs.model_validate(kwargs)
        if await asyncio.to_thread(self._store.count) == 0:
            return EMPTY_INDEX_MESSAGE
        vector = await asyncio.to_thread(self._embedder.embed_query, args.query)
        hits = await asyncio.to_thread(self._store.query, vector, args.top_k or self._default_top_k)
        return format_hits(args.query, hits, today=self._today())


def format_hits(query: str, hits: list[SearchHit], *, today: date) -> str:
    """Render hits as a compact, citable list, one entry per note (best chunk wins)."""
    if not hits:
        return f"No notes in memory matched {query!r}."
    best: dict[str, SearchHit] = {}
    for hit in hits:
        name = str(hit.metadata.get("note_name") or hit.id)
        if name not in best or hit.score > best[name].score:
            best[name] = hit
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)

    lines = [
        f"Today is {today.isoformat()}. {len(ranked)} note(s) matched {query!r}, most relevant "
        "first (relevance 0-1; low scores may be unrelated):"
    ]
    for i, hit in enumerate(ranked, start=1):
        meta = hit.metadata
        name = meta.get("note_name") or hit.id
        details = [str(meta["date"])] if meta.get("date") else []
        if meta.get("tags"):
            details.append(f"tags: {meta['tags']}")
        details.append(f"relevance {hit.score:.2f}")
        lines.append(f"{i}. [[{name}]] ({', '.join(details)})")
        lines.append(f"   {_snippet(hit.document)}")
    return "\n".join(lines)


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "…"
