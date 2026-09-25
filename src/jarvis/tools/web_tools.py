"""Web search via DuckDuckGo's free, keyless search (the `ddgs` library).

The only module that knows about ddgs. ddgs is synchronous and does its own
HTTP, so it runs on a worker thread. Read-only: no confirmation needed.
"""

import asyncio
from collections.abc import Callable
from typing import Any, Self

from pydantic import BaseModel, Field

from jarvis.logging import get_logger
from jarvis.tools.base import Tool
from jarvis.tools.context import ToolContext

SNIPPET_CHARS = 300
SEARCH_TIMEOUT_SECONDS = 10

SearchFn = Callable[[str, int], list[dict[str, Any]]]
"""(query, max_results) -> [{"title", "href", "body"}, ...]; [] when nothing is found."""

log = get_logger(__name__)


class WebSearchError(Exception):
    """The search backend failed (network, rate limit, timeout...)."""


def ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search with ddgs. Blocking. Returns [] for "no results".

    Raises:
        WebSearchError: On any backend failure.
    """
    from ddgs import DDGS  # imported lazily: only needed when a search runs
    from ddgs.exceptions import DDGSException

    try:
        return list(DDGS(timeout=SEARCH_TIMEOUT_SECONDS).text(query, max_results=max_results))
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # primp/network errors surface as assorted types
        raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc


class WebSearchArgs(BaseModel):
    """Input for `web_search`."""

    query: str = Field(description="What to search the web for.")
    max_results: int | None = Field(
        default=None, ge=1, le=20, description="Optional number of results (default 5)."
    )


class WebSearchTool(Tool):
    """Search the public web and return titles, snippets and URLs."""

    name = "web_search"
    description = (
        "Search the public web (DuckDuckGo) for current information, facts, or news. "
        "Returns titles, snippets and URLs. Cite the URLs you rely on; results may be "
        "incomplete or out of date."
    )
    args_model = WebSearchArgs
    requires_confirmation = False
    category = "web"

    def __init__(self, *, default_max_results: int = 5, search_fn: SearchFn = ddgs_search) -> None:
        """
        Args:
            default_max_results: Results returned when the LLM doesn't specify.
            search_fn: The search backend (injectable for tests).
        """
        self._default_max_results = default_max_results
        self._search = search_fn

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with `search_max_results` from settings."""
        return cls(default_max_results=context.settings.search_max_results)

    async def run(self, **kwargs: Any) -> str:
        """Search and format results; failures become a clear message, never an exception."""
        args = WebSearchArgs.model_validate(kwargs)
        limit = args.max_results or self._default_max_results
        try:
            results = await asyncio.to_thread(self._search, args.query, limit)
        except Exception as exc:
            log.warning("web_search.failed", query=args.query, error=str(exc))
            return (
                f"Web search failed ({exc}). Tell the user the search didn't work "
                "(maybe no internet connection or a rate limit); do not guess results."
            )
        log.info("web_search.done", query=args.query, results=len(results))
        return format_results(args.query, results[:limit])


def format_results(query: str, results: list[dict[str, Any]]) -> str:
    """Numbered title / snippet / URL list for the LLM."""
    if not results:
        return f"No web results found for {query!r}."
    lines = [f"Web results for {query!r}:"]
    for i, item in enumerate(results, start=1):
        title = _clean(item.get("title")) or "(untitled)"
        snippet = _clean(item.get("body"))
        url = _clean(item.get("href") or item.get("url"))
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {_truncate(snippet)}")
        if url:
            lines.append(f"   {url}")
    return "\n".join(lines)


def _clean(value: object) -> str:
    return " ".join(str(value).split()) if value else ""


def _truncate(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"
