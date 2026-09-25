"""WebSearchTool with a mocked backend: no network."""

import asyncio
from typing import Any

import pytest

from jarvis.tools import web_tools
from jarvis.tools.web_tools import WebSearchError, WebSearchTool, ddgs_search

RESULTS = [
    {"title": "Hey Jarvis model", "body": "openWakeWord ships  a pretrained\nhey_jarvis model.", "href": "https://example.com/a"},
    {"title": "Second", "body": "x" * 500, "href": "https://example.com/b"},
    {"title": "", "body": "", "href": "https://example.com/c"},
]


class RecordingSearch:
    def __init__(self, results: list[dict[str, Any]] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, max_results: int) -> list[dict[str, Any]]:
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results


def test_formats_numbered_results() -> None:
    search = RecordingSearch(RESULTS)
    text = asyncio.run(WebSearchTool(search_fn=search).run(query="hey jarvis"))

    lines = text.splitlines()
    assert lines[0] == "Web results for 'hey jarvis':"
    assert lines[1] == "1. Hey Jarvis model"
    assert lines[2] == "   openWakeWord ships a pretrained hey_jarvis model."
    assert lines[3] == "   https://example.com/a"
    assert "2. Second" in text and text.count("x") < 320  # long snippets are cut
    assert "3. (untitled)" in text
    assert search.calls == [("hey jarvis", 5)]


def test_max_results_default_and_override() -> None:
    search = RecordingSearch(RESULTS)
    tool = WebSearchTool(default_max_results=2, search_fn=search)

    text = asyncio.run(tool.run(query="q"))
    asyncio.run(tool.run(query="q", max_results=7))

    assert search.calls == [("q", 2), ("q", 7)]
    assert "3." not in text  # capped even if the backend returns more


def test_no_results_message() -> None:
    text = asyncio.run(WebSearchTool(search_fn=RecordingSearch([])).run(query="zzqqxx"))
    assert text == "No web results found for 'zzqqxx'."


def test_network_error_is_a_clear_message_not_a_crash() -> None:
    search = RecordingSearch(error=WebSearchError("ConnectError: no route to host"))
    text = asyncio.run(WebSearchTool(search_fn=search).run(query="q"))
    assert text.startswith("Web search failed (ConnectError: no route to host)")
    assert "do not guess" in text


class FakeDDGS:
    """Stands in for ddgs.DDGS so the real adapter code runs offline."""

    behaviour: Any = None

    def __init__(self, timeout: int | None = None) -> None:
        self.timeout = timeout

    def text(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if isinstance(FakeDDGS.behaviour, Exception):
            raise FakeDDGS.behaviour
        return FakeDDGS.behaviour


@pytest.fixture
def fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> type[FakeDDGS]:
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    return FakeDDGS


def test_ddgs_adapter_passes_results_through(fake_ddgs: type[FakeDDGS]) -> None:
    fake_ddgs.behaviour = RESULTS
    assert ddgs_search("q", 3) == RESULTS


def test_ddgs_no_results_exception_means_empty(fake_ddgs: type[FakeDDGS]) -> None:
    from ddgs.exceptions import DDGSException

    fake_ddgs.behaviour = DDGSException("No results found.")
    assert ddgs_search("q", 3) == []


@pytest.mark.parametrize("error_name", ["RatelimitException", "TimeoutException"])
def test_ddgs_failures_become_web_search_error(fake_ddgs: type[FakeDDGS], error_name: str) -> None:
    import ddgs.exceptions

    fake_ddgs.behaviour = getattr(ddgs.exceptions, error_name)("slow down")
    with pytest.raises(WebSearchError, match=error_name):
        ddgs_search("q", 3)


def test_unexpected_backend_errors_also_wrapped(fake_ddgs: type[FakeDDGS]) -> None:
    fake_ddgs.behaviour = OSError("network unreachable")
    with pytest.raises(WebSearchError, match="network unreachable"):
        web_tools.ddgs_search("q", 3)
