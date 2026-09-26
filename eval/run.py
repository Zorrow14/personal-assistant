"""A tiny regression eval: does the model pick the right tool for each case?

    uv run python eval/run.py --fake   # offline: a scripted LLM exercises the harness, agent and tools
    uv run python eval/run.py          # live: the LLM configured in .env (e.g. Gemini)

Each case in `eval/cases.json` runs through the real `Agent` and the real,
auto-discovered tools, on a throwaway vault in a temp folder, never your own.
Side-effect tools are auto-approved, since they can only touch that temp folder.
Memory search uses a small in-process index over a few seeded notes. In
`--fake` mode web search returns canned results, so it needs no network and no
API key, which makes it suitable for CI.

A case passes when the expected tool was called (or, for `expect_tool: null`,
when no tool was), the reply contains `expect_contains` if given, and nothing
failed. The exit code is 0 when the pass rate reaches `--min-score`.
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # for tests.fakes: the scripted LLM and in-memory index

from tests.fakes import FakeEmbedder, FakeLLMClient, FakeVectorStore  # noqa: E402

from jarvis.cli import build_tool_registry  # noqa: E402
from jarvis.config import Settings  # noqa: E402
from jarvis.core.agent import Agent  # noqa: E402
from jarvis.core.events import TOOL, EventBus  # noqa: E402
from jarvis.core.interfaces import LLMClient, LLMResponse, ToolCall  # noqa: E402
from jarvis.llm.factory import create_llm_client  # noqa: E402
from jarvis.logging import configure_logging  # noqa: E402
from jarvis.memory.factory import MemoryStack  # noqa: E402
from jarvis.memory.indexer import VaultIndexer  # noqa: E402
from jarvis.memory.vault import Vault  # noqa: E402
from jarvis.obs.metrics import MetricsRecorder, percentile  # noqa: E402
from jarvis.tools.base import ToolRegistry  # noqa: E402
from jarvis.tools.context import ToolContext  # noqa: E402
from jarvis.tools.web_tools import WebSearchTool  # noqa: E402

DEFAULT_CASES = Path(__file__).with_name("cases.json")

SEED_FILES = {
    "Jarvis/Tasks/2026-09-20-0900-finish-the-vault-module.md": (
        '---\ndate: 2026-09-20\ncommand: "Finish the vault module"\nstatus: completed\n'
        "tags: [coding]\ntools_used: [write_task_note]\n---\n\n# Finish the vault module\n\n"
        "Wrote the vault writer with a path guard that refuses writes outside Jarvis/.\n"
    ),
    "Jarvis/Tasks/2026-09-22-1400-buy-groceries.md": (
        '---\ndate: 2026-09-22\ncommand: "Buy groceries"\nstatus: completed\n'
        "tags: [errands]\ntools_used: []\n---\n\n# Buy groceries\n\nMilk, eggs and coffee.\n"
    ),
    "Projects/plan.md": "# Plan\n\nGoal: ship the eval harness this week.\n",
}

CANNED_WEB_RESULTS = [
    {
        "title": "dscripka/openWakeWord - GitHub",
        "href": "https://github.com/dscripka/openWakeWord",
        "body": "An open-source audio wake word (or phrase) detection framework.",
    }
]


@dataclass(frozen=True)
class Case:
    """One eval case (see cases.json)."""

    name: str
    input: str
    expect_tool: str | None
    expect_contains: str | None = None
    fake_args: dict[str, Any] = field(default_factory=dict)
    fake_reply: str | None = None


@dataclass
class Result:
    """How one case went."""

    case: Case
    calls: list[tuple[str, str]]
    """(tool name, final status) for every tool call, in order."""
    reply: str
    error: str | None
    seconds: float
    llm_requests: int
    input_tokens: int
    output_tokens: int

    @property
    def called(self) -> list[str]:
        """Tool names as shown in the report; failed calls are marked."""
        return [
            name if status == "completed" else f"{name} ({status})" for name, status in self.calls
        ]

    @property
    def tool_ok(self) -> bool:
        """The expected tool ran successfully at least once (retries allowed)."""
        if self.case.expect_tool is None:
            return not self.calls
        return (self.case.expect_tool, "completed") in self.calls

    @property
    def text_ok(self) -> bool:
        expected = self.case.expect_contains
        return expected is None or expected.lower() in self.reply.lower()

    @property
    def passed(self) -> bool:
        return self.error is None and self.tool_ok and self.text_ok


def load_cases(path: Path) -> list[Case]:
    """Read cases from JSON: a list, or an object with a "cases" list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["cases"] if isinstance(data, dict) else data
    return [
        Case(
            name=item.get("name") or item["input"],
            input=item["input"],
            expect_tool=item.get("expect_tool"),
            expect_contains=item.get("expect_contains"),
            fake_args=dict(item.get("fake_args") or {}),
            fake_reply=item.get("fake_reply"),
        )
        for item in items
    ]


def scripted_llm(case: Case) -> FakeLLMClient:
    """--fake: call the expected tool (if any), then reply, like an ideal model would."""
    reply = case.fake_reply or (
        f"Done: {case.expect_contains}." if case.expect_contains else "Done."
    )
    script: list[LLMResponse] = []
    if case.expect_tool is not None:
        call = ToolCall("eval-1", case.expect_tool, dict(case.fake_args))
        script.append(LLMResponse(text=None, tool_calls=[call], stop_reason="tool_use"))
    script.append(LLMResponse(text=reply))
    return FakeLLMClient(script)


def build_environment(workdir: Path, *, fake: bool) -> tuple[Settings, ToolRegistry]:
    """Settings and tools over a seeded throwaway vault inside `workdir`."""
    vault_root = workdir / "vault"
    for relative, text in SEED_FILES.items():
        target = vault_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    overrides: dict[str, Any] = {
        "vault_path": vault_root,
        "file_sandbox_root": vault_root,
        "reminders_path": workdir / "reminders.json",
        "chroma_path": workdir / "state" / "chroma",
        "metrics_path": workdir / "metrics.jsonl",
    }
    # --fake ignores .env entirely (no API key needed); live mode takes the LLM config from it.
    settings = Settings(_env_file=None, **overrides) if fake else Settings(**overrides)  # type: ignore[call-arg]

    vault = Vault(vault_root)
    embedder, store = FakeEmbedder(), FakeVectorStore()
    indexer = VaultIndexer(vault.jarvis_root, embedder, store, workdir / "state" / "manifest.json")
    indexer.reindex_all()
    context = ToolContext(settings, vault, MemoryStack(embedder, store, indexer))
    registry = build_tool_registry(settings, context)
    if fake and "web_search" in registry:
        offline = ToolRegistry()
        for name in registry.names():
            tool = registry.get(name)
            assert tool is not None
            if name == "web_search":
                tool = WebSearchTool(search_fn=lambda query, n: CANNED_WEB_RESULTS[:n])
            offline.register(tool)
        registry = offline
    return settings, registry


async def run_case(
    case: Case, llm: LLMClient, registry: ToolRegistry, settings: Settings
) -> Result:
    """Run one case through a fresh agent (no history carried over)."""
    bus = EventBus()
    events = bus.subscribe()
    agent = Agent(
        llm,
        registry,
        max_iterations=settings.agent_max_iterations,
        confirm=lambda call: True,  # everything a tool can touch here is in the temp folder
        events=bus,
    )
    recorder = MetricsRecorder(settings.metrics_path, enabled=False)
    reply, error = "", None
    started = time.perf_counter()
    with recorder.turn("eval") as record:
        try:
            reply = await agent.run(case.input)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    calls = []
    while not events.empty():
        event = events.get_nowait()
        if event.type == TOOL and event.data["status"] != "started":  # one final status per call
            calls.append((event.data["name"], event.data["status"]))
    return Result(
        case=case,
        calls=calls,
        reply=reply,
        error=error,
        seconds=time.perf_counter() - started,
        llm_requests=record.llm_requests,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
    )


async def run_all(
    cases: Sequence[Case], *, fake: bool, delay: float, workdir: Path
) -> tuple[list[Result], str]:
    """Run every case; returns the results and the model's name."""
    settings, registry = build_environment(workdir, fake=fake)
    live_llm = None if fake else create_llm_client(settings)
    results = []
    try:
        for i, case in enumerate(cases):
            if i and delay:
                await asyncio.sleep(delay)  # be gentle with free-tier rate limits
            llm = scripted_llm(case) if fake else live_llm
            assert llm is not None
            results.append(await run_case(case, llm, registry, settings))
            print_progress(results[-1], i + 1, len(cases))
    finally:
        if live_llm is not None:
            await live_llm.aclose()
    return results, "scripted (--fake)" if fake else settings.llm_model


def print_progress(result: Result, index: int, total: int) -> None:
    mark = "PASS" if result.passed else "FAIL"
    print(f"  [{index}/{total}] {mark}  {result.case.name}", flush=True)


def report(results: Sequence[Result], model: str) -> str:
    """The pass/fail table and the overall score."""
    rows = [("#", "case", "expected tool", "called", "reply", "result", "time")]
    for i, r in enumerate(results, start=1):
        text_check = "-" if r.case.expect_contains is None else ("ok" if r.text_ok else "MISSING")
        rows.append(
            (
                str(i),
                r.case.name,
                r.case.expect_tool or "(none)",
                ", ".join(r.called) or "(none)",
                text_check,
                "PASS" if r.passed else "FAIL",
                f"{r.seconds:.1f}s",
            )
        )
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows]
    failures = [r for r in results if not r.passed]
    for r in failures:
        why = r.error or (
            f"expected {r.case.expect_tool or 'no tool'}, called {', '.join(r.called) or 'none'}"
            if not r.tool_ok
            else f"reply lacks {r.case.expect_contains!r}: {r.reply[:120]!r}"
        )
        lines.append(f"  FAIL {r.case.name}: {why}")
    passed = len(results) - len(failures)
    times = sorted(r.seconds for r in results)
    lines.append("")
    lines.append(
        f"Score: {passed}/{len(results)} passed ({passed / max(1, len(results)):.0%})  "
        f"model: {model}"
    )
    lines.append(
        f"LLM: {sum(r.llm_requests for r in results)} requests, "
        f"{sum(r.input_tokens for r in results):,} input / "
        f"{sum(r.output_tokens for r in results):,} output tokens; "
        f"per case p50 {percentile(times, 0.5):.1f}s, p95 {percentile(times, 0.95):.1f}s"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the eval. Returns 0 if the pass rate reaches --min-score, else 1."""
    parser = argparse.ArgumentParser(description="Jarvis tool-selection regression eval.")
    parser.add_argument("--fake", action="store_true", help="scripted LLM, fully offline (CI)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="cases JSON file")
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="seconds between cases (default: 0 with --fake; 8 live, which keeps a ~2-request case under free-tier limits of about 15 requests/minute)",
    )
    parser.add_argument(
        "--min-score", type=float, default=1.0, help="pass rate (0-1) needed for exit code 0"
    )
    parser.add_argument("--verbose", action="store_true", help="show Jarvis's log lines")
    args = parser.parse_args(argv)

    configure_logging("INFO" if args.verbose else "WARNING")
    cases = load_cases(args.cases)
    delay = args.delay if args.delay is not None else (0.0 if args.fake else 8.0)
    mode = "fake (scripted, offline)" if args.fake else "live"
    print(f"Running {len(cases)} eval cases, mode: {mode}")
    with tempfile.TemporaryDirectory(prefix="jarvis-eval-") as tmp:
        results, model = asyncio.run(run_all(cases, fake=args.fake, delay=delay, workdir=Path(tmp)))
    print()
    print(report(results, model))
    score = sum(r.passed for r in results) / max(1, len(results))
    return 0 if score >= args.min_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
