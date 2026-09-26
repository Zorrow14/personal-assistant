"""Per-turn latency and LLM usage metrics, kept locally as JSON lines.

How it fits together:

- Whoever starts a turn (the voice loop, the panel, the text chat) opens
  `recorder.turn(source)`, which makes a `TurnRecord` current for that task.
- Code inside the turn wraps its stages in `timer("stt")`, `llm_call()` or
  `tool_call()`. Outside a turn these are no-ops, so the agent and the voice
  loop don't need a metrics object passed in.
- When the turn ends, one JSON line is appended to `metrics_path` and a
  `metrics` event is published for the panel.

Nothing here ever raises into a turn, and nothing leaves the machine.
"""

import asyncio
import contextvars
import json
import math
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from jarvis.core.events import METRICS, EventBus
from jarvis.core.interfaces import TokenUsage
from jarvis.logging import get_logger

STAGE_WAKE_TO_STT = "wake_to_stt"
"""Wake word heard -> recording handed to STT (the chime plus you speaking)."""
STAGE_STT = "stt"
STAGE_LLM = "llm_total"
"""All LLM requests in the turn, summed."""
STAGE_TOOLS = "tool_total"
"""All tool executions in the turn, summed (not the y/N wait)."""
STAGE_TTS = "tts"
STAGE_TOTAL = "total"
STAGE_ORDER = (STAGE_WAKE_TO_STT, STAGE_STT, STAGE_LLM, STAGE_TOOLS, STAGE_TTS, STAGE_TOTAL)
PROCESSING_STAGES = (STAGE_STT, STAGE_LLM, STAGE_TOOLS, STAGE_TTS)
"""Stages that are Jarvis's own latency (wake_to_stt is mostly the user talking)."""

OUTCOME_REPLIED = "replied"
OUTCOME_EMPTY = "empty"
OUTCOME_EXIT = "exit"
OUTCOME_STOPPED = "stopped"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"

log = get_logger(__name__)


@dataclass
class TurnRecord:
    """Everything measured during one turn."""

    source: str
    """Where the turn came from: "voice", "text" or "panel"."""
    model: str | None = None
    ts: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    stages_ms: dict[str, float] = field(default_factory=dict)
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    outcome: str = OUTCOME_REPLIED
    error: str | None = None

    def add_stage(self, stage: str, ms: float) -> None:
        """Add `ms` to `stage` (stages that run several times accumulate)."""
        self.stages_ms[stage] = self.stages_ms.get(stage, 0.0) + ms

    def add_usage(self, usage: TokenUsage | None) -> None:
        """Count one LLM request and its tokens."""
        self.llm_requests += 1
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.thinking_tokens += usage.thinking_tokens

    def to_dict(self) -> dict[str, Any]:
        """The JSON line for this turn (stages in pipeline order, rounded to 0.1 ms)."""
        order = {stage: i for i, stage in enumerate(STAGE_ORDER)}
        stages = sorted(self.stages_ms.items(), key=lambda kv: order.get(kv[0], len(order)))
        return {
            "ts": self.ts,
            "source": self.source,
            "model": self.model,
            "outcome": self.outcome,
            "error": self.error,
            "stages_ms": {stage: round(ms, 1) for stage, ms in stages},
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
        }


_current: contextvars.ContextVar[TurnRecord | None] = contextvars.ContextVar(
    "jarvis_metrics_turn", default=None
)


def current_turn() -> TurnRecord | None:
    """The turn being measured in this task, if any."""
    return _current.get()


class timer:
    """Time one stage into the current turn. A no-op outside a turn.

    Works as `with timer("stt"):` or `async with timer("stt"):`; the time is
    recorded even if the stage raises.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.elapsed_ms = 0.0
        self._record: TurnRecord | None = None
        self._started = 0.0

    def __enter__(self) -> Self:
        self._record = _current.get()
        self._started = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._started) * 1000
        if self._record is not None:
            self._record.add_stage(self.stage, self.elapsed_ms)
            self._finish(self._record, failed=exc is not None)

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    def _finish(self, record: TurnRecord, *, failed: bool) -> None:
        """Hook for subclasses that count as well as time."""


class llm_call(timer):
    """Time one LLM request into `llm_total` and count it, with `usage` if set."""

    def __init__(self) -> None:
        super().__init__(STAGE_LLM)
        self.usage: TokenUsage | None = None

    def _finish(self, record: TurnRecord, *, failed: bool) -> None:
        record.add_usage(self.usage)  # failed requests count too: they cost quota


class tool_call(timer):
    """Time one tool execution into `tool_total` and count it; set `error` on failure."""

    def __init__(self) -> None:
        super().__init__(STAGE_TOOLS)
        self.error = False

    def _finish(self, record: TurnRecord, *, failed: bool) -> None:
        record.tool_calls += 1
        if self.error or failed:
            record.tool_errors += 1


def set_outcome(outcome: str, error: BaseException | str | None = None) -> None:
    """Label how the current turn ended (no-op outside a turn)."""
    record = _current.get()
    if record is None:
        return
    record.outcome = outcome
    if error is not None:
        record.error = error if isinstance(error, str) else _describe(error)


class MetricsRecorder:
    """Opens measured turns and persists each one as a JSON line."""

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = True,
        bus: EventBus | None = None,
        model: str | None = None,
    ) -> None:
        """
        Args:
            path: The JSONL file (created on first write).
            enabled: False keeps measuring (for events and the session totals)
                but writes nothing to disk.
            bus: Where `metrics` events go after each turn.
            model: Recorded with every turn, to compare models over time.
        """
        self.path = Path(path)
        self.enabled = enabled
        self._bus = bus
        self._model = model
        self._lock = threading.Lock()
        self._warned = False
        self.session: dict[str, Any] = {
            "since": datetime.now().astimezone().isoformat(timespec="seconds"),
            "turns": 0,
            "failed": 0,
            "llm_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "tool_calls": 0,
        }
        """Totals since this process started."""
        self.last_turn: dict[str, Any] | None = None

    @contextmanager
    def turn(self, source: str) -> Iterator[TurnRecord]:
        """Measure one turn: everything timed inside lands in the yielded record."""
        record = TurnRecord(source=source, model=self._model)
        started = time.perf_counter()
        token = _current.set(record)
        try:
            yield record
        except BaseException as exc:
            if record.outcome != OUTCOME_FAILED:
                cancelled = isinstance(exc, asyncio.CancelledError | KeyboardInterrupt)
                record.outcome = OUTCOME_CANCELLED if cancelled else OUTCOME_FAILED
            record.error = record.error or _describe(exc)
            raise
        finally:
            with suppress(ValueError):  # exited in another context: nothing to restore
                _current.reset(token)
            record.add_stage(STAGE_TOTAL, (time.perf_counter() - started) * 1000)
            self._finish(record)

    def summary(self, *, last: int | None = None) -> dict[str, Any]:
        """Aggregates over the JSONL file, plus this process's session totals."""
        result = summarize_file(self.path, last=last)
        result["session"] = dict(self.session)
        return result

    def _finish(self, record: TurnRecord) -> None:
        try:
            data = record.to_dict()
            self.last_turn = data
            self._add_to_session(data)
            if self.enabled:
                self._append(data)
            if self._bus is not None:
                self._bus.emit(METRICS, turn=data, session=dict(self.session))
        except Exception as exc:  # metrics must never break a turn
            if not self._warned:
                self._warned = True
                log.warning("metrics.write_failed", path=str(self.path), error=_describe(exc))

    def _add_to_session(self, data: dict[str, Any]) -> None:
        session = self.session
        session["turns"] += 1
        session["failed"] += data["outcome"] == OUTCOME_FAILED
        for key in (
            "llm_requests",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "tool_calls",
        ):
            session[key] += data[key]

    def _append(self, data: dict[str, Any]) -> None:
        line = json.dumps(data, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)


# --- reading and aggregating -----------------------------------------------------


def read_records(path: Path, *, last: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Parse the JSONL file. Returns (records, number of unreadable lines skipped)."""
    path = Path(path)
    if not path.exists():
        return [], 0
    kept: deque[dict[str, Any]] = deque(maxlen=last) if last else deque()
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if isinstance(item, dict):
                kept.append(item)
            else:
                skipped += 1
    records = list(kept)
    return records, skipped


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..1) of already-sorted `values`."""
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(records: list[dict[str, Any]], *, skipped: int = 0) -> dict[str, Any]:
    """Aggregate turn records: counts, token totals and p50/p95 per stage."""
    stage_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for stage, ms in (record.get("stages_ms") or {}).items():
            if isinstance(ms, int | float):
                stage_values[stage].append(float(ms))

    def total(key: str) -> int:
        return sum(int(record.get(key) or 0) for record in records)

    order = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    stages = {
        stage: _stats(sorted(values))
        for stage, values in sorted(stage_values.items(), key=lambda kv: order.get(kv[0], 99))
    }
    turns = len(records)
    requests = total("llm_requests")
    return {
        "turns": turns,
        "skipped_lines": skipped,
        "first": records[0].get("ts") if records else None,
        "last": records[-1].get("ts") if records else None,
        "outcomes": dict(Counter(str(r.get("outcome", "?")) for r in records)),
        "models": dict(Counter(str(r.get("model") or "?") for r in records)),
        "llm_requests": requests,
        "llm_requests_per_turn": round(requests / turns, 2) if turns else 0.0,
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "thinking_tokens": total("thinking_tokens"),
        "tool_calls": total("tool_calls"),
        "tool_errors": total("tool_errors"),
        "stages_ms": stages,
        "dominant_stage": _dominant(stage_values),
    }


def summarize_file(path: Path, *, last: int | None = None) -> dict[str, Any]:
    """`summarize` over the JSONL file at `path` (missing file = zero turns)."""
    records, skipped = read_records(path, last=last)
    result = summarize(records, skipped=skipped)
    result["path"] = str(path)
    return result


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable report for `--metrics`."""
    lines = [f"Jarvis metrics: {summary.get('path', '')}"]
    turns = summary["turns"]
    if not turns:
        lines.append("  No turns recorded yet. Talk to Jarvis (any mode) and run this again.")
        return "\n".join(lines)
    outcomes = ", ".join(f"{n} {name}" for name, n in sorted(summary["outcomes"].items()))
    lines.append(f"  {turns} turns ({outcomes}), {summary['first']} .. {summary['last']}")
    models = ", ".join(f"{name} ({n})" for name, n in summary["models"].items())
    lines.append(f"  models: {models}")
    lines.append(
        f"  LLM: {summary['llm_requests']} requests ({summary['llm_requests_per_turn']}/turn), "
        f"{summary['input_tokens']:,} input / {summary['output_tokens']:,} output tokens"
        + (f" (+{summary['thinking_tokens']:,} thinking)" if summary["thinking_tokens"] else "")
    )
    lines.append(f"  tools: {summary['tool_calls']} calls, {summary['tool_errors']} errors")
    if summary.get("skipped_lines"):
        lines.append(f"  ({summary['skipped_lines']} unreadable lines skipped)")
    lines.append("")
    lines.append(f"  {'stage':<12}{'count':>7}{'p50 s':>9}{'p95 s':>9}{'mean s':>9}{'max s':>9}")
    for stage, stats in summary["stages_ms"].items():
        lines.append(
            f"  {stage:<12}{stats['count']:>7}"
            + "".join(f"{stats[key] / 1000:>9.2f}" for key in ("p50", "p95", "mean", "max"))
        )
    dominant = summary.get("dominant_stage")
    if dominant:
        lines.append("")
        lines.append(
            f"  Jarvis's own time is dominated by {dominant['stage']} "
            f"({dominant['share']:.0%} of stt+llm+tools+tts)."
        )
    return "\n".join(lines)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "p50": round(percentile(values, 0.50), 1),
        "p95": round(percentile(values, 0.95), 1),
        "mean": round(sum(values) / len(values), 1),
        "max": round(values[-1], 1),
    }


def _dominant(stage_values: dict[str, list[float]]) -> dict[str, Any] | None:
    sums = {stage: sum(stage_values.get(stage, [])) for stage in PROCESSING_STAGES}
    overall = sum(sums.values())
    if overall <= 0:
        return None
    stage = max(sums, key=lambda s: sums[s])
    return {"stage": stage, "share": round(sums[stage] / overall, 3)}


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 300 else text[:300] + "..."
