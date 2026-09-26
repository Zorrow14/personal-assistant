"""Event bus: the seam a UI hangs on to watch Jarvis work.

The assistant publishes small, JSON-friendly events while it works: state
changes, transcripts, replies, tool calls and audio levels. Each subscriber
(e.g. one open panel tab) gets its own bounded queue.

Publishing is fire-and-forget. It never blocks, never raises, and is safe to
call from worker and audio threads. A slow subscriber only loses its own oldest
events, so it can never stall the voice loop.
"""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.logging import get_logger

# --- Event vocabulary --------------------------------------------------------

STATE = "state"
"""{state}: one of the STATE_* values below."""
TRANSCRIPT = "transcript"
"""{text, source}: what the user said ("voice") or typed ("text"). Empty text = nothing heard."""
REPLY = "reply"
"""{text}: Jarvis's answer."""
TOOL = "tool"
"""{id, name, category, args, status, result?}: status is one of the TOOL_* values below."""
LEVEL = "level"
"""{rms, source}: audio amplitude for the orb, throttled; source is "mic" or "speaker"."""
ERROR = "error"
"""{message}: a turn or the voice loop failed."""
NOTICE = "notice"
"""{message}: informational, e.g. "busy" or how to quit."""
HELLO = "hello"
"""Sent once to each new panel connection: current state and what the panel can do."""
METRICS = "metrics"
"""{turn, session}: one finished turn's stage latencies and token counts, plus session totals."""
REMINDER = "reminder"
"""{id, text, due, late_seconds, message}: a reminder just came due."""

STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"

TOOL_STARTED = "started"
TOOL_COMPLETED = "completed"
TOOL_DECLINED = "declined"
TOOL_ERROR = "error"

LEVEL_MIC = "mic"
LEVEL_SPEAKER = "speaker"

DEFAULT_QUEUE_SIZE = 256
"""Events buffered per subscriber before the oldest are dropped (about 12 s of level events)."""

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened, e.g. `Event("state", {"state": "listening"})`."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    """Unix time in seconds."""

    def to_json(self) -> str:
        """Serialize as `{"type", "data", "ts"}`; values JSON can't hold become strings."""
        return json.dumps(
            {"type": self.type, "data": self.data, "ts": self.ts}, ensure_ascii=False, default=str
        )


def offer(queue: asyncio.Queue[Event], event: Event) -> bool:
    """Put `event` on `queue` without blocking, dropping the oldest event if it is full.

    Must be called on the thread running the queue's event loop.

    Returns:
        True if an older event had to be dropped to make room.
    """
    dropped = False
    if queue.full():
        try:
            queue.get_nowait()
            dropped = True
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)
    return dropped


class EventBus:
    """Async fan-out pub/sub with one bounded queue per subscriber."""

    def __init__(self, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        """
        Args:
            queue_size: Per-subscriber buffer; beyond it the oldest events are dropped.
        """
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.dropped = 0
        """Events discarded because a subscriber fell behind."""

    @property
    def has_subscribers(self) -> bool:
        """Whether anyone is listening; publishers may skip expensive work when not."""
        return bool(self._subscribers)

    def attach(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Bind to the event loop subscribers live on (default: the running one).

        Events published from other threads are handed over to this loop.
        `subscribe()` binds automatically when called inside a running loop.
        """
        self._loop = loop or asyncio.get_running_loop()

    def subscribe(self) -> asyncio.Queue[Event]:
        """Start receiving every published event on a new queue."""
        if self._loop is None:
            # Outside a loop (e.g. a synchronous test) events are delivered directly.
            self._loop = _running_loop()
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Stop delivering to `queue`. Unknown queues are ignored."""
        self._subscribers.discard(queue)

    def publish(self, event: Event) -> None:
        """Deliver `event` to every subscriber. Never blocks or raises; any thread."""
        try:
            if not self._subscribers:
                return
            loop = self._loop
            if loop is None or _running_loop() is loop:
                self._deliver(event)
            elif not loop.is_closed():
                loop.call_soon_threadsafe(self._deliver, event)
        except Exception as exc:  # an observer must never break the assistant
            log.debug("events.publish_failed", event_type=event.type, error=repr(exc))

    def emit(self, type_: str, **data: Any) -> None:
        """Shorthand for `publish(Event(type_, data))`."""
        self.publish(Event(type_, data))

    def _deliver(self, event: Event) -> None:
        for queue in tuple(self._subscribers):
            if offer(queue, event):
                self.dropped += 1


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class LevelMeter:
    """Turns raw audio blocks into throttled `level` events for the orb.

    Call it with every block of samples (from any single thread, e.g. the audio
    callback). It publishes the RMS of everything seen since its last event, at
    most `max_per_second` times a second, and does nothing while no one is
    subscribed. It never raises, so it is safe inside an audio callback.
    """

    def __init__(
        self,
        bus: EventBus,
        source: str,
        *,
        max_per_second: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            bus: Where `level` events go.
            source: LEVEL_MIC or LEVEL_SPEAKER.
            max_per_second: Throttle for published events.
            clock: Monotonic seconds (injectable for tests).
        """
        self._bus = bus
        self._source = source
        self._interval = 1.0 / max_per_second
        self._clock = clock
        self._sum_squares = 0.0
        self._count = 0
        self._last = -math.inf

    def __call__(self, samples: Any) -> None:
        """Account for one block of float samples in [-1, 1]."""
        try:
            if not self._bus.has_subscribers:
                return
            block = np.asarray(samples, dtype=np.float64).ravel()
            if block.size == 0:
                return
            self._sum_squares += float(np.dot(block, block))
            self._count += block.size
            now = self._clock()
            if now - self._last < self._interval:
                return
            rms = math.sqrt(self._sum_squares / self._count)
            self._sum_squares, self._count, self._last = 0.0, 0, now
            self._bus.emit(LEVEL, rms=round(rms, 4), source=self._source)
        except Exception:
            pass  # a meter must never disturb audio capture or playback
