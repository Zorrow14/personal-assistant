"""EventBus and LevelMeter tests: pure asyncio, no audio, no network."""

import asyncio
import json
import math
import time
from pathlib import Path

from jarvis.core.events import (
    LEVEL,
    LEVEL_MIC,
    REPLY,
    STATE,
    Event,
    EventBus,
    LevelMeter,
    offer,
)


def _drain(queue: asyncio.Queue[Event]) -> list[Event]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_published_events_reach_every_subscriber() -> None:
    async def scenario() -> None:
        bus = EventBus()
        first, second = bus.subscribe(), bus.subscribe()

        bus.emit(STATE, state="listening")

        for queue in (first, second):
            event = await asyncio.wait_for(queue.get(), 1)
            assert (event.type, event.data) == (STATE, {"state": "listening"})

    asyncio.run(scenario())


def test_unsubscribe_stops_delivery() -> None:
    async def scenario() -> None:
        bus = EventBus()
        kept, dropped = bus.subscribe(), bus.subscribe()
        bus.unsubscribe(dropped)
        bus.unsubscribe(dropped)  # twice is harmless

        bus.emit(REPLY, text="hi")

        assert [e.data["text"] for e in _drain(kept)] == ["hi"]
        assert _drain(dropped) == []

    asyncio.run(scenario())


def test_full_queue_never_blocks_and_keeps_the_newest() -> None:
    async def scenario() -> None:
        bus = EventBus(queue_size=3)
        slow = bus.subscribe()  # never read while publishing

        started = time.perf_counter()
        for i in range(1000):
            bus.emit(STATE, n=i)
        assert time.perf_counter() - started < 1.0

        assert [e.data["n"] for e in _drain(slow)] == [997, 998, 999]
        assert bus.dropped == 997

    asyncio.run(scenario())


def test_slow_subscriber_does_not_starve_others() -> None:
    async def scenario() -> None:
        bus = EventBus(queue_size=2)
        slow, fast = bus.subscribe(), bus.subscribe()
        received = []
        for i in range(5):
            bus.emit(STATE, n=i)
            received.append((await fast.get()).data["n"])
        assert received == [0, 1, 2, 3, 4]
        assert [e.data["n"] for e in _drain(slow)] == [3, 4]

    asyncio.run(scenario())


def test_publish_from_a_worker_thread_is_delivered_on_the_loop() -> None:
    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()  # binds the bus to this loop

        await asyncio.to_thread(bus.emit, LEVEL, rms=0.1, source=LEVEL_MIC)

        event = await asyncio.wait_for(queue.get(), 1)
        assert event.data == {"rms": 0.1, "source": LEVEL_MIC}

    asyncio.run(scenario())


def test_publish_without_subscribers_is_a_no_op() -> None:
    bus = EventBus()
    bus.emit(STATE, state="idle")  # no loop, no subscribers: nothing happens
    assert not bus.has_subscribers


def test_publish_never_raises() -> None:
    bus = EventBus()
    bus.subscribe()

    def explode(event: Event) -> None:
        raise RuntimeError("subscriber bug")

    bus._deliver = explode  # type: ignore[method-assign]
    bus.emit(STATE, state="idle")  # swallowed and logged


def test_publish_after_the_loop_closed_is_ignored() -> None:
    bus = EventBus()

    async def subscribe() -> None:
        bus.subscribe()

    asyncio.run(subscribe())  # the bus's loop is now closed
    bus.emit(STATE, state="idle")  # from "another thread": must not raise


def test_event_json_shape_and_unserialisable_values() -> None:
    event = Event(REPLY, {"text": "héllo", "path": Path("notes/a.md")}, ts=12.5)

    decoded = json.loads(event.to_json())

    assert decoded == {
        "type": REPLY,
        "data": {"text": "héllo", "path": str(Path("notes/a.md"))},
        "ts": 12.5,
    }


def test_offer_drops_the_oldest_when_full() -> None:
    queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1)
    assert offer(queue, Event(STATE, {"n": 1})) is False
    assert offer(queue, Event(STATE, {"n": 2})) is True
    assert queue.get_nowait().data == {"n": 2}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _level_events(bus: EventBus, queue: asyncio.Queue[Event]) -> list[float]:
    return [e.data["rms"] for e in _drain(queue) if e.type == LEVEL]


def test_level_meter_throttles_to_max_rate() -> None:
    bus, clock = EventBus(), FakeClock()
    queue = bus.subscribe()
    meter = LevelMeter(bus, LEVEL_MIC, max_per_second=16, clock=clock)

    for i in range(64):  # 64 blocks over one second (binary fractions keep the maths exact)
        clock.now = i / 64
        meter([0.5] * 250)

    levels = _level_events(bus, queue)
    assert len(levels) == 16
    assert all(math.isclose(rms, 0.5) for rms in levels)


def test_level_meter_reports_rms_over_the_whole_window() -> None:
    bus, clock = EventBus(), FakeClock()
    queue = bus.subscribe()
    meter = LevelMeter(bus, LEVEL_MIC, max_per_second=20, clock=clock)

    meter([0.0] * 100)  # t=0: published immediately (silence)
    clock.now = 0.02
    meter([1.0] * 100)  # inside the window: accumulated
    clock.now = 0.06
    meter([0.0] * 100)  # window over: RMS of the last two blocks

    assert _level_events(bus, queue) == [0.0, round(math.sqrt(0.5), 4)]


def test_level_meter_idles_without_subscribers_and_never_raises() -> None:
    bus = EventBus()
    meter = LevelMeter(bus, LEVEL_MIC)
    meter([0.5] * 10)  # nobody listening: nothing computed
    queue = bus.subscribe()
    meter("not audio")  # garbage is ignored, not raised
    meter([])
    assert _drain(queue) == []
