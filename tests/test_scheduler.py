"""Reminder scheduler: due reminders fire exactly once, via TTS / toast / the panel event."""

import asyncio
import json
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.core.agent import Agent
from jarvis.core.events import REMINDER, STATE, Event, EventBus
from jarvis.core.interfaces import AudioSamples, LLMResponse, Message, ToolSpec, WakeWordDetector
from jarvis.core.scheduler import ReminderScheduler, reminder_message
from jarvis.core.voice_loop import WakeWordLoop
from jarvis.server.controller import PanelController, PushToTalkVoice
from jarvis.tools._reminders import ReminderStore
from jarvis.tools.base import ToolRegistry
from jarvis.tools.time_tools import ListRemindersTool
from tests.fakes import FakeFrameSource, FakeLLMClient, FakeSTTEngine, FakeTTSEngine, FakeVAD

TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 26, 17, 0, tzinfo=TZ)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Delivered:
    """Records what the scheduler said and showed."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    def toast(self, title: str, message: str) -> bool:
        self.toasts.append((title, message))
        return True


def _drain(queue: asyncio.Queue[Event]) -> list[Event]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _setup(
    tmp_path: Path, *, notify: str = "tts"
) -> tuple[ReminderStore, Clock, Delivered, EventBus, ReminderScheduler]:
    store = ReminderStore(tmp_path / "reminders.json")
    clock, out, bus = Clock(), Delivered(), EventBus()
    scheduler = ReminderScheduler(
        store,
        notify=notify,  # type: ignore[arg-type]
        speak=out.speak,
        toast=out.toast,
        bus=bus,
        clock=clock,
        poll_seconds=0.01,
    )
    return store, clock, out, bus, scheduler


def test_a_due_reminder_fires_exactly_once_and_the_flag_persists(tmp_path: Path) -> None:
    store, clock, out, bus, scheduler = _setup(tmp_path)
    queue = bus.subscribe()
    store.add("call mom", NOW + timedelta(minutes=5), now=NOW)
    store.add("water the plants", NOW + timedelta(hours=2), now=NOW)

    clock.now = NOW + timedelta(minutes=1)
    assert asyncio.run(scheduler.check_once()) == []  # nothing due yet
    assert out.spoken == []

    clock.now = NOW + timedelta(minutes=5, seconds=10)
    fired = asyncio.run(scheduler.check_once())

    assert [r.text for r in fired] == ["call mom"]
    assert out.spoken == ["Reminder: call mom"]
    [event] = _drain(queue)
    assert event.type == REMINDER
    assert event.data["text"] == "call mom"
    assert event.data["message"] == "Reminder: call mom"
    assert event.data["late_seconds"] == 10

    assert asyncio.run(scheduler.check_once()) == []  # never twice
    assert out.spoken == ["Reminder: call mom"]

    # A restart (fresh store + scheduler) sees the persisted flag.
    _, _, out2, _, restarted = _setup(tmp_path)
    restarted._clock = clock  # type: ignore[attr-defined]
    assert asyncio.run(restarted.check_once()) == []
    assert out2.spoken == []
    by_text = {r.text: r for r in ReminderStore(tmp_path / "reminders.json").all()}
    assert by_text["call mom"].fired_at == clock.now
    assert by_text["water the plants"].fired_at is None


def test_not_yet_due_reminders_stay_pending(tmp_path: Path) -> None:
    store, clock, _out, _bus, scheduler = _setup(tmp_path)
    store.add("later", NOW + timedelta(hours=1), now=NOW)
    clock.now = NOW + timedelta(minutes=59, seconds=59)
    assert asyncio.run(scheduler.check_once()) == []
    assert store.all()[0].fired_at is None


def test_a_missed_reminder_says_when_it_was_due(tmp_path: Path) -> None:
    store, _clock, out, _bus, scheduler = _setup(tmp_path)
    store.add("stand up", NOW - timedelta(hours=1), now=NOW - timedelta(hours=2))
    asyncio.run(scheduler.check_once())
    assert out.spoken == ["Reminder: stand up (it was due at 16:00)"]
    yesterday = store.all()[0]
    assert reminder_message(yesterday, NOW + timedelta(days=1)).endswith("16:00 on Sat 26 Sep)")


@pytest.mark.parametrize(
    ("notify", "spoken", "toasted"),
    [("tts", True, False), ("toast", False, True), ("both", True, True)],
)
def test_notify_modes(tmp_path: Path, notify: str, spoken: bool, toasted: bool) -> None:
    store, _clock, out, bus, scheduler = _setup(tmp_path, notify=notify)
    queue = bus.subscribe()
    store.add("stretch", NOW, now=NOW - timedelta(minutes=1))

    asyncio.run(scheduler.check_once())

    assert (out.spoken == ["Reminder: stretch"]) is spoken
    assert (out.toasts == [("Jarvis reminder", "Reminder: stretch")]) is toasted
    assert [e.type for e in _drain(queue)] == [REMINDER]  # the panel always hears about it


def test_without_voice_the_reminder_is_still_shown(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    store.add("stretch", NOW, now=NOW - timedelta(minutes=1))
    bus, shown = EventBus(), []
    queue = bus.subscribe()
    scheduler = ReminderScheduler(store, speak=None, bus=bus, display=shown.append, clock=Clock())

    assert len(asyncio.run(scheduler.check_once())) == 1
    assert shown == ["⏰ Reminder: stretch"]
    assert [e.type for e in _drain(queue)] == [REMINDER]


def test_a_failing_delivery_is_contained_and_not_repeated(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    store.add("stretch", NOW, now=NOW - timedelta(minutes=1))

    async def broken_speaker(text: str) -> None:
        raise RuntimeError("speakers unplugged")

    def broken_toast(title: str, message: str) -> bool:
        raise RuntimeError("no notification service")

    scheduler = ReminderScheduler(
        store, notify="both", speak=broken_speaker, toast=broken_toast, clock=Clock()
    )
    assert len(asyncio.run(scheduler.check_once())) == 1  # no exception escaped
    assert asyncio.run(scheduler.check_once()) == []  # marked fired before delivery: no retry storm


def test_a_corrupt_file_is_reported_and_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    path.write_text("{not json", encoding="utf-8")
    scheduler = ReminderScheduler(ReminderStore(path), clock=Clock())
    assert asyncio.run(scheduler.check_once()) == []
    assert path.read_text(encoding="utf-8") == "{not json"


def test_idle_polls_do_not_reread_an_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock, _out, _bus, scheduler = _setup(tmp_path)
    store.add("later", NOW + timedelta(minutes=30), now=NOW)
    reads: list[datetime] = []
    real_claim = store.claim_due

    def counting_claim(now: datetime) -> list:  # type: ignore[type-arg]
        reads.append(now)
        return real_claim(now)

    monkeypatch.setattr(store, "claim_due", counting_claim)
    for minutes in (1, 2, 3):
        clock.now = NOW + timedelta(minutes=minutes)
        asyncio.run(scheduler.check_once())
    assert len(reads) == 1  # first poll read the file; the next two were a stat() each

    store.add("sooner", NOW + timedelta(minutes=4), now=NOW)  # the file changed
    clock.now = NOW + timedelta(minutes=5)
    fired = asyncio.run(scheduler.check_once())
    assert [r.text for r in fired] == ["sooner"]
    assert len(reads) == 2


def test_missing_file_is_cheap_and_quiet(tmp_path: Path) -> None:
    scheduler = ReminderScheduler(ReminderStore(tmp_path / "none.json"), clock=Clock())
    assert asyncio.run(scheduler.check_once()) == []


def test_old_files_without_the_fired_flag_still_load(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    path.write_text(
        json.dumps(
            {
                "reminders": [
                    {
                        "id": "a1",
                        "text": "old style",
                        "due": NOW.isoformat(),
                        "created": NOW.isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    [reminder] = ReminderStore(path).all()
    assert reminder.fired_at is None


def test_list_reminders_marks_delivered_ones(tmp_path: Path) -> None:
    store = ReminderStore(tmp_path / "reminders.json")
    store.add("done already", NOW - timedelta(minutes=1), now=NOW - timedelta(minutes=5))
    store.claim_due(NOW)
    listing = asyncio.run(ListRemindersTool(store, lambda: NOW).run())
    assert "(delivered): done already" in listing


def test_the_run_loop_polls_until_cancelled(tmp_path: Path) -> None:
    store, clock, out, _bus, scheduler = _setup(tmp_path)

    async def scenario() -> None:
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.03)
        store.add("tea", NOW + timedelta(seconds=30), now=NOW)
        clock.now = NOW + timedelta(minutes=1)
        for _ in range(100):
            if out.spoken:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert out.spoken == ["Reminder: tea"]


# --- speaking through the voice loop ----------------------------------------------


class CountingDetector(WakeWordDetector):
    """Never fires; counts frames and resets."""

    def __init__(self) -> None:
        self.frames = 0
        self.resets = 0

    @property
    def frame_samples(self) -> int:
        return 1280

    def process(self, frame: AudioSamples) -> float:
        self.frames += 1
        return 0.0

    def reset(self) -> None:
        self.resets += 1


class SlowSource(FakeFrameSource):
    def read(self, n_samples: int) -> AudioSamples:
        time.sleep(0.002)
        return super().read(n_samples)


class WatchingTTS(FakeTTSEngine):
    """Records how many frames the detector processed while it was speaking."""

    def __init__(self, detector: CountingDetector) -> None:
        super().__init__()
        self.detector = detector
        self.frames_during_speech: list[int] = []

    def speak(self, text: str) -> None:
        before = self.detector.frames
        time.sleep(0.05)
        super().speak(text)
        self.frames_during_speech.append(self.detector.frames - before)


def test_loop_announce_mutes_wake_detection_and_resumes_cleanly() -> None:
    async def scenario() -> tuple[WatchingTTS, CountingDetector, list[str]]:
        detector = CountingDetector()
        tts = WatchingTTS(detector)
        bus = EventBus()
        queue = bus.subscribe()
        loop = WakeWordLoop(
            Agent(FakeLLMClient([]), ToolRegistry()),
            FakeSTTEngine([]),
            tts,
            detector,
            FakeVAD(),
            SlowSource(),
            display=lambda _line: None,
            events=bus,
        )
        task = asyncio.create_task(loop.run())
        for _ in range(500):  # until it is listening for the wake word
            if detector.frames >= 3:
                break
            await asyncio.sleep(0.005)

        await loop.announce("Reminder: stretch")

        for _ in range(200):  # the listening thread resets the detector as it resumes
            if detector.resets:
                break
            await asyncio.sleep(0.005)
        frames_after = detector.frames
        for _ in range(200):
            if detector.frames > frames_after:
                break
            await asyncio.sleep(0.005)
        loop.stop()
        await asyncio.gather(task, return_exceptions=True)
        states = [e.data["state"] for e in _drain(queue) if e.type == STATE]
        return tts, detector, states

    tts, detector, states = asyncio.run(scenario())
    assert tts.spoken == ["Reminder: stretch"]
    assert tts.frames_during_speech == [0]  # it didn't listen to itself
    assert detector.resets == 1
    assert states[-2:] == ["speaking", "idle"]


def test_loop_announce_waits_for_the_turn_in_progress() -> None:
    class GatedLLM(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__([])
            self.gate = asyncio.Event()

        async def complete(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
        ) -> LLMResponse:
            await self.gate.wait()
            return LLMResponse(text="Here's your answer.")

    class OnceDetector(CountingDetector):
        def process(self, frame: AudioSamples) -> float:
            super().process(frame)
            return 0.9 if self.frames == 1 else 0.0

    async def scenario() -> list[str]:
        llm = GatedLLM()
        tts = FakeTTSEngine()
        loop = WakeWordLoop(
            Agent(llm, ToolRegistry()),
            FakeSTTEngine(["what's up"]),
            tts,
            OnceDetector(),
            FakeVAD(),
            SlowSource(),
            display=lambda _line: None,
        )
        task = asyncio.create_task(loop.run())
        for _ in range(500):  # until the turn is waiting on the LLM
            if loop.state is not None and loop.state.value == "processing":
                break
            await asyncio.sleep(0.005)
        announcing = asyncio.create_task(loop.announce("Reminder: stretch"))
        await asyncio.sleep(0.05)
        assert tts.spoken == []  # waits: a turn is in progress
        llm.gate.set()
        await announcing
        loop.stop()
        await asyncio.gather(task, return_exceptions=True)
        return tts.spoken

    assert asyncio.run(scenario()) == ["Here's your answer.", "Reminder: stretch"]


def test_push_to_talk_announce_speaks_and_moves_the_orb() -> None:
    async def scenario() -> tuple[list[str], list[str]]:
        bus = EventBus()
        queue = bus.subscribe()
        spoken: list[str] = []

        async def speak(text: str) -> None:
            spoken.append(text)

        voice = PushToTalkVoice(lambda d, s: None, SlowSource(), bus, speak=speak)  # type: ignore[arg-type,return-value]
        await voice.announce("Reminder: **stretch**")
        return spoken, [e.data["state"] for e in _drain(queue) if e.type == STATE]

    spoken, states = asyncio.run(scenario())
    assert spoken == ["Reminder: stretch"]  # Markdown symbols stripped, as for replies
    assert states == ["speaking", "idle"]


def test_panel_runs_the_scheduler_for_its_lifetime(tmp_path: Path) -> None:
    async def scenario() -> list[Event]:
        store, _clock, _out, bus, scheduler = _setup(tmp_path)
        queue = bus.subscribe()
        store.add("stretch", NOW, now=NOW - timedelta(minutes=1))
        controller = PanelController(
            Agent(FakeLLMClient([]), ToolRegistry()), bus, scheduler=scheduler
        )
        await controller.start()
        for _ in range(100):
            if not queue.empty():
                break
            await asyncio.sleep(0.01)
        await controller.aclose()
        return _drain(queue)

    events = asyncio.run(scenario())
    assert [e.type for e in events] == [REMINDER]
