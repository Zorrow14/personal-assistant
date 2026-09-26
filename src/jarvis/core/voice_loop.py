"""Hands-free operation: wake word -> record until silence -> agent -> spoken reply.

A small state machine around the existing pipeline. The agent is called exactly
as in text mode; STT, TTS, the wake detector, the VAD recorder and the mic are
all injected through the neutral interfaces, so this module never touches an
audio or model library.

    IDLE ──wake word──▶ LISTENING ──silence/cap──▶ PROCESSING ──text──▶ SPEAKING
     ▲                                                 │ empty             │
     └─────────────────────────────────────────────────┴───────────────────┘

A turn that fails anywhere after the wake word (recording, STT, the agent or
LLM, TTS) never takes the loop down: the error is logged and published,
Jarvis apologises aloud, and it goes back to IDLE.
"""

import asyncio
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from enum import Enum

from jarvis.core.agent import Agent
from jarvis.core.events import (
    ERROR,
    REPLY,
    STATE,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    TRANSCRIPT,
    EventBus,
)
from jarvis.core.interfaces import (
    AudioSamples,
    CommandRecorder,
    FrameSource,
    LLMError,
    STTEngine,
    TTSEngine,
    VoiceError,
    WakeWordDetector,
)
from jarvis.core.voice_session import is_exit_command, speakable
from jarvis.logging import get_logger
from jarvis.obs.metrics import (
    OUTCOME_EMPTY,
    OUTCOME_EXIT,
    OUTCOME_FAILED,
    OUTCOME_STOPPED,
    STAGE_STT,
    STAGE_TTS,
    STAGE_WAKE_TO_STT,
    MetricsRecorder,
    set_outcome,
    timer,
)

NOT_HEARD_MESSAGE = "(Didn't catch that. Say '{wake_phrase}' to try again.)"
MAX_CONSECUTIVE_FAILURES = 3
"""Failed turns in a row before the loop gives up instead of spinning on a persistent fault."""
RETRY_DELAY_SECONDS = 0.5
"""Pause after a failure while waiting for the wake word, doubled each time in a row."""

APOLOGY = "Sorry, something went wrong."
RATE_LIMITED_APOLOGY = "Sorry, I'm being rate-limited right now. Please try again in a minute."
UNAVAILABLE_APOLOGY = "Sorry, the language model isn't responding. Please try again in a moment."

log = get_logger(__name__)


class LoopState(Enum):
    """Where the wake-word loop is."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class TurnResult(Enum):
    """How one wake -> reply cycle ended."""

    REPLIED = "replied"
    EMPTY = "empty"
    EXIT = "exit"
    STOPPED = "stopped"
    FAILED = "failed"
    """Something broke after the wake word; it was reported and apologised for."""


class _Stopped(Exception):
    """Raised inside worker threads when the loop is shutting down."""


class _StoppableSource:
    """Wraps a FrameSource so blocked worker threads notice shutdown within a frame."""

    def __init__(self, source: FrameSource, stop: threading.Event) -> None:
        self._source = source
        self._stop = stop

    def read(self, n_samples: int) -> AudioSamples:
        if self._stop.is_set():
            raise _Stopped
        return self._source.read(n_samples)

    def clear(self) -> None:
        self._source.clear()


class WakeWordLoop:
    """Always-listening voice loop around an existing `Agent`."""

    def __init__(
        self,
        agent: Agent,
        stt: STTEngine,
        tts: TTSEngine,
        detector: WakeWordDetector,
        recorder: CommandRecorder,
        source: FrameSource,
        *,
        threshold: float = 0.5,
        chime: Callable[[], None] | None = None,
        wake_phrase: str = "Hey Jarvis",
        display: Callable[[str], None] = print,
        on_state: Callable[[LoopState], None] | None = None,
        events: EventBus | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """
        Args:
            agent: The same agent text mode uses.
            stt: Speech-to-text engine.
            tts: Text-to-speech engine.
            detector: Wake-word detector.
            recorder: Records one command, stopping when the speaker goes quiet.
            source: Live microphone audio.
            threshold: Wake score (0–1) that counts as a trigger.
            chime: Plays the wake cue (blocking); None for silent triggers.
            wake_phrase: Shown in the idle prompt.
            display: Where status lines, transcripts and replies are shown.
            on_state: Called on every state change.
            events: Where to publish state/transcript/reply/error events (e.g.
                for the web panel); None publishes nothing.
            metrics: Records each turn's stage latencies and token counts;
                None records nothing.
        """
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.detector = detector
        self.recorder = recorder
        self.threshold = threshold
        self._chime = chime
        self._wake_phrase = wake_phrase
        self._display = display
        self._on_state = on_state
        self._events = events
        self._metrics = metrics
        self._stop = threading.Event()
        self._source = _StoppableSource(source, self._stop)
        self.state: LoopState | None = None
        self.last_error: Exception | None = None
        """The most recent failure, re-raised if failures persist."""
        # Held for a whole turn and while announcing, so speech never overlaps.
        self._speech_lock = asyncio.Lock()
        # Set while Jarvis speaks out of turn (e.g. a reminder): wake detection ignores the mic.
        self._muted = threading.Event()

    async def run(self) -> None:
        """Cycle IDLE -> ... -> IDLE until a spoken "exit", `stop()`, or cancellation.

        A turn that fails after the wake word is reported and apologised for,
        and the loop keeps listening (see `run_once`). A failure while waiting
        for the wake word (e.g. the mic dropping out) is reported and retried
        after a short pause. Only `MAX_CONSECUTIVE_FAILURES` failures in a row,
        i.e. a persistent fault, end the loop, re-raising the last error.
        """
        self._stop.clear()
        failures = 0
        try:
            while not self._stop.is_set():
                try:
                    result = await self.run_once()
                except Exception as exc:
                    failures += 1
                    self.last_error = exc
                    self._report(exc)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise
                    await asyncio.sleep(RETRY_DELAY_SECONDS * 2 ** (failures - 1))
                    continue
                if result is TurnResult.FAILED:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES and self.last_error is not None:
                        raise self.last_error
                    continue
                failures = 0
                if result in (TurnResult.EXIT, TurnResult.STOPPED):
                    return
        finally:
            # Ensure worker threads blocked on the mic exit promptly (e.g. on Ctrl-C).
            self._stop.set()

    def stop(self) -> None:
        """Ask the loop to finish; blocked reads notice within one frame."""
        self._stop.set()

    async def run_once(self) -> TurnResult:
        """One full cycle: wait for the wake word, take a command, answer it.

        Anything that fails after the wake word is caught: it's logged,
        published as an `error` event and apologised for aloud, and the loop is
        back at IDLE. The result is then `TurnResult.FAILED`.
        """
        self._enter(LoopState.IDLE)
        # Detection is paused (the mic isn't read) while speaking, so anything
        # buffered since then, including Jarvis's own voice, is stale.
        self._source.clear()
        try:
            await asyncio.to_thread(self._wait_for_wake_word)
        except _Stopped:
            return TurnResult.STOPPED
        async with self._speech_lock:  # a reminder being announced finishes first
            with self._measured_turn():
                try:
                    return await self._take_command()
                except Exception as exc:
                    await self._recover(exc)
                    return TurnResult.FAILED

    async def announce(self, text: str) -> None:
        """Say something out of turn, such as a due reminder.

        Waits for a turn in progress to finish, and mutes wake-word detection
        while speaking so Jarvis can't wake itself up. Detection resumes with a
        reset detector and an emptied mic buffer, as it does after a reply.
        """
        async with self._speech_lock:
            self._muted.set()
            try:
                self._enter(LoopState.SPEAKING)
                await self.tts.speak_async(speakable(text))
            finally:
                self._muted.clear()
                self._enter(LoopState.IDLE)

    async def _take_command(self) -> TurnResult:
        """Everything after the wake word: record, transcribe, answer, speak."""
        with timer(STAGE_WAKE_TO_STT):
            if self._chime is not None:
                await asyncio.to_thread(self._chime)
            self.detector.reset()
            self._source.clear()  # drop the chime echo and any trailing wake audio

            self._enter(LoopState.LISTENING)
            try:
                audio = await asyncio.to_thread(self.recorder.record_command, self._source)
            except _Stopped:
                set_outcome(OUTCOME_STOPPED)
                return TurnResult.STOPPED

        self._enter(LoopState.PROCESSING)
        with timer(STAGE_STT):
            transcript = (await self.stt.transcribe_async(audio)).strip()
        self._publish(TRANSCRIPT, text=transcript, source="voice")
        if not transcript:
            self._display(NOT_HEARD_MESSAGE.format(wake_phrase=self._wake_phrase))
            self._enter(LoopState.IDLE)
            set_outcome(OUTCOME_EMPTY)
            return TurnResult.EMPTY
        self._display(f"you> {transcript}")
        if is_exit_command(transcript):
            set_outcome(OUTCOME_EXIT)
            return TurnResult.EXIT

        reply = await self.agent.run(transcript)
        self._display(f"jarvis> {reply}")
        self._publish(REPLY, text=reply)

        self._enter(LoopState.SPEAKING)
        # TODO(phase-3+): barge-in, i.e. keep detecting the wake word while
        # speaking and stop playback when it fires. Detection is paused for now.
        with timer(STAGE_TTS):
            await self.tts.speak_async(speakable(reply))
        self._enter(LoopState.IDLE)
        return TurnResult.REPLIED

    async def _recover(self, exc: Exception) -> None:
        """A turn failed: report it, apologise aloud, and settle back to IDLE."""
        self.last_error = exc
        set_outcome(OUTCOME_FAILED, exc)
        self._report(exc)
        self._enter(LoopState.SPEAKING)
        try:
            await self.tts.speak_async(apology_for(exc))
        except Exception as speak_exc:  # e.g. the speakers are what broke
            log.warning(
                "voice_loop.apology_failed", error=f"{type(speak_exc).__name__}: {speak_exc}"
            )
        self._enter(LoopState.IDLE)

    def _report(self, exc: Exception) -> None:
        """Log, show and publish a failure."""
        if isinstance(exc, LLMError):
            log.warning("voice_loop.llm_failed", error=str(exc), status=exc.status_code)
            self._display(f"jarvis> [LLM error] {exc}")
            self._publish(ERROR, message=f"LLM error: {exc}")
        elif isinstance(exc, VoiceError):
            log.warning("voice_loop.voice_failed", error=str(exc))
            self._display(f"jarvis> [voice error] {exc}")
            self._publish(ERROR, message=f"Voice error: {exc}")
        else:
            log.error("voice_loop.turn_failed", exc_info=exc)
            self._display(f"jarvis> [error] {type(exc).__name__}: {exc}")
            self._publish(ERROR, message=f"{type(exc).__name__}: {exc}")

    def _measured_turn(self) -> AbstractContextManager[object]:
        return self._metrics.turn("voice") if self._metrics is not None else nullcontext()

    def _wait_for_wake_word(self) -> float:
        """Blocking: feed frames to the detector until a score crosses the threshold."""
        frame_samples = self.detector.frame_samples
        was_muted = False
        while True:
            frame = self._source.read(frame_samples)
            if self._muted.is_set():
                was_muted = True  # Jarvis is talking out of turn: don't listen to itself
                continue
            if was_muted:
                was_muted = False
                self.detector.reset()  # flush anything it heard of Jarvis's own voice
                self._source.clear()
                continue
            score = self.detector.process(frame)
            if score >= self.threshold:
                log.info("voice_loop.wake", score=round(score, 3), threshold=self.threshold)
                return score

    def _enter(self, state: LoopState) -> None:
        if state is self.state:
            return
        log.info(
            "voice_loop.state",
            from_state=self.state.value if self.state else None,
            to_state=state.value,
        )
        self.state = state
        self._display(_STATUS[state].format(wake_phrase=self._wake_phrase))
        if self._on_state is not None:
            self._on_state(state)
        self._publish(STATE, state=event_state(state))

    def _publish(self, type_: str, **data: object) -> None:
        if self._events is not None:
            self._events.emit(type_, **data)


def apology_for(exc: BaseException) -> str:
    """What Jarvis says aloud when a turn fails: rate limits and outages get their own line."""
    if isinstance(exc, LLMError) and exc.status_code is not None:
        if exc.status_code == 429:
            return RATE_LIMITED_APOLOGY
        if exc.status_code >= 500:
            return UNAVAILABLE_APOLOGY
    return APOLOGY


def event_state(state: LoopState | None) -> str:
    """The `state` event value for a loop state (None, i.e. not started yet, counts as idle)."""
    return _EVENT_STATE[state] if state is not None else STATE_IDLE


_EVENT_STATE = {
    LoopState.IDLE: STATE_IDLE,
    LoopState.LISTENING: STATE_LISTENING,
    LoopState.PROCESSING: STATE_THINKING,
    LoopState.SPEAKING: STATE_SPEAKING,
}

_STATUS = {
    LoopState.IDLE: "\n🟢 listening for '{wake_phrase}'…",
    LoopState.LISTENING: "🎙️ go ahead…",
    LoopState.PROCESSING: "🧠 thinking…",
    LoopState.SPEAKING: "🗣️ speaking…",
}
