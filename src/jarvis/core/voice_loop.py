"""Hands-free operation: wake word -> record until silence -> agent -> spoken reply.

A small state machine around the existing pipeline. The agent is called exactly
as in text mode; STT, TTS, the wake detector, the VAD recorder and the mic are
all injected through the neutral interfaces, so this module never touches an
audio or model library.

    IDLE ──wake word──▶ LISTENING ──silence/cap──▶ PROCESSING ──text──▶ SPEAKING
     ▲                                                 │ empty             │
     └─────────────────────────────────────────────────┴───────────────────┘
"""

import asyncio
import threading
from collections.abc import Callable
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

NOT_HEARD_MESSAGE = "(Didn't catch that. Say '{wake_phrase}' to try again.)"
MAX_CONSECUTIVE_FAILURES = 3
"""Failed turns in a row before the loop gives up instead of spinning on a persistent fault."""

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
        self._stop = threading.Event()
        self._source = _StoppableSource(source, self._stop)
        self.state: LoopState | None = None

    async def run(self) -> None:
        """Cycle IDLE -> ... -> IDLE until a spoken "exit", `stop()`, or cancellation.

        LLM errors and failed turns are reported and the loop keeps listening,
        unless `MAX_CONSECUTIVE_FAILURES` turns fail in a row (then the last
        error propagates). Microphone failures (`VoiceError`) end the loop at once.
        """
        self._stop.clear()
        failures = 0
        try:
            while not self._stop.is_set():
                try:
                    result = await self.run_once()
                except VoiceError as exc:
                    self._publish(ERROR, message=f"Voice error: {exc}")
                    raise
                except Exception as exc:
                    failures += 1
                    if isinstance(exc, LLMError):
                        self._display(f"jarvis> [LLM error] {exc}")
                        self._publish(ERROR, message=f"LLM error: {exc}")
                    else:
                        log.exception("voice_loop.turn_failed")
                        self._display(f"jarvis> [error] {type(exc).__name__}: {exc}")
                        self._publish(ERROR, message=f"{type(exc).__name__}: {exc}")
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise
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
        """One full cycle: wait for the wake word, take a command, answer it."""
        self._enter(LoopState.IDLE)
        # Detection is paused (the mic isn't read) while speaking, so anything
        # buffered since then, including Jarvis's own voice, is stale.
        self._source.clear()
        try:
            await asyncio.to_thread(self._wait_for_wake_word)
        except _Stopped:
            return TurnResult.STOPPED
        if self._chime is not None:
            await asyncio.to_thread(self._chime)
        self.detector.reset()
        self._source.clear()  # drop the chime echo and any trailing wake audio

        self._enter(LoopState.LISTENING)
        try:
            audio = await asyncio.to_thread(self.recorder.record_command, self._source)
        except _Stopped:
            return TurnResult.STOPPED

        self._enter(LoopState.PROCESSING)
        transcript = (await self.stt.transcribe_async(audio)).strip()
        self._publish(TRANSCRIPT, text=transcript, source="voice")
        if not transcript:
            self._display(NOT_HEARD_MESSAGE.format(wake_phrase=self._wake_phrase))
            self._enter(LoopState.IDLE)
            return TurnResult.EMPTY
        self._display(f"you> {transcript}")
        if is_exit_command(transcript):
            return TurnResult.EXIT

        reply = await self.agent.run(transcript)
        self._display(f"jarvis> {reply}")
        self._publish(REPLY, text=reply)

        self._enter(LoopState.SPEAKING)
        # TODO(phase-3+): barge-in, i.e. keep detecting the wake word while
        # speaking and stop playback when it fires. Detection is paused for now.
        await self.tts.speak_async(speakable(reply))
        self._enter(LoopState.IDLE)
        return TurnResult.REPLIED

    def _wait_for_wake_word(self) -> float:
        """Blocking: feed frames to the detector until a score crosses the threshold."""
        frame_samples = self.detector.frame_samples
        while True:
            score = self.detector.process(self._source.read(frame_samples))
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
