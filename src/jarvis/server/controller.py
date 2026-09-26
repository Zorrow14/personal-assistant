"""What the panel's controls do: typed commands, Talk and Stop, on the one shared agent.

There is no web code here: `server/app.py` hands over parsed messages and
forwards bus events. Typed commands call `agent.run` exactly as text mode does.
Voice turns reuse the existing `WakeWordLoop`: Talk is simply a wake word fired
by a button (`ManualWakeTrigger`), so the state, transcript and reply events
come from the same code in every mode.
"""

import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from jarvis import __version__
from jarvis.core.agent import Agent
from jarvis.core.events import (
    ERROR,
    NOTICE,
    REPLY,
    STATE,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_THINKING,
    TRANSCRIPT,
    Event,
    EventBus,
)
from jarvis.core.interfaces import AudioSamples, FrameSource, LLMError, WakeWordDetector
from jarvis.core.voice_loop import LoopState, TurnResult, WakeWordLoop, event_state
from jarvis.core.voice_session import is_exit_command
from jarvis.logging import get_logger

MAX_TEXT_CHARS = 4000
"""Longest typed command the panel accepts."""
BUSY_MESSAGE = "Busy: still working on the last request."
EXIT_HINT = (
    "Jarvis keeps running while the panel is open. To quit, press Ctrl-C in the "
    "terminal you started it from."
)
NOTHING_TO_STOP = (
    "Nothing is listening right now. Requests already sent to the model finish on their own."
)
WAKE_FRAME_SAMPLES = 1280
"""Frame size the manual trigger asks for when it wraps no real detector (80 ms at 16 kHz)."""

log = get_logger(__name__)

LoopFactory = Callable[[WakeWordDetector, FrameSource], WakeWordLoop]
"""Builds a `WakeWordLoop` (wired to the agent, STT, TTS and bus) around a detector and mic."""


class ManualWakeTrigger(WakeWordDetector):
    """A wake-word detector the panel can fire, so Talk works like saying the wake word.

    Wraps the real detector in hands-free mode. With no detector (panel-only
    mode) it fires only when triggered.
    """

    def __init__(self, inner: WakeWordDetector | None = None) -> None:
        """
        Args:
            inner: The real detector to defer to between manual triggers.
        """
        self._inner = inner
        self._pending = threading.Event()  # set on the event loop, read on a worker thread

    @property
    def frame_samples(self) -> int:
        """Same framing as the wrapped detector."""
        return self._inner.frame_samples if self._inner is not None else WAKE_FRAME_SAMPLES

    def trigger(self) -> None:
        """Make the next `process` call report a certain wake word."""
        self._pending.set()

    def process(self, frame: AudioSamples) -> float:
        """1.0 once after `trigger()`; otherwise the wrapped detector's score (or 0.0)."""
        if self._pending.is_set():
            self._pending.clear()
            return 1.0
        return self._inner.process(frame) if self._inner is not None else 0.0

    def reset(self) -> None:
        """Drop any pending trigger (so a double click can't fire twice) and reset the detector."""
        self._pending.clear()
        if self._inner is not None:
            self._inner.reset()


class SwitchableMic(FrameSource, Protocol):
    """A microphone that can be opened for one turn and closed again (e.g. `MicStream`)."""

    def start(self) -> None:
        """Open the device and start buffering audio."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


class HandsFreeVoice:
    """`--serve --wake`: keeps the wake-word loop running and lets Talk and Stop steer it."""

    def __init__(
        self,
        loop: WakeWordLoop,
        trigger: ManualWakeTrigger,
        bus: EventBus,
        *,
        wake_phrase: str = "Hey Jarvis",
    ) -> None:
        """
        Args:
            loop: The always-on loop; its detector must be `trigger`.
            trigger: Fires the loop when the panel's Talk button is pressed.
            bus: Where notices about the loop go.
            wake_phrase: Shown in the panel.
        """
        self.loop = loop
        self.wake_phrase = wake_phrase
        self._trigger = trigger
        self._bus = bus
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._closing = False
        self.failure: str | None = None
        """Why the loop stopped for good, if it did."""

    @property
    def state(self) -> str | None:
        """The loop's state as an event value; None when it isn't running."""
        if self._task is None or self._task.done():
            return None
        return event_state(self.loop.state)

    async def start(self) -> None:
        """Start listening for the wake word in the background."""
        self._task = asyncio.create_task(self._keep_listening(), name="jarvis-hands-free")

    def trigger(self) -> Event | None:
        """Start a voice turn now. Returns a message for the requester if it can't."""
        if self.state is None:
            reason = f": {self.failure}" if self.failure else ""
            return _error(f"Hands-free listening isn't running{reason}.")
        if self.loop.state is not LoopState.IDLE:
            return _notice(BUSY_MESSAGE)
        self._trigger.trigger()
        return None

    def stop(self) -> bool:
        """Abandon the command being recorded, if any. Returns whether anything was stopped."""
        if self.state != STATE_LISTENING:
            return False
        self._stop_requested = True
        self.loop.stop()  # `_keep_listening` restarts it, back to waiting for the wake word
        return True

    async def aclose(self) -> None:
        """Stop the loop and wait for it to finish."""
        self._closing = True
        self.loop.stop()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _keep_listening(self) -> None:
        while not self._closing:
            self._stop_requested = False
            try:
                await self.loop.run()
            except Exception as exc:  # the loop already published the error itself
                log.exception("panel.hands_free_stopped")
                self.failure = f"{type(exc).__name__}: {exc}"
                self._bus.emit(
                    NOTICE,
                    message="Hands-free listening has stopped. Typing still works; "
                    "restart Jarvis to listen again.",
                )
                self._bus.emit(STATE, state=STATE_IDLE)
                return
            if not self._closing and not self._stop_requested:
                # `run()` only returns by itself after a spoken "exit"; the panel stays up.
                self._bus.emit(NOTICE, message=EXIT_HINT)


class PushToTalkVoice:
    """`--serve` alone: the mic opens only for a Talk turn and closes right after."""

    def __init__(
        self,
        make_loop: LoopFactory,
        mic: SwitchableMic,
        bus: EventBus,
        *,
        warm_up: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            make_loop: Builds the loop for one turn around a detector and the mic.
            mic: Opened for each Talk turn, closed after it.
            bus: Where notices go.
            warm_up: Blocking preparation to run in the background at start
                (loading the Whisper model), so the first Talk is quick.
        """
        self._make_loop = make_loop
        self._mic = mic
        self._bus = bus
        self._warm_up = warm_up
        self._warm_task: asyncio.Task[None] | None = None
        self._loop: WakeWordLoop | None = None

    @property
    def state(self) -> str | None:
        """The current Talk turn's state as an event value; None between turns."""
        return event_state(self._loop.state) if self._loop is not None else None

    async def start(self) -> None:
        """Begin warming up (e.g. loading the speech model) in the background."""
        if self._warm_up is not None:
            self._warm_task = asyncio.create_task(self._warm(), name="jarvis-voice-warm-up")

    async def talk_turn(self) -> None:
        """Open the mic and run one listen -> transcribe -> answer -> speak cycle."""
        trigger = ManualWakeTrigger()
        trigger.trigger()  # fires on the first frame: go straight to listening
        loop = self._make_loop(trigger, self._mic)
        self._loop = loop
        try:
            await asyncio.to_thread(self._mic.start)
            result = await loop.run_once()
        finally:
            self._loop = None
            await asyncio.to_thread(self._mic.close)
        if result is TurnResult.EXIT:
            self._bus.emit(NOTICE, message=EXIT_HINT)

    def stop(self) -> bool:
        """Abandon the recording in progress, if any. Returns whether anything was stopped."""
        loop = self._loop
        if loop is None or loop.state not in (None, LoopState.IDLE, LoopState.LISTENING):
            return False
        loop.stop()  # the blocked mic read notices within one frame
        return True

    async def aclose(self) -> None:
        """Abandon any recording and stop warming up."""
        if self._loop is not None:
            self._loop.stop()
        if self._warm_task is not None:
            self._warm_task.cancel()
            await asyncio.gather(self._warm_task, return_exceptions=True)

    async def _warm(self) -> None:
        assert self._warm_up is not None
        try:
            await asyncio.to_thread(self._warm_up)
        except Exception as exc:  # Talk will retry and report it properly
            log.warning("panel.voice_warm_up_failed", error=f"{type(exc).__name__}: {exc}")


PanelVoice = HandsFreeVoice | PushToTalkVoice


class PanelController:
    """Turns panel commands into actions on the shared agent and voice loop.

    One request runs at a time. While one is running, new text or Talk
    commands get a "busy" notice rather than queueing up behind it.
    """

    def __init__(
        self,
        agent: Agent,
        bus: EventBus,
        *,
        voice: PanelVoice | None = None,
        voice_error: str | None = None,
    ) -> None:
        """
        Args:
            agent: The same agent every other mode uses (publishing to `bus`).
            bus: Where this controller's events go.
            voice: Enables Talk; None for a text-only panel.
            voice_error: Why voice is unavailable, shown when Talk is pressed.
        """
        self.agent = agent
        self._bus = bus
        self._voice = voice
        self._voice_error = voice_error
        self._turn: asyncio.Task[None] | None = None

    @property
    def busy(self) -> bool:
        """Whether a panel-started request is still running."""
        return self._turn is not None and not self._turn.done()

    def current_state(self) -> str:
        """The orb state a newly opened panel should show."""
        voice_state = self._voice.state if self._voice is not None else None
        if voice_state not in (None, STATE_IDLE):
            return voice_state
        return STATE_THINKING if self.busy else STATE_IDLE

    def snapshot(self) -> dict[str, Any]:
        """Everything a new panel connection needs up front (the `hello` event's data)."""
        hands_free = isinstance(self._voice, HandsFreeVoice)
        return {
            "state": self.current_state(),
            "version": __version__,
            "voice": self._voice is not None,
            "voice_error": self._voice_error,
            "hands_free": hands_free,
            "wake_phrase": self._voice.wake_phrase
            if isinstance(self._voice, HandsFreeVoice)
            else None,
            "max_text_chars": MAX_TEXT_CHARS,
        }

    async def start(self) -> None:
        """Start background voice work (the hands-free loop, or model warm-up)."""
        if self._voice is not None:
            await self._voice.start()

    async def aclose(self) -> None:
        """Cancel whatever is running and stop the voice loop."""
        if self._turn is not None:
            self._turn.cancel()
            await asyncio.gather(self._turn, return_exceptions=True)
        if self._voice is not None:
            await self._voice.aclose()

    def handle(self, message: Any) -> Event | None:
        """Act on one decoded panel message. Never raises.

        Returns:
            An event meant only for the sender (a refusal or a notice), or None.
        """
        if not isinstance(message, dict) or not isinstance(message.get("cmd"), str):
            return _error('Expected a JSON object such as {"cmd": "text", "text": "hello"}.')
        cmd = message["cmd"]
        if cmd == "text":
            return self._on_text(message.get("text"))
        if cmd == "talk":
            return self._on_talk()
        if cmd == "stop":
            return self._on_stop()
        return _error(f"Unknown command {cmd[:40]!r}; expected text, talk or stop.")

    def _on_text(self, text: object) -> Event | None:
        if not isinstance(text, str) or not text.strip():
            return _error("Type a command first.")
        text = text.strip()
        if len(text) > MAX_TEXT_CHARS:
            return _error(f"That's {len(text)} characters; the limit is {MAX_TEXT_CHARS}.")
        if is_exit_command(text):  # text mode would quit; the panel never shuts Jarvis down
            return _notice(EXIT_HINT)
        if self._is_busy():
            return _notice(BUSY_MESSAGE)
        log.info("panel.command", cmd="text", chars=len(text))
        self._start_turn(self._text_turn(text))
        return None

    def _on_talk(self) -> Event | None:
        if self._voice is None:
            reason = self._voice_error or "voice isn't set up"
            return _error(f"Talk is unavailable: {reason}")
        if self._is_busy():
            return _notice(BUSY_MESSAGE)
        log.info("panel.command", cmd="talk")
        if isinstance(self._voice, HandsFreeVoice):
            return self._voice.trigger()
        self._start_turn(self._voice.talk_turn())
        return None

    def _on_stop(self) -> Event | None:
        if self._voice is not None and self._voice.stop():
            log.info("panel.command", cmd="stop")
            return _notice("Stopped listening.")
        return _notice(NOTHING_TO_STOP)

    def _is_busy(self) -> bool:
        return self.busy or self.current_state() != STATE_IDLE

    def _start_turn(self, turn: Coroutine[Any, Any, None]) -> None:
        self._turn = asyncio.create_task(self._guarded(turn), name="jarvis-panel-turn")

    async def _guarded(self, turn: Coroutine[Any, Any, None]) -> None:
        """Run a turn, turning failures into `error` events and always settling the orb."""
        try:
            await turn
        except LLMError as exc:
            self._bus.emit(ERROR, message=f"LLM error: {exc}")
        except Exception as exc:
            log.exception("panel.turn_failed")
            self._bus.emit(ERROR, message=f"{type(exc).__name__}: {exc}")
        finally:
            voice_state = self._voice.state if self._voice is not None else None
            self._bus.emit(STATE, state=voice_state or STATE_IDLE)

    async def _text_turn(self, text: str) -> None:
        """A typed command: exactly what text mode does, observed through events."""
        self._bus.emit(TRANSCRIPT, text=text, source="text")
        self._bus.emit(STATE, state=STATE_THINKING)
        reply = await self.agent.run(text)
        self._bus.emit(REPLY, text=reply)


def _error(message: str) -> Event:
    return Event(ERROR, {"message": message})


def _notice(message: str) -> Event:
    return Event(NOTICE, {"message": message})
