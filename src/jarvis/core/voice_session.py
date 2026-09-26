"""One push-to-talk voice turn: speech -> transcript -> agent -> spoken reply.

Voice is only an I/O shell around `Agent.run`; the agent is called exactly as
in text mode. Depends on the STT/TTS interfaces, never on a concrete engine.
"""

import re
from collections.abc import Callable
from enum import Enum

from jarvis.core.agent import Agent
from jarvis.core.interfaces import AudioSamples, STTEngine, TTSEngine
from jarvis.obs import metrics

EXIT_COMMANDS = frozenset({"exit", "quit"})
NOT_HEARD_MESSAGE = "(Didn't catch that. Press Enter and try again.)"

_MARKDOWN_NOISE = re.compile(r"[*_`#>]+")


class TurnOutcome(Enum):
    """How a voice turn ended."""

    REPLIED = "replied"
    EMPTY = "empty"
    EXIT = "exit"


def is_exit_command(text: str) -> bool:
    """True for "exit"/"quit", ignoring case and punctuation (Whisper adds "Exit.")."""
    return re.sub(r"[^\w\s]", "", text).strip().lower() in EXIT_COMMANDS


def speakable(text: str) -> str:
    """Strip Markdown symbols a voice would otherwise read aloud."""
    return " ".join(_MARKDOWN_NOISE.sub(" ", text).split())


class VoiceSession:
    """Runs voice turns against an existing agent."""

    def __init__(
        self,
        agent: Agent,
        stt: STTEngine,
        tts: TTSEngine,
        *,
        display: Callable[[str], None] = print,
    ) -> None:
        """
        Args:
            agent: The same agent text mode uses.
            stt: Speech-to-text engine.
            tts: Text-to-speech engine.
            display: Where transcripts and replies are shown.
        """
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self._display = display

    async def run_turn(self, audio: AudioSamples) -> TurnOutcome:
        """Transcribe `audio`, show it, then `respond`. Empty speech skips the agent."""
        with metrics.timer(metrics.STAGE_STT):
            transcript = (await self.stt.transcribe_async(audio)).strip()
        if not transcript:
            self._display(NOT_HEARD_MESSAGE)
            metrics.set_outcome(metrics.OUTCOME_EMPTY)
            return TurnOutcome.EMPTY
        self._display(f"you> {transcript}")
        return await self.respond(transcript)

    async def respond(self, text: str) -> TurnOutcome:
        """Run `text` through the agent, show the reply, and speak it."""
        if is_exit_command(text):
            metrics.set_outcome(metrics.OUTCOME_EXIT)
            return TurnOutcome.EXIT
        reply = await self.agent.run(text)
        self._display(f"jarvis> {reply}")
        with metrics.timer(metrics.STAGE_TTS):
            await self.tts.speak_async(speakable(reply))
        return TurnOutcome.REPLIED
