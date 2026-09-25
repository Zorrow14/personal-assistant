"""Test doubles that implement Jarvis interfaces without any network access."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from jarvis.core.interfaces import (
    AudioSamples,
    FrameSource,
    LLMClient,
    LLMResponse,
    Message,
    STTEngine,
    ToolSpec,
    TTSEngine,
    WakeWordDetector,
)
from jarvis.tools.base import Tool


class FakeSTTEngine(STTEngine):
    """Returns scripted transcripts in order and records the audio it received."""

    def __init__(self, transcripts: Sequence[str]) -> None:
        self._transcripts = list(transcripts)
        self.received: list[AudioSamples] = []

    def transcribe(self, audio: AudioSamples) -> str:
        self.received.append(audio)
        return self._transcripts.pop(0)


class FakeTTSEngine(TTSEngine):
    """Records everything it was asked to speak."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools) if tools is not None else None))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class EchoArgs(BaseModel):
    text: str


class RecordingTool(Tool):
    """Echoes its input and records each run."""

    name = "echo"
    description = "Echo the text back."
    args_model = EchoArgs

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> str:
        self.runs.append(kwargs)
        return f"echo: {kwargs['text']}"


class DangerousTool(RecordingTool):
    """Same as RecordingTool but gated behind user confirmation."""

    name = "delete_everything"
    description = "Pretend to delete everything."
    requires_confirmation = True


class ExplodingTool(RecordingTool):
    name = "explode"
    description = "Always fails."

    async def run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")


class FakeWakeWordDetector(WakeWordDetector):
    """Returns scripted wake scores, one per frame, and records resets."""

    def __init__(self, scores: Sequence[float], frame_samples: int = 1280) -> None:
        self._scores = list(scores)
        self._frame_samples = frame_samples
        self.frames_seen = 0
        self.resets = 0

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def process(self, frame: AudioSamples) -> float:
        assert len(frame) == self._frame_samples
        self.frames_seen += 1
        if not self._scores:
            raise AssertionError("FakeWakeWordDetector ran out of scripted scores")
        return self._scores.pop(0)

    def reset(self) -> None:
        self.resets += 1


class FakeFrameSource:
    """Endless silence, recording how much was read and how often it was cleared."""

    def __init__(self) -> None:
        self.samples_read = 0
        self.clears = 0

    def read(self, n_samples: int) -> AudioSamples:
        self.samples_read += n_samples
        return np.zeros(n_samples, dtype=np.float32)

    def clear(self) -> None:
        self.clears += 1


class FakeVAD:
    """A CommandRecorder that returns a fixed buffer without reading the source."""

    def __init__(self, seconds: float = 1.0, sample_rate: int = 16000) -> None:
        self.buffer = np.full(int(seconds * sample_rate), 0.1, dtype=np.float32)
        self.calls = 0

    def record_command(self, source: FrameSource) -> AudioSamples:
        self.calls += 1
        return self.buffer
