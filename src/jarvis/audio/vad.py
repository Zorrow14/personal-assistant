"""Voice-activity detection with webrtcvad: decides when a spoken command has ended.

The only module that knows about webrtcvad. Audio-only: no STT or agent knowledge.
"""

from typing import Protocol

import numpy as np

from jarvis.core.interfaces import AudioSamples, FrameSource
from jarvis.logging import get_logger

VALID_FRAME_MS = (10, 20, 30)
VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)

log = get_logger(__name__)


class _SpeechClassifier(Protocol):
    def is_speech(self, buf: bytes, sample_rate: int) -> bool: ...


def float_to_pcm16(samples: AudioSamples) -> bytes:
    """Convert float32 samples in [-1, 1] to little-endian 16-bit PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


class VoiceActivityDetector:
    """Frame-level speech detection plus end-of-command recording.

    Implements the `CommandRecorder` protocol used by the wake-word loop.
    """

    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        silence_ms: int = 800,
        max_seconds: float = 15.0,
        sample_rate: int = 16000,
        classifier: _SpeechClassifier | None = None,
    ) -> None:
        """
        Args:
            aggressiveness: webrtcvad mode 0–3; higher is stricter about what counts as speech.
            frame_ms: Frame length; webrtcvad accepts 10, 20 or 30 ms.
            silence_ms: Trailing silence (after speech) that ends a command.
            max_seconds: Hard cap on a command's length.
            sample_rate: 8, 16, 32 or 48 kHz.
            classifier: Replaces webrtcvad (injectable for tests).

        Raises:
            ValueError: If a parameter is outside what webrtcvad supports.
        """
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError(f"frame_ms must be one of {VALID_FRAME_MS}, got {frame_ms}")
        if sample_rate not in VALID_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {VALID_SAMPLE_RATES}, got {sample_rate}")
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0–3, got {aggressiveness}")
        if classifier is None:
            import webrtcvad

            classifier = webrtcvad.Vad(aggressiveness)
        self._classifier = classifier
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self._silence_frames = max(1, -(-silence_ms // frame_ms))  # ceil division
        self._max_frames = max(1, int(max_seconds * 1000 // frame_ms))

    def is_speech(self, frame: bytes) -> bool:
        """Classify one frame of 16-bit PCM (exactly `frame_samples` long)."""
        return self._classifier.is_speech(frame, self.sample_rate)

    def record_command(self, source: FrameSource) -> AudioSamples:
        """Record from `source` until speech has been heard and then stops.

        Stops once speech has been detected and `silence_ms` of silence follows,
        or when `max_seconds` is reached, whichever comes first.

        Returns:
            Everything recorded, as float32 samples at `sample_rate`.
        """
        frames: list[AudioSamples] = []
        heard_speech = False
        silent_run = 0
        reason = "max_length"
        while len(frames) < self._max_frames:
            frame = source.read(self.frame_samples)
            frames.append(frame)
            if self.is_speech(float_to_pcm16(frame)):
                heard_speech = True
                silent_run = 0
            else:
                silent_run += 1
            if heard_speech and silent_run >= self._silence_frames:
                reason = "silence"
                break
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        log.info(
            "vad.command_recorded",
            seconds=round(len(audio) / self.sample_rate, 2),
            heard_speech=heard_speech,
            stopped_by=reason,
        )
        return audio
