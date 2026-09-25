"""Local speech-to-text with faster-whisper. The only module that knows about it."""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from jarvis.config import Settings
from jarvis.core.interfaces import AudioSamples, STTEngine, VoiceError
from jarvis.logging import get_logger

WHISPER_SAMPLE_RATE = 16000

log = get_logger(__name__)

ModelFactory = Callable[[str, str, str], Any]


def _load_whisper_model(model: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel  # heavy import; only when first needed

    return WhisperModel(model, device=device, compute_type=compute_type)


class WhisperSTT(STTEngine):
    """`STTEngine` backed by faster-whisper, loading the model lazily once."""

    def __init__(
        self,
        *,
        model: str = "small.en",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_rate: int = WHISPER_SAMPLE_RATE,
        model_factory: ModelFactory = _load_whisper_model,
    ) -> None:
        """
        Args:
            model: Model size (e.g. "base.en") or local path.
            device: "cpu" or "cuda".
            compute_type: e.g. "int8" (CPU) or "float16" (GPU).
            sample_rate: Rate of the audio passed to `transcribe`.
            model_factory: Builds the model (injectable for tests).
        """
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._sample_rate = sample_rate
        self._model_factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "WhisperSTT":
        """Build from the `stt_*` and `sample_rate` settings."""
        return cls(
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            sample_rate=settings.sample_rate,
        )

    def load(self) -> None:
        """Load the model now (downloads it on first ever use).

        Raises:
            VoiceError: If the model can't be downloaded or loaded.
        """
        with self._lock:
            if self._model is not None:
                return
            log.info(
                "stt.model_loading",
                model=self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
            started = time.perf_counter()
            try:
                self._model = self._model_factory(
                    self._model_name, self._device, self._compute_type
                )
            except Exception as exc:
                raise VoiceError(
                    f"could not load Whisper model {self._model_name!r} on {self._device} "
                    f"({self._compute_type}): {exc}. Check JARVIS_STT_MODEL / "
                    "JARVIS_STT_DEVICE / JARVIS_STT_COMPUTE_TYPE (int8 for cpu, float16 for cuda)."
                ) from exc
            log.info("stt.model_loaded", seconds=round(time.perf_counter() - started, 2))

    def transcribe(self, audio: AudioSamples) -> str:
        """Transcribe mono float32 audio; returns "" for silence or empty input."""
        if audio.size == 0:
            return ""
        self.load()
        samples = resample(np.asarray(audio, dtype=np.float32), self._sample_rate, WHISPER_SAMPLE_RATE)
        started = time.perf_counter()
        # vad_filter drops silent stretches, which stops Whisper hallucinating
        # text ("Thank you.") on a recording with no speech.
        segments, _info = self._model.transcribe(samples, beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        log.info(
            "stt.transcribed",
            audio_s=round(len(samples) / WHISPER_SAMPLE_RATE, 2),
            took_ms=round((time.perf_counter() - started) * 1000),
            chars=len(text),
        )
        return text


def resample(audio: AudioSamples, from_rate: int, to_rate: int) -> AudioSamples:
    """Linearly resample mono audio. Good enough for speech recognition."""
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration = len(audio) / from_rate
    target_len = max(1, round(duration * to_rate))
    source_t = np.arange(len(audio)) / from_rate
    target_t = np.arange(target_len) / to_rate
    return np.interp(target_t, source_t, audio).astype(np.float32)
