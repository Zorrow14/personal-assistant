"""Local neural text-to-speech with Piper. The only module that knows about Piper."""

import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.audio import io as audio_io
from jarvis.core.interfaces import TTSEngine, VoiceError
from jarvis.logging import get_logger

DEFAULT_VOICE = "en_US-lessac-medium"
DOWNLOAD_HINT = (
    "Download a voice with:\n"
    f"  uv run python -m piper.download_voices {DEFAULT_VOICE} --download-dir voices\n"
    f"then set JARVIS_TTS_VOICE=voices/{DEFAULT_VOICE}.onnx in .env "
    "(or set JARVIS_TTS_PROVIDER=pyttsx3 to use the built-in OS voice)."
)

WavPlayer = Callable[[Path, int | None], None]

log = get_logger(__name__)


class PiperTTS(TTSEngine):
    """`TTSEngine` that synthesizes with Piper to a temp WAV, then plays it."""

    def __init__(
        self,
        voice_path: Path | None,
        *,
        output_device: int | None = None,
        play_wav: WavPlayer = audio_io.play_wav,
    ) -> None:
        """
        Args:
            voice_path: Piper voice `.onnx`; its `.onnx.json` config must sit beside it.
            output_device: Speaker device index; None for the system default.
            play_wav: Plays a WAV file (injectable for tests).

        Raises:
            VoiceError: If the voice or its config file is missing. The voice
                itself is loaded lazily on first `speak`.
        """
        if voice_path is None:
            raise VoiceError(f"JARVIS_TTS_VOICE is not set. {DOWNLOAD_HINT}")
        voice_path = Path(voice_path)
        config_path = voice_path.with_name(voice_path.name + ".json")
        for required in (voice_path, config_path):
            if not required.is_file():
                raise VoiceError(f"Piper voice file not found: {required}. {DOWNLOAD_HINT}")
        self._voice_path = voice_path
        self._output_device = output_device
        self._play_wav = play_wav
        self._voice: Any = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Synthesize `text` and play it, blocking until playback finishes."""
        if not text.strip():
            return
        voice = self._load_voice()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as tmp:
            wav_path = Path(tmp) / "reply.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            log.debug("tts.synthesized", chars=len(text), took_ms=_ms_since(started))
            self._play_wav(wav_path, self._output_device)

    def _load_voice(self) -> Any:
        with self._lock:
            if self._voice is None:
                from piper import PiperVoice  # heavy import; only when first needed

                started = time.perf_counter()
                self._voice = PiperVoice.load(self._voice_path)
                log.info("tts.voice_loaded", voice=self._voice_path.name, took_ms=_ms_since(started))
            return self._voice


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
