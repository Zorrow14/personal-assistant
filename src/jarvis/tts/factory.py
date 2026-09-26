"""Select a TTSEngine implementation from settings."""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from jarvis.config import Settings
from jarvis.core.interfaces import TTSEngine, VoiceError

SUPPORTED_PROVIDERS = ("piper", "pyttsx3")


def create_tts_engine(
    settings: Settings,
    *,
    level_listener: Callable[[np.ndarray], None] | None = None,
) -> TTSEngine:
    """Build the TTSEngine named by `settings.tts_provider`.

    Provider modules are imported lazily so unused engines are never loaded.

    Args:
        settings: Selects and configures the engine.
        level_listener: Receives the audio blocks as they play (for the
            panel's orb). Piper only: pyttsx3 plays through the OS, out of reach.

    Raises:
        VoiceError: If the provider is unknown or misconfigured.
    """
    if settings.tts_provider == "piper":
        from jarvis.tts.piper_tts import PiperTTS

        if level_listener is None:
            return PiperTTS(settings.tts_voice, output_device=settings.output_device)
        from jarvis.audio import io as audio_io

        def play_metered(path: Path, device: int | None) -> None:
            audio_io.play_wav(path, device, level_listener=level_listener)

        return PiperTTS(
            settings.tts_voice, output_device=settings.output_device, play_wav=play_metered
        )
    if settings.tts_provider == "pyttsx3":
        from jarvis.tts.system_tts import Pyttsx3TTS

        return Pyttsx3TTS()
    raise VoiceError(
        f"unknown tts_provider {settings.tts_provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
