"""Select a TTSEngine implementation from settings."""

from jarvis.config import Settings
from jarvis.core.interfaces import TTSEngine, VoiceError

SUPPORTED_PROVIDERS = ("piper", "pyttsx3")


def create_tts_engine(settings: Settings) -> TTSEngine:
    """Build the TTSEngine named by `settings.tts_provider`.

    Provider modules are imported lazily so unused engines are never loaded.

    Raises:
        VoiceError: If the provider is unknown or misconfigured.
    """
    if settings.tts_provider == "piper":
        from jarvis.tts.piper_tts import PiperTTS

        return PiperTTS(settings.tts_voice, output_device=settings.output_device)
    if settings.tts_provider == "pyttsx3":
        from jarvis.tts.system_tts import Pyttsx3TTS

        return Pyttsx3TTS()
    raise VoiceError(
        f"unknown tts_provider {settings.tts_provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
