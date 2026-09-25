"""Typed application settings, loaded from environment variables and `.env`."""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Jarvis configuration.

    Values come from `JARVIS_*` environment variables (the API key from
    `LLM_API_KEY`), falling back to a `.env` file in the working directory.
    Empty values are treated as unset.
    """

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    vault_path: Path
    """Root of the Obsidian vault. Jarvis writes only under `<vault_path>/Jarvis/`."""

    log_level: str = "INFO"

    llm_provider: str = "gemini"
    """Selects the LLMClient implementation (see `jarvis.llm.factory`)."""
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "JARVIS_LLM_API_KEY"),
    )
    """Read from `LLM_API_KEY`. Kept as SecretStr so it never leaks into logs or reprs."""
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    agent_max_iterations: int = Field(default=8, ge=1)
    """Safety cap on LLM round-trips per command in the tool loop."""

    # Voice (Phase 2)
    sample_rate: int = Field(default=16000, gt=0)
    """Microphone sample rate in Hz. Whisper wants 16 kHz mono; other rates are resampled."""
    input_device: int | None = None
    """Microphone device index (`--list-devices`); None = system default."""
    output_device: int | None = None
    """Speaker device index (`--list-devices`); None = system default."""
    stt_model: str = "base.en"
    """faster-whisper model size or path, e.g. tiny.en, base.en, small.en."""
    stt_device: str = "cpu"
    """Where Whisper runs: cpu, or cuda for an NVIDIA GPU."""
    stt_compute_type: str = "int8"
    """Whisper precision: int8 suits CPU, float16 suits GPU."""
    tts_provider: str = "piper"
    """Voice engine: piper (local neural voice) or pyttsx3 (OS voice, no setup)."""
    tts_voice: Path | None = None
    """Path to a Piper voice `.onnx` file (its `.onnx.json` must sit beside it)."""

    # Wake word + voice-activity detection (Phase 3, `--wake`)
    wake_word_model: str = "hey_jarvis"
    """openWakeWord pretrained model name, or a path to a custom `.onnx` model."""
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Score (0–1) that counts as the wake word; higher = fewer false triggers."""
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    """webrtcvad strictness, 0–3; higher treats more noise as non-speech."""
    vad_silence_ms: int = Field(default=800, gt=0)
    """Trailing silence that ends a command."""
    vad_frame_ms: int = 30
    """VAD frame length; webrtcvad accepts only 10, 20 or 30 ms."""
    command_max_seconds: float = Field(default=15.0, gt=0)
    """Hard cap on one command's recording length."""
    wake_chime: bool = True
    """Play a short cue when the wake word is heard."""

    @field_validator("vad_frame_ms")
    @classmethod
    def _check_vad_frame(cls, value: int) -> int:
        if value not in (10, 20, 30):
            raise ValueError("vad_frame_ms must be 10, 20 or 30")
        return value

    @field_validator("llm_provider", "tts_provider", "stt_device")
    @classmethod
    def _normalise_name(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("tts_voice")
    @classmethod
    def _expand_voice_path(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @field_validator("vault_path")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level: {value!r}")
        return level


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Raises `pydantic.ValidationError` if invalid."""
    return Settings()  # type: ignore[call-arg]  # fields are populated from the env
