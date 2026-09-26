"""Typed application settings, loaded from environment variables and `.env`."""

import ipaddress
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    # Memory / retrieval (Phase 4)
    embedder_provider: str = "local"
    """Embedding backend. Only "local" (fastembed, on-device) exists; nothing leaves the machine."""
    embed_model: str = "BAAI/bge-small-en-v1.5"
    """fastembed model name; downloaded once on first use."""
    vector_store: str = "chroma"
    """Vector store backend. Only "chroma" (embedded, on disk) exists."""
    chroma_path: Path = Path(".jarvis/chroma")
    """Where the index lives. Its parent also holds the manifest and model cache."""
    rag_top_k: int = Field(default=5, ge=1, le=50)
    """Default number of chunks `search_memory` retrieves."""
    rag_chunk_chars: int = Field(default=1000, ge=100)
    """Maximum characters per indexed chunk."""
    rag_chunk_overlap: int = Field(default=150, ge=0)
    """Characters shared between consecutive chunks of one note."""
    auto_index: bool = True
    """Index notes written during a turn right after it, so they're searchable at once."""

    # Tools (Phase 5)
    enabled_tools: Annotated[list[str] | None, NoDecode] = None
    """Whitelist of tool names; None = every discovered tool. Env: comma-separated or JSON list."""
    confirm_side_effects: bool = True
    """Master switch for the y/N gate on tools with side effects. Leave on."""
    search_max_results: int = Field(default=5, ge=1, le=20)
    """Default number of web search results."""
    reminders_path: Path = Path(".jarvis/reminders.json")
    """Where reminders are stored (local JSON, git-ignored)."""
    file_sandbox_root: Path | None = None
    """The only folder `read_local_file` may read under; None = the vault folder."""

    # Local web panel (Phase 6A, `--serve`)
    ui_host: str = "127.0.0.1"
    """Address the panel listens on. Loopback only (127.0.0.1, ::1, localhost): Jarvis
    runs tools and hears the mic, so it must never be reachable from other machines."""
    ui_port: int = Field(default=8000, ge=1, le=65535)

    # Observability + reminders (Phase 6B)
    metrics_path: Path = Path(".jarvis/metrics.jsonl")
    """One JSON line per turn: stage latencies, LLM requests and tokens. Local only."""
    metrics_enabled: bool = True
    """Write per-turn metrics to `metrics_path`."""
    reminder_poll_seconds: int = Field(default=30, ge=1)
    """How often the scheduler (running in --wake / --serve) checks for due reminders."""
    reminder_notify: Literal["tts", "toast", "both"] = "tts"
    """How a due reminder is delivered: spoken, a desktop notification, or both."""

    @field_validator("reminder_notify", mode="before")
    @classmethod
    def _normalise_notify(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("ui_host")
    @classmethod
    def _require_loopback(cls, value: str) -> str:
        host = value.strip()
        if not is_loopback_host(host):
            raise ValueError(
                f"ui_host must be a loopback address (127.0.0.1, ::1 or localhost), got {value!r}. "
                "The panel can run tools and hears the mic, so it must never be reachable "
                "from other machines."
            )
        return host

    @field_validator("enabled_tools", mode="before")
    @classmethod
    def _split_tool_list(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return json.loads(text)
            return [name.strip() for name in text.split(",") if name.strip()]
        return value

    @field_validator("embedder_provider", "vector_store")
    @classmethod
    def _normalise_backend(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_chars:
            raise ValueError("rag_chunk_overlap must be smaller than rag_chunk_chars")
        return self

    @property
    def data_dir(self) -> Path:
        """Jarvis's local state folder (default `.jarvis/`)."""
        return self.chroma_path.expanduser().parent

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


def is_loopback_host(host: str) -> bool:
    """True for "localhost" and loopback IPs (127.0.0.0/8, ::1). False for all else, incl. 0.0.0.0."""
    name = host.strip().lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Raises `pydantic.ValidationError` if invalid."""
    return Settings()  # type: ignore[call-arg]  # fields are populated from the env
