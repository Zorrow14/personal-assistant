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

    @field_validator("llm_provider")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return value.strip().lower()

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
