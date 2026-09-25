"""Settings tests. Uses monkeypatched env vars and ignores any real .env."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.config import Settings


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("LLM_API_KEY", "JARVIS_LLM_API_KEY", "JARVIS_LLM_PROVIDER", "JARVIS_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JARVIS_VAULT_PATH", str(tmp_path))


def test_defaults() -> None:
    settings = _settings()
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.llm_max_tokens == 1024
    assert settings.llm_temperature == 0.7
    assert settings.agent_max_iterations == 8
    assert settings.llm_api_key is None
    assert settings.sample_rate == 16000
    assert (settings.input_device, settings.output_device) == (None, None)
    assert (settings.stt_model, settings.stt_device, settings.stt_compute_type) == (
        "base.en",
        "cpu",
        "int8",
    )
    assert settings.tts_provider == "piper"
    assert settings.tts_voice is None


def test_api_key_read_from_llm_api_key_and_kept_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    settings = _settings()
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(settings)


def test_invalid_numbers_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_AGENT_MAX_ITERATIONS", "0")
    with pytest.raises(ValidationError):
        _settings()
