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
    assert (settings.embedder_provider, settings.embed_model, settings.vector_store) == (
        "local",
        "BAAI/bge-small-en-v1.5",
        "chroma",
    )
    assert settings.chroma_path == Path(".jarvis/chroma")
    assert settings.data_dir == Path(".jarvis")
    assert (settings.rag_top_k, settings.rag_chunk_chars, settings.rag_chunk_overlap) == (
        5,
        1000,
        150,
    )
    assert settings.auto_index is True


def test_tool_settings_defaults() -> None:
    settings = _settings()
    assert settings.enabled_tools is None
    assert settings.confirm_side_effects is True
    assert settings.search_max_results == 5
    assert settings.reminders_path == Path(".jarvis/reminders.json")
    assert settings.file_sandbox_root is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("web_search, get_datetime", ["web_search", "get_datetime"]),
        ('["web_search","set_reminder"]', ["web_search", "set_reminder"]),
        ("search_memory", ["search_memory"]),
    ],
)
def test_enabled_tools_from_env(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("JARVIS_ENABLED_TOOLS", raw)
    assert _settings().enabled_tools == expected


def test_chunk_overlap_must_be_smaller_than_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_RAG_CHUNK_CHARS", "200")
    monkeypatch.setenv("JARVIS_RAG_CHUNK_OVERLAP", "200")
    with pytest.raises(ValidationError, match="rag_chunk_overlap"):
        _settings()


def test_api_key_read_from_llm_api_key_and_kept_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    settings = _settings()
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(settings)


def test_ui_defaults_to_loopback() -> None:
    settings = _settings()
    assert (settings.ui_host, settings.ui_port) == ("127.0.0.1", 8000)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_ui_host_accepts_loopback(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    monkeypatch.setenv("JARVIS_UI_HOST", host)
    assert _settings().ui_host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "my-laptop.local"])
def test_ui_host_rejects_anything_reachable_from_other_machines(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setenv("JARVIS_UI_HOST", host)
    with pytest.raises(ValidationError, match="loopback"):
        _settings()


def test_ui_port_must_be_a_valid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_UI_PORT", "70000")
    with pytest.raises(ValidationError):
        _settings()


def test_invalid_numbers_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_AGENT_MAX_ITERATIONS", "0")
    with pytest.raises(ValidationError):
        _settings()
