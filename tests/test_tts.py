"""TTS provider selection and config validation. No voices loaded, nothing played."""

from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.interfaces import VoiceError
from jarvis.tts.factory import create_tts_engine
from jarvis.tts.piper_tts import PiperTTS
from jarvis.tts.system_tts import Pyttsx3TTS


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(vault_path=tmp_path, _env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def voice(tmp_path: Path) -> Path:
    onnx = tmp_path / "en_US-test-medium.onnx"
    onnx.write_bytes(b"fake")
    onnx.with_name(onnx.name + ".json").write_text("{}", encoding="utf-8")
    return onnx


def test_selects_piper(tmp_path: Path, voice: Path) -> None:
    engine = create_tts_engine(_settings(tmp_path, tts_provider="piper", tts_voice=voice))
    assert isinstance(engine, PiperTTS)


def test_selects_pyttsx3(tmp_path: Path) -> None:
    engine = create_tts_engine(_settings(tmp_path, tts_provider="PyTTSx3"))
    assert isinstance(engine, Pyttsx3TTS)


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    with pytest.raises(VoiceError, match="unknown tts_provider"):
        create_tts_engine(_settings(tmp_path, tts_provider="espeak"))


def test_piper_without_voice_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(VoiceError, match="download_voices") as info:
        create_tts_engine(_settings(tmp_path, tts_provider="piper", tts_voice=None))
    assert "JARVIS_TTS_PROVIDER=pyttsx3" in str(info.value)


def test_piper_voice_needs_config_json(tmp_path: Path, voice: Path) -> None:
    voice.with_name(voice.name + ".json").unlink()
    with pytest.raises(VoiceError, match=r"\.onnx\.json"):
        PiperTTS(voice)


def test_piper_skips_blank_text(voice: Path) -> None:
    played: list[Path] = []
    PiperTTS(voice, play_wav=lambda path, device: played.append(path)).speak("   ")
    assert played == []
