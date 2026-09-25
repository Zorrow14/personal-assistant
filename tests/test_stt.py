"""WhisperSTT tests with a fake model: no download, no microphone."""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from jarvis.core.interfaces import VoiceError
from jarvis.stt.whisper_stt import WhisperSTT, resample


class FakeWhisperModel:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append((audio, kwargs))
        return iter(SimpleNamespace(text=t) for t in self.texts), SimpleNamespace()


def _stt(model: FakeWhisperModel, loads: list[tuple[str, str, str]], **kwargs: Any) -> WhisperSTT:
    def factory(name: str, device: str, compute_type: str) -> FakeWhisperModel:
        loads.append((name, device, compute_type))
        return model

    return WhisperSTT(model="tiny.en", device="cpu", compute_type="int8", model_factory=factory, **kwargs)


def test_joins_segments_and_loads_model_once() -> None:
    model = FakeWhisperModel([" Log that ", "I finished. "])
    loads: list[tuple[str, str, str]] = []
    stt = _stt(model, loads)
    audio = np.ones(8000, dtype=np.float32)

    assert stt.transcribe(audio) == "Log that I finished."
    assert asyncio.run(stt.transcribe_async(audio)) == "Log that I finished."

    assert loads == [("tiny.en", "cpu", "int8")]
    assert model.calls[0][1]["vad_filter"] is True


def test_empty_audio_returns_empty_without_loading() -> None:
    loads: list[tuple[str, str, str]] = []
    stt = _stt(FakeWhisperModel(["x"]), loads)

    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert loads == []


def test_non_16k_audio_is_resampled_before_whisper() -> None:
    model = FakeWhisperModel(["hi"])
    stt = _stt(model, [], sample_rate=48000)

    stt.transcribe(np.ones(48000, dtype=np.float32))

    sent = model.calls[0][0]
    assert sent.dtype == np.float32
    assert len(sent) == 16000


def test_resample_preserves_duration_and_shape() -> None:
    tone = np.sin(np.linspace(0, 100, 44100)).astype(np.float32)
    out = resample(tone, 44100, 16000)
    assert len(out) == 16000
    assert resample(tone, 16000, 16000) is tone


def test_load_failure_becomes_voice_error() -> None:
    def broken(name: str, device: str, compute_type: str) -> Any:
        raise ValueError("float16 not supported on cpu")

    stt = WhisperSTT(model="base.en", compute_type="float16", model_factory=broken)
    with pytest.raises(VoiceError, match="float16"):
        stt.load()
