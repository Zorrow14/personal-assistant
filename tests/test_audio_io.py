"""Recording buffer logic with a fake input stream: no microphone needed."""

from typing import Any

import numpy as np
import pytest

from jarvis.audio import io as audio_io


class FakeInputStream:
    """Stands in for sounddevice.InputStream; delivers chunks when the test says so."""

    instance: "FakeInputStream | None" = None

    def __init__(self, *, callback: Any, samplerate: int, channels: int, dtype: str, device: Any) -> None:
        self.callback = callback
        self.kwargs = {"samplerate": samplerate, "channels": channels, "dtype": dtype, "device": device}
        FakeInputStream.instance = self

    def __enter__(self) -> "FakeInputStream":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def feed(self, samples: list[float]) -> None:
        block = np.array(samples, dtype=np.float32).reshape(-1, 1)
        self.callback(block, len(samples), None, None)


def test_record_until_enter_concatenates_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_io.sd, "InputStream", FakeInputStream)

    def speak_then_stop() -> None:
        stream = FakeInputStream.instance
        assert stream is not None
        stream.feed([0.1, 0.2])
        stream.feed([0.3])

    audio = audio_io.record_until_enter(16000, device=3, wait_for_stop=speak_then_stop)

    np.testing.assert_allclose(audio, [0.1, 0.2, 0.3])
    assert audio.dtype == np.float32
    assert FakeInputStream.instance is not None
    assert FakeInputStream.instance.kwargs == {
        "samplerate": 16000,
        "channels": 1,
        "dtype": "float32",
        "device": 3,
    }


def test_record_with_no_audio_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_io.sd, "InputStream", FakeInputStream)

    audio = audio_io.record_until_enter(16000, wait_for_stop=lambda: None)

    assert audio.size == 0
    assert audio.dtype == np.float32
