"""MicStream framing and play_chime tests with a fake sounddevice stream: no mic."""

from typing import Any

import numpy as np
import pytest

from jarvis.audio import io as audio_io
from jarvis.core.interfaces import VoiceError


class FakeRawStream:
    """Stands in for sounddevice.InputStream; the test pushes blocks via `deliver`."""

    last: "FakeRawStream | None" = None

    def __init__(self, *, samplerate: int, channels: int, dtype: str, device: Any, callback: Any) -> None:
        self.callback = callback
        self.samplerate = samplerate
        self.started = self.stopped = self.closed = False
        FakeRawStream.last = self

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def deliver(self, values: list[float]) -> None:
        self.callback(np.array(values, dtype=np.float32).reshape(-1, 1), len(values), None, None)


@pytest.fixture
def raw(monkeypatch: pytest.MonkeyPatch) -> type[FakeRawStream]:
    monkeypatch.setattr(audio_io.sd, "InputStream", FakeRawStream)
    return FakeRawStream


def test_reads_arbitrary_frame_sizes_across_blocks(raw: type[FakeRawStream]) -> None:
    with audio_io.MicStream() as mic:
        stream = raw.last
        assert stream is not None and stream.started and stream.samplerate == 16000
        stream.deliver([1, 2, 3, 4, 5])
        stream.deliver([6, 7, 8])

        assert mic.read(3).tolist() == [1, 2, 3]  # e.g. a VAD frame
        assert mic.read(4).tolist() == [4, 5, 6, 7]  # e.g. a wake-word frame spanning blocks
        stream.deliver([9])
        assert mic.read(2).tolist() == [8, 9]
    assert stream.stopped and stream.closed


def test_clear_drops_buffered_audio(raw: type[FakeRawStream]) -> None:
    with audio_io.MicStream() as mic:
        stream = raw.last
        assert stream is not None
        stream.deliver([1, 2, 3])
        mic.read(1)
        stream.deliver([4, 5])

        mic.clear()
        stream.deliver([6, 7])

        assert mic.read(2).tolist() == [6, 7]


def test_read_times_out_with_actionable_error(raw: type[FakeRawStream]) -> None:
    with audio_io.MicStream(read_timeout=0.05) as mic:
        with pytest.raises(VoiceError, match="JARVIS_INPUT_DEVICE"):
            mic.read(10)


def test_chime_is_short_and_in_range() -> None:
    samples = audio_io.chime_samples()
    assert 0.1 < len(samples) / audio_io.CHIME_SAMPLE_RATE < 0.5
    assert samples.dtype == np.float32
    assert np.abs(samples).max() <= 0.26
    assert abs(samples[0]) < 0.01 and abs(samples[-1]) < 0.01  # faded, no click


def test_play_chime_uses_output_device(monkeypatch: pytest.MonkeyPatch) -> None:
    played: list[tuple[int, Any]] = []
    monkeypatch.setattr(audio_io, "play", lambda audio, rate, device=None: played.append((rate, device)))

    audio_io.play_chime(device=4)

    assert played == [(audio_io.CHIME_SAMPLE_RATE, 4)]
