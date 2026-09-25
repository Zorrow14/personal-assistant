"""VoiceActivityDetector tests: scripted speech/silence, plus real webrtcvad on synthetic audio."""

import numpy as np
import pytest

from jarvis.audio.vad import VoiceActivityDetector, float_to_pcm16
from tests.fakes import FakeFrameSource


class ScriptedClassifier:
    """Stands in for webrtcvad: returns a scripted speech/non-speech pattern."""

    def __init__(self, pattern: list[bool], default: bool = False) -> None:
        self.pattern = list(pattern)
        self.default = default
        self.frame_bytes: list[int] = []

    def is_speech(self, buf: bytes, sample_rate: int) -> bool:
        self.frame_bytes.append(len(buf))
        return self.pattern.pop(0) if self.pattern else self.default


def _vad(pattern: list[bool], default: bool = False, **kwargs: object) -> tuple[VoiceActivityDetector, ScriptedClassifier]:
    classifier = ScriptedClassifier(pattern, default)
    params = {"frame_ms": 30, "silence_ms": 90, "max_seconds": 15.0, **kwargs}
    return VoiceActivityDetector(classifier=classifier, **params), classifier  # type: ignore[arg-type]


def test_stops_after_speech_then_trailing_silence() -> None:
    # 2 silent frames, 3 speech, then silence: 90 ms = 3 silent frames ends it.
    vad, classifier = _vad([False, False, True, True, True])
    source = FakeFrameSource()

    audio = vad.record_command(source)

    assert len(audio) == 8 * 480  # 2 + 3 + 3 frames of 30 ms at 16 kHz
    assert source.samples_read == 8 * 480
    assert classifier.frame_bytes[0] == 480 * 2  # 16-bit PCM


def test_short_pause_mid_sentence_does_not_end_command() -> None:
    vad, _ = _vad([True, False, False, True, True])  # 60 ms pause < 90 ms
    audio = vad.record_command(FakeFrameSource())
    assert len(audio) == (5 + 3) * 480


def test_silence_before_speech_never_ends_command() -> None:
    vad, _ = _vad([], default=False, max_seconds=0.3)  # never any speech
    audio = vad.record_command(FakeFrameSource())
    assert len(audio) == 10 * 480  # ran to the cap, not stopped by silence


def test_command_max_seconds_caps_recording() -> None:
    vad, _ = _vad([], default=True, max_seconds=0.6)  # talks forever
    source = FakeFrameSource()

    audio = vad.record_command(source)

    assert len(audio) == 20 * 480  # 0.6 s / 30 ms
    assert audio.dtype == np.float32


@pytest.mark.parametrize("frame_ms", [10, 20, 30])
def test_frame_sizes(frame_ms: int) -> None:
    vad, _ = _vad([], frame_ms=frame_ms)
    assert vad.frame_samples == 16 * frame_ms


def test_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="frame_ms"):
        VoiceActivityDetector(frame_ms=25, classifier=ScriptedClassifier([]))
    with pytest.raises(ValueError, match="aggressiveness"):
        VoiceActivityDetector(aggressiveness=4, classifier=ScriptedClassifier([]))
    with pytest.raises(ValueError, match="sample_rate"):
        VoiceActivityDetector(sample_rate=22050, classifier=ScriptedClassifier([]))


def test_float_to_pcm16_clips_and_scales() -> None:
    pcm = np.frombuffer(float_to_pcm16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)), "<i2")
    assert pcm.tolist() == [0, 32767, -32767, 32767]


def test_real_webrtcvad_treats_silence_as_non_speech() -> None:
    vad = VoiceActivityDetector(aggressiveness=2)  # real webrtcvad: a C extension, no model
    assert vad.is_speech(float_to_pcm16(np.zeros(480, dtype=np.float32))) is False
