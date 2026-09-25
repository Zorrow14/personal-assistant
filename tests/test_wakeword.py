"""OpenWakeWordDetector tests with a fake model: nothing loaded from disk."""

from typing import Any

import numpy as np
import pytest

from jarvis.core.interfaces import VoiceError
from jarvis.wakeword.openwakeword_detector import OpenWakeWordDetector, resolve_model_path


class FakeOwwModel:
    def __init__(self, scores: list[float]) -> None:
        self.models = {"hey_jarvis_v0.1": object()}
        self.scores = scores
        self.inputs: list[np.ndarray] = []
        self.resets = 0

    def predict(self, x: np.ndarray) -> dict[str, float]:
        self.inputs.append(x)
        return {"hey_jarvis_v0.1": self.scores.pop(0) if self.scores else 0.0}

    def reset(self) -> None:
        self.resets += 1


def _detector(model: FakeOwwModel, loads: list[str]) -> OpenWakeWordDetector:
    def factory(path: str) -> FakeOwwModel:
        loads.append(path)
        return model

    return OpenWakeWordDetector("hey_jarvis", model_factory=factory, path_resolver=lambda n: f"/m/{n}.onnx")


def test_scores_frames_as_int16_and_loads_once() -> None:
    model = FakeOwwModel([0.1, 0.8])
    loads: list[str] = []
    detector = _detector(model, loads)
    frame = np.full(1280, 0.5, dtype=np.float32)

    assert detector.frame_samples == 1280
    assert detector.process(frame) == pytest.approx(0.1)
    assert detector.process(frame) == pytest.approx(0.8)

    assert loads == ["/m/hey_jarvis.onnx"]
    assert model.inputs[0].dtype == np.int16
    assert int(model.inputs[0][0]) == 16383


def test_reset_flushes_feature_window_and_clears_scores() -> None:
    model = FakeOwwModel([0.9])
    detector = _detector(model, [])
    detector.process(np.zeros(1280, dtype=np.float32))

    detector.reset()

    assert model.resets == 1
    flushed = model.inputs[1:]
    assert len(flushed) >= 20  # ~2 s of silence pushed through
    assert all(not f.any() for f in flushed)


def test_reset_before_load_is_a_no_op() -> None:
    model = FakeOwwModel([])
    loads: list[str] = []
    _detector(model, loads).reset()
    assert loads == []


def test_load_failure_becomes_voice_error() -> None:
    def broken(path: str) -> Any:
        raise RuntimeError("bad onnx")

    detector = OpenWakeWordDetector("x", model_factory=broken, path_resolver=lambda n: n)
    with pytest.raises(VoiceError, match="bad onnx"):
        detector.load()


def test_resolve_pretrained_name_and_reject_unknown() -> None:
    # Resolves against the package's bundled model table; nothing is loaded or downloaded.
    assert resolve_model_path("hey_jarvis").endswith("hey_jarvis_v0.1.onnx")
    with pytest.raises(VoiceError, match="hey_jarvis"):
        resolve_model_path("hey_computer")
