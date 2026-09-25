"""Local wake-word detection with openWakeWord (ONNX models on onnxruntime).

The only module that knows about openWakeWord. openwakeword 0.4.x ships its
pretrained models (including "hey_jarvis") inside the package, so nothing is
downloaded; a path to a custom `.onnx` model also works.
"""

import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.core.interfaces import AudioSamples, VoiceError, WakeWordDetector
from jarvis.logging import get_logger

FRAME_SAMPLES = 1280
"""80 ms at 16 kHz: the chunk size openWakeWord is designed around."""
_FLUSH_SECONDS = 2.0
"""Silence fed on reset to push the triggering audio out of the feature window."""

ModelFactory = Callable[[str], Any]

log = get_logger(__name__)


def _load_openwakeword(model_path: str) -> Any:
    from openwakeword.model import Model  # heavy import; only when first needed

    with warnings.catch_warnings():
        # openwakeword 0.4 asks onnxruntime for CUDA first, then falls back to
        # CPU; the "provider not available" warning is expected noise.
        warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*", category=UserWarning)
        return Model(wakeword_model_paths=[model_path])


def resolve_model_path(name_or_path: str) -> str:
    """Map a pretrained model name ("hey_jarvis") or a file path to a model file.

    Raises:
        VoiceError: If it is neither a known pretrained name nor an existing file.
    """
    path = Path(name_or_path).expanduser()
    if path.suffix == ".onnx" and path.is_file():
        return str(path)
    import openwakeword

    pretrained: dict[str, dict[str, str]] = openwakeword.models
    if name_or_path in pretrained:
        return pretrained[name_or_path]["model_path"]
    raise VoiceError(
        f"unknown wake word model {name_or_path!r}. Use a pretrained name "
        f"({', '.join(sorted(pretrained))}) or a path to an .onnx model."
    )


class OpenWakeWordDetector(WakeWordDetector):
    """`WakeWordDetector` backed by an openWakeWord model, loaded lazily once."""

    def __init__(
        self,
        model: str = "hey_jarvis",
        *,
        model_factory: ModelFactory = _load_openwakeword,
        path_resolver: Callable[[str], str] = resolve_model_path,
    ) -> None:
        """
        Args:
            model: Pretrained model name or path to an `.onnx` model.
            model_factory: Builds the openWakeWord model (injectable for tests).
            path_resolver: Maps `model` to a file (injectable for tests).
        """
        self._model_name = model
        self._model_factory = model_factory
        self._path_resolver = path_resolver
        self._model: Any = None
        self._score_key = ""
        self._last_score = 0.0
        self._lock = threading.Lock()

    @property
    def frame_samples(self) -> int:
        """openWakeWord consumes 80 ms (1280-sample) frames."""
        return FRAME_SAMPLES

    def load(self) -> None:
        """Load the model now instead of on the first frame.

        Raises:
            VoiceError: If the model can't be found or loaded.
        """
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            model_path = self._path_resolver(self._model_name)
            try:
                model = self._model_factory(model_path)
            except Exception as exc:
                raise VoiceError(f"could not load wake word model {model_path}: {exc}") from exc
            # One model loaded -> exactly one score key (e.g. "hey_jarvis_v0.1").
            self._score_key = next(iter(model.models))
            self._model = model
            log.info(
                "wakeword.model_loaded",
                model=self._score_key,
                took_ms=round((time.perf_counter() - started) * 1000),
            )

    def process(self, frame: AudioSamples) -> float:
        """Feed one 1280-sample float32 frame; return the wake score (0–1)."""
        self.load()
        scores = self._model.predict(_to_int16(frame))
        self._last_score = float(scores[self._score_key])
        return self._last_score

    def reset(self) -> None:
        """Forget the trigger: clear scores and flush the audio-feature window.

        `Model.reset()` only clears the score history; the rolling feature
        buffer still holds the wake word, which would fire again on the next
        frames. Feeding a little silence pushes it out.
        """
        if self._model is None:
            return
        log.info("wakeword.triggered", model=self._score_key, score=round(self._last_score, 3))
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        for _ in range(int(_FLUSH_SECONDS * 16000 / FRAME_SAMPLES)):
            self._model.predict(silence)
        self._model.reset()
        self._last_score = 0.0


def _to_int16(frame: AudioSamples) -> np.ndarray:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
