"""Microphone and speaker I/O via sounddevice.

This module only moves audio samples; it knows nothing about STT or TTS.
"""

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from jarvis.core.interfaces import AudioSamples
from jarvis.logging import get_logger

RECORDING_CUE = "🎙️ recording… (press Enter to stop)"
_RECORDING_CUE_ASCII = "[recording... press Enter to stop]"
_STOP_POLL_SECONDS = 0.1

log = get_logger(__name__)


def _wait_for_enter() -> None:
    try:
        input()
    except EOFError:
        pass


def record_until_enter(
    sample_rate: int,
    device: int | None = None,
    *,
    wait_for_stop: Callable[[], None] = _wait_for_enter,
) -> AudioSamples:
    """Record mono float32 audio from the microphone until the user presses Enter.

    Recording starts immediately. A background thread waits for Enter (or
    whatever `wait_for_stop` blocks on) while the stream's callback buffers audio.

    Args:
        sample_rate: Capture rate in Hz.
        device: Input device index; None for the system default.
        wait_for_stop: Blocks until recording should stop (injectable for tests).

    Returns:
        The captured samples (empty if nothing arrived).
    """
    chunks: list[AudioSamples] = []

    def on_audio(indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags) -> None:
        if status:
            log.warning("audio.input_status", status=str(status))
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", device=device, callback=on_audio
    ):
        _print_cue(RECORDING_CUE, _RECORDING_CUE_ASCII)
        stopper = threading.Thread(target=wait_for_stop, name="record-stop", daemon=True)
        stopper.start()
        # Poll rather than join() forever so Ctrl-C stays responsive on Windows.
        while stopper.is_alive():
            stopper.join(_STOP_POLL_SECONDS)

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    log.debug("audio.recorded", seconds=round(len(audio) / sample_rate, 2))
    return audio


def play(audio: np.ndarray, sample_rate: int, device: int | None = None) -> None:
    """Play a sample buffer and block until playback finishes."""
    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()


def play_wav(path: str | Path, device: int | None = None) -> None:
    """Load a WAV file and play it, blocking until finished."""
    data, sample_rate = sf.read(str(path), dtype="float32")
    play(data, sample_rate, device)


def list_devices() -> str:
    """Human-readable table of audio devices and their indices."""
    return str(sd.query_devices())


def _print_cue(text: str, ascii_fallback: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:  # e.g. output piped through a legacy code page
        print(ascii_fallback, flush=True)
