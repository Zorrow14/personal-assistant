"""Microphone and speaker I/O via sounddevice.

This module only moves audio samples; it knows nothing about STT, TTS or
wake-word detection.
"""

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import numpy as np
import sounddevice as sd
import soundfile as sf

from jarvis.core.interfaces import AudioSamples, VoiceError
from jarvis.logging import get_logger

RECORDING_CUE = "🎙️ recording… (press Enter to stop)"
_RECORDING_CUE_ASCII = "[recording... press Enter to stop]"
_STOP_POLL_SECONDS = 0.1

STREAM_SAMPLE_RATE = 16000
"""Rate of the shared always-on stream: what openWakeWord, webrtcvad and Whisper all accept."""
CHIME_SAMPLE_RATE = 22050

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


def play(
    audio: np.ndarray,
    sample_rate: int,
    device: int | None = None,
    *,
    level_listener: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Play a sample buffer and block until playback finishes.

    Args:
        level_listener: If given, receives each 50 ms block of `audio`, paced
            to real time while it plays (drives the panel's orb).
    """
    sd.play(audio, samplerate=sample_rate, device=device)
    if level_listener is not None:
        _report_blocks_while_playing(audio, sample_rate, level_listener)
    sd.wait()


def play_wav(
    path: str | Path,
    device: int | None = None,
    *,
    level_listener: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Load a WAV file and play it, blocking until finished."""
    data, sample_rate = sf.read(str(path), dtype="float32")
    play(data, sample_rate, device, level_listener=level_listener)


def _report_blocks_while_playing(
    audio: np.ndarray, sample_rate: int, listener: Callable[[np.ndarray], None]
) -> None:
    block = max(1, sample_rate // 20)
    started = time.perf_counter()
    for offset in range(0, len(audio), block):
        listener(audio[offset : offset + block])
        ahead = started + (offset + block) / sample_rate - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)


def list_devices() -> str:
    """Human-readable table of audio devices and their indices."""
    return str(sd.query_devices())


class MicStream:
    """Continuously captured 16 kHz mono microphone audio, read in any frame size.

    One stream feeds several consumers that want different framing (the wake
    detector reads 80 ms frames, the VAD 10–30 ms frames): each simply calls
    `read(n)` with its own size. Audio keeps buffering while nobody reads, so
    call `clear()` to drop stale audio (e.g. Jarvis's own voice) before
    listening again. Use as a context manager so the device is always released.
    """

    def __init__(
        self,
        sample_rate: int = STREAM_SAMPLE_RATE,
        device: int | None = None,
        *,
        read_timeout: float = 3.0,
        audio_listener: Callable[[AudioSamples], None] | None = None,
    ) -> None:
        """
        Args:
            sample_rate: Capture rate in Hz.
            device: Input device index; None for the system default.
            read_timeout: Seconds without any audio before `read` gives up.
            audio_listener: Also receives every captured block, on the audio
                thread (e.g. a level meter). Must be quick and must not raise.
        """
        self.sample_rate = sample_rate
        self._device = device
        self._read_timeout = read_timeout
        self._audio_listener = audio_listener
        self._chunks: queue.Queue[AudioSamples] = queue.Queue()
        self._pending: AudioSamples = np.zeros(0, dtype=np.float32)
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        """Open the microphone and start buffering audio."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._on_audio,
        )
        self._stream.start()
        log.debug("audio.stream_started", sample_rate=self.sample_rate, device=self._device)

    def close(self) -> None:
        """Stop capturing and release the device. Safe to call twice."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
            log.debug("audio.stream_closed")

    def read(self, n_samples: int) -> AudioSamples:
        """Block until `n_samples` samples are available and return exactly that many.

        Raises:
            VoiceError: If no audio arrives for `read_timeout` seconds.
        """
        while len(self._pending) < n_samples:
            try:
                chunk = self._chunks.get(timeout=self._read_timeout)
            except queue.Empty:
                raise VoiceError(
                    f"no audio from the microphone for {self._read_timeout:g}s; "
                    "check it is connected and JARVIS_INPUT_DEVICE (see --list-devices)"
                ) from None
            self._pending = np.concatenate((self._pending, chunk))
        frame, self._pending = self._pending[:n_samples], self._pending[n_samples:]
        return frame

    def clear(self) -> None:
        """Discard everything buffered so far."""
        self._pending = np.zeros(0, dtype=np.float32)
        while True:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                return

    def _on_audio(
        self, indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags
    ) -> None:
        if status:
            log.warning("audio.input_status", status=str(status))
        block = indata[:, 0].copy()
        self._chunks.put(block)
        if self._audio_listener is not None:
            self._audio_listener(block)


def chime_samples(sample_rate: int = CHIME_SAMPLE_RATE) -> AudioSamples:
    """A short rising two-note cue (~0.2 s), with fades so it doesn't click."""
    notes = []
    for freq, seconds in ((880.0, 0.09), (1320.0, 0.11)):
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        tone = 0.25 * np.sin(2 * np.pi * freq * t)
        fade = min(len(t) // 4, int(sample_rate * 0.01))
        envelope = np.ones_like(t)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        notes.append(tone * envelope)
    return np.concatenate(notes).astype(np.float32)


def play_chime(device: int | None = None) -> None:
    """Play the wake cue and block until it finishes."""
    play(chime_samples(), CHIME_SAMPLE_RATE, device)


def _print_cue(text: str, ascii_fallback: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:  # e.g. output piped through a legacy code page
        print(ascii_fallback, flush=True)
