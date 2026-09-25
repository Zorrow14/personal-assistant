"""Zero-setup text-to-speech using the OS voice via pyttsx3.

The only module that knows about pyttsx3. On Windows pyttsx3 drives SAPI5
over COM, whose objects belong to the thread that created them, so every call
runs on one dedicated worker thread that initialises COM first. A fresh engine
is built per utterance: reusing one (as `pyttsx3.init()` does) hangs on the
second `runAndWait()` with pyttsx3 2.99 on Windows.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from jarvis.core.interfaces import TTSEngine
from jarvis.logging import get_logger

log = get_logger(__name__)


def _init_worker_thread() -> None:
    if sys.platform == "win32":
        import pythoncom  # from pywin32, a pyttsx3 dependency on Windows

        pythoncom.CoInitialize()


class Pyttsx3TTS(TTSEngine):
    """`TTSEngine` using the operating system's built-in voice.

    Plays through the OS default output; `output_device` is not supported.
    """

    def __init__(self) -> None:
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pyttsx3", initializer=_init_worker_thread
        )

    def speak(self, text: str) -> None:
        """Speak `text`, blocking until finished."""
        if not text.strip():
            return
        self._worker.submit(_speak_on_worker, text).result()


def _speak_on_worker(text: str) -> None:
    import pyttsx3

    engine = pyttsx3.Engine()  # deliberately not pyttsx3.init(): see module docstring
    engine.say(text)
    engine.runAndWait()
    log.debug("tts.spoken", provider="pyttsx3", chars=len(text))
