"""The desktop app's backend: `jarvis --serve`, run as a child process of the Tauri shell.

    python -m jarvis.sidecar [--wake] [--managed]

PyInstaller freezes this module into the single-file sidecar the desktop app
bundles (`packaging/jarvis-backend.spec`). On its own it is exactly
`python -m jarvis.cli --serve [--wake]`: the panel on 127.0.0.1, configured from
`.env` in the working directory, which the desktop app sets to the Jarvis home.

`--managed` (the desktop app always passes it) adapts the process to having no
terminal:

- stdin becomes a control channel. A `shutdown` line, or EOF because the app
  exited or crashed and its end of the pipe closed, stops the server the way
  Ctrl-C does. If that hasn't finished within `SHUTDOWN_GRACE_SECONDS`, the
  process exits anyway, so no orphaned backend outlives the app.
- Nobody is at a terminal to answer y/N prompts, so side-effect confirmations
  read EOF and are declined (fail closed), as with any closed stdin.
- stdout/stderr become UTF-8 and line-buffered, so the app's log gets every line
  as it happens, emoji included.

When frozen, it also clears extraction folders that earlier runs left behind (see
`remove_stale_extractions`).
"""

import argparse
import io
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TextIO

SHUTDOWN_COMMAND = "shutdown"
SHUTDOWN_GRACE_SECONDS = 8.0
PIPE_POLL_SECONDS = 0.2
# The frozen backend's own extraction folder (`runtime_tmpdir` in jarvis-backend.spec).
RUNTIME_DIR_NAME = "backend-runtime"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis-backend", description="Jarvis panel backend for the desktop app."
    )
    parser.add_argument("--wake", action="store_true", help="also listen for the wake word")
    parser.add_argument(
        "--managed",
        action="store_true",
        help="run under the desktop app: stdin controls shutdown, prompts are declined",
    )
    return parser.parse_args(argv)


def control_lines(stream: TextIO, *, poll_seconds: float = PIPE_POLL_SECONDS) -> Iterator[str]:
    """The lines arriving on the control `stream`, ending at EOF.

    On Windows a pipe is polled rather than read with a blocking call: while one thread
    waits in a synchronous read on the stdin pipe, the main thread stalls (imports never
    finish), so this only reads once data is waiting.
    """
    if sys.platform != "win32" or not _is_pipe(stream):
        yield from stream
        return
    import _winapi
    import msvcrt

    fd = stream.fileno()
    handle = msvcrt.get_osfhandle(fd)
    pending = b""
    while True:
        try:
            available, _ = _winapi.PeekNamedPipe(handle, 0)
        except OSError:  # the writing end closed: EOF
            break
        if not available:
            time.sleep(poll_seconds)
            continue
        pending += os.read(fd, available)
        *complete, pending = pending.split(b"\n")
        for raw in complete:
            yield raw.decode("utf-8", "replace") + "\n"
    if pending:
        yield pending.decode("utf-8", "replace")


def _is_pipe(stream: TextIO) -> bool:
    try:
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        return _winapi.GetFileType(handle) == _winapi.FILE_TYPE_PIPE
    except (ImportError, AttributeError, OSError, ValueError):
        return False


def remove_stale_extractions(current: Path) -> list[Path]:
    """Delete the extraction folders beside `current` (this run's) that no process is using.

    The one-file backend unpacks itself (~0.5 GB) into a new `_MEI*` folder on every start
    and deletes it on a clean exit, but a hard kill leaves it behind. Windows won't rename
    a folder whose files a running backend holds open, so each folder is renamed first and
    only deleted if that worked. Returns the folders removed.
    """
    removed: list[Path] = []
    for folder in sorted(current.parent.glob("_MEI*")):
        if folder == current or not folder.is_dir():
            continue
        doomed = folder.with_name(f"{folder.name}.stale")
        try:
            folder.rename(doomed)
        except OSError:
            continue  # in use by another running backend
        shutil.rmtree(doomed, ignore_errors=True)
        removed.append(folder)
    return removed


def _clean_up_after_earlier_runs() -> None:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is None or Path(bundle).parent.name != RUNTIME_DIR_NAME:
        return  # not the frozen desktop backend: never touch a shared temp folder
    threading.Thread(
        target=remove_stale_extractions, args=(Path(bundle),), name="sidecar-cleanup", daemon=True
    ).start()


def watch_control(lines: Iterable[str], stop: Callable[[], None]) -> None:
    """Consume control `lines` until a shutdown command or EOF, then call `stop` once."""
    for line in lines:
        if line.strip().lower() == SHUTDOWN_COMMAND:
            break
    stop()


def request_shutdown(
    *,
    grace_seconds: float = SHUTDOWN_GRACE_SECONDS,
    interrupt: Callable[[], None] = lambda: signal.raise_signal(signal.SIGINT),
    force_exit: Callable[[int], object] = os._exit,
) -> threading.Timer:
    """Stop the server like Ctrl-C; force the exit if it's still running after the grace period."""

    def overdue() -> None:
        print(f"sidecar: still running {grace_seconds:g}s after shutdown; exiting", file=sys.stderr)
        force_exit(0)

    timer = threading.Timer(grace_seconds, overdue)
    timer.daemon = True
    timer.start()
    interrupt()
    return timer


def _use_utf8_lines() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _clean_up_after_earlier_runs()
    if args.managed:
        _use_utf8_lines()
        control, sys.stdin = sys.stdin, io.StringIO()  # input() now sees EOF: prompts decline
        watcher = threading.Thread(
            target=watch_control,
            args=(control_lines(control), request_shutdown),
            name="sidecar-control",
            daemon=True,
        )
        watcher.start()

    try:
        from jarvis.cli import main as cli_main  # after stdio is set up: imports can log

        return cli_main(["--serve", "--wake"] if args.wake else ["--serve"])
    except KeyboardInterrupt:  # a shutdown that landed before the server was up
        return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # a no-op unless frozen; openWakeWord imports multiprocessing
    raise SystemExit(main())
