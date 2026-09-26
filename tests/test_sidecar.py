"""The desktop app's backend entrypoint: stdin control channel and fail-closed prompts."""

import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from jarvis import sidecar

# Runs the real sidecar entrypoint with the server replaced by a loop that just waits. It
# imports modules once the control watcher is listening, as the real server does: on
# Windows a blocking read on the stdin pipe used to stall those imports indefinitely.
FAKE_SERVER = """
import sys, time, types

def fake_main(argv):
    time.sleep(0.3)
    import numpy  # stalled behind a blocking stdin read on Windows
    print("serving", " ".join(argv), flush=True)
    while True:
        time.sleep(0.05)

fake_cli = types.ModuleType("jarvis.cli")
fake_cli.main = fake_main
sys.modules["jarvis.cli"] = fake_cli  # so nothing imports numpy before the watcher starts
from jarvis.sidecar import main
sys.exit(main(sys.argv[1:]))
"""


def test_watch_control_stops_on_shutdown_line() -> None:
    calls: list[str] = []
    lines = iter(["hello\n", "  SHUTDOWN \n", "never read\n"])
    sidecar.watch_control(lines, lambda: calls.append("stop"))
    assert calls == ["stop"]
    assert next(lines) == "never read\n"


@pytest.mark.skipif(sys.platform != "win32", reason="the polled pipe path is Windows-only")
def test_control_lines_polls_a_pipe_until_eof() -> None:
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
        lines = sidecar.control_lines(stream, poll_seconds=0.01)
        os.write(write_fd, b"hel")
        os.write(write_fd, "lo ✓\nshutdown\npartial".encode())
        os.close(write_fd)
        assert list(lines) == ["hello ✓\n", "shutdown\n", "partial"]


def test_control_lines_iterates_other_streams() -> None:
    assert list(sidecar.control_lines(io.StringIO("a\nb\n"))) == ["a\n", "b\n"]


def test_watch_control_stops_on_eof() -> None:
    calls: list[str] = []
    sidecar.watch_control(io.StringIO("noise\n"), lambda: calls.append("stop"))
    assert calls == ["stop"]


def test_request_shutdown_interrupts_then_forces_exit_when_overdue() -> None:
    forced = threading.Event()
    codes: list[int] = []
    interrupts: list[str] = []

    def force_exit(code: int) -> None:
        codes.append(code)
        forced.set()

    sidecar.request_shutdown(
        grace_seconds=0.01, interrupt=lambda: interrupts.append("sigint"), force_exit=force_exit
    )
    assert interrupts == ["sigint"]
    assert forced.wait(2)
    assert codes == [0]


def test_request_shutdown_timer_can_be_cancelled() -> None:
    codes: list[int] = []
    timer = sidecar.request_shutdown(
        grace_seconds=0.2, interrupt=lambda: None, force_exit=codes.append
    )
    timer.cancel()
    timer.join(1)
    assert codes == []


@pytest.mark.parametrize(
    ("argv", "expected"), [([], ["--serve"]), (["--wake"], ["--serve", "--wake"])]
)
def test_main_runs_serve(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: list[str]
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr("jarvis.cli.main", lambda args: seen.append(args) or 0)
    assert sidecar.main(argv) == 0
    assert seen == [expected]


def test_managed_mode_declines_prompts_and_watches_the_real_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = io.StringIO("")
    stopped = threading.Event()
    answers: list[str] = []
    monkeypatch.setattr(sys, "stdin", control)
    monkeypatch.setattr(sidecar, "_use_utf8_lines", lambda: None)
    monkeypatch.setattr(sidecar, "request_shutdown", stopped.set)

    def fake_cli(args: list[str]) -> int:
        try:
            answers.append(input("Allow this action? [y/N] "))
        except EOFError:
            answers.append("<EOF>")
        return 0

    monkeypatch.setattr("jarvis.cli.main", fake_cli)
    assert sidecar.main(["--managed"]) == 0
    assert answers == ["<EOF>"]  # a y/N prompt can't block on the control pipe
    assert stopped.wait(2)  # EOF on the control stream asked for a shutdown


def test_keyboard_interrupt_before_the_server_is_up_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(args: list[str]) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("jarvis.cli.main", interrupted)
    assert sidecar.main([]) == 0


def _start_fake_server() -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", FAKE_SERVER, "--managed"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdout is not None
    first: list[str] = []
    reader = threading.Thread(target=lambda: first.append(proc.stdout.readline()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    reader.join(20)
    if not first:
        proc.kill()
        pytest.fail("the managed backend stalled before it started serving")
    assert first[0].strip() == "serving --serve"
    return proc


@pytest.mark.parametrize("how", ["shutdown line", "eof"])
def test_managed_process_stops_gracefully(how: str) -> None:
    proc = _start_fake_server()
    assert proc.stdin is not None
    try:
        if how == "shutdown line":
            proc.stdin.write("shutdown\n")
            proc.stdin.flush()
        else:
            proc.stdin.close()  # what the OS does when the desktop app dies
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def test_remove_stale_extractions_keeps_current_and_in_use(tmp_path: Path) -> None:
    runtime = tmp_path / sidecar.RUNTIME_DIR_NAME
    current, stale, busy, other = (runtime / name for name in ("_MEI1", "_MEI2", "_MEI3", "keep"))
    for folder in (current, stale, busy, other):
        (folder / "sub").mkdir(parents=True)
        (folder / "sub" / "lib.dll").write_bytes(b"x" * 10)
    with open(busy / "sub" / "lib.dll", "rb"):  # held open, as a running backend holds its DLLs
        removed = sidecar.remove_stale_extractions(current)
        if sys.platform == "win32":
            assert removed == [stale]
            assert busy.is_dir()
    assert current.is_dir()
    assert other.is_dir()
    assert not stale.exists()
    assert not (runtime / "_MEI2.stale").exists()


def test_cleanup_only_runs_in_the_frozen_desktop_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(sidecar.threading, "Thread", lambda **kw: started.append(kw))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    sidecar._clean_up_after_earlier_runs()  # a normal Python run
    monkeypatch.setattr(sys, "_MEIPASS", str(Path("C:/Temp/_MEI123")), raising=False)
    sidecar._clean_up_after_earlier_runs()  # frozen, but in the shared temp folder
    assert started == []
