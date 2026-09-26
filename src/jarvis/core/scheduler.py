"""Reminder scheduler: delivers reminders when they come due.

`set_reminder` records reminders. This background task runs while Jarvis is in
`--wake` or `--serve` mode. Every `reminder_poll_seconds` it claims whatever is
due, writing a `fired` flag to the file first so nothing is announced twice,
and delivers each one:

- always: a `reminder` event for the panel, and a line in the terminal;
- "tts": spoken, via the voice loop's `announce()`, which never talks over a turn;
- "toast": a best-effort desktop notification.

Reminders that came due while Jarvis was off are delivered at the next check,
saying when they were due. When nothing is pending, a poll is a `stat()` call.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from jarvis.core.events import REMINDER, EventBus
from jarvis.logging import get_logger
from jarvis.tools._reminders import Reminder, ReminderStore
from jarvis.tools.context import local_now

NotifyMode = Literal["tts", "toast", "both"]
SpeakFn = Callable[[str], Awaitable[None]]
"""Says something out loud without talking over a turn (e.g. `WakeWordLoop.announce`)."""
ToastFn = Callable[[str, str], bool]
"""Shows a desktop notification (title, message); returns whether it was shown."""

LATE_AFTER = timedelta(minutes=2)
"""A reminder delivered later than this says when it was actually due."""
TOAST_TITLE = "Jarvis reminder"

log = get_logger(__name__)


def reminder_message(reminder: Reminder, now: datetime) -> str:
    """What Jarvis says and shows, e.g. "Reminder: call mom (it was due at 09:00)"."""
    due = reminder.due if reminder.due.tzinfo else reminder.due.replace(tzinfo=now.tzinfo)
    message = f"Reminder: {reminder.text}"
    if now - due > LATE_AFTER:
        when = f"{due:%H:%M}" if due.date() == now.date() else f"{due:%H:%M} on {due:%a %d %b}"
        message += f" (it was due at {when})"
    return message


class ReminderScheduler:
    """Polls the reminders file and delivers each due reminder exactly once."""

    def __init__(
        self,
        store: ReminderStore,
        *,
        notify: NotifyMode = "tts",
        speak: SpeakFn | None = None,
        toast: ToastFn | None = None,
        bus: EventBus | None = None,
        display: Callable[[str], None] | None = None,
        clock: Callable[[], datetime] = local_now,
        poll_seconds: float = 30.0,
    ) -> None:
        """
        Args:
            store: The same reminders file `set_reminder` writes.
            notify: "tts", "toast" or "both" (the panel event is always sent).
            speak: How to say a reminder; None when there's no voice (then it's
                shown in the panel and terminal only).
            toast: Desktop notifier for "toast"/"both"; None disables toasts.
            bus: Where `reminder` events go.
            display: Where a terminal line goes (e.g. print); None for none.
            clock: Current local time (injectable for tests).
            poll_seconds: Seconds between checks.
        """
        self._store = store
        self._notify = notify
        self._speak = speak
        self._toast = toast
        self._bus = bus
        self._display = display
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._stamp: tuple[int, int] | None = None
        self._next_due: datetime | None = None
        self._last_problem: str | None = None

    async def run(self) -> None:
        """Check for due reminders every `poll_seconds` until cancelled. Never raises."""
        log.info("reminders.scheduler_started", path=str(self._store.path), notify=self._notify)
        while True:
            await self.check_once()
            await asyncio.sleep(self._poll_seconds)

    async def check_once(self) -> list[Reminder]:
        """Deliver everything due now. Returns what was delivered."""
        now = self._clock()
        try:
            due = await asyncio.to_thread(self._claim_due, now)
        except Exception as exc:  # e.g. a corrupt file: report once, keep polling
            problem = f"{type(exc).__name__}: {exc}"
            if problem != self._last_problem:
                self._last_problem = problem
                log.warning("reminders.check_failed", error=problem)
            return []
        self._last_problem = None
        for reminder in due:
            await self._deliver(reminder, now)
        return due

    def _claim_due(self, now: datetime) -> list[Reminder]:
        """Worker thread: skip the read when the file hasn't changed and nothing is due yet."""
        stamp = _file_stamp(self._store.path)
        if stamp is None:
            self._stamp = self._next_due = None
            return []
        if stamp == self._stamp and (self._next_due is None or self._next_due > now):
            return []
        due = self._store.claim_due(now)
        self._next_due = self._store.next_pending(now)
        # After our own write, re-read next time rather than trust a stamp taken
        # while someone else might also have been writing.
        self._stamp = None if due else stamp
        return due

    async def _deliver(self, reminder: Reminder, now: datetime) -> None:
        message = reminder_message(reminder, now)
        due = reminder.due if reminder.due.tzinfo else reminder.due.replace(tzinfo=now.tzinfo)
        late_seconds = max(0, round((now - due).total_seconds()))
        log.info("reminders.fired", id=reminder.id, late_s=late_seconds, notify=self._notify)
        if self._bus is not None:
            self._bus.emit(
                REMINDER,
                id=reminder.id,
                text=reminder.text,
                due=due.isoformat(),
                late_seconds=late_seconds,
                message=message,
            )
        if self._display is not None:
            self._display(f"⏰ {message}")
        if self._notify in ("toast", "both"):
            await self._show_toast(message)
        if self._notify in ("tts", "both"):
            await self._say(message)

    async def _show_toast(self, message: str) -> None:
        if self._toast is None:
            log.info("reminders.toast_unavailable")
            return
        try:
            await asyncio.to_thread(self._toast, TOAST_TITLE, message)
        except Exception as exc:
            log.warning("reminders.toast_failed", error=f"{type(exc).__name__}: {exc}")

    async def _say(self, message: str) -> None:
        if self._speak is None:
            log.info("reminders.no_voice", hint="shown in the panel and terminal only")
            return
        try:
            await self._speak(message)
        except Exception as exc:
            log.warning("reminders.speak_failed", error=f"{type(exc).__name__}: {exc}")


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size
