"""Date/time and reminder tools.

`set_reminder` has a side effect (it writes to disk), so it requires
confirmation. Reminders are only recorded and listed for now: nothing alerts
the user when one is due yet.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field

from jarvis.tools._reminders import ReminderStore, parse_when
from jarvis.tools.base import Tool
from jarvis.tools.context import ToolContext, local_now

Clock = Callable[[], datetime]


class NoArgs(BaseModel):
    """For tools that take no input."""


def _fmt(moment: datetime) -> str:
    return f"{moment:%a %d %b %Y, %H:%M}"


def _until(due: datetime, now: datetime) -> str:
    minutes = int((due - now).total_seconds() // 60)
    if minutes < 0:
        return "overdue"
    if minutes < 60:
        return f"in {minutes} min"
    if minutes < 48 * 60:
        return f"in {minutes // 60} h {minutes % 60} min"
    return f"in {minutes // 1440} days"


class GetDateTimeTool(Tool):
    """Current local date and time."""

    name = "get_datetime"
    description = (
        "Get the current local date, time, weekday and timezone. Use it before reasoning "
        "about 'today', 'tomorrow', 'last week', deadlines or durations."
    )
    args_model = NoArgs
    requires_confirmation = False
    category = "time"

    def __init__(self, clock: Clock = local_now) -> None:
        self._clock = clock

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with the shared clock."""
        return cls(context.clock)

    async def run(self, **kwargs: Any) -> str:
        """Return the current local date/time in words and ISO form."""
        now = self._clock()
        offset = now.strftime("%z")
        zone = f"UTC{offset[:3]}:{offset[3:]}" if offset else "local"
        return (
            f"It is {now:%A %d %B %Y}, {now:%H:%M} ({zone}). "
            f"ISO: {now.isoformat(timespec='seconds')}"
        )


class SetReminderArgs(BaseModel):
    """Input for `set_reminder`."""

    text: str = Field(description="What to remind the user about, e.g. 'call the dentist'.")
    when: str = Field(
        description=(
            "When, in plain words or ISO: 'tomorrow at 9am', 'in 2 hours', 'friday 17:00', "
            "'october 3 at noon', '2026-10-01 14:00'."
        )
    )


class SetReminderTool(Tool):
    """Record a reminder (asks the user first)."""

    name = "set_reminder"
    description = (
        "Save a reminder for the user at a given time. The user is asked to confirm first. "
        "Reminders are recorded and can be listed; Jarvis does not alert the user yet."
    )
    args_model = SetReminderArgs
    requires_confirmation = True
    category = "time"

    def __init__(self, store: ReminderStore, clock: Clock = local_now) -> None:
        self._store = store
        self._clock = clock

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with the configured reminders file."""
        return cls(ReminderStore(context.reminders_path), context.clock)

    async def run(self, **kwargs: Any) -> str:
        """Parse `when`, save, and confirm with the resolved time.

        Raises:
            ReminderTimeError: If `when` can't be understood or is in the past.
        """
        args = SetReminderArgs.model_validate(kwargs)
        now = self._clock()
        due = parse_when(args.when, now)
        reminder = await asyncio.to_thread(self._store.add, args.text, due, now=now)
        # TODO(phase-6): a background scheduler that actually notifies when reminders are due.
        return (
            f"Reminder saved: {reminder.text!r} for {_fmt(due)} ({_until(due, now)}). "
            "Note: Jarvis records reminders but can't alert the user yet; they can ask to list them."
        )


class ListRemindersTool(Tool):
    """Show stored reminders."""

    name = "list_reminders"
    description = "List the user's saved reminders, soonest first, marking any that are overdue."
    args_model = NoArgs
    requires_confirmation = False
    category = "time"

    def __init__(self, store: ReminderStore, clock: Clock = local_now) -> None:
        self._store = store
        self._clock = clock

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Built by discovery with the configured reminders file."""
        return cls(ReminderStore(context.reminders_path), context.clock)

    async def run(self, **kwargs: Any) -> str:
        """Return every reminder with its due time."""
        reminders = await asyncio.to_thread(self._store.all)
        if not reminders:
            return "No reminders saved."
        now = self._clock()
        lines = [f"{len(reminders)} reminder(s), soonest first (now: {_fmt(now)}):"]
        for i, r in enumerate(reminders, start=1):
            lines.append(f"{i}. {_fmt(r.due)} ({_until(r.due, now)}): {r.text}")
        return "\n".join(lines)
