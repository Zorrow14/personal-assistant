"""Date/time + reminder tools and the time parser. Fixed clock, tmp files."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.tools._reminders import ReminderStore, ReminderStoreError, ReminderTimeError, parse_when
from jarvis.tools.time_tools import GetDateTimeTool, ListRemindersTool, SetReminderTool

TZ = timezone(timedelta(hours=6, minutes=30))
NOW = datetime(2026, 9, 26, 17, 45, tzinfo=TZ)  # a Saturday


def at(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow", at(2026, 9, 27, 9)),
        ("tomorrow at 7pm", at(2026, 9, 27, 19)),
        ("Tomorrow morning", at(2026, 9, 27, 9)),
        ("tomorrow at 12am", at(2026, 9, 27, 0)),
        ("in 10 minutes", NOW + timedelta(minutes=10)),
        ("in an hour", NOW + timedelta(hours=1)),
        ("in 1 hour and 30 minutes", NOW + timedelta(minutes=90)),
        ("in half an hour", NOW + timedelta(minutes=30)),
        ("in 2 days", NOW + timedelta(days=2)),
        ("at 9pm", at(2026, 9, 26, 21)),
        ("9:30 pm", at(2026, 9, 26, 21, 30)),
        ("at 9am", at(2026, 9, 27, 9)),  # already past today -> tomorrow
        ("21:15", at(2026, 9, 26, 21, 15)),
        ("at 5", at(2026, 9, 27, 17)),  # bare 1-7 means pm; 17:00 today has passed
        ("tonight", at(2026, 9, 26, 20)),
        ("tonight at 9", at(2026, 9, 26, 21)),
        ("this evening", at(2026, 9, 26, 19)),
        ("noon tomorrow", at(2026, 9, 27, 12)),
        ("midnight", at(2026, 9, 27, 0)),
        ("monday", at(2026, 9, 28, 9)),
        ("next friday at 17:00", at(2026, 10, 2, 17)),
        ("saturday 9am", at(2026, 10, 3, 9)),  # today is Saturday, 9am passed -> next week
        ("saturday 8pm", at(2026, 9, 26, 20)),  # later today
        ("on October 3rd at 5pm", at(2026, 10, 3, 17)),
        ("3 oct", at(2026, 10, 3, 9)),
        ("september 1", at(2027, 9, 1, 9)),  # passed this year -> next year
        ("2026-10-01", at(2026, 10, 1, 9)),
        ("2026-10-01 14:00", at(2026, 10, 1, 14)),
        ("2026-10-01 at 3pm", at(2026, 10, 1, 15)),
        ("day after tomorrow", at(2026, 9, 28, 9)),
    ],
)
def test_parse_when(text: str, expected: datetime) -> None:
    assert parse_when(text, NOW) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "No time given"),
        ("whenever you like", "Couldn't understand"),
        ("in a bit", "Couldn't understand"),
        ("today at 9am", "already passed"),
        ("2020-01-01 10:00", "already passed"),
        ("feb 30", "Couldn't understand"),
        ("at 25:00", "Couldn't understand"),
    ],
)
def test_parse_when_rejects(text: str, message: str) -> None:
    with pytest.raises(ReminderTimeError, match=message):
        parse_when(text, NOW)


def test_get_datetime() -> None:
    text = asyncio.run(GetDateTimeTool(lambda: NOW).run())
    assert (
        text
        == "It is Saturday 26 September 2026, 17:45 (UTC+06:30). ISO: 2026-09-26T17:45:00+06:30"
    )


def test_set_reminder_requires_confirmation_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "state" / "reminders.json"
    setter = SetReminderTool(ReminderStore(path), lambda: NOW)
    lister = ListRemindersTool(ReminderStore(path), lambda: NOW)

    assert SetReminderTool.requires_confirmation is True
    assert ListRemindersTool.requires_confirmation is False
    assert asyncio.run(lister.run()) == "No reminders saved."

    reply = asyncio.run(setter.run(text="test the gate", when="tomorrow at 10am"))
    asyncio.run(setter.run(text="stretch", when="in 20 minutes"))

    assert "Reminder saved: 'test the gate' for Sun 27 Sep 2026, 10:00 (in 16 h 15 min)" in reply
    assert "announced when due" in reply  # Phase 6B: the scheduler now delivers them
    listing = asyncio.run(lister.run()).splitlines()
    assert listing[0].startswith("2 reminder(s), soonest first")
    assert listing[1] == "1. Sat 26 Sep 2026, 18:05 (in 20 min): stretch"
    assert listing[2] == "2. Sun 27 Sep 2026, 10:00 (in 16 h 15 min): test the gate"
    # Survives a fresh store (i.e. a restart).
    assert [r.text for r in ReminderStore(path).all()] == ["stretch", "test the gate"]


def test_bad_time_raises_before_saving(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    with pytest.raises(ReminderTimeError):
        asyncio.run(SetReminderTool(ReminderStore(path), lambda: NOW).run(text="x", when="someday"))
    assert not path.exists()


def test_corrupt_reminders_file_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "reminders.json"
    path.write_text("{not json", encoding="utf-8")
    setter = SetReminderTool(ReminderStore(path), lambda: NOW)

    with pytest.raises(ReminderStoreError, match="was not modified"):
        asyncio.run(setter.run(text="x", when="tomorrow"))
    assert path.read_text(encoding="utf-8") == "{not json"
