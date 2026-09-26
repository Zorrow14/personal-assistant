"""Reminder support: natural-language time parsing and a local JSON store.

Private helper module (the leading `_` keeps tool discovery from scanning it).
The parser is deliberately small and dependency-free. It understands:

- relative: "in 10 minutes", "in an hour", "in 1 hour 30 minutes", "in half an hour", "in 2 days"
- days: "today", "tonight", "tomorrow", "day after tomorrow", weekdays ("friday",
  "next monday"), "october 3", "3 oct 2027", ISO "2026-10-01"
- times: "9am", "9:30 pm", "21:00", "at 9" (1–7 without am/pm means pm),
  "noon", "midnight", "morning" (9), "afternoon" (15), "evening" (19), "tonight" (20)
- full ISO timestamps: "2026-10-01 14:00", "2026-10-01T14:00"

A day without a time defaults to 09:00; a time without a day means the next
time it occurs. Results must be in the future.
"""

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

DEFAULT_HOUR = 9
_PART_OF_DAY = {"morning": 9, "afternoon": 15, "evening": 19, "night": 20, "tonight": 20}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
_UNIT_MINUTES = {"minute": 1, "min": 1, "hour": 60, "hr": 60, "day": 1440, "week": 10080}

_RELATIVE_PAIR = re.compile(
    r"(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s*(minute|min|hour|hr|day|week)s?\b"
)
_CLOCK_AMPM = re.compile(r"(?:\bat\s+)?\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?(?=\W|$)")
_CLOCK_24H = re.compile(r"(?:\bat\s+)?\b(\d{1,2}):(\d{2})\b")
_CLOCK_BARE = re.compile(r"\bat\s+(\d{1,2})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b"
)
_DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?("
    + "|".join(_MONTHS)
    + r")[a-z]*\.?(?:,?\s+(\d{4}))?\b"
)
_FILLER = {"on", "at", "the", "this", "next", "coming", "by", "of", "in"}

HELP = (
    "Try e.g. 'tomorrow at 9am', 'in 2 hours', 'friday 17:00', 'october 3 at noon' "
    "or '2026-10-01 14:00'."
)


class ReminderTimeError(ValueError):
    """The requested time couldn't be understood, or is in the past."""


def parse_when(text: str, now: datetime) -> datetime:
    """Turn a natural-language time into a datetime in `now`'s timezone.

    Raises:
        ReminderTimeError: If `text` isn't understood or the time has passed.
    """
    original = text
    text = " ".join(text.lower().replace(",", " ").split()).strip(" .!?")
    if not text:
        raise ReminderTimeError(f"No time given. {HELP}")

    result = (
        _parse_iso(original.strip(), now)
        or _parse_relative(text, now)
        or _parse_absolute(text, now)
    )
    if result is None:
        raise ReminderTimeError(f"Couldn't understand the time {original!r}. {HELP}")
    if result <= now:
        raise ReminderTimeError(
            f"{original!r} resolves to {result:%a %d %b %Y %H:%M}, which has already passed."
        )
    return result


def _parse_iso(text: str, now: datetime) -> datetime | None:
    if not re.match(r"^\d{4}-\d{2}-\d{2}([ T]\d{1,2}:\d{2}(:\d{2})?)?", text):
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if len(text) == 10:  # date only
        parsed = parsed.replace(hour=DEFAULT_HOUR)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=now.tzinfo)


def _parse_relative(text: str, now: datetime) -> datetime | None:
    if not text.startswith("in "):
        return None
    rest = text[3:].replace("half an hour", "30 minutes").replace("half hour", "30 minutes")
    minutes = 0.0
    for amount, unit in _RELATIVE_PAIR.findall(rest):
        value = float(_NUMBER_WORDS.get(amount, amount))
        minutes += value * _UNIT_MINUTES[unit]
    leftover = _RELATIVE_PAIR.sub(" ", rest).replace("and", " ").strip()
    if minutes <= 0 or leftover:
        return None
    return now + timedelta(minutes=minutes)


def _parse_absolute(text: str, now: datetime) -> datetime | None:
    clock, text = _extract_clock(text)

    part_hour: int | None = None
    if re.search(r"\bnoon\b|\bmidday\b", text):
        clock = clock or time(12, 0)
        text = re.sub(r"\bnoon\b|\bmidday\b", " ", text)
    if re.search(r"\bmidnight\b", text):
        clock = clock or time(0, 0)
        text = re.sub(r"\bmidnight\b", " ", text)
        if not _mentions_day(text):
            text += " tomorrow"
    for word, hour in _PART_OF_DAY.items():
        if re.search(rf"\b{word}\b", text):
            part_hour = hour
            text = re.sub(rf"\b{word}\b", " today" if word == "tonight" else " ", text)
    if clock is not None and part_hour is not None and part_hour >= 15 and clock.hour < 12:
        clock = clock.replace(hour=clock.hour + 12)  # "tonight at 9" -> 21:00

    day, explicit_day = _extract_day(text, now)
    if day is None:
        return None
    if clock is None:
        clock = time(part_hour if part_hour is not None else DEFAULT_HOUR, 0)
        if not explicit_day and part_hour is None:
            return None  # neither a day nor a time: nothing to go on
    result = datetime.combine(day.date, clock, tzinfo=now.tzinfo)
    if not explicit_day and result <= now:
        result += timedelta(days=1)  # "at 9pm" when it's 10pm means tomorrow
    elif day.weekday_based and result <= now:
        result += timedelta(days=7)  # "friday 9am" on Friday at 10am means next week
    elif day.year_flexible and result <= now:
        result = result.replace(year=result.year + 1)
    return result


@dataclass(frozen=True)
class _Day:
    date: date
    weekday_based: bool = False
    year_flexible: bool = False


def _extract_clock(text: str) -> tuple[time | None, str]:
    match = _CLOCK_AMPM.search(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None, text
        hour = hour % 12 + (12 if match.group(3) == "p" else 0)
        return time(hour, minute), _cut(text, match)
    match = _CLOCK_24H.search(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            return None, text
        return time(hour, minute), _cut(text, match)
    match = _CLOCK_BARE.search(text)
    if match:
        hour = int(match.group(1))
        if hour > 23:
            return None, text
        if 1 <= hour <= 7:
            hour += 12  # "at 5" almost always means 5 pm for a reminder
        return time(hour, 0), _cut(text, match)
    return None, text


def _extract_day(text: str, now: datetime) -> tuple[_Day | None, bool]:
    today = now.date()
    match = _ISO_DATE.search(text)
    if match:
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None, False
        return (_Day(day) if _only_filler(_cut(text, match)) else None), True
    for pattern, month_group, day_group in ((_MONTH_DAY, 1, 2), (_DAY_MONTH, 2, 1)):
        match = pattern.search(text)
        if match:
            month = _MONTHS.index(match.group(month_group)[:3]) + 1
            year = int(match.group(3)) if match.group(3) else today.year
            try:
                day = date(year, month, int(match.group(day_group)))
            except ValueError:
                return None, False
            if not _only_filler(_cut(text, match)):
                return None, False
            return _Day(day, year_flexible=match.group(3) is None), True

    words = [w for w in text.split() if w not in _FILLER]
    phrase = " ".join(words)
    if phrase == "":
        return _Day(today), False
    if phrase == "today":
        return _Day(today), True
    if phrase in {"tomorrow", "tmrw", "tmr"}:
        return _Day(today + timedelta(days=1)), True
    if phrase == "day after tomorrow":
        return _Day(today + timedelta(days=2)), True
    for index, name in enumerate(_WEEKDAYS):
        if phrase in {name, name[:3]}:
            ahead = (index - today.weekday()) % 7
            return _Day(today + timedelta(days=ahead), weekday_based=True), True
    return None, False


def _mentions_day(text: str) -> bool:
    return bool(re.search(r"\b(today|tomorrow|tmrw|" + "|".join(_WEEKDAYS) + r")\b", text))


def _only_filler(text: str) -> bool:
    return all(w in _FILLER for w in text.split())


def _cut(text: str, match: re.Match[str]) -> str:
    return " ".join((text[: match.start()] + " " + text[match.end() :]).split())


# --- storage -----------------------------------------------------------------


@dataclass(frozen=True)
class Reminder:
    """One stored reminder."""

    id: str
    text: str
    due: datetime
    created: datetime
    fired_at: datetime | None = None
    """When the scheduler delivered it; None while it's still pending."""


class ReminderStoreError(Exception):
    """The reminders file exists but can't be read; it is left untouched."""


_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """One lock per file, shared by every store instance (the tools and the scheduler)."""
    key = path.resolve()
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


class ReminderStore:
    """Reminders persisted to a local JSON file (written atomically)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = _lock_for(self.path)

    def add(self, text: str, due: datetime, *, now: datetime) -> Reminder:
        """Append a reminder and save."""
        reminder = Reminder(id=uuid.uuid4().hex[:8], text=text.strip(), due=due, created=now)
        with self._lock:
            reminders = self._load()
            reminders.append(reminder)
            self._save(reminders)
        return reminder

    def all(self) -> list[Reminder]:
        """Every stored reminder, soonest first."""
        with self._lock:
            return sorted(self._load(), key=lambda r: r.due)

    def claim_due(self, now: datetime) -> list[Reminder]:
        """Mark every pending reminder due by `now` as fired, save, and return them.

        Marking happens before anyone delivers them, so a reminder is announced
        at most once, even if Jarvis crashes halfway through announcing it.
        """
        with self._lock:
            reminders = self._load()
            due_ids = {r.id for r in reminders if r.fired_at is None and _aware(r.due, now) <= now}
            if not due_ids:
                return []
            updated = [replace(r, fired_at=now) if r.id in due_ids else r for r in reminders]
            self._save(updated)
        return sorted((r for r in updated if r.id in due_ids), key=lambda r: r.due)

    def next_pending(self, now: datetime) -> datetime | None:
        """When the next not-yet-fired reminder is due (None if there are none)."""
        pending = [_aware(r.due, now) for r in self.all() if r.fired_at is None]
        return min(pending) if pending else None

    def _load(self) -> list[Reminder]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [
                Reminder(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    due=datetime.fromisoformat(item["due"]),
                    created=datetime.fromisoformat(item["created"]),
                    fired_at=datetime.fromisoformat(item["fired"]) if item.get("fired") else None,
                )
                for item in data["reminders"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReminderStoreError(
                f"can't read reminders file {self.path} ({type(exc).__name__}: {exc}); "
                "fix or move it, it was not modified"
            ) from exc

    def _save(self, reminders: list[Reminder]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reminders": [
                {
                    "id": r.id,
                    "text": r.text,
                    "due": r.due.isoformat(),
                    "created": r.created.isoformat(),
                }
                | ({"fired": r.fired_at.isoformat()} if r.fired_at else {})
                for r in reminders
            ]
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)


def _aware(moment: datetime, now: datetime) -> datetime:
    """Treat a hand-edited timestamp without a timezone as local time."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=now.tzinfo)
