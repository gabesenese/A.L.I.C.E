"""Reminders that actually remind.

"remind me to call mom at 5pm" reached no plugin at all: the user was told "I
couldn't get a result for that", nothing was stored, and nothing would have fired.
This keeps timed reminders on disk, and a watcher delivers them into the chat when
they come due, including any that came due while Alice was closed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "data/reminders.json"


def reminders_path() -> Path:
    return Path(os.getenv("ALICE_REMINDERS_PATH") or _DEFAULT_PATH)


@dataclass
class Reminder:
    id: str
    text: str
    due: str
    created: str
    fired: bool = False
    # "daily", "weekdays" or "weekly:<0-6>"; a repeating reminder is re-armed, not retired.
    repeat: str = ""
    # "timer" for "set a timer for 10 minutes", said differently when it is up.
    kind: str = ""

    @property
    def due_at(self) -> datetime:
        return datetime.fromisoformat(self.due)


# -- understanding the request ------------------------------------------------

_COMMAND_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)\s+alice[,!]?\s*)?(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?"
    r"(?:remind\s+me(?:\s+(?:to|about|that))?|set\s+(?:a\s+|an?\s+)?reminder(?:\s+(?:to|for|about|that))?"
    r"|(?:don'?t|do\s+not)\s+(?:let\s+me\s+)?forget(?:\s+(?:to|about))?)\b\s*",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "ten": 10.0,
    "fifteen": 15.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty-five": 45.0,
}
_UNITS = {"sec": 1 / 60, "min": 1.0, "hour": 60.0, "hr": 60.0, "day": 1440.0}
_IN_RE = re.compile(
    r"\bin\s+(half\s+an|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|forty-five|\d+(?:\.\d+)?)\s*"
    r"(sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_AT_RE = re.compile(r"\b(?:at|by)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?=\W|$)", re.IGNORECASE)
_BARE_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?=\W|$)", re.IGNORECASE)
_NOON_RE = re.compile(r"\b(?:at\s+)?(noon|midnight)\b", re.IGNORECASE)
_DAY_RE = re.compile(r"\b(tomorrow|tonight|today|this\s+(?:morning|afternoon|evening))\b", re.IGNORECASE)
_DEFAULT_HOURS = {"tomorrow": 9, "today": 18, "tonight": 20, "morning": 9, "afternoon": 14, "evening": 19}


def _clock(hour: int, minute: int, meridiem: str) -> Tuple[int, int]:
    mer = meridiem.replace(".", "").lower()
    if mer == "pm" and hour < 12:
        hour += 12
    if mer == "am" and hour == 12:
        hour = 0
    return hour, minute


def _next_occurrence(now: datetime, hour: int, minute: int, meridiem: str, evening: bool) -> datetime:
    """The soonest future time matching "5" or "5pm", on or after today."""
    if meridiem or hour > 12 or hour == 0:
        hour, minute = _clock(hour, minute, meridiem)
        candidates = [now.replace(hour=hour, minute=minute, second=0, microsecond=0)]
    else:
        am = now.replace(hour=hour % 12, minute=minute, second=0, microsecond=0)
        pm = am + timedelta(hours=12)
        candidates = [pm] if evening else [am, pm]
    for moment in sorted(candidates):
        if moment > now:
            return moment
    return min(candidates) + timedelta(days=1)


def parse_reminder(text: str, now: Optional[datetime] = None) -> Optional[Tuple[str, Optional[datetime]]]:
    """(what to remind, when) from a request, or None when it is not a reminder.

    The time is None when the request names none; the caller asks for it.
    """
    now = now or datetime.now()
    raw = str(text or "").strip()
    command = _COMMAND_RE.match(raw)
    if not command:
        return None
    body = raw[command.end() :]
    due: Optional[datetime] = None
    spans: List[Tuple[int, int]] = []

    offset = _IN_RE.search(body)
    if offset:
        amount_text = offset.group(1).lower()
        amount = 0.5 if amount_text.startswith("half") else _WORD_NUMBERS.get(amount_text)
        if amount is None:
            amount = float(amount_text)
        unit = next(minutes for key, minutes in _UNITS.items() if offset.group(2).lower().startswith(key))
        due = now + timedelta(minutes=amount * unit)
        spans.append(offset.span())

    day = _DAY_RE.search(body)
    day_word = (day.group(1).lower().split()[-1] if day else "") or ""
    if day:
        spans.append(day.span())

    at = _AT_RE.search(body) or _BARE_TIME_RE.search(body)
    noon = _NOON_RE.search(body)
    if due is None and (at or noon or day):
        base = now + timedelta(days=1) if day_word == "tomorrow" else now
        if at:
            spans.append(at.span())
            hour, minute = int(at.group(1)), int(at.group(2) or 0)
            meridiem = at.group(3) or ""
            if day_word == "tomorrow":
                if meridiem or hour > 12:
                    hour, minute = _clock(hour, minute, meridiem)
                elif hour < 7:
                    hour += 12
                due = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                evening = day_word in {"tonight", "evening", "afternoon"}
                due = _next_occurrence(now, hour, minute, meridiem, evening)
        elif noon:
            spans.append(noon.span())
            hour = 12 if noon.group(1).lower() == "noon" else 0
            due = base.replace(hour=hour, minute=0, second=0, microsecond=0)
            if due <= now:
                due += timedelta(days=1)
        else:
            due = base.replace(hour=_DEFAULT_HOURS.get(day_word, 9), minute=0, second=0, microsecond=0)
            if due <= now:
                due += timedelta(days=1)

    task = body
    for start, end in sorted(spans, reverse=True):
        task = task[:start] + " " + task[end:]
    task = re.sub(r"\s+", " ", task).strip(" ,.;:!-")
    task = re.sub(r"^(?:to|about|that|for|on)\s+", "", task, flags=re.IGNORECASE)
    task = re.sub(r"\s+(?:to|at|on|for|by)$", "", task, flags=re.IGNORECASE).strip(" ,.;:!-")
    return (task, due) if task else None


# -- repeating reminders and timers ------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_REPEAT_RE = re.compile(
    r"\b(?:every\s+day|each\s+day|daily|every\s+(?P<part>morning|afternoon|evening|night)"
    r"|every\s+weekday|on\s+weekdays|weekdays"
    r"|every\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|on\s+(?P<weekdays>monday|tuesday|wednesday|thursday|friday|saturday|sunday)s)\b",
    re.IGNORECASE,
)
_PART_OF_DAY = {"morning": "this morning", "afternoon": "this afternoon", "evening": "this evening", "night": "tonight"}


@dataclass
class ReminderRequest:
    task: str
    due: Optional[datetime]
    repeat: str = ""


def parse_request(text: str, now: Optional[datetime] = None) -> Optional[ReminderRequest]:
    """Like parse_reminder, and also hears "every day", "on weekdays", "every Monday".

    "remind me every day at 8am to take my pills" was one reminder, for
    tomorrow, to "every day to take my pills".
    """
    now = now or datetime.now()
    raw = str(text or "")
    repeat = ""
    match = _REPEAT_RE.search(raw)
    if match:
        weekday = (match.group("weekday") or match.group("weekdays") or "").lower()
        if weekday:
            repeat = f"weekly:{_WEEKDAYS.index(weekday)}"
        elif "weekday" in match.group(0).lower():
            repeat = "weekdays"
        else:
            repeat = "daily"
        # "every morning" still says when: this morning's default hour.
        part = _PART_OF_DAY.get((match.group("part") or "").lower(), "")
        raw = f"{raw[: match.start()]} {part} {raw[match.end() :]}"
    parsed = parse_reminder(raw, now)
    if parsed is None:
        return None
    task, due = parsed
    if repeat and due is None:
        due = now.replace(hour=9, minute=0, second=0, microsecond=0)
    bare = _AT_RE.search(raw)
    if repeat and due is not None and bare and not bare.group(3) and int(bare.group(1)) <= 12:
        # "every Monday at 9" is the morning, as "tomorrow at 9" is: the soonest
        # 9 o'clock from now would be tonight's, on the wrong day anyway.
        hour = int(bare.group(1)) % 12
        hour += 12 if hour < 7 else 0
        due = due.replace(hour=hour, minute=int(bare.group(2) or 0))
    if repeat and due is not None:
        while due <= now or not _repeats_on(repeat, due):
            due += timedelta(days=1)
    return ReminderRequest(task=task, due=due, repeat=repeat)


def _repeats_on(repeat: str, moment: datetime) -> bool:
    if repeat == "weekdays":
        return moment.weekday() < 5
    if repeat.startswith("weekly:"):
        return moment.weekday() == int(repeat.split(":", 1)[1])
    return True


def next_repeat(due: datetime, repeat: str, now: datetime) -> datetime:
    """The next time a repeating reminder is due after ``now``."""
    following = due + timedelta(days=1)
    while following <= now or not _repeats_on(repeat, following):
        following += timedelta(days=1)
    return following


def describe_repeat(repeat: str, due: datetime) -> str:
    if repeat == "weekdays":
        return f"every weekday at {clock_time(due)}"
    if repeat.startswith("weekly:"):
        return f"every {_WEEKDAYS[int(repeat.split(':', 1)[1])].capitalize()} at {clock_time(due)}"
    return f"every day at {clock_time(due)}"


_TO_YOU = (
    (re.compile(r"\bmyself\b", re.IGNORECASE), "yourself"),
    (re.compile(r"\bmy\b", re.IGNORECASE), "your"),
    (re.compile(r"\bmine\b", re.IGNORECASE), "yours"),
    (re.compile(r"\bI'm\b", re.IGNORECASE), "you're"),
    (re.compile(r"\bI\s+am\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bme\b", re.IGNORECASE), "you"),
    (re.compile(r"\bI\b"), "you"),
)


def said_back(text: str) -> str:
    """What to remind him of, in her words: "take my pills" is "take your pills"."""
    out = str(text or "")
    for pattern, replacement in _TO_YOU:
        out = pattern.sub(replacement, out)
    return out


_TIMER_WORD_RE = re.compile(r"\btimer\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"(?P<n>\d+(?:\.\d+)?|half\s+an|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|forty-five)\s*-?\s*"
    r"(?P<unit>sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_TIMER_LABEL_RE = re.compile(r"\bfor\s+(?:the\s+|my\s+)?(?P<label>[a-z][a-z' -]{1,30}?)\s*[.!?]*$", re.IGNORECASE)


def parse_timer(text: str) -> Optional[Tuple[float, str]]:
    """(minutes, label) from "set a timer for 10 minutes for the pasta", or None."""
    raw = str(text or "")
    if not _TIMER_WORD_RE.search(raw):
        return None
    duration = _DURATION_RE.search(raw)
    if not duration:
        return None
    amount_text = duration.group("n").lower()
    amount = 0.5 if amount_text.startswith("half") else _WORD_NUMBERS.get(amount_text)
    if amount is None:
        amount = float(amount_text)
    unit = next(minutes for key, minutes in _UNITS.items() if duration.group("unit").lower().startswith(key))
    label_match = _TIMER_LABEL_RE.search(raw[duration.end() :])
    label = label_match.group("label").strip() if label_match else ""
    if _DURATION_RE.fullmatch(label) or label.lower() in {"me", "us"}:
        label = ""
    return amount * unit, label


def describe_duration(minutes: float) -> Tuple[str, str]:
    """("10 minutes", "10-minute") for a timer of ``minutes``."""
    if minutes < 1:
        seconds = int(round(minutes * 60))
        return f"{seconds} seconds", f"{seconds}-second"
    if minutes >= 60 and minutes % 60 == 0:
        hours = int(minutes // 60)
        return (f"{hours} hour" + ("s" if hours != 1 else "")), f"{hours}-hour"
    whole = int(minutes) if float(minutes).is_integer() else minutes
    return (f"{whole} minute" + ("s" if whole != 1 else "")), f"{whole}-minute"


def clock_time(due: datetime) -> str:
    return due.strftime("%I:%M %p").lstrip("0")


def describe_day(due: datetime, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    if due.date() == now.date():
        return "today"
    if due.date() == (now + timedelta(days=1)).date():
        return "tomorrow"
    return f"on {due.strftime('%A %B')} {due.day}"


def describe_time(due: datetime, now: Optional[datetime] = None) -> str:
    day = describe_day(due, now)
    return f"at {clock_time(due)}" if day == "today" else f"{day} at {clock_time(due)}"


# -- keeping them -------------------------------------------------------------


class ReminderStore:
    """Reminders on disk, read fresh on every call so every holder sees the same list."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else reminders_path()
        self._lock = threading.Lock()

    def _load(self) -> List[Reminder]:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            return [Reminder(**row) for row in rows if isinstance(row, dict)]
        except Exception as exc:
            logger.warning("Could not read reminders: %s", exc)
            return []

    def _save(self, reminders: List[Reminder]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in reminders], indent=2), encoding="utf-8")

    def add(self, text: str, due: datetime, repeat: str = "", kind: str = "") -> Reminder:
        with self._lock:
            reminders = self._load()
            reminder = Reminder(
                id=uuid.uuid4().hex[:12],
                text=str(text).strip(),
                due=due.replace(microsecond=0).isoformat(),
                created=datetime.now().replace(microsecond=0).isoformat(),
                repeat=str(repeat or ""),
                kind=str(kind or ""),
            )
            reminders.append(reminder)
            self._save(reminders)
            return reminder

    def move(self, reminder_id: str, due: datetime) -> Optional[Reminder]:
        """Give a pending reminder a new time, keeping what it is about."""
        with self._lock:
            reminders = self._load()
            for reminder in reminders:
                if reminder.id == reminder_id and not reminder.fired:
                    reminder.due = due.replace(microsecond=0).isoformat()
                    self._save(reminders)
                    return reminder
            return None

    def pending(self) -> List[Reminder]:
        return sorted((r for r in self._load() if not r.fired), key=lambda r: r.due)

    def take_due(self, now: Optional[datetime] = None) -> List[Reminder]:
        """Due reminders, marked delivered in the same step so none fires twice."""
        now = now or datetime.now()
        with self._lock:
            reminders = self._load()
            due = [r for r in reminders if not r.fired and r.due_at <= now]
            delivered = [replace(r) for r in due]
            if due:
                for reminder in due:
                    if reminder.repeat:
                        # Said now, and set again for its next time.
                        reminder.due = next_repeat(reminder.due_at, reminder.repeat, now).isoformat()
                    else:
                        reminder.fired = True
                self._save(reminders)
            return delivered

    def last_fired(self) -> Optional[Reminder]:
        fired = [r for r in self._load() if r.fired]
        return max(fired, key=lambda r: r.due) if fired else None

    def cancel(self, words: str = "") -> List[Reminder]:
        """Cancel the pending reminders whose text shares a word with ``words``;
        with no words, the next one due."""
        wanted = {w for w in re.findall(r"[a-z0-9']{3,}", str(words or "").lower())} - {
            "cancel",
            "delete",
            "remove",
            "reminder",
            "reminders",
            "the",
            "my",
            "about",
        }
        with self._lock:
            reminders = self._load()
            pending = sorted((r for r in reminders if not r.fired), key=lambda r: r.due)
            if wanted:
                chosen = [r for r in pending if wanted & set(re.findall(r"[a-z0-9']{3,}", r.text.lower()))]
            else:
                chosen = pending[:1]
            ids = {r.id for r in chosen}
            if ids:
                self._save([r for r in reminders if r.id not in ids])
            return chosen


# -- delivering them ----------------------------------------------------------


def reminder_message(reminder: Reminder, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    if reminder.kind == "timer":
        return f"Time's up. That was your {reminder.text}."
    text = said_back(reminder.text)
    if now - reminder.due_at > timedelta(minutes=2):
        return f"Reminder: {text}. (It was due {describe_time(reminder.due_at, reminder.due_at)}.)"
    return f"Reminder: {text}."


class ReminderWatcher:
    """Checks for due reminders and hands each one to ``deliver`` exactly once."""

    def __init__(
        self,
        store: ReminderStore,
        deliver: Callable[[str], None],
        interval: float = 15.0,
        ready: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.store = store
        self.deliver = deliver
        self.interval = float(interval)
        # Delivery waits while a turn is being answered, so a reminder is never
        # threaded into the middle of that turn's transcript entry.
        self.ready = ready
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check(self, now: Optional[datetime] = None) -> List[Reminder]:
        now = now or datetime.now()
        if self.ready is not None and not self.ready():
            return []
        due = self.store.take_due(now)
        for reminder in due:
            try:
                self.deliver(reminder_message(reminder, now))
            except Exception as exc:
                logger.warning("Could not deliver a reminder: %s", exc)
        return due

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check()
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ReminderWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# -- the agenda ---------------------------------------------------------------


def agenda_window(text: str, now: Optional[datetime] = None) -> Tuple[datetime, datetime, str]:
    """The stretch of time an agenda question asks about: today, tomorrow or this week."""
    now = now or datetime.now()
    low = str(text or "").lower()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "tomorrow" in low:
        return midnight + timedelta(days=1), midnight + timedelta(days=2), "tomorrow"
    if "week" in low:
        return now, midnight + timedelta(days=7), "this week"
    return now, midnight + timedelta(days=1), "today"


def _note_due(note: Any) -> Optional[Tuple[datetime, bool]]:
    """When an open note falls due, and whether that includes a time of day."""
    if getattr(note, "archived", False):
        return None
    items = getattr(note, "checklist_items", None) or []
    if items and all(bool((item or {}).get("checked")) for item in items if isinstance(item, dict)):
        return None
    raw = str(getattr(note, "due_date", "") or "").strip()
    if not raw:
        return None
    try:
        due = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if due.tzinfo is not None:
        due = due.astimezone().replace(tzinfo=None)
    if len(raw) <= 10:
        # A bare date is due by the end of that day, and has no time to read out.
        return due.replace(hour=23, minute=59), False
    return due, True


def agenda(
    store: ReminderStore, notes: Iterable[Any], start: datetime, end: datetime
) -> List[Tuple[datetime, str, bool]]:
    """Reminders and notes falling due between ``start`` and ``end``, soonest
    first, each with whether it has a time of day or only a date."""
    items = [(r.due_at, said_back(r.text), True) for r in store.pending() if start <= r.due_at < end]
    for note in notes or []:
        found = _note_due(note)
        if found is not None and start <= found[0] < end:
            items.append((found[0], f'"{getattr(note, "title", "")}" is due', found[1]))
    return sorted(items)
