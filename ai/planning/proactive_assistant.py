"""
Proactive Assistant for A.L.I.C.E
Anticipates needs, provides reminders, and delivers proactive information.
Makes A.L.I.C.E feel like a real assistant, not just a chatbot.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Thread, Event
import time

logger = logging.getLogger(__name__)

_REMINDERS_PATH = Path("app/data/reminders.json")


@dataclass
class Reminder:
    """A proactive reminder"""

    reminder_id: str
    message: str
    trigger_time: datetime
    priority: str = "normal"  # low, normal, high, urgent
    context: Dict[str, Any] = field(default_factory=dict)
    delivered: bool = False


@dataclass
class IncompleteTask:
    """A task that was started but not completed"""

    task_id: str
    description: str
    started_at: datetime
    last_activity: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class ProactiveAssistant:
    """
    Proactive assistant that:
    - Tracks incomplete tasks and follows up
    - Provides time-based reminders (meetings, deadlines)
    - Delivers proactive information (morning briefings, summaries)
    - Anticipates needs based on patterns
    """

    def __init__(self, world_state=None, calendar_plugin=None, notes_plugin=None):
        self.world_state = world_state
        self.calendar_plugin = calendar_plugin
        self.notes_plugin = notes_plugin

        self.reminders: Dict[str, Reminder] = {}
        self.incomplete_tasks: Dict[str, IncompleteTask] = {}

        self.running = False
        self._stop_event = Event()
        self._check_thread: Optional[Thread] = None

        self.notification_callback: Optional[callable] = None

        self._load_reminders()
        logger.info("[ProactiveAssistant] Initialized")

    def set_notification_callback(self, callback: callable):
        """Set callback for delivering notifications"""
        self.notification_callback = callback

    def start(self):
        """Start proactive monitoring"""
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._check_thread = Thread(target=self._monitor_loop, daemon=True)
        self._check_thread.start()
        logger.info("[ProactiveAssistant] Started monitoring")

    def stop(self):
        """Stop proactive monitoring"""
        self.running = False
        self._stop_event.set()
        if self._check_thread:
            self._check_thread.join(timeout=2)
        logger.info("[ProactiveAssistant] Stopped")

    def _monitor_loop(self):
        """Main monitoring loop - checks for reminders and proactive actions"""
        while self.running and not self._stop_event.is_set():
            try:
                self._check_reminders()
                self._check_calendar_reminders()
                self._check_incomplete_tasks()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"[ProactiveAssistant] Error in monitor loop: {e}")
                time.sleep(60)

    def _check_reminders(self):
        """Check and deliver due reminders"""
        now = datetime.now()
        to_deliver = []

        for rid, reminder in self.reminders.items():
            if not reminder.delivered and now >= reminder.trigger_time:
                to_deliver.append(reminder)

        for reminder in to_deliver:
            self._deliver_reminder(reminder)
            reminder.delivered = True

    def _check_calendar_reminders(self):
        """Check calendar for upcoming events and create reminders"""
        if not self.calendar_plugin:
            return

        try:
            # Get events in next 24 hours
            now = datetime.now()
            # Calendar plugin might have get_upcoming_events or list_today
            if hasattr(self.calendar_plugin, "get_upcoming_events"):
                events = self.calendar_plugin.get_upcoming_events(hours=24)
            elif hasattr(self.calendar_plugin, "list_today"):
                events = self.calendar_plugin.list_today() or []
            else:
                events = []

            for event in events:
                event_time = event.get("start_time")
                if not event_time:
                    continue

                if isinstance(event_time, str):
                    from dateparser import parse

                    event_time = parse(event_time)

                if not event_time:
                    continue

                # Create reminder 15 minutes before
                reminder_time = event_time - timedelta(minutes=15)

                if now < reminder_time <= now + timedelta(hours=1):
                    # Check if we already have a reminder for this
                    event_id = event.get("id") or event.get("summary", "")
                    reminder_id = f"cal_{event_id}_{int(reminder_time.timestamp())}"

                    if reminder_id not in self.reminders:
                        self.add_reminder(
                            reminder_id=reminder_id,
                            message=f"Upcoming: {event.get('summary', 'Event')} in 15 minutes",
                            trigger_time=reminder_time,
                            priority="high",
                            context={"event": event},
                        )
        except Exception as e:
            logger.debug(f"[ProactiveAssistant] Calendar check error: {e}")

    def _check_incomplete_tasks(self):
        """Check for tasks that were started but not completed"""
        now = datetime.now()
        stale_threshold = timedelta(hours=2)

        for tid, task in list(self.incomplete_tasks.items()):
            if now - task.last_activity > stale_threshold:
                # Follow up on stale incomplete task
                if self.notification_callback:
                    self.notification_callback(
                        f" You started '{task.description}' earlier. Want to continue?",
                        priority="normal",
                    )
                # Update last activity to avoid spam
                task.last_activity = now

    def _deliver_reminder(self, reminder: Reminder):
        """Deliver a reminder via notification callback"""
        if self.notification_callback:
            priority_map = {
                "low": "normal",
                "normal": "normal",
                "high": "high",
                "urgent": "critical",
            }
            from ai.infrastructure.event_bus import EventPriority

            priority = EventPriority[
                priority_map.get(reminder.priority, "normal").upper()
            ]
            self.notification_callback(reminder.message, priority)
            logger.info(
                f"[ProactiveAssistant] Delivered reminder: {reminder.message[:50]}"
            )

    def add_reminder(
        self,
        reminder_id: str,
        message: str,
        trigger_time: datetime,
        priority: str = "normal",
        context: Dict = None,
    ):
        """Add a proactive reminder and persist to disk."""
        self.reminders[reminder_id] = Reminder(
            reminder_id=reminder_id,
            message=message,
            trigger_time=trigger_time,
            priority=priority,
            context=context or {},
        )
        self._save_reminders()
        logger.debug(
            f"[ProactiveAssistant] Added reminder: {message[:50]} at {trigger_time}"
        )

    def list_reminders(self) -> List[Reminder]:
        """Return pending (not yet delivered) reminders sorted by trigger time."""
        pending = [r for r in self.reminders.values() if not r.delivered]
        return sorted(pending, key=lambda r: r.trigger_time)

    def cancel_reminder(self, keyword: str) -> bool:
        """Cancel the first pending reminder whose message contains *keyword*.

        Returns True if a reminder was cancelled, False otherwise.
        """
        keyword_lower = keyword.lower()
        for rid, reminder in list(self.reminders.items()):
            if not reminder.delivered and keyword_lower in reminder.message.lower():
                del self.reminders[rid]
                self._save_reminders()
                logger.info(f"[ProactiveAssistant] Cancelled reminder: {reminder.message[:50]}")
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_reminders(self) -> None:
        """Persist reminders to disk so they survive restarts."""
        try:
            _REMINDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "reminder_id": r.reminder_id,
                    "message": r.message,
                    "trigger_time": r.trigger_time.isoformat(),
                    "priority": r.priority,
                    "context": r.context,
                    "delivered": r.delivered,
                }
                for r in self.reminders.values()
            ]
            _REMINDERS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[ProactiveAssistant] Failed to save reminders: {exc}")

    def _load_reminders(self) -> None:
        """Load previously persisted reminders from disk."""
        if not _REMINDERS_PATH.exists():
            return
        try:
            data = json.loads(_REMINDERS_PATH.read_text(encoding="utf-8"))
            for item in data:
                if item.get("delivered"):
                    continue  # skip already-delivered reminders
                self.reminders[item["reminder_id"]] = Reminder(
                    reminder_id=item["reminder_id"],
                    message=item["message"],
                    trigger_time=datetime.fromisoformat(item["trigger_time"]),
                    priority=item.get("priority", "normal"),
                    context=item.get("context", {}),
                    delivered=False,
                )
            logger.info(
                f"[ProactiveAssistant] Loaded {len(self.reminders)} pending reminder(s) from disk"
            )
        except Exception as exc:
            logger.warning(f"[ProactiveAssistant] Failed to load reminders: {exc}")

    def track_incomplete_task(
        self, task_id: str, description: str, context: Dict = None
    ):
        """Track a task that was started but may not be complete"""
        self.incomplete_tasks[task_id] = IncompleteTask(
            task_id=task_id,
            description=description,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            context=context or {},
        )

    def mark_task_complete(self, task_id: str):
        """Mark a task as complete"""
        if task_id in self.incomplete_tasks:
            del self.incomplete_tasks[task_id]

    def get_morning_briefing(self) -> Optional[str]:
        """Generate morning briefing with today's agenda"""
        if not self.calendar_plugin:
            return None

        try:
            # Try different calendar plugin methods
            today_events = []
            if hasattr(self.calendar_plugin, "list_today"):
                result = self.calendar_plugin.execute(
                    intent="calendar:list_today", query="today", entities={}, context={}
                )
                if result.get("success") and result.get("events"):
                    today_events = result["events"]
            elif hasattr(self.calendar_plugin, "get_upcoming_events"):
                today_events = self.calendar_plugin.get_upcoming_events(hours=24)

            if not today_events:
                return None

            briefing = " **Today's Schedule:**\n\n"
            for event in today_events[:5]:
                if isinstance(event, dict):
                    summary = event.get("summary") or event.get("title", "Event")
                    start = event.get("start_time") or event.get("start", "")
                else:
                    summary = str(event)
                    start = ""
                if isinstance(start, datetime):
                    start = start.strftime("%I:%M %p")
                elif start:
                    start = str(start)
                briefing += (
                    f"  • {start} - {summary}\n" if start else f"  • {summary}\n"
                )

            return briefing
        except Exception as e:
            logger.debug(f"[ProactiveAssistant] Morning briefing error: {e}")
            return None

    def get_proactive_suggestions(self) -> List[str]:
        """Get proactive suggestions based on current context"""
        suggestions = []

        # Check for overdue notes/tasks
        if self.notes_plugin:
            try:
                notes = self.notes_plugin.manager.get_all_notes()
                overdue = [
                    n
                    for n in notes
                    if n.due_date
                    and datetime.fromisoformat(n.due_date) < datetime.now()
                ]
                if overdue:
                    suggestions.append(f"You have {len(overdue)} overdue note(s)")
            except:
                pass

        # Check for incomplete tasks
        if self.incomplete_tasks:
            suggestions.append(
                f"You have {len(self.incomplete_tasks)} incomplete task(s)"
            )

        return suggestions


_proactive_assistant: Optional[ProactiveAssistant] = None


def get_proactive_assistant(
    world_state=None, calendar_plugin=None, notes_plugin=None
) -> ProactiveAssistant:
    global _proactive_assistant
    if _proactive_assistant is None:
        _proactive_assistant = ProactiveAssistant(
            world_state=world_state,
            calendar_plugin=calendar_plugin,
            notes_plugin=notes_plugin,
        )
    return _proactive_assistant


# ---------------------------------------------------------------------------
# Time-parsing utility (used by main.py when wiring reminder:set intent)
# ---------------------------------------------------------------------------

def parse_reminder_time(text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse a natural-language time expression from *text* and return an
    absolute ``datetime``.  Returns ``None`` when no time could be parsed.

    Supports:
    * Relative: ``in 30 minutes``, ``in 2 hours``, ``in 3 days``
    * Clock time today: ``at 3pm``, ``at 14:30``, ``at 9:30 am``
    * Tomorrow: ``tomorrow at 9am``
    * Named times: ``tonight``, ``this evening``, ``morning``, ``noon``
    """
    base = now or datetime.now()
    text_lower = text.lower()

    # -- relative: "in X minutes/hours/days/weeks" --
    rel = re.search(
        r"\bin\s+(\d+)\s+(minute|min|hour|hr|day|week)s?",
        text_lower,
    )
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2)
        if unit in ("minute", "min"):
            return base + timedelta(minutes=amount)
        if unit in ("hour", "hr"):
            return base + timedelta(hours=amount)
        if unit == "day":
            return base + timedelta(days=amount)
        if unit == "week":
            return base + timedelta(weeks=amount)

    # -- named times --
    named: Dict[str, tuple] = {
        "midnight": (0, 0),
        "this morning": (9, 0),
        "morning": (9, 0),
        "noon": (12, 0),
        "this afternoon": (14, 0),
        "afternoon": (14, 0),
        "this evening": (18, 0),
        "evening": (18, 0),
        "tonight": (20, 0),
        "night": (20, 0),
    }
    for phrase, (h, m) in named.items():
        if phrase in text_lower:
            target = base.replace(hour=h, minute=m, second=0, microsecond=0)
            if target <= base:
                target += timedelta(days=1)
            return target

    # -- tomorrow flag --
    is_tomorrow = "tomorrow" in text_lower

    # -- clock time: "at 3pm", "at 14:30", "at 9:30 am" --
    clock = re.search(
        r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text_lower,
    )
    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        meridiem = clock.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            hour, minute = 9, 0  # fallback if parse produced garbage
        target = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if is_tomorrow:
            target += timedelta(days=1)
        elif target <= base:
            target += timedelta(days=1)
        return target

    if is_tomorrow:
        return (base + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    return None


def make_reminder_id() -> str:
    """Generate a short unique reminder ID."""
    return f"rem_{uuid.uuid4().hex[:8]}"
