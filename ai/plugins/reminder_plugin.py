"""Setting, listing and cancelling reminders.

No plugin handled reminder:set, so "remind me to call mom at 5pm" was answered
"I couldn't get a result for that" and nothing was ever stored.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ai.core.followups import RESCHEDULE_RE
from ai.planning.reminders import (
    ReminderStore,
    agenda,
    agenda_window,
    clock_time,
    describe_day,
    describe_time,
    parse_reminder,
)
from ai.plugins.plugin_system import PluginInterface


class ReminderPlugin(PluginInterface):
    def __init__(self, store: Optional[ReminderStore] = None, notes: Optional[Callable[[], List[Any]]] = None) -> None:
        super().__init__()
        # Notes with due dates belong on the agenda too.
        self.notes = notes
        self.name = "ReminderPlugin"
        self.description = "Set, list and cancel timed reminders"
        self.capabilities = ["reminders"]
        self.store = store or ReminderStore()
        # The reminder just set, so "actually make it 6" knows which one to move.
        self._last_set_id: Optional[str] = None

    def initialize(self) -> bool:
        return True

    def can_handle(self, intent: str, entities: Dict) -> bool:
        return str(intent or "").startswith("reminder:")

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        now = datetime.now()
        action = str(intent or "").split(":", 1)[-1]
        if action == "list":
            return self._list(now)
        if action == "agenda":
            return self._agenda(query, now)
        if action == "cancel":
            cancelled = self.store.cancel(query)
            if not cancelled:
                return {"success": True, "response": "There's no reminder like that to cancel.", "data": {}}
            names = "; ".join(r.text for r in cancelled)
            return {"success": True, "response": f"Cancelled: {names}.", "data": {"cancelled": names}}

        moving = RESCHEDULE_RE.match(str(query or "").strip())
        if moving:
            moved = self._reschedule(moving.group("when"), now)
            if moved is not None:
                return moved

        parsed = parse_reminder(query, now)
        if parsed is None:
            return {"success": False, "response": "I couldn't tell what to remind you about."}
        task, due = parsed
        if task.lower() in {"again", "me again", "it again", "that again", "about it again", "it", "that", "about it"}:
            last = self.store.last_fired()
            if last is None:
                return {"success": True, "response": "Remind you of what?", "data": {"needs_task": True}}
            task = last.text
        about = "about" if re.search(r"\b(?:remind\s+me|reminder)\s+about\b", str(query or ""), re.I) else "to"
        if due is None:
            # Asked rather than guessed: a reminder at the wrong time is worse than none.
            return {
                "success": True,
                "response": f"When should I remind you {about} {task}?",
                "data": {"task": task, "needs_time": True},
            }
        self._last_set_id = self.store.add(task, due).id
        when = describe_time(due, now)
        return {
            "success": True,
            "response": f"Okay, I'll remind you {about} {task} {when}.",
            "data": {"task": task, "due": due.isoformat(timespec="minutes"), "when": when},
        }

    def _reschedule(self, when: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Move the reminder just set: "actually make it 6" after "remind me at 5pm
        to call mom". Set again from scratch, it would have left the 5pm one."""
        last = next((r for r in self.store.pending() if r.id == self._last_set_id), None)
        if last is None:
            return None
        spoken = " ".join(str(when or "").split())
        if re.fullmatch(r"\d{1,2}(?::\d{2})?", spoken):
            spoken = "at " + spoken
        parsed = parse_reminder(f"remind me to {last.text} {spoken}", now)
        due = parsed[1] if parsed else None
        if due is None:
            return {"success": True, "response": "When should I move it to?", "data": {"needs_time": True}}
        moved_to = due
        # "make it 6" keeps the day, and the afternoon, of the reminder it changes.
        if not re.search(
            r"\b(?:today|tonight|tomorrow|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b|\bin\s+\w+", spoken, re.I
        ):
            moved_to = datetime.combine(last.due_at.date(), moved_to.time())
        if moved_to.hour < 12 <= last.due_at.hour and not re.search(
            r"\b(?:am|pm|a\.m\.|p\.m\.|noon|midnight|morning|afternoon|evening|tonight)\b", spoken, re.I
        ):
            moved_to += timedelta(hours=12)
        if moved_to <= now:
            moved_to = due
        self.store.move(last.id, moved_to)
        return {
            "success": True,
            "response": f"Moved it. I'll remind you to {last.text} {describe_time(moved_to, now)}.",
            "data": {"task": last.text, "due": moved_to.isoformat(timespec="minutes"), "moved": True},
        }

    def _list(self, now: datetime) -> Dict[str, Any]:
        pending = self.store.pending()
        if not pending:
            return {"success": True, "response": "You have no reminders set.", "data": {"count": 0}}
        items = [f"{r.text} {describe_time(r.due_at, now)}" for r in pending]
        return {
            "success": True,
            "response": "Your reminders: " + "; ".join(items) + ".",
            "data": {"count": len(items), "reminders": items},
        }

    def _agenda(self, query: str, now: datetime) -> Dict[str, Any]:
        """What he has on, from what Alice actually knows: his reminders and the
        notes falling due. Answered by the model with nothing in front of it,
        "what's on my schedule today?" could only be made up."""
        start, end, label = agenda_window(query, now)
        try:
            notes = list(self.notes() if self.notes else [])
        except Exception:
            notes = []
        items = agenda(self.store, notes, start, end)
        if not items:
            return {
                "success": True,
                "response": f"Nothing on your reminders or notes for {label}.",
                "data": {"count": 0, "window": label},
            }
        # "For tomorrow: pay rent at 9:00 AM", not "pay rent tomorrow at 9:00 AM".
        one_day = label in {"today", "tomorrow"}
        lines = []
        for due, text, timed in items:
            if timed:
                when = f"at {clock_time(due)}" if one_day else describe_time(due, now)
            else:
                when = "" if one_day else describe_day(due, now)
            lines.append(f"{text} {when}".strip())
        return {
            "success": True,
            "response": f"For {label}: " + "; ".join(lines) + ".",
            "data": {"count": len(lines), "window": label, "items": lines},
        }

    def shutdown(self) -> None:
        return None
