"""Setting, listing and cancelling reminders.

No plugin handled reminder:set, so "remind me to call mom at 5pm" was answered
"I couldn't get a result for that" and nothing was ever stored.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

from ai.planning.reminders import ReminderStore, describe_time, parse_reminder
from ai.plugins.plugin_system import PluginInterface


class ReminderPlugin(PluginInterface):
    def __init__(self, store: Optional[ReminderStore] = None) -> None:
        super().__init__()
        self.name = "ReminderPlugin"
        self.description = "Set, list and cancel timed reminders"
        self.capabilities = ["reminders"]
        self.store = store or ReminderStore()

    def initialize(self) -> bool:
        return True

    def can_handle(self, intent: str, entities: Dict) -> bool:
        return str(intent or "").startswith("reminder:")

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        now = datetime.now()
        action = str(intent or "").split(":", 1)[-1]
        if action == "list":
            return self._list(now)
        if action == "cancel":
            cancelled = self.store.cancel(query)
            if not cancelled:
                return {"success": True, "response": "There's no reminder like that to cancel.", "data": {}}
            names = "; ".join(r.text for r in cancelled)
            return {"success": True, "response": f"Cancelled: {names}.", "data": {"cancelled": names}}

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
        self.store.add(task, due)
        when = describe_time(due, now)
        return {
            "success": True,
            "response": f"Okay, I'll remind you {about} {task} {when}.",
            "data": {"task": task, "due": due.isoformat(timespec="minutes"), "when": when},
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

    def shutdown(self) -> None:
        return None
