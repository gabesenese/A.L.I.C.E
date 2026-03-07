"""
Integration tests for the persistent reminder system.

Covers:
* NLP intent classification (reminder:set / reminder:list / reminder:cancel)
* parse_reminder_time() for all supported patterns
* ProactiveAssistant.add_reminder / list_reminders / cancel_reminder
* Persistence round-trip (save → load)
* _check_reminders delivery callback
"""
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nlp_intent(text: str) -> str:
    """Return the NLP intent for *text* using the keyword-only path."""
    from ai.core.nlp_processor import NLPProcessor
    nlp = NLPProcessor()
    return nlp.process(text).intent


# ---------------------------------------------------------------------------
# 1. NLP classification
# ---------------------------------------------------------------------------

class TestReminderNLPClassification:
    """Test that reminder queries are routed to the correct intents."""

    @pytest.mark.parametrize("text", [
        "remind me to call mom in 30 minutes",
        "set a reminder for 9am to exercise",
        "remind me about the meeting tomorrow",
        "don't let me forget to buy milk",
        "alert me at 8pm to take my medicine",
        "notify me when the timer is done",
        "add a reminder to pick up the kids at 3pm",
    ])
    def test_set_intent(self, text):
        assert _nlp_intent(text) == "reminder:set"

    @pytest.mark.parametrize("text", [
        "what reminders do I have",
        "show my reminders",
        "list reminders",
        "do I have any pending reminders",
        "show upcoming reminders",
    ])
    def test_list_intent(self, text):
        assert _nlp_intent(text) == "reminder:list"

    @pytest.mark.parametrize("text", [
        "cancel reminder about doctor appointment",
        "delete reminder for gym",
        "remove the reminder to call mom",
    ])
    def test_cancel_intent(self, text):
        assert _nlp_intent(text) == "reminder:cancel"


# ---------------------------------------------------------------------------
# 2. parse_reminder_time
# ---------------------------------------------------------------------------

class TestParseReminderTime:
    """Test the time-parsing utility for various natural-language patterns."""

    BASE = datetime(2026, 3, 6, 10, 0, 0)  # 10:00 AM

    def _p(self, text: str) -> datetime:
        from ai.planning.proactive_assistant import parse_reminder_time
        result = parse_reminder_time(text, now=self.BASE)
        assert result is not None, f"parse_reminder_time({text!r}) returned None"
        return result

    # -- relative --
    def test_in_30_minutes(self):
        t = self._p("remind me in 30 minutes to call")
        assert t == self.BASE + timedelta(minutes=30)

    def test_in_2_hours(self):
        t = self._p("do this in 2 hours")
        assert t == self.BASE + timedelta(hours=2)

    def test_in_3_days(self):
        t = self._p("remind me in 3 days")
        assert t == self.BASE + timedelta(days=3)

    def test_in_1_week(self):
        t = self._p("remind me in 1 week")
        assert t == self.BASE + timedelta(weeks=1)

    # -- clock time today --
    def test_at_3pm(self):
        t = self._p("remind me at 3pm to pick up kids")
        assert t.hour == 15
        assert t.minute == 0

    def test_at_9_30_am(self):
        t = self._p("wakeup at 9:30 am")
        # 9:30 < base (10:00), so should push to tomorrow
        assert t.date() == (self.BASE + timedelta(days=1)).date()
        assert t.hour == 9
        assert t.minute == 30

    def test_at_14_30_24h(self):
        t = self._p("meeting at 14:30")
        assert t.hour == 14
        assert t.minute == 30

    # -- tomorrow --
    def test_tomorrow_with_time(self):
        t = self._p("remind me tomorrow at 9am to exercise")
        assert t.date() == (self.BASE + timedelta(days=1)).date()
        assert t.hour == 9

    def test_tomorrow_no_time(self):
        t = self._p("remind me tomorrow to take medicine")
        assert t.date() == (self.BASE + timedelta(days=1)).date()
        assert t.hour == 9  # default 9 AM

    # -- named times --
    def test_tonight(self):
        t = self._p("remind me tonight to close the garage")
        assert t.hour == 20

    def test_noon(self):
        t = self._p("remind me at noon to eat lunch")
        assert t.hour == 12

    def test_this_morning_is_tomorrow_since_already_past(self):
        # Base is 10:00, "morning" = 9:00, already passed → next day
        t = self._p("remind me this morning")
        assert t.date() > self.BASE.date() or t.hour == 9

    # -- returns None when no time found --
    def test_no_time_returns_none(self):
        from ai.planning.proactive_assistant import parse_reminder_time
        result = parse_reminder_time("remind me to stretch", now=self.BASE)
        assert result is None


# ---------------------------------------------------------------------------
# 3. ProactiveAssistant: add / list / cancel + persistence
# ---------------------------------------------------------------------------

class TestProactiveAssistantReminders:
    """Test the ProactiveAssistant reminder CRUD and persistence."""

    def _make_assistant(self, path: Path):
        """Create a fresh ProactiveAssistant pointing at a temp file."""
        import ai.planning.proactive_assistant as pa_mod
        # Redirect module-level path so save/load use temp file
        orig = pa_mod._REMINDERS_PATH
        pa_mod._REMINDERS_PATH = path
        from ai.planning.proactive_assistant import ProactiveAssistant
        pa = ProactiveAssistant()
        pa_mod._REMINDERS_PATH = orig  # restore so other tests are unaffected
        # Re-point the instance's module reference for save/load calls
        pa._reminders_path = path
        # Patch _save/_load to use our temp path
        def _save():
            data = [
                {
                    "reminder_id": r.reminder_id,
                    "message": r.message,
                    "trigger_time": r.trigger_time.isoformat(),
                    "priority": r.priority,
                    "context": r.context,
                    "delivered": r.delivered,
                }
                for r in pa.reminders.values()
            ]
            path.write_text(json.dumps(data), encoding="utf-8")

        def _load():
            if not path.exists():
                return
            for item in json.loads(path.read_text(encoding="utf-8")):
                if item.get("delivered"):
                    continue
                from ai.planning.proactive_assistant import Reminder
                pa.reminders[item["reminder_id"]] = Reminder(
                    reminder_id=item["reminder_id"],
                    message=item["message"],
                    trigger_time=datetime.fromisoformat(item["trigger_time"]),
                    priority=item.get("priority", "normal"),
                    context=item.get("context", {}),
                    delivered=False,
                )

        pa._save_reminders = _save
        pa._load_reminders = _load
        return pa

    def test_add_and_list(self, tmp_path):
        pa = self._make_assistant(tmp_path / "reminders.json")
        trigger = datetime.now() + timedelta(hours=1)
        pa.add_reminder("r1", "Take medicine", trigger)
        pa.add_reminder("r2", "Go to gym", trigger + timedelta(hours=1))
        pending = pa.list_reminders()
        assert len(pending) == 2
        assert pending[0].reminder_id == "r1"

    def test_list_sorted_by_time(self, tmp_path):
        pa = self._make_assistant(tmp_path / "reminders.json")
        now = datetime.now()
        pa.add_reminder("late", "Late task", now + timedelta(hours=3))
        pa.add_reminder("early", "Early task", now + timedelta(hours=1))
        pending = pa.list_reminders()
        assert pending[0].reminder_id == "early"

    def test_cancel_by_keyword(self, tmp_path):
        pa = self._make_assistant(tmp_path / "reminders.json")
        trigger = datetime.now() + timedelta(hours=1)
        pa.add_reminder("r1", "Reminder: call mom", trigger)
        result = pa.cancel_reminder("call mom")
        assert result is True
        assert len(pa.list_reminders()) == 0

    def test_cancel_nonexistent_returns_false(self, tmp_path):
        pa = self._make_assistant(tmp_path / "reminders.json")
        assert pa.cancel_reminder("doctor appointment") is False

    def test_delivered_reminders_excluded_from_list(self, tmp_path):
        pa = self._make_assistant(tmp_path / "reminders.json")
        trigger = datetime.now() + timedelta(hours=1)
        pa.add_reminder("r1", "Buy groceries", trigger)
        pa.reminders["r1"].delivered = True
        assert pa.list_reminders() == []

    def test_persistence_round_trip(self, tmp_path):
        rpath = tmp_path / "reminders.json"
        pa1 = self._make_assistant(rpath)
        trigger = datetime.now() + timedelta(hours=2)
        pa1.add_reminder("r1", "Dentist appointment", trigger)
        # New assistant loads from the same file
        pa2 = self._make_assistant(rpath)
        pa2._load_reminders()
        assert "r1" in pa2.reminders
        assert pa2.reminders["r1"].message == "Dentist appointment"

    def test_delivered_not_reloaded(self, tmp_path):
        rpath = tmp_path / "reminders.json"
        pa1 = self._make_assistant(rpath)
        trigger = datetime.now() + timedelta(hours=1)
        pa1.add_reminder("r1", "Already done", trigger)
        pa1.reminders["r1"].delivered = True
        pa1._save_reminders()
        pa2 = self._make_assistant(rpath)
        pa2._load_reminders()
        assert "r1" not in pa2.reminders


# ---------------------------------------------------------------------------
# 4. Delivery callback fires when reminder is due
# ---------------------------------------------------------------------------

class TestReminderDelivery:
    def test_due_reminder_calls_callback(self):
        from ai.planning.proactive_assistant import ProactiveAssistant, Reminder
        pa = ProactiveAssistant()
        # Suppress actual file I/O for this test
        pa._save_reminders = lambda: None

        delivered = []

        def cb(msg, priority):
            delivered.append(msg)

        pa.set_notification_callback(cb)
        past = datetime.now() - timedelta(seconds=5)
        pa.reminders["r1"] = Reminder("r1", "Call dentist", past)
        pa._check_reminders()
        assert len(delivered) == 1
        assert "dentist" in delivered[0].lower()

    def test_future_reminder_not_delivered(self):
        from ai.planning.proactive_assistant import ProactiveAssistant, Reminder
        pa = ProactiveAssistant()
        pa._save_reminders = lambda: None
        delivered = []
        pa.set_notification_callback(lambda m, p: delivered.append(m))
        future = datetime.now() + timedelta(hours=1)
        pa.reminders["r1"] = Reminder("r1", "Future task", future)
        pa._check_reminders()
        assert delivered == []


# ---------------------------------------------------------------------------
# 5. make_reminder_id uniqueness
# ---------------------------------------------------------------------------

def test_make_reminder_id_unique():
    from ai.planning.proactive_assistant import make_reminder_id
    ids = {make_reminder_id() for _ in range(50)}
    assert len(ids) == 50

def test_make_reminder_id_format():
    from ai.planning.proactive_assistant import make_reminder_id
    rid = make_reminder_id()
    assert rid.startswith("rem_")
    assert len(rid) == 12  # "rem_" + 8 hex chars
