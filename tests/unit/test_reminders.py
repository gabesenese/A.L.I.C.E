"""Reminders that actually remind.

No plugin handled reminder:set, so "remind me to call mom at 5pm" was answered
"I couldn't get a result for that", nothing was stored, and nothing would have
fired.
"""

from datetime import datetime, timedelta

import pytest

from ai.planning.reminders import ReminderStore, ReminderWatcher, parse_reminder
from ai.plugins.reminder_plugin import ReminderPlugin

NOW = datetime(2026, 9, 24, 14, 30)


@pytest.mark.parametrize(
    "text, task, due",
    [
        ("remind me to call mom at 5pm", "call mom", datetime(2026, 9, 24, 17, 0)),
        ("remind me in 20 minutes to check the oven", "check the oven", NOW + timedelta(minutes=20)),
        ("remind me in half an hour to move the car", "move the car", NOW + timedelta(minutes=30)),
        ("remind me to stretch in an hour", "stretch", NOW + timedelta(hours=1)),
        ("set a reminder for tomorrow at 9 to pay rent", "pay rent", datetime(2026, 9, 25, 9, 0)),
        ("remind me to take out the trash tonight", "take out the trash", datetime(2026, 9, 24, 20, 0)),
        ("remind me to call John at 3", "call John", datetime(2026, 9, 24, 15, 0)),
        ("remind me at 1:15 pm to join the call", "join the call", datetime(2026, 9, 25, 13, 15)),
        ("remind me at noon to eat", "eat", datetime(2026, 9, 25, 12, 0)),
    ],
)
def test_the_task_and_the_time_are_understood(text, task, due):
    assert parse_reminder(text, NOW) == (task, due)


def test_without_a_time_she_asks_rather_than_guesses(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))

    out = plugin.execute("reminder:set", "remind me to water the plants", {}, {})

    assert out["response"] == "When should I remind you to water the plants?"
    assert plugin.store.pending() == []


def test_setting_one_stores_it_and_says_when(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))

    out = plugin.execute("reminder:set", "remind me in 20 minutes to check the oven", {}, {})

    assert out["success"] is True
    assert out["response"].startswith("Okay, I'll remind you to check the oven at ")
    assert [r.text for r in plugin.store.pending()] == ["check the oven"]


def test_it_is_delivered_once_when_due(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("call mom", NOW)
    said = []
    watcher = ReminderWatcher(store, said.append)

    watcher.check(NOW - timedelta(minutes=1))
    watcher.check(NOW + timedelta(seconds=5))
    watcher.check(NOW + timedelta(minutes=1))

    assert said == ["Reminder: call mom."]


def test_one_that_came_due_while_she_was_closed_says_when_it_was_due(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("call mom", NOW)
    said = []

    ReminderWatcher(store, said.append).check(NOW + timedelta(hours=2))

    assert said == ["Reminder: call mom. (It was due at 2:30 PM.)"]


def test_they_are_listed_and_can_be_cancelled(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))
    plugin.execute("reminder:set", "remind me tomorrow at 9 to pay rent", {}, {})
    plugin.execute("reminder:set", "remind me tomorrow at 10 to call the bank", {}, {})

    assert "pay rent" in plugin.execute("reminder:list", "what are my reminders", {}, {})["response"]
    assert plugin.execute("reminder:cancel", "cancel the rent reminder", {}, {})["response"] == "Cancelled: pay rent."
    assert [r.text for r in plugin.store.pending()] == ["call the bank"]


def test_the_plugin_is_what_handles_reminder_intents(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))
    assert plugin.can_handle("reminder:set", {})
    assert plugin.can_handle("reminder:list", {})
    assert not plugin.can_handle("notes:create", {})
