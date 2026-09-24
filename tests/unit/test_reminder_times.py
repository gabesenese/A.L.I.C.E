"""Reminders land on the day and hour he meant.

"remind me on friday at 3 to pay rent" was set for 3 today, to "friday to pay
rent"; "tomorrow morning" left "morning" in what to remind; and the answer to
"When should I remind you to call mom?" went to the model, so nothing was set.
"""

from datetime import datetime

import pytest

from ai.planning.reminders import ReminderStore, parse_reminder
from ai.plugins.reminder_plugin import ReminderPlugin

NOW = datetime(2026, 9, 24, 14, 30)  # a Thursday


@pytest.mark.parametrize(
    "text, task, due",
    [
        ("remind me on friday at 3 to pay rent", "pay rent", datetime(2026, 9, 25, 15, 0)),
        ("remind me next monday to call the bank", "call the bank", datetime(2026, 9, 28, 9, 0)),
        ("remind me thursday at 9am to stretch", "stretch", datetime(2026, 10, 1, 9, 0)),
        ("remind me to water the plants tomorrow morning", "water the plants", datetime(2026, 9, 25, 9, 0)),
        ("remind me tomorrow evening at 7 to call Sam", "call Sam", datetime(2026, 9, 25, 19, 0)),
        ("remind me tomorrow night to lock up", "lock up", datetime(2026, 9, 25, 21, 0)),
    ],
)
def test_the_day_and_hour_he_meant(text, task, due):
    assert parse_reminder(text, NOW) == (task, due)


def test_the_answer_to_when_sets_the_reminder(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    plugin = ReminderPlugin(store)

    asked = plugin.execute("reminder:set", "don't forget to call mom", {}, {})["response"]
    answered = plugin.execute("reminder:set", "at 11:59pm", {}, {})["response"]

    assert asked == "When should I remind you to call mom?"
    assert answered.startswith("Okay, I'll remind you to call mom")
    assert [r.text for r in store.pending()] == ["call mom"]


def test_the_answer_is_routed_back_to_reminders():
    from ai.core.nlp_processor import NLPProcessor

    nlp = NLPProcessor()
    nlp.process("remind me to water the plants")

    assert nlp.process("tomorrow morning").intent == "reminder:set"
