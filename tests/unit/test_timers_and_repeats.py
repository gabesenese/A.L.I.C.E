"""Timers, and reminders that repeat.

"set a timer for 10 minutes" went to the model, which has no clock to run, and
"remind me every day at 8am to take my pills" became one reminder, for
tomorrow, to "every day to take my pills".
"""

from datetime import datetime, timedelta

import pytest

from ai.planning.reminders import ReminderStore, parse_request, parse_timer, reminder_message
from ai.plugins.reminder_plugin import ReminderPlugin

NOW = datetime(2026, 9, 24, 14, 30)  # a Thursday


@pytest.mark.parametrize(
    "text, minutes, label",
    [
        ("set a timer for 10 minutes", 10, ""),
        ("timer for 90 seconds", 1.5, ""),
        ("start a 5 minute timer", 5, ""),
        ("set a timer for an hour for the roast", 60, "roast"),
        ("set a timer for 10 minutes for the pasta", 10, "pasta"),
    ],
)
def test_a_timer_is_heard(text, minutes, label):
    assert parse_timer(text) == (minutes, label)


def test_a_timer_is_set_and_said_when_it_is_up(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    plugin = ReminderPlugin(store)

    reply = plugin._start_timer(10, "pasta", now=NOW)["response"]
    [timer] = store.take_due(NOW + timedelta(minutes=10))

    assert reply == "Okay, 10 minutes for the pasta, starting now."
    assert reminder_message(timer, NOW + timedelta(minutes=10)) == (
        "Time's up. That was your 10-minute timer for the pasta."
    )


@pytest.mark.parametrize(
    "text, repeat, due",
    [
        ("remind me every day at 8am to take my pills", "daily", datetime(2026, 9, 25, 8, 0)),
        ("remind me to stretch every weekday at 3pm", "weekdays", datetime(2026, 9, 24, 15, 0)),
        ("remind me every monday at 9 to send the report", "weekly:0", datetime(2026, 9, 28, 9, 0)),
        ("remind me to water the plants every morning", "daily", datetime(2026, 9, 25, 9, 0)),
    ],
)
def test_a_repeat_is_heard_and_left_out_of_the_task(text, repeat, due):
    request = parse_request(text, NOW)

    assert (request.repeat, request.due) == (repeat, due)
    assert "every" not in request.task and "daily" not in request.task


def test_a_repeating_reminder_is_set_again_after_it_goes_off(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("take my pills", datetime(2026, 9, 25, 8, 0), repeat="daily")

    [said] = store.take_due(datetime(2026, 9, 25, 8, 0, 30))
    [still_there] = store.pending()

    assert reminder_message(said, datetime(2026, 9, 25, 8, 0, 30)) == "Reminder: take your pills."
    assert still_there.due_at == datetime(2026, 9, 26, 8, 0)


def test_the_confirmation_says_how_often_and_in_his_terms(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))

    reply = plugin.execute("reminder:set", "remind me every day at 8am to take my pills", {}, {})["response"]

    assert reply == "Okay, I'll remind you to take your pills every day at 8:00 AM."


def test_timer_requests_are_routed_to_reminders():
    from ai.core.nlp_processor import NLPProcessor

    assert NLPProcessor().process("set a timer for 10 minutes").intent == "reminder:set"


def test_snooze_brings_back_what_just_went_off(tmp_path):
    """A repeating reminder is set again rather than marked fired, so "remind me
    again" could not find it, and "snooze" was not heard at all."""
    store = ReminderStore(tmp_path / "r.json")
    plugin = ReminderPlugin(store)
    store.add("take my pills", datetime(2026, 9, 25, 8, 0), repeat="daily")
    store.take_due(datetime(2026, 9, 25, 8, 0, 30))

    reply = plugin._snooze(None, datetime(2026, 9, 25, 8, 1))["response"]

    assert reply == "Okay, I'll remind you to take your pills again in 10 minutes."
    assert [r.due_at for r in store.pending()] == [datetime(2026, 9, 25, 8, 11), datetime(2026, 9, 26, 8, 0)]


def test_nothing_to_snooze_is_said_plainly(tmp_path):
    assert ReminderPlugin(ReminderStore(tmp_path / "r.json"))._snooze("5", NOW)["response"] == (
        "There's nothing to snooze."
    )


@pytest.mark.parametrize("text", ["snooze", "snooze 5 minutes", "ugh, snooze it for 10 min"])
def test_snooze_is_routed_to_reminders(text):
    from ai.core.nlp_processor import NLPProcessor

    assert NLPProcessor().process(text).intent == "reminder:set"


def test_the_time_left_on_a_timer_is_answered(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    plugin = ReminderPlugin(store)
    plugin._start_timer(10, "pasta", now=NOW)

    assert plugin._timer_left(NOW + timedelta(minutes=4))["response"] == "About 6 minutes left on the pasta."
    assert ReminderPlugin(ReminderStore(tmp_path / "none.json"))._timer_left(NOW)["response"] == (
        "You don't have a timer running."
    )


def test_asking_how_long_is_left_is_routed_to_the_timer():
    from ai.core.nlp_processor import NLPProcessor

    assert NLPProcessor().process("how long is left on my timer?").intent == "reminder:timer_left"
