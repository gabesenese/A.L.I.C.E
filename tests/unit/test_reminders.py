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


def test_nothing_is_delivered_in_the_middle_of_a_turn(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    store.add("call mom", NOW)
    said = []
    busy = {"turn": True}
    watcher = ReminderWatcher(store, said.append, ready=lambda: not busy["turn"])

    watcher.check(NOW + timedelta(seconds=5))
    assert said == []

    busy["turn"] = False
    watcher.check(NOW + timedelta(seconds=20))
    assert said == ["Reminder: call mom."]


def test_remind_me_again_rearms_the_one_that_just_fired(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))
    plugin.store.add("check the oven", NOW)
    ReminderWatcher(plugin.store, lambda _message: None).check(NOW)

    out = plugin.execute("reminder:set", "remind me again in 10 minutes", {}, {})

    assert out["response"].startswith("Okay, I'll remind you to check the oven at ")
    assert [r.text for r in plugin.store.pending()] == ["check the oven"]


def test_a_reminder_is_said_in_the_chat_and_kept_in_the_conversation(capsys):
    from types import SimpleNamespace

    from app.main import ALICE

    alice = ALICE.__new__(ALICE)
    alice.context = SimpleNamespace(user_prefs=SimpleNamespace(name="Gabriel"))
    alice.llm = SimpleNamespace(conversation_history=[])

    alice._say_unprompted("Reminder: check the oven.")

    assert capsys.readouterr().out == "\nA.L.I.C.E: Reminder: check the oven.\n\nGabriel: "
    assert alice.llm.conversation_history == [{"role": "assistant", "content": "Reminder: check the oven."}]


def test_a_turn_is_marked_in_progress_while_it_runs():
    from ai.runtime import turn_orchestrator
    from types import SimpleNamespace

    seen = []
    alice = SimpleNamespace()
    original = turn_orchestrator._run_default_turn
    turn_orchestrator._run_default_turn = lambda a, text, voice=False: seen.append(a._turn_in_progress) or "ok"
    try:
        assert turn_orchestrator.run_default_turn(alice, "hi") == "ok"
    finally:
        turn_orchestrator._run_default_turn = original

    assert seen == [True]
    assert alice._turn_in_progress is False


def test_a_line_said_unprompted_waits_for_the_turn_to_finish(capsys):
    import threading
    import time
    from types import SimpleNamespace

    from app.main import ALICE

    alice = ALICE.__new__(ALICE)
    alice.context = SimpleNamespace(user_prefs=SimpleNamespace(name="Gabriel"))
    alice.llm = SimpleNamespace(conversation_history=[{"role": "user", "content": "what's up?"}])
    alice._turn_in_progress = True

    speaker = threading.Thread(target=alice._say_unprompted, args=("Heads up, the build has been red for a day.",))
    speaker.start()
    time.sleep(0.5)
    assert len(alice.llm.conversation_history) == 1

    alice.llm.conversation_history.append({"role": "assistant", "content": "Not much."})
    alice._turn_in_progress = False
    speaker.join(timeout=5)

    assert [m["content"] for m in alice.llm.conversation_history][-2:] == [
        "Not much.",
        "Heads up, the build has been red for a day.",
    ]


def _note(title, due_date, archived=False):
    from types import SimpleNamespace

    return SimpleNamespace(title=title, due_date=due_date, archived=archived, checklist_items=None)


def test_the_agenda_is_what_she_actually_knows(tmp_path):
    """Answered by the model with nothing in front of it, "what's on my schedule
    today?" could only be made up."""
    store = ReminderStore(tmp_path / "r.json")
    store.add("call mom", NOW.replace(hour=17, minute=0))
    store.add("pay rent", NOW.replace(hour=9, minute=0) + timedelta(days=1))
    notes = [_note("Submit expenses", "2026-09-24T15:30"), _note("Old thing", "2026-09-24T16:00", archived=True)]
    plugin = ReminderPlugin(store, notes=lambda: notes)

    today = plugin._agenda("what's on my schedule today?", NOW)["response"]
    tomorrow = plugin._agenda("what do I have tomorrow?", NOW)["response"]

    assert today == 'For today: "Submit expenses" is due at 3:30 PM; call mom at 5:00 PM.'
    assert tomorrow == "For tomorrow: pay rent at 9:00 AM."


def test_a_note_due_on_a_date_is_not_given_a_time(tmp_path):
    """A bare due date counts until the end of the day; reading that out as
    "due at 11:59 PM" would be a time he never set."""
    notes = [_note("Tax forms", "2026-09-25")]
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"), notes=lambda: notes)

    assert plugin._agenda("what do I have tomorrow?", NOW)["response"] == 'For tomorrow: "Tax forms" is due.'
    assert plugin._agenda("anything planned for this week?", NOW)["response"] == (
        'For this week: "Tax forms" is due tomorrow.'
    )


def test_an_empty_day_is_said_plainly(tmp_path):
    plugin = ReminderPlugin(ReminderStore(tmp_path / "r.json"))
    assert plugin._agenda("what do I have today?", NOW)["response"] == "Nothing on your reminders or notes for today."


@pytest.mark.parametrize(
    "text",
    [
        "what's on my schedule today?",
        "what do I have today?",
        "what's my agenda for tomorrow",
        "anything planned for this week?",
    ],
)
def test_agenda_questions_are_routed_to_the_agenda(text):
    from ai.core.nlp_processor import NLPProcessor

    result = NLPProcessor().process(text)
    assert result.intent == "reminder:agenda"
    # Routed here but with tools still switched off by the conversation gate, the
    # question went to the model anyway.
    assert not result.parsed_command["modifiers"].get("tool_execution_disabled")


def test_calendar_questions_stay_with_the_calendar():
    from ai.core.nlp_processor import NLPProcessor

    assert NLPProcessor().process("what's on my calendar today").intent != "reminder:agenda"
