"""Without Google Calendar, "what's on my calendar?" still gets an answer.

The plugin only accepted bare intents such as "calendar", and the NLP routes to
calendar:list_events, so the question reached no plugin and was answered "I
couldn't get a result for that." Without credentials the plugin had nothing
to offer either. His reminders and the notes falling due are the schedule she
has, so they answer it, with the missing connection said once.
"""

import pytest

from ai.plugins import calendar_plugin
from ai.plugins.calendar_plugin import CalendarPlugin

HOW = " To connect it, add calendar_credentials.json to config/cred and restart me."


@pytest.fixture(autouse=True)
def google_installed(monkeypatch):
    monkeypatch.setattr(calendar_plugin, "GOOGLE_AVAILABLE", True)


def _plugin(agenda="For today: call mom at 5:00 PM."):
    return CalendarPlugin(agenda=lambda query: agenda)


def _ask(plugin, text, intent="calendar:list_events"):
    return plugin.execute(intent, text, {}, {})["response"]


def test_the_calendar_takes_its_own_namespaced_intents_but_not_reminders():
    plugin = CalendarPlugin()

    assert plugin.can_handle("calendar:list_events", {})
    assert plugin.can_handle("calendar", {})
    assert not plugin.can_handle("reminder:set", {})


def test_without_google_the_day_comes_from_reminders_and_notes():
    plugin = _plugin()

    first = _ask(plugin, "what's on my calendar today?")
    again = _ask(plugin, "what's on my calendar today?")

    assert first == (
        "Google Calendar isn't connected, so this is from your reminders and notes. For today: call mom at 5:00 PM."
        + HOW
    )
    assert again.endswith("call mom at 5:00 PM.")


def test_an_empty_day_is_said_in_one_sentence():
    plugin = _plugin("Nothing on your reminders or notes for today.")

    assert _ask(plugin, "what's on my calendar today?") == (
        "Google Calendar isn't connected, and there's nothing on your reminders or notes for today." + HOW
    )


def test_asked_to_add_an_event_she_offers_a_reminder():
    reply = _ask(_plugin(), "add lunch with Sam to my calendar tomorrow at noon", "calendar:create_event")

    assert reply.startswith(
        "Google Calendar isn't connected, so I can't add it there. Want me to set a reminder instead?"
    )


def test_without_the_google_libraries_it_still_loads_and_says_what_is_missing(monkeypatch):
    """initialize() returned False, so the plugin was never registered and the
    question reached nothing."""
    monkeypatch.setattr(calendar_plugin, "GOOGLE_AVAILABLE", False)
    plugin = _plugin()

    assert plugin.initialize() is True
    assert "install google-api-python-client" in _ask(plugin, "what's on my calendar today?")


@pytest.mark.parametrize(
    "text", ["do I have any meetings tomorrow?", "when is my next appointment?", "do I have any events today"]
)
def test_meeting_questions_go_to_the_calendar(text):
    from ai.core.nlp_processor import NLPProcessor

    result = NLPProcessor().process(text)
    assert result.intent == "calendar:list_events"
    assert not result.parsed_command["modifiers"].get("tool_execution_disabled")
