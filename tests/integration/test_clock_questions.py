"""Only a question about the clock goes to the clock.

"What's the time" matched anywhere in a sentence, so "what's the time complexity
of binary search?" went to the time plugin at 0.95 confidence and came back as
"I couldn't get a result for that." The same happened to a time difference, a
time limit, and "what's the date of the next meeting?".
"""

import pytest

from ai.core.nlp_processor import NLPProcessor


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


@pytest.mark.parametrize(
    "text",
    [
        "what time is it?",
        "what's the time?",
        "what is the current time",
        "what's the time now",
        "tell me the time",
        "what's the date today?",
        "what day is it",
    ],
)
def test_a_clock_question_goes_to_the_clock(nlp, text):
    assert nlp.process(text).intent == "time:current"


@pytest.mark.parametrize(
    "text",
    [
        "what's the time complexity of binary search?",
        "what's the time difference between tokyo and london?",
        "what is the time limit for the exam?",
        "what's the date of the next meeting?",
        "the current time complexity is quadratic, can we do better?",
    ],
)
def test_a_question_about_something_else_does_not(nlp, text):
    assert nlp.process(text).intent != "time:current"


@pytest.mark.parametrize(
    "text",
    [
        "how do I set a reminder on my iphone?",
        "how do i delete a branch in git?",
        "hey alice, how do I add a note?",
        "show me how recursion works",
        "find the bug: for i in range(10) print(i)",
        "how to create a virtual environment",
    ],
)
def test_asking_how_to_do_something_is_answered_not_acted_on(nlp, text):
    """A plugin cannot teach, and doing the thing presumes he wanted it done."""
    plugin = nlp.process(text).intent.split(":", 1)[0]
    assert plugin not in {"notes", "reminder", "memory", "file_operations", "calendar", "email", "system"}


@pytest.mark.parametrize(
    "text, intent",
    [
        ("remind me to call mom at 5pm", "reminder:set"),
        ("show me how many notes I have", "notes:list"),
    ],
)
def test_the_commands_themselves_still_act(nlp, text, intent):
    assert nlp.process(text).intent == intent


def test_the_time_plugin_answers_the_clock_intent():
    """It only accepted the bare intents "time" and "date", so time:current
    reached no plugin and "what's today's date?" failed outright."""
    from ai.plugins.plugin_system import TimePlugin

    plugin = TimePlugin()
    assert plugin.can_handle("time:current", {})
    assert plugin.execute("time:current", "what's today's date?", {}, {})["response"].startswith("Today is ")
    assert plugin.execute("time:current", "what day is it", {}, {})["response"].startswith("Today is ")
    assert plugin.execute("time:current", "what time is it?", {}, {})["response"].startswith("It's ")
