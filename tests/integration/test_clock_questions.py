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
