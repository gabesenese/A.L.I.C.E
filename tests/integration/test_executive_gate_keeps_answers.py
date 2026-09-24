"""An answer that exists is not swapped for a stand-in by a word-overlap score.

Asked "what do you know about me?", Alice had the answer, and the executive gate
replaced it with "I didn't follow that. Could you put it another way?" because
the reply shared no word with the question. That is a scripted override, so it
follows the same policy as the others: off by default, and the answer stands.
"""

import pytest

from ai.core.executive_controller import ExecutiveController
from app.main import ALICE

QUESTION = "what do you know about me?"
ANSWER = "Your sister is Ana, and you have the dentist on Friday at 3pm."


@pytest.fixture
def alice():
    bare = ALICE.__new__(ALICE)
    bare.executive_controller = ExecutiveController()
    bare._think = lambda *_args, **_kwargs: None
    return bare


def _gate(alice, response):
    return alice._executive_apply_response_gate(
        user_input=QUESTION, intent="conversation:question", response=response, route="llm"
    )


def test_the_gate_still_scores_it_as_off_topic(alice):
    """The score is unchanged; only what happens to the answer is."""
    _gate(alice, ANSWER)
    assert alice._last_exec_gate_eval["accepted"] is False


def test_the_answer_stands_by_default(alice, monkeypatch):
    monkeypatch.delenv("ALICE_ENABLE_SCRIPTED_OVERRIDES", raising=False)
    assert _gate(alice, ANSWER) == ANSWER


def test_the_old_replacement_is_still_there_under_scripted_overrides(alice, monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_SCRIPTED_OVERRIDES", "1")
    assert _gate(alice, ANSWER) != ANSWER
