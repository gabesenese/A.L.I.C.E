"""Alice learns what a phrasing meant from the conversation itself.

The only way to fix a misroute was /correct. Asked something she routed wrong,
people do not type a command; they say "no, I meant the weather". That
restatement is the correction: what it routes to is what the first phrasing
asked for, and next time the first phrasing goes there directly.
"""

import pytest

from ai.runtime.turn_orchestrator import _learn_from_restatement, run_default_turn
from tests.integration.test_every_turn_is_in_the_transcript import alice as alice  # noqa: F401  (fixture)


@pytest.fixture
def learned(alice):  # noqa: F811
    calls = []
    alice._learn_intent_correction = lambda text, intent: calls.append((text, intent)) or True
    return calls


ACCEPTED = {"verification": {"accepted": True}}


def test_saying_what_you_meant_teaches_the_first_phrasing(alice, learned):  # noqa: F811
    previous = {"user_input": "anything from the dentist?", "intent": "conversation:general"}

    _learn_from_restatement(alice, "no, I meant in my notes", {"intent": "notes:search", **ACCEPTED}, previous)

    assert learned == [("anything from the dentist?", "notes:search")]


def test_a_restatement_that_failed_teaches_nothing(alice, learned):  # noqa: F811
    previous = {"user_input": "anything from the dentist?", "intent": "conversation:general"}

    _learn_from_restatement(
        alice, "I meant in my notes", {"intent": "notes:search", "verification": {"accepted": False}}, previous
    )

    assert learned == []


def test_each_turn_is_judged_against_the_one_before_it(alice, monkeypatch):  # noqa: F811
    from ai.runtime import turn_orchestrator

    seen = []
    monkeypatch.setattr(turn_orchestrator, "_learn_from_restatement", lambda a, text, meta, prev: seen.append(prev))
    alice.replies.extend(["Nothing I can see.", "Okay."])

    run_default_turn(alice, "anything from the dentist?")
    run_default_turn(alice, "no, I meant in my notes")

    assert seen[0] is None
    assert seen[1]["user_input"] == "anything from the dentist?"


@pytest.mark.parametrize("follow_up", ["no thanks", "weather in boston", "no, that's fine"])
def test_a_follow_up_that_is_not_a_restatement_teaches_nothing(alice, learned, follow_up):  # noqa: F811
    alice.replies.extend(["Hard to say without a forecast.", "Okay."])
    run_default_turn(alice, "will I need an umbrella")

    run_default_turn(alice, follow_up)

    assert learned == []


def test_nothing_that_deletes_or_sends_is_ever_learned(learned, alice):  # noqa: F811
    previous = {"user_input": "tidy up my stuff", "intent": "conversation:general"}
    for intent in ("notes:delete", "email:send", "reminder:cancel"):
        _learn_from_restatement(alice, "I meant that", {"intent": intent, "verification": {"accepted": True}}, previous)

    assert learned == []


def test_a_restatement_within_the_same_plugin_teaches_nothing(learned, alice):  # noqa: F811
    """notes:list then "I meant the groceries one" is a new request, not a correction."""
    previous = {"user_input": "show my notes", "intent": "notes:list"}
    _learn_from_restatement(
        alice, "I meant the groceries one", {"intent": "notes:read", "verification": {"accepted": True}}, previous
    )

    assert learned == []
