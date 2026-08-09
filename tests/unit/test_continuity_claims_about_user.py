"""Claims about what the user is doing need evidence, same as claims about the past.

The guard only caught explicit recall phrasing ("you mentioned", "we were
discussing"). Assertions about the user's current state slipped through, so a plain
greeting came back as "you're still stuck on that routing refactor" for a user who
had never mentioned a routing refactor.
"""

import pytest

from ai.runtime.continuity_claim_guard import assess_continuity_claims


def assess(text, memory_items=None, operator_state=None):
    return assess_continuity_claims(
        text=text,
        memory_items=list(memory_items or []),
        operator_state=dict(operator_state or {}),
    )


@pytest.mark.parametrize(
    "text",
    [
        "I'm here, and you're still stuck on that routing refactor.",
        "You've been grinding on the caching layer for a while now.",
        "I can tell you're frustrated with the deploy.",
        "You keep running into the same import error.",
        "You were debugging the memory leak last night.",
        "I know you've got a lot on your plate.",
    ],
)
def test_invented_claims_about_the_user_are_removed(text):
    result = assess(text)
    assert result.unsupported_continuity_claim is True
    assert "routing refactor" not in result.text
    assert "caching layer" not in result.text


def test_a_claim_backed_by_the_active_objective_survives():
    result = assess(
        "You're still working on the routing refactor.",
        operator_state={
            "active_objective": "finish the routing refactor",
            "current_focus": "routing refactor",
        },
    )
    assert result.unsupported_continuity_claim is False
    assert "routing refactor" in result.text


@pytest.mark.parametrize(
    "text",
    [
        "There are four notes in total.",
        "The file has 420 lines.",
        "I'm here. What do you want to look at?",
        "That command destroys data and I don't have an undo for it.",
    ],
)
def test_grounded_statements_pass_through_unchanged(text):
    result = assess(text)
    assert result.text.strip() == text
    assert result.unsupported_continuity_claim is False


def test_empty_text_is_handled():
    assert assess("").text == ""
