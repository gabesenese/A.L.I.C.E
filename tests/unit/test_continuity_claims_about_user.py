"""Claims about what the user is doing need evidence, same as claims about the past.

The guard only caught explicit recall phrasing ("you mentioned", "we were
discussing"). Assertions about the user's current state slipped through, so a plain
greeting came back as "you're still stuck on that routing refactor" for a user who
had never mentioned a routing refactor.
"""

from types import SimpleNamespace

import pytest

from ai.runtime.boundaries.boundary_factory import _turn_evidence_text
from ai.runtime.continuity_claim_guard import UNSUPPORTED_CLAIM_REPLY, assess_continuity_claims


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


@pytest.mark.parametrize(
    "text",
    [
        "You're not exactly dressed for overcast skies and a chance of rain.",
        "You are not dressed for this weather.",
        "You look tired today.",
        "You seem stressed about the deploy.",
        "I can see you've got the terminal open.",
        "Your desk looks busy.",
        "From the looks of it, you're mid-refactor.",
    ],
)
def test_claims_needing_senses_alice_does_not_have_are_always_removed(text):
    """No camera, no microphone on the room. No amount of evidence grounds these."""
    result = assess(text)
    assert result.unsupported_continuity_claim is True
    assert "dressed" not in result.text
    assert "look" not in result.text.lower() or "looks like" in result.text.lower()


def test_a_name_the_user_never_said_is_treated_as_invented():
    result = assess("Are you heading out for that drive to Oakville?")
    assert result.unsupported_continuity_claim is True
    assert "Oakville" not in result.text


def test_the_same_name_is_kept_once_the_user_says_it():
    result = assess_continuity_claims(
        text="Are you still heading to Oakville later?",
        memory_items=[],
        operator_state={},
        evidence_text="i'm driving to Oakville later",
    )
    assert result.unsupported_continuity_claim is False
    assert "Oakville" in result.text


def test_a_name_a_tool_returned_is_kept():
    result = assess_continuity_claims(
        text="It's 26C in Kitchener, so you should be fine.",
        memory_items=[],
        operator_state={},
        evidence_text="weather for Kitchener: 26C overcast",
    )
    assert result.unsupported_continuity_claim is False
    assert "Kitchener" in result.text


def test_a_factual_sentence_survives_alongside_an_invented_one():
    """Removing the invention must not throw away the answer."""
    result = assess("26C Overcast in Kitchener. Are you heading out for that drive to Oakville?")
    assert "Kitchener" in result.text
    assert "Oakville" not in result.text


def test_grounded_memory_recall_is_not_mistaken_for_invention():
    """Pronouns are capitalised after a colon; treating "You" as a name broke recall."""
    result = assess(
        "Here is what I have saved in memory: You said your sister visited last weekend.",
        memory_items=[{"content": "You said your sister visited last weekend.", "context": {"source": "conversation"}}],
    )
    assert result.unsupported_continuity_claim is False
    assert "sister visited last weekend" in result.text


@pytest.mark.parametrize(
    "text",
    [
        "That build is taking longer than usual. I would check the test discovery step.",
        "Local models have gotten a lot better lately.",
        "Builds are slower these days because of the new linter.",
    ],
)
def test_habit_words_about_things_are_not_claims_about_the_user(text):
    """Deleting "taking longer than usual" removed the observation that the advice
    after it depended on."""
    result = assess(text)
    assert result.text == text
    assert result.unsupported_continuity_claim is False


def test_habit_words_about_the_user_still_need_evidence():
    result = assess("You're up later than usual. The build finished.")
    assert result.unsupported_continuity_claim is True
    assert result.text == "The build finished."


def test_a_callback_to_something_the_user_said_survives():
    """Turn memories carry no session source, so every in-session callback was
    deleted: she could not say "you mentioned" about anything."""
    result = assess_continuity_claims(
        text="You mentioned the tokenizer, so profile that first.",
        memory_items=[],
        operator_state={},
        evidence_text="the parser is slow and I think it's the tokenizer",
    )
    assert result.unsupported_continuity_claim is False
    assert "tokenizer" in result.text


def test_sharing_a_common_word_with_the_user_is_not_evidence():
    result = assess_continuity_claims(
        text="You mentioned your sister's wedding, so let's start there.",
        memory_items=[],
        operator_state={},
        evidence_text="so what should I work on first?",
    )
    assert result.unsupported_continuity_claim is True


def test_this_conversation_does_not_prove_an_earlier_one():
    result = assess_continuity_claims(
        text="Last time we talked about the tokenizer, you wanted to rewrite it.",
        memory_items=[],
        operator_state={},
        evidence_text="what about the tokenizer",
    )
    assert result.unsupported_continuity_claim is True


@pytest.mark.parametrize(
    "text",
    ["If you want, I'll run it now.", "You could batch the writes, but I'd profile first."],
)
def test_an_offer_is_not_an_invented_name(text):
    """The contractions "I'll" and "I'd" were read as proper nouns, so an offer
    addressed to the user was deleted as naming someone they had never mentioned."""
    result = assess(text)
    assert result.text == text
    assert result.unsupported_continuity_claim is False


def test_a_reply_that_was_all_invention_is_not_replaced_by_a_boot_banner():
    """It used to become "I am here. No active task is loaded yet...", mid-conversation."""
    result = assess("We were discussing the memory rewrite.")
    assert result.unsupported_continuity_claim is True
    assert result.text == UNSUPPORTED_CLAIM_REPLY


def test_what_the_user_said_earlier_in_the_conversation_is_evidence():
    alice = SimpleNamespace(
        llm=SimpleNamespace(
            conversation_history=[
                {"role": "user", "content": "the parser is slow, I think it's the tokenizer"},
                {"role": "assistant", "content": "Could be the regex backtracking."},
            ]
        )
    )
    req = SimpleNamespace(user_input="what would you do first?", tool_result=None)
    evidence = _turn_evidence_text(req, alice)
    assert "tokenizer" in evidence
    assert "backtracking" not in evidence
