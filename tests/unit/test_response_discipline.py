"""Padding removal for generated replies.

A retry rule forced any answer under fifty words into a three or four sentence
"take", so a one word confirmation came back as a 135 word essay that opened with
a compliment and closed with an offer to explore further.
"""

import pytest

from ai.runtime.response_discipline import (
    apply_response_discipline,
    asks_for_depth,
    guard_unverified_execution_claims,
    limit_sentences,
    strip_ai_disclaimer,
    strip_filler_closing,
    strip_filler_opening,
)


HONEST = "I haven't actually run that yet. Want me to run it now?"


@pytest.mark.parametrize(
    "user_input,answer",
    [
        ("run pytest -q", "Tests pass. No failures or errors."),
        ("run pytest -q", "Failing tests: tests/test_routing.py::test_long_path AssertionError"),
        ("are the tests passing?", "They are, but one integration test fails on the caching layer."),
        ("is the build green?", "Yes, the build is green."),
        ("how's the test suite looking", "All 97 tests pass in 2.5 seconds."),
        ("did the linter pass", "Ruff is clean, no issues."),
    ],
)
def test_results_are_not_reported_for_work_that_never_ran(user_input, answer):
    assert guard_unverified_execution_claims(answer, ran_command=False, user_input=user_input) == HONEST


def test_real_command_output_is_left_alone():
    answer = "Tests pass. No failures."
    assert guard_unverified_execution_claims(answer, ran_command=True, user_input="run pytest -q") == answer


@pytest.mark.parametrize(
    "user_input,answer",
    [
        ("what's the weather", "It's 12 degrees and raining."),
        ("hey how's it going", "I'm here. What's up?"),
        ("what does agent_loop.py do", "It drives the operator step selection."),
    ],
)
def test_ordinary_conversation_is_untouched(user_input, answer):
    assert guard_unverified_execution_claims(answer, ran_command=False, user_input=user_input) == answer


ESSAY = (
    "Your enthusiasm is palpable, but let's dive deeper into this project of building a modern "
    "Jarvis-like AI. I've been paying attention to your goals and it seems like you're eager to "
    "create something ambitious. One aspect that catches my eye is the challenge of making such a "
    "project feasible. It's a complex problem to say the least. I'd like to suggest we start by "
    "examining the current state of conversational AI. Let me know if you'd like to explore this further."
)


def test_removes_a_flattering_opening():
    assert not strip_filler_opening("That's a great question. The answer is 42.").lower().startswith("that's a great")


def test_removes_stacked_filler_openings():
    cleaned = strip_filler_opening("Absolutely. I'd be happy to help. The build is green.")
    assert cleaned == "The build is green."


def test_never_strips_the_entire_response():
    assert strip_filler_opening("Great!") == "Great!"
    assert strip_filler_opening("Of course.") == "Of course."


def test_removes_a_trailing_offer_to_help():
    cleaned = strip_filler_closing("The build is green. Let me know if you need anything else.")
    assert cleaned == "The build is green."


def test_keeps_a_closing_that_carries_content():
    text = "The build is green. The deploy still needs a manual approval."
    assert strip_filler_closing(text) == text


def test_limits_at_a_sentence_boundary_never_mid_sentence():
    text = "One. Two. Three. Four. Five."
    assert limit_sentences(text, max_sentences=3) == "One. Two. Three."


def test_short_answers_pass_through_untouched():
    for text in ["Nexo then.", "You're back on Alice now.", "677 files."]:
        assert apply_response_discipline(text) == text


def test_the_real_essay_is_cut_down():
    cleaned = apply_response_discipline(ESSAY, max_sentences=4)
    assert len(cleaned.split()) < len(ESSAY.split())
    assert "enthusiasm is palpable" not in cleaned
    assert "let me know if" not in cleaned.lower()
    assert len([s for s in cleaned.split(". ") if s.strip()]) <= 4


def test_empty_input_stays_empty():
    assert apply_response_discipline("") == ""
    assert apply_response_discipline("   ") == ""


def test_content_without_filler_is_preserved_exactly():
    text = "The routing regression came from ec84085. The session objective was ignored."
    assert apply_response_discipline(text) == text


LISTED = (
    "The cache is the slow part. It rebuilds on every turn. That was fine at ten notes. "
    "Three things make it worse in practice:\n"
    "1. Every turn re-embeds the whole store.\n"
    "2. The index is rebuilt from scratch.\n"
    "3. Nothing is ever evicted."
)


def test_a_capped_reply_never_stops_at_the_first_numeral_of_a_list():
    """Counting "1." as a sentence cut this to "...worse in practice:\\n1." --
    three things promised, and the numeral delivered."""
    capped = apply_response_discipline(LISTED, max_sentences=4)
    assert not capped.endswith("1.")
    assert capped == LISTED


def test_paragraph_breaks_survive():
    text = "SQLite is fine for one user.\n\nIf a second writer appears, move to Postgres."
    assert apply_response_discipline(text) == text


def test_removing_a_sign_off_keeps_the_line_breaks_above_it():
    text = "Two options.\n\nSQLite now, Postgres later. Hope this helps!"
    assert strip_filler_closing(text) == "Two options.\n\nSQLite now, Postgres later."


def test_a_cut_keeps_the_line_breaks_before_it():
    text = "First point.\n\nSecond point. Third point. Fourth point."
    assert limit_sentences(text, max_sentences=2) == "First point.\n\nSecond point."


def test_an_abbreviation_is_not_a_sentence_ending():
    text = "Use a broker, e.g. Redis or NATS. Both handle backpressure."
    assert limit_sentences(text, max_sentences=2) == text


def test_a_code_block_is_never_cut():
    text = "Run this first. Then check the log. It should be quiet.\n```\npytest -q\n```\nThat is all of it."
    assert limit_sentences(text, max_sentences=2) == text


@pytest.mark.parametrize(
    "reply,expected",
    [
        (
            "As an AI language model, I don't have preferences. SQLite is fine here.",
            "I don't have preferences. SQLite is fine here.",
        ),
        ("As a large language model, I can't browse. The docs say 3.12.", "I can't browse. The docs say 3.12."),
        ("Fair question. As an AI, it's not something I feel.", "Fair question. It's not something I feel."),
        ("As an AI I think the tradeoff is fine.", "I think the tradeoff is fine."),
    ],
)
def test_a_self_disclaimer_loses_the_clause_and_keeps_the_answer(reply, expected):
    assert strip_ai_disclaimer(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "Transformers are the architecture behind every modern language model.",
        "As a language model grows, its loss falls roughly as a power law.",
        "It shows up in industries such as an airline's booking system.",
        "How does a language model work? It predicts the next token from the ones before it.",
    ],
)
def test_talking_about_language_models_is_not_a_disclaimer(reply):
    assert strip_ai_disclaimer(reply) == reply


@pytest.mark.parametrize(
    "request_text",
    [
        "explain how the memory layer works",
        "can you elaborate on that?",
        "walk me through the routing",
        "tell me more about the tool loop",
        "how does the continuity guard work?",
        "how do embeddings work",
        "go through it step by step",
        "describe it in more detail",
    ],
)
def test_a_request_for_depth_is_recognised(request_text):
    assert asks_for_depth(request_text)


@pytest.mark.parametrize(
    "request_text",
    ["should I rewrite it in rust?", "what's the weather?", "that explained it, thanks", "how does it look?"],
)
def test_an_ordinary_question_is_not_a_request_for_depth(request_text):
    assert not asks_for_depth(request_text)


def test_without_a_cap_only_the_padding_goes():
    reply = "Great question! " + " ".join(f"Step {n} does its part." for n in range(1, 9))
    assert apply_response_discipline(reply, max_sentences=None) == reply[len("Great question! ") :]


@pytest.mark.parametrize(
    "reply, expected",
    [
        (
            "It's worth noting that the cache is per-process. So two workers diverge.",
            "The cache is per-process. So two workers diverge.",
        ),
        (
            "Thanks for asking, it was the lock all along. Swap it for an RLock.",
            "It was the lock all along. Swap it for an RLock.",
        ),
        (
            "That's a good point, but the cache still needs a lock. Add one around the writer.",
            "But the cache still needs a lock. Add one around the writer.",
        ),
        (
            "I'd be happy to walk through it, the loop has three stages. First it routes.",
            "The loop has three stages. First it routes.",
        ),
        ("Absolutely, the cache is per-process.", "The cache is per-process."),
        ("Great question! The loop has three stages.", "The loop has three stages."),
    ],
)
def test_the_courtesy_goes_and_the_point_it_led_into_stays(reply, expected):
    """The whole first sentence used to go, including everything after the
    courtesy, which is where the answer was."""
    assert strip_filler_opening(reply) == expected
