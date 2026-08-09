"""Padding removal for generated replies.

A retry rule forced any answer under fifty words into a three or four sentence
"take", so a one word confirmation came back as a 135 word essay that opened with
a compliment and closed with an offer to explore further.
"""

from ai.runtime.response_discipline import (
    apply_response_discipline,
    limit_sentences,
    strip_filler_closing,
    strip_filler_opening,
)

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
