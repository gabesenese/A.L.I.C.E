"""A reply's shape changes only when he asks for a shape.

"list" anywhere in a message counted as asking for bullet points, so "clear my
shopping list" came back as "- Cleared your shopping list.\\n- It had eggs and
bread." The converter also kept only the first six sentences.
"""

import pytest

from ai.core.adaptive_response_style import AdaptiveResponseStyle
from ai.core.constraint_preference_extractor import ConstraintPreferenceExtractor


@pytest.mark.parametrize(
    "text",
    [
        "clear my shopping list",
        "what's on my shopping list?",
        "play my workout playlist",
        "is the build stable?",
        "listen, I need a hand with this",
        "tell me the history of rome",
        "should I see a specialist about it?",
    ],
)
def test_mentioning_a_list_is_not_asking_for_one(text):
    assert ConstraintPreferenceExtractor().extract(text)["format"] == "default"


@pytest.mark.parametrize(
    "text, shape",
    [
        ("give me the options in bullet points", "bullet_points"),
        ("can you list them out", "bullet_points"),
        ("put it as a list", "bullet_points"),
        ("summarise it in point form", "bullet_points"),
        ("show it as a table", "table"),
        ("write it as a story", "narrative"),
    ],
)
def test_asking_for_a_shape_is_heard(text, shape):
    assert ConstraintPreferenceExtractor().extract(text)["format"] == shape


def test_bullets_keep_every_sentence():
    reply = " ".join(f"Point {n} matters." for n in range(1, 9))

    out = AdaptiveResponseStyle().apply_constraints(reply, {"format": "bullet_points"})

    assert out.splitlines() == [f"- Point {n} matters." for n in range(1, 9)]


def test_two_sentences_are_left_as_they_are():
    reply = "Cleared your shopping list. It had eggs and bread."

    assert AdaptiveResponseStyle().apply_constraints(reply, {"format": "bullet_points"}) == reply


def test_trimming_filler_leaves_the_words_that_carry_meaning():
    styler = AdaptiveResponseStyle()

    assert styler.apply_constraints("Make sure the backup ran first.", {"max_words": 20}) == (
        "Make sure the backup ran first."
    )
    assert styler.apply_constraints("Absolutely.", {"max_words": 20}) == "Absolutely."
    assert styler.apply_constraints("Sure, the backup ran.", {"max_words": 20}) == "The backup ran."
