"""What he says about himself sticks without "remember that".

The extractor runs on every turn but kept only statements that hit a domain
keyword (relationships, preferences, fitness...). Everything else was filed as
"general" and dropped, so a birthday, a job, an allergy, a dog or an appointment
was forgotten unless he explicitly asked her to remember it.
"""

import pytest

from ai.memory.memory_extractor import MemoryExtractor


def _stored(text):
    candidates = MemoryExtractor().extract_from_user_turn(user_text=text, user_name="Gabriel", source="conversation")
    return [c for c in candidates if c.should_store]


@pytest.mark.parametrize(
    "text",
    [
        "my dentist appointment is on friday at 3pm",
        "my birthday is march 3rd",
        "I work as a nurse at the general hospital",
        "I have a dog called Max",
        "I am allergic to peanuts",
        "I'm from Porto",
        "I live in Kitchener",
    ],
)
def test_a_fact_about_himself_is_kept(text):
    stored = _stored(text)
    assert [(c.domain, c.kind, c.scope) for c in stored] == [("personal_life", "personal_fact", "long_term")]
    assert text in stored[0].content


@pytest.mark.parametrize(
    "text",
    [
        "my code is broken",
        "I have a question",
        "what's my sister's name?",
        "if my flight is late, text me",
        "what time is it",
    ],
)
def test_passing_remarks_and_questions_are_not(text):
    assert [c for c in _stored(text) if c.kind == "personal_fact"] == []
