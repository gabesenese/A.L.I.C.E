"""A "that" that opens a clause refers to nothing, and is not a question to ask.

"remember that my sister's name is Ana" was treated as a short turn with an
unresolved pronoun, so it was sent for clarification, rerouted as a generic goal
statement, and never stored. With a subject in context it was worse: the word
was overwritten, giving "remember sqlite my sister's name is Ana".
"""

import pytest

from ai.reference_resolver import ReferenceResolver


@pytest.fixture
def resolver(monkeypatch):
    import ai.reference_resolver as module

    class _NoRegistry:
        def resolve_reference(self, _text):
            return ""

        def register(self, **_kwargs):
            return None

    monkeypatch.setattr(module, "get_entity_registry", lambda: _NoRegistry())
    return ReferenceResolver()


@pytest.mark.parametrize(
    "text",
    [
        "remember that my sister's name is Ana",
        "remember that I prefer tea",
        "I think that the parser is fine",
        "save this: the wifi code is on the fridge",
    ],
)
def test_a_clause_or_a_forward_pointer_is_not_an_unresolved_reference(resolver, text):
    result = resolver.resolve(text, {})
    assert result.unresolved_pronouns == []
    assert result.rewritten_input == text


def test_it_is_not_overwritten_when_a_subject_is_known(resolver):
    text = "remember that my sister's name is Ana"
    assert resolver.resolve(text, {"last_subject": "sqlite"}).rewritten_input == text


@pytest.mark.parametrize("text", ["delete that", "what does that do?", "open this"])
def test_a_bare_reference_is_still_unresolved(resolver, text):
    assert resolver.resolve(text, {}).unresolved_pronouns


def test_the_reference_is_replaced_not_the_clause_marker(resolver):
    result = resolver.resolve("I know that you said that", {"last_subject": "the tokenizer"})
    assert result.rewritten_input == "I know that you said the tokenizer"


@pytest.mark.parametrize(
    "text", ["what time is it?", "what day is it", "is it raining?", "is it going to snow", "how's it going"]
)
def test_the_it_of_the_weather_and_the_clock_is_not_a_reference(resolver, text):
    """ "What time is it?" was sent for clarification and answered as a generic
    statement, because its "it" counted as an unresolved pronoun."""
    assert resolver.resolve(text, {}).unresolved_pronouns == []
