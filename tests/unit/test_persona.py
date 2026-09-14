"""One identity, composed per path — the property the old prompts violated."""

import re

import pytest

from ai.core import persona

PATHS = {
    "conversation": persona.for_conversation,
    "tools": persona.for_tools,
    "phrasing": persona.for_phrasing,
    "brief": persona.for_brief_reply,
}


@pytest.mark.parametrize("name,build", PATHS.items())
def test_every_path_says_who_she_is(name, build):
    """The greeting path's whole identity was "You are a concise assistant"."""
    assert re.search(r"\bAlice\b", build()), name


@pytest.mark.parametrize("name,build", PATHS.items())
def test_every_path_knows_who_it_is_talking_to(name, build):
    assert re.search(r"\bGabriel\b", build()), name


@pytest.mark.parametrize("name,build", PATHS.items())
def test_no_path_disavows_being_alice(name, build):
    """The phrasing prompt said "You are a natural language generator for Alice"
    and "DO NOT add personality". Told to be a formatter, it formats."""
    text = build().lower()
    assert "generator for alice" not in text, name
    assert "do not add personality" not in text, name


@pytest.mark.parametrize("name,build", PATHS.items())
def test_no_path_is_mostly_prohibitions(name, build):
    """777 words carrying 25 negatives taught the model that the shortest bare
    declarative sentence violates the fewest rules."""
    text = build()
    negatives = len(re.findall(r"\b(?:never|don'?t|do not|avoid|must not|no longer)\b", text, re.I))
    assert negatives <= 8, f"{name}: {negatives} negative constraints"


def test_the_identity_shows_as_well_as_tells():
    """An 8B local model imitates far better than it follows."""
    text = persona.identity()
    exchanges = re.findall(r"^Alice:", text, re.M)
    assert len(exchanges) >= 2, "no worked examples to imitate"


def test_the_identity_is_the_same_object_on_every_path():
    core = persona.identity()
    for name, build in PATHS.items():
        assert core in build(), f"{name} does not carry the shared identity"


def test_each_path_adds_only_what_is_different_about_this_turn():
    """An addendum describes the job, not the speaker."""
    core_words = len(persona.identity().split())
    for name, build in PATHS.items():
        extra = len(build().split()) - core_words
        assert extra <= 120, f"{name} addendum is {extra} words — it is becoming a second persona"


def test_the_tool_path_still_insists_on_looking():
    """Personality must not cost reliability: she has to keep calling tools rather
    than guessing, or warmth just makes her confidently wrong."""
    text = persona.for_tools().lower()
    assert "call it" in text or "call the tool" in text
    assert "never guess" in text


def test_context_is_appended_not_interleaved():
    built = persona.for_conversation(context="Earlier: he chose SQLite.")
    assert built.endswith("Earlier: he chose SQLite.")
    assert persona.identity() in built


def test_empty_context_adds_nothing():
    assert persona.for_conversation(context="   ") == persona.for_conversation()


def test_the_user_name_is_configurable():
    """ "Gabriel" was hardcoded into roughly thirty prompt sites."""
    built = persona.for_conversation(user_name="Sam")
    assert "Sam" in built
    assert "Gabriel" not in built


def test_the_identity_fits_a_local_context_window():
    """It competes with memory and history for an 8B model's context."""
    assert len(persona.identity().split()) < 260
