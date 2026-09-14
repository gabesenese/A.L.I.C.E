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

# The two paths a user spends a whole conversation inside. Both need something to
# imitate; the other two are one-shot renderings of a payload Alice already has.
DEMONSTRATED_PATHS = {"conversation": persona.for_conversation, "tools": persona.for_tools}


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


@pytest.mark.parametrize("name,build", DEMONSTRATED_PATHS.items())
def test_a_conversational_path_shows_as_well_as_tells(name, build):
    """An 8B local model imitates far better than it follows, so the worked
    exchanges are the payload rather than decoration."""
    replies = re.findall(r"^Alice:", build(), re.M)
    assert len(replies) >= 3, f"{name}: only {len(replies)} examples to imitate"


def test_the_examples_are_not_all_the_same_length():
    """Terse examples alone teach terseness, which reads as curt rather than
    direct — and collapses the length spread the feel report measures."""
    replies = re.findall(r"^Alice: (.+?)(?=\n\n|\Z)", persona.for_conversation(), re.S | re.M)
    lengths = sorted(len(r.split()) for r in replies)
    assert lengths[0] <= 3, f"shortest example is {lengths[0]} words — no floor to copy"
    assert lengths[-1] >= 40, f"longest example is {lengths[-1]} words — no ceiling to copy"


def test_the_identity_is_the_same_text_on_every_path():
    """The defect this module exists for: two paths that each grew their own
    character until the user was talking to a different Alice depending on
    whether her answer happened to need a tool."""
    core = persona.identity()
    for name, build in PATHS.items():
        assert build().startswith(core), f"{name} does not open with the shared character"


@pytest.mark.parametrize("name,build", PATHS.items())
def test_no_path_redeclares_the_character(name, build):
    """A path may say what is different about this turn. The moment it says who
    is speaking, it is a second persona and the two start drifting."""
    body = build()[len(persona.identity()) :]
    assert not re.search(r"\byou are (?:alice|a\b|an\b)", body, re.I), f"{name} re-declares the speaker"


@pytest.mark.parametrize("name,build", PATHS.items())
def test_every_path_fits_a_local_context_window(name, build):
    """num_ctx is 8192 on the conversational path and the loop shares it with
    ~700 tokens of tool schemas plus up to six observations. A prompt is paying
    rent on every single turn; roughly a tenth of the window is the ceiling."""
    tokens = len(build().split()) * 1.4  # a deliberately pessimistic words→tokens ratio
    assert tokens < 820, f"{name} is about {tokens:.0f} tokens"


def test_the_tool_path_still_insists_on_looking():
    """Personality must not cost reliability: she has to keep calling tools rather
    than guessing, or warmth just makes her confidently wrong."""
    text = persona.for_tools().lower()
    assert "call it" in text or "call the tool" in text
    assert "never guess" in text


def test_the_tool_path_shows_the_act_step():
    """Candidates whose exemplars jumped straight to confident prose scored worst
    on tool reliability. What gets imitated has to be look-then-speak."""
    assert re.search(r"^\(you call \w+", persona.for_tools(), re.M)


def test_context_is_appended_not_interleaved():
    built = persona.for_conversation(context="Earlier: he chose SQLite.")
    assert built.endswith("Earlier: he chose SQLite.")
    assert persona.identity() in built


def test_empty_context_adds_nothing():
    assert persona.for_conversation(context="   ") == persona.for_conversation()


@pytest.mark.parametrize("name,build", PATHS.items())
def test_the_user_name_is_configurable(name, build):
    """ "Gabriel" was hardcoded into roughly thirty prompt sites — including,
    in the winning draft, every speaker label in the transcript."""
    built = build(user_name="Sam")
    assert "Sam" in built, name
    assert "Gabriel" not in built, name


def test_no_path_leaves_an_unfilled_placeholder():
    """The bodies are format strings; a stray brace ships a literal "{user}" to
    the model, which is worse than a wrong name."""
    for name, build in PATHS.items():
        assert "{" not in build(), name
