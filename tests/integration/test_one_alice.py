"""There should be one Alice. There are five, and the user meets the wrong ones.

Executable evidence for the finding that the greeting, the tool loop, the phrasing
path and ordinary conversation are each driven by a different system prompt, and
that the character the repo actually wrote reaches only one of them.

Every test here is marked xfail(strict=True): it documents a defect that exists
today. When the prompts are unified, these XPASS — which pytest reports as a
failure — and the marker comes off. That is the intended signal, not a problem.
"""

import re

import pytest

# Applied per test rather than to the module, so a marker can come off the
# moment its defect is fixed. A strict xfail that starts passing is reported as
# a failure, which is the signal that the fix landed.
SPLIT_IDENTITY = pytest.mark.xfail(
    strict=True, reason="Alice's identity is split across five prompts; see docs/north_star.md"
)


def _personas():
    """Every system prompt on a path a real user turn can take."""
    from ai.core import llm_engine, react_loop
    from models.coding_model import CodingModel
    from models.fast_model import FastModel
    from models.reasoning_model import ReasoningModel

    return {
        # Greeting and farewell — the first and last thing a user ever sees.
        "router:fast": FastModel().system_prompt,
        "router:reasoning": ReasoningModel().system_prompt,
        "router:coding": CodingModel().system_prompt,
        # Most substantive turns, now that ordinary turns reach the agent loop.
        "agent_loop": react_loop.SYSTEM_PROMPT,
        # Clarifications and structured payloads.
        "phraser": llm_engine.PHRASER_PROMPT,
    }


@SPLIT_IDENTITY
def test_every_user_facing_prompt_says_who_she_is():
    """A prompt whose identity is "You are a concise assistant" produces a concise
    assistant. The user opens Alice and is greeted by a generic short-answer bot."""
    anonymous = [name for name, prompt in _personas().items() if not re.search(r"\balice\b", prompt, re.I)]
    assert not anonymous, f"prompts with no identity: {anonymous}"


@SPLIT_IDENTITY
def test_every_user_facing_prompt_knows_who_it_is_talking_to():
    """Continuity is the whole point of a companion. A prompt that does not name
    the user cannot behave like one."""
    strangers = [name for name, prompt in _personas().items() if not re.search(r"\bgabriel\b", prompt, re.I)]
    assert not strangers, f"prompts that do not know the user: {strangers}"


@SPLIT_IDENTITY
def test_the_phrasing_path_is_not_told_it_is_not_alice():
    """PHRASER_PROMPT tells the model it is a text formatter and must not add
    personality, and the caller then hands it Alice's personality block. Told to
    be a formatter, it formats — which is what a rendered field reads like."""
    from ai.core.llm_engine import PHRASER_PROMPT

    disavowals = re.findall(r"(?:DO NOT act as Alice|not Alice|DO NOT add personality)", PHRASER_PROMPT, re.I)
    assert not disavowals, f"the phrasing path disavows being Alice: {disavowals}"


def _persona_text() -> str:
    """The literal persona assigned to LocalLLMEngine.system_prompt."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "ai" / "core" / "llm_engine.py"
    match = re.search(
        r'self\.system_prompt\s*=\s*"""(.*?)"""',
        source.read_text(encoding="utf-8"),
        re.S,
    )
    assert match, "could not locate the persona assignment in llm_engine.py"
    return match.group(1)


@SPLIT_IDENTITY
def test_the_persona_is_not_mostly_prohibitions():
    """The persona enumerates everything Alice must not say and almost nothing she
    should sound like. For a model optimising against dozens of prohibitions, the
    shortest bare declarative sentence violates the fewest rules — so she avoids
    sounding like a chatbot by sounding like nothing."""
    persona = _persona_text()
    negatives = len(re.findall(r"\b(?:never|don'?t|do not|avoid|must not|no longer)\b", persona, re.I))
    words = len(persona.split())
    assert negatives <= 8, f"{negatives} negative constraints in a {words}-word persona"


@SPLIT_IDENTITY
def test_the_persona_shows_rather_than_only_tells():
    """An 8B local model imitates far better than it follows. A persona with no
    worked example of an actual Alice reply gives it nothing to copy."""
    persona = _persona_text()
    has_example = bool(re.search(r"(?:^|\n)\s*(?:user|gabriel|example|e\.g\.)\s*[:>]", persona, re.I))
    assert has_example, "the persona contains no example exchange to imitate"


def _code_without_comments(path) -> str:
    """Source with comment lines dropped.

    A comment that quotes a removed prompt string reads identically to the
    string itself, so a plain grep cannot tell "we deleted this" from "this is
    still here". Only live code counts.
    """
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        kept.append(line.split("  #", 1)[0])
    return "\n".join(kept)


def test_the_prompt_does_not_instruct_her_to_offer_instead_of_act():
    """app/main.py built a block ending with an instruction to confirm she has
    code access and offer to read files — talking about looking rather than
    looking, the exact failure docs/north_star.md names as the recurring bug."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "app" / "main.py"
    offenders = re.findall(
        r"offer to read/analyze files|confirm you have it",
        _code_without_comments(source),
        re.I,
    )
    assert not offenders, f"prompt instructs her to offer rather than act: {offenders}"


@SPLIT_IDENTITY
def test_personality_is_addressed_to_alice_not_about_her():
    """build_personality_system_instructions emits "Current ALICE personality
    drift:" followed by dials. That is a monitoring readout describing a system
    called ALICE, handed to a model that is supposed to BE her. Configuration,
    not character."""
    from brain.personality import apply_personality_to_system_prompt

    rendered = apply_personality_to_system_prompt("You are a concise assistant.")
    assert "personality drift" not in rendered.lower(), (
        "personality is injected as a third-person status readout: " + rendered[:200]
    )


@SPLIT_IDENTITY
def test_stated_user_interests_are_plausible_interests():
    """The live block currently lists "wednesday" and "forecast" among the topics
    Gabriel cares about. A day of the week is not an interest; it is a word that
    appeared in a query."""
    from brain.personality import apply_personality_to_system_prompt

    rendered = apply_personality_to_system_prompt("")
    match = re.search(r"Topics this user cares about:\s*(.+)", rendered)
    if not match:
        pytest.skip("no interests currently recorded")

    topics = {t.strip().lower().rstrip(".") for t in match.group(1).split(",")}
    not_interests = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "today",
        "tomorrow",
        "yesterday",
        "forecast",
        "now",
        "week",
        "weekend",
    }
    junk = topics & not_interests
    assert not junk, f"calendar and query words listed as personal interests: {sorted(junk)}"
