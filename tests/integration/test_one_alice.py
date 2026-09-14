"""There should be one Alice. There were five, and the user met the wrong ones.

The greeting, the tool loop, the phrasing path and ordinary conversation were each
driven by a different system prompt, and the character the repo actually wrote
reached only one of them. Every test here started as xfail(strict=True) evidence
for that; they are live assertions now, and their job is to stop a sixth prompt
growing back. A new user-facing path belongs in _personas(), composed from
ai/core/persona.py — not written fresh.
"""

import re

from ai.core import persona

# Words frequent enough to clear the three-mention bar without ever being a thing
# a person is interested in.
_NOT_INTERESTS = frozenset(
    {
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
        "weather",
        "temperature",
        "now",
        "week",
        "weekend",
    }
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


def test_every_user_facing_prompt_says_who_she_is():
    """A prompt whose identity is "You are a concise assistant" produces a concise
    assistant. The user opens Alice and is greeted by a generic short-answer bot."""
    anonymous = [name for name, prompt in _personas().items() if not re.search(r"\balice\b", prompt, re.I)]
    assert not anonymous, f"prompts with no identity: {anonymous}"


def test_every_user_facing_prompt_knows_who_it_is_talking_to():
    """Continuity is the whole point of a companion. A prompt that does not name
    the user cannot behave like one."""
    strangers = [name for name, prompt in _personas().items() if not re.search(r"\bgabriel\b", prompt, re.I)]
    assert not strangers, f"prompts that do not know the user: {strangers}"


def test_the_phrasing_path_is_not_told_it_is_not_alice():
    """PHRASER_PROMPT tells the model it is a text formatter and must not add
    personality, and the caller then hands it Alice's personality block. Told to
    be a formatter, it formats — which is what a rendered field reads like."""
    from ai.core.llm_engine import PHRASER_PROMPT

    disavowals = re.findall(r"(?:DO NOT act as Alice|not Alice|DO NOT add personality)", PHRASER_PROMPT, re.I)
    assert not disavowals, f"the phrasing path disavows being Alice: {disavowals}"


def _persona_text() -> str:
    """The persona an ordinary conversational turn actually runs on.

    This used to scrape a triple-quoted literal out of llm_engine.py, because
    that is where the persona lived. It is composed now, so the engine is asked
    what it is holding rather than the source being read — which is also what
    keeps this honest if someone reintroduces a literal: a literal that is not
    assigned is not what runs.
    """
    from ai.core.llm_engine import LocalLLMEngine

    engine = LocalLLMEngine.__new__(LocalLLMEngine)
    LocalLLMEngine.__init__(engine)
    assert engine.system_prompt == persona.for_conversation(), "the engine no longer runs on the shared persona"
    return engine.system_prompt


def test_the_persona_is_not_mostly_prohibitions():
    """The persona enumerated everything Alice must not say and almost nothing she
    should sound like — 777 words carrying 25 prohibitions. For a model optimising
    against that many rules, the shortest bare declarative sentence violates the
    fewest, so she avoided sounding like a chatbot by sounding like nothing."""
    text = _persona_text()
    negatives = len(re.findall(r"\b(?:never|don'?t|do not|avoid|must not|no longer)\b", text, re.I))
    words = len(text.split())
    assert negatives <= 8, f"{negatives} negative constraints in a {words}-word persona"


def test_the_persona_shows_rather_than_only_tells():
    """An 8B local model imitates far better than it follows. A persona with no
    worked example of an actual Alice reply gives it nothing to copy."""
    text = _persona_text()
    has_example = bool(re.search(r"(?:^|\n)\s*(?:user|gabriel|example|e\.g\.)\s*[:>]", text, re.I))
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


def test_no_composed_layer_tells_her_to_offer_rather_than_act():
    """The same failure as app/main.py above, one layer down. The stale-data
    warning ended "(offer to refresh if relevant)" — an instruction to narrate a
    capability she has, on a turn where she could simply use it. docs/north_star.md
    exists because this keeps growing back somewhere new.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for relative in ("ai/core/llm_engine.py", "ai/core/react_loop.py", "ai/core/persona.py", "brain/personality.py"):
        source = _code_without_comments(root / relative)
        for hit in re.findall(r"offer to (?:refresh|read|check|look|analyze|fetch)[^\"']*", source, re.I):
            offenders.append(f"{relative}: {hit}")
    assert not offenders, f"prompt text instructs her to offer rather than act: {offenders}"


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


def test_stated_user_interests_are_plausible_interests():
    """The live block listed "wednesday" and "forecast" among the things Gabriel
    cares about. A day of the week is not an interest; it is a word that happened
    to appear in a query three times, which is the tell that the extractor counts
    tokens rather than recognising subjects."""
    from brain.personality import apply_personality_to_system_prompt

    rendered = apply_personality_to_system_prompt("")
    match = re.search(r"He has been working on:\s*(.+)", rendered)
    if not match:
        return  # nothing recorded yet; the filter is covered directly below

    topics = {t.strip().lower().rstrip(".") for t in match.group(1).split(",")}
    assert not (topics & _NOT_INTERESTS), f"query words listed as interests: {sorted(topics & _NOT_INTERESTS)}"


def test_calendar_and_weather_words_are_filtered_whatever_is_stored():
    """Independent of what this machine happens to have learned, so the guard
    still holds on a fresh clone with an empty world model."""
    from brain.personality import _meaningful_interests

    stored = ["wednesday", "forecast", "heartbeat", "tomorrow", "embeddings", "temperature"]
    assert _meaningful_interests(stored) == ["heartbeat", "embeddings"]


def test_calendar_words_never_become_interests_in_the_first_place():
    """Filtering at render leaves the junk accumulating in the world model, where
    it crowds out the real topics before the filter ever sees them."""
    from brain.personality import _topic_candidates

    candidates = set(_topic_candidates("what's the forecast for wednesday in the heartbeat runtime"))
    assert "heartbeat" in candidates
    assert not (candidates & _NOT_INTERESTS)


def test_the_prompts_agree_about_ending_with_a_question():
    """Three of the five prompts give three different answers.

    react_loop: "no offers to help further".
    PHRASER_PROMPT: "DO NOT suggest follow-up topics or ask what the user wants
    to talk about next".
    The persona: "always end with one specific follow-up that keeps the thread
    moving".

    So whether Alice ends a turn with a question depends on which prompt served
    it. A model that cannot resolve a contradiction falls back to the safest
    register it knows, which is the flat one — and the inconsistency itself reads
    as not really listening.
    """
    from ai.core import llm_engine, react_loop

    never = []
    always = []
    for name, prompt in (
        ("agent_loop", react_loop.SYSTEM_PROMPT),
        ("phraser", llm_engine.PHRASER_PROMPT),
        ("persona", _persona_text()),
    ):
        lowered = prompt.lower()
        if re.search(r"no offers to help|do not suggest follow-up|don'?t ask what", lowered):
            never.append(name)
        if re.search(r"always end with one specific follow-up|end with .{0,20}question", lowered):
            always.append(name)

    assert not (never and always), f"prompts telling her never to ask: {never}; and always to ask: {always}"
