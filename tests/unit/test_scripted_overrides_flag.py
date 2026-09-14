"""Whether a template may answer instead of the model.

Two things in this codebase look alike and are not. A template that
*substitutes* for a missing answer — the model was unreachable, or returned
nothing — is a fallback, and it stays. A template that *overrides* an answer
that already exists, because it failed a shape test, is how a direct reply
became a menu. Only the second is governed here.
"""

import pytest

from ai.infrastructure.runtime_flags import scripted_overrides_enabled
from app.main import ALICE


@pytest.fixture
def alice():
    """An uninitialised ALICE — these gates read only their arguments."""
    return ALICE.__new__(ALICE)


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("ALICE_ENABLE_SCRIPTED_OVERRIDES", raising=False)


def test_scripted_overrides_are_off_by_default():
    assert scripted_overrides_enabled() is False


def test_scripted_overrides_can_be_switched_on(monkeypatch):
    """The branches remain so the two behaviours can be compared with
    scripts/quality_harness.py rather than by argument."""
    monkeypatch.setenv("ALICE_ENABLE_SCRIPTED_OVERRIDES", "1")
    assert scripted_overrides_enabled() is True


@pytest.mark.parametrize(
    "prompt,intent",
    [
        ("teach me about nlp", "conversation:help"),
        ("i want to learn the foundations of an assistant system", "conversation:question"),
        ("give me some nlp algorithms", "conversation:help"),
    ],
)
def test_by_default_the_model_is_asked(alice, prompt, intent):
    """_self_answer_first_gate answered from a template without consulting the
    model at all — a regex deciding Alice should not think about this turn."""
    gate = alice._self_answer_first_gate(
        user_input=prompt,
        intent=intent,
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )
    assert gate["block_llm"] is False
    assert gate["reason"] == "scripted_overrides_disabled"
    assert gate["response"] == ""


def test_with_the_flag_on_the_template_answers_again(alice, monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_SCRIPTED_OVERRIDES", "1")
    gate = alice._self_answer_first_gate(
        user_input="teach me about nlp",
        intent="conversation:help",
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )
    assert gate["block_llm"] is True
    assert gate["reason"] == "structured_teaching_mode"
    assert gate["response"]


def test_the_templates_themselves_are_still_reachable(alice):
    """Turning the override off must not delete the fallbacks: when the model
    has produced nothing, one of these is the only thing left to say."""
    assert alice._structured_teaching_mode_response("teach me about nlp", "conversation:help")
    assert alice._deterministic_knowledge_fallback("what embeddings models should i use", "conversation:question")
