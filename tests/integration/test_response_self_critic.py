"""Tests for response self-critique quality checks."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.response_self_critic import ResponseSelfCritic


def test_self_critic_flags_domain_mismatch() -> None:
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="what is the weather today in toronto",
        intent="weather:current",
        entities={"location": "toronto"},
        response="Your note was saved successfully.",
        memory_snapshot=None,
    )

    assert result.passed is False
    assert any("intent-domain mismatch" in reason for reason in result.fail_reasons)


def test_self_critic_flags_unsupported_citation_pattern() -> None:
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="teach me polymorphism",
        intent="learning:study_topic",
        entities={"topic": "polymorphism"},
        response="According to 2024 study [1], polymorphism is always best.",
        memory_snapshot=None,
    )

    assert result.passed is False
    assert any("citation" in reason for reason in result.fail_reasons)


def test_self_critic_flags_memory_contradiction() -> None:
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="what is my name",
        intent="conversation:question",
        entities={},
        response="I don't know your name.",
        memory_snapshot={"user_name": "Gabriel"},
    )

    assert result.passed is False
    assert any("known user name" in reason for reason in result.fail_reasons)


def test_self_critic_passes_relevant_answer() -> None:
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="help me study polymorphism",
        intent="learning:study_topic",
        entities={"topic": "polymorphism"},
        response="Polymorphism lets one interface work with multiple concrete types in object-oriented design.",
        memory_snapshot={"user_name": "Gabriel"},
    )

    assert result.passed is True


def test_self_critic_flags_assumption_without_evidence() -> None:
    """Response that asks 'what still needs improving' after user says it's already wrong."""
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="actually that's wrong, it already exists",
        intent="conversation:correction",
        entities={},
        response="I see, what other aspects would you like me to improve or focus on?",
        memory_snapshot=None,
    )

    assert result.passed is False
    assert any("assumption-without-evidence" in r for r in result.fail_reasons)
    assert result.correction_hint is not None
    assert "retract" in result.correction_hint.lower()


def test_self_critic_flags_redundant_suggestion() -> None:
    """Response that suggests adding a feature the user confirmed already exists."""
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="that feature is already implemented and working fine",
        intent="conversation:feedback",
        entities={},
        response="You should consider adding that feature to improve the system.",
        memory_snapshot=None,
    )

    assert result.passed is False
    assert any("redundant-suggestion" in r for r in result.fail_reasons)
    assert result.correction_hint is not None
    assert "already exists" in result.correction_hint.lower()


def test_self_critic_correction_hint_is_none_for_clean_response() -> None:
    critic = ResponseSelfCritic()

    result = critic.assess(
        user_input="what time is it",
        intent="time:current",
        entities={},
        response="The current time is 3:45 PM.",
        memory_snapshot=None,
    )

    assert result.correction_hint is None
