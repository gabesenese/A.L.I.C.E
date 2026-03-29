"""
Regression tests for conversational routing and learning guardrails.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.conversational_engine import ConversationalEngine, ConversationalContext
from ai.infrastructure.router import RequestRouter, RoutingDecision
from ai.learning.phrasing_learner import PhrasingLearner


def _context(user_input: str, intent: str) -> ConversationalContext:
    return ConversationalContext(
        user_input=user_input,
        intent=intent,
        entities={},
        recent_topics=[],
        active_goal=None,
        world_state=None,
    )


def test_conversational_engine_ignores_substring_greeting_match() -> None:
    """Words like 'polymorphism' should not trigger 'hi' greeting matching."""
    engine = ConversationalEngine(memory_system=None, training_collector=None, world_state=None)
    ctx = _context("what is polymorphism", "conversation:question")

    assert engine.can_handle("what is polymorphism", "conversation:question", ctx) is False


def test_phrasing_learner_skips_high_variance_conversation_patterns(tmp_path: Path) -> None:
    """Open-ended conversation buckets should not be learned/replayed directly."""
    learner = PhrasingLearner(storage_path=str(tmp_path / "phrasing.jsonl"))
    thought = {
        "type": "conversation:general",
        "data": {"user_input": "what is polymorphism"},
    }

    for i in range(4):
        learner.record_phrasing(
            alice_thought=thought,
            ollama_phrasing=f"sample response {i}",
            context={"tone": "helpful"},
        )

    assert learner.can_phrase_myself(thought, "helpful") is False


def test_phrasing_learner_learns_low_complexity_help_opener(tmp_path: Path) -> None:
    learner = PhrasingLearner(storage_path=str(tmp_path / "phrasing_help.jsonl"))
    thought = {
        "type": "conversation:help_opener",
        "data": {"user_input": "i need help with my project"},
    }

    for _ in range(3):
        learner.record_phrasing(
            alice_thought=thought,
            ollama_phrasing="Of course. What part of your project should we focus on first?",
            context={"tone": "helpful"},
        )

    assert learner.can_phrase_myself(thought, "helpful") is True


def test_router_does_not_intercept_low_confidence_conversation_question() -> None:
    """Low-confidence conversational questions should fall through to LLM fallback."""
    router = RequestRouter()

    result = router.route(
        intent="conversation:question",
        confidence=0.52,
        entities={},
        user_text="what is ram",
    )

    assert result.decision == RoutingDecision.LLM_FALLBACK
