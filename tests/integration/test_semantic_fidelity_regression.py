from dataclasses import dataclass

from ai.context_resolver import ContextResolver
from ai.core.executive_controller import ExecutiveController
from app.main import ALICE


EXACT_PROMPT = (
    "i want to learn the foundations an advanced assistant system should have "
    "in today's world with no fiction"
)


@dataclass
class _FakeResolveResult:
    rewritten_input: str
    resolved_bindings: dict
    unresolved_pronouns: list


def test_context_resolver_keeps_raw_input_when_rewrite_has_placeholder_noise(monkeypatch):
    resolver = ContextResolver()

    def _fake_resolve(user_input, state):
        return _FakeResolveResult(
            rewritten_input="the foundations person 'an ai' assistant general_assistance",
            resolved_bindings={"it": "assistant"},
            unresolved_pronouns=[],
        )

    monkeypatch.setattr(resolver.reference_resolver, "resolve", _fake_resolve)
    monkeypatch.setattr(resolver.ambiguity_detector, "should_clarify", lambda **kwargs: False)

    result = resolver.resolve(EXACT_PROMPT, {"referenced_entities": []})

    assert result.rewritten_input == EXACT_PROMPT
    assert result.rewrite_confidence == 0.0


def test_semantic_fidelity_guard_rejects_programming_drift_response():
    controller = ExecutiveController()
    bad = (
        "Polymorphism and interface inheritance are the foundations to understand this topic."
    )

    evaluation = controller.evaluate_response(
        user_input=EXACT_PROMPT,
        intent="conversation:question",
        response=bad,
        route="llm",
        context={},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] in {
        "semantic_drift_programming_domain",
        "semantic_core_missing",
    }


def test_semantic_fidelity_guard_accepts_on_topic_foundations_response():
    controller = ExecutiveController()
    good = (
        "A real-world assistant system needs natural language understanding, "
        "memory, planning, execution, verification, and bounded autonomy."
    )

    evaluation = controller.evaluate_response(
        user_input=EXACT_PROMPT,
        intent="conversation:question",
        response=good,
        route="llm",
        context={},
    )

    assert evaluation["accepted"] is True


def test_study_flow_not_promoted_for_broad_conceptual_question_without_explicit_study_request():
    alice = ALICE.__new__(ALICE)

    intent, entities = alice._promote_learning_goal_intent(
        EXACT_PROMPT,
        "conversation:question",
        {},
    )

    assert intent == "conversation:question"
    assert entities == {}


def test_native_conceptual_mode_returns_direct_foundation_answer():
    alice = ALICE.__new__(ALICE)

    response = alice._native_conceptual_answer(
        EXACT_PROMPT,
        "conversation:question",
    )

    assert response is not None
    low = response.lower()
    assert "assistant" in low
    assert "memory" in low
    assert "planning" in low
    assert "execution" in low
    assert "autonomy" in low
    assert "person 'an ai'" not in low
    assert "polymorphism" not in low


def test_native_scaffold_handles_simple_conversation_openers_without_llm():
    alice = ALICE.__new__(ALICE)

    assert alice._native_scaffold_response("how are you?", "status_inquiry") is not None
    assert alice._native_scaffold_response("how are you?", "conversation:general") is not None
    assert alice._native_scaffold_response("hello", "conversation:general") is not None
    assert alice._native_scaffold_response("can you help me?", "conversation:help") is not None
    assert alice._native_scaffold_response("thanks", "conversation:general") is not None


def test_native_scaffold_handles_beginner_explanation_help_request():
    alice = ALICE.__new__(ALICE)

    response = alice._native_scaffold_response(
        "i am beginner so i want an explanation",
        "conversation:help",
    )

    assert response is not None
    low = response.lower()
    assert "beginner level" in low
    assert "step by step" in low


def test_native_scaffold_does_not_flatten_detailed_help_issue_report():
    alice = ALICE.__new__(ALICE)
    detailed = "my ai is not able to correctly give me some informations or it gets the intent wrong"

    assert alice._native_scaffold_response(detailed, "conversation:help") is None


def test_goal_statement_promotion_enriches_entities_and_intent():
    alice = ALICE.__new__(ALICE)
    text = "I want to build an agent and not just a chatbot"

    intent, entities, confidence = alice._promote_goal_statement_intent(
        user_input=text,
        intent="conversation:general",
        entities={},
        intent_confidence=0.41,
    )

    assert intent == "conversation:goal_statement"
    assert confidence >= 0.84
    assert entities.get("goal")
    assert entities.get("user_goal")
    assert entities.get("project_direction")


def test_native_scaffold_goal_statement_returns_alignment_response():
    alice = ALICE.__new__(ALICE)
    text = "I want to make Alice think in steps and become more autonomous"

    response = alice._native_scaffold_response(text, "conversation:goal_statement")

    assert response is not None
    low = response.lower()
    assert "agent behavior" in low
    assert "planning" in low
    assert "persistent memory" in low
    assert "tool execution" in low
