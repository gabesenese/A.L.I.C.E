from __future__ import annotations

from ai.memory.alice_memory_schema import ActiveConceptThread
from ai.runtime.alice_turn_router import route_turn


def test_hi_alice_routes_to_greeting():
    out = route_turn("hi alice")
    assert out.mode == "greeting"
    assert out.memory_required is False


def test_learn_ai_companion_routes_to_educational_explain():
    out = route_turn("i want to learn about ai companion")
    assert out.mode == "educational_explain"


def test_not_assistant_with_active_concept_routes_concept_refinement():
    thread = ActiveConceptThread(
        topic="proactive AI companion",
        constraints=["not chatbot"],
        signals=["ai companion"],
        last_user_inputs=["previous"],
        updated_at="2026-05-16T00:00:00+00:00",
    )
    out = route_turn(
        "i dont want it to be like an assistant or chatbot",
        active_concept_thread=thread,
    )
    assert out.mode == "concept_refinement"


def test_implement_in_alice_routes_operator_or_codebase_work():
    out = route_turn("how do we implement this in Alice")
    assert out.mode in {"operator_work", "codebase_work"}
    assert out.evidence_required is True


def test_check_repo_requires_codebase_work_and_evidence():
    out = route_turn("check the repo")
    assert out.mode == "codebase_work"
    assert out.evidence_required is True
    assert out.tool_required is True


def test_machine_learning_routes_external_educational_explain():
    out = route_turn("what is machine learning")
    assert out.mode == "educational_explain"
    assert out.subject == "external"
