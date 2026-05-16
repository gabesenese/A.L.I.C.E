from __future__ import annotations

from ai.memory.alice_memory_service import AliceMemoryService
from ai.runtime.context_refresh_service import ContextRefreshService


def _memory_service(tmp_path):
    svc = AliceMemoryService(db_path=str(tmp_path / "context_memory.db"))
    svc.initialize()
    return svc


def test_greeting_has_minimal_context(tmp_path):
    svc = _memory_service(tmp_path)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="hi alice",
        route="llm",
        intent="greeting",
        operator_state={},
        project_state={"active_objective": "Improve Alice"},
        memory_service=svc,
    )
    assert frame.mode == "greeting"
    assert frame.verified_memories == []
    assert frame.hint_memories == []


def test_educational_external_does_not_inject_project_state(tmp_path):
    svc = _memory_service(tmp_path)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="what is machine learning",
        route="llm",
        intent="conversation:educational_explain",
        operator_state={},
        project_state={"active_objective": "Improve Alice"},
        memory_service=svc,
    )
    assert frame.mode == "educational_explain"
    assert frame.subject == "external"
    assert frame.project_state == {}


def test_concept_refinement_injects_active_concept_thread(tmp_path):
    svc = _memory_service(tmp_path)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="i want alice to be proactive and not chatbot",
        route="llm",
        intent="conversation:concept_refinement",
        operator_state={},
        project_state={},
        memory_service=svc,
    )
    assert frame.mode == "concept_refinement"
    assert frame.active_concept_thread is not None
    assert "proactive" in [c.lower() for c in frame.active_concept_thread.constraints]


def test_implementation_request_injects_project_state_and_requires_evidence(tmp_path):
    svc = _memory_service(tmp_path)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="how do we implement this in Alice",
        route="llm",
        intent="conversation:goal_statement",
        operator_state={},
        project_state={"active_objective": "Improve Alice"},
        memory_service=svc,
    )
    assert frame.mode in {"operator_work", "codebase_work"}
    assert frame.evidence_required is True
    assert frame.project_state.get("active_objective") == "Improve Alice"


def test_codebase_request_requires_evidence(tmp_path):
    svc = _memory_service(tmp_path)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="check the repo",
        route="llm",
        intent="conversation:general",
        operator_state={},
        project_state={"active_objective": "Improve Alice"},
        memory_service=svc,
    )
    assert frame.mode == "codebase_work"
    assert frame.evidence_required is True
    assert frame.tool_required is True


def test_hint_memories_marked_as_hints(tmp_path):
    svc = _memory_service(tmp_path)
    svc.save_fact("low confidence companion idea", topic="companion", confidence=0.3)
    refresh = ContextRefreshService()
    frame = refresh.build_context_frame(
        user_input="tell me about companion ideas",
        route="llm",
        intent="conversation:educational_explain",
        operator_state={},
        project_state={},
        memory_service=svc,
    )
    assert any(item.confidence_label == "hint" for item in frame.hint_memories)
