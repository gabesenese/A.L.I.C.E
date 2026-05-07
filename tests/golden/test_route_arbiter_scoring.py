from __future__ import annotations

from ai.core.routing.route_arbiter import RouteArbiter, RouteCandidate


def test_route_arbiter_prefers_operator_next_step_with_active_objective():
    arbiter = RouteArbiter()
    result = arbiter.arbitrate_candidates(
        user_input="what's next?",
        active_mode="alice_project_operator",
        active_objective="Improve agentic companion operator runtime",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={"active_objective": "Improve agentic companion operator runtime"},
        continuation_context=True,
        candidates=[
            RouteCandidate(route="clarify", intent="clarification:context_resolution", confidence=0.6, source="test"),
            RouteCandidate(route="llm", intent="conversation:general", confidence=0.7, source="test"),
            RouteCandidate(route="local", intent="operator:next_step", confidence=0.65, source="test"),
        ],
    )
    assert result["intent"] == "operator:next_step"
    trace = dict(result.get("trace") or {})
    assert trace.get("active_objective_used") is True
