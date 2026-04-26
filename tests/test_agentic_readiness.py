from ai.core.agentic_readiness import build_agentic_focus_plan


def test_agentic_focus_plan_prioritizes_routing_and_recovery():
    plan = build_agentic_focus_plan(
        {
            "wrong_tool_rate": 0.11,
            "recovery_success_rate": 0.42,
            "stale_state_rate": 0.09,
            "long_horizon_completion_rate": 0.5,
            "personalization_satisfaction": 0.62,
            "clarification_precision": 0.7,
        }
    )

    assert plan
    assert plan[0].area == "routing_reliability"
    assert any(item.area == "recovery_graph_depth" for item in plan)


def test_agentic_focus_plan_returns_empty_when_metrics_meet_targets():
    plan = build_agentic_focus_plan(
        {
            "wrong_tool_rate": 0.01,
            "recovery_success_rate": 0.8,
            "stale_state_rate": 0.01,
            "long_horizon_completion_rate": 0.8,
            "personalization_satisfaction": 0.9,
            "clarification_precision": 0.9,
        }
    )
    assert plan == []
