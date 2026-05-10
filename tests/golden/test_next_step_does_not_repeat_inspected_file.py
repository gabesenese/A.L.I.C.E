from ai.runtime.next_step_policy import decide_next_step


def test_next_step_does_not_repeat_agent_loop_after_inspection():
    decision = decide_next_step(
        route="local",
        intent="operator:continue",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "files_inspected": ["ai/runtime/agent_loop.py"],
        },
        local_execution={},
        available_files=[],
        files_inspected=["ai/runtime/agent_loop.py"],
    )
    rec = dict(decision.last_recommended_action or {})
    assert rec.get("target") != "ai/runtime/agent_loop.py"
    assert rec.get("target") == "ai/runtime/next_step_policy.py"


def test_active_objective_still_recommends_agent_loop_initially():
    decision = decide_next_step(
        route="local",
        intent="operator:continue",
        operator_state={"active_objective": "Improve Alice into an agentic companion/operator"},
        local_execution={},
        available_files=[],
        files_inspected=[],
    )
    rec = dict(decision.last_recommended_action or {})
    assert rec.get("target") == "ai/runtime/agent_loop.py"


def test_all_primary_files_inspected_returns_summarize_findings():
    inspected = [
        "ai/runtime/agent_loop.py",
        "ai/runtime/next_step_policy.py",
        "ai/runtime/operator_state.py",
        "ai/runtime/response_momentum_policy.py",
        "ai/runtime/contract_pipeline.py",
        "ai/memory/project_memory.py",
    ]
    decision = decide_next_step(
        route="local",
        intent="operator:continue",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "files_inspected": inspected,
        },
        local_execution={},
        available_files=[],
        files_inspected=inspected,
    )
    rec = dict(decision.last_recommended_action or {})
    assert rec.get("action") == "summarize_findings"
    assert rec.get("target") == "inspected_files"
