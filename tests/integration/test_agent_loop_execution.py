from ai.runtime.agent_loop import AgentLoop


def test_agent_loop_continue_uses_last_recommended_target_priority():
    loop = AgentLoop()
    result = loop.run(
        user_input="continue",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/agent_loop.py",
                "reason": "Active objective exists; agent loop should drive next safe step.",
                "requires_approval": False,
            },
            "suggested_next_files": ["ai/runtime/turn_orchestrator.py"],
        },
        project_memory={},
        routing_result={
            "route": "local",
            "intent": "operator:continue",
            "local_execution": {},
        },
        available_files=[
            "ai/runtime/turn_orchestrator.py",
            "ai/runtime/contract_pipeline.py",
        ],
        max_steps=1,
        user_id="default",
    )
    plan = list(result.plan_steps or [])
    assert plan
    assert plan[0]["action"] == "analyze_file"
    assert plan[0]["target"] == "ai/runtime/agent_loop.py"


def test_inspected_file_is_recorded_before_next_step_decision():
    loop = AgentLoop()
    result = loop.run(
        user_input="continue",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "files_inspected": [],
            "last_inspected_file": "",
        },
        project_memory={},
        routing_result={
            "route": "local",
            "intent": "operator:continue",
            "local_execution": {
                "action": "code:analyze_file",
                "success": True,
                "inspected_file": "ai/runtime/agent_loop.py",
            },
        },
        available_files=["ai/runtime/agent_loop.py"],
        max_steps=1,
        user_id="default",
    )
    # After inspecting agent_loop.py, next recommendation should move forward.
    assert "ai/runtime/agent_loop.py" in [o.get("inspected_file") for o in result.observations]
    assert "ai/runtime/agent_loop.py" not in str(result.next_step or "").lower()
