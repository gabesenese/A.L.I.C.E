from ai.memory.project_memory import ProjectMemoryState, save_project_state
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_why_that_file_explains_stored_recommendation():
    alice = _FakeAlice()
    save_project_state(
        ProjectMemoryState(
            active_objective="Improve Alice into an agentic companion/operator",
            last_recommended_action={
                "action": "inspect_file",
                "target": "ai/runtime/agent_loop.py",
                "reason": "Active objective exists; agent loop should drive next safe step.",
                "safety_level": "safe_read",
                "requires_approval": False,
                "source": "next_step_policy",
            },
            next_recommended_action="Next best move: inspect file ai/runtime/agent_loop.py because Active objective exists; agent loop should drive next safe step.",
        ),
        user_id="default",
    )
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="why did you mention to take a look into agent_loop.py?",
        user_id="default",
        turn_number=3,
    )
    low = result.response_text.lower()
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] == "operator:explain_recommendation"
    assert "agent_loop.py" in low
    assert "active objective exists" in low
    assert "turn_orchestrator.py" not in low
