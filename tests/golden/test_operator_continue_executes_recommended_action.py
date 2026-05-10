from ai.memory.project_memory import ProjectMemoryState, load_project_state, save_project_state
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_store_structured_next_recommendation():
    alice = _FakeAlice()
    save_project_state(ProjectMemoryState(active_objective="Improve Alice into an agentic companion/operator"), user_id="default")
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="good, let's work on alice",
        user_id="default",
        turn_number=1,
    )
    assert result.metadata["intent"] in {"operator:continue", "conversation:goal_statement"}
    state = load_project_state("default")
    rec = dict(state.last_recommended_action or {})
    assert rec.get("action") == "inspect_file"
    assert rec.get("target") == "ai/runtime/agent_loop.py"
    assert rec.get("requires_approval") is False


def test_go_ahead_executes_stored_safe_recommendation():
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
        user_input="go ahead",
        user_id="default",
        turn_number=2,
    )
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] == "operator:execute_recommended_action"
    local = dict(result.metadata.get("local_execution") or {})
    assert str(local.get("inspected_file") or "").lower() == "ai/runtime/agent_loop.py"
