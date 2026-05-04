from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_file_capability_routes_local_and_not_file_plugin():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="are you able to read me a file alice has?",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] in {"code:request", "code:list_files"}
    assert result.metadata["intent"] != "file_operations:read"
    assert result.metadata["verification"]["reason"] != "tool_failed"


def test_project_objective_and_next_step():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="let's focus on making Alice more agentic",
        user_id="u1",
        turn_number=2,
    )
    state = dict(getattr(alice, "_operator_state", {}) or {})
    assert state.get("active_mode") == "alice_project_operator"
    assert "agentic companion/operator" in str(state.get("active_objective") or "")
    next_step = dict(result.metadata.get("next_step_policy") or {})
    assert isinstance(next_step.get("next_recommended_action"), str)
