from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_operator_state_sets_objective_and_focus():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(
        user_input="keep moving Alice toward agentic behavior and fix Alice's routing",
        user_id="u1",
        turn_number=1,
    )
    state = dict(getattr(alice, "_operator_state", {}) or {})
    assert state.get("active_mode") == "alice_project_operator"
    assert "agentic companion/operator" in str(state.get("active_objective") or "")
    assert str(state.get("current_focus") or "")
    assert result.metadata["decision_band"] != "clarify"


def test_next_step_query_uses_operator_state_guidance():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="make Alice more agentic",
        user_id="u1",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="what's the next step?",
        user_id="u1",
        turn_number=2,
    )
    text = str(result.response_text or "").lower()
    assert "next best move" in text or "inspect" in text

