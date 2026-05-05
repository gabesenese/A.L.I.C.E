from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_agentic_objective_flow_sets_state_and_next_action():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(
        user_input="compare Alice to an agentic companion and keep moving her toward operator behavior",
        user_id="u1",
        turn_number=1,
    )
    state = dict(getattr(alice, "_operator_state", {}) or {})
    assert state.get("active_objective")
    next_step = dict(result.metadata.get("next_step_policy") or {})
    assert isinstance(next_step.get("next_recommended_action"), str)


def test_next_step_followup_is_grounded():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn("make Alice an operator", "u1", 1)
    second = pipeline.run_turn("what's the next step?", "u1", 2)
    assert second.metadata["route"] in {"local", "llm"}
    assert "ask me to" not in str(second.response_text or "").lower()

