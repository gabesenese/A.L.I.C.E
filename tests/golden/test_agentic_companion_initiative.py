from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_goal_statement_flow_prefers_initiative_not_clarify():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="let's make Alice more agentic",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["decision_band"] != "clarify"
    assert "what exact result should i produce next" not in result.response_text.lower()


def test_operator_continue_selected_when_active_objective_exists():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="let's work on alice",
        user_id="u1",
        turn_number=1,
    )
    second = pipeline.run_turn(
        user_input="weather is weird today but i'm ready to work on alice",
        user_id="u1",
        turn_number=2,
    )
    assert second.metadata["intent"] in {
        "operator:continue",
        "conversation:goal_statement",
        "operator:next_step",
    }
