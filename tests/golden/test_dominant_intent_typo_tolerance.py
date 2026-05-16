from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_work_o_alice_typo_does_not_route_to_clarification():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="been great, it is friday after all and i am ready to work o alice",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["intent"] != "conversation:clarification_needed"
