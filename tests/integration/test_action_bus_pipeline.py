from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_operator_continue_produces_action_result_metadata():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="can you check your local code?",
        user_id="u1",
        turn_number=1,
    )
    action_result = dict(result.metadata.get("action_result") or {})
    assert action_result
    assert action_result.get("name") in {
        "inspect_file",
        "analyze_file",
        "project_status",
        "next_step",
        "list_files",
        "plan",
        "code:request",
    }
    assert isinstance(action_result.get("verified"), bool)


def test_verified_inspection_claim_survives_with_matching_action_evidence():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="read app/main.py",
        user_id="u1",
        turn_number=2,
    )
    if "i inspected " in result.response_text.lower():
        action_result = dict(result.metadata.get("action_result") or {})
        evidence = dict(action_result.get("evidence") or {})
        assert evidence.get("inspected_file")
        assert evidence.get("inspected_file") in result.response_text


def test_failed_action_does_not_produce_fake_success_claim():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=3,
    )
    action_result = dict(result.metadata.get("action_result") or {})
    if action_result and (action_result.get("success") is False):
        assert "i inspected " not in result.response_text.lower()
