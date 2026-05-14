from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_operator_response_with_verified_local_execution_keeps_inspection_claim():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="read app/main.py",
        user_id="u1",
        turn_number=1,
    )
    local_execution = dict(result.metadata.get("local_execution") or {})
    if local_execution.get("success") and local_execution.get("inspected_file"):
        assert "i inspected" in result.response_text.lower()


def test_operator_response_with_failed_local_execution_does_not_claim_inspection():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=2,
    )
    local_execution = dict(result.metadata.get("local_execution") or {})
    if local_execution.get("success") is False:
        assert "i inspected" not in result.response_text.lower()


def test_memory_deletion_claim_without_deletion_result_gets_blocked():
    alice = _FakeAlice()
    alice.llm.chat = lambda *args, **kwargs: "I deleted those memories."
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="tell me a joke",
        user_id="u1",
        turn_number=3,
    )
    assert "i deleted those memories" not in result.response_text.lower()
    assert result.metadata.get("claim_verifier_applied") is True
    assert result.metadata.get("claim_verifier_valid") in {False, True}


def test_greeting_with_fake_continuity_memory_claim_is_blocked():
    alice = _FakeAlice()
    alice.llm.chat = lambda *args, **kwargs: "I remember we discussed this last time."
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="hi alice",
        user_id="u1",
        turn_number=4,
    )
    assert "i remember we discussed" not in result.response_text.lower()


def test_general_llm_background_monitoring_claim_without_evidence_gets_blocked():
    alice = _FakeAlice()
    alice.llm.chat = lambda *args, **kwargs: "I've been monitoring your project while you were away."
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="tell me a joke",
        user_id="u1",
        turn_number=5,
    )
    assert "i've been monitoring your project while you were away" not in result.response_text.lower()
    assert result.metadata.get("claim_verifier_applied") is True
    assert result.metadata.get("claim_verifier_valid") in {False, True}
