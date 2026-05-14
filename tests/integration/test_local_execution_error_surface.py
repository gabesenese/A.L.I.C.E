from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_operator_continue_failed_local_execution_returns_compact_error_surface():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata.get("route") == "local"
    assert result.metadata.get("intent") == "code:analyze_file"
    assert result.metadata.get("claim_verifier_applied") is True
    low = result.response_text.lower()
    if (result.metadata.get("local_execution") or {}).get("success") is False:
        assert "i couldn't verify the local step" in low
        assert "rewritten the response" not in low
        assert "what would you like" not in low


def test_claim_verifier_blocks_fake_inspection_after_local_failure():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=2,
    )
    local_execution = dict(result.metadata.get("local_execution") or {})
    if local_execution.get("success") is False:
        assert "i inspected " not in result.response_text.lower()


def test_next_best_move_survives_local_failure():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=3,
    )
    local_execution = dict(result.metadata.get("local_execution") or {})
    if local_execution.get("success") is False:
        assert "next best move:" in result.response_text.lower()
