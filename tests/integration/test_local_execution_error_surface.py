from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from ai.runtime.response_momentum_policy import apply_response_momentum
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


def test_missing_file_target_returns_human_blocker():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="analyze legacy-main.py",
        user_id="u1",
        turn_number=4,
    )
    local_execution = dict(result.metadata.get("local_execution") or {})
    if local_execution.get("success") is False:
        if str(local_execution.get("error") or "") == "target_not_found":
            assert str(local_execution.get("requested_target") or "").strip()
        text = str(result.response_text or "")
        low = text.lower()
        assert "i couldn't verify the local step." in low
        assert "blocker:" in low
        assert "target_not_found" not in low
        assert "next best move:" in low
        assert "\n\nblocker:" in low
        assert "\n\nnext best move:" in low


def test_paragraph_breaks_survive_momentum_policy_for_local_failure():
    rendered = apply_response_momentum(
        user_input="analyze legacy-main.py",
        response_text="I could not find legacy-main.py in the current workspace.",
        intent="code:analyze_file",
        route="local",
        operator_state={},
        project_memory={},
        local_execution={
            "success": False,
            "error": "target_not_found",
            "requested_target": "legacy-main.py",
        },
        next_step="inspect ai/runtime/agent_loop.py because active objective exists; agent loop should drive next safe step",
        llm_generate=None,
        perception_frame={},
        companion_state={},
        response_generation_metadata={},
    )
    low = rendered.lower()
    assert "i couldn't verify the local step.\n\nblocker:" in low
    assert ".\n\nnext best move:" in low
