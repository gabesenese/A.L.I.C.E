from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_capability_question_does_not_hit_file_plugin():
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


def test_files_question_lists_files_without_clarify():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="what files can you inspect for me?",
        user_id="u1",
        turn_number=2,
    )
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] == "code:list_files"
    assert result.metadata["decision_band"] != "clarify"


def test_explicit_analysis_routes_local_and_missing_has_close_matches():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=3,
    )
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] == "code:analyze_file"
    assert result.metadata["decision_band"] != "clarify"
    local_execution = dict(result.metadata.get("local_execution") or {})
    assert local_execution.get("action") == "code:analyze_file"
    if local_execution.get("success") is False:
        op = dict(result.metadata.get("operator_context") or {})
        assert isinstance(op.get("close_matches"), list)


def test_project_mode_sets_operator_state():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="alright lets focus on my ai project",
        user_id="u1",
        turn_number=4,
    )
    state = dict(getattr(alice, "_operator_state", {}) or {})
    assert state.get("active_mode") == "alice_project_operator"
    assert result.metadata["decision_band"] != "clarify"


def test_read_without_target_does_not_execute_file_plugin():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="read a file",
        user_id="u1",
        turn_number=5,
    )
    assert result.metadata["intent"] != "file_operations:read"
    assert result.metadata["route"] == "local"
    assert "share the exact outcome" not in result.response_text.lower()


def test_explicit_file_read_kept_local():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="read app/main.py",
        user_id="u1",
        turn_number=6,
    )
    assert result.metadata["route"] == "local"
    assert result.metadata["intent"] in {"code:read_file", "code:analyze_file"}
