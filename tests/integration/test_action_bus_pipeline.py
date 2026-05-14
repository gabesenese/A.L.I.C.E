from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.action_bus import ActionBus, ActionRequest, ActionResult
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
    assert action_result.get("risk_level") == "safe_read"
    assert action_result.get("requires_approval") is False


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


def test_destructive_action_cannot_run_without_approval():
    bus = ActionBus()
    called = {"value": False}

    def _delete(req: ActionRequest) -> ActionResult:
        called["value"] = True
        return ActionResult(
            action_id=req.action_id,
            name=req.name,
            success=True,
            evidence={"deleted": True},
            risk_level="destructive",
            requires_approval=True,
        )

    bus.register("delete_file", _delete)
    out = bus.execute(
        ActionRequest(
            action_id="int_del_1",
            name="delete_file",
            risk_level="destructive",
            requires_approval=True,
            approved=False,
        )
    )
    assert called["value"] is False
    assert out.success is False
    assert out.error == "approval_required"
    assert out.evidence.get("approval_required") is True
