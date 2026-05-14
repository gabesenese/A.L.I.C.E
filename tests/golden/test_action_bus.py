from ai.runtime.action_bus import (
    ActionBus,
    ActionRequest,
    ActionResult,
    action_result_from_local_execution,
)
from ai.runtime.claim_verifier import verify_response_claims


def test_unknown_action_still_unknown_action():
    bus = ActionBus()
    out = bus.execute(ActionRequest(action_id="a1", name="does_not_exist", risk_level="safe_read"))
    assert out.success is False
    assert out.error == "unknown_action"
    assert out.verified is False


def test_safe_read_action_executes_without_approval():
    bus = ActionBus()
    bus.register(
        "inspect_file",
        lambda req: ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"inspected_file": "x"}),
    )
    out = bus.execute(
        ActionRequest(action_id="a2", name="inspect_file", risk_level="safe_read", approved=False)
    )
    assert out.success is True
    assert out.verified is True


def test_destructive_action_blocked_without_approval_id():
    bus = ActionBus()
    called = {"value": False}

    def _delete(req: ActionRequest) -> ActionResult:
        called["value"] = True
        return ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"deleted": True})

    bus.register("delete_file", _delete)
    out = bus.execute(
        ActionRequest(
            action_id="a3",
            name="delete_file",
            target="tmp/test.txt",
            risk_level="destructive",
            requires_approval=True,
            approved=True,
            approval_id="",
        )
    )
    assert called["value"] is False
    assert out.success is False
    assert out.error == "approval_required"


def test_destructive_action_blocked_with_invalid_approval_id():
    bus = ActionBus(approval_lookup=lambda _aid: None, approval_consume=lambda _aid: False)
    bus.register(
        "delete_file",
        lambda req: ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"deleted": True}),
    )
    out = bus.execute(
        ActionRequest(
            action_id="a4",
            name="delete_file",
            target="tmp/test.txt",
            risk_level="destructive",
            requires_approval=True,
            approved=True,
            approval_id="appr_missing",
        )
    )
    assert out.success is False
    assert out.error == "approval_not_found"


def test_destructive_action_blocked_with_mismatched_approval():
    approval = {
        "approval_id": "appr_1",
        "action_name": "delete_file",
        "target": "tmp/other.txt",
        "risk_level": "destructive",
        "approved": True,
        "consumed": False,
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "",
        "reason": "",
    }
    bus = ActionBus(approval_lookup=lambda _aid: approval, approval_consume=lambda _aid: False)
    bus.register(
        "delete_file",
        lambda req: ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"deleted": True}),
    )
    out = bus.execute(
        ActionRequest(
            action_id="a5",
            name="delete_file",
            target="tmp/test.txt",
            risk_level="destructive",
            requires_approval=True,
            approved=True,
            approval_id="appr_1",
        )
    )
    assert out.success is False
    assert out.error == "approval_mismatch"


def test_destructive_action_executes_with_matching_approval():
    called = {"value": False}
    consumed = {"value": False}
    approval = {
        "approval_id": "appr_2",
        "action_name": "delete_file",
        "target": "tmp/test.txt",
        "risk_level": "destructive",
        "approved": True,
        "consumed": False,
        "created_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "",
        "reason": "",
    }

    def _delete(req: ActionRequest) -> ActionResult:
        called["value"] = True
        return ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"deleted": True})

    def _consume(_aid: str) -> bool:
        consumed["value"] = True
        return True

    bus = ActionBus(approval_lookup=lambda _aid: approval, approval_consume=_consume)
    bus.register("delete_file", _delete)
    out = bus.execute(
        ActionRequest(
            action_id="a6",
            name="delete_file",
            target="tmp/test.txt",
            risk_level="destructive",
            requires_approval=True,
            approved=True,
            approval_id="appr_2",
        )
    )
    assert called["value"] is True
    assert consumed["value"] is True
    assert out.success is True
    assert out.evidence.get("approval_consumed") is True
    assert out.verified is True


def test_convert_local_execution_into_action_result():
    out = action_result_from_local_execution(
        action_name="inspect_file",
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
    )
    assert out.success is True
    assert out.evidence.get("inspected_file") == "ai/runtime/agent_loop.py"
    assert out.verified is True


def test_claim_verifier_accepts_action_result_evidence():
    out = verify_response_claims(
        "I inspected ai/runtime/agent_loop.py.",
        action_result={
            "success": True,
            "verified": True,
            "evidence": {"inspected_file": "ai/runtime/agent_loop.py"},
        },
    )
    assert out.valid is True


def test_claim_verifier_rejects_mismatched_action_result_target():
    out = verify_response_claims(
        "I inspected ai/runtime/operator_state.py.",
        action_result={
            "success": True,
            "verified": True,
            "evidence": {"inspected_file": "ai/runtime/agent_loop.py"},
        },
    )
    assert out.valid is False


def test_claim_verifier_blocks_mutation_claim_from_approval_blocked_result():
    out = verify_response_claims(
        "I deleted config.json.",
        action_result={
            "success": False,
            "error": "approval_mismatch",
            "risk_level": "destructive",
            "requires_approval": True,
            "verified": False,
            "evidence": {
                "approval_required": True,
                "approved": False,
            },
        },
    )
    assert out.valid is False
    assert "mutation_claim_without_evidence" in out.reasons
