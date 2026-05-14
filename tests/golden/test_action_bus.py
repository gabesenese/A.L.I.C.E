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

    def _executor(req: ActionRequest) -> ActionResult:
        return ActionResult(
            action_id=req.action_id,
            name=req.name,
            success=True,
            evidence={"inspected_file": "ai/runtime/agent_loop.py"},
        )

    bus.register("inspect_file", _executor)
    out = bus.execute(
        ActionRequest(
            action_id="a2",
            name="inspect_file",
            risk_level="safe_read",
            approved=False,
        )
    )
    assert out.success is True
    assert out.verified is True


def test_registered_destructive_action_blocked_without_approval():
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
            risk_level="destructive",
            requires_approval=True,
            approved=False,
        )
    )
    assert called["value"] is False
    assert out.success is False
    assert out.error == "approval_required"
    assert out.verified is False
    assert out.evidence.get("approval_required") is True
    assert out.evidence.get("approved") is False


def test_registered_destructive_action_executes_with_approval():
    bus = ActionBus()
    called = {"value": False}

    def _delete(req: ActionRequest) -> ActionResult:
        called["value"] = True
        return ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"deleted": True})

    bus.register("delete_file", _delete)
    out = bus.execute(
        ActionRequest(
            action_id="a4",
            name="delete_file",
            risk_level="destructive",
            requires_approval=True,
            approved=True,
            approval_id="appr_123",
        )
    )
    assert called["value"] is True
    assert out.success is True
    assert out.evidence.get("approved") is True
    assert out.verified is True


def test_registered_external_action_blocked_without_approval():
    bus = ActionBus()
    bus.register(
        "external_state_change",
        lambda req: ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"ok": True}),
    )
    out = bus.execute(
        ActionRequest(
            action_id="a5",
            name="external_state_change",
            risk_level="external",
            approved=False,
        )
    )
    assert out.success is False
    assert out.error == "approval_required"


def test_registered_safe_write_action_blocked_without_approval():
    bus = ActionBus()
    bus.register(
        "edit_file",
        lambda req: ActionResult(action_id=req.action_id, name=req.name, success=True, evidence={"ok": True}),
    )
    out = bus.execute(
        ActionRequest(
            action_id="a6",
            name="edit_file",
            risk_level="safe_write",
            approved=False,
        )
    )
    assert out.success is False
    assert out.error == "approval_required"


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


def test_claim_verifier_blocks_mutation_claim_from_approval_required_result():
    out = verify_response_claims(
        "I deleted config.json.",
        action_result={
            "success": False,
            "error": "approval_required",
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
