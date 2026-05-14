from ai.runtime.action_bus import (
    ActionBus,
    ActionRequest,
    ActionResult,
    action_result_from_local_execution,
)
from ai.runtime.claim_verifier import verify_response_claims


def test_unknown_action_fails_safely():
    bus = ActionBus()
    out = bus.execute(ActionRequest(action_id="a1", name="does_not_exist"))
    assert out.success is False
    assert out.error == "unknown_action"
    assert out.verified is False


def test_safe_read_action_executes():
    bus = ActionBus()

    def _executor(req: ActionRequest) -> ActionResult:
        return ActionResult(
            action_id=req.action_id,
            name=req.name,
            success=True,
            evidence={"inspected_file": "ai/runtime/agent_loop.py", "source": "test"},
            verified=True,
        )

    bus.register("inspect_file", _executor)
    out = bus.execute(ActionRequest(action_id="a2", name="inspect_file"))
    assert out.success is True
    assert out.evidence.get("inspected_file") == "ai/runtime/agent_loop.py"
    assert out.verified is True


def test_destructive_action_requires_approval():
    bus = ActionBus()
    req = ActionRequest(
        action_id="a3",
        name="delete_file",
        risk_level="destructive",
        requires_approval=True,
    )
    assert bus.can_execute(req) is False
    out = bus.execute(req)
    assert out.success is False
    assert out.error in {"unknown_action", "approval_required"}


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
