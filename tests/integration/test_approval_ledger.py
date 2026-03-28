from ai.infrastructure.approval_ledger import ApprovalLedger


def test_approval_ledger_confirm_flow(tmp_path):
    ledger = ApprovalLedger(storage_path=str(tmp_path / "approvals.jsonl"), ttl_seconds=120)
    req = ledger.create_request(action="controlled_commit", scope="write", summary="Commit repo")

    pending = ledger.get_pending(req.approval_id)
    assert pending is not None
    assert pending.action == "controlled_commit"

    rec = ledger.confirm(approval_id=req.approval_id, confirmation_text="operator approve token", actor="user")
    assert rec is not None
    assert rec.approved is True

    after = ledger.get_pending(req.approval_id)
    assert after is None


def test_approval_ledger_reject_flow(tmp_path):
    ledger = ApprovalLedger(storage_path=str(tmp_path / "approvals.jsonl"), ttl_seconds=120)
    req = ledger.create_request(action="controlled_commit", scope="write", summary="Commit repo")
    rec = ledger.reject(approval_id=req.approval_id, confirmation_text="operator reject token", actor="user")
    assert rec is not None
    assert rec.approved is False


def test_approval_ledger_scope_memory(tmp_path):
    ledger = ApprovalLedger(storage_path=str(tmp_path / "approvals.jsonl"), ttl_seconds=60)
    ledger.note_scope_approval("execute", ttl_seconds=60)
    assert ledger.is_scope_approved("execute") is True
