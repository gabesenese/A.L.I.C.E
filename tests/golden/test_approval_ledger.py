from pathlib import Path

from ai.runtime import approval_ledger as ledger
from ai.runtime.action_bus import ActionRequest


def _set_ledger_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ledger, "APPROVAL_LEDGER_PATH", tmp_path / "approval_ledger.jsonl")


def test_create_and_load_approval(monkeypatch, tmp_path):
    _set_ledger_path(monkeypatch, tmp_path)
    created = ledger.create_approval(
        action_name="delete_file",
        target="tmp/test.txt",
        risk_level="destructive",
    )
    loaded = ledger.load_approval(created.approval_id)
    assert loaded is not None
    assert loaded.action_name == "delete_file"
    assert loaded.target == "tmp/test.txt"
    assert loaded.risk_level == "destructive"


def test_approval_matches_request(monkeypatch, tmp_path):
    _set_ledger_path(monkeypatch, tmp_path)
    created = ledger.create_approval("delete_file", "tmp/test.txt", "destructive")
    req = ActionRequest(
        action_id="a1",
        name="delete_file",
        target="tmp/test.txt",
        risk_level="destructive",
    )
    assert ledger.approval_matches(created, req) is True


def test_approval_rejects_mismatched_target(monkeypatch, tmp_path):
    _set_ledger_path(monkeypatch, tmp_path)
    created = ledger.create_approval("delete_file", "tmp/test.txt", "destructive")
    req = ActionRequest(
        action_id="a2",
        name="delete_file",
        target="tmp/other.txt",
        risk_level="destructive",
    )
    assert ledger.approval_matches(created, req) is False


def test_consume_approval(monkeypatch, tmp_path):
    _set_ledger_path(monkeypatch, tmp_path)
    created = ledger.create_approval("delete_file", "tmp/test.txt", "destructive")
    assert ledger.consume_approval(created.approval_id) is True
    loaded = ledger.load_approval(created.approval_id)
    assert loaded is not None
    assert loaded.consumed is True


def test_consumed_approval_invalid(monkeypatch, tmp_path):
    _set_ledger_path(monkeypatch, tmp_path)
    created = ledger.create_approval("delete_file", "tmp/test.txt", "destructive")
    assert ledger.consume_approval(created.approval_id) is True
    loaded = ledger.load_approval(created.approval_id)
    req = ActionRequest(
        action_id="a3",
        name="delete_file",
        target="tmp/test.txt",
        risk_level="destructive",
    )
    assert ledger.approval_matches(loaded, req) is False
