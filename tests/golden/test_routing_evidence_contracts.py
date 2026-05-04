from ai.core.routing.evidence_contracts import EvidenceContracts


def test_file_operations_read_veto_without_target():
    out = EvidenceContracts.evaluate(
        intent="file_operations:read",
        user_input="read something",
        active_mode="",
    )
    assert out["accepted"] is False
    assert out["reason"] == "no_explicit_file_target"
    assert out["file_tool_vetoed"] is True
    assert out["reroute_intent"] in {"code:request", "code:list_files"}


def test_file_operations_delete_veto_without_target():
    out = EvidenceContracts.evaluate(
        intent="file_operations:delete",
        user_input="delete a file",
        active_mode="",
    )
    assert out["accepted"] is False
    assert out["reason"] == "no_explicit_file_target"
    assert out["file_tool_vetoed"] is True

