from ai.runtime.claim_verifier import verify_response_claims


def test_reject_delete_claim_without_evidence():
    out = verify_response_claims("I deleted the memories about your mom.")
    assert out.valid is False
    assert "delete_without_evidence" in out.unsupported_claims


def test_reject_inspected_claim_without_local_execution():
    out = verify_response_claims("I inspected agent_loop.py.")
    assert out.valid is False
    assert "action_without_evidence" in out.unsupported_claims


def test_reject_fictional_provenance_claim():
    out = verify_response_claims("That system was built with an exact named framework and creator has said so.")
    assert out.valid is False
    assert "fictional_provenance_claim" in out.unsupported_claims
