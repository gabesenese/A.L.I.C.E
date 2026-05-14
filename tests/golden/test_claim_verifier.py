from ai.runtime.claim_verifier import verify_response_claims


def test_allows_verified_inspection():
    out = verify_response_claims(
        "I inspected ai/runtime/agent_loop.py.",
        local_execution={"success": True, "inspected_file": "ai/runtime/agent_loop.py"},
    )
    assert out.valid is True
    assert out.verified_text == "I inspected ai/runtime/agent_loop.py."


def test_blocks_unverified_inspection():
    out = verify_response_claims("I inspected ai/runtime/agent_loop.py.", local_execution=None)
    assert out.valid is False
    assert "action_claim_without_evidence" in out.reasons
    assert "i inspected" not in out.verified_text.lower()


def test_blocks_fake_deletion():
    out = verify_response_claims("I deleted the memories about your mom.", deletion_result=None)
    assert out.valid is False
    assert "mutation_claim_without_evidence" in out.reasons
    assert "can't confirm deletion" in out.verified_text.lower()


def test_allows_verified_deletion():
    out = verify_response_claims(
        "I deleted 3 memories related to that topic.",
        deletion_result={"success": True, "deleted_count": 3, "verification_status": "cleared"},
    )
    assert out.valid is True


def test_blocks_fake_memory_claim():
    out = verify_response_claims("I remember we discussed machine learning last time.", memory_result=None)
    assert out.valid is False
    assert "memory_claim_without_evidence" in out.reasons


def test_allows_grounded_project_state_claim():
    out = verify_response_claims(
        "Current objective is Improve Alice into an agentic companion/operator.",
        project_memory={"active_objective": "Improve Alice into an agentic companion/operator"},
    )
    assert out.valid is True


def test_blocks_fake_background_monitoring():
    out = verify_response_claims(
        "I've been monitoring your project while you were away.",
        background_events=[],
    )
    assert out.valid is False
    assert "background_claim_without_evidence" in out.reasons


def test_allows_harmless_future_continuation():
    out = verify_response_claims("We can continue tomorrow.")
    assert out.valid is True


def test_rewrites_unsupported_readiness_claim():
    out = verify_response_claims("I'll be ready tomorrow.", background_events=[])
    assert out.valid is False
    assert out.verified_text == "We can continue tomorrow."
