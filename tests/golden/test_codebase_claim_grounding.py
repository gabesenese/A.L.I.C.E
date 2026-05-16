from ai.runtime.claim_verifier import verify_response_claims


def test_blocks_repo_scan_language_without_evidence():
    out = verify_response_claims(
        "I was scanning through the repo and found several plugin modules we can improve.",
        user_input="you have access to alice's code base, my ai project",
        local_execution={},
        action_result={},
    )
    assert out.valid is False
    assert "codebase_claim_without_evidence" in out.reasons
    low = out.verified_text.lower()
    assert "scanning through the repo" not in low
    assert "i have not verified the codebase yet" in low


def test_blocks_py_file_recommendation_without_evidence():
    out = verify_response_claims(
        "Take a look at self_learning/contextual_awareness.py for the next improvement area.",
        user_input="give me an area i can improve",
        local_execution={},
        action_result={},
    )
    assert out.valid is False
    assert "codebase_claim_without_evidence" in out.reasons
    low = out.verified_text.lower()
    assert "self_learning/contextual_awareness.py" not in low
    assert "i have not verified the codebase yet" in low


def test_allows_py_file_recommendation_with_listed_files_evidence():
    out = verify_response_claims(
        "Take a look at ai/runtime/response_momentum_policy.py.",
        user_input="give me an area i can improve",
        local_execution={},
        action_result={
            "success": True,
            "verified": True,
            "evidence": {
                "listed_files": ["ai/runtime/response_momentum_policy.py"],
            },
        },
    )
    assert out.valid is True
    assert out.verified_text == "Take a look at ai/runtime/response_momentum_policy.py."


def test_allows_inspected_file_claim_with_matching_local_execution():
    out = verify_response_claims(
        "I inspected ai/runtime/agent_loop.py.",
        user_input="inspect the project",
        local_execution={"success": True, "inspected_file": "ai/runtime/agent_loop.py"},
        action_result={},
    )
    assert out.valid is True
    assert out.verified_text == "I inspected ai/runtime/agent_loop.py."


def test_rewrites_unsupported_codebase_claim_to_unverified_message():
    out = verify_response_claims(
        "We have a few plugins and runtime features related to background monitoring.",
        user_input="you have access to alice's code base",
        local_execution={},
        action_result={},
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/agent_loop.py",
            }
        },
    )
    assert out.valid is False
    assert "codebase_claim_without_evidence" in out.reasons
    assert (
        out.verified_text
        == "I have not verified the codebase yet, so I should inspect the project before naming files or specific improvement areas.\n\nNext best move: inspect ai/runtime/agent_loop.py."
    )
