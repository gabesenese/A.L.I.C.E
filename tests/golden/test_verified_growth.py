from ai.runtime.verified_growth import verify_operator_surface_contract


def test_operator_response_with_service_chatter_fails_contract():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I'm here to assist you. Finding: bounded operator loop.",
        local_execution={"success": True},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is False
    assert "service_chatter_in_operator_surface" in set(check.failures)


def test_operator_response_with_finding_and_next_move_passes():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text=(
            "Finding: it owns the bounded operator loop.\n\n"
            "Next best move: inspect ai/runtime/next_step_policy.py."
        ),
        local_execution={"success": True},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is True


def test_inspection_claim_without_inspected_file_fails():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I inspected ai/runtime/agent_loop.py.",
        local_execution={"success": True},
        next_step="",
    )
    assert check.passed is False
    assert "inspection_claim_without_evidence" in set(check.failures)


def test_failed_local_execution_claiming_inspection_fails():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I inspected ai/runtime/agent_loop.py.",
        local_execution={"success": False, "error": "target_not_found"},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is False
