from ai.runtime.verified_growth import verify_operator_surface_contract


def test_operator_response_with_service_chatter_fails_contract():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I'm here to assist you. Looked at the bounded operator loop.",
        local_execution={"success": True},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is False
    assert "service_chatter_in_operator_surface" in set(check.failures)


def test_natural_operator_response_passes_contract():
    # Internal labels ("Finding:", "Next best move:", "I inspected") must be absent.
    # Natural-language content passes.
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="Looked at the operator loop — that's where plan/act/verify runs.",
        local_execution={"success": True},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is True


def test_old_finding_format_fails_contract():
    # "Finding:" is an internal label that must not appear in user output.
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text=(
            "Finding: it owns the bounded operator loop.\n\nNext best move: inspect ai/runtime/next_step_policy.py."
        ),
        local_execution={"success": True},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is False
    assert "internal_finding_label_in_user_output" in set(check.failures)
    assert "internal_next_step_label_in_user_output" in set(check.failures)


def test_inspection_label_without_inspected_file_fails():
    # "I inspected" is an internal label — forbidden in user output regardless of evidence.
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I inspected ai/runtime/agent_loop.py.",
        local_execution={"success": True},
        next_step="",
    )
    assert check.passed is False
    assert "internal_inspection_label_in_user_output" in set(check.failures)


def test_failed_local_execution_claiming_inspection_fails():
    check = verify_operator_surface_contract(
        route="local",
        intent="operator:continue",
        response_text="I inspected ai/runtime/agent_loop.py.",
        local_execution={"success": False, "error": "target_not_found"},
        next_step="inspect ai/runtime/next_step_policy.py",
    )
    assert check.passed is False
