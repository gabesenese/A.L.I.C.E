from ai.runtime.operator_response_surface import (
    render_local_execution_error_response,
    render_operator_response,
    sanitize_operator_chatter,
)


def test_operator_surface_strips_service_chatter():
    out = render_operator_response(
        user_input="continue",
        base_text='"I am here to assist you with your Alice-related tasks today."',
        operator_state={},
        local_execution={
            "success": True,
            "analysis": {
                "summary": "it owns the bounded operator loop: plan, act, observe, verify, and update state"
            },
        },
        next_step=(
            "inspect ai/runtime/next_step_policy.py because it decides what Alice should do "
            "after each safe step"
        ),
    )
    low = out.lower()
    # Internal planning labels must NOT appear in user-facing output
    assert "finding:" not in low
    assert "next best move:" not in low
    # Meaningful content should still be present
    assert "bounded operator loop" in low
    # Service chatter must be stripped
    assert "assist you" not in low
    assert "i am here to assist you" not in low


def test_local_success_with_inspected_file():
    out = render_operator_response(
        user_input="continue",
        base_text="",
        operator_state={},
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {
                "summary": "it owns the bounded operator loop: plan, act, observe, verify, and update state"
            },
        },
        next_step=(
            "inspect ai/runtime/next_step_policy.py because it decides what Alice should do "
            "after each safe step"
        ),
    )
    low = out.lower()
    # Internal labels must not appear
    assert "i inspected" not in low
    assert "finding:" not in low
    assert "next best move:" not in low
    # Natural content should be present
    assert "bounded operator loop" in low


def test_local_failure_does_not_use_base_text():
    out = render_operator_response(
        user_input="continue",
        base_text="Sure, I can help.",
        operator_state={},
        local_execution={
            "success": False,
            "error": "target_not_found",
            "requested_target": "legacy-main.py",
        },
        next_step="inspect ai/runtime/agent_loop.py",
    )
    low = out.lower()
    assert "i couldn't verify the local step." in low
    assert "blocker:" in low
    assert "sure" not in low
    assert "i can help" not in low


def test_no_hardcoded_context_ack_in_operator_surface():
    out = render_operator_response(
        user_input="pretty good, its raining today so i want to stay at home and work on alice",
        base_text="",
        operator_state={},
        local_execution={
            "success": True,
            "analysis": {
                "summary": "it owns the bounded operator loop: plan, act, observe, verify, and update state"
            },
        },
        next_step="inspect ai/runtime/next_step_policy.py because it decides what Alice should do after each safe step",
    )
    low = out.lower()
    # No hardcoded context acknowledgements
    assert "cold day. good night to work on the core." not in low
    assert "makes sense. good time for a focused pass." not in low
    # No internal labels in user output
    assert "finding:" not in low
    assert "i inspected" not in low
    assert "next best move:" not in low
    # Should have meaningful content
    assert len(out.strip()) > 5


def test_sanitize_operator_chatter_removes_forbidden_assistant_phrases():
    text = (
        "I'm here to assist you with your Alice-related tasks today. "
        "Finding: bounded operator loop. "
        "What would you like to do?"
    )
    out = sanitize_operator_chatter(text)
    low = out.lower()
    assert "assist you" not in low
    assert "what would you like" not in low
    assert "bounded operator loop." in low


def test_local_execution_error_response_preserves_paragraph_breaks():
    out = render_local_execution_error_response(
        user_input="continue",
        base_text="",
        operator_state={},
        local_execution={"success": False, "error": "target_not_found"},
        next_step="inspect ai/runtime/agent_loop.py",
    )
    assert "I couldn't verify the local step.\n\nBlocker:" in out
    assert ".\n\nNext best move:" in out
