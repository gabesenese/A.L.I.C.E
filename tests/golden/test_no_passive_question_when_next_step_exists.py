from ai.runtime.response_momentum_policy import apply_response_momentum


def test_no_passive_question_when_next_step_exists():
    out = apply_response_momentum(
        user_input="lets work on alice",
        response_text="These should give us a good starting point. What would you like to tackle first?",
        intent="operator:continue",
        route="local",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/operator_state.py",
                "reason": "It stores active objective and next-step state.",
            }
        },
        project_memory={},
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"lines": 326, "classes": 5, "functions": 5},
        },
        next_step="inspect file ai/runtime/operator_state.py because It stores active objective and next-step state.",
    )
    low = out.lower()
    # Passive open-ended question must be stripped
    assert "what would you like to tackle first" not in low
    # Natural next-step hint must appear without the "Next best move:" label
    assert "next best move:" not in low
    assert "operator_state.py" in low
    # Internal planning label must not appear in user-facing output
    assert "finding:" not in low
