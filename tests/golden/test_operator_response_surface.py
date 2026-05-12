from ai.runtime.operator_response_surface import render_operator_response


def test_operator_continue_response_is_compact_and_decisive():
    out = render_operator_response(
        user_input="good, it was a cold day today so i just stayed home, now i am gonna work on alice for a bit",
        base_text="I've taken a look at core files and here are many options...",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/operator_state.py",
                "reason": "It stores active objective, current focus, inspected files, and recommendations.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"lines": 326, "classes": 5, "functions": 5},
        },
        next_step="inspect file ai/runtime/operator_state.py because It stores active objective, current focus, inspected files, and recommendations.",
    )
    low = out.lower()
    assert "warm and cozy" not in low
    assert "what would you like to tackle first" not in low
    assert out.count("ai/runtime/agent_loop.py") == 1
    assert "next best move:" in low
    assert "ai/runtime/operator_state.py" in out
    assert "\n\n" in out


def test_operator_state_analysis_surface_is_clean_and_non_repetitive():
    out = render_operator_response(
        user_input="inspect operator_state.py",
        base_text="verbose fallback",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/memory/project_memory.py",
                "reason": "persists objective state",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/operator_state.py",
            "analysis": {"responsibility": "state management"},
        },
        next_step="inspect ai/memory/project_memory.py",
    )
    assert out.startswith("I inspected ai/runtime/operator_state.py.")
    assert "Interpretation: this is state storage, not the routing brain." in out
    assert "let me know" not in out.lower()


def test_agent_loop_finding_is_specific_and_not_routing():
    out = render_operator_response(
        user_input="inspect agent loop",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="",
    )
    assert "bounded operator loop" in out.lower()
    assert "primary responsibility is routing" not in out.lower()
    assert "because It" not in out
    assert ".." not in out


def test_long_day_ack_is_grounded_and_not_cutesy():
    out = render_operator_response(
        user_input="good, it was a long day, but its just monday so we are trying to stay positive, I want to work on alice for a little bit",
        base_text="",
        operator_state={},
        local_execution={},
        next_step="inspect ai/runtime/response_momentum_policy.py",
    )
    low = out.lower()
    assert out.startswith("Long day. We'll keep this light.")
    assert "cozy" not in low
    assert "glad you're staying" not in low
    assert "that must have been hard" not in low
    assert "you've got this" not in low


def test_next_move_reason_grammar_is_clean():
    out = render_operator_response(
        user_input="inspect",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses.",
            }
        },
        local_execution={},
        next_step="",
    )
    assert (
        "Next best move: inspect ai/runtime/response_momentum_policy.py because it shapes whether Alice advances with momentum or drifts into passive responses."
        in out
    )
    assert "because It" not in out
    assert ".." not in out

