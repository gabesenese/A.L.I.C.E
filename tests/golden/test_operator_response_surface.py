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

