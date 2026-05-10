from ai.runtime.greeting_surface_policy import render_grounded_greeting


def test_a_llm_natural_greeting_accepted():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    low = result.text.lower()
    assert result.generated_by == "llm_constrained"
    assert "current objective" not in low
    assert "machine learning" not in low
    assert result.continuity_guard_applied is True


def test_b_old_machine_learning_bug_blocked():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "We were discussing machine learning last time. Good to see you.",
    )
    low = result.text.lower()
    assert "machine learning" not in low
    assert "last time" not in low
    assert "we were discussing" not in low
    assert result.generated_by == "fallback"
    assert result.continuity_guard_applied is True
    assert result.llm_candidate_rejected is True


def test_c_broad_memory_not_used_for_plain_greeting_metadata():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to hear from you. How's it going?",
    )
    assert result.suppressed_project_menu is True
    assert result.continuity_guard_applied is True


def test_d_active_state_not_forced_into_plain_greeting():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={"active_objective": "Improve Alice", "current_focus": "routing"},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you doing?",
    )
    low = result.text.lower()
    assert "routing" not in low
    assert "current objective" not in low
    assert "next best move" not in low


def test_e_active_state_allowed_for_explicit_continuation():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={"active_objective": "Improve Alice", "current_focus": "routing"},
        session_state={},
        user_input="hi alice, where were we?",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Still on routing. Ready to continue.",
    )
    low = result.text.lower()
    assert "still on routing" in low
    assert "machine learning" not in low


def test_f_service_greeting_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. How can I help you today?",
    )
    low = result.text.lower()
    assert result.generated_by == "fallback"
    assert "how can i help" not in low


def test_g_forced_companion_phrase_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. I'm with you. Let's keep it simple.",
    )
    low = result.text.lower()
    assert result.generated_by == "fallback"
    assert "let's keep it simple" not in low


def test_h_repeated_greeting_is_shorter():
    first = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={"greeting_count": 0},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to hear from you. How's it going?",
    )
    second = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={"greeting_count": 1},
        user_input="hello",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel.",
    )
    assert len(second.text.split()) <= len(first.text.split())
    low = second.text.lower()
    assert "current objective" not in low
    assert "machine learning" not in low
