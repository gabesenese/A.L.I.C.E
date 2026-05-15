from ai.runtime.greeting_surface_policy import render_grounded_greeting
from datetime import datetime


def test_a_llm_natural_greeting_accepted():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    low = result.text.lower()
    assert result.generated_by == "llm"
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
    assert result.generated_by == "none"
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
    assert result.generated_by == "none"
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
    assert result.generated_by == "none"
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


def test_i_recent_greeting_retry_rejects_repeat_and_accepts_second():
    calls = {"n": 0}

    def _llm(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Hey Gabriel! Good to see you here. How's your day going?"
        return "Hey Gabriel. Good to hear from you. How’s it going?"

    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={
            "recent_greeting_texts": [
                "Hey Gabriel! Good to see you here. How's your day going?"
            ]
        },
        user_input="hi alice",
        llm_generate=_llm,
    )
    assert "good to see you here. how's your day going?" not in result.text.lower()
    assert result.generated_by == "llm_retry"
    assert result.continuity_claims.get("repetition_retry") is True


def test_j_exact_duplicate_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={"recent_greeting_texts": ["Hey Gabriel. Good to see you. How are you?"]},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    assert result.generated_by == "none"
    assert "repeated_candidate" in result.validation_reasons


def test_k_similar_duplicate_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={
            "recent_greeting_texts": [
                "Hey Gabriel! Good to see you here. How's your day going?"
            ]
        },
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How's your day going?",
    )
    assert result.generated_by == "none"
    assert "repeated_candidate" in result.validation_reasons


def test_l_natural_different_greeting_accepted():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={"recent_greeting_texts": ["Hey Gabriel. Good to see you. How are you?"]},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to hear from you. How’s it going?",
    )
    assert result.generated_by == "llm"
    assert result.greeting_style in {"llm", "llm_retry"}


def test_m_soft_continuity_still_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel! Long time no chat. Nice to connect with you!",
    )
    assert result.generated_by == "none"


def test_n_stale_memory_regression_still_blocked_with_metadata():
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
    assert result.continuity_claims.get("broad_memory_suppressed") is True


def test_o_session_state_stores_recent_greetings_capped_to_five():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={
            "recent_greeting_texts": [
                "one",
                "two",
                "three",
                "four",
                "five",
            ]
        },
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Glad you’re here. How’s everything?",
    )
    recent = result.session_state.get("recent_greeting_texts", [])
    assert result.session_state.get("last_greeting_text")
    assert len(recent) == 5
    assert any("glad you re here" in item or "glad you're here" in item for item in recent)


def test_p_morning_phrase_rejected_in_evening():
    at_531pm = datetime.fromisoformat("2026-05-10T17:31:00-04:00")
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=at_531pm,
        llm_generate=lambda *args, **kwargs: "Hey Gabriel! It's great to connect with you today. How's your morning going so far?",
    )
    assert "morning" not in result.text.lower()
    assert "time_period_mismatch" in result.validation_reasons
    assert result.generated_by in {"none", "llm", "llm_retry"}


def test_q_evening_phrase_accepted_in_evening():
    at_531pm = datetime.fromisoformat("2026-05-10T17:31:00-04:00")
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=at_531pm,
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good evening. How’s it going?",
    )
    assert result.generated_by == "llm"
    assert result.validation_passed is True


def test_r_timeless_greeting_accepted_any_time():
    at_531pm = datetime.fromisoformat("2026-05-10T17:31:00-04:00")
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=at_531pm,
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    assert result.generated_by == "llm"


def test_s_unknown_time_rejects_time_sensitive_phrase():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=None,
        timezone_name="",
        llm_generate=lambda *args, **kwargs: "Good morning Gabriel. How’s your morning?",
    )
    assert result.generated_by == "none"
    assert "time_period_mismatch" in result.validation_reasons


def test_t_continuity_guard_still_works():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=datetime.fromisoformat("2026-05-10T17:31:00-04:00"),
        llm_generate=lambda *args, **kwargs: "We were discussing machine learning last time.",
    )
    low = result.text.lower()
    assert "machine learning" not in low
    assert "last time" not in low
    assert result.generated_by == "none"


def test_u_soft_continuity_still_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        local_time=datetime.fromisoformat("2026-05-10T17:31:00-04:00"),
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Long time no chat.",
    )
    assert result.generated_by == "none"


def test_v_task_intake_greeting_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hi. What should we work on?",
    )
    low = result.text.lower()
    assert result.generated_by == "none"
    assert "what should we work on" not in low
    assert "assistant_service_language" in result.validation_reasons or "banned_content" in result.validation_reasons


def test_w_rejected_greeting_does_not_reuse_previous_greeting_text():
    previous = "Hey Gabriel. Good to hear from you. How's it going?"
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={
            "last_greeting_text": previous,
            "recent_greeting_texts": [previous],
            "greeting_count": 1,
        },
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hi. What should we work on?",
    )
    assert result.generated_by == "none"
    assert result.text.strip() == ""


