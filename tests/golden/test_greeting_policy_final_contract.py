from datetime import datetime

from ai.runtime.greeting_surface_policy import render_grounded_greeting


def test_a_natural_greeting_accepted():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    assert result.generated_by == "llm_constrained"


def test_a2_generic_empty_greeting_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel! It's great to chat with you!",
    )
    assert result.generated_by in {"fallback", "llm_constrained"}
    assert result.text.strip().lower() != "hey gabriel! it's great to chat with you!"


def test_a3_retry_once_after_generic_greeting():
    call_count = {"n": 0}

    def _mock(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "Hey Gabriel! It's great to chat with you!"
        return "Hey Gabriel. Good to see you. How are you?"

    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=_mock,
    )
    assert call_count["n"] >= 2
    assert "great to chat with you" not in result.text.lower()


def test_a4_fallback_avoids_generic_empty_greeting():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "",
    )
    low = result.text.lower()
    assert "great to chat with you" not in low
    assert "nice to talk to you" not in low


def test_b_immediate_context_reply_accepted():
    text = "sorry, i meant alice, was just testing you, my day is going good, weather is warmer today"
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input=text,
        llm_generate=lambda *args, **kwargs: "I caught the correction. Glad your day is going well - warmer weather definitely helps.",
    )
    assert result.generated_by == "llm_constrained"


def test_c_multiple_questions_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        llm_generate=lambda *args, **kwargs: "What are you working on? Is it new? Want ideas?",
    )
    assert result.generated_by == "fallback"
    assert "too_many_questions" in result.validation_reasons


def test_d_assistant_service_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        llm_generate=lambda *args, **kwargs: "How can I help you today?",
    )
    assert result.generated_by == "fallback"


def test_e_soft_continuity_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        llm_generate=lambda *args, **kwargs: "Long time no chat. Nice to connect with you.",
    )
    assert result.generated_by == "fallback"


def test_e2_corporate_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Nice to connect with you.",
    )
    assert result.generated_by == "fallback"


def test_f_wrong_time_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        local_time=datetime.fromisoformat("2026-05-10T17:31:00-04:00"),
        llm_generate=lambda *args, **kwargs: "How's your morning going?",
    )
    assert result.generated_by == "fallback"
    assert "time_period_mismatch" in result.validation_reasons


def test_g_repetition_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        session_state={"recent_greeting_texts": ["Hey Gabriel. Good to see you. How are you?"]},
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    assert result.generated_by == "fallback"
    assert "repeated_candidate" in result.validation_reasons


def test_h_machine_learning_stale_memory_blocked():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi",
        llm_generate=lambda *args, **kwargs: "We were discussing machine learning last time.",
    )
    assert result.generated_by == "fallback"
    assert "machine learning" not in result.text.lower()


def test_i_plain_greeting_has_no_project_status():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        operator_state={"active_objective": "Improve", "current_focus": "routing"},
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    low = result.text.lower()
    assert "current objective" not in low
    assert "current focus" not in low
    assert "next best move" not in low


def test_j_explicit_continuation_may_mention_focus():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice, where were we?",
        operator_state={"active_objective": "Improve", "current_focus": "routing"},
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Still on routing. Ready to continue.",
    )
    assert "routing" in result.text.lower()


def test_k_metadata_complete():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to hear from you. How's it going?",
    )
    meta = result.continuity_claims
    assert meta.get("greeting_policy_version") == "greeting_v1_final"
    assert meta.get("broad_memory_suppressed") is True
    assert meta.get("anti_repetition_checked") is True
    assert "time_period" in meta
    assert result.continuity_guard_applied is True


def test_l_evening_aware_greeting_accepted():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        local_time=datetime.fromisoformat("2026-05-10T20:28:00-04:00"),
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How's your evening going?",
    )
    assert result.generated_by == "llm_constrained"


def test_m_repetition_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        session_state={"recent_greeting_texts": ["Hey Gabriel. Good to see you. How are you?"]},
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Good to see you. How are you?",
    )
    assert result.generated_by == "fallback"
    assert "repeated_candidate" in result.validation_reasons
