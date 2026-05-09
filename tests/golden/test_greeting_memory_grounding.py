from ai.runtime.greeting_surface_policy import (
    filter_learned_greetings,
    render_grounded_greeting,
    validate_chat_greeting,
)


def test_a_llm_three_sentence_greeting_can_pass():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel.\n\nGood to hear from you.\n\nI'm here.",
    )
    assert result.generated_by == "llm_constrained"
    assert result.text == "Hey Gabriel.\n\nGood to hear from you.\n\nI'm here."
    assert result.validation_reasons == []


def test_b_fake_continuity_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. We were discussing machine learning last time.",
    )
    low = result.text.lower()
    assert result.generated_by == "fallback"
    assert "machine learning" not in low
    assert "last time" not in low
    assert "fake_continuity" in result.validation_reasons or "banned_content" in result.validation_reasons


def test_c_assistant_greeting_rejected():
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
    assert "banned_content" in result.validation_reasons


def test_d_device_language_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Great to see you're back online.",
    )
    low = result.text.lower()
    assert result.generated_by == "fallback"
    assert "back online" not in low
    assert "banned_content" in result.validation_reasons


def test_e_weird_ambient_phrases_rejected():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. Nothing caught fire.",
    )
    low = result.text.lower()
    assert result.generated_by == "fallback"
    assert "nothing caught fire" not in low
    assert "banned_content" in result.validation_reasons


def test_f_fallback_is_minimal_without_active_state():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. How can I help?",
    )
    assert result.text == "Hey Gabriel."


def test_g_active_state_fallback_includes_focus():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "current_focus": "routing",
        },
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel. How can I help?",
    )
    assert "Still on routing." in result.text


def test_h_learned_greeting_validation():
    learned = [
        "Hey Gabriel. How can I help?",
        "Hey Gabriel. Good to hear from you.",
    ]
    accepted = filter_learned_greetings(learned)
    assert "Hey Gabriel. How can I help?" not in accepted
    assert "Hey Gabriel. Good to hear from you." in accepted


def test_validate_chat_greeting_rejects_task_intake_for_pure_greeting():
    result = validate_chat_greeting("Hey Gabriel. What are we doing tonight?", pure_greeting=True)
    assert result.valid is False
    assert "direct_task_intake" in result.reasons
