from ai.core.system_design_response_guard import SystemDesignResponseGuard


def test_detects_architecture_style_query():
    guard = SystemDesignResponseGuard()
    assert guard.is_architecture_query("If an AI assistant existed today, what machine learning foundations would it use?")


def test_guidance_mentions_required_system_design_criteria():
    guard = SystemDesignResponseGuard()
    text = guard.guidance_text().lower()
    assert "system design" in text
    assert "rnn" in text
    assert "permissions" in text
    assert "reliability" in text


def test_direct_answer_is_structured_and_not_vendor_list_only():
    guard = SystemDesignResponseGuard()
    out = guard.direct_answer("What would an advanced assistant architecture look like today?")
    assert out is not None
    lowered = out.lower()
    assert "orchestration" in lowered
    assert "permissions" in lowered
    assert "reliability" in lowered
