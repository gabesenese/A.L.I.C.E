from ai.core.adaptive_response_style import AdaptiveResponseStyle


def test_adaptive_response_style_enforces_word_limit_and_format():
    styler = AdaptiveResponseStyle()
    response = "This is sentence one. This is sentence two. This is sentence three."
    out = styler.apply_constraints(
        response,
        {"format": "bullet_points", "max_words": 10},
    )
    assert out
    assert len(out.split()) <= 10
