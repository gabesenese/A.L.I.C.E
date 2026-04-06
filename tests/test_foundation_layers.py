from ai.core.foundation_layers import FoundationLayers


def test_staged_inference_skips_deep_for_high_confidence_shallow_route():
    layers = FoundationLayers(budget_ms=120.0)
    budget = layers.new_budget()

    assert layers.should_run_deep_stage(budget, shallow_confidence=0.97) is False


def test_clarification_policy_flags_low_margin_cases():
    layers = FoundationLayers()

    state = layers.clarification_policy(
        plugin_scores={"notes": 1.0, "email": 0.9},
        confidence=0.9,
    )

    assert state["needs_clarification"] is True
    assert state["top_margin"] < 0.35
