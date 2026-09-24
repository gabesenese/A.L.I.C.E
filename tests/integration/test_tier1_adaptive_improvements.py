from ai.core.constraint_preference_extractor import ConstraintPreferenceExtractor
from ai.core.cognitive_orchestrator import CognitiveOrchestrator


def test_constraint_preference_extractor_detects_format_and_detail():
    extractor = ConstraintPreferenceExtractor()
    prefs = extractor.extract("Give me a detailed answer in bullet points under 120 words with examples")
    assert prefs["format"] == "bullet_points"
    assert prefs["detail"] == "detailed"
    assert prefs["max_words"] == 120
    assert "include_examples" in prefs["constraints"]


def test_cognitive_orchestrator_feedback_affects_next_guidance():
    orch = CognitiveOrchestrator(tick_interval_seconds=60)
    orch.ingest_user_feedback(
        user_input="that's wrong",
        previous_intent="weather:current",
        corrected_intent="",
        severity=0.9,
    )
    guidance = orch.get_runtime_guidance()
    assert guidance["route_bias"] == "clarify_first"
    assert "feedback_adjustments" in guidance
    assert "weather:current" in guidance["feedback_adjustments"]
