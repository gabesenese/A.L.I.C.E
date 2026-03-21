from ai.core.adaptive_intent_calibrator import AdaptiveIntentCalibrator
from ai.core.constraint_preference_extractor import ConstraintPreferenceExtractor
from ai.core.context_intent_refiner import ContextIntentRefiner
from ai.core.cognitive_orchestrator import CognitiveOrchestrator


def test_adaptive_intent_calibrator_penalizes_after_corrections():
    cal = AdaptiveIntentCalibrator()
    base = cal.calibrate("weather:current", 0.82)
    cal.record_feedback("weather:current", was_correct=False)
    cal.record_feedback("weather:current", was_correct=False)
    adjusted = cal.calibrate("weather:current", 0.82)
    assert adjusted < base


def test_context_intent_refiner_uses_debug_context():
    refiner = ContextIntentRefiner()
    out = refiner.refine(
        user_input="can you analyze this",
        intent="conversation:general",
        confidence=0.52,
        recent_topic="debug traceback for parser",
        last_intent="conversation:question",
    )
    assert out["intent"] == "conversation:question"
    assert float(out["confidence"]) >= 0.7


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
