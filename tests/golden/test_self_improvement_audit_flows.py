from __future__ import annotations

from ai.runtime.self_improvement.behavior_event import create_behavior_event
from ai.runtime.self_improvement.failure_classifier import classify_failure
from ai.runtime.self_improvement.improvement_hypothesis import build_hypothesis
from ai.runtime.self_improvement.patch_plan import build_patch_plan
from ai.runtime.self_improvement.evaluation_plan import build_evaluation_plan
from ai.runtime.self_improvement.audit_report import build_audit_report


def test_greeting_failure_classification_and_audit_flow():
    event = create_behavior_event(
        source="manual_audit",
        symptom="Alice greeting feels too assistant-like and dry",
        expected_behavior="presence-first companion greeting",
        actual_behavior="How can I help today?",
    )
    classification = classify_failure(event)
    assert classification.failure_kind == "greeting_tone"
    assert "ai/runtime/greeting_surface_policy.py" in classification.likely_files

    hypothesis = build_hypothesis(event, classification)
    assert "overcorrected" in hypothesis.hypothesis.lower() or "presence" in hypothesis.hypothesis.lower()

    patch_plan = build_patch_plan(hypothesis)
    assert patch_plan.requires_approval is True
    assert patch_plan.can_auto_apply is False
    assert "tests/golden/test_greeting_memory_grounding.py" in patch_plan.tests_to_add

    evaluation = build_evaluation_plan(patch_plan)
    assert "pytest tests/golden/test_greeting_memory_grounding.py" in evaluation.commands
    assert "pytest" in evaluation.commands

    report = build_audit_report(
        event=event,
        classification=classification,
        hypothesis=hypothesis,
        patch_plan=patch_plan,
        evaluation_plan=evaluation,
    )
    assert report.approval_required is True
    assert "Failure classified as greeting_tone" in report.recommendation

