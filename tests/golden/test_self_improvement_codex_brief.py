from __future__ import annotations

from ai.runtime.self_improvement.audit_report import build_audit_report
from ai.runtime.self_improvement.behavior_event import create_behavior_event
from ai.runtime.self_improvement.codex_brief import build_codex_brief
from ai.runtime.self_improvement.evaluation_plan import build_evaluation_plan
from ai.runtime.self_improvement.failure_classifier import classify_failure
from ai.runtime.self_improvement.improvement_hypothesis import build_hypothesis
from ai.runtime.self_improvement.patch_plan import build_patch_plan


def test_codex_brief_contains_targets_tests_acceptance_and_constraints():
    event = create_behavior_event(
        source="manual_audit",
        symptom="We were discussing machine learning last time claim appeared",
        expected_behavior="no unsupported continuity claims",
        actual_behavior="We were discussing machine learning last time. How can I help?",
    )
    classification = classify_failure(event)
    hypothesis = build_hypothesis(event, classification)
    patch_plan = build_patch_plan(hypothesis)
    evaluation_plan = build_evaluation_plan(patch_plan)
    report = build_audit_report(
        event=event,
        classification=classification,
        hypothesis=hypothesis,
        patch_plan=patch_plan,
        evaluation_plan=evaluation_plan,
    )
    brief = build_codex_brief(report)
    assert "Target files:" in brief
    assert "Tests to run:" in brief
    assert "Acceptance criteria:" in brief
    assert "Do not use fictional assistant or external character names" in brief
    assert "Source-code changes require explicit approval" in brief

