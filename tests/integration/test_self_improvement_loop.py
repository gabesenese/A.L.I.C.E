from __future__ import annotations

from ai.memory.project_memory import load_project_state
from ai.runtime.self_improvement.approval_gate import (
    ApprovalDecision,
    assert_approval_for_git_operation,
    assert_approval_for_merge,
    assert_approval_for_source_change,
)
from ai.runtime.self_improvement.behavior_event import create_behavior_event
from ai.runtime.self_improvement.improvement_loop import ImprovementLoop
from ai.runtime.self_improvement.storage import read_behavior_events


def test_self_improvement_audit_pipeline_updates_project_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path / "si"))
    monkeypatch.chdir(tmp_path)

    loop = ImprovementLoop(user_id="default")
    event = loop.observe_event(
        source="manual_audit",
        user_input="hi",
        alice_response="How can I help today?",
        symptom="Alice greeting feels too assistant-like",
        expected_behavior="presence-first companion greeting",
        actual_behavior="How can I help today?",
        failure_kind="greeting_tone",
    )
    report = loop.run_audit_from_event(event)
    assert report.approval_required is True
    assert report.classification["failure_kind"] == "greeting_tone"
    assert "ai/runtime/greeting_surface_policy.py" in report.patch_plan["target_files"]
    assert (
        "pytest tests/golden/test_greeting_memory_grounding.py"
        in report.evaluation_plan["commands"]
    )

    state = load_project_state("default")
    assert state.last_self_improvement_event_id == event.event_id
    assert state.last_audit_report_id == report.report_id


def test_approval_gate_rejects_unapproved_source_change():
    try:
        assert_approval_for_source_change(
            ApprovalDecision(approved=False, approval_type="source_change")
        )
    except PermissionError as exc:
        assert "not_approved" in str(exc)
    else:
        raise AssertionError(
            "expected approval gate to reject unapproved source change"
        )


def test_auto_audit_flag_defaults_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path / "si"))
    monkeypatch.delenv("ALICE_SELF_IMPROVEMENT_AUTO_AUDIT", raising=False)
    loop = ImprovementLoop(user_id="default")
    event = loop.observe_event(
        source="manual_audit",
        symptom="unsupported continuity claim",
        actual_behavior="we were discussing something",
    )
    assert loop.maybe_auto_audit(event) is None


def test_run_audit_records_event_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path / "si"))
    monkeypatch.chdir(tmp_path)
    loop = ImprovementLoop(user_id="default")
    event = create_behavior_event(
        source="manual_audit",
        symptom="route clarify unexpectedly",
        expected_behavior="direct execution when evidence is sufficient",
        actual_behavior="route selected clarify",
        failure_kind="routing",
    )
    report = loop.run_audit_from_event(event)
    assert report.event["event_id"] == event.event_id
    rows = read_behavior_events(limit=100)
    assert any(str(row.get("event_id")) == event.event_id for row in rows)


def test_pending_status_counts_open_events(monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path / "si"))
    monkeypatch.chdir(tmp_path)
    loop = ImprovementLoop(user_id="default")
    first = loop.observe_event(
        source="manual_audit",
        symptom="first failure",
        actual_behavior="bad",
        expected_behavior="good",
    )
    second = loop.observe_event(
        source="manual_audit",
        symptom="second failure",
        actual_behavior="bad",
        expected_behavior="good",
    )
    loop.run_audit_from_event(first)
    status = loop.pending_status()
    assert int(status["pending_event_count"]) == 1
    assert second.event_id in list(status["pending_event_ids"])


def test_approval_gate_requires_git_and_merge_approval():
    for fn, token in (
        (assert_approval_for_git_operation, "git"),
        (assert_approval_for_merge, "merge"),
    ):
        try:
            fn(ApprovalDecision(approved=False, approval_type="manual"))
        except PermissionError as exc:
            assert token in str(exc).lower()
        else:
            raise AssertionError("expected approval failure")
