from __future__ import annotations

from typing import List

from ai.runtime.self_improvement.audit_report import AuditReport


def _list_lines(items: List[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {str(item)}" for item in items)


def build_codex_brief(audit_report: AuditReport) -> str:
    report = audit_report.to_dict() if hasattr(audit_report, "to_dict") else dict(audit_report or {})
    event = dict(report.get("event") or {})
    classification = dict(report.get("classification") or {})
    hypothesis = dict(report.get("hypothesis") or {})
    patch = dict(report.get("patch_plan") or {})
    evaluation = dict(report.get("evaluation_plan") or {})

    return (
        "Codex implementation brief\n\n"
        "Failure description:\n"
        f"- kind: {classification.get('failure_kind') or event.get('failure_kind') or 'unknown'}\n"
        f"- symptom: {event.get('symptom') or 'not provided'}\n"
        f"- expected: {event.get('expected_behavior') or hypothesis.get('expected_behavior') or 'not provided'}\n\n"
        "Target files:\n"
        f"{_list_lines(list(patch.get('target_files') or []))}\n\n"
        "Proposed changes:\n"
        f"{_list_lines(list(patch.get('proposed_changes') or []))}\n\n"
        "Tests to add:\n"
        f"{_list_lines(list(patch.get('tests_to_add') or []))}\n\n"
        "Tests to run:\n"
        f"{_list_lines(list(evaluation.get('commands') or patch.get('tests_to_run') or []))}\n\n"
        "Acceptance criteria:\n"
        f"{_list_lines(list(patch.get('acceptance_criteria') or []))}\n\n"
        "Constraints:\n"
        "- Do not use fictional assistant or external character names in code/comments/tests/logs/responses.\n"
        "- Do not fake memory, tool, file, or code access.\n"
        "- Do not hardcode final responses unless explicitly scoped to a small safe policy surface.\n"
        "- Preserve and extend tests.\n"
        "- Source-code changes require explicit approval before application.\n"
    )
