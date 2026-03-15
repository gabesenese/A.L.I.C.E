"""CI gate for learning-data QA reports.

This module enforces release-blocking policy around the LearningDataQAAuditor output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.auditing.training_data_auditor import LearningDataQAAuditor

EXIT_OK = 0
EXIT_CRITICAL = 2
EXIT_PARSE_STRUCTURE = 3
EXIT_SYSTEMIC_CROSS_COPY = 4
EXIT_WARNING_REGRESSION = 5
EXIT_BASELINE_ERROR = 6

STRUCTURAL_AREAS = {"learned_phrasings", "entities", "relationships"}


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_baseline(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return _load_json_file(Path(path))


def write_baseline(path: str, warning_count: int, report: Dict[str, Any]) -> None:
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "warning_count": warning_count,
        "total_findings": report.get("total_findings", 0),
        "severity_counts": report.get("severity_counts", {}),
        "source_report_generated_at": report.get("generated_at"),
    }
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _has_structural_parse_issues(report: Dict[str, Any]) -> bool:
    for finding in report.get("findings", []):
        if finding.get("area") not in STRUCTURAL_AREAS:
            continue
        code = str(finding.get("code", ""))
        if code.startswith("invalid_") or code == "missing_or_invalid_field":
            return True
    return False


def _has_cross_copy_critical_class(report: Dict[str, Any]) -> bool:
    critical_codes_in_data = set()
    critical_codes_in_app_data = set()

    for finding in report.get("findings", []):
        if finding.get("severity") != "critical":
            continue
        code = str(finding.get("code", ""))
        path = str(finding.get("path", ""))
        if path.startswith("data/"):
            critical_codes_in_data.add(code)
        elif path.startswith("app/data/"):
            critical_codes_in_app_data.add(code)

    return bool(critical_codes_in_data & critical_codes_in_app_data)


def _warning_regressed(
    warning_count: int,
    baseline: Optional[Dict[str, Any]],
    threshold_ratio: float,
) -> bool:
    if baseline is None:
        return False

    baseline_warning_count = baseline.get("warning_count")
    if not isinstance(baseline_warning_count, int) or baseline_warning_count < 0:
        raise ValueError("Baseline must contain a non-negative integer 'warning_count'.")

    allowed = int(baseline_warning_count * (1.0 + threshold_ratio))
    return warning_count > allowed


def evaluate_gate(
    report: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    warning_growth_threshold: float = 0.10,
) -> Dict[str, Any]:
    if warning_growth_threshold < 0:
        raise ValueError("warning_growth_threshold must be non-negative.")

    severity_counts = report.get("severity_counts", {})
    critical_count = int(severity_counts.get("critical", 0) or 0)
    warning_count = int(severity_counts.get("warning", 0) or 0)

    checks: List[Dict[str, Any]] = []

    critical_block = critical_count > 0
    checks.append(
        {
            "name": "critical_findings",
            "passed": not critical_block,
            "details": {"critical_count": critical_count},
            "exit_code_on_fail": EXIT_CRITICAL,
        }
    )

    parse_block = _has_structural_parse_issues(report)
    checks.append(
        {
            "name": "structural_parse_integrity",
            "passed": not parse_block,
            "details": {"areas": sorted(STRUCTURAL_AREAS)},
            "exit_code_on_fail": EXIT_PARSE_STRUCTURE,
        }
    )

    cross_copy_block = _has_cross_copy_critical_class(report)
    checks.append(
        {
            "name": "cross_copy_critical_contamination",
            "passed": not cross_copy_block,
            "details": {"rule": "same_critical_code_in_data_and_app_data"},
            "exit_code_on_fail": EXIT_SYSTEMIC_CROSS_COPY,
        }
    )

    warning_block = _warning_regressed(
        warning_count=warning_count,
        baseline=baseline,
        threshold_ratio=warning_growth_threshold,
    )
    checks.append(
        {
            "name": "warning_growth_threshold",
            "passed": not warning_block,
            "details": {
                "warning_count": warning_count,
                "baseline_warning_count": None if baseline is None else baseline.get("warning_count"),
                "threshold_ratio": warning_growth_threshold,
            },
            "exit_code_on_fail": EXIT_WARNING_REGRESSION,
        }
    )

    for check in checks:
        if not check["passed"]:
            return {
                "status": "fail",
                "exit_code": check["exit_code_on_fail"],
                "failed_check": check["name"],
                "checks": checks,
                "severity_counts": severity_counts,
            }

    return {
        "status": "pass",
        "exit_code": EXIT_OK,
        "failed_check": None,
        "checks": checks,
        "severity_counts": severity_counts,
    }


def _print_gate_summary(result: Dict[str, Any]) -> None:
    print("LEARNING DATA QA GATE")
    print("=" * 60)
    print(f"Status: {result['status'].upper()}")
    if result.get("failed_check"):
        print(f"Failed check: {result['failed_check']}")
    print(f"Exit code: {result['exit_code']}")
    severity_counts = result.get("severity_counts", {})
    print("Severity counts:")
    for severity in ("critical", "warning", "info"):
        if severity in severity_counts:
            print(f"  - {severity}: {severity_counts[severity]}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CI gate checks against learning-data QA findings.")
    parser.add_argument("--root", default=".", help="Project root to audit.")
    parser.add_argument("--report", help="Optional path for a JSON gate report.")
    parser.add_argument("--baseline", help="Optional baseline JSON with warning_count for trend checks.")
    parser.add_argument(
        "--warning-growth-threshold",
        type=float,
        default=0.10,
        help="Maximum allowed warning growth ratio versus baseline (default: 0.10 for +10%).",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write or update the baseline file from this run (requires --baseline).",
    )

    args = parser.parse_args(argv)

    try:
        baseline = load_baseline(args.baseline)
    except ValueError as exc:
        baseline_path = Path(args.baseline) if args.baseline else None
        # Allow first-run baseline creation when the user explicitly requested writing it.
        if args.write_baseline and baseline_path and not baseline_path.exists():
            baseline = None
        else:
            print(f"Baseline error: {exc}")
            return EXIT_BASELINE_ERROR

    auditor = LearningDataQAAuditor(root_dir=args.root)
    qa_report = auditor.audit()

    try:
        gate_result = evaluate_gate(
            report=qa_report,
            baseline=baseline,
            warning_growth_threshold=args.warning_growth_threshold,
        )
    except ValueError as exc:
        print(f"Gate evaluation error: {exc}")
        return EXIT_BASELINE_ERROR

    _print_gate_summary(gate_result)

    if args.write_baseline:
        if not args.baseline:
            print("Baseline error: --write-baseline requires --baseline.")
            return EXIT_BASELINE_ERROR
        write_baseline(args.baseline, int(qa_report.get("severity_counts", {}).get("warning", 0) or 0), qa_report)
        print(f"Baseline written: {args.baseline}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "gate": gate_result,
            "qa_report": qa_report,
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Gate report written: {report_path}")

    return int(gate_result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
