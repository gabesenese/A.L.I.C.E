"""Startup Doctor for A.L.I.C.E.

Runs profile-based startup diagnostics and emits a single health summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from tools.auditing.learning_data_qa_gate import evaluate_gate
from tools.auditing.training_data_auditor import LearningDataQAAuditor


SEVERITY_PENALTY = {
    "critical": 30,
    "warning": 10,
    "info": 2,
}


@dataclass
class CheckResult:
    check_id: str
    status: str
    severity: str
    started_at: str
    finished_at: str
    duration_ms: int
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None


@dataclass
class CheckDefinition:
    check_id: str
    description: str
    severity: str
    timeout_s: float
    hard_blocker: bool
    profiles: Set[str]
    phase: str
    runner: Callable[["CheckContext"], CheckResult]
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class CheckContext:
    root_dir: Path
    profile: str
    run_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StartupDoctor:
    """Dependency-aware startup diagnostics runner."""

    PROFILES = {"fast", "standard", "deep"}

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self._checks: Dict[str, CheckDefinition] = {}
        self._register_defaults()

    def run(
        self,
        profile: str = "fast",
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if profile not in self.PROFILES:
            raise ValueError(f"Unsupported profile: {profile}")

        run_id = datetime.now().strftime("startup_%Y%m%d_%H%M%S")
        context = CheckContext(
            root_dir=self.root_dir,
            profile=profile,
            run_id=run_id,
            metadata=metadata or {},
        )

        selected_checks = [
            check for check in self._checks.values() if profile in check.profiles
        ]
        selected_ids = {check.check_id for check in selected_checks}

        for check in selected_checks:
            missing = check.dependencies - selected_ids
            if missing:
                raise ValueError(
                    f"Check '{check.check_id}' depends on checks not in profile '{profile}': {sorted(missing)}"
                )

        started = datetime.now().isoformat()
        phase_groups = defaultdict(list)
        for check in selected_checks:
            phase_groups[check.phase].append(check)

        all_results: Dict[str, CheckResult] = {}
        for phase_name in ("preflight", "postflight"):
            checks = phase_groups.get(phase_name, [])
            if not checks:
                continue
            phase_results = self._run_phase(checks, context, all_results)
            all_results.update(phase_results)

        report = self._build_report(
            profile=profile,
            run_id=run_id,
            started_at=started,
            results=all_results,
        )

        report_path = Path(output_path) if output_path else self.root_dir / "data" / "qa" / "startup_health_summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report

    def _run_phase(
        self,
        checks: Sequence[CheckDefinition],
        context: CheckContext,
        completed: Dict[str, CheckResult],
    ) -> Dict[str, CheckResult]:
        pending = {check.check_id: check for check in checks}
        results: Dict[str, CheckResult] = {}

        while pending:
            ready = [
                check
                for check in pending.values()
                if all(dep in completed or dep in results for dep in check.dependencies)
            ]
            if not ready:
                unresolved = sorted(pending)
                raise ValueError(f"Dependency cycle detected among checks: {unresolved}")

            with ThreadPoolExecutor(max_workers=min(4, len(ready))) as executor:
                futures = {
                    executor.submit(self._run_single_check, check, context): check
                    for check in ready
                }
                for future, check in futures.items():
                    results[check.check_id] = future.result()

            for check in ready:
                pending.pop(check.check_id, None)

        return results

    def _run_single_check(self, check: CheckDefinition, context: CheckContext) -> CheckResult:
        started_ts = datetime.now().isoformat()
        started_perf = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(check.runner, context)
            try:
                result = future.result(timeout=check.timeout_s)
                if result.check_id != check.check_id:
                    result.check_id = check.check_id
                return result
            except FuturesTimeoutError:
                duration_ms = int((time.perf_counter() - started_perf) * 1000)
                future.cancel()
                return CheckResult(
                    check_id=check.check_id,
                    status="timeout",
                    severity=check.severity,
                    started_at=started_ts,
                    finished_at=datetime.now().isoformat(),
                    duration_ms=duration_ms,
                    summary=f"{check.description} timed out after {check.timeout_s}s.",
                    details={"timeout_s": check.timeout_s},
                    remediation="Increase timeout or optimize this check.",
                )
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started_perf) * 1000)
                return CheckResult(
                    check_id=check.check_id,
                    status="error",
                    severity=check.severity,
                    started_at=started_ts,
                    finished_at=datetime.now().isoformat(),
                    duration_ms=duration_ms,
                    summary=f"{check.description} failed: {exc}",
                    details={"exception": type(exc).__name__, "message": str(exc)},
                    remediation="Inspect logs and rerun this check manually.",
                )

    def _build_report(
        self,
        profile: str,
        run_id: str,
        started_at: str,
        results: Dict[str, CheckResult],
    ) -> Dict[str, Any]:
        finished_at = datetime.now().isoformat()
        sorted_results = [results[key] for key in sorted(results)]

        severity_failures = defaultdict(int)
        total_penalty = 0
        hard_blocked = False
        failed_checks = []

        for result in sorted_results:
            if result.status in {"fail", "error", "timeout"}:
                severity_failures[result.severity] += 1
                total_penalty += SEVERITY_PENALTY.get(result.severity, 0)
                failed_checks.append(result.check_id)
                check_def = self._checks[result.check_id]
                if check_def.hard_blocker:
                    hard_blocked = True

        health_score = max(0, 100 - total_penalty)

        if hard_blocked:
            status = "blocked"
        elif failed_checks:
            status = "degraded"
        else:
            status = "healthy"

        recommendations = self._build_recommendations(sorted_results)

        return {
            "run_id": run_id,
            "profile": profile,
            "generated_at": finished_at,
            "started_at": started_at,
            "status": status,
            "health_score": health_score,
            "severity_failures": dict(severity_failures),
            "failed_checks": failed_checks,
            "recommendations": recommendations,
            "checks": [asdict(result) for result in sorted_results],
        }

    def _build_recommendations(self, results: Iterable[CheckResult]) -> List[str]:
        recommendations: List[str] = []
        for result in results:
            if result.status == "pass":
                continue
            if result.remediation:
                recommendations.append(f"{result.check_id}: {result.remediation}")
        return recommendations[:8]

    def _register(self, check: CheckDefinition) -> None:
        if check.check_id in self._checks:
            raise ValueError(f"Duplicate check id: {check.check_id}")
        self._checks[check.check_id] = check

    def _register_defaults(self) -> None:
        self._register(
            CheckDefinition(
                check_id="preflight_required_paths",
                description="Verify required runtime paths are present.",
                severity="critical",
                timeout_s=1.0,
                hard_blocker=True,
                profiles={"fast", "standard", "deep"},
                phase="preflight",
                runner=self._check_required_paths,
            )
        )
        self._register(
            CheckDefinition(
                check_id="learning_data_qa_gate",
                description="Run learning data QA gate policy.",
                severity="critical",
                timeout_s=20.0,
                hard_blocker=True,
                profiles={"fast", "standard", "deep"},
                phase="preflight",
                runner=self._check_learning_data_gate,
                dependencies={"preflight_required_paths"},
            )
        )
        self._register(
            CheckDefinition(
                check_id="intent_diagnostics_snapshot",
                description="Build intent correction diagnostics snapshot.",
                severity="warning",
                timeout_s=5.0,
                hard_blocker=False,
                profiles={"standard", "deep"},
                phase="postflight",
                runner=self._check_intent_diagnostics,
                dependencies={"preflight_required_paths"},
            )
        )
        self._register(
            CheckDefinition(
                check_id="startup_smoke_test",
                description="Run startup smoke test script.",
                severity="warning",
                timeout_s=70.0,
                hard_blocker=False,
                profiles={"deep"},
                phase="postflight",
                runner=self._check_startup_smoke,
                dependencies={"preflight_required_paths"},
            )
        )

    def _check_required_paths(self, context: CheckContext) -> CheckResult:
        started = datetime.now().isoformat()
        start_perf = time.perf_counter()
        required = [
            context.root_dir / "app" / "main.py",
            context.root_dir / "data",
            context.root_dir / "tools" / "auditing" / "training_data_auditor.py",
        ]
        missing = [str(path.relative_to(context.root_dir)) for path in required if not path.exists()]

        status = "pass" if not missing else "fail"
        summary = "Required runtime paths are available." if not missing else "Required runtime paths are missing."
        remediation = None if not missing else "Restore missing runtime paths before startup diagnostics."

        return CheckResult(
            check_id="preflight_required_paths",
            status=status,
            severity="critical",
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
            summary=summary,
            details={"missing": missing},
            remediation=remediation,
        )

    def _check_learning_data_gate(self, context: CheckContext) -> CheckResult:
        started = datetime.now().isoformat()
        start_perf = time.perf_counter()

        auditor = LearningDataQAAuditor(root_dir=str(context.root_dir))
        qa_report = auditor.audit()
        baseline_path = context.root_dir / "data" / "qa" / "warning_baseline.json"
        baseline = None
        if baseline_path.exists():
            try:
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                baseline = None

        gate = evaluate_gate(report=qa_report, baseline=baseline, warning_growth_threshold=0.10)
        status = "pass" if gate.get("status") == "pass" else "fail"

        remediation = None
        if status == "fail":
            remediation = "Run learning data QA report and quarantine critical records before release."

        return CheckResult(
            check_id="learning_data_qa_gate",
            status=status,
            severity="critical",
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
            summary=f"Learning data QA gate status: {gate.get('status')}.",
            details={
                "gate": gate,
                "total_findings": qa_report.get("total_findings", 0),
                "severity_counts": qa_report.get("severity_counts", {}),
                "group_summaries": qa_report.get("group_summaries", {}),
            },
            remediation=remediation,
        )

    def _check_intent_diagnostics(self, context: CheckContext) -> CheckResult:
        started = datetime.now().isoformat()
        start_perf = time.perf_counter()
        corrections = context.root_dir / "memory" / "auto_generated_corrections.jsonl"
        if not corrections.exists():
            return CheckResult(
                check_id="intent_diagnostics_snapshot",
                status="skip",
                severity="warning",
                started_at=started,
                finished_at=datetime.now().isoformat(),
                duration_ms=int((time.perf_counter() - start_perf) * 1000),
                summary="No correction history yet, diagnostics snapshot skipped.",
                details={"corrections_path": str(corrections)},
                remediation="Generate correction history through normal usage, then rerun standard/deep profile.",
            )

        from tools.intent_diagnostics import _build_confusion, _load_corrections, _precision_recall

        records = _load_corrections(corrections)
        if not records:
            return CheckResult(
                check_id="intent_diagnostics_snapshot",
                status="skip",
                severity="warning",
                started_at=started,
                finished_at=datetime.now().isoformat(),
                duration_ms=int((time.perf_counter() - start_perf) * 1000),
                summary="Correction history is empty, diagnostics snapshot skipped.",
                details={"corrections_path": str(corrections)},
                remediation="Collect more correction events before relying on intent precision trends.",
            )

        matrix = _build_confusion(records)
        stats = _precision_recall(matrix)
        macro_f1_values = [entry["f1"] for entry in stats.values() if entry.get("support", 0) > 0]
        macro_f1 = round(sum(macro_f1_values) / len(macro_f1_values), 3) if macro_f1_values else 0.0

        status = "pass" if macro_f1 >= 0.60 else "fail"
        remediation = None if status == "pass" else "Review top confusion pairs and retrain intent routing hot paths."

        return CheckResult(
            check_id="intent_diagnostics_snapshot",
            status=status,
            severity="warning",
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
            summary=f"Intent diagnostics macro F1={macro_f1}.",
            details={
                "records": len(records),
                "macro_f1": macro_f1,
                "intent_count": len(stats),
                "top_intents": sorted(
                    (
                        {"intent": intent, "f1": data["f1"], "support": data["support"]}
                        for intent, data in stats.items()
                    ),
                    key=lambda item: (item["support"], item["f1"]),
                    reverse=True,
                )[:5],
            },
            remediation=remediation,
        )

    def _check_startup_smoke(self, context: CheckContext) -> CheckResult:
        started = datetime.now().isoformat()
        start_perf = time.perf_counter()

        python_exe = context.metadata.get("python_executable")
        if not python_exe:
            venv_python = context.root_dir / ".venv" / "Scripts" / "python.exe"
            python_exe = str(venv_python) if venv_python.exists() else "python"

        command = [python_exe, str(context.root_dir / "test_init.py")]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(context.root_dir)

        # Prevent nested startup diagnostics when smoke test bootstraps ALICE.
        env.setdefault("ALICE_STARTUP_DOCTOR", "0")

        smoke_timeout = int(context.metadata.get("startup_smoke_timeout_s", 60))
        try:
            completed = subprocess.run(
                command,
                cwd=str(context.root_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=smoke_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CheckResult(
                check_id="startup_smoke_test",
                status="timeout",
                severity="warning",
                started_at=started,
                finished_at=datetime.now().isoformat(),
                duration_ms=int((time.perf_counter() - start_perf) * 1000),
                summary=f"Startup smoke test timed out after {smoke_timeout}s.",
                details={
                    "timeout_s": smoke_timeout,
                    "stdout_tail": (exc.stdout or "")[-1200:],
                    "stderr_tail": (exc.stderr or "")[-1200:],
                },
                remediation="Increase smoke timeout or run test_init.py manually to profile startup bottlenecks.",
            )

        status = "pass" if completed.returncode == 0 else "fail"
        remediation = None if status == "pass" else "Inspect test_init.py output and initialization dependencies."

        return CheckResult(
            check_id="startup_smoke_test",
            status=status,
            severity="warning",
            started_at=started,
            finished_at=datetime.now().isoformat(),
            duration_ms=int((time.perf_counter() - start_perf) * 1000),
            summary="Startup smoke test completed." if status == "pass" else "Startup smoke test failed.",
            details={
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1200:],
                "stderr_tail": completed.stderr[-1200:],
            },
            remediation=remediation,
        )


def _print_console_summary(report: Dict[str, Any]) -> None:
    print("STARTUP DOCTOR")
    print("=" * 60)
    print(f"Run ID: {report['run_id']}")
    print(f"Profile: {report['profile']}")
    print(f"Status: {report['status'].upper()}")
    print(f"Health score: {report['health_score']}")

    if report.get("severity_failures"):
        print("Failure buckets:")
        for severity in ("critical", "warning", "info"):
            count = report["severity_failures"].get(severity)
            if count:
                print(f"  - {severity}: {count}")

    print("Checks:")
    for check in report.get("checks", []):
        print(f"  - {check['check_id']}: {check['status']} ({check['duration_ms']} ms)")

    recommendations = report.get("recommendations", [])
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run startup diagnostics for A.L.I.C.E.")
    parser.add_argument("--root", default=".", help="Project root path.")
    parser.add_argument(
        "--profile",
        choices=sorted(StartupDoctor.PROFILES),
        default="fast",
        help="Diagnostic profile to run.",
    )
    parser.add_argument("--output", help="Optional custom report output path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when status is blocked or degraded.")

    args = parser.parse_args(argv)

    doctor = StartupDoctor(root_dir=args.root)
    report = doctor.run(profile=args.profile, output_path=args.output)
    _print_console_summary(report)

    if args.strict and report.get("status") in {"blocked", "degraded"}:
        return 1
    if report.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
