"""Core benchmark builder, scorer, and gate for A.L.I.C.E.

Objective:
- Correct intent + useful response rate on a locked 100-scenario benchmark.

Gate policy:
- Pass only when objective improves, or objective is neutral with lower latency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BENCHMARK_PATH = Path("data/benchmarks/core_100_scenarios.json")
DEFAULT_SCORECARD_PATH = Path("data/benchmarks/scorecard_latest.json")
DEFAULT_BASELINE_PATH = Path("data/benchmarks/scorecard_baseline.json")
CRITICAL_DOMAINS = ("notes", "weather", "conversation", "email")
MAX_CRITICAL_DOMAIN_PASS_RATE_DROP = 0.10

EXCLUDED_INTENTS = {
    "generation",
    "phrase_response",
}


def _normalize_prompt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _extract_records(
    training_data_path: Path, min_quality: float
) -> List[Dict[str, Any]]:
    if not training_data_path.exists():
        raise ValueError(f"Training data file not found: {training_data_path}")

    records: List[Dict[str, Any]] = []
    for line in training_data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        user_input = str(entry.get("user_input") or "").strip()
        intent = str(
            entry.get("intent") or (entry.get("context") or {}).get("intent") or ""
        ).strip()
        if not user_input or not intent:
            continue
        if intent in EXCLUDED_INTENTS:
            continue

        quality_score = entry.get("quality_score")
        if isinstance(quality_score, (int, float)) and quality_score < min_quality:
            continue

        domain = intent.split(":", 1)[0]
        timestamp = str(entry.get("timestamp") or "")

        records.append(
            {
                "user_input": user_input,
                "intent": intent,
                "domain": domain,
                "timestamp": timestamp,
                "quality_score": float(quality_score)
                if isinstance(quality_score, (int, float))
                else 0.0,
            }
        )

    return records


def _round_robin_select(
    grouped: Dict[str, List[Dict[str, Any]]], size: int
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    domain_names = sorted(grouped)
    idx_by_domain = {domain: 0 for domain in domain_names}

    while len(selected) < size:
        made_progress = False
        for domain in domain_names:
            items = grouped[domain]
            idx = idx_by_domain[domain]
            if idx >= len(items):
                continue
            selected.append(items[idx])
            idx_by_domain[domain] += 1
            made_progress = True
            if len(selected) >= size:
                break
        if not made_progress:
            break

    return selected


def build_core_benchmark(
    training_data_path: Path,
    output_path: Path,
    size: int = 100,
    min_quality: float = 0.7,
) -> Dict[str, Any]:
    records = _extract_records(training_data_path, min_quality=min_quality)
    if len(records) < size:
        raise ValueError(
            f"Not enough candidate records ({len(records)}) to build benchmark of size {size}."
        )

    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (_normalize_prompt(record["user_input"]), record["intent"])
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
            continue

        # Keep higher quality, then newer timestamp.
        if record["quality_score"] > existing["quality_score"]:
            deduped[key] = record
        elif (
            record["quality_score"] == existing["quality_score"]
            and record["timestamp"] > existing["timestamp"]
        ):
            deduped[key] = record

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in deduped.values():
        grouped.setdefault(rec["domain"], []).append(rec)

    for domain, items in grouped.items():
        items.sort(
            key=lambda item: (
                item["quality_score"],
                item["timestamp"],
                item["user_input"],
            ),
            reverse=True,
        )

    selected = _round_robin_select(grouped, size=size)
    if len(selected) < size:
        raise ValueError(
            f"Unable to select {size} benchmark records. Selected {len(selected)}."
        )

    scenarios = []
    for idx, rec in enumerate(selected, start=1):
        scenario_id = f"core_{idx:03d}"
        scenarios.append(
            {
                "id": scenario_id,
                "suite": rec["domain"],
                "description": f"Core benchmark prompt {idx}",
                "inputs": [rec["user_input"]],
                "expected_intent": rec["intent"],
                "should_not_clarify": True,
                "should_succeed": True,
                "min_confidence": 0.0,
                "tags": ["benchmark", "core-100", rec["domain"]],
                "context": {
                    "source": "training_data",
                    "timestamp": rec["timestamp"],
                    "quality_score": rec["quality_score"],
                },
            }
        )

    domain_counts: Dict[str, int] = {}
    for scenario in scenarios:
        domain_counts[scenario["suite"]] = domain_counts.get(scenario["suite"], 0) + 1

    payload = {
        "version": "1.0",
        "locked": True,
        "generated_at": datetime.now().isoformat(),
        "objective": "correct_intent_plus_useful_response",
        "size": len(scenarios),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source": {
            "training_data": str(training_data_path),
            "min_quality": min_quality,
        },
        "scenarios": scenarios,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return float(sorted_values[idx])


def score_benchmark(benchmark_path: Path) -> Dict[str, Any]:
    from test_scenarios import ScenarioRunner

    if not benchmark_path.exists():
        raise ValueError(f"Benchmark file not found: {benchmark_path}")

    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    scenarios = benchmark_payload.get("scenarios", [])
    if len(scenarios) != 100:
        raise ValueError(
            f"Benchmark must contain exactly 100 scenarios. Found: {len(scenarios)}"
        )

    runner = ScenarioRunner(benchmark_path)
    loaded = runner.load_scenarios()
    if loaded != 100:
        raise ValueError(f"ScenarioRunner loaded {loaded} scenarios, expected 100.")

    summary = runner.run_suite()
    results = summary.get("results", [])

    if len(results) != 100:
        raise ValueError(f"Expected 100 results, got {len(results)}.")

    scenario_by_id = {scenario["id"]: scenario for scenario in scenarios}

    intent_checked = 0
    intent_correct = 0
    useful_count = 0
    failures = 0
    latencies = []
    outcomes = {}
    per_domain: Dict[str, Dict[str, Any]] = {}

    for result in results:
        asdict(result)
        scenario = scenario_by_id.get(result.scenario_id)
        if not scenario:
            continue
        domain = scenario["suite"]
        expected_intent = scenario.get("expected_intent")

        domain_stats = per_domain.setdefault(
            domain,
            {
                "total": 0,
                "passed": 0,
                "intent_checked": 0,
                "intent_correct": 0,
                "useful": 0,
            },
        )

        domain_stats["total"] += 1
        if result.passed:
            domain_stats["passed"] += 1

        if expected_intent:
            intent_checked += 1
            domain_stats["intent_checked"] += 1
            if result.actual_intent == expected_intent:
                intent_correct += 1
                domain_stats["intent_correct"] += 1

        if result.passed and (result.response or "").strip():
            useful_count += 1
            domain_stats["useful"] += 1

        if not result.passed:
            failures += 1

        latencies.append(float(result.duration_ms))
        outcomes[result.scenario_id] = {
            "passed": bool(result.passed),
            "duration_ms": float(result.duration_ms),
            "actual_intent": result.actual_intent,
            "expected_intent": expected_intent,
            "errors": list(result.errors),
        }

    for domain, stats in per_domain.items():
        stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] else 0.0
        stats["intent_accuracy"] = (
            stats["intent_correct"] / stats["intent_checked"]
            if stats["intent_checked"]
            else 0.0
        )
        stats["useful_response_rate"] = (
            stats["useful"] / stats["total"] if stats["total"] else 0.0
        )

    intent_accuracy = intent_correct / intent_checked if intent_checked else 0.0
    useful_response_rate = useful_count / len(results) if results else 0.0
    objective_score = (intent_accuracy + useful_response_rate) / 2.0

    scorecard = {
        "generated_at": datetime.now().isoformat(),
        "benchmark_path": str(benchmark_path),
        "benchmark_size": len(results),
        "objective": {
            "name": "correct_intent_plus_useful_response",
            "intent_accuracy": intent_accuracy,
            "useful_response_rate": useful_response_rate,
            "objective_score": objective_score,
        },
        "summary": {
            "accuracy": summary.get("pass_rate", 0.0),
            "failure_count": failures,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p95_latency_ms": _p95(latencies),
            "regression_count": 0,
        },
        "per_domain": dict(sorted(per_domain.items())),
        "outcomes": outcomes,
    }

    return scorecard


def build_kpi_snapshot(scorecard: Dict[str, Any]) -> Dict[str, Any]:
    """P2 benchmark KPI bundle for dashboards/automation."""
    objective = scorecard.get("objective") or {}
    summary = scorecard.get("summary") or {}
    latency = scorecard.get("latency_ms") or {"p95": summary.get("p95_latency_ms", 0.0)}
    per_domain = scorecard.get("per_domain") or {}
    useful_response_rate = float(
        objective.get(
            "useful_response_rate", scorecard.get("useful_response_rate", 0.0)
        )
    )
    clarification_rate = 1.0 - useful_response_rate
    critical = {
        domain: (per_domain.get(domain) or {}).get("pass_rate", 0.0)
        for domain in CRITICAL_DOMAINS
    }
    return {
        "generated_at": scorecard.get("generated_at"),
        "objective_score": float(
            objective.get("objective_score", scorecard.get("objective_score", 0.0))
        ),
        "intent_accuracy": float(
            objective.get("intent_accuracy", scorecard.get("intent_accuracy", 0.0))
        ),
        "useful_response_rate": useful_response_rate,
        "clarification_rate_proxy": max(0.0, min(1.0, clarification_rate)),
        "latency_p95_ms": float(latency.get("p95", 0.0)),
        "critical_domain_pass_rates": critical,
    }


def _load_scorecard(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Scorecard file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid scorecard JSON at {path}: {exc}") from exc


def compare_scorecards(
    current: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    current_obj = float(current["objective"]["objective_score"])
    baseline_obj = float(baseline["objective"]["objective_score"])
    current_latency = float(current["summary"]["avg_latency_ms"])
    baseline_latency = float(baseline["summary"]["avg_latency_ms"])

    epsilon = 1e-6
    improved = current_obj > (baseline_obj + epsilon)
    neutral = abs(current_obj - baseline_obj) <= epsilon
    lower_latency = current_latency < baseline_latency

    regression_ids = []
    baseline_outcomes = baseline.get("outcomes", {})
    current_outcomes = current.get("outcomes", {})
    for scenario_id, base_result in baseline_outcomes.items():
        base_passed = bool(base_result.get("passed"))
        curr_passed = bool(current_outcomes.get(scenario_id, {}).get("passed"))
        if base_passed and not curr_passed:
            regression_ids.append(scenario_id)

    current["summary"]["regression_count"] = len(regression_ids)

    critical_domain_regressions: List[Dict[str, Any]] = []
    baseline_domains = baseline.get("per_domain", {})
    current_domains = current.get("per_domain", {})
    for domain in CRITICAL_DOMAINS:
        base_stats = baseline_domains.get(domain, {})
        curr_stats = current_domains.get(domain, {})
        base_pass_rate = float(base_stats.get("pass_rate", 0.0) or 0.0)
        curr_pass_rate = float(curr_stats.get("pass_rate", 0.0) or 0.0)
        if base_pass_rate <= 0.0:
            continue
        floor = base_pass_rate * (1.0 - MAX_CRITICAL_DOMAIN_PASS_RATE_DROP)
        if curr_pass_rate < floor:
            critical_domain_regressions.append(
                {
                    "domain": domain,
                    "baseline_pass_rate": base_pass_rate,
                    "current_pass_rate": curr_pass_rate,
                    "floor": floor,
                    "drop": base_pass_rate - curr_pass_rate,
                }
            )

    performance_passed = improved or (neutral and lower_latency)
    passed = performance_passed and not critical_domain_regressions
    reason = (
        "objective_improved"
        if improved
        else "objective_neutral_with_lower_latency"
        if (neutral and lower_latency)
        else "objective_regressed_or_latency_not_better"
    )
    if critical_domain_regressions:
        reason = "critical_domain_regression"

    return {
        "status": "pass" if passed else "fail",
        "reason": reason,
        "metrics": {
            "current_objective": current_obj,
            "baseline_objective": baseline_obj,
            "current_avg_latency_ms": current_latency,
            "baseline_avg_latency_ms": baseline_latency,
            "objective_delta": current_obj - baseline_obj,
            "latency_delta_ms": current_latency - baseline_latency,
            "regression_count": len(regression_ids),
            "regression_ids": regression_ids[:50],
            "critical_domain_regressions": critical_domain_regressions,
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_scorecard(scorecard: Dict[str, Any]) -> None:
    objective = scorecard["objective"]
    summary = scorecard["summary"]
    print("CORE 100 BENCHMARK SCORECARD")
    print("=" * 60)
    print(f"Generated: {scorecard['generated_at']}")
    print(f"Benchmark size: {scorecard['benchmark_size']}")
    print(f"Intent accuracy: {objective['intent_accuracy']:.4f}")
    print(f"Useful response rate: {objective['useful_response_rate']:.4f}")
    print(f"Objective score: {objective['objective_score']:.4f}")
    print(f"Failure count: {summary['failure_count']}")
    print(f"Avg latency (ms): {summary['avg_latency_ms']:.2f}")
    print(f"P95 latency (ms): {summary['p95_latency_ms']:.2f}")
    print(f"Regression count: {summary['regression_count']}")


def _print_gate(gate: Dict[str, Any]) -> None:
    print("BENCHMARK GATE")
    print("=" * 60)
    print(f"Status: {gate['status'].upper()}")
    print(f"Reason: {gate['reason']}")
    metrics = gate["metrics"]
    objective_delta = metrics.get("objective_delta")
    latency_delta = metrics.get("latency_delta_ms")
    print(
        "Objective delta: "
        + (
            f"{objective_delta:+.6f}"
            if isinstance(objective_delta, (int, float))
            else "N/A (baseline bootstrap)"
        )
    )
    print(
        "Latency delta (ms): "
        + (
            f"{latency_delta:+.2f}"
            if isinstance(latency_delta, (int, float))
            else "N/A (baseline bootstrap)"
        )
    )
    print(f"Regression count: {metrics['regression_count']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, run, and gate the core 100-scenario benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Build locked core benchmark from training data."
    )
    build_parser.add_argument(
        "--training-data", default="data/training/training_data.jsonl"
    )
    build_parser.add_argument("--output", default=str(DEFAULT_BENCHMARK_PATH))
    build_parser.add_argument("--size", type=int, default=100)
    build_parser.add_argument("--min-quality", type=float, default=0.7)

    run_parser = subparsers.add_parser("run", help="Run benchmark and emit scorecard.")
    run_parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    run_parser.add_argument("--scorecard", default=str(DEFAULT_SCORECARD_PATH))

    gate_parser = subparsers.add_parser("gate", help="Gate scorecard against baseline.")
    gate_parser.add_argument("--scorecard", default=str(DEFAULT_SCORECARD_PATH))
    gate_parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    gate_parser.add_argument("--write-baseline", action="store_true")

    run_gate_parser = subparsers.add_parser(
        "run-gate", help="Run benchmark then evaluate gate in one command."
    )
    run_gate_parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    run_gate_parser.add_argument("--scorecard", default=str(DEFAULT_SCORECARD_PATH))
    run_gate_parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    run_gate_parser.add_argument("--write-baseline", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "build":
        payload = build_core_benchmark(
            training_data_path=Path(args.training_data),
            output_path=Path(args.output),
            size=args.size,
            min_quality=args.min_quality,
        )
        print(f"Built benchmark at {args.output} with {payload['size']} scenarios.")
        return 0

    if args.command == "run":
        scorecard = score_benchmark(Path(args.benchmark))
        _write_json(Path(args.scorecard), scorecard)
        _print_scorecard(scorecard)
        print(f"Scorecard written to {args.scorecard}")
        return 0

    if args.command == "gate":
        scorecard = _load_scorecard(Path(args.scorecard))
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = _load_scorecard(baseline_path)
            gate = compare_scorecards(scorecard, baseline)
        else:
            gate = {
                "status": "pass",
                "reason": "no_baseline",
                "metrics": {
                    "current_objective": scorecard["objective"]["objective_score"],
                    "baseline_objective": None,
                    "current_avg_latency_ms": scorecard["summary"]["avg_latency_ms"],
                    "baseline_avg_latency_ms": None,
                    "objective_delta": None,
                    "latency_delta_ms": None,
                    "regression_count": 0,
                    "regression_ids": [],
                },
            }

        _print_gate(gate)
        if args.write_baseline and gate["status"] == "pass":
            _write_json(baseline_path, scorecard)
            print(f"Baseline written to {baseline_path}")
        return 0 if gate["status"] == "pass" else 1

    if args.command == "run-gate":
        scorecard = score_benchmark(Path(args.benchmark))
        _write_json(Path(args.scorecard), scorecard)
        _print_scorecard(scorecard)

        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = _load_scorecard(baseline_path)
            gate = compare_scorecards(scorecard, baseline)
        else:
            gate = {
                "status": "pass",
                "reason": "no_baseline",
                "metrics": {
                    "current_objective": scorecard["objective"]["objective_score"],
                    "baseline_objective": None,
                    "current_avg_latency_ms": scorecard["summary"]["avg_latency_ms"],
                    "baseline_avg_latency_ms": None,
                    "objective_delta": None,
                    "latency_delta_ms": None,
                    "regression_count": 0,
                    "regression_ids": [],
                },
            }

        _print_gate(gate)
        if args.write_baseline and gate["status"] == "pass":
            _write_json(baseline_path, scorecard)
            print(f"Baseline written to {baseline_path}")
        return 0 if gate["status"] == "pass" else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
