"""Integration tests for core benchmark builder and gate logic."""

import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from tools.auditing.core_benchmark_gate import build_core_benchmark, compare_scorecards


def _write_training_data(path: Path, count: int = 150) -> None:
    domains = [
        ("notes", "notes:search"),
        ("weather", "weather:current"),
        ("email", "email:read"),
        ("conversation", "conversation:general"),
        ("music", "music:pause"),
    ]

    lines = []
    for idx in range(count):
        domain, intent = domains[idx % len(domains)]
        record = {
            "user_input": f"{domain} prompt {idx}",
            "intent": intent,
            "assistant_response": "ok",
            "timestamp": f"2026-03-14T12:{idx % 60:02d}:00",
            "quality_score": 0.9,
        }
        lines.append(json.dumps(record))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_core_benchmark_creates_locked_100(tmp_path):
    training_file = tmp_path / "training_data.jsonl"
    output_file = tmp_path / "core_100.json"
    _write_training_data(training_file, count=160)

    payload = build_core_benchmark(
        training_data_path=training_file,
        output_path=output_file,
        size=100,
        min_quality=0.7,
    )

    assert output_file.exists()
    assert payload["locked"] is True
    assert payload["size"] == 100
    assert len(payload["scenarios"]) == 100
    assert set(payload["domain_counts"]).issuperset({"notes", "weather", "email", "conversation", "music"})


def test_compare_scorecards_passes_on_objective_improvement():
    baseline = {
        "objective": {"objective_score": 0.80},
        "summary": {"avg_latency_ms": 100.0},
        "outcomes": {"core_001": {"passed": True}},
    }
    current = {
        "objective": {"objective_score": 0.85},
        "summary": {"avg_latency_ms": 130.0, "regression_count": 0},
        "outcomes": {"core_001": {"passed": True}},
    }

    gate = compare_scorecards(current=current, baseline=baseline)
    assert gate["status"] == "pass"
    assert gate["reason"] == "objective_improved"


def test_compare_scorecards_passes_on_neutral_objective_and_lower_latency():
    baseline = {
        "objective": {"objective_score": 0.82},
        "summary": {"avg_latency_ms": 100.0},
        "outcomes": {"core_001": {"passed": True}},
    }
    current = {
        "objective": {"objective_score": 0.82},
        "summary": {"avg_latency_ms": 95.0, "regression_count": 0},
        "outcomes": {"core_001": {"passed": True}},
    }

    gate = compare_scorecards(current=current, baseline=baseline)
    assert gate["status"] == "pass"
    assert gate["reason"] == "objective_neutral_with_lower_latency"


def test_compare_scorecards_fails_when_objective_regresses():
    baseline = {
        "objective": {"objective_score": 0.82},
        "summary": {"avg_latency_ms": 100.0},
        "outcomes": {"core_001": {"passed": True}},
    }
    current = {
        "objective": {"objective_score": 0.80},
        "summary": {"avg_latency_ms": 80.0, "regression_count": 0},
        "outcomes": {"core_001": {"passed": False}},
    }

    gate = compare_scorecards(current=current, baseline=baseline)
    assert gate["status"] == "fail"
    assert gate["metrics"]["regression_count"] == 1


def test_compare_scorecards_fails_on_critical_domain_drop():
    baseline = {
        "objective": {"objective_score": 0.82},
        "summary": {"avg_latency_ms": 100.0},
        "outcomes": {"core_001": {"passed": True}},
        "per_domain": {
            "notes": {"pass_rate": 0.80},
            "weather": {"pass_rate": 0.70},
            "conversation": {"pass_rate": 0.90},
            "email": {"pass_rate": 0.75},
        },
    }
    current = {
        "objective": {"objective_score": 0.85},
        "summary": {"avg_latency_ms": 90.0, "regression_count": 0},
        "outcomes": {"core_001": {"passed": True}},
        "per_domain": {
            "notes": {"pass_rate": 0.60},
            "weather": {"pass_rate": 0.70},
            "conversation": {"pass_rate": 0.90},
            "email": {"pass_rate": 0.75},
        },
    }

    gate = compare_scorecards(current=current, baseline=baseline)
    assert gate["status"] == "fail"
    assert gate["reason"] == "critical_domain_regression"
    assert gate["metrics"]["critical_domain_regressions"]


def test_compare_scorecards_passes_within_critical_domain_floor():
    baseline = {
        "objective": {"objective_score": 0.82},
        "summary": {"avg_latency_ms": 100.0},
        "outcomes": {"core_001": {"passed": True}},
        "per_domain": {
            "notes": {"pass_rate": 0.80},
            "weather": {"pass_rate": 0.70},
            "conversation": {"pass_rate": 0.90},
            "email": {"pass_rate": 0.75},
        },
    }
    current = {
        "objective": {"objective_score": 0.83},
        "summary": {"avg_latency_ms": 99.0, "regression_count": 0},
        "outcomes": {"core_001": {"passed": True}},
        "per_domain": {
            "notes": {"pass_rate": 0.73},
            "weather": {"pass_rate": 0.64},
            "conversation": {"pass_rate": 0.82},
            "email": {"pass_rate": 0.69},
        },
    }

    gate = compare_scorecards(current=current, baseline=baseline)
    assert gate["status"] == "pass"
    assert gate["metrics"]["critical_domain_regressions"] == []
