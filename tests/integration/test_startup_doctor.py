"""Integration tests for startup doctor orchestration."""

import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from tools.auditing.startup_doctor import StartupDoctor, main


def _seed_required_layout(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir(parents=True)
    (tmp_path / "app" / "main.py").write_text("# placeholder\n", encoding="utf-8")

    (tmp_path / "tools" / "auditing").mkdir(parents=True)
    (tmp_path / "tools" / "auditing" / "training_data_auditor.py").write_text(
        "# placeholder\n", encoding="utf-8"
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    return data_dir


def _write_clean_learning_data(tmp_path: Path) -> None:
    data_dir = _seed_required_layout(tmp_path)
    entry = {
        "pattern": "general:conversation:help",
        "alice_thought": {
            "type": "conversation:help",
            "data": {"user_input": "what can you do?"},
        },
        "ollama_phrasing": "I can help with notes and reminders.",
        "context": {"tone": "helpful", "user_input": "what can you do?"},
        "timestamp": "2026-03-14T12:00:00",
        "tone": "helpful",
    }
    (data_dir / "learned_phrasings.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )
    (data_dir / "entities.json").write_text("{}", encoding="utf-8")
    (data_dir / "relationships.json").write_text("[]", encoding="utf-8")
    (data_dir / "patterns.json").write_text(
        json.dumps({"conversation:help": 1}), encoding="utf-8"
    )


def _write_critical_learning_data(tmp_path: Path) -> None:
    data_dir = _seed_required_layout(tmp_path)
    entry = {
        "pattern": "general:weather_advice",
        "alice_thought": {
            "type": "weather_advice",
            "temperature": 6,
            "condition": "rain",
            "clothing_item": "jacket",
            "user_question": "should i bring a jacket?",
        },
        "ollama_phrasing": "For Gabriel, bring a jacket.",
        "context": {"tone": "helpful", "user_input": "should i bring a jacket?"},
        "timestamp": "2026-03-14T12:00:00",
        "tone": "helpful",
    }
    (data_dir / "learned_phrasings.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )
    (data_dir / "entities.json").write_text("{}", encoding="utf-8")
    (data_dir / "relationships.json").write_text("[]", encoding="utf-8")
    (data_dir / "patterns.json").write_text(
        json.dumps({"weather": 1}), encoding="utf-8"
    )


def test_startup_doctor_fast_profile_healthy(tmp_path):
    _write_clean_learning_data(tmp_path)

    doctor = StartupDoctor(root_dir=str(tmp_path))
    report = doctor.run(profile="fast")

    assert report["status"] == "healthy"
    assert report["health_score"] == 100

    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["preflight_required_paths"]["status"] == "pass"
    assert checks["learning_data_qa_gate"]["status"] == "pass"

    summary_path = tmp_path / "data" / "qa" / "startup_health_summary.json"
    assert summary_path.exists()


def test_startup_doctor_blocks_on_critical_gate_failure(tmp_path):
    _write_critical_learning_data(tmp_path)

    doctor = StartupDoctor(root_dir=str(tmp_path))
    report = doctor.run(profile="fast")

    assert report["status"] == "blocked"
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["learning_data_qa_gate"]["status"] == "fail"


def test_startup_doctor_cli_returns_blocked_exit_code(tmp_path):
    _write_critical_learning_data(tmp_path)

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--profile",
            "fast",
        ]
    )

    assert exit_code == 2
