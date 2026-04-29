"""Integration tests for the learning-data QA CI gate."""

import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from tools.auditing.learning_data_qa_gate import main


def _write_valid_learning_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    learned_entry = {
        "pattern": "general:conversation:help",
        "alice_thought": {
            "type": "conversation:help",
            "data": {"user_input": "what can you do?"},
        },
        "ollama_phrasing": "I can help with notes, weather, reminders, and quick tasks.",
        "context": {"tone": "helpful", "user_input": "what can you do?"},
        "timestamp": "2026-03-14T12:00:00",
        "tone": "helpful",
    }
    (data_dir / "learned_phrasings.jsonl").write_text(
        json.dumps(learned_entry) + "\n", encoding="utf-8"
    )
    (data_dir / "entities.json").write_text("{}", encoding="utf-8")
    (data_dir / "relationships.json").write_text("[]", encoding="utf-8")
    (data_dir / "patterns.json").write_text(
        json.dumps({"conversation:help": 1}), encoding="utf-8"
    )

    return data_dir


def _write_critical_learning_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    bad_entry = {
        "pattern": "general:weather_advice",
        "alice_thought": {
            "type": "weather_advice",
            "temperature": 4,
            "condition": "rain",
            "clothing_item": "jacket",
            "user_question": "should i bring a jacket?",
        },
        "ollama_phrasing": "For Gabriel, you should bring a jacket.",
        "context": {"tone": "helpful", "user_input": "should i bring a jacket?"},
        "timestamp": "2026-03-14T12:00:00",
        "tone": "helpful",
    }
    (data_dir / "learned_phrasings.jsonl").write_text(
        json.dumps(bad_entry) + "\n", encoding="utf-8"
    )
    (data_dir / "entities.json").write_text("{}", encoding="utf-8")
    (data_dir / "relationships.json").write_text("[]", encoding="utf-8")
    (data_dir / "patterns.json").write_text(
        json.dumps({"weather_advice": 1}), encoding="utf-8"
    )

    return data_dir


def _write_warning_learning_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    learned_entry = {
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
        json.dumps(learned_entry) + "\n", encoding="utf-8"
    )

    entities = {
        "for": {
            "name": "for",
            "entity_type": "unknown",
            "aliases": [],
            "first_mentioned": "2026-03-14T12:00:00",
            "last_mentioned": "2026-03-14T12:00:00",
            "mention_count": 1,
            "confidence": 0.5,
            "metadata": {},
        },
        "and": {
            "name": "and",
            "entity_type": "unknown",
            "aliases": [],
            "first_mentioned": "2026-03-14T12:00:00",
            "last_mentioned": "2026-03-14T12:00:00",
            "mention_count": 1,
            "confidence": 0.5,
            "metadata": {},
        },
    }
    (data_dir / "entities.json").write_text(
        json.dumps(entities, indent=2), encoding="utf-8"
    )
    (data_dir / "relationships.json").write_text("[]", encoding="utf-8")
    (data_dir / "patterns.json").write_text(
        json.dumps({"conversation:help": 1}), encoding="utf-8"
    )

    return data_dir


def test_gate_fails_when_critical_findings_exist(tmp_path):
    _write_critical_learning_files(tmp_path)

    exit_code = main(
        [
            "--root",
            str(tmp_path),
        ]
    )

    assert exit_code == 2


def test_gate_fails_when_warning_growth_exceeds_threshold(tmp_path):
    _write_warning_learning_files(tmp_path)

    baseline_path = tmp_path / "qa_warning_baseline.json"
    baseline_payload = {"warning_count": 1}
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--warning-growth-threshold",
            "0.10",
        ]
    )

    assert exit_code == 5


def test_gate_passes_and_writes_baseline(tmp_path):
    _write_valid_learning_files(tmp_path)

    baseline_path = tmp_path / "qa_warning_baseline.json"

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--write-baseline",
        ]
    )

    assert exit_code == 0
    assert baseline_path.exists()

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline_payload["warning_count"] == 0
