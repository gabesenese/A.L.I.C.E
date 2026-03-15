"""Integration tests for the learning-data QA auditor."""

import json
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from tools.auditing.training_data_auditor import LearningDataQAAuditor


def _write_learning_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    learned = data_dir / "learned_phrasings.jsonl"
    entries = [
        {
            "pattern": "general:weather_advice",
            "alice_thought": {
                "type": "weather_advice",
                "temperature": 8,
                "condition": "snow",
                "clothing_item": "coat",
                "user_question": "should i wear a coat?",
            },
            "ollama_phrasing": "For Gabriel, I'd definitely recommend bringing a coat.",
            "context": {"tone": "helpful", "user_input": "should i wear a coat?"},
            "timestamp": "2026-03-14T12:00:00",
            "tone": "helpful",
        },
        {
            "pattern": "general:operation_success",
            "alice_thought": {
                "type": "operation_success",
                "details": {"found": False, "count": 0},
            },
            "ollama_phrasing": "I've successfully found your notes.",
            "context": {"tone": "helpful", "user_input": "find my notes"},
            "timestamp": "2026-03-14T12:01:00",
            "tone": "helpful",
        },
        {
            "pattern": "general:music:pause",
            "alice_thought": {
                "type": "music:pause",
                "data": {"user_input": "Where do I live?"},
            },
            "ollama_phrasing": "You live in Waterloo.",
            "context": {"tone": "helpful", "user_input": "Where do I live?"},
            "timestamp": "2026-03-14T12:02:00",
            "tone": "helpful",
        },
        {
            "pattern": "general:conversation:help",
            "alice_thought": {
                "type": "conversation:help",
                "data": {"user_input": "what can you do?"},
            },
            "ollama_phrasing": "This revised response still conveys the accurate information.",
            "context": {"tone": "helpful", "user_input": "what can you do?"},
            "timestamp": "2026-03-14T12:03:00",
            "tone": "helpful",
        },
    ]
    learned.write_text("\n".join(json.dumps(item) for item in entries) + "\n", encoding="utf-8")

    entities = {
        "for": {
            "name": "for",
            "entity_type": "unknown",
            "aliases": [],
            "first_mentioned": "2026-03-14T12:00:00",
            "last_mentioned": "2026-03-14T12:00:00",
            "mention_count": 3,
            "confidence": 0.5,
            "metadata": {},
        },
        "tomorrow\n3": {
            "name": "tomorrow\n3",
            "entity_type": "unknown",
            "aliases": [],
            "first_mentioned": "2026-03-14T12:00:00",
            "last_mentioned": "2026-03-14T12:00:00",
            "mention_count": 1,
            "confidence": 0.5,
            "metadata": {},
        },
    }
    (data_dir / "entities.json").write_text(json.dumps(entities, indent=2), encoding="utf-8")

    relationships = [
        {
            "source_entity": "for",
            "target_entity": "tomorrow\n3",
            "relationship_type": "owns",
            "confidence": 0.5,
            "context": "**You have 216 notes:**\n\n1. for tomorrow\n3. tagged wrk",
            "timestamp": "2026-03-14T12:10:00",
            "source": "conversation",
        }
    ]
    (data_dir / "relationships.json").write_text(json.dumps(relationships, indent=2), encoding="utf-8")
    (data_dir / "patterns.json").write_text("{}", encoding="utf-8")
    return data_dir


def test_learning_data_qa_flags_bad_learns_and_corrupted_knowledge(tmp_path):
    _write_learning_fixture(tmp_path)

    auditor = LearningDataQAAuditor(root_dir=str(tmp_path))
    report = auditor.audit()
    codes = {finding["code"] for finding in report["findings"]}

    assert "weather_personalization_leak" in codes
    assert "contradictory_success_content" in codes
    assert "wrong_domain_learning" in codes
    assert "llm_meta_artifact" in codes
    assert "noise_entity" in codes
    assert "entity_list_artifact" in codes
    assert "relationship_from_rendered_list" in codes
    assert report["group_summaries"]["learned_phrasings"]["total_findings"] == 4
    assert report["group_summaries"]["knowledge"]["total_findings"] == 6
    assert report["area_summaries"]["entities"]["issue_counts"]["entity_list_artifact"] == 1
    assert report["area_summaries"]["relationships"]["issue_counts"]["relationship_from_rendered_list"] == 1


def test_learning_data_qa_clean_quarantines_critical_records(tmp_path):
    data_dir = _write_learning_fixture(tmp_path)
    quarantine_dir = data_dir / "qa" / "quarantine"

    auditor = LearningDataQAAuditor(root_dir=str(tmp_path))
    result = auditor.clean(min_severity="critical", quarantine_dir=str(quarantine_dir))

    cleanup = result["cleanup"]
    assert cleanup["severity_threshold"] == "critical"
    assert cleanup["total_removed_records"] == 6

    learned_lines = (data_dir / "learned_phrasings.jsonl").read_text(encoding="utf-8").splitlines()
    assert learned_lines == []

    remaining_entities = json.loads((data_dir / "entities.json").read_text(encoding="utf-8"))
    assert set(remaining_entities) == {"for"}

    remaining_relationships = json.loads((data_dir / "relationships.json").read_text(encoding="utf-8"))
    assert remaining_relationships == []

    files = cleanup["files"]
    learned_summary = files["data/learned_phrasings.jsonl"]
    assert learned_summary["removed_records"] == 4
    assert learned_summary["quarantine_path"]
    assert Path(learned_summary["quarantine_path"]).exists()

    entity_summary = files["data/entities.json"]
    assert entity_summary["removed_records"] == 1
    assert Path(entity_summary["quarantine_path"]).exists()

    relationship_summary = files["data/relationships.json"]
    assert relationship_summary["removed_records"] == 1
    assert Path(relationship_summary["quarantine_path"]).exists()