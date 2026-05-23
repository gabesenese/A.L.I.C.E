from __future__ import annotations

from ai.runtime.self_improvement.behavior_event import create_behavior_event
from ai.runtime.self_improvement.storage import (
    get_data_path,
    read_behavior_events,
    record_behavior_event,
)


def test_behavior_event_saved_to_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path))
    event = create_behavior_event(
        source="manual_audit",
        symptom="Alice greeting feels too assistant-like",
        expected_behavior="presence-first companion greeting",
        actual_behavior="How can I help today?",
        failure_kind="greeting_tone",
    )
    record_behavior_event(event)
    rows = read_behavior_events(limit=10)
    assert rows
    assert rows[-1]["failure_kind"] == "greeting_tone"
    assert get_data_path("behavior_events.jsonl").exists()
