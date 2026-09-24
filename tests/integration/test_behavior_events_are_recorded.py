"""Alice's improvement loop gets the failures it exists to learn from.

The pipeline passed user_id to ImprovementLoop.observe_event, which adds its own,
so every call raised TypeError into a bare except. No behaviour event, audit,
hypothesis or self-opinion was ever recorded, and "self improvement status" read
an empty store.
"""

import json
from pathlib import Path

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_a_failure_the_pipeline_sees_becomes_a_behaviour_event(tmp_path, monkeypatch):
    store = tmp_path / "si"
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(store))
    pipeline = ContractPipeline(build_runtime_boundaries(_FakeAlice()))

    pipeline._maybe_record_behavior_event(
        user_id="u1",
        source="user_correction",
        user_input="that's wrong",
        alice_response="It's sunny in Boston.",
        route="tool",
        intent="weather:current",
        failure_kind="wrong_answer",
        symptom="user said the answer was wrong",
    )

    rows = [json.loads(line) for line in Path(store / "behavior_events.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["user_input"] == "that's wrong"
    assert rows[0]["user_id"] == "u1"
