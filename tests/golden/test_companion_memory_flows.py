from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def _structured_rows(alice):
    return [
        row
        for row in alice.memory._stored
        if (row.get("context") or {}).get("memory_schema") == "personal_v1"
    ]


def test_a_personal_update_then_personal_recall_is_grounded():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn("I did some shopping today.", "u1", 1)
    assert first.handled is True

    second = pipeline.run_turn("what did I talk about my personal life?", "u1", 2)
    assert second.handled is True
    assert second.metadata["response_type"] == "personal_memory_grounded"
    assert "shopping" in second.response_text.lower()


def test_b_mixed_turn_keeps_code_route_and_recall_mentions_shopping():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn(
        "nothing new, just did some shopping today, are you able to check alice's codebase?",
        "u1",
        1,
    )
    assert first.metadata["route"] == "local"
    assert first.metadata["intent"] == "code:request"
    structured = _structured_rows(alice)
    assert any((r.get("context") or {}).get("domain") == "personal_life" for r in structured)

    second = pipeline.run_turn("what did I talk about my personal life?", "u1", 2)
    assert second.metadata["response_type"] == "personal_memory_grounded"
    assert "shopping" in second.response_text.lower()


def test_c_alice_project_memory_not_used_as_personal_life():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn("I want Alice to become more like Jarvis.", "u1", 1)
    assert first.handled is True

    second = pipeline.run_turn("what did I talk about my personal life?", "u1", 2)
    assert second.handled is True
    assert second.metadata["response_type"] == "personal_memory_insufficient"


def test_d_personal_concern_is_recalled():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn(
        "I feel like Alice remembers coding stuff but not my personal life.",
        "u1",
        1,
    )
    assert first.handled is True

    second = pipeline.run_turn("what did I talk about my personal life?", "u1", 2)
    assert second.handled is True
    assert second.metadata["response_type"] == "personal_memory_grounded"
    assert "personal-life context" in second.response_text.lower()


def test_e_filler_does_not_store_structured_memory():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))

    first = pipeline.run_turn("yeah", "u1", 1)
    assert first.handled is True
    assert _structured_rows(alice) == []
