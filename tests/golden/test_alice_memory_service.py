from __future__ import annotations

from ai.memory.alice_memory_schema import ActiveConceptThread
from ai.memory.alice_memory_service import AliceMemoryService


def _service(tmp_path):
    svc = AliceMemoryService(db_path=str(tmp_path / "alice_memory.db"))
    svc.initialize()
    return svc


def test_save_and_retrieve_fact(tmp_path):
    svc = _service(tmp_path)
    saved = svc.save_fact(
        "Alice should remain local-first.",
        topic="architecture",
        confidence=0.82,
        importance=8,
        source="test",
    )
    out = svc.search_memories("local-first architecture", limit=3)
    assert out
    assert any(item.record.id == saved.id for item in out)


def test_save_and_retrieve_active_concept_thread(tmp_path):
    svc = _service(tmp_path)
    thread = ActiveConceptThread(
        topic="proactive AI companion",
        constraints=["not chatbot", "proactive"],
        signals=["proactive companion"],
        last_user_inputs=["i want alice to be proactive"],
        updated_at="2026-05-16T00:00:00+00:00",
        confidence=0.9,
    )
    svc.save_concept_thread(thread)
    restored = svc.get_active_concept_thread()
    assert restored is not None
    assert restored.topic == "proactive AI companion"
    assert "proactive" in restored.constraints


def test_search_returns_verified_and_hint_labels(tmp_path):
    svc = _service(tmp_path)
    svc.save_fact("High confidence fact", topic="test", confidence=0.9, importance=6)
    svc.save_fact("Lower confidence hint", topic="test", confidence=0.3, importance=6)
    out = svc.search_memories("test", limit=6)
    labels = {item.confidence_label for item in out}
    assert "verified" in labels
    assert "hint" in labels


def test_recent_memories_returns_newest_first(tmp_path):
    svc = _service(tmp_path)
    older = svc.save_fact("older fact", topic="order", confidence=0.7)
    newer = svc.save_fact("newer fact", topic="order", confidence=0.7)
    recent = svc.get_recent_memories(limit=2)
    assert recent
    assert recent[0].id == newer.id
    assert recent[1].id == older.id


def test_fts_fallback_works_when_fts_unavailable(tmp_path):
    svc = _service(tmp_path)
    svc.save_fact("fallback keyword memory", topic="fallback", confidence=0.6)
    svc._fts_enabled = False
    out = svc.search_memories("keyword fallback", limit=3)
    assert out
    assert any("fallback" in item.record.topic.lower() for item in out)
