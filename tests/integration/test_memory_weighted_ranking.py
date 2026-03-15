"""Tests for weighted memory ranking before context injection."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.memory.memory_system import MemorySystem, MemoryEntry


def _ts(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_weighted_memory_ranking_prefers_recent_confident_reliable(monkeypatch, tmp_path) -> None:
    mem = MemorySystem(data_dir=str(tmp_path / "memory"))

    entry_map = {
        "a": MemoryEntry(
            id="a",
            content="Old but semantically very close",
            memory_type="episodic",
            timestamp=_ts(24 * 40),
            context={"confidence": 0.45, "source": "conversation"},
            importance=0.45,
        ),
        "b": MemoryEntry(
            id="b",
            content="Recent reliable procedural memory",
            memory_type="procedural",
            timestamp=_ts(6),
            context={"confidence": 0.92, "source": "system_verified"},
            importance=0.9,
        ),
        "c": MemoryEntry(
            id="c",
            content="Medium memory",
            memory_type="semantic",
            timestamp=_ts(48),
            context={"confidence": 0.7, "source": "plugin_result"},
            importance=0.7,
        ),
    }

    base_results = [
        {"id": "a", "content": entry_map["a"].content, "similarity": 0.93, "importance": 0.45, "timestamp": entry_map["a"].timestamp, "type": "episodic", "access_count": 1, "tags": []},
        {"id": "b", "content": entry_map["b"].content, "similarity": 0.86, "importance": 0.9, "timestamp": entry_map["b"].timestamp, "type": "procedural", "access_count": 1, "tags": []},
        {"id": "c", "content": entry_map["c"].content, "similarity": 0.84, "importance": 0.7, "timestamp": entry_map["c"].timestamp, "type": "semantic", "access_count": 1, "tags": []},
    ]

    monkeypatch.setattr(mem, "recall_memory", lambda query, top_k=5, min_similarity=0.35: base_results)
    monkeypatch.setattr(mem, "_get_memory_by_id", lambda memory_id: entry_map[memory_id])

    ranked = mem.recall_memory_weighted("help me with setup", top_k=3)

    assert len(ranked) == 3
    assert ranked[0]["id"] == "b"
    assert ranked[0]["weighted_score"] >= ranked[1]["weighted_score"] >= ranked[2]["weighted_score"]


def test_get_context_for_llm_uses_weighted_top_memories(monkeypatch, tmp_path) -> None:
    mem = MemorySystem(data_dir=str(tmp_path / "memory"))

    weighted = [
        {
            "id": "m1",
            "content": "Most relevant memory",
            "timestamp": _ts(2),
            "weighted_score": 0.91,
        },
        {
            "id": "m2",
            "content": "Second memory",
            "timestamp": _ts(12),
            "weighted_score": 0.73,
        },
    ]

    monkeypatch.setattr(mem, "recall_memory_weighted", lambda query, top_k=3: weighted[:top_k])

    context = mem.get_context_for_llm("what did we decide", max_memories=2)

    assert "weighted" in context.lower()
    assert "Most relevant memory" in context
    assert "Second memory" in context
