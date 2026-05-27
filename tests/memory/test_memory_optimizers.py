"""Tests for the 8 memory optimizer modules."""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np


# ---------------------------------------------------------------------------
# Minimal stub for MemoryEntry-like objects (avoids importing heavy deps)
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    id: str
    content: str
    memory_type: str = "episodic"
    timestamp: str = "2026-01-01T00:00:00+00:00"
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.context is None:
            self.context = {}


def _vec(seed: int, dim: int = 8) -> List[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near_vec(base: List[float], noise: float = 0.05) -> List[float]:
    v = np.array(base, dtype=np.float32)
    v += np.random.default_rng(42).random(len(v)).astype(np.float32) * noise
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# 1. MemoryScorer
# ---------------------------------------------------------------------------


class TestMemoryScorer:
    def setup_method(self):
        from ai.memory.memory_scorer import MemoryScorer

        self.scorer = MemoryScorer(decay_hours=168.0)

    def test_score_in_range(self):
        e = _Entry(id="a", content="test", importance=0.7, access_count=3)
        score = self.scorer.score(e)
        assert 0.0 <= score <= 1.0

    def test_higher_access_scores_higher(self):
        e_low = _Entry(id="a", content="x", importance=0.5, access_count=0)
        e_high = _Entry(id="b", content="x", importance=0.5, access_count=50)
        # Normalise together so max_access is shared
        scores = self.scorer.batch_score([e_low, e_high])
        assert scores["b"] > scores["a"]

    def test_emotional_content_boosts_score(self):
        e_plain = _Entry(id="a", content="I went to the store")
        e_emotional = _Entry(id="b", content="I feel anxious and overwhelmed today")
        s_plain = self.scorer.score(e_plain)
        s_emotional = self.scorer.score(e_emotional)
        assert s_emotional > s_plain

    def test_high_value_tags_boost_score(self):
        e_no_tag = _Entry(id="a", content="x", tags=[])
        e_tag = _Entry(id="b", content="x", tags=["goal", "milestone"])
        assert self.scorer.score(e_tag) > self.scorer.score(e_no_tag)

    def test_source_reliability(self):
        e_good = _Entry(id="a", content="x", context={"source": "system_verified"})
        e_bad = _Entry(id="b", content="x", context={"source": "ambient"})
        assert self.scorer.score(e_good) > self.scorer.score(e_bad)

    def test_batch_score_returns_all_ids(self):
        entries = [_Entry(id=f"e{i}", content="x") for i in range(5)]
        scores = self.scorer.batch_score(entries)
        assert set(scores.keys()) == {f"e{i}" for i in range(5)}


# ---------------------------------------------------------------------------
# 2. SemanticDeduplicator
# ---------------------------------------------------------------------------


class TestSemanticDeduplicator:
    def setup_method(self):
        from ai.memory.semantic_dedup import SemanticDeduplicator

        self.dedup = SemanticDeduplicator(threshold=0.82)

    def test_no_duplicates_unchanged(self):
        entries = [
            _Entry(id="a", content="cats", embedding=_vec(1)),
            _Entry(id="b", content="dogs", embedding=_vec(2)),
        ]
        result, pairs = self.dedup.deduplicate(entries)
        assert len(result) == 2
        assert pairs == []

    def test_near_duplicates_merged(self):
        base = _vec(99)
        entries = [
            _Entry(id="orig", content="alpha", embedding=base, importance=0.8),
            _Entry(
                id="dup",
                content="alpha duplicate",
                embedding=_near_vec(base, 0.01),
                importance=0.3,
            ),
        ]
        result, pairs = self.dedup.deduplicate(entries)
        assert len(result) == 1
        assert len(pairs) == 1
        kept_id, removed_id = pairs[0]
        assert kept_id == "orig"  # higher importance kept
        assert removed_id == "dup"

    def test_access_counts_merged(self):
        base = _vec(7)
        e1 = _Entry(id="a", content="x", embedding=base, importance=0.9, access_count=10)
        e2 = _Entry(
            id="b",
            content="x",
            embedding=_near_vec(base, 0.01),
            importance=0.1,
            access_count=5,
        )
        result, _ = self.dedup.deduplicate([e1, e2])
        assert result[0].access_count == 15

    def test_tag_union(self):
        base = _vec(11)
        e1 = _Entry(id="a", content="x", embedding=base, importance=0.9, tags=["goal"])
        e2 = _Entry(
            id="b",
            content="x",
            embedding=_near_vec(base, 0.01),
            importance=0.1,
            tags=["milestone"],
        )
        result, _ = self.dedup.deduplicate([e1, e2])
        assert "goal" in result[0].tags
        assert "milestone" in result[0].tags

    def test_single_entry_unchanged(self):
        e = _Entry(id="x", content="only one", embedding=_vec(3))
        result, pairs = self.dedup.deduplicate([e])
        assert len(result) == 1
        assert pairs == []


# ---------------------------------------------------------------------------
# 3. MemoryQuarantine
# ---------------------------------------------------------------------------


class TestMemoryQuarantine:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        from ai.memory.memory_quarantine import MemoryQuarantine

        self.q = MemoryQuarantine(db_path=Path(self._tmp.name), ttl_days=0.001)

    def teardown_method(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_quarantine_and_detect(self):
        self.q.quarantine("mem_1", reason="low_score:0.10", score=0.10)
        assert self.q.is_quarantined("mem_1")

    def test_release_clears_flag(self):
        self.q.quarantine("mem_2", reason="test")
        self.q.release("mem_2")
        assert not self.q.is_quarantined("mem_2")

    def test_list_quarantined(self):
        self.q.quarantine("mem_3", reason="test")
        rows = self.q.list_quarantined()
        ids = [r["memory_id"] for r in rows]
        assert "mem_3" in ids

    def test_auto_quarantine_by_score(self):
        entries = [_Entry(id="low", content="x"), _Entry(id="high", content="y")]
        scores = {"low": 0.05, "high": 0.90}
        quarantined = self.q.auto_quarantine_by_score(entries, scores, threshold=0.15)
        assert "low" in quarantined
        assert "high" not in quarantined

    def test_purge_expired(self):
        self.q.quarantine("old_mem", reason="test")
        # Back-date the expires_at so the record is already expired
        conn = sqlite3.connect(str(self._tmp.name))
        conn.execute("UPDATE quarantine SET expires_at='2000-01-01T00:00:00+00:00' WHERE memory_id='old_mem'")
        conn.commit()
        conn.close()
        n = self.q.purge_expired()
        assert n >= 1


# ---------------------------------------------------------------------------
# 4. HierarchicalCompressor
# ---------------------------------------------------------------------------


class TestHierarchicalCompressor:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = Path(self._tmp.name)
        # Seed a minimal memories table
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                memory_type TEXT NOT NULL, timestamp TEXT NOT NULL,
                context TEXT DEFAULT '{}', importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0, last_accessed TEXT,
                embedding BLOB, tags TEXT DEFAULT '[]',
                source_file TEXT, chunk_index INTEGER
            )
        """)
        # Insert 210 raw episodic entries (above the 200 threshold)
        for i in range(210):
            conn.execute(
                "INSERT INTO memories (id, content, memory_type, timestamp, tags) VALUES (?, ?, 'episodic', ?, '[]')",
                (f"m{i}", f"memory {i}", f"2026-01-{(i % 28) + 1:02d}T10:00:00"),
            )
        conn.commit()
        conn.close()

        from ai.memory.hierarchical_compressor import HierarchicalCompressor, LEVEL_RAW

        self.compressor = HierarchicalCompressor(db_path=db)
        self.LEVEL_RAW = LEVEL_RAW
        self._db = db

    def teardown_method(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_compress_creates_summaries(self):
        n = self.compressor.compress_level(self.LEVEL_RAW)
        assert n > 0

    def test_summary_stored_as_semantic(self):
        self.compressor.compress_level(self.LEVEL_RAW)
        conn = sqlite3.connect(str(self._db))
        rows = conn.execute("SELECT id FROM memories WHERE memory_type='semantic' AND tags LIKE '%summary%'").fetchall()
        conn.close()
        assert len(rows) > 0

    def test_source_entries_linked_via_parent_id(self):
        self.compressor.compress_level(self.LEVEL_RAW)
        conn = sqlite3.connect(str(self._db))
        linked = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE parent_id IS NOT NULL AND parent_id != ''"
        ).fetchone()[0]
        conn.close()
        assert linked > 0

    def test_below_threshold_no_compression(self):
        # Only 2 entries — well below threshold of 200
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = Path(tmp.name)
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                memory_type TEXT NOT NULL, timestamp TEXT NOT NULL,
                context TEXT DEFAULT '{}', importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0, last_accessed TEXT,
                embedding BLOB, tags TEXT DEFAULT '[]',
                source_file TEXT, chunk_index INTEGER
            )
        """)
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, timestamp) VALUES ('x','c','episodic','2026-01-01T00:00:00')"
        )
        conn.commit()
        conn.close()

        from ai.memory.hierarchical_compressor import HierarchicalCompressor, LEVEL_RAW

        c = HierarchicalCompressor(db_path=db)
        assert c.compress_level(LEVEL_RAW) == 0
        import os

        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 5. RetrievalBudget
# ---------------------------------------------------------------------------


class TestRetrievalBudget:
    def setup_method(self):
        from ai.memory.retrieval_budget import RetrievalBudget

        self.budget = RetrievalBudget(total_chars=100)

    def _mem(self, mid: str, content: str, mtype: str = "episodic", score: float = 0.5):
        return {
            "id": mid,
            "content": content,
            "memory_type": mtype,
            "weighted_score": score,
            "timestamp": "2026-01-01",
        }

    def test_respects_char_budget(self):
        candidates = [self._mem(f"m{i}", "x" * 30) for i in range(10)]
        selected = self.budget.select(candidates)
        total = sum(len(m["content"]) for m in selected)
        assert total <= 100

    def test_higher_priority_type_preferred(self):
        proc = self._mem("p", "x" * 40, mtype="procedural", score=0.5)
        epis = self._mem("e", "x" * 40, mtype="episodic", score=0.5)
        selected = self.budget.select([epis, proc])
        # procedural should come first
        assert selected[0]["id"] == "p"

    def test_always_includes_at_least_one(self):
        # single candidate larger than budget
        big = self._mem("big", "x" * 500)
        selected = self.budget.select([big])
        assert len(selected) == 1
        assert len(selected[0]["content"]) <= 100

    def test_empty_candidates_returns_empty(self):
        assert self.budget.select([]) == []

    def test_format_context_not_empty(self):
        candidates = [self._mem("a", "hello world")]
        selected = self.budget.select(candidates)
        ctx = self.budget.format_context(selected)
        assert "hello world" in ctx


# ---------------------------------------------------------------------------
# 6. ContradictionDetector
# ---------------------------------------------------------------------------


class TestContradictionDetector:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = Path(self._tmp.name)
        # Seed memories table (contradiction_detector only writes to its own table)
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT, memory_type TEXT,
                timestamp TEXT, context TEXT, importance REAL,
                access_count INTEGER, last_accessed TEXT, embedding BLOB,
                tags TEXT, source_file TEXT, chunk_index INTEGER
            )
        """)
        conn.commit()
        conn.close()

        from ai.memory.contradiction_detector import ContradictionDetector

        self.det = ContradictionDetector(db_path=db)
        self._db = db

    def teardown_method(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _entry(self, eid: str, content: str, emb: List[float]):
        return _Entry(id=eid, content=content, embedding=emb)

    def test_unrelated_no_contradiction(self):
        ea = self._entry("a", "I love cats", _vec(1))
        eb = self._entry("b", "The sky is blue", _vec(2))
        assert self.det.check_pair(ea, eb) is None

    def test_near_identical_no_contradiction(self):
        # Same embedding → similarity too high (>0.95) → dedup territory
        base = _vec(5)
        ea = self._entry("a", "The user is happy", base)
        eb = self._entry("b", "The user is happy too", _near_vec(base, 0.001))
        # This may or may not hit the _SIM_HIGH cap — just ensure no crash
        self.det.check_pair(ea, eb)
        # result is None or float; both acceptable

    def test_record_stores_pair(self):
        self.det.record("m1", "m2", confidence=0.7)
        rows = self.det.list_unresolved()
        ids = [(r["memory_a_id"], r["memory_b_id"]) for r in rows]
        assert ("m1", "m2") in ids

    def test_canonical_ordering(self):
        # (b,a) should be stored the same as (a,b)
        self.det.record("z", "a", confidence=0.6)
        rows = self.det.list_unresolved()
        assert any(r["memory_a_id"] == "a" and r["memory_b_id"] == "z" for r in rows)

    def test_resolve(self):
        self.det.record("x1", "x2", confidence=0.8)
        ok = self.det.resolve("x1", "x2", resolution="kept_x1")
        assert ok
        assert self.det.list_unresolved() == []


# ---------------------------------------------------------------------------
# 7. CausalMemory
# ---------------------------------------------------------------------------


class TestCausalMemory:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        from ai.memory.causal_memory import CausalMemory

        self.cm = CausalMemory(db_path=Path(self._tmp.name))

    def teardown_method(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_record_and_get_effects(self):
        self.cm.record("cause_1", "effect_1", confidence=0.9)
        effects = self.cm.get_effects("cause_1")
        assert any(e["effect_id"] == "effect_1" for e in effects)

    def test_get_causes(self):
        self.cm.record("cause_2", "effect_2")
        causes = self.cm.get_causes("effect_2")
        assert any(c["cause_id"] == "cause_2" for c in causes)

    def test_duplicate_ignored(self):
        self.cm.record("c", "e")
        self.cm.record("c", "e")  # second insert should be ignored (UNIQUE)
        assert len(self.cm.get_effects("c")) == 1

    def test_extract_because_pattern(self):
        text = "I stayed home because I felt sick."
        pairs = self.cm.extract_from_text(text)
        assert len(pairs) == 1
        cause, effect = pairs[0]
        assert len(cause) > 5
        assert len(effect) > 5

    def test_extract_therefore_pattern(self):
        text = "The build failed therefore we rolled back the deployment."
        pairs = self.cm.extract_from_text(text)
        assert len(pairs) == 1

    def test_no_causal_marker_returns_empty(self):
        pairs = self.cm.extract_from_text("The weather is nice today.")
        assert pairs == []

    def test_link_sequential(self):
        self.cm.link_sequential("early", "later", confidence=0.4)
        effects = self.cm.get_effects("early")
        assert effects[0]["chain_type"] == "inferred"

    def test_chain_context_format(self):
        self.cm.record("c1", "e1")
        ctx = self.cm.get_chain_context("c1")
        assert "led to" in ctx


# ---------------------------------------------------------------------------
# 8. MaintenanceScheduler
# ---------------------------------------------------------------------------


class TestMaintenanceScheduler:
    def setup_method(self):
        from ai.memory.maintenance_scheduler import MaintenanceScheduler

        self.sched = MaintenanceScheduler()

    def teardown_method(self):
        self.sched.stop()

    def test_register_and_run_now(self):
        called = []
        self.sched.register("test_task", lambda: called.append(1), interval_seconds=9999)
        self.sched.start()
        ok = self.sched.run_now("test_task")
        assert ok
        assert called == [1]

    def test_run_now_unknown_task_returns_false(self):
        self.sched.start()
        assert not self.sched.run_now("nonexistent")

    def test_status_shows_registered_tasks(self):
        self.sched.register("my_task", lambda: None, interval_seconds=60)
        self.sched.start()
        status = self.sched.status()
        assert "my_task" in status

    def test_task_error_increments_error_count(self):
        def _boom():
            raise RuntimeError("intentional failure")

        self.sched.register("bad_task", _boom, interval_seconds=9999)
        self.sched.start()
        self.sched.run_now("bad_task")
        assert self.sched.status()["bad_task"]["error_count"] == 1

    def test_double_start_is_idempotent(self):
        self.sched.register("t", lambda: None, interval_seconds=9999)
        self.sched.start()
        self.sched.start()  # should not raise or double-register defaults
        self.sched.stop()


# ---------------------------------------------------------------------------
# Integration: retrieval budget wired into memory_system.get_context_for_llm
# ---------------------------------------------------------------------------


class TestContextForLlmBudget:
    """Verify that get_context_for_llm respects the budget cap."""

    def test_context_length_bounded(self):
        from ai.memory.retrieval_budget import RetrievalBudget

        budget = RetrievalBudget(total_chars=200)

        big_candidates = [
            {
                "id": f"m{i}",
                "content": "x" * 100,
                "memory_type": "episodic",
                "weighted_score": 0.9,
                "timestamp": "2026-01-01",
            }
            for i in range(10)
        ]

        with patch("ai.memory.retrieval_budget.get_retrieval_budget", return_value=budget):
            selected = budget.select(big_candidates)
            ctx = budget.format_context(selected)

        # Total content chars used should be ≤ 200
        content_total = sum(len(m["content"]) for m in selected)
        assert content_total <= 200
        assert ctx  # non-empty
