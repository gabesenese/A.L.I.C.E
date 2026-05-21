#!/usr/bin/env python3
"""
ALICE Memory Health Check

Usage: python scripts/check_memory_health.py

Verifies that every memory mutation survives a simulated process restart.
Uses a temp DB — never touches data/memory/alice.db.
Exit code 0 = all pass, 1 = any failure.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy startup logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import logging
logging.disable(logging.WARNING)


def _reset(db_path: str, data_dir: str):
    """Wire both singletons to the temp DB. Call this to simulate a restart."""
    import ai.memory.memory_store as sm
    import ai.memory.memory_system as sys_mod
    from ai.memory.memory_store import SQLiteMemoryStore
    from ai.memory.memory_system import MemorySystem

    sys_mod._memory_system = None
    sm._memory_store = None

    store = SQLiteMemoryStore(db_path=db_path)
    sm._memory_store = store                   # must be set BEFORE MemorySystem()
    system = MemorySystem(data_dir=data_dir)
    sys_mod._memory_system = system

    return store, system


def _check(label: str, ok: bool) -> bool:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status}  {label}")
    return ok


def run() -> int:
    tmp     = tempfile.mkdtemp(prefix="alice_health_")
    db_path = str(Path(tmp) / "health.db")
    mem_dir = str(Path(tmp) / "memory")
    os.makedirs(mem_dir, exist_ok=True)

    results = []

    try:
        print("\nALICE Memory Health Check")
        print("-" * 50)

        # ── 1. Write-through ──────────────────────────────────────────────
        store, system = _reset(db_path, mem_dir)
        mid = system.store_memory(
            "Gabriel runs 5km daily",
            memory_type="episodic",
            importance=0.8,
        )
        store2, _ = _reset(db_path, mem_dir)
        row = store2.get_by_id(mid)
        results.append(_check(
            'write-through: "Gabriel runs 5km daily" survived restart',
            row is not None and "5km" in (row.content or ""),
        ))

        # ── 2. Forget ─────────────────────────────────────────────────────
        from ai.memory.personal_memory import PersonalMemoryStore
        store, system = _reset(db_path, mem_dir)
        pms = PersonalMemoryStore(system)
        fid = pms.store_structured_memory(
            "Gabriel prefers tea",
            domain="health", kind="preference", scope="personal",
            confidence=0.9, source="health_check",
        )
        pms.forget_recent_memory(domain="health", kind="preference")
        store2, _ = _reset(db_path, mem_dir)
        row = store2.get_by_id(fid)
        results.append(_check(
            "forget: invalid=True flag survived restart",
            row is not None and bool((row.context or {}).get("invalid")),
        ))

        # ── 3. Update ─────────────────────────────────────────────────────
        store, system = _reset(db_path, mem_dir)
        pms = PersonalMemoryStore(system)
        uid = pms.store_structured_memory(
            "Gabriel wakes at 7am",
            domain="routine", kind="habit", scope="personal",
            confidence=0.7, source="health_check",
        )
        pms.update_memory(uid, "Gabriel wakes at 6am", confidence=0.95)
        store2, _ = _reset(db_path, mem_dir)
        row = store2.get_by_id(uid)
        results.append(_check(
            'update: corrected content "Gabriel wakes at 6am" survived restart',
            row is not None
            and "6am" in (row.content or "")
            and bool((row.context or {}).get("corrected")),
        ))

        # ── 4. Consolidation ──────────────────────────────────────────────
        store, system = _reset(db_path, mem_dir)
        pms = PersonalMemoryStore(system)
        pms.store_structured_memory(
            "Gabriel runs 5km every morning",
            domain="fitness", kind="habit", scope="personal",
            confidence=0.7, source="health_check",
        )
        pms.store_structured_memory(
            "Gabriel runs 5km every morning routinely",
            domain="fitness", kind="habit", scope="personal",
            confidence=0.8, source="health_check",
        )
        store2, _ = _reset(db_path, mem_dir)
        superseded = sum(
            1 for r in store2.get_all("episodic")
            if (r.context or {}).get("superseded") is True
        )
        results.append(_check(
            f"consolidate: superseded flag persisted ({superseded} entries marked)",
            superseded >= 1,
        ))

        # ── 5. Remove ─────────────────────────────────────────────────────
        store, system = _reset(db_path, mem_dir)
        rmid = system.store_memory("Temporary delete me", memory_type="episodic")
        system._remove_memory_by_id(rmid)
        store2, _ = _reset(db_path, mem_dir)
        results.append(_check(
            "remove: hard-deleted memory absent after restart",
            store2.get_by_id(rmid) is None,
        ))

        # ── 6. Access count ───────────────────────────────────────────────
        store, system = _reset(db_path, mem_dir)
        acid = system.store_memory(
            "Gabriel enjoys hiking on weekends",
            memory_type="episodic",
            importance=0.9,
        )
        system.recall_memory("hiking", top_k=5, min_similarity=0.0)
        system.recall_memory("hiking", top_k=5, min_similarity=0.0)
        store2, _ = _reset(db_path, mem_dir)
        row = store2.get_by_id(acid)
        cnt = row.access_count if row else 0
        results.append(_check(
            f"access_count: {cnt} recall(s) accumulated in SQLite (expected >=1)",
            cnt >= 1,
        ))

    except Exception as exc:
        print(f"\n  [ERROR] Unexpected failure: {exc}")
        import traceback; traceback.print_exc()
        results.append(False)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 50)
    passed = sum(bool(r) for r in results)
    total  = len(results)
    if passed == total:
        print(f"  All checks passed: {passed}/{total}\n")
    else:
        print(f"  FAILURES DETECTED: {passed}/{total} passed\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
