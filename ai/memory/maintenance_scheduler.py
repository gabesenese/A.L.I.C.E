from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class _Task:
    __slots__ = ("name", "fn", "interval", "last_run", "run_count", "error_count")

    def __init__(self, name: str, fn: Callable, interval_seconds: float) -> None:
        self.name = name
        self.fn = fn
        self.interval = interval_seconds
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0


class MaintenanceScheduler:
    """
    Background thread that runs memory maintenance tasks on fixed intervals.

    Default schedule:
      score_refresh          30 min  — rescore all memories, persist importance
      semantic_dedup          2 h    — deduplicate episodic + semantic lists
      quarantine_purge        6 h    — expire and delete stale quarantine entries
      contradiction_scan      1 h    — scan for new contradictions
      hierarchical_compress   4 h    — compress raw episodic into summaries

    All tasks are registered at start(); custom tasks can be added via register()
    before or after start(). Tasks that fail are logged and retried next cycle.
    """

    _DEFAULT_INTERVALS: Dict[str, float] = {
        "score_refresh":        1_800,   # 30 min
        "semantic_dedup":       7_200,   # 2 h
        "quarantine_purge":    21_600,   # 6 h
        "contradiction_scan":   3_600,   # 1 h
        "hierarchical_compress": 14_400, # 4 h
    }

    _TICK = 60  # seconds between due-task checks

    def __init__(self) -> None:
        self._tasks: Dict[str, _Task] = {}
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._register_defaults()
        self._schedule_tick()
        logger.info("[MaintenanceScheduler] Started (%d tasks)", len(self._tasks))

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        logger.info("[MaintenanceScheduler] Stopped")

    # ------------------------------------------------------------------
    # Registration / manual trigger
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Callable, interval_seconds: float) -> None:
        with self._lock:
            self._tasks[name] = _Task(name, fn, interval_seconds)

    def run_now(self, name: str) -> bool:
        """Manually trigger a named task immediately. Returns True if found."""
        with self._lock:
            task = self._tasks.get(name)
        if not task:
            return False
        self._run_task(task)
        return True

    def status(self) -> Dict[str, Dict]:
        with self._lock:
            return {
                name: {
                    "last_run":        t.last_run.isoformat() if t.last_run else None,
                    "run_count":       t.run_count,
                    "error_count":     t.error_count,
                    "interval_seconds": t.interval,
                }
                for name, t in self._tasks.items()
            }

    # ------------------------------------------------------------------
    # Internal tick loop
    # ------------------------------------------------------------------

    def _schedule_tick(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._TICK, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            due = [
                t for t in self._tasks.values()
                if t.last_run is None
                or (now - t.last_run).total_seconds() >= t.interval
            ]
        for task in due:
            self._run_task(task)
        self._schedule_tick()

    def _run_task(self, task: _Task) -> None:
        try:
            task.fn()
            with self._lock:
                task.last_run = datetime.now(timezone.utc)
                task.run_count += 1
            logger.debug(
                "[MaintenanceScheduler] %s completed (run #%d)",
                task.name, task.run_count,
            )
        except Exception as exc:
            with self._lock:
                task.error_count += 1
            logger.warning(
                "[MaintenanceScheduler] %s failed (error #%d): %s",
                task.name, task.error_count, exc,
            )

    # ------------------------------------------------------------------
    # Default task definitions
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        ivs = self._DEFAULT_INTERVALS

        def _score_refresh() -> None:
            from ai.memory.memory_system import get_memory_system
            from ai.memory.memory_scorer import get_memory_scorer
            from ai.memory.memory_store import get_memory_store

            ms = get_memory_system()
            scorer = get_memory_scorer()
            store = get_memory_store()
            all_entries = (
                ms.episodic_memory + ms.semantic_memory
                + ms.procedural_memory + ms.document_memory
            )
            scores = scorer.batch_score(all_entries)
            for entry in all_entries:
                eid = getattr(entry, "id", None)
                if eid and eid in scores:
                    new_score = round(scores[eid], 4)
                    entry.importance = new_score
                    store.update(eid, {"importance": new_score})
            logger.info("[score_refresh] Scored %d memories", len(all_entries))

        def _semantic_dedup() -> None:
            from ai.memory.memory_system import get_memory_system
            from ai.memory.semantic_dedup import get_deduplicator
            from ai.memory.memory_scorer import get_memory_scorer
            from ai.memory.memory_store import get_memory_store

            ms = get_memory_system()
            dedup = get_deduplicator()
            scorer = get_memory_scorer()
            store = get_memory_store()
            total_merged = 0
            for list_name in ("episodic_memory", "semantic_memory"):
                lst = getattr(ms, list_name)
                deduped, pairs = dedup.deduplicate(lst, scorer=scorer)
                setattr(ms, list_name, deduped)
                for _, removed_id in pairs:
                    store.remove(removed_id)
                    ms.vector_store.delete(removed_id)
                total_merged += len(pairs)
            if total_merged:
                logger.info("[semantic_dedup] Merged %d duplicates", total_merged)

        def _quarantine_purge() -> None:
            from ai.memory.memory_quarantine import get_quarantine
            from ai.memory.memory_system import get_memory_system

            n = get_quarantine().purge_expired()
            if n:
                # Reload in-memory lists from SQLite to reflect purge
                get_memory_system()._load_memories()
                logger.info("[quarantine_purge] Purged %d memories", n)

        def _contradiction_scan() -> None:
            from ai.memory.memory_system import get_memory_system
            from ai.memory.contradiction_detector import get_contradiction_detector

            ms = get_memory_system()
            detector = get_contradiction_detector()
            # Cap at 200 entries to bound O(n²) cost
            candidates = (ms.episodic_memory + ms.semantic_memory)[:200]
            found = detector.scan(candidates)
            if found:
                logger.info("[contradiction_scan] %d contradictions detected", len(found))

        def _hierarchical_compress() -> None:
            from ai.memory.hierarchical_compressor import get_compressor

            result = get_compressor().run_full_pass()
            if result:
                logger.info("[hierarchical_compress] %s", result)

        self.register("score_refresh",        _score_refresh,        ivs["score_refresh"])
        self.register("semantic_dedup",       _semantic_dedup,       ivs["semantic_dedup"])
        self.register("quarantine_purge",     _quarantine_purge,     ivs["quarantine_purge"])
        self.register("contradiction_scan",   _contradiction_scan,   ivs["contradiction_scan"])
        self.register("hierarchical_compress",_hierarchical_compress,ivs["hierarchical_compress"])


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[MaintenanceScheduler] = None
_scheduler_lock = threading.Lock()


def get_maintenance_scheduler() -> MaintenanceScheduler:
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = MaintenanceScheduler()
    return _scheduler


def start_maintenance_scheduler() -> MaintenanceScheduler:
    """Convenience helper: get + start in one call."""
    sched = get_maintenance_scheduler()
    sched.start()
    return sched
