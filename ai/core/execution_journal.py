"""Execution journal for auditing action attempts and outcomes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class ExecutionJournal:
    def __init__(self, storage_path: str = "data/action_journal.jsonl") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict[str, Any]) -> None:
        line = dict(entry)
        line["timestamp"] = line.get("timestamp") or time.time()
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=True) + "\n")
        except Exception:
            return

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []

        return rows[-max(1, limit) :]

    def summary(self) -> Dict[str, Any]:
        rows = self.recent(limit=200)
        if not rows:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "retry": 0,
                "goal_satisfied": 0,
            }

        total = len(rows)
        success = sum(1 for r in rows if r.get("success") is True)
        failed = sum(1 for r in rows if r.get("success") is False)
        retry = sum(1 for r in rows if r.get("event") == "retry")
        goal_satisfied = sum(1 for r in rows if r.get("goal_satisfied") is True)
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "retry": retry,
            "goal_satisfied": goal_satisfied,
        }


_execution_journal: ExecutionJournal | None = None


def get_execution_journal(
    storage_path: str = "data/action_journal.jsonl",
) -> ExecutionJournal:
    global _execution_journal
    if _execution_journal is None:
        _execution_journal = ExecutionJournal(storage_path=storage_path)
    return _execution_journal
