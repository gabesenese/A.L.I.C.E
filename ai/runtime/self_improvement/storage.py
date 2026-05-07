from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ai.runtime.self_improvement.behavior_event import BehaviorEvent


def _data_dir() -> Path:
    root = os.getenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", "data")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(name: str) -> Path:
    return _data_dir() / name


def get_data_path(name: str) -> Path:
    return _path(name)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = _path(str(path))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path, limit: int = 100) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = _path(str(path))
    if not file_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-max(1, int(limit or 100)) :]


def record_behavior_event(event: BehaviorEvent) -> None:
    append_jsonl(_path("behavior_events.jsonl"), event.to_dict())


def record_hypothesis(hypothesis: Dict[str, Any]) -> None:
    append_jsonl(_path("improvement_hypotheses.jsonl"), dict(hypothesis or {}))


def record_patch_plan(plan: Dict[str, Any]) -> None:
    append_jsonl(_path("patch_plans.jsonl"), dict(plan or {}))


def record_evaluation_plan(plan: Dict[str, Any]) -> None:
    append_jsonl(_path("evaluation_plans.jsonl"), dict(plan or {}))


def record_audit_report(report: Dict[str, Any]) -> None:
    append_jsonl(_path("audit_reports.jsonl"), dict(report or {}))


def read_behavior_events(limit: int = 100) -> List[Dict[str, Any]]:
    return read_jsonl(_path("behavior_events.jsonl"), limit=limit)


def read_hypotheses(limit: int = 100) -> List[Dict[str, Any]]:
    return read_jsonl(_path("improvement_hypotheses.jsonl"), limit=limit)


def read_patch_plans(limit: int = 100) -> List[Dict[str, Any]]:
    return read_jsonl(_path("patch_plans.jsonl"), limit=limit)


def read_evaluation_plans(limit: int = 100) -> List[Dict[str, Any]]:
    return read_jsonl(_path("evaluation_plans.jsonl"), limit=limit)


def read_audit_reports(limit: int = 100) -> List[Dict[str, Any]]:
    return read_jsonl(_path("audit_reports.jsonl"), limit=limit)
