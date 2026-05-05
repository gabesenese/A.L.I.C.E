from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class RoutingFailureLogger:
    def __init__(self, path: str = "data/routing_failures.jsonl") -> None:
        self.path = Path(path)

    def append(self, payload: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            return

    @staticmethod
    def build_payload(**kwargs: Any) -> Dict[str, Any]:
        payload = dict(kwargs or {})
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        return payload

