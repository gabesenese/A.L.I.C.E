from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_MAX_LINES = 1000


class RoutingFailureLogger:
    def __init__(self, path: str = "data/routing_failures.jsonl") -> None:
        self.path = Path(path)
        self._write_count = 0

    def append(self, payload: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            self._write_count += 1
            if self._write_count % 50 == 0:
                self._trim()
        except Exception:
            return

    def _trim(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if len(lines) > _MAX_LINES:
                self.path.write_text("\n".join(lines[-_MAX_LINES:]) + "\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def build_payload(**kwargs: Any) -> Dict[str, Any]:
        payload = dict(kwargs or {})
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        return payload
