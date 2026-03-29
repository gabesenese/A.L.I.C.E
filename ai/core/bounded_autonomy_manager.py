"""Bounded autonomy loop registry with permission and audit controls."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AutonomyLoop:
    name: str
    permission_level: str
    scope: str
    stop_conditions: List[str]
    confidence_threshold: float
    enabled: bool = False


class BoundedAutonomyManager:
    def __init__(self, storage_path: str = "data/autonomy_loops.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.loops: Dict[str, AutonomyLoop] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("loops", []):
                loop = AutonomyLoop(**row)
                self.loops[loop.name] = loop
            self.audit_log = data.get("audit_log", [])[-200:]
        except Exception:
            return

    def _save(self) -> None:
        try:
            payload = {
                "loops": [asdict(loop) for loop in self.loops.values()],
                "audit_log": self.audit_log[-200:],
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return

    def register_loop(self, loop: AutonomyLoop) -> None:
        self.loops[loop.name] = loop
        self._audit("register", loop.name, {"scope": loop.scope, "permission": loop.permission_level})
        self._save()

    def set_enabled(self, name: str, enabled: bool, actor: str = "operator") -> bool:
        loop = self.loops.get(name)
        if not loop:
            return False
        loop.enabled = bool(enabled)
        self._audit("enable" if enabled else "disable", name, {"actor": actor})
        self._save()
        return True

    def _audit(self, event: str, name: str, data: Dict[str, Any]) -> None:
        self.audit_log.append({"event": event, "loop": name, "data": data, "at": time.time()})
        self.audit_log = self.audit_log[-200:]

    def status(self) -> Dict[str, Any]:
        enabled = [name for name, loop in self.loops.items() if loop.enabled]
        return {
            "total_loops": len(self.loops),
            "enabled_loops": enabled,
            "disabled_loops": [name for name in self.loops.keys() if name not in enabled],
            "recent_audit": self.audit_log[-10:],
        }


_bounded_autonomy_manager: BoundedAutonomyManager | None = None


def get_bounded_autonomy_manager(storage_path: str = "data/autonomy_loops.json") -> BoundedAutonomyManager:
    global _bounded_autonomy_manager
    if _bounded_autonomy_manager is None:
        _bounded_autonomy_manager = BoundedAutonomyManager(storage_path=storage_path)
    return _bounded_autonomy_manager
