from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LifecycleReport:
    started: List[str] = field(default_factory=list)
    stopped: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started": list(self.started),
            "stopped": list(self.stopped),
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }


class LifecycleManager:
    def __init__(self) -> None:
        self.report = LifecycleReport()

    def start_optional_systems(
        self, systems: Dict[str, Any], enabled: Dict[str, bool] | None = None
    ) -> LifecycleReport:
        flags = dict(enabled or {})
        for name, system in dict(systems or {}).items():
            if not bool(flags.get(name, True)):
                self.report.skipped.append(name)
                continue
            if not system:
                self.report.skipped.append(name)
                continue
            start = getattr(system, "start", None)
            if not callable(start):
                self.report.skipped.append(name)
                continue
            try:
                start()
                self.report.started.append(name)
            except Exception as exc:
                self.report.errors.append(f"start:{name}:{exc}")
        return self.report

    def stop_optional_systems(self, systems: Dict[str, Any]) -> LifecycleReport:
        for name, system in dict(systems or {}).items():
            if not system:
                self.report.skipped.append(name)
                continue
            stop = getattr(system, "stop", None)
            if not callable(stop):
                self.report.skipped.append(name)
                continue
            try:
                stop()
                self.report.stopped.append(name)
            except Exception as exc:
                self.report.errors.append(f"stop:{name}:{exc}")
        return self.report
