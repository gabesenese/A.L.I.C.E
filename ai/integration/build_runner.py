"""Build and test command runner for operator workflows."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BuildRunResult:
    success: bool
    command: str
    output: str
    error: str
    exit_code: int


class BuildRunner:
    def __init__(self, project_root: Optional[str] = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None

    def _run(self, command: str, timeout: int = 120) -> BuildRunResult:
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.project_root) if self.project_root else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
            )
            return BuildRunResult(
                success=proc.returncode == 0,
                command=command,
                output=(proc.stdout or "").strip(),
                error=(proc.stderr or "").strip(),
                exit_code=int(proc.returncode),
            )
        except Exception as exc:
            return BuildRunResult(False, command, "", str(exc), 1)

    def run_python_tests(self) -> BuildRunResult:
        py = sys.executable
        return self._run(f'"{py}" -m pytest -q', timeout=240)

    def run_python_build(self) -> BuildRunResult:
        py = sys.executable
        return self._run(f'"{py}" -m compileall -q app ai', timeout=120)


_build_runner: Optional[BuildRunner] = None


def get_build_runner(project_root: Optional[str] = None) -> BuildRunner:
    global _build_runner
    if _build_runner is None:
        _build_runner = BuildRunner(project_root=project_root)
    elif project_root:
        _build_runner.project_root = Path(project_root).resolve()
    return _build_runner
