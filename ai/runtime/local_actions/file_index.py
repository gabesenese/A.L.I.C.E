from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List


class FileIndex:
    def __init__(self, alice: Any, root: Path, ttl_seconds: int = 45):
        self.alice = alice
        self.root = root
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._cached_files: List[str] = []
        self._cached_at: float = 0.0

    def invalidate(self) -> None:
        self._cached_files = []
        self._cached_at = 0.0

    def _safe_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def list_files(self, limit: int = 400) -> List[str]:
        now = time.time()
        if self._cached_files and (now - self._cached_at) <= self.ttl_seconds:
            return list(self._cached_files[:limit])

        refl = getattr(self.alice, "self_reflection", None)
        if refl and hasattr(refl, "list_codebase"):
            try:
                rows = list(refl.list_codebase() or [])
                out = []
                for row in rows[:limit]:
                    path = row.get("path") if isinstance(row, dict) else str(row)
                    if path:
                        out.append(str(path).replace("\\", "/"))
                if out:
                    self._cached_files = out
                    self._cached_at = now
                    return out
            except Exception:
                pass

        out: List[str] = []
        for dirpath, _, filenames in os.walk(self.root):
            if any(skip in dirpath.lower() for skip in (".git", ".venv", "__pycache__")):
                continue
            for name in filenames:
                if name.endswith((".py", ".md", ".json", ".yaml", ".yml")):
                    out.append(self._safe_rel(Path(dirpath, name)))
                    if len(out) >= limit:
                        self._cached_files = out
                        self._cached_at = now
                        return out
        self._cached_files = out
        self._cached_at = now
        return out

