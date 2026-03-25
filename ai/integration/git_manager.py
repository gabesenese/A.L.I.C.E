"""Safe repository operations wrapper for ALICE runtime."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class GitCommandResult:
    success: bool
    output: str
    error: str
    exit_code: int


class GitManager:
    def __init__(self, repo_root: Optional[str] = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else None

    def _run(self, args: List[str], timeout: int = 15) -> GitCommandResult:
        try:
            proc = subprocess.run(
                args,
                cwd=str(self.repo_root) if self.repo_root else None,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return GitCommandResult(
                success=proc.returncode == 0,
                output=(proc.stdout or "").strip(),
                error=(proc.stderr or "").strip(),
                exit_code=int(proc.returncode),
            )
        except Exception as exc:
            return GitCommandResult(False, "", str(exc), 1)

    def resolve_repo_root(self) -> GitCommandResult:
        res = self._run(["git", "rev-parse", "--show-toplevel"])
        if res.success and res.output:
            self.repo_root = Path(res.output).resolve()
        return res

    def status_short(self) -> GitCommandResult:
        return self._run(["git", "status", "--short"])

    def current_branch(self) -> GitCommandResult:
        return self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def diff_unstaged(self) -> GitCommandResult:
        return self._run(["git", "diff", "--", "."])

    def recent_commits(self, limit: int = 5) -> GitCommandResult:
        lim = max(1, min(int(limit or 5), 20))
        return self._run(["git", "log", f"-{lim}", "--pretty=format:%h %ad %s", "--date=short"])

    def create_checkpoint(self, label: str) -> GitCommandResult:
        name = (label or "alice-checkpoint").strip()[:80]
        return self._run(["git", "stash", "push", "-u", "-m", name], timeout=45)

    def rollback_from_checkpoint(self, checkpoint_ref: str = "stash@{0}") -> GitCommandResult:
        ref = (checkpoint_ref or "stash@{0}").strip()
        return self._run(["git", "stash", "apply", ref], timeout=45)

    def drop_checkpoint(self, checkpoint_ref: str = "stash@{0}") -> GitCommandResult:
        ref = (checkpoint_ref or "stash@{0}").strip()
        return self._run(["git", "stash", "drop", ref], timeout=30)

    def stage_all(self) -> GitCommandResult:
        return self._run(["git", "add", "--all"], timeout=30)

    def commit(self, message: str) -> GitCommandResult:
        msg = (message or "operator commit").strip()
        return self._run(["git", "commit", "-m", msg], timeout=45)

    def has_changes(self) -> GitCommandResult:
        return self._run(["git", "status", "--porcelain"])


_git_manager: Optional[GitManager] = None


def get_git_manager(repo_root: Optional[str] = None) -> GitManager:
    global _git_manager
    if _git_manager is None:
        _git_manager = GitManager(repo_root=repo_root)
    elif repo_root:
        _git_manager.repo_root = Path(repo_root).resolve()
    return _git_manager
