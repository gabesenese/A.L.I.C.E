"""Holds the action Alice asked permission for, so a later "yes" can execute it.

The approval ledger records that a decision was made, but not what to do about it:
it stores an action label, a scope, and a summary, not the tool call itself. Without
somewhere to keep the call, asking for confirmation was a dead end, because nothing
survived the turn to be run when the user agreed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ai.infrastructure.paths import project_root

DEFAULT_TTL_SECONDS = 900


def _store_path() -> Path:
    return project_root() / "data" / "security" / "pending_actions.json"


@dataclass
class PendingAction:
    user_id: str
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    approval_id: str = ""
    reason: str = ""
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    def is_expired(self, now: Optional[float] = None) -> bool:
        return float(self.expires_at or 0.0) <= float(now if now is not None else time.time())

    def describe(self) -> str:
        if self.tool == "run_command":
            return f"run `{self.arguments.get('command', '')}`"
        path = self.arguments.get("path")
        if path:
            return f"{self.tool.replace('_', ' ')} on {path}"
        return self.tool.replace("_", " ")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_all() -> Dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_all(payload: Dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def record(
    *,
    user_id: str,
    tool: str,
    arguments: Dict[str, Any],
    approval_id: str = "",
    reason: str = "",
    summary: str = "",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> PendingAction:
    now = time.time()
    action = PendingAction(
        user_id=str(user_id or "default"),
        tool=str(tool or ""),
        arguments=dict(arguments or {}),
        approval_id=str(approval_id or ""),
        reason=str(reason or ""),
        summary=str(summary or ""),
        created_at=now,
        expires_at=now + float(max(30, int(ttl_seconds))),
    )
    payload = _read_all()
    payload[action.user_id] = action.to_dict()
    _write_all(payload)
    return action


def get(user_id: str = "default") -> Optional[PendingAction]:
    raw = _read_all().get(str(user_id or "default"))
    if not isinstance(raw, dict):
        return None
    try:
        action = PendingAction(**raw)
    except TypeError:
        return None
    if action.is_expired():
        clear(action.user_id)
        return None
    return action


def clear(user_id: str = "default") -> None:
    payload = _read_all()
    if str(user_id or "default") in payload:
        payload.pop(str(user_id or "default"), None)
        _write_all(payload)
