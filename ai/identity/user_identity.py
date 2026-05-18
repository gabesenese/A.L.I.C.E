from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_IDENTITY_DIR = Path("data/identity")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UserIdentity:
    user_id: str = "gabriel"
    name: str = "Gabriel"
    # Persistent engineering values — fed into constraint context
    values: List[str] = field(default_factory=list)
    # Long-horizon goals that survive across sessions
    long_term_goals: List[str] = field(default_factory=list)
    # Learned response style preferences
    learned_preferences: Dict[str, str] = field(default_factory=dict)
    # One-line personality read for tone calibration
    personality_read: str = ""
    known_projects: List[str] = field(default_factory=list)
    # Recent emotional state signals with timestamps — drives session continuity
    emotional_history: List[Dict[str, str]] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserIdentity:
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})


def load_identity(user_id: str = "gabriel") -> UserIdentity:
    path = _IDENTITY_DIR / f"{user_id}.json"
    if path.exists():
        try:
            return UserIdentity.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return UserIdentity(user_id=user_id)


def save_identity(identity: UserIdentity) -> None:
    _IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    identity.updated_at = _now_iso()
    path = _IDENTITY_DIR / f"{identity.user_id}.json"
    path.write_text(
        json.dumps(identity.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def update_identity(updates: Dict[str, Any], user_id: str = "gabriel") -> UserIdentity:
    identity = load_identity(user_id)
    for k, v in updates.items():
        if not hasattr(identity, k):
            continue
        existing = getattr(identity, k)
        if isinstance(existing, list) and isinstance(v, list):
            merged = list(existing)
            for item in v:
                if item not in merged:
                    merged.append(item)
            setattr(identity, k, merged)
        elif isinstance(existing, dict) and isinstance(v, dict):
            existing.update(v)
            setattr(identity, k, existing)
        else:
            setattr(identity, k, v)
    save_identity(identity)
    return identity
