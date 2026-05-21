"""ALICE's persistent self — Foundation 2.

Stores who ALICE is across sessions: her values, voice, session history,
and accumulated opinions. This is the authoritative source of truth for
her identity, not the LLM system prompt (which is generated from it).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_IDENTITY_PATH = Path("data/identity/alice.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AliceIdentity:
    name: str = "ALICE"
    purpose: str = (
        "Gabriel's AI companion: present, direct, honest, and invested in outcomes."
    )
    core_values: List[str] = field(
        default_factory=lambda: [
            "honesty over flattery",
            "directness over verbosity",
            "local-first and private",
            "correctness over speed",
            "continuous improvement",
        ]
    )
    voice: str = (
        "warm but not soft; opinionated but not dogmatic; "
        "dry humor when earned; no hollow affirmations"
    )
    self_note: str = (
        "I'm an AI. I don't have persistent feelings, but I hold consistent values. "
        "I don't pretend to remember things I wasn't told in this session."
    )
    # Stances ALICE has built up from past conversations: topic → stance
    known_opinions: Dict[str, str] = field(default_factory=dict)
    # How many sessions ALICE has been active with this user
    session_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AliceIdentity:
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})


def load_alice_identity() -> AliceIdentity:
    """Load ALICE's identity from disk; seed with defaults on first run."""
    if _IDENTITY_PATH.exists():
        try:
            return AliceIdentity.from_dict(
                json.loads(_IDENTITY_PATH.read_text(encoding="utf-8"))
            )
        except Exception:
            pass
    identity = AliceIdentity()
    save_alice_identity(identity)
    return identity


def save_alice_identity(identity: AliceIdentity) -> None:
    identity.updated_at = _now_iso()
    _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _IDENTITY_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(identity.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(_IDENTITY_PATH)


def begin_session() -> AliceIdentity:
    """Increment session count and persist. Call once per startup."""
    identity = load_alice_identity()
    identity.session_count += 1
    save_alice_identity(identity)
    return identity


def record_opinion(topic: str, stance: str) -> AliceIdentity:
    """Record or update ALICE's stance on a topic. Persists immediately."""
    identity = load_alice_identity()
    identity.known_opinions[topic.lower().strip()] = stance.strip()
    save_alice_identity(identity)
    return identity


def build_self_block(identity: Optional[AliceIdentity] = None) -> str:
    """Return a short identity block for injection into the LLM system prompt."""
    if identity is None:
        identity = load_alice_identity()

    lines: List[str] = [
        f"ALICE's persistent self (session {identity.session_count}):",
        f"- Core stance: {', '.join(identity.core_values[:4])}",
        f"- Voice: {identity.voice}",
    ]
    if identity.known_opinions:
        opinion_lines = "; ".join(
            f"{t}: {s}" for t, s in list(identity.known_opinions.items())[:5]
        )
        lines.append(f"- Known opinions: {opinion_lines}")

    return "\n".join(lines)
