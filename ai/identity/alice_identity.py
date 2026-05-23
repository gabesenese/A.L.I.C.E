"""ALICE's persistent self — Foundation 2.

Static identity (name, values, voice) lives in data/identity/alice.json.
Growing collections (opinions, sessions) live in SQLite via IdentityStore.
This module owns the lifecycle: begin_session, record_opinion, end_session,
and build_self_block which assembles a live LLM context injection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

_IDENTITY_PATH = Path("data/identity/alice.json")

# Module-level session tracking (in-memory, flushed on end_session)
_current_session_id: str = ""
_session_buffer: Dict[str, Dict[str, Any]] = {}
_confidence_samples: List[float] = []
_CONFIDENCE_WINDOW = 50


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


# ------------------------------------------------------------------ session lifecycle


def begin_session() -> str:
    """Start a new session: increment count, insert SQLite row, return session_id."""
    global _current_session_id

    identity = load_alice_identity()
    identity.session_count += 1
    save_alice_identity(identity)

    session_id = str(uuid4())
    _current_session_id = session_id

    try:
        from ai.identity.identity_store import get_identity_store

        get_identity_store().begin_session(session_id, _now_iso())
    except Exception:
        pass

    return session_id


def record_turn_outcome(
    session_id: str,
    success_score: float,
    intent: str,
    failure_kind: str,
) -> None:
    """Accumulate per-turn data in-memory; flushed at end_session."""
    buf = _session_buffer.setdefault(
        session_id,
        {
            "turn_count": 0,
            "score_sum": 0.0,
            "failure_kinds": {},
            "intent_mix": {},
            "goals_touched": set(),
        },
    )
    buf["turn_count"] += 1
    buf["score_sum"] += float(success_score)
    if failure_kind:
        buf["failure_kinds"][failure_kind] = (
            buf["failure_kinds"].get(failure_kind, 0) + 1
        )
    if intent:
        short = str(intent).split(":")[0]
        buf["intent_mix"][short] = buf["intent_mix"].get(short, 0) + 1


def record_confidence_sample(score: float) -> None:
    """Record a per-turn success score for rolling confidence calculation."""
    global _confidence_samples
    _confidence_samples.append(float(score))
    if len(_confidence_samples) > _CONFIDENCE_WINDOW:
        _confidence_samples = _confidence_samples[-_CONFIDENCE_WINDOW:]


def get_rolling_confidence() -> float:
    if not _confidence_samples:
        return 0.5
    return sum(_confidence_samples) / len(_confidence_samples)


def end_session(session_id: str) -> None:
    """Flush session buffer to SQLite. Called at shutdown."""
    if not session_id:
        return
    buf = _session_buffer.pop(session_id, {})
    turn_count = int(buf.get("turn_count") or 0)
    score_sum = float(buf.get("score_sum") or 0.0)
    avg_score = score_sum / turn_count if turn_count > 0 else 0.5
    failure_kinds = dict(buf.get("failure_kinds") or {})
    intent_mix = dict(buf.get("intent_mix") or {})
    goals_touched = list(buf.get("goals_touched") or set())

    top_intents = sorted(intent_mix.items(), key=lambda x: -x[1])[:3]
    intent_str = (
        ", ".join(f"{k}({v})" for k, v in top_intents) if top_intents else "none"
    )
    goal_str = f"; goals: {len(goals_touched)}" if goals_touched else ""
    summary = f"{turn_count} turns; intents: {intent_str}{goal_str}; avg_score: {avg_score:.2f}"

    try:
        from ai.identity.identity_store import get_identity_store

        get_identity_store().end_session(
            session_id=session_id,
            ended_at=_now_iso(),
            summary=summary,
            turn_count=turn_count,
            avg_success_score=avg_score,
            failure_kinds=failure_kinds,
            intent_mix=intent_mix,
            goals_touched=goals_touched,
        )
    except Exception:
        pass


def record_goal_touched(session_id: str, goal_id: str) -> None:
    """Record that this session touched a goal (in-memory; flushed at end_session)."""
    if not session_id or not goal_id:
        return
    buf = _session_buffer.get(session_id)
    if buf is not None:
        buf.setdefault("goals_touched", set()).add(goal_id)


# ------------------------------------------------------------------ opinions


def record_opinion(topic: str, stance: str, source: str = "auto") -> None:
    """Record or update ALICE's stance on a topic. Persists to SQLite immediately."""
    try:
        from ai.identity.identity_store import get_identity_store

        get_identity_store().upsert_opinion(topic, stance, source)
    except Exception:
        pass


# ------------------------------------------------------------------ self-block


def build_self_block(
    identity: Optional[AliceIdentity] = None,
    include_opinions: bool = False,
    include_session: bool = False,
) -> str:
    """Assemble ALICE's identity block for LLM context injection.

    Called with no args from llm_engine (lightweight prefix).
    Called with include_opinions=True, include_session=True from
    boundary_factory for the richer per-turn self-awareness block.
    """
    if identity is None:
        identity = load_alice_identity()

    conf = get_rolling_confidence()
    conf_str = f", confidence {conf:.0%}" if _confidence_samples else ""
    lines: List[str] = [
        f"ALICE's persistent self (session {identity.session_count}{conf_str}):",
        f"- Core stance: {', '.join(identity.core_values[:4])}",
        f"- Voice: {identity.voice}",
    ]

    if include_session:
        try:
            from ai.identity.identity_store import get_identity_store

            last = get_identity_store().get_last_session()
            if last and last.get("ended_at"):
                ended = last["ended_at"]
                turns = int(last.get("turn_count") or 0)
                score = float(last.get("avg_success_score") or 0.5)
                delta = datetime.now(timezone.utc) - datetime.fromisoformat(
                    str(ended).replace("Z", "+00:00")
                )
                if delta.days == 0:
                    when = "earlier today"
                elif delta.days == 1:
                    when = "yesterday"
                else:
                    when = f"{delta.days} days ago"
                lines.append(
                    f"- Last session {when}: {turns} turns, avg score {score:.0%}"
                )
        except Exception:
            pass

    if include_opinions:
        try:
            from ai.identity.identity_store import get_identity_store

            opinions = get_identity_store().get_top_opinions(n=4)
            if opinions:
                parts = []
                for op in opinions[:3]:
                    topic = op["topic"]
                    stance = op["stance"][:80]
                    count = int(op.get("evidence_count") or 1)
                    marker = f" (x{count})" if count > 1 else ""
                    parts.append(f"{topic}: {stance}{marker}")
                lines.append("- Known opinions: " + "; ".join(parts))
        except Exception:
            pass

    return "\n".join(lines)
