from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List


_CLAIM_PATTERNS = (
    r"\blast time we talked about\b",
    r"\bwe were discussing\b",
    r"\byou mentioned\b",
    r"\bwe left off on\b",
    r"\bi remember you were\b",
    r"\bour previous conversation was about\b",
    r"\bconversation history suggests\b",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ContinuityGuardResult:
    text: str
    detected_claims: List[str]
    supported_claims: List[str]
    unsupported_claims: List[str]
    evidence_sources: List[str]
    recovery_applied: bool
    unsupported_continuity_claim: bool

    def metadata(self) -> Dict[str, Any]:
        return {
            "detected_claims": list(self.detected_claims),
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "evidence_sources": list(self.evidence_sources),
            "recovery_applied": bool(self.recovery_applied),
            "unsupported_continuity_claim": bool(self.unsupported_continuity_claim),
        }


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _source_name(item: Dict[str, Any]) -> str:
    ctx = dict(item.get("context") or {})
    return str(ctx.get("source") or item.get("source") or "").strip().lower()


def _is_structured_memory(item: Dict[str, Any], *, min_confidence: float) -> bool:
    ctx = dict(item.get("context") or {})
    has_schema = all(str(ctx.get(k) or "").strip() for k in ("domain", "kind", "scope"))
    if not has_schema:
        return False
    confidence = float(ctx.get("confidence", item.get("importance", 0.0)) or 0.0)
    timestamp = str(ctx.get("timestamp") or item.get("timestamp") or "").strip()
    source = _source_name(item)
    return bool(
        confidence >= float(min_confidence)
        and timestamp
        and source
        and "vector" not in source
        and "embedding" not in source
    )


def _has_recent_session_evidence(items: List[Dict[str, Any]], *, now: datetime) -> bool:
    cutoff = now - timedelta(hours=6)
    for item in items:
        ctx = dict(item.get("context") or {})
        source = _source_name(item)
        turn_index = int(ctx.get("turn_index", -1) or -1)
        ts = _parse_iso(str(ctx.get("timestamp") or item.get("timestamp") or ""))
        if source in {"session", "session_recent", "conversation", "turn_memory"} and (
            turn_index >= 0 or (ts is not None and ts >= cutoff)
        ):
            return True
    return False


def _has_active_objective(operator_state: Dict[str, Any]) -> bool:
    state = dict(operator_state or {})
    return bool(str(state.get("active_objective") or "").strip() and str(state.get("current_focus") or "").strip())


def assess_continuity_claims(
    *,
    text: str,
    memory_items: List[Dict[str, Any]],
    operator_state: Dict[str, Any] | None,
    min_structured_confidence: float = 0.7,
) -> ContinuityGuardResult:
    content = str(text or "").strip()
    if not content:
        return ContinuityGuardResult("", [], [], [], [], False, False)

    sentences = _SENTENCE_SPLIT.split(content)
    detected: List[str] = []
    supported: List[str] = []
    unsupported: List[str] = []
    evidence_sources: List[str] = []
    now = datetime.now(timezone.utc)
    recent_session = _has_recent_session_evidence(list(memory_items or []), now=now)
    active_objective = _has_active_objective(dict(operator_state or {}))
    structured_supported = any(
        _is_structured_memory(row, min_confidence=min_structured_confidence)
        for row in list(memory_items or [])
    )

    for row in list(memory_items or []):
        source = _source_name(row)
        if source and source not in evidence_sources:
            evidence_sources.append(source)

    kept: List[str] = []
    for sentence in sentences:
        if _CLAIM_RE.search(sentence):
            claim = sentence.strip()
            detected.append(claim)
            if recent_session or active_objective or structured_supported:
                supported.append(claim)
                kept.append(sentence)
            else:
                unsupported.append(claim)
        else:
            kept.append(sentence)

    cleaned = " ".join(part.strip() for part in kept if part.strip()).strip()
    recovery_applied = bool(unsupported)
    if recovery_applied and not cleaned:
        cleaned = "I am here. No active task is loaded yet, and we can continue an existing project or start fresh."

    return ContinuityGuardResult(
        text=cleaned,
        detected_claims=detected,
        supported_claims=supported,
        unsupported_claims=unsupported,
        evidence_sources=evidence_sources,
        recovery_applied=recovery_applied,
        unsupported_continuity_claim=bool(unsupported),
    )

