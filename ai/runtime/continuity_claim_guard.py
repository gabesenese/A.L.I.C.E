from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Set


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
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "we", "you", "i", "me", "my", "our", "your", "was", "were",
    "are", "is", "to", "of", "about", "last", "time", "talked", "discussing",
    "mentioned", "remember", "previous", "conversation", "history", "suggests",
    "left", "off", "on", "and", "or", "in", "with", "for",
    "today", "day",
}


@dataclass(frozen=True)
class ContinuityGuardResult:
    text: str
    detected_claims: List[str]
    supported_claims: List[str]
    unsupported_claims: List[str]
    evidence_sources: List[str]
    claim_topic_tokens: Dict[str, List[str]]
    evidence_topic_tokens: List[str]
    overlap_passed_by_claim: Dict[str, bool]
    support_reasons: Dict[str, List[str]]
    rejection_reasons: Dict[str, List[str]]
    recovery_applied: bool
    unsupported_continuity_claim: bool

    def metadata(self) -> Dict[str, Any]:
        return {
            "detected_claims": list(self.detected_claims),
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "evidence_sources": list(self.evidence_sources),
            "claim_topic_tokens": dict(self.claim_topic_tokens),
            "evidence_topic_tokens": list(self.evidence_topic_tokens),
            "overlap_required": True,
            "overlap_passed_by_claim": dict(self.overlap_passed_by_claim),
            "support_reasons": dict(self.support_reasons),
            "rejection_reasons": dict(self.rejection_reasons),
            "recovery_applied": bool(self.recovery_applied),
            "unsupported_continuity_claim": bool(self.unsupported_continuity_claim),
        }


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _source_name(item: Dict[str, Any]) -> str:
    ctx = dict(item.get("context") or {})
    return str(ctx.get("source") or item.get("source") or "").strip().lower()


def _tokens(text: str) -> set[str]:
    words = _WORD_RE.findall(str(text or "").lower())
    return {w for w in words if w and w not in _STOPWORDS}


def _claim_topic_tokens(claim: str) -> set[str]:
    cleaned = _CLAIM_RE.sub(" ", str(claim or "").lower())
    return _tokens(cleaned)


def _operator_state_topic_tokens(operator_state: Dict[str, Any]) -> set[str]:
    state = dict(operator_state or {})
    parts: List[str] = [
        str(state.get("active_objective") or ""),
        str(state.get("current_focus") or ""),
        str(state.get("current_step") or ""),
        str(state.get("next_recommended_action") or ""),
        str(state.get("last_inspected_file") or ""),
        str(state.get("last_success") or ""),
        str(state.get("last_failure") or ""),
    ]
    for key in ("current_plan", "files_inspected", "known_blockers"):
        value = state.get(key)
        if isinstance(value, list):
            parts.extend(str(v or "") for v in value)
    return _tokens(" ".join(parts))


def _memory_item_topic_text(item: Dict[str, Any]) -> str:
    ctx = dict(item.get("context") or {})
    tags = item.get("tags")
    tag_text = ""
    if isinstance(tags, list):
        tag_text = " ".join(str(t or "") for t in tags)
    return " ".join(
        [
            str(item.get("content") or ""),
            str(ctx.get("domain") or ""),
            str(ctx.get("kind") or ""),
            str(ctx.get("scope") or ""),
            tag_text,
        ]
    )


def _memory_topic_tokens(memory_items: List[Dict[str, Any]]) -> set[str]:
    corpus = " ".join(_memory_item_topic_text(i) for i in list(memory_items or []))
    return _tokens(corpus)


def _has_topic_overlap(claim_tokens: set[str], evidence_tokens: set[str]) -> bool:
    return bool(set(claim_tokens or set()).intersection(set(evidence_tokens or set())))


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


def _recent_session_items(items: List[Dict[str, Any]], *, now: datetime) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    cutoff = now - timedelta(hours=6)
    for item in list(items or []):
        ctx = dict(item.get("context") or {})
        source = _source_name(item)
        turn_index = int(ctx.get("turn_index", -1) or -1)
        ts = _parse_iso(str(ctx.get("timestamp") or item.get("timestamp") or ""))
        if source in {"session", "session_recent", "conversation", "turn_memory"} and (
            turn_index >= 0 or (ts is not None and ts >= cutoff)
        ):
            matched.append(item)
    return matched


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
        return ContinuityGuardResult("", [], [], [], [], {}, [], {}, {}, {}, False, False)

    sentences = _SENTENCE_SPLIT.split(content)
    detected: List[str] = []
    supported: List[str] = []
    unsupported: List[str] = []
    evidence_sources: List[str] = []
    claim_topic_tokens_map: Dict[str, List[str]] = {}
    overlap_passed_by_claim: Dict[str, bool] = {}
    support_reasons: Dict[str, List[str]] = {}
    rejection_reasons: Dict[str, List[str]] = {}
    now = datetime.now(timezone.utc)
    all_items = list(memory_items or [])
    recent_items = _recent_session_items(all_items, now=now)
    structured_items = [
        row
        for row in all_items
        if _is_structured_memory(row, min_confidence=min_structured_confidence)
    ]
    state_tokens = _operator_state_topic_tokens(dict(operator_state or {}))
    operator_active = _has_active_objective(dict(operator_state or {}))

    for row in all_items:
        source = _source_name(row)
        if source and source not in evidence_sources:
            evidence_sources.append(source)
    if operator_active:
        evidence_sources.append("operator_state")

    kept: List[str] = []
    for sentence in sentences:
        if _CLAIM_RE.search(sentence):
            claim = sentence.strip()
            detected.append(claim)
            claim_tokens = _claim_topic_tokens(claim)
            claim_topic_tokens_map[claim] = sorted(claim_tokens)
            reasons: List[str] = []

            if recent_items:
                for item in recent_items:
                    if _has_topic_overlap(claim_tokens, _tokens(str(item.get("content") or ""))):
                        reasons.append("recent_session_overlap")
                        break

            if operator_active and state_tokens and _has_topic_overlap(claim_tokens, state_tokens):
                reasons.append("active_objective_overlap")

            if structured_items:
                for item in structured_items:
                    if _has_topic_overlap(claim_tokens, _tokens(_memory_item_topic_text(item))):
                        reasons.append("structured_memory_overlap")
                        break

            if reasons:
                supported.append(claim)
                overlap_passed_by_claim[claim] = True
                support_reasons[claim] = reasons
                kept.append(sentence)
            else:
                unsupported.append(claim)
                overlap_passed_by_claim[claim] = False
                reasons = []
                if not claim_tokens:
                    reasons.append("no_claim_topic_tokens")
                if not recent_items:
                    reasons.append("no_recent_session_evidence")
                elif recent_items:
                    reasons.append("no_recent_session_topic_overlap")
                if not operator_active:
                    reasons.append("no_active_operator_objective")
                elif operator_active:
                    reasons.append("no_operator_state_topic_overlap")
                if not structured_items:
                    reasons.append("no_structured_memory_evidence")
                elif structured_items:
                    reasons.append("no_structured_memory_topic_overlap")
                rejection_reasons[claim] = reasons
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
        claim_topic_tokens=claim_topic_tokens_map,
        evidence_topic_tokens=sorted(
            _memory_topic_tokens(recent_items + structured_items).union(state_tokens)
        ),
        overlap_passed_by_claim=overlap_passed_by_claim,
        support_reasons=support_reasons,
        rejection_reasons=rejection_reasons,
        recovery_applied=recovery_applied,
        unsupported_continuity_claim=bool(unsupported),
    )
