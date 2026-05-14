from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClaimVerificationResult:
    valid: bool
    original_text: str
    verified_text: str
    unsupported_claims: List[str]
    reasons: List[str] = field(default_factory=list)
    evidence_used: Dict[str, Any] = field(default_factory=dict)


_ACTION_CLAIM_PATTERNS = (
    re.compile(r"\bi inspected\b", re.IGNORECASE),
    re.compile(r"\bi checked\b", re.IGNORECASE),
    re.compile(r"\bi read\b", re.IGNORECASE),
    re.compile(r"\bi analy[sz]ed\b", re.IGNORECASE),
    re.compile(r"\bi opened\b", re.IGNORECASE),
    re.compile(r"\bi listed\b", re.IGNORECASE),
    re.compile(r"\bi searched\b", re.IGNORECASE),
    re.compile(r"\bi found\b", re.IGNORECASE),
)
_MUTATION_CLAIM_PATTERNS = (
    re.compile(r"\bi deleted\b", re.IGNORECASE),
    re.compile(r"\bi erased\b", re.IGNORECASE),
    re.compile(r"\bi removed\b", re.IGNORECASE),
    re.compile(r"\bi wiped\b", re.IGNORECASE),
    re.compile(r"\bi updated\b", re.IGNORECASE),
    re.compile(r"\bi saved\b", re.IGNORECASE),
    re.compile(r"\bi changed\b", re.IGNORECASE),
    re.compile(r"\bi created\b", re.IGNORECASE),
    re.compile(r"\bremoved from memory\b", re.IGNORECASE),
    re.compile(r"\bit won't be stored anywhere\b", re.IGNORECASE),
    re.compile(r"\bremoved from my data\b", re.IGNORECASE),
)
_MEMORY_CLAIM_PATTERNS = (
    re.compile(r"\bi remember\b", re.IGNORECASE),
    re.compile(r"\byou told me\b", re.IGNORECASE),
    re.compile(r"\bwe discussed\b", re.IGNORECASE),
    re.compile(r"\blast time\b", re.IGNORECASE),
    re.compile(r"\bfrom our previous conversation\b", re.IGNORECASE),
    re.compile(r"\bi know you said\b", re.IGNORECASE),
)
_BACKGROUND_CLAIM_PATTERNS = (
    re.compile(r"\bi(?:'|’)ve been monitoring\b", re.IGNORECASE),
    re.compile(r"\bi(?:'|’)ve been working on\b", re.IGNORECASE),
    re.compile(r"\bi was processing\b", re.IGNORECASE),
    re.compile(r"\bi kept track\b", re.IGNORECASE),
    re.compile(r"\bi watched\b", re.IGNORECASE),
    re.compile(r"\bi noticed while you were away\b", re.IGNORECASE),
    re.compile(r"\bi(?:'|’)ll be ready tomorrow\b", re.IGNORECASE),
)


def _sentences(text: str) -> List[str]:
    raw = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if chunk.strip()]
    return raw if raw else ([str(text or "").strip()] if str(text or "").strip() else [])


def _normalize_rebuild(lines: List[str]) -> str:
    cleaned = [str(line or "").strip() for line in list(lines or []) if str(line or "").strip()]
    return " ".join(cleaned).strip()


def _local_execution_has_inspection(local_execution: Dict[str, Any]) -> bool:
    return bool(
        local_execution.get("success") and str(local_execution.get("inspected_file") or "").strip()
    )


def _has_action_evidence(
    sentence: str,
    *,
    local_execution: Dict[str, Any],
    action_result: Dict[str, Any],
) -> bool:
    low = sentence.lower()
    action_evidence = dict(action_result.get("evidence") or {})
    action_verified = bool(action_result.get("verified"))
    local_success = bool(local_execution.get("success"))
    inspected_file = str(local_execution.get("inspected_file") or "").strip()
    action_success = bool(action_result.get("success"))
    search_evidence = bool(
        action_result.get("results")
        or action_result.get("matches")
        or action_result.get("search_results")
        or action_result.get("tool_results")
    )

    if "i inspected" in low:
        if action_verified and action_result.get("success"):
            action_inspected = str(action_evidence.get("inspected_file") or "").strip()
            if not action_inspected:
                return False
            claim_target_match = re.search(r"\bi inspected\s+([a-zA-Z0-9_./\\-]+)", sentence, re.IGNORECASE)
            if claim_target_match:
                claimed_target = str(claim_target_match.group(1) or "").strip("`'\".,;:!?")
                normalized_claimed = claimed_target.replace("\\", "/").lower()
                normalized_inspected = action_inspected.replace("\\", "/").strip("`'\".,;:!?").lower()
                if normalized_claimed and normalized_claimed != normalized_inspected:
                    return False
            return True
        if not (local_success and inspected_file):
            return False
        claim_target_match = re.search(r"\bi inspected\s+([a-zA-Z0-9_./\\-]+)", sentence, re.IGNORECASE)
        if claim_target_match:
            claimed_target = str(claim_target_match.group(1) or "").strip("`'\".,;:!?")
            normalized_claimed = claimed_target.replace("\\", "/").lower()
            normalized_inspected = inspected_file.replace("\\", "/").strip("`'\".,;:!?").lower()
            if normalized_claimed and normalized_claimed != normalized_inspected:
                return False
        return True
    if "i checked" in low:
        return bool(local_success or action_success or inspected_file or (action_verified and action_evidence))
    if "i read" in low or "i analyzed" in low or "i analysed" in low or "i opened" in low or "i listed" in low:
        return bool(local_success or action_success or inspected_file or (action_verified and action_evidence))
    if "i searched" in low or "i found" in low:
        return bool(search_evidence or local_success or action_success or (action_verified and action_evidence))
    return True


def _has_deletion_evidence(deletion_result: Dict[str, Any]) -> bool:
    if not bool(deletion_result.get("success")):
        return False
    deleted_count = deletion_result.get("deleted_count")
    verification_status = str(deletion_result.get("verification_status") or "").strip()
    return bool(deleted_count is not None and verification_status)


def _has_mutation_evidence(
    sentence: str,
    *,
    action_result: Dict[str, Any],
    memory_result: Dict[str, Any],
    deletion_result: Dict[str, Any],
) -> bool:
    low = sentence.lower()
    if any(token in low for token in ("i deleted", "i erased", "i removed", "i wiped", "removed from memory", "removed from my data", "it won't be stored anywhere")):
        return _has_deletion_evidence(deletion_result)
    return bool(
        action_result.get("success")
        or action_result.get("persisted")
        or action_result.get("saved")
        or memory_result.get("success")
        or memory_result.get("saved_count")
    )


def _has_memory_evidence(
    *,
    memory_result: Dict[str, Any],
    operator_state: Dict[str, Any],
    project_memory: Dict[str, Any],
) -> bool:
    if memory_result.get("source") or memory_result.get("matched_memory_id"):
        return True
    if memory_result.get("items") or memory_result.get("count"):
        return True
    return False


def _has_background_evidence(
    *,
    background_events: List[Dict[str, Any]],
    action_result: Dict[str, Any],
) -> bool:
    if list(background_events or []):
        return True
    if action_result.get("scheduled") or action_result.get("scheduled_task_id"):
        return True
    return False


def _limitation_for_reason(reason: str) -> str:
    mapping = {
        "action_claim_without_evidence": "I couldn't verify that action from available evidence.",
        "mutation_claim_without_evidence": "I can't confirm deletion because no verified delete action completed.",
        "memory_claim_without_evidence": "I don't have grounded evidence for that previous topic.",
        "background_claim_without_evidence": "We can continue tomorrow.",
    }
    return mapping.get(reason, "I couldn't verify that claim from available evidence.")


def verify_response_claims(
    response_text: str,
    *,
    route: str = "",
    intent: str = "",
    local_execution: dict | None = None,
    action_result: dict | None = None,
    memory_result: dict | None = None,
    deletion_result: dict | None = None,
    operator_state: dict | None = None,
    project_memory: dict | None = None,
    background_events: list[dict] | None = None,
) -> ClaimVerificationResult:
    original = str(response_text or "").strip()
    if not original:
        return ClaimVerificationResult(
            valid=True,
            original_text=original,
            verified_text=original,
            unsupported_claims=[],
            reasons=[],
            evidence_used={},
        )

    local = dict(local_execution or {})
    action = dict(action_result or {})
    if isinstance(action.get("action_result"), dict):
        action = dict(action.get("action_result") or {})
    memory = dict(memory_result or {})
    deletion = dict(deletion_result or {})
    op_state = dict(operator_state or {})
    project = dict(project_memory or {})
    bg_events = list(background_events or [])

    reasons: List[str] = []
    unsupported_claims: List[str] = []
    kept_sentences: List[str] = []

    for sentence in _sentences(original):
        low = sentence.lower()
        sentence_invalid = False
        sentence_reason = ""

        if any(pattern.search(sentence) for pattern in _BACKGROUND_CLAIM_PATTERNS):
            if "i'll be ready tomorrow" in low or "i’ll be ready tomorrow" in low:
                sentence_invalid = True
                sentence_reason = "background_claim_without_evidence"
                kept_sentences.append("We can continue tomorrow.")
            elif not _has_background_evidence(background_events=bg_events, action_result=action):
                sentence_invalid = True
                sentence_reason = "background_claim_without_evidence"

        if (not sentence_invalid) and any(pattern.search(sentence) for pattern in _ACTION_CLAIM_PATTERNS):
            if not _has_action_evidence(sentence, local_execution=local, action_result=action):
                sentence_invalid = True
                sentence_reason = "action_claim_without_evidence"

        if (not sentence_invalid) and any(pattern.search(sentence) for pattern in _MUTATION_CLAIM_PATTERNS):
            if not _has_mutation_evidence(
                sentence,
                action_result=action,
                memory_result=memory,
                deletion_result=deletion,
            ):
                sentence_invalid = True
                sentence_reason = "mutation_claim_without_evidence"

        if (not sentence_invalid) and any(pattern.search(sentence) for pattern in _MEMORY_CLAIM_PATTERNS):
            if not _has_memory_evidence(
                memory_result=memory,
                operator_state=op_state,
                project_memory=project,
            ):
                sentence_invalid = True
                sentence_reason = "memory_claim_without_evidence"

        if sentence_invalid:
            unsupported_claims.append(sentence)
            reasons.append(sentence_reason)
            if sentence_reason == "memory_claim_without_evidence":
                kept_sentences.append("I don't have grounded evidence for that previous topic.")
            continue

        kept_sentences.append(sentence)

    reasons = list(dict.fromkeys([r for r in reasons if r]))
    verified = _normalize_rebuild(kept_sentences)
    if not verified and reasons:
        verified = _limitation_for_reason(reasons[0])

    return ClaimVerificationResult(
        valid=not bool(reasons),
        original_text=original,
        verified_text=verified or original,
        unsupported_claims=unsupported_claims,
        reasons=reasons,
        evidence_used={
            "route": str(route or ""),
            "intent": str(intent or ""),
            "local_execution_success": bool(local.get("success")),
            "inspected_file": str(local.get("inspected_file") or ""),
            "action_success": bool(action.get("success")),
            "memory_source": str(memory.get("source") or ""),
            "memory_match": str(memory.get("matched_memory_id") or ""),
            "deletion_success": bool(deletion.get("success")),
            "deletion_count": deletion.get("deleted_count"),
            "deletion_verification_status": str(deletion.get("verification_status") or ""),
            "background_event_count": len(bg_events),
        },
    )
