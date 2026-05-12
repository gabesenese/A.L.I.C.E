from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClaimVerificationResult:
    valid: bool
    unsupported_claims: List[str]
    rewritten_text: str = ""
    reasons: List[str] = field(default_factory=list)


def verify_response_claims(
    response_text: str,
    *,
    action_result: Dict | None = None,
    memory_result: Dict | None = None,
    local_execution: Dict | None = None,
    perception_frame: Dict | None = None,
) -> ClaimVerificationResult:
    text = str(response_text or "").strip()
    low = text.lower()
    unsupported: List[str] = []
    reasons: List[str] = []

    action_ok = bool(
        (action_result or {}).get("success")
        or (local_execution or {}).get("success")
        or (local_execution or {}).get("inspected_file")
    )
    memory_ok = bool((memory_result or {}).get("items") or (memory_result or {}).get("count"))
    delete_ok = bool(
        (memory_result or {}).get("deletion_executed")
        or ((memory_result or {}).get("success") and (memory_result or {}).get("verification_status"))
    )

    if any(token in low for token in ("i inspected", "i checked", "i read", "i found")) and not action_ok:
        unsupported.append("action_without_evidence")
    if any(token in low for token in ("i remember", "you told me", "we discussed")) and not memory_ok:
        unsupported.append("memory_without_evidence")
    if any(token in low for token in ("i deleted", "i erased", "removed from my data")) and not delete_ok:
        unsupported.append("delete_without_evidence")
    if any(token in low for token in ("i've been working", "i've been monitoring", "i was processing")):
        unsupported.append("background_without_evidence")
    if any(token in low for token in ("creator has said", "known to have used", "was built with")):
        unsupported.append("fictional_provenance_claim")

    if unsupported:
        reasons.append("unsupported_claims_detected")
        rewritten = text
        if "delete_without_evidence" in unsupported:
            rewritten = "I cannot claim deletion yet. I can preview matches and delete only after confirmation with verification."
        return ClaimVerificationResult(
            valid=False,
            unsupported_claims=unsupported,
            rewritten_text=rewritten,
            reasons=reasons,
        )
    return ClaimVerificationResult(valid=True, unsupported_claims=[], rewritten_text=text, reasons=[])
