from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


_FORBIDDEN_OPERATOR_PHRASES = (
    "i'm here to assist",
    "i am here to assist",
    "assist you with",
    "ready for our conversation",
    "ready when you are",
    "how can i help",
    "what would you like",
    "let me know",
    "what should we work on",
)


@dataclass
class VerifiedGrowthCheck:
    capability_added: str
    behavior_replaced: str
    invariant_enforced: str
    evidence: Dict[str, Any]
    visible_flow: str
    deleted_or_simplified: List[str]
    passed: bool
    failures: List[str]


def _is_operator_surface(route: str, intent: str) -> bool:
    route_key = str(route or "").strip().lower()
    intent_key = str(intent or "").strip().lower()
    return route_key == "local" or intent_key.startswith("operator:")


def verify_operator_surface_contract(
    *,
    route: str,
    intent: str,
    response_text: str,
    local_execution: Dict[str, Any] | None,
    next_step: str = "",
) -> VerifiedGrowthCheck:
    local = dict(local_execution or {})
    response = str(response_text or "").strip()
    response_low = response.lower()
    failures: List[str] = []
    operator_surface = _is_operator_surface(route=route, intent=intent)

    if operator_surface:
        if any(token in response_low for token in _FORBIDDEN_OPERATOR_PHRASES):
            failures.append("service_chatter_in_operator_surface")

        has_required_marker = any(
            token in response_low
            for token in ("finding:", "i inspected", "blocker:", "next best move:")
        )
        if not has_required_marker:
            failures.append("missing_operator_evidence_markers")

        if "i inspected" in response_low and not str(local.get("inspected_file") or "").strip():
            failures.append("inspection_claim_without_evidence")

        local_success = local.get("success")
        if local_success is False:
            if "i inspected" in response_low or "verified" in response_low:
                failures.append("success_claim_on_failed_local_execution")
            if "finding:" in response_low and "blocker:" not in response_low:
                failures.append("finding_claim_on_failed_local_execution")

        if str(next_step or "").strip() and "next best move:" not in response_low:
            failures.append("missing_next_best_move_with_recommendation")

    return VerifiedGrowthCheck(
        capability_added="Operator response surface contract enforcement for local/operator turns.",
        behavior_replaced="Unverified assistant-style chatter mixed into operator output.",
        invariant_enforced="Operator responses only contain evidence-backed local execution content.",
        evidence={
            "route": str(route or ""),
            "intent": str(intent or ""),
            "local_success": local.get("success"),
            "has_inspected_file_evidence": bool(str(local.get("inspected_file") or "").strip()),
            "next_step_present": bool(str(next_step or "").strip()),
        },
        visible_flow="Operator turns now return Finding/Blocker/Next-best-move output without service chatter.",
        deleted_or_simplified=[
            "operator greeting/context acknowledgements in operator response path",
            "fallback base-text stitching for operator-local outputs",
        ],
        passed=(len(failures) == 0),
        failures=failures,
    )

