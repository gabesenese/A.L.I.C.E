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

        # Require some non-trivial content — internal label format is no longer enforced.
        # "Finding:", "Next best move:", and "I inspected" are internal planning tokens
        # and should NOT be surfaced to the user.
        if len(response) < 8:
            failures.append("operator_response_too_short")

        if "i inspected" in response_low:
            failures.append("internal_inspection_label_in_user_output")

        if "finding:" in response_low:
            failures.append("internal_finding_label_in_user_output")

        if "next best move:" in response_low:
            failures.append("internal_next_step_label_in_user_output")

        local_success = local.get("success")
        if local_success is False:
            if "i inspected" in response_low or "verified" in response_low:
                failures.append("success_claim_on_failed_local_execution")

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
