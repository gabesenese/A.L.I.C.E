from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ai.core.routing.evidence_contracts import EvidenceContracts
from ai.core.routing.routing_trace import RoutingTrace


@dataclass(frozen=True)
class RouteCandidate:
    route: str
    intent: str
    confidence: float
    source: str = "unknown"
    evidence: Dict[str, Any] | None = None
    requires_target: bool = False
    risk: float = 0.0


class RouteArbiter:
    def arbitrate_candidates(
        self,
        *,
        user_input: str,
        candidates: List[RouteCandidate],
        active_mode: str = "",
    ) -> Dict[str, Any]:
        if not candidates:
            candidates = [RouteCandidate(route="llm", intent="conversation:general", confidence=0.0, source="fallback")]

        trace = RoutingTrace(
            candidates=[{"route": c.route, "intent": c.intent, "confidence": float(c.confidence), "source": c.source} for c in candidates],
        )
        sorted_candidates = sorted(candidates, key=lambda c: float(c.confidence) - float(c.risk), reverse=True)
        best = sorted_candidates[0]
        trace.accepted_candidate = {"route": best.route, "intent": best.intent, "confidence": float(best.confidence), "source": best.source}

        for candidate in sorted_candidates:
            evidence = EvidenceContracts.evaluate(intent=candidate.intent, user_input=user_input, active_mode=active_mode)
            trace.evidence_contract_results.append({"candidate": candidate.intent, **dict(evidence)})
            if bool(evidence.get("accepted")):
                trace.accepted_candidate = {
                    "route": candidate.route,
                    "intent": candidate.intent,
                    "confidence": float(candidate.confidence),
                    "source": candidate.source,
                }
                trace.final_route = candidate.route
                trace.final_intent = candidate.intent
                return {
                    "route": candidate.route,
                    "intent": candidate.intent,
                    "trace": trace.as_dict(),
                    "evidence": evidence,
                }
            trace.rejected_candidates.append(
                {"route": candidate.route, "intent": candidate.intent, "confidence": float(candidate.confidence), "source": candidate.source}
            )
            trace.vetoes.append(
                {
                    "reason": str(evidence.get("reason") or "evidence_contract_failed"),
                    "original_intent": candidate.intent,
                    "rerouted_to": str(evidence.get("reroute_intent") or ""),
                    "file_tool_vetoed": bool(evidence.get("file_tool_vetoed")),
                }
            )

        last_evidence = dict(trace.evidence_contract_results[-1]) if trace.evidence_contract_results else {}
        reroute_intent = str(last_evidence.get("reroute_intent") or "conversation:general")
        final_route = "local" if reroute_intent.startswith("code:") else "llm"
        trace.final_route = final_route
        trace.final_intent = reroute_intent
        trace.accepted_candidate = {"route": final_route, "intent": reroute_intent, "confidence": 0.0, "source": "arbiter_fallback"}
        return {
            "route": final_route,
            "intent": reroute_intent,
            "trace": trace.as_dict(),
            "evidence": last_evidence,
        }

    def arbitrate(
        self,
        *,
        user_input: str,
        candidate_route: str,
        candidate_intent: str,
        confidence: float,
        active_mode: str = "",
    ) -> Dict[str, Any]:
        return self.arbitrate_candidates(
            user_input=user_input,
            active_mode=active_mode,
            candidates=[
                RouteCandidate(
                    route=str(candidate_route or "llm"),
                    intent=str(candidate_intent or "conversation:general"),
                    confidence=float(confidence or 0.0),
                    source="legacy_single_candidate",
                )
            ],
        )
