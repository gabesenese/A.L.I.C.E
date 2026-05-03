from __future__ import annotations

from typing import Any, Dict

from ai.core.routing.evidence_contracts import EvidenceContracts
from ai.core.routing.routing_trace import RoutingTrace


class RouteArbiter:
    def arbitrate(
        self,
        *,
        user_input: str,
        candidate_route: str,
        candidate_intent: str,
        confidence: float,
        active_mode: str = "",
    ) -> Dict[str, Any]:
        trace = RoutingTrace(
            candidates=[{"route": candidate_route, "intent": candidate_intent, "confidence": float(confidence or 0.0)}],
            accepted_candidate={"route": candidate_route, "intent": candidate_intent, "confidence": float(confidence or 0.0)},
        )

        evidence = EvidenceContracts.evaluate(intent=candidate_intent, user_input=user_input, active_mode=active_mode)
        trace.evidence_contract_results.append(dict(evidence))

        final_route = str(candidate_route or "llm")
        final_intent = str(candidate_intent or "conversation:general")

        if not bool(evidence.get("accepted")):
            veto = {
                "reason": str(evidence.get("reason") or "evidence_contract_failed"),
                "original_intent": final_intent,
                "rerouted_to": str(evidence.get("reroute_intent") or ""),
                "file_tool_vetoed": bool(evidence.get("file_tool_vetoed")),
            }
            trace.vetoes.append(veto)
            trace.rejected_candidates.append(dict(trace.accepted_candidate))
            reroute_intent = str(evidence.get("reroute_intent") or "conversation:general")
            if reroute_intent.startswith("code:"):
                final_route = "local"
            elif reroute_intent.startswith("conversation:"):
                final_route = "llm"
            final_intent = reroute_intent
            trace.accepted_candidate = {"route": final_route, "intent": final_intent, "confidence": float(confidence or 0.0)}

        trace.final_route = final_route
        trace.final_intent = final_intent

        return {
            "route": final_route,
            "intent": final_intent,
            "trace": trace.as_dict(),
            "evidence": evidence,
        }
