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
    def _score_candidate(
        self,
        *,
        candidate: RouteCandidate,
        evidence: Dict[str, Any],
        active_objective: str,
        continuation_used: bool,
    ) -> float:
        confidence = float(candidate.confidence or 0.0)
        evidence_score = float(evidence.get("evidence_score") or 0.0)
        risk = float(candidate.risk or 0.0)
        missing_slots = list(evidence.get("missing_slots") or [])

        active_objective_bonus = (
            0.18
            if active_objective and candidate.intent.startswith("operator:")
            else 0.0
        )
        safe_usefulness_bonus = 0.1 if candidate.route in {"local", "tool"} else 0.02
        continuation_bonus = 0.08 if continuation_used and candidate.intent.startswith("operator:") else 0.0
        missing_target_penalty = 0.22 if "file_target" in missing_slots else 0.0
        clarification_penalty = 0.16 if candidate.route == "clarify" else 0.0
        destructive_action_penalty = 0.4 if bool(evidence.get("unsafe_action_vetoed")) else 0.0

        return (
            confidence
            + evidence_score
            + active_objective_bonus
            + safe_usefulness_bonus
            + continuation_bonus
            - risk
            - missing_target_penalty
            - clarification_penalty
            - destructive_action_penalty
        )

    def arbitrate_candidates(
        self,
        *,
        user_input: str,
        candidates: List[RouteCandidate],
        active_mode: str = "",
        active_objective: str = "",
        operator_state: Dict[str, Any] | None = None,
        project_memory: Dict[str, Any] | None = None,
        previous_route: str = "",
        previous_intent: str = "",
        continuation_context: bool = False,
    ) -> Dict[str, Any]:
        state = dict(operator_state or {})
        project = dict(project_memory or {})
        objective = str(
            active_objective or state.get("active_objective") or project.get("active_objective") or ""
        )
        continuation_used = bool(
            continuation_context
            or any(
                token in str(user_input or "").lower()
                for token in ("continue", "what's next", "where were we", "pick up")
            )
        )

        if not candidates:
            candidates = [
                RouteCandidate(
                    route="llm",
                    intent="conversation:general",
                    confidence=0.0,
                    source="fallback",
                )
            ]

        trace = RoutingTrace(
            candidates=[
                {
                    "route": c.route,
                    "intent": c.intent,
                    "confidence": float(c.confidence),
                    "source": c.source,
                }
                for c in candidates
            ],
        )
        candidate_scores: List[Dict[str, Any]] = []
        accepted_payload: Dict[str, Any] | None = None
        rejected_reasons: List[Dict[str, Any]] = []
        clarification_avoided = False

        for candidate in candidates:
            evidence = EvidenceContracts.evaluate(
                intent=candidate.intent,
                user_input=user_input,
                active_mode=active_mode,
                operator_state=state,
                project_memory=project,
            )
            score = self._score_candidate(
                candidate=candidate,
                evidence=evidence,
                active_objective=objective,
                continuation_used=continuation_used,
            )
            candidate_scores.append(
                {
                    "route": candidate.route,
                    "intent": candidate.intent,
                    "score": score,
                    "confidence": float(candidate.confidence),
                    "evidence_score": float(evidence.get("evidence_score") or 0.0),
                    "accepted_by_contract": bool(evidence.get("accepted")),
                    "reason": str(evidence.get("reason") or "ok"),
                }
            )
            trace.evidence_contract_results.append(
                {"candidate": candidate.intent, **dict(evidence)}
            )
            if not bool(evidence.get("accepted")):
                rejected_reasons.append(
                    {"intent": candidate.intent, "reason": str(evidence.get("reason") or "contract_reject")}
                )

        scored = sorted(candidate_scores, key=lambda row: float(row["score"]), reverse=True)
        selected = None
        for row in scored:
            if row.get("accepted_by_contract"):
                selected = row
                break
        if selected is None:
            fallback_evidence = trace.evidence_contract_results[-1] if trace.evidence_contract_results else {}
            reroute_intent = str(fallback_evidence.get("reroute_intent") or "conversation:general")
            selected = {
                "route": "local" if reroute_intent.startswith(("code:", "operator:")) else "llm",
                "intent": reroute_intent,
                "score": 0.0,
                "reason": str(fallback_evidence.get("reason") or "fallback_no_candidates"),
                "confidence": 0.0,
                "accepted_by_contract": True,
            }

        selected_intent = str(selected.get("intent") or "conversation:general")
        selected_route = str(selected.get("route") or "llm")
        if selected_route == "clarify":
            # Clarify should be last resort.
            best_non_clarify = next((row for row in scored if row.get("route") != "clarify" and row.get("accepted_by_contract")), None)
            if best_non_clarify:
                clarification_avoided = True
                selected = best_non_clarify
                selected_intent = str(selected.get("intent") or "conversation:general")
                selected_route = str(selected.get("route") or "llm")

        trace.accepted_candidate = {
            "route": selected_route,
            "intent": selected_intent,
            "confidence": float(selected.get("confidence") or 0.0),
            "source": "route_arbiter",
        }
        trace.final_route = selected_route
        trace.final_intent = selected_intent
        for row in scored:
            if row is selected:
                continue
            trace.rejected_candidates.append(
                {
                    "route": row.get("route"),
                    "intent": row.get("intent"),
                    "confidence": float(row.get("confidence") or 0.0),
                    "source": "route_arbiter",
                }
            )

        selected_evidence = next(
            (
                row
                for row in trace.evidence_contract_results
                if str(row.get("intent") or row.get("candidate") or "") == selected_intent
                or str(row.get("candidate") or "") == selected_intent
            ),
            {},
        )
        accepted_payload = {
            "route": selected_route,
            "intent": selected_intent,
            "trace": {
                **trace.as_dict(),
                "candidate_scores": scored,
                "selected_reason": str(selected.get("reason") or "top_score"),
                "rejected_reasons": rejected_reasons,
                "clarification_avoided": clarification_avoided,
                "active_objective_used": bool(objective),
                "project_memory_used": bool(project),
                "continuation_used": bool(continuation_used),
            },
            "evidence": selected_evidence,
        }
        return accepted_payload

    def arbitrate(
        self,
        *,
        user_input: str,
        candidate_route: str,
        candidate_intent: str,
        confidence: float,
        active_mode: str = "",
        active_objective: str = "",
        operator_state: Dict[str, Any] | None = None,
        project_memory: Dict[str, Any] | None = None,
        previous_route: str = "",
        previous_intent: str = "",
        continuation_context: bool = False,
    ) -> Dict[str, Any]:
        return self.arbitrate_candidates(
            user_input=user_input,
            active_mode=active_mode,
            active_objective=active_objective,
            operator_state=operator_state,
            project_memory=project_memory,
            previous_route=previous_route,
            previous_intent=previous_intent,
            continuation_context=continuation_context,
            candidates=[
                RouteCandidate(
                    route=str(candidate_route or "llm"),
                    intent=str(candidate_intent or "conversation:general"),
                    confidence=float(confidence or 0.0),
                    source="legacy_single_candidate",
                )
            ],
        )
