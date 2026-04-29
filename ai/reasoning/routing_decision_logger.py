"""
Routing Decision Logger - Tier 4: Explainable Routing

Logs all routing decisions with reasoning.
Makes ALICE transparent; users understand her choices.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingDecisionType(Enum):
    """Types of routing decisions ALICE makes."""

    TOOL_DISPATCH = "tool_dispatch"
    LLM_FALLBACK = "llm_fallback"
    CLARIFICATION = "clarification"
    PATTERN_MATCH = "pattern_match"
    MEMORY_RECALL = "memory_recall"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


@dataclass
class RoutingCandidates:
    """Candidates considered during routing."""

    candidate_id: str
    name: str
    type: str  # tool, llm, pattern, etc.
    score: float  # 0.0-1.0
    reasoning: str
    pros: List[str]
    cons: List[str]


@dataclass
class RoutingDecision:
    """A single routing decision."""

    decision_id: str
    timestamp: str
    turn_number: int
    user_input: str
    classified_intent: str
    intent_confidence: float
    decision_type: RoutingDecisionType

    # Candidates evaluated
    candidates_considered: List[RoutingCandidates]
    winning_candidate: str  # name of winner
    winning_score: float

    # Reasoning
    decision_reasoning: str
    factors_used: List[str]  # What factors influenced the decision
    uncertainty_level: float  # 0.0 = certain, 1.0 = very uncertain
    fallback_route_if_failed: Optional[str] = None

    # Outcome
    execution_success: bool = None
    execution_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class RoutingDecisionLogger:
    """Logs and analyzes ALICE's routing decisions."""

    def __init__(self):
        self.decisions: List[RoutingDecision] = []
        self.decision_counter = 0
        self.routing_stats: Dict[str, int] = {}
        self.routing_success_rates: Dict[str, float] = {}

    def log_decision(
        self,
        user_input: str,
        classified_intent: str,
        intent_confidence: float,
        decision_type: RoutingDecisionType,
        candidates_considered: List[Dict[str, Any]],
        winning_candidate: str,
        winning_score: float,
        decision_reasoning: str,
        factors_used: List[str],
        uncertainty_level: float = 0.0,
        fallback_route: Optional[str] = None,
        turn_number: int = 0,
    ) -> RoutingDecision:
        """
        Log a routing decision.

        Args:
            user_input: User's message
            classified_intent: ALICE's detected intent
            intent_confidence: Confidence in intent (0.0-1.0)
            decision_type: Type of routing decision
            candidates_considered: List of candidate routes with scores
            winning_candidate: Name of the selected route
            winning_score: Score of winning candidate
            decision_reasoning: Explanation of why this route was chosen
            factors_used: List of factors that influenced the decision
            uncertainty_level: How certain is ALICE (0.0=certain, 1.0=uncertain)
            fallback_route: Alternative route if this one fails
            turn_number: Turn number in session

        Returns:
            The logged decision
        """
        self.decision_counter += 1

        # Convert candidate dicts to objects
        candidates = []
        for cand in candidates_considered:
            candidate = RoutingCandidates(
                candidate_id=f"cand_{len(candidates)}",
                name=cand.get("name", "unknown"),
                type=cand.get("type", "unknown"),
                score=cand.get("score", 0.0),
                reasoning=cand.get("reasoning", ""),
                pros=cand.get("pros", []),
                cons=cand.get("cons", []),
            )
            candidates.append(candidate)

        decision = RoutingDecision(
            decision_id=f"route_{self.decision_counter}",
            timestamp=datetime.now().isoformat(),
            turn_number=turn_number,
            user_input=user_input,
            classified_intent=classified_intent,
            intent_confidence=intent_confidence,
            decision_type=decision_type,
            candidates_considered=candidates,
            winning_candidate=winning_candidate,
            winning_score=winning_score,
            decision_reasoning=decision_reasoning,
            factors_used=factors_used,
            uncertainty_level=uncertainty_level,
            fallback_route_if_failed=fallback_route,
        )

        self.decisions.append(decision)

        # Update stats
        route_key = f"{winning_candidate}_{decision_type.value}"
        self.routing_stats[route_key] = self.routing_stats.get(route_key, 0) + 1

        logger.info(
            f"[Routing] Decision: {winning_candidate} (type={decision_type.value}, "
            f"confidence={intent_confidence:.2f}, score={winning_score:.2f}, "
            f"uncertainty={uncertainty_level:.2f})"
        )

        return decision

    def record_outcome(
        self, decision_id: str, success: bool, error_message: Optional[str] = None
    ) -> None:
        """Record the outcome of a routing decision."""
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                decision.execution_success = success
                decision.execution_error = error_message

                # Update success rate
                route_key = (
                    f"{decision.winning_candidate}_{decision.decision_type.value}"
                )
                current_rate = self.routing_success_rates.get(route_key, 0.0)
                total = self.routing_stats.get(route_key, 1)

                if success:
                    new_rate = (current_rate * (total - 1) + 1.0) / total
                else:
                    new_rate = (current_rate * (total - 1) + 0.0) / total

                self.routing_success_rates[route_key] = new_rate

                logger.info(f"[Routing] Outcome: {decision_id} → success={success}")
                break

    def get_decision_for_explanation(self, decision_id: str) -> Optional[str]:
        """
        Get human-readable explanation of a routing decision.
        Use this to show the user why ALICE made a certain choice.
        """
        for decision in self.decisions:
            if decision.decision_id == decision_id:
                explanation = f"""
I routed your request as follows:

**What I detected:**
- Your intent: {decision.classified_intent}
- Confidence: {decision.intent_confidence * 100:.0f}%

**Route I chose:**
- {decision.winning_candidate} (score: {decision.winning_score:.2f})

**Why I chose this:**
- {decision.decision_reasoning}

**Factors I considered:**
- {", ".join(decision.factors_used[:3])}

**Alternative routes I considered:**
"""
                for cand in decision.candidates_considered:
                    if cand.name != decision.winning_candidate:
                        explanation += f"\n  • {cand.name} ({cand.score:.2f}): {cand.reasoning[:50]}"

                if decision.uncertainty_level > 0.3:
                    explanation += (
                        "\n\n⚠️ Note: I'm not very confident in this decision. "
                    )
                    explanation += f"Please correct me if I misunderstood (uncertainty: {decision.uncertainty_level * 100:.0f}%)"

                return explanation.strip()

        return None

    def get_routing_effectiveness(self) -> Dict[str, Any]:
        """Analyze routing effectiveness."""
        if not self.decisions:
            return {"total_decisions": 0}

        # Group by route
        by_route = {}
        for decision in self.decisions:
            route = f"{decision.winning_candidate}_{decision.decision_type.value}"
            if route not in by_route:
                by_route[route] = {"total": 0, "successful": 0}
            by_route[route]["total"] += 1
            if decision.execution_success:
                by_route[route]["successful"] += 1

        # Calculate effectiveness
        effectiveness = {}
        for route, stats in by_route.items():
            success_rate = (
                stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            effectiveness[route] = {
                "usage_count": stats["total"],
                "success_rate": success_rate,
                "reliability": (
                    "high"
                    if success_rate > 0.8
                    else "medium"
                    if success_rate > 0.6
                    else "low"
                ),
            }

        return {
            "total_decisions": len(self.decisions),
            "by_route": effectiveness,
            "most_used": max(by_route.items(), key=lambda x: x[1]["total"])[0],
            "most_reliable": max(
                by_route.items(), key=lambda x: x[1]["successful"] / x[1]["total"]
            )[0],
        }

    def get_problem_routes(self) -> List[Dict[str, Any]]:
        """Identify routes with high failure rates."""
        if not self.decisions:
            return []

        # Group by route
        by_route = {}
        for decision in self.decisions:
            route = f"{decision.winning_candidate}_{decision.decision_type.value}"
            if route not in by_route:
                by_route[route] = {"total": 0, "successful": 0}
            by_route[route]["total"] += 1
            if decision.execution_success:
                by_route[route]["successful"] += 1

        # Find problem routes
        problems = []
        for route, stats in by_route.items():
            if stats["total"] >= 3:  # Only flag if used multiple times
                success_rate = stats["successful"] / stats["total"]
                if success_rate < 0.7:
                    problems.append(
                        {
                            "route": route,
                            "usage_count": stats["total"],
                            "success_rate": success_rate,
                            "failure_count": stats["total"] - stats["successful"],
                            "severity": "high" if success_rate < 0.4 else "medium",
                        }
                    )

        return sorted(problems, key=lambda x: x["failure_count"], reverse=True)

    def get_user_predictability_score(self) -> float:
        """
        Score: How predictable is user input for ALICE?
        High = consistent intents, low = random/varied

        Based on: consistency of intent_confidence and routing path diversity
        """
        if len(self.decisions) < 5:
            return 0.5  # insufficient data

        # Calculate confidence consistency
        recent = self.decisions[-20:]
        confidences = [d.intent_confidence for d in recent]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(
            confidences
        )
        consistency_score = 1.0 / (1.0 + confidence_variance)

        # Calculate routing diversity (lower = more predictable)
        routes = [d.winning_candidate for d in recent]
        unique_routes = len(set(routes))
        diversity_score = 1.0 - (
            unique_routes / len(routes)
        )  # More routes = less predictable

        # Combine scores
        predictability = (consistency_score + diversity_score) / 2.0
        return round(predictability, 2)

    def format_decision_log(self, max_entries: int = 10) -> str:
        """Get formatted decision log for display."""
        if not self.decisions:
            return "No routing decisions recorded."

        lines = ["Recent Routing Decisions:", "=" * 70]

        for decision in self.decisions[-max_entries:]:
            symbol = (
                "✓"
                if decision.execution_success
                else "✗"
                if decision.execution_success is False
                else "?"
            )
            lines.append(
                f"{symbol} {decision.decision_id}: {decision.classified_intent}"
            )
            lines.append(
                f"   → {decision.winning_candidate} (conf={decision.intent_confidence:.0%})"
            )
            if decision.execution_error:
                lines.append(f"   Error: {decision.execution_error[:60]}")

        return "\n".join(lines)
