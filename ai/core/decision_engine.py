"""
Advanced Decision Engine
========================
Sophisticated decision-making algorithms using multi-criteria optimization,
probabilistic reasoning, and utility theory.
"""

import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Represents a decision option with weighted criteria"""
    name: str
    description: str
    scores: Dict[str, float] = field(default_factory=dict)
    utility: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Criterion:
    """Evaluation criterion with weight and optimization direction"""
    name: str
    weight: float
    maximize: bool = True  # True to maximize, False to minimize
    threshold: Optional[float] = None  # Minimum acceptable value


class DecisionEngine:
    """
    Advanced decision-making engine using multiple algorithms:

    - Multi-Criteria Decision Analysis (MCDA)
    - Weighted Sum Model (WSM)
    - TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    - Analytic Hierarchy Process (AHP) concepts
    - Probabilistic reasoning
    - Risk assessment
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1

    def evaluate_options(
        self,
        options: List[Decision],
        criteria: List[Criterion],
        method: str = 'weighted_sum'
    ) -> List[Decision]:
        """
        Evaluate and rank decision options.

        Args:
            options: List of decision options to evaluate
            criteria: List of evaluation criteria with weights
            method: Evaluation method ('weighted_sum', 'topsis', 'utility')

        Returns:
            Ranked list of decisions (best first)
        """
        if not options or not criteria:
            return []

        # Normalize weights
        total_weight = sum(c.weight for c in criteria)
        if total_weight > 0:
            for criterion in criteria:
                criterion.weight /= total_weight

        # Apply chosen method
        if method == 'weighted_sum':
            evaluated = self._weighted_sum(options, criteria)
        elif method == 'topsis':
            evaluated = self._topsis(options, criteria)
        elif method == 'utility':
            evaluated = self._utility_theory(options, criteria)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by utility (descending)
        evaluated.sort(key=lambda d: d.utility, reverse=True)

        # Calculate confidence scores
        self._calculate_confidence(evaluated)

        return evaluated

    def _weighted_sum(
        self,
        options: List[Decision],
        criteria: List[Criterion]
    ) -> List[Decision]:
        """Weighted Sum Model - simple but effective"""
        for option in options:
            total_utility = 0.0

            for criterion in criteria:
                score = option.scores.get(criterion.name, 0.0)

                # Normalize to 0-1 range
                normalized_score = max(0.0, min(1.0, score))

                # Invert if minimizing
                if not criterion.maximize:
                    normalized_score = 1.0 - normalized_score

                # Apply weight
                total_utility += normalized_score * criterion.weight

            option.utility = total_utility

        return options

    def _topsis(
        self,
        options: List[Decision],
        criteria: List[Criterion]
    ) -> List[Decision]:
        """
        TOPSIS - finds option closest to ideal and farthest from worst.
        More sophisticated than weighted sum.
        """
        if not options:
            return []

        # Find ideal and anti-ideal solutions
        ideal = {}
        anti_ideal = {}

        for criterion in criteria:
            scores = [opt.scores.get(criterion.name, 0.0) for opt in options]

            if criterion.maximize:
                ideal[criterion.name] = max(scores)
                anti_ideal[criterion.name] = min(scores)
            else:
                ideal[criterion.name] = min(scores)
                anti_ideal[criterion.name] = max(scores)

        # Calculate distances to ideal and anti-ideal
        for option in options:
            distance_to_ideal = 0.0
            distance_to_anti_ideal = 0.0

            for criterion in criteria:
                score = option.scores.get(criterion.name, 0.0)

                # Euclidean distance
                diff_ideal = (score - ideal[criterion.name]) ** 2
                diff_anti = (score - anti_ideal[criterion.name]) ** 2

                distance_to_ideal += criterion.weight * diff_ideal
                distance_to_anti_ideal += criterion.weight * diff_anti

            distance_to_ideal = math.sqrt(distance_to_ideal)
            distance_to_anti_ideal = math.sqrt(distance_to_anti_ideal)

            # Calculate relative closeness to ideal
            if distance_to_ideal + distance_to_anti_ideal > 0:
                option.utility = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)
            else:
                option.utility = 0.5

        return options

    def _utility_theory(
        self,
        options: List[Decision],
        criteria: List[Criterion]
    ) -> List[Decision]:
        """
        Utility theory with risk adjustment.
        Uses logarithmic utility function for risk-averse behavior.
        """
        for option in options:
            total_utility = 0.0

            for criterion in criteria:
                score = option.scores.get(criterion.name, 0.0)

                # Apply logarithmic utility (risk-averse)
                # U(x) = log(1 + x) for x >= 0
                if score >= 0:
                    utility = math.log(1 + score)
                else:
                    utility = -math.log(1 - score)

                # Invert if minimizing
                if not criterion.maximize:
                    utility = -utility

                # Apply weight
                total_utility += utility * criterion.weight

            option.utility = total_utility

        # Normalize utilities to 0-1 range
        if options:
            utilities = [opt.utility for opt in options]
            min_util = min(utilities)
            max_util = max(utilities)

            if max_util > min_util:
                for option in options:
                    option.utility = (option.utility - min_util) / (max_util - min_util)

        return options

    def _calculate_confidence(self, evaluated: List[Decision]):
        """
        Calculate confidence scores based on:
        - Separation from other options
        - Data quality/completeness
        """
        if len(evaluated) < 2:
            if evaluated:
                evaluated[0].confidence = 1.0
            return

        # Calculate utility spread
        utilities = [d.utility for d in evaluated]
        best_utility = utilities[0]
        second_best = utilities[1] if len(utilities) > 1 else 0.0

        # Confidence based on separation
        for i, decision in enumerate(evaluated):
            if i == 0:
                # Best option - confidence based on lead over second
                separation = best_utility - second_best
                decision.confidence = min(1.0, 0.5 + separation)
            else:
                # Other options - confidence decreases with rank
                decision.confidence = max(0.1, 1.0 - (i * 0.15))

            # Adjust for data completeness
            criteria_coverage = len(decision.scores) / max(1, len(evaluated[0].scores))
            decision.confidence *= criteria_coverage

    def probabilistic_choice(
        self,
        options: List[Decision],
        temperature: float = 1.0
    ) -> Decision:
        """
        Make a probabilistic choice using softmax distribution.
        Higher temperature = more exploration, lower = more exploitation.

        Args:
            options: List of evaluated decisions
            temperature: Softmax temperature (0 to inf)

        Returns:
            Chosen decision
        """
        if not options:
            raise ValueError("No options to choose from")

        if len(options) == 1:
            return options[0]

        # Calculate softmax probabilities
        utilities = [opt.utility / temperature for opt in options]
        max_utility = max(utilities)

        # Subtract max for numerical stability
        exp_utilities = [math.exp(u - max_utility) for u in utilities]
        total = sum(exp_utilities)

        probabilities = [exp_u / total for exp_u in exp_utilities]

        # For now, return highest probability (deterministic)
        # In production, would sample from distribution
        best_idx = probabilities.index(max(probabilities))
        return options[best_idx]

    def compare_pairwise(
        self,
        option1: Decision,
        option2: Decision,
        criteria: List[Criterion]
    ) -> Tuple[Decision, float]:
        """
        Pairwise comparison of two options.

        Returns:
            (better_option, confidence)
        """
        score1 = 0.0
        score2 = 0.0

        for criterion in criteria:
            val1 = option1.scores.get(criterion.name, 0.0)
            val2 = option2.scores.get(criterion.name, 0.0)

            if criterion.maximize:
                if val1 > val2:
                    score1 += criterion.weight
                else:
                    score2 += criterion.weight
            else:
                if val1 < val2:
                    score1 += criterion.weight
                else:
                    score2 += criterion.weight

        if score1 > score2:
            confidence = score1 / (score1 + score2)
            return option1, confidence
        else:
            confidence = score2 / (score1 + score2)
            return option2, confidence

    def learn_from_outcome(
        self,
        decision: Decision,
        actual_utility: float,
        context: Dict[str, Any] = None
    ):
        """
        Learn from decision outcomes to improve future choices.

        Args:
            decision: The decision that was made
            actual_utility: Actual utility achieved (0-1)
            context: Additional context information
        """
        # Record outcome
        self.history.append({
            'decision': decision.name,
            'predicted_utility': decision.utility,
            'actual_utility': actual_utility,
            'error': abs(decision.utility - actual_utility),
            'context': context or {}
        })

        # Simple learning: could adjust criterion weights based on prediction errors
        # For now, just log for future analysis
        logger.info(
            f"Decision outcome: {decision.name}, "
            f"predicted={decision.utility:.3f}, "
            f"actual={actual_utility:.3f}"
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate decision-making performance metrics"""
        if not self.history:
            return {}

        errors = [entry['error'] for entry in self.history]

        return {
            'mean_absolute_error': sum(errors) / len(errors),
            'decision_count': len(self.history),
            'accuracy': 1.0 - (sum(errors) / len(errors)) if errors else 0.0
        }


class RiskAssessment:
    """
    Risk assessment and management for decision-making.
    """

    @staticmethod
    def calculate_risk_score(
        probability: float,
        impact: float,
        mitigation: float = 0.0
    ) -> float:
        """
        Calculate risk score using standard risk matrix.

        Args:
            probability: Probability of risk occurring (0-1)
            impact: Impact if risk occurs (0-1, where 1 is catastrophic)
            mitigation: Effectiveness of mitigation measures (0-1)

        Returns:
            Risk score (0-1)
        """
        # Base risk = probability * impact
        base_risk = probability * impact

        # Apply mitigation
        mitigated_risk = base_risk * (1.0 - mitigation)

        return mitigated_risk

    @staticmethod
    def risk_category(risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score >= 0.7:
            return 'CRITICAL'
        elif risk_score >= 0.5:
            return 'HIGH'
        elif risk_score >= 0.3:
            return 'MEDIUM'
        elif risk_score >= 0.1:
            return 'LOW'
        else:
            return 'MINIMAL'

    @staticmethod
    def expected_value(
        outcomes: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate expected value from probability-weighted outcomes.

        Args:
            outcomes: List of (probability, value) tuples

        Returns:
            Expected value
        """
        return sum(prob * value for prob, value in outcomes)


# Global singleton
_decision_engine = None


def get_decision_engine() -> DecisionEngine:
    """Get global decision engine instance"""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
