"""
Result Quality Scorer - Tier 1: Tighter Error Recovery

Evaluates tool results and triggers retries if quality is below threshold.
Handles structured vs. unstructured outputs.
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Represents quality evaluation of a result."""

    overall: float  # 0.0 to 1.0
    structure_confidence: float  # How well-formed is the result
    relevance_confidence: float  # How relevant to the query
    completeness_confidence: float  # How complete is the answer
    issues: list  # List of detected issues
    recommendations: list  # Suggestions to improve
    should_retry: bool = False
    retry_reason: str = ""


class ResultQualityScorer:
    """Scores quality of tool/plugin results."""

    def __init__(
        self,
        structure_weight: float = 0.3,
        relevance_weight: float = 0.4,
        completeness_weight: float = 0.3,
    ):
        """
        Args:
            structure_weight: Impact of structure quality on overall score
            relevance_weight: Impact of relevance on overall score
            completeness_weight: Impact of completeness on overall score
        """
        self.structure_weight = structure_weight
        self.relevance_weight = relevance_weight
        self.completeness_weight = completeness_weight
        self.retry_threshold = 0.6  # Retry if score below this
        self.high_stakes_threshold = 0.75  # High stakes need higher confidence

    def score_result(
        self,
        result: Dict[str, Any],
        user_input: str,
        intent: str = "",
        is_high_stakes: bool = False,
    ) -> QualityScore:
        """
        Score the quality of a tool result.

        Args:
            result: Tool/plugin result dict
            user_input: Original user query
            intent: Classified intent
            is_high_stakes: If True, use stricter threshold

        Returns:
            QualityScore with overall score and recommendations
        """
        issues = []
        recommendations = []

        # Check structure
        struct_score = self._score_structure(result, issues)

        # Check relevance to query
        rel_score = self._score_relevance(result, user_input, intent, issues)

        # Check completeness
        comp_score = self._score_completeness(result, issues)

        # Calculate weighted overall score
        overall = (
            struct_score * self.structure_weight
            + rel_score * self.relevance_weight
            + comp_score * self.completeness_weight
        )

        # Add recommendations based on issues
        recommendations = self._generate_recommendations(
            issues, struct_score, rel_score, comp_score
        )

        # Determine if retry is needed
        threshold = (
            self.high_stakes_threshold if is_high_stakes else self.retry_threshold
        )
        should_retry = overall < threshold and len(issues) > 0

        retry_reason = ""
        if should_retry:
            retry_reason = self._build_retry_reason(issues, overall)

        score = QualityScore(
            overall=round(overall, 3),
            structure_confidence=round(struct_score, 3),
            relevance_confidence=round(rel_score, 3),
            completeness_confidence=round(comp_score, 3),
            issues=issues,
            recommendations=recommendations,
            should_retry=should_retry,
            retry_reason=retry_reason,
        )

        logger.info(
            f"[Quality] Score {score.overall:.2f}: struct={score.structure_confidence:.2f}, "
            f"rel={score.relevance_confidence:.2f}, comp={score.completeness_confidence:.2f} "
            f"(retry={should_retry})"
        )

        return score

    def _score_structure(self, result: Dict[str, Any], issues: list) -> float:
        """Score how well-formed the result is."""
        score = 1.0

        if not isinstance(result, dict):
            issues.append("result is not a dict")
            return 0.1

        # Check for essential fields
        if "success" not in result:
            issues.append("missing 'success' field")
            score -= 0.2

        if not isinstance(result.get("success", False), bool):
            issues.append("'success' field is not boolean")
            score -= 0.15

        # Check for meaningful content
        response = str(result.get("response") or "").strip()
        data = result.get("data", {})

        if not response and not data:
            issues.append("empty response and no data")
            score -= 0.3

        # Check for error field when failed
        if result.get("success") is False and not result.get("error"):
            issues.append("failed result lacks error explanation")
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _score_relevance(
        self, result: Dict[str, Any], user_input: str, intent: str, issues: list
    ) -> float:
        """Score how relevant the result is to the original query."""
        score = 1.0

        user_input.lower()
        response = str(result.get("response") or "").lower()
        data = result.get("data", {})

        # Check if response addresses the query
        if not response:
            issues.append("empty response - no textual answer")
            score -= 0.3

        # Check for common irrelevance patterns
        bad_patterns = [
            "error occurred",
            "unable to",
            "failed to",
            "request timeout",
            "connection refused",
            "not found",
        ]

        for pattern in bad_patterns:
            if pattern in response and "error" not in intent:
                issues.append(f"response contains failure pattern: '{pattern}'")
                score -= 0.15
                break

        # Check if data is relevant
        if intent and "weather" in intent and isinstance(data, dict):
            expected_keys = ["temperature", "condition", "humidity"]
            found_keys = [k for k in expected_keys if k in data]
            if not found_keys:
                issues.append("weather result missing expected fields")
                score -= 0.2

        # Check for suspiciously generic responses
        generic_phrases = ["as an ai", "i cannot", "it depends", "unclear"]
        if any(phrase in response for phrase in generic_phrases):
            if len(response.split()) < 10:  # Short + generic = bad
                issues.append("response is too generic/short")
                score -= 0.15

        return max(0.0, min(1.0, score))

    def _score_completeness(self, result: Dict[str, Any], issues: list) -> float:
        """Score how complete/thorough the result is."""
        score = 1.0

        response = str(result.get("response") or "").strip()
        data = result.get("data", {})

        # Minimum word count for completeness
        response_words = len(response.split()) if response else 0
        if response_words < 5:
            issues.append("response is too short (< 5 words)")
            score -= 0.2

        # Check if data feels complete
        if isinstance(data, dict):
            if len(data) == 0:
                issues.append("result data is empty dict")
                score -= 0.15
            elif len(data) == 1:
                issues.append("result data is minimal (single field)")
                score -= 0.05  # Minor penalty

        elif isinstance(data, list):
            if len(data) == 0:
                issues.append("result data is empty list")
                score -= 0.15
            elif len(data) == 1:
                issues.append("result data has only 1 item (might be incomplete)")
                score -= 0.05

        # Check if response ends abruptly (truncation indicator)
        if response and response.endswith("..."):
            issues.append("response appears truncated")
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self, issues: list, struct_score: float, rel_score: float, comp_score: float
    ) -> list:
        """Generate actionable recommendations based on scores."""
        recommendations = []

        if struct_score < 0.7:
            recommendations.append(
                "Ensure result has required fields: success, response, data, error"
            )

        if rel_score < 0.7:
            recommendations.append(
                "Check if tool output is addressing the original query"
            )

        if comp_score < 0.7:
            recommendations.append(
                "Expand result with more detail or complete the operation"
            )

        # Add issue-specific recommendations
        for issue in issues:
            if "missing" in issue:
                recommendations.append(f"Fix: {issue}")
            elif "empty" in issue:
                recommendations.append(f"Provide data for: {issue}")

        return recommendations

    def _build_retry_reason(self, issues: list, overall: float) -> str:
        """Build human-readable retry reason."""
        if not issues:
            return f"Quality too low ({overall:.2f})"

        main_issue = issues[0]  # Dominant issue
        return f"Retry: {main_issue} (score={overall:.2f})"

    def is_quality_acceptable(self, score: QualityScore, strict: bool = False) -> bool:
        """Check if result quality is acceptable."""
        threshold = self.high_stakes_threshold if strict else self.retry_threshold
        return score.overall >= threshold

    def can_improve_with_retry(self, score: QualityScore) -> bool:
        """Check if retrying might improve results."""
        # If completely failed, retrying won't help
        if score.structure_confidence < 0.1:
            return False

        # If it's just incomplete/short, retry might help
        if "is too short" in str(score.issues):
            return True

        # If generic response, retry might give better answer
        if any("generic" in str(i) for i in score.issues):
            return True

        # If truncated, retry might complete it
        if any("truncated" in str(i) for i in score.issues):
            return True

        return False
