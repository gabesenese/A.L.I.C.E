"""
Ollama Evaluator for Alice
===========================
Ollama automatically evaluates Alice's responses and provides feedback.
User only audits aggregated metrics, not individual interactions.

Philosophy:
- Ollama is the teacher, user is the auditor
- Automated feedback loop for continuous improvement
- Scores every interaction (0-100)
- Generates corrections for failures
- No human intervention needed for learning
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    """Ollama's evaluation of Alice's response"""
    interaction_id: str
    timestamp: str

    # Input/Output
    user_input: str
    alice_response: str
    expected_data: Dict[str, Any]

    # Evaluation scores (0-100)
    overall_score: int
    accuracy_score: int      # Did she understand correctly?
    completeness_score: int  # Did she address everything?
    naturalness_score: int   # Does it sound human?
    conciseness_score: int   # Not too verbose?

    # Feedback
    what_worked: str
    what_needs_improvement: str
    suggested_improvement: Optional[str]

    # Metadata
    action_type: str
    alice_confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def passed(self) -> bool:
        """Score >= 85 is passing"""
        return self.overall_score >= 85

    @property
    def failed(self) -> bool:
        """Score < 70 is failing"""
        return self.overall_score < 70

    @property
    def critical_failure(self) -> bool:
        """Score < 50 requires human audit"""
        return self.overall_score < 50


class OllamaEvaluator:
    """
    Uses Ollama to automatically evaluate and teach Alice.
    Provides continuous feedback without human intervention.
    """

    def __init__(
        self,
        llm_engine=None,
        storage_path: str = "data/evaluations"
    ):
        self.llm = llm_engine
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Evaluation logs (append-only)
        self.eval_log = self.storage_path / "evaluations.jsonl"

        # Calibration data (track Ollama's evaluation patterns)
        self.calibration_data = self.storage_path / "calibration.json"

        logger.info("OllamaEvaluator initialized - automated feedback active")

    def evaluate_response(
        self,
        user_input: str,
        alice_response: str,
        expected_data: Dict[str, Any],
        action_type: str,
        alice_confidence: float = 0.5,
        context: Dict[str, Any] = None
    ) -> Evaluation:
        """
        Ollama evaluates Alice's response quality.

        Returns detailed evaluation with scores and suggestions.
        """

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            user_input=user_input,
            alice_response=alice_response,
            expected_data=expected_data,
            context=context or {}
        )

        # Get Ollama's evaluation
        try:
            raw_evaluation = self.llm.query(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=500
            )

            # Parse evaluation
            evaluation = self._parse_evaluation(
                raw_evaluation=raw_evaluation,
                user_input=user_input,
                alice_response=alice_response,
                expected_data=expected_data,
                action_type=action_type,
                alice_confidence=alice_confidence
            )

            # Log evaluation
            self._log_evaluation(evaluation)

            logger.info(f"[Evaluator] Score: {evaluation.overall_score}/100 for {action_type}")

            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return neutral evaluation on error
            return self._create_neutral_evaluation(
                user_input=user_input,
                alice_response=alice_response,
                expected_data=expected_data,
                action_type=action_type,
                alice_confidence=alice_confidence
            )

    def _build_evaluation_prompt(
        self,
        user_input: str,
        alice_response: str,
        expected_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for Ollama to evaluate Alice's response"""

        return f"""You are evaluating Alice, an AI assistant's response quality.

USER REQUEST: "{user_input}"

ALICE'S RESPONSE: "{alice_response}"

EXPECTED TO ACCOMPLISH: {json.dumps(expected_data, indent=2)}

CONTEXT: {json.dumps(context, indent=2) if context else "None"}

Evaluate Alice's response on four criteria (0-100 each):

1. ACCURACY: Did Alice understand the request correctly and respond appropriately?
   - 100: Perfect understanding, correct action
   - 75: Mostly correct, minor misunderstanding
   - 50: Partially correct, significant confusion
   - 25: Misunderstood the request
   - 0: Completely wrong

2. COMPLETENESS: Did Alice address all aspects of the request?
   - 100: Fully complete, nothing missing
   - 75: Mostly complete, minor omission
   - 50: Partial response, missing key info
   - 25: Incomplete, major gaps
   - 0: Barely addressed request

3. NATURALNESS: Does the response sound human and conversational?
   - 100: Perfectly natural, human-like
   - 75: Mostly natural, slightly robotic
   - 50: Somewhat natural, clearly AI
   - 25: Awkward phrasing
   - 0: Robotic, unnatural

4. CONCISENESS: Is the response appropriately brief?
   - 100: Perfect length, no fluff
   - 75: Slightly verbose but acceptable
   - 50: Too verbose or too terse
   - 25: Significantly too long/short
   - 0: Extremely verbose or incomplete

Format your response EXACTLY as:
ACCURACY: [0-100]
COMPLETENESS: [0-100]
NATURALNESS: [0-100]
CONCISENESS: [0-100]
OVERALL: [0-100]
GOOD: [What Alice did well, be specific]
IMPROVE: [What needs improvement, be specific]
SUGGESTION: [Better version of the response if overall < 85, or "None" if good]

Be objective and constructive. Focus on helping Alice improve."""

    def _parse_evaluation(
        self,
        raw_evaluation: str,
        user_input: str,
        alice_response: str,
        expected_data: Dict[str, Any],
        action_type: str,
        alice_confidence: float
    ) -> Evaluation:
        """Parse Ollama's evaluation response"""

        # Extract scores using regex
        accuracy = self._extract_score(raw_evaluation, "ACCURACY")
        completeness = self._extract_score(raw_evaluation, "COMPLETENESS")
        naturalness = self._extract_score(raw_evaluation, "NATURALNESS")
        conciseness = self._extract_score(raw_evaluation, "CONCISENESS")
        overall = self._extract_score(raw_evaluation, "OVERALL")

        # Extract feedback
        good = self._extract_field(raw_evaluation, "GOOD")
        improve = self._extract_field(raw_evaluation, "IMPROVE")
        suggestion = self._extract_field(raw_evaluation, "SUGGESTION")

        # Clean up suggestion
        if suggestion and suggestion.lower() in ["none", "n/a", "not needed"]:
            suggestion = None

        return Evaluation(
            interaction_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            alice_response=alice_response,
            expected_data=expected_data,
            overall_score=overall,
            accuracy_score=accuracy,
            completeness_score=completeness,
            naturalness_score=naturalness,
            conciseness_score=conciseness,
            what_worked=good or "No specific feedback",
            what_needs_improvement=improve or "No issues identified",
            suggested_improvement=suggestion,
            action_type=action_type,
            alice_confidence=alice_confidence
        )

    def _extract_score(self, text: str, field: str) -> int:
        """Extract numeric score from evaluation text"""
        pattern = rf'{field}:\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(100, score))  # Clamp to 0-100
        return 50  # Default middle score if not found

    def _extract_field(self, text: str, field: str) -> str:
        """Extract text field from evaluation"""
        pattern = rf'{field}:\s*(.+?)(?=\n[A-Z]+:|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _generate_id(self) -> str:
        """Generate unique evaluation ID"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"eval_{timestamp}"

    def _log_evaluation(self, evaluation: Evaluation):
        """Log evaluation to JSONL file"""
        try:
            with open(self.eval_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(evaluation.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")

    def _create_neutral_evaluation(
        self,
        user_input: str,
        alice_response: str,
        expected_data: Dict[str, Any],
        action_type: str,
        alice_confidence: float
    ) -> Evaluation:
        """Create neutral evaluation when Ollama evaluation fails"""
        return Evaluation(
            interaction_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            alice_response=alice_response,
            expected_data=expected_data,
            overall_score=50,
            accuracy_score=50,
            completeness_score=50,
            naturalness_score=50,
            conciseness_score=50,
            what_worked="Evaluation failed",
            what_needs_improvement="Could not evaluate",
            suggested_improvement=None,
            action_type=action_type,
            alice_confidence=alice_confidence
        )

    def get_recent_evaluations(
        self,
        days: int = 7,
        min_score: int = 0,
        max_score: int = 100,
        action_type: Optional[str] = None
    ) -> List[Evaluation]:
        """Load recent evaluations with filtering"""

        if not self.eval_log.exists():
            return []

        evaluations = []
        cutoff = datetime.now().timestamp() - (days * 86400)

        try:
            with open(self.eval_log, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        eval_time = datetime.fromisoformat(data['timestamp']).timestamp()

                        # Filter by date
                        if eval_time < cutoff:
                            continue

                        # Filter by score
                        if not (min_score <= data['overall_score'] <= max_score):
                            continue

                        # Filter by action type
                        if action_type and data['action_type'] != action_type:
                            continue

                        # Reconstruct evaluation object
                        evaluations.append(Evaluation(**data))

        except Exception as e:
            logger.error(f"Failed to load evaluations: {e}")

        return evaluations

    def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get evaluation statistics for audit dashboard"""

        evaluations = self.get_recent_evaluations(days=days)

        if not evaluations:
            return {
                'total': 0,
                'average_score': 0,
                'passing_rate': 0,
                'failing_rate': 0
            }

        total = len(evaluations)
        avg_score = sum(e.overall_score for e in evaluations) / total
        passing = sum(1 for e in evaluations if e.passed)
        failing = sum(1 for e in evaluations if e.failed)
        critical = sum(1 for e in evaluations if e.critical_failure)

        # By action type
        by_action = {}
        for eval in evaluations:
            action = eval.action_type
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(eval.overall_score)

        action_stats = {
            action: {
                'count': len(scores),
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            }
            for action, scores in by_action.items()
        }

        return {
            'total': total,
            'average_score': round(avg_score, 1),
            'passing_rate': round(passing / total * 100, 1),
            'failing_rate': round(failing / total * 100, 1),
            'critical_failures': critical,
            'by_action': action_stats
        }


# Singleton instance
_evaluator = None

def get_ollama_evaluator(llm_engine=None) -> OllamaEvaluator:
    """Get or create the Ollama evaluator singleton"""
    global _evaluator
    if _evaluator is None:
        _evaluator = OllamaEvaluator(llm_engine=llm_engine)
    return _evaluator
