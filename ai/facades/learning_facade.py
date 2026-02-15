"""
Learning Facade for A.L.I.C.E
Pattern learning, evaluation, and self-improvement
"""

from ai.learning.phrasing_learner import PhrasingLearner
from ai.training.autolearn import get_autolearn
from ai.training.ollama_evaluator import get_ollama_evaluator
from ai.training.realtime_logger import RealtimeLogger
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LearningFacade:
    """Facade for all learning and self-improvement systems"""

    def __init__(self) -> None:
        # Phrasing learner
        try:
            self.phrasing = PhrasingLearner()
        except Exception as e:
            logger.error(f"Failed to initialize phrasing learner: {e}")
            self.phrasing = None

        # Evaluator
        try:
            self.evaluator = get_ollama_evaluator()
        except Exception as e:
            logger.warning(f"Evaluator not available: {e}")
            self.evaluator = None

        # Realtime logger
        try:
            self.realtime_logger = RealtimeLogger()
        except Exception as e:
            logger.warning(f"Realtime logger not available: {e}")
            self.realtime_logger = None

        # AutoLearn (deferred initialization)
        self._autolearn = None

        logger.info("[LearningFacade] Initialized learning systems")

    @property
    def autolearn(self):
        """Lazy initialize autolearn to avoid circular dependencies"""
        if self._autolearn is None:
            try:
                self._autolearn = get_autolearn(
                    ollama_evaluator=self.evaluator,
                    realtime_logger=self.realtime_logger,
                    auto_start=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize autolearn: {e}")
        return self._autolearn

    def can_phrase_independently(
        self,
        thought: Dict[str, Any],
        tone: str
    ) -> bool:
        """
        Check if Alice can phrase this thought independently

        Args:
            thought: Alice's structured thought
            tone: Desired tone

        Returns:
            True if can phrase without LLM help
        """
        if not self.phrasing:
            return False

        try:
            return self.phrasing.can_phrase_myself(thought, tone)
        except Exception as e:
            logger.error(f"Failed to check phrasing independence: {e}")
            return False

    def phrase_independently(
        self,
        thought: Dict[str, Any],
        tone: str
    ) -> Optional[str]:
        """
        Phrase thought independently using learned patterns

        Args:
            thought: Alice's structured thought
            tone: Desired tone

        Returns:
            Phrased response or None if can't phrase
        """
        if not self.phrasing:
            return None

        try:
            return self.phrasing.phrase_myself(thought, tone)
        except Exception as e:
            logger.error(f"Failed to phrase independently: {e}")
            return None

    def learn_phrasing(
        self,
        thought: Dict[str, Any],
        phrasing: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Record new phrasing example for learning

        Args:
            thought: Alice's thought structure
            phrasing: How it was phrased
            context: Context including tone

        Returns:
            True if recorded successfully
        """
        if not self.phrasing:
            return False

        try:
            self.phrasing.record_phrasing(thought, phrasing, context)
            return True
        except Exception as e:
            logger.error(f"Failed to record phrasing: {e}")
            return False

    def log_interaction(
        self,
        user_input: str,
        alice_response: str,
        intent: str,
        confidence: float,
        success: bool
    ) -> bool:
        """
        Log interaction for realtime learning

        Args:
            user_input: User's input
            alice_response: Alice's response
            intent: Classified intent
            confidence: Intent confidence
            success: Whether interaction succeeded

        Returns:
            True if logged successfully
        """
        if not self.realtime_logger:
            return False

        try:
            if success:
                self.realtime_logger.log_success(
                    event_type='interaction',
                    user_input=user_input,
                    alice_response=alice_response,
                    intent=intent,
                    route=intent,
                    confidence=confidence
                )
            else:
                self.realtime_logger.log_error(
                    error_type='interaction_failure',
                    user_input=user_input,
                    expected="successful response",
                    actual=alice_response,
                    intent=intent,
                    severity='medium'
                )
            return True
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
            return False

    def start_autolearn(self, check_interval_hours: int = 6) -> bool:
        """
        Start automated learning loop

        Args:
            check_interval_hours: Hours between learning cycles

        Returns:
            True if started successfully
        """
        if not self.autolearn:
            return False

        try:
            self.autolearn.check_interval_hours = check_interval_hours
            self.autolearn.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start autolearn: {e}")
            return False

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics

        Returns:
            Statistics dictionary
        """
        stats = {}

        # Phrasing stats
        if self.phrasing:
            try:
                stats['phrasing'] = self.phrasing.get_stats()
            except Exception as e:
                logger.error(f"Failed to get phrasing stats: {e}")
                stats['phrasing'] = {"error": str(e)}

        # AutoLearn stats
        if self.autolearn:
            try:
                stats['autolearn'] = self.autolearn.get_status()
            except Exception as e:
                logger.error(f"Failed to get autolearn stats: {e}")
                stats['autolearn'] = {"error": str(e)}

        # Evaluator stats
        if self.evaluator:
            try:
                stats['evaluator'] = self.evaluator.get_statistics(days=7)
            except Exception as e:
                logger.error(f"Failed to get evaluator stats: {e}")
                stats['evaluator'] = {"error": str(e)}

        return stats


# Singleton instance
_learning_facade: Optional[LearningFacade] = None


def get_learning_facade() -> LearningFacade:
    """Get or create the LearningFacade singleton"""
    global _learning_facade
    if _learning_facade is None:
        _learning_facade = LearningFacade()
    return _learning_facade
