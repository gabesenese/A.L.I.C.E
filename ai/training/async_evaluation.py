"""
Async Evaluation Wrapper
=========================
Non-blocking evaluation of Alice's responses.

After Alice responds to user:
1. Return response immediately (don't block)
2. Async: Ollama evaluates response
3. Async: Alice learns from evaluation
4. Async: Log for audit

User never waits for evaluation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncEvaluationWrapper:
    """
    Wraps evaluation in async calls so user isn't blocked.
    Ollama evaluates after response is returned.
    """

    def __init__(
        self,
        ollama_evaluator=None,
        response_formulator=None,
        realtime_logger=None
    ):
        self.ollama_evaluator = ollama_evaluator
        self.response_formulator = response_formulator
        self.realtime_logger = realtime_logger

        self.evaluation_queue = []
        self.processing = False

    def queue_evaluation(
        self,
        user_input: str,
        alice_response: str,
        plugin_result: Dict[str, Any],
        alice_confidence: float = 0.5
    ):
        """
        Queue evaluation for async processing.
        Returns immediately - doesn't block user.
        """

        evaluation_task = {
            'user_input': user_input,
            'alice_response': alice_response,
            'action': plugin_result.get('action', 'unknown'),
            'data': plugin_result.get('data', {}),
            'success': plugin_result.get('success', True),
            'alice_confidence': alice_confidence,
            'timestamp': datetime.now().isoformat()
        }

        self.evaluation_queue.append(evaluation_task)

        # Process async (non-blocking)
        try:
            import threading
            thread = threading.Thread(
                target=self._process_evaluation,
                args=(evaluation_task,),
                daemon=True
            )
            thread.start()
        except Exception as e:
            logger.error(f"Failed to queue evaluation: {e}")

    def _process_evaluation(self, task: Dict[str, Any]):
        """
        Process evaluation in background thread.
        User never waits for this.
        """
        try:
            if not self.ollama_evaluator:
                return

            # Ollama evaluates (this is slow, but user already has response)
            evaluation = self.ollama_evaluator.evaluate_response(
                user_input=task['user_input'],
                alice_response=task['alice_response'],
                expected_data=task['data'],
                action_type=task['action'],
                alice_confidence=task['alice_confidence']
            )

            # Alice learns from evaluation
            if evaluation.failed and evaluation.suggested_improvement:
                self._apply_learning(task, evaluation)

            logger.debug(
                f"[AsyncEval] Evaluated: {task['action']} "
                f"score={evaluation.overall_score}/100"
            )

        except Exception as e:
            logger.error(f"[AsyncEval] Error processing evaluation: {e}")

    def _apply_learning(self, task: Dict[str, Any], evaluation):
        """Apply learning from evaluation"""
        try:
            if not self.response_formulator:
                return

            # Learn from Ollama's suggestion
            self.response_formulator.phrasing_learner.learn_from_example(
                alice_thought={
                    "type": task['action'],
                    "data": task['data']
                },
                ollama_phrasing=evaluation.suggested_improvement,
                tone="helpful"
            )

            logger.debug(f"[AsyncEval] Learned correction for {task['action']}")

        except Exception as e:
            logger.error(f"[AsyncEval] Error applying learning: {e}")


# Singleton instance
_async_eval = None

def get_async_evaluator(
    ollama_evaluator=None,
    response_formulator=None,
    realtime_logger=None
) -> AsyncEvaluationWrapper:
    """Get or create the async evaluator singleton"""
    global _async_eval
    if _async_eval is None:
        _async_eval = AsyncEvaluationWrapper(
            ollama_evaluator=ollama_evaluator,
            response_formulator=response_formulator,
            realtime_logger=realtime_logger
        )
    return _async_eval
