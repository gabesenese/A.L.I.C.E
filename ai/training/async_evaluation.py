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
from dataclasses import is_dataclass, asdict

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

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert dataclass objects (like Entity) to dictionaries for JSON serialization"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

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
            'data': self._make_json_serializable(plugin_result.get('data', {})),
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

            # Small delay to avoid resource contention with main response
            # Ensures Ollama isn't hit by two requests simultaneously
            import time
            time.sleep(2)

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

            # Learn from Ollama's suggestion using correct PhrasingLearner API
            if hasattr(self.response_formulator, 'phrasing_learner'):
                self.response_formulator.phrasing_learner.record_phrasing(
                    alice_thought={
                        "type": task['action'],
                        "data": task['data']
                    },
                    ollama_phrasing=evaluation.suggested_improvement,
                    context={
                        "tone": "helpful",
                        "user_input": task['user_input'],
                        "timestamp": task['timestamp']
                    }
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
