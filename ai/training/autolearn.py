"""
AutoLearn - Automated Learning Loop for Alice
==============================================
Continuous self-improvement driven by Ollama's evaluations.

Runs 24/7:
- Evaluates every interaction
- Learns from failures automatically
- Reinforces successful patterns
- No human intervention needed

User only audits aggregated metrics weekly.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class AutoLearn:
    """
    Automated learning loop - learns from Ollama's evaluations
    without human intervention.
    """

    def __init__(
        self,
        ollama_evaluator=None,
        learning_engine=None,
        response_formulator=None,
        realtime_logger=None,
        check_interval_hours: int = 6,
        auto_start: bool = False
    ):
        self.ollama_evaluator = ollama_evaluator
        self.learning_engine = learning_engine
        self.response_formulator = response_formulator
        self.realtime_logger = realtime_logger

        self.check_interval_hours = check_interval_hours
        self.running = False
        self.paused = False
        self.thread = None

        # Metrics
        self.cycles_completed = 0
        self.total_evaluations = 0
        self.total_improvements = 0
        self.last_run = None

        # Storage
        self.storage_path = Path("data/autolearn")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        if auto_start:
            self.start()

        logger.info(f"AutoLearn initialized - will run every {check_interval_hours} hours")

    def start(self):
        """Start the automated learning loop"""
        if self.running:
            logger.warning("AutoLearn already running")
            return

        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("AutoLearn started - automated improvement active")

    def stop(self):
        """Stop the automated learning loop"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("AutoLearn stopped")

    def pause(self):
        """Pause learning temporarily"""
        self.paused = True
        logger.info("AutoLearn paused")

    def resume(self):
        """Resume learning"""
        self.paused = False
        logger.info("AutoLearn resumed")

    def _run_loop(self):
        """Main loop - runs continuously in background"""
        logger.info("[AutoLearn] Loop started")

        while self.running:
            try:
                if not self.paused:
                    self._process_learning_cycle()

                # Sleep until next cycle
                time.sleep(self.check_interval_hours * 3600)

            except Exception as e:
                logger.error(f"[AutoLearn] Error in loop: {e}", exc_info=True)
                time.sleep(300)  # Sleep 5 minutes on error

    def _process_learning_cycle(self):
        """
        Process one learning cycle:
        1. Analyze recent evaluations from Ollama
        2. Identify patterns in failures
        3. Apply automated corrections
        4. Reinforce successful patterns
        5. Update metrics
        """
        cycle_start = datetime.now()
        logger.info("=" * 70)
        logger.info(f"[AutoLearn] CYCLE #{self.cycles_completed + 1}")
        logger.info(f"[AutoLearn] Started at: {cycle_start.isoformat()}")
        logger.info("=" * 70)

        improvements_made = 0
        stats = {
            'cycle_number': self.cycles_completed + 1,
            'timestamp': cycle_start.isoformat(),
            'evaluations_analyzed': 0,
            'improvements_made': 0,
            'patterns_reinforced': 0
        }

        try:
            # Step 1: Get evaluations since last cycle
            logger.info(f"[Step 1] Loading evaluations from last {self.check_interval_hours} hours...")

            if not self.ollama_evaluator:
                logger.warning("[AutoLearn] No evaluator available - skipping cycle")
                return

            evaluations = self.ollama_evaluator.get_recent_evaluations(
                days=self.check_interval_hours / 24
            )

            stats['evaluations_analyzed'] = len(evaluations)
            logger.info(f"[Step 1] Loaded {len(evaluations)} evaluations")

            if len(evaluations) == 0:
                logger.info("[AutoLearn] No evaluations - Alice hasn't been used recently")
                self._finalize_cycle(stats)
                return

            # Step 2: Separate failures from successes
            failures = [e for e in evaluations if e.failed]
            successes = [e for e in evaluations if e.passed]

            logger.info(f"[Step 2] Failures: {len(failures)}, Successes: {len(successes)}")

            # Step 3: Learn from failures
            if failures:
                logger.info(f"[Step 3] Learning from {len(failures)} failures...")
                improvements = self._learn_from_failures(failures)
                improvements_made += improvements
                stats['improvements_made'] = improvements

            # Step 4: Reinforce successes
            if successes:
                logger.info(f"[Step 4] Reinforcing {len(successes)} successful patterns...")
                reinforcements = self._reinforce_successes(successes)
                stats['patterns_reinforced'] = reinforcements

            # Step 5: Log cycle results
            self._finalize_cycle(stats)

            # Update metrics
            self.total_evaluations += len(evaluations)
            self.total_improvements += improvements_made

            logger.info(f"[AutoLearn] Cycle complete: {improvements_made} improvements made")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[AutoLearn] Error in cycle: {e}", exc_info=True)

    def _learn_from_failures(self, failures: List) -> int:
        """
        Automatically learn from failures.
        Uses Ollama's suggestions to improve.
        """
        improvements = 0

        # Group failures by action type
        by_action = defaultdict(list)
        for failure in failures:
            by_action[failure.action_type].append(failure)

        for action_type, action_failures in by_action.items():
            logger.info(f"  - {action_type}: {len(action_failures)} failures")

            for failure in action_failures:
                try:
                    # Learn from Ollama's suggestion
                    if failure.suggested_improvement and self.response_formulator:
                        # Add as learned example
                        self.response_formulator.phrasing_learner.record_phrasing(
                            alice_thought={
                                "type": action_type,
                                "data": failure.expected_data
                            },
                            ollama_phrasing=failure.suggested_improvement,
                            context={'tone': 'helpful'}
                        )
                        improvements += 1

                        logger.debug(f"    Learned correction for {action_type}")

                    # Log error to realtime logger for continuous learning
                    if self.realtime_logger:
                        self.realtime_logger.log_error(
                            error_type=f"{action_type}_quality",
                            user_input=failure.user_input,
                            expected=failure.suggested_improvement,
                            actual=failure.alice_response,
                            intent=action_type,
                            entities=failure.expected_data,
                            context={'ollama_score': failure.overall_score},
                            severity='high' if failure.critical_failure else 'medium'
                        )

                except Exception as e:
                    logger.error(f"  Error learning from failure: {e}")

        return improvements

    def _reinforce_successes(self, successes: List) -> int:
        """
        Reinforce successful patterns.
        Makes Alice more confident in what works.
        """
        reinforcements = 0

        # Group by action type
        by_action = defaultdict(list)
        for success in successes:
            by_action[success.action_type].append(success)

        for action_type, action_successes in by_action.items():
            # Only reinforce if consistently successful
            if len(action_successes) >= 3:
                logger.info(f"  - {action_type}: {len(action_successes)} successes - reinforcing")

                try:
                    # Log successes to realtime logger
                    if self.realtime_logger:
                        for success in action_successes:
                            self.realtime_logger.log_success(
                                event_type='evaluated_success',
                                user_input=success.user_input,
                                alice_response=success.alice_response,
                                intent=action_type,
                                route=action_type,
                                confidence=success.overall_score / 100.0
                            )

                    reinforcements += len(action_successes)

                except Exception as e:
                    logger.error(f"  Error reinforcing successes: {e}")

        return reinforcements

    def _finalize_cycle(self, stats: Dict[str, Any]):
        """Finalize cycle and save metrics"""
        self.cycles_completed += 1
        self.last_run = datetime.now().isoformat()

        # Save cycle stats
        cycle_file = self.storage_path / "cycle_history.jsonl"
        try:
            with open(cycle_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
        except Exception as e:
            logger.error(f"Failed to save cycle stats: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'running': self.running,
            'paused': self.paused,
            'check_interval_hours': self.check_interval_hours,
            'cycles_completed': self.cycles_completed,
            'total_evaluations': self.total_evaluations,
            'total_improvements': self.total_improvements,
            'last_run': self.last_run
        }

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance report for user audit.
        Shows aggregate metrics, not individual interactions.
        """
        if not self.ollama_evaluator:
            return {'error': 'No evaluator available'}

        # Get statistics from evaluator
        stats = self.ollama_evaluator.get_statistics(days=days)

        # Get recent failures for attention
        failures = self.ollama_evaluator.get_recent_evaluations(
            days=days,
            max_score=69  # Failed evaluations
        )

        # Identify problem areas
        problem_areas = {}
        for failure in failures:
            action = failure.action_type
            if action not in problem_areas:
                problem_areas[action] = {
                    'count': 0,
                    'avg_score': [],
                    'examples': []
                }

            problem_areas[action]['count'] += 1
            problem_areas[action]['avg_score'].append(failure.overall_score)

            if len(problem_areas[action]['examples']) < 3:
                problem_areas[action]['examples'].append({
                    'input': failure.user_input,
                    'response': failure.alice_response,
                    'score': failure.overall_score,
                    'issue': failure.what_needs_improvement
                })

        # Calculate averages
        for action in problem_areas:
            scores = problem_areas[action]['avg_score']
            problem_areas[action]['avg_score'] = sum(scores) / len(scores)

        return {
            'period_days': days,
            'overall_stats': stats,
            'autolearn_stats': {
                'cycles_run': self.cycles_completed,
                'total_improvements': self.total_improvements,
                'last_run': self.last_run
            },
            'problem_areas': problem_areas,
            'recommendation': self._generate_recommendation(stats, problem_areas)
        }

    def _generate_recommendation(
        self,
        stats: Dict[str, Any],
        problem_areas: Dict[str, Any]
    ) -> str:
        """Generate recommendation for user based on performance"""

        avg_score = stats.get('average_score', 0)

        if avg_score >= 90:
            return "Excellent performance. Alice is learning well across all domains."
        elif avg_score >= 80:
            return "Good performance. Minor improvements in progress."
        elif avg_score >= 70:
            return "Acceptable performance. AutoLearn is addressing issues automatically."
        elif avg_score >= 60:
            return "Below target. Review problem areas - AutoLearn may need more training data."
        else:
            return "Poor performance. Manual review recommended - check problem areas."


# Singleton instance
_autolearn = None

def get_autolearn(
    ollama_evaluator=None,
    learning_engine=None,
    response_formulator=None,
    realtime_logger=None,
    check_interval_hours: int = 6,
    auto_start: bool = False
) -> AutoLearn:
    """Get or create the AutoLearn singleton"""
    global _autolearn
    if _autolearn is None:
        _autolearn = AutoLearn(
            ollama_evaluator=ollama_evaluator,
            learning_engine=learning_engine,
            response_formulator=response_formulator,
            realtime_logger=realtime_logger,
            check_interval_hours=check_interval_hours,
            auto_start=auto_start
        )
    return _autolearn
