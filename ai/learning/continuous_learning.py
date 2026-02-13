"""
Continuous Learning Loop for A.L.I.C.E
======================================
Continuous improvement philosophy: Alice learns and improves autonomously.

Background thread that continuously learns from real-time errors:
- Runs every 6 hours automatically
- Applies micro-corrections without restarting Alice
- Tracks learning velocity
- Self-adjusts based on what's working

Key Features:
- No-downtime learning
- Automatic pattern updates
- Learning velocity monitoring
- Self-optimization
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ContinuousLearningLoop:
    """
    24/7 continuous learning background thread for autonomous improvement.
    """

    def __init__(
        self,
        learning_engine=None,
        realtime_logger=None,
        check_interval_hours: int = 6,
        auto_start: bool = False
    ):
        """
        Initialize continuous learning loop

        Args:
            learning_engine: LearningEngine instance for applying corrections
            realtime_logger: RealtimeLearningLogger instance
            check_interval_hours: How often to process errors (default: 6 hours)
            auto_start: Start the loop immediately
        """
        self.learning_engine = learning_engine
        self.realtime_logger = realtime_logger
        self.check_interval_hours = check_interval_hours

        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.paused = False

        # Statistics
        self.cycles_completed = 0
        self.total_corrections_applied = 0
        self.last_run = None

        # Storage for tracking
        self.stats_file = Path("data/realtime_learning/continuous_stats.json")
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        self._load_stats()

        if auto_start:
            self.start()

    def start(self):
        """Start the continuous learning loop"""
        if self.running:
            logger.warning("[ContinuousLearning] Already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.thread.start()

        logger.info(f"[ContinuousLearning] Started - checking every {self.check_interval_hours} hours")
        logger.info("[ContinuousLearning] Alice will now learn continuously, 24/7")

    def stop(self):
        """Stop the continuous learning loop"""
        if not self.running:
            return

        logger.info("[ContinuousLearning] Stopping continuous learning loop...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=5)

        self._save_stats()
        logger.info("[ContinuousLearning] Stopped")

    def pause(self):
        """Pause learning (still running, but not processing)"""
        self.paused = True
        logger.info("[ContinuousLearning] Paused")

    def resume(self):
        """Resume learning"""
        self.paused = False
        logger.info("[ContinuousLearning] Resumed")

    def _learning_loop(self):
        """
        Main continuous learning loop

        Runs every N hours, processes recent errors, applies micro-corrections
        """
        logger.info("[ContinuousLearning] Learning loop started")

        while self.running:
            try:
                if not self.paused:
                    # Process recent errors and apply corrections
                    self._process_learning_cycle()

                # Sleep until next check (check every minute if we should run)
                sleep_time = self.check_interval_hours * 3600

                for _ in range(int(sleep_time / 60)):  # Check every minute
                    if not self.running:
                        break
                    time.sleep(60)

            except Exception as e:
                logger.error(f"[ContinuousLearning] Error in learning loop: {e}", exc_info=True)
                time.sleep(300)  # Sleep 5 minutes on error

    def _process_learning_cycle(self):
        """
        Process one learning cycle:
        1. Gather errors from last 6 hours
        2. Identify patterns
        3. Apply micro-corrections
        4. Update metrics
        5. Self-evaluate results
        """
        cycle_start = datetime.now()
        logger.info("=" * 70)
        logger.info(f"[ContinuousLearning] LEARNING CYCLE #{self.cycles_completed + 1}")
        logger.info(f"[ContinuousLearning] Started at: {cycle_start.isoformat()}")
        logger.info("=" * 70)

        corrections_applied = 0
        stats = {
            'cycle_number': self.cycles_completed + 1,
            'timestamp': cycle_start.isoformat(),
            'errors_processed': 0,
            'corrections_applied': 0,
            'patterns_identified': 0
        }

        try:
            # Step 1: Gather recent errors
            logger.info(f"[Step 1] Gathering errors from last {self.check_interval_hours} hours...")
            errors = self.realtime_logger.get_errors_since(hours=self.check_interval_hours)
            stats['errors_processed'] = len(errors)

            logger.info(f"[Step 1] Found {len(errors)} errors to learn from")

            if len(errors) == 0:
                logger.info("[ContinuousLearning] No errors found - Alice is performing well!")
                self._finalize_cycle(stats)
                return

            # Step 2: Group errors by type
            logger.info("[Step 2] Analyzing error patterns...")
            error_groups = self._group_errors(errors)

            for error_type, error_list in error_groups.items():
                logger.info(f"  - {error_type}: {len(error_list)} occurrences")

            # Step 3: Apply micro-corrections
            logger.info("[Step 3] Applying micro-corrections...")

            for error_type, error_list in error_groups.items():
                if len(error_list) >= 2:  # Only correct if pattern appears 2+ times
                    corrections = self._apply_micro_corrections(error_type, error_list)
                    corrections_applied += corrections
                    logger.info(f"  - {error_type}: {corrections} corrections applied")

            stats['corrections_applied'] = corrections_applied

            # Step 4: Update learning velocity
            logger.info("[Step 4] Updating learning metrics...")
            velocity = self.realtime_logger.get_learning_velocity()
            logger.info(f"  - Learning trend: {velocity['trend']}")
            logger.info(f"  - Success rate: {velocity['success_rate']:.2%}")

            # Step 5: Self-evaluation
            logger.info("[Step 5] Self-evaluating corrections...")
            if velocity['trend'] == 'improving':
                logger.info("  - Corrections are working! Error rate decreasing.")
            elif velocity['trend'] == 'degrading':
                logger.warning("  - Error rate increasing. Need to adjust learning strategy.")
            else:
                logger.info("  - Error rate stable.")

        except Exception as e:
            logger.error(f"[ContinuousLearning] Error in learning cycle: {e}", exc_info=True)

        # Finalize
        self._finalize_cycle(stats)

        duration = (datetime.now() - cycle_start).total_seconds()
        logger.info("=" * 70)
        logger.info(f"[ContinuousLearning] CYCLE COMPLETE in {duration:.1f}s")
        logger.info(f"[ContinuousLearning] Total corrections applied: {corrections_applied}")
        logger.info(f"[ContinuousLearning] Next cycle in {self.check_interval_hours} hours")
        logger.info("=" * 70)

    def _group_errors(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group errors by type for pattern identification"""
        groups = {}

        for error in errors:
            error_type = error['error_type']
            if error_type not in groups:
                groups[error_type] = []
            groups[error_type].append(error)

        return groups

    def _apply_micro_corrections(self, error_type: str, errors: List[Dict]) -> int:
        """
        Apply micro-corrections based on error patterns.

        Small, frequent adjustments beat big, infrequent overhauls.
        """
        if not self.learning_engine:
            logger.warning("[ContinuousLearning] No learning engine available for corrections")
            return 0

        corrections_applied = 0

        try:
            if error_type == 'intent_mismatch':
                # Correct intent classification errors
                for error in errors:
                    if error.get('expected') and error.get('actual'):
                        self.learning_engine.add_correction(
                            user_input=error['user_input'],
                            wrong_intent=error['actual'],
                            correct_intent=error['expected'],
                            confidence=0.7  # Micro-corrections have moderate confidence
                        )
                        corrections_applied += 1

            elif error_type == 'route_mismatch':
                # Correct routing errors
                for error in errors:
                    if error.get('expected') and error.get('actual'):
                        # Log for nightly processing (routes are more complex)
                        pass  # Routes handled by nightly training

            elif error_type == 'plugin_failure':
                # Log plugin failures for investigation
                logger.warning(f"[ContinuousLearning] Plugin failures detected: {len(errors)}")
                # Don't auto-correct plugin failures - need manual review

        except Exception as e:
            logger.error(f"[ContinuousLearning] Error applying corrections: {e}")

        return corrections_applied

    def _finalize_cycle(self, stats: Dict):
        """Finalize learning cycle and update statistics"""
        self.cycles_completed += 1
        self.total_corrections_applied += stats['corrections_applied']
        self.last_run = datetime.now()

        self._save_stats()

        # Save cycle stats
        cycle_log = Path("data/realtime_learning/learning_cycles.jsonl")
        with open(cycle_log, 'a') as f:
            f.write(json.dumps(stats) + '\n')

    def _load_stats(self):
        """Load statistics from disk"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                    self.cycles_completed = stats.get('cycles_completed', 0)
                    self.total_corrections_applied = stats.get('total_corrections_applied', 0)
                    last_run_str = stats.get('last_run')
                    if last_run_str:
                        self.last_run = datetime.fromisoformat(last_run_str)
            except Exception as e:
                logger.error(f"[ContinuousLearning] Error loading stats: {e}")

    def _save_stats(self):
        """Save statistics to disk"""
        stats = {
            'cycles_completed': self.cycles_completed,
            'total_corrections_applied': self.total_corrections_applied,
            'last_run': self.last_run.isoformat() if self.last_run else None
        }

        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"[ContinuousLearning] Error saving stats: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of continuous learning"""
        return {
            'running': self.running,
            'paused': self.paused,
            'cycles_completed': self.cycles_completed,
            'total_corrections_applied': self.total_corrections_applied,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'check_interval_hours': self.check_interval_hours
        }


# Global singleton
_continuous_loop = None


def get_continuous_learning_loop(
    learning_engine=None,
    realtime_logger=None,
    check_interval_hours: int = 6,
    auto_start: bool = False
) -> ContinuousLearningLoop:
    """Get or create global continuous learning loop"""
    global _continuous_loop
    if _continuous_loop is None:
        _continuous_loop = ContinuousLearningLoop(
            learning_engine=learning_engine,
            realtime_logger=realtime_logger,
            check_interval_hours=check_interval_hours,
            auto_start=auto_start
        )
    return _continuous_loop
