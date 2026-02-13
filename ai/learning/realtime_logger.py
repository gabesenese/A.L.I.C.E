"""
Real-Time Learning Logger for A.L.I.C.E
========================================
Continuous learning approach: Learn immediately, not just nightly.

Logs errors and learning opportunities immediately as they happen,
enabling continuous learning throughout the day, not just nightly.

Key Features:
- Immediate error logging (no waiting for batch processing)
- Structured error format for easy analysis
- Learning velocity metrics
- Automatic micro-corrections every 6 hours
- No-downtime learning (updates patterns without restarting)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from threading import Lock
import os
import sys

# Platform-specific file locking
if sys.platform == 'win32':
    import msvcrt
    HAS_FCNTL = False
    HAS_MSVCRT = True
else:
    try:
        import fcntl
        HAS_FCNTL = True
        HAS_MSVCRT = False
    except ImportError:
        HAS_FCNTL = False
        HAS_MSVCRT = False

logger = logging.getLogger(__name__)


@dataclass
class RealtimeError:
    """Real-time error record"""
    timestamp: str
    error_type: str  # intent_mismatch, route_mismatch, plugin_failure, llm_error
    user_input: str
    expected: Optional[str]
    actual: Optional[str]
    intent: str
    entities: Dict[str, Any]
    context: Dict[str, Any]
    severity: str  # critical, high, medium, low
    session_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LearningEvent:
    """Positive learning event - what worked well"""
    timestamp: str
    event_type: str  # successful_pattern, user_satisfied, goal_completed
    user_input: str
    alice_response: str
    intent: str
    route: str
    confidence: float
    user_feedback: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RealtimeLearningLogger:
    """
    Continuous learning logger for immediate error capture and learning.
    """

    def __init__(self, storage_path: str = "data/realtime_learning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Error log (append-only, fast)
        self.error_log = self.storage_path / "errors_realtime.jsonl"

        # Success log (what worked well)
        self.success_log = self.storage_path / "successes_realtime.jsonl"

        # Metrics
        self.metrics_file = self.storage_path / "learning_velocity.json"

        # Thread safety for concurrent writes
        self.write_lock = Lock()

        # In-memory buffer for fast access
        self.recent_errors: List[RealtimeError] = []
        self.max_buffer_size = 100

        logger.info("[RealtimeLogger] Initialized - Learning 24/7 mode active")

    def log_error(
        self,
        error_type: str,
        user_input: str,
        expected: Optional[str],
        actual: Optional[str],
        intent: str,
        entities: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
        severity: str = "medium",
        session_id: Optional[str] = None
    ):
        """
        Log an error immediately - no batching, no delay.

        Every mistake is a learning opportunity - learn from it immediately.
        """
        error = RealtimeError(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            user_input=user_input,
            expected=expected,
            actual=actual,
            intent=intent,
            entities=entities or {},
            context=context or {},
            severity=severity,
            session_id=session_id
        )

        # Write immediately to disk (append-only, fast)
        self._write_to_log(self.error_log, error.to_dict())

        # Add to in-memory buffer
        with self.write_lock:
            self.recent_errors.append(error)
            if len(self.recent_errors) > self.max_buffer_size:
                self.recent_errors.pop(0)

        # Update metrics
        self._update_metrics('error')

        logger.debug(f"[RealtimeLogger] Logged {error_type}: {user_input[:50]}...")

    def log_success(
        self,
        event_type: str,
        user_input: str,
        alice_response: str,
        intent: str,
        route: str,
        confidence: float,
        user_feedback: Optional[str] = None
    ):
        """
        Log successful interactions - learn from what works.

        Success leaves clues - track what's working.
        """
        success = LearningEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_input=user_input,
            alice_response=alice_response,
            intent=intent,
            route=route,
            confidence=confidence,
            user_feedback=user_feedback
        )

        self._write_to_log(self.success_log, success.to_dict())
        self._update_metrics('success')

    def _write_to_log(self, log_file: Path, data: Dict):
        """Write to log with platform-independent file locking for concurrent access"""
        try:
            with self.write_lock:
                with open(log_file, 'a', encoding='utf-8') as f:
                    # Platform-specific file locking
                    if HAS_FCNTL:
                        # Unix/Linux/Mac file locking
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except (IOError, OSError):
                            pass
                    elif HAS_MSVCRT:
                        # Windows file locking
                        try:
                            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        except (IOError, OSError):
                            pass

                    f.write(json.dumps(data) + '\n')
                    f.flush()

                    # Release lock
                    if HAS_FCNTL:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except (IOError, OSError):
                            pass
                    elif HAS_MSVCRT:
                        try:
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        except (IOError, OSError):
                            pass
        except Exception as e:
            logger.error(f"[RealtimeLogger] Error writing to log: {e}")

    def _update_metrics(self, event_type: str):
        """Update learning velocity metrics"""
        try:
            # Load current metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    'total_errors': 0,
                    'total_successes': 0,
                    'last_updated': datetime.now().isoformat(),
                    'errors_per_hour': [],
                    'learning_velocity': 0.0
                }

            # Update counts
            if event_type == 'error':
                metrics['total_errors'] += 1
            else:
                metrics['total_successes'] += 1

            metrics['last_updated'] = datetime.now().isoformat()

            # Calculate learning velocity (errors per hour)
            current_hour = datetime.now().strftime('%Y-%m-%d %H:00')
            if not metrics['errors_per_hour'] or metrics['errors_per_hour'][-1]['hour'] != current_hour:
                metrics['errors_per_hour'].append({'hour': current_hour, 'count': 1})
            else:
                metrics['errors_per_hour'][-1]['count'] += 1

            # Keep only last 24 hours
            metrics['errors_per_hour'] = metrics['errors_per_hour'][-24:]

            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.error(f"[RealtimeLogger] Error updating metrics: {e}")

    def get_recent_errors(self, count: int = 10) -> List[Dict]:
        """Get recent errors from buffer (fast)"""
        with self.write_lock:
            return [e.to_dict() for e in self.recent_errors[-count:]]

    def get_errors_since(self, hours: int = 6) -> List[Dict]:
        """
        Get errors from last N hours for micro-correction processing.

        Regular review and learning from recent mistakes.
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        errors = []

        if not self.error_log.exists():
            return errors

        with open(self.error_log, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    error = json.loads(line)
                    error_time = datetime.fromisoformat(error['timestamp'])
                    if error_time >= cutoff:
                        errors.append(error)

        return errors

    def get_learning_velocity(self) -> Dict[str, Any]:
        """
        Get learning velocity metrics

        Returns:
            - Errors per hour (trending up/down)
            - Success rate
            - Learning opportunities identified
        """
        if not self.metrics_file.exists():
            return {
                'errors_per_hour': [],
                'total_errors': 0,
                'total_successes': 0,
                'success_rate': 0.0,
                'trend': 'stable'
            }

        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)

        # Calculate success rate
        total = metrics['total_errors'] + metrics['total_successes']
        success_rate = metrics['total_successes'] / total if total > 0 else 0.0

        # Determine trend
        if len(metrics['errors_per_hour']) >= 2:
            recent_avg = sum(h['count'] for h in metrics['errors_per_hour'][-3:]) / 3
            older_avg = sum(h['count'] for h in metrics['errors_per_hour'][-6:-3]) / 3 if len(metrics['errors_per_hour']) >= 6 else recent_avg

            if recent_avg < older_avg * 0.8:
                trend = 'improving'
            elif recent_avg > older_avg * 1.2:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'errors_per_hour': metrics['errors_per_hour'],
            'total_errors': metrics['total_errors'],
            'total_successes': metrics['total_successes'],
            'success_rate': success_rate,
            'trend': trend
        }

    def clear_old_logs(self, days: int = 7):
        """Clear logs older than N days to save space"""
        cutoff = datetime.now() - timedelta(days=days)

        for log_file in [self.error_log, self.success_log]:
            if not log_file.exists():
                continue

            temp_file = log_file.with_suffix('.tmp')
            kept = 0

            with open(log_file, 'r') as f_in, open(temp_file, 'w') as f_out:
                for line in f_in:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time >= cutoff:
                                f_out.write(line)
                                kept += 1
                        except Exception:
                            pass  # Skip malformed lines

            # Replace old file with cleaned version
            temp_file.replace(log_file)
            logger.info(f"[RealtimeLogger] Cleaned {log_file.name}, kept {kept} entries")


# Global singleton
_realtime_logger = None


def get_realtime_logger(storage_path: str = "data/realtime_learning") -> RealtimeLearningLogger:
    """Get or create global realtime logger"""
    global _realtime_logger
    if _realtime_logger is None:
        _realtime_logger = RealtimeLearningLogger(storage_path=storage_path)
    return _realtime_logger
