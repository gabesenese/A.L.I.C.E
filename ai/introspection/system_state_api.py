"""
System State API - Tier 3: Live Architecture Introspection

Provides runtime query API for system state.
Enables ALICE to describe what she's actually doing right now.
"""

import logging
import psutil
import os
from typing import Dict, Any, List
from dataclasses import asdict
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemStateAPI:
    """API for querying ALICE's live state at runtime."""

    def __init__(self, alice_ref=None):
        """
        Args:
            alice_ref: Reference to main ALICE instance (set after ALICE init)
        """
        self.alice = alice_ref
        self.state_snapshot_timestamp = None

    def set_alice_reference(self, alice_instance) -> None:
        """Set reference to main ALICE after initialization."""
        self.alice = alice_instance
        logger.info("[SystemState] Alice reference set")

    def get_processor_state(self) -> Dict[str, Any]:
        """Get state of NLP/reasoning processors."""
        if not self.alice:
            return {}

        return {
            "nlp_processor": (
                "active"
                if hasattr(self.alice, "nlp") and self.alice.nlp
                else "inactive"
            ),
            "router_initialized": hasattr(self.alice, "router")
            and self.alice.router is not None,
            "reasoning_engine": (
                "active"
                if (
                    getattr(self.alice, "reasoning_engine", None)
                    or getattr(self.alice, "reasoning", None)
                )
                else "inactive"
            ),
            "learning_engine": (
                "active"
                if (
                    getattr(self.alice, "learning_engine", None)
                    or getattr(self.alice, "learning", None)
                )
                else "inactive"
            ),
        }

    def get_memory_state(self) -> Dict[str, Any]:
        """Get state of memory systems."""
        if not self.alice or not hasattr(self.alice, "memory"):
            return {}

        mem_sys = self.alice.memory

        state = {
            "memory_initialized": mem_sys is not None,
            "episodic_memories": 0,
            "semantic_memories": 0,
            "total_embeddings": 0,
            "memory_usage_mb": 0.0,
        }

        if mem_sys:
            try:
                # Estimate memory entries
                if hasattr(mem_sys, "episodic_memory"):
                    state["episodic_memories"] = (
                        len(mem_sys.episodic_memory)
                        if hasattr(mem_sys.episodic_memory, "__len__")
                        else 0
                    )

                if hasattr(mem_sys, "semantic_memory"):
                    state["semantic_memories"] = (
                        len(mem_sys.semantic_memory)
                        if hasattr(mem_sys.semantic_memory, "__len__")
                        else 0
                    )

                # Get process memory
                process = psutil.Process(os.getpid())
                state["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
            except Exception as e:
                logger.debug(f"[SystemState] Memory stat error: {e}")

        return state

    def get_active_goals(self) -> List[str]:
        """Get list of actively tracked goals."""
        if not self.alice:
            return []

        # Try multiple goal sources
        if hasattr(self.alice, "goal_system") and self.alice.goal_system:
            if hasattr(self.alice.goal_system, "get_active_goals"):
                try:
                    goals = self.alice.goal_system.get_active_goals()
                    return [self._format_goal(goal) for goal in list(goals or [])[:5]]
                except Exception:
                    pass
            if hasattr(self.alice.goal_system, "active_goals"):
                return [
                    self._format_goal(goal)
                    for goal in list(self.alice.goal_system.active_goals or [])[:5]
                ]

        summarizer = getattr(
            self.alice,
            "session_summarizer",
            getattr(self.alice, "_session_summarizer", None),
        )
        if summarizer and hasattr(summarizer, "active_goals"):
            return [
                self._format_goal(goal)
                for goal in list(summarizer.active_goals or [])[:5]
            ]

        return []

    @staticmethod
    def _format_goal(goal: Any) -> str:
        """Normalize goal objects or raw strings into compact status labels."""
        if isinstance(goal, str):
            return goal

        title = str(
            getattr(goal, "title", "")
            or getattr(goal, "description", "")
            or getattr(goal, "goal_id", "")
            or goal
        ).strip()
        status = str(getattr(goal, "status", "") or "").strip()
        if status and hasattr(getattr(goal, "status", None), "value"):
            status = str(goal.status.value)
        return f"{title} ({status})" if title and status else title

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of currently executing/queued tasks."""
        if not self.alice:
            return []

        tasks = []

        if (
            hasattr(self.alice, "persistent_task_queue")
            and self.alice.persistent_task_queue
        ):
            queue = self.alice.persistent_task_queue
            if hasattr(queue, "get_active_tasks"):
                active = queue.get_active_tasks()
                for task in active[:5]:
                    tasks.append(
                        {
                            "task_id": getattr(task, "id", "unknown"),
                            "task_type": getattr(task, "type", "unknown"),
                            "status": getattr(task, "status", "unknown"),
                        }
                    )

        return tasks

    def get_plugin_state(self) -> Dict[str, Any]:
        """Get state of registered plugins."""
        if not self.alice or not hasattr(self.alice, "plugins"):
            return {}

        plugins = self.alice.plugins

        registered = getattr(
            plugins,
            "plugins",
            getattr(plugins, "registered_plugins", {}),
        )
        registered = registered if isinstance(registered, dict) else {}

        state = {
            "plugins_loaded": bool(registered),
            "plugin_count": len(registered),
            "available_plugins": list(registered.keys())[:10],
            "failed_plugins": [],
        }

        if hasattr(plugins, "failed_plugins"):
            state["failed_plugins"] = plugins.failed_plugins[:5]

        return state

    def get_error_state(self) -> Dict[str, Any]:
        """Get recent errors and failures."""
        if not self.alice:
            return {}

        errors = {
            "recent_errors": [],
            "error_count_session": 0,
            "last_error": None,
        }

        # Try to get error log
        if hasattr(self.alice, "error_logger") and self.alice.error_logger:
            logger_obj = self.alice.error_logger
            if hasattr(logger_obj, "get_recent_errors"):
                errors["recent_errors"] = logger_obj.get_recent_errors(count=3)

        if hasattr(self.alice, "_last_error"):
            errors["last_error"] = str(self.alice._last_error)

        return errors

    def get_performance_state(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.alice:
            return {}

        state = {
            "avg_response_time_ms": 0.0,
            "request_latency_p50_ms": 0.0,
            "request_latency_p99_ms": 0.0,
            "tool_execution_time_ms": 0.0,
        }

        if hasattr(self.alice, "metrics") and self.alice.metrics:
            metrics = self.alice.metrics
            if hasattr(metrics, "avg_response_latency"):
                state["avg_response_time_ms"] = getattr(
                    metrics, "avg_response_latency", 0.0
                )

        return state

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        if not self.alice:
            return {}

        stats = {
            "session_id": "unknown",
            "turn_count": 0,
            "user_name": getattr(self.alice, "user_name", "User"),
            "llm_policy": getattr(self.alice, "llm_policy", "default"),
            "uptime_seconds": 0,
        }

        if (
            hasattr(self.alice, "_session_summarizer")
            and self.alice._session_summarizer
        ):
            summarizer = self.alice._session_summarizer
            stats["session_id"] = summarizer.session_id[:8] + "..."
            stats["turn_count"] = summarizer.turn_count

        if hasattr(self.alice, "_start_time"):
            import time

            stats["uptime_seconds"] = int(time.time() - self.alice._start_time)

        return stats

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the entire system state."""
        self.state_snapshot_timestamp = datetime.now().isoformat()

        snapshot = {
            "timestamp": self.state_snapshot_timestamp,
            "processors": self.get_processor_state(),
            "memory": self.get_memory_state(),
            "active_goals": self.get_active_goals(),
            "active_tasks": self.get_active_tasks(),
            "plugins": self.get_plugin_state(),
            "errors": self.get_error_state(),
            "performance": self.get_performance_state(),
            "session_stats": self.get_session_stats(),
        }

        return snapshot

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        snapshot = self.get_system_snapshot()

        health = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
        }

        # Check memory
        memory = snapshot.get("memory", {})
        if memory.get("memory_usage_mb", 0) > 500:
            health["warnings"].append("High memory usage (>500MB)")

        # Check errors
        errors = snapshot.get("errors", {})
        if errors.get("recent_errors"):
            health["issues"].append(f"{len(errors['recent_errors'])} recent errors")
            health["status"] = "degraded"

        # Check plugins
        plugins = snapshot.get("plugins", {})
        if plugins.get("failed_plugins"):
            health["warnings"].append(
                f"{len(plugins['failed_plugins'])} plugins failed to load"
            )

        # Check tasks
        tasks = snapshot.get("active_tasks", [])
        if len(tasks) > 5:
            health["warnings"].append(f"Many active tasks ({len(tasks)})")

        # Get system-level metrics
        try:
            process = psutil.Process(os.getpid())
            health["cpu_percent"] = process.cpu_percent()
            health["memory_percent"] = process.memory_percent()
        except:
            pass

        return health

    def format_system_report(self) -> str:
        """Generate a human-readable system report."""
        snapshot = self.get_system_snapshot()
        health = self.get_system_health()

        lines = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║                  A.L.I.C.E SYSTEM STATE REPORT                 ║",
            "╚═══════════════════════════════════════════════════════════════╝",
            "",
        ]

        # Session info
        session = snapshot.get("session_stats", {})
        lines.append(
            f"Session: {session.get('session_id')} | Turns: {session.get('turn_count')} | Uptime: {session.get('uptime_seconds')}s"
        )
        lines.append("")

        # System health
        lines.append(f"Health: {health['status'].upper()}")
        if health.get("issues"):
            for issue in health["issues"]:
                lines.append(f"  ✗ {issue}")
        if health.get("warnings"):
            for warning in health["warnings"]:
                lines.append(f"  ⚠ {warning}")
        lines.append("")

        # Active state
        memory = snapshot.get("memory", {})
        lines.append(
            f"Memory: {memory.get('episodic_memories', 0)} episodic, {memory.get('semantic_memories', 0)} semantic ({memory.get('memory_usage_mb', 0):.0f}MB)"
        )

        goals = snapshot.get("active_goals", [])
        if goals:
            lines.append(f"Goals: {', '.join(goals[:3])}")

        tasks = snapshot.get("active_tasks", [])
        if tasks:
            lines.append(f"Tasks: {len(tasks)} active")

        plugins = snapshot.get("plugins", {})
        lines.append(f"Plugins: {plugins.get('plugin_count', 0)} loaded")

        lines.append("")

        return "\n".join(lines)


# Singleton
_api: SystemStateAPI = None


def get_system_state_api() -> SystemStateAPI:
    """Get or create the system state API."""
    global _api
    if _api is None:
        _api = SystemStateAPI()
    return _api
