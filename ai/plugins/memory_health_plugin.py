"""
Memory Health Plugin for A.L.I.C.E

Handles intent: memory:test
Triggered by phrases like:
  "run memory tests"
  "check memory health"
  "test memory"
  "memory diagnostics"

Executes scripts/check_memory_health.py and returns a human-readable summary.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from ai.plugins.plugin_system import PluginInterface

logger = logging.getLogger(__name__)

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "check_memory_health.py"


class MemoryHealthPlugin(PluginInterface):
    """Runs the memory health check and reports results to the user."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "MemoryHealthPlugin"
        self.version = "1.0.0"
        self.description = "Run memory persistence tests and report results"
        self.enabled = True
        self.capabilities = ["memory_test", "memory_health", "memory_diagnostics"]

    def initialize(self) -> bool:
        if not _SCRIPT.exists():
            logger.warning(f"[MemoryHealthPlugin] Script not found: {_SCRIPT}")
            return False
        logger.info("[MemoryHealthPlugin] Ready")
        return True

    def shutdown(self) -> None:
        pass

    def can_handle(self, intent: str, entities: Dict, query: str = "") -> bool:
        return intent == "memory:test"

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                [sys.executable, str(_SCRIPT)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout.strip()
            passed, total, lines = self._parse_output(output)

            if passed == total and total > 0:
                summary = f"All {total} memory checks passed."
            elif total == 0:
                summary = "Could not parse test results."
            else:
                failed = total - passed
                summary = f"{failed} of {total} memory checks failed."

            response = f"{summary}\n\n{self._format_lines(lines)}"

            return {
                "success": result.returncode == 0,
                "response": response.strip(),
                "data": {
                    "passed": passed,
                    "total": total,
                    "exit_code": result.returncode,
                    "plugin_type": "memory_health",
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "response": "Memory health check timed out after 120 seconds.",
                "data": {"error": "timeout"},
            }
        except Exception as exc:
            logger.error(f"[MemoryHealthPlugin] Error: {exc}")
            return {
                "success": False,
                "response": f"Could not run memory health check: {exc}",
                "data": {"error": str(exc)},
            }

    @staticmethod
    def _parse_output(output: str) -> tuple:
        """Return (passed, total, result_lines) from health check stdout."""
        lines = [l.strip() for l in output.splitlines() if "[PASS]" in l or "[FAIL]" in l]
        passed = sum(1 for l in lines if "[PASS]" in l)
        return passed, len(lines), lines

    @staticmethod
    def _format_lines(lines: list) -> str:
        """Format result lines for display."""
        return "\n".join(lines)
