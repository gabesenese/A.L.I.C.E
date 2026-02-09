"""
Error Recovery for A.L.I.C.E
When actions fail, tries alternatives and provides helpful feedback.
Makes A.L.I.C.E resilient and helpful even when things go wrong.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecoveryStrategy:
    """A recovery strategy for a failed action"""
    strategy_id: str
    description: str
    action: callable
    confidence: float = 0.5


class ErrorRecovery:
    """
    Error recovery engine that:
    - Tries alternative approaches when actions fail
    - Provides helpful error messages
    - Suggests next steps
    """
    
    def __init__(self):
        self.recovery_strategies: Dict[str, List[RecoveryStrategy]] = {}
    
    def recover_from_error(
        self,
        error_type: str,
        original_intent: str,
        failed_action: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to recover from an error by using LLM to generate recovery response.
        Returns: (recovery_response, alternative_suggestion) or (None, None) if LLM should handle
        """
        # Let LLM generate context-aware error messages instead of hardcoding
        # This preserves ALICE's ability to think and adapt
        return None, None
    
    def suggest_alternative(self, intent: str, failed_action: str) -> Optional[str]:
        """Suggest an alternative action - LLM generates these based on context"""
        # Don't hardcode alternatives - let LLM suggest them based on context
        return None


_error_recovery: Optional[ErrorRecovery] = None


def get_error_recovery() -> ErrorRecovery:
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = ErrorRecovery()
    return _error_recovery
