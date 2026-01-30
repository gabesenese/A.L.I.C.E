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
        Attempt to recover from an error.
        Returns: (recovery_response, alternative_suggestion) or (None, None)
        """
        # For "note not found" errors, suggest listing notes
        if "not found" in error_message.lower() or "couldn't find" in error_message.lower():
            if "note" in original_intent.lower():
                return (
                    f"I couldn't find that note. {error_message}",
                    "Would you like me to list your notes so you can see what's available?"
                )
            elif "email" in original_intent.lower():
                return (
                    f"I couldn't find that email. {error_message}",
                    "Would you like me to show your recent emails?"
                )
        
        # For "permission denied" or "access" errors
        if "permission" in error_message.lower() or "access" in error_message.lower():
            return (
                f"I don't have permission to do that. {error_message}",
                "You may need to grant permissions or check your settings."
            )
        
        # For generic failures, suggest retry or alternative
        if "failed" in error_message.lower() or "error" in error_message.lower():
            return (
                f"That didn't work. {error_message}",
                "Would you like me to try a different approach, or would you prefer to try again later?"
            )
        
        return None, None
    
    def suggest_alternative(self, intent: str, failed_action: str) -> Optional[str]:
        """Suggest an alternative action when one fails"""
        alternatives = {
            "notes:delete": "You could archive the note instead, or I can list your notes so you can choose.",
            "notes:create": "I can try creating it with a different title, or you can check if a similar note already exists.",
            "email:send": "I can save it as a draft, or you can try again with a different recipient.",
            "calendar:create": "I can add it to your notes as a reminder, or you can try creating it manually.",
        }
        
        return alternatives.get(intent)


_error_recovery: Optional[ErrorRecovery] = None


def get_error_recovery() -> ErrorRecovery:
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = ErrorRecovery()
    return _error_recovery
