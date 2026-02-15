"""
Infrastructure Facade for A.L.I.C.E
Events, monitoring, and error recovery
"""

from ai.infrastructure.event_bus import get_event_bus
from ai.infrastructure.error_recovery import ErrorRecovery
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class InfrastructureFacade:
    """Facade for events, monitoring, and reliability"""

    def __init__(self) -> None:
        # Event bus
        try:
            self.event_bus = get_event_bus()
        except Exception as e:
            logger.warning(f"Event bus not available: {e}")
            self.event_bus = None

        # Error recovery
        try:
            self.error_recovery = ErrorRecovery()
        except Exception as e:
            logger.warning(f"Error recovery not available: {e}")
            self.error_recovery = None

        logger.info("[InfrastructureFacade] Initialized infrastructure systems")

    def emit_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Emit system event

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            True if emitted successfully
        """
        if not self.event_bus:
            return False

        try:
            self.event_bus.emit(event_type, data)
            return True
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
            return False

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """
        Subscribe to system events

        Args:
            event_type: Event type to subscribe to
            callback: Callback function

        Returns:
            True if subscribed successfully
        """
        if not self.event_bus:
            return False

        try:
            self.event_bus.subscribe(event_type, callback)
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to event: {e}")
            return False

    def recover_from_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Attempt to recover from error

        Args:
            error: Exception that occurred
            context: Error context

        Returns:
            Recovery result or None
        """
        if not self.error_recovery:
            return None

        try:
            return self.error_recovery.recover(error, context)
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return None


# Singleton instance
_infrastructure_facade: Optional[InfrastructureFacade] = None


def get_infrastructure_facade() -> InfrastructureFacade:
    """Get or create the InfrastructureFacade singleton"""
    global _infrastructure_facade
    if _infrastructure_facade is None:
        _infrastructure_facade = InfrastructureFacade()
    return _infrastructure_facade
