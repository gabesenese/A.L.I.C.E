"""
Optimization Facade for A.L.I.C.E
Performance tuning and runtime optimization
"""

from ai.optimization.runtime_thresholds import get_thresholds, update_thresholds
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizationFacade:
    """Facade for performance optimization"""

    def __init__(self) -> None:
        # Load runtime thresholds
        try:
            self.thresholds = get_thresholds()
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            self.thresholds = {}

        logger.info("[OptimizationFacade] Initialized optimization systems")

    def get_threshold(self, key: str, default: float = 0.7) -> float:
        """
        Get runtime threshold value

        Args:
            key: Threshold key
            default: Default value if not found

        Returns:
            Threshold value
        """
        try:
            return self.thresholds.get(key, default)
        except Exception as e:
            logger.error(f"Failed to get threshold '{key}': {e}")
            return default

    def set_threshold(self, key: str, value: float) -> bool:
        """
        Update runtime threshold

        Args:
            key: Threshold key
            value: New value

        Returns:
            True if updated successfully
        """
        try:
            update_thresholds({key: value})
            # Reload thresholds
            self.thresholds = get_thresholds()
            return True
        except Exception as e:
            logger.error(f"Failed to set threshold '{key}': {e}")
            return False

    def get_all_thresholds(self) -> Dict[str, float]:
        """
        Get all runtime thresholds

        Returns:
            Dictionary of all thresholds
        """
        try:
            return self.thresholds.copy()
        except Exception as e:
            logger.error(f"Failed to get all thresholds: {e}")
            return {}

    def reload_thresholds(self) -> bool:
        """
        Reload thresholds from configuration

        Returns:
            True if reloaded successfully
        """
        try:
            self.thresholds = get_thresholds()
            logger.info("Runtime thresholds reloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to reload thresholds: {e}")
            return False


# Singleton instance
_optimization_facade: Optional[OptimizationFacade] = None


def get_optimization_facade() -> OptimizationFacade:
    """Get or create the OptimizationFacade singleton"""
    global _optimization_facade
    if _optimization_facade is None:
        _optimization_facade = OptimizationFacade()
    return _optimization_facade
