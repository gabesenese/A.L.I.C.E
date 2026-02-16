"""
Knowledge Facade for A.L.I.C.E
Knowledge graphs and user profiling
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    from ai.core.knowledge_engine import KnowledgeEngine
    _knowledge_engine_available = True
except ImportError:
    _knowledge_engine_available = False

try:
    from ai.learning.user_profile_engine import UserProfileEngine
    _user_profile_available = True
except ImportError:
    _user_profile_available = False


class KnowledgeFacade:
    """Facade for knowledge management"""

    def __init__(self) -> None:
        # Knowledge engine
        try:
            self.graph = KnowledgeEngine() if _knowledge_engine_available else None
        except Exception as e:
            logger.warning(f"Knowledge engine not available: {e}")
            self.graph = None

        # User profile
        try:
            self.profile = UserProfileEngine() if _user_profile_available else None
        except Exception as e:
            logger.warning(f"User profile not available: {e}")
            self.profile = None

        logger.info("[KnowledgeFacade] Initialized knowledge systems")

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0
    ) -> bool:
        """
        Add fact to knowledge graph

        Args:
            subject: Subject entity
            predicate: Relationship
            object: Object entity
            confidence: Confidence score

        Returns:
            True if added successfully
        """
        if not self.graph:
            return False

        try:
            self.graph.add_triple(subject, predicate, object, confidence)
            return True
        except Exception as e:
            logger.error(f"Failed to add fact: {e}")
            return False

    def query_knowledge(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge graph

        Args:
            query: Query string

        Returns:
            List of matching facts
        """
        if not self.graph:
            return []

        try:
            return self.graph.query(query)
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return []

    def update_user_preference(
        self,
        category: str,
        preference: str,
        value: Any
    ) -> bool:
        """
        Update user preference

        Args:
            category: Preference category
            preference: Preference name
            value: Preference value

        Returns:
            True if updated successfully
        """
        if not self.profile:
            return False

        try:
            self.profile.set_preference(category, preference, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update preference: {e}")
            return False

    def get_user_preference(
        self,
        category: str,
        preference: str,
        default: Any = None
    ) -> Any:
        """
        Get user preference

        Args:
            category: Preference category
            preference: Preference name
            default: Default value

        Returns:
            Preference value or default
        """
        if not self.profile:
            return default

        try:
            return self.profile.get_preference(category, preference, default)
        except Exception as e:
            logger.error(f"Failed to get preference: {e}")
            return default


# Singleton instance
_knowledge_facade: Optional[KnowledgeFacade] = None


def get_knowledge_facade() -> KnowledgeFacade:
    """Get or create the KnowledgeFacade singleton"""
    global _knowledge_facade
    if _knowledge_facade is None:
        _knowledge_facade = KnowledgeFacade()
    return _knowledge_facade
