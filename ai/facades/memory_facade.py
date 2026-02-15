"""
Memory Facade for A.L.I.C.E
Unified memory management interface
"""

from ai.memory.memory_system import MemorySystem, get_memory_system
from ai.memory.context_engine import get_context_engine
from ai.memory.smart_context_cache import get_context_cache
from ai.memory.adaptive_context_selector import get_context_selector
from ai.memory.conversation_summarizer import ConversationSummarizer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryFacade:
    """Unified memory management facade"""

    def __init__(self) -> None:
        # Core memory system
        self.core = get_memory_system()

        # Context management components
        try:
            self.context_engine = get_context_engine()
        except Exception as e:
            logger.warning(f"Context engine not available: {e}")
            self.context_engine = None

        try:
            self.cache = get_context_cache()
        except Exception as e:
            logger.warning(f"Context cache not available: {e}")
            self.cache = None

        try:
            self.selector = get_context_selector()
        except Exception as e:
            logger.warning(f"Context selector not available: {e}")
            self.selector = None

        try:
            self.summarizer = ConversationSummarizer()
        except Exception as e:
            logger.warning(f"Conversation summarizer not available: {e}")
            self.summarizer = None

        logger.info("[MemoryFacade] Initialized memory management facade")

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> bool:
        """
        Store memory with automatic categorization

        Args:
            content: Memory content to store
            memory_type: Type of memory (episodic, semantic, procedural, document)
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization

        Returns:
            True if storage succeeded
        """
        try:
            return self.core.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or []
            )
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search memories with semantic similarity

        Args:
            query: Search query
            memory_type: Filter by memory type (optional)
            top_k: Number of results to return

        Returns:
            List of relevant memories
        """
        try:
            return self.core.recall_memory(
                query=query,
                memory_type=memory_type,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Failed to recall memory: {e}")
            return []

    def get_relevant_context(
        self,
        user_input: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Get best context for LLM response

        Args:
            user_input: Current user input
            max_tokens: Maximum context size in tokens

        Returns:
            Relevant context string for LLM
        """
        try:
            return self.core.get_context_for_llm(
                user_input=user_input,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return ""

    def store_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        intent: str,
        entities: Dict[str, Any]
    ) -> bool:
        """
        Store a complete conversation turn

        Args:
            user_input: User's message
            assistant_response: Alice's response
            intent: Classified intent
            entities: Extracted entities

        Returns:
            True if storage succeeded
        """
        try:
            conversation_memory = {
                "user": user_input,
                "assistant": assistant_response,
                "intent": intent,
                "entities": entities
            }

            return self.core.store_memory(
                content=f"User: {user_input}\nAlice: {assistant_response}",
                memory_type="episodic",
                importance=0.6,
                tags=["conversation", intent],
                metadata=conversation_memory
            )
        except Exception as e:
            logger.error(f"Failed to store conversation turn: {e}")
            return False

    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Run memory consolidation to optimize storage

        Returns:
            Consolidation statistics
        """
        try:
            return self.core.consolidate_memories()
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return {"error": str(e)}

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics

        Returns:
            Statistics dictionary
        """
        try:
            return self.core.get_stats()
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search through ingested documents

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Relevant document chunks
        """
        try:
            return self.core.search_documents(query=query, top_k=top_k)
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ingest a document into the memory system

        Args:
            file_path: Path to document file
            metadata: Optional metadata

        Returns:
            True if ingestion succeeded
        """
        try:
            return self.core.ingest_document(
                file_path=file_path,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False


# Singleton instance
_memory_facade: Optional[MemoryFacade] = None


def get_memory_facade() -> MemoryFacade:
    """Get or create the MemoryFacade singleton"""
    global _memory_facade
    if _memory_facade is None:
        _memory_facade = MemoryFacade()
    return _memory_facade
