"""
RAG Engine for A.L.I.C.E
Retrieval-Augmented Generation capabilities
"""

from ai.memory.memory_store import MemoryStore, MemoryEntry
from ai.memory.embedding_manager import EmbeddingManager
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    """Handles retrieval-augmented generation queries"""

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_manager: EmbeddingManager
    ) -> None:
        self.store = memory_store
        self.embedding = embedding_manager

    def query(
        self,
        query_text: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Query memories with semantic similarity

        Args:
            query_text: Search query
            memory_type: Filter by memory type (optional)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant memory dictionaries
        """
        try:
            # Create query embedding
            query_embedding = self.embedding.create_embedding(query_text)

            # Find similar memories
            similar_memories = self.store.find_by_similarity(
                embedding=query_embedding,
                threshold=min_similarity,
                top_k=top_k,
                memory_type=memory_type
            )

            # Convert to dictionaries with similarity scores
            results = []
            for memory in similar_memories:
                mem_embedding = memory.embedding
                if mem_embedding:
                    import numpy as np
                    similarity = self.embedding.calculate_similarity(
                        query_embedding,
                        np.array(mem_embedding)
                    )
                else:
                    similarity = 0.0

                results.append({
                    'id': memory.id,
                    'content': memory.content,
                    'memory_type': memory.memory_type,
                    'timestamp': memory.timestamp,
                    'importance': memory.importance,
                    'similarity': similarity,
                    'tags': memory.tags
                })

            return results

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []

    def get_llm_context(
        self,
        user_input: str,
        max_tokens: int = 4000,
        max_memories: int = 3
    ) -> str:
        """
        Get relevant context for LLM response

        Args:
            user_input: User's current input
            max_tokens: Maximum context size
            max_memories: Maximum number of memories to include

        Returns:
            Formatted context string
        """
        try:
            # Query relevant memories
            memories = self.query(
                query_text=user_input,
                top_k=max_memories,
                min_similarity=0.6
            )

            if not memories:
                return ""

            # Build context string
            context_parts = []
            total_chars = 0
            max_chars = max_tokens * 4  # Rough estimate: 4 chars per token

            for mem in memories:
                mem_text = f"[Memory from {mem['timestamp']}]: {mem['content']}"

                if total_chars + len(mem_text) > max_chars:
                    break

                context_parts.append(mem_text)
                total_chars += len(mem_text)

            if context_parts:
                return "Relevant memories:\n" + "\n\n".join(context_parts)

            return ""

        except Exception as e:
            logger.error(f"Failed to get LLM context: {e}")
            return ""

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search through ingested document chunks

        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant document chunks
        """
        try:
            # Query document type memories
            results = self.query(
                query_text=query,
                memory_type="document",
                top_k=top_k,
                min_similarity=min_similarity
            )

            # Add document-specific info
            for result in results:
                memory = self.store.get_by_id(result['id'])
                if memory:
                    result['source_file'] = memory.source_file
                    result['chunk_index'] = memory.chunk_index

            return results

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []


from typing import Optional as OptionalType

# Singleton requires store and embedding manager instances
_rag_engine: OptionalType[RAGEngine] = None


def get_rag_engine(
    memory_store: MemoryStore = None,
    embedding_manager: EmbeddingManager = None
) -> RAGEngine:
    """Get or create the RAGEngine singleton"""
    global _rag_engine
    if _rag_engine is None:
        from ai.memory.memory_store import get_memory_store
        from ai.memory.embedding_manager import get_embedding_manager

        store = memory_store or get_memory_store()
        embedding = embedding_manager or get_embedding_manager()
        _rag_engine = RAGEngine(store, embedding)
    return _rag_engine
