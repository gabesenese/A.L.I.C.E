"""
Memory Store for A.L.I.C.E
Abstract storage interface for memory entries
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural", "document"
    timestamp: str
    context: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    source_file: Optional[str] = None  # Source document path
    chunk_index: Optional[int] = None  # For document chunks

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MemoryStore(ABC):
    """Abstract interface for memory storage backends"""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> bool:
        """
        Add a single memory entry

        Args:
            entry: Memory entry to add

        Returns:
            True if added successfully
        """
        pass

    @abstractmethod
    def bulk_add(self, entries: List[MemoryEntry]) -> int:
        """
        Add multiple entries efficiently

        Args:
            entries: List of memory entries

        Returns:
            Count of entries added
        """
        pass

    @abstractmethod
    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve memory by ID

        Args:
            memory_id: Memory identifier

        Returns:
            Memory entry or None
        """
        pass

    @abstractmethod
    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Get all memories, optionally filtered by type

        Args:
            memory_type: Filter by type (optional)

        Returns:
            List of memory entries
        """
        pass

    @abstractmethod
    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Find similar memories by vector similarity

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            memory_type: Filter by type (optional)

        Returns:
            List of similar memories
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        Remove memory by ID

        Args:
            memory_id: Memory identifier

        Returns:
            True if removed successfully
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update memory metadata

        Args:
            memory_id: Memory identifier
            updates: Fields to update

        Returns:
            True if updated successfully
        """
        pass

    @abstractmethod
    def count(self, memory_type: Optional[str] = None) -> int:
        """
        Count memories, optionally by type

        Args:
            memory_type: Filter by type (optional)

        Returns:
            Count of memories
        """
        pass


class InMemoryMemoryStore(MemoryStore):
    """In-memory implementation of MemoryStore"""

    def __init__(self) -> None:
        self.memories: Dict[str, MemoryEntry] = {}
        logger.info("[InMemoryMemoryStore] Initialized")

    def add(self, entry: MemoryEntry) -> bool:
        """Add single memory entry"""
        try:
            self.memories[entry.id] = entry
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False

    def bulk_add(self, entries: List[MemoryEntry]) -> int:
        """Add multiple entries"""
        count = 0
        for entry in entries:
            if self.add(entry):
                count += 1
        return count

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory by ID"""
        return self.memories.get(memory_id)

    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """Get all memories"""
        if memory_type is None:
            return list(self.memories.values())

        return [
            mem for mem in self.memories.values()
            if mem.memory_type == memory_type
        ]

    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Find similar memories"""
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = self.get_all(memory_type=memory_type)

        # Filter memories with embeddings
        memories_with_embeddings = [
            mem for mem in candidates
            if mem.embedding is not None
        ]

        if not memories_with_embeddings:
            return []

        try:
            # Calculate similarities
            similarities = []
            for mem in memories_with_embeddings:
                mem_embedding = np.array(mem.embedding)
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    mem_embedding.reshape(1, -1)
                )[0][0]

                if similarity >= threshold:
                    similarities.append((mem, similarity))

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in similarities[:top_k]]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def remove(self, memory_id: str) -> bool:
        """Remove memory by ID"""
        try:
            if memory_id in self.memories:
                del self.memories[memory_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove memory: {e}")
            return False

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata"""
        if memory_id not in self.memories:
            return False

        try:
            memory = self.memories[memory_id]
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False

    def count(self, memory_type: Optional[str] = None) -> int:
        """Count memories"""
        if memory_type is None:
            return len(self.memories)

        return sum(
            1 for mem in self.memories.values()
            if mem.memory_type == memory_type
        )


# Singleton instance
_memory_store: Optional[InMemoryMemoryStore] = None


def get_memory_store() -> InMemoryMemoryStore:
    """Get or create the MemoryStore singleton"""
    global _memory_store
    if _memory_store is None:
        _memory_store = InMemoryMemoryStore()
    return _memory_store
