"""
Consolidation Engine for A.L.I.C.E Memory System
Handles memory consolidation and deduplication
"""

from ai.memory.memory_store import MemoryStore, MemoryEntry
from ai.memory.embedding_manager import EmbeddingManager
from ai.memory.persistence_manager import PersistenceManager
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Handles memory consolidation and optimization"""

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_manager: EmbeddingManager,
        persistence_manager: PersistenceManager
    ) -> None:
        self.store = memory_store
        self.embedding = embedding_manager
        self.persistence = persistence_manager

        self.turns_since_consolidation = 0
        self.consolidation_interval = 100  # Consolidate every N turns

    def periodic_check(self) -> bool:
        """
        Check if periodic consolidation is needed

        Returns:
            True if consolidation was performed
        """
        self.turns_since_consolidation += 1

        if self.turns_since_consolidation >= self.consolidation_interval:
            logger.info(f"⏰ Running periodic consolidation (after {self.turns_since_consolidation} turns)")
            result = self.consolidate(max_episodic=1000, auto_deduplicate=True)
            self.turns_since_consolidation = 0
            return result

        return False

    def calculate_importance(self, memory: MemoryEntry) -> float:
        """
        Calculate memory importance score

        Args:
            memory: Memory entry

        Returns:
            Importance score (0-1)
        """
        try:
            from datetime import datetime

            # Base importance
            importance = memory.importance

            # Recency score (newer = more important)
            try:
                mem_time = datetime.fromisoformat(memory.timestamp)
                age_days = (datetime.now() - mem_time).days
                recency_score = max(0, (30 - age_days) / 30.0) * 0.3
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to calculate recency score: {e}")
                recency_score = 0.0

            # Access frequency score
            access_score = min(memory.access_count / 10.0, 1.0) * 0.2

            # Context richness score
            context_score = min(len(memory.context) / 5.0, 1.0) * 0.2

            # Tag richness score
            tag_score = min(len(memory.tags) / 3.0, 1.0) * 0.1

            # Combined score
            total_score = importance + recency_score + access_score + context_score + tag_score

            return min(total_score, 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate importance: {e}")
            return memory.importance

    def deduplicate(self, similarity_threshold: float = 0.95) -> int:
        """
        Remove duplicate memories

        Args:
            similarity_threshold: Similarity threshold for deduplication

        Returns:
            Number of duplicates removed
        """
        try:
            all_memories = self.store.get_all()
            memories_with_embeddings = [
                m for m in all_memories
                if m.embedding is not None
            ]

            if len(memories_with_embeddings) < 2:
                return 0

            duplicates_to_remove = set()

           for i, mem1 in enumerate(memories_with_embeddings):
                if mem1.id in duplicates_to_remove:
                    continue

                for mem2 in memories_with_embeddings[i + 1:]:
                    if mem2.id in duplicates_to_remove:
                        continue

                    # Calculate similarity
                    emb1 = np.array(mem1.embedding)
                    emb2 = np.array(mem2.embedding)
                    similarity = self.embedding.calculate_similarity(emb1, emb2)

                    if similarity >= similarity_threshold:
                        # Keep the more important one
                        imp1 = self.calculate_importance(mem1)
                        imp2 = self.calculate_importance(mem2)

                        if imp1 >= imp2:
                            duplicates_to_remove.add(mem2.id)
                        else:
                            duplicates_to_remove.add(mem1.id)
                            break  # mem1 is being removed, move to next

            # Remove duplicates
            for mem_id in duplicates_to_remove:
                self.store.remove(mem_id)

            if duplicates_to_remove:
                logger.info(f"Removed {len(duplicates_to_remove)} duplicate memories")

            return len(duplicates_to_remove)

        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return 0

    def consolidate(
        self,
        max_episodic: int = 1000,
        auto_deduplicate: bool = True
    ) -> bool:
        """
        Consolidate memories by archiving old/unimportant ones

        Args:
            max_episodic: Maximum episodic memories to keep
            auto_deduplicate: Automatically remove duplicates

        Returns:
            True if consolidation succeeded
        """
        try:
            logger.info("🧹 Starting memory consolidation...")

            # Step 1: Deduplicate if requested
            if auto_deduplicate:
                duplicates_removed = self.deduplicate(similarity_threshold=0.95)
                logger.info(f"   Removed {duplicates_removed} duplicates")

            # Step 2: Get episodic memories
            episodic_memories = self.store.get_all(memory_type="episodic")

            if len(episodic_memories) <= max_episodic:
                logger.info(f"   Episodic memory count ({len(episodic_memories)}) within limit ({max_episodic})")
                return True

            # Step 3: Calculate importance scores
            scored_memories = [
                (mem, self.calculate_importance(mem))
                for mem in episodic_memories
            ]

            # Step 4: Sort by importance
            scored_memories.sort(key=lambda x: x[1], reverse=True)

            # Step 5: Archive low-importance memories
            to_keep = scored_memories[:max_episodic]
            to_archive = scored_memories[max_episodic:]

            archived_memories = [mem for mem, _ in to_archive]

            # Remove from active memory
            for memory, _ in to_archive:
                self.store.remove(memory.id)

            # Save to archive
            if archived_memories:
                self.persistence.save_archived_memories(archived_memories)

            logger.info(f"✅ Consolidation complete:")
            logger.info(f"   Kept: {len(to_keep)} memories")
            logger.info(f"   Archived: {len(archived_memories)} memories")

            return True

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return False


from typing import Optional as OptionalType

# Singleton instance
_consolidation_engine: OptionalType[ConsolidationEngine] = None


def get_consolidation_engine(
    memory_store: MemoryStore = None,
    embedding_manager: EmbeddingManager = None,
    persistence_manager: PersistenceManager = None
) -> ConsolidationEngine:
    """Get or create the ConsolidationEngine singleton"""
    global _consolidation_engine
    if _consolidation_engine is None:
        from ai.memory.memory_store import get_memory_store
        from ai.memory.embedding_manager import get_embedding_manager
        from ai.memory.persistence_manager import get_persistence_manager

        store = memory_store or get_memory_store()
        embedding = embedding_manager or get_embedding_manager()
        persistence = persistence_manager or get_persistence_manager()

        _consolidation_engine = ConsolidationEngine(store, embedding, persistence)
    return _consolidation_engine
