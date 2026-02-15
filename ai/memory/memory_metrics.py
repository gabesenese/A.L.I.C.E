"""
Memory Metrics for A.L.I.C.E
Statistics and analytics for memory system
"""

from ai.memory.memory_store import MemoryStore
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemoryMetrics:
    """Provides statistics and metrics for memory system"""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.store = memory_store

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics

        Returns:
            Statistics dictionary
        """
        try:
            all_memories = self.store.get_all()

            # Count by type
            counts_by_type = {
                'episodic': self.store.count('episodic'),
                'semantic': self.store.count('semantic'),
                'procedural': self.store.count('procedural'),
                'document': self.store.count('document')
            }

            # Calculate importance distribution
            importances = [m.importance for m in all_memories]
            avg_importance = sum(importances) / len(importances) if importances else 0.0

            # Access statistics
            access_counts = [m.access_count for m in all_memories]
            total_accesses = sum(access_counts)
            avg_accesses = total_accesses / len(access_counts) if access_counts else 0.0

            # Tag statistics
            all_tags = set()
            for mem in all_memories:
                all_tags.update(mem.tags)

            stats = {
                'total_memories': len(all_memories),
                'by_type': counts_by_type,
                'average_importance': round(avg_importance, 3),
                'total_accesses': total_accesses,
                'average_accesses': round(avg_accesses, 2),
                'unique_tags': len(all_tags),
                'tags': list(all_tags)
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

    def get_top_memories(
        self,
        n: int = 10,
        by: str = 'importance'
    ) -> list[Dict[str, Any]]:
        """
        Get top N memories by specified metric

        Args:
            n: Number of memories to return
            by: Sort key ('importance', 'access_count', 'recency')

        Returns:
            List of top memory dictionaries
        """
        try:
            all_memories = self.store.get_all()

            if by == 'importance':
                sorted_memories = sorted(all_memories, key=lambda m: m.importance, reverse=True)
            elif by == 'access_count':
                sorted_memories = sorted(all_memories, key=lambda m: m.access_count, reverse=True)
            elif by == 'recency':
                sorted_memories = sorted(all_memories, key=lambda m: m.timestamp, reverse=True)
            else:
                logger.warning(f"Unknown sort key: {by}, using importance")
                sorted_memories = sorted(all_memories, key=lambda m: m.importance, reverse=True)

            top_memories = sorted_memories[:n]

            return [
                {
                    'id': m.id,
                    'content': m.content[:100] + '...' if len(m.content) > 100 else m.content,
                    'memory_type': m.memory_type,
                    'importance': m.importance,
                    'access_count': m.access_count,
                    'timestamp': m.timestamp
                }
                for m in top_memories
            ]

        except Exception as e:
            logger.error(f"Failed to get top memories: {e}")
            return []

    def get_tag_distribution(self) -> Dict[str, int]:
        """
        Get distribution of tags across memories

        Returns:
            Dictionary mapping tags to counts
        """
        try:
            all_memories = self.store.get_all()

            tag_counts: Dict[str, int] = {}
            for mem in all_memories:
                for tag in mem.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Sort by count
            sorted_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))

            return sorted_tags

        except Exception as e:
            logger.error(f"Failed to get tag distribution: {e}")
            return {}

    def get_memory_health(self) -> Dict[str, Any]:
        """
        Get memory system health metrics

        Returns:
            Health metrics dictionary
        """
        try:
            stats = self.get_stats()

            total = stats['total_memories']
            avg_importance = stats['average_importance']
            avg_accesses = stats['average_accesses']

            # Determine health status
            if total == 0:
                health = 'empty'
            elif avg_importance < 0.3:
                health = 'low_quality'
            elif avg_accesses < 0.5:
                health = 'underutilized'
            elif total > 10000:
                health = 'needs_consolidation'
            else:
                health = 'healthy'

            return {
                'status': health,
                'total_memories': total,
                'average_importance': avg_importance,
                'average_accesses': avg_accesses,
                'recommendations': self._get_recommendations(health, stats)
            }

        except Exception as e:
            logger.error(f"Failed to get memory health: {e}")
            return {'status': 'error', 'error': str(e)}

    def _get_recommendations(self, health: str, stats: Dict) -> list[str]:
        """Generate recommendations based on health status"""
        recommendations = []

        if health == 'empty':
            recommendations.append("Memory system is empty. Start storing memories.")
        elif health == 'low_quality':
            recommendations.append("Average importance is low. Consider reviewing memory quality.")
        elif health == 'underutilized':
            recommendations.append("Memories are rarely accessed. Improve memory retrieval.")
        elif health == 'needs_consolidation':
            recommendations.append("Large number of memories. Run consolidation to optimize.")

        return recommendations


from typing import Optional

# Singleton instance
_memory_metrics: Optional[MemoryMetrics] = None


def get_memory_metrics(memory_store: MemoryStore = None) -> MemoryMetrics:
    """Get or create the MemoryMetrics singleton"""
    global _memory_metrics
    if _memory_metrics is None:
        from ai.memory.memory_store import get_memory_store
        store = memory_store or get_memory_store()
        _memory_metrics = MemoryMetrics(store)
    return _memory_metrics
