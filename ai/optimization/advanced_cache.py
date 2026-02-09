"""
Advanced Caching System
=======================
Sophisticated caching with multiple eviction policies, predictive prefetching,
and adaptive optimization.
"""

import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import heapq
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction"""
    key: str
    value: Any
    size: int = 1
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds
    priority: int = 0  # Higher priority = less likely to evict
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created > self.ttl

    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)"""
        age = time.time() - self.created
        if age <= 0:
            return float('inf')
        return self.access_count / age

    def recency_score(self) -> float:
        """Calculate recency score (0-1, higher = more recent)"""
        time_since_access = time.time() - self.last_access
        # Decay with half-life of 1 hour
        half_life = 3600
        return 2 ** (-time_since_access / half_life)


class AdvancedCache:
    """
    Advanced caching system with multiple eviction policies:

    - LRU (Least Recently Used)
    - LFU (Least Frequently Used)
    - ARC (Adaptive Replacement Cache) - combines LRU and LFU
    - TTL (Time To Live) - expire based on age
    - SIZE - size-aware eviction
    - PRIORITY - priority-based retention
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100_000_000,  # 100 MB default
        eviction_policy: str = 'arc',
        ttl_default: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.ttl_default = ttl_default

        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.current_memory = 0

        # LRU tracking
        self.lru_order: OrderedDict[str, bool] = OrderedDict()

        # LFU tracking
        self.frequency_map: Dict[str, int] = defaultdict(int)

        # ARC - split cache into recent and frequent
        self.recent_keys: OrderedDict[str, bool] = OrderedDict()  # T1 - recent items
        self.frequent_keys: OrderedDict[str, bool] = OrderedDict()  # T2 - frequent items
        self.ghost_recent: OrderedDict[str, bool] = OrderedDict()  # B1 - evicted from recent
        self.ghost_frequent: OrderedDict[str, bool] = OrderedDict()  # B2 - evicted from frequent
        self.arc_target = max_size // 2  # Target size for T1

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'total_accesses': 0
        }

        # Predictive prefetching
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)  # key -> [next_keys]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats['total_accesses'] += 1

        # Clean expired entries periodically
        if self.stats['total_accesses'] % 100 == 0:
            self._clean_expired()

        if key not in self.entries:
            self.stats['misses'] += 1
            return None

        entry = self.entries[key]

        # Check if expired
        if entry.is_expired():
            self._remove(key)
            self.stats['expirations'] += 1
            self.stats['misses'] += 1
            return None

        # Update access metadata
        entry.access_count += 1
        entry.last_access = time.time()
        self.stats['hits'] += 1

        # Update tracking structures based on policy
        if self.eviction_policy == 'lru':
            self._update_lru(key)
        elif self.eviction_policy == 'lfu':
            self.frequency_map[key] += 1
        elif self.eviction_policy == 'arc':
            self._update_arc(key, hit=True)

        return entry.value

    def put(
        self,
        key: str,
        value: Any,
        size: int = 1,
        ttl: Optional[float] = None,
        priority: int = 0
    ) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to store
            size: Size of entry (for size-aware eviction)
            ttl: Time to live in seconds (None = use default)
            priority: Priority level (higher = more important)

        Returns:
            True if stored successfully
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.ttl_default

        # Check if we need to evict
        while (len(self.entries) >= self.max_size or
               self.current_memory + size > self.max_memory):
            if not self._evict():
                # Couldn't evict anything
                logger.warning(f"Cache full, cannot store key: {key}")
                return False

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            size=size,
            ttl=ttl,
            priority=priority
        )

        # Remove old entry if exists
        if key in self.entries:
            self._remove(key)

        # Store entry
        self.entries[key] = entry
        self.current_memory += size

        # Update tracking structures
        if self.eviction_policy == 'lru':
            self._update_lru(key)
        elif self.eviction_policy == 'lfu':
            self.frequency_map[key] = 1
        elif self.eviction_policy == 'arc':
            self._update_arc(key, hit=False)

        return True

    def _update_lru(self, key: str):
        """Update LRU order"""
        if key in self.lru_order:
            del self.lru_order[key]
        self.lru_order[key] = True

    def _update_arc(self, key: str, hit: bool):
        """Update ARC (Adaptive Replacement Cache) structures"""
        if hit:
            # Cache hit - move to frequent if was in recent
            if key in self.recent_keys:
                del self.recent_keys[key]
                self.frequent_keys[key] = True
            elif key in self.frequent_keys:
                # Move to end (most recent)
                del self.frequent_keys[key]
                self.frequent_keys[key] = True
        else:
            # Cache miss
            if key in self.ghost_recent:
                # Was recently evicted from recent - increase target
                self.arc_target = min(self.max_size, self.arc_target + 1)
                del self.ghost_recent[key]
                self.frequent_keys[key] = True
            elif key in self.ghost_frequent:
                # Was evicted from frequent - decrease target
                self.arc_target = max(0, self.arc_target - 1)
                del self.ghost_frequent[key]
                self.frequent_keys[key] = True
            else:
                # Completely new
                self.recent_keys[key] = True

    def _evict(self) -> bool:
        """Evict one entry based on policy"""
        if not self.entries:
            return False

        victim_key = None

        if self.eviction_policy == 'lru':
            victim_key = self._evict_lru()
        elif self.eviction_policy == 'lfu':
            victim_key = self._evict_lfu()
        elif self.eviction_policy == 'arc':
            victim_key = self._evict_arc()
        elif self.eviction_policy == 'priority':
            victim_key = self._evict_priority()
        else:
            # Default to LRU
            victim_key = self._evict_lru()

        if victim_key:
            self._remove(victim_key)
            self.stats['evictions'] += 1
            return True

        return False

    def _evict_lru(self) -> Optional[str]:
        """Evict least recently used"""
        if not self.lru_order:
            return None
        return next(iter(self.lru_order))

    def _evict_lfu(self) -> Optional[str]:
        """Evict least frequently used"""
        if not self.frequency_map:
            return None

        # Find key with minimum frequency, breaking ties with age
        min_key = None
        min_freq = float('inf')
        oldest_time = float('inf')

        for key, freq in self.frequency_map.items():
            if key in self.entries:
                entry = self.entries[key]
                if freq < min_freq or (freq == min_freq and entry.created < oldest_time):
                    min_key = key
                    min_freq = freq
                    oldest_time = entry.created

        return min_key

    def _evict_arc(self) -> Optional[str]:
        """Evict using ARC policy"""
        # Prefer evicting from recent if over target
        if len(self.recent_keys) > self.arc_target:
            if self.recent_keys:
                victim = next(iter(self.recent_keys))
                # Move to ghost recent
                del self.recent_keys[victim]
                self.ghost_recent[victim] = True
                # Limit ghost size
                if len(self.ghost_recent) > self.max_size:
                    self.ghost_recent.popitem(last=False)
                return victim

        # Otherwise evict from frequent
        if self.frequent_keys:
            victim = next(iter(self.frequent_keys))
            del self.frequent_keys[victim]
            self.ghost_frequent[victim] = True
            if len(self.ghost_frequent) > self.max_size:
                self.ghost_frequent.popitem(last=False)
            return victim

        return None

    def _evict_priority(self) -> Optional[str]:
        """Evict lowest priority, least recently used"""
        if not self.entries:
            return None

        # Find lowest priority entries
        min_priority = min(entry.priority for entry in self.entries.values())
        candidates = [
            key for key, entry in self.entries.items()
            if entry.priority == min_priority
        ]

        # Among those, pick LRU
        oldest_access = float('inf')
        victim = None

        for key in candidates:
            entry = self.entries[key]
            if entry.last_access < oldest_access:
                oldest_access = entry.last_access
                victim = key

        return victim

    def _remove(self, key: str):
        """Remove entry from cache"""
        if key not in self.entries:
            return

        entry = self.entries[key]
        self.current_memory -= entry.size
        del self.entries[key]

        # Clean up tracking structures
        if key in self.lru_order:
            del self.lru_order[key]
        if key in self.frequency_map:
            del self.frequency_map[key]
        if key in self.recent_keys:
            del self.recent_keys[key]
        if key in self.frequent_keys:
            del self.frequent_keys[key]

    def _clean_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self.entries.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            self._remove(key)
            self.stats['expirations'] += 1

    def prefetch(self, key: str) -> List[str]:
        """
        Predict and return keys that might be accessed next.

        Args:
            key: Current key being accessed

        Returns:
            List of predicted next keys
        """
        if key not in self.access_patterns:
            return []

        # Get most common next accesses
        next_keys = self.access_patterns[key]

        # Count frequencies
        freq = defaultdict(int)
        for next_key in next_keys[-10:]:  # Look at last 10 accesses
            freq[next_key] += 1

        # Return top 3 predictions
        predictions = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [key for key, _ in predictions[:3]]

    def record_access_pattern(self, prev_key: str, current_key: str):
        """Record access pattern for predictive prefetching"""
        self.access_patterns[prev_key].append(current_key)

        # Limit history
        if len(self.access_patterns[prev_key]) > 100:
            self.access_patterns[prev_key] = self.access_patterns[prev_key][-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats['total_accesses'] > 0:
            hit_rate = self.stats['hits'] / self.stats['total_accesses']

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'size': len(self.entries),
            'memory_used': self.current_memory,
            'memory_capacity': self.max_memory,
            'capacity': self.max_size
        }

    def clear(self):
        """Clear all cache entries"""
        self.entries.clear()
        self.current_memory = 0
        self.lru_order.clear()
        self.frequency_map.clear()
        self.recent_keys.clear()
        self.frequent_keys.clear()
        self.ghost_recent.clear()
        self.ghost_frequent.clear()


# Global cache instance
_global_cache = None


def get_cache(
    max_size: int = 1000,
    eviction_policy: str = 'arc'
) -> AdvancedCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdvancedCache(
            max_size=max_size,
            eviction_policy=eviction_policy
        )
    return _global_cache
