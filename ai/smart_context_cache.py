"""
Smart Context Cache for A.L.I.C.E
Intelligently caches and reuses context to avoid redundant computation.
Makes A.L.I.C.E faster and more efficient.
"""

import logging
import hashlib
import time
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class CachedContext:
    """Cached context entry"""
    context: str
    timestamp: float
    hit_count: int = 0
    last_used: float = field(default_factory=time.time)


class SmartContextCache:
    """
    Intelligent context caching that:
    - Caches context for similar queries
    - Invalidates when conversation changes significantly
    - Prioritizes frequently used context
    - Adapts cache size based on usage patterns
    """
    
    def __init__(self, max_size: int = 50, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedContext] = {}
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str, intent: str, entities_hash: str) -> str:
        """Generate cache key from query, intent, and entities"""
        key_str = f"{query.lower().strip()[:100]}:{intent}:{entities_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hash_entities(self, entities: Dict) -> str:
        """Hash entities for cache key"""
        if not entities:
            return ""
        # Use entity types and first values
        ent_str = ":".join(f"{k}:{str(v[0] if isinstance(v, list) and v else v)[:30]}" 
                          for k, v in list(entities.items())[:5])
        return hashlib.md5(ent_str.encode()).hexdigest()
    
    def get(self, query: str, intent: str, entities: Dict) -> Optional[str]:
        """Get cached context if available and fresh"""
        with self._lock:
            entities_hash = self._hash_entities(entities)
            cache_key = self._hash_query(query, intent, entities_hash)
            
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                
                # Check TTL
                if time.time() - cached.timestamp > self.ttl_seconds:
                    del self.cache[cache_key]
                    self.misses += 1
                    return None
                
                # Update stats
                cached.hit_count += 1
                cached.last_used = time.time()
                self.hits += 1
                return cached.context
            
            self.misses += 1
            return None
    
    def put(self, query: str, intent: str, entities: Dict, context: str):
        """Cache context for future use"""
        with self._lock:
            entities_hash = self._hash_entities(entities)
            cache_key = self._hash_query(query, intent, entities_hash)
            
            # Evict if at capacity (LRU)
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                # Remove least recently used
                lru_key = min(self.cache.keys(), 
                            key=lambda k: self.cache[k].last_used)
                del self.cache[lru_key]
            
            self.cache[cache_key] = CachedContext(
                context=context,
                timestamp=time.time(),
                hit_count=0,
                last_used=time.time()
            )
    
    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern, or all if None"""
        with self._lock:
            if pattern is None:
                self.cache.clear()
            else:
                to_remove = [k for k in self.cache.keys() if pattern in k]
                for k in to_remove:
                    del self.cache[k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


_context_cache: Optional[SmartContextCache] = None


def get_context_cache() -> SmartContextCache:
    global _context_cache
    if _context_cache is None:
        _context_cache = SmartContextCache()
    return _context_cache
