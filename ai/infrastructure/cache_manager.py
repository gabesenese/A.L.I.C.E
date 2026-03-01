"""
Production-Grade Distributed Cache Manager
Uses Redis for high-performance caching with intelligent invalidation strategies
"""

import json
import hashlib
import logging
from typing import Any, Optional, Callable, Union
from functools import wraps
from datetime import datetime, timedelta
import pickle
import time

logger = logging.getLogger(__name__)

# Try to import Redis, fall back to in-memory cache if not available
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache fallback")


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half_open'
                logger.info("[CircuitBreaker] Attempting recovery (half-open)")
            else:
                raise Exception(f"Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("[CircuitBreaker] Recovery successful (closed)")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"[CircuitBreaker] OPENED after {self.failure_count} failures")
            
            raise


class CacheManager:
    """
    Intelligent caching with multiple strategies:
    - Write-through: Write to cache and backend simultaneously
    - Write-back: Write to cache immediately, backend asynchronously
    - Cache-aside: Application manages cache explicitly
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,
        enable_circuit_breaker: bool = True
    ):
        self.default_ttl = default_ttl
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=False,  # We'll handle encoding
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"[Cache] Connected to Redis at {redis_host}:{redis_port}")
            except (RedisError, RedisConnectionError) as e:
                logger.warning(f"[Cache] Redis connection failed: {e}, using in-memory fallback")
                self.redis_client = None
        else:
            logger.info("[Cache] Using in-memory fallback cache")
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Generate namespaced cache key"""
        return f"alice:{namespace}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try JSON first (faster, human-readable in Redis)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(data)
    
    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        cache_key = self._make_key(namespace, key)
        
        try:
            if self.redis_client and self.circuit_breaker:
                data = self.circuit_breaker.call(self.redis_client.get, cache_key)
            elif self.redis_client:
                data = self.redis_client.get(cache_key)
            else:
                data = self.fallback_cache.get(cache_key)
            
            if data is not None:
                self.stats['hits'] += 1
                return self._deserialize(data) if isinstance(data, bytes) else data
            else:
                self.stats['misses'] += 1
                return default
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.debug(f"[Cache] Get error: {e}")
            # Try fallback
            return self.fallback_cache.get(cache_key, default)
    
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """
        Set value in cache
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (None = default_ttl)
            nx: Only set if not exists
        
        Returns:
            True if successful
        """
        cache_key = self._make_key(namespace, key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        try:
            serialized = self._serialize(value)
            
            if self.redis_client and self.circuit_breaker:
                result = self.circuit_breaker.call(
                    self.redis_client.set,
                    cache_key,
                    serialized,
                    ex=ttl,
                    nx=nx
                )
            elif self.redis_client:
                result = self.redis_client.set(cache_key, serialized, ex=ttl, nx=nx)
            else:
                if nx and cache_key in self.fallback_cache:
                    return False
                self.fallback_cache[cache_key] = serialized
                result = True
            
            self.stats['sets'] += 1
            return bool(result)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.debug(f"[Cache] Set error: {e}")
            # Store in fallback anyway
            self.fallback_cache[cache_key] = self._serialize(value)
            return False
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._make_key(namespace, key)
        
        try:
            if self.redis_client and self.circuit_breaker:
                result = self.circuit_breaker.call(self.redis_client.delete, cache_key)
            elif self.redis_client:
                result = self.redis_client.delete(cache_key)
            else:
                result = 1 if self.fallback_cache.pop(cache_key, None) is not None else 0
            
            self.stats['deletes'] += 1
            return bool(result)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.debug(f"[Cache] Delete error: {e}")
            self.fallback_cache.pop(cache_key, None)
            return False
    
    def invalidate_pattern(self, namespace: str, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        full_pattern = self._make_key(namespace, pattern)
        count = 0
        
        try:
            if self.redis_client:
                for key in self.redis_client.scan_iter(match=full_pattern):
                    self.redis_client.delete(key)
                    count += 1
            else:
                # Fallback pattern matching
                prefix = self._make_key(namespace, '')
                keys_to_delete = [
                    k for k in self.fallback_cache.keys()
                    if k.startswith(prefix)
                ]
                for key in keys_to_delete:
                    del self.fallback_cache[key]
                    count += 1
            
            logger.info(f"[Cache] Invalidated {count} keys matching {full_pattern}")
            return count
            
        except Exception as e:
            logger.error(f"[Cache] Pattern invalidation error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            **self.stats,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests,
            'backend': 'redis' if self.redis_client else 'memory'
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info('stats')
                stats['redis_keys'] = self.redis_client.dbsize()
                stats['redis_memory'] = info.get('used_memory_human', 'unknown')
            except:
                pass
        
        return stats
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache (optionally by namespace)"""
        try:
            if namespace:
                return self.invalidate_pattern(namespace, '*')
            else:
                if self.redis_client:
                    self.redis_client.flushdb()
                self.fallback_cache.clear()
                logger.info("[Cache] Cleared all caches")
                return -1
        except Exception as e:
            logger.error(f"[Cache] Clear error: {e}")
            return 0


def cached(
    namespace: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Usage:
        @cached('nlp', ttl=300)
        def expensive_nlp_operation(text: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager from kwargs or use global
            cache = kwargs.pop('_cache', None) or get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash function name + args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
            
            # Try cache first
            cached_result = cache.get(namespace, cache_key)
            if cached_result is not None:
                logger.debug(f"[Cache] HIT: {func.__name__}")
                return cached_result
            
            # Cache miss - execute function
            logger.debug(f"[Cache] MISS: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(namespace, cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def initialize_cache(
    redis_host: str = 'localhost',
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    default_ttl: int = 3600
) -> CacheManager:
    """Initialize global cache manager with custom config"""
    global _cache_manager
    _cache_manager = CacheManager(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        default_ttl=default_ttl
    )
    return _cache_manager
