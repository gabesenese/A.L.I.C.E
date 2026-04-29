#!/usr/bin/env python3
"""
Quick test script to verify production infrastructure components.
Tests: Redis caching, Prometheus metrics, structured logging, task queues, database pooling.
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_cache_manager():
    """Test Redis caching with fallback to in-memory"""
    print("\n=== Testing Cache Manager ===")
    from ai.infrastructure.cache_manager import initialize_cache

    # Initialize (will fallback to memory if Redis unavailable)
    cache = initialize_cache(redis_host="localhost", redis_port=6379, default_ttl=60)
    backend = "redis" if cache.redis_client else "memory"
    print(f"✓ Cache initialized (backend: {backend})")

    # Test set/get
    namespace = "test"
    key = "hello"
    value = "world"

    cache.set(namespace, key, value, ttl=10)
    print(f"✓ Set {namespace}:{key} = {value}")

    retrieved = cache.get(namespace, key)
    assert retrieved == value, f"Expected {value}, got {retrieved}"
    print(f"✓ Retrieved {namespace}:{key} = {retrieved}")

    # Test delete
    cache.delete(namespace, key)
    print(f"✓ Deleted {namespace}:{key}")

    retrieved_after_delete = cache.get(namespace, key)
    assert retrieved_after_delete is None, "Expected None after delete"
    print("✓ Confirmed deletion (get returned None)")

    # Test stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: {stats}")

    print("✅ Cache Manager: PASS")
    return True


def test_metrics_collector():
    """Test Prometheus metrics collection"""
    print("\n=== Testing Metrics Collector ===")
    from ai.infrastructure.metrics_collector import (
        initialize_metrics,
    )

    # Initialize
    metrics = initialize_metrics(enable_prometheus=False)  # Use in-memory for testing
    print(f"✓ Metrics initialized (Prometheus: {metrics.enable_prometheus})")

    # Track various metrics
    metrics.track_request("greeting", True, 0.123, "plugin")
    print("✓ Tracked request metric")

    metrics.track_llm_call("llama3.1", 1.5, 2000, True)
    print("✓ Tracked LLM call metric")

    metrics.track_plugin("notes", "create", 0.05, True)
    print("✓ Tracked plugin metric")

    metrics.track_cache("get", "hit")
    print("✓ Tracked cache metric")

    metrics.track_error("ValueError", "test_component")
    print("✓ Tracked error metric")

    # Track learning
    metrics.track_learning(examples_total=100, quality_score=0.95)
    print("✓ Tracked learning metric")

    # Track confidence
    metrics.track_intent_confidence("greeting", 0.92)
    print("✓ Tracked confidence metric")

    # Get stats
    stats = metrics.get_metrics_summary()
    print(
        f"✓ Metrics stats: requests={stats.get('total_requests', 0)}, llm_calls={stats.get('total_llm_calls', 0)}"
    )

    print("✅ Metrics Collector: PASS")
    return True


def test_structured_logging():
    """Test JSON structured logging"""
    print("\n=== Testing Structured Logging ===")
    from ai.infrastructure.structured_logging import (
        configure_logging,
        get_structured_logger,
    )

    # Configure
    configure_logging(
        level="DEBUG", enable_json=False
    )  # Disable JSON for console readability
    logger = get_structured_logger("test")
    print("✓ Structured logger initialized")

    # Set context
    logger.set_context(user_id="test_user", session_id="test_session")
    print("✓ Context set")

    # Log various levels
    logger.debug("Debug message", extra_field="value1")
    logger.info("Info message", extra_field="value2")
    logger.warning("Warning message", extra_field="value3")
    print("✓ Logged at multiple levels")

    # Test component loggers
    nlp_logger = get_structured_logger("nlp")
    nlp_logger.info("NLP processing", intent="greeting", confidence=0.95)
    print("✓ Component logger works")

    print("✅ Structured Logging: PASS")
    return True


def test_database_pool():
    """Test database connection pooling"""
    print("\n=== Testing Database Pool ===")
    from ai.infrastructure.database_pool import (
        initialize_database,
        DatabaseConfig,
        DatabaseType,
    )

    # Initialize with SQLite for testing
    config = DatabaseConfig(
        db_type=DatabaseType.SQLITE, database="data/test_infrastructure.db"
    )
    pool = initialize_database(config)
    print(f"✓ Database pool initialized (type: {config.db_type.value})")

    # Test connection
    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        print("✓ Database connection works")

    # Test health check
    is_healthy = pool.health_check()
    print(f"✓ Health check: {'✓' if is_healthy else '✗'}")

    # Get stats
    stats = pool.get_stats()
    print(
        f"✓ Pool stats: size={stats.get('pool_size', 0)}, active={stats.get('active_connections', 0)}"
    )

    print("✅ Database Pool: PASS")
    return True


def test_task_queue():
    """Test async task queue"""
    print("\n=== Testing Task Queue ===")
    from ai.infrastructure.task_queue import (
        initialize_task_queue,
    )

    # Initialize with thread fallback (no RabbitMQ needed)
    queue = initialize_task_queue(broker_url=None, num_workers=2)
    backend = "celery" if queue.celery_app else "threads"
    print(f"✓ Task queue initialized (backend: {backend})")

    # Define a test task
    @queue.register_task(name="test_task")
    def example_task(x, y):
        return x + y

    print("✓ Registered test task")

    # Submit task
    result = example_task.delay(5, 3)
    print(f"✓ Submitted task (async: {result is not None})")

    # Wait a bit for async execution
    time.sleep(0.5)

    # Get stats
    stats = queue.get_stats()
    print(
        f"✓ Queue stats: total={stats.get('total_tasks', 0)}, completed={stats.get('completed_tasks', 0)}"
    )

    print("✅ Task Queue: PASS")
    return True


def main():
    """Run all infrastructure tests"""
    print("=" * 60)
    print("PRODUCTION INFRASTRUCTURE TEST SUITE")
    print("=" * 60)

    results = {}

    try:
        results["cache"] = test_cache_manager()
    except Exception as e:
        print(f"❌ Cache Manager: FAIL - {e}")
        results["cache"] = False

    try:
        results["metrics"] = test_metrics_collector()
    except Exception as e:
        print(f"❌ Metrics Collector: FAIL - {e}")
        results["metrics"] = False

    try:
        results["logging"] = test_structured_logging()
    except Exception as e:
        print(f"❌ Structured Logging: FAIL - {e}")
        results["logging"] = False

    try:
        results["database"] = test_database_pool()
    except Exception as e:
        print(f"❌ Database Pool: FAIL - {e}")
        results["database"] = False

    try:
        results["queue"] = test_task_queue()
    except Exception as e:
        print(f"❌ Task Queue: FAIL - {e}")
        results["queue"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{component.upper():20s} {status}")

    print("=" * 60)
    print(f"TOTAL: {passed}/{total} components passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
