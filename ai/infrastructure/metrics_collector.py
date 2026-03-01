"""
Production-Grade Metrics Collection System
Prometheus-compatible metrics with automatic export
"""

import time
import logging
from typing import Dict, Callable, Any, Optional
from functools import wraps
from collections import defaultdict
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Try to import prometheus client
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        generate_latest,
        REGISTRY,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available, using basic metrics")


class MetricsCollector:
    """
    Comprehensive metrics collection for A.L.I.C.E
    Tracks: latency, throughput, errors, resource usage
    """

    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.lock = threading.Lock()

        # Basic metrics storage (fallback if Prometheus unavailable)
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = {}

        if self.enable_prometheus:
            self._init_prometheus_metrics()
        else:
            logger.info("[Metrics] Using basic metrics (Prometheus unavailable)")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""

        # Request metrics
        self.request_total = Counter(
            "alice_requests_total",
            "Total number of requests processed",
            ["intent", "success"],
        )

        self.request_duration = Histogram(
            "alice_request_duration_seconds",
            "Request processing duration in seconds",
            ["intent", "route"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        # LLM metrics
        self.llm_calls = Counter(
            "alice_llm_calls_total", "Total LLM API calls", ["model", "success"]
        )

        self.llm_duration = Histogram(
            "alice_llm_duration_seconds",
            "LLM call duration",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )

        self.llm_tokens = Counter(
            "alice_llm_tokens_total",
            "Total tokens processed by LLM",
            ["model", "type"],  # type: input/output
        )

        # Plugin metrics
        self.plugin_calls = Counter(
            "alice_plugin_calls_total",
            "Plugin invocations",
            ["plugin", "action", "success"],
        )

        self.plugin_duration = Histogram(
            "alice_plugin_duration_seconds",
            "Plugin execution duration",
            ["plugin", "action"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )

        # Cache metrics
        self.cache_operations = Counter(
            "alice_cache_operations_total",
            "Cache operations",
            [
                "operation",
                "result",
            ],  # operation: get/set/delete, result: hit/miss/success/fail
        )

        # Error metrics
        self.errors_total = Counter(
            "alice_errors_total", "Total errors", ["type", "component"]
        )

        # Learning metrics
        self.learning_examples = Gauge(
            "alice_learning_examples_total", "Total learning examples collected"
        )

        self.learning_quality = Histogram(
            "alice_learning_quality_score",
            "Quality score of learning examples",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        # System metrics
        self.active_conversations = Gauge(
            "alice_active_conversations", "Number of active conversation threads"
        )

        self.memory_usage = Gauge(
            "alice_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # type: context/notes/cache/etc
        )

        # NLP metrics
        self.intent_confidence = Histogram(
            "alice_intent_confidence",
            "Intent classification confidence",
            ["intent"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        logger.info("[Metrics] Prometheus metrics initialized")

    # ===== Convenience Methods =====

    def track_request(self, intent: str, success: bool, duration: float, route: str):
        """Track a request from start to finish"""
        if self.enable_prometheus:
            self.request_total.labels(intent=intent, success=str(success)).inc()
            self.request_duration.labels(intent=intent, route=route).observe(duration)
        else:
            with self.lock:
                self.counters[f"requests_{intent}_{success}"] += 1
                self.histograms[f"duration_{intent}_{route}"].append(duration)

    def track_llm_call(
        self,
        model: str,
        duration: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Track LLM API call"""
        if self.enable_prometheus:
            self.llm_calls.labels(model=model, success=str(success)).inc()
            self.llm_duration.labels(model=model).observe(duration)
            if input_tokens > 0:
                self.llm_tokens.labels(model=model, type="input").inc(input_tokens)
            if output_tokens > 0:
                self.llm_tokens.labels(model=model, type="output").inc(output_tokens)
        else:
            with self.lock:
                self.counters[f"llm_{model}_{success}"] += 1
                self.histograms[f"llm_duration_{model}"].append(duration)

    def track_plugin(self, plugin: str, action: str, duration: float, success: bool):
        """Track plugin execution"""
        if self.enable_prometheus:
            self.plugin_calls.labels(
                plugin=plugin, action=action, success=str(success)
            ).inc()
            self.plugin_duration.labels(plugin=plugin, action=action).observe(duration)
        else:
            with self.lock:
                self.counters[f"plugin_{plugin}_{action}_{success}"] += 1
                self.histograms[f"plugin_duration_{plugin}_{action}"].append(duration)

    def track_cache(self, operation: str, result: str):
        """Track cache operation"""
        if self.enable_prometheus:
            self.cache_operations.labels(operation=operation, result=result).inc()
        else:
            with self.lock:
                self.counters[f"cache_{operation}_{result}"] += 1

    def track_error(self, error_type: str, component: str):
        """Track error occurrence"""
        if self.enable_prometheus:
            self.errors_total.labels(type=error_type, component=component).inc()
        else:
            with self.lock:
                self.counters[f"error_{error_type}_{component}"] += 1

    def track_learning(self, examples_total: int, quality_score: float):
        """Track learning metrics"""
        if self.enable_prometheus:
            self.learning_examples.set(examples_total)
            self.learning_quality.observe(quality_score)
        else:
            with self.lock:
                self.gauges["learning_examples"] = examples_total
                self.histograms["learning_quality"].append(quality_score)

    def track_intent_confidence(self, intent: str, confidence: float):
        """Track intent classification confidence"""
        if self.enable_prometheus:
            self.intent_confidence.labels(intent=intent).observe(confidence)
        else:
            with self.lock:
                self.histograms[f"confidence_{intent}"].append(confidence)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Set gauge value"""
        if self.enable_prometheus:
            # Dynamic gauge setting - would need pre-registration
            pass
        with self.lock:
            self.gauges[name] = value

    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ):
        """Increment counter"""
        with self.lock:
            self.counters[name] += value

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Record histogram value"""
        with self.lock:
            self.histograms[name].append(value)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get human-readable metrics summary"""
        with self.lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms_summary": {},
            }

            # Calculate histogram stats
            for name, values in self.histograms.items():
                if values:
                    summary["histograms_summary"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": sorted(values)[len(values) // 2] if values else 0,
                        "p95": (
                            sorted(values)[int(len(values) * 0.95)]
                            if len(values) > 1
                            else 0
                        ),
                        "p99": (
                            sorted(values)[int(len(values) * 0.99)]
                            if len(values) > 1
                            else 0
                        ),
                    }

            return summary

    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format"""
        if self.enable_prometheus:
            return generate_latest(REGISTRY)
        return b"# Prometheus not available\n"

    def reset(self):
        """Reset all metrics (for testing)"""
        with self.lock:
            self.counters.clear()
            self.histograms.clear()
            self.gauges.clear()
        logger.info("[Metrics] Reset all metrics")


# Decorator for automatic function timing
def track_time(
    metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None
):
    """
    Decorator to automatically track function execution time

    Usage:
        @track_time('nlp_processing')
        def process_nlp(text):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = kwargs.pop("_metrics", None) or get_metrics_collector()

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                name = metric_name or func.__name__

                # Record timing
                metrics.record_histogram(f"{name}_duration", duration, labels)
                metrics.increment_counter(
                    f"{name}_total", labels={"success": str(success)}
                )

        return wrapper

    return decorator


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(enable_prometheus: bool = True) -> MetricsCollector:
    """Initialize global metrics collector"""
    global _metrics_collector
    _metrics_collector = MetricsCollector(enable_prometheus=enable_prometheus)
    return _metrics_collector
