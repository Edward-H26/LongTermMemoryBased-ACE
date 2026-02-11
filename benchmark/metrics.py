"""
Token usage and latency metrics collector for benchmarking.

Re-exports RequestMetrics and MetricsCollector from src.llm for
benchmark-specific usage with additional export utilities.
"""

from src.llm import RequestMetrics, MetricsCollector, get_metrics_collector

__all__ = ["RequestMetrics", "MetricsCollector", "get_metrics_collector"]
