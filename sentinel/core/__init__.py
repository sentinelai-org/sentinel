"""Sentinel core package for settings, logging, and error handling."""

from sentinel.core.config import Settings, get_settings
from sentinel.core.errors import (
    CacheError,
    ProviderError,
    ProviderRateLimitError,
    SentinelError,
    VectorStoreError,
)
from sentinel.core.logging import get_logger, setup_logging
from sentinel.core.telemetry import (
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    setup_telemetry,
    shutdown_telemetry,
)

__all__ = [
    "Settings",
    "get_settings",
    "SentinelError",
    "ProviderError",
    "ProviderRateLimitError",
    "VectorStoreError",
    "CacheError",
    "get_logger",
    "setup_logging",
    "get_tracer",
    "setup_telemetry",
    "shutdown_telemetry",
    "get_current_trace_id",
    "get_current_span_id",
]
