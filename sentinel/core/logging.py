"""Structured JSON logging with contextvars-based request_id propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

# Context variable storing the current request ID
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def bind_request_id(request_id: str | None) -> None:
    """Bind a request_id to the current async context."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    """Retrieve the current context request_id."""
    return _request_id_ctx.get()


def _get_trace_context() -> tuple[str | None, str | None]:
    """Extract trace_id and span_id from current active OpenTelemetry span."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid and ctx.trace_id != 0:
        return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    return None, None


class JSONFormatter(logging.Formatter):
    """Custom logging formatter outputting standard JSON objects with trace correlation."""

    def format(self, record: logging.LogRecord) -> str:
        trace_id, span_id = _get_trace_context()
        log_record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "trace_id": trace_id,
            "span_id": span_id,
        }

        # Include exception trace if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields passed to logger
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extras:
            log_record["extra"] = extras

        return json.dumps(log_record, default=str)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger to output structured JSON."""
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Avoid duplicate handlers if setup_logging is called multiple times
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Silence overly noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
