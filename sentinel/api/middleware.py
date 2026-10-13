"""HTTP middleware for request tracing and timing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from sentinel.core.logging import bind_request_id, get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds request_id to contextvars and logs request lifecycle."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        start_time = time.perf_counter()

        # Extract or generate request ID
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        bind_request_id(req_id)

        # Store in request state for convenient route access
        request.state.request_id = req_id
        request.state.start_time = start_time

        logger.info(
            "HTTP request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

            logger.info(
                "HTTP request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "HTTP request unhandled exception",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                    "latency_ms": round(elapsed_ms, 2),
                },
                exc_info=True,
            )
            raise
