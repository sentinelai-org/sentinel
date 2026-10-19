"""Enterprise API Security: OWASP headers, Sliding Window Rate Limiting, and API Key Auth."""

from __future__ import annotations

import hmac
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

# Security scheme for FastAPI OpenAPI docs
bearer_scheme = HTTPBearer(auto_error=False)

# Paths exempt from rate limits and API key authentication
PUBLIC_PATHS: set[str] = {
    "/",
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/docs",
    "/openapi.json",
    "/ui",
    "/admin",
    "/favicon.ico",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces OWASP-recommended HTTP response security headers."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response = await call_next(request)

        # Apply security headers to protect against clickjacking, sniffing, and XSS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"

        # Content-Security-Policy (allows self and cdnjs/jsdelivr for local UI rendering)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self';"
        )
        response.headers["Content-Security-Policy"] = csp

        return response


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter supporting Redis and thread-safe in-memory fallback."""

    def __init__(self, max_requests: int = 120, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # In-memory store: client_key -> list of timestamp floats
        self._memory_store: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)."""
        now = time.time()
        window_start = now - self.window_seconds

        # Prune old timestamps
        history = self._memory_store.get(client_id, [])
        valid_history = [ts for ts in history if ts > window_start]

        if len(valid_history) >= self.max_requests:
            oldest_ts = valid_history[0]
            retry_after = max(1, int(oldest_ts + self.window_seconds - now))
            return False, retry_after

        valid_history.append(now)
        self._memory_store[client_id] = valid_history
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces rate limiting per IP or API key to prevent DoS and brute-force attacks."""

    def __init__(self, app: Any, rate_limiter: SlidingWindowRateLimiter) -> None:
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/v1/admin"):
            return await call_next(request)

        settings = getattr(request.app.state, "settings", None) or get_settings()
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Derive identifier: Prefer API key if provided, otherwise client IP
        auth_header = request.headers.get("Authorization") or ""
        client_ip = request.client.host if request.client else "127.0.0.1"
        client_key = auth_header.strip() if auth_header else client_ip

        allowed, retry_after = self.rate_limiter.is_allowed(client_key)
        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={"client": client_ip, "path": path, "retry_after": retry_after},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Protects server memory by enforcing a maximum incoming request payload size."""

    def __init__(self, app: Any, max_size_bytes: int = 2 * 1024 * 1024) -> None:
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size_bytes:
            logger.warning(
                "Request payload size exceeded limit",
                extra={"content_length": content_length, "max_allowed": self.max_size_bytes},
            )
            status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
            return JSONResponse(
                status_code=status_413,
                content={
                    "error": "Payload Too Large",
                    "message": f"Request body exceeds maximum allowed size ({self.max_size_bytes} bytes).",
                },
            )

        return await call_next(request)


async def verify_api_key(
    request: Request,
    auth: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str:
    """FastAPI Dependency for verifying Bearer token or X-API-Key with Enterprise Authorization."""
    settings = getattr(request.app.state, "settings", None) or get_settings()

    # Check path whitelist
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/v1/admin"):
        return "public"

    # Check Bearer token first, then fallback to X-API-Key header
    provided_key: str | None = None
    if auth and auth.credentials:
        provided_key = auth.credentials
    else:
        provided_key = request.headers.get("X-API-Key")

    enterprise_manager = getattr(request.app.state, "enterprise_manager", None)
    if provided_key and enterprise_manager:
        is_valid, enterprise, env, is_authorized, error_msg = enterprise_manager.validate_key(
            provided_key
        )
        if is_valid:
            if not is_authorized:
                logger.warning(
                    "Enterprise authorization denied",
                    extra={
                        "enterprise_id": enterprise.id if enterprise else "unknown",
                        "environment": env,
                        "path": request.url.path,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=error_msg,
                )
            # Record telemetry / usage
            if enterprise:
                enterprise_manager.record_usage(enterprise.id, env)
                request.state.enterprise = enterprise
                request.state.environment = env
            return provided_key

    if not settings.api_key_auth_enabled:
        return provided_key or "anonymous"

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide Authorization: Bearer <key> or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison for static keys configured in .env
    valid = any(
        hmac.compare_digest(provided_key, valid_key) for valid_key in settings.sentinel_api_keys
    )

    if not valid:
        logger.warning("Invalid API key attempt", extra={"path": request.url.path})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_key
