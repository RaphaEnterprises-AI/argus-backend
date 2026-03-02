"""
Centralized Streaming Configuration and Utilities

This module provides a single source of truth for streaming endpoint detection,
solving the BaseHTTPMiddleware + StreamingResponse incompatibility issue.

The Problem:
    Starlette's BaseHTTPMiddleware buffers the entire response body, which breaks
    StreamingResponse and causes "Unexpected message received: http.request" errors.

The Solution:
    1. Centralized streaming endpoint registry (this module)
    2. All middleware imports from here - single source of truth
    3. Pure ASGI middleware base class for new middleware (recommended)

Usage:
    from src.api.middleware.streaming import is_streaming_endpoint, STREAMING_ENDPOINTS

    # In middleware dispatch:
    if is_streaming_endpoint(request.url.path):
        return await call_next(request)  # Bypass processing
"""

from typing import Callable, Set
from starlette.requests import Request
from starlette.responses import Response

# =============================================================================
# STREAMING ENDPOINTS REGISTRY - Single Source of Truth
# =============================================================================
# Add all streaming endpoints here. All middleware will respect this list.
# Use path prefixes for flexibility (e.g., "/api/v1/chat/stream" matches
# "/api/v1/chat/stream" and "/api/v1/chat/stream/session-123")

STREAMING_ENDPOINTS: Set[str] = {
    # Chat streaming (SSE)
    "/api/v1/chat/stream",

    # Swarm AG-UI streaming (SSE via EventSource)
    "/api/v1/swarms/",

    # Test streaming endpoint
    "/api/v1/stream/test",

    # Browser automation (may stream results)
    "/api/v1/browser/execute",

    # LangGraph streaming (if used)
    "/api/v1/runs/stream",

    # WebSocket upgrade paths (not SSE but also shouldn't be buffered)
    "/api/v1/ws",
}

# Paths that should completely bypass all middleware processing
# (e.g., health checks that need to be fast)
BYPASS_MIDDLEWARE_PATHS: Set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/metrics",
}


def is_streaming_endpoint(path: str) -> bool:
    """
    Check if a request path is a streaming endpoint.

    Uses prefix matching so /api/v1/chat/stream/session-123 matches.

    Args:
        path: The request URL path

    Returns:
        True if this is a streaming endpoint that should bypass body processing
    """
    return any(path.startswith(endpoint) for endpoint in STREAMING_ENDPOINTS)


def should_bypass_middleware(path: str) -> bool:
    """
    Check if a request should completely bypass middleware.

    Args:
        path: The request URL path

    Returns:
        True if middleware should be skipped entirely
    """
    return path in BYPASS_MIDDLEWARE_PATHS or is_streaming_endpoint(path)


# =============================================================================
# PURE ASGI MIDDLEWARE BASE CLASS
# =============================================================================
# Use this instead of BaseHTTPMiddleware for new middleware to avoid issues.

class PureASGIMiddleware:
    """
    Base class for pure ASGI middleware that properly handles streaming.

    Unlike BaseHTTPMiddleware, this doesn't buffer responses and works
    correctly with StreamingResponse, EventSourceResponse, and WebSockets.

    Usage:
        class MyMiddleware(PureASGIMiddleware):
            async def process_request(self, scope, receive):
                # Pre-request processing
                # Return None to continue, or (status, headers, body) to short-circuit
                return None

            async def process_response(self, scope, message):
                # Intercept response headers/body
                # Return modified message or original
                return message

    Example:
        class LoggingMiddleware(PureASGIMiddleware):
            async def process_request(self, scope, receive):
                logger.info(f"Request: {scope['path']}")
                return None

            async def process_response(self, scope, message):
                if message["type"] == "http.response.start":
                    logger.info(f"Response status: {message['status']}")
                return message
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # Pass through non-HTTP requests (WebSocket, lifespan, etc.)
            await self.app(scope, receive, send)
            return

        # Check for bypass
        path = scope.get("path", "")
        if should_bypass_middleware(path):
            await self.app(scope, receive, send)
            return

        # Pre-request processing
        short_circuit = await self.process_request(scope, receive)
        if short_circuit is not None:
            status, headers, body = short_circuit
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
            })
            await send({
                "type": "http.response.body",
                "body": body.encode() if isinstance(body, str) else body,
            })
            return

        # Wrap send to intercept responses
        async def send_wrapper(message):
            processed = await self.process_response(scope, message)
            await send(processed)

        await self.app(scope, receive, send_wrapper)

    async def process_request(self, scope: dict, receive: Callable) -> tuple | None:
        """
        Override to process requests before they reach the application.

        Args:
            scope: ASGI scope dict
            receive: ASGI receive callable

        Returns:
            None to continue processing, or (status, headers_dict, body) to short-circuit
        """
        return None

    async def process_response(self, scope: dict, message: dict) -> dict:
        """
        Override to process/modify response messages.

        Args:
            scope: ASGI scope dict
            message: ASGI message dict (http.response.start or http.response.body)

        Returns:
            The message dict (modified or original)
        """
        return message


# =============================================================================
# HELPER: Streaming-Safe Response Wrapper
# =============================================================================

def add_streaming_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """
    Add headers that prevent infrastructure buffering for streaming responses.

    Args:
        headers: List of (name, value) header tuples

    Returns:
        Headers with streaming-safe additions
    """
    streaming_headers = [
        (b"x-accel-buffering", b"no"),  # Nginx
        (b"cache-control", b"no-cache, no-transform"),
    ]

    # Don't duplicate headers
    existing_names = {h[0].lower() for h in headers}
    for name, value in streaming_headers:
        if name not in existing_names:
            headers.append((name, value))

    return headers
