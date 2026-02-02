"""CamelCase Response Middleware.

Automatically converts all JSON API responses from snake_case to camelCase.
This allows Python code to use idiomatic snake_case while providing
JavaScript-friendly camelCase responses.

Industry Standard: Stripe, GitHub, Twilio APIs all return camelCase JSON.

Usage:
    app.add_middleware(CamelCaseMiddleware)
"""

import json
import re
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

logger = structlog.get_logger()

# Pre-compiled regex patterns for better performance
_CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
_CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')

# Maximum body size to process (10MB) - prevents memory exhaustion
MAX_BODY_SIZE = 10 * 1024 * 1024

# Maximum recursion depth for nested objects
MAX_RECURSION_DEPTH = 50


def to_camel_case(snake_str: str) -> str:
    """Convert snake_case string to camelCase.

    Examples:
        user_name -> userName
        created_at -> createdAt
        api_key_id -> apiKeyId
    """
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def convert_keys_to_camel_case(obj, _depth: int = 0):
    """Recursively convert all dictionary keys from snake_case to camelCase."""
    if _depth > MAX_RECURSION_DEPTH:
        logger.warning(
            "camelcase_middleware.max_depth_exceeded",
            depth=_depth,
            note="Returning unconverted value to prevent stack overflow",
        )
        return obj

    if isinstance(obj, dict):
        return {
            to_camel_case(k) if isinstance(k, str) and '_' in k else k: convert_keys_to_camel_case(v, _depth + 1)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_keys_to_camel_case(item, _depth + 1) for item in obj]
    else:
        return obj


def to_snake_case(camel_str: str) -> str:
    """Convert camelCase string to snake_case.

    Examples:
        userName -> user_name
        createdAt -> created_at
        apiKeyId -> api_key_id
    """
    # Only convert if string has camelCase pattern (lowercase followed by uppercase)
    if not _CAMEL_PATTERN_2.search(camel_str):
        return camel_str
    # Insert underscore before uppercase letters and lowercase them
    s1 = _CAMEL_PATTERN_1.sub(r'\1_\2', camel_str)
    return _CAMEL_PATTERN_2.sub(r'\1_\2', s1).lower()


def convert_keys_to_snake_case(obj, _depth: int = 0):
    """Recursively convert all dictionary keys from camelCase to snake_case."""
    if _depth > MAX_RECURSION_DEPTH:
        logger.warning(
            "camelcase_middleware.max_depth_exceeded",
            depth=_depth,
            note="Returning unconverted value to prevent stack overflow",
        )
        return obj

    if isinstance(obj, dict):
        return {
            to_snake_case(k) if isinstance(k, str) else k: convert_keys_to_snake_case(v, _depth + 1)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_keys_to_snake_case(item, _depth + 1) for item in obj]
    else:
        return obj


class CamelCaseMiddleware(BaseHTTPMiddleware):
    """Middleware that converts API responses to camelCase and requests to snake_case.

    This provides a clean API contract:
    - Frontend sends/receives camelCase (JavaScript convention)
    - Backend uses snake_case internally (Python convention)

    Excluded paths:
    - /health, /metrics (internal endpoints)
    - /docs, /openapi.json (documentation)
    - Non-JSON responses
    """

    # Paths to exclude from conversion
    EXCLUDED_PATHS = {
        '/health',
        '/metrics',
        '/docs',
        '/redoc',
        '/openapi.json',
        '/favicon.ico',
    }

    # Streaming endpoints that must bypass all body processing
    STREAMING_ENDPOINTS = {
        '/api/v1/chat/stream',
        '/api/v1/stream/test',
        '/api/v1/browser/execute',
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            return await call_next(request)

        # CRITICAL: Skip ALL processing for streaming endpoints
        # BaseHTTPMiddleware doesn't work well with StreamingResponse
        if any(request.url.path.startswith(ep) for ep in self.STREAMING_ENDPOINTS):
            logger.debug(
                "camelcase_middleware.streaming_bypass",
                path=request.url.path,
                note="Bypassing camelcase middleware for streaming endpoint",
            )
            return await call_next(request)

        # Convert request body from camelCase to snake_case for JSON requests
        if request.method in ('POST', 'PUT', 'PATCH') and request.headers.get('content-type', '').startswith('application/json'):
            await self._convert_request_body(request)

        # Call the next handler
        response = await call_next(request)

        # Convert response body from snake_case to camelCase
        return await self._convert_response_body(request, response)

    async def _convert_request_body(self, request: Request) -> None:
        """Convert request body keys from camelCase to snake_case."""
        try:
            body = await request.body()
            if not body:
                return

            if len(body) > MAX_BODY_SIZE:
                logger.warning(
                    "camelcase_middleware.request_too_large",
                    path=request.url.path,
                    size=len(body),
                    limit=MAX_BODY_SIZE,
                )
                return

            json_body = json.loads(body)
            converted_body = convert_keys_to_snake_case(json_body)
            converted_bytes = json.dumps(converted_body).encode('utf-8')

            # Create a proper receive function that follows ASGI protocol
            async def receive():
                return {
                    'type': 'http.request',
                    'body': converted_bytes,
                    'more_body': False,
                }

            # Replace the receive function
            request._receive = receive

        except json.JSONDecodeError as e:
            logger.warning(
                "camelcase_middleware.request_parse_failed",
                error=str(e),
                path=request.url.path,
                method=request.method,
            )
        except UnicodeDecodeError as e:
            logger.warning(
                "camelcase_middleware.request_unicode_error",
                error=str(e),
                path=request.url.path,
                method=request.method,
            )
        except Exception as e:
            logger.error(
                "camelcase_middleware.request_conversion_error",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
            )

    async def _convert_response_body(self, request: Request, response: Response) -> Response:
        """Convert response body keys from snake_case to camelCase."""
        # Skip non-JSON responses
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('application/json'):
            return response

        # Skip streaming responses (SSE, file downloads)
        if isinstance(response, StreamingResponse):
            logger.debug(
                "camelcase_middleware.streaming_skipped",
                path=request.url.path,
                note="Streaming responses are not converted",
            )
            return response

        try:
            # Read the response body
            body = b''
            async for chunk in response.body_iterator:
                body += chunk
                if len(body) > MAX_BODY_SIZE:
                    logger.warning(
                        "camelcase_middleware.response_too_large",
                        path=request.url.path,
                        size=len(body),
                        limit=MAX_BODY_SIZE,
                    )
                    # Return unconverted response
                    return self._create_response(body, response, content_type)

            if not body:
                return response

            json_body = json.loads(body)
            converted_body = convert_keys_to_camel_case(json_body)
            new_body = json.dumps(converted_body, default=str)

            # Create new response with converted body
            # Note: We exclude Content-Length from copied headers - Response will recalculate
            headers = {k: v for k, v in response.headers.items() if k.lower() != 'content-length'}

            return Response(
                content=new_body,
                status_code=response.status_code,
                headers=headers,
                media_type='application/json',
            )

        except json.JSONDecodeError as e:
            logger.error(
                "camelcase_middleware.response_parse_failed",
                error=str(e),
                path=request.url.path,
                status_code=response.status_code,
            )
            return self._create_response(body, response, content_type)

        except UnicodeDecodeError as e:
            logger.error(
                "camelcase_middleware.response_unicode_error",
                error=str(e),
                path=request.url.path,
                status_code=response.status_code,
            )
            return self._create_response(body, response, content_type)

        except Exception as e:
            logger.error(
                "camelcase_middleware.response_conversion_error",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
            )
            return self._create_response(body, response, content_type)

    def _create_response(self, body: bytes, original_response: Response, content_type: str) -> Response:
        """Create a response with the original body when conversion fails."""
        headers = {k: v for k, v in original_response.headers.items() if k.lower() != 'content-length'}
        return Response(
            content=body,
            status_code=original_response.status_code,
            headers=headers,
            media_type=content_type,
        )
