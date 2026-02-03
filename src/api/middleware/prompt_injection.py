"""
Prompt Injection Protection Middleware

RAP-331: Provides prompt injection detection and input sanitization
at the API layer. Uses the existing PromptInjectionDetector and
InputSanitizer from the guardrails module.

This middleware:
1. Detects adversarial patterns in user input
2. Sanitizes input before it reaches AI prompts
3. Logs blocked attempts to the audit trail
4. Provides a dependency for endpoint-level protection
"""

import json
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.middleware.streaming import is_streaming_endpoint
from src.guardrails.validators import (
    InputSanitizer,
    PromptInjectionDetector,
    ValidationResult,
)
from src.services.audit_logger import (
    AuditAction,
    AuditStatus,
    ResourceType,
    get_audit_logger,
)

logger = structlog.get_logger()

# Singleton instances for efficiency
_injection_detector: PromptInjectionDetector | None = None
_input_sanitizer: InputSanitizer | None = None


def get_injection_detector() -> PromptInjectionDetector:
    """Get singleton PromptInjectionDetector instance."""
    global _injection_detector
    if _injection_detector is None:
        _injection_detector = PromptInjectionDetector()
    return _injection_detector


def get_input_sanitizer() -> InputSanitizer:
    """Get singleton InputSanitizer instance."""
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = InputSanitizer()
    return _input_sanitizer


@dataclass
class PromptProtectionResult:
    """Result of prompt injection protection check."""

    is_safe: bool
    original_text: str
    sanitized_text: str | None = None
    blocked_reason: str | None = None
    detected_patterns: list[str] | None = None
    confidence: float = 0.0
    sanitization_issues: list[str] | None = None


async def check_prompt_injection(
    text: str,
    context: dict[str, Any] | None = None,
    block_on_detection: bool = True,
    sanitize: bool = True,
    user_id: str | None = None,
    request_path: str | None = None,
) -> PromptProtectionResult:
    """Check text for prompt injection and optionally sanitize.

    This is the core function for prompt injection protection.
    Can be used directly in endpoints or via the middleware.

    Args:
        text: The text to check (user input, message content, etc.)
        context: Optional context for multi-turn detection
        block_on_detection: If True, raises HTTPException on detection
        sanitize: If True, sanitizes input even if no injection detected
        user_id: User ID for audit logging
        request_path: Request path for audit logging

    Returns:
        PromptProtectionResult with safety status and sanitized text

    Raises:
        HTTPException: 400 if block_on_detection=True and injection detected
    """
    detector = get_injection_detector()
    sanitizer = get_input_sanitizer()
    audit = get_audit_logger()

    # Check for prompt injection
    if context:
        # Use multi-turn detection if context provided
        previous_messages = context.get("previous_messages", [])
        detection_result = detector.detect_with_context(text, previous_messages)
    else:
        detection_result = detector.detect(text)

    # If injection detected
    if not detection_result.is_valid:
        # Log to audit trail
        await audit.log(
            action=AuditAction.SECURITY_PROMPT_INJECTION_BLOCKED,
            resource_type=ResourceType.SECURITY,
            description=f"Prompt injection attempt blocked: {detection_result.issues[0][:100] if detection_result.issues else 'Unknown pattern'}",
            user_id=user_id or "anonymous",
            status=AuditStatus.SUCCESS,
            metadata={
                "detected_patterns": detection_result.issues,
                "confidence": detection_result.confidence,
                "input_length": len(text),
                "input_preview": text[:200] + "..." if len(text) > 200 else text,
                "request_path": request_path,
            },
        )

        logger.warning(
            "Prompt injection detected and blocked",
            user_id=user_id,
            patterns=detection_result.issues,
            confidence=detection_result.confidence,
            request_path=request_path,
        )

        if block_on_detection:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "prompt_injection_detected",
                    "message": "Your input contains patterns that are not allowed. Please rephrase your request.",
                    "code": "SECURITY_001",
                },
            )

        return PromptProtectionResult(
            is_safe=False,
            original_text=text,
            sanitized_text=None,
            blocked_reason="Prompt injection patterns detected",
            detected_patterns=detection_result.issues,
            confidence=detection_result.confidence,
        )

    # Sanitize input if requested
    sanitized_text = text
    sanitization_issues: list[str] = []

    if sanitize:
        sanitization_result = sanitizer.sanitize(text)
        sanitized_text = sanitization_result.sanitized_input or text
        sanitization_issues = sanitization_result.issues

        # Log if sanitization made changes
        if sanitization_issues:
            await audit.log(
                action=AuditAction.SECURITY_INPUT_SANITIZED,
                resource_type=ResourceType.SECURITY,
                description=f"Input sanitized: {', '.join(sanitization_issues[:3])}",
                user_id=user_id or "anonymous",
                status=AuditStatus.SUCCESS,
                metadata={
                    "sanitization_issues": sanitization_issues,
                    "original_length": len(text),
                    "sanitized_length": len(sanitized_text),
                    "request_path": request_path,
                },
            )

            logger.info(
                "Input sanitized",
                user_id=user_id,
                issues=sanitization_issues,
                request_path=request_path,
            )

    return PromptProtectionResult(
        is_safe=True,
        original_text=text,
        sanitized_text=sanitized_text,
        sanitization_issues=sanitization_issues if sanitization_issues else None,
    )


async def validate_chat_messages(
    messages: list[dict[str, str]],
    user_id: str | None = None,
    request_path: str | None = None,
) -> list[dict[str, str]]:
    """Validate and sanitize a list of chat messages.

    Checks each user message for prompt injection and sanitizes content.

    Args:
        messages: List of chat messages with 'role' and 'content' keys
        user_id: User ID for audit logging
        request_path: Request path for audit logging

    Returns:
        List of sanitized messages

    Raises:
        HTTPException: 400 if prompt injection detected in any message
    """
    sanitized_messages = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Only check user messages (assistant/system messages are trusted)
        if role == "user" and content:
            result = await check_prompt_injection(
                text=content,
                block_on_detection=True,
                sanitize=True,
                user_id=user_id,
                request_path=request_path,
            )
            sanitized_messages.append({
                **msg,
                "content": result.sanitized_text or content,
            })
        else:
            sanitized_messages.append(msg)

    return sanitized_messages


# =============================================================================
# FastAPI Dependencies
# =============================================================================


class PromptProtectionDep:
    """Dependency for endpoint-level prompt injection protection.

    Usage:
        @router.post("/chat")
        async def chat(
            request: ChatRequest,
            protection: PromptProtectionDep = Depends()
        ):
            # Validate messages
            safe_messages = await protection.validate_messages(
                request.messages,
                user_id=user.user_id
            )
    """

    def __init__(self, request: Request):
        self.request = request
        self.user_id = getattr(
            getattr(request.state, "user", None), "user_id", None
        )
        self.request_path = str(request.url.path)

    async def check_text(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        block_on_detection: bool = True,
    ) -> PromptProtectionResult:
        """Check single text for prompt injection."""
        return await check_prompt_injection(
            text=text,
            context=context,
            block_on_detection=block_on_detection,
            sanitize=True,
            user_id=self.user_id,
            request_path=self.request_path,
        )

    async def validate_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Validate and sanitize chat messages."""
        return await validate_chat_messages(
            messages=messages,
            user_id=self.user_id,
            request_path=self.request_path,
        )


# =============================================================================
# Middleware (Optional - for global protection)
# =============================================================================


class PromptInjectionMiddleware(BaseHTTPMiddleware):
    """Middleware for global prompt injection protection.

    This middleware checks POST/PUT/PATCH request bodies for prompt injection.
    Only applies to JSON requests with 'content' or 'messages' fields.

    Note: For more granular control, use PromptProtectionDep in endpoints.

    Usage:
        app.add_middleware(PromptInjectionMiddleware)
    """

    # Paths to protect (others are skipped)
    PROTECTED_PATHS = [
        "/api/v1/chat/",
        "/api/v1/tests/generate",
        "/api/v1/healing/",
    ]

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and check for prompt injection."""
        # Only check mutation methods
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        # CRITICAL: Skip body reading for streaming endpoints
        # BaseHTTPMiddleware doesn't work well with StreamingResponse
        # Uses centralized config from streaming.py
        if is_streaming_endpoint(request.url.path):
            logger.debug(
                "prompt_injection_middleware.streaming_bypass",
                path=request.url.path,
            )
            return await call_next(request)

        # Only check protected paths
        path = request.url.path
        if not any(path.startswith(p) for p in self.PROTECTED_PATHS):
            return await call_next(request)

        # Only check JSON content
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return await call_next(request)

        # Read and check body
        try:
            body = await request.body()
            if not body:
                return await call_next(request)

            data = json.loads(body)

            # Get user context
            user = getattr(request.state, "user", None)
            user_id = getattr(user, "user_id", None) if user else None

            # Check for content field (single text)
            if "content" in data and isinstance(data["content"], str):
                await check_prompt_injection(
                    text=data["content"],
                    block_on_detection=True,
                    sanitize=False,  # Don't modify body in middleware
                    user_id=user_id,
                    request_path=path,
                )

            # Check for messages field (chat messages)
            if "messages" in data and isinstance(data["messages"], list):
                for msg in data["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if content:
                            await check_prompt_injection(
                                text=content,
                                block_on_detection=True,
                                sanitize=False,
                                user_id=user_id,
                                request_path=path,
                            )

        except json.JSONDecodeError:
            # Not valid JSON, let the endpoint handle it
            pass
        except HTTPException:
            # Re-raise our security exceptions
            raise
        except Exception as e:
            # Log but don't block on unexpected errors
            logger.warning(
                "Error in prompt injection middleware",
                error=str(e),
                path=path,
            )

        return await call_next(request)
