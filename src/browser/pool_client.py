"""
Unified Browser Pool Client

This client replaces ALL previous browser automation clients:
- e2e_client.py (E2EBrowserClient)
- stagehand_client.py (StagehandClient)
- browser_worker_client.py (BrowserWorkerClient)
- cloudflare_browser.py (CloudflareBrowserClient)

It provides a single, unified interface for browser automation
that routes to the Vultr Browser Pool (or fallback to Claude Computer Use).
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass

import httpx
import structlog

from src.browser.pool_models import (
    ActionResult,
    ActResult,
    BrowserPoolConfig,
    BrowserType,
    ElementInfo,
    ExecutionMode,
    ExtractResult,
    ObserveResult,
    PoolHealth,
    StepResult,
    TestResult,
)
from src.config import get_settings
from src.services.audit_logger import get_audit_logger

logger = structlog.get_logger(__name__)


# =====================================================
# JWT Token Signing for Browser Pool (Production-Grade)
# =====================================================

@dataclass
class UserContext:
    """User context for audit logging."""
    user_id: str
    org_id: str | None = None
    email: str | None = None
    ip: str | None = None


def _base64url_encode(data: bytes) -> str:
    """Base64URL encode (JWT-safe)."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def sign_pool_token(
    user_context: UserContext,
    secret: str,
    action: str = "",
    expires_in_seconds: int = 300
) -> str:
    """
    Sign a JWT token for browser pool authentication.

    Args:
        user_context: User information for audit
        secret: HMAC secret for signing
        action: Action being performed (observe, act, test)
        expires_in_seconds: Token validity period (default 5 minutes)

    Returns:
        Signed JWT token string
    """
    now = int(time.time())

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "argus-backend",
        "sub": user_context.user_id,
        "aud": "browser-pool",
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": str(uuid.uuid4()),
        "org_id": user_context.org_id,
        "email": user_context.email,
        "action": action,
        "ip": user_context.ip,
    }

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    header_b64 = _base64url_encode(json.dumps(header).encode())
    payload_b64 = _base64url_encode(json.dumps(payload).encode())

    signature = hmac.new(
        secret.encode(),
        f"{header_b64}.{payload_b64}".encode(),
        hashlib.sha256
    ).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


class BrowserPoolError(Exception):
    """Base exception for browser pool errors."""
    pass


class BrowserPoolTimeoutError(BrowserPoolError):
    """Timeout error from browser pool."""
    pass


class BrowserPoolUnavailableError(BrowserPoolError):
    """Browser pool is unavailable."""
    pass


class BrowserPoolClient:
    """
    Unified client for the Vultr Browser Pool.

    This is the SINGLE entry point for all browser automation in Argus.
    It provides MCP-compatible endpoints and handles:
    - Connection pooling
    - Automatic retries
    - Vision fallback (Claude Computer Use)
    - Selector caching
    - Health monitoring

    Usage:
        async with BrowserPoolClient() as client:
            # Discover elements
            result = await client.observe("https://example.com")

            # Execute action
            result = await client.act("https://example.com", "Click the login button")

            # Run multi-step test
            result = await client.test("https://example.com", [
                "Click login",
                "Fill email with test@example.com",
                "Click submit"
            ])
    """

    def __init__(
        self,
        pool_url: str | None = None,
        jwt_secret: str | None = None,
        api_key: str | None = None,  # Deprecated: Use jwt_secret
        user_context: UserContext | None = None,
        config: BrowserPoolConfig | None = None,
    ):
        """
        Initialize the browser pool client.

        Args:
            pool_url: URL of the browser pool (defaults to BROWSER_POOL_URL env var)
            jwt_secret: JWT secret for signing tokens (defaults to BROWSER_POOL_JWT_SECRET env var)
            api_key: Legacy API key (deprecated, use jwt_secret instead)
            user_context: User context for audit logging (optional)
            config: Optional configuration override
        """
        self.pool_url = pool_url or os.getenv(
            "BROWSER_POOL_URL",
            os.getenv("BROWSER_WORKER_URL", "http://localhost:8080")
        )

        # Production: JWT secret for signing tokens
        self.jwt_secret = jwt_secret or os.getenv("BROWSER_POOL_JWT_SECRET")

        # Legacy: Static API key (deprecated)
        self.api_key = api_key or os.getenv("BROWSER_POOL_API_KEY")

        # User context for audit logging
        self.user_context = user_context or UserContext(
            user_id=os.getenv("ARGUS_USER_ID", "system"),
            org_id=os.getenv("ARGUS_ORG_ID"),
        )

        self.config = config or BrowserPoolConfig(
            pool_url=self.pool_url,
            timeout_ms=int(os.getenv("BROWSER_TIMEOUT_MS", "60000")),
            retry_count=int(os.getenv("BROWSER_RETRY_COUNT", "3")),
            vision_fallback_enabled=os.getenv("VISION_FALLBACK_ENABLED", "true").lower() == "true",
        )

        self._client: httpx.AsyncClient | None = None
        self._selector_cache: dict[str, str] = {}
        self._health_cache: PoolHealth | None = None
        self._health_cache_time: float = 0

        # Vision fallback client (lazy initialized)
        self._vision_client = None

        # Log auth mode
        if self.jwt_secret:
            logger.info("BrowserPoolClient initialized with JWT authentication")
        elif self.api_key:
            logger.warning("BrowserPoolClient using deprecated API key auth. Migrate to JWT.")
        else:
            logger.warning("BrowserPoolClient running without authentication")

    async def __aenter__(self) -> "BrowserPoolClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            # Base headers (auth added per-request for JWT)
            headers = {"Content-Type": "application/json"}

            self._client = httpx.AsyncClient(
                base_url=self.pool_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout_ms / 1000),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )

    def _get_auth_header(self, action: str = "") -> dict[str, str]:
        """Get authentication header for request.

        Production: Signs a short-lived JWT with user context.
        Legacy: Uses static API key (deprecated).
        """
        headers = {}

        if self.jwt_secret:
            # Production: Sign JWT per-request
            token = sign_pool_token(
                user_context=self.user_context,
                secret=self.jwt_secret,
                action=action,
            )
            headers["Authorization"] = f"Bearer {token}"
        elif self.api_key:
            # Legacy: Static API key
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        retry_count: int | None = None,
    ) -> dict:
        """
        Make a request to the browser pool with automatic retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            retry_count: Number of retries (defaults to config)

        Returns:
            Response JSON

        Raises:
            BrowserPoolError: On request failure after retries
        """
        await self._ensure_client()

        retries = retry_count if retry_count is not None else self.config.retry_count
        last_error = None

        # Get auth header (JWT signed per-request)
        action = endpoint.lstrip("/").split("/")[0]  # e.g., "observe", "act", "test"
        auth_headers = self._get_auth_header(action=action)

        for attempt in range(retries + 1):
            try:
                if method.upper() == "GET":
                    response = await self._client.get(endpoint, headers=auth_headers)
                else:
                    response = await self._client.post(endpoint, json=data or {}, headers=auth_headers)

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_error = BrowserPoolTimeoutError(f"Request timed out: {e}")
                logger.warning(
                    "Browser pool request timeout",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    max_attempts=retries + 1,
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    last_error = BrowserPoolUnavailableError("No browsers available")
                else:
                    last_error = BrowserPoolError(f"HTTP error {e.response.status_code}: {e.response.text}")
                logger.warning(
                    "Browser pool HTTP error",
                    endpoint=endpoint,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                )

            except Exception as e:
                last_error = BrowserPoolError(f"Request failed: {e}")
                logger.warning(
                    "Browser pool request failed",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                )

            # Wait before retry
            if attempt < retries:
                await asyncio.sleep(self.config.retry_delay_ms / 1000)

        raise last_error

    def _cache_key(self, url: str, description: str) -> str:
        """Generate a cache key for selector caching."""
        content = f"{url}:{description}"
        return hashlib.md5(content.encode()).hexdigest()

    async def health(self, use_cache: bool = True) -> PoolHealth:
        """
        Check the health of the browser pool.

        Args:
            use_cache: Use cached health (valid for 30 seconds)

        Returns:
            PoolHealth with status information
        """
        # Check cache
        if use_cache and self._health_cache:
            if time.time() - self._health_cache_time < 30:
                return self._health_cache

        try:
            data = await self._request("GET", "/health")

            health = PoolHealth(
                healthy=data.get("status") == "healthy",
                total_pods=data.get("poolSize", 0),
                available_pods=data.get("available", data.get("availablePods", 0)),
                active_sessions=data.get("activeSessions", 0),
                pool_url=self.pool_url,
            )

            self._health_cache = health
            self._health_cache_time = time.time()

            return health

        except Exception as e:
            logger.error("Failed to check browser pool health", error=str(e))
            return PoolHealth(healthy=False, pool_url=self.pool_url)

    async def observe(
        self,
        url: str,
        instruction: str | None = None,
        use_cache: bool = True,
    ) -> ObserveResult:
        """
        Observe/discover interactive elements on a page.

        This is the MCP-compatible endpoint for element discovery.
        Equivalent to: argus_discover

        Args:
            url: URL of the page to analyze
            instruction: Optional instruction for what to look for
            use_cache: Use cached selectors if available

        Returns:
            ObserveResult with discovered elements
        """
        logger.info("Observing page", url=url, instruction=instruction)
        start_time = time.time()

        try:
            data = await self._request("POST", "/observe", {
                "url": url,
                "instruction": instruction or "What actions can I take on this page?",
            })

            # Parse elements
            elements = []
            for action in data.get("actions", []):
                elements.append(ElementInfo(
                    selector=action.get("selector", ""),
                    type=action.get("type", "unknown"),
                    tag_name=action.get("tagName", action.get("type", "unknown")),
                    text=action.get("text", ""),
                    value=action.get("value", ""),
                    placeholder=action.get("placeholder", ""),
                    description=action.get("description", ""),
                    confidence=action.get("confidence", 0.5),
                    bounds=action.get("bounds"),
                    attributes=action.get("attributes"),
                ))

            # Cache selectors
            if use_cache and self.config.enable_caching:
                for element in elements:
                    if element.description and element.confidence >= 0.8:
                        cache_key = self._cache_key(url, element.description)
                        self._selector_cache[cache_key] = element.selector

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info("Observe completed", url=url, elements_found=len(elements), duration_ms=duration_ms)

            # Log to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="observe",
                url=url,
                success=True,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                metadata={"elements_found": len(elements), "instruction": instruction},
            )

            return ObserveResult(
                success=data.get("success", True),
                url=data.get("url", url),
                title=data.get("title", ""),
                elements=elements,
            )

        except BrowserPoolError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Observe failed", url=url, error=str(e))

            # Log failure to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="observe",
                url=url,
                success=False,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                error=str(e),
            )

            return ObserveResult(
                success=False,
                url=url,
                error=str(e),
            )

    async def act(
        self,
        url: str,
        instruction: str,
        screenshot: bool = True,
        use_vision_fallback: bool = True,
    ) -> ActResult:
        """
        Execute a browser action using natural language instruction.

        This is the MCP-compatible endpoint for action execution.
        Equivalent to: argus_act

        Args:
            url: URL of the page
            instruction: Natural language instruction (e.g., "Click the login button")
            screenshot: Capture screenshot after action
            use_vision_fallback: Fall back to Claude Computer Use on failure

        Returns:
            ActResult with action outcome
        """
        logger.info("Executing action", url=url, instruction=instruction)
        start_time = time.time()

        try:
            data = await self._request("POST", "/act", {
                "url": url,
                "instruction": instruction,
                "screenshot": screenshot,
                "selfHeal": self.config.enable_self_healing,
            })

            # Parse actions
            actions = []
            for action in data.get("actions", []):
                actions.append(ActionResult(
                    action=action.get("action", "unknown"),
                    success=action.get("success", False),
                    selector=action.get("selector"),
                    value=action.get("value"),
                    url=action.get("url"),
                    error=action.get("error"),
                ))

            success = data.get("success", False)

            # Vision fallback if DOM execution failed
            if not success and use_vision_fallback and self.config.vision_fallback_enabled:
                logger.info("DOM execution failed, trying vision fallback", url=url, instruction=instruction)
                vision_result = await self._vision_fallback(url, instruction)
                if vision_result.success:
                    return vision_result

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info("Action completed", url=url, success=success, duration_ms=duration_ms)

            # Log to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="act",
                url=url,
                success=success,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                metadata={"instruction": instruction, "actions_count": len(actions)},
                error=data.get("error"),
            )

            return ActResult(
                success=success,
                message=data.get("message", ""),
                actions=actions,
                url=data.get("url", url),
                screenshot=data.get("screenshot"),
                error=data.get("error"),
                execution_mode=ExecutionMode.DOM,
            )

        except BrowserPoolError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Action failed", url=url, instruction=instruction, error=str(e))

            # Log failure to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="act",
                url=url,
                success=False,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                error=str(e),
                metadata={"instruction": instruction, "fallback_attempted": use_vision_fallback},
            )

            # Try vision fallback on pool failure
            if use_vision_fallback and self.config.vision_fallback_enabled:
                logger.info("Pool unavailable, trying vision fallback", url=url)
                return await self._vision_fallback(url, instruction)

            return ActResult(
                success=False,
                error=str(e),
                url=url,
            )

    async def test(
        self,
        url: str,
        steps: list[str],
        browser: BrowserType = BrowserType.CHROMIUM,
        capture_screenshots: bool = True,
    ) -> TestResult:
        """
        Run a multi-step test on a web application.

        This is the MCP-compatible endpoint for test execution.
        Equivalent to: argus_test

        Args:
            url: Starting URL for the test
            steps: List of test step instructions
            browser: Browser type to use
            capture_screenshots: Capture screenshot at each step

        Returns:
            TestResult with detailed step results
        """
        logger.info("Running test", url=url, step_count=len(steps))
        start_time = time.time()

        try:
            data = await self._request("POST", "/test", {
                "url": url,
                "steps": steps,
                "browser": browser.value,
                "captureScreenshots": capture_screenshots,
            })

            # Parse step results
            step_results = []
            for step_data in data.get("steps", []):
                actions = [
                    ActionResult(
                        action=a.get("action", ""),
                        success=a.get("success", False),
                        selector=a.get("selector"),
                        value=a.get("value"),
                        error=a.get("error"),
                    )
                    for a in step_data.get("actions", [])
                ]

                step_results.append(StepResult(
                    step_index=step_data.get("stepIndex", 0),
                    instruction=step_data.get("instruction", ""),
                    success=step_data.get("success", False),
                    actions=actions,
                    duration_ms=step_data.get("duration_ms", step_data.get("duration", 0)),
                    screenshot=step_data.get("screenshot"),
                    error=step_data.get("error"),
                ))

            success = data.get("success", False)
            duration_ms = int((time.time() - start_time) * 1000)

            steps_passed = sum(1 for s in step_results if s.success)
            logger.info(
                "Test completed",
                url=url,
                success=success,
                steps_passed=steps_passed,
                steps_total=len(step_results),
                duration_ms=duration_ms,
            )

            # Log to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="test",
                url=url,
                success=success,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                metadata={
                    "steps_total": len(steps),
                    "steps_passed": steps_passed,
                    "browser": browser.value,
                },
            )

            return TestResult(
                success=success,
                steps=step_results,
                final_screenshot=data.get("finalScreenshot"),
            )

        except BrowserPoolError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Test failed", url=url, error=str(e))

            # Log failure to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="test",
                url=url,
                success=False,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                error=str(e),
                metadata={"steps_total": len(steps), "browser": browser.value},
            )

            return TestResult(
                success=False,
                error=str(e),
            )

    async def extract(
        self,
        url: str,
        schema: dict[str, str],
        instruction: str | None = None,
    ) -> ExtractResult:
        """
        Extract structured data from a page.

        Args:
            url: URL of the page
            schema: Schema defining what to extract (e.g., {"title": "string", "price": "number"})
            instruction: Optional instruction for extraction

        Returns:
            ExtractResult with extracted data
        """
        logger.info("Extracting data", url=url, schema=schema)
        start_time = time.time()

        try:
            data = await self._request("POST", "/extract", {
                "url": url,
                "schema": schema,
                "instruction": instruction or f"Extract the following data: {json.dumps(schema)}",
            })

            duration_ms = int((time.time() - start_time) * 1000)

            # Log to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="extract",
                url=url,
                success=data.get("success", True),
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                metadata={"schema_fields": list(schema.keys())},
            )

            return ExtractResult(
                success=data.get("success", True),
                data=data.get("data", {}),
                url=data.get("url", url),
                screenshot=data.get("screenshot"),
            )

        except BrowserPoolError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Extraction failed", url=url, error=str(e))

            # Log failure to audit trail
            audit = get_audit_logger()
            await audit.log_browser_operation(
                operation="extract",
                url=url,
                success=False,
                duration_ms=duration_ms,
                user_id=self.user_context.user_id,
                organization_id=self.user_context.org_id,
                error=str(e),
            )

            return ExtractResult(
                success=False,
                url=url,
                error=str(e),
            )

    async def screenshot(
        self,
        url: str,
        full_page: bool = False,
        selector: str | None = None,
    ) -> str | None:
        """
        Capture a screenshot of a page.

        Args:
            url: URL of the page
            full_page: Capture full page or viewport only
            selector: Optional selector to capture specific element

        Returns:
            Base64 encoded screenshot or None on failure
        """
        try:
            data = await self._request("POST", "/screenshot", {
                "url": url,
                "fullPage": full_page,
                "selector": selector,
            })
            return data.get("screenshot")
        except Exception as e:
            logger.error("Screenshot failed", url=url, error=str(e))
            return None

    async def _vision_fallback(
        self,
        url: str,
        instruction: str,
    ) -> ActResult:
        """
        Execute action using Claude Computer Use as fallback.

        This is invoked when DOM-based execution fails.
        """
        try:
            # Lazy import to avoid circular dependencies
            from src.computer_use.client import ComputerUseClient

            if self._vision_client is None:
                settings = get_settings()
                self._vision_client = ComputerUseClient(settings)

            logger.info("Executing with vision fallback", url=url, instruction=instruction)

            # Execute with Computer Use
            result = await self._vision_client.execute_task(
                task=f"Go to {url} and {instruction}",
                max_iterations=10,
            )

            return ActResult(
                success=result.get("success", False),
                message=f"Executed via Claude Computer Use: {result.get('message', '')}",
                actions=[ActionResult(
                    action=instruction,
                    success=result.get("success", False),
                )],
                url=url,
                screenshot=result.get("screenshot"),
                execution_mode=ExecutionMode.VISION,
            )

        except ImportError:
            logger.warning("Computer Use client not available for vision fallback")
            return ActResult(
                success=False,
                error="Vision fallback not available (Computer Use client not installed)",
                url=url,
                execution_mode=ExecutionMode.VISION,
            )
        except Exception as e:
            logger.error("Vision fallback failed", url=url, error=str(e))
            return ActResult(
                success=False,
                error=f"Vision fallback failed: {e}",
                url=url,
                execution_mode=ExecutionMode.VISION,
            )


# Singleton instance for convenience
_default_client: BrowserPoolClient | None = None


def get_browser_pool_client() -> BrowserPoolClient:
    """
    Get the default browser pool client instance.

    Usage:
        client = get_browser_pool_client()
        result = await client.observe("https://example.com")
    """
    global _default_client
    if _default_client is None:
        _default_client = BrowserPoolClient()
    return _default_client


async def observe(url: str, instruction: str | None = None) -> ObserveResult:
    """Convenience function for observing a page."""
    return await get_browser_pool_client().observe(url, instruction)


async def act(url: str, instruction: str, **kwargs) -> ActResult:
    """Convenience function for executing an action."""
    return await get_browser_pool_client().act(url, instruction, **kwargs)


async def test(url: str, steps: list[str], **kwargs) -> TestResult:
    """Convenience function for running a test."""
    return await get_browser_pool_client().test(url, steps, **kwargs)
