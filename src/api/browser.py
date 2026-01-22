"""Browser Pool API endpoints.

Provides REST endpoints for browser automation via the Vultr Browser Pool.
This replaces the deprecated Cloudflare Worker endpoints.
"""

from enum import Enum

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.teams import get_current_user
from src.browser.pool_client import (
    BrowserPoolClient,
    BrowserPoolError,
    BrowserPoolTimeoutError,
    BrowserPoolUnavailableError,
    UserContext,
)
from src.browser.pool_models import BrowserType

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/browser", tags=["Browser Pool"])


# =============================================================================
# Request/Response Models
# =============================================================================


class BrowserTypeEnum(str, Enum):
    """Browser type selection."""
    chrome = "chromium"
    chromium = "chromium"
    firefox = "firefox"
    edge = "edge"
    webkit = "webkit"


class TestStepResult(BaseModel):
    """Result of a single test step."""
    step_index: int
    instruction: str
    success: bool
    duration_ms: int = 0
    screenshot: str | None = None
    error: str | None = None
    actions: list[dict] = Field(default_factory=list)


class TestRequest(BaseModel):
    """Request to execute a browser test.

    Compatible with the legacy Cloudflare Worker API for easy migration.
    """
    url: str = Field(..., description="URL to test")
    steps: list[str] = Field(..., description="List of test step instructions")
    browser: BrowserTypeEnum = Field(default=BrowserTypeEnum.chrome, description="Browser type")
    screenshot: bool = Field(default=True, description="Capture screenshots at each step")
    record_video: bool = Field(default=False, description="Record video of test execution")
    timeout: int = Field(default=45000, description="Timeout in milliseconds per step")
    verbose: bool = Field(default=False, description="Include verbose output")


class TestResponse(BaseModel):
    """Response from test execution.

    Compatible with the legacy Cloudflare Worker API response format.
    """
    success: bool
    steps: list[TestStepResult]
    final_screenshot: str | None = None
    video_artifact_id: str | None = None
    error: str | None = None
    duration_ms: int = 0
    browser: str = "chromium"


class ObserveRequest(BaseModel):
    """Request to observe/discover elements on a page."""
    url: str = Field(..., description="URL to observe")
    instruction: str | None = Field(None, description="What to look for on the page")


class ObserveResponse(BaseModel):
    """Response from page observation."""
    success: bool
    url: str
    title: str = ""
    elements: list[dict] = Field(default_factory=list)
    error: str | None = None


class ActRequest(BaseModel):
    """Request to execute a single browser action."""
    url: str = Field(..., description="URL of the page")
    instruction: str = Field(..., description="Action to perform (natural language)")
    screenshot: bool = Field(default=True, description="Capture screenshot after action")


class ActResponse(BaseModel):
    """Response from action execution."""
    success: bool
    message: str = ""
    url: str
    screenshot: str | None = None
    error: str | None = None
    actions: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Browser pool health status."""
    healthy: bool
    total_pods: int = 0
    available_pods: int = 0
    active_sessions: int = 0
    pool_url: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def _get_browser_type(browser: BrowserTypeEnum) -> BrowserType:
    """Convert API browser enum to pool client browser type."""
    mapping = {
        BrowserTypeEnum.chrome: BrowserType.CHROMIUM,
        BrowserTypeEnum.chromium: BrowserType.CHROMIUM,
        BrowserTypeEnum.firefox: BrowserType.FIREFOX,
        BrowserTypeEnum.edge: BrowserType.EDGE,
        BrowserTypeEnum.webkit: BrowserType.WEBKIT,
    }
    return mapping.get(browser, BrowserType.CHROMIUM)


async def _get_browser_client(request: Request) -> BrowserPoolClient:
    """Get a browser pool client with user context for audit logging."""
    try:
        user = await get_current_user(request)
        user_context = UserContext(
            user_id=user["user_id"],
            org_id=user.get("org_id"),
            email=user.get("email"),
            ip=request.client.host if request.client else None,
        )
    except Exception:
        # Allow unauthenticated requests with anonymous context
        user_context = UserContext(
            user_id="anonymous",
            ip=request.client.host if request.client else None,
        )

    return BrowserPoolClient(user_context=user_context)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def get_browser_pool_health(request: Request):
    """
    Check the health of the browser pool.

    Returns information about available browser instances.
    """
    try:
        async with BrowserPoolClient() as client:
            health = await client.health(use_cache=False)
            return HealthResponse(
                healthy=health.healthy,
                total_pods=health.total_pods,
                available_pods=health.available_pods,
                active_sessions=health.active_sessions,
                pool_url=health.pool_url,
            )
    except Exception as e:
        logger.error("Failed to check browser pool health", error=str(e))
        return HealthResponse(healthy=False, pool_url="")


@router.post("/test", response_model=TestResponse)
async def execute_browser_test(body: TestRequest, request: Request):
    """
    Execute a multi-step browser test.

    This endpoint replaces the deprecated Cloudflare Worker /test endpoint.
    It routes tests to the Vultr Browser Pool for execution.

    The response format is compatible with the legacy API for easy migration.
    """
    logger.info(
        "Executing browser test",
        url=body.url,
        step_count=len(body.steps),
        browser=body.browser.value,
        record_video=body.record_video,
    )

    import time
    start_time = time.time()

    try:
        client = await _get_browser_client(request)
        async with client:
            result = await client.test(
                url=body.url,
                steps=body.steps,
                browser=_get_browser_type(body.browser),
                capture_screenshots=body.screenshot,
                record_video=body.record_video,
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Convert to response format
        step_results = [
            TestStepResult(
                step_index=step.step_index,
                instruction=step.instruction,
                success=step.success,
                duration_ms=step.duration_ms,
                screenshot=step.screenshot,
                error=step.error,
                actions=[a.model_dump() if hasattr(a, 'model_dump') else a.__dict__ for a in step.actions] if step.actions else [],
            )
            for step in result.steps
        ]

        logger.info(
            "Browser test completed",
            url=body.url,
            success=result.success,
            steps_passed=sum(1 for s in step_results if s.success),
            steps_total=len(step_results),
            duration_ms=duration_ms,
        )

        return TestResponse(
            success=result.success,
            steps=step_results,
            final_screenshot=result.final_screenshot,
            video_artifact_id=result.video_artifact_id,
            error=result.error,
            duration_ms=duration_ms,
            browser=body.browser.value,
        )

    except BrowserPoolUnavailableError as e:
        logger.error("Browser pool unavailable", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Browser pool is unavailable. No browsers are currently available to run tests."
        )
    except BrowserPoolTimeoutError as e:
        logger.error("Browser test timed out", error=str(e))
        raise HTTPException(
            status_code=504,
            detail="Browser test timed out. The page may be slow or elements not found."
        )
    except BrowserPoolError as e:
        logger.error("Browser pool error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in browser test", error=str(e))
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


@router.post("/observe", response_model=ObserveResponse)
async def observe_page(body: ObserveRequest, request: Request):
    """
    Discover interactive elements on a web page.

    Returns a list of elements that can be interacted with (buttons, inputs, links, etc.)
    """
    logger.info("Observing page", url=body.url, instruction=body.instruction)

    try:
        client = await _get_browser_client(request)
        async with client:
            result = await client.observe(
                url=body.url,
                instruction=body.instruction,
            )

        return ObserveResponse(
            success=result.success,
            url=result.url,
            title=result.title,
            elements=[e.to_dict() if hasattr(e, 'to_dict') else e.__dict__ for e in result.elements] if result.elements else [],
            error=result.error,
        )

    except BrowserPoolError as e:
        logger.error("Observe failed", url=body.url, error=str(e))
        return ObserveResponse(
            success=False,
            url=body.url,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error in observe", error=str(e))
        return ObserveResponse(
            success=False,
            url=body.url,
            error=f"Observation failed: {str(e)}",
        )


@router.post("/act", response_model=ActResponse)
async def execute_action(body: ActRequest, request: Request):
    """
    Execute a single browser action using natural language.

    Example instructions:
    - "Click the login button"
    - "Fill the email field with test@example.com"
    - "Select 'United States' from the country dropdown"
    """
    logger.info("Executing action", url=body.url, instruction=body.instruction)

    try:
        client = await _get_browser_client(request)
        async with client:
            result = await client.act(
                url=body.url,
                instruction=body.instruction,
                screenshot=body.screenshot,
            )

        return ActResponse(
            success=result.success,
            message=result.message,
            url=result.url,
            screenshot=result.screenshot,
            error=result.error,
            actions=[a.model_dump() if hasattr(a, 'model_dump') else a.__dict__ for a in result.actions] if result.actions else [],
        )

    except BrowserPoolError as e:
        logger.error("Action failed", url=body.url, error=str(e))
        return ActResponse(
            success=False,
            url=body.url,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error in action", error=str(e))
        return ActResponse(
            success=False,
            url=body.url,
            error=f"Action failed: {str(e)}",
        )
