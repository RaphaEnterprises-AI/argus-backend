"""Browser Pool API endpoints.

Provides REST endpoints for browser automation via the Vultr Browser Pool.
This replaces the deprecated Cloudflare Worker endpoints.

Supports two backends:
- BrowserPoolClient (Cloudflare Worker): Fast, no video recording
- SeleniumGridClient (Vultr K8s): Video recording via ffmpeg sidecars
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
from src.browser.selenium_grid_client import SeleniumGridClient

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
    recording_url: str | None = None  # Signed URL for video playback
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


def _convert_screenshot(screenshot: str | dict | None) -> str | None:
    """Convert screenshot from various formats to base64 string.

    The browser pool may return screenshots as:
    - A base64 string (pass through)
    - A JavaScript Buffer object: {'type': 'Buffer', 'data': [bytes...]}
    - None (pass through)
    """
    import base64

    if screenshot is None:
        return None

    if isinstance(screenshot, str):
        return screenshot

    if isinstance(screenshot, dict) and screenshot.get('type') == 'Buffer':
        # Convert Buffer data array to base64
        data = screenshot.get('data', [])
        if data:
            return base64.b64encode(bytes(data)).decode('utf-8')
        return None

    # Unknown format, try to convert to string
    return str(screenshot) if screenshot else None


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

    Backends:
    - record_video=False: Uses BrowserPoolClient (Cloudflare Worker) - fast
    - record_video=True: Uses SeleniumGridClient (Vultr K8s) - video recording

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

    # Use Selenium Grid for video recording, otherwise use browser pool
    if body.record_video:
        return await _execute_test_with_selenium_grid(body, request, start_time)
    else:
        return await _execute_test_with_browser_pool(body, request, start_time)


async def _execute_test_with_browser_pool(body: TestRequest, request: Request, start_time: float) -> TestResponse:
    """Execute test using BrowserPoolClient (fast, no video)."""
    import time

    try:
        client = await _get_browser_client(request)
        async with client:
            result = await client.test(
                url=body.url,
                steps=body.steps,
                browser=_get_browser_type(body.browser),
                capture_screenshots=body.screenshot,
                record_video=False,
            )

        duration_ms = int((time.time() - start_time) * 1000)

        # Convert to response format with screenshot conversion
        step_results = [
            TestStepResult(
                step_index=step.step_index,
                instruction=step.instruction,
                success=step.success,
                duration_ms=step.duration_ms,
                screenshot=_convert_screenshot(step.screenshot),
                error=step.error,
                actions=[a.model_dump() if hasattr(a, 'model_dump') else a.__dict__ for a in step.actions] if step.actions else [],
            )
            for step in result.steps
        ]

        logger.info(
            "Browser test completed (pool)",
            url=body.url,
            success=result.success,
            steps_passed=sum(1 for s in step_results if s.success),
            steps_total=len(step_results),
            duration_ms=duration_ms,
        )

        return TestResponse(
            success=result.success,
            steps=step_results,
            final_screenshot=_convert_screenshot(result.final_screenshot),
            video_artifact_id=None,
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


async def _execute_test_with_selenium_grid(body: TestRequest, request: Request, start_time: float) -> TestResponse:
    """Execute test using SeleniumGridClient (video recording enabled)."""
    import asyncio
    import re
    import time

    logger.info(
        "Using Selenium Grid for video recording",
        url=body.url,
        step_count=len(body.steps),
    )

    try:
        # Get user context for audit
        try:
            user = await get_current_user(request)
            user_id = user["user_id"]
        except Exception:
            user_id = "anonymous"

        # Create Selenium Grid client
        grid_client = SeleniumGridClient()

        # Start session (video recording begins automatically)
        session_id = await grid_client.start_session()
        logger.info("Selenium session started", session_id=session_id)

        step_results: list[TestStepResult] = []
        all_success = True

        try:
            # Navigate to starting URL
            await grid_client.navigate(body.url)
            await asyncio.sleep(2)  # Wait for page load

            # Execute each step
            for idx, step_instruction in enumerate(body.steps):
                step_start = time.time()
                step_success = False
                step_error = None
                step_screenshot = None
                actions_taken = []

                try:
                    # Execute the step instruction using Selenium
                    instruction_lower = step_instruction.lower()
                    logger.info(f"Executing step {idx}: {step_instruction}")

                    if instruction_lower.startswith("goto ") or instruction_lower.startswith("navigate to "):
                        # Extract URL from instruction
                        url = step_instruction.split(" ", 2)[-1].strip()
                        success = await grid_client.navigate(url)
                        step_success = success
                        actions_taken.append({"action": "navigate", "url": url, "success": success})
                        await asyncio.sleep(1)

                    elif "wait" in instruction_lower:
                        # Extract wait duration if specified
                        match = re.search(r'(\d+)\s*(second|sec|s|ms|millisecond)', instruction_lower)
                        if match:
                            duration = int(match.group(1))
                            if 'ms' in match.group(2) or 'millisecond' in match.group(2):
                                duration = duration / 1000
                            await asyncio.sleep(duration)
                        else:
                            await asyncio.sleep(2)
                        step_success = True
                        actions_taken.append({"action": "wait", "success": True})

                    elif "screenshot" in instruction_lower:
                        step_success = True
                        actions_taken.append({"action": "screenshot", "success": True})

                    elif "press enter" in instruction_lower or "hit enter" in instruction_lower or "press return" in instruction_lower:
                        # Press Enter key
                        success = await grid_client.press_key("enter")
                        step_success = success
                        actions_taken.append({"action": "press_key", "key": "enter", "success": success})
                        await asyncio.sleep(0.5)

                    elif "press tab" in instruction_lower:
                        success = await grid_client.press_key("tab")
                        step_success = success
                        actions_taken.append({"action": "press_key", "key": "tab", "success": success})
                        await asyncio.sleep(0.3)

                    elif "press escape" in instruction_lower or "press esc" in instruction_lower:
                        success = await grid_client.press_key("escape")
                        step_success = success
                        actions_taken.append({"action": "press_key", "key": "escape", "success": success})
                        await asyncio.sleep(0.3)

                    elif ("click" in instruction_lower or "tap" in instruction_lower or
                          "select" in instruction_lower or "choose" in instruction_lower):
                        # Try to find and click element
                        element_id = await _find_element_for_instruction(grid_client, step_instruction)
                        if element_id:
                            success = await grid_client.click_element(element_id)
                            step_success = success
                            actions_taken.append({"action": "click", "element_id": element_id, "success": success})
                            await asyncio.sleep(0.5)
                        else:
                            step_error = f"Could not find element to click for: {step_instruction}"
                            step_success = False
                            actions_taken.append({"action": "click", "error": "element not found"})

                    elif ("type" in instruction_lower or "enter" in instruction_lower or
                          "fill" in instruction_lower or "input" in instruction_lower or
                          "write" in instruction_lower):
                        # Type text into an element
                        element_id, text_to_type = await _find_element_and_text_for_instruction(grid_client, step_instruction)
                        if element_id and text_to_type:
                            # Clear first, then type
                            await grid_client.clear_element(element_id)
                            success = await grid_client.send_keys(element_id, text_to_type)
                            step_success = success
                            actions_taken.append({"action": "type", "element_id": element_id, "text": text_to_type, "success": success})
                            await asyncio.sleep(0.3)
                        elif text_to_type:
                            # No specific element found, try typing into active element
                            success = await grid_client.send_keys_to_active_element(text_to_type)
                            step_success = success
                            actions_taken.append({"action": "type_active", "text": text_to_type, "success": success})
                            await asyncio.sleep(0.3)
                        else:
                            step_error = f"Could not determine what to type for: {step_instruction}"
                            step_success = False
                            actions_taken.append({"action": "type", "error": "could not parse instruction"})

                    elif "verify" in instruction_lower or "check" in instruction_lower or "assert" in instruction_lower or "should" in instruction_lower:
                        # Verification step - try to find the element/text mentioned
                        verification_result = await _verify_instruction(grid_client, step_instruction)
                        step_success = verification_result["success"]
                        if not step_success:
                            step_error = verification_result.get("error", "Verification failed")
                        actions_taken.append({"action": "verify", "result": verification_result})

                    elif "scroll" in instruction_lower:
                        # Scroll the page
                        if "down" in instruction_lower:
                            await grid_client.execute_script("window.scrollBy(0, 300)")
                        elif "up" in instruction_lower:
                            await grid_client.execute_script("window.scrollBy(0, -300)")
                        elif "top" in instruction_lower:
                            await grid_client.execute_script("window.scrollTo(0, 0)")
                        elif "bottom" in instruction_lower:
                            await grid_client.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                        else:
                            await grid_client.execute_script("window.scrollBy(0, 300)")
                        step_success = True
                        actions_taken.append({"action": "scroll", "success": True})
                        await asyncio.sleep(0.5)

                    else:
                        # Unknown instruction - try to interpret as click or type
                        # First, try to find an input field
                        element_id = await grid_client.find_element("css selector", "input:focus, textarea:focus")
                        if element_id:
                            # There's a focused input, assume we should type
                            success = await grid_client.send_keys(element_id, step_instruction)
                            step_success = success
                            actions_taken.append({"action": "type_fallback", "text": step_instruction, "success": success})
                        else:
                            # Try to find a clickable element with matching text
                            element_id = await _find_element_for_instruction(grid_client, step_instruction)
                            if element_id:
                                success = await grid_client.click_element(element_id)
                                step_success = success
                                actions_taken.append({"action": "click_fallback", "element_id": element_id, "success": success})
                            else:
                                step_error = f"Could not interpret instruction: {step_instruction}"
                                step_success = False
                                actions_taken.append({"action": "unknown", "error": "could not interpret"})

                    # Capture screenshot after step
                    if body.screenshot:
                        step_screenshot = await grid_client.screenshot()

                except Exception as e:
                    step_error = str(e)
                    step_success = False
                    logger.warning("Step failed", step=idx, error=step_error)

                step_duration = int((time.time() - step_start) * 1000)

                step_results.append(TestStepResult(
                    step_index=idx,
                    instruction=step_instruction,
                    success=step_success,
                    duration_ms=step_duration,
                    screenshot=step_screenshot,
                    error=step_error,
                    actions=actions_taken,
                ))

                if not step_success:
                    all_success = False
                    # Don't stop on failure - continue to get full results
                    # but log it clearly
                    logger.warning(f"Step {idx} failed: {step_error}")

        finally:
            # End session (video stops, uploads to R2)
            end_result = await grid_client.end_session()
            logger.info("Selenium session ended", session_id=session_id, result=end_result)

        duration_ms = int((time.time() - start_time) * 1000)

        # Get video artifact info
        video_artifact_id = end_result.get("video_artifact_id")
        final_screenshot = step_results[-1].screenshot if step_results else None

        # Generate signed URL for video playback
        recording_url = None
        if video_artifact_id:
            try:
                from src.services.cloudflare_storage import get_cloudflare_client, is_cloudflare_configured
                if is_cloudflare_configured():
                    cf_client = get_cloudflare_client()
                    recording_url = cf_client.r2.generate_signed_url(
                        video_artifact_id,
                        artifact_type="video",
                        expiry_seconds=3600  # 1 hour expiry
                    )
                    logger.info("Generated signed video URL", video_artifact_id=video_artifact_id)
            except Exception as e:
                logger.warning("Failed to generate signed video URL", error=str(e))

        logger.info(
            "Browser test completed (selenium grid)",
            url=body.url,
            success=all_success,
            steps_passed=sum(1 for s in step_results if s.success),
            steps_total=len(step_results),
            duration_ms=duration_ms,
            video_artifact_id=video_artifact_id,
            recording_url=recording_url,
        )

        return TestResponse(
            success=all_success,
            steps=step_results,
            final_screenshot=final_screenshot,
            video_artifact_id=video_artifact_id,
            recording_url=recording_url,
            error=None if all_success else "One or more steps failed",
            duration_ms=duration_ms,
            browser=body.browser.value,
        )

    except Exception as e:
        logger.exception("Selenium Grid test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


async def _find_element_for_instruction(grid_client: SeleniumGridClient, instruction: str) -> str | None:
    """Find an element based on natural language instruction."""
    import re

    instruction_lower = instruction.lower()

    # Common patterns to extract element identifiers
    # Pattern: "click the X button", "click on X", "tap X"
    patterns = [
        r'(?:click|tap|select|choose|press)\s+(?:the\s+)?(?:on\s+)?["\']?([^"\']+)["\']?\s*(?:button|link|checkbox|radio|option)?',
        r'(?:button|link)\s+["\']?([^"\']+)["\']?',
        r'["\']([^"\']+)["\']',  # Quoted text
    ]

    target_text = None
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target_text = match.group(1).strip()
            break

    if not target_text:
        # Use the whole instruction as fallback
        target_text = instruction_lower.replace("click", "").replace("tap", "").replace("the", "").replace("on", "").strip()

    # Try various selectors to find the element
    selectors_to_try = [
        # By text content (buttons, links)
        f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        # By aria-label
        f"//*[@aria-label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        # By placeholder
        f"//input[@placeholder[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        # By id or name containing the text
        f"//*[@id[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        f"//*[@name[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
    ]

    for selector in selectors_to_try:
        element_id = await grid_client.find_element("xpath", selector)
        if element_id:
            logger.debug(f"Found element for '{target_text}' with selector: {selector}")
            return element_id

    # Also try CSS selectors for common elements
    css_selectors = [
        f"button",  # Any button
        f"input[type='submit']",
        f"a",  # Any link
        f"[role='button']",
    ]

    for css in css_selectors:
        elements = await grid_client.find_elements(css)
        for el in elements:
            element_id = el.get("element_id")
            if element_id:
                # Check if element text matches
                text = el.get("text", "").lower()
                if target_text in text:
                    logger.debug(f"Found element by CSS and text match: {css}")
                    return element_id

    logger.warning(f"Could not find element for instruction: {instruction}")
    return None


async def _find_element_and_text_for_instruction(grid_client: SeleniumGridClient, instruction: str) -> tuple[str | None, str | None]:
    """Find an element and extract text to type from instruction."""
    import re

    instruction_lower = instruction.lower()

    # Extract the text to type
    # Patterns: "type 'hello'", "enter 'text' in field", "fill field with 'value'"
    text_patterns = [
        r'["\']([^"\']+)["\']',  # Quoted text
        r'(?:type|enter|fill|input|write)\s+(.+?)(?:\s+in|\s+into|\s+to|\s+on|$)',
    ]

    text_to_type = None
    for pattern in text_patterns:
        match = re.search(pattern, instruction)
        if match:
            text_to_type = match.group(1).strip()
            break

    if not text_to_type:
        return None, None

    # Extract the target field
    field_patterns = [
        r'(?:in|into|to|on)\s+(?:the\s+)?["\']?([^"\']+)["\']?\s*(?:field|input|box|area)?',
        r'(?:field|input|box|area)\s+["\']?([^"\']+)["\']?',
    ]

    target_field = None
    for pattern in field_patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target_field = match.group(1).strip()
            break

    # Find the input element
    element_id = None
    if target_field:
        selectors = [
            f"//input[@placeholder[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_field}')]]",
            f"//input[@name[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_field}')]]",
            f"//input[@id[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_field}')]]",
            f"//textarea[@placeholder[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_field}')]]",
            f"//label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_field}')]/following::input[1]",
        ]
        for selector in selectors:
            element_id = await grid_client.find_element("xpath", selector)
            if element_id:
                break

    if not element_id:
        # Try to find any visible input field
        element_id = await grid_client.find_element("css selector", "input:not([type='hidden']):not([type='submit']):not([type='button'])")

    if not element_id:
        element_id = await grid_client.find_element("css selector", "textarea")

    return element_id, text_to_type


async def _verify_instruction(grid_client: SeleniumGridClient, instruction: str) -> dict:
    """Verify a condition described in natural language."""
    import re

    instruction_lower = instruction.lower()

    # Extract what to verify
    # Patterns: "verify X appears", "check X exists", "should see X"
    patterns = [
        r'(?:verify|check|assert|should\s+see|should\s+contain|should\s+have|should\s+show)\s+(?:that\s+)?["\']?(.+?)["\']?(?:\s+appears|\s+exists|\s+is\s+visible|\s+is\s+shown|\s+in\s+list)?$',
        r'["\']([^"\']+)["\']',  # Quoted text
    ]

    target_text = None
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target_text = match.group(1).strip()
            break

    if not target_text:
        target_text = instruction_lower.replace("verify", "").replace("check", "").replace("that", "").strip()

    if not target_text:
        return {"success": False, "error": "Could not determine what to verify"}

    # Try to find element with the text
    selectors = [
        f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
    ]

    for selector in selectors:
        element_id = await grid_client.find_element("xpath", selector)
        if element_id:
            return {"success": True, "found": True, "element_id": element_id, "target": target_text}

    # Also check page source for the text
    page_source = await grid_client.execute_script("return document.body.innerText")
    if page_source and target_text in page_source.lower():
        return {"success": True, "found": True, "in_page_text": True, "target": target_text}

    return {"success": False, "error": f"Could not find '{target_text}' on page"}


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
            screenshot=_convert_screenshot(result.screenshot),
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
