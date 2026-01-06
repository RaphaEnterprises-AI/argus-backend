"""UI Tester Agent - Executes browser-based tests using Playwright or Browser Worker.

This agent:
- Executes UI test specifications step by step
- Takes screenshots for verification
- Uses Claude for visual verification when needed
- Reports detailed results with evidence
- Supports both local Playwright and remote Cloudflare Worker
"""

import base64
import time
from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING

from .base import BaseAgent, AgentResult
from .test_planner import TestSpec, TestStep

if TYPE_CHECKING:
    from src.tools.browser_worker_client import BrowserWorkerClient


@dataclass
class StepResult:
    """Result from executing a single test step."""

    step_index: int
    action: str
    success: bool
    duration_ms: int
    error: Optional[str] = None
    screenshot: Optional[bytes] = None

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "has_screenshot": self.screenshot is not None,
        }


@dataclass
class AssertionResult:
    """Result from checking an assertion."""

    type: str
    target: Optional[str]
    expected: Optional[str]
    actual: Optional[str]
    passed: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target": self.target,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "error": self.error,
        }


@dataclass
class UITestResult:
    """Complete result from a UI test execution."""

    test_id: str
    test_name: str
    status: str  # passed, failed, error
    step_results: list[StepResult] = field(default_factory=list)
    assertion_results: list[AssertionResult] = field(default_factory=list)
    total_duration_ms: int = 0
    error_message: Optional[str] = None
    final_screenshot: Optional[bytes] = None
    failure_screenshot: Optional[bytes] = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status,
            "step_results": [s.to_dict() for s in self.step_results],
            "assertion_results": [a.to_dict() for a in self.assertion_results],
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message,
            "has_final_screenshot": self.final_screenshot is not None,
            "has_failure_screenshot": self.failure_screenshot is not None,
        }


class UITesterAgent(BaseAgent):
    """Agent that executes UI tests using Playwright or Browser Worker.

    Integrates with:
    - PlaywrightTools for local browser automation
    - BrowserWorkerClient for remote Cloudflare Worker browser automation
    - Claude for visual verification
    - Screenshot capture for evidence
    """

    def __init__(
        self,
        playwright_tools=None,
        browser_worker: Optional["BrowserWorkerClient"] = None,
        use_worker: bool = True,
        **kwargs,
    ):
        """Initialize with optional Playwright tools or Browser Worker.

        Args:
            playwright_tools: Instance of PlaywrightTools for local browser automation
            browser_worker: Instance of BrowserWorkerClient for remote automation
            use_worker: If True, prefer using Worker for browser operations
        """
        super().__init__(**kwargs)
        self._playwright = playwright_tools
        self._browser_worker = browser_worker
        self._use_worker = use_worker

    def _get_system_prompt(self) -> str:
        return """You are an expert UI test executor. Your task is to verify visual elements and states in screenshots.

When asked to verify a screenshot:
1. Look for the specific elements mentioned
2. Check if text, buttons, forms are visible and correct
3. Identify any error messages or unexpected states
4. Report exactly what you observe

Respond with JSON containing:
- passed: boolean indicating if verification succeeded
- observations: list of what you see
- issues: list of any problems found
- confidence: 0.0-1.0 confidence in your assessment"""

    async def execute_via_worker(
        self,
        test_spec: TestSpec | dict,
        app_url: str,
    ) -> AgentResult[UITestResult]:
        """Execute a UI test using the remote Browser Worker.

        This method sends the test to the Cloudflare Browser Worker for execution,
        which provides better scalability and browser compatibility.

        Args:
            test_spec: Test specification to execute
            app_url: Base application URL

        Returns:
            AgentResult containing UITestResult
        """
        from src.tools.browser_worker_client import get_browser_client

        worker = self._browser_worker or get_browser_client()

        # Check Worker health
        is_healthy = await worker.health_check()
        if not is_healthy:
            return AgentResult(
                success=False,
                error="Browser Worker is not available",
            )

        # Convert test spec to dict if needed
        if isinstance(test_spec, TestSpec):
            test_id = test_spec.id
            test_name = test_spec.name
            steps = [step.to_natural_language() if hasattr(step, 'to_natural_language')
                     else f"{step.action} {step.target or ''} {step.value or ''}".strip()
                     for step in test_spec.steps]
        else:
            test_id = test_spec.get("id", "unknown")
            test_name = test_spec.get("name", "Unknown Test")
            steps = []
            for step in test_spec.get("steps", []):
                if isinstance(step, str):
                    steps.append(step)
                else:
                    action = step.get("action", "")
                    target = step.get("target", "")
                    value = step.get("value", "")
                    steps.append(f"{action} {target} {value}".strip())

        self.log.info(
            "Executing UI test via Worker",
            test_id=test_id,
            test_name=test_name,
            steps_count=len(steps),
        )

        start_time = time.time()

        try:
            # Build full URL for first step if needed
            full_url = app_url

            # Execute test via Worker
            result = await worker.run_test(
                url=full_url,
                steps=steps,
                browser="chrome",
                capture_screenshots=True,
            )

            total_duration = int((time.time() - start_time) * 1000)

            # Convert Worker result to UITestResult
            step_results = []
            if result.steps:
                for idx, step_data in enumerate(result.steps):
                    step_results.append(
                        StepResult(
                            step_index=idx,
                            action=steps[idx] if idx < len(steps) else "unknown",
                            success=step_data.get("success", False),
                            duration_ms=step_data.get("duration_ms", 0),
                            error=step_data.get("error"),
                        )
                    )

            # Decode final screenshot if present
            final_screenshot = None
            if result.final_screenshot:
                try:
                    final_screenshot = base64.b64decode(result.final_screenshot)
                except Exception:
                    pass

            ui_result = UITestResult(
                test_id=test_id,
                test_name=test_name,
                status="passed" if result.success else "failed",
                step_results=step_results,
                assertion_results=[],  # Worker doesn't return assertion details
                total_duration_ms=total_duration,
                error_message=result.error,
                final_screenshot=final_screenshot,
            )

            self.log.info(
                "Worker test complete",
                test_id=test_id,
                status="passed" if result.success else "failed",
                duration_ms=total_duration,
            )

            return AgentResult(
                success=result.success,
                data=ui_result,
            )

        except Exception as e:
            self.log.exception("Worker test execution failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )

    async def execute(
        self,
        test_spec: TestSpec | dict,
        app_url: str,
        playwright_tools=None,
        use_worker: Optional[bool] = None,
    ) -> AgentResult[UITestResult]:
        """Execute a UI test specification.

        Args:
            test_spec: Test specification to execute
            app_url: Base application URL
            playwright_tools: Optional PlaywrightTools instance
            use_worker: Override to force Worker or local Playwright

        Returns:
            AgentResult containing UITestResult
        """
        # Determine if we should use Worker
        should_use_worker = use_worker if use_worker is not None else self._use_worker

        if should_use_worker:
            # Try Worker first, fall back to Playwright if Worker unavailable
            result = await self.execute_via_worker(test_spec, app_url)
            if result.success or result.error != "Browser Worker is not available":
                return result
            self.log.info("Worker unavailable, falling back to local Playwright")
        # Convert dict to TestSpec if needed
        if isinstance(test_spec, dict):
            from .test_planner import TestStep, TestAssertion

            steps = [
                TestStep(**s) if isinstance(s, dict) else s
                for s in test_spec.get("steps", [])
            ]
            assertions = [
                TestAssertion(**a) if isinstance(a, dict) else a
                for a in test_spec.get("assertions", [])
            ]
            test_spec = TestSpec(
                id=test_spec.get("id", "unknown"),
                name=test_spec.get("name", "Unknown Test"),
                type=test_spec.get("type", "ui"),
                priority=test_spec.get("priority", "medium"),
                description=test_spec.get("description", ""),
                steps=steps,
                assertions=assertions,
            )

        self.log.info(
            "Executing UI test",
            test_id=test_spec.id,
            test_name=test_spec.name,
            steps_count=len(test_spec.steps),
        )

        pw = playwright_tools or self._playwright
        if pw is None:
            return AgentResult(
                success=False,
                error="PlaywrightTools not provided - cannot execute UI test",
            )

        start_time = time.time()
        step_results = []
        assertion_results = []
        failure_screenshot = None
        final_screenshot = None
        error_message = None
        status = "passed"

        try:
            # Execute each step
            for idx, step in enumerate(test_spec.steps):
                step_start = time.time()

                try:
                    await self._execute_step(pw, step, app_url)
                    step_results.append(
                        StepResult(
                            step_index=idx,
                            action=step.action,
                            success=True,
                            duration_ms=int((time.time() - step_start) * 1000),
                        )
                    )
                except Exception as e:
                    # Capture failure screenshot
                    try:
                        failure_screenshot = await pw.screenshot()
                    except Exception as screenshot_err:
                        self.log.warning(
                            "Failed to capture failure screenshot",
                            step=idx,
                            error=str(screenshot_err)
                        )

                    step_results.append(
                        StepResult(
                            step_index=idx,
                            action=step.action,
                            success=False,
                            duration_ms=int((time.time() - step_start) * 1000),
                            error=str(e),
                            screenshot=failure_screenshot,
                        )
                    )
                    error_message = f"Step {idx} ({step.action}) failed: {str(e)}"
                    status = "failed"
                    break

            # Check assertions if all steps passed
            if status == "passed":
                for assertion in test_spec.assertions:
                    result = await self._check_assertion(pw, assertion)
                    assertion_results.append(result)
                    if not result.passed:
                        status = "failed"
                        error_message = f"Assertion failed: {assertion.type}"
                        try:
                            failure_screenshot = await pw.screenshot()
                        except Exception as screenshot_err:
                            self.log.warning(
                                "Failed to capture assertion failure screenshot",
                                assertion_type=assertion.type,
                                error=str(screenshot_err)
                            )

            # Take final screenshot
            try:
                final_screenshot = await pw.screenshot()
            except Exception as screenshot_err:
                self.log.warning("Failed to capture final screenshot", error=str(screenshot_err))

        except Exception as e:
            status = "error"
            error_message = str(e)
            self.log.error("Test execution error", error=str(e))

        total_duration = int((time.time() - start_time) * 1000)

        result = UITestResult(
            test_id=test_spec.id,
            test_name=test_spec.name,
            status=status,
            step_results=step_results,
            assertion_results=assertion_results,
            total_duration_ms=total_duration,
            error_message=error_message,
            final_screenshot=final_screenshot,
            failure_screenshot=failure_screenshot,
        )

        self.log.info(
            "UI test complete",
            test_id=test_spec.id,
            status=status,
            duration_ms=total_duration,
        )

        return AgentResult(
            success=status == "passed",
            data=result,
        )

    async def _execute_step(self, pw, step: TestStep, app_url: str) -> None:
        """Execute a single test step."""
        action = step.action.lower()
        target = step.target
        value = step.value
        timeout = step.timeout

        if action == "goto":
            url = target if target.startswith("http") else f"{app_url.rstrip('/')}{target}"
            await pw.goto(url)

        elif action == "click":
            await pw.click(target, timeout_ms=timeout)

        elif action == "double_click":
            await pw.double_click(target, timeout_ms=timeout)

        elif action == "fill":
            await pw.fill(target, value or "")

        elif action == "type":
            await pw.type_text(target, value or "")

        elif action == "select":
            await pw.select_option(target, value)

        elif action == "hover":
            await pw.hover(target)

        elif action == "wait":
            if target:
                await pw.wait_for_selector(target, timeout_ms=timeout)
            else:
                await pw.wait(int(value) if value else 1000)

        elif action == "scroll":
            direction = value or "down"
            delta = 300 if direction == "down" else -300
            # Use JavaScript scroll
            await pw._page.evaluate(f"window.scrollBy(0, {delta})")

        elif action == "press_key":
            await pw.press_key(value or target)

        elif action == "screenshot":
            await pw.screenshot()

        else:
            raise ValueError(f"Unknown action: {action}")

    async def _check_assertion(self, pw, assertion) -> AssertionResult:
        """Check a single assertion."""
        assertion_type = assertion.type.lower()
        target = assertion.target
        expected = assertion.expected

        try:
            if assertion_type == "element_visible":
                visible = await pw.is_visible(target)
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected="visible",
                    actual="visible" if visible else "not visible",
                    passed=visible,
                )

            elif assertion_type == "element_hidden":
                visible = await pw.is_visible(target)
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected="hidden",
                    actual="hidden" if not visible else "visible",
                    passed=not visible,
                )

            elif assertion_type == "text_contains":
                text = await pw.get_text(target)
                passed = expected in (text or "")
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected=expected,
                    actual=text,
                    passed=passed,
                )

            elif assertion_type == "text_equals":
                text = await pw.get_text(target)
                passed = text == expected
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected=expected,
                    actual=text,
                    passed=passed,
                )

            elif assertion_type == "url_matches" or assertion_type == "url_contains":
                current_url = await pw.get_current_url()
                passed = expected in current_url
                return AssertionResult(
                    type=assertion_type,
                    target=None,
                    expected=expected,
                    actual=current_url,
                    passed=passed,
                )

            elif assertion_type == "value_equals":
                value = await pw.get_input_value(target)
                passed = value == expected
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected=expected,
                    actual=value,
                    passed=passed,
                )

            else:
                return AssertionResult(
                    type=assertion_type,
                    target=target,
                    expected=expected,
                    actual=None,
                    passed=False,
                    error=f"Unknown assertion type: {assertion_type}",
                )

        except Exception as e:
            return AssertionResult(
                type=assertion_type,
                target=target,
                expected=expected,
                actual=None,
                passed=False,
                error=str(e),
            )

    async def verify_with_vision(
        self,
        screenshot: bytes,
        verification_prompt: str,
    ) -> AgentResult[dict]:
        """Use Claude vision to verify a screenshot.

        Args:
            screenshot: Screenshot bytes (PNG)
            verification_prompt: What to verify in the screenshot

        Returns:
            AgentResult with verification data
        """
        self.log.info("Running visual verification")

        if not self._check_cost_limit():
            return AgentResult(
                success=False,
                error="Cost limit exceeded",
            )

        try:
            response = self._call_claude(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(screenshot).decode(),
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Verify: {verification_prompt}\n\nRespond with JSON: {{\"passed\": boolean, \"observations\": [...], \"issues\": [...], \"confidence\": 0.0-1.0}}",
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )

            content = self._extract_text_response(response)
            result = self._parse_json_response(content)

            return AgentResult(
                success=True,
                data=result or {"passed": False, "error": "Failed to parse response"},
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except Exception as e:
            self.log.error("Visual verification failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )
