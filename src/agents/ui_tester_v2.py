"""
UI Tester Agent V2 - Unified Browser Pool Integration

This is the updated UITesterAgent that uses the unified BrowserPoolClient
instead of the fragmented browser clients (BrowserWorkerClient, PlaywrightTools, etc.)

Key improvements:
- Single browser client (BrowserPoolClient) for all browser automation
- MCP-compatible endpoints (/observe, /act, /test)
- Built-in vision fallback (Claude Computer Use)
- Automatic retries and self-healing
- Simplified API

Migration from ui_tester.py:
- execute_via_worker() → execute_via_pool()
- execute() → execute_via_pool() (unified)
- execute_hybrid() → execute_via_pool() (hybrid is built-in)
"""

import base64
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .base import BaseAgent, AgentResult
from .prompts import get_enhanced_prompt
from .test_planner import TestSpec, TestStep as PlannerTestStep, TestAssertion

if TYPE_CHECKING:
    from src.browser.pool_client import BrowserPoolClient


@dataclass
class StepResult:
    """Result from executing a single test step."""

    step_index: int
    action: str
    success: bool
    duration_ms: int
    error: Optional[str] = None
    screenshot: Optional[bytes] = None
    execution_mode: str = "dom"  # "dom", "vision", "hybrid"
    fallback_triggered: bool = False

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "has_screenshot": self.screenshot is not None,
            "execution_mode": self.execution_mode,
            "fallback_triggered": self.fallback_triggered,
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
    execution_mode: str = "pool"  # "pool", "local", "hybrid"
    fallback_count: int = 0
    fallback_rate: float = 0.0

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
            "execution_mode": self.execution_mode,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_rate,
        }


class UITesterAgentV2(BaseAgent):
    """
    UI Tester Agent using the unified BrowserPoolClient.

    This agent executes UI tests using the Hetzner Browser Pool,
    with automatic vision fallback for reliability.

    Usage:
        agent = UITesterAgentV2()

        result = await agent.execute(
            test_spec=test_spec,
            app_url="https://example.com"
        )
    """

    def __init__(
        self,
        pool_client: Optional["BrowserPoolClient"] = None,
        **kwargs,
    ):
        """
        Initialize the UI Tester Agent.

        Args:
            pool_client: Optional BrowserPoolClient instance.
                        If not provided, a default client will be created.
        """
        super().__init__(**kwargs)
        self._pool_client = pool_client
        self._owns_client = False

    async def _ensure_client(self) -> "BrowserPoolClient":
        """Ensure we have a browser pool client."""
        if self._pool_client is None:
            from src.browser.pool_client import BrowserPoolClient
            self._pool_client = BrowserPoolClient()
            self._owns_client = True
            await self._pool_client._ensure_client()
        return self._pool_client

    async def close(self) -> None:
        """Close the client if we own it."""
        if self._owns_client and self._pool_client:
            await self._pool_client.close()
            self._pool_client = None

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for UI testing."""
        enhanced = get_enhanced_prompt("ui_tester")
        if enhanced:
            return enhanced

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

    async def execute(
        self,
        test_spec: TestSpec | dict,
        app_url: str,
        capture_screenshots: bool = True,
    ) -> AgentResult[UITestResult]:
        """
        Execute a UI test using the Browser Pool.

        This is the PRIMARY entry point for test execution.
        It uses the unified BrowserPoolClient which:
        - Routes to Hetzner Browser Pool
        - Automatically falls back to Claude Computer Use on failure
        - Handles retries and self-healing

        Args:
            test_spec: Test specification to execute
            app_url: Base application URL
            capture_screenshots: Whether to capture screenshots at each step

        Returns:
            AgentResult containing UITestResult
        """
        client = await self._ensure_client()

        # Convert dict to TestSpec if needed
        if isinstance(test_spec, dict):
            steps = [
                PlannerTestStep(**s) if isinstance(s, dict) else s
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
            "Executing UI test via Browser Pool",
            test_id=test_spec.id,
            test_name=test_spec.name,
            steps_count=len(test_spec.steps),
            app_url=app_url,
        )

        start_time = time.time()

        # Check pool health
        health = await client.health()
        if not health.healthy:
            self.log.warning("Browser pool unhealthy, will use vision fallback")

        # Convert steps to natural language instructions
        step_instructions = []
        for step in test_spec.steps:
            instruction = self._step_to_instruction(step, app_url)
            step_instructions.append(instruction)

        try:
            # Execute test via pool
            pool_result = await client.test(
                url=app_url,
                steps=step_instructions,
                capture_screenshots=capture_screenshots,
            )

            total_duration = int((time.time() - start_time) * 1000)

            # Convert pool result to UITestResult
            step_results = []
            fallback_count = 0

            for pool_step in pool_result.steps:
                # Decode screenshot if present
                screenshot_bytes = None
                if pool_step.screenshot:
                    try:
                        screenshot_bytes = base64.b64decode(pool_step.screenshot)
                    except Exception:
                        pass

                fallback_triggered = pool_step.execution_mode.value == "vision"
                if fallback_triggered:
                    fallback_count += 1

                step_results.append(StepResult(
                    step_index=pool_step.step_index,
                    action=pool_step.instruction,
                    success=pool_step.success,
                    duration_ms=pool_step.duration_ms,
                    error=pool_step.error,
                    screenshot=screenshot_bytes,
                    execution_mode=pool_step.execution_mode.value,
                    fallback_triggered=fallback_triggered,
                ))

            # Decode final screenshot
            final_screenshot = None
            if pool_result.final_screenshot:
                try:
                    final_screenshot = base64.b64decode(pool_result.final_screenshot)
                except Exception:
                    pass

            # Calculate fallback rate
            fallback_rate = fallback_count / len(step_results) if step_results else 0.0

            # Determine status
            status = "passed" if pool_result.success else "failed"
            error_message = pool_result.error

            # Get failure screenshot (last failed step)
            failure_screenshot = None
            for step in reversed(step_results):
                if not step.success and step.screenshot:
                    failure_screenshot = step.screenshot
                    break

            result = UITestResult(
                test_id=test_spec.id,
                test_name=test_spec.name,
                status=status,
                step_results=step_results,
                assertion_results=[],  # Assertions are handled separately
                total_duration_ms=total_duration,
                error_message=error_message,
                final_screenshot=final_screenshot,
                failure_screenshot=failure_screenshot,
                execution_mode="pool",
                fallback_count=fallback_count,
                fallback_rate=fallback_rate,
            )

            self.log.info(
                "UI test complete",
                test_id=test_spec.id,
                status=status,
                duration_ms=total_duration,
                fallback_rate=f"{fallback_rate:.1%}",
            )

            return AgentResult(
                success=pool_result.success,
                data=result,
            )

        except Exception as e:
            self.log.exception("UI test execution failed", error=str(e))

            total_duration = int((time.time() - start_time) * 1000)

            result = UITestResult(
                test_id=test_spec.id,
                test_name=test_spec.name,
                status="error",
                total_duration_ms=total_duration,
                error_message=str(e),
                execution_mode="pool",
            )

            return AgentResult(
                success=False,
                data=result,
                error=str(e),
            )

    def _step_to_instruction(self, step: PlannerTestStep, app_url: str) -> str:
        """Convert a test step to a natural language instruction."""
        action = step.action.lower()
        target = step.target or ""
        value = step.value or ""

        if action == "goto":
            url = target if target.startswith("http") else f"{app_url.rstrip('/')}{target}"
            return f"Navigate to {url}"

        elif action == "click":
            return f"Click on {target}"

        elif action == "double_click":
            return f"Double click on {target}"

        elif action == "fill":
            return f"Fill '{value}' into {target}"

        elif action == "type":
            return f"Type '{value}' into {target}"

        elif action == "select":
            return f"Select '{value}' from {target}"

        elif action == "hover":
            return f"Hover over {target}"

        elif action == "wait":
            if target:
                return f"Wait for {target} to be visible"
            else:
                return f"Wait {value}ms"

        elif action == "scroll":
            direction = value or "down"
            return f"Scroll {direction}"

        elif action == "press_key":
            key = value or target
            return f"Press {key} key"

        elif action == "screenshot":
            return "Take a screenshot"

        else:
            return f"{action} {target} {value}".strip()

    async def observe(
        self,
        url: str,
        instruction: Optional[str] = None,
    ) -> AgentResult[dict]:
        """
        Observe/discover interactive elements on a page.

        This is useful for:
        - Discovering available actions on a page
        - Building test steps dynamically
        - Debugging failed tests

        Args:
            url: URL of the page to observe
            instruction: Optional instruction for what to look for

        Returns:
            AgentResult with discovered elements
        """
        client = await self._ensure_client()

        try:
            result = await client.observe(url, instruction)

            return AgentResult(
                success=result.success,
                data={
                    "url": result.url,
                    "title": result.title,
                    "elements": [e.to_dict() for e in result.elements],
                    "error": result.error,
                },
            )

        except Exception as e:
            self.log.error("Observe failed", url=url, error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )

    async def act(
        self,
        url: str,
        instruction: str,
        capture_screenshot: bool = True,
    ) -> AgentResult[dict]:
        """
        Execute a single browser action.

        This is useful for:
        - One-off actions during debugging
        - Building tests interactively
        - Self-healing verification

        Args:
            url: URL of the page
            instruction: Natural language instruction
            capture_screenshot: Whether to capture screenshot after action

        Returns:
            AgentResult with action outcome
        """
        client = await self._ensure_client()

        try:
            result = await client.act(url, instruction, screenshot=capture_screenshot)

            return AgentResult(
                success=result.success,
                data=result.to_dict(),
            )

        except Exception as e:
            self.log.error("Act failed", url=url, instruction=instruction, error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )

    async def verify_with_vision(
        self,
        screenshot: bytes,
        verification_prompt: str,
    ) -> AgentResult[dict]:
        """
        Use Claude vision to verify a screenshot.

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


# Alias for backward compatibility
UITesterAgent = UITesterAgentV2
