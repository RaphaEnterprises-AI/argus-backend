"""
Real Browser Pool Integration Tests.

Tests actual browser pool operations:
- Real page navigation
- Element observation
- Action execution (click, fill, etc.)
- Screenshot capture
- Self-healing selectors
- Multi-step test execution

Requires:
    BROWSER_POOL_URL=<your-pool-url>
    BROWSER_POOL_JWT_SECRET=<your-secret>

Run with:
    pytest tests/integration/test_browser_pool_real.py -v -s
"""

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import structlog

logger = structlog.get_logger()


@dataclass
class BrowserTestResult:
    """Result from a browser pool test."""
    test_name: str
    action_type: str  # observe, act, test, extract
    passed: bool
    latency_ms: float
    execution_mode: str | None = None  # dom, vision, hybrid, cached
    error: str | None = None
    details: dict = field(default_factory=dict)
    screenshot: bytes | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "action_type": self.action_type,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "execution_mode": self.execution_mode,
            "error": self.error,
            "details": self.details,
            "has_screenshot": self.screenshot is not None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BrowserTestReport:
    """Aggregated browser test report."""
    tests: list[BrowserTestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def add(self, result: BrowserTestResult):
        self.tests.append(result)

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.tests)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(len(self.tests), 1)

    def execution_mode_stats(self) -> dict[str, int]:
        stats = {}
        for t in self.tests:
            mode = t.execution_mode or "unknown"
            stats[mode] = stats.get(mode, 0) + 1
        return stats

    def to_dict(self) -> dict:
        self.completed_at = datetime.utcnow()
        return {
            "summary": {
                "total_tests": len(self.tests),
                "passed": self.passed_count,
                "failed": len(self.tests) - self.passed_count,
                "total_latency_ms": round(self.total_latency_ms, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
            },
            "execution_modes": self.execution_mode_stats(),
            "tests": [t.to_dict() for t in self.tests],
            "duration_seconds": (self.completed_at - self.started_at).total_seconds(),
        }


class BrowserPoolTester:
    """Tests real browser pool operations."""

    def __init__(self):
        self.report = BrowserTestReport()
        self.log = logger.bind(component="browser_pool_tester")
        self.pool_url = os.environ.get("BROWSER_POOL_URL")
        self.jwt_secret = os.environ.get("BROWSER_POOL_JWT_SECRET")

    def _check_config(self) -> bool:
        """Check if browser pool is configured."""
        if not self.pool_url:
            self.log.warning("BROWSER_POOL_URL not configured")
            return False
        if not self.jwt_secret:
            self.log.warning("BROWSER_POOL_JWT_SECRET not configured")
            return False
        return True

    async def _get_client(self):
        """Get a configured browser pool client."""
        from src.browser.pool_client import BrowserPoolClient, UserContext

        user_context = UserContext(
            user_id="integration-test",
            org_id="test-org",
            email="test@example.com",
        )

        return BrowserPoolClient(
            pool_url=self.pool_url,
            jwt_secret=self.jwt_secret,
            user_context=user_context,
        )

    async def test_observe_simple_page(self) -> BrowserTestResult:
        """Test observing a simple webpage."""
        self.log.info("Testing observe on example.com")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                result = await client.observe(url="https://example.com")
                latency = (time.perf_counter() - start) * 1000

                test_result = BrowserTestResult(
                    test_name="observe_simple_page",
                    action_type="observe",
                    passed=result.success,
                    latency_ms=latency,
                    details={
                        "url": result.url,
                        "title": result.title,
                        "elements_found": len(result.elements) if result.elements else 0,
                        "element_types": list(set(e.type for e in (result.elements or []))),
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="observe_simple_page",
                action_type="observe",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_observe_complex_page(self) -> BrowserTestResult:
        """Test observing a more complex webpage."""
        self.log.info("Testing observe on github.com")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                result = await client.observe(url="https://github.com")
                latency = (time.perf_counter() - start) * 1000

                test_result = BrowserTestResult(
                    test_name="observe_complex_page",
                    action_type="observe",
                    passed=result.success and len(result.elements or []) > 5,
                    latency_ms=latency,
                    details={
                        "url": result.url,
                        "title": result.title,
                        "elements_found": len(result.elements) if result.elements else 0,
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="observe_complex_page",
                action_type="observe",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_act_screenshot(self) -> BrowserTestResult:
        """Test taking a screenshot via act."""
        self.log.info("Testing screenshot capture")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                result = await client.act(
                    url="https://example.com",
                    instruction="Take a screenshot of the page",
                    screenshot=True,
                )
                latency = (time.perf_counter() - start) * 1000

                has_screenshot = result.screenshot is not None

                test_result = BrowserTestResult(
                    test_name="act_screenshot",
                    action_type="act",
                    passed=result.success and has_screenshot,
                    latency_ms=latency,
                    execution_mode=result.execution_mode.value if result.execution_mode else None,
                    details={
                        "url": result.url,
                        "has_screenshot": has_screenshot,
                        "screenshot_size": len(result.screenshot) if result.screenshot else 0,
                    },
                    screenshot=base64.b64decode(result.screenshot) if result.screenshot else None,
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="act_screenshot",
                action_type="act",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_act_click(self) -> BrowserTestResult:
        """Test clicking an element."""
        self.log.info("Testing click action")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                # Click the "More information" link on example.com
                result = await client.act(
                    url="https://example.com",
                    instruction="Click the 'More information' link",
                    screenshot=True,
                )
                latency = (time.perf_counter() - start) * 1000

                test_result = BrowserTestResult(
                    test_name="act_click",
                    action_type="act",
                    passed=result.success,
                    latency_ms=latency,
                    execution_mode=result.execution_mode.value if result.execution_mode else None,
                    details={
                        "url": result.url,
                        "message": result.message,
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="act_click",
                action_type="act",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_extract_data(self) -> BrowserTestResult:
        """Test extracting data from a page."""
        self.log.info("Testing data extraction")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                result = await client.extract(
                    url="https://example.com",
                    instruction="Extract the main heading text from this page",
                )
                latency = (time.perf_counter() - start) * 1000

                has_data = result.data is not None and len(str(result.data)) > 0

                test_result = BrowserTestResult(
                    test_name="extract_data",
                    action_type="extract",
                    passed=result.success and has_data,
                    latency_ms=latency,
                    details={
                        "url": result.url,
                        "extracted_data": str(result.data)[:200] if result.data else None,
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="extract_data",
                action_type="extract",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_multi_step_flow(self) -> BrowserTestResult:
        """Test a multi-step browser flow."""
        self.log.info("Testing multi-step flow")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                result = await client.test(
                    url="https://example.com",
                    steps=[
                        "Verify the page has loaded",
                        "Find and read the main heading",
                        "Take a screenshot",
                    ],
                    browser="chrome",
                    capture_screenshots=True,
                )
                latency = (time.perf_counter() - start) * 1000

                test_result = BrowserTestResult(
                    test_name="multi_step_flow",
                    action_type="test",
                    passed=result.success,
                    latency_ms=latency,
                    details={
                        "steps_count": len(result.steps) if result.steps else 0,
                        "screenshots_count": len(result.screenshots) if result.screenshots else 0,
                        "healing_applied": result.healing_report is not None,
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="multi_step_flow",
                action_type="test",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def test_self_healing_selector(self) -> BrowserTestResult:
        """Test self-healing when selector doesn't match."""
        self.log.info("Testing self-healing selector")
        start = time.perf_counter()

        try:
            client = await self._get_client()
            async with client:
                # Try to click a non-existent selector - should self-heal
                result = await client.act(
                    url="https://example.com",
                    instruction="Click the link that goes to more information (even if selector is wrong)",
                    self_heal=True,
                    screenshot=True,
                )
                latency = (time.perf_counter() - start) * 1000

                test_result = BrowserTestResult(
                    test_name="self_healing_selector",
                    action_type="act",
                    passed=result.success,
                    latency_ms=latency,
                    execution_mode=result.execution_mode.value if result.execution_mode else None,
                    details={
                        "url": result.url,
                        "message": result.message,
                        "used_fallback": result.execution_mode and result.execution_mode.value in ["vision", "hybrid"],
                    },
                    error=result.error,
                )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = BrowserTestResult(
                test_name="self_healing_selector",
                action_type="act",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

        self.report.add(test_result)
        return test_result

    async def run_all(self) -> BrowserTestReport:
        """Run all browser pool tests."""
        if not self._check_config():
            self.log.error("Browser pool not configured, skipping tests")
            return self.report

        self.log.info("Starting browser pool tests")

        await self.test_observe_simple_page()
        await self.test_observe_complex_page()
        await self.test_act_screenshot()
        await self.test_act_click()
        await self.test_extract_data()
        await self.test_multi_step_flow()
        await self.test_self_healing_selector()

        self.log.info(
            "Browser pool tests complete",
            total=len(self.report.tests),
            passed=self.report.passed_count,
            avg_latency=f"{self.report.avg_latency_ms:.0f}ms",
        )

        return self.report


# =========================================================================
# PYTEST TESTS
# =========================================================================

@pytest.fixture
def browser_tester():
    return BrowserPoolTester()


def skip_if_no_browser_pool():
    """Skip test if browser pool not configured."""
    if not os.environ.get("BROWSER_POOL_URL"):
        pytest.skip("BROWSER_POOL_URL not configured")
    if not os.environ.get("BROWSER_POOL_JWT_SECRET"):
        pytest.skip("BROWSER_POOL_JWT_SECRET not configured")


@pytest.mark.asyncio
async def test_observe_page():
    """Test observing a webpage."""
    skip_if_no_browser_pool()

    tester = BrowserPoolTester()
    result = await tester.test_observe_simple_page()

    print(f"\nObserve Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Details: {result.details}")

    assert result.passed, f"Observe failed: {result.error}"


@pytest.mark.asyncio
async def test_act_screenshot():
    """Test taking a screenshot."""
    skip_if_no_browser_pool()

    tester = BrowserPoolTester()
    result = await tester.test_act_screenshot()

    print(f"\nScreenshot Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Mode: {result.execution_mode}")
    print(f"  Details: {result.details}")

    assert result.passed, f"Screenshot failed: {result.error}"


@pytest.mark.asyncio
async def test_multi_step():
    """Test multi-step flow."""
    skip_if_no_browser_pool()

    tester = BrowserPoolTester()
    result = await tester.test_multi_step_flow()

    print(f"\nMulti-step Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Details: {result.details}")

    assert result.passed, f"Multi-step failed: {result.error}"


@pytest.mark.asyncio
async def test_all_browser_operations(browser_tester):
    """Run all browser pool tests."""
    skip_if_no_browser_pool()

    report = await browser_tester.run_all()

    # Print report
    import json
    print("\n" + "=" * 60)
    print("BROWSER POOL TEST REPORT")
    print("=" * 60)
    print(json.dumps(report.to_dict(), indent=2))

    assert report.passed_count > 0, "No browser tests passed"
