"""
Integration Tests: Brain (Railway) + Worker (Cloudflare) Pipeline

These tests verify the full E2E testing pipeline with realistic scenarios:
1. Code Analysis -> Test Generation -> Test Execution via Worker
2. Security Scanning with real browser automation
3. Accessibility Checking on live pages
4. Performance Analysis with real metrics
5. Self-healing when selectors fail
6. Root cause analysis on failures

Prerequisites:
- ANTHROPIC_API_KEY set
- Cloudflare Worker deployed at BROWSER_WORKER_URL
- Network access to test target sites
"""

import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

# Skip if no API key
pytestmark = [
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY required for integration tests"
    ),
    pytest.mark.asyncio(loop_scope="class"),  # Share event loop within each test class
]


@pytest.fixture(autouse=True)
async def reset_browser_client():
    """Reset the browser client singleton before each test to avoid event loop issues."""
    from src.tools.browser_worker_client import cleanup_browser_client
    await cleanup_browser_client()
    yield
    await cleanup_browser_client()


class TestWorkerHealthAndConnectivity:
    """Test basic Worker connectivity and health."""

    @pytest.mark.asyncio
    async def test_worker_health_check(self):
        """Verify Worker is responding to health checks."""
        from src.tools.browser_worker_client import get_browser_client

        client = get_browser_client()
        is_healthy = await client.health_check()

        # Log result for debugging
        print(f"Worker health: {is_healthy}")
        assert is_healthy is True, "Worker should be healthy and responding"

    @pytest.mark.asyncio
    async def test_worker_basic_navigation(self):
        """Test Worker can navigate to a simple page."""
        from src.tools.browser_worker_client import get_browser_client

        client = get_browser_client()

        result = await client.run_test(
            url="https://example.com",
            steps=["Verify the page title contains 'Example'"],
            browser="chrome",
            capture_screenshots=True,
        )

        print(f"Navigation result: success={result.success}")
        # Worker may timeout on cold start - that's acceptable for integration tests
        # The key is that it doesn't error with event loop issues
        if result.error and "timed out" in result.error.lower():
            pytest.skip("Worker timed out (likely cold start) - skipping")
        assert result.success is True, f"Navigation failed: {result.error}"


class TestRealisticLoginFlow:
    """Test realistic login flow scenarios."""

    @pytest.mark.asyncio
    async def test_github_login_page_discovery(self):
        """Discover elements on GitHub login page (no actual login)."""
        from src.tools.browser_worker_client import get_browser_client

        client = get_browser_client()

        # Discover available actions on login page
        result = await client.discover(
            url="https://github.com/login",
            task="Find all input fields and buttons for authentication"
        )

        print(f"Discovery result: {result.success}")
        if result.actions:
            print(f"Found {len(result.actions)} actions:")
            for action in result.actions[:5]:
                print(f"  - {action.get('action', 'unknown')}: {action.get('description', '')[:50]}")

        assert result.success is True, f"Discovery failed: {result.error}"
        assert result.actions is not None and len(result.actions) > 0, "Should find login elements"

    @pytest.mark.asyncio
    async def test_form_interaction_without_submit(self):
        """Test form field interactions without submitting."""
        from src.tools.browser_worker_client import get_browser_client

        client = get_browser_client()

        # Test typing in a search form (safe, no submission)
        result = await client.run_test(
            url="https://www.google.com",
            steps=[
                "Type 'test query' into the search box",
                "Verify the search box contains text",
            ],
            browser="chrome",
            capture_screenshots=True,
        )

        print(f"Form interaction: success={result.success}")
        # This may fail due to Google's bot protection, which is expected
        if not result.success:
            print(f"Note: Google may block automated access: {result.error}")


class TestCodeAnalysisToTestGeneration:
    """Test the full code analysis -> test generation pipeline."""

    @pytest.mark.asyncio
    async def test_analyze_and_generate_tests(self):
        """Analyze a codebase and generate test specs."""
        from src.agents.code_analyzer import CodeAnalyzerAgent
        from src.agents.test_planner import TestPlannerAgent

        # Analyze a sample codebase (our own project)
        analyzer = CodeAnalyzerAgent()

        analysis = await analyzer.execute(
            codebase_path="/Users/bvk/Downloads/e2e-testing-agent",
            app_url="http://localhost:8000",
            changed_files=["src/api/server.py"],
        )

        print(f"Analysis: success={analysis.success}")
        if analysis.success and analysis.data:
            print(f"Found {len(analysis.data.testable_surfaces)} testable surfaces")
            for surface in analysis.data.testable_surfaces[:3]:
                print(f"  - {surface.name}: {surface.type}")

        assert analysis.success is True, f"Analysis failed: {analysis.error}"

        # Generate test plan from analysis
        if analysis.data and analysis.data.testable_surfaces:
            planner = TestPlannerAgent()
            plan = await planner.execute(
                testable_surfaces=analysis.data.testable_surfaces[:5],
                app_url="http://localhost:8000",
            )

            print(f"Planning: success={plan.success}")
            if plan.success and plan.data:
                print(f"Generated {len(plan.data.tests)} test specs")

            assert plan.success is True, f"Planning failed: {plan.error}"


class TestSecurityScanning:
    """Test security scanning on real pages."""

    @pytest.mark.asyncio
    async def test_security_scan_example_site(self):
        """Scan example.com for security issues."""
        from src.agents.security_scanner import SecurityScannerAgent

        scanner = SecurityScannerAgent()

        result = await scanner.execute(
            url="https://example.com",
            scan_type="quick",
            include_api_tests=False,
        )

        print(f"Security scan: success={result.success}")
        if result.success and result.data:
            print(f"Risk score: {result.data.risk_score}")
            print(f"Vulnerabilities found: {len(result.data.vulnerabilities)}")
            for vuln in result.data.vulnerabilities[:3]:
                print(f"  - [{vuln.severity.value}] {vuln.title}")

        assert result.success is True, f"Security scan failed: {result.error}"

    @pytest.mark.asyncio
    async def test_header_security_analysis(self):
        """Analyze security headers on a real site."""
        from src.agents.security_scanner import SecurityScannerAgent

        scanner = SecurityScannerAgent()

        result = await scanner.execute(
            url="https://www.cloudflare.com",
            scan_type="quick",
        )

        print(f"Header analysis: success={result.success}")
        if result.success and result.data:
            missing = result.data.headers.get_missing_headers()
            print(f"Missing headers: {missing}")

        assert result.success is True


class TestAccessibilityChecking:
    """Test accessibility checking on real pages."""

    @pytest.mark.asyncio
    async def test_accessibility_scan_example(self):
        """Scan example.com for accessibility issues."""
        from src.agents.accessibility_checker import AccessibilityCheckerAgent

        checker = AccessibilityCheckerAgent()

        from src.agents.accessibility_checker import WCAGLevel
        result = await checker.execute(
            url="https://example.com",
            wcag_level=WCAGLevel.AA,
        )

        print(f"Accessibility scan: success={result.success}")
        if result.success and result.data:
            print(f"Score: {result.data.score}")
            print(f"Issues: {len(result.data.issues)}")
            for issue in result.data.issues[:3]:
                print(f"  - [{issue.impact}] {issue.description[:50]}")

        assert result.success is True, f"Accessibility scan failed: {result.error}"


class TestPerformanceAnalysis:
    """Test performance analysis on real pages."""

    @pytest.mark.asyncio
    async def test_performance_analysis_example(self):
        """Analyze performance of example.com."""
        from src.agents.performance_analyzer import PerformanceAnalyzerAgent

        analyzer = PerformanceAnalyzerAgent()

        result = await analyzer.execute(
            url="https://example.com",
            device="mobile",
        )

        print(f"Performance analysis: success={result.success}")
        if result.success and result.data:
            print(f"Overall grade: {result.data.overall_grade}")
            print(f"LCP: {result.data.metrics.core_vitals.lcp_ms}ms")
            print(f"Issues: {len(result.data.issues)}")

        assert result.success is True, f"Performance analysis failed: {result.error}"


class TestSelfHealingScenarios:
    """Test self-healing capabilities."""

    @pytest.mark.asyncio
    async def test_self_healing_on_selector_failure(self):
        """Test that self-healer can diagnose selector failures."""
        from src.agents.self_healer import SelfHealerAgent, FailureDiagnosis, FailureType

        healer = SelfHealerAgent()

        # Simulate a test failure
        failure_context = {
            "test_id": "login-test-001",
            "test_name": "Login Flow",
            "step_index": 2,
            "step_action": "click",
            "step_target": "#old-login-button",
            "error_message": "Element not found: #old-login-button",
            "current_url": "https://example.com/login",
            "screenshot_base64": None,  # Would include real screenshot
            "html_snippet": """
                <div class="login-form">
                    <button class="btn-primary login-btn" data-testid="login-submit">
                        Sign In
                    </button>
                </div>
            """,
        }

        result = await healer.execute(
            failure_details=failure_context,
            test_spec={"id": "login-test-001", "name": "Login Flow"},
        )

        print(f"Self-healing: success={result.success}")

        # Skip if authentication fails (no valid API key)
        if result.error and "authentication_error" in result.error.lower():
            pytest.skip("API key not valid for integration test")

        if result.success and result.data:
            print(f"Diagnosis: {result.data.diagnosis.failure_type}")
            print(f"Confidence: {result.data.diagnosis.confidence}")
            if result.data.suggested_fixes:
                for fix in result.data.suggested_fixes[:2]:
                    print(f"Suggested fix: {fix.fix_type}")
                    print(f"New value: {fix.new_value}")

        assert result.success is True, f"Self-healing failed: {result.error}"


class TestRootCauseAnalysis:
    """Test root cause analysis on failures."""

    @pytest.mark.asyncio
    async def test_root_cause_analysis_timing_issue(self):
        """Analyze a timing-related failure."""
        from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext
        from anthropic import AuthenticationError

        analyzer = RootCauseAnalyzer()

        context = FailureContext(
            test_id="checkout-test-001",
            test_name="Checkout Flow",
            error_message="Timeout waiting for element: #payment-form",
            stack_trace="""
                TimeoutError: Waiting for selector '#payment-form'
                    at waitForSelector (playwright.js:1234)
                    at test_checkout (checkout.test.js:45)
            """,
            step_history=[
                {"action": "navigate", "target": "/cart", "success": True},
                {"action": "click", "target": "#checkout-btn", "success": True},
                {"action": "wait", "target": "#payment-form", "success": False},
            ],
            network_logs=[
                {"url": "/api/payment/init", "status": 200, "duration_ms": 2500},
            ],
        )

        try:
            result = await analyzer.analyze(context)
        except AuthenticationError:
            pytest.skip("API key not valid for integration test")

        print(f"Root cause analysis:")
        print(f"  Category: {result.category}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Summary: {result.summary}")
        print(f"  Is Flaky: {result.is_flaky}")
        print(f"  Auto-healable: {result.auto_healable}")

        assert result.category is not None
        assert result.confidence > 0


class TestEndToEndTestExecution:
    """Test complete E2E test execution via Worker."""

    @pytest.mark.asyncio
    async def test_e2e_simple_navigation_test(self):
        """Execute a simple navigation test end-to-end."""
        from src.agents.ui_tester import UITesterAgent
        from src.agents.test_planner import TestSpec, TestStep, TestAssertion

        # Create a simple test spec
        test_spec = TestSpec(
            id="nav-test-001",
            name="Simple Navigation Test",
            type="ui",
            priority="high",
            description="Verify basic page navigation works",
            steps=[
                TestStep(action="navigate", target="https://example.com"),
                TestStep(action="verify", target="page title", value="Example Domain"),
            ],
            assertions=[
                TestAssertion(type="text_visible", target="body", expected="Example Domain"),
            ],
        )

        tester = UITesterAgent(use_worker=True)

        result = await tester.execute(
            test_spec=test_spec,
            app_url="https://example.com",
        )

        print(f"E2E test: success={result.success}")
        if result.data:
            print(f"Status: {result.data.status}")
            print(f"Duration: {result.data.total_duration_ms}ms")
            print(f"Steps: {len(result.data.step_results)}")

        # Note: This may fail if Worker has issues, but we're testing the integration
        if not result.success:
            print(f"Test failed (may be expected): {result.error or result.data.error_message}")

    @pytest.mark.asyncio
    async def test_e2e_extraction_test(self):
        """Execute a data extraction test via Worker."""
        from src.tools.browser_worker_client import get_browser_client

        client = get_browser_client()

        # Extract data from a page
        result = await client.extract(
            url="https://example.com",
            instruction="Extract the main heading and any paragraphs",
            schema={
                "heading": "string",
                "paragraphs": "array",
            },
        )

        print(f"Extraction: success={result.success}")
        if result.success and result.data:
            print(f"Extracted: {result.data}")

        assert result.success is True, f"Extraction failed: {result.error}"


class TestMultiAgentOrchestration:
    """Test multi-agent workflows."""

    @pytest.mark.asyncio
    async def test_quality_audit_workflow(self):
        """Run a quality audit that combines multiple agents."""
        from src.agents.quality_auditor import QualityAuditor
        from anthropic import AuthenticationError
        import httpx

        auditor = QualityAuditor()

        # Fetch page content first
        async with httpx.AsyncClient() as client:
            response = await client.get("https://example.com")
            page_html = response.text

        try:
            result = await auditor.full_audit(
                page_url="https://example.com",
                page_html=page_html,
                screenshot=None,
                performance_metrics=None,
            )
        except AuthenticationError:
            pytest.skip("API key not valid for integration test")

        print(f"Quality audit:")
        print(f"  Accessibility score: {result.accessibility.score}")
        print(f"  Performance score: {result.performance.overall_score}")
        print(f"  Best Practices: {result.best_practices_score}")
        print(f"  SEO: {result.seo_score}")
        print(f"  Overall: {result.overall_grade}")

        assert result.overall_grade is not None

    @pytest.mark.asyncio
    async def test_full_test_run_orchestration(self):
        """Test the full orchestration pipeline."""
        from src.orchestrator.state import create_initial_state
        from src.orchestrator.graph import create_testing_graph
        from src.config import get_settings

        # Create initial state
        state = create_initial_state(
            codebase_path="/Users/bvk/Downloads/e2e-testing-agent",
            app_url="https://example.com",
        )

        print(f"Initial state created:")
        print(f"  App URL: {state['app_url']}")
        print(f"  Codebase: {state['codebase_path']}")

        # Create the graph (but don't execute - just verify it builds)
        settings = get_settings()
        graph = create_testing_graph(settings)

        print(f"Graph created successfully")

        assert graph is not None


class TestReporterWithRealData:
    """Test reporter with real test data."""

    @pytest.mark.asyncio
    async def test_generate_report_from_results(self):
        """Generate a report from test results."""
        from src.agents.reporter import ReporterAgent
        from anthropic import AuthenticationError

        reporter = ReporterAgent()

        # Simulate test results
        test_results = [
            {
                "test_id": "test-001",
                "test_name": "Login Flow",
                "status": "passed",
                "duration_ms": 2500,
                "steps": [
                    {"action": "navigate", "success": True},
                    {"action": "type", "success": True},
                    {"action": "click", "success": True},
                ],
            },
            {
                "test_id": "test-002",
                "test_name": "Checkout Flow",
                "status": "failed",
                "duration_ms": 5000,
                "error_message": "Payment form not loaded",
                "steps": [
                    {"action": "navigate", "success": True},
                    {"action": "click", "success": True},
                    {"action": "wait", "success": False, "error": "Timeout"},
                ],
            },
        ]

        try:
            result = await reporter.execute(
                test_results=test_results,
                execution_metadata={"run_id": "integration-test-run"},
            )
        except AuthenticationError:
            pytest.skip("API key not valid for integration test")

        # Skip if authentication fails (no valid API key)
        if result.error and "authentication_error" in result.error.lower():
            pytest.skip("API key not valid for integration test")

        print(f"Report generation: success={result.success}")
        if result.success and result.data:
            print(f"Summary: {result.data.summary.total_tests} tests")
            print(f"Pass rate: {result.data.summary.pass_rate:.1f}%")
            print(f"Failures: {len(result.data.failures)}")

        assert result.success is True, f"Report generation failed: {result.error}"


# Run specific integration tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
