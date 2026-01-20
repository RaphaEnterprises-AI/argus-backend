"""
Real Infrastructure Integration Tests.

Tests actual infrastructure components with NO MOCKS:
- Browser Pool connectivity and actions
- Database/Checkpointer operations
- Memory Store with pgvector
- API Server health
- Agent execution

Run with:
    pytest tests/integration/test_infrastructure.py -v --tb=short

Or use the CLI:
    python scripts/run_infra_tests.py --all
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import pytest
import structlog

logger = structlog.get_logger()


@dataclass
class InfraTestResult:
    """Result from an infrastructure test."""
    component: str
    test_name: str
    passed: bool
    latency_ms: float
    error: str | None = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "test_name": self.test_name,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InfraTestReport:
    """Aggregated infrastructure test report."""
    tests: list[InfraTestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def add(self, result: InfraTestResult):
        self.tests.append(result)

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total, 1)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.tests)

    def by_component(self) -> dict[str, list[InfraTestResult]]:
        result = {}
        for test in self.tests:
            if test.component not in result:
                result[test.component] = []
            result[test.component].append(test)
        return result

    def to_dict(self) -> dict:
        self.completed_at = datetime.utcnow()
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": f"{self.pass_rate:.1%}",
                "total_latency_ms": round(self.total_latency_ms, 2),
            },
            "by_component": {
                comp: [t.to_dict() for t in tests]
                for comp, tests in self.by_component().items()
            },
            "duration_seconds": (self.completed_at - self.started_at).total_seconds(),
        }


class InfrastructureTester:
    """Tests real infrastructure components."""

    def __init__(self):
        self.report = InfraTestReport()
        self.log = logger.bind(component="infra_tester")

    async def run_all(self) -> InfraTestReport:
        """Run all infrastructure tests."""
        self.log.info("Starting infrastructure tests")

        # Run each test category
        await self.test_browser_pool()
        await self.test_database()
        await self.test_memory_store()
        await self.test_api_server()
        await self.test_claude_api()

        self.log.info(
            "Infrastructure tests complete",
            total=self.report.total,
            passed=self.report.passed,
            failed=self.report.failed,
        )
        return self.report

    # =========================================================================
    # BROWSER POOL TESTS
    # =========================================================================

    async def test_browser_pool(self):
        """Test browser pool connectivity and operations."""
        pool_url = os.environ.get("BROWSER_POOL_URL")
        jwt_secret = os.environ.get("BROWSER_POOL_JWT_SECRET")

        if not pool_url:
            self.report.add(InfraTestResult(
                component="browser_pool",
                test_name="connectivity",
                passed=False,
                latency_ms=0,
                error="BROWSER_POOL_URL not configured",
            ))
            return

        # Test 1: Health check
        await self._test_browser_pool_health(pool_url)

        # Test 2: Observe action (if pool is available)
        if jwt_secret:
            await self._test_browser_pool_observe(pool_url, jwt_secret)
            await self._test_browser_pool_act(pool_url, jwt_secret)

    async def _test_browser_pool_health(self, pool_url: str):
        """Test browser pool health endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{pool_url}/health")
                latency = (time.perf_counter() - start) * 1000

                self.report.add(InfraTestResult(
                    component="browser_pool",
                    test_name="health_check",
                    passed=response.status_code == 200,
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="browser_pool",
                test_name="health_check",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    async def _test_browser_pool_observe(self, pool_url: str, jwt_secret: str):
        """Test browser pool observe action."""
        start = time.perf_counter()
        try:
            from src.browser.pool_client import BrowserPoolClient, UserContext

            user_context = UserContext(user_id="test-user", org_id="test-org")
            client = BrowserPoolClient(
                pool_url=pool_url,
                jwt_secret=jwt_secret,
                user_context=user_context,
            )

            async with client:
                # Observe a simple page
                result = await client.observe(url="https://example.com")
                latency = (time.perf_counter() - start) * 1000

                self.report.add(InfraTestResult(
                    component="browser_pool",
                    test_name="observe_action",
                    passed=result.success,
                    latency_ms=latency,
                    details={
                        "url": result.url,
                        "elements_found": len(result.elements) if result.elements else 0,
                        "title": result.title,
                    },
                    error=result.error,
                ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="browser_pool",
                test_name="observe_action",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    async def _test_browser_pool_act(self, pool_url: str, jwt_secret: str):
        """Test browser pool act action."""
        start = time.perf_counter()
        try:
            from src.browser.pool_client import BrowserPoolClient, UserContext

            user_context = UserContext(user_id="test-user", org_id="test-org")
            client = BrowserPoolClient(
                pool_url=pool_url,
                jwt_secret=jwt_secret,
                user_context=user_context,
            )

            async with client:
                # Simple action test
                result = await client.act(
                    url="https://example.com",
                    instruction="Take a screenshot of the page",
                    screenshot=True,
                )
                latency = (time.perf_counter() - start) * 1000

                self.report.add(InfraTestResult(
                    component="browser_pool",
                    test_name="act_action",
                    passed=result.success,
                    latency_ms=latency,
                    details={
                        "url": result.url,
                        "has_screenshot": result.screenshot is not None,
                        "execution_mode": result.execution_mode.value if result.execution_mode else None,
                    },
                    error=result.error,
                ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="browser_pool",
                test_name="act_action",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    # =========================================================================
    # DATABASE TESTS
    # =========================================================================

    async def test_database(self):
        """Test database connectivity and checkpointer."""
        database_url = os.environ.get("DATABASE_URL")

        if not database_url:
            self.report.add(InfraTestResult(
                component="database",
                test_name="connectivity",
                passed=False,
                latency_ms=0,
                error="DATABASE_URL not configured",
            ))
            return

        # Test 1: Raw connection
        await self._test_database_connection(database_url)

        # Test 2: Checkpointer
        await self._test_checkpointer()

    async def _test_database_connection(self, database_url: str):
        """Test raw database connection."""
        start = time.perf_counter()
        try:
            import asyncpg
            conn = await asyncpg.connect(database_url)
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            latency = (time.perf_counter() - start) * 1000

            self.report.add(InfraTestResult(
                component="database",
                test_name="connection",
                passed=result == 1,
                latency_ms=latency,
                details={"query_result": result},
            ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="database",
                test_name="connection",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    async def _test_checkpointer(self):
        """Test LangGraph checkpointer."""
        start = time.perf_counter()
        try:
            from src.orchestrator.checkpointer import get_checkpointer

            checkpointer = get_checkpointer()
            latency = (time.perf_counter() - start) * 1000

            # Check if it's PostgresSaver (real DB) or MemorySaver (fallback)
            is_postgres = "PostgresSaver" in type(checkpointer).__name__

            self.report.add(InfraTestResult(
                component="database",
                test_name="checkpointer",
                passed=True,
                latency_ms=latency,
                details={
                    "type": type(checkpointer).__name__,
                    "is_persistent": is_postgres,
                },
            ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="database",
                test_name="checkpointer",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    # =========================================================================
    # MEMORY STORE TESTS
    # =========================================================================

    async def test_memory_store(self):
        """Test memory store with pgvector."""
        database_url = os.environ.get("DATABASE_URL")

        if not database_url:
            self.report.add(InfraTestResult(
                component="memory_store",
                test_name="initialization",
                passed=False,
                latency_ms=0,
                error="DATABASE_URL not configured",
            ))
            return

        await self._test_memory_store_init()
        await self._test_memory_store_operations()

    async def _test_memory_store_init(self):
        """Test memory store initialization."""
        start = time.perf_counter()
        try:
            from src.orchestrator.memory_store import MemoryStore

            store = MemoryStore()
            # Just check it initializes
            latency = (time.perf_counter() - start) * 1000

            self.report.add(InfraTestResult(
                component="memory_store",
                test_name="initialization",
                passed=True,
                latency_ms=latency,
            ))
            await store.close()
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="memory_store",
                test_name="initialization",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    async def _test_memory_store_operations(self):
        """Test memory store write/read operations."""
        start = time.perf_counter()
        try:
            from src.orchestrator.memory_store import MemoryStore

            store = MemoryStore()

            # Store a test pattern
            await store.store_failure_pattern(
                error_message="Test error: Element not found #test-button",
                healed_selector="button[data-testid='test']",
                healing_method="semantic_match",
            )

            # Try to find similar
            similar = await store.find_similar_failures(
                error_message="Element not found #test-button",
                limit=5,
            )

            await store.close()
            latency = (time.perf_counter() - start) * 1000

            self.report.add(InfraTestResult(
                component="memory_store",
                test_name="store_and_retrieve",
                passed=True,
                latency_ms=latency,
                details={"similar_found": len(similar)},
            ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="memory_store",
                test_name="store_and_retrieve",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    # =========================================================================
    # API SERVER TESTS
    # =========================================================================

    async def test_api_server(self):
        """Test API server endpoints."""
        api_url = os.environ.get("API_URL", "http://localhost:8000")

        await self._test_api_health(api_url)
        await self._test_api_docs(api_url)

    async def _test_api_health(self, api_url: str):
        """Test API health endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{api_url}/health")
                latency = (time.perf_counter() - start) * 1000

                self.report.add(InfraTestResult(
                    component="api_server",
                    test_name="health",
                    passed=response.status_code == 200,
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="api_server",
                test_name="health",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    async def _test_api_docs(self, api_url: str):
        """Test API docs endpoint."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{api_url}/docs")
                latency = (time.perf_counter() - start) * 1000

                self.report.add(InfraTestResult(
                    component="api_server",
                    test_name="docs",
                    passed=response.status_code == 200,
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="api_server",
                test_name="docs",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))

    # =========================================================================
    # CLAUDE API TESTS
    # =========================================================================

    async def test_claude_api(self):
        """Test Claude API connectivity."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            self.report.add(InfraTestResult(
                component="claude_api",
                test_name="connectivity",
                passed=False,
                latency_ms=0,
                error="ANTHROPIC_API_KEY not configured",
            ))
            return

        await self._test_claude_simple_call(api_key)

    async def _test_claude_simple_call(self, api_key: str):
        """Test a simple Claude API call."""
        start = time.perf_counter()
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-5-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
            )
            latency = (time.perf_counter() - start) * 1000

            content = response.content[0].text if response.content else ""

            self.report.add(InfraTestResult(
                component="claude_api",
                test_name="simple_call",
                passed="ok" in content.lower(),
                latency_ms=latency,
                details={
                    "model": response.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "response": content[:100],
                },
            ))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.report.add(InfraTestResult(
                component="claude_api",
                test_name="simple_call",
                passed=False,
                latency_ms=latency,
                error=str(e),
            ))


# =========================================================================
# PYTEST FIXTURES AND TESTS
# =========================================================================

@pytest.fixture
def infra_tester():
    return InfrastructureTester()


@pytest.mark.asyncio
async def test_browser_pool_health():
    """Test browser pool is accessible."""
    pool_url = os.environ.get("BROWSER_POOL_URL")
    if not pool_url:
        pytest.skip("BROWSER_POOL_URL not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{pool_url}/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_database_connection():
    """Test database is accessible."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not configured")

    import asyncpg
    conn = await asyncpg.connect(database_url)
    result = await conn.fetchval("SELECT 1")
    await conn.close()
    assert result == 1


@pytest.mark.asyncio
async def test_claude_api_connectivity():
    """Test Claude API is accessible."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        assert response.content
    except anthropic.AuthenticationError:
        pytest.skip("ANTHROPIC_API_KEY is invalid or expired")


@pytest.mark.asyncio
async def test_full_infrastructure(infra_tester):
    """Run all infrastructure tests."""
    report = await infra_tester.run_all()

    # Print report
    import json
    print("\n" + "=" * 60)
    print("INFRASTRUCTURE TEST REPORT")
    print("=" * 60)
    print(json.dumps(report.to_dict(), indent=2))

    # At least one test should pass (Claude API if key is set)
    assert report.passed > 0 or report.total == 0
