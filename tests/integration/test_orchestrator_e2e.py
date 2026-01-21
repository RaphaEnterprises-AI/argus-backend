"""
Real Orchestrator End-to-End Tests.

Tests the actual LangGraph orchestrator with real:
- Agent execution (CodeAnalyzer, TestPlanner, SelfHealer, etc.)
- State persistence (checkpointer)
- Browser pool integration
- Cost/token tracking

Run with:
    pytest tests/integration/test_orchestrator_e2e.py -v -s

Or specific tests:
    pytest tests/integration/test_orchestrator_e2e.py::test_code_analysis_flow -v -s
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import structlog

logger = structlog.get_logger()


@dataclass
class OrchestratorTestResult:
    """Result from an orchestrator test."""
    test_name: str
    phase: str  # analysis, planning, execution, healing, reporting
    passed: bool
    latency_ms: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    state_snapshot: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "phase": self.phase,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
            "state_keys": list(self.state_snapshot.keys()),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrchestratorTestReport:
    """Aggregated orchestrator test report."""
    tests: list[OrchestratorTestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def add(self, result: OrchestratorTestResult):
        self.tests.append(result)

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens_used for t in self.tests)

    @property
    def total_cost(self) -> float:
        return sum(t.cost_usd for t in self.tests)

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.tests)

    def to_dict(self) -> dict:
        self.completed_at = datetime.utcnow()
        return {
            "summary": {
                "total_tests": len(self.tests),
                "passed": self.passed_count,
                "failed": len(self.tests) - self.passed_count,
                "total_tokens": self.total_tokens,
                "total_cost_usd": f"${self.total_cost:.4f}",
                "total_latency_ms": round(self.total_latency_ms, 2),
            },
            "tests": [t.to_dict() for t in self.tests],
            "duration_seconds": (self.completed_at - self.started_at).total_seconds(),
        }


# Sample test application code for analysis
SAMPLE_REACT_APP = '''
// src/App.tsx
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import ProductList from './pages/ProductList';
import Cart from './pages/Cart';
import Checkout from './pages/Checkout';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/products" element={<ProductList />} />
        <Route path="/cart" element={<Cart />} />
        <Route path="/checkout" element={<Checkout />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
'''

SAMPLE_LOGIN_PAGE = '''
// src/pages/LoginPage.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../api/auth';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container" data-testid="login-page">
      <h1>Sign In</h1>
      <form onSubmit={handleSubmit} data-testid="login-form">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          data-testid="email-input"
          required
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          data-testid="password-input"
          required
        />
        {error && <div className="error" data-testid="error-message">{error}</div>}
        <button type="submit" disabled={loading} data-testid="login-button">
          {loading ? 'Signing in...' : 'Sign In'}
        </button>
      </form>
      <a href="/forgot-password" data-testid="forgot-password-link">Forgot password?</a>
    </div>
  );
}
'''

SAMPLE_API_AUTH = '''
// src/api/auth.ts
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

export async function login(email: string, password: string): Promise<{ token: string }> {
  const response = await axios.post(`${API_URL}/auth/login`, { email, password });
  localStorage.setItem('token', response.data.token);
  return response.data;
}

export async function logout(): Promise<void> {
  localStorage.removeItem('token');
}

export function getToken(): string | null {
  return localStorage.getItem('token');
}
'''


def create_sample_codebase() -> str:
    """Create a temporary sample codebase for testing."""
    temp_dir = tempfile.mkdtemp(prefix="argus_test_")

    # Create directory structure
    src_dir = Path(temp_dir) / "src"
    pages_dir = src_dir / "pages"
    api_dir = src_dir / "api"

    pages_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)

    # Write files
    (src_dir / "App.tsx").write_text(SAMPLE_REACT_APP)
    (pages_dir / "LoginPage.tsx").write_text(SAMPLE_LOGIN_PAGE)
    (api_dir / "auth.ts").write_text(SAMPLE_API_AUTH)

    # Create package.json
    package_json = {
        "name": "sample-react-app",
        "version": "1.0.0",
        "dependencies": {
            "react": "^18.0.0",
            "react-router-dom": "^6.0.0",
            "axios": "^1.0.0"
        }
    }
    (Path(temp_dir) / "package.json").write_text(json.dumps(package_json, indent=2))

    return temp_dir


class OrchestratorE2ETester:
    """Tests real orchestrator flows."""

    def __init__(self):
        self.report = OrchestratorTestReport()
        self.log = logger.bind(component="orchestrator_e2e")

    async def test_code_analysis_phase(self) -> OrchestratorTestResult:
        """Test the code analysis phase with real CodeAnalyzerAgent."""
        self.log.info("Testing code analysis phase")
        start = time.perf_counter()

        try:
            from src.agents.code_analyzer import CodeAnalyzerAgent

            # Create sample codebase
            codebase_path = create_sample_codebase()

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path=codebase_path,
                app_url="http://localhost:3000",
            )

            latency = (time.perf_counter() - start) * 1000

            if result.success and result.data:
                data = result.data
                test_result = OrchestratorTestResult(
                    test_name="code_analysis",
                    phase="analysis",
                    passed=len(data.testable_surfaces) > 0,
                    latency_ms=latency,
                    tokens_used=result.input_tokens + result.output_tokens,
                    cost_usd=result.cost or 0,
                    state_snapshot={
                        "surfaces_found": len(data.testable_surfaces),
                        "framework": data.framework_detected,
                        "language": data.language,
                        "summary_length": len(data.summary),
                    },
                )
            else:
                test_result = OrchestratorTestResult(
                    test_name="code_analysis",
                    phase="analysis",
                    passed=False,
                    latency_ms=latency,
                    error=result.error,
                )

            self.report.add(test_result)
            return test_result

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = OrchestratorTestResult(
                test_name="code_analysis",
                phase="analysis",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            self.report.add(test_result)
            return test_result

    async def test_test_planning_phase(self) -> OrchestratorTestResult:
        """Test the test planning phase with real TestPlannerAgent."""
        self.log.info("Testing test planning phase")
        start = time.perf_counter()

        try:
            from src.agents.test_planner import TestPlannerAgent

            # Sample testable surfaces (simulating output from code analysis)
            testable_surfaces = [
                {
                    "type": "ui",
                    "name": "Login Page",
                    "path": "/login",
                    "priority": "critical",
                    "description": "User authentication page",
                    "test_scenarios": ["successful_login", "invalid_credentials", "empty_fields"],
                },
                {
                    "type": "ui",
                    "name": "Dashboard",
                    "path": "/dashboard",
                    "priority": "high",
                    "description": "Main user dashboard",
                    "test_scenarios": ["loads_correctly", "displays_user_data"],
                },
            ]

            agent = TestPlannerAgent()
            result = await agent.execute(
                testable_surfaces=testable_surfaces,
                app_url="http://localhost:3000",
                codebase_summary="React SPA with login, dashboard, and e-commerce features",
            )

            latency = (time.perf_counter() - start) * 1000

            if result.success and result.data:
                data = result.data
                test_result = OrchestratorTestResult(
                    test_name="test_planning",
                    phase="planning",
                    passed=len(data.tests) > 0,
                    latency_ms=latency,
                    tokens_used=result.input_tokens + result.output_tokens,
                    cost_usd=result.cost or 0,
                    state_snapshot={
                        "tests_generated": len(data.tests),
                        "estimated_duration_min": data.estimated_duration_minutes,
                        "test_names": [t.name for t in data.tests[:5]],
                    },
                )
            else:
                test_result = OrchestratorTestResult(
                    test_name="test_planning",
                    phase="planning",
                    passed=False,
                    latency_ms=latency,
                    error=result.error,
                )

            self.report.add(test_result)
            return test_result

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = OrchestratorTestResult(
                test_name="test_planning",
                phase="planning",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            self.report.add(test_result)
            return test_result

    async def test_self_healing_phase(self) -> OrchestratorTestResult:
        """Test the self-healing phase with real SelfHealerAgent."""
        self.log.info("Testing self-healing phase")
        start = time.perf_counter()

        try:
            from src.agents.self_healer import SelfHealerAgent

            # Simulate a test failure
            test_spec = {
                "id": "login_test_001",
                "name": "Login with valid credentials",
                "steps": [
                    {"action": "goto", "target": "/login"},
                    {"action": "fill", "target": "#email-input", "value": "test@example.com"},
                    {"action": "fill", "target": "#password-input", "value": "password123"},
                    {"action": "click", "target": "#submit-btn"},
                ],
            }

            failure_details = {
                "type": "selector_changed",
                "selector": "#submit-btn",
                "message": "Element not found: #submit-btn",
                "html_context": '''
                <form data-testid="login-form">
                    <input data-testid="email-input" type="email" />
                    <input data-testid="password-input" type="password" />
                    <button data-testid="login-button" type="submit">Sign In</button>
                </form>
                ''',
            }

            agent = SelfHealerAgent()
            result = await agent.execute(
                test_spec=test_spec,
                failure_details=failure_details,
            )

            latency = (time.perf_counter() - start) * 1000

            if result.success and result.data:
                data = result.data
                test_result = OrchestratorTestResult(
                    test_name="self_healing",
                    phase="healing",
                    passed=data.auto_healed or len(data.suggested_fixes) > 0,
                    latency_ms=latency,
                    tokens_used=result.input_tokens + result.output_tokens,
                    cost_usd=result.cost or 0,
                    state_snapshot={
                        "auto_healed": data.auto_healed,
                        "diagnosis_type": data.diagnosis.failure_type.value if data.diagnosis else None,
                        "confidence": data.diagnosis.confidence if data.diagnosis else 0,
                        "fixes_suggested": len(data.suggested_fixes),
                    },
                )
            else:
                test_result = OrchestratorTestResult(
                    test_name="self_healing",
                    phase="healing",
                    passed=False,
                    latency_ms=latency,
                    error=result.error,
                )

            self.report.add(test_result)
            return test_result

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = OrchestratorTestResult(
                test_name="self_healing",
                phase="healing",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            self.report.add(test_result)
            return test_result

    async def test_nlp_test_creation(self) -> OrchestratorTestResult:
        """Test NLP test creation with real NLPTestCreator."""
        self.log.info("Testing NLP test creation")
        start = time.perf_counter()

        try:
            from src.agents.nlp_test_creator import NLPTestCreator

            creator = NLPTestCreator(app_url="http://localhost:3000")

            # Test natural language to test conversion
            result = await creator.create(
                description="Log in with email test@example.com and password secret123, then verify the dashboard loads with a welcome message"
            )

            latency = (time.perf_counter() - start) * 1000

            if result and result.steps:
                test_result = OrchestratorTestResult(
                    test_name="nlp_test_creation",
                    phase="planning",
                    passed=len(result.steps) >= 3,  # Expect at least goto, fill, click
                    latency_ms=latency,
                    state_snapshot={
                        "steps_generated": len(result.steps),
                        "assertions_generated": len(result.assertions),
                        "test_name": result.name,
                        "step_actions": [s.action for s in result.steps[:5]],
                    },
                )
            else:
                test_result = OrchestratorTestResult(
                    test_name="nlp_test_creation",
                    phase="planning",
                    passed=False,
                    latency_ms=latency,
                    error="No steps generated",
                )

            self.report.add(test_result)
            return test_result

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = OrchestratorTestResult(
                test_name="nlp_test_creation",
                phase="planning",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            self.report.add(test_result)
            return test_result

    async def test_full_orchestrator_flow(self) -> OrchestratorTestResult:
        """Test the full orchestrator flow end-to-end."""
        self.log.info("Testing full orchestrator flow")
        start = time.perf_counter()

        try:
            from src.orchestrator.graph import TestingOrchestrator

            # Create sample codebase
            codebase_path = create_sample_codebase()

            # Create orchestrator
            orchestrator = TestingOrchestrator(
                codebase_path=codebase_path,
                app_url="http://localhost:3000",
            )

            # Run orchestrator (this runs the full graph)
            # Note: This may take several minutes and cost money
            result = await orchestrator.run(thread_id=f"e2e-test-{int(time.time())}")

            latency = (time.perf_counter() - start) * 1000

            test_result = OrchestratorTestResult(
                test_name="full_orchestrator_flow",
                phase="full",
                passed=result.get("error") is None,
                latency_ms=latency,
                tokens_used=result.get("total_input_tokens", 0) + result.get("total_output_tokens", 0),
                cost_usd=result.get("total_cost", 0),
                state_snapshot={
                    "surfaces_found": len(result.get("testable_surfaces", [])),
                    "tests_planned": len(result.get("test_plan", [])),
                    "tests_executed": result.get("passed_count", 0) + result.get("failed_count", 0),
                    "passed": result.get("passed_count", 0),
                    "failed": result.get("failed_count", 0),
                    "iterations": result.get("iteration", 0),
                },
                error=result.get("error"),
            )

            self.report.add(test_result)
            return test_result

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            test_result = OrchestratorTestResult(
                test_name="full_orchestrator_flow",
                phase="full",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )
            self.report.add(test_result)
            return test_result

    async def run_all(self) -> OrchestratorTestReport:
        """Run all orchestrator tests."""
        self.log.info("Starting orchestrator E2E tests")

        # Run each phase test
        await self.test_code_analysis_phase()
        await self.test_test_planning_phase()
        await self.test_self_healing_phase()
        await self.test_nlp_test_creation()

        self.log.info(
            "Orchestrator E2E tests complete",
            total=len(self.report.tests),
            passed=self.report.passed_count,
            tokens=self.report.total_tokens,
            cost=f"${self.report.total_cost:.4f}",
        )

        return self.report


# =========================================================================
# PYTEST TESTS
# =========================================================================

@pytest.fixture
def orchestrator_tester():
    return OrchestratorE2ETester()


@pytest.mark.asyncio
async def test_code_analysis_flow():
    """Test code analysis with real agent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    tester = OrchestratorE2ETester()
    result = await tester.test_code_analysis_phase()

    print("\nCode Analysis Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Tokens: {result.tokens_used}")
    print(f"  State: {result.state_snapshot}")
    if result.error:
        print(f"  Error: {result.error}")

    assert result.passed, f"Code analysis failed: {result.error}"


@pytest.mark.asyncio
async def test_test_planning_flow():
    """Test test planning with real agent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    tester = OrchestratorE2ETester()
    result = await tester.test_test_planning_phase()

    print("\nTest Planning Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Tokens: {result.tokens_used}")
    print(f"  State: {result.state_snapshot}")

    assert result.passed, f"Test planning failed: {result.error}"


@pytest.mark.asyncio
async def test_self_healing_flow():
    """Test self-healing with real agent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    tester = OrchestratorE2ETester()
    result = await tester.test_self_healing_phase()

    print("\nSelf-Healing Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Tokens: {result.tokens_used}")
    print(f"  State: {result.state_snapshot}")

    assert result.passed, f"Self-healing failed: {result.error}"


@pytest.mark.asyncio
async def test_nlp_test_creation_flow():
    """Test NLP test creation with real agent."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    tester = OrchestratorE2ETester()
    result = await tester.test_nlp_test_creation()

    print("\nNLP Test Creation Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  State: {result.state_snapshot}")

    assert result.passed, f"NLP test creation failed: {result.error}"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_orchestrator():
    """Test full orchestrator flow (slow, costs money)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    tester = OrchestratorE2ETester()
    result = await tester.test_full_orchestrator_flow()

    print("\nFull Orchestrator Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    print(f"  Tokens: {result.tokens_used}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  State: {result.state_snapshot}")

    assert result.passed, f"Full orchestrator failed: {result.error}"


@pytest.mark.asyncio
async def test_all_orchestrator_phases(orchestrator_tester):
    """Run all orchestrator phase tests."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not configured")

    report = await orchestrator_tester.run_all()

    # Print report
    print("\n" + "=" * 60)
    print("ORCHESTRATOR E2E TEST REPORT")
    print("=" * 60)
    print(json.dumps(report.to_dict(), indent=2))

    assert report.passed_count > 0, "No orchestrator tests passed"
