"""Tests for the swarm orchestrator, throttler, and intent verifier.

Covers:
- 5 newly wired agent runners (ui_tester, api_tester, self_healer, flaky_detector, visual_ai)
- Grok 4 model registration
- Tiered throttling (FREE/PRO/ENTERPRISE)
- Discovery swarm mode
- Intent verification (fast path, heuristic fallback, LLM path)
- Swarm lifecycle (launch → execute → verify → finish)
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def swarm_config():
    """Minimal SwarmConfig for testing."""
    from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode

    return SwarmConfig(
        mode=SwarmMode.FULL_CRAWL,
        org_id="test-org",
        project_id="test-project",
        user_id="test-user",
        target_url="https://example.com",
    )


@pytest.fixture
def pr_config():
    """SwarmConfig for PR analysis mode."""
    from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode

    return SwarmConfig(
        mode=SwarmMode.PR_ANALYSIS,
        org_id="test-org",
        project_id="test-project",
        user_id="test-user",
        pr_number=42,
        changed_files=[
            {"path": "src/auth.py", "status": "modified", "additions": 10, "deletions": 5},
            {"path": "tests/test_auth.py", "status": "modified", "additions": 20, "deletions": 0},
        ],
    )


@pytest.fixture
def discovery_config():
    """SwarmConfig for discovery swarm mode."""
    from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode

    return SwarmConfig(
        mode=SwarmMode.DISCOVERY_SWARM,
        org_id="test-org",
        project_id="test-project",
        user_id="test-user",
        target_url="https://example.com",
        max_discovery_agents=5,
    )


@pytest.fixture
def mock_emitter():
    """Mock AG-UI emitter that collects events."""
    emitter = AsyncMock()
    emitter.emit = AsyncMock()
    emitter.close = AsyncMock()
    return emitter


@pytest.fixture
def worker_results():
    """Mixed worker results for testing consensus and intent verification."""
    from src.orchestrator.swarm_orchestrator import WorkerResult

    return [
        WorkerResult(
            agent_id="sec_1",
            agent_type="security_scanner",
            success=True,
            duration_ms=1200,
            cost_usd=0.02,
            findings=[{"type": "xss", "severity": "high"}],
            summary="Found 1 XSS vulnerability",
            confidence=0.85,
        ),
        WorkerResult(
            agent_id="ui_1",
            agent_type="ui_tester",
            success=True,
            duration_ms=2000,
            cost_usd=0.01,
            findings=[{"type": "ui_step", "status": "passed"}],
            summary="UI check passed",
            confidence=0.8,
        ),
        WorkerResult(
            agent_id="api_1",
            agent_type="api_tester",
            success=False,
            duration_ms=500,
            cost_usd=0.0,
            error="Connection refused",
        ),
    ]


# ===========================================================================
# 1. Wired Agent Runner Tests
# ===========================================================================


class TestUITesterRunner:
    """Test _run_ui_tester dispatches correctly."""

    @pytest.mark.asyncio
    async def test_ui_tester_dispatches_with_target_url(self, swarm_config, mock_emitter):
        """ui_tester should be dispatched (not error-skipped) when target_url is set."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost = 0.01
        mock_result.data = MagicMock(
            status="passed",
            steps=[],
        )

        with patch("src.agents.UITesterAgent") as MockUI:
            instance = AsyncMock()
            instance.execute = AsyncMock(return_value=mock_result)
            MockUI.return_value = instance

            result = await orch._run_ui_tester(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] > 0, "Should have non-zero confidence"
        assert "passed" in result["summary"]

    @pytest.mark.asyncio
    async def test_ui_tester_falls_back_to_error_result_on_error(self, swarm_config, mock_emitter):
        """If UITesterAgent raises, should gracefully return an error result."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        with patch("src.agents.UITesterAgent", side_effect=ImportError("no browser")):
            result = await orch._run_ui_tester(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 0.0
        assert "no browser" in result["summary"]


class TestAPITesterRunner:
    """Test _run_api_tester dispatches correctly."""

    @pytest.mark.asyncio
    async def test_api_tester_probes_common_endpoints(self, swarm_config, mock_emitter):
        """api_tester should probe /, /api, /health, /api/v1/health."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cost = 0.005
        mock_result.data = MagicMock(
            requests=[
                MagicMock(path="/", status_code=200, success=True),
                MagicMock(path="/api", status_code=404, success=False),
                MagicMock(path="/health", status_code=200, success=True),
                MagicMock(path="/api/v1/health", status_code=200, success=True),
            ],
        )

        with patch("src.agents.APITesterAgent") as MockAPI:
            instance = AsyncMock()
            instance.execute = AsyncMock(return_value=mock_result)
            MockAPI.return_value = instance

            result = await orch._run_api_tester(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 0.7
        assert "3/4 endpoints reachable" in result["summary"]
        assert len(result["findings"]) == 4

    @pytest.mark.asyncio
    async def test_api_tester_error_without_url(self, mock_emitter):
        """api_tester should return error result if no target_url."""
        from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode, SwarmOrchestrator

        config = SwarmConfig(
            mode=SwarmMode.TARGETED_BLITZ,
            org_id="test-org",
            project_id="test-project",
            user_id="test-user",
            target_url=None,
        )
        orch = SwarmOrchestrator()
        result = await orch._execute_agent("api_tester", config, mock_emitter, "agent_1")
        assert result["confidence"] == 0.0, "Should be error result when no URL"


class TestSelfHealerRunner:
    """Test _run_self_healer queries DB and heals."""

    @pytest.mark.asyncio
    async def test_self_healer_returns_nothing_to_heal_when_no_failures(
        self, swarm_config, mock_emitter,
    ):
        """If DB has no failures, should return clean result."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        mock_supabase = AsyncMock()
        mock_supabase.select = AsyncMock(return_value=[])

        with patch(
            "src.integrations.supabase.get_supabase",
            new_callable=AsyncMock,
            return_value=mock_supabase,
        ):
            result = await orch._run_self_healer(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 1.0
        assert "No recent failures" in result["summary"]
        assert result["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_self_healer_error_when_supabase_unavailable(
        self, swarm_config, mock_emitter,
    ):
        """Should gracefully return error result when Supabase is down."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        with patch(
            "src.integrations.supabase.get_supabase",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await orch._run_self_healer(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 0.0
        assert "Supabase not available" in result["summary"]


class TestFlakyDetectorRunner:
    """Test _run_flaky_detector does statistical analysis."""

    @pytest.mark.asyncio
    async def test_flaky_detector_identifies_flaky_tests(self, swarm_config, mock_emitter):
        """Should detect flakiness from mixed pass/fail history."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        # Simulate 10 runs of 2 tests: one stable, one flaky
        rows = []
        for i in range(10):
            rows.append({
                "id": f"run_{i}_stable",
                "test_id": "test_login",
                "test_name": "test_login",
                "status": "passed",
                "duration_ms": 100,
                "created_at": f"2026-02-28T10:0{i}:00Z",
            })
            rows.append({
                "id": f"run_{i}_flaky",
                "test_id": "test_checkout",
                "test_name": "test_checkout",
                "status": "passed" if i % 3 != 0 else "failed",  # Fails 33% of the time
                "duration_ms": 200 + (i * 50),
                "created_at": f"2026-02-28T10:0{i}:00Z",
            })

        mock_supabase = AsyncMock()
        mock_supabase.select = AsyncMock(return_value=rows)

        with patch(
            "src.integrations.supabase.get_supabase",
            new_callable=AsyncMock,
            return_value=mock_supabase,
        ):
            result = await orch._run_flaky_detector(swarm_config, mock_emitter, "agent_1")

        assert result["cost_usd"] == 0.0, "Flaky detection should have zero AI cost"
        assert result["confidence"] == 0.85
        assert "Analyzed 2 tests" in result["summary"]
        # The flaky test should appear in findings
        flaky_findings = [f for f in result["findings"] if f.get("test_id") == "test_checkout"]
        assert len(flaky_findings) > 0, "Should detect test_checkout as flaky"
        assert flaky_findings[0]["flakiness_score"] > 0.1

    @pytest.mark.asyncio
    async def test_flaky_detector_empty_history(self, swarm_config, mock_emitter):
        """Should handle empty test history gracefully."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()
        mock_supabase = AsyncMock()
        mock_supabase.select = AsyncMock(return_value=[])

        with patch(
            "src.integrations.supabase.get_supabase",
            new_callable=AsyncMock,
            return_value=mock_supabase,
        ):
            result = await orch._run_flaky_detector(swarm_config, mock_emitter, "agent_1")

        assert "No test history" in result["summary"]
        assert result["confidence"] == 1.0


class TestVisualAIRunner:
    """Test _run_visual_ai screenshot analysis."""

    @pytest.mark.asyncio
    async def test_visual_ai_analyzes_screenshot(self, swarm_config, mock_emitter):
        """Should take screenshot and analyze it."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        mock_analysis = {
            "page_loaded": True,
            "overall_health": "healthy",
            "main_elements": ["header", "nav", "main", "footer"],
            "issues": [],
        }

        with (
            patch("src.browser.pool_client.BrowserPoolClient") as MockBrowser,
            patch("src.agents.VisualAI") as MockVisual,
        ):
            # Mock browser screenshot
            browser_instance = AsyncMock()
            browser_instance.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            browser_instance.__aenter__ = AsyncMock(return_value=browser_instance)
            browser_instance.__aexit__ = AsyncMock(return_value=False)
            MockBrowser.return_value = browser_instance

            # Mock visual analysis
            visual_instance = MagicMock()
            visual_instance.analyze_single = AsyncMock(return_value=mock_analysis)
            MockVisual.return_value = visual_instance

            result = await orch._run_visual_ai(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 0.9  # "healthy" maps to 0.9
        assert "healthy" in result["summary"]
        assert len(result["findings"]) == 4  # 4 elements

    @pytest.mark.asyncio
    async def test_visual_ai_error_when_browser_unavailable(self, swarm_config, mock_emitter):
        """Should return error result if browser can't take screenshot."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        with patch(
            "src.browser.pool_client.BrowserPoolClient",
            side_effect=ConnectionError("No browser worker"),
        ):
            result = await orch._run_visual_ai(swarm_config, mock_emitter, "agent_1")

        assert result["confidence"] == 0.0
        assert "visual_ai" in result["summary"]


# ===========================================================================
# 2. Grok 4 Model Registration
# ===========================================================================


class TestGrok4Models:
    """Verify Grok 4 models are registered in both provider and router."""

    def test_xai_provider_has_grok4_models(self):
        from src.core.providers.xai_provider import XAIProvider

        expected = ["grok-4-0709", "grok-4-fast-reasoning", "grok-4-1-fast-reasoning", "grok-code-fast-1"]
        for model_id in expected:
            assert model_id in XAIProvider.MODELS, f"Missing {model_id} in XAIProvider.MODELS"

    def test_model_router_has_grok4_entries(self):
        from src.core.model_router import MODELS, ModelProvider

        grok_keys = {"grok-4", "grok-4-fast", "grok-4.1-fast", "grok-code"}
        for key in grok_keys:
            assert key in MODELS, f"Missing {key} in model_router.MODELS"
            assert MODELS[key].provider == ModelProvider.XAI

    def test_grok4_fast_has_2m_context(self):
        """Grok 4 fast models should have 2M context window."""
        from src.core.model_router import MODELS

        assert MODELS["grok-4-fast"].context_window == 2097152
        assert MODELS["grok-4.1-fast"].context_window == 2097152

    def test_grok4_pricing_tiers(self):
        """Expert model should be expensive, fast models should be cheap."""
        from src.core.model_router import MODELS

        assert MODELS["grok-4"].input_cost_per_1m == 3.00  # Expert tier
        assert MODELS["grok-4-fast"].input_cost_per_1m == 0.20  # Value tier
        assert MODELS["grok-code"].input_cost_per_1m == 0.20  # Value tier


# ===========================================================================
# 3. Tiered Throttling
# ===========================================================================


class TestThrottlerTiers:
    """Test FREE/PRO/ENTERPRISE tier configurations."""

    def test_tier_configs_exist(self):
        from src.orchestrator.swarm_throttler import TIER_CONFIGS, ThrottlerTier

        assert ThrottlerTier.FREE in TIER_CONFIGS
        assert ThrottlerTier.PRO in TIER_CONFIGS
        assert ThrottlerTier.ENTERPRISE in TIER_CONFIGS

    def test_free_tier_limits(self):
        from src.orchestrator.swarm_throttler import TIER_CONFIGS, ThrottlerTier

        free = TIER_CONFIGS[ThrottlerTier.FREE]
        assert free.max_concurrent_calls == 8
        assert free.max_workers_per_swarm == 8
        assert free.max_cost_per_swarm == 2.00

    def test_enterprise_tier_scales_to_100(self):
        from src.orchestrator.swarm_throttler import TIER_CONFIGS, ThrottlerTier

        ent = TIER_CONFIGS[ThrottlerTier.ENTERPRISE]
        assert ent.max_concurrent_calls == 50
        assert ent.max_workers_per_swarm == 100
        assert ent.max_cost_per_swarm == 50.00

    def test_set_org_tier_changes_limits(self):
        from src.orchestrator.swarm_throttler import SwarmThrottler, ThrottlerTier

        throttler = SwarmThrottler()

        # Default: free-tier config
        cfg = throttler._get_config_for_org("org-1")
        assert cfg.max_workers_per_swarm == 8

        # Upgrade to enterprise
        throttler.set_org_tier("org-1", ThrottlerTier.ENTERPRISE)
        cfg = throttler._get_config_for_org("org-1")
        assert cfg.max_workers_per_swarm == 100

    def test_per_org_semaphore_created_on_tier_set(self):
        from src.orchestrator.swarm_throttler import SwarmThrottler, ThrottlerTier

        throttler = SwarmThrottler()
        assert "org-1" not in throttler._org_semaphores

        throttler.set_org_tier("org-1", ThrottlerTier.PRO)
        assert "org-1" in throttler._org_semaphores
        # PRO tier = 25 concurrent
        assert throttler._org_semaphores["org-1"]._value == 25

    def test_reserve_swarm_respects_tier_limit(self):
        from src.orchestrator.swarm_throttler import SwarmThrottler, ThrottlerTier

        throttler = SwarmThrottler()
        throttler.set_org_tier("org-1", ThrottlerTier.FREE)

        # FREE allows 3 concurrent swarms
        throttler.reserve_swarm("org-1", "swarm-1")
        throttler.reserve_swarm("org-1", "swarm-2")
        throttler.reserve_swarm("org-1", "swarm-3")

        with pytest.raises(RuntimeError, match="maximum of 3"):
            throttler.reserve_swarm("org-1", "swarm-4")

    def test_budget_check_respects_tier(self):
        from src.orchestrator.swarm_throttler import SwarmThrottler, ThrottlerTier

        throttler = SwarmThrottler()
        throttler.set_org_tier("org-1", ThrottlerTier.FREE)  # $2 per swarm

        throttler.reserve_swarm("org-1", "swarm-1")

        # Should pass under budget
        throttler.check_budget("org-1", "swarm-1", additional_cost=1.50)

        # Should fail over budget
        with pytest.raises(RuntimeError, match="exceed budget"):
            throttler.check_budget("org-1", "swarm-1", additional_cost=3.00)


# ===========================================================================
# 4. Discovery Swarm
# ===========================================================================


class TestDiscoverySwarm:
    """Test DISCOVERY_SWARM mode agents and target selection."""

    def test_discovery_swarm_mode_exists(self):
        from src.orchestrator.swarm_orchestrator import SwarmMode

        assert SwarmMode.DISCOVERY_SWARM == "discovery_swarm"

    def test_select_agent_for_target(self):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        assert SwarmOrchestrator._select_agent_for_target("page") == "ui_tester"
        assert SwarmOrchestrator._select_agent_for_target("form") == "ui_tester"
        assert SwarmOrchestrator._select_agent_for_target("api_endpoint") == "api_tester"
        assert SwarmOrchestrator._select_agent_for_target("endpoint") == "api_tester"
        assert SwarmOrchestrator._select_agent_for_target("unknown") == "ui_tester"  # default

    def test_build_discovery_targets_deduplicates_urls(self):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        @dataclass
        class FakePage:
            url: str

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list
            flows_discovered: list
            suggested_tests: list

        result = FakeDiscoveryResult(
            pages_discovered=[FakePage("https://a.com"), FakePage("https://b.com"), FakePage("https://a.com")],
            flows_discovered=[],
            suggested_tests=[],
        )

        targets = SwarmOrchestrator._build_discovery_targets(result, max_agents=10)
        urls = [url for _, url in targets]
        assert len(urls) == len(set(urls)), "Should deduplicate by URL"
        assert len(targets) == 2

    def test_build_discovery_targets_respects_max_cap(self):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        @dataclass
        class FakePage:
            url: str

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list
            flows_discovered: list
            suggested_tests: list

        pages = [FakePage(f"https://example.com/page{i}") for i in range(50)]
        result = FakeDiscoveryResult(pages_discovered=pages, flows_discovered=[], suggested_tests=[])

        targets = SwarmOrchestrator._build_discovery_targets(result, max_agents=5)
        assert len(targets) == 5, "Should cap at max_agents"

    def test_discovery_config_max_agents_field(self, discovery_config):
        assert discovery_config.max_discovery_agents == 5

    def test_launch_swarm_request_accepts_discovery_mode(self):
        """API model should accept discovery_swarm mode."""
        from src.api.swarms import LaunchSwarmRequest

        req = LaunchSwarmRequest(
            mode="discovery_swarm",
            project_id="proj-1",
            target_url="https://example.com",
            max_discovery_agents=20,
        )
        assert req.max_discovery_agents == 20

    def test_launch_swarm_request_caps_max_at_100(self):
        """max_discovery_agents should reject > 100."""
        from pydantic import ValidationError

        from src.api.swarms import LaunchSwarmRequest

        with pytest.raises(ValidationError):
            LaunchSwarmRequest(
                mode="discovery_swarm",
                project_id="proj-1",
                target_url="https://example.com",
                max_discovery_agents=200,
            )


# ===========================================================================
# 5. Intent Verification
# ===========================================================================


class TestIntentVerifier:
    """Test the lightweight intent verifier."""

    def test_extract_intent_full_crawl(self, swarm_config):
        from src.orchestrator.intent_verifier import extract_intent

        intent = extract_intent(swarm_config)
        assert "Comprehensive" in intent
        assert "example.com" in intent

    def test_extract_intent_pr_analysis(self, pr_config):
        from src.orchestrator.intent_verifier import extract_intent

        intent = extract_intent(pr_config)
        assert "2 changed files" in intent
        assert "PR #42" in intent

    def test_extract_intent_discovery(self, discovery_config):
        from src.orchestrator.intent_verifier import extract_intent

        intent = extract_intent(discovery_config)
        assert "Discover" in intent

    @pytest.mark.asyncio
    async def test_fast_path_skips_llm_when_all_succeed(self, swarm_config):
        """When all agents succeed with good confidence, should skip LLM call."""
        from src.orchestrator.intent_verifier import verify_intent
        from src.orchestrator.swarm_orchestrator import WorkerResult

        results = [
            WorkerResult("a1", "security_scanner", True, 100, 0.01, [{"x": 1}], "ok", 0.9),
            WorkerResult("a2", "ui_tester", True, 100, 0.01, [{"y": 1}], "ok", 0.8),
        ]

        # Should NOT call the model router at all (ModelRouter is lazy-imported
        # inside verify_intent, so we patch where it lives, not where it's used)
        with patch("src.core.model_router.ModelRouter") as MockRouter:
            verdict = await verify_intent(swarm_config, results, 0.02)
            MockRouter.assert_not_called()

        assert verdict.verdict == "complete"
        assert verdict.completion_score >= 0.8
        assert verdict.gaps == []

    @pytest.mark.asyncio
    async def test_heuristic_fallback_on_llm_failure(self, swarm_config, worker_results):
        """When LLM call fails, should fall back to heuristic."""
        from src.orchestrator.intent_verifier import verify_intent

        with patch("src.core.model_router.ModelRouter") as MockRouter:
            instance = MagicMock()
            instance.complete = AsyncMock(side_effect=RuntimeError("No API key"))
            MockRouter.return_value = instance

            verdict = await verify_intent(swarm_config, worker_results, 0.03)

        assert verdict.verdict in ("partial", "incomplete")
        # Should identify the failed agent
        gap_text = " ".join(verdict.gaps)
        assert "api_tester" in gap_text

    @pytest.mark.asyncio
    async def test_heuristic_verdict_all_failed(self):
        """All agents failed → incomplete."""
        from src.orchestrator.intent_verifier import _heuristic_verdict
        from src.orchestrator.swarm_orchestrator import WorkerResult

        results = [
            WorkerResult("a1", "sec", False, 100, 0, error="timeout"),
            WorkerResult("a2", "ui", False, 100, 0, error="crash"),
        ]

        verdict = _heuristic_verdict(results)
        assert verdict.verdict == "incomplete"
        assert verdict.completion_score == 0.0
        assert len(verdict.gaps) == 2

    @pytest.mark.asyncio
    async def test_heuristic_flags_empty_results(self):
        """Workers that returned empty results (confidence=0, no findings) should be flagged."""
        from src.orchestrator.intent_verifier import _heuristic_verdict
        from src.orchestrator.swarm_orchestrator import WorkerResult

        results = [
            WorkerResult("a1", "security_scanner", True, 100, 0.01, [{"x": 1}], "ok", 0.9),
            WorkerResult("a2", "visual_ai", True, 100, 0.0, [], "error", 0.0),  # error result
        ]

        verdict = _heuristic_verdict(results)
        gap_text = " ".join(verdict.gaps)
        assert "visual_ai" in gap_text
        assert "empty" in gap_text.lower() or "error" in gap_text.lower()

    @pytest.mark.asyncio
    async def test_intent_verdict_in_swarm_result(self, swarm_config):
        """SwarmResult should carry the intent verdict."""
        from src.orchestrator.intent_verifier import IntentVerdict
        from src.orchestrator.swarm_orchestrator import SwarmMode, SwarmResult

        verdict = IntentVerdict(
            completion_score=0.75,
            verdict="partial",
            gaps=["api_tester failed"],
            summary="3/4 goals met",
        )
        result = SwarmResult(
            swarm_id="test",
            mode=SwarmMode.FULL_CRAWL,
            success=True,
            total_duration_ms=5000,
            total_cost_usd=0.05,
            intent_verdict=verdict,
        )
        assert result.intent_verdict.verdict == "partial"
        assert result.intent_verdict.gaps == ["api_tester failed"]


# ===========================================================================
# 6. Swarm Lifecycle / Dispatch Tests
# ===========================================================================


class TestSwarmDispatch:
    """Test the _execute_agent dispatch routes to the correct runner."""

    @pytest.mark.asyncio
    async def test_dispatch_routes_all_13_agent_types(self, swarm_config, mock_emitter):
        """All 13 agent types should have a dispatch path (no unknown error)."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()

        # These should NOT return "Unknown agent type"
        known_agents = [
            "security_scanner", "accessibility_checker", "performance_analyzer",
            "auto_discovery", "code_analyzer", "mr_analyzer",
            "test_impact_analyzer", "smart_test_selector",
            "ui_tester", "api_tester", "self_healer",
            "flaky_detector", "visual_ai",
        ]

        # Mock all imports to avoid real agent initialization
        mock_patches = [
            "src.agents.SecurityScannerAgent",
            "src.agents.AccessibilityCheckerAgent",
            "src.agents.PerformanceAnalyzerAgent",
            "src.agents.AutoDiscovery",
            "src.agents.CodeAnalyzerAgent",
            "src.agents.MRAnalyzerAgent",
            "src.agents.UITesterAgent",
            "src.agents.APITesterAgent",
            "src.agents.SelfHealerAgent",
            "src.agents.FlakyTestDetector",
            "src.agents.VisualAI",
            "src.agents.TestImpactAnalyzer",
            "src.agents.SmartTestSelector",
        ]

        for agent_type in known_agents:
            # Use a config that has all the fields populated
            config = swarm_config
            if agent_type in ("mr_analyzer", "test_impact_analyzer", "smart_test_selector"):
                from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode

                config = SwarmConfig(
                    mode=SwarmMode.PR_ANALYSIS,
                    org_id="test-org",
                    project_id="test-project",
                    user_id="test-user",
                    changed_files=[{"path": "a.py", "status": "modified"}],
                )

            result = await orch._execute_agent(agent_type, config, mock_emitter, f"{agent_type}_1")
            assert "unrecognized agent type" not in result.get("summary", ""), (
                f"{agent_type} dispatched as unknown"
            )

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_error(self, swarm_config, mock_emitter):
        """A truly unknown agent type should get the fallback error result."""
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()
        result = await orch._execute_agent("nonexistent_agent", swarm_config, mock_emitter, "x_1")
        assert "unrecognized agent type" in result["summary"]
        assert result["confidence"] == 0.0


class TestSwarmConsensus:
    """Test consensus scoring across workers."""

    def test_consensus_all_high_confidence(self):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator, WorkerResult

        orch = SwarmOrchestrator()
        results = [
            WorkerResult("a1", "sec", True, 100, 0, confidence=0.9),
            WorkerResult("a2", "ui", True, 100, 0, confidence=0.8),
        ]
        score = orch._compute_consensus(results)
        assert 0.8 <= score <= 1.0

    def test_consensus_zero_when_all_failed(self):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator, WorkerResult

        orch = SwarmOrchestrator()
        results = [
            WorkerResult("a1", "sec", False, 100, 0, error="fail"),
            WorkerResult("a2", "ui", False, 100, 0, error="fail"),
        ]
        assert orch._compute_consensus(results) == 0.0

    def test_consensus_mixed_results(self, worker_results):
        from src.orchestrator.swarm_orchestrator import SwarmOrchestrator

        orch = SwarmOrchestrator()
        score = orch._compute_consensus(worker_results)
        assert 0.0 < score < 1.0


# ===========================================================================
# 7. QA Discovery Swarm
# ===========================================================================


class TestQADiscoverySwarm:
    """Test QA_DISCOVERY mode: enum, agent list, phase chaining, conversion."""

    def test_qa_discovery_mode_exists(self):
        """QA_DISCOVERY should be a valid SwarmMode enum value."""
        from src.orchestrator.swarm_orchestrator import SwarmMode

        assert SwarmMode.QA_DISCOVERY == "qa_discovery"

    def test_qa_discovery_in_swarm_mode_agents(self):
        """SWARM_MODE_AGENTS should list auto_discovery + testgen."""
        from src.orchestrator.swarm_orchestrator import SWARM_MODE_AGENTS, SwarmMode

        agents = SWARM_MODE_AGENTS[SwarmMode.QA_DISCOVERY]
        assert "auto_discovery" in agents
        assert "testgen" in agents
        assert len(agents) == 2

    def test_format_discovery_queue(self):
        """_format_discovery_queue should format pages and flows as text."""
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {"url": "https://example.com/", "title": "Home", "description": "Landing page", "forms": [], "type": "page"},
            {"name": "Login Flow", "description": "User logs in", "category": "auth", "steps": [{"action": "click", "target": "Login"}], "type": "flow"},
        ]
        text = _format_discovery_queue(queue, "https://example.com")
        assert "Home" in text
        assert "Landing page" in text
        assert "Login Flow" in text
        assert "auth" in text

    @pytest.mark.asyncio
    async def test_qa_discovery_uses_supervisor(self, mock_emitter):
        """QA discovery should delegate to the supervisor graph."""
        from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode, SwarmOrchestrator

        orch = SwarmOrchestrator()
        config = SwarmConfig(
            mode=SwarmMode.QA_DISCOVERY,
            org_id="test-org",
            project_id="test-project",
            user_id="test-user",
            target_url="https://example.com",
        )

        # Mock the supervisor graph to return a final state
        mock_final_state = {
            "task_complete": True,
            "current_phase": "complete",
            "iteration": 5,
            "total_cost": 0.05,
            "discovery_pages_found": 3,
            "discovery_batch_index": 1,
            "discovery_complete": True,
            "testable_surfaces": [
                {"url": "https://example.com/", "title": "Home", "type": "page"},
                {"url": "https://example.com/login", "title": "Login", "type": "page"},
            ],
            "generated_tests": [
                {"name": "test_home_loads", "framework": "playwright"},
                {"name": "test_login_flow", "framework": "playwright"},
            ],
            "generated_requirements": [{"title": "Home page loads"}],
            "results": {},
        }

        # Mock astream to yield the final state
        async def mock_astream(initial_state, config, stream_mode="values"):
            yield mock_final_state

        mock_graph = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.astream = mock_astream
        mock_graph.compile = MagicMock(return_value=mock_compiled)

        with (
            patch("src.orchestrator.supervisor.create_supervisor_graph", return_value=mock_graph),
            patch("src.orchestrator.supervisor.create_initial_supervisor_state", return_value={
                "messages": [], "discovery_complete": False, "discovery_queue": [],
                "generated_tests": [], "org_id": "test-org", "project_id": "test-project",
                "discovery_pages_found": 0, "discovery_batch_index": 0,
                "testable_surfaces": [], "results": {}, "total_cost": 0,
            }),
            patch("src.orchestrator.checkpointer.get_checkpointer", return_value=MagicMock()),
            patch(
                "src.orchestrator.swarm_orchestrator.SwarmOrchestrator._verify_intent",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await orch._run_qa_discovery_swarm(
                swarm_id="swarm_test",
                config=config,
                emitter=mock_emitter,
            )

        assert result.success is True
        assert len(result.worker_results) == 2
        assert result.worker_results[0].agent_type == "auto_discovery"
        assert result.worker_results[1].agent_type == "testgen"
        assert result.total_cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_qa_discovery_handles_supervisor_failure(self, mock_emitter):
        """If supervisor graph fails, should return error result."""
        from src.orchestrator.swarm_orchestrator import SwarmConfig, SwarmMode, SwarmOrchestrator

        orch = SwarmOrchestrator()
        config = SwarmConfig(
            mode=SwarmMode.QA_DISCOVERY,
            org_id="test-org",
            project_id="test-project",
            user_id="test-user",
            target_url="https://example.com",
        )

        with (
            patch(
                "src.orchestrator.supervisor.create_supervisor_graph",
                side_effect=RuntimeError("Graph compilation failed"),
            ),
            patch("src.orchestrator.supervisor.create_initial_supervisor_state", return_value={
                "messages": [], "discovery_complete": False, "discovery_queue": [],
                "generated_tests": [], "org_id": "test-org", "project_id": "test-project",
                "discovery_pages_found": 0, "discovery_batch_index": 0,
                "testable_surfaces": [], "results": {}, "total_cost": 0,
            }),
            patch(
                "src.orchestrator.swarm_orchestrator.SwarmOrchestrator._verify_intent",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await orch._run_qa_discovery_swarm(
                swarm_id="swarm_test",
                config=config,
                emitter=mock_emitter,
            )

        assert result.success is False
        assert "QA discovery failed" in result.summary

    def test_launch_swarm_request_accepts_qa_discovery_mode(self):
        """API model should accept qa_discovery mode."""
        from src.api.swarms import LaunchSwarmRequest

        req = LaunchSwarmRequest(
            mode="qa_discovery",
            project_id="proj-1",
            target_url="https://example.com",
        )
        assert req.mode == "qa_discovery"
