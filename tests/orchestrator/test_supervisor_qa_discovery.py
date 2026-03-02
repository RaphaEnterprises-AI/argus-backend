"""Tests for supervisor auto_discovery node and discovery-testgen interleaving."""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSupervisorAutoDiscoveryNode:
    """Test the supervisor_auto_discovery_node wrapper."""

    @pytest.mark.asyncio
    async def test_discovery_node_returns_batch(self):
        """Should discover pages and write them to discovery_queue."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        fake_result = FakeDiscoveryResult(
            pages_discovered=[
                FakePage(url="https://example.com/", title="Home"),
                FakePage(url="https://example.com/about", title="About"),
            ],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 0,
            "discovery_pages_found": 0,
            "testable_surfaces": [],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        assert len(result["discovery_queue"]) == 2
        assert result["discovery_pages_found"] == 2
        assert result["discovery_batch_index"] == 1
        # Found fewer than BATCH_SIZE(5) pages → discovery_complete
        assert result["discovery_complete"] is True

    @pytest.mark.asyncio
    async def test_discovery_node_filters_already_seen(self):
        """Should not add pages that are already in testable_surfaces."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        fake_result = FakeDiscoveryResult(
            pages_discovered=[
                FakePage(url="https://example.com/", title="Home"),
                FakePage(url="https://example.com/new", title="New Page"),
            ],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 1,
            "discovery_pages_found": 1,
            "testable_surfaces": [
                {"url": "https://example.com/", "title": "Home", "type": "page"},
            ],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        # Only the new page should be in queue
        assert len(result["discovery_queue"]) == 1
        assert result["discovery_queue"][0]["url"] == "https://example.com/new"

    @pytest.mark.asyncio
    async def test_discovery_node_max_batches(self):
        """Should set discovery_complete when max batches reached."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 10,  # Already at max
            "discovery_pages_found": 50,
            "testable_surfaces": [],
            "results": {},
        }

        result = await supervisor_auto_discovery_node(state)
        assert result["discovery_complete"] is True

    @pytest.mark.asyncio
    async def test_discovery_node_no_url(self):
        """Should handle missing app_url gracefully."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        state = {"app_url": None, "results": {}}
        result = await supervisor_auto_discovery_node(state)
        assert result["discovery_complete"] is True

    @pytest.mark.asyncio
    async def test_discovery_node_not_complete_when_full_batch(self):
        """Should NOT set discovery_complete when batch is full (more pages likely)."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        # Return exactly BATCH_SIZE (5) pages → not done yet
        fake_result = FakeDiscoveryResult(
            pages_discovered=[
                FakePage(url=f"https://example.com/page{i}", title=f"Page {i}")
                for i in range(5)
            ],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 0,
            "discovery_pages_found": 0,
            "testable_surfaces": [],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        # Full batch → more pages might exist → not complete
        assert result["discovery_complete"] is False
        assert result["discovery_pages_found"] == 5
        assert result["current_phase"] == "test_generation"

    @pytest.mark.asyncio
    async def test_discovery_node_includes_flows(self):
        """Should include discovered flows in the queue."""
        from src.orchestrator.supervisor import supervisor_auto_discovery_node

        @dataclass
        class FakePage:
            url: str
            title: str = ""
            description: str = ""
            forms: list = field(default_factory=list)

        @dataclass
        class FakeFlow:
            name: str = "Login"
            description: str = "Login flow"
            category: str = "auth"
            priority: str = "high"
            steps: list = field(default_factory=list)

        @dataclass
        class FakeDiscoveryResult:
            pages_discovered: list = field(default_factory=list)
            flows_discovered: list = field(default_factory=list)
            suggested_tests: list = field(default_factory=list)

        fake_result = FakeDiscoveryResult(
            pages_discovered=[FakePage(url="https://example.com/", title="Home")],
            flows_discovered=[FakeFlow()],
        )

        mock_discovery = AsyncMock()
        mock_discovery.discover = AsyncMock(return_value=fake_result)

        state = {
            "app_url": "https://example.com",
            "discovery_batch_index": 0,
            "discovery_pages_found": 0,
            "testable_surfaces": [],
            "results": {},
        }

        with patch("src.agents.AutoDiscovery", return_value=mock_discovery):
            result = await supervisor_auto_discovery_node(state)

        # 1 page + 1 flow in queue
        assert len(result["discovery_queue"]) == 2
        flow_items = [q for q in result["discovery_queue"] if q["type"] == "flow"]
        assert len(flow_items) == 1
        assert flow_items[0]["name"] == "Login"


class TestFormatDiscoveryQueue:
    """Test the _format_discovery_queue helper."""

    def test_formats_pages_and_flows(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {"url": "https://a.com/", "title": "Home", "description": "Main page", "forms": [], "type": "page"},
            {"name": "Checkout", "description": "Buy stuff", "category": "ecommerce", "steps": [], "type": "flow"},
        ]
        text = _format_discovery_queue(queue, "https://a.com")
        assert "Home" in text
        assert "Checkout" in text
        assert "ecommerce" in text

    def test_empty_queue(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        text = _format_discovery_queue([], "https://a.com")
        assert "https://a.com" in text

    def test_formats_form_info(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {
                "url": "https://a.com/login",
                "title": "Login",
                "description": "",
                "forms": [{"action": "/auth/login", "method": "POST"}, "search-form"],
                "type": "page",
            },
        ]
        text = _format_discovery_queue(queue, "https://a.com")
        assert "/auth/login" in text
        assert "search-form" in text

    def test_formats_flow_steps(self):
        from src.orchestrator.supervisor import _format_discovery_queue

        queue = [
            {
                "name": "Signup",
                "description": "New user signup",
                "category": "auth",
                "steps": [
                    {"action": "click", "target": "Sign Up"},
                    {"action": "fill", "target": "email input"},
                    "Submit form",
                ],
                "type": "flow",
            },
        ]
        text = _format_discovery_queue(queue, "https://a.com")
        assert "click" in text
        assert "Sign Up" in text
        assert "Submit form" in text


class TestSupervisorDiscoveryDecisionRules:
    """Test that the supervisor prompt includes discovery rules."""

    def test_prompt_has_auto_discovery(self):
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()
        assert "auto_discovery" in prompt
        assert "discovery_complete" in prompt

    def test_auto_discovery_in_agents_list(self):
        from src.orchestrator.supervisor import AGENTS, AGENT_DESCRIPTIONS

        assert "auto_discovery" in AGENTS
        assert "auto_discovery" in AGENT_DESCRIPTIONS

    def test_discovery_phase_exists(self):
        from src.orchestrator.supervisor import PHASE_DESCRIPTIONS

        assert "discovery" in PHASE_DESCRIPTIONS

    def test_prompt_has_interleaving_rule(self):
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()
        assert "alternate between auto_discovery" in prompt
        assert "discovery_queue" in prompt
