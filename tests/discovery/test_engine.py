"""Tests for the Discovery Engine module.

This module tests the main DiscoveryEngine class which orchestrates
the autonomous discovery process including crawling, element extraction,
flow inference, and result persistence.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discovery.engine import (
    DiscoveryEngine,
    DiscoveryError,
    create_discovery_engine,
)
from src.discovery.models import (
    CrawlError,
    CrawlResult,
    DiscoveredElement,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoveryConfig,
    DiscoveryMode,
    DiscoverySession,
    DiscoveryStatus,
    ElementCategory,
    ExplorationStrategy,
    FlowCategory,
    FlowStep,
    PageCategory,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_repository():
    """Create a mock DiscoveryRepository."""
    repo = AsyncMock()
    repo.save_session = AsyncMock()
    repo.get_session = AsyncMock(return_value=None)
    repo.save_pages = AsyncMock()
    repo.get_pages = AsyncMock(return_value=[])
    repo.save_flows = AsyncMock()
    repo.get_flows = AsyncMock(return_value=[])
    repo.list_sessions = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_crawlee_bridge():
    """Create a mock CrawleeBridge."""
    bridge = MagicMock()
    bridge.set_progress_callback = MagicMock()
    bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))
    return bridge


@pytest.fixture
def mock_claude_client():
    """Create a mock Anthropic Claude client."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text='{"flows": []}')]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def sample_discovery_config():
    """Create a sample discovery configuration."""
    return DiscoveryConfig(
        mode=DiscoveryMode.standard_crawl,
        strategy=ExplorationStrategy.breadth_first,
        max_pages=10,
        max_depth=2,
        capture_screenshots=False,
    )


@pytest.fixture
def sample_discovered_page():
    """Create a sample discovered page."""
    return DiscoveredPage(
        id=str(uuid.uuid4()),
        url="https://example.com/login",
        title="Login Page",
        category=PageCategory.auth_login,
        elements=[
            DiscoveredElement(
                id=str(uuid.uuid4()),
                page_url="https://example.com/login",
                selector="#username",
                category=ElementCategory.form,
                label="Username",
                tag_name="input",
                html_attributes={"type": "text"},
            ),
            DiscoveredElement(
                id=str(uuid.uuid4()),
                page_url="https://example.com/login",
                selector="#password",
                category=ElementCategory.authentication,
                label="Password",
                tag_name="input",
                html_attributes={"type": "password"},
            ),
        ],
        outgoing_links={"/dashboard", "/signup"},
    )


@pytest.fixture
def sample_crawl_result(sample_discovered_page):
    """Create a sample crawl result."""
    return CrawlResult(
        pages={"https://example.com/login": sample_discovered_page},
        duration_ms=1000,
        errors=[],
    )


# ==============================================================================
# Test DiscoveryEngine Initialization
# ==============================================================================


class TestDiscoveryEngineInit:
    """Tests for DiscoveryEngine initialization."""

    def test_init_with_no_client(self, mock_repository):
        """Test initialization without Supabase client."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            assert engine.supabase is None
            assert engine.current_session is None

    def test_init_with_supabase_client(self, mock_repository):
        """Test initialization with Supabase client."""
        mock_supabase = MagicMock()
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(
                supabase_client=mock_supabase, repository=mock_repository
            )
            assert engine.supabase == mock_supabase

    def test_init_with_anthropic_api_key(self, mock_repository):
        """Test initialization with Anthropic API key."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key
            with patch("src.discovery.engine.anthropic.Anthropic") as mock_anthropic:
                engine = DiscoveryEngine(repository=mock_repository)
                mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_init_with_plain_string_api_key(self, mock_repository):
        """Test initialization with plain string API key (no SecretStr)."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "sk-ant-test-key-plain"
            with patch("src.discovery.engine.anthropic.Anthropic") as mock_anthropic:
                engine = DiscoveryEngine(repository=mock_repository)
                mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key-plain")


# ==============================================================================
# Test Start Discovery
# ==============================================================================


class TestStartDiscovery:
    """Tests for the start_discovery method."""

    @pytest.mark.asyncio
    async def test_start_discovery_creates_session(self, mock_repository):
        """Test that start_discovery creates a new session."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = AsyncMock()
            engine.bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))

            session = await engine.start_discovery(
                project_id="test-project",
                app_url="https://example.com",
            )

            assert session.project_id == "test-project"
            assert session.status == DiscoveryStatus.running
            mock_repository.save_session.assert_called()

    @pytest.mark.asyncio
    async def test_start_discovery_with_custom_config(self, mock_repository):
        """Test start_discovery with custom configuration."""
        config = DiscoveryConfig(
            mode=DiscoveryMode.deep,
            max_pages=50,
            max_depth=5,
        )
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = AsyncMock()
            engine.bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))

            session = await engine.start_discovery(
                project_id="test-project",
                app_url="https://example.com",
                config=config,
            )

            assert session.mode == DiscoveryMode.deep
            assert session.config.max_pages == 50

    @pytest.mark.asyncio
    async def test_start_discovery_emits_start_event(self, mock_repository):
        """Test that start_discovery emits a start event."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = AsyncMock()
            engine.bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))

            session = await engine.start_discovery(
                project_id="test-project",
                app_url="https://example.com",
            )

            # Verify event subscribers dict is set up
            assert session.id in engine._cancellation_flags


# ==============================================================================
# Test Run Discovery
# ==============================================================================


class TestRunDiscovery:
    """Tests for the run_discovery method."""

    @pytest.mark.asyncio
    async def test_run_discovery_saves_pages(
        self, mock_repository, sample_crawl_result
    ):
        """Test that discovered pages are saved."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = MagicMock()
            engine.bridge.set_progress_callback = MagicMock()
            engine.bridge.run_crawl = AsyncMock(return_value=sample_crawl_result)

            session = DiscoverySession(
                id=str(uuid.uuid4()),
                project_id="test-project",
                status=DiscoveryStatus.running,
            )
            engine.current_session = session

            config = DiscoveryConfig()
            result = await engine.run_discovery(
                session, "https://example.com", config
            )

            assert result.total_pages == 1
            mock_repository.save_pages.assert_called()

    @pytest.mark.asyncio
    async def test_run_discovery_handles_cancellation(self, mock_repository):
        """Test that cancellation is handled properly."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = MagicMock()
            engine.bridge.set_progress_callback = MagicMock()
            engine.bridge.run_crawl = AsyncMock(return_value=CrawlResult(pages={}))

            session = DiscoverySession(
                id=str(uuid.uuid4()),
                project_id="test-project",
                status=DiscoveryStatus.running,
            )
            engine._cancellation_flags[session.id] = True

            config = DiscoveryConfig()
            result = await engine.run_discovery(
                session, "https://example.com", config
            )

            assert session.status == DiscoveryStatus.cancelled

    @pytest.mark.asyncio
    async def test_run_discovery_infers_flows(
        self, mock_repository, sample_crawl_result
    ):
        """Test that flows are inferred from discovered pages."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine.bridge = MagicMock()
            engine.bridge.set_progress_callback = MagicMock()
            engine.bridge.run_crawl = AsyncMock(return_value=sample_crawl_result)

            session = DiscoverySession(
                id=str(uuid.uuid4()),
                project_id="test-project",
                status=DiscoveryStatus.running,
            )
            engine.current_session = session

            config = DiscoveryConfig()
            await engine.run_discovery(session, "https://example.com", config)

            # Flows should be saved
            mock_repository.save_flows.assert_called()


# ==============================================================================
# Test Flow Inference
# ==============================================================================


class TestFlowInference:
    """Tests for flow inference methods."""

    def test_infer_flows_heuristic_finds_login(self, mock_repository):
        """Test heuristic flow inference finds login flows."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            pages = [
                DiscoveredPage(
                    id=str(uuid.uuid4()),
                    url="https://example.com/login",
                    title="Login",
                    category=PageCategory.auth_login,
                )
            ]

            flows = engine._infer_flows_heuristic(pages)
            assert len(flows) > 0
            assert flows[0].category == FlowCategory.authentication

    def test_infer_flows_heuristic_finds_signup(self, mock_repository):
        """Test heuristic flow inference finds signup flows."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            pages = [
                DiscoveredPage(
                    id=str(uuid.uuid4()),
                    url="https://example.com/signup",
                    title="Sign Up",
                    category=PageCategory.auth_signup,
                )
            ]

            flows = engine._infer_flows_heuristic(pages)
            # Should find registration flow
            registration_flows = [f for f in flows if f.category == FlowCategory.registration]
            assert len(registration_flows) > 0

    def test_infer_flows_heuristic_finds_forms(self, mock_repository):
        """Test heuristic flow inference finds form flows."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            pages = [
                DiscoveredPage(
                    id=str(uuid.uuid4()),
                    url="https://example.com/contact",
                    title="Contact Us",
                    category=PageCategory.form,
                )
            ]

            flows = engine._infer_flows_heuristic(pages)
            form_flows = [f for f in flows if f.category == FlowCategory.crud]
            assert len(form_flows) > 0

    def test_infer_flows_heuristic_finds_navigation(self, mock_repository):
        """Test heuristic flow inference finds navigation flows."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            pages = [
                DiscoveredPage(
                    id=str(uuid.uuid4()),
                    url="https://example.com/",
                    title="Home",
                    category=PageCategory.landing,
                )
            ]

            flows = engine._infer_flows_heuristic(pages)
            nav_flows = [f for f in flows if f.category == FlowCategory.navigation]
            assert len(nav_flows) > 0

    def test_infer_flows_returns_empty_for_no_pages(self, mock_repository):
        """Test that no flows are returned for empty pages list."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            flows = engine._infer_flows_heuristic([])
            assert flows == []


# ==============================================================================
# Test JSON Response Parsing
# ==============================================================================


class TestJsonParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_response_direct(self, mock_repository):
        """Test parsing clean JSON."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = '{"flows": []}'
            result = engine._parse_json_response(content)
            assert result == {"flows": []}

    def test_parse_json_response_markdown_block(self, mock_repository):
        """Test parsing JSON from markdown code block."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = '```json\n{"flows": []}\n```'
            result = engine._parse_json_response(content)
            assert result == {"flows": []}

    def test_parse_json_response_generic_code_block(self, mock_repository):
        """Test parsing JSON from generic code block."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = '```\n{"flows": []}\n```'
            result = engine._parse_json_response(content)
            assert result == {"flows": []}

    def test_parse_json_response_trailing_comma(self, mock_repository):
        """Test parsing JSON with trailing commas."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = '{"flows": [1, 2,]}'
            result = engine._parse_json_response(content)
            assert result == {"flows": [1, 2]}

    def test_parse_json_response_extracts_object(self, mock_repository):
        """Test extracting JSON object from text."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = 'Here is the result: {"flows": []} More text.'
            result = engine._parse_json_response(content)
            assert result == {"flows": []}

    def test_parse_json_response_returns_empty_on_failure(self, mock_repository):
        """Test that invalid JSON returns empty dict."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            content = "This is not JSON at all"
            result = engine._parse_json_response(content)
            assert result == {}


# ==============================================================================
# Test Session Management
# ==============================================================================


class TestSessionManagement:
    """Tests for session management methods."""

    @pytest.mark.asyncio
    async def test_get_session_status(self, mock_repository):
        """Test getting session status."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.running,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.get_session_status("test-session")

            assert result == session
            mock_repository.get_session.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_pause_discovery(self, mock_repository):
        """Test pausing a discovery session."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.running,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.pause_discovery("test-session")

            assert result is True
            assert engine._cancellation_flags["test-session"] is True

    @pytest.mark.asyncio
    async def test_pause_discovery_not_found(self, mock_repository):
        """Test pausing non-existent session raises error."""
        mock_repository.get_session.return_value = None

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            with pytest.raises(DiscoveryError) as exc_info:
                await engine.pause_discovery("non-existent")
            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pause_discovery_wrong_status(self, mock_repository):
        """Test pausing session with wrong status raises error."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.completed,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            with pytest.raises(DiscoveryError) as exc_info:
                await engine.pause_discovery("test-session")
            assert "not running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resume_discovery(self, mock_repository):
        """Test resuming a paused discovery session."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.paused,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            engine._cancellation_flags["test-session"] = True

            result = await engine.resume_discovery("test-session")

            assert result.status == DiscoveryStatus.running
            assert engine._cancellation_flags["test-session"] is False

    @pytest.mark.asyncio
    async def test_resume_discovery_wrong_status(self, mock_repository):
        """Test resuming session with wrong status raises error."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.running,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            with pytest.raises(DiscoveryError) as exc_info:
                await engine.resume_discovery("test-session")
            assert "not paused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_discovery(self, mock_repository):
        """Test cancelling a discovery session."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.running,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.cancel_discovery("test-session")

            assert result is True
            assert engine._cancellation_flags["test-session"] is True

    @pytest.mark.asyncio
    async def test_cancel_discovery_wrong_status(self, mock_repository):
        """Test cancelling session with wrong status raises error."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.completed,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            with pytest.raises(DiscoveryError) as exc_info:
                await engine.cancel_discovery("test-session")
            assert "cannot be cancelled" in str(exc_info.value)


# ==============================================================================
# Test Event Streaming
# ==============================================================================


class TestEventStreaming:
    """Tests for SSE event streaming."""

    @pytest.mark.asyncio
    async def test_discovery_events_not_found(self, mock_repository):
        """Test streaming events for non-existent session."""
        mock_repository.get_session.return_value = None

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            events = []
            async for event in engine.discovery_events("non-existent"):
                events.append(event)

            assert len(events) == 1
            assert events[0]["event"] == "error"

    @pytest.mark.asyncio
    async def test_discovery_events_initial_status(self, mock_repository):
        """Test that initial status is emitted."""
        session = DiscoverySession(
            id="test-session",
            project_id="test-project",
            status=DiscoveryStatus.completed,
            progress_percentage=100.0,
            pages_found=5,
            flows_found=2,
        )
        mock_repository.get_session.return_value = session

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            events = []
            async for event in engine.discovery_events("test-session"):
                events.append(event)
                if event.get("event") == "status":
                    break

            assert len(events) >= 1
            status_events = [e for e in events if e.get("event") == "status"]
            assert len(status_events) > 0

    @pytest.mark.asyncio
    async def test_emit_event(self, mock_repository):
        """Test emitting events to subscribers."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)

            # Create a subscriber queue
            queue = asyncio.Queue(maxsize=100)
            engine._event_subscribers["test-session"] = [queue]

            await engine._emit_event(
                "test-session",
                "test_event",
                {"message": "Hello"},
            )

            event = await queue.get()
            assert event["event"] == "test_event"
            assert event["data"]["message"] == "Hello"


# ==============================================================================
# Test List Sessions
# ==============================================================================


class TestListSessions:
    """Tests for listing sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions_no_filter(self, mock_repository):
        """Test listing all sessions."""
        sessions = [
            DiscoverySession(id="s1", project_id="p1"),
            DiscoverySession(id="s2", project_id="p2"),
        ]
        mock_repository.list_sessions.return_value = sessions

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.list_sessions()

            assert len(result) == 2
            mock_repository.list_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_sessions_with_project_filter(self, mock_repository):
        """Test listing sessions filtered by project."""
        sessions = [DiscoverySession(id="s1", project_id="p1")]
        mock_repository.list_sessions.return_value = sessions

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.list_sessions(project_id="p1")

            assert len(result) == 1
            mock_repository.list_sessions.assert_called_with(
                project_id="p1", status=None, limit=50
            )

    @pytest.mark.asyncio
    async def test_list_sessions_with_status_filter(self, mock_repository):
        """Test listing sessions filtered by status."""
        sessions = []
        mock_repository.list_sessions.return_value = sessions

        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = DiscoveryEngine(repository=mock_repository)
            result = await engine.list_sessions(status=DiscoveryStatus.running)

            mock_repository.list_sessions.assert_called_with(
                project_id=None, status="running", limit=50
            )


# ==============================================================================
# Test Factory Function
# ==============================================================================


class TestFactoryFunction:
    """Tests for the create_discovery_engine factory function."""

    def test_create_discovery_engine_no_client(self):
        """Test creating engine without Supabase client."""
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = create_discovery_engine()
            assert isinstance(engine, DiscoveryEngine)

    def test_create_discovery_engine_with_client(self):
        """Test creating engine with Supabase client."""
        mock_supabase = MagicMock()
        with patch("src.discovery.engine.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            engine = create_discovery_engine(supabase_client=mock_supabase)
            assert engine.supabase == mock_supabase
