"""Tests for Discovery API endpoints."""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException


class TestDiscoveryModels:
    """Tests for discovery request/response models."""

    def test_start_discovery_request_defaults(self, mock_env_vars):
        """Test StartDiscoveryRequest default values."""
        from src.api.discovery import StartDiscoveryRequest

        request = StartDiscoveryRequest(
            project_id="test-project",
            app_url="https://example.com",
        )

        assert request.mode == "standard_crawl"
        assert request.strategy == "breadth_first"
        assert request.max_pages == 50
        assert request.max_depth == 3
        assert request.capture_screenshots is True
        assert request.use_vision_ai is True
        assert request.timeout_seconds == 30

    def test_start_discovery_request_custom_values(self, mock_env_vars):
        """Test StartDiscoveryRequest with custom values."""
        from src.api.discovery import StartDiscoveryRequest, AuthConfig

        auth_config = AuthConfig(
            type="bearer",
            credentials={"token": "test-token"},
            login_url="https://example.com/login",
        )

        request = StartDiscoveryRequest(
            project_id="test-project",
            app_url="https://example.com",
            mode="deep_analysis",
            strategy="smart_adaptive",
            max_pages=100,
            max_depth=5,
            include_patterns=["/dashboard/*"],
            exclude_patterns=["/admin/*"],
            focus_areas=["authentication", "checkout"],
            capture_screenshots=False,
            use_vision_ai=False,
            auth_config=auth_config,
            custom_headers={"X-API-Key": "test-key"},
            timeout_seconds=60,
        )

        assert request.mode == "deep_analysis"
        assert request.strategy == "smart_adaptive"
        assert request.max_pages == 100
        assert request.max_depth == 5
        assert request.auth_config is not None
        assert request.auth_config.type == "bearer"

    def test_discovery_session_response(self, mock_env_vars):
        """Test DiscoverySessionResponse model."""
        from src.api.discovery import DiscoverySessionResponse

        response = DiscoverySessionResponse(
            id="session-123",
            project_id="project-456",
            status="running",
            progress_percentage=50.0,
            pages_found=10,
            flows_found=5,
            elements_found=100,
            forms_found=3,
            errors_count=0,
            started_at="2024-01-01T00:00:00Z",
            app_url="https://example.com",
            mode="standard_crawl",
            strategy="breadth_first",
            max_pages=50,
            max_depth=3,
        )

        assert response.id == "session-123"
        assert response.progress_percentage == 50.0

    def test_update_flow_request(self, mock_env_vars):
        """Test UpdateFlowRequest model."""
        from src.api.discovery import UpdateFlowRequest

        request = UpdateFlowRequest(
            name="Updated Flow",
            description="Updated description",
            priority="high",
            steps=[{"action": "click", "target": "#button"}],
            category="checkout",
        )

        assert request.name == "Updated Flow"
        assert request.priority == "high"

    def test_generate_test_request(self, mock_env_vars):
        """Test GenerateTestRequest model."""
        from src.api.discovery import GenerateTestRequest

        request = GenerateTestRequest(
            framework="playwright",
            language="typescript",
            include_assertions=True,
            include_screenshots=True,
            parameterize=True,
        )

        assert request.framework == "playwright"
        assert request.parameterize is True


class TestDiscoveryEnums:
    """Tests for discovery enums."""

    def test_discovery_mode_enum(self, mock_env_vars):
        """Test DiscoveryMode enum values."""
        from src.api.discovery import DiscoveryMode

        assert DiscoveryMode.STANDARD_CRAWL.value == "standard_crawl"
        assert DiscoveryMode.QUICK_SCAN.value == "quick_scan"
        assert DiscoveryMode.DEEP_ANALYSIS.value == "deep_analysis"
        assert DiscoveryMode.AUTHENTICATED.value == "authenticated"
        assert DiscoveryMode.API_FIRST.value == "api_first"

    def test_discovery_strategy_enum(self, mock_env_vars):
        """Test DiscoveryStrategy enum values."""
        from src.api.discovery import DiscoveryStrategy

        assert DiscoveryStrategy.BREADTH_FIRST.value == "breadth_first"
        assert DiscoveryStrategy.DEPTH_FIRST.value == "depth_first"
        assert DiscoveryStrategy.PRIORITY_BASED.value == "priority_based"
        assert DiscoveryStrategy.SMART_ADAPTIVE.value == "smart_adaptive"

    def test_session_status_enum(self, mock_env_vars):
        """Test SessionStatus enum values."""
        from src.api.discovery import SessionStatus

        assert SessionStatus.PENDING.value == "pending"
        assert SessionStatus.RUNNING.value == "running"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.CANCELLED.value == "cancelled"
        assert SessionStatus.FAILED.value == "failed"


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.asyncio
    async def test_get_session_or_404_found(self, mock_env_vars):
        """Test get_session_or_404 when session exists."""
        from src.api.discovery import get_session_or_404, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {"id": session_id, "status": "running"}

        try:
            session = await get_session_or_404(session_id)
            assert session["id"] == session_id
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_session_or_404_not_found(self, mock_env_vars):
        """Test get_session_or_404 when session doesn't exist."""
        from src.api.discovery import get_session_or_404, _discovery_sessions

        _discovery_sessions.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_session_or_404("nonexistent")

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_flow_or_404_found(self, mock_env_vars):
        """Test get_flow_or_404 when flow exists."""
        from src.api.discovery import get_flow_or_404, _discovered_flows

        flow_id = "test-flow"
        _discovered_flows[flow_id] = {"id": flow_id, "name": "Test Flow"}

        try:
            flow = await get_flow_or_404(flow_id)
            assert flow["id"] == flow_id
        finally:
            _discovered_flows.clear()

    @pytest.mark.asyncio
    async def test_get_flow_or_404_not_found(self, mock_env_vars):
        """Test get_flow_or_404 when flow doesn't exist."""
        from src.api.discovery import get_flow_or_404, _discovered_flows

        _discovered_flows.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_flow_or_404("nonexistent")

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()

    def test_build_session_response(self, mock_env_vars):
        """Test build_session_response helper."""
        from src.api.discovery import build_session_response

        session = {
            "id": "session-123",
            "project_id": "project-456",
            "status": "running",
            "started_at": "2024-01-01T00:00:00Z",
            "app_url": "https://example.com",
            "config": {
                "mode": "standard_crawl",
                "strategy": "breadth_first",
                "max_pages": 50,
                "max_depth": 3,
            },
            "pages": [{"elements_count": 10, "forms_count": 2}],
            "flows": [{"name": "Flow 1"}],
            "errors": [],
        }

        response = build_session_response(session)

        assert response.id == "session-123"
        assert response.pages_found == 1
        assert response.flows_found == 1
        assert response.elements_found == 10
        assert response.forms_found == 2
        assert response.progress_percentage == 2.0  # 1/50 * 100


class TestDiscoverySessionEndpoints:
    """Tests for discovery session endpoints."""

    @pytest.mark.asyncio
    async def test_list_discovery_sessions_empty(self, mock_env_vars):
        """Test listing sessions when none exist."""
        from src.api.discovery import list_discovery_sessions, _discovery_sessions

        _discovery_sessions.clear()

        # Mock the repository
        with patch("src.api.discovery.get_discovery_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_instance.list_sessions = AsyncMock(return_value=[])
            mock_repo.return_value = mock_instance

            response = await list_discovery_sessions()

            assert response.total == 0
            assert len(response.sessions) == 0

    @pytest.mark.asyncio
    async def test_list_discovery_sessions_with_data(self, mock_env_vars):
        """Test listing sessions with data."""
        from src.api.discovery import list_discovery_sessions, _discovery_sessions

        _discovery_sessions.clear()
        _discovery_sessions["session-1"] = {
            "id": "session-1",
            "project_id": "project-1",
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "app_url": "https://example.com",
            "config": {"mode": "standard_crawl", "strategy": "breadth_first", "max_pages": 50, "max_depth": 3},
            "pages": [],
            "flows": [],
            "errors": [],
        }

        try:
            # Mock the repository to return empty from DB
            with patch("src.api.discovery.get_discovery_repository") as mock_repo:
                mock_instance = MagicMock()
                mock_instance.list_sessions = AsyncMock(return_value=[])
                mock_repo.return_value = mock_instance

                response = await list_discovery_sessions()

                assert response.total == 1
                assert len(response.sessions) == 1
                assert response.sessions[0].id == "session-1"
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_list_discovery_sessions_filter_by_project(self, mock_env_vars):
        """Test filtering sessions by project_id."""
        from src.api.discovery import list_discovery_sessions, _discovery_sessions

        _discovery_sessions.clear()
        _discovery_sessions["session-1"] = {
            "id": "session-1",
            "project_id": "project-1",
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "app_url": "https://example.com",
            "config": {"mode": "standard_crawl", "strategy": "breadth_first", "max_pages": 50, "max_depth": 3},
            "pages": [],
            "flows": [],
            "errors": [],
        }
        _discovery_sessions["session-2"] = {
            "id": "session-2",
            "project_id": "project-2",
            "status": "completed",
            "started_at": "2024-01-02T00:00:00Z",
            "app_url": "https://other.com",
            "config": {"mode": "standard_crawl", "strategy": "breadth_first", "max_pages": 50, "max_depth": 3},
            "pages": [],
            "flows": [],
            "errors": [],
        }

        try:
            with patch("src.api.discovery.get_discovery_repository") as mock_repo:
                mock_instance = MagicMock()
                mock_instance.list_sessions = AsyncMock(return_value=[])
                mock_repo.return_value = mock_instance

                response = await list_discovery_sessions(project_id="project-1")

                assert response.total == 1
                assert response.sessions[0].id == "session-1"
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_start_discovery_invalid_mode(self, mock_env_vars):
        """Test starting discovery with invalid mode."""
        from src.api.discovery import start_discovery, StartDiscoveryRequest

        request = StartDiscoveryRequest(
            project_id="test-project",
            app_url="https://example.com",
            mode="invalid_mode",
        )

        mock_background = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await start_discovery(request, mock_background)

        assert exc_info.value.status_code == 400
        assert "Invalid mode" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_start_discovery_invalid_strategy(self, mock_env_vars):
        """Test starting discovery with invalid strategy."""
        from src.api.discovery import start_discovery, StartDiscoveryRequest

        request = StartDiscoveryRequest(
            project_id="test-project",
            app_url="https://example.com",
            strategy="invalid_strategy",
        )

        mock_background = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await start_discovery(request, mock_background)

        assert exc_info.value.status_code == 400
        assert "Invalid strategy" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_start_discovery_success(self, mock_env_vars):
        """Test successfully starting a discovery session."""
        from src.api.discovery import start_discovery, StartDiscoveryRequest, _discovery_sessions

        _discovery_sessions.clear()

        request = StartDiscoveryRequest(
            project_id="test-project",
            app_url="https://example.com",
        )

        mock_background = MagicMock()

        try:
            response = await start_discovery(request, mock_background)

            assert response.project_id == "test-project"
            assert response.app_url == "https://example.com"
            assert response.status == "pending"
            assert response.id in _discovery_sessions
            mock_background.add_task.assert_called_once()
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_session(self, mock_env_vars):
        """Test getting a specific session."""
        from src.api.discovery import get_session, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "project_id": "project-1",
            "status": "running",
            "started_at": "2024-01-01T00:00:00Z",
            "app_url": "https://example.com",
            "config": {"mode": "standard_crawl", "strategy": "breadth_first", "max_pages": 50, "max_depth": 3},
            "pages": [],
            "flows": [],
            "errors": [],
        }

        try:
            response = await get_session(session_id)
            assert response.id == session_id
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_env_vars):
        """Test getting a non-existent session."""
        from src.api.discovery import get_session, _discovery_sessions

        _discovery_sessions.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_session("nonexistent")

        assert exc_info.value.status_code == 404


class TestDiscoveryControlEndpoints:
    """Tests for discovery control endpoints (pause, resume, cancel)."""

    @pytest.mark.asyncio
    async def test_pause_discovery_success(self, mock_env_vars):
        """Test pausing a running discovery session."""
        from src.api.discovery import pause_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.RUNNING.value,
            "events_queue": asyncio.Queue(),
        }

        try:
            response = await pause_discovery(session_id)

            assert response["success"] is True
            assert _discovery_sessions[session_id]["status"] == SessionStatus.PAUSED.value
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_pause_discovery_invalid_status(self, mock_env_vars):
        """Test pausing a session that's not running."""
        from src.api.discovery import pause_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.COMPLETED.value,
        }

        try:
            with pytest.raises(HTTPException) as exc_info:
                await pause_discovery(session_id)

            assert exc_info.value.status_code == 400
            assert "Cannot pause" in str(exc_info.value.detail)
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_resume_discovery_success(self, mock_env_vars):
        """Test resuming a paused discovery session."""
        from src.api.discovery import resume_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.PAUSED.value,
            "events_queue": asyncio.Queue(),
        }

        mock_background = MagicMock()

        try:
            response = await resume_discovery(session_id, mock_background)

            assert response["success"] is True
            assert _discovery_sessions[session_id]["status"] == SessionStatus.RUNNING.value
            mock_background.add_task.assert_called_once()
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_resume_discovery_invalid_status(self, mock_env_vars):
        """Test resuming a session that's not paused."""
        from src.api.discovery import resume_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.RUNNING.value,
        }

        mock_background = MagicMock()

        try:
            with pytest.raises(HTTPException) as exc_info:
                await resume_discovery(session_id, mock_background)

            assert exc_info.value.status_code == 400
            assert "Cannot resume" in str(exc_info.value.detail)
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_cancel_discovery_success(self, mock_env_vars):
        """Test cancelling a running discovery session."""
        from src.api.discovery import cancel_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.RUNNING.value,
            "pages": [{"url": "https://example.com"}],
            "flows": [{"name": "Test Flow"}],
            "events_queue": asyncio.Queue(),
        }

        try:
            response = await cancel_discovery(session_id)

            assert response["success"] is True
            assert _discovery_sessions[session_id]["status"] == SessionStatus.CANCELLED.value
            assert response["pages_discovered"] == 1
            assert response["flows_discovered"] == 1
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_cancel_discovery_already_completed(self, mock_env_vars):
        """Test cancelling an already completed session."""
        from src.api.discovery import cancel_discovery, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.COMPLETED.value,
        }

        try:
            with pytest.raises(HTTPException) as exc_info:
                await cancel_discovery(session_id)

            assert exc_info.value.status_code == 400
            assert "Cannot cancel" in str(exc_info.value.detail)
        finally:
            _discovery_sessions.clear()


class TestDiscoveredPagesEndpoints:
    """Tests for discovered pages endpoints."""

    @pytest.mark.asyncio
    async def test_get_discovered_pages(self, mock_env_vars):
        """Test getting pages for a session."""
        from src.api.discovery import get_discovered_pages, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "pages": [
                {
                    "id": "page-1",
                    "url": "https://example.com",
                    "title": "Home",
                    "description": "Homepage",
                    "page_type": "landing",
                    "elements_count": 50,
                    "forms_count": 1,
                    "links_count": 10,
                },
                {
                    "id": "page-2",
                    "url": "https://example.com/about",
                    "title": "About",
                    "description": "About page",
                    "page_type": "content",
                    "elements_count": 30,
                    "forms_count": 0,
                    "links_count": 5,
                },
            ],
        }

        try:
            pages = await get_discovered_pages(session_id)

            assert len(pages) == 2
            assert pages[0].url == "https://example.com"
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_discovered_pages_with_filter(self, mock_env_vars):
        """Test filtering pages by type."""
        from src.api.discovery import get_discovered_pages, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "pages": [
                {"id": "page-1", "url": "https://example.com", "page_type": "landing"},
                {"id": "page-2", "url": "https://example.com/form", "page_type": "form"},
            ],
        }

        try:
            pages = await get_discovered_pages(session_id, page_type="form")

            assert len(pages) == 1
            assert pages[0].page_type == "form"
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_page_details(self, mock_env_vars):
        """Test getting details for a specific page."""
        from src.api.discovery import get_page_details, _discovery_sessions

        session_id = "test-session"
        page_id = "page-1"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "pages": [
                {"id": page_id, "url": "https://example.com", "title": "Home"},
            ],
        }

        try:
            response = await get_page_details(session_id, page_id)

            assert response["success"] is True
            assert response["page"]["id"] == page_id
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_page_details_not_found(self, mock_env_vars):
        """Test getting details for a non-existent page."""
        from src.api.discovery import get_page_details, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "pages": [],
        }

        try:
            with pytest.raises(HTTPException) as exc_info:
                await get_page_details(session_id, "nonexistent")

            assert exc_info.value.status_code == 404
        finally:
            _discovery_sessions.clear()


class TestDiscoveredFlowsEndpoints:
    """Tests for discovered flows endpoints."""

    @pytest.mark.asyncio
    async def test_get_discovered_flows(self, mock_env_vars):
        """Test getting flows for a session."""
        from src.api.discovery import get_discovered_flows, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "flows": [
                {
                    "id": "flow-1",
                    "name": "Login Flow",
                    "description": "User login flow",
                    "category": "authentication",
                    "priority": "high",
                    "start_url": "https://example.com/login",
                    "steps": [{"action": "fill", "target": "#username"}],
                    "pages_involved": ["https://example.com/login"],
                },
            ],
        }

        try:
            flows = await get_discovered_flows(session_id)

            assert len(flows) == 1
            assert flows[0].name == "Login Flow"
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_discovered_flows_with_filters(self, mock_env_vars):
        """Test filtering flows by category and priority."""
        from src.api.discovery import get_discovered_flows, _discovery_sessions

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "flows": [
                {"id": "flow-1", "name": "Login", "category": "authentication", "priority": "high"},
                {"id": "flow-2", "name": "Checkout", "category": "checkout", "priority": "high"},
                {"id": "flow-3", "name": "Browse", "category": "navigation", "priority": "low"},
            ],
        }

        try:
            flows = await get_discovered_flows(session_id, category="authentication")
            assert len(flows) == 1
            assert flows[0].name == "Login"

            flows = await get_discovered_flows(session_id, priority="high")
            assert len(flows) == 2
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_update_flow(self, mock_env_vars):
        """Test updating a discovered flow."""
        from src.api.discovery import update_flow, UpdateFlowRequest, _discovered_flows

        flow_id = "test-flow"
        _discovered_flows[flow_id] = {
            "id": flow_id,
            "name": "Original Name",
            "description": "Original description",
            "category": "user_journey",
            "priority": "medium",
            "start_url": "https://example.com",
            "steps": [],
            "pages_involved": [],
            "created_at": "2024-01-01T00:00:00Z",
        }

        request = UpdateFlowRequest(
            name="Updated Name",
            priority="high",
        )

        try:
            response = await update_flow(flow_id, request)

            assert response.name == "Updated Name"
            assert response.priority == "high"
            assert _discovered_flows[flow_id]["name"] == "Updated Name"
        finally:
            _discovered_flows.clear()

    @pytest.mark.asyncio
    async def test_validate_flow(self, mock_env_vars):
        """Test validating a discovered flow."""
        from src.api.discovery import validate_flow, FlowValidationRequest, _discovered_flows, _discovery_sessions

        flow_id = "test-flow"
        session_id = "test-session"
        _discovered_flows[flow_id] = {
            "id": flow_id,
            "session_id": session_id,
            "name": "Test Flow",
            "steps": [{"action": "click", "target": "#button"}],
        }
        _discovery_sessions[session_id] = {
            "id": session_id,
            "app_url": "https://example.com",
        }

        request = FlowValidationRequest(timeout_seconds=30)

        try:
            response = await validate_flow(flow_id, request)

            assert response["success"] is True
            assert response["validation_result"]["success"] is True
            assert _discovered_flows[flow_id]["validated"] is True
        finally:
            _discovered_flows.clear()
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_generate_test_from_flow(self, mock_env_vars):
        """Test generating a test from a discovered flow."""
        from src.api.discovery import generate_test_from_flow, GenerateTestRequest, _discovered_flows

        flow_id = "test-flow"
        _discovered_flows[flow_id] = {
            "id": flow_id,
            "name": "Login Flow",
            "description": "Test login flow",
            "category": "authentication",
            "priority": "high",
            "start_url": "https://example.com/login",
            "steps": [
                {"action": "fill", "target": "#username", "value": "testuser"},
                {"action": "fill", "target": "#password", "value": "password"},
                {"action": "click", "target": "#submit"},
            ],
        }

        request = GenerateTestRequest(
            framework="playwright",
            language="typescript",
        )

        try:
            response = await generate_test_from_flow(flow_id, request)

            assert response["success"] is True
            assert "test" in response
            assert response["test"]["name"] == "Test: Login Flow"
            assert len(response["test"]["steps"]) == 3
            assert _discovered_flows[flow_id]["test_generated"] is True
        finally:
            _discovered_flows.clear()


class TestDiscoveryHistoryEndpoints:
    """Tests for discovery history and comparison endpoints."""

    @pytest.mark.asyncio
    async def test_get_discovery_history(self, mock_env_vars):
        """Test getting discovery history for a project."""
        from src.api.discovery import get_discovery_history, _discovery_sessions

        project_id = "project-1"
        _discovery_sessions["session-1"] = {
            "id": "session-1",
            "project_id": project_id,
            "status": "completed",
            "started_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T01:00:00Z",
            "pages": [{"id": "page-1"}],
            "flows": [{"id": "flow-1"}],
        }

        try:
            history = await get_discovery_history(project_id)

            assert len(history) == 1
            assert history[0].pages_found == 1
            assert history[0].flows_found == 1
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_compare_discoveries(self, mock_env_vars):
        """Test comparing two discovery sessions."""
        from src.api.discovery import compare_discoveries, _discovery_sessions

        project_id = "project-1"
        _discovery_sessions["session-1"] = {
            "id": "session-1",
            "project_id": project_id,
            "pages": [
                {"url": "https://example.com/page1"},
                {"url": "https://example.com/page2"},
            ],
            "flows": [{"name": "Flow A"}],
        }
        _discovery_sessions["session-2"] = {
            "id": "session-2",
            "project_id": project_id,
            "pages": [
                {"url": "https://example.com/page2"},
                {"url": "https://example.com/page3"},
            ],
            "flows": [{"name": "Flow A"}, {"name": "Flow B"}],
        }

        try:
            comparison = await compare_discoveries(project_id, "session-1", "session-2")

            assert "https://example.com/page3" in comparison.new_pages
            assert "https://example.com/page1" in comparison.removed_pages
            assert "Flow B" in comparison.new_flows
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_compare_discoveries_different_projects(self, mock_env_vars):
        """Test comparing sessions from different projects fails."""
        from src.api.discovery import compare_discoveries, _discovery_sessions

        _discovery_sessions["session-1"] = {"id": "session-1", "project_id": "project-1"}
        _discovery_sessions["session-2"] = {"id": "session-2", "project_id": "project-2"}

        try:
            with pytest.raises(HTTPException) as exc_info:
                await compare_discoveries("project-1", "session-1", "session-2")

            assert exc_info.value.status_code == 400
            assert "belong to the specified project" in str(exc_info.value.detail)
        finally:
            _discovery_sessions.clear()

    @pytest.mark.asyncio
    async def test_delete_session(self, mock_env_vars):
        """Test deleting a discovery session."""
        from src.api.discovery import delete_session, _discovery_sessions, _discovered_flows, SessionStatus

        session_id = "test-session"
        flow_id = "test-flow"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.COMPLETED.value,
        }
        _discovered_flows[flow_id] = {
            "id": flow_id,
            "session_id": session_id,
        }

        try:
            response = await delete_session(session_id)

            assert response["success"] is True
            assert session_id not in _discovery_sessions
            assert flow_id not in _discovered_flows
        finally:
            _discovery_sessions.clear()
            _discovered_flows.clear()

    @pytest.mark.asyncio
    async def test_delete_running_session_fails(self, mock_env_vars):
        """Test that deleting a running session fails."""
        from src.api.discovery import delete_session, _discovery_sessions, SessionStatus

        session_id = "test-session"
        _discovery_sessions[session_id] = {
            "id": session_id,
            "status": SessionStatus.RUNNING.value,
        }

        try:
            with pytest.raises(HTTPException) as exc_info:
                await delete_session(session_id)

            assert exc_info.value.status_code == 400
            assert "Cannot delete a running session" in str(exc_info.value.detail)
        finally:
            _discovery_sessions.clear()


class TestDiscoveryPatternsEndpoints:
    """Tests for discovery patterns endpoints."""

    @pytest.mark.asyncio
    async def test_list_discovery_patterns(self, mock_env_vars):
        """Test listing discovery patterns."""
        from src.api.discovery import list_discovery_patterns

        with patch("src.api.discovery.get_supabase_client") as mock_supabase:
            mock_instance = MagicMock()
            mock_instance.request = AsyncMock(return_value={
                "data": [
                    {
                        "id": "pattern-1",
                        "pattern_type": "form",
                        "pattern_name": "Login Form",
                        "pattern_signature": "form#login",
                        "pattern_data": {},
                        "times_seen": 10,
                        "created_at": "2024-01-01T00:00:00Z",
                    },
                ],
            })
            mock_supabase.return_value = mock_instance

            response = await list_discovery_patterns()

            assert len(response.patterns) == 1
            assert response.patterns[0].pattern_name == "Login Form"

    @pytest.mark.asyncio
    async def test_list_discovery_patterns_empty(self, mock_env_vars):
        """Test listing patterns when table doesn't exist."""
        from src.api.discovery import list_discovery_patterns

        with patch("src.api.discovery.get_supabase_client") as mock_supabase:
            mock_instance = MagicMock()
            mock_instance.request = AsyncMock(return_value={
                "error": "Table not found",
            })
            mock_supabase.return_value = mock_instance

            response = await list_discovery_patterns()

            assert len(response.patterns) == 0
            assert response.total == 0
