"""Tests for the Discovery Repository module.

This module tests the DiscoveryRepository class which provides database-first
persistence layer for discovery sessions, pages, flows, and elements.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discovery.repository import (
    DatabaseUnavailableError,
    DiscoveryRepository,
    RecordNotFoundError,
    RepositoryError,
)
from src.discovery.models import (
    DiscoveredElement,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoveryConfig,
    DiscoveryMode,
    DiscoverySession,
    DiscoveryStatus,
    ElementBounds,
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
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()

    # Mock the table chain
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_eq = MagicMock()
    mock_single = MagicMock()
    mock_order = MagicMock()
    mock_range = MagicMock()
    mock_upsert = MagicMock()
    mock_delete = MagicMock()

    # Chain the methods
    client.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_table.upsert.return_value = mock_upsert
    mock_table.delete.return_value = mock_delete
    mock_select.eq.return_value = mock_eq
    mock_eq.single.return_value = mock_single
    mock_eq.execute.return_value = MagicMock(data=[])
    mock_single.execute.return_value = MagicMock(data=None)
    mock_select.order.return_value = mock_order
    mock_order.range.return_value = mock_range
    mock_range.execute.return_value = MagicMock(data=[])
    mock_upsert.execute.return_value = MagicMock(data=[{}])
    mock_delete.eq.return_value = mock_eq

    return client


@pytest.fixture
def sample_session():
    """Create a sample discovery session."""
    return DiscoverySession(
        id=str(uuid.uuid4()),
        project_id="test-project",
        status=DiscoveryStatus.running,
        mode=DiscoveryMode.standard_crawl,
        strategy=ExplorationStrategy.breadth_first,
        config=DiscoveryConfig(),
        progress_percentage=50.0,
        current_page="https://example.com/page",
        started_at=datetime.now(timezone.utc),
        pages_found=5,
        flows_found=2,
        elements_found=20,
    )


@pytest.fixture
def sample_page():
    """Create a sample discovered page."""
    return DiscoveredPage(
        id=str(uuid.uuid4()),
        url="https://example.com/login",
        title="Login Page",
        description="Sign in to your account",
        category=PageCategory.auth_login,
        elements=[
            DiscoveredElement(
                id=str(uuid.uuid4()),
                page_url="https://example.com/login",
                selector="#username",
                category=ElementCategory.form,
                label="Username",
                tag_name="input",
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
        incoming_links={"/"},
        importance_score=0.9,
        depth=1,
        load_time_ms=500,
        requires_auth=False,
    )


@pytest.fixture
def sample_flow():
    """Create a sample discovered flow."""
    return DiscoveredFlow(
        id=str(uuid.uuid4()),
        name="User Login",
        description="Standard login flow",
        category=FlowCategory.authentication,
        priority=1,
        start_url="https://example.com/login",
        pages=["/login", "/dashboard"],
        steps=[
            FlowStep(
                order=1,
                page_url="/login",
                action="navigate",
            ),
            FlowStep(
                order=2,
                page_url="/login",
                action="fill",
                element_selector="#username",
                input_value="test@example.com",
            ),
            FlowStep(
                order=3,
                page_url="/login",
                action="click",
                element_selector="#submit",
            ),
        ],
        success_criteria=["Redirected to dashboard"],
        complexity_score=0.3,
        business_value_score=0.9,
        confidence_score=0.85,
        validated=False,
    )


# ==============================================================================
# Test Repository Initialization
# ==============================================================================


class TestRepositoryInit:
    """Tests for DiscoveryRepository initialization."""

    def test_init_with_no_client(self):
        """Test initialization without Supabase client."""
        repo = DiscoveryRepository()
        assert repo.supabase is None
        assert repo.is_database_available is False

    def test_init_with_client(self, mock_supabase):
        """Test initialization with Supabase client."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        assert repo.supabase == mock_supabase
        assert repo.is_database_available is True

    def test_init_with_custom_retry_settings(self, mock_supabase):
        """Test initialization with custom retry settings."""
        repo = DiscoveryRepository(
            supabase_client=mock_supabase,
            max_retries=5,
            retry_delay=1.0,
        )
        assert repo.max_retries == 5
        assert repo.retry_delay == 1.0

    def test_init_empty_caches(self, mock_supabase):
        """Test that caches are initialized empty."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        assert len(repo._session_cache) == 0
        assert len(repo._pages_cache) == 0
        assert len(repo._flows_cache) == 0


# ==============================================================================
# Test Session Operations
# ==============================================================================


class TestSessionOperations:
    """Tests for session CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_session(self, mock_supabase, sample_session):
        """Test saving a session."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        result = await repo.save_session(sample_session)

        assert result == sample_session
        assert sample_session.id in repo._session_cache

    @pytest.mark.asyncio
    async def test_save_session_without_db(self, sample_session):
        """Test saving session when DB is unavailable."""
        repo = DiscoveryRepository(supabase_client=None)
        result = await repo.save_session(sample_session)

        assert result == sample_session
        assert sample_session.id in repo._session_cache

    @pytest.mark.asyncio
    async def test_get_session_from_db(self, mock_supabase, sample_session):
        """Test getting a session from database."""
        # Setup mock to return session data
        db_record = {
            "id": sample_session.id,
            "project_id": sample_session.project_id,
            "status": sample_session.status.value,
            "mode": sample_session.mode.value,
            "strategy": sample_session.strategy.value,
            "progress_percentage": sample_session.progress_percentage,
            "pages_discovered": sample_session.pages_found,
        }

        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_eq.single.return_value.execute.return_value = MagicMock(data=db_record)

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        result = await repo.get_session(sample_session.id)

        assert result is not None
        assert result.id == sample_session.id
        assert result.id in repo._session_cache

    @pytest.mark.asyncio
    async def test_get_session_from_cache(self, mock_supabase, sample_session):
        """Test getting a session from cache."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._session_cache[sample_session.id] = sample_session
        repo._db_available = False  # Force cache usage

        result = await repo.get_session(sample_session.id)

        assert result == sample_session

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_supabase):
        """Test getting non-existent session."""
        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        # Simulate PGRST116 error (no rows returned)
        mock_eq.single.return_value.execute.side_effect = Exception("PGRST116")

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        result = await repo.get_session("non-existent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_supabase, sample_session):
        """Test listing sessions."""
        db_records = [
            {
                "id": sample_session.id,
                "project_id": sample_session.project_id,
                "status": sample_session.status.value,
                "mode": sample_session.mode.value,
                "strategy": sample_session.strategy.value,
                "progress_percentage": sample_session.progress_percentage,
            }
        ]

        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_order = mock_select.order.return_value
        mock_range = mock_order.range.return_value
        mock_range.execute.return_value = MagicMock(data=db_records)

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        results = await repo.list_sessions()

        assert len(results) == 1
        assert results[0].id == sample_session.id

    @pytest.mark.asyncio
    async def test_list_sessions_with_filters(self, mock_supabase):
        """Test listing sessions with filters."""
        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq1 = MagicMock()
        mock_eq2 = MagicMock()
        mock_order = MagicMock()
        mock_range = MagicMock()

        mock_select.eq.return_value = mock_eq1
        mock_eq1.eq.return_value = mock_eq2
        mock_eq2.order.return_value = mock_order
        mock_order.range.return_value = mock_range
        mock_range.execute.return_value = MagicMock(data=[])

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        await repo.list_sessions(
            project_id="project-123",
            status=DiscoveryStatus.running,
        )

        # Verify filters were applied
        mock_select.eq.assert_called()

    @pytest.mark.asyncio
    async def test_list_sessions_from_cache(self, mock_supabase, sample_session):
        """Test listing sessions from cache when DB unavailable."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._session_cache[sample_session.id] = sample_session
        repo._db_available = False

        results = await repo.list_sessions()

        assert len(results) == 1
        assert results[0].id == sample_session.id

    @pytest.mark.asyncio
    async def test_delete_session(self, mock_supabase, sample_session):
        """Test deleting a session."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._session_cache[sample_session.id] = sample_session

        result = await repo.delete_session(sample_session.id)

        assert result is True
        assert sample_session.id not in repo._session_cache


# ==============================================================================
# Test Page Operations
# ==============================================================================


class TestPageOperations:
    """Tests for page CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_pages(self, mock_supabase, sample_page):
        """Test saving pages."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        result = await repo.save_pages(
            session_id="session-123",
            project_id="project-456",
            pages=[sample_page],
        )

        assert len(result) == 1
        assert "session-123" in repo._pages_cache

    @pytest.mark.asyncio
    async def test_save_pages_empty_list(self, mock_supabase):
        """Test saving empty pages list."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        result = await repo.save_pages(
            session_id="session-123",
            project_id="project-456",
            pages=[],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_pages_from_db(self, mock_supabase, sample_page):
        """Test getting pages from database."""
        db_record = {
            "id": sample_page.id,
            "url": sample_page.url,
            "title": sample_page.title,
            "category": sample_page.category.value,
            "outgoing_links": list(sample_page.outgoing_links),
            "incoming_links": list(sample_page.incoming_links),
            "importance_score": 90,
            "coverage_score": 0,
            "risk_score": 50,
            "depth_from_start": 1,
        }

        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_eq.execute.return_value = MagicMock(data=[db_record])

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        results = await repo.get_pages("session-123")

        assert len(results) == 1
        assert results[0].url == sample_page.url

    @pytest.mark.asyncio
    async def test_get_pages_from_cache(self, mock_supabase, sample_page):
        """Test getting pages from cache."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._pages_cache["session-123"] = [sample_page]
        repo._db_available = False

        results = await repo.get_pages("session-123")

        assert len(results) == 1
        assert results[0].id == sample_page.id


# ==============================================================================
# Test Flow Operations
# ==============================================================================


class TestFlowOperations:
    """Tests for flow CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_flows(self, mock_supabase, sample_flow):
        """Test saving flows."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        result = await repo.save_flows(
            session_id="session-123",
            project_id="project-456",
            flows=[sample_flow],
        )

        assert len(result) == 1
        assert "session-123" in repo._flows_cache

    @pytest.mark.asyncio
    async def test_save_flows_empty_list(self, mock_supabase):
        """Test saving empty flows list."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        result = await repo.save_flows(
            session_id="session-123",
            project_id="project-456",
            flows=[],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_flows_from_db(self, mock_supabase, sample_flow):
        """Test getting flows from database."""
        db_record = {
            "id": sample_flow.id,
            "name": sample_flow.name,
            "description": sample_flow.description,
            "flow_type": "authentication",
            "category": sample_flow.category.value,
            "steps": [s.to_dict() for s in sample_flow.steps],
            "entry_points": [{"url": sample_flow.start_url}],
            "success_criteria": {"checks": sample_flow.success_criteria},
            "failure_indicators": {"checks": []},
            "complexity_score": 30,
            "business_value_score": 90,
            "confidence_score": 85,
            "validated": False,
        }

        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_eq.execute.return_value = MagicMock(data=[db_record])

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        results = await repo.get_flows("session-123")

        assert len(results) == 1
        assert results[0].name == sample_flow.name

    @pytest.mark.asyncio
    async def test_get_flows_from_cache(self, mock_supabase, sample_flow):
        """Test getting flows from cache."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._flows_cache["session-123"] = [sample_flow]
        repo._db_available = False

        results = await repo.get_flows("session-123")

        assert len(results) == 1
        assert results[0].id == sample_flow.id


# ==============================================================================
# Test Bulk Operations
# ==============================================================================


class TestBulkOperations:
    """Tests for bulk operations."""

    @pytest.mark.asyncio
    async def test_get_session_with_data(
        self, mock_supabase, sample_session, sample_page, sample_flow
    ):
        """Test getting session with all related data."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._session_cache[sample_session.id] = sample_session
        repo._pages_cache[sample_session.id] = [sample_page]
        repo._flows_cache[sample_session.id] = [sample_flow]
        repo._db_available = False

        result = await repo.get_session_with_data(sample_session.id)

        assert result is not None
        assert result["session"] == sample_session
        assert len(result["pages"]) == 1
        assert len(result["flows"]) == 1

    @pytest.mark.asyncio
    async def test_get_session_with_data_not_found(self, mock_supabase):
        """Test getting non-existent session with data."""
        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_eq = mock_select.eq.return_value
        mock_eq.single.return_value.execute.side_effect = Exception("PGRST116")

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        result = await repo.get_session_with_data("non-existent")

        assert result is None


# ==============================================================================
# Test Retry Logic
# ==============================================================================


class TestRetryLogic:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_attempt(self, mock_supabase):
        """Test that successful operations don't retry."""
        repo = DiscoveryRepository(
            supabase_client=mock_supabase,
            max_retries=3,
        )

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return MagicMock(data=[{}])

        result = await repo._retry_operation(operation, "test_operation")

        assert call_count == 1
        assert repo._db_available is True

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self, mock_supabase):
        """Test that operations retry on transient failures."""
        repo = DiscoveryRepository(
            supabase_client=mock_supabase,
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return MagicMock(data=[{}])

        result = await repo._retry_operation(operation, "test_operation")

        assert call_count == 3
        assert repo._db_available is True

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_error(self, mock_supabase):
        """Test that exhausted retries raise DatabaseUnavailableError."""
        repo = DiscoveryRepository(
            supabase_client=mock_supabase,
            max_retries=3,
            retry_delay=0.01,
        )

        def operation():
            raise Exception("Persistent error")

        with pytest.raises(DatabaseUnavailableError):
            await repo._retry_operation(operation, "test_operation")

        assert repo._db_available is False


# ==============================================================================
# Test Data Conversion
# ==============================================================================


class TestDataConversion:
    """Tests for data conversion methods."""

    def test_session_to_db_record(self, mock_supabase, sample_session):
        """Test converting session to DB record."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = repo._session_to_db_record(sample_session)

        assert record["id"] == sample_session.id
        assert record["project_id"] == sample_session.project_id
        assert record["status"] == sample_session.status.value
        assert record["mode"] == sample_session.mode.value

    def test_db_record_to_session(self, mock_supabase):
        """Test converting DB record to session."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = {
            "id": "test-id",
            "project_id": "project-123",
            "status": "running",
            "mode": "standard_crawl",
            "strategy": "breadth_first",
            "progress_percentage": 50,
            "pages_discovered": 5,
        }

        session = repo._db_record_to_session(record)

        assert session.id == "test-id"
        assert session.status == DiscoveryStatus.running
        assert session.mode == DiscoveryMode.standard_crawl

    def test_page_to_db_record(self, mock_supabase, sample_page):
        """Test converting page to DB record."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = repo._page_to_db_record(
            sample_page, "session-123", "project-456"
        )

        assert record["id"] == sample_page.id
        assert record["url"] == sample_page.url
        assert record["discovery_session_id"] == "session-123"
        assert "url_hash" in record

    def test_db_record_to_page(self, mock_supabase):
        """Test converting DB record to page."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = {
            "id": "page-123",
            "url": "https://example.com",
            "title": "Test Page",
            "category": "landing",
            "outgoing_links": ["/link1", "/link2"],
            "incoming_links": ["/"],
            "importance_score": 80,
            "coverage_score": 0,
            "risk_score": 50,
            "depth_from_start": 0,
        }

        page = repo._db_record_to_page(record)

        assert page.id == "page-123"
        assert page.category == PageCategory.landing
        assert len(page.outgoing_links) == 2

    def test_element_to_db_record(self, mock_supabase, sample_page):
        """Test converting element to DB record."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        element = sample_page.elements[0]
        record = repo._element_to_db_record(element, sample_page.id, "session-123")

        assert record["id"] == element.id
        assert record["page_id"] == sample_page.id
        assert record["selector"] == element.selector

    def test_db_record_to_element(self, mock_supabase):
        """Test converting DB record to element."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = {
            "id": "elem-123",
            "selector": "#test",
            "tag_name": "button",
            "category": "button",
            "importance_score": 70,
            "stability_score": 80,
            "is_visible": True,
            "is_enabled": True,
        }

        element = repo._db_record_to_element(record)

        assert element.id == "elem-123"
        assert element.selector == "#test"

    def test_flow_to_db_record(self, mock_supabase, sample_flow):
        """Test converting flow to DB record."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = repo._flow_to_db_record(sample_flow, "session-123", "project-456")

        assert record["id"] == sample_flow.id
        assert record["name"] == sample_flow.name
        assert record["discovery_session_id"] == "session-123"
        assert len(record["steps"]) == 3

    def test_db_record_to_flow(self, mock_supabase):
        """Test converting DB record to flow."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        record = {
            "id": "flow-123",
            "name": "Test Flow",
            "description": "A test flow",
            "flow_type": "authentication",
            "category": "authentication",
            "steps": [
                {"order": 1, "page_url": "/login", "action": "navigate"},
            ],
            "entry_points": [{"url": "/login"}],
            "success_criteria": {"checks": ["Logged in"]},
            "failure_indicators": {"checks": []},
            "complexity_score": 30,
            "business_value_score": 90,
            "confidence_score": 85,
            "validated": False,
        }

        flow = repo._db_record_to_flow(record)

        assert flow.id == "flow-123"
        assert flow.category == FlowCategory.authentication
        assert len(flow.steps) == 1

    def test_map_page_category_to_type(self, mock_supabase):
        """Test page category to type mapping."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        test_cases = [
            ("landing", "landing"),
            ("auth_login", "auth"),
            ("auth_signup", "auth"),
            ("dashboard", "dashboard"),
            ("form", "form"),
            ("settings", "settings"),
            ("error", "error"),
            ("unknown", "unknown"),
        ]

        for category, expected in test_cases:
            result = repo._map_page_category_to_type(category)
            assert result == expected, f"Failed for {category}"

    def test_map_element_category(self, mock_supabase):
        """Test element category mapping."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)

        test_cases = [
            ("navigation", "link"),
            ("form", "input"),
            ("action", "button"),
            ("content", "other"),
            ("authentication", "input"),
        ]

        for category, expected in test_cases:
            result = repo._map_element_category(category)
            assert result == expected, f"Failed for {category}"


# ==============================================================================
# Test Health Check
# ==============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_with_db(self, mock_supabase):
        """Test health check when DB is configured."""
        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_limit = mock_select.limit.return_value
        mock_limit.execute.return_value = MagicMock(data=[{"id": "test"}])

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        repo._session_cache["s1"] = MagicMock()
        repo._pages_cache["s1"] = [MagicMock()]

        status = await repo.health_check()

        assert status["database_configured"] is True
        assert status["database_available"] is True
        assert status["cache_stats"]["sessions"] == 1
        assert status["cache_stats"]["pages"] == 1

    @pytest.mark.asyncio
    async def test_health_check_without_db(self):
        """Test health check when DB is not configured."""
        repo = DiscoveryRepository(supabase_client=None)

        status = await repo.health_check()

        assert status["database_configured"] is False
        assert status["database_available"] is False

    @pytest.mark.asyncio
    async def test_health_check_db_error(self, mock_supabase):
        """Test health check when DB ping fails."""
        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value
        mock_limit = mock_select.limit.return_value
        mock_limit.execute.side_effect = Exception("Connection error")

        repo = DiscoveryRepository(supabase_client=mock_supabase)

        status = await repo.health_check()

        assert status["database_configured"] is True
        assert status["database_available"] is False
        assert "ping_error" in status


# ==============================================================================
# Test Exception Classes
# ==============================================================================


class TestExceptionClasses:
    """Tests for exception classes."""

    def test_repository_error(self):
        """Test RepositoryError base class."""
        error = RepositoryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_database_unavailable_error(self):
        """Test DatabaseUnavailableError."""
        error = DatabaseUnavailableError("DB down")
        assert str(error) == "DB down"
        assert isinstance(error, RepositoryError)

    def test_record_not_found_error(self):
        """Test RecordNotFoundError."""
        error = RecordNotFoundError("Not found")
        assert str(error) == "Not found"
        assert isinstance(error, RepositoryError)


# ==============================================================================
# Test Element Summary
# ==============================================================================


class TestElementSummary:
    """Tests for element summary generation."""

    def test_summarize_element_categories(self, mock_supabase, sample_page):
        """Test element category summarization."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        summary = repo._summarize_element_categories(sample_page.elements)

        assert "form" in summary
        assert "authentication" in summary
        assert summary["form"] == 1
        assert summary["authentication"] == 1

    def test_summarize_empty_elements(self, mock_supabase):
        """Test summarization of empty element list."""
        repo = DiscoveryRepository(supabase_client=mock_supabase)
        summary = repo._summarize_element_categories([])

        assert summary == {}


# ==============================================================================
# Test Cache Refresh
# ==============================================================================


class TestCacheRefresh:
    """Tests for cache refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_cache_no_db(self):
        """Test cache refresh without database."""
        repo = DiscoveryRepository(supabase_client=None)
        count = await repo.refresh_cache_from_db()
        assert count == 0

    @pytest.mark.asyncio
    async def test_refresh_cache_with_db(self, mock_supabase, sample_session):
        """Test cache refresh with database."""
        db_records = [
            {
                "id": sample_session.id,
                "project_id": sample_session.project_id,
                "status": "completed",
                "mode": "standard_crawl",
                "strategy": "breadth_first",
            }
        ]

        mock_table = mock_supabase.table.return_value
        mock_select = mock_table.select.return_value

        # For list_sessions
        mock_order = mock_select.order.return_value
        mock_range = mock_order.range.return_value
        mock_range.execute.return_value = MagicMock(data=db_records)

        # For get_pages and get_flows
        mock_eq = mock_select.eq.return_value
        mock_eq.execute.return_value = MagicMock(data=[])

        repo = DiscoveryRepository(supabase_client=mock_supabase)
        count = await repo.refresh_cache_from_db()

        assert count == 1
