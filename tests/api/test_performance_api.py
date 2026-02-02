"""Tests for Performance API endpoints (src/api/performance.py).

Covers:
- List performance tests
- Get single performance test
- Get latest performance test
- Get performance trends
- Get performance summary
- Run performance test
- Update performance test
- Delete performance test
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock = MagicMock()
    mock.request = AsyncMock()
    mock.insert = AsyncMock()
    mock.update = AsyncMock()
    return mock


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent"}
    request.state = MagicMock()
    request.state.user_id = "user-123"
    request.state.organization_id = "org-456"
    request.json = AsyncMock(return_value={})
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_performance_test():
    """Create sample performance test data."""
    return {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "url": "https://example.com",
        "device": "mobile",
        "status": "completed",
        "lcp_ms": 2500.0,
        "fid_ms": 100.0,
        "cls": 0.1,
        "inp_ms": 200.0,
        "ttfb_ms": 500.0,
        "fcp_ms": 1500.0,
        "speed_index": 3000.0,
        "tti_ms": 3500.0,
        "tbt_ms": 300.0,
        "total_requests": 50,
        "total_transfer_size_kb": 1024.5,
        "js_execution_time_ms": 1000.0,
        "dom_content_loaded_ms": 2000.0,
        "load_time_ms": 4000.0,
        "performance_score": 85,
        "accessibility_score": 90,
        "best_practices_score": 88,
        "seo_score": 92,
        "overall_grade": "good",
        "recommendations": ["Optimize images", "Reduce JavaScript"],
        "issues": [
            {
                "category": "performance",
                "severity": "medium",
                "title": "Large images",
                "description": "Some images are not optimized",
                "savings_ms": 500.0,
                "fix_suggestion": "Use WebP format",
            }
        ],
        "summary": "Good performance overall",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "triggered_by": "user-123",
        "created_at": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestDbToResponse:
    """Tests for _db_to_response helper function."""

    def test_db_to_response_complete(self, mock_env_vars, sample_performance_test):
        """Test converting complete performance test row."""
        from src.api.performance import _db_to_response

        response = _db_to_response(sample_performance_test)

        assert response.id == sample_performance_test["id"]
        assert response.lcp_ms == 2500.0
        assert response.performance_score == 85
        assert len(response.issues) == 1
        assert response.issues[0].category == "performance"

    def test_db_to_response_minimal(self, mock_env_vars):
        """Test converting minimal performance test row."""
        from src.api.performance import _db_to_response

        minimal = {
            "id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "url": "https://example.com",
            "created_at": datetime.now(UTC).isoformat(),
        }

        response = _db_to_response(minimal)

        assert response.id == minimal["id"]
        assert response.device == "mobile"  # Default
        assert response.status == "pending"  # Default
        assert response.lcp_ms is None
        assert response.issues == []

    def test_db_to_response_with_null_issues(self, mock_env_vars):
        """Test converting row with null issues."""
        from src.api.performance import _db_to_response

        data = {
            "id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "url": "https://example.com",
            "created_at": datetime.now(UTC).isoformat(),
            "issues": None,
            "recommendations": None,
        }

        response = _db_to_response(data)
        assert response.issues == []
        assert response.recommendations == []


# ============================================================================
# List Performance Tests
# ============================================================================


class TestListPerformanceTests:
    """Tests for GET /api/v1/performance/tests endpoint."""

    @pytest.mark.asyncio
    async def test_list_performance_tests_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test successful performance test listing."""
        from src.api.performance import list_performance_tests

        mock_supabase.request.return_value = {"data": [sample_performance_test], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await list_performance_tests(
                mock_request,
                project_id=sample_performance_test["project_id"]
            )

            assert len(response.tests) == 1
            assert response.tests[0].id == sample_performance_test["id"]
            assert response.limit == 10

    @pytest.mark.asyncio
    async def test_list_performance_tests_with_filters(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing with status and device filters."""
        from src.api.performance import list_performance_tests

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await list_performance_tests(
                mock_request,
                project_id=str(uuid.uuid4()),
                status="completed",
                device="desktop",
                limit=20,
                offset=10
            )

            assert response.tests == []
            assert response.limit == 20
            assert response.offset == 10

    @pytest.mark.asyncio
    async def test_list_performance_tests_no_membership(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing returns empty when user has no memberships."""
        from src.api.performance import list_performance_tests

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)):

            response = await list_performance_tests(mock_request)

            assert response.tests == []
            assert response.total == 0

    @pytest.mark.asyncio
    async def test_list_performance_tests_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing handles database errors."""
        from src.api.performance import list_performance_tests

        # First call for count, second for data
        mock_supabase.request.side_effect = [
            {"data": [], "error": None},  # count query
            {"data": None, "error": "Database error"},  # data query
        ]

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            with pytest.raises(HTTPException) as exc_info:
                await list_performance_tests(
                    mock_request,
                    project_id=str(uuid.uuid4())
                )

            assert exc_info.value.status_code == 500


# ============================================================================
# Get Latest Performance Test
# ============================================================================


class TestGetLatestPerformanceTest:
    """Tests for GET /api/v1/performance/tests/latest endpoint."""

    @pytest.mark.asyncio
    async def test_get_latest_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test getting latest performance test."""
        from src.api.performance import get_latest_performance_test

        mock_supabase.request.return_value = {"data": [sample_performance_test], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_latest_performance_test(
                mock_request,
                project_id=sample_performance_test["project_id"]
            )

            assert response.id == sample_performance_test["id"]

    @pytest.mark.asyncio
    async def test_get_latest_none(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test getting latest when no tests exist."""
        from src.api.performance import get_latest_performance_test

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_latest_performance_test(
                mock_request,
                project_id=str(uuid.uuid4())
            )

            assert response is None

    @pytest.mark.asyncio
    async def test_get_latest_invalid_uuid(self, mock_env_vars, mock_request, mock_user):
        """Test get latest with invalid UUID."""
        from src.api.performance import get_latest_performance_test

        with patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_performance_test(mock_request, project_id="invalid-uuid")

            assert exc_info.value.status_code == 400


# ============================================================================
# Get Performance Trends
# ============================================================================


class TestGetPerformanceTrends:
    """Tests for GET /api/v1/performance/trends endpoint."""

    @pytest.mark.asyncio
    async def test_get_trends_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test getting performance trends."""
        from src.api.performance import get_performance_trends

        trend_data = [
            {
                "created_at": (datetime.now(UTC) - timedelta(days=i)).isoformat(),
                "lcp_ms": 2500 - i * 100,
                "fid_ms": 100,
                "cls": 0.1,
                "performance_score": 80 + i,
            }
            for i in range(5)
        ]

        mock_supabase.request.return_value = {"data": trend_data, "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_performance_trends(
                mock_request,
                project_id=str(uuid.uuid4()),
                days=30
            )

            assert len(response.trends) == 5
            assert response.days == 30

    @pytest.mark.asyncio
    async def test_get_trends_empty(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test getting trends when no data."""
        from src.api.performance import get_performance_trends

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_performance_trends(
                mock_request,
                project_id=str(uuid.uuid4())
            )

            assert response.trends == []


# ============================================================================
# Get Performance Summary
# ============================================================================


class TestGetPerformanceSummary:
    """Tests for GET /api/v1/performance/summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_summary_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test getting performance summary."""
        from src.api.performance import get_performance_summary

        # Recent tests query
        mock_supabase.request.side_effect = [
            {"data": [sample_performance_test], "error": None},  # tests query
            {"data": [sample_performance_test], "error": None},  # trends query
        ]

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_performance_summary(
                mock_request,
                project_id=sample_performance_test["project_id"]
            )

            assert response.latest_test is not None
            assert response.averages is not None
            assert response.total_tests == 1

    @pytest.mark.asyncio
    async def test_get_summary_no_tests(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test summary when no tests exist."""
        from src.api.performance import get_performance_summary

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_performance_summary(
                mock_request,
                project_id=str(uuid.uuid4())
            )

            assert response.latest_test is None
            assert response.averages is None
            assert response.total_tests == 0


# ============================================================================
# Get Single Performance Test
# ============================================================================


class TestGetPerformanceTest:
    """Tests for GET /api/v1/performance/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test getting single performance test."""
        from src.api.performance import get_performance_test

        mock_supabase.request.return_value = {"data": [sample_performance_test], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await get_performance_test(sample_performance_test["id"], mock_request)

            assert response.id == sample_performance_test["id"]
            assert response.performance_score == 85

    @pytest.mark.asyncio
    async def test_get_test_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test getting non-existent performance test."""
        from src.api.performance import get_performance_test

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)):

            with pytest.raises(HTTPException) as exc_info:
                await get_performance_test(str(uuid.uuid4()), mock_request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Run Performance Test
# ============================================================================


class TestRunPerformanceTest:
    """Tests for POST /api/v1/performance/tests endpoint."""

    @pytest.mark.asyncio
    async def test_run_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test running a new performance test."""
        from src.api.performance import RunPerformanceTestRequest, run_performance_test

        project_id = str(uuid.uuid4())
        new_test = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "url": "https://example.com",
            "device": "mobile",
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
        }

        mock_supabase.insert.return_value = {"data": [new_test], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()), \
             patch("src.api.performance.get_project_org_id", AsyncMock(return_value="org-123")), \
             patch("src.api.performance.log_audit", AsyncMock()):

            body = RunPerformanceTestRequest(
                project_id=project_id,
                url="https://example.com",
                device="mobile"
            )
            response = await run_performance_test(body, mock_request)

            assert response.status == "running"
            assert response.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_run_test_insert_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test running test handles insert errors."""
        from src.api.performance import RunPerformanceTestRequest, run_performance_test

        mock_supabase.insert.return_value = {"data": None, "error": "Insert failed"}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            body = RunPerformanceTestRequest(
                project_id=str(uuid.uuid4()),
                url="https://example.com"
            )

            with pytest.raises(HTTPException) as exc_info:
                await run_performance_test(body, mock_request)

            assert exc_info.value.status_code == 500


# ============================================================================
# Update Performance Test
# ============================================================================


class TestUpdatePerformanceTest:
    """Tests for PATCH /api/v1/performance/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test updating performance test results."""
        from src.api.performance import update_performance_test

        mock_request.json = AsyncMock(return_value={
            "status": "completed",
            "performance_score": 90,
            "lcp_ms": 2000.0,
        })

        mock_supabase.request.return_value = {"data": [sample_performance_test], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            response = await update_performance_test(sample_performance_test["id"], mock_request)

            mock_supabase.update.assert_called_once()
            assert response.id == sample_performance_test["id"]

    @pytest.mark.asyncio
    async def test_update_test_no_fields(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test update with no valid fields."""
        from src.api.performance import update_performance_test

        mock_request.json = AsyncMock(return_value={"invalid_field": "value"})

        mock_supabase.request.return_value = {"data": [sample_performance_test], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()):

            with pytest.raises(HTTPException) as exc_info:
                await update_performance_test(sample_performance_test["id"], mock_request)

            assert exc_info.value.status_code == 400


# ============================================================================
# Delete Performance Test
# ============================================================================


class TestDeletePerformanceTest:
    """Tests for DELETE /api/v1/performance/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_performance_test
    ):
        """Test deleting performance test."""
        from src.api.performance import delete_performance_test

        mock_supabase.request.side_effect = [
            {"data": [sample_performance_test], "error": None},  # verify access
            {"data": [], "error": None},  # delete
        ]

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.performance.verify_project_access", AsyncMock()), \
             patch("src.api.performance.get_project_org_id", AsyncMock(return_value="org-123")), \
             patch("src.api.performance.log_audit", AsyncMock()):

            response = await delete_performance_test(sample_performance_test["id"], mock_request)

            assert response["success"] is True

    @pytest.mark.asyncio
    async def test_delete_test_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test deleting non-existent test."""
        from src.api.performance import delete_performance_test

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.performance.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.performance.get_current_user", AsyncMock(return_value=mock_user)):

            with pytest.raises(HTTPException) as exc_info:
                await delete_performance_test(str(uuid.uuid4()), mock_request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_run_performance_test_request(self, mock_env_vars):
        """Test RunPerformanceTestRequest validation."""
        from src.api.performance import RunPerformanceTestRequest

        request = RunPerformanceTestRequest(
            project_id=str(uuid.uuid4()),
            url="https://example.com",
            device="desktop"
        )
        assert request.device == "desktop"

    def test_run_performance_test_default_device(self, mock_env_vars):
        """Test default device type."""
        from src.api.performance import RunPerformanceTestRequest

        request = RunPerformanceTestRequest(
            project_id=str(uuid.uuid4()),
            url="https://example.com"
        )
        assert request.device == "mobile"


class TestResponseModels:
    """Tests for response model validation."""

    def test_performance_test_response(self, mock_env_vars, sample_performance_test):
        """Test PerformanceTestResponse model."""
        from src.api.performance import PerformanceTestResponse

        response = PerformanceTestResponse(**sample_performance_test)

        assert response.id == sample_performance_test["id"]
        assert response.performance_score == 85
        assert response.overall_grade == "good"

    def test_performance_trends_response(self, mock_env_vars):
        """Test PerformanceTrendsResponse model."""
        from src.api.performance import PerformanceTrendPoint, PerformanceTrendsResponse

        trend = PerformanceTrendPoint(
            date="Jan 15",
            lcp_ms=2500.0,
            fid_ms=100.0,
            cls=0.1,
            performance_score=85.0
        )

        response = PerformanceTrendsResponse(trends=[trend], days=30)

        assert len(response.trends) == 1
        assert response.days == 30

    def test_performance_issue_model(self, mock_env_vars):
        """Test PerformanceIssue model."""
        from src.api.performance import PerformanceIssue

        issue = PerformanceIssue(
            category="performance",
            severity="high",
            title="Large JavaScript bundle",
            description="The main JS bundle is too large",
            savings_ms=500.0,
            fix_suggestion="Use code splitting"
        )

        assert issue.severity == "high"
        assert issue.savings_ms == 500.0
