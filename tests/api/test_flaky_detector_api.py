"""Tests for Flaky Test Detection API endpoints (src/api/flaky_detector.py).

Covers:
- List flaky tests
- Get flakiness trend
- Get flaky test stats
- Toggle quarantine
- FlakyDetector class methods
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
    return request


@pytest.fixture
def mock_user_context():
    """Create a mock UserContext."""
    from src.api.security.auth import UserContext
    return UserContext(
        user_id="user-123",
        email="test@example.com",
        organization_id="org-456",
        roles=["admin"],
    )


@pytest.fixture
def sample_test_result():
    """Create sample test result data."""
    return {
        "id": str(uuid.uuid4()),
        "test_id": str(uuid.uuid4()),
        "test_run_id": str(uuid.uuid4()),
        "name": "Login Flow Test",
        "status": "passed",
        "duration_ms": 5000,
        "error_message": None,
        "completed_at": datetime.now(UTC).isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    return {
        "id": str(uuid.uuid4()),
        "name": "Login Flow Test",
        "description": "Tests the login functionality",
        "project_id": str(uuid.uuid4()),
        "tags": ["smoke", "auth"],
        "settings": {},
        "is_active": True,
    }


@pytest.fixture
def sample_project_data():
    """Create sample project data."""
    return {
        "id": str(uuid.uuid4()),
        "name": "Test Project",
        "organization_id": "org-456",
    }


# ============================================================================
# FlakyDetector Class Tests
# ============================================================================


class TestFlakyDetectorCalculateScore:
    """Tests for FlakyDetector.calculate_flaky_score method."""

    @pytest.mark.asyncio
    async def test_calculate_flaky_score_stable(self, mock_env_vars, mock_supabase):
        """Test flaky score calculation for stable test."""
        from src.api.flaky_detector import FlakyDetector

        # All passed results
        results = [
            {"test_id": "t1", "test_run_id": f"r{i}", "status": "passed", "completed_at": datetime.now(UTC).isoformat(), "duration_ms": 1000}
            for i in range(10)
        ]

        mock_supabase.request.return_value = {"data": results, "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            detector = FlakyDetector()
            score = await detector.calculate_flaky_score("test-123")

            assert score == 0.0  # No status changes = stable

    @pytest.mark.asyncio
    async def test_calculate_flaky_score_flaky(self, mock_env_vars, mock_supabase):
        """Test flaky score calculation for flaky test."""
        from src.api.flaky_detector import FlakyDetector

        # Alternating results
        results = [
            {"test_id": "t1", "test_run_id": f"r{i}", "status": "passed" if i % 2 == 0 else "failed", "completed_at": datetime.now(UTC).isoformat(), "duration_ms": 1000}
            for i in range(10)
        ]

        mock_supabase.request.return_value = {"data": results, "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            detector = FlakyDetector()
            score = await detector.calculate_flaky_score("test-123")

            assert score > 0.5  # High flakiness due to alternating results

    @pytest.mark.asyncio
    async def test_calculate_flaky_score_insufficient_data(self, mock_env_vars, mock_supabase):
        """Test flaky score with insufficient data."""
        from src.api.flaky_detector import FlakyDetector

        # Only 1 result
        mock_supabase.request.return_value = {
            "data": [{"test_id": "t1", "test_run_id": "r1", "status": "passed", "completed_at": datetime.now(UTC).isoformat(), "duration_ms": 1000}],
            "error": None
        }

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            detector = FlakyDetector()
            score = await detector.calculate_flaky_score("test-123")

            assert score == 0.0  # Not enough data


class TestFlakyDetectorAnalyzeTest:
    """Tests for FlakyDetector.analyze_test method."""

    @pytest.mark.asyncio
    async def test_analyze_test_returns_result(self, mock_env_vars, mock_supabase):
        """Test analyze_test returns proper result."""
        from src.api.flaky_detector import FlakyDetector

        results = [
            {"test_id": "t1", "test_run_id": f"r{i}", "status": "passed" if i % 3 != 0 else "failed", "completed_at": datetime.now(UTC).isoformat(), "duration_ms": 1000}
            for i in range(10)
        ]

        mock_supabase.request.side_effect = [
            {"data": results, "error": None},  # outcomes
            {"data": [{"id": "t1", "name": "Test 1", "project_id": "p1"}], "error": None},  # test info
            {"data": [{"settings": {}}], "error": None},  # first flaky detection
        ]

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            detector = FlakyDetector()
            result = await detector.analyze_test("test-123")

            assert result is not None
            assert result.test_id == "test-123"
            assert result.total_runs == 10

    @pytest.mark.asyncio
    async def test_analyze_test_insufficient_runs(self, mock_env_vars, mock_supabase):
        """Test analyze_test with insufficient runs."""
        from src.api.flaky_detector import FlakyDetector

        mock_supabase.request.return_value = {
            "data": [{"test_id": "t1", "test_run_id": "r1", "status": "passed", "completed_at": datetime.now(UTC).isoformat(), "duration_ms": 1000}],
            "error": None
        }

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            detector = FlakyDetector()
            result = await detector.analyze_test("test-123")

            assert result is None  # Not enough runs


class TestFlakyDetectorRecommendation:
    """Tests for FlakyDetector._get_recommendation method."""

    def test_recommendation_quarantine(self, mock_env_vars):
        """Test quarantine recommendation."""
        from src.api.flaky_detector import FlakyDetector, FlakyRecommendation

        detector = FlakyDetector()
        recommendation = detector._get_recommendation(0.6)

        assert recommendation == FlakyRecommendation.QUARANTINE

    def test_recommendation_investigate(self, mock_env_vars):
        """Test investigate recommendation."""
        from src.api.flaky_detector import FlakyDetector, FlakyRecommendation

        detector = FlakyDetector()
        recommendation = detector._get_recommendation(0.4)

        assert recommendation == FlakyRecommendation.INVESTIGATE

    def test_recommendation_monitor(self, mock_env_vars):
        """Test monitor recommendation."""
        from src.api.flaky_detector import FlakyDetector, FlakyRecommendation

        detector = FlakyDetector()
        recommendation = detector._get_recommendation(0.2)

        assert recommendation == FlakyRecommendation.MONITOR

    def test_recommendation_stable(self, mock_env_vars):
        """Test stable recommendation."""
        from src.api.flaky_detector import FlakyDetector, FlakyRecommendation

        detector = FlakyDetector()
        recommendation = detector._get_recommendation(0.05)

        assert recommendation == FlakyRecommendation.STABLE


# ============================================================================
# List Flaky Tests API Tests
# ============================================================================


class TestListFlakyTests:
    """Tests for GET /api/v1/flaky-tests endpoint."""

    @pytest.mark.asyncio
    async def test_list_flaky_tests_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_project_data
    ):
        """Test listing flaky tests successfully."""
        from src.api.flaky_detector import list_flaky_tests

        test_results = [
            {"test_id": "t1", "name": "Test 1", "status": "passed", "duration_ms": 1000, "error_message": None, "created_at": datetime.now(UTC).isoformat(), "test_run_id": "r1"},
            {"test_id": "t1", "name": "Test 1", "status": "failed", "duration_ms": 1500, "error_message": "timeout error", "created_at": datetime.now(UTC).isoformat(), "test_run_id": "r2"},
            {"test_id": "t1", "name": "Test 1", "status": "passed", "duration_ms": 1000, "error_message": None, "created_at": datetime.now(UTC).isoformat(), "test_run_id": "r3"},
            {"test_id": "t1", "name": "Test 1", "status": "failed", "duration_ms": 2000, "error_message": "timeout error", "created_at": datetime.now(UTC).isoformat(), "test_run_id": "r4"},
        ]

        run_ids = ["r1", "r2", "r3", "r4"]
        test_runs = [{"id": rid, "project_id": sample_project_data["id"]} for rid in run_ids]

        # Use AsyncMock for proper async return values
        async def mock_request_handler(path):
            if "projects" in path:
                return {"data": [sample_project_data], "error": None}
            elif "test_runs" in path:
                return {"data": test_runs, "error": None}
            elif "test_results" in path:
                return {"data": test_results, "error": None}
            elif "tests" in path:
                return {"data": [{"id": "t1", "name": "Test 1", "project_id": sample_project_data["id"]}], "error": None}
            elif "healing_patterns" in path:
                return {"data": [], "error": None}
            return {"data": [], "error": None}

        mock_supabase.request = AsyncMock(side_effect=mock_request_handler)

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            # Pass actual parameter values instead of Query objects
            response = await list_flaky_tests(
                request=mock_request,
                project_id=None,
                min_score=0.0,
                days=30,
                limit=100,
                user=mock_user_context
            )

            assert isinstance(response.tests, list)

    @pytest.mark.asyncio
    async def test_list_flaky_tests_empty(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context
    ):
        """Test listing flaky tests when none exist."""
        from src.api.flaky_detector import list_flaky_tests

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            # Pass actual parameter values instead of Query objects
            response = await list_flaky_tests(
                request=mock_request,
                project_id=None,
                min_score=0.0,
                days=30,
                limit=100,
                user=mock_user_context
            )

            assert response.tests == []
            assert response.total == 0

    @pytest.mark.asyncio
    async def test_list_flaky_tests_with_project_filter(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context
    ):
        """Test listing flaky tests with project filter."""
        from src.api.flaky_detector import list_flaky_tests

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            response = await list_flaky_tests(
                mock_request,
                project_id=project_id,
                user=mock_user_context
            )

            assert response.tests == []


# ============================================================================
# Flakiness Trend API Tests
# ============================================================================


class TestGetFlakinessTrend:
    """Tests for GET /api/v1/flaky-tests/trend endpoint."""

    @pytest.mark.asyncio
    async def test_get_trend_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_project_data
    ):
        """Test getting flakiness trend successfully."""
        from src.api.flaky_detector import get_flakiness_trend

        call_count = [0]

        async def mock_request_handler(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"data": [sample_project_data], "error": None}  # projects
            return {"data": [], "error": None}  # weeks of data

        mock_supabase.request = AsyncMock(side_effect=mock_request_handler)

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            # Pass actual parameter values instead of Query objects
            response = await get_flakiness_trend(
                request=mock_request,
                project_id=None,
                weeks=8,
                user=mock_user_context
            )

            assert len(response.trend) == 8

    @pytest.mark.asyncio
    async def test_get_trend_empty(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context
    ):
        """Test getting trend when no projects exist."""
        from src.api.flaky_detector import get_flakiness_trend

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            # Pass actual parameter values instead of Query objects
            response = await get_flakiness_trend(
                request=mock_request,
                project_id=None,
                weeks=8,
                user=mock_user_context
            )

            assert response.trend == []


# ============================================================================
# Flaky Test Stats API Tests
# ============================================================================


class TestGetFlakyTestStats:
    """Tests for GET /api/v1/flaky-tests/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context
    ):
        """Test getting flaky test stats successfully."""
        from src.api.flaky_detector import get_flaky_test_stats

        # Mock list_flaky_tests response - needs to be AsyncMock for proper await
        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            # Pass actual parameter values instead of Query objects
            response = await get_flaky_test_stats(
                request=mock_request,
                project_id=None,
                user=mock_user_context
            )

            assert response.total >= 0
            assert response.high >= 0
            assert response.medium >= 0
            assert response.low >= 0


# ============================================================================
# Toggle Quarantine API Tests
# ============================================================================


class TestToggleQuarantine:
    """Tests for POST /api/v1/flaky-tests/{test_id}/quarantine endpoint."""

    @pytest.mark.asyncio
    async def test_toggle_quarantine_enable(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_test_data
    ):
        """Test enabling quarantine for a test."""
        from src.api.flaky_detector import ToggleQuarantineRequest, toggle_quarantine

        mock_supabase.request.return_value = {"data": [sample_test_data], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            request = ToggleQuarantineRequest(quarantine=True)
            response = await toggle_quarantine(
                sample_test_data["id"],
                request,
                mock_request,
                user=mock_user_context
            )

            assert response.success is True
            assert response.is_quarantined is True

    @pytest.mark.asyncio
    async def test_toggle_quarantine_disable(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_test_data
    ):
        """Test disabling quarantine for a test."""
        from src.api.flaky_detector import ToggleQuarantineRequest, toggle_quarantine

        test_with_quarantine = {**sample_test_data, "tags": ["smoke", "quarantined"]}
        mock_supabase.request.return_value = {"data": [test_with_quarantine], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            request = ToggleQuarantineRequest(quarantine=False)
            response = await toggle_quarantine(
                sample_test_data["id"],
                request,
                mock_request,
                user=mock_user_context
            )

            assert response.success is True
            assert response.is_quarantined is False

    @pytest.mark.asyncio
    async def test_toggle_quarantine_test_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user_context
    ):
        """Test quarantine toggle for non-existent test."""
        from src.api.flaky_detector import ToggleQuarantineRequest, toggle_quarantine

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            request = ToggleQuarantineRequest(quarantine=True)

            with pytest.raises(HTTPException) as exc_info:
                await toggle_quarantine(
                    str(uuid.uuid4()),
                    request,
                    mock_request,
                    user=mock_user_context
                )

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_quarantine_invalid_uuid(
        self, mock_env_vars, mock_request, mock_user_context
    ):
        """Test quarantine toggle with invalid UUID."""
        from src.api.flaky_detector import ToggleQuarantineRequest, toggle_quarantine

        request = ToggleQuarantineRequest(quarantine=True)

        with pytest.raises(HTTPException) as exc_info:
            await toggle_quarantine(
                "invalid-uuid",
                request,
                mock_request,
                user=mock_user_context
            )

        assert exc_info.value.status_code == 400


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_toggle_quarantine_request(self, mock_env_vars):
        """Test ToggleQuarantineRequest validation."""
        from src.api.flaky_detector import ToggleQuarantineRequest

        request = ToggleQuarantineRequest(quarantine=True)
        assert request.quarantine is True


class TestResponseModels:
    """Tests for response model validation."""

    def test_flaky_test_response(self, mock_env_vars):
        """Test FlakyTestResponse model."""
        from src.api.flaky_detector import FlakyTestResponse, RootCause

        response = FlakyTestResponse(
            id="test-123",
            name="Login Test",
            path="tests/login.spec.ts",
            flakiness_score=0.35,
            total_runs=20,
            pass_count=13,
            fail_count=7,
            last_run=datetime.now(UTC).isoformat(),
            trend="increasing",
            is_quarantined=False,
            root_causes=[
                RootCause(
                    type="timing",
                    description="Timeout during login",
                    confidence=0.8
                )
            ],
            recent_results=[True, False, True, True, False],
            avg_duration=5000.0,
            suggested_fix="Add explicit waits",
            project_id="proj-123",
            project_name="Test Project"
        )

        assert response.flakiness_score == 0.35
        assert len(response.root_causes) == 1

    def test_flaky_test_trend_response(self, mock_env_vars):
        """Test FlakyTestTrendResponse model."""
        from src.api.flaky_detector import FlakyTestTrendItem, FlakyTestTrendResponse

        response = FlakyTestTrendResponse(
            trend=[
                FlakyTestTrendItem(date="Jan 15", flaky=5, fixed=2),
                FlakyTestTrendItem(date="Jan 22", flaky=3, fixed=4),
            ]
        )

        assert len(response.trend) == 2
        assert response.trend[0].flaky == 5

    def test_flaky_test_stats_response(self, mock_env_vars):
        """Test FlakyTestStatsResponse model."""
        from src.api.flaky_detector import FlakyTestStatsResponse

        response = FlakyTestStatsResponse(
            total=20,
            high=5,
            medium=8,
            low=7,
            quarantined=3,
            avg_score=0.25
        )

        assert response.total == 20
        assert response.high + response.medium + response.low == 20


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestNormalizeStatus:
    """Tests for _normalize_status helper method."""

    def test_normalize_passed(self, mock_env_vars):
        """Test normalizing passed statuses."""
        from src.api.flaky_detector import FlakyDetector

        detector = FlakyDetector()

        assert detector._normalize_status("passed") == "passed"
        assert detector._normalize_status("PASSED") == "passed"
        assert detector._normalize_status("success") == "passed"

    def test_normalize_failed(self, mock_env_vars):
        """Test normalizing failed statuses."""
        from src.api.flaky_detector import FlakyDetector

        detector = FlakyDetector()

        assert detector._normalize_status("failed") == "failed"
        assert detector._normalize_status("FAILED") == "failed"
        assert detector._normalize_status("failure") == "failed"
        assert detector._normalize_status("error") == "failed"

    def test_normalize_other(self, mock_env_vars):
        """Test normalizing other statuses."""
        from src.api.flaky_detector import FlakyDetector

        detector = FlakyDetector()

        assert detector._normalize_status("skipped") == "other"
        assert detector._normalize_status("pending") == "other"
        assert detector._normalize_status("running") == "other"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_calculate_flaky_score_function(self, mock_env_vars, mock_supabase):
        """Test calculate_flaky_score convenience function."""
        from src.api.flaky_detector import calculate_flaky_score

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            score = await calculate_flaky_score("test-123")
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_detect_flaky_tests_function(self, mock_env_vars, mock_supabase):
        """Test detect_flaky_tests convenience function."""
        from src.api.flaky_detector import detect_flaky_tests

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.flaky_detector.get_supabase_client", return_value=mock_supabase):
            result = await detect_flaky_tests("schedule-123")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_flaky_detector(self, mock_env_vars):
        """Test get_flaky_detector returns singleton."""
        from src.api.flaky_detector import get_flaky_detector

        detector1 = get_flaky_detector()
        detector2 = get_flaky_detector()

        assert detector1 is detector2
