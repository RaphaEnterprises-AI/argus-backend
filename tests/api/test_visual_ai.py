"""Tests for Visual AI API endpoints (src/api/visual_ai.py).

Covers:
- Snapshot capture endpoints
- Visual comparison endpoints
- Responsive testing
- Cross-browser testing
- Baseline management
- Approval/rejection workflows
- AI explanation endpoints
"""

import uuid
from datetime import UTC, datetime
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
    mock.select = AsyncMock()
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
def sample_snapshot_data():
    """Create sample snapshot data."""
    return {
        "id": str(uuid.uuid4()),
        "url": "https://example.com",
        "page_url": "https://example.com",
        "screenshot_path": "data:image/png;base64,test",
        "screenshot_url": "data:image/png;base64,test",
        "viewport_width": 1440,
        "viewport_height": 900,
        "browser": "chromium",
        "metadata": {"captured_at": datetime.now(UTC).isoformat()},
        "project_id": str(uuid.uuid4()),
        "name": "test-snapshot",
        "captured_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_baseline_data():
    """Create sample baseline data."""
    return {
        "id": str(uuid.uuid4()),
        "name": "homepage-baseline",
        "project_id": str(uuid.uuid4()),
        "url": "https://example.com",
        "page_url": "https://example.com",
        "screenshot_path": "data:image/png;base64,test",
        "screenshot_url": "data:image/png;base64,test",
        "viewport_width": 1440,
        "viewport_height": 900,
        "browser": "chromium",
        "version": 1,
        "metadata": {},
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_comparison_data():
    """Create sample comparison data."""
    return {
        "id": str(uuid.uuid4()),
        "baseline_id": str(uuid.uuid4()),
        "current_snapshot_id": str(uuid.uuid4()),
        "match": True,
        "match_percentage": 98.5,
        "has_regressions": False,
        "differences": [],
        "summary": "Visual comparison completed successfully",
        "cost_usd": 0.01,
        "context": None,
        "status": "passed",
        "compared_at": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# Capture Endpoint Tests
# ============================================================================


class TestCaptureSnapshot:
    """Tests for POST /api/v1/visual/capture endpoint."""

    @pytest.mark.asyncio
    async def test_capture_snapshot_success(self, mock_env_vars, mock_supabase):
        """Test successful snapshot capture."""
        from src.api.visual_ai import CaptureRequest, capture_snapshot

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = CaptureRequest(url="https://example.com")
            response = await capture_snapshot(request)

            assert response.url == "https://example.com"
            assert response.browser == "chromium"
            assert response.viewport["width"] == 1440

    @pytest.mark.asyncio
    async def test_capture_snapshot_with_custom_viewport(self, mock_env_vars):
        """Test snapshot capture with custom viewport."""
        from src.api.visual_ai import CaptureRequest, ViewportConfig, capture_snapshot

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = CaptureRequest(
                url="https://example.com",
                viewport=ViewportConfig(width=375, height=667),
                browser="webkit"
            )
            response = await capture_snapshot(request)

            assert response.viewport["width"] == 375
            assert response.viewport["height"] == 667
            assert response.browser == "webkit"

    @pytest.mark.asyncio
    async def test_capture_snapshot_failure(self, mock_env_vars):
        """Test snapshot capture handles errors."""
        from src.api.visual_ai import CaptureRequest, capture_snapshot

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(side_effect=Exception("Browser error"))):
            request = CaptureRequest(url="https://example.com")

            with pytest.raises(HTTPException) as exc_info:
                await capture_snapshot(request)

            assert exc_info.value.status_code == 500
            assert "Capture failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_capture_snapshot_full_page(self, mock_env_vars):
        """Test full page screenshot capture."""
        from src.api.visual_ai import CaptureRequest, capture_snapshot

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))) as mock_capture, \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = CaptureRequest(url="https://example.com", full_page=True)
            await capture_snapshot(request)

            # Verify full_page was passed
            mock_capture.assert_called_once()
            call_kwargs = mock_capture.call_args[1]
            assert call_kwargs["full_page"] is True


class TestCaptureResponsiveMatrix:
    """Tests for POST /api/v1/visual/responsive/capture endpoint."""

    @pytest.mark.asyncio
    async def test_capture_responsive_success(self, mock_env_vars):
        """Test responsive capture at multiple viewports."""
        from src.api.visual_ai import ResponsiveRequest, capture_responsive_matrix

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = ResponsiveRequest(url="https://example.com")
            response = await capture_responsive_matrix(request)

            assert response["success"] is True
            assert len(response["results"]) == 4  # Default viewports
            assert all(r["success"] for r in response["results"])

    @pytest.mark.asyncio
    async def test_capture_responsive_partial_failure(self, mock_env_vars):
        """Test responsive capture handles partial failures."""
        from src.api.visual_ai import ResponsiveRequest, capture_responsive_matrix

        call_count = [0]

        async def mock_capture(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Browser error")
            return b"fake_screenshot_data", {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", mock_capture), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = ResponsiveRequest(url="https://example.com")
            response = await capture_responsive_matrix(request)

            # Should report partial success
            assert response["success"] is False
            success_count = sum(1 for r in response["results"] if r["success"])
            failure_count = sum(1 for r in response["results"] if not r["success"])
            assert success_count == 3
            assert failure_count == 1


class TestCaptureBrowserMatrix:
    """Tests for POST /api/v1/visual/browsers/capture endpoint."""

    @pytest.mark.asyncio
    async def test_capture_browser_matrix_success(self, mock_env_vars):
        """Test cross-browser capture."""
        from src.api.visual_ai import BrowserMatrixRequest, capture_browser_matrix

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        with patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai._save_snapshot_to_db", AsyncMock(return_value={})):

            request = BrowserMatrixRequest(url="https://example.com")
            response = await capture_browser_matrix(request)

            assert response["success"] is True
            assert len(response["results"]) == 3  # chromium, firefox, webkit


# ============================================================================
# Comparison Endpoint Tests
# ============================================================================


class TestCompareSnapshots:
    """Tests for POST /api/v1/visual/compare endpoint."""

    @pytest.mark.asyncio
    async def test_compare_snapshots_baseline_not_found(self, mock_env_vars, mock_supabase):
        """Test comparison fails when baseline not found."""
        from src.api.visual_ai import CompareRequest, compare_snapshots

        mock_supabase.select.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            request = CompareRequest(
                baseline_id="nonexistent",
                current_url="https://example.com"
            )

            with pytest.raises(HTTPException) as exc_info:
                await compare_snapshots(request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_snapshots_table_not_configured(self, mock_env_vars, mock_supabase):
        """Test comparison handles missing tables gracefully."""
        from src.api.visual_ai import CompareRequest, compare_snapshots

        mock_supabase.select.return_value = {"data": None, "error": "relation does not exist"}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            request = CompareRequest(
                baseline_id=str(uuid.uuid4()),
                current_url="https://example.com"
            )

            with pytest.raises(HTTPException) as exc_info:
                await compare_snapshots(request)

            assert exc_info.value.status_code == 503
            assert "not configured" in exc_info.value.detail


# ============================================================================
# Baseline Management Tests
# ============================================================================


class TestCreateBaseline:
    """Tests for POST /api/v1/visual/baselines endpoint."""

    @pytest.mark.asyncio
    async def test_create_baseline_success(
        self, mock_env_vars, mock_supabase, mock_user_context, sample_baseline_data
    ):
        """Test successful baseline creation."""
        from src.api.visual_ai import create_baseline

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        mock_supabase.select.return_value = {"data": [], "error": None}  # No existing baseline
        mock_supabase.insert.return_value = {"data": [sample_baseline_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):

            response = await create_baseline(
                url="https://example.com",
                name="homepage-baseline",
                project_id=sample_baseline_data["project_id"],
                user=mock_user_context
            )

            assert response.name == "homepage-baseline"
            assert response.version == 1

    @pytest.mark.asyncio
    async def test_create_baseline_updates_existing(
        self, mock_env_vars, mock_supabase, mock_user_context, sample_baseline_data
    ):
        """Test baseline update when name already exists."""
        from src.api.visual_ai import create_baseline

        mock_screenshot = b"fake_screenshot_data"
        mock_metadata = {"captured_at": datetime.now(UTC).isoformat()}

        existing_baseline = {**sample_baseline_data, "version": 3}
        mock_supabase.select.return_value = {"data": [existing_baseline], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}
        mock_supabase.insert.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai._capture_screenshot", AsyncMock(return_value=(mock_screenshot, mock_metadata))), \
             patch("src.api.visual_ai._store_screenshot", AsyncMock(return_value="data:image/png;base64,test")), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):

            response = await create_baseline(
                url="https://example.com",
                name="homepage-baseline",
                project_id=sample_baseline_data["project_id"],
                user=mock_user_context
            )

            assert response.version == 4  # Incremented version


class TestGetBaseline:
    """Tests for GET /api/v1/visual/baselines/{baseline_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_baseline_success(self, mock_env_vars, mock_supabase, sample_baseline_data):
        """Test successful baseline retrieval."""
        from src.api.visual_ai import get_baseline

        mock_supabase.select.return_value = {"data": [sample_baseline_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            response = await get_baseline(sample_baseline_data["id"])

            assert response["baseline"]["id"] == sample_baseline_data["id"]
            assert response["baseline"]["name"] == "homepage-baseline"

    @pytest.mark.asyncio
    async def test_get_baseline_not_found(self, mock_env_vars, mock_supabase):
        """Test baseline not found."""
        from src.api.visual_ai import get_baseline

        mock_supabase.select.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_baseline("nonexistent")

            assert exc_info.value.status_code == 404


class TestListBaselines:
    """Tests for GET /api/v1/visual/baselines endpoint."""

    @pytest.mark.asyncio
    async def test_list_baselines_success(
        self, mock_env_vars, mock_supabase, mock_user_context, sample_baseline_data
    ):
        """Test successful baseline listing."""
        from src.api.visual_ai import list_baselines

        mock_supabase.request.return_value = {"data": [sample_baseline_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):

            response = await list_baselines(
                project_id=sample_baseline_data["project_id"],
                user=mock_user_context
            )

            assert len(response["baselines"]) == 1
            assert response["total"] == 1

    @pytest.mark.asyncio
    async def test_list_baselines_empty(self, mock_env_vars, mock_supabase, mock_user_context):
        """Test empty baseline list."""
        from src.api.visual_ai import list_baselines

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):

            response = await list_baselines(
                project_id=str(uuid.uuid4()),
                user=mock_user_context
            )

            assert response["baselines"] == []
            assert response["total"] == 0


# ============================================================================
# Approval/Rejection Tests
# ============================================================================


class TestApproveChanges:
    """Tests for POST /api/v1/visual/comparisons/{comparison_id}/approve endpoint."""

    @pytest.mark.asyncio
    async def test_approve_changes_success(
        self, mock_env_vars, mock_supabase, sample_comparison_data
    ):
        """Test successful change approval."""
        from src.api.visual_ai import ApprovalRequest, approve_changes

        mock_supabase.select.return_value = {"data": [sample_comparison_data], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            request = ApprovalRequest(notes="Looks good")
            response = await approve_changes(sample_comparison_data["id"], request)

            assert response["success"] is True
            assert response["comparison_id"] == sample_comparison_data["id"]

    @pytest.mark.asyncio
    async def test_approve_changes_not_found(self, mock_env_vars, mock_supabase):
        """Test approval fails when comparison not found."""
        from src.api.visual_ai import ApprovalRequest, approve_changes

        mock_supabase.select.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            request = ApprovalRequest(notes="Test")

            with pytest.raises(HTTPException) as exc_info:
                await approve_changes("nonexistent", request)

            assert exc_info.value.status_code == 404


class TestRejectChanges:
    """Tests for POST /api/v1/visual/comparisons/{comparison_id}/reject endpoint."""

    @pytest.mark.asyncio
    async def test_reject_changes_success(
        self, mock_env_vars, mock_supabase, sample_comparison_data
    ):
        """Test successful change rejection."""
        from src.api.visual_ai import RejectionRequest, reject_changes

        mock_supabase.select.return_value = {"data": [sample_comparison_data], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            request = RejectionRequest(notes="Visual regression detected")
            response = await reject_changes(sample_comparison_data["id"], request)

            assert response["success"] is True
            assert response["status"] == "rejected"


# ============================================================================
# Analysis Endpoint Tests
# ============================================================================


class TestAnalyzeSnapshot:
    """Tests for POST /api/v1/visual/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_snapshot_not_found(self, mock_env_vars):
        """Test analysis fails when snapshot not found."""
        from src.api.visual_ai import analyze_snapshot

        # Create a mock Path that properly chains __truediv__ and exists()
        mock_screenshot_path = MagicMock()
        mock_screenshot_path.exists.return_value = False

        mock_screenshots_dir = MagicMock()
        mock_screenshots_dir.__truediv__ = MagicMock(return_value=mock_screenshot_path)

        with patch("src.api.visual_ai.Path") as mock_path_class:
            # Path(settings.output_dir) / "screenshots" returns mock_screenshots_dir
            mock_path_class.return_value.__truediv__ = MagicMock(return_value=mock_screenshots_dir)

            with pytest.raises(HTTPException) as exc_info:
                await analyze_snapshot("nonexistent")

            assert exc_info.value.status_code == 404


class TestAnalyzeAccessibility:
    """Tests for POST /api/v1/visual/accessibility/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_accessibility_analyze_not_found(self, mock_env_vars):
        """Test accessibility analysis fails when snapshot not found."""
        from src.api.visual_ai import analyze_accessibility

        with pytest.raises(HTTPException) as exc_info:
            await analyze_accessibility("nonexistent")

        assert exc_info.value.status_code == 404


# ============================================================================
# AI Explanation Tests
# ============================================================================


class TestExplainChanges:
    """Tests for POST /api/v1/visual/ai/explain endpoint."""

    @pytest.mark.asyncio
    async def test_explain_no_differences(self, mock_env_vars, mock_supabase, sample_comparison_data):
        """Test explanation when no differences."""
        from src.api.visual_ai import explain_changes

        mock_supabase.select.return_value = {"data": [sample_comparison_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            response = await explain_changes(sample_comparison_data["id"])

            assert response["success"] is True
            assert "No visual differences" in response["explanation"]

    @pytest.mark.asyncio
    async def test_explain_comparison_not_found(self, mock_env_vars, mock_supabase):
        """Test explanation fails when comparison not found."""
        from src.api.visual_ai import explain_changes

        mock_supabase.select.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await explain_changes("nonexistent")

            assert exc_info.value.status_code == 404


# ============================================================================
# Query Endpoint Tests
# ============================================================================


class TestListComparisons:
    """Tests for GET /api/v1/visual/comparisons endpoint."""

    @pytest.mark.asyncio
    async def test_list_comparisons_success(
        self, mock_env_vars, mock_supabase, mock_user_context, sample_comparison_data
    ):
        """Test successful comparison listing."""
        from src.api.visual_ai import list_comparisons

        mock_supabase.request.return_value = {"data": [sample_comparison_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):
            # Call without project_id so validation is skipped
            response = await list_comparisons(user=mock_user_context)

            assert len(response["comparisons"]) == 1
            assert response["total"] == 1

    @pytest.mark.asyncio
    async def test_list_comparisons_with_filters(
        self, mock_env_vars, mock_supabase, mock_user_context
    ):
        """Test comparison listing with filters."""
        from src.api.visual_ai import list_comparisons

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.visual_ai.validate_project_ownership", AsyncMock()):

            response = await list_comparisons(
                project_id=str(uuid.uuid4()),
                status="passed",
                user=mock_user_context
            )

            assert response["comparisons"] == []


class TestGetSnapshot:
    """Tests for GET /api/v1/visual/snapshots/{snapshot_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_snapshot_success(self, mock_env_vars, mock_supabase, sample_snapshot_data):
        """Test successful snapshot retrieval."""
        from src.api.visual_ai import get_snapshot

        mock_supabase.select.return_value = {"data": [sample_snapshot_data], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            response = await get_snapshot(sample_snapshot_data["id"])

            assert response["snapshot"]["id"] == sample_snapshot_data["id"]

    @pytest.mark.asyncio
    async def test_get_snapshot_not_found(self, mock_env_vars, mock_supabase):
        """Test snapshot not found."""
        from src.api.visual_ai import get_snapshot

        mock_supabase.select.return_value = {"data": [], "error": None}

        with patch("src.api.visual_ai.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_snapshot("nonexistent")

            assert exc_info.value.status_code == 404


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_capture_request_validation(self, mock_env_vars):
        """Test CaptureRequest validation."""
        from src.api.visual_ai import CaptureRequest

        request = CaptureRequest(url="https://example.com")
        assert request.url == "https://example.com"
        assert request.browser == "chromium"

    def test_viewport_config_validation(self, mock_env_vars):
        """Test ViewportConfig validation."""
        from src.api.visual_ai import ViewportConfig

        viewport = ViewportConfig(width=1920, height=1080)
        assert viewport.width == 1920
        assert viewport.height == 1080

    def test_viewport_config_bounds(self, mock_env_vars):
        """Test ViewportConfig bounds validation."""
        from pydantic import ValidationError
        from src.api.visual_ai import ViewportConfig

        with pytest.raises(ValidationError):
            ViewportConfig(width=100, height=1080)  # Too small

        with pytest.raises(ValidationError):
            ViewportConfig(width=5000, height=1080)  # Too large

    def test_compare_request_validation(self, mock_env_vars):
        """Test CompareRequest validation."""
        from src.api.visual_ai import CompareRequest

        request = CompareRequest(
            baseline_id=str(uuid.uuid4()),
            current_url="https://example.com",
            sensitivity="high"
        )
        assert request.sensitivity == "high"

    def test_approval_request_validation(self, mock_env_vars):
        """Test ApprovalRequest validation."""
        from src.api.visual_ai import ApprovalRequest

        request = ApprovalRequest(
            update_baseline=True,
            notes="Approved"
        )
        assert request.update_baseline is True

    def test_rejection_request_validation(self, mock_env_vars):
        """Test RejectionRequest validation."""
        from src.api.visual_ai import RejectionRequest

        request = RejectionRequest(
            notes="Visual regression detected",
            create_issue=True
        )
        assert request.create_issue is True
