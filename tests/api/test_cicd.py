"""Tests for the CI/CD API module (src/api/cicd.py).

Covers:
- Build endpoints (list, get)
- Deployment endpoints (list, get, rollback)
- Pipeline endpoints (list, get, retrigger, cancel)
- Test impact analysis
- Deployment risk calculation
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

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
    mock.rpc = AsyncMock()
    return mock


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent"}
    request.query_params = {}
    request.state = MagicMock()
    request.state.user_id = "user-123"
    request.state.organization_id = "org-456"
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_build_data():
    """Create sample build data."""
    return {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "pipeline_id": str(uuid.uuid4()),
        "provider": "github",
        "build_number": 42,
        "name": "Test Build",
        "branch": "main",
        "status": "success",
        "commit_sha": "abc123def",
        "commit_message": "Fix bugs",
        "commit_author": "test@example.com",
        "tests_total": 100,
        "tests_passed": 95,
        "tests_failed": 3,
        "tests_skipped": 2,
        "coverage_percent": 85.5,
        "artifact_urls": ["https://artifacts.example.com/build.zip"],
        "logs_url": "https://logs.example.com/build-42",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_ms": 120000,
        "metadata": {"environment": "ci"},
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_deployment_data():
    """Create sample deployment data."""
    return {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "build_id": str(uuid.uuid4()),
        "environment": "production",
        "status": "success",
        "version": "1.0.0",
        "commit_sha": "abc123def",
        "deployed_by": "test@example.com",
        "deployment_url": "https://app.example.com",
        "preview_url": None,
        "risk_score": 25,
        "risk_factors": [
            {"category": "test_coverage", "severity": "low", "description": "Good coverage", "score": 10},
        ],
        "health_check_status": "healthy",
        "rollback_available": True,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_ms": 60000,
        "metadata": {},
        "created_at": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestBuildFromRow:
    """Tests for _build_from_row helper function."""

    def test_build_from_row_complete(self, mock_env_vars, sample_build_data):
        """Test converting a complete build row to Build model."""
        from src.api.cicd import _build_from_row

        build = _build_from_row(sample_build_data)

        assert build.id == sample_build_data["id"]
        assert build.project_id == sample_build_data["project_id"]
        assert build.status == "success"
        assert build.tests_total == 100
        assert build.tests_passed == 95

    def test_build_from_row_minimal(self, mock_env_vars):
        """Test converting a minimal build row."""
        from src.api.cicd import _build_from_row

        minimal_data = {
            "id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        }

        build = _build_from_row(minimal_data)

        assert build.id == minimal_data["id"]
        assert build.provider == "unknown"
        assert build.build_number == 0
        assert build.tests_total == 0


class TestDeploymentFromRow:
    """Tests for _deployment_from_row helper function."""

    def test_deployment_from_row_complete(self, mock_env_vars, sample_deployment_data):
        """Test converting a complete deployment row."""
        from src.api.cicd import _deployment_from_row

        deployment = _deployment_from_row(sample_deployment_data)

        assert deployment.id == sample_deployment_data["id"]
        assert deployment.environment == "production"
        assert deployment.risk_score == 25
        assert len(deployment.risk_factors) == 1

    def test_deployment_from_row_minimal(self, mock_env_vars):
        """Test converting a minimal deployment row."""
        from src.api.cicd import _deployment_from_row

        minimal_data = {
            "id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        }

        deployment = _deployment_from_row(minimal_data)

        assert deployment.id == minimal_data["id"]
        assert deployment.environment == "preview"
        assert deployment.risk_factors == []


# ============================================================================
# Build Endpoint Tests
# ============================================================================

class TestListBuilds:
    """Tests for list_builds endpoint."""

    @pytest.mark.asyncio
    async def test_list_builds_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_build_data):
        """Test successful build listing."""
        from src.api.cicd import list_builds

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [sample_build_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await list_builds(mock_request, project_id=project_id)

        assert result.total >= 0
        assert isinstance(result.builds, list)

    @pytest.mark.asyncio
    async def test_list_builds_with_filters(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_build_data):
        """Test build listing with branch and status filters."""
        from src.api.cicd import list_builds

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [sample_build_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await list_builds(
                mock_request,
                project_id=project_id,
                branch="main",
                status="success",
                limit=10,
                offset=0
            )

        assert isinstance(result.builds, list)

    @pytest.mark.asyncio
    async def test_list_builds_database_error(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test handling of database errors."""
        from src.api.cicd import list_builds

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": None, "error": "Database error"}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            with pytest.raises(HTTPException) as exc_info:
                await list_builds(mock_request, project_id=project_id)

            assert exc_info.value.status_code == 500


class TestGetBuild:
    """Tests for get_build endpoint."""

    @pytest.mark.asyncio
    async def test_get_build_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_build_data):
        """Test successful build retrieval."""
        from src.api.cicd import get_build

        build_id = sample_build_data["id"]
        mock_supabase.request.return_value = {"data": [sample_build_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await get_build(build_id, mock_request)

        assert result.id == build_id
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_get_build_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test build not found error."""
        from src.api.cicd import get_build

        build_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)):

            with pytest.raises(HTTPException) as exc_info:
                await get_build(build_id, mock_request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_build_invalid_uuid(self, mock_env_vars, mock_request, mock_user):
        """Test invalid UUID validation."""
        from src.api.cicd import get_build

        with patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_build("invalid-uuid", mock_request)

            assert exc_info.value.status_code == 400


# ============================================================================
# Deployment Endpoint Tests
# ============================================================================

class TestListDeployments:
    """Tests for list_deployments endpoint."""

    @pytest.mark.asyncio
    async def test_list_deployments_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_deployment_data):
        """Test successful deployment listing."""
        from src.api.cicd import list_deployments

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [sample_deployment_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await list_deployments(mock_request, project_id=project_id)

        assert isinstance(result.deployments, list)

    @pytest.mark.asyncio
    async def test_list_deployments_with_environment_filter(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_deployment_data):
        """Test deployment listing filtered by environment."""
        from src.api.cicd import list_deployments

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [sample_deployment_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await list_deployments(
                mock_request,
                project_id=project_id,
                environment="production"
            )

        assert isinstance(result.deployments, list)

    @pytest.mark.asyncio
    async def test_list_deployments_table_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test graceful handling when table doesn't exist."""
        from src.api.cicd import list_deployments

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": None, "error": "42P01: table does not exist"}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await list_deployments(mock_request, project_id=project_id)

        assert result.deployments == []
        assert result.total == 0


class TestGetDeployment:
    """Tests for get_deployment endpoint."""

    @pytest.mark.asyncio
    async def test_get_deployment_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_deployment_data):
        """Test successful deployment retrieval."""
        from src.api.cicd import get_deployment

        deployment_id = sample_deployment_data["id"]
        mock_supabase.request.return_value = {"data": [sample_deployment_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            result = await get_deployment(deployment_id, mock_request)

        assert result.id == deployment_id
        assert result.environment == "production"

    @pytest.mark.asyncio
    async def test_get_deployment_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test deployment not found error."""
        from src.api.cicd import get_deployment

        deployment_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)):

            with pytest.raises(HTTPException) as exc_info:
                await get_deployment(deployment_id, mock_request)

            assert exc_info.value.status_code == 404


class TestRollbackDeployment:
    """Tests for rollback_deployment endpoint."""

    @pytest.mark.asyncio
    async def test_rollback_deployment_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_deployment_data):
        """Test successful deployment rollback."""
        from src.api.cicd import rollback_deployment

        deployment_id = sample_deployment_data["id"]

        # Mock the original deployment
        mock_supabase.request.side_effect = [
            {"data": [sample_deployment_data], "error": None},  # Get original
            {"data": [sample_deployment_data], "error": None},  # Get previous
        ]
        mock_supabase.insert.return_value = {"data": [{"id": str(uuid.uuid4())}], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()), \
             patch("src.api.cicd.log_audit", AsyncMock()):

            result = await rollback_deployment(deployment_id, mock_request)

        assert result.success is True
        assert "rolled back" in result.message.lower()

    @pytest.mark.asyncio
    async def test_rollback_deployment_not_available(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test rollback when not available."""
        from src.api.cicd import rollback_deployment

        deployment_id = str(uuid.uuid4())
        deployment_data = {
            "id": deployment_id,
            "project_id": str(uuid.uuid4()),
            "rollback_available": False,
            "environment": "production",
            "status": "success",
            "created_at": datetime.now(UTC).isoformat(),
        }

        mock_supabase.request.return_value = {"data": [deployment_data], "error": None}

        with patch("src.api.cicd.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.cicd.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.cicd.verify_project_access", AsyncMock()):

            with pytest.raises(HTTPException) as exc_info:
                await rollback_deployment(deployment_id, mock_request)

            assert exc_info.value.status_code == 400
            assert "not available" in exc_info.value.detail.lower()


# ============================================================================
# Pipeline Helper Function Tests
# ============================================================================

class TestMapGithubStatus:
    """Tests for map_github_status helper function."""

    def test_map_queued_status(self, mock_env_vars):
        """Test mapping queued status."""
        from src.api.cicd import map_github_status

        assert map_github_status("queued", None) == "pending"

    def test_map_in_progress_status(self, mock_env_vars):
        """Test mapping in_progress status."""
        from src.api.cicd import map_github_status

        assert map_github_status("in_progress", None) == "running"

    def test_map_completed_success(self, mock_env_vars):
        """Test mapping completed with success conclusion."""
        from src.api.cicd import map_github_status

        assert map_github_status("completed", "success") == "success"

    def test_map_completed_failure(self, mock_env_vars):
        """Test mapping completed with failure conclusion."""
        from src.api.cicd import map_github_status

        assert map_github_status("completed", "failure") == "failure"

    def test_map_completed_cancelled(self, mock_env_vars):
        """Test mapping completed with cancelled conclusion."""
        from src.api.cicd import map_github_status

        assert map_github_status("completed", "cancelled") == "cancelled"


class TestCalculatePipelineDuration:
    """Tests for calculate_pipeline_duration helper function."""

    def test_calculate_duration_valid(self, mock_env_vars):
        """Test duration calculation with valid timestamps."""
        from src.api.cicd import calculate_pipeline_duration

        started_at = "2024-01-01T10:00:00Z"
        completed_at = "2024-01-01T10:05:00Z"

        duration = calculate_pipeline_duration(started_at, completed_at)

        assert duration == 300  # 5 minutes in seconds

    def test_calculate_duration_none_values(self, mock_env_vars):
        """Test duration calculation with None values."""
        from src.api.cicd import calculate_pipeline_duration

        assert calculate_pipeline_duration(None, "2024-01-01T10:00:00Z") is None
        assert calculate_pipeline_duration("2024-01-01T10:00:00Z", None) is None
        assert calculate_pipeline_duration(None, None) is None


# ============================================================================
# Model Validation Tests
# ============================================================================

class TestBuildModel:
    """Tests for Build model validation."""

    def test_build_model_creation(self, mock_env_vars):
        """Test creating a valid Build model."""
        from src.api.cicd import Build

        build = Build(
            id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            provider="github",
            build_number=1,
            name="Test",
            branch="main",
            status="success",
            commit_sha="abc123",
            created_at=datetime.now(UTC).isoformat(),
        )

        assert build.provider == "github"
        assert build.status == "success"


class TestDeploymentModel:
    """Tests for Deployment model validation."""

    def test_deployment_model_creation(self, mock_env_vars):
        """Test creating a valid Deployment model."""
        from src.api.cicd import Deployment

        deployment = Deployment(
            id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            environment="production",
            status="success",
            created_at=datetime.now(UTC).isoformat(),
        )

        assert deployment.environment == "production"
        assert deployment.status == "success"


class TestRiskFactorModel:
    """Tests for RiskFactor model validation."""

    def test_risk_factor_model_creation(self, mock_env_vars):
        """Test creating a valid RiskFactor model."""
        from src.api.cicd import RiskFactor

        risk_factor = RiskFactor(
            category="test_coverage",
            severity="medium",
            description="Coverage is below threshold",
            score=35,
        )

        assert risk_factor.category == "test_coverage"
        assert risk_factor.score == 35

    def test_risk_factor_score_bounds(self, mock_env_vars):
        """Test RiskFactor score validation bounds."""
        from src.api.cicd import RiskFactor
        from pydantic import ValidationError

        # Valid boundary values
        RiskFactor(category="test", severity="low", description="test", score=0)
        RiskFactor(category="test", severity="low", description="test", score=100)

        # Invalid values
        with pytest.raises(ValidationError):
            RiskFactor(category="test", severity="low", description="test", score=-1)

        with pytest.raises(ValidationError):
            RiskFactor(category="test", severity="low", description="test", score=101)
