"""Tests for the Projects API module (src/api/projects.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

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
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_project_data():
    """Create sample project data."""
    return {
        "id": "project-123",
        "organization_id": "org-456",
        "name": "Test Project",
        "description": "A test project",
        "app_url": "http://localhost:3000",
        "codebase_path": "/path/to/code",
        "repository_url": "https://github.com/test/repo",
        "settings": {"key": "value"},
        "is_active": True,
        "last_run_at": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": None,
    }


# ============================================================================
# Validation Helper Tests
# ============================================================================

class TestValidateUrl:
    """Tests for validate_url helper function."""

    def test_validate_url_http(self, mock_env_vars):
        """Test validate_url accepts http URLs."""
        from src.api.projects import validate_url

        result = validate_url("http://localhost:3000", "app_url")
        assert result == "http://localhost:3000"

    def test_validate_url_https(self, mock_env_vars):
        """Test validate_url accepts https URLs."""
        from src.api.projects import validate_url

        result = validate_url("https://example.com", "app_url")
        assert result == "https://example.com"

    def test_validate_url_none(self, mock_env_vars):
        """Test validate_url accepts None."""
        from src.api.projects import validate_url

        result = validate_url(None, "app_url")
        assert result is None

    def test_validate_url_invalid(self, mock_env_vars):
        """Test validate_url rejects invalid URLs."""
        from src.api.projects import validate_url

        with pytest.raises(ValueError) as exc_info:
            validate_url("ftp://example.com", "app_url")

        assert "must start with http://" in str(exc_info.value)


# ============================================================================
# Model Tests
# ============================================================================

class TestRequestModels:
    """Tests for request model validation."""

    def test_create_project_request(self, mock_env_vars):
        """Test CreateProjectRequest model."""
        from src.api.projects import CreateProjectRequest

        request = CreateProjectRequest(
            name="My Project",
            description="A new project",
            app_url="http://localhost:3000",
            codebase_path="/path/to/code",
            repository_url="https://github.com/test/repo",
            settings={"key": "value"},
        )

        assert request.name == "My Project"
        assert request.app_url == "http://localhost:3000"

    def test_create_project_request_minimal(self, mock_env_vars):
        """Test CreateProjectRequest with only required fields."""
        from src.api.projects import CreateProjectRequest

        request = CreateProjectRequest(name="Minimal Project")

        assert request.name == "Minimal Project"
        assert request.description is None
        assert request.app_url is None

    def test_create_project_request_name_validation(self, mock_env_vars):
        """Test CreateProjectRequest name length validation."""
        from src.api.projects import CreateProjectRequest

        # Name too short
        with pytest.raises(ValidationError):
            CreateProjectRequest(name="A")

        # Name too long
        with pytest.raises(ValidationError):
            CreateProjectRequest(name="x" * 101)

    def test_create_project_request_url_validation(self, mock_env_vars):
        """Test CreateProjectRequest URL validation."""
        from src.api.projects import CreateProjectRequest

        with pytest.raises(ValidationError):
            CreateProjectRequest(
                name="Project",
                app_url="invalid-url",
            )

    def test_update_project_request(self, mock_env_vars):
        """Test UpdateProjectRequest model."""
        from src.api.projects import UpdateProjectRequest

        request = UpdateProjectRequest(
            name="Updated Name",
            is_active=False,
        )

        assert request.name == "Updated Name"
        assert request.is_active is False

    def test_update_project_request_empty(self, mock_env_vars):
        """Test UpdateProjectRequest with no fields."""
        from src.api.projects import UpdateProjectRequest

        request = UpdateProjectRequest()

        assert request.name is None
        assert request.description is None
        assert request.is_active is None


class TestResponseModels:
    """Tests for response model validation."""

    def test_project_response(self, mock_env_vars):
        """Test ProjectResponse model."""
        from src.api.projects import ProjectResponse

        response = ProjectResponse(
            id="project-123",
            organization_id="org-456",
            name="Test Project",
            description="Description",
            app_url="http://localhost:3000",
            codebase_path="/path",
            repository_url="https://github.com/test/repo",
            settings={},
            is_active=True,
            test_count=5,
            last_run_at=None,
            created_at="2024-01-01T00:00:00Z",
            updated_at=None,
        )

        assert response.id == "project-123"
        assert response.test_count == 5

    def test_project_list_response(self, mock_env_vars):
        """Test ProjectListResponse model."""
        from src.api.projects import ProjectListResponse

        response = ProjectListResponse(
            id="project-123",
            organization_id="org-456",
            name="Test Project",
            description=None,
            app_url="http://localhost:3000",
            is_active=True,
            test_count=10,
            last_run_at=None,
            created_at="2024-01-01T00:00:00Z",
        )

        assert response.test_count == 10


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.asyncio
    async def test_get_project_test_count(self, mock_env_vars, mock_supabase):
        """Test get_project_test_count returns correct count."""
        from src.api.projects import get_project_test_count

        mock_supabase.request = AsyncMock(return_value={
            "data": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase):
            count = await get_project_test_count("project-123")
            assert count == 3

    @pytest.mark.asyncio
    async def test_get_project_test_count_empty(self, mock_env_vars, mock_supabase):
        """Test get_project_test_count with no tests."""
        from src.api.projects import get_project_test_count

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase):
            count = await get_project_test_count("project-123")
            assert count == 0

    @pytest.mark.asyncio
    async def test_get_project_test_counts_batch(self, mock_env_vars, mock_supabase):
        """Test get_project_test_counts_batch returns correct mapping."""
        from src.api.projects import get_project_test_counts_batch

        mock_supabase.rpc = AsyncMock(return_value={
            "data": [
                {"project_id": "project-1", "count": 5},
                {"project_id": "project-2", "count": 10},
            ],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase):
            result = await get_project_test_counts_batch(["project-1", "project-2"])

            assert result["project-1"] == 5
            assert result["project-2"] == 10

    @pytest.mark.asyncio
    async def test_get_project_test_counts_batch_empty(self, mock_env_vars):
        """Test get_project_test_counts_batch with empty list."""
        from src.api.projects import get_project_test_counts_batch

        result = await get_project_test_counts_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_project_test_counts_batch_fallback(self, mock_env_vars, mock_supabase):
        """Test get_project_test_counts_batch falls back on RPC error."""
        from src.api.projects import get_project_test_counts_batch

        mock_supabase.rpc = AsyncMock(return_value={"data": None, "error": "RPC error"})
        mock_supabase.request = AsyncMock(return_value={
            "data": [{"id": "t1"}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase):
            result = await get_project_test_counts_batch(["project-1"])
            assert "project-1" in result

    @pytest.mark.asyncio
    async def test_verify_project_access_success(
        self, mock_env_vars, mock_supabase, mock_request
    ):
        """Test verify_project_access when user has access."""
        from src.api.projects import verify_project_access

        mock_supabase.request = AsyncMock(return_value={
            "data": [{"id": "project-123", "organization_id": "org-456", "name": "Test"}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            project = await verify_project_access(
                "project-123", "user-123", "test@example.com", mock_request
            )
            assert project["id"] == "project-123"

    @pytest.mark.asyncio
    async def test_verify_project_access_not_found(
        self, mock_env_vars, mock_supabase, mock_request
    ):
        """Test verify_project_access when project not found."""
        from src.api.projects import verify_project_access

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await verify_project_access("nonexistent", "user-123")

            assert exc_info.value.status_code == 404


# ============================================================================
# Organization-Scoped Endpoint Tests
# ============================================================================

class TestListOrganizationProjects:
    """Tests for GET /api/v1/organizations/{org_id}/projects endpoint."""

    @pytest.mark.asyncio
    async def test_list_organization_projects_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project listing for organization."""
        from src.api.projects import list_organization_projects

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-123", "count": 5}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            response = await list_organization_projects("org-456", mock_request)

            assert len(response) == 1
            assert response[0].id == "project-123"
            assert response[0].test_count == 5

    @pytest.mark.asyncio
    async def test_list_organization_projects_with_filters(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test project listing with filters."""
        from src.api.projects import list_organization_projects

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})
        mock_supabase.rpc = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            response = await list_organization_projects(
                "org-456", mock_request,
                is_active=True,
                limit=10,
                offset=5,
            )

            assert response == []

    @pytest.mark.asyncio
    async def test_list_organization_projects_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test project listing handles errors."""
        from src.api.projects import list_organization_projects

        mock_supabase.request = AsyncMock(return_value={"data": None, "error": "DB error"})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            with pytest.raises(HTTPException) as exc_info:
                await list_organization_projects("org-456", mock_request)

            assert exc_info.value.status_code == 500


class TestCreateOrganizationProject:
    """Tests for POST /api/v1/organizations/{org_id}/projects endpoint."""

    @pytest.mark.asyncio
    async def test_create_organization_project_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project creation."""
        from src.api.projects import CreateProjectRequest, create_organization_project

        mock_supabase.insert = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })

        body = CreateProjectRequest(
            name="Test Project",
            description="A test project",
            app_url="http://localhost:3000",
        )

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()):
            response = await create_organization_project("org-456", body, mock_request)

            assert response.id == "project-123"
            assert response.name == "Test Project"
            assert response.test_count == 0

    @pytest.mark.asyncio
    async def test_create_organization_project_insert_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test project creation handles insert errors."""
        from src.api.projects import CreateProjectRequest, create_organization_project

        mock_supabase.insert = AsyncMock(return_value={
            "data": None,
            "error": "Insert failed",
        })

        body = CreateProjectRequest(name="Test")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            with pytest.raises(HTTPException) as exc_info:
                await create_organization_project("org-456", body, mock_request)

            assert exc_info.value.status_code == 500


# ============================================================================
# Project Endpoints Tests
# ============================================================================

class TestListProjects:
    """Tests for GET /api/v1/projects endpoint."""

    @pytest.mark.asyncio
    async def test_list_projects_with_org_context(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test listing projects with organization context."""
        from src.api.projects import list_projects

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-123", "count": 5}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.get_current_organization_id", AsyncMock(return_value="org-456")), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            response = await list_projects(mock_request)

            assert len(response) == 1

    @pytest.mark.asyncio
    async def test_list_projects_without_org_context(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test listing projects without organization context gets all user's projects."""
        from src.api.projects import list_projects

        mock_supabase.request = AsyncMock(side_effect=[
            # Memberships query
            {"data": [{"organization_id": "org-456"}], "error": None},
            # Projects query
            {"data": [sample_project_data], "error": None},
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-123", "count": 5}],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.get_current_organization_id", AsyncMock(return_value=None)):
            response = await list_projects(mock_request)

            assert len(response) == 1

    @pytest.mark.asyncio
    async def test_list_projects_no_memberships(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing projects when user has no org memberships."""
        from src.api.projects import list_projects

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.get_current_organization_id", AsyncMock(return_value=None)):
            response = await list_projects(mock_request)

            assert response == []


class TestCreateProject:
    """Tests for POST /api/v1/projects endpoint."""

    @pytest.mark.asyncio
    async def test_create_project_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project creation with org context."""
        from src.api.projects import CreateProjectRequest, create_project

        mock_supabase.insert = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })

        body = CreateProjectRequest(name="Test Project")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.require_organization_id", AsyncMock(return_value="org-456")), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()):
            response = await create_project(body, mock_request)

            assert response.id == "project-123"


class TestGetProject:
    """Tests for GET /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_project_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project retrieval."""
        from src.api.projects import get_project

        mock_supabase.request = AsyncMock(side_effect=[
            # verify_project_access
            {"data": [sample_project_data], "error": None},
            # get_project_test_count
            {"data": [{"id": "t1"}, {"id": "t2"}], "error": None},
        ])

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            response = await get_project("project-123", mock_request)

            assert response.id == "project-123"
            assert response.name == "Test Project"
            assert response.test_count == 2

    @pytest.mark.asyncio
    async def test_get_project_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test get project returns 404 when not found."""
        from src.api.projects import get_project

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_project("nonexistent", mock_request)

            assert exc_info.value.status_code == 404


class TestUpdateProject:
    """Tests for PUT/PATCH /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_project_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project update."""
        from src.api.projects import UpdateProjectRequest, update_project

        updated_data = {**sample_project_data, "name": "Updated Project"}

        mock_supabase.request = AsyncMock(side_effect=[
            # verify_project_access
            {"data": [sample_project_data], "error": None},
            # get_project_test_count (for get_project call)
            {"data": [], "error": None},
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateProjectRequest(name="Updated Project")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()), \
             patch("src.api.projects.verify_project_access", AsyncMock(return_value=updated_data)):
            await update_project("project-123", body, mock_request)

            mock_supabase.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_project_partial(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test partial project update."""
        from src.api.projects import UpdateProjectRequest, update_project

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_project_data], "error": None},
            {"data": [], "error": None},
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateProjectRequest(is_active=False)

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()), \
             patch("src.api.projects.verify_project_access", AsyncMock(return_value=sample_project_data)):
            await update_project("project-123", body, mock_request)

            # Verify only is_active and updated_at were updated
            call_args = mock_supabase.update.call_args
            update_data = call_args[0][2]
            assert "is_active" in update_data
            assert update_data["is_active"] is False
            assert "name" not in update_data

    @pytest.mark.asyncio
    async def test_update_project_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test project update handles errors."""
        from src.api.projects import UpdateProjectRequest, update_project

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })
        mock_supabase.update = AsyncMock(return_value={"data": None, "error": "Update failed"})

        body = UpdateProjectRequest(name="Updated")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            with pytest.raises(HTTPException) as exc_info:
                await update_project("project-123", body, mock_request)

            assert exc_info.value.status_code == 500


class TestDeleteProject:
    """Tests for DELETE /api/v1/projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_project_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test successful project deletion."""
        from src.api.projects import delete_project

        mock_supabase.request = AsyncMock(side_effect=[
            # verify_project_access
            {"data": [sample_project_data], "error": None},
            # DELETE
            {"data": [], "error": None},
        ])

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()):
            response = await delete_project("project-123", mock_request)

            assert response["success"] is True
            assert "Test Project" in response["message"]

    @pytest.mark.asyncio
    async def test_delete_project_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test delete project returns 404 when not found."""
        from src.api.projects import delete_project

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await delete_project("nonexistent", mock_request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test delete project handles errors."""
        from src.api.projects import delete_project

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_project_data], "error": None},
            {"data": None, "error": "Delete failed"},
        ])

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))):
            with pytest.raises(HTTPException) as exc_info:
                await delete_project("project-123", mock_request)

            assert exc_info.value.status_code == 500


# ============================================================================
# Access Control Tests
# ============================================================================

class TestAccessControl:
    """Tests for access control in project operations."""

    @pytest.mark.asyncio
    async def test_create_project_requires_admin(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test that create project requires admin or owner role."""
        from src.api.projects import CreateProjectRequest, create_organization_project

        body = CreateProjectRequest(name="Test")

        with patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Insufficient permissions"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await create_organization_project("org-456", body, mock_request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_update_project_requires_admin(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test that update project requires admin or owner role."""
        from src.api.projects import UpdateProjectRequest, update_project

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })

        body = UpdateProjectRequest(name="Updated")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_project_access", AsyncMock(return_value=sample_project_data)), \
             patch("src.api.projects.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Insufficient permissions"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await update_project("project-123", body, mock_request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_project_requires_owner(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test that delete project requires owner role."""
        from src.api.projects import delete_project

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_project_access", AsyncMock(return_value=sample_project_data)), \
             patch("src.api.projects.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Owner role required"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await delete_project("project-123", mock_request)

            assert exc_info.value.status_code == 403


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestProjectIntegration:
    """Integration-style tests for project operations."""

    @pytest.mark.asyncio
    async def test_create_and_list_project_flow(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_project_data
    ):
        """Test creating and then listing a project."""
        from src.api.projects import (
            CreateProjectRequest,
            create_organization_project,
            list_organization_projects,
        )

        mock_supabase.insert = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })
        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_project_data],
            "error": None,
        })
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-123", "count": 0}],
            "error": None,
        })

        body = CreateProjectRequest(name="Test Project")

        with patch("src.api.projects.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.projects.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_org_access", AsyncMock(return_value=({}, "org-456"))), \
             patch("src.api.projects.log_audit", AsyncMock()):
            # Create project
            created = await create_organization_project("org-456", body, mock_request)
            assert created.id == "project-123"

            # List projects
            projects = await list_organization_projects("org-456", mock_request)
            assert len(projects) == 1
            assert projects[0].id == "project-123"
