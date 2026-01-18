"""Tests for the Tests CRUD API module (src/api/tests.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
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
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    return {
        "id": "test-123",
        "project_id": "project-456",
        "name": "Login Test",
        "description": "Test login functionality",
        "steps": [
            {"action": "navigate", "target": "/login", "description": "Go to login page"},
            {"action": "fill", "target": "#email", "value": "test@example.com"},
            {"action": "click", "target": "#submit"},
        ],
        "tags": ["auth", "login"],
        "priority": "high",
        "is_active": True,
        "source": "manual",
        "created_by": "user-123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": None,
    }


# ============================================================================
# Model Tests
# ============================================================================

class TestRequestModels:
    """Tests for request model validation."""

    def test_test_step_model(self, mock_env_vars):
        """Test TestStep model."""
        from src.api.tests import TestStep

        step = TestStep(
            action="click",
            target="#submit-btn",
            value=None,
            description="Click submit button",
        )

        assert step.action == "click"
        assert step.target == "#submit-btn"
        assert step.description == "Click submit button"

    def test_create_test_request(self, mock_env_vars):
        """Test CreateTestRequest model."""
        from src.api.tests import CreateTestRequest

        request = CreateTestRequest(
            project_id="project-123",
            name="Login Test",
            description="Test login flow",
            steps=[{"action": "click", "target": "#submit"}],
            tags=["auth", "login"],
            priority="high",
            is_active=True,
            source="manual",
        )

        assert request.project_id == "project-123"
        assert request.name == "Login Test"
        assert request.priority == "high"
        assert len(request.steps) == 1

    def test_create_test_request_minimal(self, mock_env_vars):
        """Test CreateTestRequest with only required fields."""
        from src.api.tests import CreateTestRequest

        request = CreateTestRequest(
            project_id="project-123",
            name="Test",
        )

        assert request.project_id == "project-123"
        assert request.name == "Test"
        assert request.priority == "medium"  # default
        assert request.is_active is True  # default
        assert request.source == "manual"  # default

    def test_create_test_request_name_length_validation(self, mock_env_vars):
        """Test CreateTestRequest name length validation."""
        from src.api.tests import CreateTestRequest
        from pydantic import ValidationError

        # Empty name should fail
        with pytest.raises(ValidationError):
            CreateTestRequest(project_id="project-123", name="")

        # Name too long should fail
        with pytest.raises(ValidationError):
            CreateTestRequest(project_id="project-123", name="x" * 256)

    def test_update_test_request(self, mock_env_vars):
        """Test UpdateTestRequest model."""
        from src.api.tests import UpdateTestRequest

        request = UpdateTestRequest(
            name="Updated Test Name",
            priority="critical",
            is_active=False,
        )

        assert request.name == "Updated Test Name"
        assert request.priority == "critical"
        assert request.is_active is False
        assert request.description is None

    def test_update_test_request_empty(self, mock_env_vars):
        """Test UpdateTestRequest with no fields."""
        from src.api.tests import UpdateTestRequest

        request = UpdateTestRequest()

        assert request.name is None
        assert request.description is None
        assert request.steps is None

    def test_bulk_delete_request(self, mock_env_vars):
        """Test BulkDeleteRequest model."""
        from src.api.tests import BulkDeleteRequest

        request = BulkDeleteRequest(
            test_ids=["test-1", "test-2", "test-3"]
        )

        assert len(request.test_ids) == 3

    def test_bulk_delete_request_empty_fails(self, mock_env_vars):
        """Test BulkDeleteRequest with empty list fails."""
        from src.api.tests import BulkDeleteRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BulkDeleteRequest(test_ids=[])

    def test_bulk_update_request(self, mock_env_vars):
        """Test BulkUpdateRequest model."""
        from src.api.tests import BulkUpdateRequest

        request = BulkUpdateRequest(
            test_ids=["test-1", "test-2"],
            is_active=True,
            priority="high",
            tags_add=["new-tag"],
            tags_remove=["old-tag"],
        )

        assert len(request.test_ids) == 2
        assert request.is_active is True
        assert request.priority == "high"


class TestResponseModels:
    """Tests for response model creation."""

    def test_test_response(self, mock_env_vars):
        """Test TestResponse model."""
        from src.api.tests import TestResponse

        response = TestResponse(
            id="test-123",
            project_id="project-456",
            name="Login Test",
            description="Test login functionality",
            steps=[{"action": "click", "target": "#submit"}],
            tags=["auth"],
            priority="high",
            is_active=True,
            source="manual",
            created_by="user-123",
            created_at="2024-01-01T00:00:00Z",
            updated_at=None,
        )

        assert response.id == "test-123"
        assert response.name == "Login Test"

    def test_test_list_response(self, mock_env_vars):
        """Test TestListResponse model."""
        from src.api.tests import TestListResponse

        response = TestListResponse(
            id="test-123",
            project_id="project-456",
            name="Login Test",
            description=None,
            tags=["auth"],
            priority="high",
            is_active=True,
            source="manual",
            step_count=5,
            created_at="2024-01-01T00:00:00Z",
        )

        assert response.step_count == 5

    def test_test_list_paginated_response(self, mock_env_vars):
        """Test TestListPaginatedResponse model."""
        from src.api.tests import TestListPaginatedResponse, TestListResponse

        response = TestListPaginatedResponse(
            tests=[
                TestListResponse(
                    id="test-1",
                    project_id="project-1",
                    name="Test 1",
                    description=None,
                    tags=[],
                    priority="medium",
                    is_active=True,
                    source="manual",
                    step_count=3,
                    created_at="2024-01-01T00:00:00Z",
                )
            ],
            total=100,
            limit=50,
            offset=0,
        )

        assert len(response.tests) == 1
        assert response.total == 100
        assert response.limit == 50
        assert response.offset == 0


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.asyncio
    async def test_verify_test_access_success(self, mock_env_vars, mock_supabase, mock_request):
        """Test verify_test_access when user has access."""
        from src.api.tests import verify_test_access

        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "test-123",
                "project_id": "project-456",
                "name": "Test",
            }],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            test = await verify_test_access("test-123", "user-123", "test@example.com", mock_request)

            assert test["id"] == "test-123"

    @pytest.mark.asyncio
    async def test_verify_test_access_not_found(self, mock_env_vars, mock_supabase, mock_request):
        """Test verify_test_access when test doesn't exist."""
        from src.api.tests import verify_test_access

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await verify_test_access("nonexistent", "user-123")

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_get_project_org_id_success(self, mock_env_vars, mock_supabase):
        """Test get_project_org_id returns correct org_id."""
        from src.api.tests import get_project_org_id

        mock_supabase.request = AsyncMock(return_value={
            "data": [{"organization_id": "org-123"}],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            org_id = await get_project_org_id("project-456")

            assert org_id == "org-123"

    @pytest.mark.asyncio
    async def test_get_project_org_id_not_found(self, mock_env_vars, mock_supabase):
        """Test get_project_org_id when project doesn't exist."""
        from src.api.tests import get_project_org_id

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_project_org_id("nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_project_org_ids_batch(self, mock_env_vars, mock_supabase):
        """Test get_project_org_ids_batch returns correct mapping."""
        from src.api.tests import get_project_org_ids_batch

        mock_supabase.rpc = AsyncMock(return_value={
            "data": [
                {"project_id": "project-1", "organization_id": "org-1"},
                {"project_id": "project-2", "organization_id": "org-2"},
            ],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            result = await get_project_org_ids_batch(["project-1", "project-2"])

            assert result["project-1"] == "org-1"
            assert result["project-2"] == "org-2"

    @pytest.mark.asyncio
    async def test_get_project_org_ids_batch_empty(self, mock_env_vars):
        """Test get_project_org_ids_batch with empty list."""
        from src.api.tests import get_project_org_ids_batch

        result = await get_project_org_ids_batch([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_verify_test_access_success(self, mock_env_vars, mock_supabase):
        """Test batch_verify_test_access with all tests accessible."""
        from src.api.tests import batch_verify_test_access

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query
            {
                "data": [
                    {"id": "test-1", "project_id": "project-1", "name": "Test 1"},
                    {"id": "test-2", "project_id": "project-1", "name": "Test 2"},
                ],
                "error": None,
            },
            # Membership query
            {
                "data": [{"organization_id": "org-1"}],
                "error": None,
            },
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            accessible, failed = await batch_verify_test_access(
                ["test-1", "test-2"], "user-123"
            )

            assert "test-1" in accessible
            assert "test-2" in accessible
            assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_batch_verify_test_access_some_missing(self, mock_env_vars, mock_supabase):
        """Test batch_verify_test_access with some tests not found."""
        from src.api.tests import batch_verify_test_access

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query - only test-1 exists
            {
                "data": [{"id": "test-1", "project_id": "project-1", "name": "Test 1"}],
                "error": None,
            },
            # Membership query
            {
                "data": [{"organization_id": "org-1"}],
                "error": None,
            },
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            accessible, failed = await batch_verify_test_access(
                ["test-1", "test-2"], "user-123"
            )

            assert "test-1" in accessible
            assert "test-2" in failed
            assert "not found" in failed["test-2"].lower()

    @pytest.mark.asyncio
    async def test_batch_insert_audit_logs_success(self, mock_env_vars, mock_supabase):
        """Test batch_insert_audit_logs inserts entries."""
        from src.api.tests import batch_insert_audit_logs

        mock_supabase.insert = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase):
            await batch_insert_audit_logs([
                {"action": "test.delete", "resource_id": "test-1"},
                {"action": "test.delete", "resource_id": "test-2"},
            ])

            mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_insert_audit_logs_empty(self, mock_env_vars):
        """Test batch_insert_audit_logs with empty list does nothing."""
        from src.api.tests import batch_insert_audit_logs

        # Should not raise any exception
        await batch_insert_audit_logs([])


# ============================================================================
# Endpoint Tests
# ============================================================================

class TestListTestsEndpoint:
    """Tests for GET /api/v1/tests endpoint."""

    @pytest.mark.asyncio
    async def test_list_tests_with_project_id(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing tests with specific project_id filter."""
        from src.api.tests import list_tests

        mock_supabase.request = AsyncMock(side_effect=[
            # Count query
            {"data": [{"id": "test-1"}, {"id": "test-2"}], "error": None},
            # Tests query
            {
                "data": [
                    {
                        "id": "test-1",
                        "project_id": "project-123",
                        "name": "Test 1",
                        "description": None,
                        "steps": [{"action": "click"}],
                        "tags": [],
                        "priority": "medium",
                        "is_active": True,
                        "source": "manual",
                        "created_at": "2024-01-01T00:00:00Z",
                    },
                ],
                "error": None,
            },
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            response = await list_tests(
                request=mock_request,
                project_id="project-123",
            )

            assert response.total == 2
            assert len(response.tests) == 1
            assert response.limit == 50

    @pytest.mark.asyncio
    async def test_list_tests_without_project_id(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing tests without project_id gets tests from user's orgs."""
        from src.api.tests import list_tests

        mock_supabase.request = AsyncMock(side_effect=[
            # Memberships query
            {"data": [{"organization_id": "org-123"}], "error": None},
            # Projects query
            {"data": [{"id": "project-123"}], "error": None},
            # Count query
            {"data": [{"id": "test-1"}], "error": None},
            # Tests query
            {
                "data": [
                    {
                        "id": "test-1",
                        "project_id": "project-123",
                        "name": "Test 1",
                        "description": None,
                        "steps": [],
                        "tags": [],
                        "priority": "medium",
                        "is_active": True,
                        "source": "manual",
                        "created_at": "2024-01-01T00:00:00Z",
                    },
                ],
                "error": None,
            },
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await list_tests(request=mock_request)

            assert response.total == 1

    @pytest.mark.asyncio
    async def test_list_tests_no_memberships(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing tests when user has no org memberships."""
        from src.api.tests import list_tests

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await list_tests(request=mock_request)

            assert response.total == 0
            assert len(response.tests) == 0

    @pytest.mark.asyncio
    async def test_list_tests_with_filters(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test listing tests with various filters."""
        from src.api.tests import list_tests

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [], "error": None},
            {"data": [], "error": None},
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            response = await list_tests(
                request=mock_request,
                project_id="project-123",
                is_active=True,
                priority="high",
                source="generated",
                tags="auth,login",
                search="login",
                limit=10,
                offset=5,
            )

            assert response.total == 0

    @pytest.mark.asyncio
    async def test_list_tests_error_handling(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test list tests handles Supabase errors."""
        from src.api.tests import list_tests

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [], "error": None},
            {"data": None, "error": "Database error"},
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await list_tests(request=mock_request, project_id="project-123")

            assert exc_info.value.status_code == 500


class TestCreateTestEndpoint:
    """Tests for POST /api/v1/tests endpoint."""

    @pytest.mark.asyncio
    async def test_create_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test successful test creation."""
        from src.api.tests import create_test, CreateTestRequest

        mock_supabase.insert = AsyncMock(return_value={
            "data": [sample_test_data],
            "error": None,
        })
        mock_supabase.request = AsyncMock(return_value={
            "data": [{"organization_id": "org-123"}],
            "error": None,
        })

        body = CreateTestRequest(
            project_id="project-456",
            name="Login Test",
            description="Test login functionality",
            steps=[{"action": "click", "target": "#submit"}],
            tags=["auth"],
            priority="high",
        )

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()), \
             patch("src.api.tests.log_audit", AsyncMock()):
            response = await create_test(body, mock_request)

            assert response.id == "test-123"
            assert response.name == "Login Test"

    @pytest.mark.asyncio
    async def test_create_test_insert_failure(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test create test handles insert errors."""
        from src.api.tests import create_test, CreateTestRequest

        mock_supabase.insert = AsyncMock(return_value={
            "data": None,
            "error": "Insert failed",
        })

        body = CreateTestRequest(
            project_id="project-456",
            name="Test",
        )

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await create_test(body, mock_request)

            assert exc_info.value.status_code == 500


class TestGetTestEndpoint:
    """Tests for GET /api/v1/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test successful test retrieval."""
        from src.api.tests import get_test

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_test_data],
            "error": None,
        })

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            response = await get_test("test-123", mock_request)

            assert response.id == "test-123"
            assert response.name == "Login Test"

    @pytest.mark.asyncio
    async def test_get_test_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test get test returns 404 when not found."""
        from src.api.tests import get_test

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_test("nonexistent", mock_request)

            assert exc_info.value.status_code == 404


class TestUpdateTestEndpoint:
    """Tests for PUT/PATCH /api/v1/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test successful test update."""
        from src.api.tests import update_test, UpdateTestRequest

        updated_data = {**sample_test_data, "name": "Updated Test"}
        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_test_data], "error": None},  # verify_test_access
            {"data": [{"organization_id": "org-123"}], "error": None},  # get_project_org_id
            {"data": [updated_data], "error": None},  # get_test
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateTestRequest(name="Updated Test")

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()), \
             patch("src.api.tests.log_audit", AsyncMock()):
            response = await update_test("test-123", body, mock_request)

            assert response.name == "Updated Test"

    @pytest.mark.asyncio
    async def test_update_test_partial(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test partial update only updates provided fields."""
        from src.api.tests import update_test, UpdateTestRequest

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_test_data], "error": None},
            {"data": [{"organization_id": "org-123"}], "error": None},
            {"data": [sample_test_data], "error": None},
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateTestRequest(priority="critical")

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()), \
             patch("src.api.tests.log_audit", AsyncMock()):
            await update_test("test-123", body, mock_request)

            # Verify only priority was included in update
            call_args = mock_supabase.update.call_args
            update_data = call_args[0][2]
            assert "priority" in update_data
            assert update_data["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_update_test_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test update test handles errors."""
        from src.api.tests import update_test, UpdateTestRequest

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_test_data],
            "error": None,
        })
        mock_supabase.update = AsyncMock(return_value={"data": None, "error": "Update failed"})

        body = UpdateTestRequest(name="Updated")

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await update_test("test-123", body, mock_request)

            assert exc_info.value.status_code == 500


class TestDeleteTestEndpoint:
    """Tests for DELETE /api/v1/tests/{test_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_test_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test successful test deletion."""
        from src.api.tests import delete_test

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_test_data], "error": None},  # verify_test_access
            {"data": [], "error": None},  # DELETE
            {"data": [{"organization_id": "org-123"}], "error": None},  # get_project_org_id
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()), \
             patch("src.api.tests.log_audit", AsyncMock()):
            response = await delete_test("test-123", mock_request)

            assert response["success"] is True

    @pytest.mark.asyncio
    async def test_delete_test_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test delete test returns 404 when not found."""
        from src.api.tests import delete_test

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await delete_test("nonexistent", mock_request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_test_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_test_data
    ):
        """Test delete test handles errors."""
        from src.api.tests import delete_test

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_test_data], "error": None},
            {"data": None, "error": "Delete failed"},
        ])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.tests.verify_project_access", AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await delete_test("test-123", mock_request)

            assert exc_info.value.status_code == 500


# ============================================================================
# Bulk Operation Tests
# ============================================================================

class TestBulkDeleteEndpoint:
    """Tests for POST /api/v1/tests/bulk-delete endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_delete_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test successful bulk delete."""
        from src.api.tests import bulk_delete_tests, BulkDeleteRequest

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query
            {
                "data": [
                    {"id": "test-1", "project_id": "project-1", "name": "Test 1"},
                    {"id": "test-2", "project_id": "project-1", "name": "Test 2"},
                ],
                "error": None,
            },
            # Membership query
            {"data": [{"organization_id": "org-1"}], "error": None},
            # DELETE
            {"data": [], "error": None},
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })
        mock_supabase.insert = AsyncMock(return_value={"data": [], "error": None})

        body = BulkDeleteRequest(test_ids=["test-1", "test-2"])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_delete_tests(body, mock_request)

            assert response["success"] is True
            assert response["deleted_count"] == 2
            assert response["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_delete_partial_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test bulk delete with some tests not accessible."""
        from src.api.tests import bulk_delete_tests, BulkDeleteRequest

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query - only test-1 exists
            {
                "data": [{"id": "test-1", "project_id": "project-1", "name": "Test 1"}],
                "error": None,
            },
            # Membership query
            {"data": [{"organization_id": "org-1"}], "error": None},
            # DELETE
            {"data": [], "error": None},
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })
        mock_supabase.insert = AsyncMock(return_value={"data": [], "error": None})

        body = BulkDeleteRequest(test_ids=["test-1", "test-2"])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_delete_tests(body, mock_request)

            assert response["success"] is False
            assert response["deleted_count"] == 1
            assert response["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_delete_none_accessible(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test bulk delete when no tests are accessible."""
        from src.api.tests import bulk_delete_tests, BulkDeleteRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        body = BulkDeleteRequest(test_ids=["test-1", "test-2"])

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_delete_tests(body, mock_request)

            assert response["deleted_count"] == 0
            assert response["failed_count"] == 2


class TestBulkUpdateEndpoint:
    """Tests for POST /api/v1/tests/bulk-update endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_update_simple(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test bulk update without tag modifications."""
        from src.api.tests import bulk_update_tests, BulkUpdateRequest

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query
            {
                "data": [
                    {"id": "test-1", "project_id": "project-1", "name": "Test 1", "tags": []},
                    {"id": "test-2", "project_id": "project-1", "name": "Test 2", "tags": []},
                ],
                "error": None,
            },
            # Membership query
            {"data": [{"organization_id": "org-1"}], "error": None},
            # PATCH (single batch update)
            {"data": [], "error": None},
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })
        mock_supabase.insert = AsyncMock(return_value={"data": [], "error": None})

        body = BulkUpdateRequest(
            test_ids=["test-1", "test-2"],
            is_active=False,
            priority="critical",
        )

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_update_tests(body, mock_request)

            assert response["success"] is True
            assert response["updated_count"] == 2

    @pytest.mark.asyncio
    async def test_bulk_update_with_tags(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test bulk update with tag modifications."""
        from src.api.tests import bulk_update_tests, BulkUpdateRequest

        mock_supabase.request = AsyncMock(side_effect=[
            # Tests query
            {
                "data": [
                    {"id": "test-1", "project_id": "project-1", "name": "Test 1", "tags": ["old-tag"]},
                ],
                "error": None,
            },
            # Membership query
            {"data": [{"organization_id": "org-1"}], "error": None},
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"project_id": "project-1", "organization_id": "org-1"}],
            "error": None,
        })
        mock_supabase.insert = AsyncMock(return_value={"data": [], "error": None})

        body = BulkUpdateRequest(
            test_ids=["test-1"],
            tags_add=["new-tag"],
            tags_remove=["old-tag"],
        )

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_update_tests(body, mock_request)

            # Verify tags were updated correctly
            call_args = mock_supabase.update.call_args
            update_data = call_args[0][2]
            assert "new-tag" in update_data["tags"]
            assert "old-tag" not in update_data["tags"]

    @pytest.mark.asyncio
    async def test_bulk_update_none_accessible(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test bulk update when no tests are accessible."""
        from src.api.tests import bulk_update_tests, BulkUpdateRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        body = BulkUpdateRequest(
            test_ids=["test-1"],
            is_active=False,
        )

        with patch("src.api.tests.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.tests.get_current_user", AsyncMock(return_value=mock_user)):
            response = await bulk_update_tests(body, mock_request)

            assert response["updated_count"] == 0
            assert response["failed_count"] == 1
