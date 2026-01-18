"""Tests for the Organizations API module (src/api/organizations.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
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
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_org_data():
    """Create sample organization data."""
    return {
        "id": "org-123",
        "name": "Test Organization",
        "slug": "test-organization",
        "plan": "pro",
        "ai_budget_daily": 5.0,
        "ai_budget_monthly": 100.0,
        "settings": {"theme": "dark"},
        "features": {"advanced_analytics": True},
        "stripe_customer_id": "cus_123",
        "stripe_subscription_id": "sub_456",
        "logo_url": "https://example.com/logo.png",
        "domain": "example.com",
        "sso_enabled": True,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": None,
    }


@pytest.fixture
def sample_member_data():
    """Create sample member data."""
    return {
        "id": "member-123",
        "organization_id": "org-123",
        "user_id": "user-123",
        "email": "test@example.com",
        "role": "owner",
        "status": "active",
    }


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestGenerateSlug:
    """Tests for generate_slug helper function."""

    def test_generate_slug_basic(self, mock_env_vars):
        """Test basic slug generation."""
        from src.api.organizations import generate_slug

        result = generate_slug("Test Organization")
        assert result == "test-organization"

    def test_generate_slug_special_chars(self, mock_env_vars):
        """Test slug generation with special characters."""
        from src.api.organizations import generate_slug

        result = generate_slug("Test@#$Organization!")
        assert result == "test-organization"

    def test_generate_slug_consecutive_special(self, mock_env_vars):
        """Test slug generation with consecutive special characters."""
        from src.api.organizations import generate_slug

        result = generate_slug("Test---Organization")
        assert result == "test-organization"

    def test_generate_slug_numbers(self, mock_env_vars):
        """Test slug generation with numbers."""
        from src.api.organizations import generate_slug

        result = generate_slug("Org 123 Test")
        assert result == "org-123-test"

    def test_generate_slug_leading_trailing_special(self, mock_env_vars):
        """Test slug generation trims special chars from ends."""
        from src.api.organizations import generate_slug

        result = generate_slug("---Test Organization---")
        assert result == "test-organization"


class TestEnsureUniqueSlug:
    """Tests for ensure_unique_slug helper function."""

    @pytest.mark.asyncio
    async def test_ensure_unique_slug_already_unique(self, mock_env_vars, mock_supabase):
        """Test when slug is already unique."""
        from src.api.organizations import ensure_unique_slug

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            result = await ensure_unique_slug("test-org")
            assert result == "test-org"

    @pytest.mark.asyncio
    async def test_ensure_unique_slug_collision(self, mock_env_vars, mock_supabase):
        """Test when slug collides and needs suffix."""
        from src.api.organizations import ensure_unique_slug

        # First call returns existing, subsequent calls return empty
        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [{"id": "existing"}], "error": None},  # First check - collision
            {"data": [], "error": None},  # Second check with suffix - unique
        ])

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            result = await ensure_unique_slug("test-org")
            assert result.startswith("test-org-")
            assert len(result) > len("test-org")


class TestGetMemberCount:
    """Tests for get_member_count helper function."""

    @pytest.mark.asyncio
    async def test_get_member_count(self, mock_env_vars, mock_supabase):
        """Test getting member count."""
        from src.api.organizations import get_member_count

        mock_supabase.request = AsyncMock(return_value={
            "data": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
            "error": None,
        })

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            count = await get_member_count("org-123")
            assert count == 3

    @pytest.mark.asyncio
    async def test_get_member_count_empty(self, mock_env_vars, mock_supabase):
        """Test getting member count with no members."""
        from src.api.organizations import get_member_count

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            count = await get_member_count("org-123")
            assert count == 0


class TestGetMemberCountsBatch:
    """Tests for get_member_counts_batch helper function."""

    @pytest.mark.asyncio
    async def test_get_member_counts_batch(self, mock_env_vars, mock_supabase):
        """Test batch member count retrieval."""
        from src.api.organizations import get_member_counts_batch

        mock_supabase.rpc = AsyncMock(return_value={
            "data": [
                {"organization_id": "org-1", "count": 5},
                {"organization_id": "org-2", "count": 10},
            ],
            "error": None,
        })

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            result = await get_member_counts_batch(["org-1", "org-2"])

            assert result["org-1"] == 5
            assert result["org-2"] == 10

    @pytest.mark.asyncio
    async def test_get_member_counts_batch_empty(self, mock_env_vars):
        """Test batch member count with empty list."""
        from src.api.organizations import get_member_counts_batch

        result = await get_member_counts_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_member_counts_batch_fallback(self, mock_env_vars, mock_supabase):
        """Test batch member count falls back on RPC error."""
        from src.api.organizations import get_member_counts_batch

        mock_supabase.rpc = AsyncMock(return_value={"data": None, "error": "RPC error"})
        mock_supabase.request = AsyncMock(return_value={
            "data": [{"id": "m1"}],
            "error": None,
        })

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase):
            result = await get_member_counts_batch(["org-1"])
            assert "org-1" in result


# ============================================================================
# Model Tests
# ============================================================================

class TestRequestModels:
    """Tests for request model validation."""

    def test_create_organization_request(self, mock_env_vars):
        """Test CreateOrganizationRequest model."""
        from src.api.organizations import CreateOrganizationRequest

        request = CreateOrganizationRequest(name="My Organization")

        assert request.name == "My Organization"

    def test_create_organization_request_name_validation(self, mock_env_vars):
        """Test CreateOrganizationRequest name validation."""
        from src.api.organizations import CreateOrganizationRequest

        # Name too short
        with pytest.raises(ValidationError):
            CreateOrganizationRequest(name="A")

        # Name too long
        with pytest.raises(ValidationError):
            CreateOrganizationRequest(name="x" * 101)

    def test_update_organization_request(self, mock_env_vars):
        """Test UpdateOrganizationRequest model."""
        from src.api.organizations import UpdateOrganizationRequest

        request = UpdateOrganizationRequest(
            name="Updated Name",
            logo_url="https://example.com/logo.png",
            domain="example.com",
            ai_budget_daily=10.0,
            ai_budget_monthly=200.0,
            settings={"theme": "light"},
            features={"new_feature": True},
            sso_enabled=True,
        )

        assert request.name == "Updated Name"
        assert request.ai_budget_daily == 10.0

    def test_update_organization_request_empty(self, mock_env_vars):
        """Test UpdateOrganizationRequest with no fields."""
        from src.api.organizations import UpdateOrganizationRequest

        request = UpdateOrganizationRequest()

        assert request.name is None
        assert request.ai_budget_daily is None

    def test_update_organization_request_budget_validation(self, mock_env_vars):
        """Test UpdateOrganizationRequest budget validation."""
        from src.api.organizations import UpdateOrganizationRequest

        # Negative daily budget
        with pytest.raises(ValidationError):
            UpdateOrganizationRequest(ai_budget_daily=-1.0)

        # Excessive monthly budget
        with pytest.raises(ValidationError):
            UpdateOrganizationRequest(ai_budget_monthly=100001.0)

    def test_transfer_ownership_request(self, mock_env_vars):
        """Test TransferOwnershipRequest model."""
        from src.api.organizations import TransferOwnershipRequest

        request = TransferOwnershipRequest(new_owner_user_id="user-456")

        assert request.new_owner_user_id == "user-456"

    def test_transfer_ownership_request_empty_fails(self, mock_env_vars):
        """Test TransferOwnershipRequest with empty user_id fails."""
        from src.api.organizations import TransferOwnershipRequest

        with pytest.raises(ValidationError):
            TransferOwnershipRequest(new_owner_user_id="")


class TestResponseModels:
    """Tests for response model validation."""

    def test_organization_response(self, mock_env_vars):
        """Test OrganizationResponse model."""
        from src.api.organizations import OrganizationResponse

        response = OrganizationResponse(
            id="org-123",
            name="Test Org",
            slug="test-org",
            plan="pro",
            ai_budget_daily=5.0,
            ai_budget_monthly=100.0,
            settings={},
            features={},
            stripe_customer_id=None,
            stripe_subscription_id=None,
            logo_url=None,
            domain=None,
            sso_enabled=False,
            member_count=5,
            created_at="2024-01-01T00:00:00Z",
            updated_at=None,
        )

        assert response.id == "org-123"
        assert response.member_count == 5

    def test_organization_list_response(self, mock_env_vars):
        """Test OrganizationListResponse model."""
        from src.api.organizations import OrganizationListResponse

        response = OrganizationListResponse(
            id="org-123",
            name="Test Org",
            slug="test-org",
            plan="free",
            logo_url=None,
            member_count=3,
            role="admin",
            created_at="2024-01-01T00:00:00Z",
        )

        assert response.role == "admin"


# ============================================================================
# Endpoint Tests
# ============================================================================

class TestCreateOrganization:
    """Tests for POST /api/v1/organizations endpoint."""

    @pytest.mark.asyncio
    async def test_create_organization_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test successful organization creation."""
        from src.api.organizations import create_organization, CreateOrganizationRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})
        mock_supabase.insert = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},  # org insert
            {"data": [{"id": "member-1"}], "error": None},  # member insert
        ])

        body = CreateOrganizationRequest(name="Test Organization")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            response = await create_organization(body, mock_request)

            assert response.id == "org-123"
            assert response.name == "Test Organization"
            assert response.member_count == 1

    @pytest.mark.asyncio
    async def test_create_organization_invalid_name(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization creation with invalid name."""
        from src.api.organizations import create_organization, CreateOrganizationRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        body = CreateOrganizationRequest(name="###")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await create_organization(body, mock_request)

            assert exc_info.value.status_code == 400
            assert "alphanumeric" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_organization_insert_failure(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization creation handles insert failure."""
        from src.api.organizations import create_organization, CreateOrganizationRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})
        mock_supabase.insert = AsyncMock(return_value={
            "data": None,
            "error": "Insert failed",
        })

        body = CreateOrganizationRequest(name="Test Org")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await create_organization(body, mock_request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_create_organization_member_failure_rollback(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test organization creation rolls back on member insert failure."""
        from src.api.organizations import create_organization, CreateOrganizationRequest

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [], "error": None},  # slug check
            {"data": [], "error": None},  # delete for rollback
        ])
        mock_supabase.insert = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},  # org insert
            {"data": None, "error": "Member insert failed"},  # member insert
        ])

        body = CreateOrganizationRequest(name="Test Org")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await create_organization(body, mock_request)

            assert exc_info.value.status_code == 500


class TestListOrganizations:
    """Tests for GET /api/v1/organizations endpoint."""

    @pytest.mark.asyncio
    async def test_list_organizations_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test successful organization listing."""
        from src.api.organizations import list_organizations

        mock_supabase.request = AsyncMock(side_effect=[
            # memberships query
            {"data": [{"organization_id": "org-123", "role": "owner"}], "error": None},
            # orgs query
            {"data": [sample_org_data], "error": None},
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"organization_id": "org-123", "count": 5}],
            "error": None,
        })

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            response = await list_organizations(mock_request)

            assert len(response) == 1
            assert response[0].id == "org-123"
            assert response[0].role == "owner"
            assert response[0].member_count == 5

    @pytest.mark.asyncio
    async def test_list_organizations_no_memberships(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization listing with no memberships."""
        from src.api.organizations import list_organizations

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            response = await list_organizations(mock_request)

            assert response == []

    @pytest.mark.asyncio
    async def test_list_organizations_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization listing handles errors."""
        from src.api.organizations import list_organizations

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [{"organization_id": "org-123", "role": "owner"}], "error": None},
            {"data": None, "error": "DB error"},
        ])

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await list_organizations(mock_request)

            assert exc_info.value.status_code == 500


class TestGetOrganization:
    """Tests for GET /api/v1/organizations/{org_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_organization_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test successful organization retrieval."""
        from src.api.organizations import get_organization

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},  # org query
            {"data": [{"id": "m1"}, {"id": "m2"}], "error": None},  # member count
        ])

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            response = await get_organization("org-123", mock_request)

            assert response.id == "org-123"
            assert response.name == "Test Organization"
            assert response.member_count == 2

    @pytest.mark.asyncio
    async def test_get_organization_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization retrieval when not found."""
        from src.api.organizations import get_organization

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await get_organization("org-123", mock_request)

            assert exc_info.value.status_code == 404


class TestUpdateOrganization:
    """Tests for PUT /api/v1/organizations/{org_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_organization_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test successful organization update."""
        from src.api.organizations import update_organization, UpdateOrganizationRequest

        updated_data = {**sample_org_data, "name": "Updated Org"}
        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [updated_data], "error": None},  # get org
            {"data": [{"id": "m1"}], "error": None},  # member count
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateOrganizationRequest(name="Updated Org")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            response = await update_organization("org-123", body, mock_request)

            assert response.name == "Updated Org"
            mock_supabase.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_organization_partial(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test partial organization update."""
        from src.api.organizations import update_organization, UpdateOrganizationRequest

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},
            {"data": [{"id": "m1"}], "error": None},
        ])
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = UpdateOrganizationRequest(ai_budget_daily=20.0)

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            await update_organization("org-123", body, mock_request)

            # Verify only budget was updated
            call_args = mock_supabase.update.call_args
            update_data = call_args[0][2]
            assert "ai_budget_daily" in update_data
            assert update_data["ai_budget_daily"] == 20.0
            assert "name" not in update_data

    @pytest.mark.asyncio
    async def test_update_organization_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization update handles errors."""
        from src.api.organizations import update_organization, UpdateOrganizationRequest

        mock_supabase.update = AsyncMock(return_value={"data": None, "error": "Update failed"})

        body = UpdateOrganizationRequest(name="Updated")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await update_organization("org-123", body, mock_request)

            assert exc_info.value.status_code == 500


class TestDeleteOrganization:
    """Tests for DELETE /api/v1/organizations/{org_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_organization_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test successful organization deletion."""
        from src.api.organizations import delete_organization

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},  # get org
            {"data": [], "error": None},  # delete members
            {"data": [], "error": None},  # delete org
        ])

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            response = await delete_organization("org-123", mock_request)

            assert response["success"] is True
            assert "Test Organization" in response["message"]

    @pytest.mark.asyncio
    async def test_delete_organization_not_found(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test organization deletion when not found."""
        from src.api.organizations import delete_organization

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await delete_organization("org-123", mock_request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_organization_error(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test organization deletion handles errors."""
        from src.api.organizations import delete_organization

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},
            {"data": [], "error": None},  # delete members
            {"data": None, "error": "Delete failed"},  # delete org
        ])

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=({}, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await delete_organization("org-123", mock_request)

            assert exc_info.value.status_code == 500


class TestTransferOwnership:
    """Tests for POST /api/v1/organizations/{org_id}/transfer endpoint."""

    @pytest.mark.asyncio
    async def test_transfer_ownership_success(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_member_data
    ):
        """Test successful ownership transfer."""
        from src.api.organizations import transfer_ownership, TransferOwnershipRequest

        new_owner_member = {
            **sample_member_data,
            "id": "member-456",
            "user_id": "user-456",
            "email": "newowner@example.com",
            "role": "admin",
        }

        mock_supabase.request = AsyncMock(return_value={
            "data": [new_owner_member],
            "error": None,
        })
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = TransferOwnershipRequest(new_owner_user_id="user-456")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=(sample_member_data, "org-123"))), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            response = await transfer_ownership("org-123", body, mock_request)

            assert response["success"] is True
            assert response["new_owner_id"] == "user-456"
            assert response["previous_owner_role"] == "admin"

    @pytest.mark.asyncio
    async def test_transfer_ownership_to_non_member(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_member_data
    ):
        """Test transfer to non-member fails."""
        from src.api.organizations import transfer_ownership, TransferOwnershipRequest

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        body = TransferOwnershipRequest(new_owner_user_id="nonexistent-user")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=(sample_member_data, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await transfer_ownership("org-123", body, mock_request)

            assert exc_info.value.status_code == 400
            assert "not an active member" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_transfer_ownership_to_self(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_member_data
    ):
        """Test transfer to self fails."""
        from src.api.organizations import transfer_ownership, TransferOwnershipRequest

        mock_supabase.request = AsyncMock(return_value={
            "data": [sample_member_data],  # Same user
            "error": None,
        })

        body = TransferOwnershipRequest(new_owner_user_id="user-123")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=(sample_member_data, "org-123"))):
            with pytest.raises(HTTPException) as exc_info:
                await transfer_ownership("org-123", body, mock_request)

            assert exc_info.value.status_code == 400
            assert "yourself" in str(exc_info.value.detail)


# ============================================================================
# Access Control Tests
# ============================================================================

class TestAccessControl:
    """Tests for access control in organization operations."""

    @pytest.mark.asyncio
    async def test_update_requires_admin(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test that update requires admin or owner role."""
        from src.api.organizations import update_organization, UpdateOrganizationRequest

        body = UpdateOrganizationRequest(name="Updated")

        with patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Insufficient permissions"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await update_organization("org-123", body, mock_request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_requires_owner(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test that delete requires owner role."""
        from src.api.organizations import delete_organization

        with patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Owner role required"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await delete_organization("org-123", mock_request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_transfer_requires_owner(
        self, mock_env_vars, mock_supabase, mock_request, mock_user
    ):
        """Test that transfer requires owner role."""
        from src.api.organizations import transfer_ownership, TransferOwnershipRequest

        body = TransferOwnershipRequest(new_owner_user_id="user-456")

        with patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(side_effect=HTTPException(
                 status_code=403, detail="Owner role required"
             ))):
            with pytest.raises(HTTPException) as exc_info:
                await transfer_ownership("org-123", body, mock_request)

            assert exc_info.value.status_code == 403


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestOrganizationIntegration:
    """Integration-style tests for organization operations."""

    @pytest.mark.asyncio
    async def test_create_and_list_organization_flow(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_org_data
    ):
        """Test creating and then listing an organization."""
        from src.api.organizations import (
            create_organization,
            list_organizations,
            CreateOrganizationRequest,
        )

        mock_supabase.request = AsyncMock(side_effect=[
            {"data": [], "error": None},  # slug check
            {"data": [{"organization_id": "org-123", "role": "owner"}], "error": None},  # memberships
            {"data": [sample_org_data], "error": None},  # orgs
        ])
        mock_supabase.insert = AsyncMock(side_effect=[
            {"data": [sample_org_data], "error": None},  # org insert
            {"data": [{"id": "member-1"}], "error": None},  # member insert
        ])
        mock_supabase.rpc = AsyncMock(return_value={
            "data": [{"organization_id": "org-123", "count": 1}],
            "error": None,
        })

        body = CreateOrganizationRequest(name="Test Organization")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            # Create organization
            created = await create_organization(body, mock_request)
            assert created.id == "org-123"

            # List organizations
            orgs = await list_organizations(mock_request)
            assert len(orgs) == 1
            assert orgs[0].id == "org-123"

    @pytest.mark.asyncio
    async def test_full_ownership_transfer_flow(
        self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_member_data
    ):
        """Test full ownership transfer flow."""
        from src.api.organizations import transfer_ownership, TransferOwnershipRequest

        current_owner = sample_member_data
        new_owner = {
            **sample_member_data,
            "id": "member-456",
            "user_id": "user-456",
            "role": "admin",
        }

        mock_supabase.request = AsyncMock(return_value={
            "data": [new_owner],
            "error": None,
        })
        mock_supabase.update = AsyncMock(return_value={"data": [], "error": None})

        body = TransferOwnershipRequest(new_owner_user_id="user-456")

        with patch("src.api.organizations.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.organizations.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.organizations.verify_org_access", AsyncMock(return_value=(current_owner, "org-123"))), \
             patch("src.api.organizations.log_audit", AsyncMock()):
            response = await transfer_ownership("org-123", body, mock_request)

            assert response["success"] is True
            assert response["new_owner_id"] == "user-456"

            # Verify both updates were called (new owner and old owner)
            assert mock_supabase.update.call_count == 2
