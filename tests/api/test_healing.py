"""Tests for the Self-Healing API module (src/api/healing.py).

Covers:
- Healing configuration endpoints (get, update)
- Healing patterns endpoints (list, delete)
- Healing statistics endpoint
- Approval workflow endpoints (approve, reject, pending)
- Intelligence layer endpoints (suggest, analyze-root-cause, similar-errors)
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
    return request


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_healing_config():
    """Create sample healing configuration."""
    return {
        "id": str(uuid.uuid4()),
        "organization_id": str(uuid.uuid4()),
        "project_id": None,
        "enabled": True,
        "auto_apply": False,
        "min_confidence_auto": 0.95,
        "min_confidence_suggest": 0.70,
        "heal_selectors": True,
        "max_selector_variations": 9,
        "preferred_selector_strategies": ["id", "data-testid", "role", "text", "css"],
        "heal_timeouts": True,
        "max_wait_time_ms": 30000,
        "heal_text_content": True,
        "text_similarity_threshold": 0.85,
        "learn_from_success": True,
        "learn_from_manual_fixes": True,
        "share_patterns_across_projects": False,
        "notify_on_heal": True,
        "notify_on_suggestion": True,
        "notification_channels": {"email": True, "slack": False},
        "require_approval": True,
        "auto_approve_after_hours": None,
        "approvers": None,
        "max_heals_per_hour": 50,
        "max_heals_per_test": 5,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_healing_pattern():
    """Create sample healing pattern."""
    return {
        "id": str(uuid.uuid4()),
        "fingerprint": "fp-12345",
        "original_selector": "#submit-btn",
        "healed_selector": "button[data-testid='submit']",
        "error_type": "selector_changed",
        "success_count": 10,
        "failure_count": 2,
        "confidence": 0.85,
        "project_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def mock_query_router():
    """Create a mock QueryRouter."""
    mock_router = MagicMock()
    mock_result = MagicMock()
    mock_result.source = "vector"
    mock_result.confidence = 0.85
    mock_result.latency_ms = 50
    mock_result.cache_key = "test-cache-key"
    mock_result.intent = MagicMock()
    mock_result.intent.value = "SIMILAR_ERRORS"
    mock_result.data = {
        "suggestions": [
            {
                "fix_type": "update_selector",
                "old_value": "#old-btn",
                "new_value": "[data-testid='new-btn']",
                "confidence": 0.85,
                "explanation": "Similar pattern found",
            }
        ]
    }
    mock_result.metadata = {}
    mock_router.route = AsyncMock(return_value=mock_result)
    return mock_router


# ============================================================================
# Configuration Endpoint Tests
# ============================================================================

class TestGetHealingConfig:
    """Tests for get_healing_config endpoint."""

    @pytest.mark.asyncio
    async def test_get_config_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_config):
        """Test successful config retrieval."""
        from src.api.healing import get_healing_config

        org_id = sample_healing_config["organization_id"]
        mock_supabase.request.return_value = {"data": [sample_healing_config], "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await get_healing_config(org_id, mock_request)

        assert result.organization_id == org_id
        assert result.enabled is True

    @pytest.mark.asyncio
    async def test_get_config_creates_default(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_config):
        """Test that default config is created when none exists."""
        from src.api.healing import get_healing_config

        org_id = str(uuid.uuid4())

        # First request returns empty, then insert returns new config
        mock_supabase.request.return_value = {"data": [], "error": None}
        mock_supabase.insert.return_value = {"data": [sample_healing_config], "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await get_healing_config(org_id, mock_request)

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_config_invalid_org_id(self, mock_env_vars, mock_request, mock_user):
        """Test invalid organization ID validation."""
        from src.api.healing import get_healing_config

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_healing_config("invalid-uuid", mock_request)

            assert exc_info.value.status_code == 400


class TestUpdateHealingConfig:
    """Tests for update_healing_config endpoint."""

    @pytest.mark.asyncio
    async def test_update_config_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_config):
        """Test successful config update."""
        from src.api.healing import update_healing_config, HealingConfigRequest

        org_id = sample_healing_config["organization_id"]
        update_request = HealingConfigRequest(enabled=False, auto_apply=True)

        mock_supabase.request.return_value = {"data": [{"id": sample_healing_config["id"]}], "error": None}
        mock_supabase.update.return_value = {"data": [], "error": None}

        # Create a patched version of get_healing_config
        async def patched_get_config(*args, **kwargs):
            from src.api.healing import HealingConfigResponse
            return HealingConfigResponse(**sample_healing_config)

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.log_audit", AsyncMock()), \
             patch("src.api.healing.get_healing_config", patched_get_config):

            result = await update_healing_config(org_id, update_request, mock_request)

        assert result is not None

    @pytest.mark.asyncio
    async def test_update_config_validation(self, mock_env_vars):
        """Test config request validation."""
        from src.api.healing import HealingConfigRequest
        from pydantic import ValidationError

        # Valid request
        request = HealingConfigRequest(min_confidence_auto=0.95)
        assert request.min_confidence_auto == 0.95

        # Invalid confidence values
        with pytest.raises(ValidationError):
            HealingConfigRequest(min_confidence_auto=0.3)  # Below 0.5

        with pytest.raises(ValidationError):
            HealingConfigRequest(min_confidence_auto=1.5)  # Above 1.0


# ============================================================================
# Pattern Endpoint Tests
# ============================================================================

class TestListHealingPatterns:
    """Tests for list_healing_patterns endpoint."""

    @pytest.mark.asyncio
    async def test_list_patterns_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_pattern):
        """Test successful pattern listing."""
        from src.api.healing import list_healing_patterns

        org_id = str(uuid.uuid4())

        # Mock project lookup and pattern query
        mock_supabase.request.side_effect = [
            {"data": [{"id": sample_healing_pattern["project_id"]}], "error": None},  # Projects
            {"data": [sample_healing_pattern], "error": None},  # Patterns
        ]

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await list_healing_patterns(org_id, mock_request)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_patterns_empty(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test listing patterns when none exist."""
        from src.api.healing import list_healing_patterns

        org_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await list_healing_patterns(org_id, mock_request)

        assert result == []


class TestDeleteHealingPattern:
    """Tests for delete_healing_pattern endpoint."""

    @pytest.mark.asyncio
    async def test_delete_pattern_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_pattern):
        """Test successful pattern deletion."""
        from src.api.healing import delete_healing_pattern

        org_id = str(uuid.uuid4())
        pattern_id = sample_healing_pattern["id"]

        mock_supabase.request.side_effect = [
            {"data": [sample_healing_pattern], "error": None},  # Get pattern
            {"data": [], "error": None},  # Delete
        ]

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.log_audit", AsyncMock()):

            result = await delete_healing_pattern(org_id, pattern_id, mock_request)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_pattern_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test deleting non-existent pattern."""
        from src.api.healing import delete_healing_pattern

        org_id = str(uuid.uuid4())
        pattern_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            with pytest.raises(HTTPException) as exc_info:
                await delete_healing_pattern(org_id, pattern_id, mock_request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Statistics Endpoint Tests
# ============================================================================

class TestGetHealingStats:
    """Tests for get_healing_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, mock_env_vars, mock_supabase, mock_request, mock_user, sample_healing_pattern):
        """Test successful stats retrieval."""
        from src.api.healing import get_healing_stats

        org_id = str(uuid.uuid4())

        # Mock projects and patterns
        mock_supabase.request.side_effect = [
            {"data": [{"id": sample_healing_pattern["project_id"], "name": "Test Project"}], "error": None},
            {"data": [sample_healing_pattern], "error": None},
        ]

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await get_healing_stats(org_id, mock_request)

        assert result.total_patterns >= 0
        assert isinstance(result.top_error_types, dict)

    @pytest.mark.asyncio
    async def test_get_stats_no_projects(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test stats when organization has no projects."""
        from src.api.healing import get_healing_stats

        org_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await get_healing_stats(org_id, mock_request)

        assert result.total_patterns == 0
        assert result.success_rate == 0


# ============================================================================
# Approval Endpoint Tests
# ============================================================================

class TestApproveHealing:
    """Tests for approve_healing endpoint."""

    @pytest.mark.asyncio
    async def test_approve_healing_success(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test successful healing approval."""
        from src.api.healing import approve_healing

        org_id = str(uuid.uuid4())
        pattern_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": {"success": True}, "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.log_audit", AsyncMock()):

            result = await approve_healing(org_id, pattern_id, mock_request)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_approve_healing_rpc_error(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test handling of RPC errors during approval."""
        from src.api.healing import approve_healing

        org_id = str(uuid.uuid4())
        pattern_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": None, "error": "RPC failed"}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            with pytest.raises(HTTPException) as exc_info:
                await approve_healing(org_id, pattern_id, mock_request)

            assert exc_info.value.status_code == 500


class TestRejectHealing:
    """Tests for reject_healing endpoint."""

    @pytest.mark.asyncio
    async def test_reject_healing_success(self, mock_env_vars, mock_supabase, mock_request, mock_user):
        """Test successful healing rejection."""
        from src.api.healing import reject_healing

        org_id = str(uuid.uuid4())
        pattern_id = str(uuid.uuid4())

        mock_supabase.request.return_value = {"data": {"success": True}, "error": None}

        with patch("src.api.healing.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.log_audit", AsyncMock()):

            result = await reject_healing(org_id, pattern_id, mock_request)

        assert result["success"] is True


class TestGetPendingApprovals:
    """Tests for get_pending_approvals endpoint."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_success(self, mock_env_vars, mock_request, mock_user):
        """Test getting pending approvals."""
        from src.api.healing import get_pending_approvals

        org_id = str(uuid.uuid4())

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))):

            result = await get_pending_approvals(org_id, mock_request)

        assert "pending" in result
        assert result["total"] == 0


# ============================================================================
# Intelligence Layer Endpoint Tests
# ============================================================================

class TestGetHealingSuggestions:
    """Tests for get_healing_suggestions endpoint."""

    @pytest.mark.asyncio
    async def test_get_suggestions_success(self, mock_env_vars, mock_request, mock_user, mock_query_router):
        """Test successful suggestion retrieval."""
        from src.api.healing import get_healing_suggestions, HealingSuggestionRequest

        org_id = str(uuid.uuid4())
        suggestion_request = HealingSuggestionRequest(
            error_message="Element not found: #submit-btn",
            error_type="selector_changed",
            selector="#submit-btn",
        )

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.get_query_router", return_value=mock_query_router):

            result = await get_healing_suggestions(org_id, suggestion_request, mock_request)

        assert len(result.suggestions) > 0
        assert result.source == "vector"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_suggestions_with_cache_skip(self, mock_env_vars, mock_request, mock_user, mock_query_router):
        """Test suggestions with cache skip option."""
        from src.api.healing import get_healing_suggestions, HealingSuggestionRequest

        org_id = str(uuid.uuid4())
        suggestion_request = HealingSuggestionRequest(
            error_message="Timeout waiting for element",
            skip_cache=True,
        )

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.get_query_router", return_value=mock_query_router):

            result = await get_healing_suggestions(org_id, suggestion_request, mock_request)

        # Verify route was called with skip_cache=True
        mock_query_router.route.assert_called()


class TestAnalyzeRootCause:
    """Tests for analyze_root_cause endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_root_cause_success(self, mock_env_vars, mock_request, mock_user, mock_query_router):
        """Test successful root cause analysis."""
        from src.api.healing import analyze_root_cause, RootCauseRequest

        org_id = str(uuid.uuid4())
        root_cause_request = RootCauseRequest(
            error_message="Element not found: #submit-btn",
            error_type="selector_changed",
        )

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.get_query_router", return_value=mock_query_router):

            result = await analyze_root_cause(org_id, root_cause_request, mock_request)

        assert result.source == "vector"
        assert result.confidence > 0


class TestFindSimilarErrors:
    """Tests for find_similar_errors endpoint."""

    @pytest.mark.asyncio
    async def test_find_similar_errors_success(self, mock_env_vars, mock_request, mock_user):
        """Test successful similar error search."""
        from src.api.healing import find_similar_errors, HealingSuggestionRequest

        org_id = str(uuid.uuid4())
        search_request = HealingSuggestionRequest(
            error_message="Button not found",
            selector="#missing-btn",
        )

        # Create a mock router with properly structured data for similar errors
        mock_router = MagicMock()
        mock_result = MagicMock()
        mock_result.source = "vector"
        mock_result.confidence = 0.85
        mock_result.latency_ms = 50
        mock_result.data = {
            "results": [
                {
                    "id": str(uuid.uuid4()),
                    "error_message": "Button element not found",
                    "error_type": "selector_changed",
                    "selector": "#old-btn",
                    "healed_selector": "[data-testid='btn']",
                    "healing_method": "update_selector",
                    "similarity": 0.9,
                    "success_rate": 0.85,
                    "success_count": 10,
                }
            ]
        }
        mock_router.route = AsyncMock(return_value=mock_result)

        with patch("src.api.healing.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.healing.verify_org_access", AsyncMock(return_value=("org-name", org_id))), \
             patch("src.api.healing.get_query_router", return_value=mock_router):

            result = await find_similar_errors(org_id, search_request, mock_request, limit=5)

        assert "similar_errors" in result
        assert "count" in result


# ============================================================================
# Model Validation Tests
# ============================================================================

class TestHealingConfigRequestModel:
    """Tests for HealingConfigRequest model validation."""

    def test_valid_request(self, mock_env_vars):
        """Test creating a valid request."""
        from src.api.healing import HealingConfigRequest

        request = HealingConfigRequest(
            enabled=True,
            auto_apply=False,
            min_confidence_auto=0.9,
        )

        assert request.enabled is True
        assert request.min_confidence_auto == 0.9

    def test_confidence_bounds(self, mock_env_vars):
        """Test confidence field validation bounds."""
        from src.api.healing import HealingConfigRequest
        from pydantic import ValidationError

        # min_confidence_auto: 0.5 to 1.0
        HealingConfigRequest(min_confidence_auto=0.5)
        HealingConfigRequest(min_confidence_auto=1.0)

        with pytest.raises(ValidationError):
            HealingConfigRequest(min_confidence_auto=0.4)

        # min_confidence_suggest: 0.3 to 1.0
        HealingConfigRequest(min_confidence_suggest=0.3)
        HealingConfigRequest(min_confidence_suggest=1.0)

        with pytest.raises(ValidationError):
            HealingConfigRequest(min_confidence_suggest=0.2)


class TestHealingSuggestionModel:
    """Tests for HealingSuggestion model validation."""

    def test_valid_suggestion(self, mock_env_vars):
        """Test creating a valid suggestion."""
        from src.api.healing import HealingSuggestion

        suggestion = HealingSuggestion(
            fix_type="update_selector",
            old_value="#old-btn",
            new_value="[data-testid='new-btn']",
            confidence=0.85,
            explanation="Similar pattern found in history",
        )

        assert suggestion.fix_type == "update_selector"
        assert suggestion.confidence == 0.85

    def test_confidence_bounds(self, mock_env_vars):
        """Test confidence field validation bounds."""
        from src.api.healing import HealingSuggestion
        from pydantic import ValidationError

        HealingSuggestion(fix_type="test", confidence=0.0, explanation="test")
        HealingSuggestion(fix_type="test", confidence=1.0, explanation="test")

        with pytest.raises(ValidationError):
            HealingSuggestion(fix_type="test", confidence=-0.1, explanation="test")

        with pytest.raises(ValidationError):
            HealingSuggestion(fix_type="test", confidence=1.1, explanation="test")


class TestHealingPatternResponseModel:
    """Tests for HealingPatternResponse model validation."""

    def test_valid_pattern_response(self, mock_env_vars, sample_healing_pattern):
        """Test creating a valid pattern response."""
        from src.api.healing import HealingPatternResponse

        response = HealingPatternResponse(**sample_healing_pattern)

        assert response.fingerprint == "fp-12345"
        assert response.error_type == "selector_changed"
        assert response.success_count == 10
