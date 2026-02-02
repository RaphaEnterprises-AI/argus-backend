"""Tests for Accessibility API endpoints (src/api/accessibility.py).

Covers:
- Get latest audit
- Get audit by ID
- Get audit history
- Get accessibility issues
- Run audit
- Update audit
- Create issues batch
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
    return mock


@pytest.fixture
def sample_audit_data():
    """Create sample audit data."""
    return {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "url": "https://example.com",
        "status": "completed",
        "accessibility_score": 85.0,
        "performance_score": 90.0,
        "best_practices_score": 88.0,
        "seo_score": 92.0,
        "lcp_ms": 2500.0,
        "fid_ms": 100.0,
        "cls": 0.1,
        "ttfb_ms": 500.0,
        "fcp_ms": 1500.0,
        "tti_ms": 3500.0,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "triggered_by": "user-123",
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_issue_data():
    """Create sample accessibility issue data."""
    return {
        "id": str(uuid.uuid4()),
        "audit_id": str(uuid.uuid4()),
        "rule": "color-contrast",
        "severity": "serious",
        "element_selector": "#main-content p",
        "description": "Text color has insufficient contrast ratio",
        "suggested_fix": "Use a darker text color for better contrast",
        "wcag_criteria": ["1.4.3", "1.4.6"],
        "created_at": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# Get Latest Audit Tests
# ============================================================================


class TestGetLatestAudit:
    """Tests for GET /api/v1/accessibility/audit/latest endpoint."""

    @pytest.mark.asyncio
    async def test_get_latest_audit_success(
        self, mock_env_vars, mock_supabase, sample_audit_data, sample_issue_data
    ):
        """Test getting latest audit successfully."""
        from src.api.accessibility import get_latest_audit

        sample_issue_data["audit_id"] = sample_audit_data["id"]

        mock_supabase.request.side_effect = [
            {"data": [sample_audit_data], "error": None},  # audits query
            {"data": [sample_issue_data], "error": None},  # issues query
        ]

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_latest_audit(project_id=sample_audit_data["project_id"])

            assert response is not None
            assert response.audit.id == sample_audit_data["id"]
            assert len(response.issues) == 1

    @pytest.mark.asyncio
    async def test_get_latest_audit_none(self, mock_env_vars, mock_supabase):
        """Test getting latest audit when none exists."""
        from src.api.accessibility import get_latest_audit

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_latest_audit(project_id=str(uuid.uuid4()))

            assert response is None

    @pytest.mark.asyncio
    async def test_get_latest_audit_table_not_exists(self, mock_env_vars, mock_supabase):
        """Test handling when table doesn't exist."""
        from src.api.accessibility import get_latest_audit

        mock_supabase.request.return_value = {"data": None, "error": "relation does not exist"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_latest_audit(project_id=str(uuid.uuid4()))

            assert response is None

    @pytest.mark.asyncio
    async def test_get_latest_audit_invalid_uuid(self, mock_env_vars):
        """Test with invalid UUID."""
        from src.api.accessibility import get_latest_audit

        with pytest.raises(HTTPException) as exc_info:
            await get_latest_audit(project_id="invalid-uuid")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_latest_audit_db_error(self, mock_env_vars, mock_supabase):
        """Test handling database errors."""
        from src.api.accessibility import get_latest_audit

        mock_supabase.request.return_value = {"data": None, "error": "Connection refused"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_audit(project_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 500


# ============================================================================
# Get Audit by ID Tests
# ============================================================================


class TestGetAudit:
    """Tests for GET /api/v1/accessibility/audit/{audit_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_audit_success(
        self, mock_env_vars, mock_supabase, sample_audit_data, sample_issue_data
    ):
        """Test getting audit by ID successfully."""
        from src.api.accessibility import get_audit

        sample_issue_data["audit_id"] = sample_audit_data["id"]

        mock_supabase.request.side_effect = [
            {"data": [sample_audit_data], "error": None},
            {"data": [sample_issue_data], "error": None},
        ]

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_audit(audit_id=sample_audit_data["id"])

            assert response.audit.id == sample_audit_data["id"]
            assert response.audit.accessibility_score == 85.0

    @pytest.mark.asyncio
    async def test_get_audit_not_found(self, mock_env_vars, mock_supabase):
        """Test getting non-existent audit."""
        from src.api.accessibility import get_audit

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_audit(audit_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_audit_invalid_uuid(self, mock_env_vars):
        """Test with invalid UUID."""
        from src.api.accessibility import get_audit

        with pytest.raises(HTTPException) as exc_info:
            await get_audit(audit_id="invalid-uuid")

        assert exc_info.value.status_code == 400


# ============================================================================
# Get Audit History Tests
# ============================================================================


class TestGetAuditHistory:
    """Tests for GET /api/v1/accessibility/audits/history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self, mock_env_vars, mock_supabase, sample_audit_data):
        """Test getting audit history successfully."""
        from src.api.accessibility import get_audit_history

        audits = [sample_audit_data, {**sample_audit_data, "id": str(uuid.uuid4())}]
        mock_supabase.request.return_value = {"data": audits, "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_audit_history(
                project_id=sample_audit_data["project_id"],
                limit=10,
                offset=0
            )

            assert response["total"] == 2
            assert len(response["audits"]) == 2

    @pytest.mark.asyncio
    async def test_get_history_empty(self, mock_env_vars, mock_supabase):
        """Test getting empty audit history."""
        from src.api.accessibility import get_audit_history

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_audit_history(project_id=str(uuid.uuid4()))

            assert response["total"] == 0
            assert response["audits"] == []

    @pytest.mark.asyncio
    async def test_get_history_table_not_exists(self, mock_env_vars, mock_supabase):
        """Test handling when table doesn't exist."""
        from src.api.accessibility import get_audit_history

        mock_supabase.request.return_value = {"data": None, "error": "42P01"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_audit_history(project_id=str(uuid.uuid4()))

            assert response["audits"] == []
            assert response["total"] == 0

    @pytest.mark.asyncio
    async def test_get_history_with_pagination(self, mock_env_vars, mock_supabase, sample_audit_data):
        """Test history with pagination parameters."""
        from src.api.accessibility import get_audit_history

        mock_supabase.request.return_value = {"data": [sample_audit_data], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_audit_history(
                project_id=sample_audit_data["project_id"],
                limit=5,
                offset=10
            )

            assert response["total"] == 1


# ============================================================================
# Get Issues Tests
# ============================================================================


class TestGetIssues:
    """Tests for GET /api/v1/accessibility/issues endpoint."""

    @pytest.mark.asyncio
    async def test_get_issues_success(self, mock_env_vars, mock_supabase, sample_issue_data):
        """Test getting issues successfully."""
        from src.api.accessibility import get_issues

        mock_supabase.request.return_value = {"data": [sample_issue_data], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_issues(audit_id=sample_issue_data["audit_id"])

            assert len(response["issues"]) == 1
            assert response["issues"][0].rule == "color-contrast"

    @pytest.mark.asyncio
    async def test_get_issues_with_severity_filter(
        self, mock_env_vars, mock_supabase, sample_issue_data
    ):
        """Test getting issues with severity filter."""
        from src.api.accessibility import get_issues

        mock_supabase.request.return_value = {"data": [sample_issue_data], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_issues(
                audit_id=sample_issue_data["audit_id"],
                severity="serious"
            )

            assert len(response["issues"]) == 1

    @pytest.mark.asyncio
    async def test_get_issues_empty(self, mock_env_vars, mock_supabase):
        """Test getting issues when none exist."""
        from src.api.accessibility import get_issues

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_issues(audit_id=str(uuid.uuid4()))

            assert response["issues"] == []

    @pytest.mark.asyncio
    async def test_get_issues_table_not_exists(self, mock_env_vars, mock_supabase):
        """Test handling when table doesn't exist."""
        from src.api.accessibility import get_issues

        mock_supabase.request.return_value = {"data": None, "error": "42P01"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await get_issues(audit_id=str(uuid.uuid4()))

            assert response["issues"] == []


# ============================================================================
# Run Audit Tests
# ============================================================================


class TestRunAudit:
    """Tests for POST /api/v1/accessibility/audit endpoint."""

    @pytest.mark.asyncio
    async def test_run_audit_success(self, mock_env_vars, mock_supabase):
        """Test running audit successfully."""
        from src.api.accessibility import RunAuditRequest, run_audit

        project_id = str(uuid.uuid4())
        audit_data = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "url": "https://example.com",
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
        }

        mock_supabase.insert.return_value = {"data": [audit_data], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            request = RunAuditRequest(
                project_id=project_id,
                url="https://example.com",
                wcag_level="AA"
            )
            response = await run_audit(request)

            assert response.success is True
            assert response.status == "running"
            assert response.audit_id is not None

    @pytest.mark.asyncio
    async def test_run_audit_insert_error(self, mock_env_vars, mock_supabase):
        """Test running audit with insert error."""
        from src.api.accessibility import RunAuditRequest, run_audit

        mock_supabase.insert.return_value = {"data": None, "error": "Insert failed"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            request = RunAuditRequest(
                project_id=str(uuid.uuid4()),
                url="https://example.com"
            )

            with pytest.raises(HTTPException) as exc_info:
                await run_audit(request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_run_audit_with_options(self, mock_env_vars, mock_supabase):
        """Test running audit with all options."""
        from src.api.accessibility import RunAuditRequest, run_audit

        project_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        mock_supabase.insert.return_value = {
            "data": [{
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "status": "running",
                "created_at": datetime.now(UTC).isoformat(),
            }],
            "error": None
        }

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            request = RunAuditRequest(
                project_id=project_id,
                url="https://example.com",
                wcag_level="AAA",
                include_best_practices=True,
                triggered_by=user_id
            )
            response = await run_audit(request)

            assert response.success is True


# ============================================================================
# Update Audit Tests
# ============================================================================


class TestUpdateAudit:
    """Tests for PATCH /api/v1/accessibility/audit/{audit_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_audit_status(self, mock_env_vars, mock_supabase):
        """Test updating audit status."""
        from src.api.accessibility import update_audit

        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await update_audit(
                audit_id=str(uuid.uuid4()),
                status="completed"
            )

            assert response["success"] is True

    @pytest.mark.asyncio
    async def test_update_audit_score(self, mock_env_vars, mock_supabase):
        """Test updating audit score."""
        from src.api.accessibility import update_audit

        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await update_audit(
                audit_id=str(uuid.uuid4()),
                accessibility_score=92.5
            )

            assert response["success"] is True

    @pytest.mark.asyncio
    async def test_update_audit_no_data(self, mock_env_vars, mock_supabase):
        """Test update with no data provided."""
        from src.api.accessibility import update_audit

        # The function calls get_supabase_client() before checking for empty data
        # So we need to mock it to avoid side effects
        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                # Pass actual None values instead of Query defaults
                await update_audit(audit_id=str(uuid.uuid4()), status=None, accessibility_score=None)

            assert exc_info.value.status_code == 400
            assert "No update data provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_audit_error(self, mock_env_vars, mock_supabase):
        """Test update with database error."""
        from src.api.accessibility import update_audit

        mock_supabase.update.return_value = {"data": None, "error": "Update failed"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await update_audit(
                    audit_id=str(uuid.uuid4()),
                    status="completed"
                )

            assert exc_info.value.status_code == 500


# ============================================================================
# Create Issues Batch Tests
# ============================================================================


class TestCreateIssuesBatch:
    """Tests for POST /api/v1/accessibility/issues/batch endpoint."""

    @pytest.mark.asyncio
    async def test_create_issues_batch_success(self, mock_env_vars, mock_supabase):
        """Test creating issues batch successfully."""
        from src.api.accessibility import create_issues_batch

        issues = [
            {
                "rule": "color-contrast",
                "severity": "serious",
                "description": "Insufficient contrast",
                "element_selector": "#header",
                "wcag_criteria": ["1.4.3"],
            },
            {
                "rule": "alt-text",
                "severity": "critical",
                "description": "Missing alt text",
                "element_selector": "img.hero",
            },
        ]

        mock_supabase.insert.return_value = {"data": issues, "error": None}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await create_issues_batch(
                audit_id=str(uuid.uuid4()),
                issues=issues
            )

            assert response["success"] is True
            assert response["created"] == 2

    @pytest.mark.asyncio
    async def test_create_issues_batch_empty(self, mock_env_vars, mock_supabase):
        """Test creating empty issues batch."""
        from src.api.accessibility import create_issues_batch

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            response = await create_issues_batch(
                audit_id=str(uuid.uuid4()),
                issues=[]
            )

            assert response["success"] is True
            assert response["created"] == 0

    @pytest.mark.asyncio
    async def test_create_issues_batch_error(self, mock_env_vars, mock_supabase):
        """Test creating issues batch with error."""
        from src.api.accessibility import create_issues_batch

        mock_supabase.insert.return_value = {"data": None, "error": "Insert failed"}

        with patch("src.api.accessibility.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await create_issues_batch(
                    audit_id=str(uuid.uuid4()),
                    issues=[{"rule": "test", "severity": "minor", "description": "test"}]
                )

            assert exc_info.value.status_code == 500


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_run_audit_request(self, mock_env_vars):
        """Test RunAuditRequest validation."""
        from src.api.accessibility import RunAuditRequest

        request = RunAuditRequest(
            project_id=str(uuid.uuid4()),
            url="https://example.com",
            wcag_level="AA"
        )

        assert request.wcag_level == "AA"
        assert request.include_best_practices is True

    def test_run_audit_request_defaults(self, mock_env_vars):
        """Test RunAuditRequest default values."""
        from src.api.accessibility import RunAuditRequest

        request = RunAuditRequest(
            project_id=str(uuid.uuid4()),
            url="https://example.com"
        )

        assert request.wcag_level == "AA"
        assert request.include_best_practices is True
        assert request.triggered_by is None


class TestResponseModels:
    """Tests for response model validation."""

    def test_accessibility_issue_response(self, mock_env_vars, sample_issue_data):
        """Test AccessibilityIssueResponse model."""
        from src.api.accessibility import AccessibilityIssueResponse

        response = AccessibilityIssueResponse(**sample_issue_data)

        assert response.rule == "color-contrast"
        assert response.severity == "serious"
        assert "1.4.3" in response.wcag_criteria

    def test_quality_audit_response(self, mock_env_vars, sample_audit_data):
        """Test QualityAuditResponse model."""
        from src.api.accessibility import QualityAuditResponse

        response = QualityAuditResponse(**sample_audit_data)

        assert response.accessibility_score == 85.0
        assert response.status == "completed"

    def test_run_audit_response(self, mock_env_vars):
        """Test RunAuditResponse model."""
        from src.api.accessibility import RunAuditResponse

        response = RunAuditResponse(
            success=True,
            message="Audit started",
            audit_id=str(uuid.uuid4()),
            status="running"
        )

        assert response.success is True
        assert response.status == "running"


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestMapAuditResponse:
    """Tests for map_audit_response helper function."""

    def test_map_audit_response_complete(self, mock_env_vars, sample_audit_data):
        """Test mapping complete audit data."""
        from src.api.accessibility import map_audit_response

        response = map_audit_response(sample_audit_data)

        assert response.id == sample_audit_data["id"]
        assert response.accessibility_score == 85.0
        assert response.lcp_ms == 2500.0

    def test_map_audit_response_minimal(self, mock_env_vars):
        """Test mapping minimal audit data."""
        from src.api.accessibility import map_audit_response

        minimal = {
            "id": str(uuid.uuid4()),
            "project_id": str(uuid.uuid4()),
            "url": "https://example.com",
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
        }

        response = map_audit_response(minimal)

        assert response.id == minimal["id"]
        assert response.accessibility_score is None


class TestMapIssueResponse:
    """Tests for map_issue_response helper function."""

    def test_map_issue_response_complete(self, mock_env_vars, sample_issue_data):
        """Test mapping complete issue data."""
        from src.api.accessibility import map_issue_response

        response = map_issue_response(sample_issue_data)

        assert response.rule == "color-contrast"
        assert response.suggested_fix is not None
        assert response.wcag_criteria is not None

    def test_map_issue_response_minimal(self, mock_env_vars):
        """Test mapping minimal issue data."""
        from src.api.accessibility import map_issue_response

        minimal = {
            "id": str(uuid.uuid4()),
            "audit_id": str(uuid.uuid4()),
            "rule": "test-rule",
            "severity": "minor",
            "description": "Test issue",
            "created_at": datetime.now(UTC).isoformat(),
        }

        response = map_issue_response(minimal)

        assert response.rule == "test-rule"
        assert response.element_selector is None
        assert response.suggested_fix is None
