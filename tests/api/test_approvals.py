"""Tests for Approvals API endpoints (src/api/approvals.py).

Covers:
- List pending approvals
- Get pending approval
- Approve action
- Resume execution
- Get thread state
- Update thread state
- Abort execution
- Get approval config
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
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.require_healing_approval = True
    settings.require_test_plan_approval = True
    settings.require_human_approval_for_healing = True
    settings.approval_timeout_seconds = 3600
    return settings


@pytest.fixture
def mock_state():
    """Create a mock graph state at an approval breakpoint."""
    state = MagicMock()
    state.config = {"configurable": {"checkpoint_id": "cp-123"}}
    state.next = ["self_heal"]  # Paused at self_heal
    state.values = {
        "run_id": "run-123",
        "started_at": datetime.now(UTC).isoformat(),
        "iteration": 5,
        "passed_count": 10,
        "failed_count": 3,
        "skipped_count": 1,
        "total_cost": 0.05,
        "current_test_index": 14,
        "test_plan": [{"id": "t1"}, {"id": "t2"}],
        "healing_queue": ["test-1", "test-2"],
        "failures": [
            {"test_id": "test-1", "failure_type": "selector", "root_cause": "Element not found"},
            {"test_id": "test-2", "failure_type": "timeout", "root_cause": "Page timeout"},
        ],
        "error": None,
        "should_continue": True,
    }
    state.metadata = {"created_at": datetime.now(UTC).isoformat()}
    return state


@pytest.fixture
def mock_completed_state():
    """Create a mock completed graph state (not paused)."""
    state = MagicMock()
    state.next = None  # Not paused
    state.values = {
        "iteration": 10,
        "passed_count": 15,
        "failed_count": 2,
        "skipped_count": 1,
        "error": None,
        "should_continue": False,
    }
    return state


@pytest.fixture
def mock_compiled_app(mock_state):
    """Create a mock compiled LangGraph app."""
    app = AsyncMock()
    app.aget_state = AsyncMock(return_value=mock_state)
    app.aupdate_state = AsyncMock()
    app.ainvoke = AsyncMock(return_value={
        "passed_count": 15,
        "failed_count": 2,
        "error": None,
    })
    return app


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock CheckpointManager."""
    manager = MagicMock()
    manager.get_pending_approvals = AsyncMock(return_value=[
        {"thread_id": "thread-1", "paused_at": "self_heal", "created_at": datetime.now(UTC).isoformat()},
        {"thread_id": "thread-2", "paused_at": "execute_test", "created_at": datetime.now(UTC).isoformat()},
    ])
    manager.get_approval_details = AsyncMock(return_value={
        "description": "Self-heal 2 failed tests",
        "context": {
            "iteration": 5,
            "passed_count": 10,
            "failed_count": 3,
        },
    })
    return manager


# ============================================================================
# List Pending Approvals Tests
# ============================================================================


class TestListPendingApprovals:
    """Tests for GET /api/v1/approvals/pending endpoint."""

    @pytest.mark.asyncio
    async def test_list_pending_success(self, mock_env_vars, mock_checkpoint_manager):
        """Test listing pending approvals successfully."""
        from src.api.approvals import list_pending_approvals

        with patch("src.api.approvals.CheckpointManager", return_value=mock_checkpoint_manager):
            response = await list_pending_approvals()

            assert len(response) == 2
            assert response[0].thread_id == "thread-1"

    @pytest.mark.asyncio
    async def test_list_pending_empty(self, mock_env_vars):
        """Test listing when no pending approvals."""
        from src.api.approvals import list_pending_approvals

        mock_manager = MagicMock()
        mock_manager.get_pending_approvals = AsyncMock(return_value=[])

        with patch("src.api.approvals.CheckpointManager", return_value=mock_manager):
            response = await list_pending_approvals()

            assert response == []

    @pytest.mark.asyncio
    async def test_list_pending_error(self, mock_env_vars):
        """Test listing handles errors."""
        from src.api.approvals import list_pending_approvals

        mock_manager = MagicMock()
        mock_manager.get_pending_approvals = AsyncMock(side_effect=Exception("Database error"))

        with patch("src.api.approvals.CheckpointManager", return_value=mock_manager):
            with pytest.raises(HTTPException) as exc_info:
                await list_pending_approvals()

            assert exc_info.value.status_code == 500


# ============================================================================
# Get Pending Approval Tests
# ============================================================================


class TestGetPendingApproval:
    """Tests for GET /api/v1/approvals/pending/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_pending_success(self, mock_env_vars, mock_compiled_app, mock_settings, mock_state):
        """Test getting pending approval details."""
        from src.api.approvals import get_pending_approval

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)), \
             patch("src.api.approvals.get_settings", return_value=mock_settings), \
             patch("src.api.approvals.get_interrupt_nodes", return_value=["self_heal", "execute_test"]):

            response = await get_pending_approval(thread_id=str(uuid.uuid4()))

            assert response.paused_before == "self_heal"
            assert "healing_queue_count" in response.state_summary

    @pytest.mark.asyncio
    async def test_get_pending_not_found(self, mock_env_vars, mock_completed_state):
        """Test getting approval when not paused."""
        from src.api.approvals import get_pending_approval

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_completed_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            with pytest.raises(HTTPException) as exc_info:
                await get_pending_approval(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_pending_not_at_approval_point(self, mock_env_vars, mock_state, mock_settings):
        """Test getting approval when paused at non-approval node."""
        from src.api.approvals import get_pending_approval

        mock_state.next = ["report"]  # Not an approval node

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)), \
             patch("src.api.approvals.get_settings", return_value=mock_settings), \
             patch("src.api.approvals.get_interrupt_nodes", return_value=["self_heal", "execute_test"]):

            with pytest.raises(HTTPException) as exc_info:
                await get_pending_approval(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_pending_invalid_uuid(self, mock_env_vars):
        """Test getting approval with invalid UUID."""
        from src.api.approvals import get_pending_approval

        with pytest.raises(HTTPException) as exc_info:
            await get_pending_approval(thread_id="invalid-uuid")

        assert exc_info.value.status_code == 400


# ============================================================================
# Approve Action Tests
# ============================================================================


class TestApproveAction:
    """Tests for POST /api/v1/approvals/approve endpoint."""

    @pytest.mark.asyncio
    async def test_approve_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test approving action successfully."""
        from src.api.approvals import ApprovalRequest, approve_action

        # After approval, state should show not paused
        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            request = ApprovalRequest(
                thread_id=str(uuid.uuid4()),
                approved=True,
                reason="Looks good"
            )
            response = await approve_action(request)

            assert response.status == "approved"
            assert response.result["completed"] is True

    @pytest.mark.asyncio
    async def test_approve_with_modifications(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test approving with state modifications."""
        from src.api.approvals import ApprovalRequest, approve_action

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            request = ApprovalRequest(
                thread_id=str(uuid.uuid4()),
                approved=True,
                modifications={"healing_queue": ["test-1"]},  # Only heal first test
            )
            response = await approve_action(request)

            mock_compiled_app.aupdate_state.assert_called_once()
            assert response.status == "approved"

    @pytest.mark.asyncio
    async def test_reject_action(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test rejecting action."""
        from src.api.approvals import ApprovalRequest, approve_action

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            request = ApprovalRequest(
                thread_id=str(uuid.uuid4()),
                approved=False,
                reason="Too risky"
            )
            response = await approve_action(request)

            assert response.status == "rejected"
            # State should be updated to skip healing
            call_args = mock_compiled_app.aupdate_state.call_args
            assert call_args[0][1]["healing_queue"] == []

    @pytest.mark.asyncio
    async def test_approve_no_pending(self, mock_env_vars, mock_completed_state):
        """Test approving when no pending action."""
        from src.api.approvals import ApprovalRequest, approve_action

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=mock_completed_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            request = ApprovalRequest(
                thread_id=str(uuid.uuid4()),
                approved=True
            )

            with pytest.raises(HTTPException) as exc_info:
                await approve_action(request)

            assert exc_info.value.status_code == 404


# ============================================================================
# Resume Execution Tests
# ============================================================================


class TestResumeExecution:
    """Tests for POST /api/v1/approvals/resume/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_resume_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test resuming execution successfully."""
        from src.api.approvals import ResumeRequest, resume_execution

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            response = await resume_execution(thread_id=str(uuid.uuid4()))

            assert response.status == "resumed"
            assert response.result["completed"] is True

    @pytest.mark.asyncio
    async def test_resume_with_modifications(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test resuming with modifications."""
        from src.api.approvals import ResumeRequest, resume_execution

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            request = ResumeRequest(modifications={"should_continue": False})
            response = await resume_execution(
                thread_id=str(uuid.uuid4()),
                request=request
            )

            mock_compiled_app.aupdate_state.assert_called_once()
            assert response.status == "resumed"

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, mock_env_vars, mock_completed_state):
        """Test resuming when not paused."""
        from src.api.approvals import resume_execution

        mock_app = AsyncMock()
        mock_completed_state.next = None
        mock_app.aget_state = AsyncMock(return_value=mock_completed_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            with pytest.raises(HTTPException) as exc_info:
                await resume_execution(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_resume_thread_not_found(self, mock_env_vars):
        """Test resuming non-existent thread."""
        from src.api.approvals import resume_execution

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            with pytest.raises(HTTPException) as exc_info:
                await resume_execution(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 404


# ============================================================================
# Get Thread State Tests
# ============================================================================


class TestGetThreadState:
    """Tests for GET /api/v1/approvals/state/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_state_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test getting thread state successfully."""
        from src.api.approvals import get_thread_state

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            response = await get_thread_state(thread_id=str(uuid.uuid4()))

            assert response["is_paused"] is True
            assert response["next_node"] == "self_heal"
            assert "values" in response

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, mock_env_vars):
        """Test getting state for non-existent thread."""
        from src.api.approvals import get_thread_state

        mock_app = AsyncMock()
        mock_app.aget_state = AsyncMock(return_value=None)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            with pytest.raises(HTTPException) as exc_info:
                await get_thread_state(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 404


# ============================================================================
# Update Thread State Tests
# ============================================================================


class TestUpdateThreadState:
    """Tests for PATCH /api/v1/approvals/state/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_state_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test updating thread state successfully."""
        from src.api.approvals import StateModification, update_thread_state

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            modifications = StateModification(
                healing_queue=["test-1"],
                should_continue=True
            )
            response = await update_thread_state(
                thread_id=str(uuid.uuid4()),
                modifications=modifications
            )

            assert response["success"] is True
            assert "healing_queue" in response["modified_keys"]

    @pytest.mark.asyncio
    async def test_update_state_no_modifications(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test update with no modifications."""
        from src.api.approvals import StateModification, update_thread_state

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            modifications = StateModification()  # No modifications

            with pytest.raises(HTTPException) as exc_info:
                await update_thread_state(
                    thread_id=str(uuid.uuid4()),
                    modifications=modifications
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_update_completed_thread(self, mock_env_vars, mock_completed_state):
        """Test updating completed thread."""
        from src.api.approvals import StateModification, update_thread_state

        mock_app = AsyncMock()
        mock_completed_state.next = None
        mock_app.aget_state = AsyncMock(return_value=mock_completed_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            modifications = StateModification(should_continue=True)

            with pytest.raises(HTTPException) as exc_info:
                await update_thread_state(
                    thread_id=str(uuid.uuid4()),
                    modifications=modifications
                )

            assert exc_info.value.status_code == 400


# ============================================================================
# Abort Execution Tests
# ============================================================================


class TestAbortExecution:
    """Tests for DELETE /api/v1/approvals/{thread_id} endpoint."""

    @pytest.mark.asyncio
    async def test_abort_success(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test aborting execution successfully."""
        from src.api.approvals import abort_execution

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            response = await abort_execution(
                thread_id=str(uuid.uuid4()),
                reason="Manual abort"
            )

            assert response["success"] is True
            assert "Manual abort" in response["reason"]

    @pytest.mark.asyncio
    async def test_abort_with_default_reason(self, mock_env_vars, mock_compiled_app, mock_state):
        """Test aborting with default reason."""
        from src.api.approvals import abort_execution

        completed_state = MagicMock()
        completed_state.next = None

        mock_compiled_app.aget_state = AsyncMock(side_effect=[mock_state, completed_state])

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_compiled_app)):
            response = await abort_execution(thread_id=str(uuid.uuid4()))

            assert "aborted by user" in response["reason"]

    @pytest.mark.asyncio
    async def test_abort_not_paused(self, mock_env_vars, mock_completed_state):
        """Test aborting when not paused."""
        from src.api.approvals import abort_execution

        mock_app = AsyncMock()
        mock_completed_state.next = None
        mock_app.aget_state = AsyncMock(return_value=mock_completed_state)

        with patch("src.api.approvals._get_compiled_graph", AsyncMock(return_value=mock_app)):
            with pytest.raises(HTTPException) as exc_info:
                await abort_execution(thread_id=str(uuid.uuid4()))

            assert exc_info.value.status_code == 400


# ============================================================================
# Get Approval Config Tests
# ============================================================================


class TestGetApprovalConfig:
    """Tests for GET /api/v1/approvals/config endpoint."""

    @pytest.mark.asyncio
    async def test_get_config_with_breakpoints(self, mock_env_vars, mock_settings):
        """Test getting config with breakpoints enabled."""
        from src.api.approvals import get_approval_config

        with patch("src.api.approvals.get_settings", return_value=mock_settings), \
             patch("src.api.approvals.get_interrupt_nodes", return_value=["self_heal", "execute_test"]):

            response = await get_approval_config()

            assert response["breakpoints_enabled"] is True
            assert "self_heal" in response["interrupt_before"]
            assert response["settings"]["require_healing_approval"] is True

    @pytest.mark.asyncio
    async def test_get_config_no_breakpoints(self, mock_env_vars, mock_settings):
        """Test getting config with no breakpoints."""
        from src.api.approvals import get_approval_config

        mock_settings.require_healing_approval = False
        mock_settings.require_test_plan_approval = False

        with patch("src.api.approvals.get_settings", return_value=mock_settings), \
             patch("src.api.approvals.get_interrupt_nodes", return_value=[]):

            response = await get_approval_config()

            assert response["breakpoints_enabled"] is False
            assert response["interrupt_before"] == []


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_approval_request(self, mock_env_vars):
        """Test ApprovalRequest validation."""
        from src.api.approvals import ApprovalRequest

        request = ApprovalRequest(
            thread_id="thread-123",
            approved=True,
            reason="LGTM"
        )

        assert request.approved is True
        assert request.reason == "LGTM"

    def test_resume_request(self, mock_env_vars):
        """Test ResumeRequest validation."""
        from src.api.approvals import ResumeRequest

        request = ResumeRequest(
            modifications={"key": "value"}
        )

        assert request.modifications == {"key": "value"}

    def test_state_modification(self, mock_env_vars):
        """Test StateModification validation."""
        from src.api.approvals import StateModification

        mod = StateModification(
            healing_queue=["t1", "t2"],
            should_continue=True,
            test_plan=[{"id": "t1"}],
            custom={"extra_key": "extra_value"}
        )

        assert len(mod.healing_queue) == 2
        assert mod.custom["extra_key"] == "extra_value"


class TestResponseModels:
    """Tests for response model validation."""

    def test_pending_approval(self, mock_env_vars):
        """Test PendingApproval model."""
        from src.api.approvals import PendingApproval

        approval = PendingApproval(
            thread_id="thread-123",
            paused_at="2024-01-15T12:00:00Z",
            paused_before="self_heal",
            state_summary={"iteration": 5, "passed_count": 10},
            action_description="Self-heal 2 failed tests"
        )

        assert approval.thread_id == "thread-123"
        assert approval.paused_before == "self_heal"

    def test_approval_response(self, mock_env_vars):
        """Test ApprovalResponse model."""
        from src.api.approvals import ApprovalResponse

        response = ApprovalResponse(
            status="approved",
            thread_id="thread-123",
            message="Action approved",
            result={"passed": 15, "failed": 2}
        )

        assert response.status == "approved"
        assert response.result["passed"] == 15


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestBuildActionDescription:
    """Tests for _build_action_description helper function."""

    def test_build_description_self_heal(self, mock_env_vars):
        """Test building description for self_heal node."""
        from src.api.approvals import _build_action_description

        values = {
            "failures": [{"test_id": "t1"}, {"test_id": "t2"}],
            "healing_queue": ["t1", "t2"],
        }

        desc = _build_action_description("self_heal", values)

        assert "Self-heal" in desc
        assert "2" in desc

    def test_build_description_execute_test(self, mock_env_vars):
        """Test building description for execute_test node."""
        from src.api.approvals import _build_action_description

        values = {"test_plan": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]}

        desc = _build_action_description("execute_test", values)

        assert "Execute test plan" in desc
        assert "3 tests" in desc

    def test_build_description_report(self, mock_env_vars):
        """Test building description for report node."""
        from src.api.approvals import _build_action_description

        desc = _build_action_description("report", {})

        assert "report" in desc.lower()

    def test_build_description_unknown(self, mock_env_vars):
        """Test building description for unknown node."""
        from src.api.approvals import _build_action_description

        desc = _build_action_description("custom_node", {})

        assert "custom_node" in desc


class TestBuildStateSummary:
    """Tests for _build_state_summary helper function."""

    def test_build_summary(self, mock_env_vars):
        """Test building state summary."""
        from src.api.approvals import _build_state_summary

        values = {
            "iteration": 5,
            "passed_count": 10,
            "failed_count": 3,
            "skipped_count": 1,
            "healing_queue": ["t1", "t2"],
            "total_cost": 0.05,
            "error": None,
        }

        summary = _build_state_summary(values)

        assert summary["iteration"] == 5
        assert summary["passed_count"] == 10
        assert summary["healing_queue_count"] == 2
        assert summary["error"] is None

    def test_build_summary_missing_values(self, mock_env_vars):
        """Test building summary with missing values."""
        from src.api.approvals import _build_state_summary

        summary = _build_state_summary({})

        assert summary["iteration"] == 0
        assert summary["passed_count"] == 0
        assert summary["healing_queue_count"] == 0
