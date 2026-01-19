"""Tests for Scheduling API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestSchedulingModels:
    """Tests for scheduling request/response models."""

    def test_schedule_create_request_basic(self, mock_env_vars):
        """Test ScheduleCreateRequest with basic values."""
        from src.api.scheduling import ScheduleCreateRequest

        request = ScheduleCreateRequest(
            project_id="project-123",
            name="Daily Test Run",
            cron_expression="0 9 * * *",
            app_url="https://example.com",
        )

        assert request.project_id == "project-123"
        assert request.name == "Daily Test Run"
        assert request.cron_expression == "0 9 * * *"
        assert request.enabled is True
        assert request.timeout_minutes == 60

    def test_schedule_create_request_full(self, mock_env_vars):
        """Test ScheduleCreateRequest with all values."""
        from src.api.scheduling import ScheduleCreateRequest

        request = ScheduleCreateRequest(
            project_id="project-123",
            name="Comprehensive Tests",
            cron_expression="0 0 * * 1-5",
            test_ids=["test-1", "test-2"],
            app_url="https://staging.example.com",
            enabled=True,
            notify_on_failure=True,
            notification_channels={"email": True, "slack": True},
            description="Run all tests on weekdays",
            timeout_minutes=120,
            retry_count=2,
            environment_variables={"ENV": "staging"},
            tags=["regression", "smoke"],
        )

        assert request.test_ids == ["test-1", "test-2"]
        assert request.timeout_minutes == 120
        assert request.retry_count == 2
        assert "regression" in request.tags

    def test_schedule_create_request_invalid_cron(self, mock_env_vars):
        """Test ScheduleCreateRequest with invalid cron expression."""
        from pydantic import ValidationError

        from src.api.scheduling import ScheduleCreateRequest

        with pytest.raises(ValidationError) as exc_info:
            ScheduleCreateRequest(
                project_id="project-123",
                name="Invalid Schedule",
                cron_expression="invalid cron",
                app_url="https://example.com",
            )

        assert "cron_expression" in str(exc_info.value).lower() or "5 fields" in str(exc_info.value)

    def test_schedule_update_request(self, mock_env_vars):
        """Test ScheduleUpdateRequest model."""
        from src.api.scheduling import ScheduleUpdateRequest

        request = ScheduleUpdateRequest(
            name="Updated Name",
            cron_expression="0 12 * * *",
            enabled=False,
        )

        assert request.name == "Updated Name"
        assert request.enabled is False

    def test_schedule_response_model(self, mock_env_vars):
        """Test ScheduleResponse model."""
        from src.api.scheduling import ScheduleResponse

        response = ScheduleResponse(
            id="schedule-123",
            project_id="project-456",
            name="Daily Tests",
            cron_expression="0 9 * * *",
            cron_readable="Daily at 9:00 AM",
            test_ids=None,
            app_url="https://example.com",
            enabled=True,
            status="active",
            notify_on_failure=True,
            notification_channels={"email": True},
            description="Daily regression",
            timeout_minutes=60,
            retry_count=0,
            environment_variables=None,
            tags=None,
            next_run_at="2024-01-02T09:00:00Z",
            last_run_at="2024-01-01T09:00:00Z",
            last_run_status="success",
            run_count=10,
            success_count=9,
            failure_count=1,
            avg_duration_seconds=120.5,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            created_by="user-123",
        )

        assert response.status == "active"
        assert response.success_count == 9

    def test_schedule_run_response_model(self, mock_env_vars):
        """Test ScheduleRunResponse model."""
        from src.api.scheduling import ScheduleRunResponse

        response = ScheduleRunResponse(
            id="run-123",
            schedule_id="schedule-456",
            status="success",
            started_at="2024-01-01T09:00:00Z",
            completed_at="2024-01-01T09:02:00Z",
            duration_seconds=120,
            trigger_type="scheduled",
            triggered_by=None,
            test_results={"passed": 10, "failed": 0},
            error_message=None,
            retry_attempt=0,
            logs_url="https://logs.example.com/run-123",
        )

        assert response.status == "success"
        assert response.duration_seconds == 120

    def test_trigger_response_model(self, mock_env_vars):
        """Test TriggerResponse model."""
        from src.api.scheduling import TriggerResponse

        response = TriggerResponse(
            success=True,
            message="Test run started",
            run_id="run-123",
            schedule_id="schedule-456",
            started_at="2024-01-01T09:00:00Z",
        )

        assert response.success is True


class TestCronValidation:
    """Tests for cron expression validation."""

    def test_validate_cron_expression_valid(self, mock_env_vars):
        """Test validate_cron_expression with valid expressions."""
        from src.api.scheduling import validate_cron_expression

        valid_expressions = [
            "* * * * *",
            "0 9 * * *",
            "0 0 1 * *",
            "*/15 * * * *",
            "0 9 * * 1-5",
            "0,30 9,17 * * *",
            "0 */2 * * *",
        ]

        for expr in valid_expressions:
            is_valid, error = validate_cron_expression(expr)
            assert is_valid, f"Expected {expr} to be valid, got error: {error}"
            assert error is None

    def test_validate_cron_expression_invalid(self, mock_env_vars):
        """Test validate_cron_expression with invalid expressions."""
        from src.api.scheduling import validate_cron_expression

        invalid_expressions = [
            ("", "cannot be empty"),
            ("* * *", "5 fields"),
            ("* * * * * *", "5 fields"),
            ("60 * * * *", "out of range"),
            ("* 24 * * *", "out of range"),
            ("* * 32 * *", "out of range"),
            ("* * * 13 *", "out of range"),
            ("* * * * 7", "out of range"),
        ]

        for expr, expected_error in invalid_expressions:
            is_valid, error = validate_cron_expression(expr)
            assert not is_valid, f"Expected {expr} to be invalid"
            assert error is not None

    def test_validate_cron_expression_special_characters(self, mock_env_vars):
        """Test cron validation with special characters."""
        from src.api.scheduling import validate_cron_expression

        # Step values
        is_valid, _ = validate_cron_expression("*/10 * * * *")
        assert is_valid

        # Range
        is_valid, _ = validate_cron_expression("0 9-17 * * *")
        assert is_valid

        # List
        is_valid, _ = validate_cron_expression("0,15,30,45 * * * *")
        assert is_valid

    def test_validate_cron_expression_invalid_step(self, mock_env_vars):
        """Test cron validation with invalid step value."""
        from src.api.scheduling import validate_cron_expression

        is_valid, error = validate_cron_expression("*/0 * * * *")
        assert not is_valid
        assert "step" in error.lower()


class TestCronCalculation:
    """Tests for cron next run time calculation."""

    def test_calculate_next_run_basic(self, mock_env_vars):
        """Test calculate_next_run with basic expressions."""
        from src.api.scheduling import calculate_next_run

        # Every minute
        from_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        next_run = calculate_next_run("* * * * *", from_time)
        assert next_run is not None
        assert next_run > from_time

    def test_calculate_next_run_invalid(self, mock_env_vars):
        """Test calculate_next_run with invalid expression."""
        from src.api.scheduling import calculate_next_run

        next_run = calculate_next_run("invalid")
        assert next_run is None

    def test_calculate_next_run_daily(self, mock_env_vars):
        """Test calculate_next_run for daily schedule."""
        from src.api.scheduling import calculate_next_run

        from_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        next_run = calculate_next_run("0 9 * * *", from_time)
        assert next_run is not None
        assert next_run.hour == 9
        assert next_run.minute == 0

    def test_calculate_next_run_with_step(self, mock_env_vars):
        """Test calculate_next_run with step values."""
        from src.api.scheduling import calculate_next_run

        from_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        next_run = calculate_next_run("*/15 * * * *", from_time)
        assert next_run is not None
        assert next_run.minute in [0, 15, 30, 45]


class TestCronToReadable:
    """Tests for cron to human-readable conversion."""

    def test_cron_to_readable_midnight(self, mock_env_vars):
        """Test cron_to_readable for midnight."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 0 * * *")
        assert "midnight" in readable.lower()

    def test_cron_to_readable_daily_9am(self, mock_env_vars):
        """Test cron_to_readable for 9 AM daily."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 9 * * *")
        assert "daily" in readable.lower()
        assert "9:00" in readable

    def test_cron_to_readable_weekdays(self, mock_env_vars):
        """Test cron_to_readable for weekdays."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 9 * * 1-5")
        assert "weekday" in readable.lower()

    def test_cron_to_readable_every_minutes(self, mock_env_vars):
        """Test cron_to_readable for minute intervals."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("*/15 * * * *")
        assert "15 minutes" in readable.lower()

    def test_cron_to_readable_every_hours(self, mock_env_vars):
        """Test cron_to_readable for hour intervals."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 */2 * * *")
        assert "2 hours" in readable.lower()

    def test_cron_to_readable_monthly(self, mock_env_vars):
        """Test cron_to_readable for monthly."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 0 1 * *")
        assert "monthly" in readable.lower()

    def test_cron_to_readable_weekly_sunday(self, mock_env_vars):
        """Test cron_to_readable for weekly on Sunday."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("0 0 * * 0")
        assert "sunday" in readable.lower()

    def test_cron_to_readable_default(self, mock_env_vars):
        """Test cron_to_readable for unrecognized pattern."""
        from src.api.scheduling import cron_to_readable

        readable = cron_to_readable("15 10 15 3 2")
        assert "cron:" in readable.lower()


class TestScheduleHelperFunctions:
    """Tests for schedule helper functions."""

    def test_schedule_to_response(self, mock_env_vars):
        """Test schedule_to_response function."""
        from src.api.scheduling import schedule_runs, schedule_to_response

        schedule = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Test Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": True,
            "description": "Daily tests",
            "timeout_minutes": 60,
            "retry_count": 0,
            "created_at": "2024-01-01T00:00:00Z",
        }

        # Add some test runs
        schedule_runs["schedule-123"] = [
            {"id": "run-1", "status": "success", "duration_seconds": 100, "started_at": "2024-01-01T09:00:00Z"},
            {"id": "run-2", "status": "success", "duration_seconds": 120, "started_at": "2024-01-02T09:00:00Z"},
        ]

        try:
            response = schedule_to_response(schedule)

            assert response.id == "schedule-123"
            assert response.success_count == 2
            assert response.failure_count == 0
            assert response.avg_duration_seconds == 110.0
        finally:
            schedule_runs.clear()


class TestScheduleDBHelpers:
    """Tests for schedule database helper functions."""

    @pytest.mark.asyncio
    async def test_get_schedule_from_db_in_memory(self, mock_env_vars):
        """Test _get_schedule_from_db with in-memory fallback."""
        from src.api.scheduling import _get_schedule_from_db, schedules

        schedules["test-schedule"] = {"id": "test-schedule", "name": "Test"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            result = await _get_schedule_from_db("test-schedule")
            assert result is not None
            assert result["id"] == "test-schedule"

        schedules.clear()

    @pytest.mark.asyncio
    async def test_get_schedule_from_db_supabase(self, mock_env_vars):
        """Test _get_schedule_from_db with Supabase."""
        from src.api.scheduling import _get_schedule_from_db

        mock_supabase = MagicMock()
        mock_supabase.select = AsyncMock(return_value=[
            {"id": "schedule-123", "name": "Test Schedule"}
        ])

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=mock_supabase)):
            result = await _get_schedule_from_db("schedule-123")
            assert result is not None
            assert result["id"] == "schedule-123"

    @pytest.mark.asyncio
    async def test_save_schedule_to_db_in_memory(self, mock_env_vars):
        """Test _save_schedule_to_db with in-memory fallback."""
        from src.api.scheduling import _save_schedule_to_db, schedules

        schedules.clear()

        schedule = {"id": "new-schedule", "name": "New Schedule"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            result = await _save_schedule_to_db(schedule)
            assert result is True
            assert "new-schedule" in schedules

        schedules.clear()

    @pytest.mark.asyncio
    async def test_update_schedule_in_db(self, mock_env_vars):
        """Test _update_schedule_in_db."""
        from src.api.scheduling import _update_schedule_in_db, schedules

        schedules["test-schedule"] = {"id": "test-schedule", "name": "Original"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            result = await _update_schedule_in_db("test-schedule", {"name": "Updated"})
            assert result is True
            assert schedules["test-schedule"]["name"] == "Updated"

        schedules.clear()

    @pytest.mark.asyncio
    async def test_delete_schedule_from_db(self, mock_env_vars):
        """Test _delete_schedule_from_db."""
        from src.api.scheduling import _delete_schedule_from_db, schedules

        schedules["to-delete"] = {"id": "to-delete"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            result = await _delete_schedule_from_db("to-delete")
            assert result is True
            assert "to-delete" not in schedules

        schedules.clear()

    @pytest.mark.asyncio
    async def test_save_schedule_run_to_db(self, mock_env_vars):
        """Test _save_schedule_run_to_db."""
        from src.api.scheduling import _save_schedule_run_to_db, schedule_runs

        schedule_runs.clear()

        run = {"id": "run-123", "schedule_id": "schedule-456", "status": "pending"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            result = await _save_schedule_run_to_db(run)
            assert result is True
            assert "schedule-456" in schedule_runs
            assert len(schedule_runs["schedule-456"]) == 1

        schedule_runs.clear()

    @pytest.mark.asyncio
    async def test_get_schedule_runs_from_db(self, mock_env_vars):
        """Test _get_schedule_runs_from_db."""
        from src.api.scheduling import _get_schedule_runs_from_db, schedule_runs

        schedule_runs["schedule-123"] = [
            {"id": "run-1", "status": "success"},
            {"id": "run-2", "status": "failure"},
        ]

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            # All runs
            result = await _get_schedule_runs_from_db("schedule-123")
            assert len(result) == 2

            # Filtered by status
            result = await _get_schedule_runs_from_db("schedule-123", status="success")
            assert len(result) == 1

        schedule_runs.clear()


class TestScheduleEndpoints:
    """Tests for schedule API endpoints."""

    @pytest.mark.asyncio
    async def test_create_schedule_success(self, mock_env_vars):
        """Test create_schedule endpoint."""
        from src.api.scheduling import ScheduleCreateRequest, create_schedule, schedules

        schedules.clear()

        request = ScheduleCreateRequest(
            project_id="project-123",
            name="New Schedule",
            cron_expression="0 9 * * *",
            app_url="https://example.com",
        )

        mock_request = MagicMock()
        mock_request.headers = {"x-user-id": "user-123"}

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await create_schedule(request, mock_request)

            assert response.name == "New Schedule"
            assert response.project_id == "project-123"

        schedules.clear()

    @pytest.mark.asyncio
    async def test_list_schedules(self, mock_env_vars):
        """Test list_schedules endpoint."""
        from src.api.scheduling import list_schedules, schedules

        schedules.clear()
        schedules["schedule-1"] = {
            "id": "schedule-1",
            "project_id": "project-123",
            "name": "Schedule 1",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await list_schedules(mock_request, project_id="project-123")

            assert response.total >= 1
            assert len(response.schedules) >= 1

        schedules.clear()

    @pytest.mark.asyncio
    async def test_get_schedule(self, mock_env_vars):
        """Test get_schedule endpoint."""
        from src.api.scheduling import get_schedule, schedules

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Test Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await get_schedule(mock_request, "schedule-123")

            assert response.id == "schedule-123"
            assert response.name == "Test Schedule"

        schedules.clear()

    @pytest.mark.asyncio
    async def test_get_schedule_not_found(self, mock_env_vars):
        """Test get_schedule with non-existent schedule."""
        from src.api.scheduling import get_schedule, schedules

        schedules.clear()

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await get_schedule(mock_request, "nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_schedule(self, mock_env_vars):
        """Test update_schedule endpoint."""
        from src.api.scheduling import ScheduleUpdateRequest, schedules, update_schedule

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Original Name",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        }

        request = ScheduleUpdateRequest(name="Updated Name")

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await update_schedule("schedule-123", request)

            assert response.name == "Updated Name"

        schedules.clear()

    @pytest.mark.asyncio
    async def test_delete_schedule(self, mock_env_vars):
        """Test delete_schedule endpoint."""
        from src.api.scheduling import delete_schedule, schedules

        schedules["to-delete"] = {
            "id": "to-delete",
            "project_id": "project-123",
            "name": "To Delete",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await delete_schedule(mock_request, "to-delete")

            assert response["success"] is True
            assert "to-delete" not in schedules

        schedules.clear()

    @pytest.mark.asyncio
    async def test_trigger_schedule_manual(self, mock_env_vars):
        """Test trigger_schedule endpoint."""
        from src.api.scheduling import schedules, trigger_schedule

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Test Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "app_url_override": "https://example.com",
            "enabled": True,
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()
        mock_background = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await trigger_schedule(mock_request, "schedule-123", mock_background)

            assert response.success is True
            assert response.schedule_id == "schedule-123"
            mock_background.add_task.assert_called_once()

        schedules.clear()

    @pytest.mark.asyncio
    async def test_trigger_disabled_schedule(self, mock_env_vars):
        """Test triggering a disabled schedule."""
        from src.api.scheduling import schedules, trigger_schedule

        schedules["disabled-schedule"] = {
            "id": "disabled-schedule",
            "project_id": "project-123",
            "name": "Disabled Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "enabled": False,
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()
        mock_background = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await trigger_schedule(mock_request, "disabled-schedule", mock_background)

            assert exc_info.value.status_code == 400
            assert "disabled" in str(exc_info.value.detail).lower()

        schedules.clear()


class TestScheduleRunEndpoints:
    """Tests for schedule run endpoints."""

    @pytest.mark.asyncio
    async def test_get_schedule_runs(self, mock_env_vars):
        """Test get_schedule_runs endpoint."""
        from src.api.scheduling import get_schedule_runs, schedule_runs, schedules

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Test Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "created_at": "2024-01-01T00:00:00Z",
        }
        schedule_runs["schedule-123"] = [
            {"id": "run-1", "schedule_id": "schedule-123", "status": "success", "started_at": "2024-01-01T09:00:00Z"},
            {"id": "run-2", "schedule_id": "schedule-123", "status": "failure", "started_at": "2024-01-02T09:00:00Z"},
        ]

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await get_schedule_runs(mock_request, "schedule-123")

            assert response["total"] == 2

        schedules.clear()
        schedule_runs.clear()

    @pytest.mark.asyncio
    async def test_get_schedule_runs_filtered(self, mock_env_vars):
        """Test get_schedule_runs with status filter."""
        from src.api.scheduling import get_schedule_runs, schedule_runs, schedules

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "project_id": "project-456",
            "name": "Test Schedule",
            "cron_expression": "0 9 * * *",
            "app_url": "https://example.com",
            "created_at": "2024-01-01T00:00:00Z",
        }
        schedule_runs["schedule-123"] = [
            {"id": "run-1", "status": "success"},
            {"id": "run-2", "status": "failure"},
            {"id": "run-3", "status": "success"},
        ]

        mock_request = MagicMock()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            response = await get_schedule_runs(mock_request, "schedule-123", status="success")

            assert len(response["runs"]) == 2

        schedules.clear()
        schedule_runs.clear()


class TestBackgroundTask:
    """Tests for background task functions."""

    @pytest.mark.asyncio
    async def test_run_scheduled_tests_not_found(self, mock_env_vars):
        """Test run_scheduled_tests with non-existent schedule."""
        from src.api.scheduling import run_scheduled_tests, schedules

        schedules.clear()

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            # Should not raise, just return
            await run_scheduled_tests("nonexistent", "run-123")

    @pytest.mark.asyncio
    async def test_run_scheduled_tests_success(self, mock_env_vars):
        """Test run_scheduled_tests executes successfully."""
        from src.api.scheduling import run_scheduled_tests, schedule_runs, schedules

        schedules["schedule-123"] = {
            "id": "schedule-123",
            "name": "Test Schedule",
            "app_url_override": "https://example.com",
            "notify_on_failure": False,
        }
        schedule_runs["schedule-123"] = [
            {"id": "run-123", "status": "pending"}
        ]

        with patch("src.api.scheduling.get_supabase", AsyncMock(return_value=None)):
            with patch("src.api.scheduling._update_schedule_run_in_db", AsyncMock(return_value=True)):
                with patch("src.api.scheduling._update_schedule_in_db", AsyncMock(return_value=True)):
                    with patch("asyncio.sleep", AsyncMock()):
                        await run_scheduled_tests("schedule-123", "run-123")

        schedules.clear()
        schedule_runs.clear()
