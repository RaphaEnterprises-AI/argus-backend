"""Tests for Notifications API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestNotificationModels:
    """Tests for notification request/response models."""

    def test_slack_configure_request(self, mock_env_vars):
        """Test SlackConfigureRequest model."""
        from src.api.notifications import SlackConfigureRequest

        request = SlackConfigureRequest(
            webhook_url="https://hooks.slack.com/services/xxx/yyy/zzz",
            bot_token="xoxb-test-token",
            default_channel="#testing",
        )

        assert request.webhook_url.startswith("https://")
        assert request.default_channel == "#testing"

    def test_slack_configure_request_invalid_url(self, mock_env_vars):
        """Test SlackConfigureRequest with invalid webhook URL."""
        from pydantic import ValidationError

        from src.api.notifications import SlackConfigureRequest

        with pytest.raises(ValidationError) as exc_info:
            SlackConfigureRequest(
                webhook_url="invalid-url",
                default_channel="#testing",
            )

        assert "webhook_url" in str(exc_info.value).lower()

    def test_slack_test_request(self, mock_env_vars):
        """Test SlackTestRequest model."""
        from src.api.notifications import SlackTestRequest

        request = SlackTestRequest(
            channel="#test-channel",
            message_type="simple",
            custom_message="Hello, World!",
        )

        assert request.message_type == "simple"
        assert request.custom_message == "Hello, World!"

    def test_test_result_notification_request(self, mock_env_vars):
        """Test TestResultNotificationRequest model."""
        from src.api.notifications import TestResultNotificationRequest

        request = TestResultNotificationRequest(
            channel="#qa",
            title="Test Results",
            total=100,
            passed=95,
            failed=3,
            skipped=2,
            duration_seconds=120.5,
            cost_usd=0.50,
            failures=[{"test": "test-1", "error": "Assertion failed"}],
            report_url="https://reports.example.com/123",
            pr_number=456,
            branch="feature/test",
        )

        assert request.total == 100
        assert request.passed == 95
        assert len(request.failures) == 1

    def test_failure_alert_request(self, mock_env_vars):
        """Test FailureAlertRequest model."""
        from src.api.notifications import FailureAlertRequest

        request = FailureAlertRequest(
            channel="#alerts",
            test_id="test-123",
            test_name="Login Test",
            error_message="Element not found",
            stack_trace="at LoginPage.submit (login.ts:45)",
            screenshot_url="https://screenshots.example.com/123.png",
            root_cause="Selector changed",
            component="LoginForm",
            url="https://app.example.com/login",
            duration_ms=5000,
            retry_count=2,
        )

        assert request.test_id == "test-123"
        assert request.retry_count == 2

    def test_schedule_reminder_request(self, mock_env_vars):
        """Test ScheduleReminderRequest model."""
        from src.api.notifications import ScheduleReminderRequest

        request = ScheduleReminderRequest(
            channel="#schedule",
            schedule_id="schedule-123",
            schedule_name="Daily Regression",
            next_run_at=datetime.now(UTC),
            test_suite="Full Suite",
            estimated_duration_minutes=45,
            environment="staging",
        )

        assert request.schedule_name == "Daily Regression"
        assert request.estimated_duration_minutes == 45

    def test_quality_report_request(self, mock_env_vars):
        """Test QualityReportRequest model."""
        from src.api.notifications import QualityReportRequest

        request = QualityReportRequest(
            channel="#quality",
            project_id="project-123",
            project_name="My App",
            overall_score=85.5,
            grade="B",
            test_coverage=78.0,
            error_count=5,
            resolved_count=12,
            risk_level="medium",
            trends={"score_change": 2.5},
            recommendations=["Add more unit tests"],
            report_url="https://reports.example.com/quality/123",
        )

        assert request.overall_score == 85.5
        assert request.grade == "B"

    def test_notification_response(self, mock_env_vars):
        """Test NotificationResponse model."""
        from src.api.notifications import NotificationResponse

        response = NotificationResponse(
            success=True,
            message="Notification sent",
            details={"message_ts": "123.456"},
        )

        assert response.success is True
        assert response.details is not None

    def test_slack_status_response(self, mock_env_vars):
        """Test SlackStatusResponse model."""
        from src.api.notifications import SlackStatusResponse

        response = SlackStatusResponse(
            configured=True,
            webhook_configured=True,
            bot_configured=False,
            default_channel="#testing",
            webhook_status="ok",
            api_status="ok",
            bot_info=None,
        )

        assert response.configured is True


class TestChannelModels:
    """Tests for notification channel models."""

    def test_channel_create_request(self, mock_env_vars):
        """Test ChannelCreateRequest model."""
        from src.api.notifications import ChannelCreateRequest

        request = ChannelCreateRequest(
            organization_id="org-123",
            project_id="project-456",
            name="Slack Alerts",
            channel_type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/xxx"},
            enabled=True,
            rate_limit_per_hour=100,
        )

        assert request.channel_type == "slack"
        assert request.rate_limit_per_hour == 100

    def test_channel_create_request_invalid_webhook(self, mock_env_vars):
        """Test ChannelCreateRequest with invalid webhook in config."""
        from pydantic import ValidationError

        from src.api.notifications import ChannelCreateRequest

        with pytest.raises(ValidationError) as exc_info:
            ChannelCreateRequest(
                organization_id="org-123",
                name="Invalid Channel",
                channel_type="webhook",
                config={"webhook_url": "invalid-url"},
            )

        assert "webhook_url" in str(exc_info.value).lower()

    def test_channel_update_request(self, mock_env_vars):
        """Test ChannelUpdateRequest model."""
        from src.api.notifications import ChannelUpdateRequest

        request = ChannelUpdateRequest(
            name="Updated Name",
            enabled=False,
            rate_limit_per_hour=50,
        )

        assert request.name == "Updated Name"
        assert request.enabled is False

    def test_channel_response(self, mock_env_vars):
        """Test ChannelResponse model."""
        from src.api.notifications import ChannelResponse

        response = ChannelResponse(
            id="channel-123",
            organization_id="org-456",
            project_id="project-789",
            name="Slack Channel",
            channel_type="slack",
            config={"webhook_url": "https://hooks.slack.com/xxx"},
            enabled=True,
            verified=True,
            rate_limit_per_hour=100,
            last_sent_at="2024-01-01T12:00:00Z",
            sent_today=25,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
        )

        assert response.verified is True
        assert response.sent_today == 25


class TestRuleModels:
    """Tests for notification rule models."""

    def test_rule_create_request(self, mock_env_vars):
        """Test RuleCreateRequest model."""
        from src.api.notifications import RuleCreateRequest

        request = RuleCreateRequest(
            channel_id="channel-123",
            name="Failure Alert Rule",
            event_type="test_failure",
            conditions={"severity": "high"},
            message_template="Test {{test_name}} failed!",
            priority="high",
            cooldown_minutes=30,
            enabled=True,
        )

        assert request.event_type == "test_failure"
        assert request.priority == "high"

    def test_rule_response(self, mock_env_vars):
        """Test RuleResponse model."""
        from src.api.notifications import RuleResponse

        response = RuleResponse(
            id="rule-123",
            channel_id="channel-456",
            name="Test Rule",
            event_type="test_complete",
            conditions={},
            message_template=None,
            priority="normal",
            cooldown_minutes=0,
            enabled=True,
            last_triggered_at="2024-01-01T12:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
        )

        assert response.enabled is True


class TestUserNotificationModels:
    """Tests for user notification models."""

    def test_user_notification_model(self, mock_env_vars):
        """Test UserNotification model."""
        from src.api.notifications import UserNotification

        notification = UserNotification(
            id="notif-123",
            user_id="user-456",
            organization_id="org-789",
            type="test_result",
            title="Test Complete",
            message="Your tests have completed",
            read=False,
            priority="normal",
            action_url="https://app.example.com/results/123",
            metadata={"test_run_id": "run-123"},
            created_at="2024-01-01T12:00:00Z",
        )

        assert notification.read is False
        assert notification.priority == "normal"

    def test_user_notification_list_response(self, mock_env_vars):
        """Test UserNotificationListResponse model."""
        from src.api.notifications import UserNotification, UserNotificationListResponse

        response = UserNotificationListResponse(
            notifications=[
                UserNotification(
                    id="notif-1",
                    user_id="user-123",
                    type="test_result",
                    title="Test 1",
                    message="Message 1",
                    created_at="2024-01-01T12:00:00Z",
                )
            ],
            total=10,
            unread_count=5,
            has_more=True,
        )

        assert response.total == 10
        assert response.unread_count == 5

    def test_notification_preferences(self, mock_env_vars):
        """Test NotificationPreferences model."""
        from src.api.notifications import NotificationPreferences

        prefs = NotificationPreferences(
            user_id="user-123",
            email_enabled=True,
            email_frequency="daily",
            email_types=["failure", "quality"],
            in_app_enabled=True,
            slack_enabled=False,
            quiet_hours_enabled=True,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
        )

        assert prefs.email_frequency == "daily"
        assert prefs.quiet_hours_enabled is True


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_url_valid(self, mock_env_vars):
        """Test validate_url with valid URLs."""
        from src.api.notifications import validate_url

        assert validate_url("https://example.com", "test") == "https://example.com"
        assert validate_url("http://localhost:3000", "test") == "http://localhost:3000"
        assert validate_url(None, "test") is None

    def test_validate_url_invalid(self, mock_env_vars):
        """Test validate_url with invalid URLs."""
        from src.api.notifications import validate_url

        with pytest.raises(ValueError) as exc_info:
            validate_url("ftp://example.com", "test_field")

        assert "test_field" in str(exc_info.value)


class TestSlackNotifierHelpers:
    """Tests for Slack notifier helper functions."""

    def test_get_slack_notifier_with_config(self, mock_env_vars):
        """Test get_slack_notifier with configured settings."""
        from src.api.notifications import get_slack_notifier, set_slack_config
        from src.integrations.slack import SlackConfig

        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            default_channel="#test",
        )

        set_slack_config(config)

        notifier = get_slack_notifier()
        assert notifier is not None

        # Reset
        from src.api import notifications
        notifications._slack_config = None

    def test_get_slack_notifier_from_env(self, mock_env_vars):
        """Test get_slack_notifier falls back to env vars."""
        from src.api.notifications import get_slack_notifier

        with patch("src.api.notifications.create_slack_notifier") as mock_create:
            mock_notifier = MagicMock()
            mock_create.return_value = mock_notifier

            notifier = get_slack_notifier()
            assert notifier is mock_notifier


class TestChannelDBHelpers:
    """Tests for channel database helper functions."""

    @pytest.mark.asyncio
    async def test_get_channel_from_db_in_memory(self, mock_env_vars):
        """Test _get_channel_from_db with in-memory fallback."""
        from src.api.notifications import _channels, _get_channel_from_db

        _channels["channel-123"] = {"id": "channel-123", "name": "Test Channel"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _get_channel_from_db("channel-123")
            assert result is not None
            assert result["id"] == "channel-123"

        _channels.clear()

    @pytest.mark.asyncio
    async def test_list_channels_from_db(self, mock_env_vars):
        """Test _list_channels_from_db."""
        from src.api.notifications import _channels, _list_channels_from_db

        _channels["channel-1"] = {"id": "channel-1", "organization_id": "org-1", "channel_type": "slack", "enabled": True}
        _channels["channel-2"] = {"id": "channel-2", "organization_id": "org-1", "channel_type": "email", "enabled": False}
        _channels["channel-3"] = {"id": "channel-3", "organization_id": "org-2", "channel_type": "slack", "enabled": True}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            # Filter by org
            result = await _list_channels_from_db(organization_id="org-1")
            assert len(result) == 2

            # Filter by type
            result = await _list_channels_from_db(channel_type="slack")
            assert len(result) == 2

            # Filter by enabled
            result = await _list_channels_from_db(enabled=True)
            assert len(result) == 2

        _channels.clear()

    @pytest.mark.asyncio
    async def test_save_channel_to_db(self, mock_env_vars):
        """Test _save_channel_to_db."""
        from src.api.notifications import _channels, _save_channel_to_db

        _channels.clear()

        channel = {"id": "new-channel", "name": "New Channel"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _save_channel_to_db(channel)
            assert result is True
            assert "new-channel" in _channels

        _channels.clear()

    @pytest.mark.asyncio
    async def test_update_channel_in_db(self, mock_env_vars):
        """Test _update_channel_in_db."""
        from src.api.notifications import _channels, _update_channel_in_db

        _channels["channel-123"] = {"id": "channel-123", "name": "Original"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _update_channel_in_db("channel-123", {"name": "Updated"})
            assert result is True
            assert _channels["channel-123"]["name"] == "Updated"

        _channels.clear()

    @pytest.mark.asyncio
    async def test_delete_channel_from_db(self, mock_env_vars):
        """Test _delete_channel_from_db."""
        from src.api.notifications import _channels, _delete_channel_from_db

        _channels["to-delete"] = {"id": "to-delete"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _delete_channel_from_db("to-delete")
            assert result is True
            assert "to-delete" not in _channels

        _channels.clear()


class TestRuleDBHelpers:
    """Tests for rule database helper functions."""

    @pytest.mark.asyncio
    async def test_get_rule_from_db(self, mock_env_vars):
        """Test _get_rule_from_db."""
        from src.api.notifications import _get_rule_from_db, _rules

        _rules["rule-123"] = {"id": "rule-123", "name": "Test Rule"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _get_rule_from_db("rule-123")
            assert result is not None

        _rules.clear()

    @pytest.mark.asyncio
    async def test_list_rules_from_db(self, mock_env_vars):
        """Test _list_rules_from_db."""
        from src.api.notifications import _list_rules_from_db, _rules

        _rules["rule-1"] = {"id": "rule-1", "channel_id": "channel-1", "event_type": "test_failure"}
        _rules["rule-2"] = {"id": "rule-2", "channel_id": "channel-1", "event_type": "test_success"}
        _rules["rule-3"] = {"id": "rule-3", "channel_id": "channel-2", "event_type": "test_failure"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            # Filter by channel
            result = await _list_rules_from_db(channel_id="channel-1")
            assert len(result) == 2

            # Filter by event type
            result = await _list_rules_from_db(event_type="test_failure")
            assert len(result) == 2

        _rules.clear()

    @pytest.mark.asyncio
    async def test_save_rule_to_db(self, mock_env_vars):
        """Test _save_rule_to_db."""
        from src.api.notifications import _rules, _save_rule_to_db

        _rules.clear()

        rule = {"id": "new-rule", "name": "New Rule"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _save_rule_to_db(rule)
            assert result is True
            assert "new-rule" in _rules

        _rules.clear()

    @pytest.mark.asyncio
    async def test_update_rule_in_db(self, mock_env_vars):
        """Test _update_rule_in_db."""
        from src.api.notifications import _rules, _update_rule_in_db

        _rules["rule-123"] = {"id": "rule-123", "enabled": True}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _update_rule_in_db("rule-123", {"enabled": False})
            assert result is True
            assert _rules["rule-123"]["enabled"] is False

        _rules.clear()

    @pytest.mark.asyncio
    async def test_delete_rule_from_db(self, mock_env_vars):
        """Test _delete_rule_from_db."""
        from src.api.notifications import _delete_rule_from_db, _rules

        _rules["to-delete"] = {"id": "to-delete"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _delete_rule_from_db("to-delete")
            assert result is True
            assert "to-delete" not in _rules

        _rules.clear()


class TestNotificationLogHelpers:
    """Tests for notification log helper functions."""

    @pytest.mark.asyncio
    async def test_save_notification_log(self, mock_env_vars):
        """Test _save_notification_log."""
        from src.api.notifications import _logs, _save_notification_log

        _logs.clear()

        log = {"id": "log-123", "channel_id": "channel-456", "status": "sent"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _save_notification_log(log)
            assert result is True
            assert len(_logs) == 1

        _logs.clear()

    @pytest.mark.asyncio
    async def test_list_notification_logs(self, mock_env_vars):
        """Test _list_notification_logs."""
        from src.api.notifications import _list_notification_logs, _logs

        _logs.clear()
        _logs.append({"id": "log-1", "channel_id": "channel-1", "status": "sent"})
        _logs.append({"id": "log-2", "channel_id": "channel-1", "status": "failed"})
        _logs.append({"id": "log-3", "channel_id": "channel-2", "status": "sent"})

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            # Filter by channel
            result = await _list_notification_logs(channel_id="channel-1")
            assert len(result) == 2

            # Filter by status
            result = await _list_notification_logs(status="sent")
            assert len(result) == 2

        _logs.clear()


class TestUserNotificationHelpers:
    """Tests for user notification helper functions."""

    @pytest.mark.asyncio
    async def test_get_user_notification_from_db(self, mock_env_vars):
        """Test _get_user_notification_from_db."""
        from src.api.notifications import _get_user_notification_from_db, _user_notifications

        _user_notifications["notif-123"] = {"id": "notif-123", "title": "Test"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _get_user_notification_from_db("notif-123")
            assert result is not None

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_list_user_notifications_from_db(self, mock_env_vars):
        """Test _list_user_notifications_from_db."""
        from src.api.notifications import _list_user_notifications_from_db, _user_notifications

        _user_notifications.clear()
        _user_notifications["notif-1"] = {"id": "notif-1", "user_id": "user-1", "read": False, "type": "test_result", "created_at": "2024-01-01T12:00:00Z"}
        _user_notifications["notif-2"] = {"id": "notif-2", "user_id": "user-1", "read": True, "type": "failure", "created_at": "2024-01-01T13:00:00Z"}
        _user_notifications["notif-3"] = {"id": "notif-3", "user_id": "user-2", "read": False, "type": "test_result", "created_at": "2024-01-01T14:00:00Z"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            # All for user
            result, total = await _list_user_notifications_from_db("user-1")
            assert len(result) == 2
            assert total == 2

            # Filter by read status
            result, total = await _list_user_notifications_from_db("user-1", read=False)
            assert len(result) == 1

            # Filter by type
            result, total = await _list_user_notifications_from_db("user-1", notification_type="failure")
            assert len(result) == 1

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_get_unread_count_from_db(self, mock_env_vars):
        """Test _get_unread_count_from_db."""
        from src.api.notifications import _get_unread_count_from_db, _user_notifications

        _user_notifications.clear()
        _user_notifications["notif-1"] = {"id": "notif-1", "user_id": "user-1", "read": False, "priority": "high"}
        _user_notifications["notif-2"] = {"id": "notif-2", "user_id": "user-1", "read": False, "priority": "normal"}
        _user_notifications["notif-3"] = {"id": "notif-3", "user_id": "user-1", "read": True, "priority": "normal"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _get_unread_count_from_db("user-1")

            assert result["unread_count"] == 2
            assert result["by_priority"]["high"] == 1
            assert result["by_priority"]["normal"] == 1

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_save_user_notification_to_db(self, mock_env_vars):
        """Test _save_user_notification_to_db."""
        from src.api.notifications import _save_user_notification_to_db, _user_notifications

        _user_notifications.clear()

        notification = {"id": "new-notif", "user_id": "user-1", "title": "New"}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _save_user_notification_to_db(notification)
            assert result is True
            assert "new-notif" in _user_notifications

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_update_user_notification_in_db(self, mock_env_vars):
        """Test _update_user_notification_in_db."""
        from src.api.notifications import _update_user_notification_in_db, _user_notifications

        _user_notifications["notif-123"] = {"id": "notif-123", "read": False}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            result = await _update_user_notification_in_db("notif-123", {"read": True})
            assert result is True
            assert _user_notifications["notif-123"]["read"] is True

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_mark_all_notifications_read(self, mock_env_vars):
        """Test _mark_all_notifications_read."""
        from src.api.notifications import _mark_all_notifications_read, _user_notifications

        _user_notifications.clear()
        _user_notifications["notif-1"] = {"id": "notif-1", "user_id": "user-1", "read": False}
        _user_notifications["notif-2"] = {"id": "notif-2", "user_id": "user-1", "read": False}
        _user_notifications["notif-3"] = {"id": "notif-3", "user_id": "user-2", "read": False}

        with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
            count = await _mark_all_notifications_read("user-1")

            assert count == 2
            assert _user_notifications["notif-1"]["read"] is True
            assert _user_notifications["notif-2"]["read"] is True
            assert _user_notifications["notif-3"]["read"] is False  # Different user

        _user_notifications.clear()


class TestSlackConfigEndpoints:
    """Tests for Slack configuration endpoints."""

    @pytest.mark.asyncio
    async def test_configure_slack(self, mock_env_vars):
        """Test configure_slack endpoint."""
        from src.api.notifications import SlackConfigureRequest, configure_slack

        request = SlackConfigureRequest(
            webhook_url="https://hooks.slack.com/services/xxx",
            default_channel="#testing",
        )

        response = await configure_slack(request)

        assert response.success is True
        assert "configured" in response.message.lower()

    @pytest.mark.asyncio
    async def test_get_slack_status(self, mock_env_vars):
        """Test get_slack_status endpoint."""
        from src.api.notifications import get_slack_status

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.config = MagicMock()
            mock_notifier.config.webhook_url = "https://hooks.slack.com/xxx"
            mock_notifier.config.bot_token = None
            mock_notifier.config.default_channel = "#testing"
            mock_get.return_value = mock_notifier

            response = await get_slack_status()

            assert response.webhook_configured is True
            assert response.default_channel == "#testing"


class TestSlackNotificationEndpoints:
    """Tests for Slack notification sending endpoints."""

    @pytest.mark.asyncio
    async def test_send_test_notification(self, mock_env_vars):
        """Test send_test_notification endpoint."""
        from src.api.notifications import SlackTestRequest, send_test_notification

        request = SlackTestRequest(
            channel="#test",
            message_type="simple",
            custom_message="Test message",
        )

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_message = AsyncMock(return_value=True)
            mock_get.return_value = mock_notifier

            response = await send_test_notification(request)

            assert response.success is True

    @pytest.mark.asyncio
    async def test_send_test_result_notification(self, mock_env_vars):
        """Test send_test_result_notification endpoint."""
        from src.api.notifications import (
            TestResultNotificationRequest,
            send_test_result_notification,
        )

        request = TestResultNotificationRequest(
            channel="#qa",
            title="Test Results",
            total=10,
            passed=9,
            failed=1,
            skipped=0,
            duration_seconds=60.0,
        )

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_test_results = AsyncMock(return_value=True)
            mock_get.return_value = mock_notifier

            response = await send_test_result_notification(request)

            assert response.success is True

    @pytest.mark.asyncio
    async def test_send_failure_alert(self, mock_env_vars):
        """Test send_failure_alert endpoint."""
        from src.api.notifications import FailureAlertRequest, send_failure_alert

        request = FailureAlertRequest(
            test_id="test-123",
            test_name="Login Test",
            error_message="Element not found",
        )

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_failure_alert = AsyncMock(return_value=True)
            mock_get.return_value = mock_notifier

            response = await send_failure_alert(request)

            assert response.success is True

    @pytest.mark.asyncio
    async def test_send_schedule_reminder(self, mock_env_vars):
        """Test send_schedule_reminder endpoint."""
        from src.api.notifications import ScheduleReminderRequest, send_schedule_reminder

        request = ScheduleReminderRequest(
            schedule_id="schedule-123",
            schedule_name="Daily Tests",
            next_run_at=datetime.now(UTC),
            test_suite="Regression Suite",
        )

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_schedule_reminder = AsyncMock(return_value=True)
            mock_get.return_value = mock_notifier

            response = await send_schedule_reminder(request)

            assert response.success is True

    @pytest.mark.asyncio
    async def test_send_quality_report(self, mock_env_vars):
        """Test send_quality_report endpoint."""
        from src.api.notifications import QualityReportRequest, send_quality_report

        request = QualityReportRequest(
            project_id="project-123",
            project_name="My App",
            overall_score=85.0,
            grade="B",
            test_coverage=80.0,
        )

        with patch("src.api.notifications.get_slack_notifier") as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_quality_report = AsyncMock(return_value=True)
            mock_get.return_value = mock_notifier

            response = await send_quality_report(request)

            assert response.success is True


class TestChannelEndpoints:
    """Tests for notification channel endpoints."""

    @pytest.mark.asyncio
    async def test_create_channel(self, mock_env_vars):
        """Test create_channel endpoint."""
        from src.api.notifications import ChannelCreateRequest, _channels, create_channel

        _channels.clear()

        request = ChannelCreateRequest(
            organization_id="org-123",
            name="New Channel",
            channel_type="slack",
            config={"webhook_url": "https://hooks.slack.com/xxx"},
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123", "organization_id": "org-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await create_channel(mock_request, request)

                assert response.name == "New Channel"
                assert response.channel_type == "slack"

        _channels.clear()

    @pytest.mark.asyncio
    async def test_list_channels(self, mock_env_vars):
        """Test list_channels endpoint."""
        from src.api.notifications import _channels, list_channels

        _channels.clear()
        _channels["channel-1"] = {
            "id": "channel-1",
            "organization_id": "org-123",
            "name": "Channel 1",
            "channel_type": "slack",
            "config": {},
            "enabled": True,
            "verified": True,
            "rate_limit_per_hour": 100,
            "sent_today": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123", "organization_id": "org-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await list_channels(mock_request, organization_id="org-123")

                assert response["total"] >= 1

        _channels.clear()

    @pytest.mark.asyncio
    async def test_get_channel(self, mock_env_vars):
        """Test get_channel endpoint."""
        from src.api.notifications import _channels, get_channel

        _channels["channel-123"] = {
            "id": "channel-123",
            "organization_id": "org-456",
            "name": "Test Channel",
            "channel_type": "slack",
            "config": {},
            "enabled": True,
            "verified": True,
            "rate_limit_per_hour": 100,
            "sent_today": 5,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123", "organization_id": "org-456"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await get_channel(mock_request, "channel-123")

                assert response.id == "channel-123"

        _channels.clear()

    @pytest.mark.asyncio
    async def test_get_channel_not_found(self, mock_env_vars):
        """Test get_channel with non-existent channel."""
        from src.api.notifications import _channels, get_channel

        _channels.clear()

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                with pytest.raises(HTTPException) as exc_info:
                    await get_channel(mock_request, "nonexistent")

                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_channel(self, mock_env_vars):
        """Test delete_channel endpoint."""
        from src.api.notifications import _channels, delete_channel

        _channels["to-delete"] = {
            "id": "to-delete",
            "organization_id": "org-123",
            "name": "To Delete",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123", "organization_id": "org-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await delete_channel(mock_request, "to-delete")

                assert response["success"] is True
                assert "to-delete" not in _channels

        _channels.clear()


class TestUserNotificationEndpoints:
    """Tests for user notification endpoints."""

    @pytest.mark.asyncio
    async def test_list_user_notifications(self, mock_env_vars):
        """Test list_user_notifications endpoint."""
        from src.api.notifications import _user_notifications, list_user_notifications

        _user_notifications.clear()
        _user_notifications["notif-1"] = {
            "id": "notif-1",
            "user_id": "user-123",
            "type": "test_result",
            "title": "Test Complete",
            "message": "Your tests passed",
            "read": False,
            "priority": "normal",
            "created_at": "2024-01-01T12:00:00Z",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await list_user_notifications(mock_request)

                assert response.total >= 1
                assert response.unread_count >= 0

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_get_unread_count(self, mock_env_vars):
        """Test get_unread_count endpoint."""
        from src.api.notifications import _user_notifications, get_unread_count

        _user_notifications.clear()
        _user_notifications["notif-1"] = {"id": "notif-1", "user_id": "user-123", "read": False, "priority": "high"}
        _user_notifications["notif-2"] = {"id": "notif-2", "user_id": "user-123", "read": False, "priority": "normal"}

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await get_unread_count(mock_request)

                assert response.unread_count == 2

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_mark_notification_read(self, mock_env_vars):
        """Test mark_notification_read endpoint."""
        from src.api.notifications import _user_notifications, mark_notification_read

        _user_notifications["notif-123"] = {
            "id": "notif-123",
            "user_id": "user-456",
            "type": "test_result",
            "title": "Test",
            "message": "Message",
            "read": False,
            "priority": "normal",
            "created_at": "2024-01-01T12:00:00Z",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "user-456"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await mark_notification_read(mock_request, "notif-123")

                assert response.read is True

        _user_notifications.clear()

    @pytest.mark.asyncio
    async def test_mark_all_read(self, mock_env_vars):
        """Test mark_all_read endpoint."""
        from src.api.notifications import _user_notifications, mark_all_read

        _user_notifications.clear()
        _user_notifications["notif-1"] = {"id": "notif-1", "user_id": "user-123", "read": False}
        _user_notifications["notif-2"] = {"id": "notif-2", "user_id": "user-123", "read": False}

        mock_request = MagicMock()
        mock_user = {"user_id": "user-123"}

        with patch("src.api.notifications.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.notifications.get_supabase", AsyncMock(return_value=None)):
                response = await mark_all_read(mock_request)

                assert response["success"] is True
                assert response["count"] == 2

        _user_notifications.clear()
