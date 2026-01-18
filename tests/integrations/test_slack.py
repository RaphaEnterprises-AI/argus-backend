"""Tests for Slack integration module (slack.py)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNotificationStatus:
    """Tests for NotificationStatus enum."""

    def test_notification_status_values(self, mock_env_vars):
        """Test NotificationStatus enum values."""
        from src.integrations.slack import NotificationStatus

        assert NotificationStatus.SUCCESS.value == "success"
        assert NotificationStatus.FAILURE.value == "failure"
        assert NotificationStatus.WARNING.value == "warning"
        assert NotificationStatus.INFO.value == "info"


class TestSlackConfig:
    """Tests for SlackConfig dataclass."""

    def test_config_creation(self, mock_env_vars):
        """Test creating SlackConfig."""
        from src.integrations.slack import SlackConfig

        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            bot_token="xoxb-test-token",
            default_channel="#testing",
        )

        assert config.webhook_url == "https://hooks.slack.com/test"
        assert config.bot_token == "xoxb-test-token"
        assert config.default_channel == "#testing"
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3

    def test_config_from_env(self, mock_env_vars, monkeypatch):
        """Test creating SlackConfig from environment."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env-token")
        monkeypatch.setenv("SLACK_DEFAULT_CHANNEL", "#env-channel")

        from src.integrations.slack import SlackConfig

        config = SlackConfig.from_env()

        assert config.webhook_url == "https://hooks.slack.com/env"
        assert config.bot_token == "xoxb-env-token"
        assert config.default_channel == "#env-channel"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_test_result_creation(self, mock_env_vars):
        """Test creating TestResult."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=100,
            passed=95,
            failed=3,
            skipped=2,
            duration_seconds=120.5,
            cost_usd=0.5,
        )

        assert result.total == 100
        assert result.passed == 95
        assert result.failed == 3

    def test_test_result_pass_rate(self, mock_env_vars):
        """Test TestResult pass_rate property."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=100,
            passed=80,
            failed=20,
            skipped=0,
            duration_seconds=60.0,
        )

        assert result.pass_rate == 80.0

    def test_test_result_pass_rate_zero_total(self, mock_env_vars):
        """Test TestResult pass_rate with zero total."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=0,
            passed=0,
            failed=0,
            skipped=0,
            duration_seconds=0.0,
        )

        assert result.pass_rate == 0.0

    def test_test_result_status_success(self, mock_env_vars):
        """Test TestResult status when all pass."""
        from src.integrations.slack import NotificationStatus, TestResult

        result = TestResult(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.0,
        )

        assert result.status == NotificationStatus.SUCCESS

    def test_test_result_status_failure(self, mock_env_vars):
        """Test TestResult status when many fail."""
        from src.integrations.slack import NotificationStatus, TestResult

        result = TestResult(
            total=10,
            passed=3,
            failed=7,
            skipped=0,
            duration_seconds=30.0,
        )

        assert result.status == NotificationStatus.FAILURE

    def test_test_result_status_warning(self, mock_env_vars):
        """Test TestResult status when some fail."""
        from src.integrations.slack import NotificationStatus, TestResult

        result = TestResult(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=30.0,
        )

        assert result.status == NotificationStatus.WARNING


class TestFailureDetails:
    """Tests for FailureDetails dataclass."""

    def test_failure_details_creation(self, mock_env_vars):
        """Test creating FailureDetails."""
        from src.integrations.slack import FailureDetails

        failure = FailureDetails(
            test_id="test-123",
            test_name="Login Test",
            error_message="Element not found",
            stack_trace="Traceback...",
            screenshot_url="https://example.com/screenshot.png",
            root_cause="Selector changed",
            component="LoginForm",
            url="https://app.example.com/login",
            duration_ms=5000,
            retry_count=2,
        )

        assert failure.test_id == "test-123"
        assert failure.error_message == "Element not found"
        assert failure.retry_count == 2


class TestScheduleInfo:
    """Tests for ScheduleInfo dataclass."""

    def test_schedule_info_creation(self, mock_env_vars):
        """Test creating ScheduleInfo."""
        from src.integrations.slack import ScheduleInfo

        next_run = datetime.now(UTC) + timedelta(minutes=30)

        schedule = ScheduleInfo(
            schedule_id="sched-123",
            schedule_name="Nightly Tests",
            next_run_at=next_run,
            test_suite="full",
            estimated_duration_minutes=45,
            environment="staging",
            notify_channel="#qa",
        )

        assert schedule.schedule_id == "sched-123"
        assert schedule.schedule_name == "Nightly Tests"


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_quality_report_creation(self, mock_env_vars):
        """Test creating QualityReport."""
        from src.integrations.slack import QualityReport

        report = QualityReport(
            project_id="proj-123",
            project_name="My App",
            overall_score=85.5,
            grade="B",
            test_coverage=75.0,
            error_count=5,
            resolved_count=10,
            risk_level="low",
            trends={"score": 5.0, "coverage": 2.0},
            recommendations=["Add more tests"],
            report_url="https://example.com/report",
        )

        assert report.overall_score == 85.5
        assert report.grade == "B"


class TestSlackNotifierInit:
    """Tests for SlackNotifier initialization."""

    def test_notifier_with_config(self, mock_env_vars):
        """Test creating notifier with config."""
        from src.integrations.slack import SlackConfig, SlackNotifier

        config = SlackConfig(
            webhook_url="https://hooks.slack.com/test",
            default_channel="#test",
        )

        notifier = SlackNotifier(config=config)

        assert notifier.config.webhook_url == "https://hooks.slack.com/test"

    def test_notifier_with_params(self, mock_env_vars):
        """Test creating notifier with parameters."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/test",
            bot_token="xoxb-test",
            default_channel="#testing",
        )

        assert notifier.config.webhook_url == "https://hooks.slack.com/test"
        assert notifier.config.bot_token == "xoxb-test"

    def test_notifier_is_configured(self, mock_env_vars):
        """Test is_configured property."""
        from src.integrations.slack import SlackNotifier

        notifier_with_webhook = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        notifier_with_token = SlackNotifier(bot_token="xoxb-test")
        notifier_empty = SlackNotifier()

        assert notifier_with_webhook.is_configured is True
        assert notifier_with_token.is_configured is True
        assert notifier_empty.is_configured is False


class TestSlackNotifierHelpers:
    """Tests for SlackNotifier helper methods."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    def test_get_status_color(self, notifier):
        """Test _get_status_color method."""
        from src.integrations.slack import NotificationStatus

        assert notifier._get_status_color(NotificationStatus.SUCCESS) == "#36a64f"
        assert notifier._get_status_color(NotificationStatus.FAILURE) == "#dc3545"
        assert notifier._get_status_color(NotificationStatus.WARNING) == "#ffc107"
        assert notifier._get_status_color(NotificationStatus.INFO) == "#17a2b8"

    def test_get_status_emoji(self, notifier):
        """Test _get_status_emoji method."""
        from src.integrations.slack import NotificationStatus

        assert ":white_check_mark:" in notifier._get_status_emoji(NotificationStatus.SUCCESS)
        assert ":x:" in notifier._get_status_emoji(NotificationStatus.FAILURE)
        assert ":warning:" in notifier._get_status_emoji(NotificationStatus.WARNING)

    def test_format_duration_seconds(self, notifier):
        """Test _format_duration for seconds."""
        assert notifier._format_duration(30.5) == "30.5s"

    def test_format_duration_minutes(self, notifier):
        """Test _format_duration for minutes."""
        assert notifier._format_duration(120.0) == "2.0m"

    def test_format_duration_hours(self, notifier):
        """Test _format_duration for hours."""
        assert notifier._format_duration(3600.0) == "1.0h"

    def test_truncate_short_text(self, notifier):
        """Test _truncate for short text."""
        text = "Short text"
        assert notifier._truncate(text, 100) == text

    def test_truncate_long_text(self, notifier):
        """Test _truncate for long text."""
        text = "x" * 100
        result = notifier._truncate(text, 50)

        assert len(result) == 50
        assert result.endswith("...")


class TestSlackNotifierBuildBlocks:
    """Tests for SlackNotifier block building methods."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    def test_build_test_result_blocks(self, notifier):
        """Test _build_test_result_blocks method."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=60.0,
        )

        blocks = notifier._build_test_result_blocks(result)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"

    def test_build_test_result_blocks_with_failures(self, notifier):
        """Test building blocks with failure details."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=60.0,
            failures=[
                {"test_id": "test-1", "error_message": "Error 1"},
                {"test_id": "test-2", "error_message": "Error 2"},
            ],
        )

        blocks = notifier._build_test_result_blocks(result)

        # Should have divider and failure sections
        block_types = [b["type"] for b in blocks]
        assert "divider" in block_types

    def test_build_test_result_blocks_with_urls(self, notifier):
        """Test building blocks with URLs."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=60.0,
            report_url="https://example.com/report",
            pr_url="https://github.com/org/repo/pull/123",
            pr_number=123,
            branch="feature-branch",
            commit_sha="abc123def456",
        )

        blocks = notifier._build_test_result_blocks(result)

        # Should have actions section with buttons
        has_actions = any(b["type"] == "actions" for b in blocks)
        assert has_actions

    def test_build_failure_alert_blocks(self, notifier):
        """Test _build_failure_alert_blocks method."""
        from src.integrations.slack import FailureDetails

        failure = FailureDetails(
            test_id="test-123",
            test_name="Login Test",
            error_message="Element not found",
            stack_trace="Error at line 42",
            root_cause="Selector changed",
        )

        blocks = notifier._build_failure_alert_blocks(failure)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        assert "Test Failure Alert" in blocks[0]["text"]["text"]

    def test_build_schedule_reminder_blocks(self, notifier):
        """Test _build_schedule_reminder_blocks method."""
        from src.integrations.slack import ScheduleInfo

        next_run = datetime.now(UTC) + timedelta(minutes=30)

        schedule = ScheduleInfo(
            schedule_id="sched-123",
            schedule_name="Nightly Tests",
            next_run_at=next_run,
            test_suite="full",
        )

        blocks = notifier._build_schedule_reminder_blocks(schedule)

        assert len(blocks) > 0
        assert "Scheduled Test Run" in blocks[0]["text"]["text"]

    def test_build_quality_report_blocks(self, notifier):
        """Test _build_quality_report_blocks method."""
        from src.integrations.slack import QualityReport

        report = QualityReport(
            project_id="proj-123",
            project_name="My App",
            overall_score=85.5,
            grade="B",
            test_coverage=75.0,
            error_count=5,
            resolved_count=10,
            risk_level="low",
        )

        blocks = notifier._build_quality_report_blocks(report)

        assert len(blocks) > 0
        assert "Quality Report" in blocks[0]["text"]["text"]


class TestSlackNotifierSendMessage:
    """Tests for SlackNotifier.send_message method."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_message_via_webhook(self, notifier):
        """Test sending message via webhook."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.send_message(message="Test message")

            assert result is True
            mock_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_send_message_via_api(self, mock_env_vars):
        """Test sending message via Slack API."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.send_message(
                channel="#testing",
                message="Test message",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self, mock_env_vars):
        """Test send_message when not configured."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier()

        result = await notifier.send_message(message="Test")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_webhook_failure(self, notifier):
        """Test send_message handles webhook failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.send_message(message="Test")

            assert result is False


class TestSlackNotifierSendTestResult:
    """Tests for SlackNotifier.send_test_result method."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_test_result_success(self, notifier):
        """Test sending test result notification."""
        from src.integrations.slack import TestResult

        result = TestResult(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=60.0,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            success = await notifier.send_test_result(result)

            assert success is True


class TestSlackNotifierSendFailureAlert:
    """Tests for SlackNotifier.send_failure_alert method."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_failure_alert_success(self, notifier):
        """Test sending failure alert notification."""
        from src.integrations.slack import FailureDetails

        failure = FailureDetails(
            test_id="test-123",
            test_name="Login Test",
            error_message="Element not found",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            success = await notifier.send_failure_alert(failure)

            assert success is True


class TestSlackNotifierAddReaction:
    """Tests for SlackNotifier.add_reaction method."""

    @pytest.mark.asyncio
    async def test_add_reaction_success(self, mock_env_vars):
        """Test adding reaction to message."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.add_reaction(
                channel="C123",
                timestamp="123.456",
                emoji="thumbsup",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_add_reaction_no_bot_token(self, mock_env_vars):
        """Test add_reaction requires bot token."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")

        result = await notifier.add_reaction(
            channel="C123",
            timestamp="123.456",
            emoji="thumbsup",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_add_reaction_already_reacted(self, mock_env_vars):
        """Test add_reaction handles already_reacted."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "already_reacted"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.add_reaction(
                channel="C123",
                timestamp="123.456",
                emoji="thumbsup",
            )

            assert result is True


class TestSlackNotifierCheckConnection:
    """Tests for SlackNotifier.check_connection method."""

    @pytest.mark.asyncio
    async def test_check_connection_not_configured(self, mock_env_vars):
        """Test check_connection when not configured."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier()

        result = await notifier.check_connection()

        assert result["configured"] is False

    @pytest.mark.asyncio
    async def test_check_connection_with_bot(self, mock_env_vars):
        """Test check_connection with bot token."""
        from src.integrations.slack import SlackNotifier

        notifier = SlackNotifier(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "team": "Test Team",
            "user": "bot",
            "team_id": "T123",
            "user_id": "U123",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.check_connection()

            assert result["configured"] is True
            assert result["api_status"] == "connected"
            assert result["bot_info"]["team"] == "Test Team"


class TestCreateSlackNotifier:
    """Tests for create_slack_notifier factory function."""

    def test_create_with_webhook(self, mock_env_vars):
        """Test creating notifier with webhook."""
        from src.integrations.slack import create_slack_notifier

        notifier = create_slack_notifier(webhook_url="https://hooks.slack.com/test")

        assert notifier.config.webhook_url == "https://hooks.slack.com/test"

    def test_create_with_bot_token(self, mock_env_vars):
        """Test creating notifier with bot token."""
        from src.integrations.slack import create_slack_notifier

        notifier = create_slack_notifier(bot_token="xoxb-test")

        assert notifier.config.bot_token == "xoxb-test"
