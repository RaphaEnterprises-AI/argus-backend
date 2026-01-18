"""Tests for Slack integration module (slack_integration.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import httpx


class TestTestSummary:
    """Tests for TestSummary dataclass."""

    def test_test_summary_creation(self, mock_env_vars):
        """Test creating TestSummary."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=100,
            passed=95,
            failed=3,
            skipped=2,
            duration_seconds=120.5,
            cost_usd=0.5,
            failures=[{"test_id": "test-1", "error_message": "Failed"}],
            report_url="https://example.com/report",
            pr_url="https://github.com/org/repo/pull/123",
        )

        assert summary.total == 100
        assert summary.passed == 95
        assert summary.report_url == "https://example.com/report"


class TestSlackIntegrationInit:
    """Tests for SlackIntegration initialization."""

    def test_initialization_with_webhook(self, mock_env_vars):
        """Test initialization with webhook URL."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(webhook_url="https://hooks.slack.com/test")

        assert slack.webhook_url == "https://hooks.slack.com/test"
        assert slack.default_channel == "#testing"

    def test_initialization_with_bot_token(self, mock_env_vars):
        """Test initialization with bot token."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(bot_token="xoxb-test-token")

        assert slack.bot_token == "xoxb-test-token"

    def test_initialization_from_env(self, mock_env_vars, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env-token")

        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration()

        assert slack.webhook_url == "https://hooks.slack.com/env"
        assert slack.bot_token == "xoxb-env-token"

    def test_initialization_custom_channel(self, mock_env_vars):
        """Test initialization with custom default channel."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(
            webhook_url="https://hooks.slack.com/test",
            default_channel="#custom-channel",
        )

        assert slack.default_channel == "#custom-channel"

    def test_initialization_no_credentials(self, mock_env_vars, monkeypatch):
        """Test initialization without credentials."""
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration()

        assert slack.webhook_url is None
        assert slack.bot_token is None


class TestSlackIntegrationFormatResultsBlocks:
    """Tests for SlackIntegration._format_results_blocks method."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    def test_format_results_blocks_success(self, slack):
        """Test formatting blocks for successful test run."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
        )

        blocks = slack._format_results_blocks(summary)

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        # Success emoji should be in header
        assert ":white_check_mark:" in blocks[0]["text"]["text"]

    def test_format_results_blocks_failure(self, slack):
        """Test formatting blocks with failures."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[
                {"test_id": "test-1", "error_message": "Error 1"},
                {"test_id": "test-2", "error_message": "Error 2"},
            ],
        )

        blocks = slack._format_results_blocks(summary)

        # Should include failure section
        block_types = [b["type"] for b in blocks]
        assert "divider" in block_types
        # Failure emoji in header
        assert ":x:" in blocks[0]["text"]["text"]

    def test_format_results_blocks_with_urls(self, slack):
        """Test formatting blocks with report and PR URLs."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
            report_url="https://example.com/report",
            pr_url="https://github.com/org/repo/pull/123",
        )

        blocks = slack._format_results_blocks(summary)

        # Should have actions section with buttons
        has_actions = any(b["type"] == "actions" for b in blocks)
        assert has_actions

    def test_format_results_blocks_many_failures(self, slack):
        """Test formatting blocks truncates many failures."""
        from src.integrations.slack_integration import TestSummary

        failures = [
            {"test_id": f"test-{i}", "error_message": f"Error {i}"}
            for i in range(10)
        ]

        summary = TestSummary(
            total=10,
            passed=0,
            failed=10,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=failures,
        )

        blocks = slack._format_results_blocks(summary)

        # Should show max 5 failures and mention remaining
        block_texts = str(blocks)
        assert "test-4" in block_texts  # 5th failure (0-indexed)
        assert "5 more failures" in block_texts

    def test_format_results_blocks_custom_title(self, slack):
        """Test formatting blocks with custom title."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=5,
            passed=5,
            failed=0,
            skipped=0,
            duration_seconds=10.0,
            cost_usd=0.01,
            failures=[],
        )

        blocks = slack._format_results_blocks(summary, title="Custom Title")

        assert "Custom Title" in blocks[0]["text"]["text"]


class TestSlackIntegrationFormatFailureBlocks:
    """Tests for SlackIntegration._format_failure_blocks method."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    def test_format_failure_blocks_basic(self, slack):
        """Test formatting basic failure blocks."""
        blocks = slack._format_failure_blocks(
            test_id="test-123",
            error="Element not found: #login-button",
        )

        assert len(blocks) > 0
        assert blocks[0]["type"] == "header"
        assert "Test Failure Alert" in blocks[0]["text"]["text"]

        # Should contain test ID and error
        block_text = str(blocks)
        assert "test-123" in block_text
        assert "Element not found" in block_text

    def test_format_failure_blocks_with_root_cause(self, slack):
        """Test formatting failure blocks with root cause."""
        blocks = slack._format_failure_blocks(
            test_id="test-123",
            error="Element not found",
            root_cause="Selector changed in recent deploy",
        )

        block_text = str(blocks)
        assert "Root Cause Analysis" in block_text
        assert "Selector changed" in block_text

    def test_format_failure_blocks_with_screenshot(self, slack):
        """Test formatting failure blocks with screenshot."""
        blocks = slack._format_failure_blocks(
            test_id="test-123",
            error="Element not found",
            screenshot_url="https://example.com/screenshot.png",
        )

        # Should have image block
        has_image = any(b["type"] == "image" for b in blocks)
        assert has_image


class TestSlackIntegrationSendTestResults:
    """Tests for SlackIntegration.send_test_results method."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    @pytest.fixture
    def test_summary(self, mock_env_vars):
        """Create a test summary."""
        from src.integrations.slack_integration import TestSummary

        return TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
        )

    @pytest.mark.asyncio
    async def test_send_test_results_via_webhook(self, slack, test_summary):
        """Test sending test results via webhook."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_test_results(test_summary)

            assert result is True
            mock_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_send_test_results_not_configured(self, mock_env_vars):
        """Test send_test_results when not configured."""
        from src.integrations.slack_integration import SlackIntegration, TestSummary

        slack = SlackIntegration()

        summary = TestSummary(
            total=10,
            passed=10,
            failed=0,
            skipped=0,
            duration_seconds=30.5,
            cost_usd=0.05,
            failures=[],
        )

        result = await slack.send_test_results(summary)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_test_results_custom_channel(self, slack, test_summary):
        """Test sending test results to custom channel."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_test_results(
                test_summary,
                channel="#custom-channel",
            )

            assert result is True


class TestSlackIntegrationSendFailureAlert:
    """Tests for SlackIntegration.send_failure_alert method."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_failure_alert_success(self, slack):
        """Test sending failure alert."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_failure_alert(
                test_id="test-123",
                error="Element not found",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_send_failure_alert_with_details(self, slack):
        """Test sending failure alert with all details."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_failure_alert(
                test_id="test-123",
                error="Element not found",
                root_cause="Selector changed",
                screenshot_url="https://example.com/screenshot.png",
                channel="#alerts",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_send_failure_alert_not_configured(self, mock_env_vars):
        """Test send_failure_alert when not configured."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration()

        result = await slack.send_failure_alert(
            test_id="test-123",
            error="Error",
        )

        assert result is False


class TestSlackIntegrationSendSimpleMessage:
    """Tests for SlackIntegration.send_simple_message method."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_simple_message_success(self, slack):
        """Test sending simple message."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_simple_message("Hello, Slack!")

            assert result is True

    @pytest.mark.asyncio
    async def test_send_simple_message_not_configured(self, mock_env_vars):
        """Test send_simple_message when not configured."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration()

        result = await slack.send_simple_message("Hello")

        assert result is False


class TestSlackIntegrationSendMessage:
    """Tests for SlackIntegration._send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_via_webhook(self, mock_env_vars):
        """Test _send_message via webhook."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(webhook_url="https://hooks.slack.com/test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack._send_message(text="Test")

            assert result is True
            # Verify webhook was called
            call_args = mock_client.post.call_args
            assert "hooks.slack.com" in str(call_args)

    @pytest.mark.asyncio
    async def test_send_message_via_api(self, mock_env_vars):
        """Test _send_message via Slack API."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack._send_message(
                text="Test",
                channel="#testing",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_send_message_webhook_fallback_to_api(self, mock_env_vars):
        """Test _send_message falls back to API on webhook failure."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(
            webhook_url="https://hooks.slack.com/test",
            bot_token="xoxb-test-token",
        )

        mock_webhook_response = MagicMock()
        mock_webhook_response.status_code = 500

        mock_api_response = MagicMock()
        mock_api_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[mock_webhook_response, mock_api_response]
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack._send_message(
                text="Test",
                channel="#testing",
            )

            assert result is True
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_message_webhook_error(self, mock_env_vars):
        """Test _send_message handles webhook error."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(webhook_url="https://hooks.slack.com/test")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack._send_message(text="Test")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, mock_env_vars):
        """Test _send_message handles API error."""
        from src.integrations.slack_integration import SlackIntegration

        slack = SlackIntegration(bot_token="xoxb-test-token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack._send_message(
                text="Test",
                channel="#invalid",
            )

            assert result is False


class TestSlackIntegrationTestStartedCompleted:
    """Tests for send_test_started and send_test_completed methods."""

    @pytest.fixture
    def slack(self, mock_env_vars):
        """Create a SlackIntegration instance."""
        from src.integrations.slack_integration import SlackIntegration

        return SlackIntegration(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_send_test_started(self, slack):
        """Test sending test started notification."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_test_started(test_count=50)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_test_started_with_pr(self, slack):
        """Test sending test started notification with PR number."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_test_started(test_count=50, pr_number=123)

            assert result is True
            # Verify PR number in message
            call_args = mock_client.post.call_args
            assert "123" in str(call_args)

    @pytest.mark.asyncio
    async def test_send_test_completed(self, slack):
        """Test sending test completed notification."""
        from src.integrations.slack_integration import TestSummary

        summary = TestSummary(
            total=50,
            passed=48,
            failed=2,
            skipped=0,
            duration_seconds=120.0,
            cost_usd=0.5,
            failures=[],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await slack.send_test_completed(summary)

            assert result is True


class TestSlackNotifier:
    """Tests for SlackNotifier class (high-level wrapper)."""

    @pytest.fixture
    def notifier(self, mock_env_vars):
        """Create a SlackNotifier instance."""
        from src.integrations.slack_integration import SlackNotifier

        return SlackNotifier(webhook_url="https://hooks.slack.com/test")

    @pytest.mark.asyncio
    async def test_notify_start(self, notifier):
        """Test notify_start method."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.notify_start(test_count=50)

            assert result is True

    @pytest.mark.asyncio
    async def test_notify_complete(self, notifier):
        """Test notify_complete method."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.notify_complete(
                passed=48,
                failed=2,
                skipped=0,
                duration=120.0,
                cost=0.5,
                failures=[],
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_notify_failure(self, notifier):
        """Test notify_failure method."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await notifier.notify_failure(
                test_id="test-123",
                error="Element not found",
                root_cause="Selector changed",
            )

            assert result is True


class TestCreateSlackIntegration:
    """Tests for create_slack_integration factory function."""

    def test_create_with_webhook(self, mock_env_vars):
        """Test creating integration with webhook."""
        from src.integrations.slack_integration import create_slack_integration

        slack = create_slack_integration(webhook_url="https://hooks.slack.com/test")

        assert slack.webhook_url == "https://hooks.slack.com/test"

    def test_create_with_bot_token(self, mock_env_vars):
        """Test creating integration with bot token."""
        from src.integrations.slack_integration import create_slack_integration

        slack = create_slack_integration(bot_token="xoxb-test-token")

        assert slack.bot_token == "xoxb-test-token"

    def test_create_without_credentials(self, mock_env_vars, monkeypatch):
        """Test creating integration without credentials."""
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        from src.integrations.slack_integration import create_slack_integration

        slack = create_slack_integration()

        assert slack.webhook_url is None
        assert slack.bot_token is None
