"""Tests for Webhooks API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestWebhookModels:
    """Tests for webhook request/response models."""

    def test_sentry_exception_model(self, mock_env_vars):
        """Test SentryException model."""
        from src.api.webhooks import SentryException

        exc = SentryException(
            type="TypeError",
            value="Cannot read property 'length' of undefined",
        )

        assert exc.type == "TypeError"
        assert "undefined" in exc.value

    def test_sentry_metadata_model(self, mock_env_vars):
        """Test SentryMetadata model."""
        from src.api.webhooks import SentryMetadata

        metadata = SentryMetadata(
            type="TypeError",
            value="Error message",
            filename="app.js",
            function="handleClick",
        )

        assert metadata.filename == "app.js"
        assert metadata.function == "handleClick"

    def test_sentry_issue_model(self, mock_env_vars):
        """Test SentryIssue model."""
        from src.api.webhooks import SentryIssue

        issue = SentryIssue(
            id="12345",
            title="TypeError: Cannot read property",
            culprit="handleClick",
            level="error",
            message="Error in component",
            platform="javascript",
            project="frontend",
            url="https://sentry.io/issues/12345",
            shortId="PROJ-123",
            count="5",
            userCount=3,
        )

        assert issue.id == "12345"
        assert issue.userCount == 3

    def test_sentry_webhook_payload_model(self, mock_env_vars):
        """Test SentryWebhookPayload model."""
        from src.api.webhooks import SentryWebhookPayload

        payload = SentryWebhookPayload(
            action="created",
            data={"issue": {"id": "123"}},
            installation={"uuid": "install-uuid"},
            actor={"type": "user", "id": "user-123"},
        )

        assert payload.action == "created"
        assert payload.data["issue"]["id"] == "123"

    def test_datadog_error_model(self, mock_env_vars):
        """Test DatadogError model."""
        from src.api.webhooks import DatadogError

        error = DatadogError(
            type="NetworkError",
            message="Failed to fetch",
            stack="Error at fetchData (app.js:10)",
            source="https://example.com/app.js",
        )

        assert error.type == "NetworkError"
        assert "fetchData" in error.stack

    def test_datadog_event_model(self, mock_env_vars):
        """Test DatadogEvent model."""
        from src.api.webhooks import DatadogError, DatadogEvent, DatadogView

        event = DatadogEvent(
            id="event-123",
            title="Error Alert",
            message="Application error detected",
            priority="high",
            host="web-server-1",
            tags=["env:production", "service:api"],
            alert_type="error",
            error=DatadogError(type="Error", message="Test"),
            view=DatadogView(url="/dashboard", name="Dashboard"),
        )

        assert event.title == "Error Alert"
        assert "env:production" in event.tags

    def test_production_event_model(self, mock_env_vars):
        """Test ProductionEvent model."""
        from src.api.webhooks import ProductionEvent

        event = ProductionEvent(
            project_id="project-123",
            source="sentry",
            external_id="ext-456",
            event_type="error",
            severity="error",
            title="Application Error",
            message="Something went wrong",
            fingerprint="abc123",
            occurrence_count=5,
            affected_users=3,
            tags=["frontend", "critical"],
        )

        assert event.source == "sentry"
        assert event.severity == "error"
        assert event.occurrence_count == 5

    def test_webhook_response_model(self, mock_env_vars):
        """Test WebhookResponse model."""
        from src.api.webhooks import WebhookResponse

        response = WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id="event-123",
            fingerprint="abc123",
        )

        assert response.success is True
        assert response.event_id == "event-123"


class TestHelperFunctions:
    """Tests for webhook helper functions."""

    def test_parse_severity_fatal(self, mock_env_vars):
        """Test parse_severity with fatal level."""
        from src.api.webhooks import parse_severity

        assert parse_severity("fatal") == "fatal"
        assert parse_severity("FATAL") == "fatal"

    def test_parse_severity_error(self, mock_env_vars):
        """Test parse_severity with error levels."""
        from src.api.webhooks import parse_severity

        assert parse_severity("error") == "error"
        assert parse_severity("ERROR") == "error"
        assert parse_severity("high") == "error"

    def test_parse_severity_warning(self, mock_env_vars):
        """Test parse_severity with warning levels."""
        from src.api.webhooks import parse_severity

        assert parse_severity("warning") == "warning"
        assert parse_severity("warn") == "warning"
        assert parse_severity("normal") == "warning"

    def test_parse_severity_info(self, mock_env_vars):
        """Test parse_severity defaults to info."""
        from src.api.webhooks import parse_severity

        assert parse_severity("info") == "info"
        assert parse_severity("low") == "info"
        assert parse_severity("unknown") == "info"

    def test_extract_component_react(self, mock_env_vars):
        """Test extract_component_from_stack with React stack trace."""
        from src.api.webhooks import extract_component_from_stack

        stack = """Error: Something went wrong
        at LoginForm (LoginForm.tsx:45:12)
        at render (react-dom.js:100:10)"""

        component = extract_component_from_stack(stack)
        assert component == "LoginForm"

    def test_extract_component_vue(self, mock_env_vars):
        """Test extract_component_from_stack with Vue stack trace."""
        from src.api.webhooks import extract_component_from_stack

        stack = """Error in component
        VueComponent.handleSubmit (Dashboard.vue:30:5)"""

        component = extract_component_from_stack(stack)
        assert component == "handleSubmit"

    def test_extract_component_angular(self, mock_env_vars):
        """Test extract_component_from_stack with Angular stack trace."""
        from src.api.webhooks import extract_component_from_stack

        stack = """Error occurred
        at UserProfileComponent.ngOnInit (user-profile.component.ts:25:10)"""

        component = extract_component_from_stack(stack)
        assert component == "UserProfileComponent"

    def test_extract_component_none(self, mock_env_vars):
        """Test extract_component_from_stack returns None for no match."""
        from src.api.webhooks import extract_component_from_stack

        assert extract_component_from_stack(None) is None
        assert extract_component_from_stack("") is None
        assert extract_component_from_stack("plain text without component") is None

    def test_generate_fingerprint_basic(self, mock_env_vars):
        """Test generate_fingerprint creates consistent hash."""
        from src.api.webhooks import generate_fingerprint

        fp1 = generate_fingerprint("TypeError", "Cannot read property", "LoginForm", "/login")
        fp2 = generate_fingerprint("TypeError", "Cannot read property", "LoginForm", "/login")

        assert fp1 == fp2
        assert len(fp1) == 12  # SHA256 truncated to 12 chars

    def test_generate_fingerprint_normalizes_url(self, mock_env_vars):
        """Test generate_fingerprint normalizes URLs."""
        from src.api.webhooks import generate_fingerprint

        # Different IDs should produce same fingerprint
        fp1 = generate_fingerprint("Error", "msg", None, "/users/123/profile")
        fp2 = generate_fingerprint("Error", "msg", None, "/users/456/profile")

        assert fp1 == fp2  # IDs normalized to /:id

    def test_generate_fingerprint_normalizes_uuid(self, mock_env_vars):
        """Test generate_fingerprint normalizes UUIDs."""
        from src.api.webhooks import generate_fingerprint

        fp1 = generate_fingerprint("Error", "msg", None, "/orders/a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        fp2 = generate_fingerprint("Error", "msg", None, "/orders/11111111-2222-3333-4444-555555555555")

        assert fp1 == fp2  # UUIDs normalized to /:uuid


class TestSentryWebhook:
    """Tests for Sentry webhook handler."""

    @pytest.mark.asyncio
    async def test_sentry_webhook_issue_created(self, mock_env_vars):
        """Test handling Sentry issue created webhook."""
        from src.api.webhooks import handle_sentry_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "created",
            "data": {
                "issue": {
                    "id": "12345",
                    "title": "TypeError: Cannot read property",
                    "level": "error",
                    "message": "Error in component",
                    "count": "1",
                    "userCount": 1,
                    "tags": [{"key": "env", "value": "production"}],
                },
                "event": {
                    "event_id": "event-123",
                    "platform": "javascript",
                    "event": {
                        "exception": {
                            "values": [{"type": "TypeError", "value": "Test error"}]
                        },
                        "request": {"url": "https://example.com/page"},
                        "contexts": {},
                    },
                },
            },
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_client.request = AsyncMock(return_value={"data": [{"id": "project-1"}]})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_sentry_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True
                            assert response.event_id == "prod-event-1"

    @pytest.mark.asyncio
    async def test_sentry_webhook_issue_resolved(self, mock_env_vars):
        """Test handling Sentry issue resolved webhook."""
        from src.api.webhooks import handle_sentry_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "resolved",
            "data": {
                "issue": {"id": "12345"},
            },
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.update = AsyncMock(return_value={"data": [{}], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        response = await handle_sentry_webhook(
                            mock_request,
                            organization_id="org-123",
                            project_id="project-1",
                        )

                        assert response.success is True
                        assert "resolved" in response.message.lower()

    @pytest.mark.asyncio
    async def test_sentry_webhook_invalid_project(self, mock_env_vars):
        """Test Sentry webhook with invalid project for org."""
        from src.api.webhooks import handle_sentry_webhook

        mock_request = MagicMock()

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=False)):
                with pytest.raises(HTTPException) as exc_info:
                    await handle_sentry_webhook(
                        mock_request,
                        organization_id="org-123",
                        project_id="wrong-project",
                    )

                assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_sentry_webhook_no_issue_data(self, mock_env_vars):
        """Test Sentry webhook with missing issue data."""
        from src.api.webhooks import handle_sentry_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "created",
            "data": {},  # No issue or event data
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with pytest.raises(HTTPException) as exc_info:
                        await handle_sentry_webhook(
                            mock_request,
                            organization_id="org-123",
                            project_id="project-1",
                        )

                    assert exc_info.value.status_code == 400
                    assert "No issue or event data" in str(exc_info.value.detail)


class TestDatadogWebhook:
    """Tests for Datadog webhook handler."""

    @pytest.mark.asyncio
    async def test_datadog_webhook_single_event(self, mock_env_vars):
        """Test handling single Datadog event."""
        from src.api.webhooks import handle_datadog_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "id": "dd-event-1",
            "title": "High Error Rate Alert",
            "message": "Error rate exceeded threshold",
            "priority": "high",
            "alert_type": "error",
            "tags": ["service:api", "env:prod"],
            "error": {
                "type": "NetworkError",
                "message": "Connection refused",
                "stack": "at connect (http.js:10)",
            },
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_client.request = AsyncMock(return_value={"data": [{"id": "project-1"}]})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_datadog_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True

    @pytest.mark.asyncio
    async def test_datadog_webhook_multiple_events(self, mock_env_vars):
        """Test handling multiple Datadog events in batch."""
        from src.api.webhooks import handle_datadog_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value=[
            {"title": "Event 1", "message": "First event", "alert_type": "error"},
            {"title": "Event 2", "message": "Second event", "alert_type": "warning"},
        ])

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_client.request = AsyncMock(return_value={"data": [{"id": "project-1"}]})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_datadog_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True
                            assert "2 events" in response.message

    @pytest.mark.asyncio
    async def test_datadog_webhook_no_project(self, mock_env_vars):
        """Test Datadog webhook without project falls back to default."""
        from src.api.webhooks import handle_datadog_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "title": "Alert",
            "message": "Test",
            "alert_type": "error",
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_client.request = AsyncMock(return_value={"data": [{"id": "default-project"}]})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.log_webhook", AsyncMock()):
                with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                    with patch("src.api.webhooks.get_default_project_id", AsyncMock(return_value="default-project")):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_datadog_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id=None,
                            )

                            assert response.success is True


class TestFullStoryWebhook:
    """Tests for FullStory webhook handler."""

    @pytest.mark.asyncio
    async def test_fullstory_webhook_rage_click(self, mock_env_vars):
        """Test handling FullStory rage click event."""
        from src.api.webhooks import handle_fullstory_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "id": "fs-event-1",
            "type": "rage_click",
            "title": "Rage Click Detected",
            "message": "User rage clicked on button",
            "session": {"url": "https://fullstory.com/session/123"},
            "page": {"url": "https://example.com/checkout"},
            "element": {"selector": "#submit-btn"},
            "count": 5,
            "userCount": 1,
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_fullstory_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True
                            assert response.fingerprint is not None

    @pytest.mark.asyncio
    async def test_fullstory_webhook_dead_click(self, mock_env_vars):
        """Test handling FullStory dead click event."""
        from src.api.webhooks import handle_fullstory_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "type": "dead_click",
            "pageUrl": "https://example.com/broken",
            "selector": ".broken-link",
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_fullstory_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True


class TestLogRocketWebhook:
    """Tests for LogRocket webhook handler."""

    @pytest.mark.asyncio
    async def test_logrocket_webhook_error(self, mock_env_vars):
        """Test handling LogRocket error event."""
        from src.api.webhooks import handle_logrocket_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "id": "lr-event-1",
            "error": {
                "type": "TypeError",
                "name": "TypeError",
                "message": "Cannot read property 'foo' of null",
                "stack": "at Component.render (app.js:100)",
            },
            "session": {
                "id": "session-123",
                "url": "https://example.com/page",
                "sessionUrl": "https://app.logrocket.com/sessions/123",
                "browser": "Chrome 120",
                "os": "Windows 11",
            },
            "severity": "error",
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "prod-event-1"}], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with patch("src.api.webhooks.index_production_event", AsyncMock()):
                            response = await handle_logrocket_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                            assert response.success is True
                            assert response.fingerprint is not None

    @pytest.mark.asyncio
    async def test_logrocket_webhook_no_project(self, mock_env_vars):
        """Test LogRocket webhook fails without project."""
        from src.api.webhooks import handle_logrocket_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "error": {"message": "Test error"},
            "session": {},
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={"data": []})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.log_webhook", AsyncMock()):
                with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                    with patch("src.api.webhooks.get_default_project_id", AsyncMock(return_value=None)):
                        with pytest.raises(HTTPException) as exc_info:
                            await handle_logrocket_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id=None,
                            )

                        assert exc_info.value.status_code == 400
                        assert "No project found" in str(exc_info.value.detail)


class TestWebhookSecurityValidation:
    """Tests for webhook security validation."""

    @pytest.mark.asyncio
    async def test_validate_project_org_success(self, mock_env_vars):
        """Test successful project-org validation."""
        from src.api.webhooks import validate_project_org

        with patch("src.api.webhooks.get_supabase_client"):
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={
                "data": [{"id": "project-1"}],
            })

            result = await validate_project_org(mock_client, "project-1", "org-1")
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_project_org_failure(self, mock_env_vars):
        """Test failed project-org validation."""
        from src.api.webhooks import validate_project_org

        with patch("src.api.webhooks.get_supabase_client"):
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={
                "data": [],
            })

            result = await validate_project_org(mock_client, "project-1", "wrong-org")
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_project_org_empty_params(self, mock_env_vars):
        """Test validation with empty params."""
        from src.api.webhooks import validate_project_org

        mock_client = MagicMock()

        assert await validate_project_org(mock_client, "", "org-1") is False
        assert await validate_project_org(mock_client, "project-1", "") is False
        assert await validate_project_org(mock_client, None, "org-1") is False

    @pytest.mark.asyncio
    async def test_get_default_project_id_success(self, mock_env_vars):
        """Test getting default project for org."""
        from src.api.webhooks import get_default_project_id

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={
            "data": [{"id": "default-project"}],
        })

        result = await get_default_project_id(mock_client, "org-1")
        assert result == "default-project"

    @pytest.mark.asyncio
    async def test_get_default_project_id_no_org(self, mock_env_vars):
        """Test getting default project without org."""
        from src.api.webhooks import get_default_project_id

        mock_client = MagicMock()

        result = await get_default_project_id(mock_client, "")
        assert result is None

        result = await get_default_project_id(mock_client, None)
        assert result is None


class TestWebhookLogging:
    """Tests for webhook logging functionality."""

    @pytest.mark.asyncio
    async def test_log_webhook(self, mock_env_vars):
        """Test webhook logging."""
        from src.api.webhooks import log_webhook

        mock_supabase = MagicMock()
        mock_supabase.insert = AsyncMock(return_value={"data": [{}]})

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = {"Content-Type": "application/json"}

        await log_webhook(
            mock_supabase,
            webhook_id="webhook-123",
            source="sentry",
            request=mock_request,
            body={"test": "data"},
            status="processing",
        )

        mock_supabase.insert.assert_called_once()
        call_args = mock_supabase.insert.call_args
        assert call_args[0][0] == "webhook_logs"

    @pytest.mark.asyncio
    async def test_update_webhook_log(self, mock_env_vars):
        """Test updating webhook log."""
        from src.api.webhooks import update_webhook_log

        mock_supabase = MagicMock()
        mock_supabase.update = AsyncMock(return_value={"data": [{}]})

        await update_webhook_log(
            mock_supabase,
            webhook_id="webhook-123",
            status="processed",
            error_message=None,
            event_id="event-456",
        )

        mock_supabase.update.assert_called_once()
        call_args = mock_supabase.update.call_args
        assert call_args[0][0] == "webhook_logs"
        assert "processed_at" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_update_webhook_log_with_error(self, mock_env_vars):
        """Test updating webhook log with error."""
        from src.api.webhooks import update_webhook_log

        mock_supabase = MagicMock()
        mock_supabase.update = AsyncMock(return_value={"data": [{}]})

        await update_webhook_log(
            mock_supabase,
            webhook_id="webhook-123",
            status="failed",
            error_message="Something went wrong",
        )

        call_args = mock_supabase.update.call_args
        assert "error_message" in call_args[0][2]
        assert call_args[0][2]["error_message"] == "Something went wrong"


class TestWebhookErrorHandling:
    """Tests for webhook error handling."""

    @pytest.mark.asyncio
    async def test_sentry_webhook_insert_failure(self, mock_env_vars):
        """Test Sentry webhook handles insert failure."""
        from src.api.webhooks import handle_sentry_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "created",
            "data": {
                "issue": {"id": "123", "title": "Error", "level": "error"},
                "event": {"event_id": "evt-1"},
            },
        })

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": None, "error": "Insert failed"})
            mock_client.request = AsyncMock(return_value={"data": [{"id": "project-1"}]})
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with patch("src.api.webhooks.log_webhook", AsyncMock()):
                    with patch("src.api.webhooks.update_webhook_log", AsyncMock()):
                        with pytest.raises(HTTPException) as exc_info:
                            await handle_sentry_webhook(
                                mock_request,
                                organization_id="org-123",
                                project_id="project-1",
                            )

                        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_datadog_webhook_exception_handling(self, mock_env_vars):
        """Test Datadog webhook handles exceptions."""
        from src.api.webhooks import handle_datadog_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=Exception("JSON parse error"))

        with patch("src.api.webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_supabase.return_value = mock_client

            with patch("src.api.webhooks.validate_project_org", AsyncMock(return_value=True)):
                with pytest.raises(HTTPException) as exc_info:
                    await handle_datadog_webhook(
                        mock_request,
                        organization_id="org-123",
                        project_id="project-1",
                    )

                assert exc_info.value.status_code == 500
