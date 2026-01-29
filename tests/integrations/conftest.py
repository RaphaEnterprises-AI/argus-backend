"""Shared fixtures for integration plugin tests.

This module provides fixtures for testing the plugin architecture including:
- Mock HTTP clients and responses
- Sample credentials for various platforms
- Mock API response factories
"""

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    OAuthConfig,
    SyncResult,
)


# ==============================================================================
# Mock HTTP Client Fixtures
# ==============================================================================


@pytest.fixture
def mock_httpx_async_client():
    """Create a mock httpx.AsyncClient for testing.

    This fixture creates a fully mocked async HTTP client that can be
    configured to return specific responses for different endpoints.
    """
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Default successful response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}
    mock_response.text = '{"success": true}'
    mock_response.content = b'{"success": true}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.raise_for_status = MagicMock()

    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_response)
    mock_client.patch = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=mock_response)
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()

    return mock_client


@pytest.fixture
def mock_response_factory():
    """Factory for creating mock HTTP responses.

    Returns a function that creates mock responses with specified
    status codes and JSON data.
    """

    def create_response(
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str | None = None,
        headers: dict[str, str] | None = None,
        raise_for_status: bool = False,
    ) -> MagicMock:
        """Create a mock HTTP response.

        Args:
            status_code: HTTP status code (default: 200)
            json_data: JSON response data (default: {"success": True})
            text: Text response (auto-generated from json_data if not provided)
            headers: Response headers
            raise_for_status: If True, raise_for_status will raise an exception

        Returns:
            MagicMock configured as an httpx.Response
        """
        if json_data is None:
            json_data = {"success": True}

        if text is None:
            text = json.dumps(json_data)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data
        mock_response.text = text
        mock_response.content = text.encode()
        mock_response.headers = headers or {"content-type": "application/json"}

        if raise_for_status and status_code >= 400:
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=mock_response,
            )
        else:
            mock_response.raise_for_status = MagicMock()

        return mock_response

    return create_response


# ==============================================================================
# Sample Credentials Fixtures
# ==============================================================================


@pytest.fixture
def github_credentials() -> IntegrationCredentials:
    """Sample GitHub credentials for testing."""
    return IntegrationCredentials(
        access_token="ghp_test_token_1234567890",
        org_slug="test-org",
    )


@pytest.fixture
def gitlab_credentials() -> IntegrationCredentials:
    """Sample GitLab credentials for testing."""
    return IntegrationCredentials(
        access_token="glpat-test-token-1234567890",
        instance_url="https://gitlab.example.com",
    )


@pytest.fixture
def slack_credentials() -> IntegrationCredentials:
    """Sample Slack credentials for testing."""
    return IntegrationCredentials(
        access_token="xoxb-test-slack-token",
        workspace_id="T0123456789",
    )


@pytest.fixture
def slack_webhook_credentials() -> IntegrationCredentials:
    """Sample Slack webhook credentials for testing."""
    return IntegrationCredentials(
        webhook_url="https://hooks.slack.com/services/T00/B00/XXX",
        webhook_secret="webhook_signing_secret_123",
    )


@pytest.fixture
def jira_credentials() -> IntegrationCredentials:
    """Sample Jira credentials for testing."""
    return IntegrationCredentials(
        email="test@example.com",
        api_key="jira_api_token_12345",
        instance_url="https://test.atlassian.net",
    )


@pytest.fixture
def sentry_credentials() -> IntegrationCredentials:
    """Sample Sentry credentials for testing."""
    return IntegrationCredentials(
        api_key="sentry_api_key_12345",
        org_slug="test-org",
        project_slug="test-project",
    )


@pytest.fixture
def datadog_credentials() -> IntegrationCredentials:
    """Sample Datadog credentials for testing."""
    return IntegrationCredentials(
        api_key="datadog_api_key_12345",
        api_secret="datadog_app_key_12345",  # Datadog uses app_key as second key
        site="datadoghq.com",
    )


@pytest.fixture
def vercel_credentials() -> IntegrationCredentials:
    """Sample Vercel credentials for testing."""
    return IntegrationCredentials(
        access_token="vercel_test_token_12345",
        team_id="team_12345",
    )


@pytest.fixture
def linear_credentials() -> IntegrationCredentials:
    """Sample Linear credentials for testing."""
    return IntegrationCredentials(
        api_key="lin_api_key_12345",
        team_id="team_12345",
    )


@pytest.fixture
def pagerduty_credentials() -> IntegrationCredentials:
    """Sample PagerDuty credentials for testing."""
    return IntegrationCredentials(
        api_key="pd_api_key_12345",
    )


@pytest.fixture
def launchdarkly_credentials() -> IntegrationCredentials:
    """Sample LaunchDarkly credentials for testing."""
    return IntegrationCredentials(
        api_key="api-key-12345",
        extra={"sdk_key": "sdk-key-12345"},
    )


@pytest.fixture
def oauth_credentials() -> IntegrationCredentials:
    """Sample OAuth credentials for testing token refresh."""
    return IntegrationCredentials(
        access_token="access_token_12345",
        refresh_token="refresh_token_12345",
        client_id="client_id_12345",
        client_secret="client_secret_12345",
        token_expires_at=datetime.utcnow() + timedelta(hours=1),
    )


@pytest.fixture
def expired_oauth_credentials() -> IntegrationCredentials:
    """Sample expired OAuth credentials for testing token refresh."""
    return IntegrationCredentials(
        access_token="expired_access_token",
        refresh_token="refresh_token_12345",
        token_expires_at=datetime.utcnow() - timedelta(hours=1),
    )


@pytest.fixture
def empty_credentials() -> IntegrationCredentials:
    """Empty credentials for testing validation failures."""
    return IntegrationCredentials()


# ==============================================================================
# Mock API Response Fixtures
# ==============================================================================


@pytest.fixture
def github_api_responses():
    """Mock GitHub API responses."""
    return {
        "user": {
            "login": "testuser",
            "id": 12345,
            "name": "Test User",
            "email": "test@example.com",
            "type": "User",
        },
        "rate_limit": {
            "resources": {
                "core": {"limit": 5000, "remaining": 4999, "reset": 1234567890}
            }
        },
        "repos": [
            {
                "id": 1,
                "name": "test-repo",
                "full_name": "test-org/test-repo",
                "private": False,
                "default_branch": "main",
            }
        ],
        "issues": [
            {
                "id": 1,
                "number": 1,
                "title": "Test Issue",
                "state": "open",
                "body": "Test issue body",
            }
        ],
        "pull_requests": [
            {
                "id": 1,
                "number": 1,
                "title": "Test PR",
                "state": "open",
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"},
            }
        ],
    }


@pytest.fixture
def slack_api_responses():
    """Mock Slack API responses."""
    return {
        "auth.test": {
            "ok": True,
            "url": "https://test.slack.com/",
            "team": "Test Team",
            "user": "testbot",
            "team_id": "T0123456789",
            "user_id": "U0123456789",
        },
        "conversations.list": {
            "ok": True,
            "channels": [
                {"id": "C0123456789", "name": "general", "is_private": False}
            ],
        },
        "chat.postMessage": {
            "ok": True,
            "channel": "C0123456789",
            "ts": "1234567890.123456",
            "message": {"text": "Test message"},
        },
        "error_response": {"ok": False, "error": "invalid_auth"},
    }


@pytest.fixture
def jira_api_responses():
    """Mock Jira API responses."""
    return {
        "myself": {
            "accountId": "12345",
            "displayName": "Test User",
            "emailAddress": "test@example.com",
            "active": True,
        },
        "serverInfo": {
            "version": "8.0.0",
            "deploymentType": "Cloud",
            "baseUrl": "https://test.atlassian.net",
        },
        "search": {
            "total": 1,
            "issues": [
                {
                    "id": "10001",
                    "key": "TEST-1",
                    "fields": {
                        "summary": "Test Issue",
                        "status": {"name": "Open"},
                    },
                }
            ],
        },
        "projects": [
            {"id": "10000", "key": "TEST", "name": "Test Project"}
        ],
    }


@pytest.fixture
def sentry_api_responses():
    """Mock Sentry API responses."""
    return {
        "organization": {
            "id": "12345",
            "slug": "test-org",
            "name": "Test Organization",
        },
        "projects": [
            {
                "id": "12345",
                "slug": "test-project",
                "name": "Test Project",
                "platform": "python",
            }
        ],
        "issues": [
            {
                "id": "12345",
                "shortId": "TEST-1",
                "title": "Test Error",
                "culprit": "test.py in main",
                "count": 10,
                "firstSeen": "2024-01-01T00:00:00Z",
                "lastSeen": "2024-01-15T00:00:00Z",
            }
        ],
    }


@pytest.fixture
def datadog_api_responses():
    """Mock Datadog API responses."""
    return {
        "validate": {"valid": True},
        "monitors": {
            "monitors": [
                {
                    "id": 12345,
                    "name": "Test Monitor",
                    "type": "metric alert",
                    "overall_state": "OK",
                }
            ]
        },
        "metrics": {
            "series": [
                {"metric": "system.cpu.user", "pointlist": [[1234567890, 50.0]]}
            ]
        },
    }


@pytest.fixture
def vercel_api_responses():
    """Mock Vercel API responses."""
    return {
        "user": {"id": "user_12345", "username": "testuser", "email": "test@example.com"},
        "teams": [{"id": "team_12345", "name": "Test Team", "slug": "test-team"}],
        "projects": [
            {
                "id": "prj_12345",
                "name": "test-project",
                "framework": "nextjs",
            }
        ],
        "deployments": [
            {
                "uid": "dpl_12345",
                "name": "test-project",
                "url": "test-project-abc123.vercel.app",
                "state": "READY",
            }
        ],
    }


# ==============================================================================
# Mock Plugin Fixtures
# ==============================================================================


class MockPlugin(IntegrationPlugin):
    """A mock plugin implementation for testing the base class."""

    def __init__(
        self,
        slug: str = "mock",
        name: str = "Mock Plugin",
        category: IntegrationCategory = IntegrationCategory.TESTING,
        auth_type: AuthType = AuthType.API_KEY,
        required_fields: list[FieldDefinition] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        super().__init__(http_client)
        self._slug = slug
        self._name = name
        self._category = category
        self._auth_type = auth_type
        self._required_fields = required_fields or [
            FieldDefinition(name="api_key", label="API Key", type="password")
        ]

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug=self._slug,
            name=self._name,
            description=f"Mock {self._name} plugin for testing",
            category=self._category,
            auth_type=self._auth_type,
            features=["test_connection", "sync_data"],
            required_fields=self._required_fields,
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["items", "events"],
        )

    async def test_connection(
        self, credentials: IntegrationCredentials
    ) -> ConnectionTestResult:
        """Mock test_connection that checks for api_key."""
        if credentials.api_key:
            return ConnectionTestResult(
                success=True,
                message="Connection successful",
                latency_ms=50.0,
                api_version="1.0",
                account_info={"account": "test"},
            )
        return ConnectionTestResult(
            success=False,
            message="Invalid credentials",
            error_code="INVALID_CREDENTIALS",
        )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> SyncResult:
        """Mock sync_data that returns test data."""
        if not credentials.api_key:
            return SyncResult(
                success=False,
                message="Invalid credentials",
                errors=["Missing API key"],
            )

        return SyncResult(
            success=True,
            message="Sync completed",
            data_points_synced=10,
            data={
                "items": [{"id": i, "name": f"Item {i}"} for i in range(min(limit, 10))]
            },
        )


class MockOAuthPlugin(IntegrationPlugin):
    """A mock OAuth plugin implementation for testing OAuth flows."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        super().__init__(http_client)

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="mock-oauth",
            name="Mock OAuth Plugin",
            description="Mock OAuth plugin for testing",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
            features=["oauth"],
            required_fields=[],
            oauth_config=OAuthConfig(
                authorize_url="https://oauth.example.com/authorize",
                token_url="https://oauth.example.com/token",
                scopes=["read", "write"],
                client_id_env_var="MOCK_CLIENT_ID",
                client_secret_env_var="MOCK_CLIENT_SECRET",
            ),
        )

    async def test_connection(
        self, credentials: IntegrationCredentials
    ) -> ConnectionTestResult:
        if credentials.access_token:
            return ConnectionTestResult(
                success=True,
                message="OAuth connection successful",
                api_version="2.0",
            )
        return ConnectionTestResult(
            success=False,
            message="No access token",
            error_code="NO_TOKEN",
        )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> SyncResult:
        return SyncResult(
            success=True,
            message="OAuth sync completed",
            data_points_synced=5,
        )


@pytest.fixture
def mock_plugin(mock_httpx_async_client):
    """Create a mock plugin instance."""
    return MockPlugin(http_client=mock_httpx_async_client)


@pytest.fixture
def mock_oauth_plugin(mock_httpx_async_client):
    """Create a mock OAuth plugin instance."""
    return MockOAuthPlugin(http_client=mock_httpx_async_client)


@pytest.fixture
def mock_plugin_class():
    """Return the MockPlugin class for registration tests."""
    return MockPlugin


@pytest.fixture
def mock_oauth_plugin_class():
    """Return the MockOAuthPlugin class for registration tests."""
    return MockOAuthPlugin


# ==============================================================================
# Helper Functions
# ==============================================================================


def generate_webhook_signature(secret: str, payload: dict) -> str:
    """Generate a webhook signature for testing.

    Args:
        secret: The webhook signing secret
        payload: The payload to sign

    Returns:
        The signature in format "sha256=<hex_digest>"
    """
    body = json.dumps(payload)
    signature = hmac.new(
        secret.encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


@pytest.fixture
def webhook_signature_generator():
    """Fixture that provides the webhook signature generator function."""
    return generate_webhook_signature
