"""Mock tests for integration plugins.

This module tests plugin implementations with mocked HTTP responses:
- Test test_connection succeeds with valid credentials
- Test test_connection fails with invalid credentials
- Test sync_data returns expected format
- Test error handling for API failures

These tests use the MockPlugin and MockOAuthPlugin classes to verify
the plugin interface contract without making actual API calls.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
# Mock Plugin Implementation Tests
# ==============================================================================


class TestMockPluginTestConnection:
    """Tests for test_connection with mock plugins."""

    @pytest.mark.asyncio
    async def test_connection_success_with_valid_credentials(self, mock_plugin):
        """Test that test_connection succeeds with valid API key."""
        credentials = IntegrationCredentials(api_key="valid_api_key")

        result = await mock_plugin.test_connection(credentials)

        assert result.success is True
        assert result.message == "Connection successful"
        assert result.latency_ms == 50.0
        assert result.api_version == "1.0"
        assert result.account_info == {"account": "test"}

    @pytest.mark.asyncio
    async def test_connection_fails_with_missing_api_key(self, mock_plugin):
        """Test that test_connection fails without API key."""
        credentials = IntegrationCredentials()  # No api_key

        result = await mock_plugin.test_connection(credentials)

        assert result.success is False
        assert result.message == "Invalid credentials"
        assert result.error_code == "INVALID_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_connection_fails_with_empty_api_key(self, mock_plugin):
        """Test that test_connection fails with empty API key."""
        credentials = IntegrationCredentials(api_key="")

        result = await mock_plugin.test_connection(credentials)

        assert result.success is False


class TestMockPluginSyncData:
    """Tests for sync_data with mock plugins."""

    @pytest.mark.asyncio
    async def test_sync_data_success(self, mock_plugin):
        """Test that sync_data succeeds with valid credentials."""
        credentials = IntegrationCredentials(api_key="valid_api_key")

        result = await mock_plugin.sync_data(credentials)

        assert result.success is True
        assert result.message == "Sync completed"
        assert result.data_points_synced == 10
        assert "items" in result.data
        assert len(result.data["items"]) == 10

    @pytest.mark.asyncio
    async def test_sync_data_with_limit(self, mock_plugin):
        """Test that sync_data respects limit parameter."""
        credentials = IntegrationCredentials(api_key="valid_api_key")

        result = await mock_plugin.sync_data(credentials, limit=5)

        assert result.success is True
        assert len(result.data["items"]) <= 5

    @pytest.mark.asyncio
    async def test_sync_data_with_resources(self, mock_plugin):
        """Test that sync_data accepts resources parameter."""
        credentials = IntegrationCredentials(api_key="valid_api_key")

        result = await mock_plugin.sync_data(
            credentials,
            resources=["items"],
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_data_with_since(self, mock_plugin):
        """Test that sync_data accepts since parameter for incremental sync."""
        credentials = IntegrationCredentials(api_key="valid_api_key")
        since = datetime(2024, 1, 1)

        result = await mock_plugin.sync_data(credentials, since=since)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_data_fails_with_invalid_credentials(self, mock_plugin):
        """Test that sync_data fails without valid credentials."""
        credentials = IntegrationCredentials()  # No api_key

        result = await mock_plugin.sync_data(credentials)

        assert result.success is False
        assert "Missing API key" in result.errors


class TestMockOAuthPluginTestConnection:
    """Tests for test_connection with OAuth plugins."""

    @pytest.mark.asyncio
    async def test_oauth_connection_success(self, mock_oauth_plugin):
        """Test that OAuth test_connection succeeds with access token."""
        credentials = IntegrationCredentials(access_token="valid_access_token")

        result = await mock_oauth_plugin.test_connection(credentials)

        assert result.success is True
        assert result.message == "OAuth connection successful"
        assert result.api_version == "2.0"

    @pytest.mark.asyncio
    async def test_oauth_connection_fails_without_token(self, mock_oauth_plugin):
        """Test that OAuth test_connection fails without access token."""
        credentials = IntegrationCredentials()

        result = await mock_oauth_plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "NO_TOKEN"


class TestMockOAuthPluginSyncData:
    """Tests for sync_data with OAuth plugins."""

    @pytest.mark.asyncio
    async def test_oauth_sync_data_success(self, mock_oauth_plugin):
        """Test that OAuth sync_data succeeds."""
        credentials = IntegrationCredentials(access_token="valid_token")

        result = await mock_oauth_plugin.sync_data(credentials)

        assert result.success is True
        assert result.message == "OAuth sync completed"
        assert result.data_points_synced == 5


# ==============================================================================
# Simulated Platform Plugin Tests
# ==============================================================================


class GitHubPluginMock(IntegrationPlugin):
    """Mock GitHub plugin for testing."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        super().__init__(http_client)

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="github",
            name="GitHub",
            description="GitHub source control integration",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
            features=["repos", "issues", "pull_requests", "webhooks"],
            required_fields=[],
            oauth_config=OAuthConfig(
                authorize_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                scopes=["repo", "user"],
                client_id_env_var="GITHUB_CLIENT_ID",
                client_secret_env_var="GITHUB_CLIENT_SECRET",
            ),
            supports_webhooks=True,
            sync_resources=["repos", "issues", "pull_requests"],
        )

    async def test_connection(
        self, credentials: IntegrationCredentials
    ) -> ConnectionTestResult:
        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="No access token provided",
                error_code="NO_TOKEN",
            )

        try:
            response = await self.http_client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {credentials.access_token}"},
            )
            response.raise_for_status()
            user_data = response.json()

            return ConnectionTestResult(
                success=True,
                message="Connected to GitHub",
                latency_ms=100.0,
                api_version="v3",
                account_info={"login": user_data.get("login")},
                permissions=["repo", "user"],
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                error_code="API_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> SyncResult:
        if not credentials.access_token:
            return SyncResult(
                success=False,
                message="No access token",
                errors=["Missing access token"],
            )

        try:
            response = await self.http_client.get(
                "https://api.github.com/user/repos",
                headers={"Authorization": f"Bearer {credentials.access_token}"},
                params={"per_page": limit},
            )
            response.raise_for_status()
            repos = response.json()

            return SyncResult(
                success=True,
                message="Synced GitHub data",
                data_points_synced=len(repos),
                data={"repos": repos},
            )
        except Exception as e:
            return SyncResult(
                success=False,
                message=str(e),
                errors=[str(e)],
            )


class SlackPluginMock(IntegrationPlugin):
    """Mock Slack plugin for testing."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        super().__init__(http_client)

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="slack",
            name="Slack",
            description="Slack team communication integration",
            category=IntegrationCategory.CHAT,
            auth_type=AuthType.OAUTH2,
            features=["notifications", "channels", "messages"],
            required_fields=[],
            oauth_config=OAuthConfig(
                authorize_url="https://slack.com/oauth/v2/authorize",
                token_url="https://slack.com/api/oauth.v2.access",
                scopes=["chat:write", "channels:read"],
                client_id_env_var="SLACK_CLIENT_ID",
                client_secret_env_var="SLACK_CLIENT_SECRET",
            ),
            supports_webhooks=True,
            sync_resources=["channels", "users"],
        )

    async def test_connection(
        self, credentials: IntegrationCredentials
    ) -> ConnectionTestResult:
        if not credentials.access_token:
            return ConnectionTestResult(
                success=False,
                message="No access token provided",
                error_code="NO_TOKEN",
            )

        try:
            response = await self.http_client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {credentials.access_token}"},
            )
            data = response.json()

            if data.get("ok"):
                return ConnectionTestResult(
                    success=True,
                    message="Connected to Slack",
                    latency_ms=80.0,
                    account_info={
                        "team": data.get("team"),
                        "user": data.get("user"),
                    },
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=data.get("error", "Unknown error"),
                    error_code=data.get("error"),
                )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                error_code="API_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> SyncResult:
        if not credentials.access_token:
            return SyncResult(
                success=False,
                message="No access token",
                errors=["Missing access token"],
            )

        try:
            response = await self.http_client.get(
                "https://slack.com/api/conversations.list",
                headers={"Authorization": f"Bearer {credentials.access_token}"},
                params={"limit": limit},
            )
            data = response.json()

            if data.get("ok"):
                channels = data.get("channels", [])
                return SyncResult(
                    success=True,
                    message="Synced Slack data",
                    data_points_synced=len(channels),
                    data={"channels": channels},
                )
            else:
                return SyncResult(
                    success=False,
                    message=data.get("error", "Unknown error"),
                    errors=[data.get("error")],
                )
        except Exception as e:
            return SyncResult(
                success=False,
                message=str(e),
                errors=[str(e)],
            )


class SentryPluginMock(IntegrationPlugin):
    """Mock Sentry plugin for testing."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        super().__init__(http_client)

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="sentry",
            name="Sentry",
            description="Sentry error monitoring integration",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.API_KEY,
            features=["issues", "events", "alerts"],
            required_fields=[
                FieldDefinition(name="api_key", label="API Key", type="password"),
                FieldDefinition(name="org_slug", label="Organization Slug"),
            ],
            optional_fields=[
                FieldDefinition(name="project_slug", label="Project Slug", required=False),
            ],
            supports_webhooks=True,
            sync_resources=["issues", "events"],
        )

    async def test_connection(
        self, credentials: IntegrationCredentials
    ) -> ConnectionTestResult:
        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="No API key provided",
                error_code="NO_API_KEY",
            )

        try:
            response = await self.http_client.get(
                f"https://sentry.io/api/0/organizations/{credentials.org_slug}/",
                headers={"Authorization": f"Bearer {credentials.api_key}"},
            )
            response.raise_for_status()
            org_data = response.json()

            return ConnectionTestResult(
                success=True,
                message="Connected to Sentry",
                latency_ms=120.0,
                account_info={
                    "org_name": org_data.get("name"),
                    "org_slug": org_data.get("slug"),
                },
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=str(e),
                error_code="API_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> SyncResult:
        if not credentials.api_key or not credentials.org_slug:
            return SyncResult(
                success=False,
                message="Missing credentials",
                errors=["API key and org_slug required"],
            )

        try:
            url = f"https://sentry.io/api/0/organizations/{credentials.org_slug}/issues/"
            if credentials.project_slug:
                url = f"https://sentry.io/api/0/projects/{credentials.org_slug}/{credentials.project_slug}/issues/"

            response = await self.http_client.get(
                url,
                headers={"Authorization": f"Bearer {credentials.api_key}"},
                params={"limit": limit},
            )
            response.raise_for_status()
            issues = response.json()

            return SyncResult(
                success=True,
                message="Synced Sentry issues",
                data_points_synced=len(issues),
                data={"issues": issues},
            )
        except Exception as e:
            return SyncResult(
                success=False,
                message=str(e),
                errors=[str(e)],
            )


# ==============================================================================
# Platform Plugin Tests with Mock HTTP
# ==============================================================================


class TestGitHubPluginMock:
    """Tests for the mock GitHub plugin."""

    @pytest.fixture
    def github_plugin(self, mock_httpx_async_client):
        """Create a GitHub plugin with mock HTTP client."""
        return GitHubPluginMock(http_client=mock_httpx_async_client)

    @pytest.mark.asyncio
    async def test_connection_success(
        self, github_plugin, mock_httpx_async_client, github_api_responses
    ):
        """Test successful GitHub connection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = github_api_responses["user"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(access_token="ghp_valid_token")
        result = await github_plugin.test_connection(credentials)

        assert result.success is True
        assert result.account_info["login"] == "testuser"
        assert result.api_version == "v3"

    @pytest.mark.asyncio
    async def test_connection_no_token(self, github_plugin):
        """Test GitHub connection fails without token."""
        credentials = IntegrationCredentials()
        result = await github_plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "NO_TOKEN"

    @pytest.mark.asyncio
    async def test_connection_api_error(
        self, github_plugin, mock_httpx_async_client
    ):
        """Test GitHub connection handles API errors."""
        mock_httpx_async_client.get.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401),
        )

        credentials = IntegrationCredentials(access_token="invalid_token")
        result = await github_plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "API_ERROR"

    @pytest.mark.asyncio
    async def test_sync_data_success(
        self, github_plugin, mock_httpx_async_client, github_api_responses
    ):
        """Test successful GitHub data sync."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = github_api_responses["repos"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(access_token="ghp_valid_token")
        result = await github_plugin.sync_data(credentials)

        assert result.success is True
        assert "repos" in result.data
        assert result.data_points_synced == 1


class TestSlackPluginMock:
    """Tests for the mock Slack plugin."""

    @pytest.fixture
    def slack_plugin(self, mock_httpx_async_client):
        """Create a Slack plugin with mock HTTP client."""
        return SlackPluginMock(http_client=mock_httpx_async_client)

    @pytest.mark.asyncio
    async def test_connection_success(
        self, slack_plugin, mock_httpx_async_client, slack_api_responses
    ):
        """Test successful Slack connection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = slack_api_responses["auth.test"]
        mock_httpx_async_client.post.return_value = mock_response

        credentials = IntegrationCredentials(access_token="xoxb-valid-token")
        result = await slack_plugin.test_connection(credentials)

        assert result.success is True
        assert result.account_info["team"] == "Test Team"

    @pytest.mark.asyncio
    async def test_connection_invalid_auth(
        self, slack_plugin, mock_httpx_async_client, slack_api_responses
    ):
        """Test Slack connection with invalid auth."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = slack_api_responses["error_response"]
        mock_httpx_async_client.post.return_value = mock_response

        credentials = IntegrationCredentials(access_token="invalid_token")
        result = await slack_plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "invalid_auth"

    @pytest.mark.asyncio
    async def test_sync_data_success(
        self, slack_plugin, mock_httpx_async_client, slack_api_responses
    ):
        """Test successful Slack data sync."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = slack_api_responses["conversations.list"]
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(access_token="xoxb-valid-token")
        result = await slack_plugin.sync_data(credentials)

        assert result.success is True
        assert "channels" in result.data


class TestSentryPluginMock:
    """Tests for the mock Sentry plugin."""

    @pytest.fixture
    def sentry_plugin(self, mock_httpx_async_client):
        """Create a Sentry plugin with mock HTTP client."""
        return SentryPluginMock(http_client=mock_httpx_async_client)

    @pytest.mark.asyncio
    async def test_connection_success(
        self, sentry_plugin, mock_httpx_async_client, sentry_api_responses
    ):
        """Test successful Sentry connection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sentry_api_responses["organization"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(
            api_key="sentry_api_key",
            org_slug="test-org",
        )
        result = await sentry_plugin.test_connection(credentials)

        assert result.success is True
        assert result.account_info["org_slug"] == "test-org"

    @pytest.mark.asyncio
    async def test_connection_no_api_key(self, sentry_plugin):
        """Test Sentry connection fails without API key."""
        credentials = IntegrationCredentials(org_slug="test-org")
        result = await sentry_plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "NO_API_KEY"

    @pytest.mark.asyncio
    async def test_sync_data_success(
        self, sentry_plugin, mock_httpx_async_client, sentry_api_responses
    ):
        """Test successful Sentry data sync."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sentry_api_responses["issues"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(
            api_key="sentry_api_key",
            org_slug="test-org",
        )
        result = await sentry_plugin.sync_data(credentials)

        assert result.success is True
        assert "issues" in result.data

    @pytest.mark.asyncio
    async def test_sync_data_with_project(
        self, sentry_plugin, mock_httpx_async_client, sentry_api_responses
    ):
        """Test Sentry sync with project slug."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sentry_api_responses["issues"]
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(
            api_key="sentry_api_key",
            org_slug="test-org",
            project_slug="test-project",
        )
        result = await sentry_plugin.sync_data(credentials)

        assert result.success is True

        # Verify the project-specific URL was called
        call_args = mock_httpx_async_client.get.call_args
        assert "test-project" in str(call_args)


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestPluginErrorHandling:
    """Tests for plugin error handling."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, mock_httpx_async_client):
        """Test that plugins handle network timeouts gracefully."""
        plugin = GitHubPluginMock(http_client=mock_httpx_async_client)
        mock_httpx_async_client.get.side_effect = httpx.TimeoutException("Timeout")

        credentials = IntegrationCredentials(access_token="valid_token")
        result = await plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "API_ERROR"

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_httpx_async_client):
        """Test that plugins handle connection errors gracefully."""
        plugin = SlackPluginMock(http_client=mock_httpx_async_client)
        mock_httpx_async_client.post.side_effect = httpx.ConnectError("Connection refused")

        credentials = IntegrationCredentials(access_token="valid_token")
        result = await plugin.test_connection(credentials)

        assert result.success is False
        assert "Connection refused" in result.message

    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self, mock_httpx_async_client):
        """Test that plugins handle malformed JSON responses."""
        plugin = SentryPluginMock(http_client=mock_httpx_async_client)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(
            api_key="api_key",
            org_slug="test-org",
        )
        result = await plugin.test_connection(credentials)

        assert result.success is False
        assert result.error_code == "API_ERROR"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_httpx_async_client):
        """Test that plugins handle rate limiting."""
        plugin = GitHubPluginMock(http_client=mock_httpx_async_client)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_async_client.get.return_value = mock_response

        credentials = IntegrationCredentials(access_token="valid_token")
        result = await plugin.test_connection(credentials)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_server_error_handling(self, mock_httpx_async_client):
        """Test that plugins handle server errors (5xx)."""
        plugin = SlackPluginMock(http_client=mock_httpx_async_client)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Server error")
        mock_httpx_async_client.post.return_value = mock_response

        credentials = IntegrationCredentials(access_token="valid_token")
        result = await plugin.test_connection(credentials)

        assert result.success is False


# ==============================================================================
# Credential Validation Tests
# ==============================================================================


class TestPluginCredentialValidation:
    """Tests for credential validation across different plugins."""

    def test_github_metadata(self, mock_httpx_async_client):
        """Test GitHub plugin metadata configuration."""
        plugin = GitHubPluginMock(http_client=mock_httpx_async_client)
        metadata = plugin.metadata

        assert metadata.slug == "github"
        assert metadata.category == IntegrationCategory.SOURCE_CODE
        assert metadata.auth_type == AuthType.OAUTH2
        assert metadata.oauth_config is not None
        assert "repo" in metadata.oauth_config.scopes

    def test_slack_metadata(self, mock_httpx_async_client):
        """Test Slack plugin metadata configuration."""
        plugin = SlackPluginMock(http_client=mock_httpx_async_client)
        metadata = plugin.metadata

        assert metadata.slug == "slack"
        assert metadata.category == IntegrationCategory.CHAT
        assert metadata.auth_type == AuthType.OAUTH2
        assert metadata.supports_webhooks is True

    def test_sentry_metadata(self, mock_httpx_async_client):
        """Test Sentry plugin metadata configuration."""
        plugin = SentryPluginMock(http_client=mock_httpx_async_client)
        metadata = plugin.metadata

        assert metadata.slug == "sentry"
        assert metadata.category == IntegrationCategory.OBSERVABILITY
        assert metadata.auth_type == AuthType.API_KEY
        assert len(metadata.required_fields) == 2

    def test_sentry_credential_validation(self, mock_httpx_async_client):
        """Test Sentry plugin credential validation."""
        plugin = SentryPluginMock(http_client=mock_httpx_async_client)

        # Valid credentials
        valid_creds = IntegrationCredentials(
            api_key="api_key_123",
            org_slug="test-org",
        )
        is_valid, missing = plugin.validate_credentials(valid_creds)
        assert is_valid is True

        # Missing org_slug
        invalid_creds = IntegrationCredentials(api_key="api_key_123")
        is_valid, missing = plugin.validate_credentials(invalid_creds)
        assert is_valid is False
        assert "org_slug" in missing


# ==============================================================================
# Integration Tests for Multiple Plugins
# ==============================================================================


class TestMultiplePluginInteraction:
    """Tests for scenarios involving multiple plugins."""

    @pytest.mark.asyncio
    async def test_different_auth_types(self, mock_httpx_async_client):
        """Test plugins with different auth types work correctly."""
        oauth_plugin = GitHubPluginMock(http_client=mock_httpx_async_client)
        api_key_plugin = SentryPluginMock(http_client=mock_httpx_async_client)

        # Setup mock responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"login": "user", "slug": "org"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response

        # Test OAuth plugin
        oauth_creds = IntegrationCredentials(access_token="oauth_token")
        oauth_result = await oauth_plugin.test_connection(oauth_creds)
        assert oauth_result.success is True

        # Test API key plugin
        api_key_creds = IntegrationCredentials(
            api_key="api_key",
            org_slug="test-org",
        )
        api_result = await api_key_plugin.test_connection(api_key_creds)
        assert api_result.success is True

    @pytest.mark.asyncio
    async def test_plugin_isolation(self, mock_httpx_async_client):
        """Test that plugins are isolated from each other."""
        plugin1 = GitHubPluginMock(http_client=mock_httpx_async_client)
        plugin2 = SlackPluginMock(http_client=mock_httpx_async_client)

        # Verify metadata is different
        assert plugin1.metadata.slug != plugin2.metadata.slug
        assert plugin1.metadata.category != plugin2.metadata.category

        # Verify both can use shared HTTP client
        assert plugin1.http_client is plugin2.http_client
