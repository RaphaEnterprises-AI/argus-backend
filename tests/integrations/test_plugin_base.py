"""Tests for the base plugin class and related data structures.

This module tests:
- IntegrationCredentials dataclass
- Credential validation
- OAuth URL generation
- Webhook signature generation
- IntegrationPlugin base class methods
"""

import hashlib
import hmac
import json
import os
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


class TestIntegrationCredentials:
    """Tests for the IntegrationCredentials dataclass."""

    def test_empty_credentials(self):
        """Test creating empty credentials."""
        creds = IntegrationCredentials()

        assert creds.api_key is None
        assert creds.api_secret is None
        assert creds.access_token is None
        assert creds.refresh_token is None
        assert creds.extra == {}

    def test_api_key_credentials(self):
        """Test creating credentials with API key."""
        creds = IntegrationCredentials(api_key="test_api_key_12345")

        assert creds.api_key == "test_api_key_12345"
        assert creds.api_secret is None

    def test_oauth_credentials(self):
        """Test creating credentials with OAuth tokens."""
        expires_at = datetime.utcnow() + timedelta(hours=1)
        creds = IntegrationCredentials(
            access_token="access_token_123",
            refresh_token="refresh_token_456",
            client_id="client_id_789",
            client_secret="client_secret_abc",
            token_expires_at=expires_at,
        )

        assert creds.access_token == "access_token_123"
        assert creds.refresh_token == "refresh_token_456"
        assert creds.client_id == "client_id_789"
        assert creds.client_secret == "client_secret_abc"
        assert creds.token_expires_at == expires_at

    def test_platform_specific_credentials(self):
        """Test credentials with platform-specific fields."""
        creds = IntegrationCredentials(
            api_key="api_key_123",
            instance_url="https://jira.example.com",
            org_slug="my-org",
            project_slug="my-project",
            workspace_id="workspace_123",
            team_id="team_456",
            site="datadoghq.eu",
        )

        assert creds.instance_url == "https://jira.example.com"
        assert creds.org_slug == "my-org"
        assert creds.project_slug == "my-project"
        assert creds.workspace_id == "workspace_123"
        assert creds.team_id == "team_456"
        assert creds.site == "datadoghq.eu"

    def test_basic_auth_credentials(self):
        """Test credentials with basic auth fields."""
        creds = IntegrationCredentials(
            username="testuser",
            password="testpass",
            email="test@example.com",
        )

        assert creds.username == "testuser"
        assert creds.password == "testpass"
        assert creds.email == "test@example.com"

    def test_webhook_credentials(self):
        """Test credentials with webhook fields."""
        creds = IntegrationCredentials(
            webhook_url="https://hooks.example.com/webhook",
            webhook_secret="secret_key_123",
        )

        assert creds.webhook_url == "https://hooks.example.com/webhook"
        assert creds.webhook_secret == "secret_key_123"

    def test_extra_fields(self):
        """Test credentials with extra custom fields."""
        creds = IntegrationCredentials(
            api_key="api_key_123",
            extra={
                "custom_field": "custom_value",
                "another_field": 12345,
            },
        )

        assert creds.extra["custom_field"] == "custom_value"
        assert creds.extra["another_field"] == 12345

    def test_credentials_immutability(self):
        """Test that credentials are mutable (dataclass without frozen)."""
        creds = IntegrationCredentials(api_key="original_key")
        creds.api_key = "new_key"

        assert creds.api_key == "new_key"


class TestConnectionTestResult:
    """Tests for the ConnectionTestResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful connection result."""
        result = ConnectionTestResult(
            success=True,
            message="Connection successful",
            latency_ms=45.5,
            api_version="2.0",
            permissions=["read", "write"],
            account_info={"name": "Test Account"},
        )

        assert result.success is True
        assert result.message == "Connection successful"
        assert result.latency_ms == 45.5
        assert result.api_version == "2.0"
        assert "read" in result.permissions
        assert result.account_info["name"] == "Test Account"
        assert result.error_code is None

    def test_failed_result(self):
        """Test creating a failed connection result."""
        result = ConnectionTestResult(
            success=False,
            message="Authentication failed",
            error_code="AUTH_ERROR",
            error_details={"reason": "Invalid API key"},
        )

        assert result.success is False
        assert result.message == "Authentication failed"
        assert result.error_code == "AUTH_ERROR"
        assert result.error_details["reason"] == "Invalid API key"


class TestSyncResult:
    """Tests for the SyncResult dataclass."""

    def test_successful_sync(self):
        """Test creating a successful sync result."""
        started = datetime.utcnow()
        completed = datetime.utcnow()

        result = SyncResult(
            success=True,
            message="Sync completed",
            data_points_synced=100,
            sync_id="sync_123",
            started_at=started,
            completed_at=completed,
            data={"items": [{"id": 1}, {"id": 2}]},
        )

        assert result.success is True
        assert result.data_points_synced == 100
        assert result.sync_id == "sync_123"
        assert len(result.data["items"]) == 2

    def test_failed_sync_with_errors(self):
        """Test creating a failed sync result with errors."""
        result = SyncResult(
            success=False,
            message="Sync failed",
            errors=["Network timeout", "Rate limit exceeded"],
            warnings=["Some items were skipped"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


class TestFieldDefinition:
    """Tests for the FieldDefinition dataclass."""

    def test_basic_field(self):
        """Test creating a basic field definition."""
        field = FieldDefinition(
            name="api_key",
            label="API Key",
            type="password",
            required=True,
        )

        assert field.name == "api_key"
        assert field.label == "API Key"
        assert field.type == "password"
        assert field.required is True

    def test_select_field(self):
        """Test creating a select field with options."""
        field = FieldDefinition(
            name="region",
            label="Region",
            type="select",
            required=True,
            options=[
                {"value": "us", "label": "United States"},
                {"value": "eu", "label": "Europe"},
            ],
        )

        assert field.type == "select"
        assert len(field.options) == 2
        assert field.options[0]["value"] == "us"


class TestIntegrationMetadata:
    """Tests for the IntegrationMetadata dataclass."""

    def test_basic_metadata(self):
        """Test creating basic integration metadata."""
        metadata = IntegrationMetadata(
            slug="github",
            name="GitHub",
            description="GitHub integration",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
        )

        assert metadata.slug == "github"
        assert metadata.name == "GitHub"
        assert metadata.category == IntegrationCategory.SOURCE_CODE
        assert metadata.auth_type == AuthType.OAUTH2

    def test_metadata_with_oauth_config(self):
        """Test metadata with OAuth configuration."""
        oauth_config = OAuthConfig(
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            scopes=["repo", "user"],
            client_id_env_var="GITHUB_CLIENT_ID",
            client_secret_env_var="GITHUB_CLIENT_SECRET",
        )

        metadata = IntegrationMetadata(
            slug="github",
            name="GitHub",
            description="GitHub integration",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
            oauth_config=oauth_config,
        )

        assert metadata.oauth_config is not None
        assert metadata.oauth_config.scopes == ["repo", "user"]


class TestOAuthConfig:
    """Tests for the OAuthConfig dataclass."""

    def test_basic_oauth_config(self):
        """Test creating basic OAuth config."""
        config = OAuthConfig(
            authorize_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
            scopes=["read", "write"],
            client_id_env_var="CLIENT_ID",
            client_secret_env_var="CLIENT_SECRET",
        )

        assert config.authorize_url == "https://oauth.example.com/authorize"
        assert config.token_url == "https://oauth.example.com/token"
        assert config.scopes == ["read", "write"]
        assert config.pkce_required is False
        assert config.state_required is True

    def test_oauth_config_with_pkce(self):
        """Test OAuth config with PKCE enabled."""
        config = OAuthConfig(
            authorize_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
            scopes=["read"],
            client_id_env_var="CLIENT_ID",
            client_secret_env_var="CLIENT_SECRET",
            pkce_required=True,
        )

        assert config.pkce_required is True


class TestCredentialValidation:
    """Tests for credential validation in IntegrationPlugin."""

    @pytest.fixture
    def plugin_with_required_fields(self, mock_httpx_async_client):
        """Create a plugin with specific required fields."""
        from tests.integrations.conftest import MockPlugin

        return MockPlugin(
            required_fields=[
                FieldDefinition(name="api_key", label="API Key", required=True),
                FieldDefinition(name="instance_url", label="Instance URL", required=True),
            ],
            http_client=mock_httpx_async_client,
        )

    def test_validate_complete_credentials(self, plugin_with_required_fields):
        """Test validation passes with all required fields."""
        creds = IntegrationCredentials(
            api_key="test_key",
            instance_url="https://example.com",
        )

        is_valid, missing = plugin_with_required_fields.validate_credentials(creds)

        assert is_valid is True
        assert len(missing) == 0

    def test_validate_missing_credentials(self, plugin_with_required_fields):
        """Test validation fails with missing required fields."""
        creds = IntegrationCredentials(api_key="test_key")

        is_valid, missing = plugin_with_required_fields.validate_credentials(creds)

        assert is_valid is False
        assert "instance_url" in missing

    def test_validate_empty_credentials(self, plugin_with_required_fields):
        """Test validation fails with empty credentials."""
        creds = IntegrationCredentials()

        is_valid, missing = plugin_with_required_fields.validate_credentials(creds)

        assert is_valid is False
        assert "api_key" in missing
        assert "instance_url" in missing

    def test_validate_credentials_from_extra(self, mock_httpx_async_client):
        """Test validation checks extra field for custom fields."""
        from tests.integrations.conftest import MockPlugin

        plugin = MockPlugin(
            required_fields=[
                FieldDefinition(name="custom_field", label="Custom Field", required=True),
            ],
            http_client=mock_httpx_async_client,
        )

        creds = IntegrationCredentials(extra={"custom_field": "custom_value"})

        is_valid, missing = plugin.validate_credentials(creds)

        assert is_valid is True


class TestOAuthUrlGeneration:
    """Tests for OAuth URL generation in IntegrationPlugin."""

    @pytest.fixture
    def oauth_plugin(self, mock_httpx_async_client):
        """Create an OAuth plugin for testing."""
        from tests.integrations.conftest import MockOAuthPlugin

        return MockOAuthPlugin(http_client=mock_httpx_async_client)

    def test_get_oauth_url_with_env_vars(self, oauth_plugin, monkeypatch):
        """Test OAuth URL generation with environment variables."""
        monkeypatch.setenv("MOCK_CLIENT_ID", "test_client_id_123")

        url = oauth_plugin.get_oauth_url(
            redirect_uri="https://app.example.com/callback",
            state="random_state_string",
        )

        assert url is not None
        assert "https://oauth.example.com/authorize" in url
        assert "client_id=test_client_id_123" in url
        assert "redirect_uri=https%3A%2F%2Fapp.example.com%2Fcallback" in url
        assert "state=random_state_string" in url
        assert "response_type=code" in url
        assert "scope=read+write" in url

    def test_get_oauth_url_with_custom_scopes(self, oauth_plugin, monkeypatch):
        """Test OAuth URL generation with custom scopes."""
        monkeypatch.setenv("MOCK_CLIENT_ID", "test_client_id")

        url = oauth_plugin.get_oauth_url(
            redirect_uri="https://app.example.com/callback",
            state="state_123",
            scopes=["admin", "repo"],
        )

        assert url is not None
        assert "scope=admin+repo" in url

    def test_get_oauth_url_non_oauth_plugin(self, mock_plugin):
        """Test that non-OAuth plugins return None for OAuth URL."""
        url = mock_plugin.get_oauth_url(
            redirect_uri="https://app.example.com/callback",
            state="state_123",
        )

        assert url is None


class TestOAuthTokenExchange:
    """Tests for OAuth token exchange in IntegrationPlugin."""

    @pytest.fixture
    def oauth_plugin(self, mock_httpx_async_client):
        """Create an OAuth plugin for testing."""
        from tests.integrations.conftest import MockOAuthPlugin

        return MockOAuthPlugin(http_client=mock_httpx_async_client)

    @pytest.mark.asyncio
    async def test_exchange_oauth_code_success(
        self, oauth_plugin, mock_httpx_async_client, monkeypatch
    ):
        """Test successful OAuth code exchange."""
        monkeypatch.setenv("MOCK_CLIENT_ID", "client_id_123")
        monkeypatch.setenv("MOCK_CLIENT_SECRET", "client_secret_456")

        # Mock successful token response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.post.return_value = mock_response

        result = await oauth_plugin.exchange_oauth_code(
            code="auth_code_123",
            redirect_uri="https://app.example.com/callback",
        )

        assert result is not None
        assert result.access_token == "new_access_token"
        assert result.refresh_token == "new_refresh_token"
        assert result.token_expires_at is not None

    @pytest.mark.asyncio
    async def test_exchange_oauth_code_failure(
        self, oauth_plugin, mock_httpx_async_client, monkeypatch
    ):
        """Test failed OAuth code exchange."""
        monkeypatch.setenv("MOCK_CLIENT_ID", "client_id_123")
        monkeypatch.setenv("MOCK_CLIENT_SECRET", "client_secret_456")

        # Mock failed token response
        mock_httpx_async_client.post.side_effect = Exception("Token exchange failed")

        result = await oauth_plugin.exchange_oauth_code(
            code="invalid_code",
            redirect_uri="https://app.example.com/callback",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_exchange_oauth_code_non_oauth_plugin(self, mock_plugin):
        """Test that non-OAuth plugins return None for token exchange."""
        result = await mock_plugin.exchange_oauth_code(
            code="code_123",
            redirect_uri="https://app.example.com/callback",
        )

        assert result is None


class TestOAuthTokenRefresh:
    """Tests for OAuth token refresh in IntegrationPlugin."""

    @pytest.fixture
    def oauth_plugin(self, mock_httpx_async_client):
        """Create an OAuth plugin for testing."""
        from tests.integrations.conftest import MockOAuthPlugin

        return MockOAuthPlugin(http_client=mock_httpx_async_client)

    @pytest.mark.asyncio
    async def test_refresh_oauth_token_success(
        self, oauth_plugin, mock_httpx_async_client, monkeypatch
    ):
        """Test successful OAuth token refresh."""
        monkeypatch.setenv("MOCK_CLIENT_ID", "client_id_123")
        monkeypatch.setenv("MOCK_CLIENT_SECRET", "client_secret_456")

        # Mock successful refresh response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.post.return_value = mock_response

        credentials = IntegrationCredentials(
            access_token="old_access_token",
            refresh_token="refresh_token_123",
        )

        result = await oauth_plugin.refresh_oauth_token(credentials)

        assert result is not None
        assert result.access_token == "refreshed_access_token"
        # Refresh token should be preserved if not returned
        assert result.refresh_token == "refresh_token_123"

    @pytest.mark.asyncio
    async def test_refresh_oauth_token_no_refresh_token(self, oauth_plugin):
        """Test refresh fails when no refresh token is provided."""
        credentials = IntegrationCredentials(access_token="access_token_only")

        result = await oauth_plugin.refresh_oauth_token(credentials)

        assert result is None


class TestWebhookSignatureGeneration:
    """Tests for webhook signature generation in IntegrationPlugin."""

    @pytest.mark.asyncio
    async def test_send_webhook_with_signature(
        self, mock_plugin, mock_httpx_async_client
    ):
        """Test webhook sending with signature generation."""
        credentials = IntegrationCredentials(
            webhook_url="https://hooks.example.com/webhook",
            webhook_secret="secret_123",
        )

        payload = {"event": "test", "data": {"id": 1}}

        # Mock successful webhook response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_async_client.post.return_value = mock_response

        result = await mock_plugin.send_webhook(
            credentials=credentials,
            event_type="test_event",
            payload=payload,
        )

        assert result is True

        # Verify the signature was included in headers
        call_kwargs = mock_httpx_async_client.post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "X-Signature-256" in headers
        assert headers["X-Signature-256"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_send_webhook_without_secret(
        self, mock_plugin, mock_httpx_async_client
    ):
        """Test webhook sending without signature when no secret is provided."""
        credentials = IntegrationCredentials(
            webhook_url="https://hooks.example.com/webhook",
        )

        payload = {"event": "test"}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_async_client.post.return_value = mock_response

        result = await mock_plugin.send_webhook(
            credentials=credentials,
            event_type="test_event",
            payload=payload,
        )

        assert result is True

        # Verify no signature header when no secret
        call_kwargs = mock_httpx_async_client.post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "X-Signature-256" not in headers

    @pytest.mark.asyncio
    async def test_send_webhook_no_url(self, mock_plugin):
        """Test webhook fails when no URL is provided."""
        credentials = IntegrationCredentials()

        result = await mock_plugin.send_webhook(
            credentials=credentials,
            event_type="test_event",
            payload={"test": "data"},
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_failure(
        self, mock_plugin, mock_httpx_async_client
    ):
        """Test webhook returns False on HTTP error."""
        credentials = IntegrationCredentials(
            webhook_url="https://hooks.example.com/webhook",
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_httpx_async_client.post.return_value = mock_response

        result = await mock_plugin.send_webhook(
            credentials=credentials,
            event_type="test_event",
            payload={"test": "data"},
        )

        assert result is False

    def test_webhook_signature_correctness(self, webhook_signature_generator):
        """Test that webhook signature is correctly computed."""
        secret = "test_secret"
        payload = {"event": "test", "data": {"id": 123}}

        signature = webhook_signature_generator(secret, payload)

        # Verify manually
        body = json.dumps(payload)
        expected = hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert signature == f"sha256={expected}"


class TestPluginContextManager:
    """Tests for the plugin async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self):
        """Test that entering context creates HTTP client."""
        from tests.integrations.conftest import MockPlugin

        plugin = MockPlugin()

        async with plugin:
            assert plugin._http_client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_owned_client(self):
        """Test that exiting context closes owned HTTP client."""
        from tests.integrations.conftest import MockPlugin

        plugin = MockPlugin()
        plugin._owns_client = True

        # Create a mock client to track closure
        mock_client = AsyncMock()
        plugin._http_client = mock_client

        async with plugin:
            pass

        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_keeps_external_client(self):
        """Test that context manager doesn't close externally provided client."""
        from tests.integrations.conftest import MockPlugin

        external_client = AsyncMock(spec=httpx.AsyncClient)
        plugin = MockPlugin(http_client=external_client)

        async with plugin:
            pass

        # External client should not be closed
        external_client.aclose.assert_not_called()


class TestAuthTypeEnum:
    """Tests for the AuthType enum."""

    def test_auth_type_values(self):
        """Test that AuthType enum has expected values."""
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.OAUTH2.value == "oauth2"
        assert AuthType.BEARER_TOKEN.value == "bearer_token"
        assert AuthType.BASIC_AUTH.value == "basic_auth"
        assert AuthType.WEBHOOK.value == "webhook"
        assert AuthType.SERVICE_ACCOUNT.value == "service_account"
        assert AuthType.APP_PASSWORD.value == "app_password"
        assert AuthType.MCP.value == "mcp"


class TestIntegrationCategoryEnum:
    """Tests for the IntegrationCategory enum."""

    def test_category_values(self):
        """Test that IntegrationCategory enum has expected values."""
        assert IntegrationCategory.SOURCE_CODE.value == "source_code"
        assert IntegrationCategory.ISSUE_TRACKING.value == "issue_tracking"
        assert IntegrationCategory.CHAT.value == "chat"
        assert IntegrationCategory.CICD.value == "cicd"
        assert IntegrationCategory.DEPLOYMENT.value == "deployment"
        assert IntegrationCategory.OBSERVABILITY.value == "observability"
        assert IntegrationCategory.SESSION_REPLAY.value == "session_replay"
        assert IntegrationCategory.ANALYTICS.value == "analytics"
        assert IntegrationCategory.INCIDENT_MANAGEMENT.value == "incident_management"
        assert IntegrationCategory.FEATURE_FLAGS.value == "feature_flags"
        assert IntegrationCategory.TESTING.value == "testing"
        assert IntegrationCategory.WEBHOOKS.value == "webhooks"
        assert IntegrationCategory.AI_AGENTS.value == "ai_agents"
        assert IntegrationCategory.DATABASE.value == "database"
