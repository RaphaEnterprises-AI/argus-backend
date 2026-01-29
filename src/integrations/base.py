"""Base class for all integration plugins.

This module defines the abstract interface that all integration plugins must implement.
It provides a consistent API for connecting, testing, syncing, and managing integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx


class AuthType(str, Enum):
    """Authentication types supported by integrations."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    WEBHOOK = "webhook"
    SERVICE_ACCOUNT = "service_account"
    APP_PASSWORD = "app_password"
    MCP = "mcp"  # Model Context Protocol


class IntegrationCategory(str, Enum):
    """Categories of integrations."""
    SOURCE_CODE = "source_code"
    ISSUE_TRACKING = "issue_tracking"
    CHAT = "chat"
    CICD = "cicd"
    DEPLOYMENT = "deployment"
    OBSERVABILITY = "observability"
    SESSION_REPLAY = "session_replay"
    ANALYTICS = "analytics"
    INCIDENT_MANAGEMENT = "incident_management"
    FEATURE_FLAGS = "feature_flags"
    TESTING = "testing"
    WEBHOOKS = "webhooks"
    AI_AGENTS = "ai_agents"
    DATABASE = "database"


@dataclass
class IntegrationCredentials:
    """Credentials for an integration connection."""
    # Common fields
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None

    # OAuth specific
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    oauth_code: Optional[str] = None
    oauth_state: Optional[str] = None
    token_expires_at: Optional[datetime] = None

    # Platform specific
    instance_url: Optional[str] = None  # Jira, GitLab self-hosted
    org_slug: Optional[str] = None  # Sentry, GitHub org
    project_slug: Optional[str] = None  # Sentry project
    workspace_id: Optional[str] = None  # Slack, Notion
    team_id: Optional[str] = None  # Linear, Vercel
    site: Optional[str] = None  # Datadog site (datadoghq.com, datadoghq.eu)

    # Basic auth
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None

    # Webhook
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

    # Additional config
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionTestResult:
    """Result of testing an integration connection."""
    success: bool
    message: str
    latency_ms: Optional[float] = None
    api_version: Optional[str] = None
    permissions: list[str] = field(default_factory=list)
    account_info: dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None


@dataclass
class SyncResult:
    """Result of syncing data from an integration."""
    success: bool
    message: str
    data_points_synced: int = 0
    sync_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthConfig:
    """OAuth configuration for an integration."""
    authorize_url: str
    token_url: str
    scopes: list[str]
    client_id_env_var: str  # Environment variable name for client ID
    client_secret_env_var: str  # Environment variable name for client secret
    redirect_uri_path: str = "/api/v1/integrations/oauth/callback"
    pkce_required: bool = False
    state_required: bool = True


@dataclass
class FieldDefinition:
    """Definition of a credential field."""
    name: str
    label: str
    type: str = "text"  # text, password, url, email, select
    required: bool = True
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    options: list[dict[str, str]] = field(default_factory=list)  # For select type


@dataclass
class IntegrationMetadata:
    """Metadata about an integration plugin."""
    slug: str
    name: str
    description: str
    category: IntegrationCategory
    auth_type: AuthType
    icon_url: Optional[str] = None
    docs_url: Optional[str] = None
    website_url: Optional[str] = None
    features: list[str] = field(default_factory=list)
    required_fields: list[FieldDefinition] = field(default_factory=list)
    optional_fields: list[FieldDefinition] = field(default_factory=list)
    oauth_config: Optional[OAuthConfig] = None
    supports_webhooks: bool = False
    supports_sync: bool = True
    sync_resources: list[str] = field(default_factory=list)  # What can be synced


class IntegrationPlugin(ABC):
    """Abstract base class for all integration plugins.

    All integration plugins must inherit from this class and implement
    the required abstract methods.

    Example:
        class SlackPlugin(IntegrationPlugin):
            @property
            def metadata(self) -> IntegrationMetadata:
                return IntegrationMetadata(
                    slug="slack",
                    name="Slack",
                    description="Team communication and notifications",
                    category=IntegrationCategory.CHAT,
                    auth_type=AuthType.OAUTH2,
                    features=["notifications", "channels", "webhooks"],
                )

            async def test_connection(self, credentials: IntegrationCredentials) -> ConnectionTestResult:
                # Test the Slack connection
                ...

            async def sync_data(self, credentials: IntegrationCredentials, ...) -> SyncResult:
                # Sync data from Slack
                ...
    """

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """Initialize the plugin.

        Args:
            http_client: Optional HTTP client for making API requests.
                        If not provided, a new client will be created.
        """
        self._http_client = http_client
        self._owns_client = http_client is None

    async def __aenter__(self):
        """Async context manager entry."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owns_client and self._http_client:
            await self._http_client.aclose()

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, creating one if needed."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    @property
    @abstractmethod
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about this integration.

        Returns:
            IntegrationMetadata describing this integration's capabilities.
        """
        pass

    @abstractmethod
    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test if the connection to the integration works.

        Args:
            credentials: The credentials to test.

        Returns:
            ConnectionTestResult with success status and details.
        """
        pass

    @abstractmethod
    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from the integration.

        Args:
            credentials: The credentials to use.
            resources: Optional list of specific resources to sync.
                      If None, syncs all available resources.
            since: Optional datetime to sync data since (incremental sync).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        pass

    def get_oauth_url(
        self,
        redirect_uri: str,
        state: str,
        scopes: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Get the OAuth authorization URL for this integration.

        Args:
            redirect_uri: The URI to redirect to after authorization.
            state: State parameter for CSRF protection.
            scopes: Optional list of scopes to request.

        Returns:
            The OAuth authorization URL, or None if not an OAuth integration.
        """
        if self.metadata.auth_type != AuthType.OAUTH2:
            return None

        oauth_config = self.metadata.oauth_config
        if not oauth_config:
            return None

        import os
        from urllib.parse import urlencode

        params = {
            "client_id": os.environ.get(oauth_config.client_id_env_var, ""),
            "redirect_uri": redirect_uri,
            "state": state,
            "response_type": "code",
        }

        requested_scopes = scopes or oauth_config.scopes
        if requested_scopes:
            params["scope"] = " ".join(requested_scopes)

        return f"{oauth_config.authorize_url}?{urlencode(params)}"

    async def exchange_oauth_code(
        self,
        code: str,
        redirect_uri: str,
    ) -> Optional[IntegrationCredentials]:
        """Exchange an OAuth authorization code for tokens.

        Args:
            code: The authorization code from OAuth callback.
            redirect_uri: The redirect URI used in the authorization request.

        Returns:
            IntegrationCredentials with access token, or None if exchange failed.
        """
        if self.metadata.auth_type != AuthType.OAUTH2:
            return None

        oauth_config = self.metadata.oauth_config
        if not oauth_config:
            return None

        import os

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": os.environ.get(oauth_config.client_id_env_var, ""),
            "client_secret": os.environ.get(oauth_config.client_secret_env_var, ""),
        }

        try:
            response = await self.http_client.post(
                oauth_config.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

            expires_at = None
            if "expires_in" in token_data:
                from datetime import timedelta
                expires_at = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])

            return IntegrationCredentials(
                access_token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
                token_expires_at=expires_at,
            )
        except Exception:
            return None

    async def refresh_oauth_token(
        self,
        credentials: IntegrationCredentials,
    ) -> Optional[IntegrationCredentials]:
        """Refresh an expired OAuth token.

        Args:
            credentials: Credentials containing the refresh token.

        Returns:
            New credentials with fresh access token, or None if refresh failed.
        """
        if not credentials.refresh_token:
            return None

        oauth_config = self.metadata.oauth_config
        if not oauth_config:
            return None

        import os

        data = {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": os.environ.get(oauth_config.client_id_env_var, ""),
            "client_secret": os.environ.get(oauth_config.client_secret_env_var, ""),
        }

        try:
            response = await self.http_client.post(
                oauth_config.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

            expires_at = None
            if "expires_in" in token_data:
                from datetime import timedelta
                expires_at = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])

            return IntegrationCredentials(
                access_token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token", credentials.refresh_token),
                token_expires_at=expires_at,
            )
        except Exception:
            return None

    async def send_webhook(
        self,
        credentials: IntegrationCredentials,
        event_type: str,
        payload: dict[str, Any],
    ) -> bool:
        """Send a webhook event to the integration.

        Args:
            credentials: Credentials containing webhook URL.
            event_type: Type of event being sent.
            payload: Event payload data.

        Returns:
            True if webhook was sent successfully.
        """
        if not credentials.webhook_url:
            return False

        try:
            headers = {"Content-Type": "application/json"}

            # Add signature if webhook secret is provided
            if credentials.webhook_secret:
                import hashlib
                import hmac
                import json

                body = json.dumps(payload)
                signature = hmac.new(
                    credentials.webhook_secret.encode(),
                    body.encode(),
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Signature-256"] = f"sha256={signature}"

            response = await self.http_client.post(
                credentials.webhook_url,
                json={"event": event_type, "data": payload},
                headers=headers,
            )
            return response.status_code < 400
        except Exception:
            return False

    def validate_credentials(
        self,
        credentials: IntegrationCredentials,
    ) -> tuple[bool, list[str]]:
        """Validate that required credentials are provided.

        Args:
            credentials: The credentials to validate.

        Returns:
            Tuple of (is_valid, list_of_missing_fields).
        """
        missing = []

        for field_def in self.metadata.required_fields:
            value = getattr(credentials, field_def.name, None)
            if value is None:
                value = credentials.extra.get(field_def.name)

            if not value:
                missing.append(field_def.name)

        return (len(missing) == 0, missing)
