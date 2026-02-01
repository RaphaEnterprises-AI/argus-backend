"""Integrations API endpoints.

Provides endpoints for:
- Listing user integrations
- Connecting/disconnecting integrations (Slack, GitHub, observability platforms)
- Testing connections before saving
- Triggering data syncs
- Listing available platforms from the plugin registry

The API now uses a plugin-based architecture where integrations are discovered
dynamically from the PluginRegistry. This allows for easy addition of new
integrations without modifying this file.

For backwards compatibility, legacy platforms without plugins still use the
hardcoded test_*_connection functions until they are migrated to plugins.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional, Union

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.context import get_current_organization_id
from src.api.middleware.tenant import validate_uuid, validate_uuid_optional
from src.api.teams import get_current_user, log_audit, verify_org_access
from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    SyncResult,
)
from src.integrations.registry import PluginRegistry, get_registry
from src.services.key_encryption import decrypt_api_key, encrypt_api_key
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/integrations", tags=["Integrations"])


# ============================================================================
# Enums (Legacy - kept for backwards compatibility)
# ============================================================================

class Platform(str, Enum):
    """Supported integration platforms.

    Note: This enum is kept for backwards compatibility.
    New platforms should be added as plugins in src/integrations/plugins/
    and will be automatically discovered by the PluginRegistry.
    """
    SLACK = "slack"
    GITHUB = "github"
    JIRA = "jira"
    VERCEL = "vercel"
    DATADOG = "datadog"
    SENTRY = "sentry"
    NEW_RELIC = "new_relic"
    FULLSTORY = "fullstory"
    POSTHOG = "posthog"
    HONEYCOMB = "honeycomb"
    GRAFANA = "grafana"
    # Tier 2 platforms
    PAGERDUTY = "pagerduty"
    LAUNCHDARKLY = "launchdarkly"
    AMPLITUDE = "amplitude"


class PlatformType(str, Enum):
    """Platform categories.

    Note: Maps to IntegrationCategory from the plugin system.
    """
    NOTIFICATION = "notification"
    CI_CD = "ci_cd"
    DEPLOYMENT = "deployment"
    ISSUE_TRACKER = "issue_tracker"
    OBSERVABILITY = "observability"
    ANALYTICS = "analytics"


# ============================================================================
# Category Mapping (Plugin System <-> Legacy)
# ============================================================================

# Map IntegrationCategory to legacy PlatformType
CATEGORY_TO_PLATFORM_TYPE: dict[IntegrationCategory, PlatformType] = {
    IntegrationCategory.CHAT: PlatformType.NOTIFICATION,
    IntegrationCategory.SOURCE_CODE: PlatformType.CI_CD,
    IntegrationCategory.CICD: PlatformType.CI_CD,
    IntegrationCategory.ISSUE_TRACKING: PlatformType.ISSUE_TRACKER,
    IntegrationCategory.DEPLOYMENT: PlatformType.DEPLOYMENT,
    IntegrationCategory.OBSERVABILITY: PlatformType.OBSERVABILITY,
    IntegrationCategory.INCIDENT_MANAGEMENT: PlatformType.OBSERVABILITY,
    IntegrationCategory.SESSION_REPLAY: PlatformType.ANALYTICS,
    IntegrationCategory.ANALYTICS: PlatformType.ANALYTICS,
    IntegrationCategory.FEATURE_FLAGS: PlatformType.ANALYTICS,
    IntegrationCategory.TESTING: PlatformType.CI_CD,
    IntegrationCategory.WEBHOOKS: PlatformType.NOTIFICATION,
    IntegrationCategory.AI_AGENTS: PlatformType.CI_CD,
    IntegrationCategory.DATABASE: PlatformType.CI_CD,
}


# Platform metadata
PLATFORM_INFO = {
    Platform.SLACK: {
        "type": PlatformType.NOTIFICATION,
        "name": "Slack",
        "description": "Send test results and alerts to Slack channels",
        "required_fields": [],
        "optional_fields": ["webhook_url", "bot_token", "default_channel"],
        "docs_url": "https://api.slack.com/messaging/webhooks",
        "auth_type": "oauth",
        "features": ["Test notifications", "Failure alerts", "Interactive messages", "Channel selector"],
    },
    Platform.GITHUB: {
        "type": PlatformType.CI_CD,
        "name": "GitHub",
        "description": "Post test results to PRs and create check runs",
        "required_fields": ["token"],
        "optional_fields": ["api_base"],
        "docs_url": "https://docs.github.com/en/rest",
        "auth_type": "oauth",
        "features": ["PR comments", "Check runs", "Webhooks", "Repository sync"],
    },
    Platform.JIRA: {
        "type": PlatformType.ISSUE_TRACKER,
        "name": "Jira",
        "description": "Sync issues and create tickets from test failures",
        "required_fields": [],
        "optional_fields": ["project_key", "issue_type"],
        "docs_url": "https://developer.atlassian.com/cloud/jira/platform/rest/v3/",
        "auth_type": "oauth",
        "features": ["Issue sync", "Auto-create tickets", "JQL queries", "Webhooks"],
    },
    Platform.VERCEL: {
        "type": PlatformType.DEPLOYMENT,
        "name": "Vercel",
        "description": "Track deployments and trigger tests on preview URLs",
        "required_fields": ["api_token"],
        "optional_fields": ["team_id"],
        "docs_url": "https://vercel.com/docs/rest-api",
        "auth_type": "api_key",
        "features": ["Deployment tracking", "Preview URL testing", "Webhooks", "Build logs"],
    },
    Platform.DATADOG: {
        "type": PlatformType.OBSERVABILITY,
        "name": "Datadog",
        "description": "Ingest RUM sessions and errors from Datadog",
        "required_fields": ["api_key", "app_key"],
        "optional_fields": ["site"],
        "docs_url": "https://docs.datadoghq.com/api/",
        "auth_type": "api_key",
        "features": ["RUM sessions", "Error tracking", "APM data", "Webhooks"],
    },
    Platform.SENTRY: {
        "type": PlatformType.OBSERVABILITY,
        "name": "Sentry",
        "description": "Import errors and issues from Sentry",
        "required_fields": ["auth_token"],
        "optional_fields": ["organization", "project"],
        "docs_url": "https://docs.sentry.io/api/",
        "auth_type": "api_key",
        "features": ["Error sync", "Issue tracking", "Webhooks", "Performance data"],
    },
    Platform.NEW_RELIC: {
        "type": PlatformType.OBSERVABILITY,
        "name": "New Relic",
        "description": "Import browser sessions and errors from New Relic",
        "required_fields": ["api_key"],
        "optional_fields": ["account_id"],
        "docs_url": "https://docs.newrelic.com/docs/apis/",
        "auth_type": "api_key",
        "features": ["APM data", "Browser monitoring", "Error tracking", "Alerts"],
    },
    Platform.FULLSTORY: {
        "type": PlatformType.ANALYTICS,
        "name": "FullStory",
        "description": "Import session recordings and frustration signals",
        "required_fields": ["api_key"],
        "optional_fields": [],
        "docs_url": "https://developer.fullstory.com/",
        "auth_type": "api_key",
        "features": ["Session replay", "Rage clicks", "Dead clicks", "Error tracking"],
    },
    Platform.POSTHOG: {
        "type": PlatformType.ANALYTICS,
        "name": "PostHog",
        "description": "Import product analytics and session recordings",
        "required_fields": ["api_key"],
        "optional_fields": ["host"],
        "docs_url": "https://posthog.com/docs/api",
        "auth_type": "api_key",
        "features": ["Analytics", "Session recordings", "Feature flags", "Events"],
    },
    Platform.HONEYCOMB: {
        "type": PlatformType.OBSERVABILITY,
        "name": "Honeycomb",
        "description": "Import traces and events from Honeycomb",
        "required_fields": ["api_key", "dataset"],
        "optional_fields": [],
        "docs_url": "https://docs.honeycomb.io/api/",
        "auth_type": "api_key",
        "features": ["Traces", "Events", "Query results", "SLOs"],
    },
    Platform.GRAFANA: {
        "type": PlatformType.OBSERVABILITY,
        "name": "Grafana",
        "description": "Import dashboards and alerts from Grafana",
        "required_fields": ["api_key", "url"],
        "optional_fields": [],
        "docs_url": "https://grafana.com/docs/grafana/latest/developers/http_api/",
        "auth_type": "api_key",
        "features": ["Dashboards", "Alerts", "Metrics", "Annotations"],
    },
    Platform.PAGERDUTY: {
        "type": PlatformType.OBSERVABILITY,
        "name": "PagerDuty",
        "description": "Sync incidents and post-mortems for correlation with deployments",
        "required_fields": ["api_token"],
        "optional_fields": ["default_service_id"],
        "docs_url": "https://developer.pagerduty.com/api-reference/",
        "auth_type": "api_key",
        "features": ["Incident sync", "Timeline tracking", "Post-mortem extraction", "MTTR metrics", "Webhooks"],
    },
    Platform.LAUNCHDARKLY: {
        "type": PlatformType.ANALYTICS,
        "name": "LaunchDarkly",
        "description": "Track feature flags to correlate flag changes with errors",
        "required_fields": ["api_token"],
        "optional_fields": ["project_key", "default_environment"],
        "docs_url": "https://apidocs.launchdarkly.com/",
        "auth_type": "api_key",
        "features": ["Flag sync", "Rollout tracking", "Audit log", "Environment awareness", "Webhooks"],
    },
    Platform.AMPLITUDE: {
        "type": PlatformType.ANALYTICS,
        "name": "Amplitude",
        "description": "Use product analytics to prioritize tests by feature usage",
        "required_fields": ["api_key", "secret_key"],
        "optional_fields": [],
        "docs_url": "https://www.docs.developers.amplitude.com/analytics/apis/",
        "auth_type": "api_key",
        "features": ["Event sync", "Funnel analysis", "User paths", "Retention data", "Test prioritization"],
    },
}


# ============================================================================
# Request/Response Models
# ============================================================================

class ConnectRequest(BaseModel):
    """Request to connect an integration."""
    credentials: dict[str, str] = Field(
        ...,
        description="Platform-specific credentials (api_key, token, etc.)",
    )
    name: str | None = Field(
        None,
        max_length=100,
        description="Optional name for this integration instance",
    )
    project_id: str | None = Field(
        None,
        description="Project ID to scope this integration to (optional)",
    )
    settings: dict[str, Any] | None = Field(
        None,
        description="Platform-specific settings",
    )
    sync_frequency_minutes: int = Field(
        60,
        ge=5,
        le=1440,
        description="How often to sync data (in minutes)",
    )


class IntegrationResponse(BaseModel):
    """Integration details response."""
    id: str
    platform: str
    platform_type: str
    name: str
    is_connected: bool
    project_id: str | None
    last_sync_at: str | None
    sync_status: str | None
    sync_frequency_minutes: int
    data_points_synced: int
    features_enabled: list[str]
    error_message: str | None
    created_at: str
    updated_at: str | None


class IntegrationListResponse(BaseModel):
    """Response for listing integrations."""
    integrations: list[IntegrationResponse]
    total: int
    available_platforms: list[dict]


class TestConnectionRequest(BaseModel):
    """Request to test a connection before saving."""
    credentials: dict[str, str] = Field(
        ...,
        description="Platform-specific credentials to test",
    )


class TestConnectionResponse(BaseModel):
    """Response for connection test."""
    success: bool
    message: str
    details: dict | None = None


class SyncResponse(BaseModel):
    """Response for sync trigger."""
    success: bool
    message: str
    sync_id: str | None = None
    data_points_synced: int = 0


class OAuthRedirectResponse(BaseModel):
    """Response when OAuth authentication is required.

    Returned by connect_integration when the platform uses OAuth.
    The frontend should redirect the user to the oauth_url.
    """
    oauth_url: str = Field(
        ...,
        description="URL to redirect the user to for OAuth authorization",
    )
    state: str = Field(
        ...,
        description="OAuth state parameter for CSRF protection",
    )
    platform: str = Field(
        ...,
        description="Platform name",
    )
    message: str = Field(
        default="Redirect to OAuth provider to authorize",
        description="Human-readable message",
    )
    requires_oauth: bool = Field(
        default=True,
        description="Indicates this platform requires OAuth flow",
    )


# ============================================================================
# Helper Functions
# ============================================================================

def encrypt_credentials(credentials: dict[str, str]) -> dict[str, str]:
    """Encrypt sensitive credential values before storage.

    Uses AES-256-GCM encryption via KeyEncryptionService.
    """
    encrypted = {}
    for key, value in credentials.items():
        if value and isinstance(value, str):
            encrypted_key = encrypt_api_key(value)
            encrypted[key] = encrypted_key.encrypted_data
    return encrypted


def decrypt_credentials(encrypted_credentials: dict[str, str]) -> dict[str, str]:
    """Decrypt stored credentials for use.

    Returns decrypted credential values.
    """
    decrypted = {}
    for key, value in encrypted_credentials.items():
        if value and isinstance(value, str):
            try:
                decrypted[key] = decrypt_api_key(value)
            except Exception as e:
                logger.warning(
                    "Failed to decrypt credential",
                    key=key,
                    error=str(e),
                )
                decrypted[key] = ""
    return decrypted


def validate_credentials(platform: Platform, credentials: dict[str, str]) -> list[str]:
    """Validate that required credentials are provided.

    Returns list of missing fields.
    """
    info = PLATFORM_INFO.get(platform)
    if not info:
        return ["Unknown platform"]

    missing = []
    for field in info["required_fields"]:
        if not credentials.get(field):
            missing.append(field)

    return missing


# ============================================================================
# Plugin System Integration Helpers
# ============================================================================

def dict_to_integration_credentials(credentials: dict[str, str]) -> IntegrationCredentials:
    """Convert a flat credentials dict to IntegrationCredentials.

    Maps common field names to the structured IntegrationCredentials dataclass.
    Unmapped fields go into the 'extra' dict.
    """
    # Known field mappings
    known_fields = {
        "api_key": "api_key",
        "api_secret": "api_secret",
        "access_token": "access_token",
        "token": "access_token",  # Alias for access_token
        "auth_token": "access_token",  # Alias for access_token
        "refresh_token": "refresh_token",
        "client_id": "client_id",
        "client_secret": "client_secret",
        "instance_url": "instance_url",
        "api_base": "instance_url",  # Alias
        "org_slug": "org_slug",
        "organization": "org_slug",  # Alias
        "project_slug": "project_slug",
        "project": "project_slug",  # Alias
        "workspace_id": "workspace_id",
        "team_id": "team_id",
        "site": "site",
        "username": "username",
        "password": "password",
        "email": "email",
        "webhook_url": "webhook_url",
        "webhook_secret": "webhook_secret",
        "app_key": "api_secret",  # Datadog app_key maps to api_secret
        "secret_key": "api_secret",  # Amplitude secret_key maps to api_secret
        "bot_token": "access_token",  # Slack bot token
    }

    mapped = {}
    extra = {}

    for key, value in credentials.items():
        if key in known_fields:
            field_name = known_fields[key]
            if field_name not in mapped:  # Don't overwrite if already set
                mapped[field_name] = value
        else:
            extra[key] = value

    return IntegrationCredentials(**mapped, extra=extra)


def is_platform_in_registry(platform: str) -> bool:
    """Check if a platform has a registered plugin."""
    registry = get_registry()
    return registry.has_plugin(platform)


async def test_platform_connection_via_registry(
    platform: str,
    credentials: dict[str, str],
) -> tuple[bool, str, dict]:
    """Test a platform connection using the plugin registry.

    Returns:
        Tuple of (success, message, details)
    """
    registry = get_registry()

    if not registry.has_plugin(platform):
        return False, f"No plugin found for platform: {platform}", {}

    creds = dict_to_integration_credentials(credentials)

    try:
        result: ConnectionTestResult = await registry.test_connection(platform, creds)
        return (
            result.success,
            result.message,
            {
                "latency_ms": result.latency_ms,
                "api_version": result.api_version,
                "permissions": result.permissions,
                "account_info": result.account_info,
            } if result.success else {"error_code": result.error_code, "error_details": result.error_details},
        )
    except ValueError as e:
        return False, str(e), {}
    except Exception as e:
        logger.error("Plugin connection test failed", platform=platform, error=str(e))
        return False, f"Connection test failed: {str(e)}", {}


async def sync_platform_data_via_registry(
    platform: str,
    credentials: dict[str, str],
    resources: Optional[list[str]] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
) -> SyncResult:
    """Sync data from a platform using the plugin registry.

    Returns:
        SyncResult from the plugin
    """
    registry = get_registry()

    if not registry.has_plugin(platform):
        return SyncResult(success=False, message=f"No plugin found for platform: {platform}")

    creds = dict_to_integration_credentials(credentials)

    try:
        return await registry.sync_data(
            platform,
            creds,
            resources=resources,
            since=since,
            limit=limit,
        )
    except ValueError as e:
        return SyncResult(success=False, message=str(e))
    except Exception as e:
        logger.error("Plugin sync failed", platform=platform, error=str(e))
        return SyncResult(success=False, message=f"Sync failed: {str(e)}")


def metadata_to_platform_info(metadata: IntegrationMetadata) -> dict:
    """Convert plugin IntegrationMetadata to legacy PLATFORM_INFO format."""
    platform_type = CATEGORY_TO_PLATFORM_TYPE.get(
        metadata.category,
        PlatformType.OBSERVABILITY,
    )

    return {
        "type": platform_type,
        "name": metadata.name,
        "description": metadata.description,
        "required_fields": [f.name for f in metadata.required_fields],
        "optional_fields": [f.name for f in metadata.optional_fields],
        "docs_url": metadata.docs_url,
        "auth_type": metadata.auth_type.value,
        "features": metadata.features,
    }


def get_all_available_platforms() -> list[dict]:
    """Get all available platforms from both PLATFORM_INFO and PluginRegistry.

    Returns a deduplicated list combining legacy platforms with registered plugins.
    """
    platforms = []
    seen_slugs = set()

    # First add all registered plugins (they take precedence)
    registry = get_registry()
    for metadata in registry.list_plugins():
        platform_type = CATEGORY_TO_PLATFORM_TYPE.get(
            metadata.category,
            PlatformType.OBSERVABILITY,
        )
        platforms.append({
            "platform": metadata.slug,
            "type": platform_type.value,
            "name": metadata.name,
            "description": metadata.description,
            "docs_url": metadata.docs_url,
            "category": metadata.category.value,
            "auth_type": metadata.auth_type.value,
            "features": metadata.features,
            "supports_sync": metadata.supports_sync,
            "supports_webhooks": metadata.supports_webhooks,
            "source": "plugin",
        })
        seen_slugs.add(metadata.slug)

    # Then add legacy platforms that don't have plugins yet
    for p, info in PLATFORM_INFO.items():
        if p.value not in seen_slugs:
            platforms.append({
                "platform": p.value,
                "type": info["type"].value,
                "name": info["name"],
                "description": info["description"],
                "docs_url": info["docs_url"],
                "category": info["type"].value,  # Approximation
                "auth_type": info["auth_type"],
                "features": info.get("features", []),
                "supports_sync": True,  # Assume true for legacy
                "supports_webhooks": False,  # Unknown for legacy
                "source": "legacy",
            })
            seen_slugs.add(p.value)

    return platforms


# ============================================================================
# Legacy Test Connection Functions (kept for backwards compatibility)
# ============================================================================

async def test_slack_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test Slack webhook or bot connection."""
    webhook_url = credentials.get("webhook_url")
    bot_token = credentials.get("bot_token")

    if webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook_url,
                    json={"text": "Argus connection test - this message confirms your webhook is working!"},
                )
                if response.status_code == 200:
                    return True, "Webhook connection successful", {"method": "webhook"}
                return False, f"Webhook returned status {response.status_code}", {}
        except Exception as e:
            return False, f"Webhook connection failed: {str(e)}", {}

    if bot_token:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": f"Bearer {bot_token}"},
                )
                data = response.json()
                if data.get("ok"):
                    return True, f"Bot authenticated as {data.get('user', 'unknown')}", {
                        "team": data.get("team"),
                        "user": data.get("user"),
                    }
                return False, f"Bot auth failed: {data.get('error', 'unknown')}", {}
        except Exception as e:
            return False, f"Bot connection failed: {str(e)}", {}

    return False, "Either webhook_url or bot_token is required", {}


async def test_github_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test GitHub token connection."""
    token = credentials.get("token")
    if not token:
        return False, "GitHub token is required", {}

    api_base = credentials.get("api_base", "https://api.github.com")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{api_base}/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                return True, f"Authenticated as {data.get('login', 'unknown')}", {
                    "login": data.get("login"),
                    "name": data.get("name"),
                    "scopes": response.headers.get("x-oauth-scopes", "").split(", "),
                }
            return False, f"GitHub returned status {response.status_code}", {}
    except Exception as e:
        return False, f"GitHub connection failed: {str(e)}", {}


async def test_datadog_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test Datadog API connection."""
    api_key = credentials.get("api_key")
    app_key = credentials.get("app_key")
    site = credentials.get("site", "datadoghq.com")

    if not api_key or not app_key:
        return False, "Both api_key and app_key are required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.{site}/api/v1/validate",
                headers={
                    "DD-API-KEY": api_key,
                    "DD-APPLICATION-KEY": app_key,
                },
            )
            if response.status_code == 200:
                return True, "Datadog API keys validated", {"site": site}
            return False, f"Datadog returned status {response.status_code}", {}
    except Exception as e:
        return False, f"Datadog connection failed: {str(e)}", {}


async def test_sentry_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test Sentry API connection."""
    auth_token = credentials.get("auth_token")
    organization = credentials.get("organization")
    project = credentials.get("project")

    if not all([auth_token, organization, project]):
        return False, "auth_token, organization, and project are required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://sentry.io/api/0/projects/{organization}/{project}/",
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            if response.status_code == 200:
                data = response.json()
                return True, f"Connected to project: {data.get('name', project)}", {
                    "project_name": data.get("name"),
                    "organization": organization,
                }
            return False, f"Sentry returned status {response.status_code}", {}
    except Exception as e:
        return False, f"Sentry connection failed: {str(e)}", {}


async def test_new_relic_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test New Relic API connection."""
    api_key = credentials.get("api_key")
    account_id = credentials.get("account_id")

    if not api_key or not account_id:
        return False, "Both api_key and account_id are required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Use NerdGraph to validate
            response = await client.post(
                "https://api.newrelic.com/graphql",
                headers={
                    "API-Key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": "{ actor { account(id: %s) { name } } }" % account_id,
                },
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("data", {}).get("actor", {}).get("account"):
                    account_name = data["data"]["actor"]["account"].get("name", "Unknown")
                    return True, f"Connected to account: {account_name}", {
                        "account_name": account_name,
                        "account_id": account_id,
                    }
            return False, f"New Relic returned status {response.status_code}", {}
    except Exception as e:
        return False, f"New Relic connection failed: {str(e)}", {}


async def test_pagerduty_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test PagerDuty API connection."""
    api_token = credentials.get("api_token")

    if not api_token:
        return False, "api_token is required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.pagerduty.com/abilities",
                headers={
                    "Authorization": f"Token token={api_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/vnd.pagerduty+json;version=2",
                },
            )
            if response.status_code == 200:
                data = response.json()
                abilities = data.get("abilities", [])
                return True, f"PagerDuty connected with {len(abilities)} abilities", {
                    "abilities_count": len(abilities),
                }
            elif response.status_code == 401:
                return False, "Invalid PagerDuty API token", {}
            return False, f"PagerDuty returned status {response.status_code}", {}
    except Exception as e:
        return False, f"PagerDuty connection failed: {str(e)}", {}


async def test_launchdarkly_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test LaunchDarkly API connection."""
    api_token = credentials.get("api_token")
    project_key = credentials.get("project_key", "default")

    if not api_token:
        return False, "api_token is required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://app.launchdarkly.com/api/v2/projects/{project_key}",
                headers={
                    "Authorization": api_token,
                    "Content-Type": "application/json",
                },
            )
            if response.status_code == 200:
                data = response.json()
                return True, f"Connected to LaunchDarkly project: {data.get('name', project_key)}", {
                    "project_key": data.get("key"),
                    "project_name": data.get("name"),
                }
            elif response.status_code == 401:
                return False, "Invalid LaunchDarkly API token", {}
            elif response.status_code == 404:
                return False, f"LaunchDarkly project '{project_key}' not found", {}
            return False, f"LaunchDarkly returned status {response.status_code}", {}
    except Exception as e:
        return False, f"LaunchDarkly connection failed: {str(e)}", {}


async def test_amplitude_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test Amplitude API connection."""
    api_key = credentials.get("api_key")
    secret_key = credentials.get("secret_key")

    if not api_key or not secret_key:
        return False, "Both api_key and secret_key are required", {}

    try:
        from datetime import datetime, timedelta
        start = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
        end = datetime.utcnow().strftime("%Y%m%d")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://amplitude.com/api/2/users",
                auth=(api_key, secret_key),
                params={"start": start, "end": end},
            )
            if response.status_code == 200:
                return True, "Amplitude API credentials validated", {"api_key": api_key[:8] + "..."}
            elif response.status_code == 401:
                return False, "Invalid Amplitude API credentials", {}
            return False, f"Amplitude returned status {response.status_code}", {}
    except Exception as e:
        return False, f"Amplitude connection failed: {str(e)}", {}


async def test_fullstory_connection(credentials: dict[str, str]) -> tuple[bool, str, dict]:
    """Test FullStory API connection."""
    api_key = credentials.get("api_key")

    if not api_key:
        return False, "api_key is required", {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # FullStory uses Basic auth with the API key
            response = await client.get(
                "https://api.fullstory.com/operations/v1",
                headers={
                    "Authorization": f"Basic {api_key}",
                    "Content-Type": "application/json",
                },
            )
            # FullStory returns 200 for valid keys, 401 for invalid
            if response.status_code == 200:
                return True, "FullStory API key validated", {}
            elif response.status_code == 401:
                return False, "Invalid FullStory API key", {}
            # Some endpoints may return 404 but still validate the key
            return True, "FullStory API key appears valid", {}
    except Exception as e:
        return False, f"FullStory connection failed: {str(e)}", {}


async def test_platform_connection(
    platform: Platform | str,
    credentials: dict[str, str],
) -> tuple[bool, str, dict]:
    """Test connection for a specific platform.

    This function first checks if the platform has a registered plugin
    in the PluginRegistry. If so, it uses the plugin's test_connection method.
    Otherwise, it falls back to legacy test functions for backwards compatibility.

    Args:
        platform: The platform identifier (Platform enum or string)
        credentials: Platform-specific credentials

    Returns:
        Tuple of (success, message, details)
    """
    # Normalize platform to string
    platform_slug = platform.value if isinstance(platform, Platform) else platform

    # First, try the plugin registry
    if is_platform_in_registry(platform_slug):
        return await test_platform_connection_via_registry(platform_slug, credentials)

    # Fall back to legacy test functions
    if isinstance(platform, str):
        try:
            platform = Platform(platform)
        except ValueError:
            # Unknown platform and no plugin - return error
            return False, f"Unknown platform: {platform_slug}. No plugin registered.", {}

    if platform == Platform.SLACK:
        return await test_slack_connection(credentials)
    elif platform == Platform.GITHUB:
        return await test_github_connection(credentials)
    elif platform == Platform.DATADOG:
        return await test_datadog_connection(credentials)
    elif platform == Platform.SENTRY:
        return await test_sentry_connection(credentials)
    elif platform == Platform.NEW_RELIC:
        return await test_new_relic_connection(credentials)
    elif platform == Platform.PAGERDUTY:
        return await test_pagerduty_connection(credentials)
    elif platform == Platform.LAUNCHDARKLY:
        return await test_launchdarkly_connection(credentials)
    elif platform == Platform.AMPLITUDE:
        return await test_amplitude_connection(credentials)
    elif platform == Platform.FULLSTORY:
        return await test_fullstory_connection(credentials)
    else:
        # For platforms without specific test, just validate credentials exist
        missing = validate_credentials(platform, credentials)
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}", {}
        return True, "Credentials provided (connection not verified)", {}


def format_integration_response(integration: dict) -> IntegrationResponse:
    """Convert database record to response model.

    Checks both the plugin registry and legacy PLATFORM_INFO for metadata.
    """
    platform = integration.get("type") or integration.get("platform", "unknown")

    # First check plugin registry
    registry = get_registry()
    plugin_metadata = registry.get_metadata(platform)

    if plugin_metadata:
        # Use plugin metadata
        platform_type = CATEGORY_TO_PLATFORM_TYPE.get(
            plugin_metadata.category,
            PlatformType.OBSERVABILITY,
        ).value
        platform_name = plugin_metadata.name
    else:
        # Fall back to legacy PLATFORM_INFO
        platform_info = PLATFORM_INFO.get(Platform(platform) if platform in [p.value for p in Platform] else None, {})
        platform_type = platform_info.get("type", PlatformType.OBSERVABILITY).value if platform_info else "unknown"
        platform_name = platform_info.get("name", platform) if platform_info else platform

    return IntegrationResponse(
        id=integration["id"],
        platform=platform,
        platform_type=platform_type,
        name=integration.get("name") or platform_name,
        is_connected=integration.get("status") == "connected" or integration.get("is_connected", False),
        project_id=integration.get("project_id"),
        last_sync_at=integration.get("last_sync_at"),
        sync_status=integration.get("sync_status"),
        sync_frequency_minutes=integration.get("sync_frequency_minutes") or 60,
        data_points_synced=integration.get("data_points_synced") or 0,
        features_enabled=integration.get("features_enabled") or [],
        error_message=integration.get("error_message"),
        created_at=integration.get("created_at") or datetime.now(UTC).isoformat(),
        updated_at=integration.get("updated_at"),
    )


# ============================================================================
# Response Models for Platforms Endpoint
# ============================================================================

class PlatformInfoResponse(BaseModel):
    """Detailed information about a platform."""
    platform: str
    name: str
    description: str
    type: str
    category: str
    auth_type: str
    docs_url: Optional[str] = None
    features: list[str] = []
    supports_sync: bool = True
    supports_webhooks: bool = False
    source: str  # "plugin" or "legacy"


class PlatformsListResponse(BaseModel):
    """Response for listing available platforms."""
    platforms: list[PlatformInfoResponse]
    total: int
    categories: list[str]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/platforms", response_model=PlatformsListResponse)
async def list_available_platforms(
    request: Request,
    category: Optional[str] = None,
):
    """
    List all available integration platforms.

    Returns platforms from both the plugin registry (new plugins) and
    legacy PLATFORM_INFO (backwards compatibility).

    Parameters:
    - category: Optional filter by category (e.g., "observability", "chat")
    """
    await get_current_user(request)

    all_platforms = get_all_available_platforms()

    # Filter by category if provided
    if category:
        all_platforms = [p for p in all_platforms if p.get("category") == category]

    # Extract unique categories
    categories = sorted(set(p.get("category", "unknown") for p in all_platforms))

    # Convert to response model
    platforms = [PlatformInfoResponse(**p) for p in all_platforms]

    logger.info(
        "Listed available platforms",
        total=len(platforms),
        filter_category=category,
    )

    return PlatformsListResponse(
        platforms=platforms,
        total=len(platforms),
        categories=categories,
    )


@router.get("/platforms/{platform}", response_model=PlatformInfoResponse)
async def get_platform_info(
    platform: str,
    request: Request,
):
    """
    Get detailed information about a specific platform.

    Returns platform metadata from either the plugin registry or legacy PLATFORM_INFO.
    """
    await get_current_user(request)

    # Check plugin registry first
    registry = get_registry()
    plugin_metadata = registry.get_metadata(platform)

    if plugin_metadata:
        platform_type = CATEGORY_TO_PLATFORM_TYPE.get(
            plugin_metadata.category,
            PlatformType.OBSERVABILITY,
        )
        return PlatformInfoResponse(
            platform=plugin_metadata.slug,
            name=plugin_metadata.name,
            description=plugin_metadata.description,
            type=platform_type.value,
            category=plugin_metadata.category.value,
            auth_type=plugin_metadata.auth_type.value,
            docs_url=plugin_metadata.docs_url,
            features=plugin_metadata.features,
            supports_sync=plugin_metadata.supports_sync,
            supports_webhooks=plugin_metadata.supports_webhooks,
            source="plugin",
        )

    # Check legacy PLATFORM_INFO
    try:
        platform_enum = Platform(platform)
        info = PLATFORM_INFO.get(platform_enum)
        if info:
            return PlatformInfoResponse(
                platform=platform_enum.value,
                name=info["name"],
                description=info["description"],
                type=info["type"].value,
                category=info["type"].value,  # Approximation
                auth_type=info["auth_type"],
                docs_url=info["docs_url"],
                features=info.get("features", []),
                supports_sync=True,
                supports_webhooks=False,
                source="legacy",
            )
    except ValueError:
        pass

    # Platform not found
    valid_platforms = list(registry.list_slugs()) + [p.value for p in Platform]
    raise HTTPException(
        status_code=404,
        detail=f"Platform not found: {platform}. Valid platforms: {sorted(set(valid_platforms))}",
    )


@router.get("", response_model=IntegrationListResponse)
async def list_integrations(
    request: Request,
    project_id: str | None = None,
    platform: str | None = None,
):
    """
    List all integrations for the current user/organization.

    Optionally filter by project_id or platform.
    """
    # Validate UUID parameters
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    supabase = get_supabase_client()

    # Build query path
    path = "/integrations?select=*"

    # Filter by organization via project membership
    if project_id:
        path += f"&project_id=eq.{project_id}"
    elif org_id:
        # Get all projects in the organization
        projects_result = await supabase.request(
            f"/projects?organization_id=eq.{org_id}&select=id"
        )
        if projects_result.get("data"):
            project_ids = [p["id"] for p in projects_result["data"]]
            path += f"&project_id=in.({','.join(project_ids)})"

    if platform:
        path += f"&platform=eq.{platform}"

    path += "&order=created_at.desc"

    result = await supabase.request(path)

    integrations = []
    if result.get("data"):
        for integration in result["data"]:
            integrations.append(format_integration_response(integration))

    # Build available platforms list from both plugins and legacy
    available_platforms = get_all_available_platforms()

    logger.info(
        "Listed integrations",
        user_id=user["user_id"],
        count=len(integrations),
        available_platforms_count=len(available_platforms),
    )

    return IntegrationListResponse(
        integrations=integrations,
        total=len(integrations),
        available_platforms=available_platforms,
    )


@router.post("/{platform}/connect", response_model=Union[IntegrationResponse, OAuthRedirectResponse])
async def connect_integration(
    platform: str,
    body: ConnectRequest,
    request: Request,
):
    """
    Connect a new integration.

    For API key/token-based integrations:
    - Credentials are encrypted before storage using AES-256-GCM
    - The integration is validated before being saved

    For OAuth-based integrations (GitHub, Slack, Jira, Linear):
    - Returns an OAuth authorization URL instead
    - The frontend should redirect the user to complete authorization
    - The OAuth callback will handle saving the credentials

    Supports both plugin-based platforms (discovered from PluginRegistry)
    and legacy platforms defined in PLATFORM_INFO.
    """
    # Validate UUID parameters
    validate_uuid_optional(body.project_id, "project_id")

    # Check if platform exists in registry or legacy
    registry = get_registry()
    plugin_metadata = registry.get_metadata(platform)

    # Validate platform - check both plugin registry and legacy enum
    platform_enum: Optional[Platform] = None
    if not plugin_metadata:
        try:
            platform_enum = Platform(platform)
        except ValueError:
            # Get list of all valid platforms
            valid_platforms = list(registry.list_slugs()) + [p.value for p in Platform]
            raise HTTPException(
                status_code=400,
                detail=f"Unknown platform: {platform}. Valid platforms: {sorted(set(valid_platforms))}",
            )

    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    # Determine authentication type
    auth_type: Optional[str] = None
    if plugin_metadata:
        # Plugin-based platforms have AuthType enum
        auth_type = plugin_metadata.auth_type.value if plugin_metadata.auth_type else None
    elif platform_enum:
        # Legacy platforms have auth_type string in PLATFORM_INFO
        auth_type = PLATFORM_INFO.get(platform_enum, {}).get("auth_type")

    # For OAuth integrations, return OAuth authorization URL instead of validating credentials
    if auth_type in ("oauth", "oauth2"):
        from src.api.oauth import generate_oauth_url, is_oauth_platform

        # Check if this platform has OAuth support
        if is_oauth_platform(platform):
            try:
                oauth_result = await generate_oauth_url(
                    platform=platform,
                    user_id=user["user_id"],
                    org_id=org_id,
                )
                logger.info(
                    "OAuth redirect generated for integration connect",
                    platform=platform,
                    user_id=user["user_id"],
                    org_id=org_id,
                )
                return OAuthRedirectResponse(
                    oauth_url=oauth_result["authorize_url"],
                    state=oauth_result["state"],
                    platform=oauth_result["platform"],
                    message=f"Redirect to {platform.title()} to authorize access",
                )
            except ValueError as e:
                # OAuth not configured for this platform, log and continue
                # This allows the endpoint to fall back to credential-based auth
                logger.warning(
                    "OAuth not configured, falling back to credential auth",
                    platform=platform,
                    error=str(e),
                )
        else:
            # Platform marked as OAuth but not in our OAuth platform list
            # This is a configuration mismatch, log warning
            logger.warning(
                "Platform marked as OAuth but not in OAuth platform list",
                platform=platform,
                auth_type=auth_type,
            )

    # Validate required credentials (for non-OAuth or OAuth fallback)
    if plugin_metadata:
        # Use plugin's credential validation
        creds = dict_to_integration_credentials(body.credentials)
        plugin = registry.get_plugin(platform)
        if plugin:
            is_valid, missing = plugin.validate_credentials(creds)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required credentials: {', '.join(missing)}",
                )
    elif platform_enum:
        # Use legacy validation
        missing = validate_credentials(platform_enum, body.credentials)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required credentials: {', '.join(missing)}",
            )

    # Test connection before saving (works with both plugins and legacy)
    success, message, details = await test_platform_connection(platform, body.credentials)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Connection test failed: {message}",
        )

    supabase = get_supabase_client()

    # If project_id provided, verify access
    if body.project_id:
        project_result = await supabase.request(
            f"/projects?id=eq.{body.project_id}&select=organization_id"
        )
        if not project_result.get("data"):
            raise HTTPException(status_code=404, detail="Project not found")
        _, _ = await verify_org_access(
            project_result["data"][0]["organization_id"],
            user["user_id"],
            user_email=user.get("email"),
            request=request,
        )

    # Check for existing integration
    check_path = f"/integrations?platform=eq.{platform}"
    if body.project_id:
        check_path += f"&project_id=eq.{body.project_id}"
    existing = await supabase.request(check_path)

    if existing.get("data"):
        # Update existing integration
        integration_id = existing["data"][0]["id"]
        encrypted_creds = encrypt_credentials(body.credentials)

        update_data = {
            "credentials": encrypted_creds,
            "status": "connected",
            "error_message": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if body.name:
            update_data["name"] = body.name
        if body.settings:
            update_data["settings"] = body.settings
        if body.sync_frequency_minutes:
            update_data["sync_frequency_minutes"] = body.sync_frequency_minutes

        result = await supabase.update(
            "integrations",
            {"id": f"eq.{integration_id}"},
            update_data,
        )

        logger.info(
            "Updated integration",
            platform=platform,
            integration_id=integration_id,
            user_id=user["user_id"],
        )

        # Fetch updated record
        updated = await supabase.request(f"/integrations?id=eq.{integration_id}")
        if updated.get("data"):
            return format_integration_response(updated["data"][0])
    else:
        # Create new integration
        integration_id = str(uuid.uuid4())
        encrypted_creds = encrypt_credentials(body.credentials)

        # Get platform name from plugin or legacy
        if plugin_metadata:
            platform_name = plugin_metadata.name
        elif platform_enum:
            platform_info = PLATFORM_INFO[platform_enum]
            platform_name = platform_info["name"]
        else:
            platform_name = platform.title()

        # Determine platform_type from plugin or legacy metadata
        if plugin_metadata:
            platform_type_value = CATEGORY_TO_PLATFORM_TYPE.get(
                plugin_metadata.category,
                PlatformType.OBSERVABILITY,
            ).value
        elif platform_enum:
            platform_type_value = PLATFORM_INFO[platform_enum]["type"].value
        else:
            platform_type_value = "observability"

        integration_data = {
            "id": integration_id,
            "platform": platform,  # DB column is 'platform', not 'type'
            "platform_type": platform_type_value,
            "name": body.name or platform_name,
            "user_id": user["user_id"],
            "project_id": body.project_id,
            "credentials": encrypted_creds,
            "config": body.settings or {},  # DB column is 'config', not 'settings'
            "status": "connected",
            "is_connected": True,
            "sync_frequency_minutes": body.sync_frequency_minutes,
            "data_points_synced": 0,
            "created_at": datetime.now(UTC).isoformat(),
        }

        result = await supabase.insert("integrations", integration_data)

        if result.get("error"):
            logger.error(
                "Failed to create integration",
                platform=platform,
                error=result["error"],
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save integration",
            )

        logger.info(
            "Created integration",
            platform=platform,
            integration_id=integration_id,
            user_id=user["user_id"],
        )

        # Log audit event
        if org_id:
            await log_audit(
                organization_id=org_id,
                user_id=user["user_id"],
                user_email=user.get("email"),
                action="integration.connect",
                resource_type="integration",
                resource_id=integration_id,
                description=f"Connected {platform_name} integration",
                metadata={"platform": platform, "project_id": body.project_id},
                request=request,
            )

        integration_data["is_connected"] = True
        return format_integration_response(integration_data)


@router.post("/{platform}/disconnect", response_model=dict)
async def disconnect_integration(
    platform: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Disconnect an integration.

    This removes the stored credentials but keeps the integration record
    for audit purposes.
    """
    # Validate UUID parameters
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?platform=eq.{platform}"
    if project_id:
        path += f"&project_id=eq.{project_id}"

    result = await supabase.request(path)

    if not result.get("data"):
        raise HTTPException(status_code=404, detail="Integration not found")

    integration = result["data"][0]
    integration_id = integration["id"]

    # Update to disconnected state
    update_result = await supabase.update(
        "integrations",
        {"id": f"eq.{integration_id}"},
        {
            "status": "disconnected",
            "credentials": {},
            "error_message": None,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )

    if update_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to disconnect integration")

    logger.info(
        "Disconnected integration",
        platform=platform,
        integration_id=integration_id,
        user_id=user["user_id"],
    )

    # Log audit event
    if org_id:
        await log_audit(
            organization_id=org_id,
            user_id=user["user_id"],
            user_email=user.get("email"),
            action="integration.disconnect",
            resource_type="integration",
            resource_id=integration_id,
            description=f"Disconnected {platform} integration",
            metadata={"platform": platform},
            request=request,
        )

    return {
        "success": True,
        "message": f"{platform} integration disconnected",
        "integration_id": integration_id,
    }


@router.post("/{platform}/test", response_model=TestConnectionResponse)
async def test_connection(
    platform: str,
    body: TestConnectionRequest,
    request: Request,
):
    """
    Test a connection before saving credentials.

    This validates the credentials against the platform's API without
    storing them. Supports both plugin-based and legacy platforms.
    """
    await get_current_user(request)

    # Check if platform exists in registry or legacy
    registry = get_registry()
    plugin_metadata = registry.get_metadata(platform)
    platform_enum: Optional[Platform] = None

    if not plugin_metadata:
        try:
            platform_enum = Platform(platform)
        except ValueError:
            valid_platforms = list(registry.list_slugs()) + [p.value for p in Platform]
            raise HTTPException(
                status_code=400,
                detail=f"Unknown platform: {platform}. Valid platforms: {sorted(set(valid_platforms))}",
            )

    # Validate required credentials are present
    if plugin_metadata:
        creds = dict_to_integration_credentials(body.credentials)
        plugin = registry.get_plugin(platform)
        if plugin:
            is_valid, missing = plugin.validate_credentials(creds)
            if not is_valid:
                return TestConnectionResponse(
                    success=False,
                    message=f"Missing required credentials: {', '.join(missing)}",
                )
    elif platform_enum:
        missing = validate_credentials(platform_enum, body.credentials)
        if missing:
            return TestConnectionResponse(
                success=False,
                message=f"Missing required credentials: {', '.join(missing)}",
            )

    # Test the connection (works with both plugins and legacy)
    success, message, details = await test_platform_connection(platform, body.credentials)

    logger.info(
        "Tested integration connection",
        platform=platform,
        success=success,
        source="plugin" if plugin_metadata else "legacy",
    )

    return TestConnectionResponse(
        success=success,
        message=message,
        details=details if success else None,
    )


@router.post("/{platform}/sync", response_model=SyncResponse)
async def trigger_sync(
    platform: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Trigger a data sync for an integration.

    This fetches the latest data from the connected platform.
    """
    # Validate UUID parameters
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?platform=eq.{platform}&status=eq.connected"
    if project_id:
        path += f"&project_id=eq.{project_id}"

    result = await supabase.request(path)

    if not result.get("data"):
        raise HTTPException(
            status_code=404,
            detail=f"No connected {platform} integration found",
        )

    integration = result["data"][0]
    integration_id = integration["id"]

    # Decrypt credentials
    encrypted_creds = integration.get("credentials", {})
    if not encrypted_creds:
        raise HTTPException(
            status_code=400,
            detail="Integration has no stored credentials",
        )

    try:
        credentials = decrypt_credentials(encrypted_creds)
    except Exception as e:
        logger.error("Failed to decrypt credentials", integration_id=integration_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to decrypt integration credentials",
        )

    # Update sync status to "syncing"
    await supabase.update(
        "integrations",
        {"id": f"eq.{integration_id}"},
        {
            "sync_status": "syncing",
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )

    # Perform the sync based on platform
    sync_id = str(uuid.uuid4())
    data_points_synced = 0

    try:
        # First, try using the plugin registry
        registry = get_registry()
        if registry.has_plugin(platform):
            # Use plugin-based sync
            sync_result = await sync_platform_data_via_registry(
                platform,
                credentials,
                limit=100,
            )
            if sync_result.success:
                data_points_synced = sync_result.data_points_synced

                # Store synced data in SDLC events if project_id exists
                project_id_for_events = integration.get("project_id")
                if project_id_for_events and sync_result.data:
                    # Store the raw synced data for downstream processing
                    for resource_type, items in sync_result.data.items():
                        for item in items if isinstance(items, list) else [items]:
                            if isinstance(item, dict):
                                await supabase.insert("sdlc_events", {
                                    "project_id": project_id_for_events,
                                    "source_platform": platform,
                                    "event_type": resource_type,
                                    "external_id": item.get("id", str(uuid.uuid4())),
                                    "title": item.get("title", item.get("name", f"{platform} {resource_type}")),
                                    "metadata": item,
                                })
            else:
                # Sync failed via plugin - raise to trigger error handling
                raise Exception(sync_result.message)
        else:
            # Fall back to legacy sync logic
            platform_enum = Platform(platform)

            if platform_enum == Platform.SLACK:
                # Slack doesn't need syncing - it's push-based
                data_points_synced = 0

            elif platform_enum == Platform.GITHUB:
                # GitHub sync would fetch recent PRs/issues
                # For now, just validate connection works
                success, _, _ = await test_github_connection(credentials)
                if success:
                    data_points_synced = 0  # Would be actual PR count

            elif platform_enum in [Platform.DATADOG, Platform.SENTRY, Platform.NEW_RELIC]:
                # Import from observability hub
                from src.integrations.observability_hub import (
                    DatadogProvider,
                    NewRelicProvider,
                    SentryProvider,
                )

                if platform_enum == Platform.DATADOG:
                    provider = DatadogProvider(
                        api_key=credentials.get("api_key", ""),
                        app_key=credentials.get("app_key", ""),
                        site=credentials.get("site", "datadoghq.com"),
                    )
                    errors = await provider.get_errors(limit=100)
                    data_points_synced = len(errors)
                    await provider.close()

                elif platform_enum == Platform.SENTRY:
                    provider = SentryProvider(
                        auth_token=credentials.get("auth_token", ""),
                        organization=credentials.get("organization", ""),
                        project=credentials.get("project", ""),
                    )
                    errors = await provider.get_errors(limit=100)
                    data_points_synced = len(errors)
                    await provider.close()

                elif platform_enum == Platform.NEW_RELIC:
                    provider = NewRelicProvider(
                        api_key=credentials.get("api_key", ""),
                        account_id=credentials.get("account_id", ""),
                    )
                    errors = await provider.get_errors(limit=100)
                    data_points_synced = len(errors)
                    await provider.close()

            elif platform_enum == Platform.PAGERDUTY:
                # Sync PagerDuty incidents
                from src.integrations.pagerduty_integration import PagerDutyIntegration

                pd = PagerDutyIntegration(
                    api_token=credentials.get("api_token", ""),
                    default_since_days=30,
                )
                incidents = await pd.get_incidents(limit=100)
                data_points_synced = len(incidents)

                # Store incidents as SDLC events
                project_id_for_events = integration.get("project_id")
                if project_id_for_events:
                    for incident in incidents:
                        await supabase.insert("sdlc_events", {
                            "project_id": project_id_for_events,
                            "source_platform": "pagerduty",
                            "event_type": "incident",
                            "external_id": incident.incident_id,
                            "external_key": str(incident.incident_number),
                            "external_url": incident.html_url,
                            "title": incident.title,
                            "description": incident.description,
                            "status": incident.status.value,
                            "priority": incident.priority,
                            "incident_id": incident.incident_id,
                            "occurred_at": incident.created_at.isoformat() if incident.created_at else None,
                            "metadata": {
                                "service_id": incident.service_id,
                                "service_name": incident.service_name,
                                "urgency": incident.urgency.value,
                                "duration_seconds": incident.duration_seconds,
                                "time_to_acknowledge_seconds": incident.time_to_acknowledge_seconds,
                            },
                        })
                await pd.close()

            elif platform_enum == Platform.LAUNCHDARKLY:
                # Sync LaunchDarkly flags
                from src.integrations.launchdarkly_integration import LaunchDarklyIntegration

                ld = LaunchDarklyIntegration(
                    api_token=credentials.get("api_token"),
                    project_key=credentials.get("project_key", "default"),
                )
                environment = credentials.get("default_environment", "production")
                flags = await ld.get_flags(environment=environment, limit=100)
                data_points_synced = len(flags)

                # Store flags as SDLC events
                project_id_for_events = integration.get("project_id")
                if project_id_for_events:
                    for flag in flags:
                        await supabase.insert("sdlc_events", {
                            "project_id": project_id_for_events,
                            "source_platform": "launchdarkly",
                            "event_type": "flag_change" if flag.on else "flag_disabled",
                            "external_id": flag.key,
                            "external_key": flag.key,
                            "title": flag.name,
                            "description": flag.description,
                            "status": "on" if flag.on else "off",
                            "flag_key": flag.key,
                            "occurred_at": flag.created_at.isoformat() if flag.created_at else None,
                            "metadata": {
                                "kind": flag.kind,
                                "environment": flag.environment,
                                "rollout_percentage": flag.rollout_percentage,
                                "variations": flag.variations,
                                "tags": flag.tags,
                            },
                        })
                await ld.close()

            elif platform_enum == Platform.AMPLITUDE:
                # Sync Amplitude top events for test prioritization
                from src.integrations.amplitude_integration import AmplitudeIntegration

                amp = AmplitudeIntegration(
                    api_key=credentials.get("api_key", ""),
                    secret_key=credentials.get("secret_key", ""),
                )
                events = await amp.get_top_events(limit=50, days=30)
                data_points_synced = len(events)

                # Store as SDLC events for correlation
                project_id_for_events = integration.get("project_id")
                if project_id_for_events:
                    for event in events:
                        await supabase.insert("sdlc_events", {
                            "project_id": project_id_for_events,
                            "source_platform": "amplitude",
                            "event_type": "user_event",
                            "external_id": event.event_type,
                            "external_key": event.event_type,
                            "title": event.event_type,
                            "description": f"Top event with {event.unique_users} unique users",
                            "metadata": {
                                "total_count": event.total_count,
                                "unique_users": event.unique_users,
                                "avg_per_user": event.avg_per_user,
                            },
                        })
                await amp.close()

            elif platform_enum == Platform.FULLSTORY:
                # Sync FullStory sessions
                from src.integrations.observability_hub import FullStoryProvider

                fs = FullStoryProvider(api_key=credentials.get("api_key", ""))
                sessions = await fs.get_recent_sessions(limit=100)
                data_points_synced = len(sessions)

                # Store sessions as SDLC events
                project_id_for_events = integration.get("project_id")
                if project_id_for_events:
                    for session in sessions:
                        # Only store sessions with frustration signals
                        if session.frustration_signals:
                            await supabase.insert("sdlc_events", {
                                "project_id": project_id_for_events,
                                "source_platform": "fullstory",
                                "event_type": "frustration_signal",
                                "external_id": session.session_id,
                                "external_url": session.replay_url,
                                "title": f"Frustration signals in session {session.session_id[:8]}",
                                "description": f"Session with {len(session.frustration_signals)} frustration signals",
                                "session_id": session.session_id,
                                "occurred_at": session.started_at.isoformat() if session.started_at else None,
                                "metadata": {
                                    "user_id": session.user_id,
                                    "duration_ms": session.duration_ms,
                                    "page_views": session.page_views,
                                    "frustration_count": len(session.frustration_signals),
                                    "device": session.device,
                                },
                            })
                await fs.close()

        # Update sync status to completed
        await supabase.update(
            "integrations",
            {"id": f"eq.{integration_id}"},
            {
                "sync_status": "completed",
                "last_sync_at": datetime.now(UTC).isoformat(),
                "data_points_synced": integration.get("data_points_synced", 0) + data_points_synced,
                "error_message": None,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(
            "Completed integration sync",
            platform=platform,
            integration_id=integration_id,
            data_points_synced=data_points_synced,
            user_id=user["user_id"],
        )

        return SyncResponse(
            success=True,
            message=f"Sync completed. {data_points_synced} data points synced.",
            sync_id=sync_id,
            data_points_synced=data_points_synced,
        )

    except Exception as e:
        # Update sync status to failed
        await supabase.update(
            "integrations",
            {"id": f"eq.{integration_id}"},
            {
                "sync_status": "failed",
                "error_message": str(e),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

        logger.error(
            "Integration sync failed",
            platform=platform,
            integration_id=integration_id,
            error=str(e),
        )

        return SyncResponse(
            success=False,
            message=f"Sync failed: {str(e)}",
            sync_id=sync_id,
        )


@router.get("/{platform}", response_model=IntegrationResponse)
async def get_integration(
    platform: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Get details for a specific integration.
    """
    # Validate UUID parameters
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)

    supabase = get_supabase_client()

    path = f"/integrations?platform=eq.{platform}"
    if project_id:
        path += f"&project_id=eq.{project_id}"

    result = await supabase.request(path)

    if not result.get("data"):
        raise HTTPException(status_code=404, detail="Integration not found")

    return format_integration_response(result["data"][0])


@router.delete("/{platform}")
async def delete_integration(
    platform: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Permanently delete an integration.

    This removes all stored data including credentials and sync history.
    """
    # Validate UUID parameters
    validate_uuid_optional(project_id, "project_id")

    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?platform=eq.{platform}"
    if project_id:
        path += f"&project_id=eq.{project_id}"

    result = await supabase.request(path)

    if not result.get("data"):
        raise HTTPException(status_code=404, detail="Integration not found")

    integration_id = result["data"][0]["id"]

    # Delete the integration
    delete_result = await supabase.request(
        f"/integrations?id=eq.{integration_id}",
        method="DELETE",
    )

    if delete_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to delete integration")

    logger.info(
        "Deleted integration",
        platform=platform,
        integration_id=integration_id,
        user_id=user["user_id"],
    )

    # Log audit event
    if org_id:
        await log_audit(
            organization_id=org_id,
            user_id=user["user_id"],
            user_email=user.get("email"),
            action="integration.delete",
            resource_type="integration",
            resource_id=integration_id,
            description=f"Deleted {platform} integration",
            metadata={"platform": platform},
            request=request,
        )

    return {
        "success": True,
        "message": f"{platform} integration deleted",
        "integration_id": integration_id,
    }
