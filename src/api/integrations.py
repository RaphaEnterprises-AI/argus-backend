"""Integrations API endpoints.

Provides endpoints for:
- Listing user integrations
- Connecting/disconnecting integrations (Slack, GitHub, observability platforms)
- Testing connections before saving
- Triggering data syncs

Supported platforms:
- slack: Slack notifications
- github: GitHub PR comments and checks
- datadog: Datadog RUM + APM
- sentry: Sentry error tracking
- new_relic: New Relic APM + Browser
- fullstory: FullStory session replay
- posthog: PostHog analytics
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.context import get_current_organization_id
from src.api.teams import get_current_user, log_audit, verify_org_access
from src.services.key_encryption import encrypt_api_key, decrypt_api_key
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/integrations", tags=["Integrations"])


# ============================================================================
# Enums
# ============================================================================

class Platform(str, Enum):
    """Supported integration platforms."""
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


class PlatformType(str, Enum):
    """Platform categories."""
    NOTIFICATION = "notification"
    CI_CD = "ci_cd"
    DEPLOYMENT = "deployment"
    ISSUE_TRACKER = "issue_tracker"
    OBSERVABILITY = "observability"
    ANALYTICS = "analytics"


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


async def test_platform_connection(
    platform: Platform,
    credentials: dict[str, str],
) -> tuple[bool, str, dict]:
    """Test connection for a specific platform."""
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
    else:
        # For platforms without specific test, just validate credentials exist
        missing = validate_credentials(platform, credentials)
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}", {}
        return True, "Credentials provided (connection not verified)", {}


def format_integration_response(integration: dict) -> IntegrationResponse:
    """Convert database record to response model."""
    platform = integration.get("type") or integration.get("platform", "unknown")
    platform_info = PLATFORM_INFO.get(Platform(platform) if platform in [p.value for p in Platform] else None, {})

    return IntegrationResponse(
        id=integration["id"],
        platform=platform,
        platform_type=platform_info.get("type", PlatformType.OBSERVABILITY).value if platform_info else "unknown",
        name=integration.get("name", platform_info.get("name", platform) if platform_info else platform),
        is_connected=integration.get("status") == "connected" or integration.get("is_connected", False),
        project_id=integration.get("project_id"),
        last_sync_at=integration.get("last_sync_at"),
        sync_status=integration.get("sync_status"),
        sync_frequency_minutes=integration.get("sync_frequency_minutes", 60),
        data_points_synced=integration.get("data_points_synced", 0),
        features_enabled=integration.get("features_enabled", []),
        error_message=integration.get("error_message"),
        created_at=integration.get("created_at", datetime.now(UTC).isoformat()),
        updated_at=integration.get("updated_at"),
    )


# ============================================================================
# Endpoints
# ============================================================================

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
        path += f"&type=eq.{platform}"

    path += "&order=created_at.desc"

    result = await supabase.request(path)

    integrations = []
    if result.get("data"):
        for integration in result["data"]:
            integrations.append(format_integration_response(integration))

    # Build available platforms list
    available_platforms = [
        {
            "platform": p.value,
            "type": info["type"].value,
            "name": info["name"],
            "description": info["description"],
            "docs_url": info["docs_url"],
        }
        for p, info in PLATFORM_INFO.items()
    ]

    logger.info(
        "Listed integrations",
        user_id=user["user_id"],
        count=len(integrations),
    )

    return IntegrationListResponse(
        integrations=integrations,
        total=len(integrations),
        available_platforms=available_platforms,
    )


@router.post("/{platform}/connect", response_model=IntegrationResponse)
async def connect_integration(
    platform: str,
    body: ConnectRequest,
    request: Request,
):
    """
    Connect a new integration.

    Credentials are encrypted before storage using AES-256-GCM.
    The integration is validated before being saved.
    """
    # Validate platform
    try:
        platform_enum = Platform(platform)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform: {platform}. Valid platforms: {[p.value for p in Platform]}",
        )

    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    # Validate required credentials
    missing = validate_credentials(platform_enum, body.credentials)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required credentials: {', '.join(missing)}",
        )

    # Test connection before saving
    success, message, details = await test_platform_connection(platform_enum, body.credentials)
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
    check_path = f"/integrations?type=eq.{platform}"
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

        platform_info = PLATFORM_INFO[platform_enum]

        integration_data = {
            "id": integration_id,
            "type": platform,
            "name": body.name or platform_info["name"],
            "project_id": body.project_id,
            "credentials": encrypted_creds,
            "settings": body.settings or {},
            "status": "connected",
            "sync_frequency_minutes": body.sync_frequency_minutes,
            "data_points_synced": 0,
            "features_enabled": [],
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
                description=f"Connected {platform_info['name']} integration",
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
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?type=eq.{platform}"
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
    storing them.
    """
    await get_current_user(request)

    # Validate platform
    try:
        platform_enum = Platform(platform)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown platform: {platform}",
        )

    # Validate required credentials are present
    missing = validate_credentials(platform_enum, body.credentials)
    if missing:
        return TestConnectionResponse(
            success=False,
            message=f"Missing required credentials: {', '.join(missing)}",
        )

    # Test the connection
    success, message, details = await test_platform_connection(platform_enum, body.credentials)

    logger.info(
        "Tested integration connection",
        platform=platform,
        success=success,
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
    user = await get_current_user(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?type=eq.{platform}&status=eq.connected"
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
                SentryProvider,
                NewRelicProvider,
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
    user = await get_current_user(request)

    supabase = get_supabase_client()

    path = f"/integrations?type=eq.{platform}"
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
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    supabase = get_supabase_client()

    # Find the integration
    path = f"/integrations?type=eq.{platform}"
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
