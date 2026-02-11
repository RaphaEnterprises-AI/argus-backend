"""Batch 6 — Platform Operations live integration tests.

Covers:
  • data_layer_health.py  (2 endpoints)
  • browser.py            (6 endpoints — most need active pool, test what we can)
  • artifacts.py          (9 endpoints — mostly 404/503 expected w/o R2)
  • notifications.py      (~20 endpoints — Slack, channels, rules, user notifs, prefs)
  • audit.py              (4 endpoints)
  • integrations.py       (9 endpoints)

Every test tolerates HTTP 429 (rate-limit) as a valid production response.
"""

from __future__ import annotations

import os

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Configuration — env vars with production defaults
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get(
    "ARGUS_BASE_URL",
    "https://skopaq-brain-production.up.railway.app",
)
API_KEY = os.environ.get(
    "ARGUS_API_KEY",
    "argus_sk_9a1819e0e9faa47011c28ff75bbeb442ffa285779233ea3e81aed7e992367198",
)
ORG_ID = os.environ.get(
    "ARGUS_ORG_ID",
    "229904aa-3084-4b25-8bf8-c07ae749888c",
)
PROJ_ID = os.environ.get(
    "ARGUS_PROJECT_ID",
    "e98fc22d-ae11-490d-86f2-116fdd29941b",
)
FAKE_UUID = "00000000-0000-0000-0000-000000000000"

HEADERS = {
    "X-API-Key": API_KEY,
    "X-Organization-Id": ORG_ID,
    "Content-Type": "application/json",
}

DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
SLOW_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


# ---------------------------------------------------------------------------
# Pydantic response models for strict validation
# ---------------------------------------------------------------------------

class ComponentHealthModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    name: str
    status: str
    latency_ms: float | None = Field(None, alias="latencyMs")
    message: str | None = None
    checked_at: str | None = Field(None, alias="checkedAt")


class DataLayerHealthModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    overall_status: str = Field(alias="overallStatus")
    components: list[dict]
    healthy_count: int = Field(alias="healthyCount")
    total_count: int = Field(alias="totalCount")
    checked_at: str = Field(alias="checkedAt")


class BrowserHealthModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    healthy: bool


class ArtifactResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    artifact_id: str = Field(alias="artifactId")
    type: str = ""


class ArtifactListModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    artifacts: list[dict]
    total: int


class NotificationResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    message: str


class SlackStatusModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    configured: bool
    webhook_configured: bool = Field(alias="webhookConfigured")
    bot_configured: bool = Field(alias="botConfigured")
    default_channel: str = Field(alias="defaultChannel")
    webhook_status: str = Field(alias="webhookStatus")
    api_status: str = Field(alias="apiStatus")


class ChannelResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    name: str
    channel_type: str = Field(alias="channelType")
    enabled: bool


class RuleResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str
    channel_id: str = Field(alias="channelId")
    event_type: str = Field(alias="eventType")
    enabled: bool


class UserNotificationListModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    notifications: list[dict]
    total: int
    unread_count: int = Field(alias="unreadCount")
    has_more: bool = Field(alias="hasMore")


class UnreadCountModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    unread_count: int = Field(alias="unreadCount")
    by_priority: dict = Field(default_factory=dict, alias="byPriority")


class NotificationPrefsModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    user_id: str = Field(alias="userId")
    email_enabled: bool = Field(alias="emailEnabled")
    in_app_enabled: bool = Field(alias="inAppEnabled")


class AuditLogResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    logs: list[dict]
    total: int
    page: int
    page_size: int = Field(alias="pageSize")
    has_more: bool = Field(alias="hasMore")


class AuditSummaryModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    total_events: int = Field(alias="totalEvents")
    events_today: int = Field(alias="eventsToday")
    events_this_week: int = Field(alias="eventsThisWeek")
    events_this_month: int = Field(alias="eventsThisMonth")
    by_action: dict = Field(alias="byAction")


class IntegrationListModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    integrations: list[dict]
    total: int
    available_platforms: list[dict] = Field(alias="availablePlatforms")


class PlatformsListModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    platforms: list[dict]
    total: int
    categories: list[str]


class PlatformInfoModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    platform: str
    name: str
    description: str


# ===========================================================================
# 1. Data Layer Health
# ===========================================================================

class TestDataLayerHealth:
    """Tests for GET /api/v1/health/data-layer and /data-layer/{component}."""

    @pytest.mark.asyncio
    async def test_data_layer_health_returns_200(self):
        """GET /api/v1/health/data-layer — all 9 components."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/health/data-layer", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            model = DataLayerHealthModel.model_validate(data)
            assert model.total_count >= 7  # at least 7 components expected
            assert model.overall_status in ("healthy", "degraded", "unhealthy", "unknown")

    @pytest.mark.asyncio
    async def test_individual_component_health(self):
        """GET /api/v1/health/data-layer/{component} — cognee."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/health/data-layer/cognee", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["name"] == "Cognee"
            assert data["status"] in ("healthy", "degraded", "unhealthy", "unknown")

    @pytest.mark.asyncio
    async def test_unknown_component_returns_200_unknown(self):
        """GET /api/v1/health/data-layer/nonexistent — returns status=unknown."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/health/data-layer/nonexistent", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_supabase_component_health(self):
        """GET /api/v1/health/data-layer/supabase — should be healthy."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/health/data-layer/supabase", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["name"] == "Supabase"
            assert data["status"] == "healthy"


# ===========================================================================
# 2. Browser Pool
# ===========================================================================

class TestBrowserPool:
    """Tests for /api/v1/browser/* endpoints.

    Most browser endpoints require an active browser pool or Selenium Grid.
    We test health and verify error handling for the rest.
    """

    @pytest.mark.asyncio
    async def test_browser_health_returns_200(self):
        """GET /api/v1/browser/health — pool may or may not be running."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/browser/health", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            BrowserHealthModel.model_validate(data)

    @pytest.mark.asyncio
    async def test_browser_test_requires_body(self):
        """POST /api/v1/browser/test — 422 without body."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v1/browser/test", headers=HEADERS
            )
        assert resp.status_code in (422, 429), resp.text

    @pytest.mark.asyncio
    async def test_browser_observe_requires_body(self):
        """POST /api/v1/browser/observe — 422 without body."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v1/browser/observe", headers=HEADERS
            )
        assert resp.status_code in (422, 429), resp.text


# ===========================================================================
# 3. Artifacts
# ===========================================================================

class TestArtifacts:
    """Tests for /api/v1/artifacts/* endpoints."""

    @pytest.mark.asyncio
    async def test_list_artifacts_returns_200(self):
        """GET /api/v1/artifacts/ — list recent artifacts."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/artifacts/", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ArtifactListModel.model_validate(data)

    @pytest.mark.asyncio
    async def test_get_artifact_fake_id_returns_404(self):
        """GET /api/v1/artifacts/{id} — nonexistent."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/artifacts/screenshot_0000000000000000_20260101_000000",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_artifact_raw_invalid_id_returns_400(self):
        """GET /api/v1/artifacts/{id}/raw — invalid format returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/artifacts/bad-id-format/raw",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_video_artifact_invalid_returns_400(self):
        """GET /api/v1/artifacts/videos/{id} — non-video prefix returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/artifacts/videos/screenshot_fake",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 429), resp.text

    @pytest.mark.asyncio
    async def test_resolve_artifacts_empty_list(self):
        """POST /api/v1/artifacts/resolve — empty list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v1/artifacts/resolve",
                headers=HEADERS,
                json=[],
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["total"] == 0
            assert data["resolved"] == 0


# ===========================================================================
# 4. Notifications
# ===========================================================================

class TestNotifications:
    """Tests for /api/v1/notifications/* endpoints."""

    # -- Slack status --
    @pytest.mark.asyncio
    async def test_slack_status_returns_200(self):
        """GET /api/v1/notifications/slack/status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/slack/status", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            SlackStatusModel.model_validate(resp.json())

    # -- Event types --
    @pytest.mark.asyncio
    async def test_event_types_returns_200(self):
        """GET /api/v1/notifications/event-types."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/event-types", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            event_types = data.get("event_types") or data.get("eventTypes", [])
            assert len(event_types) >= 5

    # -- Channels --
    @pytest.mark.asyncio
    async def test_list_channels_returns_200(self):
        """GET /api/v1/notifications/channels."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/channels",
                headers=HEADERS,
                params={"organization_id": ORG_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_channel_fake_id_returns_404(self):
        """GET /api/v1/notifications/channels/{id} — nonexistent."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/channels/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_channel_fake_id_returns_404(self):
        """DELETE /api/v1/notifications/channels/{id} — nonexistent."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{BASE_URL}/api/v1/notifications/channels/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    # -- Rules --
    @pytest.mark.asyncio
    async def test_list_rules_returns_200(self):
        """GET /api/v1/notifications/rules."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/rules", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_rule_fake_id_returns_404(self):
        """GET /api/v1/notifications/rules/{id} — nonexistent."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/rules/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    # -- Logs --
    @pytest.mark.asyncio
    async def test_list_notification_logs_returns_200(self):
        """GET /api/v1/notifications/logs."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/logs", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    # -- User notifications --
    @pytest.mark.asyncio
    async def test_list_user_notifications_returns_200(self):
        """GET /api/v1/notifications — user notifications."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            UserNotificationListModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_unread_count_returns_200(self):
        """GET /api/v1/notifications/unread-count."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/unread-count", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            UnreadCountModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_mark_notification_fake_id_returns_404(self):
        """PUT /api/v1/notifications/{id}/read — nonexistent."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.put(
                f"{BASE_URL}/api/v1/notifications/{FAKE_UUID}/read",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_mark_all_read_returns_200(self):
        """PUT /api/v1/notifications/mark-all-read."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.put(
                f"{BASE_URL}/api/v1/notifications/mark-all-read",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["success"] is True

    # -- Preferences --
    @pytest.mark.asyncio
    async def test_get_notification_preferences_returns_200(self):
        """GET /api/v1/notifications/preferences."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/preferences", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            NotificationPrefsModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_update_notification_preferences_returns_200(self):
        """PUT /api/v1/notifications/preferences — partial update."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.put(
                f"{BASE_URL}/api/v1/notifications/preferences",
                headers=HEADERS,
                json={"email_enabled": True},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            NotificationPrefsModel.model_validate(resp.json())


# ===========================================================================
# 5. Audit Logs
# ===========================================================================

class TestAuditLogs:
    """Tests for /api/v1/audit/organizations/{org_id}/* endpoints."""

    @pytest.mark.asyncio
    async def test_get_audit_logs_returns_200(self):
        """GET /api/v1/audit/organizations/{org_id}/logs — paginated."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/audit/organizations/{ORG_ID}/logs",
                headers=HEADERS,
                params={"page": 1, "page_size": 10},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            AuditLogResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_get_audit_summary_returns_200(self):
        """GET /api/v1/audit/organizations/{org_id}/summary."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/audit/organizations/{ORG_ID}/summary",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            AuditSummaryModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_export_audit_logs_json(self):
        """GET /api/v1/audit/organizations/{org_id}/export?format=json."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/audit/organizations/{ORG_ID}/export",
                headers=HEADERS,
                params={"format": "json"},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "logs" in data or "log_count" in data or "logCount" in data

    @pytest.mark.asyncio
    async def test_export_audit_logs_csv(self):
        """GET /api/v1/audit/organizations/{org_id}/export?format=csv."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/audit/organizations/{ORG_ID}/export",
                headers=HEADERS,
                params={"format": "csv"},
            )
        # CSV returns 200 with text/csv content type
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            # Could be CSV or JSON (if table missing, returns JSON fallback)
            assert "csv" in ct or "json" in ct or "text" in ct

    @pytest.mark.asyncio
    async def test_get_available_actions_returns_200(self):
        """GET /api/v1/audit/organizations/{org_id}/actions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/audit/organizations/{ORG_ID}/actions",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            categories = data.get("categories", {})
            assert len(categories) >= 3
            assert "resource_types" in data or "resourceTypes" in data


# ===========================================================================
# 6. Integrations
# ===========================================================================

class TestIntegrations:
    """Tests for /api/v1/integrations/* endpoints."""

    @pytest.mark.asyncio
    async def test_list_available_platforms_returns_200(self):
        """GET /api/v1/integrations/platforms."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations/platforms", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            PlatformsListModel.model_validate(data)
            assert data["total"] >= 5  # at least 5 platforms

    @pytest.mark.asyncio
    async def test_get_platform_info_github(self):
        """GET /api/v1/integrations/platforms/github."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations/platforms/github", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            PlatformInfoModel.model_validate(data)
            assert data["platform"] == "github"

    @pytest.mark.asyncio
    async def test_get_platform_info_unknown_returns_404(self):
        """GET /api/v1/integrations/platforms/nonexistent — 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations/platforms/nonexistent_platform",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_list_integrations_returns_200(self):
        """GET /api/v1/integrations — list user/org integrations."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations", headers=HEADERS
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            IntegrationListModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_list_integrations_with_project_filter(self):
        """GET /api/v1/integrations?project_id=... — filtered."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            IntegrationListModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_test_connection_missing_creds_returns_error(self):
        """POST /api/v1/integrations/github/test — missing token."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v1/integrations/github/test",
                headers=HEADERS,
                json={"credentials": {}},
            )
        # Returns 200 with success=false or 400 for validation
        assert resp.status_code in (200, 400, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_returns_404(self):
        """POST /api/v1/integrations/nonexistent/disconnect — 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/api/v1/integrations/nonexistent_platform/disconnect",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_integration_nonexistent_returns_404(self):
        """GET /api/v1/integrations/nonexistent — 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/integrations/nonexistent_platform_xyz",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_integration_nonexistent_returns_404(self):
        """DELETE /api/v1/integrations/nonexistent — 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{BASE_URL}/api/v1/integrations/nonexistent_platform_xyz",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text


# ===========================================================================
# 7. Cross-Module Integration Tests
# ===========================================================================

class TestCrossModuleIntegration:
    """Cross-module tests that verify consistency between related endpoints."""

    @pytest.mark.asyncio
    async def test_health_components_match_individual_checks(self):
        """Verify /data-layer lists all individually-checkable components."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(
                f"{BASE_URL}/api/v1/health/data-layer", headers=HEADERS
            )
        if resp.status_code == 429:
            return
        assert resp.status_code == 200, resp.text
        data = resp.json()
        component_names = {
            c.get("name", "").lower() for c in data.get("components", [])
        }
        # These should always be present
        expected = {"cognee", "supabase"}
        assert expected.issubset(component_names), f"Missing: {expected - component_names}"

    @pytest.mark.asyncio
    async def test_platforms_appear_in_integration_list(self):
        """Verify platforms from /platforms are available in integration list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            platforms_resp = await client.get(
                f"{BASE_URL}/api/v1/integrations/platforms", headers=HEADERS
            )
            integrations_resp = await client.get(
                f"{BASE_URL}/api/v1/integrations", headers=HEADERS
            )
        if platforms_resp.status_code == 429 or integrations_resp.status_code == 429:
            return
        assert platforms_resp.status_code == 200
        assert integrations_resp.status_code == 200

        platforms_data = platforms_resp.json()
        integrations_data = integrations_resp.json()

        # The available_platforms in integrations list should include all platforms
        available = {
            p.get("platform") for p in integrations_data.get("availablePlatforms", integrations_data.get("available_platforms", []))
        }
        listed = {p.get("platform") for p in platforms_data.get("platforms", [])}
        assert listed.issubset(available), f"Platforms missing from integration list: {listed - available}"

    @pytest.mark.asyncio
    async def test_notification_preferences_consistent_with_unread_count(self):
        """Verify notification preferences and unread count are both accessible."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            prefs_resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/preferences", headers=HEADERS
            )
            unread_resp = await client.get(
                f"{BASE_URL}/api/v1/notifications/unread-count", headers=HEADERS
            )
        if prefs_resp.status_code == 429 or unread_resp.status_code == 429:
            return
        assert prefs_resp.status_code == 200
        assert unread_resp.status_code == 200

        prefs = prefs_resp.json()
        unread = unread_resp.json()

        # Verify both are well-formed
        NotificationPrefsModel.model_validate(prefs)
        UnreadCountModel.model_validate(unread)

        # Unread count should be non-negative
        count = unread.get("unread_count") or unread.get("unreadCount", 0)
        assert count >= 0
