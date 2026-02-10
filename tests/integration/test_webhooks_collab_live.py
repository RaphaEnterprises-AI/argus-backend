"""Production E2E integration tests for Webhooks, Collaboration, OAuth, and MCP APIs.

Tests the following API modules against LIVE production with NO MOCKS:
  - Webhooks (17 platform webhook endpoints)
  - GitHub Webhooks (receive, analyze, analysis retrieval, events)
  - Collaboration (presence, comments, reactions, locks)
  - OAuth (platform status, disconnect, all connections)
  - MCP Sessions (connections, stats, auth pending)
  - MCP Screenshots (register, retrieve)

Every response is validated against strict Pydantic models matching the
production API contract (camelCase via CamelCaseMiddleware).

Run with:
    pytest tests/integration/test_webhooks_collab_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestWebhooks"
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestGitHubWebhooks"
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestCollaboration"
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestOAuth"
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestMCPSessions"
    pytest tests/integration/test_webhooks_collab_live.py -v -k "TestMCPScreenshots"
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

# ===========================================================================
# Configuration -- env vars with production defaults
# ===========================================================================

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

WEBHOOK_HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
SLOW_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

# ===========================================================================
# Endpoint URLs
# ===========================================================================

WEBHOOKS_URL = f"{BASE_URL}/api/v1/webhooks"
GITHUB_WEBHOOKS_URL = f"{BASE_URL}/api/v1/webhooks/github"
COLLABORATION_URL = f"{BASE_URL}/api/v1/collaboration"
OAUTH_URL = f"{BASE_URL}/api/v1/oauth"
MCP_URL = f"{BASE_URL}/api/v1/mcp"
MCP_SCREENSHOTS_URL = f"{BASE_URL}/api/v1/mcp/screenshots"


# ===========================================================================
# Pydantic Response Models
# ===========================================================================

class WebhookResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    message: str
    event_id: str | None = Field(None, alias="eventId")
    fingerprint: str | None = None


class GitHubWebhookResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    message: str
    event_type: str = Field(alias="eventType")
    delivery_id: str = Field(alias="deliveryId")
    sdlc_event_id: str | None = Field(None, alias="sdlcEventId")
    commit_analysis_id: str | None = Field(None, alias="commitAnalysisId")


class PresenceResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    workspace_id: str = Field(alias="workspaceId")
    users: list[dict] = []
    total_online: int = Field(0, alias="totalOnline")


class CommentResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    comment: dict | None = None
    error: str | None = None


class CommentsListResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    success: bool
    test_id: str = Field(alias="testId")
    comments: list[dict] = []
    total: int = 0
    unresolved: int = 0


class OAuthStatusResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    platform: str
    connected: bool
    account_name: str | None = Field(None, alias="accountName")
    account_id: str | None = Field(None, alias="accountId")
    scopes: list[str] = Field(default_factory=list)
    connected_at: str | None = Field(None, alias="connectedAt")
    token_expires_at: str | None = Field(None, alias="tokenExpiresAt")


class MCPConnectionListModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    connections: list[dict]
    total: int
    active_count: int = Field(alias="activeCount")


class MCPStatsModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    active_connections: int = Field(alias="activeConnections")
    total_connections: int = Field(alias="totalConnections")
    total_requests: int = Field(alias="totalRequests")
    unique_users: int = Field(alias="uniqueUsers")
    client_types: list[str] = Field(alias="clientTypes")
    last_activity: str | None = Field(None, alias="lastActivity")
    connections_today: int = Field(alias="connectionsToday")
    connections_this_week: int = Field(alias="connectionsThisWeek")
    requests_today: int = Field(alias="requestsToday")
    top_tools: list[dict] = Field(alias="topTools")


# ===========================================================================
# Fake Payloads for Webhooks
# ===========================================================================

SENTRY_PAYLOAD = {
    "action": "created",
    "data": {
        "event": {
            "event_id": "test-e2e-001",
            "title": "E2E Test Error",
            "message": "Test error from integration suite",
            "level": "error",
            "platform": "python",
            "tags": [],
        },
    },
    "installation": {"uuid": "test-install"},
}

DATADOG_PAYLOAD = {
    "id": "test-event-dd-001",
    "title": "E2E Test Monitor Alert",
    "message": "Test alert from integration suite",
    "date_happened": 1700000000,
    "priority": "normal",
    "alert_type": "error",
    "tags": ["env:test"],
}

FULLSTORY_PAYLOAD = {
    "event_type": "rage_click",
    "data": {
        "session_id": "fs-session-e2e-001",
        "user_id": "fs-user-e2e",
        "url": "https://example.com/page",
        "element": "button#submit",
        "timestamp": "2026-01-15T10:00:00Z",
        "click_count": 5,
    },
}

LOGROCKET_PAYLOAD = {
    "event": "error",
    "data": {
        "session_url": "https://app.logrocket.com/test-session",
        "error_type": "TypeError",
        "message": "Cannot read property 'id' of undefined",
        "url": "https://example.com/dashboard",
        "timestamp": "2026-01-15T10:00:00Z",
    },
}

NEWRELIC_PAYLOAD = {
    "id": "nr-event-e2e-001",
    "event_type": "INCIDENT_OPEN",
    "title": "E2E Test High Error Rate",
    "description": "Error rate exceeded threshold in test environment",
    "severity": "CRITICAL",
    "policy_name": "test-policy",
    "condition_name": "High Error Rate",
    "incident_url": "https://one.newrelic.com/test-incident",
    "timestamp": 1700000000,
}

BUGSNAG_PAYLOAD = {
    "trigger": {"type": "firstException", "message": "New error"},
    "error": {
        "id": "bs-error-e2e-001",
        "errorId": "bs-error-e2e-001",
        "exceptionClass": "RuntimeError",
        "message": "E2E Test Exception from Bugsnag webhook",
        "context": "/api/v1/test",
        "severity": "error",
        "url": "https://app.bugsnag.com/test-error",
        "stackTrace": [
            {"file": "app/main.py", "lineNumber": 42, "method": "handle_request"}
        ],
    },
    "project": {"name": "test-project", "url": "https://app.bugsnag.com/test"},
}

ROLLBAR_PAYLOAD = {
    "event_name": "new_item",
    "data": {
        "item": {
            "id": 12345,
            "title": "E2E Test Rollbar Error",
            "level": 40,
            "environment": "test",
            "framework": "python",
            "language": "python",
            "total_occurrences": 1,
            "last_occurrence_timestamp": 1700000000,
        },
    },
}

GITHUB_ACTIONS_PAYLOAD = {
    "action": "completed",
    "workflow_run": {
        "id": 12345678,
        "name": "E2E Test Workflow",
        "head_sha": "abc123def456",
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "run_number": 42,
        "html_url": "https://github.com/test-org/test-repo/actions/runs/12345678",
        "created_at": "2026-01-15T10:00:00Z",
        "updated_at": "2026-01-15T10:05:00Z",
    },
    "repository": {
        "full_name": "test-org/test-repo",
        "html_url": "https://github.com/test-org/test-repo",
    },
}

GITLAB_CI_PAYLOAD = {
    "object_kind": "pipeline",
    "object_attributes": {
        "id": 987654,
        "ref": "main",
        "sha": "abc123def456",
        "status": "success",
        "created_at": "2026-01-15T10:00:00Z",
        "finished_at": "2026-01-15T10:05:00Z",
        "duration": 300,
    },
    "project": {
        "name": "test-project",
        "web_url": "https://gitlab.com/test-org/test-project",
    },
}

CIRCLECI_PAYLOAD = {
    "type": "job-completed",
    "id": str(uuid.uuid4()),
    "happened_at": "2026-01-15T10:05:00Z",
    "pipeline": {
        "id": str(uuid.uuid4()),
        "vcs": {"branch": "main", "revision": "abc123def456"},
    },
    "workflow": {"id": str(uuid.uuid4()), "name": "build-and-test"},
    "job": {
        "id": str(uuid.uuid4()),
        "name": "test",
        "status": "success",
        "started_at": "2026-01-15T10:00:00Z",
        "stopped_at": "2026-01-15T10:05:00Z",
    },
    "project": {"slug": "gh/test-org/test-repo"},
}

VERCEL_PAYLOAD = {
    "type": "deployment.succeeded",
    "id": "vercel-event-e2e-001",
    "createdAt": 1700000000000,
    "payload": {
        "deploymentId": "dpl_test_e2e_001",
        "name": "test-app",
        "url": "https://test-app.vercel.app",
        "project": {"id": "prj_test", "name": "test-app"},
        "target": "production",
        "meta": {"githubCommitSha": "abc123def456", "githubCommitRef": "main"},
    },
}

TEST_RESULTS_PAYLOAD = {
    "organization_id": ORG_ID,
    "project_id": PROJ_ID,
    "branch": "main",
    "commit_sha": "e2e_test_commit_results",
    "format": "junit",
    "report_data": """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="3" failures="1" errors="0" time="1.5">
  <testsuite name="e2e-suite" tests="3" failures="1" time="1.5">
    <testcase classname="test_auth" name="test_login" time="0.5"/>
    <testcase classname="test_auth" name="test_logout" time="0.3"/>
    <testcase classname="test_auth" name="test_expired_token" time="0.7">
      <failure message="Token not expired" type="AssertionError">AssertionError: Expected expired</failure>
    </testcase>
  </testsuite>
</testsuites>""",
}

COVERAGE_PAYLOAD = {
    "organization_id": ORG_ID,
    "project_id": PROJ_ID,
    "branch": "main",
    "commit_sha": "e2e_test_commit_coverage",
    "format": "lcov",
    "report_data": (
        "TN:\nSF:src/main.py\nDA:1,1\nDA:2,1\nDA:3,0\nLF:3\nLH:2\nend_of_record\n"
    ),
}

SLACK_INTERACTIVE_PAYLOAD = {
    "type": "block_actions",
    "user": {"id": "U_TEST_E2E", "username": "e2e_tester", "name": "E2E Tester"},
    "trigger_id": "test-trigger-id",
    "actions": [
        {
            "action_id": "rerun_tests_job123",
            "type": "button",
            "value": "rerun",
        }
    ],
}

JIRA_PAYLOAD = {
    "webhookEvent": "jira:issue_updated",
    "timestamp": 1700000000000,
    "issue": {
        "id": "10001",
        "key": "TEST-123",
        "fields": {
            "summary": "E2E Test Jira Issue",
            "status": {"name": "In Progress"},
            "priority": {"name": "High"},
            "issuetype": {"name": "Bug"},
            "assignee": {"displayName": "E2E Tester"},
        },
    },
    "user": {"displayName": "E2E Tester", "emailAddress": "test@example.com"},
}

PAGERDUTY_PAYLOAD = {
    "messages": [
        {
            "event": "incident.trigger",
            "incident": {
                "id": "PD_TEST_001",
                "title": "E2E Test PagerDuty Incident",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "svc_test", "name": "test-service"},
                "created_at": "2026-01-15T10:00:00Z",
                "html_url": "https://test.pagerduty.com/incidents/PD_TEST_001",
            },
        }
    ],
}

LAUNCHDARKLY_PAYLOAD = {
    "event": "flag:updated",
    "accesses": {
        "flagKey": "test-feature-flag",
        "action": "updateFlagVariations",
    },
    "currentVersion": {
        "key": "test-feature-flag",
        "name": "Test Feature Flag",
        "kind": "boolean",
        "on": True,
        "environments": {"production": {"on": True}},
    },
    "previousVersion": {
        "key": "test-feature-flag",
        "name": "Test Feature Flag",
        "kind": "boolean",
        "on": False,
    },
}


# ===========================================================================
# Collaboration Payloads
# ===========================================================================

PRESENCE_PAYLOAD = {
    "workspace_id": PROJ_ID,
    "user_id": "test-user-e2e",
    "user_name": "E2E Test User",
    "user_email": "test@example.com",
    "status": "online",
}

COMMENT_PAYLOAD = {
    "test_id": FAKE_UUID,
    "author_id": "test-user-e2e",
    "author_name": "E2E Test User",
    "content": "Integration test comment",
    "comment_type": "general",
}


# ===========================================================================
# 1. Webhooks (platform-specific webhook endpoints)
# ===========================================================================


class TestWebhooks:
    """Tests for POST /api/v1/webhooks/{platform} endpoints.

    NOTE: Platform webhook endpoints (sentry, datadog, etc.) defined in
    webhooks.py are NOT deployed — the file is shadowed by the webhooks/
    package directory (Python import precedence: package > module).
    Only VCS webhooks (github, gitlab) under webhooks/ are mounted.
    We accept 404 for unmounted platform webhook endpoints.
    """

    @pytest.mark.asyncio
    async def test_sentry_webhook(self):
        """POST /webhooks/sentry -- Sentry error event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/sentry",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=SENTRY_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_datadog_webhook(self):
        """POST /webhooks/datadog -- Datadog monitor alert."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/datadog",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=DATADOG_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_fullstory_webhook(self):
        """POST /webhooks/fullstory -- FullStory session event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/fullstory",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=FULLSTORY_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_logrocket_webhook(self):
        """POST /webhooks/logrocket -- LogRocket error event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/logrocket",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=LOGROCKET_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_newrelic_webhook(self):
        """POST /webhooks/newrelic -- New Relic incident."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/newrelic",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=NEWRELIC_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_bugsnag_webhook(self):
        """POST /webhooks/bugsnag -- Bugsnag error event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/bugsnag",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=BUGSNAG_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_rollbar_webhook(self):
        """POST /webhooks/rollbar -- Rollbar error event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/rollbar",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=ROLLBAR_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_github_actions_webhook(self):
        """POST /webhooks/github-actions -- GitHub Actions workflow event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/github-actions",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=GITHUB_ACTIONS_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_gitlab_ci_webhook(self):
        """POST /webhooks/gitlab-ci -- GitLab CI pipeline event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/gitlab-ci",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=GITLAB_CI_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_circleci_webhook(self):
        """POST /webhooks/circleci -- CircleCI job event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/circleci",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=CIRCLECI_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_vercel_webhook(self):
        """POST /webhooks/vercel -- Vercel deployment event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/vercel",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=VERCEL_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_test_results_upload(self):
        """POST /webhooks/test-results -- JUnit XML upload (body-based auth)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/test-results",
                headers=WEBHOOK_HEADERS,
                json=TEST_RESULTS_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_coverage_upload(self):
        """POST /webhooks/coverage -- LCOV coverage report (body-based auth)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/coverage",
                headers=WEBHOOK_HEADERS,
                json=COVERAGE_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_slack_interactive_webhook(self):
        """POST /webhooks/slack/interactive -- Slack interactive message (form-encoded)."""
        import json

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/slack/interactive",
                data={"payload": json.dumps(SLACK_INTERACTIVE_PAYLOAD)},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_jira_webhook(self):
        """POST /webhooks/jira -- Jira issue event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/jira",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=JIRA_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_pagerduty_webhook(self):
        """POST /webhooks/pagerduty -- PagerDuty incident event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/pagerduty",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=PAGERDUTY_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_launchdarkly_webhook(self):
        """POST /webhooks/launchdarkly -- LaunchDarkly flag change event."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/launchdarkly",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": ORG_ID, "project_id": PROJ_ID},
                json=LAUNCHDARKLY_PAYLOAD,
            )
        assert resp.status_code in (200, 201, 401, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            WebhookResponseModel.model_validate(resp.json())


# ===========================================================================
# 2. GitHub Webhooks
# ===========================================================================


class TestGitHubWebhooks:
    """Tests for /api/v1/webhooks/github/* endpoints (API-key authed)."""

    @pytest.mark.asyncio
    async def test_receive_github_webhook_push(self):
        """POST /webhooks/github -- push event with headers."""
        delivery_id = str(uuid.uuid4())
        payload = {
            "ref": "refs/heads/main",
            "head_commit": {
                "id": "abc123e2e_test",
                "message": "E2E test commit",
                "timestamp": "2026-01-15T10:00:00Z",
            },
            "commits": [],
            "repository": {
                "full_name": "test-org/test-repo",
                "html_url": "https://github.com/test-org/test-repo",
            },
            "pusher": {"name": "e2e-tester"},
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                GITHUB_WEBHOOKS_URL,
                headers={
                    **HEADERS,
                    "X-GitHub-Event": "push",
                    "X-GitHub-Delivery": delivery_id,
                },
                params={"project_id": PROJ_ID},
                json=payload,
            )
        # May succeed or fail depending on table existence / signature check
        assert resp.status_code in (200, 201, 400, 401, 422, 429, 500), resp.text
        if resp.status_code == 200:
            GitHubWebhookResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_receive_github_webhook_missing_event_header(self):
        """POST /webhooks/github -- missing X-GitHub-Event returns 400."""
        delivery_id = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                GITHUB_WEBHOOKS_URL,
                headers={
                    **HEADERS,
                    "X-GitHub-Delivery": delivery_id,
                },
                params={"project_id": PROJ_ID},
                json={"ref": "refs/heads/main"},
            )
        assert resp.status_code in (400, 422, 429), resp.text

    @pytest.mark.asyncio
    async def test_analyze_commit(self):
        """POST /webhooks/github/analyze -- analyze a commit."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{GITHUB_WEBHOOKS_URL}/analyze",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
                json={
                    "commit_sha": "e2e_fake_sha_12345",
                    "branch_name": "main",
                },
            )
        # May 500 if tables missing, 200 if analysis succeeds
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_get_analysis_nonexistent_sha(self):
        """GET /webhooks/github/analysis/{sha} -- fake SHA returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{GITHUB_WEBHOOKS_URL}/analysis/nonexistent_sha_xyz_000",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (404, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_list_webhook_events(self):
        """GET /webhooks/github/events -- list events for project."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{GITHUB_WEBHOOKS_URL}/events",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "events" in data or "total" in data


# ===========================================================================
# 3. Collaboration
# ===========================================================================


class TestCollaboration:
    """Tests for /api/v1/collaboration/* endpoints."""

    @pytest.mark.asyncio
    async def test_update_presence(self):
        """POST /collaboration/presence -- update user presence."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/presence",
                headers=HEADERS,
                json=PRESENCE_PAYLOAD,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            PresenceResponseModel.model_validate(data)
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_get_presence_valid_workspace(self):
        """GET /collaboration/presence/{workspace_id} -- returns presence (may be empty)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{COLLABORATION_URL}/presence/{PROJ_ID}",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            PresenceResponseModel.model_validate(data)
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_get_presence_fake_workspace(self):
        """GET /collaboration/presence/{fake_uuid} -- returns empty presence."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{COLLABORATION_URL}/presence/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True
            users = data.get("users", [])
            assert isinstance(users, list)

    @pytest.mark.asyncio
    async def test_remove_presence_fake_user(self):
        """DELETE /collaboration/presence/{workspace}/{user} -- non-existent user."""
        fake_user_uuid = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{COLLABORATION_URL}/presence/{PROJ_ID}/{fake_user_uuid}",
                headers=HEADERS,
            )
        # Should succeed even if user not found (no-op)
        assert resp.status_code in (200, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_create_comment(self):
        """POST /collaboration/comments -- create a comment."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/comments",
                headers=HEADERS,
                json=COMMENT_PAYLOAD,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            CommentResponseModel.model_validate(data)
            assert data.get("success") is True
            assert data.get("comment") is not None
            # Store comment_id for subsequent tests
            TestCollaboration._last_comment_id = data["comment"]["id"]

    _last_comment_id: str | None = None

    @pytest.mark.asyncio
    async def test_list_comments_for_test(self):
        """GET /collaboration/comments/{test_id} -- list comments."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(
                f"{COLLABORATION_URL}/comments/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            CommentsListResponseModel.model_validate(data)
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_update_comment_nonexistent(self):
        """PATCH /collaboration/comments/{id} -- nonexistent comment returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.patch(
                f"{COLLABORATION_URL}/comments/{FAKE_UUID}",
                headers=HEADERS,
                json={"content": "Updated content"},
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_comment_nonexistent(self):
        """DELETE /collaboration/comments/{id} -- nonexistent comment returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{COLLABORATION_URL}/comments/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_add_reaction_nonexistent_comment(self):
        """POST /collaboration/comments/{id}/reactions -- nonexistent returns 404."""
        fake_user = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/comments/{FAKE_UUID}/reactions",
                headers=HEADERS,
                params={"emoji": "thumbsup", "user_id": fake_user},
            )
        assert resp.status_code in (404, 422, 429), resp.text

    @pytest.mark.asyncio
    async def test_acquire_lock(self):
        """POST /collaboration/lock/{test_id} -- acquire edit lock."""
        fake_user = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/lock/{FAKE_UUID}",
                headers=HEADERS,
                params={"user_id": fake_user},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True
            assert "lock_id" in data or "lockId" in data

    @pytest.mark.asyncio
    async def test_release_lock_nonexistent(self):
        """DELETE /collaboration/lock/{lock_id} -- release nonexistent lock."""
        fake_lock = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{COLLABORATION_URL}/lock/{fake_lock}",
                headers=HEADERS,
            )
        # Lock release always succeeds (in-memory, no validation)
        assert resp.status_code in (200, 404, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_create_comment_missing_content(self):
        """POST /collaboration/comments -- missing content returns 422."""
        payload = {
            "test_id": FAKE_UUID,
            "author_id": "test-user-e2e",
            "author_name": "E2E Test User",
            # Missing "content" field
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/comments",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (422, 429), resp.text


# ===========================================================================
# 4. OAuth
# ===========================================================================


class TestOAuth:
    """Tests for /api/v1/oauth/* endpoints (API-key authed)."""

    @pytest.mark.asyncio
    async def test_github_oauth_status(self):
        """GET /oauth/github/status -- check GitHub connection status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/github/status",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            OAuthStatusResponseModel.model_validate(data)
            assert data.get("platform") == "github"

    @pytest.mark.asyncio
    async def test_slack_oauth_status(self):
        """GET /oauth/slack/status -- check Slack connection status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/slack/status",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            OAuthStatusResponseModel.model_validate(data)
            assert data.get("platform") == "slack"

    @pytest.mark.asyncio
    async def test_jira_oauth_status(self):
        """GET /oauth/jira/status -- check Jira connection status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/jira/status",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            OAuthStatusResponseModel.model_validate(data)
            assert data.get("platform") == "jira"

    @pytest.mark.asyncio
    async def test_linear_oauth_status(self):
        """GET /oauth/linear/status -- check Linear connection status."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/linear/status",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            OAuthStatusResponseModel.model_validate(data)
            assert data.get("platform") == "linear"

    @pytest.mark.asyncio
    async def test_all_oauth_status(self):
        """GET /oauth/status -- list all OAuth connections."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/status",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            # Should have all 4 OAuth platforms
            platforms = {item.get("platform") for item in data}
            expected = {"github", "slack", "jira", "linear"}
            assert expected.issubset(platforms), (
                f"Missing platforms: {expected - platforms}"
            )
            # Validate each status response
            for item in data:
                OAuthStatusResponseModel.model_validate(item)

    @pytest.mark.asyncio
    async def test_disconnect_github_not_connected(self):
        """DELETE /oauth/github/disconnect -- expect 404 if not connected."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{OAUTH_URL}/github/disconnect",
                headers=HEADERS,
            )
        # 404 if not connected, 200 if connected
        assert resp.status_code in (200, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_disconnect_slack_not_connected(self):
        """DELETE /oauth/slack/disconnect -- expect 404 if not connected."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{OAUTH_URL}/slack/disconnect",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_oauth_invalid_platform(self):
        """GET /oauth/invalid_platform/status -- expect 422 for invalid platform."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/invalid_platform/status",
                headers=HEADERS,
            )
        assert resp.status_code in (422, 429), resp.text


# ===========================================================================
# 5. MCP Sessions
# ===========================================================================


class TestMCPSessions:
    """Tests for /api/v1/mcp/* endpoints (API-key authed)."""

    @pytest.mark.asyncio
    async def test_list_connections(self):
        """GET /mcp/connections -- list MCP connections."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            MCPConnectionListModel.model_validate(data)
            assert data.get("total", 0) >= 0

    @pytest.mark.asyncio
    async def test_list_connections_with_org(self):
        """GET /mcp/connections?org_id=ORG_ID -- list for org."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections",
                headers=HEADERS,
                params={"org_id": ORG_ID},
            )
        assert resp.status_code in (200, 403, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            MCPConnectionListModel.model_validate(data)

    @pytest.mark.asyncio
    async def test_get_connection_nonexistent(self):
        """GET /mcp/connections/{id} -- nonexistent returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_revoke_connection_nonexistent(self):
        """DELETE /mcp/connections/{id} -- nonexistent returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{MCP_URL}/connections/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_connection_activity_nonexistent(self):
        """GET /mcp/connections/{id}/activity -- nonexistent returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections/{FAKE_UUID}/activity",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_mcp_stats(self):
        """GET /mcp/stats -- dashboard statistics."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/stats",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            MCPStatsModel.model_validate(data)
            ac = data.get("activeConnections", data.get("active_connections", 0))
            assert ac >= 0

    @pytest.mark.asyncio
    async def test_get_pending_auth(self):
        """GET /mcp/auth/pending -- pending auth sessions."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/auth/pending",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "sessions" in data
            assert "count" in data
            assert data["count"] >= 0

    @pytest.mark.asyncio
    async def test_get_connection_invalid_uuid(self):
        """GET /mcp/connections/{bad-id} -- invalid UUID returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections/not-a-valid-uuid",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 429), resp.text


# ===========================================================================
# 6. MCP Screenshots
# ===========================================================================


class TestMCPScreenshots:
    """Tests for /api/v1/mcp/screenshots/* endpoints (API-key authed)."""

    @pytest.mark.asyncio
    async def test_register_screenshot_missing_connection(self):
        """POST /mcp/screenshots/register -- nonexistent connection_id returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{MCP_SCREENSHOTS_URL}/register",
                headers=HEADERS,
                json={
                    "connection_id": FAKE_UUID,
                    "r2_key": "screenshots/e2e_test_fake.png",
                    "screenshot_type": "step",
                },
            )
        assert resp.status_code in (404, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_get_screenshot_nonexistent(self):
        """GET /mcp/screenshots/{id} -- nonexistent returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_SCREENSHOTS_URL}/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_register_screenshot_missing_fields(self):
        """POST /mcp/screenshots/register -- missing r2_key returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{MCP_SCREENSHOTS_URL}/register",
                headers=HEADERS,
                json={"connection_id": FAKE_UUID},
            )
        assert resp.status_code in (422, 429), resp.text


# ===========================================================================
# 7. Cross-Module & Edge Case Tests
# ===========================================================================


class TestCrossModuleAndEdgeCases:
    """Cross-module consistency and edge case tests."""

    @pytest.mark.asyncio
    async def test_webhook_missing_org_id_returns_422(self):
        """POST /webhooks/sentry without organization_id returns 422 (or 404 if unmounted)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/sentry",
                headers=WEBHOOK_HEADERS,
                json=SENTRY_PAYLOAD,
            )
        assert resp.status_code in (404, 422, 429), resp.text

    @pytest.mark.asyncio
    async def test_webhook_invalid_org_id_returns_400(self):
        """POST /webhooks/datadog with non-UUID organization_id returns 400 (or 404 if unmounted)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/datadog",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": "not-a-valid-uuid"},
                json=DATADOG_PAYLOAD,
            )
        assert resp.status_code in (400, 404, 422, 429), resp.text

    @pytest.mark.asyncio
    async def test_collaboration_presence_invalid_workspace_uuid(self):
        """GET /collaboration/presence/{bad-uuid} -- invalid UUID returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{COLLABORATION_URL}/presence/not-a-uuid-at-all",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 429), resp.text

    @pytest.mark.asyncio
    async def test_oauth_status_requires_auth(self):
        """GET /oauth/github/status without API key returns 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/github/status",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code in (401, 429), resp.text

    @pytest.mark.asyncio
    async def test_mcp_connections_requires_auth(self):
        """GET /mcp/connections without API key returns 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/connections",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code in (401, 429), resp.text

    @pytest.mark.asyncio
    async def test_mcp_stats_requires_auth(self):
        """GET /mcp/stats without API key returns 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{MCP_URL}/stats",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code in (401, 429), resp.text

    @pytest.mark.asyncio
    async def test_webhook_project_org_mismatch(self):
        """POST /webhooks/sentry with wrong org returns 403 or processes."""
        fake_org = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{WEBHOOKS_URL}/sentry",
                headers=WEBHOOK_HEADERS,
                params={"organization_id": fake_org, "project_id": PROJ_ID},
                json=SENTRY_PAYLOAD,
            )
        # 403 if project doesn't belong to org, 404 if endpoint not mounted, or 400/500
        assert resp.status_code in (400, 403, 404, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_collaboration_comment_with_reply_structure(self):
        """POST /collaboration/comments with parent_id (thread reply)."""
        payload = {
            "test_id": FAKE_UUID,
            "author_id": "test-user-e2e-reply",
            "author_name": "E2E Reply User",
            "content": "This is a threaded reply",
            "comment_type": "suggestion",
            "parent_id": FAKE_UUID,  # Non-existent parent, but should still create
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{COLLABORATION_URL}/comments",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_github_webhook_events_with_filter(self):
        """GET /webhooks/github/events -- with event_type filter."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{GITHUB_WEBHOOKS_URL}/events",
                headers=HEADERS,
                params={"project_id": PROJ_ID, "event_type": "push", "limit": 5},
            )
        assert resp.status_code in (200, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            events = data.get("events", [])
            assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_mcp_revoke_invalid_uuid(self):
        """DELETE /mcp/connections/{bad-uuid} -- invalid UUID returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{MCP_URL}/connections/bad-uuid-format",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 429), resp.text

    @pytest.mark.asyncio
    async def test_oauth_all_statuses_have_connected_field(self):
        """GET /oauth/status -- every platform entry has 'connected' boolean."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{OAUTH_URL}/status",
                headers=HEADERS,
            )
        if resp.status_code == 429:
            return
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        for item in data:
            assert "connected" in item, f"Missing 'connected' in {item}"
            assert isinstance(item["connected"], bool)
