"""Sentry integration plugin.

This module provides integration with Sentry for error tracking and monitoring.
Sentry is an application monitoring platform that helps developers monitor
and fix crashes in real-time.

Features:
- Fetch issues, events, and releases
- Real-time alert webhooks
- Project and organization management
- Performance monitoring data
"""

import time
from datetime import datetime
from typing import Any, Optional

import httpx

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)


class SentryPlugin(IntegrationPlugin):
    """Sentry integration for error tracking and monitoring.

    Sentry provides real-time error tracking that gives developers insight into
    production deployments and helps them fix crashes in real-time.

    Authentication:
        Uses Bearer token authentication with an Auth Token from Sentry.
        Token can be created at: https://sentry.io/settings/account/api/auth-tokens/

    Required Permissions:
        - project:read (for reading project data)
        - event:read (for reading events)
        - org:read (for reading organization data)
        - issue:read (for reading issues)
        - release:read (for reading releases)

    Example:
        async with SentryPlugin() as plugin:
            credentials = IntegrationCredentials(
                access_token="your-auth-token",
                org_slug="your-org",
                project_slug="your-project",
            )
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://sentry.io/api/0"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Sentry integration metadata."""
        return IntegrationMetadata(
            slug="sentry",
            name="Sentry",
            description="Application monitoring and error tracking platform",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://sentry.io/favicon.ico",
            docs_url="https://docs.sentry.io/api/",
            website_url="https://sentry.io",
            features=[
                "error_tracking",
                "performance_monitoring",
                "release_tracking",
                "issue_management",
                "alert_webhooks",
                "custom_integrations",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Auth Token",
                    type="password",
                    required=True,
                    placeholder="sntrys_...",
                    help_text="Sentry Auth Token. Create at Settings > Account > API > Auth Tokens",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Organization Slug",
                    type="text",
                    required=True,
                    placeholder="my-organization",
                    help_text="Your Sentry organization slug (from URL: sentry.io/organizations/{slug}/)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project Slug",
                    type="text",
                    required=False,
                    placeholder="my-project",
                    help_text="Specific project to sync (leave empty for all projects)",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Self-Hosted URL",
                    type="url",
                    required=False,
                    placeholder="https://sentry.mycompany.com",
                    help_text="For self-hosted Sentry instances (leave empty for sentry.io)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["issues", "events", "releases", "projects"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for API calls."""
        if credentials.instance_url:
            return f"{credentials.instance_url.rstrip('/')}/api/0"
        return self.BASE_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Sentry connection.

        Validates the auth token by fetching organization details.

        Args:
            credentials: Must contain access_token and org_slug.

        Returns:
            ConnectionTestResult with organization info if successful.
        """
        is_valid, missing = self.validate_credentials(credentials)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=f"Missing required fields: {', '.join(missing)}",
                error_code="MISSING_CREDENTIALS",
            )

        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/organizations/{credentials.org_slug}/"

        start_time = time.time()
        try:
            response = await self.http_client.get(url, headers=headers)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid auth token",
                    latency_ms=latency_ms,
                    error_code="INVALID_TOKEN",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Insufficient permissions. Ensure token has org:read scope.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message=f"Organization '{credentials.org_slug}' not found",
                    latency_ms=latency_ms,
                    error_code="ORG_NOT_FOUND",
                )

            response.raise_for_status()
            org_data = response.json()

            # Fetch available projects if no specific project is set
            projects = []
            if not credentials.project_slug:
                projects_url = f"{base_url}/organizations/{credentials.org_slug}/projects/"
                projects_response = await self.http_client.get(
                    projects_url, headers=headers
                )
                if projects_response.status_code == 200:
                    projects = [p["slug"] for p in projects_response.json()]

            # Extract permissions from token (if available in response)
            permissions = []
            if "access" in org_data:
                permissions = org_data.get("access", [])

            return ConnectionTestResult(
                success=True,
                message=f"Connected to organization: {org_data.get('name', credentials.org_slug)}",
                latency_ms=latency_ms,
                api_version="0",
                permissions=permissions,
                account_info={
                    "organization_name": org_data.get("name"),
                    "organization_slug": org_data.get("slug"),
                    "organization_id": org_data.get("id"),
                    "date_created": org_data.get("dateCreated"),
                    "projects": projects,
                    "is_early_adopter": org_data.get("isEarlyAdopter", False),
                    "require_2fa": org_data.get("require2FA", False),
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Sentry.

        Fetches issues, events, and releases from Sentry projects.

        Args:
            credentials: Must contain access_token, org_slug, and optionally project_slug.
            resources: List of resources to sync: "issues", "events", "releases", "projects".
            since: Only fetch data since this datetime.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or self.metadata.sync_resources
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Determine which projects to sync
        project_slugs: list[str] = []
        if credentials.project_slug:
            project_slugs = [credentials.project_slug]
        elif "projects" in sync_resources:
            # Fetch all projects for the organization
            projects_url = f"{base_url}/organizations/{credentials.org_slug}/projects/"
            try:
                response = await self.http_client.get(projects_url, headers=headers)
                response.raise_for_status()
                project_slugs = [p["slug"] for p in response.json()]
            except Exception as e:
                errors.append(f"Failed to fetch projects: {str(e)}")

        # Sync projects
        if "projects" in sync_resources:
            try:
                projects_data = await self._sync_projects(
                    credentials, base_url, headers, limit
                )
                data["projects"] = projects_data
                total_synced += len(projects_data)
            except Exception as e:
                errors.append(f"Failed to sync projects: {str(e)}")

        # Sync issues
        if "issues" in sync_resources:
            try:
                issues_data = await self._sync_issues(
                    credentials, base_url, headers, project_slugs, since, limit
                )
                data["issues"] = issues_data
                total_synced += len(issues_data)
            except Exception as e:
                errors.append(f"Failed to sync issues: {str(e)}")

        # Sync events (latest events per project)
        if "events" in sync_resources:
            try:
                events_data = await self._sync_events(
                    credentials, base_url, headers, project_slugs, since, limit
                )
                data["events"] = events_data
                total_synced += len(events_data)
            except Exception as e:
                errors.append(f"Failed to sync events: {str(e)}")

        # Sync releases
        if "releases" in sync_resources:
            try:
                releases_data = await self._sync_releases(
                    credentials, base_url, headers, since, limit
                )
                data["releases"] = releases_data
                total_synced += len(releases_data)
            except Exception as e:
                errors.append(f"Failed to sync releases: {str(e)}")

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from Sentry"
            if len(errors) == 0
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_projects(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync projects from Sentry."""
        url = f"{base_url}/organizations/{credentials.org_slug}/projects/"
        response = await self.http_client.get(
            url, headers=headers, params={"per_page": min(limit, 100)}
        )
        response.raise_for_status()

        projects = response.json()
        return [
            {
                "id": p.get("id"),
                "slug": p.get("slug"),
                "name": p.get("name"),
                "platform": p.get("platform"),
                "date_created": p.get("dateCreated"),
                "is_bookmarked": p.get("isBookmarked", False),
                "has_access": p.get("hasAccess", True),
                "status": p.get("status"),
            }
            for p in projects[:limit]
        ]

    async def _sync_issues(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        headers: dict[str, str],
        project_slugs: list[str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync issues from Sentry."""
        issues: list[dict[str, Any]] = []

        # Build query parameters
        params: dict[str, Any] = {
            "per_page": min(limit, 100),
            "query": "is:unresolved",
        }

        if since:
            params["start"] = since.isoformat()

        # Fetch issues for organization (across all projects or specific ones)
        url = f"{base_url}/organizations/{credentials.org_slug}/issues/"

        if project_slugs:
            params["project"] = project_slugs

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        for issue in response.json()[:limit]:
            issues.append(
                {
                    "id": issue.get("id"),
                    "short_id": issue.get("shortId"),
                    "title": issue.get("title"),
                    "culprit": issue.get("culprit"),
                    "level": issue.get("level"),
                    "status": issue.get("status"),
                    "platform": issue.get("platform"),
                    "project": {
                        "id": issue.get("project", {}).get("id"),
                        "slug": issue.get("project", {}).get("slug"),
                        "name": issue.get("project", {}).get("name"),
                    },
                    "type": issue.get("type"),
                    "metadata": issue.get("metadata", {}),
                    "count": issue.get("count"),
                    "user_count": issue.get("userCount"),
                    "first_seen": issue.get("firstSeen"),
                    "last_seen": issue.get("lastSeen"),
                    "is_bookmarked": issue.get("isBookmarked", False),
                    "is_subscribed": issue.get("isSubscribed", False),
                    "has_seen": issue.get("hasSeen", False),
                    "annotations": issue.get("annotations", []),
                    "permalink": issue.get("permalink"),
                }
            )

        return issues

    async def _sync_events(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        headers: dict[str, str],
        project_slugs: list[str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync events from Sentry using Discover API."""
        events: list[dict[str, Any]] = []

        # Use Discover API to fetch events
        url = f"{base_url}/organizations/{credentials.org_slug}/events/"

        params: dict[str, Any] = {
            "per_page": min(limit, 100),
            "field": [
                "id",
                "title",
                "project",
                "timestamp",
                "event.type",
                "platform",
                "message",
                "user.email",
            ],
            "sort": "-timestamp",
        }

        if project_slugs:
            params["project"] = project_slugs

        if since:
            params["start"] = since.isoformat()

        try:
            response = await self.http_client.get(url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()

            for event in result.get("data", [])[:limit]:
                events.append(
                    {
                        "id": event.get("id"),
                        "title": event.get("title"),
                        "project": event.get("project"),
                        "timestamp": event.get("timestamp"),
                        "event_type": event.get("event.type"),
                        "platform": event.get("platform"),
                        "message": event.get("message"),
                        "user_email": event.get("user.email"),
                    }
                )
        except httpx.HTTPStatusError:
            # Discover API might not be available, try legacy endpoint
            if project_slugs:
                for project_slug in project_slugs[:5]:  # Limit to 5 projects
                    legacy_url = f"{base_url}/projects/{credentials.org_slug}/{project_slug}/events/"
                    try:
                        response = await self.http_client.get(
                            legacy_url,
                            headers=headers,
                            params={"per_page": min(limit // len(project_slugs), 100)},
                        )
                        response.raise_for_status()
                        for event in response.json():
                            events.append(
                                {
                                    "id": event.get("eventID"),
                                    "title": event.get("title"),
                                    "project": project_slug,
                                    "timestamp": event.get("dateReceived"),
                                    "event_type": event.get("type"),
                                    "platform": event.get("platform"),
                                    "message": event.get("message"),
                                    "tags": event.get("tags", []),
                                }
                            )
                    except Exception:
                        continue

        return events[:limit]

    async def _sync_releases(
        self,
        credentials: IntegrationCredentials,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync releases from Sentry."""
        url = f"{base_url}/organizations/{credentials.org_slug}/releases/"

        params: dict[str, Any] = {"per_page": min(limit, 100)}

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        releases: list[dict[str, Any]] = []
        for release in response.json()[:limit]:
            release_date = release.get("dateCreated")
            if since and release_date:
                release_datetime = datetime.fromisoformat(
                    release_date.replace("Z", "+00:00")
                )
                if release_datetime.replace(tzinfo=None) < since:
                    continue

            releases.append(
                {
                    "version": release.get("version"),
                    "short_version": release.get("shortVersion"),
                    "date_created": release.get("dateCreated"),
                    "date_released": release.get("dateReleased"),
                    "first_event": release.get("firstEvent"),
                    "last_event": release.get("lastEvent"),
                    "new_groups": release.get("newGroups"),
                    "projects": [
                        {"id": p.get("id"), "slug": p.get("slug"), "name": p.get("name")}
                        for p in release.get("projects", [])
                    ],
                    "commit_count": release.get("commitCount"),
                    "deploy_count": release.get("deployCount"),
                    "status": release.get("status"),
                    "url": release.get("url"),
                }
            )

        return releases

    async def get_issue_details(
        self,
        credentials: IntegrationCredentials,
        issue_id: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific issue.

        Args:
            credentials: Sentry credentials.
            issue_id: The Sentry issue ID.

        Returns:
            Detailed issue data including tags, events count, and activity.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/issues/{issue_id}/"

        response = await self.http_client.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    async def get_issue_events(
        self,
        credentials: IntegrationCredentials,
        issue_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get events for a specific issue.

        Args:
            credentials: Sentry credentials.
            issue_id: The Sentry issue ID.
            limit: Maximum number of events to fetch.

        Returns:
            List of events for the issue.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        url = f"{base_url}/issues/{issue_id}/events/"

        response = await self.http_client.get(
            url, headers=headers, params={"per_page": min(limit, 100)}
        )
        response.raise_for_status()

        return response.json()

    async def configure_webhook(
        self,
        credentials: IntegrationCredentials,
        webhook_url: str,
        events: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Configure a webhook for Sentry alerts.

        Note: This creates an internal integration with webhook capabilities.

        Args:
            credentials: Sentry credentials with org:write permissions.
            webhook_url: URL to receive webhook events.
            events: List of event types to subscribe to. Defaults to all.

        Returns:
            Webhook configuration details.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        default_events = events or [
            "issue.created",
            "issue.resolved",
            "issue.ignored",
            "issue.assigned",
            "error.created",
        ]

        # Create an internal integration with webhook
        url = f"{base_url}/organizations/{credentials.org_slug}/sentry-app-installations/"

        payload = {
            "name": "Argus Webhook",
            "webhookUrl": webhook_url,
            "events": default_events,
            "isAlertable": True,
        }

        response = await self.http_client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()

    def parse_webhook_payload(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Parse and validate a Sentry webhook payload.

        Args:
            payload: The webhook payload.
            headers: Request headers for signature validation.

        Returns:
            Parsed event data with normalized structure.
        """
        action = payload.get("action")
        data = payload.get("data", {})
        actor = payload.get("actor", {})

        # Map Sentry event types to normalized types
        event_type_map = {
            "created": "issue.created",
            "resolved": "issue.resolved",
            "ignored": "issue.ignored",
            "assigned": "issue.assigned",
        }

        normalized_type = event_type_map.get(action, f"sentry.{action}")

        issue_data = data.get("issue", {})

        return {
            "event_type": normalized_type,
            "timestamp": payload.get("timestamp"),
            "actor": {
                "type": actor.get("type"),
                "id": actor.get("id"),
                "name": actor.get("name"),
            },
            "issue": {
                "id": issue_data.get("id"),
                "short_id": issue_data.get("shortId"),
                "title": issue_data.get("title"),
                "culprit": issue_data.get("culprit"),
                "level": issue_data.get("level"),
                "status": issue_data.get("status"),
                "project": issue_data.get("project", {}).get("slug"),
                "permalink": issue_data.get("permalink"),
            },
            "raw_payload": payload,
        }
