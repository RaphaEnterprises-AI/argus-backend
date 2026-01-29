"""LogRocket Integration Plugin for session replay.

LogRocket provides session replay and error tracking for web applications.
This plugin enables syncing sessions, issues, and user data from LogRocket.

API Docs: https://docs.logrocket.com/reference/introduction
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog

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

logger = structlog.get_logger()


@dataclass
class LogRocketSession:
    """A LogRocket session replay."""

    id: str
    url: str
    user_id: str | None
    user_email: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int
    page_count: int
    event_count: int
    error_count: int
    rage_click_count: int
    device_type: str | None
    browser: str | None
    os: str | None
    country: str | None
    tags: list[str] = field(default_factory=list)
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogRocketIssue:
    """A LogRocket issue (grouped errors)."""

    id: str
    title: str
    type: str  # error, rage_click, dead_click, etc.
    status: str
    first_seen: datetime
    last_seen: datetime
    session_count: int
    user_count: int
    occurrence_count: int
    sample_session_url: str | None
    tags: list[str] = field(default_factory=list)


@dataclass
class LogRocketUser:
    """A LogRocket identified user."""

    id: str
    email: str | None
    name: str | None
    first_seen: datetime
    last_seen: datetime
    session_count: int
    custom_properties: dict[str, Any] = field(default_factory=dict)


class LogRocketPlugin(IntegrationPlugin):
    """LogRocket integration plugin for session replay data.

    Features:
    - Sync session replays with filters
    - Sync issues (grouped errors, rage clicks, etc.)
    - Sync identified users
    - Test connection via organization endpoint

    Usage:
        async with LogRocketPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://api.logrocket.com/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get LogRocket integration metadata."""
        return IntegrationMetadata(
            slug="logrocket",
            name="LogRocket",
            description="Session replay and error tracking for web applications",
            category=IntegrationCategory.SESSION_REPLAY,
            auth_type=AuthType.API_KEY,
            icon_url="https://logrocket.com/static/logrocket-icon.svg",
            docs_url="https://docs.logrocket.com/reference/introduction",
            website_url="https://logrocket.com",
            features=[
                "session_replay",
                "error_tracking",
                "rage_click_detection",
                "user_identification",
                "performance_monitoring",
                "console_logging",
                "network_logging",
                "redux_state_tracking",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="your-api-key",
                    help_text="Found in LogRocket Settings > API > API Key",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization Slug",
                    type="text",
                    required=False,
                    placeholder="your-org",
                    help_text="Your LogRocket organization slug",
                ),
                FieldDefinition(
                    name="extra",
                    label="App ID",
                    type="text",
                    required=False,
                    placeholder="your-app-id",
                    help_text="Filter sessions by specific app ID",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["sessions", "issues", "users"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for LogRocket API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test LogRocket connection by fetching organization info.

        Args:
            credentials: The credentials to test.

        Returns:
            ConnectionTestResult with success status and details.
        """
        log = logger.bind(plugin="logrocket", action="test_connection")

        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="API key is required",
                error_code="MISSING_API_KEY",
            )

        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/orgs",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                orgs = data.get("orgs", [])

                log.info(
                    "LogRocket connection successful",
                    org_count=len(orgs),
                    latency_ms=latency_ms,
                )

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to LogRocket with {len(orgs)} organization(s)",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["read:sessions", "read:issues", "read:users"],
                    account_info={
                        "organizations": [
                            {"slug": org.get("slug"), "name": org.get("name")}
                            for org in orgs
                        ],
                    },
                )

            elif response.status_code == 401:
                log.warning("LogRocket authentication failed")
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            elif response.status_code == 403:
                log.warning("LogRocket access forbidden")
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have sufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            else:
                log.error(
                    "LogRocket connection failed",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code=f"HTTP_{response.status_code}",
                    error_details={"response": response.text[:500]},
                )

        except httpx.TimeoutException:
            log.error("LogRocket connection timed out")
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except Exception as e:
            log.error("LogRocket connection error", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from LogRocket.

        Args:
            credentials: The credentials to use.
            resources: Optional list of resources to sync (sessions, issues, users).
            since: Optional datetime to sync data since.
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        log = logger.bind(plugin="logrocket", action="sync_data")

        started_at = datetime.utcnow()
        sync_resources = resources or ["sessions", "issues", "users"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        log.info("Starting LogRocket sync", resources=sync_resources, limit=limit)

        # Get organization slug from credentials or fetch first org
        org_slug = credentials.org_slug
        if not org_slug:
            org_result = await self._get_default_org(credentials)
            if org_result:
                org_slug = org_result
            else:
                return SyncResult(
                    success=False,
                    message="Could not determine organization",
                    errors=["No organization slug provided and could not fetch default"],
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

        app_id = credentials.extra.get("app_id") if credentials.extra else None

        # Sync sessions
        if "sessions" in sync_resources:
            try:
                sessions = await self._sync_sessions(
                    credentials, org_slug, app_id, since, limit
                )
                data["sessions"] = [self._session_to_dict(s) for s in sessions]
                total_synced += len(sessions)
                log.info("Synced LogRocket sessions", count=len(sessions))
            except Exception as e:
                errors.append(f"Failed to sync sessions: {str(e)}")
                log.error("Failed to sync sessions", error=str(e))

        # Sync issues
        if "issues" in sync_resources:
            try:
                issues = await self._sync_issues(
                    credentials, org_slug, app_id, since, limit
                )
                data["issues"] = [self._issue_to_dict(i) for i in issues]
                total_synced += len(issues)
                log.info("Synced LogRocket issues", count=len(issues))
            except Exception as e:
                errors.append(f"Failed to sync issues: {str(e)}")
                log.error("Failed to sync issues", error=str(e))

        # Sync users
        if "users" in sync_resources:
            try:
                users = await self._sync_users(
                    credentials, org_slug, app_id, since, limit
                )
                data["users"] = [self._user_to_dict(u) for u in users]
                total_synced += len(users)
                log.info("Synced LogRocket users", count=len(users))
            except Exception as e:
                errors.append(f"Failed to sync users: {str(e)}")
                log.error("Failed to sync users", error=str(e))

        completed_at = datetime.utcnow()
        success = len(errors) == 0

        return SyncResult(
            success=success,
            message=f"Synced {total_synced} items from LogRocket"
            if success
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _get_default_org(
        self, credentials: IntegrationCredentials
    ) -> str | None:
        """Get the default organization slug."""
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/orgs",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                orgs = response.json().get("orgs", [])
                if orgs:
                    return orgs[0].get("slug")
        except Exception:
            pass
        return None

    async def _sync_sessions(
        self,
        credentials: IntegrationCredentials,
        org_slug: str,
        app_id: str | None,
        since: datetime | None,
        limit: int,
    ) -> list[LogRocketSession]:
        """Sync sessions from LogRocket."""
        sessions = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["start"] = int(since.timestamp() * 1000)

        if app_id:
            params["appId"] = app_id

        url = f"{self.BASE_URL}/orgs/{org_slug}/sessions"

        response = await self.http_client.get(
            url,
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for session_data in data.get("sessions", []):
            session = self._parse_session(session_data)
            if session:
                sessions.append(session)

            if len(sessions) >= limit:
                break

        return sessions

    async def _sync_issues(
        self,
        credentials: IntegrationCredentials,
        org_slug: str,
        app_id: str | None,
        since: datetime | None,
        limit: int,
    ) -> list[LogRocketIssue]:
        """Sync issues from LogRocket."""
        issues = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["updatedAfter"] = int(since.timestamp() * 1000)

        if app_id:
            params["appId"] = app_id

        url = f"{self.BASE_URL}/orgs/{org_slug}/issues"

        response = await self.http_client.get(
            url,
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for issue_data in data.get("issues", []):
            issue = self._parse_issue(issue_data)
            if issue:
                issues.append(issue)

            if len(issues) >= limit:
                break

        return issues

    async def _sync_users(
        self,
        credentials: IntegrationCredentials,
        org_slug: str,
        app_id: str | None,
        since: datetime | None,
        limit: int,
    ) -> list[LogRocketUser]:
        """Sync identified users from LogRocket."""
        users = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["lastSeenAfter"] = int(since.timestamp() * 1000)

        if app_id:
            params["appId"] = app_id

        url = f"{self.BASE_URL}/orgs/{org_slug}/users"

        response = await self.http_client.get(
            url,
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for user_data in data.get("users", []):
            user = self._parse_user(user_data)
            if user:
                users.append(user)

            if len(users) >= limit:
                break

        return users

    def _parse_session(self, data: dict[str, Any]) -> LogRocketSession | None:
        """Parse a session from API response."""
        try:
            started_at = datetime.fromtimestamp(data.get("startedAt", 0) / 1000)
            ended_at = None
            if data.get("endedAt"):
                ended_at = datetime.fromtimestamp(data["endedAt"] / 1000)

            return LogRocketSession(
                id=data.get("id", ""),
                url=data.get("url", ""),
                user_id=data.get("userId"),
                user_email=data.get("userEmail"),
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=data.get("duration", 0),
                page_count=data.get("pageCount", 0),
                event_count=data.get("eventCount", 0),
                error_count=data.get("errorCount", 0),
                rage_click_count=data.get("rageClickCount", 0),
                device_type=data.get("deviceType"),
                browser=data.get("browser"),
                os=data.get("os"),
                country=data.get("country"),
                tags=data.get("tags", []),
                custom_properties=data.get("customProperties", {}),
            )
        except Exception:
            return None

    def _parse_issue(self, data: dict[str, Any]) -> LogRocketIssue | None:
        """Parse an issue from API response."""
        try:
            first_seen = datetime.fromtimestamp(data.get("firstSeen", 0) / 1000)
            last_seen = datetime.fromtimestamp(data.get("lastSeen", 0) / 1000)

            return LogRocketIssue(
                id=data.get("id", ""),
                title=data.get("title", ""),
                type=data.get("type", "error"),
                status=data.get("status", "open"),
                first_seen=first_seen,
                last_seen=last_seen,
                session_count=data.get("sessionCount", 0),
                user_count=data.get("userCount", 0),
                occurrence_count=data.get("occurrenceCount", 0),
                sample_session_url=data.get("sampleSessionUrl"),
                tags=data.get("tags", []),
            )
        except Exception:
            return None

    def _parse_user(self, data: dict[str, Any]) -> LogRocketUser | None:
        """Parse a user from API response."""
        try:
            first_seen = datetime.fromtimestamp(data.get("firstSeen", 0) / 1000)
            last_seen = datetime.fromtimestamp(data.get("lastSeen", 0) / 1000)

            return LogRocketUser(
                id=data.get("id", ""),
                email=data.get("email"),
                name=data.get("name"),
                first_seen=first_seen,
                last_seen=last_seen,
                session_count=data.get("sessionCount", 0),
                custom_properties=data.get("customProperties", {}),
            )
        except Exception:
            return None

    def _session_to_dict(self, session: LogRocketSession) -> dict[str, Any]:
        """Convert a session to a dictionary."""
        return {
            "id": session.id,
            "url": session.url,
            "user_id": session.user_id,
            "user_email": session.user_email,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_ms": session.duration_ms,
            "page_count": session.page_count,
            "event_count": session.event_count,
            "error_count": session.error_count,
            "rage_click_count": session.rage_click_count,
            "device_type": session.device_type,
            "browser": session.browser,
            "os": session.os,
            "country": session.country,
            "tags": session.tags,
            "custom_properties": session.custom_properties,
        }

    def _issue_to_dict(self, issue: LogRocketIssue) -> dict[str, Any]:
        """Convert an issue to a dictionary."""
        return {
            "id": issue.id,
            "title": issue.title,
            "type": issue.type,
            "status": issue.status,
            "first_seen": issue.first_seen.isoformat(),
            "last_seen": issue.last_seen.isoformat(),
            "session_count": issue.session_count,
            "user_count": issue.user_count,
            "occurrence_count": issue.occurrence_count,
            "sample_session_url": issue.sample_session_url,
            "tags": issue.tags,
        }

    def _user_to_dict(self, user: LogRocketUser) -> dict[str, Any]:
        """Convert a user to a dictionary."""
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "first_seen": user.first_seen.isoformat(),
            "last_seen": user.last_seen.isoformat(),
            "session_count": user.session_count,
            "custom_properties": user.custom_properties,
        }

    # Additional helper methods for session replay specific operations

    async def get_session_details(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
    ) -> LogRocketSession | None:
        """Get detailed information about a specific session.

        Args:
            credentials: The credentials to use.
            session_id: The session ID to fetch.

        Returns:
            LogRocketSession or None if not found.
        """
        org_slug = credentials.org_slug
        if not org_slug:
            org_slug = await self._get_default_org(credentials)
            if not org_slug:
                return None

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/orgs/{org_slug}/sessions/{session_id}",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                return self._parse_session(response.json())
        except Exception:
            pass

        return None

    async def get_session_events(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get events from a specific session.

        Args:
            credentials: The credentials to use.
            session_id: The session ID.
            event_types: Optional list of event types to filter.

        Returns:
            List of event dictionaries.
        """
        org_slug = credentials.org_slug
        if not org_slug:
            org_slug = await self._get_default_org(credentials)
            if not org_slug:
                return []

        params: dict[str, Any] = {}
        if event_types:
            params["types"] = ",".join(event_types)

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/orgs/{org_slug}/sessions/{session_id}/events",
                headers=self._get_headers(credentials),
                params=params,
            )
            if response.status_code == 200:
                return response.json().get("events", [])
        except Exception:
            pass

        return []

    async def search_sessions(
        self,
        credentials: IntegrationCredentials,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[LogRocketSession]:
        """Search sessions with advanced filters.

        Args:
            credentials: The credentials to use.
            query: Search query string.
            filters: Additional filters (e.g., has_errors, user_id, etc.).
            limit: Maximum number of results.

        Returns:
            List of matching sessions.
        """
        org_slug = credentials.org_slug
        if not org_slug:
            org_slug = await self._get_default_org(credentials)
            if not org_slug:
                return []

        params: dict[str, Any] = {
            "q": query,
            "limit": min(limit, 100),
        }

        if filters:
            params.update(filters)

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/orgs/{org_slug}/sessions/search",
                headers=self._get_headers(credentials),
                params=params,
            )
            if response.status_code == 200:
                sessions = []
                for session_data in response.json().get("sessions", []):
                    session = self._parse_session(session_data)
                    if session:
                        sessions.append(session)
                return sessions
        except Exception:
            pass

        return []


def create_logrocket_plugin(
    http_client: httpx.AsyncClient | None = None,
) -> LogRocketPlugin:
    """Factory function for LogRocketPlugin.

    Args:
        http_client: Optional HTTP client to use.

    Returns:
        LogRocketPlugin instance.
    """
    return LogRocketPlugin(http_client=http_client)
