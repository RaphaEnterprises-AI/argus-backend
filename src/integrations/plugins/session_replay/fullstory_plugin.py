"""FullStory Integration Plugin for session replay.

FullStory provides digital experience intelligence including session replay,
product analytics, and user insights. This plugin enables syncing sessions,
events, users, and segments from FullStory.

API Docs: https://developer.fullstory.com/
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
class FullStorySession:
    """A FullStory session replay."""

    id: str
    user_id: str | None
    created_time: datetime
    updated_time: datetime
    replay_url: str
    page_count: int
    event_count: int
    duration_ms: int
    browser: str | None
    browser_version: str | None
    device_type: str | None
    operating_system: str | None
    operating_system_version: str | None
    country: str | None
    city: str | None
    app_id: str | None
    app_version: str | None
    # Session quality metrics
    frustration_signals: int = 0
    rage_clicks: int = 0
    error_clicks: int = 0
    dead_clicks: int = 0
    thrashed_cursor: int = 0
    custom_vars: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullStoryEvent:
    """A FullStory event."""

    id: str
    session_id: str
    user_id: str | None
    event_type: str
    event_name: str
    timestamp: datetime
    page_url: str | None
    page_title: str | None
    element_selector: str | None
    element_text: str | None
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullStoryUser:
    """A FullStory identified user."""

    uid: str
    display_name: str | None
    email: str | None
    created_time: datetime
    updated_time: datetime
    num_sessions: int
    first_seen_time: datetime | None
    last_seen_time: datetime | None
    custom_vars: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullStorySegment:
    """A FullStory segment (saved filter)."""

    id: str
    name: str
    description: str | None
    type: str  # user, session
    created_time: datetime
    updated_time: datetime
    created_by: str | None
    filter_expression: str | None
    user_count: int | None
    session_count: int | None


class FullStoryPlugin(IntegrationPlugin):
    """FullStory integration plugin for session replay and analytics.

    Features:
    - Sync session replays
    - Sync events (clicks, pages, custom events)
    - Sync identified users
    - Sync segments (saved filters)
    - Test connection via users endpoint

    Usage:
        async with FullStoryPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://api.fullstory.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get FullStory integration metadata."""
        return IntegrationMetadata(
            slug="fullstory",
            name="FullStory",
            description="Digital experience intelligence with session replay and analytics",
            category=IntegrationCategory.SESSION_REPLAY,
            auth_type=AuthType.API_KEY,
            icon_url="https://www.fullstory.com/favicon.ico",
            docs_url="https://developer.fullstory.com/",
            website_url="https://www.fullstory.com",
            features=[
                "session_replay",
                "event_tracking",
                "user_identification",
                "frustration_signals",
                "rage_click_detection",
                "dead_click_detection",
                "error_click_detection",
                "segment_management",
                "funnel_analysis",
                "journey_mapping",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Key",
                    type="password",
                    required=True,
                    placeholder="your-api-key",
                    help_text="Found in FullStory Settings > Integrations > API Keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization ID",
                    type="text",
                    required=False,
                    placeholder="your-org-id",
                    help_text="Your FullStory organization ID (for multi-org accounts)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["sessions", "events", "users", "segments"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for FullStory API requests."""
        return {
            "Authorization": f"Basic {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test FullStory connection by fetching user info.

        Args:
            credentials: The credentials to test.

        Returns:
            ConnectionTestResult with success status and details.
        """
        log = logger.bind(plugin="fullstory", action="test_connection")

        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="API key is required",
                error_code="MISSING_API_KEY",
            )

        start_time = time.time()

        try:
            # Use the users endpoint to test connection
            response = await self.http_client.get(
                f"{self.BASE_URL}/users/v1/individual",
                headers=self._get_headers(credentials),
                params={"limit": 1},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                log.info(
                    "FullStory connection successful",
                    latency_ms=latency_ms,
                )

                return ConnectionTestResult(
                    success=True,
                    message="Connected to FullStory successfully",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=[
                        "read:sessions",
                        "read:events",
                        "read:users",
                        "read:segments",
                    ],
                    account_info={
                        "org_id": credentials.org_slug,
                    },
                )

            elif response.status_code == 401:
                log.warning("FullStory authentication failed")
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            elif response.status_code == 403:
                log.warning("FullStory access forbidden")
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have sufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            else:
                log.error(
                    "FullStory connection failed",
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
            log.error("FullStory connection timed out")
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except Exception as e:
            log.error("FullStory connection error", error=str(e))
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
        """Sync data from FullStory.

        Args:
            credentials: The credentials to use.
            resources: Optional list of resources to sync (sessions, events, users, segments).
            since: Optional datetime to sync data since.
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        log = logger.bind(plugin="fullstory", action="sync_data")

        started_at = datetime.utcnow()
        sync_resources = resources or ["sessions", "events", "users", "segments"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        log.info("Starting FullStory sync", resources=sync_resources, limit=limit)

        # Sync sessions
        if "sessions" in sync_resources:
            try:
                sessions = await self._sync_sessions(credentials, since, limit)
                data["sessions"] = [self._session_to_dict(s) for s in sessions]
                total_synced += len(sessions)
                log.info("Synced FullStory sessions", count=len(sessions))
            except Exception as e:
                errors.append(f"Failed to sync sessions: {str(e)}")
                log.error("Failed to sync sessions", error=str(e))

        # Sync events
        if "events" in sync_resources:
            try:
                events = await self._sync_events(credentials, since, limit)
                data["events"] = [self._event_to_dict(e) for e in events]
                total_synced += len(events)
                log.info("Synced FullStory events", count=len(events))
            except Exception as e:
                errors.append(f"Failed to sync events: {str(e)}")
                log.error("Failed to sync events", error=str(e))

        # Sync users
        if "users" in sync_resources:
            try:
                users = await self._sync_users(credentials, since, limit)
                data["users"] = [self._user_to_dict(u) for u in users]
                total_synced += len(users)
                log.info("Synced FullStory users", count=len(users))
            except Exception as e:
                errors.append(f"Failed to sync users: {str(e)}")
                log.error("Failed to sync users", error=str(e))

        # Sync segments
        if "segments" in sync_resources:
            try:
                segments = await self._sync_segments(credentials, limit)
                data["segments"] = [self._segment_to_dict(s) for s in segments]
                total_synced += len(segments)
                log.info("Synced FullStory segments", count=len(segments))
            except Exception as e:
                errors.append(f"Failed to sync segments: {str(e)}")
                log.error("Failed to sync segments", error=str(e))

        completed_at = datetime.utcnow()
        success = len(errors) == 0

        return SyncResult(
            success=success,
            message=f"Synced {total_synced} items from FullStory"
            if success
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_sessions(
        self,
        credentials: IntegrationCredentials,
        since: datetime | None,
        limit: int,
    ) -> list[FullStorySession]:
        """Sync sessions from FullStory."""
        sessions = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["start"] = since.isoformat() + "Z"

        response = await self.http_client.get(
            f"{self.BASE_URL}/sessions/v1",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for session_data in data.get("sessions", data.get("results", [])):
            session = self._parse_session(session_data)
            if session:
                sessions.append(session)

            if len(sessions) >= limit:
                break

        return sessions

    async def _sync_events(
        self,
        credentials: IntegrationCredentials,
        since: datetime | None,
        limit: int,
    ) -> list[FullStoryEvent]:
        """Sync events from FullStory."""
        events = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["start"] = since.isoformat() + "Z"

        response = await self.http_client.get(
            f"{self.BASE_URL}/events/v1",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for event_data in data.get("events", data.get("results", [])):
            event = self._parse_event(event_data)
            if event:
                events.append(event)

            if len(events) >= limit:
                break

        return events

    async def _sync_users(
        self,
        credentials: IntegrationCredentials,
        since: datetime | None,
        limit: int,
    ) -> list[FullStoryUser]:
        """Sync identified users from FullStory."""
        users = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["updatedAfter"] = since.isoformat() + "Z"

        response = await self.http_client.get(
            f"{self.BASE_URL}/users/v1/individual",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for user_data in data.get("users", data.get("results", [])):
            user = self._parse_user(user_data)
            if user:
                users.append(user)

            if len(users) >= limit:
                break

        return users

    async def _sync_segments(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[FullStorySegment]:
        """Sync segments from FullStory."""
        segments = []

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        response = await self.http_client.get(
            f"{self.BASE_URL}/segments/v1",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for segment_data in data.get("segments", data.get("results", [])):
            segment = self._parse_segment(segment_data)
            if segment:
                segments.append(segment)

            if len(segments) >= limit:
                break

        return segments

    def _parse_session(self, data: dict[str, Any]) -> FullStorySession | None:
        """Parse a session from API response."""
        try:
            created_time = datetime.fromisoformat(
                data.get("CreatedTime", data.get("createdTime", "")).replace("Z", "+00:00")
            )
            updated_time = datetime.fromisoformat(
                data.get("UpdatedTime", data.get("updatedTime", "")).replace("Z", "+00:00")
            )

            return FullStorySession(
                id=data.get("SessionId", data.get("sessionId", data.get("id", ""))),
                user_id=data.get("UserId", data.get("userId")),
                created_time=created_time,
                updated_time=updated_time,
                replay_url=data.get("FsUrl", data.get("fsUrl", data.get("replayUrl", ""))),
                page_count=data.get("PageCount", data.get("pageCount", 0)),
                event_count=data.get("EventCount", data.get("eventCount", 0)),
                duration_ms=data.get("Duration", data.get("duration", 0)),
                browser=data.get("Browser", data.get("browser")),
                browser_version=data.get("BrowserVersion", data.get("browserVersion")),
                device_type=data.get("DeviceType", data.get("deviceType")),
                operating_system=data.get("OperatingSystem", data.get("operatingSystem")),
                operating_system_version=data.get(
                    "OperatingSystemVersion", data.get("operatingSystemVersion")
                ),
                country=data.get("Country", data.get("country")),
                city=data.get("City", data.get("city")),
                app_id=data.get("AppId", data.get("appId")),
                app_version=data.get("AppVersion", data.get("appVersion")),
                frustration_signals=data.get(
                    "FrustrationSignals", data.get("frustrationSignals", 0)
                ),
                rage_clicks=data.get("RageClicks", data.get("rageClicks", 0)),
                error_clicks=data.get("ErrorClicks", data.get("errorClicks", 0)),
                dead_clicks=data.get("DeadClicks", data.get("deadClicks", 0)),
                thrashed_cursor=data.get("ThrashedCursor", data.get("thrashedCursor", 0)),
                custom_vars=data.get("CustomVars", data.get("customVars", {})),
            )
        except Exception:
            return None

    def _parse_event(self, data: dict[str, Any]) -> FullStoryEvent | None:
        """Parse an event from API response."""
        try:
            timestamp = datetime.fromisoformat(
                data.get("Timestamp", data.get("timestamp", "")).replace("Z", "+00:00")
            )

            return FullStoryEvent(
                id=data.get("EventId", data.get("eventId", data.get("id", ""))),
                session_id=data.get("SessionId", data.get("sessionId", "")),
                user_id=data.get("UserId", data.get("userId")),
                event_type=data.get("EventType", data.get("eventType", "")),
                event_name=data.get("EventName", data.get("eventName", "")),
                timestamp=timestamp,
                page_url=data.get("PageUrl", data.get("pageUrl")),
                page_title=data.get("PageTitle", data.get("pageTitle")),
                element_selector=data.get("ElementSelector", data.get("elementSelector")),
                element_text=data.get("ElementText", data.get("elementText")),
                custom_properties=data.get(
                    "CustomProperties", data.get("customProperties", {})
                ),
            )
        except Exception:
            return None

    def _parse_user(self, data: dict[str, Any]) -> FullStoryUser | None:
        """Parse a user from API response."""
        try:
            created_time = datetime.fromisoformat(
                data.get("CreatedTime", data.get("createdTime", "")).replace("Z", "+00:00")
            )
            updated_time = datetime.fromisoformat(
                data.get("UpdatedTime", data.get("updatedTime", "")).replace("Z", "+00:00")
            )

            first_seen = None
            if data.get("FirstSeenTime", data.get("firstSeenTime")):
                first_seen = datetime.fromisoformat(
                    data.get("FirstSeenTime", data.get("firstSeenTime")).replace(
                        "Z", "+00:00"
                    )
                )

            last_seen = None
            if data.get("LastSeenTime", data.get("lastSeenTime")):
                last_seen = datetime.fromisoformat(
                    data.get("LastSeenTime", data.get("lastSeenTime")).replace(
                        "Z", "+00:00"
                    )
                )

            return FullStoryUser(
                uid=data.get("Uid", data.get("uid", data.get("id", ""))),
                display_name=data.get("DisplayName", data.get("displayName")),
                email=data.get("Email", data.get("email")),
                created_time=created_time,
                updated_time=updated_time,
                num_sessions=data.get("NumSessions", data.get("numSessions", 0)),
                first_seen_time=first_seen,
                last_seen_time=last_seen,
                custom_vars=data.get("CustomVars", data.get("customVars", {})),
            )
        except Exception:
            return None

    def _parse_segment(self, data: dict[str, Any]) -> FullStorySegment | None:
        """Parse a segment from API response."""
        try:
            created_time = datetime.fromisoformat(
                data.get("CreatedTime", data.get("createdTime", "")).replace("Z", "+00:00")
            )
            updated_time = datetime.fromisoformat(
                data.get("UpdatedTime", data.get("updatedTime", "")).replace("Z", "+00:00")
            )

            return FullStorySegment(
                id=data.get("Id", data.get("id", "")),
                name=data.get("Name", data.get("name", "")),
                description=data.get("Description", data.get("description")),
                type=data.get("Type", data.get("type", "session")),
                created_time=created_time,
                updated_time=updated_time,
                created_by=data.get("CreatedBy", data.get("createdBy")),
                filter_expression=data.get(
                    "FilterExpression", data.get("filterExpression")
                ),
                user_count=data.get("UserCount", data.get("userCount")),
                session_count=data.get("SessionCount", data.get("sessionCount")),
            )
        except Exception:
            return None

    def _session_to_dict(self, session: FullStorySession) -> dict[str, Any]:
        """Convert a session to a dictionary."""
        return {
            "id": session.id,
            "user_id": session.user_id,
            "created_time": session.created_time.isoformat(),
            "updated_time": session.updated_time.isoformat(),
            "replay_url": session.replay_url,
            "page_count": session.page_count,
            "event_count": session.event_count,
            "duration_ms": session.duration_ms,
            "browser": session.browser,
            "browser_version": session.browser_version,
            "device_type": session.device_type,
            "operating_system": session.operating_system,
            "operating_system_version": session.operating_system_version,
            "country": session.country,
            "city": session.city,
            "app_id": session.app_id,
            "app_version": session.app_version,
            "frustration_signals": session.frustration_signals,
            "rage_clicks": session.rage_clicks,
            "error_clicks": session.error_clicks,
            "dead_clicks": session.dead_clicks,
            "thrashed_cursor": session.thrashed_cursor,
            "custom_vars": session.custom_vars,
        }

    def _event_to_dict(self, event: FullStoryEvent) -> dict[str, Any]:
        """Convert an event to a dictionary."""
        return {
            "id": event.id,
            "session_id": event.session_id,
            "user_id": event.user_id,
            "event_type": event.event_type,
            "event_name": event.event_name,
            "timestamp": event.timestamp.isoformat(),
            "page_url": event.page_url,
            "page_title": event.page_title,
            "element_selector": event.element_selector,
            "element_text": event.element_text,
            "custom_properties": event.custom_properties,
        }

    def _user_to_dict(self, user: FullStoryUser) -> dict[str, Any]:
        """Convert a user to a dictionary."""
        return {
            "uid": user.uid,
            "display_name": user.display_name,
            "email": user.email,
            "created_time": user.created_time.isoformat(),
            "updated_time": user.updated_time.isoformat(),
            "num_sessions": user.num_sessions,
            "first_seen_time": user.first_seen_time.isoformat()
            if user.first_seen_time
            else None,
            "last_seen_time": user.last_seen_time.isoformat()
            if user.last_seen_time
            else None,
            "custom_vars": user.custom_vars,
        }

    def _segment_to_dict(self, segment: FullStorySegment) -> dict[str, Any]:
        """Convert a segment to a dictionary."""
        return {
            "id": segment.id,
            "name": segment.name,
            "description": segment.description,
            "type": segment.type,
            "created_time": segment.created_time.isoformat(),
            "updated_time": segment.updated_time.isoformat(),
            "created_by": segment.created_by,
            "filter_expression": segment.filter_expression,
            "user_count": segment.user_count,
            "session_count": segment.session_count,
        }

    # Additional helper methods for session replay specific operations

    async def get_session_by_id(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
    ) -> FullStorySession | None:
        """Get detailed information about a specific session.

        Args:
            credentials: The credentials to use.
            session_id: The session ID to fetch.

        Returns:
            FullStorySession or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/v1/{session_id}",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                return self._parse_session(response.json())
        except Exception:
            pass

        return None

    async def get_user_by_uid(
        self,
        credentials: IntegrationCredentials,
        uid: str,
    ) -> FullStoryUser | None:
        """Get detailed information about a specific user.

        Args:
            credentials: The credentials to use.
            uid: The user ID to fetch.

        Returns:
            FullStoryUser or None if not found.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/users/v1/individual/{uid}",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                return self._parse_user(response.json())
        except Exception:
            pass

        return None

    async def get_sessions_with_frustration(
        self,
        credentials: IntegrationCredentials,
        min_signals: int = 1,
        limit: int = 50,
    ) -> list[FullStorySession]:
        """Get sessions that have frustration signals.

        Args:
            credentials: The credentials to use.
            min_signals: Minimum number of frustration signals.
            limit: Maximum number of results.

        Returns:
            List of sessions with frustration signals.
        """
        sessions = await self._sync_sessions(credentials, None, limit * 2)

        # Filter sessions with frustration signals
        return [
            s
            for s in sessions
            if s.frustration_signals >= min_signals
            or s.rage_clicks > 0
            or s.dead_clicks > 0
            or s.error_clicks > 0
        ][:limit]

    async def get_segment_sessions(
        self,
        credentials: IntegrationCredentials,
        segment_id: str,
        limit: int = 50,
    ) -> list[FullStorySession]:
        """Get sessions belonging to a specific segment.

        Args:
            credentials: The credentials to use.
            segment_id: The segment ID.
            limit: Maximum number of results.

        Returns:
            List of sessions in the segment.
        """
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/segments/v1/{segment_id}/sessions",
                headers=self._get_headers(credentials),
                params={"limit": min(limit, 100)},
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


def create_fullstory_plugin(
    http_client: httpx.AsyncClient | None = None,
) -> FullStoryPlugin:
    """Factory function for FullStoryPlugin.

    Args:
        http_client: Optional HTTP client to use.

    Returns:
        FullStoryPlugin instance.
    """
    return FullStoryPlugin(http_client=http_client)
