"""
BrowserStack Integration Plugin.

BrowserStack is a cloud-based cross-browser testing platform that provides
instant access to real browsers and devices for testing web applications.

API Docs: https://www.browserstack.com/docs/automate/api-reference/

Features:
- Cross-browser testing on real browsers and devices
- Automated testing with Selenium/Playwright
- Session management and logs
- Build organization and tracking
- Screenshots, videos, and console logs
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

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


class SessionStatus(str, Enum):
    """BrowserStack session status."""
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    TIMEOUT = "timeout"


class BuildStatus(str, Enum):
    """BrowserStack build status."""
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class BrowserStackSession:
    """A BrowserStack test session."""
    session_id: str
    name: str
    status: SessionStatus
    reason: Optional[str]  # Error reason if failed

    # Browser/device info
    os: str
    os_version: str
    browser: Optional[str]
    browser_version: Optional[str]
    device: Optional[str]  # For mobile testing

    # Build info
    build_id: str
    build_name: str
    project_name: Optional[str]

    # Timestamps
    created_at: datetime
    duration: int  # seconds

    # URLs
    public_url: Optional[str]  # Shareable session URL
    browser_url: Optional[str]  # Dashboard URL

    # Logs and artifacts
    video_url: Optional[str]
    logs_url: Optional[str]
    console_logs_url: Optional[str]
    network_logs_url: Optional[str]

    # Test metadata
    test_priority: Optional[str]
    test_status: Optional[str]  # Custom test status (passed/failed)


@dataclass
class BrowserStackBuild:
    """A BrowserStack build (collection of sessions)."""
    build_id: str
    name: str
    status: BuildStatus
    session_count: int
    created_at: datetime

    # Statistics (with defaults)
    sessions_running: int = 0
    sessions_done: int = 0
    sessions_passed: int = 0
    sessions_failed: int = 0

    # Timestamps
    updated_at: Optional[datetime] = None
    duration: int = 0  # Total duration in seconds

    # Project info
    project_name: Optional[str] = None

    # Tags
    tags: list[str] = field(default_factory=list)


@dataclass
class BrowserInfo:
    """Information about an available browser."""
    os: str
    os_version: str
    browser: str
    browser_version: str
    device: Optional[str] = None  # For mobile
    real_mobile: bool = False


class BrowserStackPlugin(IntegrationPlugin):
    """
    BrowserStack Integration Plugin.

    Provides integration with BrowserStack's Automate API for:
    - Managing test sessions and builds
    - Retrieving test artifacts (screenshots, videos, logs)
    - Listing available browsers and devices

    Authentication: Basic auth with username + access_key
    API Base URL: https://api.browserstack.com/automate

    Usage:
        async with BrowserStackPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://api.browserstack.com/automate"

    def __init__(self, http_client=None):
        super().__init__(http_client)
        self.log = logger.bind(component="browserstack")

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="browserstack",
            name="BrowserStack",
            description="Cross-browser testing platform with real browsers and devices",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://www.browserstack.com/favicon.ico",
            docs_url="https://www.browserstack.com/docs/automate/api-reference/",
            website_url="https://www.browserstack.com",
            features=[
                "cross_browser_testing",
                "real_device_testing",
                "session_management",
                "build_tracking",
                "screenshots",
                "video_recording",
                "console_logs",
                "network_logs",
            ],
            required_fields=[
                FieldDefinition(
                    name="username",
                    label="BrowserStack Username",
                    type="text",
                    required=True,
                    placeholder="your_username",
                    help_text="Your BrowserStack username from account settings",
                ),
                FieldDefinition(
                    name="api_key",
                    label="Access Key",
                    type="password",
                    required=True,
                    placeholder="your_access_key",
                    help_text="Your BrowserStack access key from account settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Default Project",
                    type="text",
                    required=False,
                    placeholder="my-project",
                    help_text="Default project name for filtering",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["builds", "sessions", "browsers"],
        )

    def _get_auth_header(self, credentials: IntegrationCredentials) -> dict:
        """Generate Basic auth header from credentials."""
        username = credentials.username or ""
        access_key = credentials.api_key or ""
        auth_string = f"{username}:{access_key}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """
        Test if the BrowserStack credentials are valid.

        Uses GET /automate/plan.json to verify credentials and get account info.

        Args:
            credentials: BrowserStack credentials with username and api_key

        Returns:
            ConnectionTestResult with success status and account info
        """
        start_time = datetime.utcnow()

        try:
            # Validate required credentials
            is_valid, missing = self.validate_credentials(credentials)
            if not is_valid:
                return ConnectionTestResult(
                    success=False,
                    message=f"Missing required credentials: {', '.join(missing)}",
                    error_code="MISSING_CREDENTIALS",
                )

            headers = self._get_auth_header(credentials)

            response = await self.http_client.get(
                f"{self.BASE_URL}/plan.json",
                headers=headers,
            )

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()

                self.log.info(
                    "BrowserStack connection successful",
                    username=credentials.username,
                    parallel_sessions_max=data.get("parallel_sessions_max_allowed"),
                )

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to BrowserStack",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["automate:read", "automate:write"],
                    account_info={
                        "username": credentials.username,
                        "automate_plan": data.get("automate_plan"),
                        "parallel_sessions_max_allowed": data.get("parallel_sessions_max_allowed"),
                        "parallel_sessions_running": data.get("parallel_sessions_running"),
                        "queued_sessions": data.get("queued_sessions"),
                        "queued_sessions_max_allowed": data.get("queued_sessions_max_allowed"),
                    },
                )

            elif response.status_code == 401:
                self.log.error(
                    "BrowserStack authentication failed",
                    username=credentials.username,
                )
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed - invalid username or access key",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                )

            else:
                self.log.error(
                    "BrowserStack connection failed",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code=f"HTTP_{response.status_code}",
                    error_details={"response": response.text[:500]},
                )

        except Exception as e:
            self.log.error("BrowserStack connection error", error=str(e))
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
        """
        Sync data from BrowserStack.

        Syncs:
        - builds: Test builds with session counts and status
        - sessions: Individual test sessions with artifacts
        - browsers: Available browsers and devices

        Args:
            credentials: BrowserStack credentials
            resources: Optional list of resources to sync (builds, sessions, browsers)
            since: Optional datetime for incremental sync (not supported by BrowserStack API)
            limit: Maximum number of items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        resources_to_sync = resources or ["builds", "sessions", "browsers"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            headers = self._get_auth_header(credentials)

            # Sync builds
            if "builds" in resources_to_sync:
                builds, build_errors = await self._sync_builds(headers, limit)
                data["builds"] = builds
                total_synced += len(builds)
                errors.extend(build_errors)

            # Sync sessions (from recent builds)
            if "sessions" in resources_to_sync:
                sessions, session_errors = await self._sync_sessions(
                    headers,
                    limit,
                    data.get("builds", [])
                )
                data["sessions"] = sessions
                total_synced += len(sessions)
                errors.extend(session_errors)

            # Sync available browsers
            if "browsers" in resources_to_sync:
                browsers, browser_errors = await self._sync_browsers(headers)
                data["browsers"] = browsers
                total_synced += len(browsers)
                errors.extend(browser_errors)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from BrowserStack",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            self.log.error("BrowserStack sync error", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_builds(
        self,
        headers: dict,
        limit: int
    ) -> tuple[list[dict], list[str]]:
        """Sync builds from BrowserStack."""
        builds = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/builds.json",
                headers=headers,
                params={"limit": min(limit, 100)},
            )

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    build_data = item.get("automation_build", {})
                    build = self._parse_build(build_data)
                    if build:
                        builds.append(build.__dict__)

                self.log.info("Synced BrowserStack builds", count=len(builds))
            else:
                errors.append(f"Failed to fetch builds: HTTP {response.status_code}")
                self.log.error("Failed to fetch builds", status_code=response.status_code)

        except Exception as e:
            errors.append(f"Error fetching builds: {str(e)}")
            self.log.error("Error fetching builds", error=str(e))

        return builds, errors

    async def _sync_sessions(
        self,
        headers: dict,
        limit: int,
        builds: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Sync sessions from BrowserStack builds."""
        sessions = []
        errors = []

        # Get sessions from recent builds
        build_ids = [b["build_id"] for b in builds[:10]]  # Limit to 10 builds

        for build_id in build_ids:
            try:
                response = await self.http_client.get(
                    f"{self.BASE_URL}/builds/{build_id}/sessions.json",
                    headers=headers,
                    params={"limit": min(limit // len(build_ids) if build_ids else limit, 100)},
                )

                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        session_data = item.get("automation_session", {})
                        session = self._parse_session(session_data, build_id)
                        if session:
                            sessions.append(session.__dict__)
                else:
                    errors.append(f"Failed to fetch sessions for build {build_id}")

            except Exception as e:
                errors.append(f"Error fetching sessions for build {build_id}: {str(e)}")

        self.log.info("Synced BrowserStack sessions", count=len(sessions))
        return sessions, errors

    async def _sync_browsers(self, headers: dict) -> tuple[list[dict], list[str]]:
        """Sync available browsers from BrowserStack."""
        browsers = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/browsers.json",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    browser = BrowserInfo(
                        os=item.get("os", ""),
                        os_version=item.get("os_version", ""),
                        browser=item.get("browser", ""),
                        browser_version=item.get("browser_version", ""),
                        device=item.get("device"),
                        real_mobile=item.get("real_mobile", False),
                    )
                    browsers.append(browser.__dict__)

                self.log.info("Synced BrowserStack browsers", count=len(browsers))
            else:
                errors.append(f"Failed to fetch browsers: HTTP {response.status_code}")

        except Exception as e:
            errors.append(f"Error fetching browsers: {str(e)}")
            self.log.error("Error fetching browsers", error=str(e))

        return browsers, errors

    def _parse_build(self, data: dict) -> Optional[BrowserStackBuild]:
        """Parse build data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "done").lower()
            try:
                status = BuildStatus(status_str)
            except ValueError:
                status = BuildStatus.DONE

            # Parse timestamps
            created_at = datetime.utcnow()
            updated_at = None
            if data.get("created_at"):
                created_at = self._parse_datetime(data["created_at"])
            if data.get("updated_at"):
                updated_at = self._parse_datetime(data["updated_at"])

            return BrowserStackBuild(
                build_id=data.get("hashed_id", ""),
                name=data.get("name", "Unnamed Build"),
                status=status,
                session_count=data.get("session_count", 0),
                sessions_running=data.get("sessions_running", 0),
                created_at=created_at,
                updated_at=updated_at,
                duration=data.get("duration", 0),
                project_name=data.get("project_name"),
                tags=data.get("tags", []) or [],
            )
        except Exception as e:
            self.log.warning("Failed to parse build", error=str(e))
            return None

    def _parse_session(self, data: dict, build_id: str) -> Optional[BrowserStackSession]:
        """Parse session data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "done").lower()
            try:
                status = SessionStatus(status_str)
            except ValueError:
                status = SessionStatus.DONE

            # Parse timestamps
            created_at = datetime.utcnow()
            if data.get("created_at"):
                created_at = self._parse_datetime(data["created_at"])

            return BrowserStackSession(
                session_id=data.get("hashed_id", ""),
                name=data.get("name", "Unnamed Session"),
                status=status,
                reason=data.get("reason"),
                os=data.get("os", ""),
                os_version=data.get("os_version", ""),
                browser=data.get("browser"),
                browser_version=data.get("browser_version"),
                device=data.get("device"),
                build_id=build_id,
                build_name=data.get("build_name", ""),
                project_name=data.get("project_name"),
                created_at=created_at,
                duration=data.get("duration", 0),
                public_url=data.get("public_url"),
                browser_url=data.get("browser_url"),
                video_url=data.get("video_url"),
                logs_url=data.get("logs"),
                console_logs_url=data.get("browser_console_logs_url"),
                network_logs_url=data.get("har_logs_url"),
                test_priority=data.get("test_priority"),
                test_status=data.get("test_status"),
            )
        except Exception as e:
            self.log.warning("Failed to parse session", error=str(e))
            return None

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string from BrowserStack API."""
        try:
            # BrowserStack uses ISO 8601 format
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str.replace("+00:00", ""))
        except (ValueError, TypeError):
            return datetime.utcnow()

    # Additional helper methods for specific operations

    async def get_session_logs(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Get all logs for a specific session.

        Returns:
            Dictionary containing text_logs, console_logs, and network_logs
        """
        headers = self._get_auth_header(credentials)
        logs: dict[str, Any] = {}

        try:
            # Get text logs
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/{session_id}/logs",
                headers=headers,
            )
            if response.status_code == 200:
                logs["text_logs"] = response.text

            # Get console logs
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/{session_id}/consolelogs",
                headers=headers,
            )
            if response.status_code == 200:
                logs["console_logs"] = response.json()

            # Get network logs (HAR)
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/{session_id}/networklogs",
                headers=headers,
            )
            if response.status_code == 200:
                logs["network_logs"] = response.json()

        except Exception as e:
            self.log.error("Failed to get session logs", session_id=session_id, error=str(e))

        return logs

    async def update_session_status(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
        status: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Update the status of a session (mark as passed/failed).

        Args:
            credentials: BrowserStack credentials
            session_id: Session ID to update
            status: New status ("passed" or "failed")
            reason: Optional reason for the status

        Returns:
            True if update was successful
        """
        headers = self._get_auth_header(credentials)

        payload = {"status": status}
        if reason:
            payload["reason"] = reason

        try:
            response = await self.http_client.put(
                f"{self.BASE_URL}/sessions/{session_id}.json",
                headers=headers,
                json=payload,
            )
            return response.status_code == 200
        except Exception as e:
            self.log.error("Failed to update session status", session_id=session_id, error=str(e))
            return False


def create_browserstack_plugin() -> BrowserStackPlugin:
    """Factory function for BrowserStackPlugin."""
    return BrowserStackPlugin()
