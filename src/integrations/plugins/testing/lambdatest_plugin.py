"""
LambdaTest Integration Plugin.

LambdaTest is a cloud-based cross-browser testing platform that enables
testing of web and mobile applications across 3000+ browsers and devices.

API Docs: https://www.lambdatest.com/support/api-doc/

Features:
- Cross-browser testing on real browsers
- Automated Selenium and Cypress testing
- Real device testing for mobile apps
- Screenshot testing
- Build and session management
- Video recording and logs
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
    """LambdaTest session status."""
    RUNNING = "running"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


class BuildStatus(str, Enum):
    """LambdaTest build status."""
    RUNNING = "running"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class LambdaTestSession:
    """A LambdaTest test session."""
    session_id: str
    name: str
    status: SessionStatus
    status_ind: str  # User-defined status (passed/failed)
    os: str
    os_version: str
    build_id: str
    build_name: str
    created_at: datetime
    duration: int  # seconds

    # Optional browser/device info
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    device: Optional[str] = None  # For mobile/real device testing
    resolution: Optional[str] = None

    # Project info
    project_name: Optional[str] = None

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # URLs
    console_logs_url: Optional[str] = None
    network_logs_url: Optional[str] = None
    command_logs_url: Optional[str] = None
    selenium_logs_url: Optional[str] = None
    video_url: Optional[str] = None

    # Test metadata
    test_type: Optional[str] = None  # selenium, cypress, etc.
    remarks: Optional[str] = None

    # Tags
    tags: list[str] = field(default_factory=list)


@dataclass
class LambdaTestBuild:
    """A LambdaTest build (collection of sessions)."""
    build_id: str
    name: str
    status: BuildStatus
    status_ind: str  # Overall status indicator

    # Statistics
    session_count: int = 0
    sessions_running: int = 0
    sessions_completed: int = 0
    sessions_passed: int = 0
    sessions_failed: int = 0
    sessions_error: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    duration: int = 0  # Total duration in seconds

    # Project info
    project_name: Optional[str] = None

    # Tags
    tags: list[str] = field(default_factory=list)


@dataclass
class LambdaTestScreenshot:
    """A screenshot from LambdaTest screenshot testing."""
    screenshot_id: str
    name: str
    url: str  # URL that was screenshotted
    status: str
    os: str
    os_version: str
    browser: str
    browser_version: str
    resolution: str
    screenshot_url: str
    created_at: datetime

    # Optional fields
    thumbnail_url: Optional[str] = None
    test_id: Optional[str] = None


@dataclass
class LambdaTestUser:
    """LambdaTest user information."""
    user_id: int
    username: str
    email: str
    name: str

    # Plan info
    plan: str
    max_concurrent_sessions: int
    max_queue_size: int

    # Usage
    minutes_used: int = 0
    minutes_remaining: int = 0


class LambdaTestPlugin(IntegrationPlugin):
    """
    LambdaTest Integration Plugin.

    Provides integration with LambdaTest REST API for:
    - Managing test sessions and builds
    - Retrieving test artifacts (screenshots, videos, logs)
    - Screenshot testing
    - Getting user and organization info

    Authentication: Basic auth with username + access_key
    API Base URL: https://api.lambdatest.com/automation/api/v1

    Usage:
        async with LambdaTestPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    BASE_URL = "https://api.lambdatest.com/automation/api/v1"
    SCREENSHOTS_API_URL = "https://api.lambdatest.com/screenshots/v1"

    def __init__(self, http_client=None):
        super().__init__(http_client)
        self.log = logger.bind(component="lambdatest")

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="lambdatest",
            name="LambdaTest",
            description="Cloud-based cross-browser testing platform with 3000+ browsers",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://www.lambdatest.com/favicon.ico",
            docs_url="https://www.lambdatest.com/support/api-doc/",
            website_url="https://www.lambdatest.com",
            features=[
                "cross_browser_testing",
                "real_device_testing",
                "selenium_testing",
                "cypress_testing",
                "session_management",
                "build_tracking",
                "screenshots",
                "video_recording",
                "console_logs",
                "network_logs",
                "screenshot_testing",
            ],
            required_fields=[
                FieldDefinition(
                    name="username",
                    label="LambdaTest Username",
                    type="text",
                    required=True,
                    placeholder="your_username",
                    help_text="Your LambdaTest username from Profile section",
                ),
                FieldDefinition(
                    name="api_key",
                    label="Access Key",
                    type="password",
                    required=True,
                    placeholder="your_access_key",
                    help_text="Your LambdaTest access key from Profile section",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Default Project",
                    type="text",
                    required=False,
                    placeholder="my-project",
                    help_text="Default project name for filtering sessions",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["builds", "sessions", "screenshots"],
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
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """
        Test if the LambdaTest credentials are valid.

        Uses GET /user to verify credentials and get account info.

        Args:
            credentials: LambdaTest credentials with username and api_key

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

            # Use /user/me endpoint for verification (LambdaTest's user info endpoint)
            response = await self.http_client.get(
                f"{self.BASE_URL}/user",
                headers=headers,
            )

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()
                user_data = data.get("data", {})

                self.log.info(
                    "LambdaTest connection successful",
                    username=credentials.username,
                )

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to LambdaTest",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["automation:read", "automation:write", "screenshots:read"],
                    account_info={
                        "username": user_data.get("username"),
                        "email": user_data.get("email"),
                        "name": user_data.get("name"),
                        "organization": user_data.get("organization_name"),
                        "max_parallel_sessions": user_data.get("max_parallel_sessions"),
                        "max_queue_size": user_data.get("max_queue_size"),
                        "plan_name": user_data.get("plan_name"),
                        "credits_used": user_data.get("credits_used"),
                        "credits_remaining": user_data.get("credits_remaining"),
                    },
                )

            elif response.status_code == 401:
                self.log.error(
                    "LambdaTest authentication failed",
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
                    "LambdaTest connection failed",
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
            self.log.error("LambdaTest connection error", error=str(e))
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
        Sync data from LambdaTest.

        Syncs:
        - builds: Test builds with session counts and status
        - sessions: Individual test sessions with artifacts
        - screenshots: Screenshot testing results

        Args:
            credentials: LambdaTest credentials
            resources: Optional list of resources to sync (builds, sessions, screenshots)
            since: Optional datetime for incremental sync
            limit: Maximum number of items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        resources_to_sync = resources or ["builds", "sessions", "screenshots"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            headers = self._get_auth_header(credentials)

            # Sync builds
            if "builds" in resources_to_sync:
                builds, build_errors = await self._sync_builds(headers, since, limit)
                data["builds"] = builds
                total_synced += len(builds)
                errors.extend(build_errors)

            # Sync sessions
            if "sessions" in resources_to_sync:
                sessions, session_errors = await self._sync_sessions(
                    headers,
                    since,
                    limit,
                    data.get("builds", [])
                )
                data["sessions"] = sessions
                total_synced += len(sessions)
                errors.extend(session_errors)

            # Sync screenshots
            if "screenshots" in resources_to_sync:
                screenshots, screenshot_errors = await self._sync_screenshots(headers, limit)
                data["screenshots"] = screenshots
                total_synced += len(screenshots)
                errors.extend(screenshot_errors)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from LambdaTest",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            self.log.error("LambdaTest sync error", error=str(e))
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
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync builds from LambdaTest."""
        builds = []
        errors = []

        try:
            params: dict[str, Any] = {"limit": min(limit, 100), "offset": 0}

            # Add time filter if provided
            if since:
                params["fromdate"] = since.strftime("%Y-%m-%d")
                params["todate"] = datetime.utcnow().strftime("%Y-%m-%d")

            response = await self.http_client.get(
                f"{self.BASE_URL}/builds",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("data", [])

                for item in items:
                    build = self._parse_build(item)
                    if build:
                        builds.append(build.__dict__)

                self.log.info("Synced LambdaTest builds", count=len(builds))
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
        since: Optional[datetime],
        limit: int,
        builds: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Sync sessions from LambdaTest builds."""
        sessions = []
        errors = []

        # Get sessions from recent builds (or all sessions if no builds)
        build_ids = [b["build_id"] for b in builds[:10]] if builds else []

        if build_ids:
            # Fetch sessions per build
            for build_id in build_ids:
                try:
                    params: dict[str, Any] = {
                        "limit": min(limit // len(build_ids) if build_ids else limit, 100),
                        "offset": 0,
                    }

                    response = await self.http_client.get(
                        f"{self.BASE_URL}/builds/{build_id}/sessions",
                        headers=headers,
                        params=params,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("data", [])
                        for item in items:
                            session = self._parse_session(item, build_id)
                            if session:
                                sessions.append(session.__dict__)
                    else:
                        errors.append(f"Failed to fetch sessions for build {build_id}")

                except Exception as e:
                    errors.append(f"Error fetching sessions for build {build_id}: {str(e)}")
        else:
            # Fetch sessions directly (latest)
            try:
                params = {"limit": min(limit, 100), "offset": 0}
                if since:
                    params["fromdate"] = since.strftime("%Y-%m-%d")
                    params["todate"] = datetime.utcnow().strftime("%Y-%m-%d")

                response = await self.http_client.get(
                    f"{self.BASE_URL}/sessions",
                    headers=headers,
                    params=params,
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("data", [])
                    for item in items:
                        session = self._parse_session(item, item.get("build_id", ""))
                        if session:
                            sessions.append(session.__dict__)
                else:
                    errors.append(f"Failed to fetch sessions: HTTP {response.status_code}")

            except Exception as e:
                errors.append(f"Error fetching sessions: {str(e)}")

        self.log.info("Synced LambdaTest sessions", count=len(sessions))
        return sessions, errors

    async def _sync_screenshots(
        self,
        headers: dict,
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync screenshot tests from LambdaTest."""
        screenshots = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{self.SCREENSHOTS_API_URL}/screenshots",
                headers=headers,
                params={"limit": min(limit, 100)},
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("data", [])

                for item in items:
                    # Parse each screenshot test
                    test_id = item.get("test_id")
                    for screenshot in item.get("screenshots", []):
                        parsed = self._parse_screenshot(screenshot, test_id)
                        if parsed:
                            screenshots.append(parsed.__dict__)

                self.log.info("Synced LambdaTest screenshots", count=len(screenshots))
            elif response.status_code == 404:
                # Screenshot API might not be enabled
                self.log.warning("Screenshot API not available or no screenshots found")
            else:
                errors.append(f"Failed to fetch screenshots: HTTP {response.status_code}")
                self.log.error("Failed to fetch screenshots", status_code=response.status_code)

        except Exception as e:
            errors.append(f"Error fetching screenshots: {str(e)}")
            self.log.error("Error fetching screenshots", error=str(e))

        return screenshots, errors

    def _parse_build(self, data: dict) -> Optional[LambdaTestBuild]:
        """Parse build data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "completed").lower()
            try:
                status = BuildStatus(status_str)
            except ValueError:
                status = BuildStatus.COMPLETED

            # Parse timestamps
            created_at = None
            updated_at = None
            if data.get("create_timestamp"):
                created_at = self._parse_datetime(data["create_timestamp"])
            if data.get("update_timestamp"):
                updated_at = self._parse_datetime(data["update_timestamp"])

            return LambdaTestBuild(
                build_id=str(data.get("build_id", "")),
                name=data.get("name", "Unnamed Build"),
                status=status,
                status_ind=data.get("status_ind", ""),
                session_count=data.get("sessions_count", 0),
                sessions_running=data.get("running_sessions", 0),
                sessions_completed=data.get("completed_sessions", 0),
                sessions_passed=data.get("passed_sessions", 0),
                sessions_failed=data.get("failed_sessions", 0),
                sessions_error=data.get("error_sessions", 0),
                created_at=created_at,
                updated_at=updated_at,
                duration=data.get("duration", 0),
                project_name=data.get("project_name"),
                tags=data.get("tags", []) or [],
            )
        except Exception as e:
            self.log.warning("Failed to parse build", error=str(e))
            return None

    def _parse_session(self, data: dict, build_id: str) -> Optional[LambdaTestSession]:
        """Parse session data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "completed").lower()
            try:
                status = SessionStatus(status_str)
            except ValueError:
                status = SessionStatus.COMPLETED

            # Parse timestamps
            created_at = datetime.utcnow()
            start_time = None
            end_time = None

            if data.get("create_timestamp"):
                created_at = self._parse_datetime(data["create_timestamp"])
            if data.get("start_timestamp"):
                start_time = self._parse_datetime(data["start_timestamp"])
            if data.get("end_timestamp"):
                end_time = self._parse_datetime(data["end_timestamp"])

            return LambdaTestSession(
                session_id=data.get("session_id", ""),
                name=data.get("name", "Unnamed Session"),
                status=status,
                status_ind=data.get("status_ind", ""),
                os=data.get("os", ""),
                os_version=data.get("os_version", ""),
                browser=data.get("browser"),
                browser_version=data.get("browser_version"),
                device=data.get("device"),
                resolution=data.get("resolution"),
                build_id=build_id or str(data.get("build_id", "")),
                build_name=data.get("build_name", ""),
                project_name=data.get("project_name"),
                created_at=created_at,
                start_time=start_time,
                end_time=end_time,
                duration=data.get("duration", 0),
                console_logs_url=data.get("console_logs_url"),
                network_logs_url=data.get("network_logs_url"),
                command_logs_url=data.get("command_logs_url"),
                selenium_logs_url=data.get("selenium_logs_url"),
                video_url=data.get("video_url"),
                test_type=data.get("test_type"),
                remarks=data.get("remarks"),
                tags=data.get("tags", []) or [],
            )
        except Exception as e:
            self.log.warning("Failed to parse session", error=str(e))
            return None

    def _parse_screenshot(self, data: dict, test_id: Optional[str]) -> Optional[LambdaTestScreenshot]:
        """Parse screenshot data from API response."""
        try:
            created_at = datetime.utcnow()
            if data.get("created_at"):
                created_at = self._parse_datetime(data["created_at"])

            return LambdaTestScreenshot(
                screenshot_id=data.get("screenshot_id", ""),
                name=data.get("name", ""),
                url=data.get("url", ""),
                status=data.get("status", ""),
                os=data.get("os", ""),
                os_version=data.get("os_version", ""),
                browser=data.get("browser", ""),
                browser_version=data.get("browser_version", ""),
                resolution=data.get("resolution", ""),
                screenshot_url=data.get("screenshot_url", ""),
                thumbnail_url=data.get("thumbnail_url"),
                created_at=created_at,
                test_id=test_id,
            )
        except Exception as e:
            self.log.warning("Failed to parse screenshot", error=str(e))
            return None

    def _parse_datetime(self, dt_value: Any) -> datetime:
        """Parse datetime from LambdaTest API (can be string or timestamp)."""
        try:
            if isinstance(dt_value, (int, float)):
                # Unix timestamp
                return datetime.fromtimestamp(dt_value)
            elif isinstance(dt_value, str):
                # ISO format or other string format
                if dt_value.endswith("Z"):
                    dt_value = dt_value[:-1] + "+00:00"
                return datetime.fromisoformat(dt_value.replace("+00:00", ""))
            return datetime.utcnow()
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
            Dictionary containing console_logs, network_logs, command_logs, and selenium_logs
        """
        headers = self._get_auth_header(credentials)
        logs: dict[str, Any] = {}

        try:
            # Get session details first
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/{session_id}",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                session_data = data.get("data", {})

                # Fetch each log type if URL is available
                log_urls = {
                    "console_logs": session_data.get("console_logs_url"),
                    "network_logs": session_data.get("network_logs_url"),
                    "command_logs": session_data.get("command_logs_url"),
                    "selenium_logs": session_data.get("selenium_logs_url"),
                }

                for log_type, log_url in log_urls.items():
                    if log_url:
                        try:
                            log_response = await self.http_client.get(
                                log_url,
                                headers=headers,
                            )
                            if log_response.status_code == 200:
                                logs[log_type] = log_response.json()
                        except Exception as e:
                            self.log.warning(
                                f"Failed to fetch {log_type}",
                                session_id=session_id,
                                error=str(e),
                            )

        except Exception as e:
            self.log.error("Failed to get session logs", session_id=session_id, error=str(e))

        return logs

    async def get_session_video(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
    ) -> Optional[str]:
        """
        Get the video URL for a session.

        Args:
            credentials: LambdaTest credentials
            session_id: Session ID

        Returns:
            Video URL or None if not available
        """
        headers = self._get_auth_header(credentials)

        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/sessions/{session_id}",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("video_url")

        except Exception as e:
            self.log.error("Failed to get session video", session_id=session_id, error=str(e))

        return None

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
            credentials: LambdaTest credentials
            session_id: Session ID to update
            status: New status ("passed" or "failed")
            reason: Optional reason/remarks for the status

        Returns:
            True if update was successful
        """
        headers = self._get_auth_header(credentials)

        payload: dict[str, Any] = {"status_ind": status}
        if reason:
            payload["reason"] = reason

        try:
            response = await self.http_client.patch(
                f"{self.BASE_URL}/sessions/{session_id}",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                self.log.info("Updated session status", session_id=session_id, status=status)
                return True
            else:
                self.log.error(
                    "Failed to update session status",
                    session_id=session_id,
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            self.log.error("Error updating session status", session_id=session_id, error=str(e))
            return False

    async def stop_session(
        self,
        credentials: IntegrationCredentials,
        session_id: str,
    ) -> bool:
        """
        Stop a running session.

        Args:
            credentials: LambdaTest credentials
            session_id: Session ID to stop

        Returns:
            True if session was stopped successfully
        """
        headers = self._get_auth_header(credentials)

        try:
            response = await self.http_client.delete(
                f"{self.BASE_URL}/sessions/{session_id}",
                headers=headers,
            )

            if response.status_code in [200, 204]:
                self.log.info("Stopped session", session_id=session_id)
                return True
            else:
                self.log.error(
                    "Failed to stop session",
                    session_id=session_id,
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            self.log.error("Error stopping session", session_id=session_id, error=str(e))
            return False

    async def create_screenshot_test(
        self,
        credentials: IntegrationCredentials,
        url: str,
        browsers: list[dict[str, str]],
        name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a new screenshot test.

        Args:
            credentials: LambdaTest credentials
            url: URL to capture screenshots of
            browsers: List of browser configurations, e.g.:
                [{"os": "Windows 10", "browser": "Chrome", "browser_version": "latest"}]
            name: Optional test name

        Returns:
            Test ID if created successfully, None otherwise
        """
        headers = self._get_auth_header(credentials)

        payload = {
            "url": url,
            "configs": browsers,
        }
        if name:
            payload["name"] = name

        try:
            response = await self.http_client.post(
                f"{self.SCREENSHOTS_API_URL}/screenshots",
                headers=headers,
                json=payload,
            )

            if response.status_code in [200, 201]:
                data = response.json()
                test_id = data.get("data", {}).get("test_id")
                self.log.info("Created screenshot test", test_id=test_id, url=url)
                return test_id
            else:
                self.log.error(
                    "Failed to create screenshot test",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return None

        except Exception as e:
            self.log.error("Error creating screenshot test", error=str(e))
            return None


def create_lambdatest_plugin() -> LambdaTestPlugin:
    """Factory function for LambdaTestPlugin."""
    return LambdaTestPlugin()
