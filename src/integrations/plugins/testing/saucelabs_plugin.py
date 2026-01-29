"""
Sauce Labs Integration Plugin.

Sauce Labs is a cloud-based testing platform that provides comprehensive
automated testing infrastructure for web and mobile applications.

API Docs: https://docs.saucelabs.com/dev/api/

Features:
- Cross-browser and cross-platform testing
- Real device cloud for mobile testing
- Automated testing with Selenium/Appium
- Sauce Connect tunnels for testing behind firewalls
- Build and job management
- Test analytics and insights
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


class DataCenter(str, Enum):
    """Sauce Labs data center regions."""
    US_WEST = "us-west-1"
    US_EAST = "us-east-4"
    EU_CENTRAL = "eu-central-1"
    APAC_SOUTHEAST = "apac-southeast-1"


class JobStatus(str, Enum):
    """Sauce Labs job status."""
    IN_PROGRESS = "in progress"
    PASSED = "passed"
    FAILED = "failed"
    COMPLETE = "complete"
    ERROR = "error"


class TunnelStatus(str, Enum):
    """Sauce Labs tunnel status."""
    RUNNING = "running"
    TERMINATED = "terminated"
    BOOTING = "booting"


@dataclass
class SauceLabsJob:
    """A Sauce Labs test job."""
    job_id: str
    name: str
    status: JobStatus
    os: str
    owner: str
    created_at: datetime
    duration: int  # seconds

    # Optional fields
    passed: Optional[bool] = None  # True if test passed

    # Browser/device info
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    device: Optional[str] = None  # For mobile testing

    # Build info
    build: Optional[str] = None
    build_id: Optional[str] = None

    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # URLs
    video_url: Optional[str] = None
    log_url: Optional[str] = None

    # Error info
    error: Optional[str] = None

    # Tags and metadata
    tags: list[str] = field(default_factory=list)
    custom_data: dict = field(default_factory=dict)

    # Automation framework
    automation_backend: Optional[str] = None  # selenium, appium, etc.


@dataclass
class SauceLabsBuild:
    """A Sauce Labs build (collection of jobs)."""
    build_id: str
    name: str
    status: str

    # Statistics
    jobs_total: int = 0
    jobs_passed: int = 0
    jobs_failed: int = 0
    jobs_in_progress: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Owner
    owner: Optional[str] = None


@dataclass
class SauceLabsTunnel:
    """A Sauce Connect tunnel."""
    tunnel_id: str
    tunnel_name: str
    status: TunnelStatus
    owner: str

    # Connection info (with defaults)
    host: Optional[str] = None
    direct_domains: list[str] = field(default_factory=list)
    no_proxy_caching: bool = False

    # Timestamps
    created_at: Optional[datetime] = None
    last_connected: Optional[datetime] = None

    # Metadata
    tunnel_identifier: Optional[str] = None
    vm_version: Optional[str] = None


DATA_CENTER_URLS = {
    DataCenter.US_WEST: "https://api.us-west-1.saucelabs.com",
    DataCenter.US_EAST: "https://api.us-east-4.saucelabs.com",
    DataCenter.EU_CENTRAL: "https://api.eu-central-1.saucelabs.com",
    DataCenter.APAC_SOUTHEAST: "https://api.apac-southeast-1.saucelabs.com",
}


class SauceLabsPlugin(IntegrationPlugin):
    """
    Sauce Labs Integration Plugin.

    Provides integration with Sauce Labs REST API for:
    - Managing test jobs and builds
    - Monitoring Sauce Connect tunnels
    - Retrieving test artifacts (videos, logs)
    - Getting test analytics

    Authentication: Basic auth with username + access_key
    API Base URL varies by data center (site parameter)

    Usage:
        async with SauceLabsPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    def __init__(self, http_client=None):
        super().__init__(http_client)
        self.log = logger.bind(component="saucelabs")

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="saucelabs",
            name="Sauce Labs",
            description="Comprehensive cloud testing platform for web and mobile apps",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://saucelabs.com/favicon.ico",
            docs_url="https://docs.saucelabs.com/dev/api/",
            website_url="https://saucelabs.com",
            features=[
                "cross_browser_testing",
                "mobile_testing",
                "real_device_cloud",
                "sauce_connect_tunnels",
                "job_management",
                "build_tracking",
                "video_recording",
                "test_logs",
                "multiple_data_centers",
            ],
            required_fields=[
                FieldDefinition(
                    name="username",
                    label="Sauce Labs Username",
                    type="text",
                    required=True,
                    placeholder="your_username",
                    help_text="Your Sauce Labs username",
                ),
                FieldDefinition(
                    name="api_key",
                    label="Access Key",
                    type="password",
                    required=True,
                    placeholder="your_access_key",
                    help_text="Your Sauce Labs access key from User Settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="site",
                    label="Data Center",
                    type="select",
                    required=False,
                    help_text="Sauce Labs data center region",
                    options=[
                        {"value": "us-west-1", "label": "US West (default)"},
                        {"value": "us-east-4", "label": "US East"},
                        {"value": "eu-central-1", "label": "EU Central"},
                        {"value": "apac-southeast-1", "label": "APAC Southeast"},
                    ],
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["jobs", "builds", "tunnels"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL based on the data center."""
        site = credentials.site or "us-west-1"
        try:
            data_center = DataCenter(site)
            return DATA_CENTER_URLS.get(data_center, DATA_CENTER_URLS[DataCenter.US_WEST])
        except ValueError:
            return DATA_CENTER_URLS[DataCenter.US_WEST]

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
        Test if the Sauce Labs credentials are valid.

        Uses GET /rest/v1/users/{username} to verify credentials.

        Args:
            credentials: Sauce Labs credentials with username and api_key

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

            base_url = self._get_base_url(credentials)
            headers = self._get_auth_header(credentials)
            username = credentials.username

            response = await self.http_client.get(
                f"{base_url}/rest/v1/users/{username}",
                headers=headers,
            )

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()

                self.log.info(
                    "Sauce Labs connection successful",
                    username=username,
                    data_center=credentials.site or "us-west-1",
                )

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Sauce Labs",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["jobs:read", "jobs:write", "tunnels:read"],
                    account_info={
                        "username": data.get("username"),
                        "email": data.get("email"),
                        "name": data.get("name"),
                        "id": data.get("id"),
                        "access_key_present": bool(data.get("access_key")),
                        "verified": data.get("verified", False),
                        "ancestor": data.get("ancestor_allows_subaccounts", False),
                        "data_center": credentials.site or "us-west-1",
                        "minutes": data.get("minutes"),
                        "manual_minutes": data.get("manual_minutes"),
                    },
                )

            elif response.status_code == 401:
                self.log.error(
                    "Sauce Labs authentication failed",
                    username=username,
                )
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed - invalid username or access key",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                )

            elif response.status_code == 404:
                self.log.error(
                    "Sauce Labs user not found",
                    username=username,
                )
                return ConnectionTestResult(
                    success=False,
                    message="User not found - check username",
                    latency_ms=latency_ms,
                    error_code="USER_NOT_FOUND",
                )

            else:
                self.log.error(
                    "Sauce Labs connection failed",
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
            self.log.error("Sauce Labs connection error", error=str(e))
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
        Sync data from Sauce Labs.

        Syncs:
        - jobs: Test jobs with status and artifacts
        - builds: Build summaries with job counts
        - tunnels: Active Sauce Connect tunnels

        Args:
            credentials: Sauce Labs credentials
            resources: Optional list of resources to sync (jobs, builds, tunnels)
            since: Optional datetime for incremental sync
            limit: Maximum number of items per resource

        Returns:
            SyncResult with synced data
        """
        started_at = datetime.utcnow()
        resources_to_sync = resources or ["jobs", "builds", "tunnels"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            base_url = self._get_base_url(credentials)
            headers = self._get_auth_header(credentials)
            username = credentials.username

            # Sync jobs
            if "jobs" in resources_to_sync:
                jobs, job_errors = await self._sync_jobs(
                    base_url, headers, username, since, limit
                )
                data["jobs"] = jobs
                total_synced += len(jobs)
                errors.extend(job_errors)

            # Sync builds
            if "builds" in resources_to_sync:
                builds, build_errors = await self._sync_builds(
                    base_url, headers, username, limit
                )
                data["builds"] = builds
                total_synced += len(builds)
                errors.extend(build_errors)

            # Sync tunnels
            if "tunnels" in resources_to_sync:
                tunnels, tunnel_errors = await self._sync_tunnels(
                    base_url, headers, username
                )
                data["tunnels"] = tunnels
                total_synced += len(tunnels)
                errors.extend(tunnel_errors)

            completed_at = datetime.utcnow()

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Sauce Labs",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=completed_at,
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            self.log.error("Sauce Labs sync error", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_jobs(
        self,
        base_url: str,
        headers: dict,
        username: str,
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync jobs from Sauce Labs."""
        jobs = []
        errors = []

        try:
            params: dict[str, Any] = {"limit": min(limit, 500)}

            # Add time filter if provided
            if since:
                # Sauce Labs uses Unix timestamp in seconds
                params["from"] = int(since.timestamp())

            response = await self.http_client.get(
                f"{base_url}/rest/v1/{username}/jobs",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    job = self._parse_job(item)
                    if job:
                        jobs.append(job.__dict__)

                self.log.info("Synced Sauce Labs jobs", count=len(jobs))
            else:
                errors.append(f"Failed to fetch jobs: HTTP {response.status_code}")
                self.log.error("Failed to fetch jobs", status_code=response.status_code)

        except Exception as e:
            errors.append(f"Error fetching jobs: {str(e)}")
            self.log.error("Error fetching jobs", error=str(e))

        return jobs, errors

    async def _sync_builds(
        self,
        base_url: str,
        headers: dict,
        username: str,
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync builds from Sauce Labs."""
        builds = []
        errors = []

        try:
            # Get builds via the builds API (v2)
            response = await self.http_client.get(
                f"{base_url}/rest/v1/{username}/builds",
                headers=headers,
                params={"limit": min(limit, 100)},
            )

            if response.status_code == 200:
                data = response.json()
                for item in data:
                    build = self._parse_build(item)
                    if build:
                        builds.append(build.__dict__)

                self.log.info("Synced Sauce Labs builds", count=len(builds))
            else:
                errors.append(f"Failed to fetch builds: HTTP {response.status_code}")
                self.log.error("Failed to fetch builds", status_code=response.status_code)

        except Exception as e:
            errors.append(f"Error fetching builds: {str(e)}")
            self.log.error("Error fetching builds", error=str(e))

        return builds, errors

    async def _sync_tunnels(
        self,
        base_url: str,
        headers: dict,
        username: str,
    ) -> tuple[list[dict], list[str]]:
        """Sync active tunnels from Sauce Labs."""
        tunnels = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{base_url}/rest/v1/{username}/tunnels",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                # Response is a dict with tunnel IDs as keys
                for tunnel_id, tunnel_info in data.items():
                    tunnel = self._parse_tunnel(tunnel_id, tunnel_info)
                    if tunnel:
                        tunnels.append(tunnel.__dict__)

                self.log.info("Synced Sauce Labs tunnels", count=len(tunnels))
            else:
                errors.append(f"Failed to fetch tunnels: HTTP {response.status_code}")
                self.log.error("Failed to fetch tunnels", status_code=response.status_code)

        except Exception as e:
            errors.append(f"Error fetching tunnels: {str(e)}")
            self.log.error("Error fetching tunnels", error=str(e))

        return tunnels, errors

    def _parse_job(self, data: dict) -> Optional[SauceLabsJob]:
        """Parse job data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "complete")
            try:
                status = JobStatus(status_str)
            except ValueError:
                status = JobStatus.COMPLETE

            # Parse timestamps
            created_at = datetime.utcnow()
            start_time = None
            end_time = None

            if data.get("creation_time"):
                created_at = datetime.fromtimestamp(data["creation_time"])
            if data.get("start_time"):
                start_time = datetime.fromtimestamp(data["start_time"])
            if data.get("end_time"):
                end_time = datetime.fromtimestamp(data["end_time"])

            # Calculate duration
            duration = data.get("duration", 0)
            if not duration and start_time and end_time:
                duration = int((end_time - start_time).total_seconds())

            return SauceLabsJob(
                job_id=data.get("id", ""),
                name=data.get("name", "Unnamed Job"),
                status=status,
                passed=data.get("passed"),
                os=data.get("os", ""),
                browser=data.get("browser"),
                browser_version=data.get("browser_version"),
                device=data.get("device"),
                build=data.get("build"),
                build_id=data.get("build_id"),
                owner=data.get("owner", ""),
                created_at=created_at,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                video_url=data.get("video_url"),
                log_url=data.get("log_url"),
                error=data.get("error"),
                tags=data.get("tags", []) or [],
                custom_data=data.get("custom-data", {}) or {},
                automation_backend=data.get("automation_backend"),
            )
        except Exception as e:
            self.log.warning("Failed to parse job", error=str(e))
            return None

    def _parse_build(self, data: dict) -> Optional[SauceLabsBuild]:
        """Parse build data from API response."""
        try:
            # Parse timestamps
            created_at = None
            if data.get("creation_time"):
                created_at = datetime.fromtimestamp(data["creation_time"])

            return SauceLabsBuild(
                build_id=data.get("id", ""),
                name=data.get("name", "Unnamed Build"),
                status=data.get("status", "complete"),
                jobs_total=data.get("jobs", {}).get("total", 0),
                jobs_passed=data.get("jobs", {}).get("passed", 0),
                jobs_failed=data.get("jobs", {}).get("failed", 0),
                jobs_in_progress=data.get("jobs", {}).get("running", 0),
                created_at=created_at,
                owner=data.get("owner"),
            )
        except Exception as e:
            self.log.warning("Failed to parse build", error=str(e))
            return None

    def _parse_tunnel(self, tunnel_id: str, data: dict) -> Optional[SauceLabsTunnel]:
        """Parse tunnel data from API response."""
        try:
            # Parse status
            status_str = data.get("status", "running")
            try:
                status = TunnelStatus(status_str)
            except ValueError:
                status = TunnelStatus.RUNNING

            # Parse timestamps
            created_at = None
            last_connected = None
            if data.get("creation_time"):
                created_at = datetime.fromtimestamp(data["creation_time"])
            if data.get("last_connected"):
                last_connected = datetime.fromtimestamp(data["last_connected"])

            return SauceLabsTunnel(
                tunnel_id=tunnel_id,
                tunnel_name=data.get("tunnel_identifier", tunnel_id),
                status=status,
                owner=data.get("owner", ""),
                host=data.get("host"),
                direct_domains=data.get("direct_domains", []) or [],
                no_proxy_caching=data.get("no_proxy_caching", False),
                created_at=created_at,
                last_connected=last_connected,
                tunnel_identifier=data.get("tunnel_identifier"),
                vm_version=data.get("vm_version"),
            )
        except Exception as e:
            self.log.warning("Failed to parse tunnel", error=str(e))
            return None

    # Additional helper methods for specific operations

    async def get_job_assets(
        self,
        credentials: IntegrationCredentials,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Get all assets for a specific job.

        Returns:
            Dictionary containing video, logs, screenshots, etc.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_auth_header(credentials)
        username = credentials.username
        assets: dict[str, Any] = {}

        try:
            response = await self.http_client.get(
                f"{base_url}/rest/v1/{username}/jobs/{job_id}/assets",
                headers=headers,
            )

            if response.status_code == 200:
                assets = response.json()
                self.log.info("Retrieved job assets", job_id=job_id, asset_count=len(assets))

        except Exception as e:
            self.log.error("Failed to get job assets", job_id=job_id, error=str(e))

        return assets

    async def download_job_asset(
        self,
        credentials: IntegrationCredentials,
        job_id: str,
        asset_name: str,
    ) -> Optional[bytes]:
        """
        Download a specific asset (video, screenshot, log) for a job.

        Args:
            credentials: Sauce Labs credentials
            job_id: Job ID
            asset_name: Asset filename (e.g., "video.mp4", "selenium-log", "screenshot.png")

        Returns:
            Asset content as bytes, or None if not found
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_auth_header(credentials)
        username = credentials.username

        try:
            response = await self.http_client.get(
                f"{base_url}/rest/v1/{username}/jobs/{job_id}/assets/{asset_name}",
                headers=headers,
            )

            if response.status_code == 200:
                return response.content
            else:
                self.log.warning(
                    "Asset not found",
                    job_id=job_id,
                    asset_name=asset_name,
                    status_code=response.status_code,
                )
                return None

        except Exception as e:
            self.log.error("Failed to download asset", job_id=job_id, error=str(e))
            return None

    async def update_job(
        self,
        credentials: IntegrationCredentials,
        job_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a job with custom data, status, or tags.

        Args:
            credentials: Sauce Labs credentials
            job_id: Job ID to update
            updates: Dictionary with fields to update (name, passed, tags, custom-data, etc.)

        Returns:
            True if update was successful
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_auth_header(credentials)
        username = credentials.username

        try:
            response = await self.http_client.put(
                f"{base_url}/rest/v1/{username}/jobs/{job_id}",
                headers=headers,
                json=updates,
            )

            if response.status_code == 200:
                self.log.info("Updated job", job_id=job_id, updates=list(updates.keys()))
                return True
            else:
                self.log.error(
                    "Failed to update job",
                    job_id=job_id,
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            self.log.error("Error updating job", job_id=job_id, error=str(e))
            return False

    async def stop_job(
        self,
        credentials: IntegrationCredentials,
        job_id: str,
    ) -> bool:
        """
        Stop a running job.

        Args:
            credentials: Sauce Labs credentials
            job_id: Job ID to stop

        Returns:
            True if job was stopped successfully
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_auth_header(credentials)
        username = credentials.username

        try:
            response = await self.http_client.put(
                f"{base_url}/rest/v1/{username}/jobs/{job_id}/stop",
                headers=headers,
            )

            if response.status_code == 200:
                self.log.info("Stopped job", job_id=job_id)
                return True
            else:
                self.log.error(
                    "Failed to stop job",
                    job_id=job_id,
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            self.log.error("Error stopping job", job_id=job_id, error=str(e))
            return False


def create_saucelabs_plugin() -> SauceLabsPlugin:
    """Factory function for SauceLabsPlugin."""
    return SauceLabsPlugin()
