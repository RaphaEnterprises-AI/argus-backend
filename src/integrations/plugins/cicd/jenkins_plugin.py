"""Jenkins integration plugin.

Provides integration with Jenkins CI server for job management,
build monitoring, and queue status.
"""

import base64
import time
from datetime import datetime
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


class JenkinsPlugin(IntegrationPlugin):
    """Jenkins CI integration plugin.

    Features:
    - Monitor jobs and builds
    - View build queue status
    - Trigger builds with parameters
    - Access console output and artifacts
    - Track node/agent status
    - Pipeline support (Jenkinsfile)

    Authentication:
    - Basic Auth with username + API token
    - API tokens are generated per-user in Jenkins settings

    API Reference:
    - https://www.jenkins.io/doc/book/using/remote-access-api/
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Return plugin metadata."""
        return IntegrationMetadata(
            slug="jenkins",
            name="Jenkins",
            description="Open-source automation server for CI/CD pipelines",
            category=IntegrationCategory.CICD,
            auth_type=AuthType.BASIC_AUTH,
            icon_url="https://www.jenkins.io/images/logos/jenkins/jenkins.svg",
            docs_url="https://www.jenkins.io/doc/",
            website_url="https://www.jenkins.io/",
            features=[
                "jobs",
                "builds",
                "queue",
                "nodes",
                "pipelines",
                "artifacts",
                "console_output",
                "build_triggers",
            ],
            required_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Jenkins URL",
                    type="url",
                    required=True,
                    placeholder="https://jenkins.example.com",
                    help_text="URL to your Jenkins server",
                ),
                FieldDefinition(
                    name="username",
                    label="Username",
                    type="text",
                    required=True,
                    placeholder="jenkins-user",
                    help_text="Jenkins username",
                ),
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Jenkins API token (generate in User Settings > Configure > API Token)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.folder",
                    label="Folder Path",
                    type="text",
                    required=False,
                    placeholder="my-folder/sub-folder",
                    help_text="Optional folder path to scope job listing",
                ),
                FieldDefinition(
                    name="extra.crumb_enabled",
                    label="CSRF Protection",
                    type="select",
                    required=False,
                    help_text="Enable CSRF crumb fetching (required if enabled on server)",
                    options=[
                        {"label": "Auto-detect", "value": "auto"},
                        {"label": "Enabled", "value": "true"},
                        {"label": "Disabled", "value": "false"},
                    ],
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["jobs", "builds", "queue", "nodes"],
        )

    def _get_api_base(self, credentials: IntegrationCredentials) -> str:
        """Get the Jenkins API base URL."""
        base = (credentials.instance_url or "").rstrip("/")
        return base

    def _get_auth_header(self, credentials: IntegrationCredentials) -> str:
        """Get the Basic Auth header value."""
        auth_string = f"{credentials.username}:{credentials.api_key}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {encoded}"

    def _get_headers(self, credentials: IntegrationCredentials, crumb: Optional[dict] = None) -> dict[str, str]:
        """Get headers for Jenkins API requests."""
        headers = {
            "Authorization": self._get_auth_header(credentials),
            "Accept": "application/json",
        }
        if crumb:
            headers[crumb["crumbRequestField"]] = crumb["crumb"]
        return headers

    async def _get_crumb(self, credentials: IntegrationCredentials) -> Optional[dict]:
        """Fetch CSRF crumb if enabled on the server.

        Jenkins CSRF protection requires a crumb for POST requests.
        """
        api_base = self._get_api_base(credentials)
        crumb_enabled = credentials.extra.get("crumb_enabled", "auto")

        if crumb_enabled == "false":
            return None

        try:
            response = await self.http_client.get(
                f"{api_base}/crumbIssuer/api/json",
                headers={
                    "Authorization": self._get_auth_header(credentials),
                    "Accept": "application/json",
                },
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404 and crumb_enabled == "auto":
                # CSRF protection not enabled
                return None
            return None
        except Exception:
            return None

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the Jenkins connection.

        Validates the credentials by fetching server info.

        Args:
            credentials: Integration credentials with instance_url, username, api_key.

        Returns:
            ConnectionTestResult with success status and server info.
        """
        start_time = time.time()
        api_base = self._get_api_base(credentials)

        if not api_base:
            return ConnectionTestResult(
                success=False,
                message="Jenkins URL is required",
                error_code="MISSING_URL",
            )

        try:
            # Test authentication by getting server info
            response = await self.http_client.get(
                f"{api_base}/api/json",
                headers=self._get_headers(credentials),
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid username or API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check user permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            server_data = response.json()

            # Get additional system info
            permissions = ["read:jobs"]

            # Check if we can access build queue
            queue_resp = await self.http_client.get(
                f"{api_base}/queue/api/json",
                headers=self._get_headers(credentials),
            )
            if queue_resp.status_code == 200:
                permissions.append("read:queue")

            # Check CSRF crumb availability
            crumb = await self._get_crumb(credentials)
            if crumb:
                permissions.append("csrf:crumb")

            return ConnectionTestResult(
                success=True,
                message=f"Connected to Jenkins ({server_data.get('mode', 'unknown')} mode)",
                latency_ms=latency_ms,
                api_version=server_data.get("version"),
                permissions=permissions,
                account_info={
                    "mode": server_data.get("mode"),
                    "node_name": server_data.get("nodeName"),
                    "node_description": server_data.get("nodeDescription"),
                    "num_executors": server_data.get("numExecutors"),
                    "quieting_down": server_data.get("quietingDown"),
                    "slave_agent_port": server_data.get("slaveAgentPort"),
                    "use_crumbs": server_data.get("useCrumbs"),
                    "use_security": server_data.get("useSecurity"),
                    "primary_view": server_data.get("primaryView", {}).get("name"),
                    "job_count": len(server_data.get("jobs", [])),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Jenkins connection test failed", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=latency_ms,
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
        """Sync data from Jenkins.

        Fetches jobs, builds, queue, and node information.

        Args:
            credentials: Integration credentials.
            resources: Optional list of resources to sync (jobs, builds, queue, nodes).
            since: Optional datetime to fetch builds since.
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced Jenkins data.
        """
        started_at = datetime.utcnow()
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        data_points = 0

        resources_to_sync = resources or self.metadata.sync_resources
        api_base = self._get_api_base(credentials)
        headers = self._get_headers(credentials)

        try:
            # Get folder path if specified
            folder_path = credentials.extra.get("folder", "")
            base_path = f"{api_base}/job/{folder_path.replace('/', '/job/')}" if folder_path else api_base

            # Sync jobs
            if "jobs" in resources_to_sync:
                data["jobs"] = []
                try:
                    # Fetch jobs with details
                    tree_param = "jobs[name,url,color,buildable,fullName,fullDisplayName,description,lastBuild[number,timestamp,result,duration],healthReport[score,description]]"
                    resp = await self.http_client.get(
                        f"{base_path}/api/json",
                        headers=headers,
                        params={"tree": tree_param},
                    )
                    if resp.status_code == 200:
                        server_data = resp.json()
                        jobs = server_data.get("jobs", [])

                        for job in jobs[:limit]:
                            last_build = job.get("lastBuild") or {}
                            health_reports = job.get("healthReport", [])

                            data["jobs"].append({
                                "name": job.get("name"),
                                "full_name": job.get("fullName"),
                                "full_display_name": job.get("fullDisplayName"),
                                "url": job.get("url"),
                                "color": job.get("color"),
                                "buildable": job.get("buildable"),
                                "description": job.get("description"),
                                "last_build_number": last_build.get("number"),
                                "last_build_timestamp": last_build.get("timestamp"),
                                "last_build_result": last_build.get("result"),
                                "last_build_duration_ms": last_build.get("duration"),
                                "health_score": health_reports[0].get("score") if health_reports else None,
                                "health_description": health_reports[0].get("description") if health_reports else None,
                            })
                            data_points += 1
                except Exception as e:
                    errors.append(f"Failed to sync jobs: {str(e)}")

            # Sync builds
            if "builds" in resources_to_sync:
                data["builds"] = []
                # Get builds from each job
                jobs_to_check = data.get("jobs", [])
                if not jobs_to_check:
                    # Fetch job list if not already done
                    resp = await self.http_client.get(
                        f"{base_path}/api/json",
                        headers=headers,
                        params={"tree": "jobs[name,url]"},
                    )
                    if resp.status_code == 200:
                        jobs_to_check = resp.json().get("jobs", [])

                for job in jobs_to_check[:20]:  # Limit to 20 jobs for builds
                    try:
                        job_url = job.get("url", "")
                        if not job_url:
                            continue

                        # Get recent builds for this job
                        tree_param = "builds[number,url,timestamp,result,duration,displayName,description,building,estimatedDuration,executor,actions[causes[shortDescription,userId,userName]]]"
                        resp = await self.http_client.get(
                            f"{job_url}api/json",
                            headers=headers,
                            params={"tree": tree_param},
                        )
                        if resp.status_code == 200:
                            job_data = resp.json()
                            builds = job_data.get("builds", [])

                            for build in builds[:min(limit // len(jobs_to_check) + 1, 10)]:
                                # Filter by since if provided
                                timestamp = build.get("timestamp")
                                if since and timestamp:
                                    build_time = datetime.fromtimestamp(timestamp / 1000)
                                    if build_time < since:
                                        continue

                                # Extract cause info
                                causes = []
                                for action in build.get("actions", []):
                                    if action and "causes" in action:
                                        for cause in action["causes"]:
                                            causes.append({
                                                "description": cause.get("shortDescription"),
                                                "user_id": cause.get("userId"),
                                                "user_name": cause.get("userName"),
                                            })

                                data["builds"].append({
                                    "job_name": job.get("name"),
                                    "number": build.get("number"),
                                    "url": build.get("url"),
                                    "timestamp": timestamp,
                                    "result": build.get("result"),
                                    "duration_ms": build.get("duration"),
                                    "estimated_duration_ms": build.get("estimatedDuration"),
                                    "display_name": build.get("displayName"),
                                    "description": build.get("description"),
                                    "building": build.get("building"),
                                    "causes": causes,
                                })
                                data_points += 1

                    except Exception as e:
                        warnings.append(f"Failed to sync builds for {job.get('name')}: {str(e)}")

            # Sync queue
            if "queue" in resources_to_sync:
                data["queue"] = []
                try:
                    resp = await self.http_client.get(
                        f"{api_base}/queue/api/json",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        queue_data = resp.json()
                        items = queue_data.get("items", [])

                        for item in items[:limit]:
                            data["queue"].append({
                                "id": item.get("id"),
                                "task_name": item.get("task", {}).get("name"),
                                "task_url": item.get("task", {}).get("url"),
                                "why": item.get("why"),
                                "blocked": item.get("blocked"),
                                "buildable": item.get("buildable"),
                                "stuck": item.get("stuck"),
                                "in_queue_since": item.get("inQueueSince"),
                                "buildable_start_milliseconds": item.get("buildableStartMilliseconds"),
                            })
                            data_points += 1
                except Exception as e:
                    warnings.append(f"Failed to sync queue: {str(e)}")

            # Sync nodes
            if "nodes" in resources_to_sync:
                data["nodes"] = []
                try:
                    resp = await self.http_client.get(
                        f"{api_base}/computer/api/json",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        computer_data = resp.json()
                        computers = computer_data.get("computer", [])

                        for node in computers[:limit]:
                            monitor_data = node.get("monitorData", {})
                            disk_space = monitor_data.get("hudson.node_monitors.DiskSpaceMonitor", {})
                            swap_space = monitor_data.get("hudson.node_monitors.SwapSpaceMonitor", {})

                            data["nodes"].append({
                                "display_name": node.get("displayName"),
                                "description": node.get("description"),
                                "icon": node.get("icon"),
                                "idle": node.get("idle"),
                                "jnlp_agent": node.get("jnlpAgent"),
                                "launch_supported": node.get("launchSupported"),
                                "manual_launch_allowed": node.get("manualLaunchAllowed"),
                                "num_executors": node.get("numExecutors"),
                                "offline": node.get("offline"),
                                "offline_cause_reason": node.get("offlineCauseReason"),
                                "temporarily_offline": node.get("temporarilyOffline"),
                                "disk_space_path": disk_space.get("path") if disk_space else None,
                                "disk_space_bytes": disk_space.get("size") if disk_space else None,
                                "available_physical_memory": swap_space.get("availablePhysicalMemory") if swap_space else None,
                                "total_physical_memory": swap_space.get("totalPhysicalMemory") if swap_space else None,
                            })
                            data_points += 1
                except Exception as e:
                    warnings.append(f"Failed to sync nodes: {str(e)}")

            return SyncResult(
                success=True,
                message=f"Synced {data_points} items from Jenkins",
                data_points_synced=data_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error("Jenkins sync failed", error=str(e))
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def trigger_build(
        self,
        credentials: IntegrationCredentials,
        job_name: str,
        parameters: Optional[dict[str, Any]] = None,
        folder: Optional[str] = None,
    ) -> dict[str, Any]:
        """Trigger a build for a job.

        Args:
            credentials: Integration credentials.
            job_name: Name of the job to build.
            parameters: Optional build parameters.
            folder: Optional folder path containing the job.

        Returns:
            Dict with trigger result.
        """
        api_base = self._get_api_base(credentials)
        crumb = await self._get_crumb(credentials)
        headers = self._get_headers(credentials, crumb)

        try:
            # Build job URL path
            if folder:
                job_path = f"/job/{folder.replace('/', '/job/')}/job/{job_name}"
            else:
                job_path = f"/job/{job_name}"

            # Determine endpoint based on whether parameters are provided
            if parameters:
                url = f"{api_base}{job_path}/buildWithParameters"
                response = await self.http_client.post(
                    url,
                    headers=headers,
                    params=parameters,
                )
            else:
                url = f"{api_base}{job_path}/build"
                response = await self.http_client.post(
                    url,
                    headers=headers,
                )

            if response.status_code == 201:
                # Build queued - get queue item URL from Location header
                queue_url = response.headers.get("Location", "")
                return {
                    "success": True,
                    "message": f"Build triggered for {job_name}",
                    "queue_url": queue_url,
                }
            elif response.status_code == 409:
                return {
                    "success": False,
                    "message": "Build already in queue or running",
                    "error_code": "ALREADY_QUEUED",
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to trigger build: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to trigger build", error=str(e))
            return {
                "success": False,
                "message": f"Failed to trigger build: {str(e)}",
            }

    async def get_build_console_output(
        self,
        credentials: IntegrationCredentials,
        job_name: str,
        build_number: int,
        folder: Optional[str] = None,
        start: int = 0,
    ) -> dict[str, Any]:
        """Get console output for a build.

        Args:
            credentials: Integration credentials.
            job_name: Name of the job.
            build_number: Build number.
            folder: Optional folder path containing the job.
            start: Byte offset to start from (for progressive output).

        Returns:
            Dict with console text and metadata.
        """
        api_base = self._get_api_base(credentials)
        headers = self._get_headers(credentials)

        try:
            # Build job URL path
            if folder:
                job_path = f"/job/{folder.replace('/', '/job/')}/job/{job_name}"
            else:
                job_path = f"/job/{job_name}"

            response = await self.http_client.get(
                f"{api_base}{job_path}/{build_number}/logText/progressiveText",
                headers=headers,
                params={"start": start},
            )
            response.raise_for_status()

            # Get the X-Text-Size header for progressive fetching
            text_size = int(response.headers.get("X-Text-Size", 0))
            more_data = response.headers.get("X-More-Data", "false") == "true"

            return {
                "success": True,
                "text": response.text,
                "text_size": text_size,
                "more_data": more_data,
            }

        except Exception as e:
            logger.error("Failed to get console output", error=str(e))
            return {
                "success": False,
                "message": f"Failed to get console output: {str(e)}",
            }

    async def stop_build(
        self,
        credentials: IntegrationCredentials,
        job_name: str,
        build_number: int,
        folder: Optional[str] = None,
    ) -> dict[str, Any]:
        """Stop a running build.

        Args:
            credentials: Integration credentials.
            job_name: Name of the job.
            build_number: Build number.
            folder: Optional folder path containing the job.

        Returns:
            Dict with stop result.
        """
        api_base = self._get_api_base(credentials)
        crumb = await self._get_crumb(credentials)
        headers = self._get_headers(credentials, crumb)

        try:
            # Build job URL path
            if folder:
                job_path = f"/job/{folder.replace('/', '/job/')}/job/{job_name}"
            else:
                job_path = f"/job/{job_name}"

            response = await self.http_client.post(
                f"{api_base}{job_path}/{build_number}/stop",
                headers=headers,
            )

            if response.status_code in (200, 302):
                return {
                    "success": True,
                    "message": f"Build #{build_number} of {job_name} stopped",
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to stop build: {response.status_code}",
                    "error": response.text,
                }

        except Exception as e:
            logger.error("Failed to stop build", error=str(e))
            return {
                "success": False,
                "message": f"Failed to stop build: {str(e)}",
            }

    async def download_artifact(
        self,
        credentials: IntegrationCredentials,
        job_name: str,
        build_number: int,
        artifact_path: str,
        folder: Optional[str] = None,
    ) -> Optional[bytes]:
        """Download a build artifact.

        Args:
            credentials: Integration credentials.
            job_name: Name of the job.
            build_number: Build number.
            artifact_path: Relative path to the artifact.
            folder: Optional folder path containing the job.

        Returns:
            Artifact content as bytes, or None on failure.
        """
        api_base = self._get_api_base(credentials)
        headers = self._get_headers(credentials)

        try:
            # Build job URL path
            if folder:
                job_path = f"/job/{folder.replace('/', '/job/')}/job/{job_name}"
            else:
                job_path = f"/job/{job_name}"

            response = await self.http_client.get(
                f"{api_base}{job_path}/{build_number}/artifact/{artifact_path}",
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error("Failed to download artifact", error=str(e))
            return None
