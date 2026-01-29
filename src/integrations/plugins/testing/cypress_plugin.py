"""Cypress Cloud integration plugin.

Cypress Cloud (formerly Cypress Dashboard) is a service for recording test runs,
managing test artifacts, and analyzing test results from Cypress tests.

API Documentation: https://cloud.cypress.io/docs/api
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


class CypressPlugin(IntegrationPlugin):
    """Cypress Cloud integration for test recording and analytics.

    Features:
    - Test run recording and playback
    - Screenshot and video artifacts
    - Test analytics and insights
    - Flaky test detection
    - Parallelization insights

    Authentication:
    - Record Key (Bearer authentication)
    - API Key for project access

    API Endpoints:
    - GET /projects - List projects
    - GET /projects/{id}/runs - List runs for project
    - GET /runs/{id}/instances - Get run instances
    - GET /instances/{id} - Get instance details

    Example:
        async with CypressPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    CYPRESS_API_BASE = "https://cloud.cypress.io/api/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Cypress Cloud integration metadata."""
        return IntegrationMetadata(
            slug="cypress",
            name="Cypress Cloud",
            description="Test recording, analytics, and CI/CD integration for Cypress tests",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://cloud.cypress.io/favicon.ico",
            docs_url="https://docs.cypress.io/guides/cloud/introduction",
            website_url="https://cloud.cypress.io",
            features=[
                "test_recording",
                "screenshot_capture",
                "video_recording",
                "test_analytics",
                "flaky_test_detection",
                "parallelization",
                "ci_integration",
                "failure_analysis",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Record Key",
                    type="password",
                    required=True,
                    placeholder="Enter your Cypress record key",
                    help_text="Found in Cypress Cloud project settings",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="abc123",
                    help_text="Cypress Cloud project ID (6 character string)",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Organization Slug",
                    type="text",
                    required=False,
                    placeholder="my-organization",
                    help_text="Organization slug for multi-org accounts",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=[
                "projects",
                "runs",
                "instances",
                "specs",
                "tests",
                "screenshots",
                "videos",
            ],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for Cypress Cloud API."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Cypress Cloud connection by fetching projects.

        Args:
            credentials: Cypress credentials with record key.

        Returns:
            ConnectionTestResult with connection status.
        """
        start_time = time.time()

        try:
            # List projects accessible with this key
            url = f"{self.CYPRESS_API_BASE}/projects"
            if credentials.org_slug:
                url = f"{self.CYPRESS_API_BASE}/orgs/{credentials.org_slug}/projects"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Cypress record key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Record key does not have access to this organization",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )

            response.raise_for_status()
            data = response.json()

            # Extract projects from response
            projects = data if isinstance(data, list) else data.get("projects", [])

            # Get more details if specific project is provided
            project_details = None
            if credentials.project_slug:
                project_details = await self._get_project(
                    credentials, credentials.project_slug
                )

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Cypress Cloud",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=["read:projects", "read:runs", "read:artifacts"],
                account_info={
                    "total_projects": len(projects),
                    "projects": [
                        {
                            "id": p.get("id") or p.get("projectId"),
                            "name": p.get("name"),
                            "slug": p.get("slug"),
                            "public": p.get("public", False),
                        }
                        for p in projects[:5]
                    ],
                    "selected_project": project_details,
                    "organization": credentials.org_slug,
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Cypress Cloud connection test failed: {e}")
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                latency_ms=latency_ms,
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Cypress Cloud (runs, instances, specs).

        Args:
            credentials: Cypress credentials.
            resources: List of resources to sync.
            since: Sync runs created after this datetime.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with synced test data.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or [
            "runs",
            "instances",
            "specs",
            "tests",
            "screenshots",
        ]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            # Get project(s) first
            projects = await self._get_projects(credentials)
            if not projects:
                return SyncResult(
                    success=False,
                    message="No projects found for this record key",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=["No accessible projects"],
                )

            # Filter to specific project if provided
            if credentials.project_slug:
                projects = [
                    p for p in projects
                    if p.get("id") == credentials.project_slug
                    or p.get("projectId") == credentials.project_slug
                ]

            data["projects"] = projects[:limit]
            total_synced += len(data["projects"])

            # Sync runs for each project
            if "runs" in sync_resources:
                all_runs = []
                for project in projects[:5]:  # Limit to 5 projects
                    project_id = project.get("id") or project.get("projectId")
                    runs, run_errors = await self._sync_runs(
                        credentials, project_id, since, limit
                    )
                    all_runs.extend(runs)
                    errors.extend(run_errors)
                data["runs"] = all_runs
                total_synced += len(all_runs)

                # Sync instances for recent runs
                if "instances" in sync_resources and all_runs:
                    all_instances = []
                    for run in all_runs[:20]:  # Limit to 20 runs
                        instances, inst_errors = await self._sync_instances(
                            credentials, run["id"]
                        )
                        all_instances.extend(instances)
                        errors.extend(inst_errors)
                    data["instances"] = all_instances
                    total_synced += len(all_instances)

                    # Sync specs from instances
                    if "specs" in sync_resources:
                        data["specs"] = self._extract_specs(all_instances)
                        total_synced += len(data["specs"])

                    # Sync test results
                    if "tests" in sync_resources:
                        all_tests = []
                        for instance in all_instances[:50]:
                            tests = await self._sync_tests(
                                credentials, instance["id"]
                            )
                            all_tests.extend(tests)
                        data["tests"] = all_tests
                        total_synced += len(all_tests)

                    # Extract screenshots from instances
                    if "screenshots" in sync_resources:
                        data["screenshots"] = self._extract_screenshots(all_instances)
                        total_synced += len(data["screenshots"])

                    # Extract videos
                    if "videos" in sync_resources:
                        data["videos"] = self._extract_videos(all_instances)
                        total_synced += len(data["videos"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Cypress Cloud",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error(f"Cypress Cloud sync failed: {e}")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _get_project(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
    ) -> Optional[dict]:
        """Get details for a specific project."""
        try:
            url = f"{self.CYPRESS_API_BASE}/projects/{project_id}"
            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            return self._transform_project(response.json())
        except Exception as e:
            logger.error(f"Failed to get Cypress project: {e}")
            return None

    async def _get_projects(
        self,
        credentials: IntegrationCredentials,
    ) -> list[dict]:
        """Get all accessible projects."""
        try:
            url = f"{self.CYPRESS_API_BASE}/projects"
            if credentials.org_slug:
                url = f"{self.CYPRESS_API_BASE}/orgs/{credentials.org_slug}/projects"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            projects = data if isinstance(data, list) else data.get("projects", [])
            return [self._transform_project(p) for p in projects]

        except Exception as e:
            logger.error(f"Failed to get Cypress projects: {e}")
            return []

    async def _sync_runs(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync runs for a project."""
        runs = []
        errors = []

        try:
            url = f"{self.CYPRESS_API_BASE}/projects/{project_id}/runs"
            params = {"page[size]": min(limit, 100)}

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            runs_data = data if isinstance(data, list) else data.get("runs", [])

            for run in runs_data:
                run_data = self._transform_run(run, project_id)

                # Filter by since date if provided
                if since:
                    created_at = run_data.get("created_at")
                    if created_at:
                        try:
                            run_dt = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                            if run_dt.replace(tzinfo=None) < since.replace(tzinfo=None):
                                continue
                        except ValueError:
                            pass

                runs.append(run_data)

                if len(runs) >= limit:
                    break

        except Exception as e:
            errors.append(f"Failed to sync runs for project {project_id}: {str(e)}")
            logger.error(f"Cypress runs sync error: {e}")

        return runs, errors

    async def _sync_instances(
        self,
        credentials: IntegrationCredentials,
        run_id: str,
    ) -> tuple[list[dict], list[str]]:
        """Sync instances (spec files) for a run."""
        instances = []
        errors = []

        try:
            url = f"{self.CYPRESS_API_BASE}/runs/{run_id}/instances"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            instances_data = (
                data if isinstance(data, list) else data.get("instances", [])
            )

            for instance in instances_data:
                instances.append(self._transform_instance(instance, run_id))

        except Exception as e:
            errors.append(f"Failed to sync instances for run {run_id}: {str(e)}")
            logger.error(f"Cypress instances sync error: {e}")

        return instances, errors

    async def _sync_tests(
        self,
        credentials: IntegrationCredentials,
        instance_id: str,
    ) -> list[dict]:
        """Sync test results for an instance."""
        tests = []

        try:
            url = f"{self.CYPRESS_API_BASE}/instances/{instance_id}"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            # Extract tests from instance details
            tests_data = data.get("tests", [])
            for test in tests_data:
                tests.append(self._transform_test(test, instance_id))

        except Exception as e:
            logger.error(f"Failed to sync tests for instance {instance_id}: {e}")

        return tests

    def _extract_specs(self, instances: list[dict]) -> list[dict]:
        """Extract unique specs from instances."""
        specs_seen = set()
        specs = []

        for instance in instances:
            spec_path = instance.get("spec")
            if spec_path and spec_path not in specs_seen:
                specs_seen.add(spec_path)
                specs.append({
                    "path": spec_path,
                    "project_id": instance.get("project_id"),
                    "last_run_id": instance.get("run_id"),
                    "last_instance_id": instance.get("id"),
                    "last_status": instance.get("status"),
                })

        return specs

    def _extract_screenshots(self, instances: list[dict]) -> list[dict]:
        """Extract screenshots from instances."""
        screenshots = []

        for instance in instances:
            for screenshot in instance.get("screenshots", []):
                screenshots.append({
                    "id": screenshot.get("screenshotId"),
                    "instance_id": instance.get("id"),
                    "run_id": instance.get("run_id"),
                    "name": screenshot.get("name"),
                    "url": screenshot.get("url"),
                    "path": screenshot.get("path"),
                    "height": screenshot.get("height"),
                    "width": screenshot.get("width"),
                    "taken_at": screenshot.get("takenAt"),
                    "test_id": screenshot.get("testId"),
                    "test_attempt_index": screenshot.get("testAttemptIndex"),
                })

        return screenshots

    def _extract_videos(self, instances: list[dict]) -> list[dict]:
        """Extract videos from instances."""
        videos = []

        for instance in instances:
            video = instance.get("video")
            if video:
                videos.append({
                    "instance_id": instance.get("id"),
                    "run_id": instance.get("run_id"),
                    "url": video.get("url") if isinstance(video, dict) else video,
                    "uploaded": instance.get("videoUploaded", False),
                })

        return videos

    def _transform_project(self, project: dict) -> dict:
        """Transform Cypress project to standard format."""
        return {
            "id": project.get("id") or project.get("projectId"),
            "name": project.get("name"),
            "slug": project.get("slug"),
            "public": project.get("public", False),
            "created_at": project.get("createdAt"),
            "last_build_at": project.get("lastBuildAt"),
            "repo": project.get("repo"),
            "org_id": project.get("orgId"),
        }

    def _transform_run(self, run: dict, project_id: str) -> dict:
        """Transform Cypress run to standard format."""
        return {
            "id": run.get("id") or run.get("runId"),
            "project_id": project_id,
            "build_number": run.get("buildNumber"),
            "status": run.get("status"),
            "created_at": run.get("createdAt"),
            "completed_at": run.get("completedAt"),
            "branch": run.get("commit", {}).get("branch"),
            "commit_sha": run.get("commit", {}).get("sha"),
            "commit_message": run.get("commit", {}).get("message"),
            "commit_author": run.get("commit", {}).get("authorName"),
            "total_tests": run.get("totalTests"),
            "total_passed": run.get("totalPassed"),
            "total_failed": run.get("totalFailed"),
            "total_pending": run.get("totalPending"),
            "total_skipped": run.get("totalSkipped"),
            "total_duration": run.get("totalDuration"),
            "parallelization": run.get("parallelizationEnabled"),
            "ci_provider": run.get("ci", {}).get("provider") if run.get("ci") else None,
            "ci_build_id": run.get("ci", {}).get("buildId") if run.get("ci") else None,
            "tags": run.get("tags", []),
            "group": run.get("group"),
            "os": run.get("os", {}).get("name") if run.get("os") else None,
            "browser_name": (
                run.get("browser", {}).get("name") if run.get("browser") else None
            ),
            "browser_version": (
                run.get("browser", {}).get("version") if run.get("browser") else None
            ),
        }

    def _transform_instance(self, instance: dict, run_id: str) -> dict:
        """Transform Cypress instance to standard format."""
        return {
            "id": instance.get("id") or instance.get("instanceId"),
            "run_id": run_id,
            "spec": instance.get("spec"),
            "status": instance.get("status"),
            "started_at": instance.get("startedAt"),
            "completed_at": instance.get("completedAt"),
            "duration": instance.get("duration"),
            "tests_count": instance.get("testsCount"),
            "passed_count": instance.get("passedCount"),
            "failed_count": instance.get("failedCount"),
            "pending_count": instance.get("pendingCount"),
            "skipped_count": instance.get("skippedCount"),
            "flaky_count": instance.get("flakyCount", 0),
            "group_id": instance.get("groupId"),
            "machine_id": instance.get("machineId"),
            "screenshots": instance.get("screenshots", []),
            "video_url": instance.get("video"),
            "video_uploaded": instance.get("videoUploaded", False),
            "stdout": instance.get("stdout"),
            "claimed_at": instance.get("claimedAt"),
        }

    def _transform_test(self, test: dict, instance_id: str) -> dict:
        """Transform Cypress test to standard format."""
        return {
            "id": test.get("testId"),
            "instance_id": instance_id,
            "title": test.get("title", [])[-1] if test.get("title") else None,
            "full_title": " > ".join(test.get("title", [])),
            "state": test.get("state"),
            "body": test.get("body"),
            "display_error": test.get("displayError"),
            "duration": test.get("duration"),
            "attempts": len(test.get("attempts", [])),
            "is_flaky": any(
                a.get("state") != test.get("state")
                for a in test.get("attempts", [])
            ),
            "stack": test.get("stack"),
        }

    async def get_run_details(
        self,
        credentials: IntegrationCredentials,
        run_id: str,
    ) -> Optional[dict]:
        """Get detailed information about a specific run.

        Args:
            credentials: Cypress credentials.
            run_id: The run ID.

        Returns:
            Run details or None if not found.
        """
        try:
            url = f"{self.CYPRESS_API_BASE}/runs/{run_id}"
            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            return self._transform_run(response.json(), "unknown")
        except Exception as e:
            logger.error(f"Failed to get Cypress run details: {e}")
            return None

    async def get_flaky_tests(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get flaky tests for a project.

        Args:
            credentials: Cypress credentials.
            project_id: The project ID.
            limit: Maximum number of flaky tests to return.

        Returns:
            List of flaky test information.
        """
        try:
            url = f"{self.CYPRESS_API_BASE}/projects/{project_id}/flaky-tests"
            params = {"page[size]": min(limit, 100)}

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            flaky_tests = data if isinstance(data, list) else data.get("flakyTests", [])

            return [
                {
                    "spec": ft.get("spec"),
                    "test_title": ft.get("title"),
                    "flake_rate": ft.get("flakyRate"),
                    "flaky_runs": ft.get("flakyRuns"),
                    "total_runs": ft.get("totalRuns"),
                    "last_flaky_at": ft.get("lastFlakyAt"),
                }
                for ft in flaky_tests
            ]

        except Exception as e:
            logger.error(f"Failed to get Cypress flaky tests: {e}")
            return []

    async def get_test_analytics(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        days: int = 30,
    ) -> Optional[dict]:
        """Get test analytics for a project.

        Args:
            credentials: Cypress credentials.
            project_id: The project ID.
            days: Number of days to analyze.

        Returns:
            Analytics data or None.
        """
        try:
            url = f"{self.CYPRESS_API_BASE}/projects/{project_id}/analytics"
            params = {"timeRange": f"{days}d"}

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "total_runs": data.get("totalRuns"),
                "total_tests": data.get("totalTests"),
                "pass_rate": data.get("passRate"),
                "flaky_rate": data.get("flakyRate"),
                "avg_duration": data.get("avgDuration"),
                "top_failures": data.get("topFailures", []),
                "trend": data.get("trend", {}),
            }

        except Exception as e:
            logger.error(f"Failed to get Cypress analytics: {e}")
            return None
