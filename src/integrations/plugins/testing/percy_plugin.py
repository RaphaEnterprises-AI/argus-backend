"""Percy (BrowserStack) visual testing integration plugin.

Percy is a visual testing platform that captures screenshots across browsers
and compares them to detect visual regressions.

API Documentation: https://www.browserstack.com/docs/percy/api-reference
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


class PercyPlugin(IntegrationPlugin):
    """Percy (BrowserStack) visual testing integration.

    Features:
    - Visual regression testing
    - Cross-browser screenshot comparisons
    - Build and snapshot management
    - Baseline approval workflow

    Authentication:
    - Project Token (Bearer authentication)
    - Token obtained from Percy project settings

    API Endpoints:
    - GET /projects - List projects
    - GET /builds - List builds
    - GET /builds/{id}/snapshots - Get build snapshots
    - GET /snapshots/{id}/comparisons - Get snapshot comparisons

    Example:
        async with PercyPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    PERCY_API_BASE = "https://percy.io/api/v1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Percy integration metadata."""
        return IntegrationMetadata(
            slug="percy",
            name="Percy",
            description="Visual testing and review platform by BrowserStack",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://percy.io/favicon.ico",
            docs_url="https://www.browserstack.com/docs/percy",
            website_url="https://percy.io",
            features=[
                "visual_regression",
                "cross_browser_testing",
                "screenshot_comparisons",
                "build_management",
                "snapshot_approvals",
                "baseline_management",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Project Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Percy project token",
                    help_text="Found in Percy project settings under 'Project token'",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project Slug",
                    type="text",
                    required=False,
                    placeholder="org-slug/project-slug",
                    help_text="Project identifier in format 'org-slug/project-slug'",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "builds", "snapshots", "comparisons"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for Percy API."""
        return {
            "Authorization": f"Token {credentials.access_token}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Percy connection by fetching projects.

        Args:
            credentials: Percy credentials with project token.

        Returns:
            ConnectionTestResult with connection status.
        """
        start_time = time.time()

        try:
            # Percy API uses token-based auth, verify by listing projects
            # If a project_slug is provided, try to get that specific project
            if credentials.project_slug:
                url = f"{self.PERCY_API_BASE}/projects/{credentials.project_slug}"
            else:
                # List all projects accessible with this token
                url = f"{self.PERCY_API_BASE}/projects"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Percy project token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message="Project not found or token does not have access",
                    latency_ms=latency_ms,
                    error_code="NOT_FOUND",
                    error_details={"project_slug": credentials.project_slug},
                )

            response.raise_for_status()
            data = response.json()

            # Extract project info
            if credentials.project_slug:
                project = data.get("data", {})
                project_attrs = project.get("attributes", {})
                account_info = {
                    "project_id": project.get("id"),
                    "project_name": project_attrs.get("name"),
                    "project_slug": project_attrs.get("full-slug"),
                    "default_branch": project_attrs.get("default-base-branch"),
                }
            else:
                projects = data.get("data", [])
                account_info = {
                    "total_projects": len(projects),
                    "projects": [
                        {
                            "id": p.get("id"),
                            "name": p.get("attributes", {}).get("name"),
                            "slug": p.get("attributes", {}).get("full-slug"),
                        }
                        for p in projects[:5]  # Limit to first 5
                    ],
                }

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Percy",
                latency_ms=latency_ms,
                api_version="v1",
                permissions=["read:projects", "read:builds", "read:snapshots"],
                account_info=account_info,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Percy connection test failed: {e}")
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
        """Sync data from Percy (builds, snapshots, comparisons).

        Args:
            credentials: Percy credentials.
            resources: List of resources to sync (builds, snapshots, comparisons).
            since: Sync builds created after this datetime.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with synced visual testing data.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["builds", "snapshots", "comparisons"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            # Get project info first
            project_slug = credentials.project_slug
            if not project_slug:
                # Try to get first project from list
                projects_response = await self.http_client.get(
                    f"{self.PERCY_API_BASE}/projects",
                    headers=self._get_headers(credentials),
                )
                projects_response.raise_for_status()
                projects_data = projects_response.json()
                projects = projects_data.get("data", [])

                if not projects:
                    return SyncResult(
                        success=False,
                        message="No projects found for this token",
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        errors=["No accessible projects"],
                    )

                project_slug = projects[0].get("attributes", {}).get("full-slug")
                data["projects"] = [
                    self._transform_project(p) for p in projects[:limit]
                ]
                total_synced += len(projects[:limit])

            # Sync builds
            if "builds" in sync_resources:
                builds, build_errors = await self._sync_builds(
                    credentials, project_slug, since, limit
                )
                data["builds"] = builds
                total_synced += len(builds)
                errors.extend(build_errors)

                # Sync snapshots for recent builds
                if "snapshots" in sync_resources and builds:
                    all_snapshots = []
                    for build in builds[:10]:  # Limit to 10 most recent builds
                        snapshots, snap_errors = await self._sync_snapshots(
                            credentials, build["id"], limit
                        )
                        all_snapshots.extend(snapshots)
                        errors.extend(snap_errors)
                    data["snapshots"] = all_snapshots
                    total_synced += len(all_snapshots)

                    # Sync comparisons for snapshots
                    if "comparisons" in sync_resources and all_snapshots:
                        all_comparisons = []
                        for snapshot in all_snapshots[:50]:  # Limit to 50 snapshots
                            comparisons, comp_errors = await self._sync_comparisons(
                                credentials, snapshot["id"]
                            )
                            all_comparisons.extend(comparisons)
                            errors.extend(comp_errors)
                        data["comparisons"] = all_comparisons
                        total_synced += len(all_comparisons)

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Percy",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error(f"Percy sync failed: {e}")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_builds(
        self,
        credentials: IntegrationCredentials,
        project_slug: str,
        since: Optional[datetime],
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync builds for a project."""
        builds = []
        errors = []

        try:
            url = f"{self.PERCY_API_BASE}/projects/{project_slug}/builds"
            params = {"page[limit]": min(limit, 100)}

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            for build in data.get("data", []):
                build_data = self._transform_build(build)

                # Filter by since date if provided
                if since:
                    created_at = build_data.get("created_at")
                    if created_at and datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    ) < since.replace(tzinfo=None):
                        continue

                builds.append(build_data)

                if len(builds) >= limit:
                    break

        except Exception as e:
            errors.append(f"Failed to sync builds: {str(e)}")
            logger.error(f"Percy builds sync error: {e}")

        return builds, errors

    async def _sync_snapshots(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync snapshots for a build."""
        snapshots = []
        errors = []

        try:
            url = f"{self.PERCY_API_BASE}/builds/{build_id}/snapshots"
            params = {"page[limit]": min(limit, 100)}

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            for snapshot in data.get("data", []):
                snapshots.append(self._transform_snapshot(snapshot, build_id))

        except Exception as e:
            errors.append(f"Failed to sync snapshots for build {build_id}: {str(e)}")
            logger.error(f"Percy snapshots sync error: {e}")

        return snapshots, errors

    async def _sync_comparisons(
        self,
        credentials: IntegrationCredentials,
        snapshot_id: str,
    ) -> tuple[list[dict], list[str]]:
        """Sync comparisons for a snapshot."""
        comparisons = []
        errors = []

        try:
            url = f"{self.PERCY_API_BASE}/snapshots/{snapshot_id}/comparisons"

            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()

            for comparison in data.get("data", []):
                comparisons.append(self._transform_comparison(comparison, snapshot_id))

        except Exception as e:
            errors.append(
                f"Failed to sync comparisons for snapshot {snapshot_id}: {str(e)}"
            )
            logger.error(f"Percy comparisons sync error: {e}")

        return comparisons, errors

    def _transform_project(self, project: dict) -> dict:
        """Transform Percy project to standard format."""
        attrs = project.get("attributes", {})
        return {
            "id": project.get("id"),
            "name": attrs.get("name"),
            "slug": attrs.get("full-slug"),
            "default_branch": attrs.get("default-base-branch"),
            "is_enabled": attrs.get("is-enabled"),
            "created_at": attrs.get("created-at"),
            "updated_at": attrs.get("updated-at"),
        }

    def _transform_build(self, build: dict) -> dict:
        """Transform Percy build to standard format."""
        attrs = build.get("attributes", {})
        return {
            "id": build.get("id"),
            "branch": attrs.get("branch"),
            "commit_sha": attrs.get("commit-sha"),
            "commit_message": attrs.get("commit-html-url"),
            "state": attrs.get("state"),
            "review_state": attrs.get("review-state"),
            "review_state_reason": attrs.get("review-state-reason"),
            "total_snapshots": attrs.get("total-snapshots"),
            "total_snapshots_requesting_changes": attrs.get(
                "total-snapshots-requesting-changes"
            ),
            "total_snapshots_unreviewed": attrs.get("total-snapshots-unreviewed"),
            "total_comparisons": attrs.get("total-comparisons"),
            "total_comparisons_diff": attrs.get("total-comparisons-diff"),
            "total_comparisons_finished": attrs.get("total-comparisons-finished"),
            "created_at": attrs.get("created-at"),
            "updated_at": attrs.get("updated-at"),
            "finished_at": attrs.get("finished-at"),
            "approved_at": attrs.get("approved-at"),
            "web_url": attrs.get("web-url"),
            "is_pull_request": attrs.get("is-pull-request"),
            "pull_request_number": attrs.get("pull-request-number"),
            "pull_request_title": attrs.get("pull-request-title"),
        }

    def _transform_snapshot(self, snapshot: dict, build_id: str) -> dict:
        """Transform Percy snapshot to standard format."""
        attrs = snapshot.get("attributes", {})
        return {
            "id": snapshot.get("id"),
            "build_id": build_id,
            "name": attrs.get("name"),
            "review_state": attrs.get("review-state"),
            "review_state_reason": attrs.get("review-state-reason"),
            "fingerprint": attrs.get("fingerprint"),
            "total_comparisons": attrs.get("total-comparisons"),
            "created_at": attrs.get("created-at"),
            "updated_at": attrs.get("updated-at"),
        }

    def _transform_comparison(self, comparison: dict, snapshot_id: str) -> dict:
        """Transform Percy comparison to standard format."""
        attrs = comparison.get("attributes", {})
        return {
            "id": comparison.get("id"),
            "snapshot_id": snapshot_id,
            "state": attrs.get("state"),
            "width": attrs.get("width"),
            "diff_ratio": attrs.get("diff-ratio"),
            "diff_image_url": attrs.get("diff-image-url"),
            "head_image_url": attrs.get("head-image-url"),
            "base_image_url": attrs.get("base-image-url"),
            "created_at": attrs.get("created-at"),
            "updated_at": attrs.get("updated-at"),
        }

    async def get_build_details(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
    ) -> Optional[dict]:
        """Get detailed information about a specific build.

        Args:
            credentials: Percy credentials.
            build_id: The build ID.

        Returns:
            Build details or None if not found.
        """
        try:
            url = f"{self.PERCY_API_BASE}/builds/{build_id}"
            response = await self.http_client.get(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            data = response.json()
            return self._transform_build(data.get("data", {}))
        except Exception as e:
            logger.error(f"Failed to get Percy build details: {e}")
            return None

    async def approve_build(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
    ) -> bool:
        """Approve all snapshots in a build.

        Args:
            credentials: Percy credentials.
            build_id: The build ID to approve.

        Returns:
            True if approval succeeded.
        """
        try:
            url = f"{self.PERCY_API_BASE}/builds/{build_id}/approve"
            response = await self.http_client.post(
                url,
                headers=self._get_headers(credentials),
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to approve Percy build: {e}")
            return False
