"""Chromatic visual testing integration plugin.

Chromatic is a visual testing and review platform for UI components,
integrated with Storybook for component-based visual regression testing.

API Documentation: https://www.chromatic.com/docs/api
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


class ChromaticPlugin(IntegrationPlugin):
    """Chromatic visual testing integration.

    Features:
    - Storybook visual testing
    - Component snapshot comparisons
    - Build and approval workflow
    - UI review for design systems

    Authentication:
    - Project Token for API access
    - Token obtained from Chromatic project settings

    API Types:
    - REST API for project and build info
    - GraphQL API for detailed queries

    Example:
        async with ChromaticPlugin() as plugin:
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    CHROMATIC_API_BASE = "https://www.chromatic.com/api"
    CHROMATIC_GRAPHQL_URL = "https://www.chromatic.com/api/graphql"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Chromatic integration metadata."""
        return IntegrationMetadata(
            slug="chromatic",
            name="Chromatic",
            description="Visual testing and review platform for Storybook components",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.chromatic.com/favicon.ico",
            docs_url="https://www.chromatic.com/docs/",
            website_url="https://www.chromatic.com",
            features=[
                "visual_regression",
                "storybook_integration",
                "component_testing",
                "ui_review",
                "build_management",
                "baseline_management",
                "design_system_testing",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Project Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Chromatic project token",
                    help_text="Found in Chromatic project settings under 'Manage' > 'Configure'",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="project-id",
                    help_text="Chromatic project identifier",
                ),
                FieldDefinition(
                    name="extra",
                    label="App ID",
                    type="text",
                    required=False,
                    placeholder="app-id",
                    help_text="Chromatic app ID (found in project URL)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "builds", "components", "stories", "snapshots"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authentication headers for Chromatic API."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Chromatic connection by validating the project token.

        Args:
            credentials: Chromatic credentials with project token.

        Returns:
            ConnectionTestResult with connection status.
        """
        start_time = time.time()

        try:
            # Use GraphQL to query project information
            query = """
            query ValidateToken {
                viewer {
                    account {
                        id
                        name
                        avatarUrl
                    }
                    projects(first: 5) {
                        nodes {
                            id
                            name
                            slug
                            webUrl
                            lastBuild {
                                id
                                number
                                status
                                createdAt
                            }
                        }
                    }
                }
            }
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": query},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid Chromatic project token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            response.raise_for_status()
            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                error_msg = data["errors"][0].get("message", "Unknown GraphQL error")
                return ConnectionTestResult(
                    success=False,
                    message=f"Chromatic API error: {error_msg}",
                    latency_ms=latency_ms,
                    error_code="GRAPHQL_ERROR",
                    error_details={"errors": data["errors"]},
                )

            viewer = data.get("data", {}).get("viewer", {})
            account = viewer.get("account", {})
            projects = viewer.get("projects", {}).get("nodes", [])

            return ConnectionTestResult(
                success=True,
                message="Successfully connected to Chromatic",
                latency_ms=latency_ms,
                api_version="graphql",
                permissions=["read:projects", "read:builds", "read:stories"],
                account_info={
                    "account_id": account.get("id"),
                    "account_name": account.get("name"),
                    "avatar_url": account.get("avatarUrl"),
                    "total_projects": len(projects),
                    "projects": [
                        {
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "slug": p.get("slug"),
                            "web_url": p.get("webUrl"),
                        }
                        for p in projects
                    ],
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Chromatic connection test failed: {e}")
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
        """Sync data from Chromatic (builds, components, stories).

        Args:
            credentials: Chromatic credentials.
            resources: List of resources to sync (builds, components, stories).
            since: Sync builds created after this datetime.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with synced visual testing data.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["builds", "components", "stories", "snapshots"]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            # Get project and build information using GraphQL
            builds_data, builds_errors = await self._sync_builds(
                credentials, since, limit
            )
            data["builds"] = builds_data.get("builds", [])
            data["projects"] = builds_data.get("projects", [])
            total_synced += len(data["builds"]) + len(data["projects"])
            errors.extend(builds_errors)

            # Sync components/stories for recent builds
            if "components" in sync_resources or "stories" in sync_resources:
                for build in data["builds"][:10]:  # Limit to 10 recent builds
                    components, comp_errors = await self._sync_build_stories(
                        credentials, build["id"]
                    )
                    if "components" not in data:
                        data["components"] = []
                    if "stories" not in data:
                        data["stories"] = []

                    data["components"].extend(components.get("components", []))
                    data["stories"].extend(components.get("stories", []))
                    total_synced += len(components.get("components", []))
                    total_synced += len(components.get("stories", []))
                    errors.extend(comp_errors)

            # Sync snapshots for builds with changes
            if "snapshots" in sync_resources:
                builds_with_changes = [
                    b for b in data["builds"] if b.get("changes_count", 0) > 0
                ]
                all_snapshots = []
                for build in builds_with_changes[:5]:
                    snapshots, snap_errors = await self._sync_snapshots(
                        credentials, build["id"], limit
                    )
                    all_snapshots.extend(snapshots)
                    errors.extend(snap_errors)
                data["snapshots"] = all_snapshots
                total_synced += len(all_snapshots)

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Chromatic",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error(f"Chromatic sync failed: {e}")
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
        since: Optional[datetime],
        limit: int,
    ) -> tuple[dict, list[str]]:
        """Sync builds using GraphQL."""
        errors = []
        result = {"builds": [], "projects": []}

        try:
            # Determine date filter
            since_filter = ""
            if since:
                since_iso = since.isoformat()
                since_filter = f', after: "{since_iso}"'

            query = f"""
            query GetBuilds {{
                viewer {{
                    projects(first: 10) {{
                        nodes {{
                            id
                            name
                            slug
                            webUrl
                            repository
                            builds(first: {min(limit, 50)}{since_filter}) {{
                                nodes {{
                                    id
                                    number
                                    branch
                                    commit
                                    commitUrl
                                    status
                                    result
                                    createdAt
                                    completedAt
                                    webUrl
                                    storybookUrl
                                    changeCount
                                    componentCount
                                    specCount
                                    errorCount
                                    isAutoAcceptBranchBuild
                                    uncommittedHash
                                    actualCaptureCount
                                    inheritedCaptureCount
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                errors.append(f"GraphQL error: {data['errors'][0].get('message')}")
                return result, errors

            projects = (
                data.get("data", {}).get("viewer", {}).get("projects", {}).get("nodes", [])
            )

            for project in projects:
                result["projects"].append(self._transform_project(project))

                for build in project.get("builds", {}).get("nodes", []):
                    result["builds"].append(
                        self._transform_build(build, project.get("id"))
                    )

        except Exception as e:
            errors.append(f"Failed to sync builds: {str(e)}")
            logger.error(f"Chromatic builds sync error: {e}")

        return result, errors

    async def _sync_build_stories(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
    ) -> tuple[dict, list[str]]:
        """Sync stories and components for a build."""
        errors = []
        result = {"components": [], "stories": []}

        try:
            query = f"""
            query GetBuildStories {{
                build(id: "{build_id}") {{
                    id
                    specs {{
                        nodes {{
                            id
                            name
                            component {{
                                id
                                name
                                displayName
                            }}
                            story {{
                                id
                                name
                                displayName
                            }}
                            parameters
                        }}
                    }}
                }}
            }}
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                errors.append(f"GraphQL error: {data['errors'][0].get('message')}")
                return result, errors

            specs = (
                data.get("data", {}).get("build", {}).get("specs", {}).get("nodes", [])
            )

            components_seen = set()
            for spec in specs:
                component = spec.get("component", {})
                story = spec.get("story", {})

                # Add unique components
                comp_id = component.get("id")
                if comp_id and comp_id not in components_seen:
                    components_seen.add(comp_id)
                    result["components"].append({
                        "id": comp_id,
                        "name": component.get("name"),
                        "display_name": component.get("displayName"),
                        "build_id": build_id,
                    })

                # Add story
                if story:
                    result["stories"].append({
                        "id": story.get("id"),
                        "name": story.get("name"),
                        "display_name": story.get("displayName"),
                        "component_id": comp_id,
                        "build_id": build_id,
                        "spec_id": spec.get("id"),
                        "parameters": spec.get("parameters"),
                    })

        except Exception as e:
            errors.append(f"Failed to sync stories for build {build_id}: {str(e)}")
            logger.error(f"Chromatic stories sync error: {e}")

        return result, errors

    async def _sync_snapshots(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
        limit: int,
    ) -> tuple[list[dict], list[str]]:
        """Sync snapshots (visual comparisons) for a build."""
        errors = []
        snapshots = []

        try:
            query = f"""
            query GetBuildSnapshots {{
                build(id: "{build_id}") {{
                    id
                    tests(first: {min(limit, 100)}) {{
                        nodes {{
                            id
                            status
                            result
                            spec {{
                                name
                                component {{
                                    name
                                }}
                            }}
                            comparisons {{
                                id
                                browser {{
                                    id
                                    name
                                }}
                                viewport {{
                                    width
                                    height
                                }}
                                result
                                headCapture {{
                                    id
                                    captureUrl
                                }}
                                baseCapture {{
                                    id
                                    captureUrl
                                }}
                                diffImage {{
                                    imageUrl
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                errors.append(f"GraphQL error: {data['errors'][0].get('message')}")
                return snapshots, errors

            tests = (
                data.get("data", {}).get("build", {}).get("tests", {}).get("nodes", [])
            )

            for test in tests:
                spec = test.get("spec", {})
                for comparison in test.get("comparisons", []):
                    viewport = comparison.get("viewport", {})
                    browser = comparison.get("browser", {})

                    snapshots.append({
                        "id": comparison.get("id"),
                        "test_id": test.get("id"),
                        "build_id": build_id,
                        "test_status": test.get("status"),
                        "test_result": test.get("result"),
                        "spec_name": spec.get("name"),
                        "component_name": spec.get("component", {}).get("name"),
                        "browser": browser.get("name"),
                        "viewport_width": viewport.get("width"),
                        "viewport_height": viewport.get("height"),
                        "comparison_result": comparison.get("result"),
                        "head_capture_url": comparison.get("headCapture", {}).get(
                            "captureUrl"
                        ),
                        "base_capture_url": comparison.get("baseCapture", {}).get(
                            "captureUrl"
                        ),
                        "diff_image_url": comparison.get("diffImage", {}).get(
                            "imageUrl"
                        ),
                    })

        except Exception as e:
            errors.append(f"Failed to sync snapshots for build {build_id}: {str(e)}")
            logger.error(f"Chromatic snapshots sync error: {e}")

        return snapshots, errors

    def _transform_project(self, project: dict) -> dict:
        """Transform Chromatic project to standard format."""
        return {
            "id": project.get("id"),
            "name": project.get("name"),
            "slug": project.get("slug"),
            "web_url": project.get("webUrl"),
            "repository": project.get("repository"),
        }

    def _transform_build(self, build: dict, project_id: str) -> dict:
        """Transform Chromatic build to standard format."""
        return {
            "id": build.get("id"),
            "project_id": project_id,
            "number": build.get("number"),
            "branch": build.get("branch"),
            "commit": build.get("commit"),
            "commit_url": build.get("commitUrl"),
            "status": build.get("status"),
            "result": build.get("result"),
            "created_at": build.get("createdAt"),
            "completed_at": build.get("completedAt"),
            "web_url": build.get("webUrl"),
            "storybook_url": build.get("storybookUrl"),
            "changes_count": build.get("changeCount"),
            "component_count": build.get("componentCount"),
            "spec_count": build.get("specCount"),
            "error_count": build.get("errorCount"),
            "is_auto_accept": build.get("isAutoAcceptBranchBuild"),
            "actual_capture_count": build.get("actualCaptureCount"),
            "inherited_capture_count": build.get("inheritedCaptureCount"),
        }

    async def accept_build(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
    ) -> bool:
        """Accept all changes in a build.

        Args:
            credentials: Chromatic credentials.
            build_id: The build ID to accept.

        Returns:
            True if acceptance succeeded.
        """
        try:
            mutation = """
            mutation AcceptBuild($buildId: ID!) {
                acceptBuild(buildId: $buildId) {
                    id
                    status
                }
            }
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": mutation, "variables": {"buildId": build_id}},
            )
            response.raise_for_status()
            data = response.json()

            return "errors" not in data

        except Exception as e:
            logger.error(f"Failed to accept Chromatic build: {e}")
            return False

    async def get_build_url(
        self,
        credentials: IntegrationCredentials,
        build_id: str,
    ) -> Optional[str]:
        """Get the web URL for a build.

        Args:
            credentials: Chromatic credentials.
            build_id: The build ID.

        Returns:
            Build web URL or None.
        """
        try:
            query = f"""
            query GetBuildUrl {{
                build(id: "{build_id}") {{
                    webUrl
                }}
            }}
            """

            response = await self.http_client.post(
                self.CHROMATIC_GRAPHQL_URL,
                headers=self._get_headers(credentials),
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()

            return data.get("data", {}).get("build", {}).get("webUrl")

        except Exception as e:
            logger.error(f"Failed to get Chromatic build URL: {e}")
            return None
