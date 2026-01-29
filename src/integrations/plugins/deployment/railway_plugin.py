"""Railway deployment integration plugin.

This plugin provides integration with Railway's deployment platform,
allowing sync of projects, services, and deployments via GraphQL API.
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


class RailwayPlugin(IntegrationPlugin):
    """Railway deployment integration plugin.

    Integrates with Railway's GraphQL API for:
    - Projects: List and manage Railway projects
    - Services: Access services within projects
    - Deployments: Monitor deployment status and history

    Authentication: API Token (Bearer)
    API Documentation: https://docs.railway.app/reference/public-api
    """

    BASE_URL = "https://backboard.railway.app/graphql/v2"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Railway integration metadata."""
        return IntegrationMetadata(
            slug="railway",
            name="Railway",
            description="Deploy applications with instant infrastructure provisioning",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://railway.app/brand/logo-light.png",
            docs_url="https://docs.railway.app/reference/public-api",
            website_url="https://railway.app",
            features=[
                "projects",
                "services",
                "deployments",
                "environments",
                "variables",
                "domains",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="Enter your Railway API token",
                    help_text="Generate from Railway Dashboard > Account Settings > Tokens",
                ),
            ],
            optional_fields=[],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "services", "deployments"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get authorization headers for Railway API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }

    async def _execute_graphql(
        self,
        credentials: IntegrationCredentials,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query against Railway's API.

        Args:
            credentials: IntegrationCredentials with access_token.
            query: GraphQL query string.
            variables: Optional variables for the query.

        Returns:
            The data field from the GraphQL response.

        Raises:
            Exception: If the GraphQL response contains errors.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await self.http_client.post(
            self.BASE_URL,
            headers=self._get_headers(credentials),
            json=payload,
        )
        response.raise_for_status()

        result = response.json()

        if "errors" in result and result["errors"]:
            error_messages = [e.get("message", str(e)) for e in result["errors"]]
            raise Exception(f"GraphQL errors: {'; '.join(error_messages)}")

        return result.get("data", {})

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Railway API.

        Tests authentication by fetching the current user's information.

        Args:
            credentials: IntegrationCredentials with access_token.

        Returns:
            ConnectionTestResult indicating success or failure.
        """
        start_time = time.time()

        query = """
        query {
            me {
                id
                email
                name
                avatar
                createdAt
            }
        }
        """

        try:
            data = await self._execute_graphql(credentials, query)
            latency_ms = (time.time() - start_time) * 1000

            me = data.get("me", {})

            if me:
                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Railway",
                    latency_ms=latency_ms,
                    api_version="v2 (GraphQL)",
                    permissions=["read", "write"],
                    account_info={
                        "id": me.get("id"),
                        "email": me.get("email"),
                        "name": me.get("name"),
                        "avatar": me.get("avatar"),
                        "created_at": me.get("createdAt"),
                    },
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message="Failed to retrieve user information",
                    latency_ms=latency_ms,
                    error_code="NO_USER_DATA",
                )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            if e.response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired API token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )
            elif e.response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check token permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"HTTP error: {e.response.status_code}",
                    latency_ms=latency_ms,
                    error_code="HTTP_ERROR",
                    error_details={
                        "status_code": e.response.status_code,
                        "body": e.response.text[:500],
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"exception": str(e)},
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                success=False,
                message=f"Error: {str(e)}",
                latency_ms=latency_ms,
                error_code="GRAPHQL_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Railway.

        Syncs projects, services, and deployments from Railway.

        Args:
            credentials: IntegrationCredentials with access_token.
            resources: List of resources to sync (projects, services, deployments).
            since: Optional datetime for incremental sync.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with synced data.
        """
        started_at = datetime.utcnow()
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        # Default to all resources if none specified
        if resources is None:
            resources = ["projects", "services", "deployments"]

        try:
            # Sync projects first (needed for services and deployments)
            projects_list = []
            if "projects" in resources or "services" in resources or "deployments" in resources:
                try:
                    projects_list = await self._sync_projects(credentials, limit)
                    if "projects" in resources:
                        data["projects"] = projects_list
                        total_points += len(projects_list)
                except Exception as e:
                    errors.append(f"Failed to sync projects: {str(e)}")

            # Sync services for each project
            if "services" in resources and projects_list:
                try:
                    all_services = []

                    for project in projects_list[:20]:  # Limit to 20 projects
                        try:
                            project_services = await self._sync_project_services(
                                credentials, project["id"]
                            )
                            all_services.extend(project_services)
                        except Exception as e:
                            warnings.append(
                                f"Failed to sync services for project {project.get('name', project['id'])}: {str(e)}"
                            )

                    data["services"] = all_services
                    total_points += len(all_services)
                except Exception as e:
                    errors.append(f"Failed to sync services: {str(e)}")

            # Sync deployments for each project
            if "deployments" in resources and projects_list:
                try:
                    all_deployments = []
                    deploys_per_project = max(1, limit // len(projects_list)) if projects_list else limit

                    for project in projects_list[:20]:  # Limit to 20 projects
                        try:
                            project_deployments = await self._sync_project_deployments(
                                credentials, project["id"], deploys_per_project, since
                            )
                            all_deployments.extend(project_deployments)
                        except Exception as e:
                            warnings.append(
                                f"Failed to sync deployments for project {project.get('name', project['id'])}: {str(e)}"
                            )

                    data["deployments"] = all_deployments
                    total_points += len(all_deployments)
                except Exception as e:
                    errors.append(f"Failed to sync deployments: {str(e)}")

            return SyncResult(
                success=len(errors) == 0,
                message=(
                    f"Synced {total_points} items from Railway"
                    if len(errors) == 0
                    else f"Partial sync completed with {len(errors)} errors"
                ),
                data_points_synced=total_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
            )

    async def _sync_projects(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync projects from Railway."""
        query = """
        query($first: Int) {
            projects(first: $first) {
                edges {
                    node {
                        id
                        name
                        description
                        createdAt
                        updatedAt
                        isPublic
                        prDeploys
                        prForks
                        team {
                            id
                            name
                        }
                        environments {
                            edges {
                                node {
                                    id
                                    name
                                    isEphemeral
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        data = await self._execute_graphql(
            credentials, query, {"first": min(limit, 100)}
        )

        projects = data.get("projects", {}).get("edges", [])

        return [
            {
                "id": project["node"].get("id"),
                "name": project["node"].get("name"),
                "description": project["node"].get("description"),
                "created_at": project["node"].get("createdAt"),
                "updated_at": project["node"].get("updatedAt"),
                "is_public": project["node"].get("isPublic"),
                "pr_deploys": project["node"].get("prDeploys"),
                "pr_forks": project["node"].get("prForks"),
                "team": project["node"].get("team"),
                "environments": [
                    {
                        "id": env["node"].get("id"),
                        "name": env["node"].get("name"),
                        "is_ephemeral": env["node"].get("isEphemeral"),
                    }
                    for env in project["node"].get("environments", {}).get("edges", [])
                ],
            }
            for project in projects
        ]

    async def _sync_project_services(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
    ) -> list[dict[str, Any]]:
        """Sync services for a specific project."""
        query = """
        query($projectId: String!) {
            project(id: $projectId) {
                services {
                    edges {
                        node {
                            id
                            name
                            icon
                            createdAt
                            updatedAt
                            templateServiceId
                            templateThreadSlug
                        }
                    }
                }
            }
        }
        """

        data = await self._execute_graphql(
            credentials, query, {"projectId": project_id}
        )

        services = data.get("project", {}).get("services", {}).get("edges", [])

        return [
            {
                "id": service["node"].get("id"),
                "project_id": project_id,
                "name": service["node"].get("name"),
                "icon": service["node"].get("icon"),
                "created_at": service["node"].get("createdAt"),
                "updated_at": service["node"].get("updatedAt"),
                "template_service_id": service["node"].get("templateServiceId"),
                "template_thread_slug": service["node"].get("templateThreadSlug"),
            }
            for service in services
        ]

    async def _sync_project_deployments(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        limit: int,
        since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Sync deployments for a specific project."""
        query = """
        query($projectId: String!, $first: Int) {
            project(id: $projectId) {
                deployments(first: $first) {
                    edges {
                        node {
                            id
                            status
                            createdAt
                            updatedAt
                            staticUrl
                            service {
                                id
                                name
                            }
                            environment {
                                id
                                name
                            }
                            meta {
                                branch
                                commitHash
                                commitMessage
                                commitAuthor
                                repo
                            }
                        }
                    }
                }
            }
        }
        """

        data = await self._execute_graphql(
            credentials, query, {"projectId": project_id, "first": min(limit, 100)}
        )

        deployments = data.get("project", {}).get("deployments", {}).get("edges", [])

        result = []
        for deployment in deployments:
            node = deployment["node"]

            # Filter by since date if provided
            if since:
                created_at = node.get("createdAt")
                if created_at and created_at < since.isoformat():
                    continue

            meta = node.get("meta") or {}
            result.append({
                "id": node.get("id"),
                "project_id": project_id,
                "service": node.get("service"),
                "environment": node.get("environment"),
                "status": node.get("status"),
                "static_url": node.get("staticUrl"),
                "created_at": node.get("createdAt"),
                "updated_at": node.get("updatedAt"),
                "meta": {
                    "branch": meta.get("branch"),
                    "commit_hash": meta.get("commitHash"),
                    "commit_message": meta.get("commitMessage"),
                    "commit_author": meta.get("commitAuthor"),
                    "repo": meta.get("repo"),
                },
            })

        return result

    async def get_deployment_status(
        self,
        credentials: IntegrationCredentials,
        deployment_id: str,
    ) -> dict[str, Any]:
        """Get detailed status of a specific deployment.

        Args:
            credentials: IntegrationCredentials with access_token.
            deployment_id: The deployment ID to check.

        Returns:
            Dictionary with deployment status details.
        """
        query = """
        query($deploymentId: String!) {
            deployment(id: $deploymentId) {
                id
                status
                createdAt
                updatedAt
                staticUrl
                service {
                    id
                    name
                }
                environment {
                    id
                    name
                }
                meta {
                    branch
                    commitHash
                    commitMessage
                    commitAuthor
                    repo
                }
            }
        }
        """

        data = await self._execute_graphql(
            credentials, query, {"deploymentId": deployment_id}
        )

        deployment = data.get("deployment", {})
        meta = deployment.get("meta") or {}

        return {
            "id": deployment.get("id"),
            "status": deployment.get("status"),
            "static_url": deployment.get("staticUrl"),
            "created_at": deployment.get("createdAt"),
            "updated_at": deployment.get("updatedAt"),
            "service": deployment.get("service"),
            "environment": deployment.get("environment"),
            "meta": {
                "branch": meta.get("branch"),
                "commit_hash": meta.get("commitHash"),
                "commit_message": meta.get("commitMessage"),
                "commit_author": meta.get("commitAuthor"),
                "repo": meta.get("repo"),
            },
        }

    async def trigger_deployment(
        self,
        credentials: IntegrationCredentials,
        service_id: str,
        environment_id: str,
    ) -> dict[str, Any]:
        """Trigger a new deployment for a service.

        Args:
            credentials: IntegrationCredentials with access_token.
            service_id: The service ID to deploy.
            environment_id: The environment ID to deploy to.

        Returns:
            Dictionary with the new deployment information.
        """
        mutation = """
        mutation($serviceId: String!, $environmentId: String!) {
            serviceInstanceRedeploy(
                serviceId: $serviceId
                environmentId: $environmentId
            )
        }
        """

        data = await self._execute_graphql(
            credentials,
            mutation,
            {"serviceId": service_id, "environmentId": environment_id},
        )

        return {
            "success": data.get("serviceInstanceRedeploy", False),
            "service_id": service_id,
            "environment_id": environment_id,
            "triggered_at": datetime.utcnow().isoformat(),
        }

    async def get_service_logs(
        self,
        credentials: IntegrationCredentials,
        deployment_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get logs for a deployment.

        Args:
            credentials: IntegrationCredentials with access_token.
            deployment_id: The deployment ID.
            limit: Maximum number of log entries.

        Returns:
            List of log entries.
        """
        query = """
        query($deploymentId: String!, $limit: Int) {
            deploymentLogs(deploymentId: $deploymentId, limit: $limit) {
                timestamp
                message
                severity
            }
        }
        """

        data = await self._execute_graphql(
            credentials, query, {"deploymentId": deployment_id, "limit": limit}
        )

        logs = data.get("deploymentLogs", [])

        return [
            {
                "timestamp": log.get("timestamp"),
                "message": log.get("message"),
                "severity": log.get("severity"),
            }
            for log in logs
        ]
