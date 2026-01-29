"""Azure DevOps integration plugin.

Provides integration with Azure DevOps for repositories, pull requests,
work items, and pipelines via Personal Access Token and REST API.
"""

import base64
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


class AzureDevOpsPlugin(IntegrationPlugin):
    """Azure DevOps integration using Personal Access Token (PAT).

    Features:
    - Personal Access Token authentication
    - Project and repository management
    - Pull request tracking
    - Work items integration
    - Pipelines and builds
    - Webhook support for code, PR, and work item events

    API Documentation: https://learn.microsoft.com/en-us/rest/api/azure/devops/
    """

    DEFAULT_BASE_URL = "https://dev.azure.com"
    API_VERSION = "7.1"

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="azure_devops",
            name="Azure DevOps",
            description="Source code, pull requests, work items, pipelines, and CI/CD",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://cdn.vsassets.io/content/icons/favicon.ico",
            docs_url="https://learn.microsoft.com/en-us/rest/api/azure/devops/",
            website_url="https://dev.azure.com",
            features=[
                "repositories",
                "pull_requests",
                "work_items",
                "pipelines",
                "builds",
                "webhooks",
                "commits",
                "branches",
                "boards",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Personal Access Token",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help_text="Personal Access Token (PAT) with appropriate scopes",
                ),
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=True,
                    placeholder="my-organization",
                    help_text="Your Azure DevOps organization name",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="project_slug",
                    label="Project",
                    type="text",
                    required=False,
                    placeholder="my-project",
                    help_text="Optional: Filter to a specific project",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://dev.azure.com",
                    help_text="For Azure DevOps Server. Leave empty for Azure DevOps Services",
                ),
            ],
            oauth_config=None,  # Using PAT instead of OAuth
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["projects", "repositories", "pull_requests"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the Azure DevOps instance URL."""
        if credentials.instance_url:
            return credentials.instance_url.rstrip("/")
        return self.DEFAULT_BASE_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Azure DevOps API requests with Basic Auth (PAT)."""
        # Azure DevOps uses Basic Auth with PAT: username can be empty
        auth_string = f":{credentials.access_token}"
        auth_bytes = auth_string.encode("utf-8")
        auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")

        return {
            "Authorization": f"Basic {auth_b64}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Argus-E2E-Testing-Agent/1.0",
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        credentials: IntegrationCredentials,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        use_org_level: bool = True,
    ) -> httpx.Response:
        """Make an API request to Azure DevOps.

        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            credentials: Azure DevOps credentials
            params: Query parameters
            json_data: JSON body data
            use_org_level: If True, includes org in URL path

        Returns:
            HTTP response
        """
        base_url = self._get_base_url(credentials)

        if use_org_level:
            url = f"{base_url}/{credentials.org_slug}{endpoint}"
        else:
            url = f"{base_url}{endpoint}"

        # Add API version to params
        if params is None:
            params = {}
        params["api-version"] = self.API_VERSION

        headers = self._get_headers(credentials)

        response = await self.http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        )

        return response

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Azure DevOps connection by fetching connection data.

        Uses GET /_apis/connectionData endpoint to verify token validity.
        """
        start_time = time.time()

        try:
            response = await self._make_request(
                method="GET",
                endpoint="/_apis/connectionData",
                credentials=credentials,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid or expired Personal Access Token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check PAT permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 404:
                return ConnectionTestResult(
                    success=False,
                    message=f"Organization '{credentials.org_slug}' not found",
                    latency_ms=latency_ms,
                    error_code="NOT_FOUND",
                )

            response.raise_for_status()
            connection_data = response.json()

            # Extract user info
            authenticated_user = connection_data.get("authenticatedUser", {})

            return ConnectionTestResult(
                success=True,
                message=f"Connected to {credentials.org_slug}",
                latency_ms=latency_ms,
                api_version=self.API_VERSION,
                permissions=[],  # Azure DevOps doesn't expose PAT scopes easily
                account_info={
                    "id": authenticated_user.get("id"),
                    "display_name": authenticated_user.get("providerDisplayName"),
                    "descriptor": authenticated_user.get("descriptor"),
                    "organization": credentials.org_slug,
                    "instance_id": connection_data.get("instanceId"),
                    "deployment_type": connection_data.get("deploymentType"),
                    "location_url": connection_data.get("locationServiceData", {}).get("locationUrl"),
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="TIMEOUT",
            )
        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to Azure DevOps: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="CONNECTION_ERROR",
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                error_code="UNKNOWN_ERROR",
                error_details={"exception": type(e).__name__},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Azure DevOps.

        Syncs projects, repositories, and pull requests.

        Args:
            credentials: Azure DevOps credentials
            resources: List of resources to sync (projects, repositories, pull_requests)
            since: Only fetch data updated since this datetime
            limit: Maximum items per resource type

        Returns:
            SyncResult containing synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"azure_devops_{started_at.timestamp()}"

        if resources is None:
            resources = ["projects", "repositories", "pull_requests"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync projects
            if "projects" in resources:
                projects_result = await self._sync_projects(
                    credentials, since, limit
                )
                data["projects"] = projects_result["data"]
                total_points += projects_result["count"]
                errors.extend(projects_result.get("errors", []))
                warnings.extend(projects_result.get("warnings", []))

            # Sync repositories
            if "repositories" in resources:
                repos_result = await self._sync_repositories(
                    credentials,
                    projects=data.get("projects", []),
                    since=since,
                    limit=limit,
                )
                data["repositories"] = repos_result["data"]
                total_points += repos_result["count"]
                errors.extend(repos_result.get("errors", []))
                warnings.extend(repos_result.get("warnings", []))

            # Sync pull requests
            if "pull_requests" in resources:
                prs_result = await self._sync_pull_requests(
                    credentials,
                    repos=data.get("repositories", []),
                    since=since,
                    limit=limit,
                )
                data["pull_requests"] = prs_result["data"]
                total_points += prs_result["count"]
                errors.extend(prs_result.get("errors", []))
                warnings.extend(prs_result.get("warnings", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} data points from Azure DevOps",
                data_points_synced=total_points,
                sync_id=sync_id,
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
                data_points_synced=total_points,
                sync_id=sync_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                warnings=warnings,
                data=data,
            )

    async def _sync_projects(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch projects for the organization."""
        projects: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            params: dict[str, Any] = {
                "$top": min(limit, 100),
                "stateFilter": "all",
            }

            # If specific project is specified, only fetch that
            if credentials.project_slug:
                response = await self._make_request(
                    method="GET",
                    endpoint=f"/_apis/projects/{credentials.project_slug}",
                    credentials=credentials,
                )

                if response.status_code == 200:
                    project = response.json()
                    projects.append(self._format_project(project))
                else:
                    errors.append(f"Failed to fetch project: {response.status_code}")
            else:
                response = await self._make_request(
                    method="GET",
                    endpoint="/_apis/projects",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    errors.append(f"Failed to fetch projects: {response.status_code}")
                else:
                    response_data = response.json()
                    for project in response_data.get("value", [])[:limit]:
                        last_update = project.get("lastUpdateTime")
                        if since and last_update:
                            project_date = datetime.fromisoformat(
                                last_update.replace("Z", "+00:00")
                            )
                            if project_date.replace(tzinfo=None) < since:
                                continue

                        projects.append(self._format_project(project))

        except Exception as e:
            errors.append(f"Error syncing projects: {str(e)}")

        return {
            "data": projects,
            "count": len(projects),
            "errors": errors,
            "warnings": warnings,
        }

    def _format_project(self, project: dict[str, Any]) -> dict[str, Any]:
        """Format project data to a consistent structure."""
        return {
            "id": project.get("id"),
            "name": project.get("name"),
            "description": project.get("description"),
            "url": project.get("url"),
            "state": project.get("state"),
            "revision": project.get("revision"),
            "visibility": project.get("visibility"),
            "last_update_time": project.get("lastUpdateTime"),
            "default_team": project.get("defaultTeam", {}).get("name"),
        }

    async def _sync_repositories(
        self,
        credentials: IntegrationCredentials,
        projects: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch repositories for projects."""
        repos: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Limit projects to check
        projects_to_check = projects[:10] if len(projects) > 10 else projects

        if len(projects) > 10:
            warnings.append(f"Limited repo sync to first 10 projects out of {len(projects)}")

        for project in projects_to_check:
            try:
                response = await self._make_request(
                    method="GET",
                    endpoint=f"/{project['name']}/_apis/git/repositories",
                    credentials=credentials,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch repos for {project['name']}")
                    continue

                response_data = response.json()
                for repo in response_data.get("value", []):
                    repos.append({
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "project_id": project.get("id"),
                        "project_name": project.get("name"),
                        "url": repo.get("url"),
                        "remote_url": repo.get("remoteUrl"),
                        "ssh_url": repo.get("sshUrl"),
                        "web_url": repo.get("webUrl"),
                        "default_branch": repo.get("defaultBranch", "refs/heads/main"),
                        "size": repo.get("size"),
                        "is_disabled": repo.get("isDisabled", False),
                    })

                    if len(repos) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching repos for {project['name']}: {str(e)}")

        return {
            "data": repos[:limit],
            "count": len(repos[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def _sync_pull_requests(
        self,
        credentials: IntegrationCredentials,
        repos: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch recent pull requests from repositories."""
        pull_requests: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Limit repos to check
        repos_to_check = repos[:10] if len(repos) > 10 else repos

        if len(repos) > 10:
            warnings.append(f"Limited PR sync to first 10 repos out of {len(repos)}")

        for repo in repos_to_check:
            try:
                params: dict[str, Any] = {
                    "$top": min(limit // max(len(repos_to_check), 1) + 1, 50),
                    "searchCriteria.status": "all",
                }

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/{repo['project_name']}/_apis/git/repositories/{repo['id']}/pullrequests",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch PRs for {repo['name']}")
                    continue

                response_data = response.json()
                for pr in response_data.get("value", []):
                    created_date = pr.get("creationDate")
                    if since and created_date:
                        pr_date = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
                        if pr_date.replace(tzinfo=None) < since:
                            continue

                    pull_requests.append({
                        "id": pr.get("pullRequestId"),
                        "title": pr.get("title"),
                        "description": pr.get("description"),
                        "status": pr.get("status"),
                        "repository_id": repo.get("id"),
                        "repository_name": repo.get("name"),
                        "project_name": repo.get("project_name"),
                        "url": pr.get("url"),
                        "source_ref_name": pr.get("sourceRefName"),
                        "target_ref_name": pr.get("targetRefName"),
                        "created_by": pr.get("createdBy", {}).get("displayName"),
                        "created_by_id": pr.get("createdBy", {}).get("id"),
                        "creation_date": created_date,
                        "closed_date": pr.get("closedDate"),
                        "is_draft": pr.get("isDraft", False),
                        "merge_status": pr.get("mergeStatus"),
                        "reviewers": [
                            {
                                "display_name": r.get("displayName"),
                                "vote": r.get("vote"),
                            }
                            for r in pr.get("reviewers", [])
                        ],
                    })

                    if len(pull_requests) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching PRs for {repo['name']}: {str(e)}")

        return {
            "data": pull_requests[:limit],
            "count": len(pull_requests[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def get_project(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific project by ID or name."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/_apis/projects/{project_id}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def get_repository(
        self,
        credentials: IntegrationCredentials,
        project_name: str,
        repo_id: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific repository."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/{project_name}/_apis/git/repositories/{repo_id}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def get_pull_request(
        self,
        credentials: IntegrationCredentials,
        project_name: str,
        repo_id: str,
        pr_id: int,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific pull request."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/{project_name}/_apis/git/repositories/{repo_id}/pullrequests/{pr_id}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def get_work_items(
        self,
        credentials: IntegrationCredentials,
        project_name: str,
        ids: list[int],
    ) -> list[dict[str, Any]]:
        """Fetch work items by IDs.

        Args:
            credentials: Azure DevOps credentials
            project_name: Project name
            ids: List of work item IDs

        Returns:
            List of work item data
        """
        if not ids:
            return []

        try:
            # Azure DevOps limits to 200 IDs per request
            ids_str = ",".join(str(i) for i in ids[:200])

            response = await self._make_request(
                method="GET",
                endpoint=f"/{project_name}/_apis/wit/workitems",
                credentials=credentials,
                params={
                    "ids": ids_str,
                    "$expand": "all",
                },
            )

            if response.status_code == 200:
                return response.json().get("value", [])
            return []
        except Exception:
            return []

    async def create_service_hook(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        webhook_url: str,
        event_type: str,
        secret: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a service hook (webhook) for events.

        Args:
            credentials: Azure DevOps credentials
            project_id: Project ID
            webhook_url: URL to receive webhook events
            event_type: Event type (e.g., "git.push", "git.pullrequest.created")
            secret: Optional secret for verification

        Returns:
            Service hook configuration if created successfully
        """
        try:
            payload = {
                "publisherId": "tfs",
                "eventType": event_type,
                "resourceVersion": "1.0",
                "consumerId": "webHooks",
                "consumerActionId": "httpRequest",
                "publisherInputs": {
                    "projectId": project_id,
                },
                "consumerInputs": {
                    "url": webhook_url,
                },
            }

            if secret:
                payload["consumerInputs"]["basicAuthPassword"] = secret

            response = await self._make_request(
                method="POST",
                endpoint="/_apis/hooks/subscriptions",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def trigger_build(
        self,
        credentials: IntegrationCredentials,
        project_name: str,
        definition_id: int,
        source_branch: Optional[str] = None,
        parameters: Optional[dict[str, str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Queue a new build.

        Args:
            credentials: Azure DevOps credentials
            project_name: Project name
            definition_id: Build definition ID
            source_branch: Optional source branch (e.g., "refs/heads/main")
            parameters: Optional build parameters

        Returns:
            Build data if queued successfully
        """
        try:
            payload: dict[str, Any] = {
                "definition": {
                    "id": definition_id,
                },
            }

            if source_branch:
                payload["sourceBranch"] = source_branch

            if parameters:
                payload["parameters"] = parameters

            response = await self._make_request(
                method="POST",
                endpoint=f"/{project_name}/_apis/build/builds",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
