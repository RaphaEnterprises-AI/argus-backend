"""Bitbucket integration plugin.

Provides integration with Bitbucket Cloud for repositories, pull requests,
and pipelines via App Password authentication and REST API 2.0.
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


class BitbucketPlugin(IntegrationPlugin):
    """Bitbucket integration using App Password and REST API 2.0.

    Features:
    - App Password authentication (recommended for API access)
    - Repository listing and metadata
    - Pull request management
    - Pipelines integration
    - Webhook support for push, PR events

    API Documentation: https://developer.atlassian.com/cloud/bitbucket/rest/
    """

    BASE_URL = "https://api.bitbucket.org/2.0"

    # Rate limit tracking (Bitbucket uses a different model)
    _rate_limit_retry_after: Optional[float] = None

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="bitbucket",
            name="Bitbucket",
            description="Source code hosting, pull requests, and Bitbucket Pipelines CI/CD",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.APP_PASSWORD,
            icon_url="https://wac-cdn.atlassian.com/dam/jcr:c942540c-53aa-4f14-9e43-2f8c0de7c888/Bitbucket-blue.svg",
            docs_url="https://developer.atlassian.com/cloud/bitbucket/rest/",
            website_url="https://bitbucket.org",
            features=[
                "repositories",
                "pull_requests",
                "pipelines",
                "webhooks",
                "code_search",
                "commits",
                "branches",
            ],
            required_fields=[
                FieldDefinition(
                    name="username",
                    label="Username",
                    type="text",
                    required=True,
                    placeholder="your-username",
                    help_text="Your Bitbucket username (not email)",
                ),
                FieldDefinition(
                    name="password",
                    label="App Password",
                    type="password",
                    required=True,
                    placeholder="xxxxxxxxxxxxxxxx",
                    help_text="App Password created in Bitbucket settings (not your login password)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="workspace_id",
                    label="Workspace",
                    type="text",
                    required=False,
                    placeholder="my-workspace",
                    help_text="Optional: Filter to a specific workspace (formerly team)",
                ),
            ],
            oauth_config=None,  # Using App Password instead of OAuth
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["repositories", "pull_requests"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Bitbucket API requests with Basic Auth."""
        auth_string = f"{credentials.username}:{credentials.password}"
        auth_bytes = auth_string.encode("utf-8")
        auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")

        return {
            "Authorization": f"Basic {auth_b64}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Argus-E2E-Testing-Agent/1.0",
        }

    async def _wait_for_rate_limit(self) -> None:
        """Wait if rate limited."""
        if self._rate_limit_retry_after and self._rate_limit_retry_after > time.time():
            wait_time = self._rate_limit_retry_after - time.time()
            if wait_time > 0:
                wait_time = min(wait_time, 60)
                import asyncio
                await asyncio.sleep(wait_time)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        credentials: IntegrationCredentials,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an API request with rate limit handling."""
        await self._wait_for_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers(credentials)

        response = await self.http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        )

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            self._rate_limit_retry_after = time.time() + float(retry_after)

        return response

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Bitbucket connection by fetching authenticated user info.

        Uses GET /user endpoint to verify credentials.
        """
        start_time = time.time()

        try:
            response = await self._make_request(
                method="GET",
                endpoint="/user",
                credentials=credentials,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid username or app password",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - check app password permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            if response.status_code == 429:
                return ConnectionTestResult(
                    success=False,
                    message="Rate limit exceeded. Please try again later.",
                    latency_ms=latency_ms,
                    error_code="RATE_LIMITED",
                    error_details={
                        "retry_after": self._rate_limit_retry_after,
                    },
                )

            response.raise_for_status()
            user_data = response.json()

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('username', 'unknown')}",
                latency_ms=latency_ms,
                api_version="2.0",
                permissions=[],  # Bitbucket doesn't easily expose app password scopes
                account_info={
                    "uuid": user_data.get("uuid"),
                    "username": user_data.get("username"),
                    "display_name": user_data.get("display_name"),
                    "nickname": user_data.get("nickname"),
                    "account_id": user_data.get("account_id"),
                    "links": {
                        "html": user_data.get("links", {}).get("html", {}).get("href"),
                        "avatar": user_data.get("links", {}).get("avatar", {}).get("href"),
                    },
                    "type": user_data.get("type"),
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
                message=f"Failed to connect to Bitbucket API: {str(e)}",
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
        """Sync data from Bitbucket.

        Syncs repositories and pull requests based on the requested resources.

        Args:
            credentials: Bitbucket credentials
            resources: List of resources to sync (repositories, pull_requests)
            since: Only fetch data updated since this datetime
            limit: Maximum items per resource type

        Returns:
            SyncResult containing synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"bitbucket_{started_at.timestamp()}"

        if resources is None:
            resources = ["repositories", "pull_requests"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_points = 0

        try:
            # Sync repositories
            if "repositories" in resources:
                repos_result = await self._sync_repositories(
                    credentials, since, limit
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
                message=f"Synced {total_points} data points from Bitbucket",
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

    async def _sync_repositories(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch repositories for the authenticated user."""
        repos: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            params: dict[str, Any] = {
                "pagelen": min(limit, 100),
                "sort": "-updated_on",
            }

            # Fetch from specific workspace if specified
            if credentials.workspace_id:
                endpoint = f"/repositories/{credentials.workspace_id}"
            else:
                endpoint = f"/repositories/{credentials.username}"

            response = await self._make_request(
                method="GET",
                endpoint=endpoint,
                credentials=credentials,
                params=params,
            )

            if response.status_code != 200:
                errors.append(f"Failed to fetch repositories: {response.status_code}")
            else:
                response_data = response.json()
                repos_data = response_data.get("values", [])

                for repo in repos_data[:limit]:
                    updated_on = repo.get("updated_on")
                    if since and updated_on:
                        repo_date = datetime.fromisoformat(updated_on.replace("Z", "+00:00"))
                        if repo_date.replace(tzinfo=None) < since:
                            continue

                    repos.append({
                        "uuid": repo.get("uuid"),
                        "name": repo.get("name"),
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "is_private": repo.get("is_private"),
                        "html_url": repo.get("links", {}).get("html", {}).get("href"),
                        "clone_url": self._get_clone_url(repo),
                        "mainbranch": repo.get("mainbranch", {}).get("name", "main"),
                        "language": repo.get("language"),
                        "updated_on": updated_on,
                        "created_on": repo.get("created_on"),
                        "size": repo.get("size"),
                        "has_issues": repo.get("has_issues", False),
                        "has_wiki": repo.get("has_wiki", False),
                        "fork_policy": repo.get("fork_policy"),
                        "project": repo.get("project", {}).get("name"),
                        "workspace": repo.get("workspace", {}).get("slug"),
                    })

        except Exception as e:
            errors.append(f"Error syncing repositories: {str(e)}")

        return {
            "data": repos,
            "count": len(repos),
            "errors": errors,
            "warnings": warnings,
        }

    def _get_clone_url(self, repo: dict[str, Any]) -> Optional[str]:
        """Extract HTTPS clone URL from repository data."""
        clone_links = repo.get("links", {}).get("clone", [])
        for link in clone_links:
            if link.get("name") == "https":
                return link.get("href")
        return None

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
                    "state": "OPEN,MERGED,DECLINED,SUPERSEDED",
                    "pagelen": min(limit // max(len(repos_to_check), 1) + 1, 50),
                    "sort": "-updated_on",
                }

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/repositories/{repo['full_name']}/pullrequests",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch PRs for {repo['full_name']}")
                    continue

                response_data = response.json()
                prs_data = response_data.get("values", [])

                for pr in prs_data:
                    updated_on = pr.get("updated_on")
                    if since and updated_on:
                        pr_date = datetime.fromisoformat(updated_on.replace("Z", "+00:00"))
                        if pr_date.replace(tzinfo=None) < since:
                            continue

                    pull_requests.append({
                        "id": pr.get("id"),
                        "title": pr.get("title"),
                        "state": pr.get("state"),
                        "html_url": pr.get("links", {}).get("html", {}).get("href"),
                        "repository": repo["full_name"],
                        "author": pr.get("author", {}).get("display_name"),
                        "author_username": pr.get("author", {}).get("username"),
                        "created_on": pr.get("created_on"),
                        "updated_on": updated_on,
                        "merge_commit": pr.get("merge_commit", {}).get("hash"),
                        "source_branch": pr.get("source", {}).get("branch", {}).get("name"),
                        "destination_branch": pr.get("destination", {}).get("branch", {}).get("name"),
                        "comment_count": pr.get("comment_count", 0),
                        "task_count": pr.get("task_count", 0),
                    })

                    if len(pull_requests) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching PRs for {repo['full_name']}: {str(e)}")

        return {
            "data": pull_requests[:limit],
            "count": len(pull_requests[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def get_repository(
        self,
        credentials: IntegrationCredentials,
        workspace: str,
        repo_slug: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific repository."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/repositories/{workspace}/{repo_slug}",
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
        workspace: str,
        repo_slug: str,
        pr_id: int,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific pull request."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}",
                credentials=credentials,
            )

            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    async def create_webhook(
        self,
        credentials: IntegrationCredentials,
        workspace: str,
        repo_slug: str,
        webhook_url: str,
        events: list[str],
        description: str = "Argus Integration Webhook",
        secret: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a webhook for a repository.

        Args:
            credentials: Bitbucket credentials
            workspace: Workspace slug
            repo_slug: Repository slug
            webhook_url: URL to receive webhook events
            events: List of events to subscribe to (e.g., ["repo:push", "pullrequest:created"])
            description: Webhook description
            secret: Optional webhook secret for verification

        Returns:
            Webhook configuration if created successfully
        """
        try:
            payload: dict[str, Any] = {
                "description": description,
                "url": webhook_url,
                "active": True,
                "events": events,
            }

            if secret:
                payload["secret"] = secret

            response = await self._make_request(
                method="POST",
                endpoint=f"/repositories/{workspace}/{repo_slug}/hooks",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 201:
                return response.json()
            return None
        except Exception:
            return None

    async def list_workspaces(
        self,
        credentials: IntegrationCredentials,
    ) -> list[dict[str, Any]]:
        """List all workspaces the user has access to."""
        workspaces: list[dict[str, Any]] = []

        try:
            response = await self._make_request(
                method="GET",
                endpoint="/user/permissions/workspaces",
                credentials=credentials,
                params={"pagelen": 100},
            )

            if response.status_code == 200:
                response_data = response.json()
                for item in response_data.get("values", []):
                    workspace = item.get("workspace", {})
                    workspaces.append({
                        "uuid": workspace.get("uuid"),
                        "slug": workspace.get("slug"),
                        "name": workspace.get("name"),
                        "permission": item.get("permission"),
                    })

        except Exception:
            pass

        return workspaces
