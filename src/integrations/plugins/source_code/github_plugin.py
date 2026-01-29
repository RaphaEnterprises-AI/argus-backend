"""GitHub integration plugin.

Provides integration with GitHub for repositories, pull requests, issues,
and webhook events via GitHub Apps OAuth2 and REST API v3.
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
    OAuthConfig,
    SyncResult,
)


class GitHubPlugin(IntegrationPlugin):
    """GitHub integration using OAuth2 (GitHub Apps) and REST API v3.

    Features:
    - OAuth2 flow with GitHub Apps for secure authentication
    - Repository listing and metadata
    - Pull request management
    - Issue tracking
    - Webhook support for push, PR, and issue events

    API Documentation: https://docs.github.com/en/rest
    """

    BASE_URL = "https://api.github.com"

    # Rate limit tracking
    _rate_limit_remaining: int = 5000
    _rate_limit_reset: Optional[float] = None

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug="github",
            name="GitHub",
            description="Source code hosting, pull requests, issues, and CI/CD with GitHub Actions",
            category=IntegrationCategory.SOURCE_CODE,
            auth_type=AuthType.OAUTH2,
            icon_url="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
            docs_url="https://docs.github.com/en/rest",
            website_url="https://github.com",
            features=[
                "repositories",
                "pull_requests",
                "issues",
                "webhooks",
                "code_search",
                "actions",
                "commits",
                "branches",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Access Token",
                    type="password",
                    required=True,
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                    help_text="Personal access token or OAuth token from GitHub App",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=False,
                    placeholder="my-organization",
                    help_text="Optional: Filter to a specific organization",
                ),
            ],
            oauth_config=OAuthConfig(
                authorize_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                scopes=["repo", "read:org", "read:user", "user:email"],
                client_id_env_var="GITHUB_CLIENT_ID",
                client_secret_env_var="GITHUB_CLIENT_SECRET",
                redirect_uri_path="/api/v1/integrations/oauth/callback/github",
                pkce_required=False,
                state_required=True,
            ),
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["repositories", "pull_requests", "issues"],
        )

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for GitHub API requests."""
        return {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Argus-E2E-Testing-Agent/1.0",
        }

    def _update_rate_limits(self, response: httpx.Response) -> None:
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Remaining" in response.headers:
            self._rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in response.headers:
            self._rate_limit_reset = float(response.headers["X-RateLimit-Reset"])

    async def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exhausted."""
        if self._rate_limit_remaining <= 1 and self._rate_limit_reset:
            wait_time = self._rate_limit_reset - time.time()
            if wait_time > 0:
                # Cap wait time at 60 seconds to avoid indefinite waits
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

        self._update_rate_limits(response)
        return response

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test GitHub connection by fetching authenticated user info.

        Uses GET /user endpoint to verify token validity and permissions.
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
                    message="Invalid or expired access token",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            if response.status_code == 403:
                # Check if rate limited
                if self._rate_limit_remaining == 0:
                    return ConnectionTestResult(
                        success=False,
                        message="Rate limit exceeded. Please try again later.",
                        latency_ms=latency_ms,
                        error_code="RATE_LIMITED",
                        error_details={
                            "reset_at": self._rate_limit_reset,
                        },
                    )
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - insufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            user_data = response.json()

            # Fetch scopes from a separate request to /user endpoint
            scopes = response.headers.get("X-OAuth-Scopes", "").split(", ")
            scopes = [s.strip() for s in scopes if s.strip()]

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {user_data.get('login', 'unknown')}",
                latency_ms=latency_ms,
                api_version="v3",
                permissions=scopes,
                account_info={
                    "login": user_data.get("login"),
                    "name": user_data.get("name"),
                    "email": user_data.get("email"),
                    "avatar_url": user_data.get("avatar_url"),
                    "type": user_data.get("type"),  # User or Organization
                    "public_repos": user_data.get("public_repos"),
                    "private_repos": user_data.get("total_private_repos"),
                    "rate_limit_remaining": self._rate_limit_remaining,
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
                message=f"Failed to connect to GitHub API: {str(e)}",
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
        """Sync data from GitHub.

        Syncs repositories, pull requests, and issues based on the requested resources.

        Args:
            credentials: GitHub credentials with access token
            resources: List of resources to sync (repositories, pull_requests, issues)
            since: Only fetch data updated since this datetime
            limit: Maximum items per resource type

        Returns:
            SyncResult containing synced data
        """
        started_at = datetime.utcnow()
        sync_id = f"github_{started_at.timestamp()}"

        # Default to all resources if not specified
        if resources is None:
            resources = ["repositories", "pull_requests", "issues"]

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

            # Sync pull requests (requires repositories to be synced first or org specified)
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

            # Sync issues
            if "issues" in resources:
                issues_result = await self._sync_issues(
                    credentials,
                    repos=data.get("repositories", []),
                    since=since,
                    limit=limit,
                )
                data["issues"] = issues_result["data"]
                total_points += issues_result["count"]
                errors.extend(issues_result.get("errors", []))
                warnings.extend(issues_result.get("warnings", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_points} data points from GitHub",
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
                "per_page": min(limit, 100),
                "sort": "updated",
                "direction": "desc",
            }

            if since:
                params["since"] = since.isoformat() + "Z"

            # Fetch user repos
            if credentials.org_slug:
                # Fetch org repos
                endpoint = f"/orgs/{credentials.org_slug}/repos"
            else:
                # Fetch user repos
                endpoint = "/user/repos"

            response = await self._make_request(
                method="GET",
                endpoint=endpoint,
                credentials=credentials,
                params=params,
            )

            if response.status_code != 200:
                errors.append(f"Failed to fetch repositories: {response.status_code}")
            else:
                repos_data = response.json()
                for repo in repos_data[:limit]:
                    repos.append({
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": repo.get("description"),
                        "private": repo["private"],
                        "html_url": repo["html_url"],
                        "clone_url": repo["clone_url"],
                        "default_branch": repo.get("default_branch", "main"),
                        "language": repo.get("language"),
                        "updated_at": repo.get("updated_at"),
                        "pushed_at": repo.get("pushed_at"),
                        "open_issues_count": repo.get("open_issues_count", 0),
                    })

        except Exception as e:
            errors.append(f"Error syncing repositories: {str(e)}")

        return {
            "data": repos,
            "count": len(repos),
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

        # Limit repos to check for PRs
        repos_to_check = repos[:10] if len(repos) > 10 else repos

        if len(repos) > 10:
            warnings.append(f"Limited PR sync to first 10 repos out of {len(repos)}")

        for repo in repos_to_check:
            try:
                params: dict[str, Any] = {
                    "state": "all",
                    "per_page": min(limit // len(repos_to_check) + 1, 30),
                    "sort": "updated",
                    "direction": "desc",
                }

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/repos/{repo['full_name']}/pulls",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch PRs for {repo['full_name']}")
                    continue

                prs_data = response.json()
                for pr in prs_data:
                    pr_updated = pr.get("updated_at")
                    if since and pr_updated:
                        pr_date = datetime.fromisoformat(pr_updated.replace("Z", "+00:00"))
                        if pr_date.replace(tzinfo=None) < since:
                            continue

                    pull_requests.append({
                        "id": pr["id"],
                        "number": pr["number"],
                        "title": pr["title"],
                        "state": pr["state"],
                        "html_url": pr["html_url"],
                        "repository": repo["full_name"],
                        "user": pr["user"]["login"] if pr.get("user") else None,
                        "created_at": pr.get("created_at"),
                        "updated_at": pr.get("updated_at"),
                        "merged_at": pr.get("merged_at"),
                        "head_ref": pr.get("head", {}).get("ref"),
                        "base_ref": pr.get("base", {}).get("ref"),
                        "draft": pr.get("draft", False),
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

    async def _sync_issues(
        self,
        credentials: IntegrationCredentials,
        repos: list[dict[str, Any]],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Fetch recent issues from repositories."""
        issues: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Limit repos to check for issues
        repos_to_check = repos[:10] if len(repos) > 10 else repos

        if len(repos) > 10:
            warnings.append(f"Limited issue sync to first 10 repos out of {len(repos)}")

        for repo in repos_to_check:
            try:
                params: dict[str, Any] = {
                    "state": "all",
                    "per_page": min(limit // len(repos_to_check) + 1, 30),
                    "sort": "updated",
                    "direction": "desc",
                }

                if since:
                    params["since"] = since.isoformat() + "Z"

                response = await self._make_request(
                    method="GET",
                    endpoint=f"/repos/{repo['full_name']}/issues",
                    credentials=credentials,
                    params=params,
                )

                if response.status_code != 200:
                    warnings.append(f"Failed to fetch issues for {repo['full_name']}")
                    continue

                issues_data = response.json()
                for issue in issues_data:
                    # Skip pull requests (GitHub returns PRs in issues endpoint)
                    if "pull_request" in issue:
                        continue

                    issues.append({
                        "id": issue["id"],
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "html_url": issue["html_url"],
                        "repository": repo["full_name"],
                        "user": issue["user"]["login"] if issue.get("user") else None,
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "closed_at": issue.get("closed_at"),
                        "labels": [l["name"] for l in issue.get("labels", [])],
                        "assignees": [a["login"] for a in issue.get("assignees", [])],
                    })

                    if len(issues) >= limit:
                        break

            except Exception as e:
                warnings.append(f"Error fetching issues for {repo['full_name']}: {str(e)}")

        return {
            "data": issues[:limit],
            "count": len(issues[:limit]),
            "errors": errors,
            "warnings": warnings,
        }

    async def get_repository(
        self,
        credentials: IntegrationCredentials,
        owner: str,
        repo: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific repository by owner and name."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/repos/{owner}/{repo}",
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
        owner: str,
        repo: str,
        pr_number: int,
    ) -> Optional[dict[str, Any]]:
        """Fetch a specific pull request."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/repos/{owner}/{repo}/pulls/{pr_number}",
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
        owner: str,
        repo: str,
        webhook_url: str,
        events: list[str],
        secret: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a webhook for a repository.

        Args:
            credentials: GitHub credentials
            owner: Repository owner
            repo: Repository name
            webhook_url: URL to receive webhook events
            events: List of events to subscribe to (e.g., ["push", "pull_request"])
            secret: Optional webhook secret for verification

        Returns:
            Webhook configuration if created successfully
        """
        try:
            config: dict[str, Any] = {
                "url": webhook_url,
                "content_type": "json",
            }
            if secret:
                config["secret"] = secret

            payload = {
                "name": "web",
                "active": True,
                "events": events,
                "config": config,
            }

            response = await self._make_request(
                method="POST",
                endpoint=f"/repos/{owner}/{repo}/hooks",
                credentials=credentials,
                json_data=payload,
            )

            if response.status_code == 201:
                return response.json()
            return None
        except Exception:
            return None
