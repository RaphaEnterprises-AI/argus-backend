"""Splunk integration plugin.

This plugin provides integration with Splunk for log management,
security information and event management (SIEM), and observability.

Splunk is an enterprise platform for searching, monitoring, and analyzing
machine-generated data via a web-style interface.
"""

import logging
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

logger = logging.getLogger(__name__)


class SplunkPlugin(IntegrationPlugin):
    """Splunk integration plugin.

    Supports:
    - Saved searches
    - Alerts
    - Notable events (Enterprise Security)
    - Search job execution

    Authentication:
    - Bearer Token (Splunk Authentication Token)
    - Requires instance_url for self-hosted or cloud instances

    Example:
        plugin = SplunkPlugin()
        credentials = IntegrationCredentials(
            access_token="your-splunk-auth-token",
            instance_url="https://your-instance.splunkcloud.com:8089",
        )
        result = await plugin.test_connection(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Splunk integration metadata."""
        return IntegrationMetadata(
            slug="splunk",
            name="Splunk",
            description="Enterprise platform for searching, monitoring, and analyzing machine-generated data",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://www.splunk.com/content/dam/splunk2/images/2020-splunk-planet.svg",
            docs_url="https://docs.splunk.com/Documentation/Splunk/latest/RESTREF/RESTprolog",
            website_url="https://www.splunk.com",
            features=[
                "saved_searches",
                "alerts",
                "notable_events",
                "search",
                "dashboards",
                "reports",
            ],
            required_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Authentication Token",
                    type="password",
                    required=True,
                    placeholder="your-splunk-auth-token",
                    help_text="Generate a token from Settings > Tokens in Splunk Web",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Splunk Instance URL",
                    type="url",
                    required=True,
                    placeholder="https://your-instance.splunkcloud.com:8089",
                    help_text="Your Splunk instance URL with management port (usually 8089)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="username",
                    label="Username (for Basic Auth)",
                    type="text",
                    required=False,
                    placeholder="admin",
                    help_text="Username if using Basic Auth instead of token",
                ),
                FieldDefinition(
                    name="password",
                    label="Password (for Basic Auth)",
                    type="password",
                    required=False,
                    help_text="Password if using Basic Auth instead of token",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["saved_searches", "alerts", "notable_events"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base API URL from credentials."""
        instance_url = credentials.instance_url or ""
        # Remove trailing slash if present
        return instance_url.rstrip("/")

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get the request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if credentials.access_token:
            headers["Authorization"] = f"Bearer {credentials.access_token}"
        elif credentials.username and credentials.password:
            import base64
            auth_string = f"{credentials.username}:{credentials.password}"
            encoded = base64.b64encode(auth_string.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Splunk API.

        Uses GET /services/server/info to verify authentication and retrieve
        server information.

        Args:
            credentials: Splunk credentials with access_token and instance_url.

        Returns:
            ConnectionTestResult with server details on success.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        start_time = time.time()

        try:
            # Test connection using /services/server/info endpoint
            response = await self.http_client.get(
                f"{base_url}/services/server/info",
                headers=headers,
                params={"output_mode": "json"},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                entry = data.get("entry", [{}])[0]
                content = entry.get("content", {})

                # Get current user info
                user_response = await self.http_client.get(
                    f"{base_url}/services/authentication/current-context",
                    headers=headers,
                    params={"output_mode": "json"},
                )
                user_data = {}
                if user_response.status_code == 200:
                    user_entry = user_response.json().get("entry", [{}])[0]
                    user_data = user_entry.get("content", {})

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to Splunk server: {content.get('serverName', 'Unknown')}",
                    latency_ms=latency_ms,
                    api_version=content.get("version", "unknown"),
                    permissions=self._extract_permissions(user_data),
                    account_info={
                        "server_name": content.get("serverName"),
                        "version": content.get("version"),
                        "build": content.get("build"),
                        "os_name": content.get("os_name"),
                        "os_version": content.get("os_version"),
                        "cpu_arch": content.get("cpu_arch"),
                        "license_state": content.get("licenseState"),
                        "license_key": content.get("licenseKey", "")[:8] + "***" if content.get("licenseKey") else None,
                        "is_free_license": content.get("isFree", False),
                        "is_trial": content.get("isTrial", False),
                        "current_user": user_data.get("username"),
                        "user_roles": user_data.get("roles", []),
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Please check your token or credentials.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. The token may lack required capabilities.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )

            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response from Splunk: {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="UNEXPECTED_RESPONSE",
                    error_details={
                        "status_code": response.status_code,
                        "response_body": response.text[:500],
                    },
                )

        except httpx.ConnectError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Failed to connect to Splunk at {base_url}. Please verify the URL and port.",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to Splunk timed out.",
                error_code="TIMEOUT",
            )

        except Exception as e:
            logger.exception("Unexpected error testing Splunk connection")
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"error": str(e)},
            )

    def _extract_permissions(self, user_data: dict) -> list[str]:
        """Extract permissions from user data."""
        permissions = []

        roles = user_data.get("roles", [])
        for role in roles:
            permissions.append(f"role:{role}")

        capabilities = user_data.get("capabilities", [])
        for cap in capabilities[:20]:  # Limit to first 20 capabilities
            permissions.append(f"cap:{cap}")

        return permissions

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Splunk.

        Syncs saved searches, alerts, and notable events.

        Args:
            credentials: Splunk credentials.
            resources: Optional list of resources to sync.
                      Options: saved_searches, alerts, notable_events
            since: Optional datetime for incremental sync.
            limit: Maximum items to sync per resource.

        Returns:
            SyncResult with synced data.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        started_at = datetime.utcnow()
        sync_resources = resources or ["saved_searches", "alerts", "notable_events"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Sync saved searches
            if "saved_searches" in sync_resources:
                searches_result = await self._sync_saved_searches(
                    base_url, headers, limit
                )
                data["saved_searches"] = searches_result["data"]
                total_synced += searches_result["count"]
                errors.extend(searches_result.get("errors", []))

            # Sync alerts
            if "alerts" in sync_resources:
                alerts_result = await self._sync_alerts(base_url, headers, limit)
                data["alerts"] = alerts_result["data"]
                total_synced += alerts_result["count"]
                errors.extend(alerts_result.get("errors", []))

            # Sync notable events (Enterprise Security)
            if "notable_events" in sync_resources:
                notable_result = await self._sync_notable_events(
                    base_url, headers, since, limit
                )
                data["notable_events"] = notable_result["data"]
                total_synced += notable_result["count"]
                errors.extend(notable_result.get("errors", []))
                warnings.extend(notable_result.get("warnings", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Splunk"
                if len(errors) == 0
                else f"Synced {total_synced} items with {len(errors)} errors",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.exception("Error syncing Splunk data")
            return SyncResult(
                success=False,
                message=f"Failed to sync Splunk data: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_saved_searches(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync saved searches from Splunk."""
        saved_searches = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{base_url}/services/saved/searches",
                headers=headers,
                params={
                    "output_mode": "json",
                    "count": limit,
                    "search": "is_scheduled=1 OR is_visible=1",
                },
            )

            if response.status_code == 200:
                data = response.json()
                entries = data.get("entry", [])

                for entry in entries[:limit]:
                    content = entry.get("content", {})
                    saved_searches.append({
                        "name": entry.get("name"),
                        "id": entry.get("id"),
                        "author": entry.get("author"),
                        "search": content.get("search"),
                        "description": content.get("description"),
                        "is_scheduled": content.get("is_scheduled", False),
                        "is_visible": content.get("is_visible", True),
                        "cron_schedule": content.get("cron_schedule"),
                        "next_scheduled_time": content.get("next_scheduled_time"),
                        "dispatch_earliest_time": content.get("dispatch.earliest_time"),
                        "dispatch_latest_time": content.get("dispatch.latest_time"),
                        "updated": entry.get("updated"),
                        "app": entry.get("acl", {}).get("app"),
                    })
            else:
                errors.append(f"Failed to fetch saved searches: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing saved searches: {str(e)}")

        return {"data": saved_searches, "count": len(saved_searches), "errors": errors}

    async def _sync_alerts(
        self,
        base_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync alerts from Splunk."""
        alerts = []
        errors = []

        try:
            # Get fired alerts
            response = await self.http_client.get(
                f"{base_url}/services/alerts/fired_alerts",
                headers=headers,
                params={
                    "output_mode": "json",
                    "count": limit,
                },
            )

            if response.status_code == 200:
                data = response.json()
                entries = data.get("entry", [])

                for entry in entries[:limit]:
                    content = entry.get("content", {})
                    alerts.append({
                        "name": entry.get("name"),
                        "id": entry.get("id"),
                        "savedsearch_name": content.get("savedsearch_name"),
                        "severity": content.get("severity"),
                        "trigger_time": content.get("trigger_time"),
                        "trigger_time_rendered": content.get("trigger_time_rendered"),
                        "triggered_alerts_count": content.get("triggered_alerts"),
                        "app": entry.get("acl", {}).get("app"),
                        "author": entry.get("author"),
                    })
            else:
                errors.append(f"Failed to fetch alerts: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing alerts: {str(e)}")

        return {"data": alerts, "count": len(alerts), "errors": errors}

    async def _sync_notable_events(
        self,
        base_url: str,
        headers: dict[str, str],
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync notable events from Splunk Enterprise Security."""
        notable_events = []
        errors = []
        warnings = []

        try:
            # Build the search query for notable events
            search_query = "| `notable` | head {limit}".format(limit=limit)

            if since:
                earliest = since.strftime("%Y-%m-%dT%H:%M:%S")
                search_params = {
                    "search": search_query,
                    "earliest_time": earliest,
                    "latest_time": "now",
                }
            else:
                search_params = {
                    "search": search_query,
                    "earliest_time": "-24h",
                    "latest_time": "now",
                }

            # Create a search job
            job_response = await self.http_client.post(
                f"{base_url}/services/search/jobs",
                headers=headers,
                data={
                    **search_params,
                    "output_mode": "json",
                    "exec_mode": "blocking",  # Wait for results
                },
            )

            if job_response.status_code in (200, 201):
                job_data = job_response.json()
                sid = job_data.get("sid")

                if sid:
                    # Get the results
                    results_response = await self.http_client.get(
                        f"{base_url}/services/search/jobs/{sid}/results",
                        headers=headers,
                        params={"output_mode": "json", "count": limit},
                    )

                    if results_response.status_code == 200:
                        results_data = results_response.json()
                        results = results_data.get("results", [])

                        for event in results:
                            notable_events.append({
                                "event_id": event.get("event_id"),
                                "rule_name": event.get("rule_name"),
                                "rule_title": event.get("rule_title"),
                                "urgency": event.get("urgency"),
                                "status": event.get("status"),
                                "owner": event.get("owner"),
                                "security_domain": event.get("security_domain"),
                                "src": event.get("src"),
                                "dest": event.get("dest"),
                                "user": event.get("user"),
                                "time": event.get("_time"),
                                "raw": event.get("_raw"),
                            })
            elif job_response.status_code == 404:
                warnings.append(
                    "Notable events endpoint not found. "
                    "This feature requires Splunk Enterprise Security."
                )
            else:
                errors.append(
                    f"Failed to search notable events: {job_response.status_code}"
                )

        except Exception as e:
            errors.append(f"Error syncing notable events: {str(e)}")

        return {
            "data": notable_events,
            "count": len(notable_events),
            "errors": errors,
            "warnings": warnings,
        }

    async def run_search(
        self,
        credentials: IntegrationCredentials,
        search_query: str,
        earliest_time: str = "-24h",
        latest_time: str = "now",
        max_results: int = 100,
    ) -> Optional[list[dict[str, Any]]]:
        """Run a search query in Splunk.

        Args:
            credentials: Splunk credentials.
            search_query: SPL search query (without leading 'search' command).
            earliest_time: Earliest time for search (default: -24h).
            latest_time: Latest time for search (default: now).
            max_results: Maximum number of results to return.

        Returns:
            List of search results or None on failure.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        # Ensure query starts with 'search' command if not a generating command
        if not search_query.strip().startswith("|"):
            if not search_query.strip().lower().startswith("search"):
                search_query = f"search {search_query}"

        try:
            # Create search job
            job_response = await self.http_client.post(
                f"{base_url}/services/search/jobs",
                headers=headers,
                data={
                    "search": search_query,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "output_mode": "json",
                    "exec_mode": "blocking",
                },
                timeout=120.0,  # Longer timeout for searches
            )

            if job_response.status_code not in (200, 201):
                logger.error(f"Failed to create search job: {job_response.status_code}")
                return None

            job_data = job_response.json()
            sid = job_data.get("sid")

            if not sid:
                logger.error("No search ID returned")
                return None

            # Get results
            results_response = await self.http_client.get(
                f"{base_url}/services/search/jobs/{sid}/results",
                headers=headers,
                params={"output_mode": "json", "count": max_results},
            )

            if results_response.status_code == 200:
                results_data = results_response.json()
                return results_data.get("results", [])

            logger.error(f"Failed to get search results: {results_response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error running search: {e}")
            return None

    async def get_saved_search(
        self,
        credentials: IntegrationCredentials,
        name: str,
        app: str = "search",
    ) -> Optional[dict[str, Any]]:
        """Get a specific saved search by name.

        Args:
            credentials: Splunk credentials.
            name: Saved search name.
            app: App context (default: search).

        Returns:
            Saved search data or None if not found.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/servicesNS/-/{app}/saved/searches/{name}",
                headers=headers,
                params={"output_mode": "json"},
            )

            if response.status_code == 200:
                data = response.json()
                entry = data.get("entry", [{}])[0]
                return {
                    "name": entry.get("name"),
                    "content": entry.get("content", {}),
                    "acl": entry.get("acl", {}),
                    "author": entry.get("author"),
                    "updated": entry.get("updated"),
                }

            return None

        except Exception as e:
            logger.error(f"Error fetching saved search {name}: {e}")
            return None
