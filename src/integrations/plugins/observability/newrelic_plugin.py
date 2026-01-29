"""New Relic integration plugin.

This module provides integration with New Relic for application performance
monitoring, infrastructure monitoring, and observability. New Relic provides
full-stack observability for modern software teams.

Features:
- Fetch applications and their metrics
- Sync alerts, policies, and incidents
- Query via NerdGraph (GraphQL API)
- Support for US and EU regions
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


class NewRelicPlugin(IntegrationPlugin):
    """New Relic integration for full-stack observability.

    New Relic provides unified observability across the entire software stack,
    including APM, infrastructure, logs, browser monitoring, and more.

    Authentication:
        Uses User API Key for authentication.
        Key can be created at: https://one.newrelic.com/api-keys

    Supported Regions:
        - US (api.newrelic.com) - default
        - EU (api.eu.newrelic.com)

    APIs:
        - REST API v2: Legacy API for basic operations
        - NerdGraph (GraphQL): Modern API for advanced queries

    Example:
        async with NewRelicPlugin() as plugin:
            credentials = IntegrationCredentials(
                api_key="NRAK-...",
                extra={"region": "us"},
            )
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    # Region to base URL mapping
    REGION_URLS = {
        "us": {
            "rest": "https://api.newrelic.com/v2",
            "nerdgraph": "https://api.newrelic.com/graphql",
            "infrastructure": "https://infra-api.newrelic.com/v2",
        },
        "eu": {
            "rest": "https://api.eu.newrelic.com/v2",
            "nerdgraph": "https://api.eu.newrelic.com/graphql",
            "infrastructure": "https://infra-api.eu.newrelic.com/v2",
        },
    }

    DEFAULT_REGION = "us"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get New Relic integration metadata."""
        return IntegrationMetadata(
            slug="newrelic",
            name="New Relic",
            description="Full-stack observability platform for APM, infrastructure, logs, and more",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.API_KEY,
            icon_url="https://newrelic.com/favicon.ico",
            docs_url="https://docs.newrelic.com/docs/apis/",
            website_url="https://newrelic.com",
            features=[
                "apm",
                "infrastructure_monitoring",
                "log_management",
                "browser_monitoring",
                "synthetics",
                "alerts",
                "dashboards",
                "nrql_queries",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="User API Key",
                    type="password",
                    required=True,
                    placeholder="NRAK-...",
                    help_text="New Relic User API Key. Create at one.newrelic.com/api-keys",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.region",
                    label="Region",
                    type="select",
                    required=False,
                    placeholder="us",
                    help_text="Your New Relic data region",
                    options=[
                        {"value": "us", "label": "US (api.newrelic.com)"},
                        {"value": "eu", "label": "EU (api.eu.newrelic.com)"},
                    ],
                ),
                FieldDefinition(
                    name="extra.account_id",
                    label="Account ID",
                    type="text",
                    required=False,
                    placeholder="1234567",
                    help_text="Your New Relic account ID (optional, will auto-detect if not provided)",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["applications", "alerts", "incidents", "dashboards", "entities"],
        )

    def _get_region(self, credentials: IntegrationCredentials) -> str:
        """Get the region from credentials."""
        return credentials.extra.get("region", self.DEFAULT_REGION)

    def _get_urls(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get API URLs based on region."""
        region = self._get_region(credentials)
        return self.REGION_URLS.get(region, self.REGION_URLS[self.DEFAULT_REGION])

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for REST API requests."""
        return {
            "Api-Key": credentials.api_key or "",
            "Content-Type": "application/json",
        }

    def _get_graphql_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for NerdGraph (GraphQL) requests."""
        return {
            "API-Key": credentials.api_key or "",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the New Relic connection.

        Validates the API key by fetching user information.

        Args:
            credentials: Must contain api_key.

        Returns:
            ConnectionTestResult with user and account info if successful.
        """
        is_valid, missing = self.validate_credentials(credentials)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=f"Missing required fields: {', '.join(missing)}",
                error_code="MISSING_CREDENTIALS",
            )

        urls = self._get_urls(credentials)
        headers = self._get_headers(credentials)

        # Use REST API to validate - GET /users is a simple endpoint
        url = f"{urls['rest']}/users.json"

        start_time = time.time()
        try:
            response = await self.http_client.get(url, headers=headers)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="INVALID_API_KEY",
                )

            if response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have required permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            response.raise_for_status()
            result = response.json()

            users = result.get("users", [])
            current_user = users[0] if users else {}

            # Get account information using NerdGraph
            account_info = await self._get_account_info(credentials)

            return ConnectionTestResult(
                success=True,
                message=f"Connected as {current_user.get('email', 'Unknown')}",
                latency_ms=latency_ms,
                api_version="v2",
                permissions=[],
                account_info={
                    "user_id": current_user.get("id"),
                    "email": current_user.get("email"),
                    "first_name": current_user.get("first_name"),
                    "last_name": current_user.get("last_name"),
                    "role": current_user.get("role"),
                    "region": self._get_region(credentials),
                    **account_info,
                },
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.HTTPStatusError as e:
            return ConnectionTestResult(
                success=False,
                message=f"HTTP error: {e.response.status_code}",
                error_code=f"HTTP_{e.response.status_code}",
                error_details={"response": e.response.text},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
            )

    async def _get_account_info(
        self,
        credentials: IntegrationCredentials,
    ) -> dict[str, Any]:
        """Get account information via NerdGraph."""
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        query = """
        {
            actor {
                accounts {
                    id
                    name
                }
                user {
                    id
                    email
                    name
                }
            }
        }
        """

        try:
            response = await self.http_client.post(
                urls["nerdgraph"],
                headers=headers,
                json={"query": query},
            )
            response.raise_for_status()
            result = response.json()

            actor = result.get("data", {}).get("actor", {})
            accounts = actor.get("accounts", [])
            user = actor.get("user", {})

            return {
                "accounts": [
                    {"id": a.get("id"), "name": a.get("name")}
                    for a in accounts
                ],
                "account_count": len(accounts),
                "nerdgraph_user": {
                    "id": user.get("id"),
                    "email": user.get("email"),
                    "name": user.get("name"),
                },
            }
        except Exception:
            return {}

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from New Relic.

        Fetches applications, alerts, incidents, and dashboards.

        Args:
            credentials: Must contain api_key.
            resources: List of resources to sync: "applications", "alerts", "incidents", "dashboards", "entities".
            since: Only fetch data since this datetime (for incidents).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced data organized by resource type.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or self.metadata.sync_resources

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        # Sync applications
        if "applications" in sync_resources:
            try:
                apps_data = await self._sync_applications(credentials, limit)
                data["applications"] = apps_data
                total_synced += len(apps_data)
            except Exception as e:
                errors.append(f"Failed to sync applications: {str(e)}")

        # Sync alert policies
        if "alerts" in sync_resources:
            try:
                alerts_data = await self._sync_alerts(credentials, limit)
                data["alerts"] = alerts_data
                total_synced += len(alerts_data)
            except Exception as e:
                errors.append(f"Failed to sync alerts: {str(e)}")

        # Sync incidents
        if "incidents" in sync_resources:
            try:
                incidents_data = await self._sync_incidents(credentials, since, limit)
                data["incidents"] = incidents_data
                total_synced += len(incidents_data)
            except Exception as e:
                errors.append(f"Failed to sync incidents: {str(e)}")

        # Sync dashboards via NerdGraph
        if "dashboards" in sync_resources:
            try:
                dashboards_data = await self._sync_dashboards(credentials, limit)
                data["dashboards"] = dashboards_data
                total_synced += len(dashboards_data)
            except Exception as e:
                errors.append(f"Failed to sync dashboards: {str(e)}")

        # Sync entities via NerdGraph
        if "entities" in sync_resources:
            try:
                entities_data = await self._sync_entities(credentials, limit)
                data["entities"] = entities_data
                total_synced += len(entities_data)
            except Exception as e:
                errors.append(f"Failed to sync entities: {str(e)}")

        completed_at = datetime.utcnow()

        return SyncResult(
            success=len(errors) == 0,
            message=f"Synced {total_synced} items from New Relic"
            if len(errors) == 0
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _sync_applications(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync applications from New Relic REST API."""
        urls = self._get_urls(credentials)
        headers = self._get_headers(credentials)
        url = f"{urls['rest']}/applications.json"

        params: dict[str, Any] = {"page": 1}

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        applications = result.get("applications", [])

        return [
            {
                "id": app.get("id"),
                "name": app.get("name"),
                "language": app.get("language"),
                "health_status": app.get("health_status"),
                "reporting": app.get("reporting"),
                "last_reported_at": app.get("last_reported_at"),
                "application_summary": {
                    "response_time": app.get("application_summary", {}).get("response_time"),
                    "throughput": app.get("application_summary", {}).get("throughput"),
                    "error_rate": app.get("application_summary", {}).get("error_rate"),
                    "apdex_score": app.get("application_summary", {}).get("apdex_score"),
                    "apdex_target": app.get("application_summary", {}).get("apdex_target"),
                    "host_count": app.get("application_summary", {}).get("host_count"),
                    "instance_count": app.get("application_summary", {}).get("instance_count"),
                },
                "end_user_summary": app.get("end_user_summary", {}),
                "settings": {
                    "app_apdex_threshold": app.get("settings", {}).get("app_apdex_threshold"),
                    "end_user_apdex_threshold": app.get("settings", {}).get("end_user_apdex_threshold"),
                    "enable_real_user_monitoring": app.get("settings", {}).get("enable_real_user_monitoring"),
                },
                "links": app.get("links", {}),
            }
            for app in applications[:limit]
        ]

    async def _sync_alerts(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync alert policies from New Relic REST API."""
        urls = self._get_urls(credentials)
        headers = self._get_headers(credentials)
        url = f"{urls['rest']}/alerts_policies.json"

        response = await self.http_client.get(url, headers=headers)
        response.raise_for_status()

        result = response.json()
        policies = result.get("policies", [])

        policies_data: list[dict[str, Any]] = []
        for policy in policies[:limit]:
            policy_data = {
                "id": policy.get("id"),
                "name": policy.get("name"),
                "incident_preference": policy.get("incident_preference"),
                "created_at": policy.get("created_at"),
                "updated_at": policy.get("updated_at"),
                "conditions": [],
            }

            # Get conditions for this policy
            try:
                conditions_url = f"{urls['rest']}/alerts_conditions.json"
                conditions_params = {"policy_id": policy.get("id")}
                conditions_response = await self.http_client.get(
                    conditions_url, headers=headers, params=conditions_params
                )
                if conditions_response.status_code == 200:
                    conditions_result = conditions_response.json()
                    policy_data["conditions"] = [
                        {
                            "id": c.get("id"),
                            "name": c.get("name"),
                            "type": c.get("type"),
                            "enabled": c.get("enabled"),
                            "metric": c.get("metric"),
                        }
                        for c in conditions_result.get("conditions", [])
                    ]
            except Exception:
                pass

            policies_data.append(policy_data)

        return policies_data

    async def _sync_incidents(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync incidents via NerdGraph."""
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        # Get account ID
        account_id = credentials.extra.get("account_id")
        if not account_id:
            # Try to get first account
            account_info = await self._get_account_info(credentials)
            accounts = account_info.get("accounts", [])
            if accounts:
                account_id = accounts[0].get("id")

        if not account_id:
            return []

        # Build time filter
        time_filter = ""
        if since:
            since_ms = int(since.timestamp() * 1000)
            time_filter = f", filter: {{startTime: {since_ms}}}"

        query = f"""
        {{
            actor {{
                account(id: {account_id}) {{
                    aiIssues {{
                        issues(first: {min(limit, 100)}{time_filter}) {{
                            issues {{
                                issueId
                                title
                                priority
                                state
                                activatedAt
                                acknowledgedAt
                                closedAt
                                origins
                                sources
                                totalIncidents
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

        try:
            response = await self.http_client.post(
                urls["nerdgraph"],
                headers=headers,
                json={"query": query},
            )
            response.raise_for_status()
            result = response.json()

            issues = (
                result.get("data", {})
                .get("actor", {})
                .get("account", {})
                .get("aiIssues", {})
                .get("issues", {})
                .get("issues", [])
            )

            return [
                {
                    "id": issue.get("issueId"),
                    "title": issue.get("title"),
                    "priority": issue.get("priority"),
                    "state": issue.get("state"),
                    "activated_at": issue.get("activatedAt"),
                    "acknowledged_at": issue.get("acknowledgedAt"),
                    "closed_at": issue.get("closedAt"),
                    "origins": issue.get("origins", []),
                    "sources": issue.get("sources", []),
                    "total_incidents": issue.get("totalIncidents"),
                }
                for issue in issues[:limit]
            ]
        except Exception:
            # Fall back to REST API for alerts_incidents
            return await self._sync_incidents_rest(credentials, since, limit)

    async def _sync_incidents_rest(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync incidents from REST API (fallback)."""
        urls = self._get_urls(credentials)
        headers = self._get_headers(credentials)
        url = f"{urls['rest']}/alerts_incidents.json"

        params: dict[str, Any] = {"only_open": False}

        response = await self.http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

        result = response.json()
        incidents = result.get("incidents", [])

        return [
            {
                "id": i.get("id"),
                "opened_at": i.get("opened_at"),
                "closed_at": i.get("closed_at"),
                "incident_preference": i.get("incident_preference"),
                "links": {
                    "policy_id": i.get("links", {}).get("policy_id"),
                    "violations": i.get("links", {}).get("violations", []),
                },
            }
            for i in incidents[:limit]
        ]

    async def _sync_dashboards(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync dashboards via NerdGraph."""
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        query = f"""
        {{
            actor {{
                entitySearch(queryBuilder: {{type: DASHBOARD}}) {{
                    results(cursor: null) {{
                        entities {{
                            guid
                            name
                            accountId
                            ... on DashboardEntityOutline {{
                                dashboardParentGuid
                                permissions
                                owner {{
                                    email
                                    userId
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

        try:
            response = await self.http_client.post(
                urls["nerdgraph"],
                headers=headers,
                json={"query": query},
            )
            response.raise_for_status()
            result = response.json()

            entities = (
                result.get("data", {})
                .get("actor", {})
                .get("entitySearch", {})
                .get("results", {})
                .get("entities", [])
            )

            return [
                {
                    "guid": e.get("guid"),
                    "name": e.get("name"),
                    "account_id": e.get("accountId"),
                    "parent_guid": e.get("dashboardParentGuid"),
                    "permissions": e.get("permissions"),
                    "owner": e.get("owner"),
                }
                for e in entities[:limit]
            ]
        except Exception:
            return []

    async def _sync_entities(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Sync entities via NerdGraph."""
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        query = f"""
        {{
            actor {{
                entitySearch(query: "reporting = 'true'") {{
                    results(cursor: null) {{
                        entities {{
                            guid
                            name
                            entityType
                            domain
                            type
                            accountId
                            alertSeverity
                            reporting
                            tags {{
                                key
                                values
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

        try:
            response = await self.http_client.post(
                urls["nerdgraph"],
                headers=headers,
                json={"query": query},
            )
            response.raise_for_status()
            result = response.json()

            entities = (
                result.get("data", {})
                .get("actor", {})
                .get("entitySearch", {})
                .get("results", {})
                .get("entities", [])
            )

            return [
                {
                    "guid": e.get("guid"),
                    "name": e.get("name"),
                    "entity_type": e.get("entityType"),
                    "domain": e.get("domain"),
                    "type": e.get("type"),
                    "account_id": e.get("accountId"),
                    "alert_severity": e.get("alertSeverity"),
                    "reporting": e.get("reporting"),
                    "tags": {
                        t.get("key"): t.get("values")
                        for t in e.get("tags", [])
                    },
                }
                for e in entities[:limit]
            ]
        except Exception:
            return []

    async def run_nrql_query(
        self,
        credentials: IntegrationCredentials,
        account_id: int,
        nrql: str,
    ) -> dict[str, Any]:
        """Run a NRQL query via NerdGraph.

        Args:
            credentials: New Relic credentials.
            account_id: The New Relic account ID to query.
            nrql: The NRQL query string.

        Returns:
            Query results including metadata.
        """
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        # Escape the NRQL query for GraphQL
        escaped_nrql = nrql.replace('"', '\\"')

        query = f"""
        {{
            actor {{
                account(id: {account_id}) {{
                    nrql(query: "{escaped_nrql}") {{
                        results
                        metadata {{
                            eventTypes
                            facets
                            messages
                            rawSince
                            rawUntil
                            rawCompareWith
                        }}
                    }}
                }}
            }}
        }}
        """

        response = await self.http_client.post(
            urls["nerdgraph"],
            headers=headers,
            json={"query": query},
        )
        response.raise_for_status()
        result = response.json()

        nrql_result = (
            result.get("data", {})
            .get("actor", {})
            .get("account", {})
            .get("nrql", {})
        )

        return {
            "results": nrql_result.get("results", []),
            "metadata": nrql_result.get("metadata", {}),
        }

    async def get_entity_details(
        self,
        credentials: IntegrationCredentials,
        entity_guid: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific entity.

        Args:
            credentials: New Relic credentials.
            entity_guid: The entity GUID.

        Returns:
            Detailed entity data.
        """
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        query = f"""
        {{
            actor {{
                entity(guid: "{entity_guid}") {{
                    guid
                    name
                    entityType
                    domain
                    type
                    accountId
                    alertSeverity
                    reporting
                    tags {{
                        key
                        values
                    }}
                    ... on ApmApplicationEntity {{
                        applicationId
                        language
                        settings {{
                            apdexTarget
                        }}
                        runningAgentVersions {{
                            maxVersion
                            minVersion
                        }}
                        deployments {{
                            changelog
                            description
                            timestamp
                            user
                        }}
                    }}
                    ... on BrowserApplicationEntity {{
                        applicationId
                        servingApmApplicationId
                        settings {{
                            apdexTarget
                        }}
                    }}
                    ... on InfrastructureHostEntity {{
                        hostSummary {{
                            cpuUtilizationPercent
                            diskUsedPercent
                            memoryUsedPercent
                            networkReceiveRate
                            networkTransmitRate
                            servicesCount
                        }}
                    }}
                }}
            }}
        }}
        """

        response = await self.http_client.post(
            urls["nerdgraph"],
            headers=headers,
            json={"query": query},
        )
        response.raise_for_status()
        result = response.json()

        return result.get("data", {}).get("actor", {}).get("entity", {})

    async def get_golden_metrics(
        self,
        credentials: IntegrationCredentials,
        entity_guid: str,
    ) -> dict[str, Any]:
        """Get golden metrics for an entity.

        Args:
            credentials: New Relic credentials.
            entity_guid: The entity GUID.

        Returns:
            Golden metrics data (latency, throughput, errors, saturation).
        """
        urls = self._get_urls(credentials)
        headers = self._get_graphql_headers(credentials)

        query = f"""
        {{
            actor {{
                entity(guid: "{entity_guid}") {{
                    goldenMetrics {{
                        metrics {{
                            name
                            title
                            unit
                            query
                        }}
                    }}
                }}
            }}
        }}
        """

        response = await self.http_client.post(
            urls["nerdgraph"],
            headers=headers,
            json={"query": query},
        )
        response.raise_for_status()
        result = response.json()

        return (
            result.get("data", {})
            .get("actor", {})
            .get("entity", {})
            .get("goldenMetrics", {})
        )

    def parse_webhook_payload(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Parse and normalize a New Relic webhook payload.

        New Relic sends different payloads for different notification channels.

        Args:
            payload: The webhook payload from New Relic.
            headers: Request headers.

        Returns:
            Parsed event data with normalized structure.
        """
        # Determine the webhook type
        current_state = payload.get("current_state", payload.get("state", ""))
        severity = payload.get("severity", payload.get("priority", ""))

        # Normalize event type
        normalized_type = "newrelic.alert"
        state_lower = current_state.lower()
        if state_lower == "open" or state_lower == "triggered":
            normalized_type = "newrelic.alert.triggered"
        elif state_lower == "acknowledged":
            normalized_type = "newrelic.alert.acknowledged"
        elif state_lower == "closed":
            normalized_type = "newrelic.alert.closed"

        return {
            "event_type": normalized_type,
            "timestamp": payload.get("timestamp"),
            "incident_id": payload.get("incident_id"),
            "issue_id": payload.get("issue_id"),
            "current_state": current_state,
            "severity": severity,
            "condition": {
                "id": payload.get("condition_id"),
                "name": payload.get("condition_name"),
                "type": payload.get("condition_type"),
            },
            "policy": {
                "id": payload.get("policy_id"),
                "name": payload.get("policy_name"),
            },
            "account": {
                "id": payload.get("account_id"),
                "name": payload.get("account_name"),
            },
            "targets": payload.get("targets", []),
            "incident_url": payload.get("incident_url"),
            "violation_chart_url": payload.get("violation_chart_url"),
            "details": payload.get("details", ""),
            "raw_payload": payload,
        }
