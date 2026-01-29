"""Dynatrace integration plugin.

This plugin provides integration with Dynatrace for application performance
monitoring (APM), infrastructure monitoring, and AI-powered observability.

Dynatrace is an AI-powered, full-stack, automated monitoring solution that
provides automatic discovery and mapping of applications, infrastructure,
and user experience.
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


class DynatracePlugin(IntegrationPlugin):
    """Dynatrace integration plugin.

    Supports:
    - Problems and incidents
    - Metrics
    - Events
    - Entities (hosts, services, applications)
    - Synthetic monitors

    Authentication:
    - API Token
    - Requires environment_url (stored in instance_url)

    Example:
        plugin = DynatracePlugin()
        credentials = IntegrationCredentials(
            api_key="dt0c01.XXXXXXXX.YYYYYYYY",
            instance_url="https://abc12345.live.dynatrace.com",
        )
        result = await plugin.test_connection(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Dynatrace integration metadata."""
        return IntegrationMetadata(
            slug="dynatrace",
            name="Dynatrace",
            description="AI-powered, full-stack, automated application performance monitoring and observability",
            category=IntegrationCategory.OBSERVABILITY,
            auth_type=AuthType.API_KEY,
            icon_url="https://assets.dynatrace.com/content/dam/dynatrace/misc/dynatrace_web.png",
            docs_url="https://www.dynatrace.com/support/help/dynatrace-api",
            website_url="https://www.dynatrace.com",
            features=[
                "problems",
                "metrics",
                "events",
                "entities",
                "synthetic_monitors",
                "logs",
                "traces",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="dt0c01.XXXXXXXXXXXXXXXX.YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
                    help_text="Generate an API token from Settings > Integration > Dynatrace API",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Environment URL",
                    type="url",
                    required=True,
                    placeholder="https://abc12345.live.dynatrace.com",
                    help_text="Your Dynatrace environment URL (SaaS or Managed)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra",
                    label="Additional Options",
                    type="text",
                    required=False,
                    help_text="JSON object with additional configuration",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["problems", "metrics", "events", "entities"],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base API URL from credentials."""
        instance_url = credentials.instance_url or ""
        # Remove trailing slash if present
        base = instance_url.rstrip("/")
        # Dynatrace API v2 uses /api/v2 prefix
        return f"{base}/api/v2"

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get the request headers with authentication."""
        return {
            "Authorization": f"Api-Token {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Dynatrace API.

        Uses GET /api/v2/extensions to verify authentication and API access.

        Args:
            credentials: Dynatrace credentials with api_key and instance_url (environment_url).

        Returns:
            ConnectionTestResult with environment details on success.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        start_time = time.time()

        try:
            # Test connection using /api/v2/extensions endpoint
            # This is a lightweight endpoint that validates the token
            response = await self.http_client.get(
                f"{base_url}/extensions",
                headers=headers,
                params={"pageSize": 1},
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Get additional environment info
                env_info = await self._get_environment_info(credentials)

                # Get token permissions
                token_info = await self._get_token_info(credentials)

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to Dynatrace environment: {env_info.get('name', 'Unknown')}",
                    latency_ms=latency_ms,
                    api_version="v2",
                    permissions=self._extract_permissions(token_info),
                    account_info={
                        "environment_id": env_info.get("id"),
                        "environment_name": env_info.get("name"),
                        "environment_url": credentials.instance_url,
                        "cluster_version": env_info.get("version"),
                        "token_name": token_info.get("name"),
                        "token_owner": token_info.get("owner"),
                        "token_created": token_info.get("created"),
                        "token_expires": token_info.get("expirationDate"),
                        "token_scopes": token_info.get("scopes", []),
                    },
                )

            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed. Please check your API token.",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                    error_details={"status_code": 401},
                )

            elif response.status_code == 403:
                error_data = response.json() if response.content else {}
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden. The API token may lack required scopes.",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={
                        "status_code": 403,
                        "missing_scopes": error_data.get("error", {}).get("missingScopes", []),
                    },
                )

            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Unexpected response from Dynatrace: {response.status_code}",
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
                message=f"Failed to connect to Dynatrace at {credentials.instance_url}. Please verify the URL.",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection to Dynatrace timed out.",
                error_code="TIMEOUT",
            )

        except Exception as e:
            logger.exception("Unexpected error testing Dynatrace connection")
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"error": str(e)},
            )

    async def _get_environment_info(
        self,
        credentials: IntegrationCredentials,
    ) -> dict[str, Any]:
        """Get environment information."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            # Try to get cluster version
            response = await self.http_client.get(
                f"{base_url}/settings/schemas",
                headers=headers,
                params={"pageSize": 1},
            )

            if response.status_code == 200:
                # Extract environment ID from URL
                env_url = credentials.instance_url or ""
                env_id = env_url.split("//")[-1].split(".")[0] if "//" in env_url else ""

                return {
                    "id": env_id,
                    "name": env_id,
                    "url": env_url,
                }

        except Exception:
            pass

        return {}

    async def _get_token_info(
        self,
        credentials: IntegrationCredentials,
    ) -> dict[str, Any]:
        """Get API token information."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            # Use token lookup endpoint
            response = await self.http_client.get(
                f"{base_url}/apiTokens/lookup",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()

        except Exception:
            pass

        return {}

    def _extract_permissions(self, token_info: dict) -> list[str]:
        """Extract permissions from token info."""
        scopes = token_info.get("scopes", [])
        return [f"scope:{scope}" for scope in scopes]

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Dynatrace.

        Syncs problems, metrics, events, and entities.

        Args:
            credentials: Dynatrace credentials.
            resources: Optional list of resources to sync.
                      Options: problems, metrics, events, entities
            since: Optional datetime for incremental sync.
            limit: Maximum items to sync per resource.

        Returns:
            SyncResult with synced data.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["problems", "metrics", "events", "entities"]

        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Sync problems
            if "problems" in sync_resources:
                problems_result = await self._sync_problems(credentials, since, limit)
                data["problems"] = problems_result["data"]
                total_synced += problems_result["count"]
                errors.extend(problems_result.get("errors", []))

            # Sync metrics metadata
            if "metrics" in sync_resources:
                metrics_result = await self._sync_metrics(credentials, limit)
                data["metrics"] = metrics_result["data"]
                total_synced += metrics_result["count"]
                errors.extend(metrics_result.get("errors", []))

            # Sync events
            if "events" in sync_resources:
                events_result = await self._sync_events(credentials, since, limit)
                data["events"] = events_result["data"]
                total_synced += events_result["count"]
                errors.extend(events_result.get("errors", []))

            # Sync entities
            if "entities" in sync_resources:
                entities_result = await self._sync_entities(credentials, limit)
                data["entities"] = entities_result["data"]
                total_synced += entities_result["count"]
                errors.extend(entities_result.get("errors", []))

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Dynatrace"
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
            logger.exception("Error syncing Dynatrace data")
            return SyncResult(
                success=False,
                message=f"Failed to sync Dynatrace data: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_problems(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync problems from Dynatrace."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        problems = []
        errors = []

        try:
            params: dict[str, Any] = {
                "pageSize": min(limit, 500),
            }

            if since:
                # Convert to milliseconds since epoch
                params["from"] = int(since.timestamp() * 1000)
            else:
                # Default to last 72 hours
                params["from"] = "now-72h"

            response = await self.http_client.get(
                f"{base_url}/problems",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                problem_list = data.get("problems", [])

                for problem in problem_list[:limit]:
                    problems.append({
                        "problem_id": problem.get("problemId"),
                        "display_id": problem.get("displayId"),
                        "title": problem.get("title"),
                        "impact_level": problem.get("impactLevel"),
                        "severity_level": problem.get("severityLevel"),
                        "status": problem.get("status"),
                        "affected_entities": [
                            {
                                "entity_id": e.get("entityId", {}).get("id"),
                                "name": e.get("name"),
                                "type": e.get("entityId", {}).get("type"),
                            }
                            for e in problem.get("affectedEntities", [])
                        ],
                        "root_cause_entity": {
                            "entity_id": problem.get("rootCauseEntity", {}).get("entityId", {}).get("id"),
                            "name": problem.get("rootCauseEntity", {}).get("name"),
                        } if problem.get("rootCauseEntity") else None,
                        "start_time": problem.get("startTime"),
                        "end_time": problem.get("endTime"),
                        "management_zones": [
                            mz.get("name") for mz in problem.get("managementZones", [])
                        ],
                    })
            else:
                errors.append(f"Failed to fetch problems: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing problems: {str(e)}")

        return {"data": problems, "count": len(problems), "errors": errors}

    async def _sync_metrics(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync metric descriptors from Dynatrace."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        metrics = []
        errors = []

        try:
            response = await self.http_client.get(
                f"{base_url}/metrics",
                headers=headers,
                params={
                    "pageSize": min(limit, 500),
                    "fields": "displayName,description,unit,aggregationTypes,dimensionDefinitions",
                },
            )

            if response.status_code == 200:
                data = response.json()
                metric_list = data.get("metrics", [])

                for metric in metric_list[:limit]:
                    metrics.append({
                        "metric_id": metric.get("metricId"),
                        "display_name": metric.get("displayName"),
                        "description": metric.get("description"),
                        "unit": metric.get("unit"),
                        "aggregation_types": metric.get("aggregationTypes", []),
                        "dimension_definitions": [
                            {
                                "key": dim.get("key"),
                                "name": dim.get("name"),
                                "display_name": dim.get("displayName"),
                            }
                            for dim in metric.get("dimensionDefinitions", [])
                        ],
                    })
            else:
                errors.append(f"Failed to fetch metrics: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing metrics: {str(e)}")

        return {"data": metrics, "count": len(metrics), "errors": errors}

    async def _sync_events(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync events from Dynatrace."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        events = []
        errors = []

        try:
            params: dict[str, Any] = {
                "pageSize": min(limit, 1000),
            }

            if since:
                params["from"] = int(since.timestamp() * 1000)
            else:
                params["from"] = "now-24h"

            response = await self.http_client.get(
                f"{base_url}/events",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                event_list = data.get("events", [])

                for event in event_list[:limit]:
                    events.append({
                        "event_id": event.get("eventId"),
                        "event_type": event.get("eventType"),
                        "title": event.get("title"),
                        "start_time": event.get("startTime"),
                        "end_time": event.get("endTime"),
                        "entity_id": event.get("entityId"),
                        "properties": event.get("properties", {}),
                        "status": event.get("status"),
                        "under_maintenance": event.get("underMaintenance", False),
                        "suppress_alert": event.get("suppressAlert", False),
                        "suppress_problem": event.get("suppressProblem", False),
                    })
            else:
                errors.append(f"Failed to fetch events: {response.status_code}")

        except Exception as e:
            errors.append(f"Error syncing events: {str(e)}")

        return {"data": events, "count": len(events), "errors": errors}

    async def _sync_entities(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync entities from Dynatrace."""
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)
        entities = []
        errors = []

        # Sync key entity types
        entity_types = ["HOST", "SERVICE", "APPLICATION", "PROCESS_GROUP"]

        try:
            for entity_type in entity_types:
                response = await self.http_client.get(
                    f"{base_url}/entities",
                    headers=headers,
                    params={
                        "pageSize": min(limit // len(entity_types), 200),
                        "entitySelector": f'type("{entity_type}")',
                        "fields": "properties,tags,managementZones",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    entity_list = data.get("entities", [])

                    for entity in entity_list:
                        entities.append({
                            "entity_id": entity.get("entityId"),
                            "display_name": entity.get("displayName"),
                            "type": entity.get("type"),
                            "first_seen_timestamp": entity.get("firstSeenTms"),
                            "last_seen_timestamp": entity.get("lastSeenTms"),
                            "properties": entity.get("properties", {}),
                            "tags": [
                                {
                                    "context": tag.get("context"),
                                    "key": tag.get("key"),
                                    "value": tag.get("value"),
                                }
                                for tag in entity.get("tags", [])
                            ],
                            "management_zones": [
                                mz.get("name") for mz in entity.get("managementZones", [])
                            ],
                        })
                elif response.status_code != 403:  # Ignore permission errors for specific types
                    errors.append(
                        f"Failed to fetch {entity_type} entities: {response.status_code}"
                    )

        except Exception as e:
            errors.append(f"Error syncing entities: {str(e)}")

        return {"data": entities, "count": len(entities), "errors": errors}

    async def get_problem(
        self,
        credentials: IntegrationCredentials,
        problem_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific problem by ID.

        Args:
            credentials: Dynatrace credentials.
            problem_id: Problem ID.

        Returns:
            Problem data or None if not found.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/problems/{problem_id}",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Error fetching problem {problem_id}: {e}")
            return None

    async def query_metrics(
        self,
        credentials: IntegrationCredentials,
        metric_selector: str,
        resolution: str = "1h",
        time_from: str = "now-24h",
        time_to: str = "now",
    ) -> Optional[list[dict[str, Any]]]:
        """Query metric data points.

        Args:
            credentials: Dynatrace credentials.
            metric_selector: Metric selector (e.g., "builtin:host.cpu.usage").
            resolution: Time resolution (e.g., "1h", "15m").
            time_from: Start time (e.g., "now-24h", timestamp in ms).
            time_to: End time (e.g., "now", timestamp in ms).

        Returns:
            List of metric data or None on failure.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/metrics/query",
                headers=headers,
                params={
                    "metricSelector": metric_selector,
                    "resolution": resolution,
                    "from": time_from,
                    "to": time_to,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("result", [])

            logger.error(f"Failed to query metrics: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return None

    async def create_event(
        self,
        credentials: IntegrationCredentials,
        event_type: str,
        title: str,
        entity_selector: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Create an event in Dynatrace.

        Args:
            credentials: Dynatrace credentials.
            event_type: Event type (e.g., "CUSTOM_INFO", "CUSTOM_DEPLOYMENT").
            title: Event title.
            entity_selector: Optional entity selector to attach event.
            properties: Optional custom properties.
            start_time: Start time in milliseconds (defaults to now).
            end_time: End time in milliseconds.

        Returns:
            Created event data or None on failure.
        """
        base_url = self._get_base_url(credentials)
        headers = self._get_headers(credentials)

        payload: dict[str, Any] = {
            "eventType": event_type,
            "title": title,
            "properties": properties or {},
        }

        if entity_selector:
            payload["entitySelector"] = entity_selector
        if start_time:
            payload["startTime"] = start_time
        if end_time:
            payload["endTime"] = end_time

        try:
            response = await self.http_client.post(
                f"{base_url}/events/ingest",
                headers=headers,
                json=payload,
            )

            if response.status_code in (200, 201):
                return response.json()

            logger.error(f"Failed to create event: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return None
