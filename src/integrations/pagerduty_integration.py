"""
PagerDuty Integration for incident tracking.

Syncs incidents to learn from outages and generate preventive tests.
Every incident is a test that should have existed - we correlate
incidents with deployments and code changes to identify gaps.

API Docs: https://developer.pagerduty.com/api-reference/
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class IncidentStatus(str, Enum):
    """PagerDuty incident status."""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class IncidentUrgency(str, Enum):
    """PagerDuty incident urgency level."""
    HIGH = "high"
    LOW = "low"


class TimelineEntryType(str, Enum):
    """Types of incident timeline entries."""
    TRIGGER = "trigger"
    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"
    ASSIGN = "assign"
    ESCALATE = "escalate"
    ANNOTATE = "annotate"
    REACH_TRIGGER_LIMIT = "reach_trigger_limit"
    NOTIFY = "notify"
    PRIORITY_CHANGE = "priority_change"
    SNOOZE = "snooze"
    UNACKNOWLEDGE = "unacknowledge"


@dataclass
class PagerDutyService:
    """A PagerDuty service."""
    service_id: str
    name: str
    description: str | None
    status: str
    created_at: datetime
    html_url: str
    integration_type: str | None = None
    escalation_policy_id: str | None = None


@dataclass
class PagerDutyUser:
    """A PagerDuty user."""
    user_id: str
    name: str
    email: str
    html_url: str


@dataclass
class TimelineEntry:
    """An entry in the incident timeline."""
    entry_id: str
    entry_type: str
    created_at: datetime
    message: str | None
    agent_type: str | None  # "user_reference", "service_reference", etc.
    agent_id: str | None
    agent_name: str | None
    channel_type: str | None  # "web_ui", "api", "email", etc.


@dataclass
class PagerDutyIncident:
    """A PagerDuty incident with full context."""
    incident_id: str
    incident_number: int
    title: str
    description: str | None
    status: IncidentStatus
    urgency: IncidentUrgency
    priority: str | None

    # Service info
    service_id: str
    service_name: str

    # Timestamps
    created_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None
    last_status_change_at: datetime | None

    # Duration metrics
    duration_seconds: int | None
    time_to_acknowledge_seconds: int | None
    time_to_resolve_seconds: int | None

    # People
    assigned_to: list[str] = field(default_factory=list)
    acknowledged_by: str | None = None
    resolved_by: str | None = None

    # Links
    html_url: str = ""

    # Related data
    alert_count: int = 0
    escalation_policy_id: str | None = None
    escalation_policy_name: str | None = None

    # Root cause analysis
    summary: str | None = None
    body_text: str | None = None  # Incident body/notes

    # Related changes (for correlation)
    related_change_events: list[dict] = field(default_factory=list)


@dataclass
class IncidentAnalysis:
    """AI-enhanced analysis of an incident for test generation."""
    incident_id: str
    title: str

    # Impact assessment
    duration_seconds: int
    is_critical: bool  # Based on priority/urgency
    affected_service: str

    # Root cause signals
    likely_root_cause: str | None
    error_patterns: list[str]
    affected_endpoints: list[str]

    # Correlation with code changes
    recent_deployments: list[dict]
    suspicious_commits: list[dict]

    # Test suggestions
    suggested_test_scenarios: list[str]
    coverage_gaps_identified: list[str]

    # Timeline summary
    timeline_summary: str


class PagerDutyIntegration:
    """
    PagerDuty Integration for incident tracking and analysis.

    API Docs: https://developer.pagerduty.com/api-reference/

    Features:
    - Fetch recent incidents
    - Get incident details and timeline
    - Correlate with deployments via Change Events API
    - Analyze incidents for test generation
    - Track MTTR (Mean Time To Resolution)
    """

    def __init__(self, api_token: str, default_since_days: int = 30):
        """
        Initialize PagerDuty integration.

        Args:
            api_token: PagerDuty API token (REST API key with read access)
            default_since_days: Default number of days to look back for incidents
        """
        self.api_token = api_token
        self.base_url = "https://api.pagerduty.com"
        self.http = httpx.AsyncClient(timeout=30.0)
        self.default_since_days = default_since_days
        self.log = logger.bind(component="pagerduty")

    @property
    def headers(self) -> dict:
        """Get headers for PagerDuty API requests."""
        return {
            "Authorization": f"Token token={self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2",
        }

    async def test_connection(self) -> bool:
        """
        Test if API token is valid.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = await self.http.get(
                f"{self.base_url}/abilities",
                headers=self.headers,
            )

            if response.status_code == 200:
                data = response.json()
                abilities = data.get("abilities", [])
                self.log.info(
                    "PagerDuty connection successful",
                    abilities_count=len(abilities)
                )
                return True
            elif response.status_code == 401:
                self.log.error("PagerDuty authentication failed - invalid token")
                return False
            else:
                self.log.error(
                    "PagerDuty connection failed",
                    status_code=response.status_code,
                    response=response.text[:200]
                )
                return False

        except Exception as e:
            self.log.error("PagerDuty connection error", error=str(e))
            return False

    async def get_incidents(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        statuses: list[IncidentStatus] | None = None,
        service_ids: list[str] | None = None,
        urgencies: list[IncidentUrgency] | None = None,
        limit: int = 25,
        offset: int = 0,
        include_body: bool = False,
    ) -> list[PagerDutyIncident]:
        """
        Get incidents within a time range.

        Args:
            since: Start of time range (defaults to default_since_days ago)
            until: End of time range (defaults to now)
            statuses: Filter by incident statuses
            service_ids: Filter by service IDs
            urgencies: Filter by urgency levels
            limit: Maximum number of incidents to return (max 100)
            offset: Pagination offset
            include_body: Whether to fetch full incident body/notes

        Returns:
            List of PagerDuty incidents
        """
        # Set defaults
        if since is None:
            since = datetime.utcnow() - timedelta(days=self.default_since_days)
        if until is None:
            until = datetime.utcnow()

        # Build query parameters
        params: dict[str, Any] = {
            "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": min(limit, 100),
            "offset": offset,
            "sort_by": "created_at:desc",
            "include[]": ["acknowledgers", "assignees", "services", "escalation_policies"],
        }

        if statuses:
            params["statuses[]"] = [s.value for s in statuses]
        if service_ids:
            params["service_ids[]"] = service_ids
        if urgencies:
            params["urgencies[]"] = [u.value for u in urgencies]

        try:
            response = await self.http.get(
                f"{self.base_url}/incidents",
                headers=self.headers,
                params=params,
            )

            if response.status_code != 200:
                self.log.error(
                    "Failed to fetch incidents",
                    status_code=response.status_code,
                    response=response.text[:200]
                )
                return []

            data = response.json()
            incidents = []

            for item in data.get("incidents", []):
                incident = self._parse_incident(item)

                # Optionally fetch full body
                if include_body and incident:
                    body = await self._get_incident_body(incident.incident_id)
                    if body:
                        incident.body_text = body

                if incident:
                    incidents.append(incident)

            self.log.info(
                "Fetched PagerDuty incidents",
                count=len(incidents),
                since=since.isoformat(),
                until=until.isoformat()
            )

            return incidents

        except Exception as e:
            self.log.error("Error fetching incidents", error=str(e))
            return []

    def _parse_incident(self, item: dict) -> PagerDutyIncident | None:
        """Parse a raw incident API response into a PagerDutyIncident."""
        try:
            # Parse timestamps
            created_at = self._parse_datetime(item.get("created_at"))
            acknowledged_at = self._parse_datetime(item.get("last_status_change_at")) if item.get("status") in ["acknowledged", "resolved"] else None
            resolved_at = self._parse_datetime(item.get("last_status_change_at")) if item.get("status") == "resolved" else None
            last_status_change_at = self._parse_datetime(item.get("last_status_change_at"))

            # Calculate durations
            duration_seconds = None
            time_to_acknowledge_seconds = None
            time_to_resolve_seconds = None

            if created_at and resolved_at:
                duration_seconds = int((resolved_at - created_at).total_seconds())

            # Extract service info
            service = item.get("service", {})
            service_id = service.get("id", "")
            service_name = service.get("summary", "Unknown Service")

            # Extract escalation policy
            escalation_policy = item.get("escalation_policy", {})

            # Extract assignees
            assigned_to = [
                a.get("assignee", {}).get("summary", "")
                for a in item.get("assignments", [])
            ]

            # Extract acknowledgers
            acknowledgers = item.get("acknowledgements", [])
            acknowledged_by = None
            if acknowledgers:
                first_ack = acknowledgers[0]
                acknowledged_by = first_ack.get("acknowledger", {}).get("summary")
                ack_at = self._parse_datetime(first_ack.get("at"))
                if ack_at and created_at:
                    time_to_acknowledge_seconds = int((ack_at - created_at).total_seconds())

            # Get resolver from last status change if resolved
            resolved_by = None
            if item.get("status") == "resolved" and acknowledgers:
                # In a real implementation, we'd need to check the timeline
                # For now, use the last acknowledger as a proxy
                resolved_by = acknowledgers[-1].get("acknowledger", {}).get("summary")

            # Priority
            priority = None
            if item.get("priority"):
                priority = item["priority"].get("summary")

            return PagerDutyIncident(
                incident_id=item.get("id", ""),
                incident_number=item.get("incident_number", 0),
                title=item.get("title", ""),
                description=item.get("description"),
                status=IncidentStatus(item.get("status", "triggered")),
                urgency=IncidentUrgency(item.get("urgency", "high")),
                priority=priority,
                service_id=service_id,
                service_name=service_name,
                created_at=created_at,
                acknowledged_at=acknowledged_at,
                resolved_at=resolved_at,
                last_status_change_at=last_status_change_at,
                duration_seconds=duration_seconds,
                time_to_acknowledge_seconds=time_to_acknowledge_seconds,
                time_to_resolve_seconds=time_to_resolve_seconds,
                assigned_to=assigned_to,
                acknowledged_by=acknowledged_by,
                resolved_by=resolved_by,
                html_url=item.get("html_url", ""),
                alert_count=item.get("alert_counts", {}).get("all", 0),
                escalation_policy_id=escalation_policy.get("id"),
                escalation_policy_name=escalation_policy.get("summary"),
                summary=item.get("summary"),
            )

        except Exception as e:
            self.log.warning(
                "Failed to parse incident",
                incident_id=item.get("id"),
                error=str(e)
            )
            return None

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse a datetime string from PagerDuty API."""
        if not dt_str:
            return None
        try:
            # Handle both formats
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str.replace("+00:00", ""))
        except (ValueError, TypeError):
            return None

    async def _get_incident_body(self, incident_id: str) -> str | None:
        """Get the full body/notes of an incident."""
        try:
            response = await self.http.get(
                f"{self.base_url}/incidents/{incident_id}",
                headers=self.headers,
            )

            if response.status_code == 200:
                data = response.json()
                incident = data.get("incident", {})
                return incident.get("body", {}).get("details")
            return None

        except Exception as e:
            self.log.warning(
                "Failed to fetch incident body",
                incident_id=incident_id,
                error=str(e)
            )
            return None

    async def get_incident(self, incident_id: str) -> PagerDutyIncident | None:
        """
        Get a single incident by ID.

        Args:
            incident_id: PagerDuty incident ID

        Returns:
            PagerDutyIncident or None if not found
        """
        try:
            response = await self.http.get(
                f"{self.base_url}/incidents/{incident_id}",
                headers=self.headers,
                params={
                    "include[]": ["acknowledgers", "assignees", "services", "escalation_policies"]
                }
            )

            if response.status_code == 404:
                self.log.warning("Incident not found", incident_id=incident_id)
                return None

            if response.status_code != 200:
                self.log.error(
                    "Failed to fetch incident",
                    incident_id=incident_id,
                    status_code=response.status_code
                )
                return None

            data = response.json()
            incident = self._parse_incident(data.get("incident", {}))

            # Also fetch body
            if incident:
                body = await self._get_incident_body(incident_id)
                if body:
                    incident.body_text = body

            return incident

        except Exception as e:
            self.log.error(
                "Error fetching incident",
                incident_id=incident_id,
                error=str(e)
            )
            return None

    async def get_incident_timeline(self, incident_id: str) -> list[TimelineEntry]:
        """
        Get the timeline/log entries for an incident.

        The timeline provides a detailed chronological record of all
        events that occurred during the incident lifecycle.

        Args:
            incident_id: PagerDuty incident ID

        Returns:
            List of timeline entries
        """
        try:
            response = await self.http.get(
                f"{self.base_url}/incidents/{incident_id}/log_entries",
                headers=self.headers,
                params={
                    "include[]": ["channels"],
                    "is_overview": "false",
                }
            )

            if response.status_code != 200:
                self.log.error(
                    "Failed to fetch incident timeline",
                    incident_id=incident_id,
                    status_code=response.status_code
                )
                return []

            data = response.json()
            entries = []

            for item in data.get("log_entries", []):
                entry = self._parse_timeline_entry(item)
                if entry:
                    entries.append(entry)

            # Sort by created_at ascending
            entries.sort(key=lambda e: e.created_at)

            self.log.info(
                "Fetched incident timeline",
                incident_id=incident_id,
                entry_count=len(entries)
            )

            return entries

        except Exception as e:
            self.log.error(
                "Error fetching incident timeline",
                incident_id=incident_id,
                error=str(e)
            )
            return []

    def _parse_timeline_entry(self, item: dict) -> TimelineEntry | None:
        """Parse a timeline entry from the API response."""
        try:
            # Extract agent info
            agent = item.get("agent", {})
            agent_type = agent.get("type")
            agent_id = agent.get("id")
            agent_name = agent.get("summary")

            # Extract channel info
            channel = item.get("channel", {})
            channel_type = channel.get("type")

            # Build message from various sources
            message = None
            if item.get("note"):
                message = item["note"].get("content")
            elif item.get("summary"):
                message = item["summary"]

            return TimelineEntry(
                entry_id=item.get("id", ""),
                entry_type=item.get("type", "unknown"),
                created_at=self._parse_datetime(item.get("created_at")) or datetime.utcnow(),
                message=message,
                agent_type=agent_type,
                agent_id=agent_id,
                agent_name=agent_name,
                channel_type=channel_type,
            )

        except Exception as e:
            self.log.warning(
                "Failed to parse timeline entry",
                entry_id=item.get("id"),
                error=str(e)
            )
            return None

    async def get_services(self) -> list[PagerDutyService]:
        """
        List all services.

        Returns:
            List of PagerDuty services
        """
        try:
            services = []
            offset = 0
            limit = 100

            while True:
                response = await self.http.get(
                    f"{self.base_url}/services",
                    headers=self.headers,
                    params={
                        "offset": offset,
                        "limit": limit,
                        "include[]": ["integrations", "escalation_policies"],
                    }
                )

                if response.status_code != 200:
                    self.log.error(
                        "Failed to fetch services",
                        status_code=response.status_code
                    )
                    break

                data = response.json()
                items = data.get("services", [])

                for item in items:
                    service = self._parse_service(item)
                    if service:
                        services.append(service)

                # Check for more pages
                if not data.get("more", False):
                    break

                offset += limit

            self.log.info("Fetched PagerDuty services", count=len(services))
            return services

        except Exception as e:
            self.log.error("Error fetching services", error=str(e))
            return []

    def _parse_service(self, item: dict) -> PagerDutyService | None:
        """Parse a service from the API response."""
        try:
            # Get integration type if available
            integrations = item.get("integrations", [])
            integration_type = None
            if integrations:
                integration_type = integrations[0].get("type")

            return PagerDutyService(
                service_id=item.get("id", ""),
                name=item.get("name", ""),
                description=item.get("description"),
                status=item.get("status", "active"),
                created_at=self._parse_datetime(item.get("created_at")) or datetime.utcnow(),
                html_url=item.get("html_url", ""),
                integration_type=integration_type,
                escalation_policy_id=item.get("escalation_policy", {}).get("id"),
            )

        except Exception as e:
            self.log.warning(
                "Failed to parse service",
                service_id=item.get("id"),
                error=str(e)
            )
            return None

    async def get_change_events(
        self,
        service_ids: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """
        Get change events (deployments, config changes, etc.).

        Change events can be correlated with incidents to identify
        which changes may have caused outages.

        Args:
            service_ids: Filter by service IDs
            since: Start of time range
            until: End of time range
            limit: Maximum number of events

        Returns:
            List of change events
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)
        if until is None:
            until = datetime.utcnow()

        params: dict[str, Any] = {
            "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": min(limit, 100),
        }

        if service_ids:
            params["routing_key"] = service_ids[0]  # Change events use routing keys

        try:
            response = await self.http.get(
                f"{self.base_url}/change_events",
                headers=self.headers,
                params=params,
            )

            if response.status_code != 200:
                self.log.warning(
                    "Failed to fetch change events",
                    status_code=response.status_code
                )
                return []

            data = response.json()
            events = data.get("change_events", [])

            self.log.info("Fetched change events", count=len(events))
            return events

        except Exception as e:
            self.log.error("Error fetching change events", error=str(e))
            return []

    async def get_incident_with_correlation(
        self,
        incident_id: str,
        correlation_window_hours: int = 24,
    ) -> PagerDutyIncident | None:
        """
        Get incident with correlated change events.

        This is useful for identifying which deployments or config
        changes may have caused the incident.

        Args:
            incident_id: PagerDuty incident ID
            correlation_window_hours: Hours before incident to look for changes

        Returns:
            PagerDutyIncident with related_change_events populated
        """
        incident = await self.get_incident(incident_id)
        if not incident:
            return None

        # Find change events around the incident time
        since = incident.created_at - timedelta(hours=correlation_window_hours)
        until = incident.created_at

        change_events = await self.get_change_events(
            service_ids=[incident.service_id],
            since=since,
            until=until,
        )

        incident.related_change_events = change_events

        return incident

    async def analyze_incident_for_tests(
        self,
        incident_id: str,
    ) -> IncidentAnalysis | None:
        """
        Analyze an incident to suggest tests that could have prevented it.

        This performs a comprehensive analysis of the incident:
        - Fetches full incident details and timeline
        - Correlates with recent change events
        - Identifies patterns and root cause signals
        - Generates test suggestions

        Args:
            incident_id: PagerDuty incident ID

        Returns:
            IncidentAnalysis with test suggestions
        """
        # Get incident with timeline
        incident = await self.get_incident_with_correlation(incident_id)
        if not incident:
            return None

        timeline = await self.get_incident_timeline(incident_id)

        # Analyze timeline for patterns
        error_patterns = []
        affected_endpoints = []

        for entry in timeline:
            if entry.message:
                # Look for error patterns in messages
                if "error" in entry.message.lower():
                    error_patterns.append(entry.message)
                # Look for URLs/endpoints
                if "http" in entry.message.lower() or "/" in entry.message:
                    # Simple extraction - could be more sophisticated
                    words = entry.message.split()
                    for word in words:
                        if word.startswith("/") or word.startswith("http"):
                            affected_endpoints.append(word)

        # Generate timeline summary
        timeline_events = []
        for entry in timeline:
            event_summary = f"{entry.created_at.strftime('%H:%M:%S')} - {entry.entry_type}"
            if entry.agent_name:
                event_summary += f" by {entry.agent_name}"
            timeline_events.append(event_summary)
        timeline_summary = "\n".join(timeline_events[:20])  # Limit to 20 events

        # Determine if critical
        is_critical = (
            incident.urgency == IncidentUrgency.HIGH or
            (incident.priority and "P1" in incident.priority) or
            (incident.duration_seconds and incident.duration_seconds > 3600)
        )

        # Generate test suggestions based on incident type
        suggested_tests = self._generate_test_suggestions(
            incident, error_patterns, affected_endpoints
        )

        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(
            incident, error_patterns, affected_endpoints
        )

        return IncidentAnalysis(
            incident_id=incident.incident_id,
            title=incident.title,
            duration_seconds=incident.duration_seconds or 0,
            is_critical=is_critical,
            affected_service=incident.service_name,
            likely_root_cause=incident.body_text or incident.description,
            error_patterns=error_patterns[:10],
            affected_endpoints=list(set(affected_endpoints))[:10],
            recent_deployments=[
                {
                    "summary": e.get("summary", ""),
                    "timestamp": e.get("timestamp"),
                    "source": e.get("source", {}).get("type"),
                }
                for e in incident.related_change_events
            ],
            suspicious_commits=[],  # Would need GitHub integration
            suggested_test_scenarios=suggested_tests,
            coverage_gaps_identified=coverage_gaps,
            timeline_summary=timeline_summary,
        )

    def _generate_test_suggestions(
        self,
        incident: PagerDutyIncident,
        error_patterns: list[str],
        affected_endpoints: list[str],
    ) -> list[str]:
        """Generate test suggestions based on incident analysis."""
        suggestions = []

        # Service-level health check
        suggestions.append(
            f"Add health check test for {incident.service_name} service"
        )

        # Endpoint tests
        for endpoint in affected_endpoints[:5]:
            suggestions.append(
                f"Add API test for endpoint: {endpoint}"
            )

        # Error-specific tests
        if "timeout" in incident.title.lower():
            suggestions.append(
                "Add performance test with timeout assertions"
            )

        if "database" in incident.title.lower() or "db" in incident.title.lower():
            suggestions.append(
                "Add database connection and query performance tests"
            )

        if "authentication" in incident.title.lower() or "auth" in incident.title.lower():
            suggestions.append(
                "Add authentication flow E2E test"
            )

        if "memory" in incident.title.lower() or "oom" in incident.title.lower():
            suggestions.append(
                "Add memory usage monitoring test"
            )

        # If incident was long-running, suggest alerting tests
        if incident.duration_seconds and incident.duration_seconds > 1800:
            suggestions.append(
                "Add alerting/monitoring test to verify faster detection"
            )

        return suggestions

    def _identify_coverage_gaps(
        self,
        incident: PagerDutyIncident,
        error_patterns: list[str],
        affected_endpoints: list[str],
    ) -> list[str]:
        """Identify test coverage gaps based on incident."""
        gaps = []

        # Critical path coverage
        if incident.urgency == IncidentUrgency.HIGH:
            gaps.append(
                f"Critical path for {incident.service_name} needs comprehensive E2E coverage"
            )

        # Endpoint coverage
        if affected_endpoints:
            gaps.append(
                f"Endpoints {', '.join(affected_endpoints[:3])} lack test coverage"
            )

        # Error scenario coverage
        for pattern in error_patterns[:3]:
            gaps.append(
                f"Error scenario not covered: {pattern[:100]}"
            )

        # Integration coverage
        if incident.related_change_events:
            gaps.append(
                "Post-deployment smoke tests needed for this service"
            )

        return gaps

    async def get_incident_metrics(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        service_ids: list[str] | None = None,
    ) -> dict:
        """
        Get aggregated incident metrics for reporting.

        Args:
            since: Start of time range
            until: End of time range
            service_ids: Filter by services

        Returns:
            Dictionary with incident metrics
        """
        incidents = await self.get_incidents(
            since=since,
            until=until,
            service_ids=service_ids,
            limit=100,
        )

        if not incidents:
            return {
                "total_incidents": 0,
                "by_status": {},
                "by_urgency": {},
                "mean_time_to_acknowledge": None,
                "mean_time_to_resolve": None,
                "top_services": [],
            }

        # Count by status
        by_status = {}
        for incident in incidents:
            status = incident.status.value
            by_status[status] = by_status.get(status, 0) + 1

        # Count by urgency
        by_urgency = {}
        for incident in incidents:
            urgency = incident.urgency.value
            by_urgency[urgency] = by_urgency.get(urgency, 0) + 1

        # Calculate MTTA and MTTR
        tta_values = [
            i.time_to_acknowledge_seconds
            for i in incidents
            if i.time_to_acknowledge_seconds
        ]
        ttr_values = [
            i.duration_seconds
            for i in incidents
            if i.duration_seconds
        ]

        mtta = sum(tta_values) / len(tta_values) if tta_values else None
        mttr = sum(ttr_values) / len(ttr_values) if ttr_values else None

        # Top services by incident count
        service_counts: dict[str, int] = {}
        for incident in incidents:
            service_counts[incident.service_name] = service_counts.get(
                incident.service_name, 0
            ) + 1

        top_services = sorted(
            service_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_incidents": len(incidents),
            "by_status": by_status,
            "by_urgency": by_urgency,
            "mean_time_to_acknowledge_seconds": mtta,
            "mean_time_to_resolve_seconds": mttr,
            "top_services": [
                {"service": name, "count": count}
                for name, count in top_services
            ],
        }

    async def close(self):
        """Close the HTTP client."""
        await self.http.aclose()
        self.log.info("PagerDuty integration closed")


def create_pagerduty_integration(
    api_token: str | None = None,
    default_since_days: int = 30,
) -> PagerDutyIntegration | None:
    """
    Factory function for PagerDutyIntegration.

    Args:
        api_token: PagerDuty API token. If not provided, reads from
                   PAGERDUTY_API_TOKEN environment variable.
        default_since_days: Default lookback period for incidents

    Returns:
        PagerDutyIntegration instance or None if no token available
    """
    import os

    token = api_token or os.environ.get("PAGERDUTY_API_TOKEN")

    if not token:
        logger.warning("No PagerDuty API token provided")
        return None

    return PagerDutyIntegration(
        api_token=token,
        default_since_days=default_since_days,
    )
