"""Cross-Correlation Engine for SDLC Events.

This is the intelligence layer that correlates events across the entire SDLC.
It provides algorithms for:
- Event correlation based on shared keys (commit SHA, PR number, Jira key, etc.)
- Time-based correlation (proximity scoring)
- Pattern detection across the timeline
- AI-powered insight generation using Claude

The engine is designed to be used by both the API layer and background jobs.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


# =============================================================================
# Data Classes
# =============================================================================


class InsightType(str, Enum):
    """Types of correlation insights that can be generated."""
    RISK_PATTERN = "risk_pattern"
    PERFORMANCE_TREND = "performance_trend"
    FAILURE_CLUSTER = "failure_cluster"
    DEPLOYMENT_RISK = "deployment_risk"
    COVERAGE_GAP = "coverage_gap"
    FLAKY_TEST = "flaky_test"
    DEPENDENCY_ISSUE = "dependency_issue"
    REGRESSION = "regression"
    RECOMMENDATION = "recommendation"


class Severity(str, Enum):
    """Severity levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CorrelationKey:
    """A key that can be used to correlate events."""
    key_type: str  # commit_sha, pr_number, jira_key, deploy_id, branch_name
    value: str | int

    def __hash__(self) -> int:
        return hash((self.key_type, str(self.value)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CorrelationKey):
            return False
        return self.key_type == other.key_type and str(self.value) == str(other.value)


@dataclass
class SDLCEventData:
    """Parsed SDLC event data for correlation analysis."""
    id: str
    project_id: str
    event_type: str
    source_platform: str
    external_id: str
    external_url: str | None
    title: str | None
    occurred_at: datetime
    commit_sha: str | None
    pr_number: int | None
    jira_key: str | None
    deploy_id: str | None
    branch_name: str | None
    data: dict = field(default_factory=dict)

    @classmethod
    def from_db_row(cls, row: dict) -> "SDLCEventData":
        """Create from database row."""
        occurred_at = row.get("occurred_at")
        if isinstance(occurred_at, str):
            # Handle various ISO format variations
            if occurred_at.endswith("Z"):
                occurred_at = occurred_at[:-1] + "+00:00"
            occurred_at = datetime.fromisoformat(occurred_at)

        return cls(
            id=row["id"],
            project_id=row["project_id"],
            event_type=row["event_type"],
            source_platform=row["source_platform"],
            external_id=row["external_id"],
            external_url=row.get("external_url"),
            title=row.get("title"),
            occurred_at=occurred_at,
            commit_sha=row.get("commit_sha"),
            pr_number=row.get("pr_number"),
            jira_key=row.get("jira_key"),
            deploy_id=row.get("deploy_id"),
            branch_name=row.get("branch_name"),
            data=row.get("data") or {},
        )

    def get_correlation_keys(self) -> list[CorrelationKey]:
        """Extract all correlation keys from this event."""
        keys = []
        if self.commit_sha:
            keys.append(CorrelationKey("commit_sha", self.commit_sha))
        if self.pr_number:
            keys.append(CorrelationKey("pr_number", self.pr_number))
        if self.jira_key:
            keys.append(CorrelationKey("jira_key", self.jira_key))
        if self.deploy_id:
            keys.append(CorrelationKey("deploy_id", self.deploy_id))
        if self.branch_name:
            keys.append(CorrelationKey("branch_name", self.branch_name))
        return keys


@dataclass
class CorrelationResult:
    """Result of correlating two events."""
    source_event_id: str
    target_event_id: str
    correlation_type: str
    confidence: float
    factors: list[dict] = field(default_factory=list)
    shared_keys: list[CorrelationKey] = field(default_factory=list)


@dataclass
class GeneratedInsight:
    """An AI-generated insight from correlation analysis."""
    insight_type: InsightType
    severity: Severity
    title: str
    description: str
    recommendations: list[dict]
    event_ids: list[str]
    confidence: float = 0.0


# =============================================================================
# Correlation Engine
# =============================================================================


class CorrelationEngine:
    """Core correlation engine for SDLC events.

    This class provides methods for:
    - Finding correlated events based on shared keys
    - Scoring correlations based on time proximity and other factors
    - Detecting patterns across the SDLC timeline
    - Generating AI-powered insights
    """

    def __init__(self):
        self.supabase = get_supabase_client()
        self.settings = get_settings()

    # =========================================================================
    # Event Retrieval
    # =========================================================================

    async def get_events_in_window(
        self,
        project_id: str,
        start_time: datetime,
        end_time: datetime,
        event_types: list[str] | None = None,
        limit: int = 500,
    ) -> list[SDLCEventData]:
        """Retrieve SDLC events within a time window.

        Args:
            project_id: Project to query
            start_time: Start of time window
            end_time: End of time window
            event_types: Optional filter for event types
            limit: Maximum events to return

        Returns:
            List of SDLCEventData objects
        """
        query_path = (
            f"/sdlc_events?project_id=eq.{project_id}"
            f"&occurred_at=gte.{start_time.isoformat()}"
            f"&occurred_at=lte.{end_time.isoformat()}"
            f"&order=occurred_at.asc"
            f"&limit={limit}"
        )

        if event_types:
            types_filter = ",".join(event_types)
            query_path += f"&event_type=in.({types_filter})"

        result = await self.supabase.request(query_path)

        if result.get("error"):
            logger.error("Failed to fetch events", error=result["error"])
            return []

        events_data = result.get("data") or []
        return [SDLCEventData.from_db_row(row) for row in events_data]

    async def get_events_by_key(
        self,
        project_id: str,
        key: CorrelationKey,
        limit: int = 100,
    ) -> list[SDLCEventData]:
        """Get all events matching a correlation key.

        Args:
            project_id: Project to query
            key: Correlation key to search by
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        query_path = (
            f"/sdlc_events?project_id=eq.{project_id}"
            f"&{key.key_type}=eq.{key.value}"
            f"&order=occurred_at.asc"
            f"&limit={limit}"
        )

        result = await self.supabase.request(query_path)

        if result.get("error"):
            logger.error("Failed to fetch events by key", error=result["error"], key=key)
            return []

        events_data = result.get("data") or []
        return [SDLCEventData.from_db_row(row) for row in events_data]

    # =========================================================================
    # Correlation Algorithms
    # =========================================================================

    def calculate_time_proximity_score(
        self,
        event1: SDLCEventData,
        event2: SDLCEventData,
        max_hours: int = 24,
    ) -> float:
        """Calculate a score based on time proximity between two events.

        Closer events get higher scores. Max score is 0.3.

        Args:
            event1: First event
            event2: Second event
            max_hours: Maximum time window to consider

        Returns:
            Score from 0 to 0.3
        """
        time_diff = abs((event1.occurred_at - event2.occurred_at).total_seconds())
        hours_diff = time_diff / 3600

        if hours_diff > max_hours:
            return 0.0

        # Linear decay: closer = higher score
        proximity_ratio = 1 - (hours_diff / max_hours)
        return proximity_ratio * 0.3

    def calculate_key_overlap_score(
        self,
        event1: SDLCEventData,
        event2: SDLCEventData,
    ) -> tuple[float, list[CorrelationKey]]:
        """Calculate correlation score based on shared keys.

        Args:
            event1: First event
            event2: Second event

        Returns:
            Tuple of (score 0-0.5, list of shared keys)
        """
        keys1 = set(event1.get_correlation_keys())
        keys2 = set(event2.get_correlation_keys())

        shared = keys1 & keys2

        if not shared:
            return 0.0, []

        # Weight different key types differently
        key_weights = {
            "commit_sha": 0.5,  # Strongest correlation
            "pr_number": 0.4,
            "deploy_id": 0.4,
            "jira_key": 0.3,
            "branch_name": 0.2,  # Weakest (many events on same branch)
        }

        total_weight = sum(key_weights.get(k.key_type, 0.1) for k in shared)
        # Cap at 0.5
        return min(0.5, total_weight), list(shared)

    def calculate_correlation_confidence(
        self,
        event1: SDLCEventData,
        event2: SDLCEventData,
        max_hours: int = 48,
    ) -> CorrelationResult:
        """Calculate overall correlation confidence between two events.

        Combines time proximity and key overlap scores.

        Args:
            event1: First event (earlier in time)
            event2: Second event (later in time)
            max_hours: Max time window for proximity scoring

        Returns:
            CorrelationResult with confidence and factors
        """
        factors = []

        # Key overlap score (0-0.5)
        key_score, shared_keys = self.calculate_key_overlap_score(event1, event2)
        if key_score > 0:
            factors.append({
                "factor": "shared_keys",
                "score": round(key_score, 3),
                "description": f"Shared keys: {', '.join(f'{k.key_type}={k.value}' for k in shared_keys)}",
            })

        # Time proximity score (0-0.3)
        time_score = self.calculate_time_proximity_score(event1, event2, max_hours)
        if time_score > 0:
            time_diff = abs((event1.occurred_at - event2.occurred_at).total_seconds()) / 3600
            factors.append({
                "factor": "time_proximity",
                "score": round(time_score, 3),
                "description": f"Events occurred {time_diff:.1f}h apart",
            })

        # Event type relationship score (0-0.2)
        type_score = self._calculate_event_type_relationship_score(event1, event2)
        if type_score > 0:
            factors.append({
                "factor": "event_relationship",
                "score": round(type_score, 3),
                "description": f"{event1.event_type} -> {event2.event_type} is a common pattern",
            })

        total_confidence = min(1.0, key_score + time_score + type_score)

        # Determine correlation type based on event types
        correlation_type = self._determine_correlation_type(event1, event2)

        return CorrelationResult(
            source_event_id=event1.id,
            target_event_id=event2.id,
            correlation_type=correlation_type,
            confidence=round(total_confidence, 3),
            factors=factors,
            shared_keys=shared_keys,
        )

    def _calculate_event_type_relationship_score(
        self,
        event1: SDLCEventData,
        event2: SDLCEventData,
    ) -> float:
        """Score based on common event type relationships.

        Certain event type pairs are more likely to be correlated.
        """
        # Common causal relationships (source -> target)
        relationships = {
            ("commit", "build"): 0.2,
            ("commit", "test_run"): 0.15,
            ("commit", "deploy"): 0.15,
            ("pr", "commit"): 0.2,
            ("pr", "build"): 0.15,
            ("pr", "test_run"): 0.15,
            ("deploy", "error"): 0.2,
            ("deploy", "incident"): 0.2,
            ("commit", "error"): 0.15,
            ("feature_flag", "error"): 0.15,
            ("feature_flag", "incident"): 0.15,
            ("build", "deploy"): 0.2,
            ("test_run", "deploy"): 0.1,
            ("requirement", "pr"): 0.15,
            ("requirement", "commit"): 0.1,
        }

        pair = (event1.event_type, event2.event_type)
        return relationships.get(pair, 0.0)

    def _determine_correlation_type(
        self,
        source: SDLCEventData,
        target: SDLCEventData,
    ) -> str:
        """Determine the type of correlation between two events."""
        # Causal relationships
        causal_pairs = {
            ("commit", "error"): "introduced_by",
            ("commit", "incident"): "introduced_by",
            ("deploy", "error"): "caused_by",
            ("deploy", "incident"): "caused_by",
            ("feature_flag", "error"): "caused_by",
            ("pr", "commit"): "related_to",
            ("requirement", "pr"): "related_to",
        }

        pair = (source.event_type, target.event_type)
        return causal_pairs.get(pair, "related_to")

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    async def detect_failure_clusters(
        self,
        project_id: str,
        days: int = 7,
        min_cluster_size: int = 3,
    ) -> list[dict]:
        """Detect clusters of related failures.

        Looks for groups of errors/incidents that share common causes.

        Args:
            project_id: Project to analyze
            days: Number of days to look back
            min_cluster_size: Minimum events to form a cluster

        Returns:
            List of failure cluster dicts
        """
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=days)

        # Get all error and incident events
        events = await self.get_events_in_window(
            project_id,
            start_time,
            end_time,
            event_types=["error", "incident"],
        )

        if len(events) < min_cluster_size:
            return []

        # Group by correlation keys
        key_to_events: dict[CorrelationKey, list[SDLCEventData]] = {}
        for event in events:
            for key in event.get_correlation_keys():
                if key not in key_to_events:
                    key_to_events[key] = []
                key_to_events[key].append(event)

        # Find clusters (groups with at least min_cluster_size events)
        clusters = []
        for key, cluster_events in key_to_events.items():
            if len(cluster_events) >= min_cluster_size:
                clusters.append({
                    "key_type": key.key_type,
                    "key_value": key.value,
                    "event_count": len(cluster_events),
                    "event_ids": [e.id for e in cluster_events],
                    "event_types": list(set(e.event_type for e in cluster_events)),
                    "time_span_hours": (
                        max(e.occurred_at for e in cluster_events) -
                        min(e.occurred_at for e in cluster_events)
                    ).total_seconds() / 3600,
                })

        # Sort by event count descending
        clusters.sort(key=lambda c: c["event_count"], reverse=True)

        return clusters

    async def detect_deployment_risks(
        self,
        project_id: str,
        days: int = 30,
    ) -> list[dict]:
        """Analyze recent deployments and identify risky patterns.

        Looks for deployments that led to errors or incidents.

        Args:
            project_id: Project to analyze
            days: Number of days to look back

        Returns:
            List of deployment risk dicts
        """
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=days)

        # Get deployments
        deploys = await self.get_events_in_window(
            project_id,
            start_time,
            end_time,
            event_types=["deploy", "deployment_status"],
        )

        if not deploys:
            return []

        # For each deployment, look for errors in the following 24 hours
        risks = []
        for deploy in deploys:
            error_window_end = deploy.occurred_at + timedelta(hours=24)

            # Get errors after this deployment
            errors = await self.get_events_in_window(
                project_id,
                deploy.occurred_at,
                min(error_window_end, end_time),
                event_types=["error", "incident"],
            )

            # Filter to errors with shared correlation keys
            related_errors = []
            deploy_keys = set(deploy.get_correlation_keys())

            for error in errors:
                error_keys = set(error.get_correlation_keys())
                if deploy_keys & error_keys:
                    related_errors.append(error)

            if related_errors:
                risks.append({
                    "deploy_id": deploy.id,
                    "deploy_title": deploy.title,
                    "deploy_time": deploy.occurred_at.isoformat(),
                    "commit_sha": deploy.commit_sha,
                    "error_count": len(related_errors),
                    "error_ids": [e.id for e in related_errors],
                    "time_to_first_error_hours": (
                        min(e.occurred_at for e in related_errors) - deploy.occurred_at
                    ).total_seconds() / 3600,
                })

        # Sort by error count descending
        risks.sort(key=lambda r: r["error_count"], reverse=True)

        return risks

    async def analyze_commit_impact(
        self,
        project_id: str,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Analyze the downstream impact of a specific commit.

        Finds all events related to this commit: builds, tests, deploys, errors.

        Args:
            project_id: Project to analyze
            commit_sha: Commit SHA to analyze

        Returns:
            Impact analysis dict
        """
        events = await self.get_events_by_key(
            project_id,
            CorrelationKey("commit_sha", commit_sha),
        )

        if not events:
            return {
                "commit_sha": commit_sha,
                "found": False,
                "message": "No events found for this commit",
            }

        # Categorize events by type
        by_type: dict[str, list[SDLCEventData]] = {}
        for event in events:
            if event.event_type not in by_type:
                by_type[event.event_type] = []
            by_type[event.event_type].append(event)

        # Calculate risk score
        risk_score = 0.0
        risk_factors = []

        if "error" in by_type:
            error_count = len(by_type["error"])
            risk_score += min(0.4, error_count * 0.1)
            risk_factors.append({
                "factor": "production_errors",
                "count": error_count,
                "contribution": min(0.4, error_count * 0.1),
            })

        if "incident" in by_type:
            incident_count = len(by_type["incident"])
            risk_score += min(0.5, incident_count * 0.25)
            risk_factors.append({
                "factor": "incidents",
                "count": incident_count,
                "contribution": min(0.5, incident_count * 0.25),
            })

        # Test failures
        test_events = by_type.get("test_run", [])
        failed_tests = [e for e in test_events if e.data.get("status") in ("failed", "error")]
        if failed_tests:
            fail_ratio = len(failed_tests) / len(test_events) if test_events else 0
            risk_score += fail_ratio * 0.3
            risk_factors.append({
                "factor": "test_failures",
                "count": len(failed_tests),
                "total_tests": len(test_events),
                "contribution": fail_ratio * 0.3,
            })

        # Time span
        time_span = (
            max(e.occurred_at for e in events) - min(e.occurred_at for e in events)
        ).total_seconds() / 3600

        return {
            "commit_sha": commit_sha,
            "found": True,
            "total_events": len(events),
            "events_by_type": {k: len(v) for k, v in by_type.items()},
            "event_ids": [e.id for e in events],
            "risk_score": round(min(1.0, risk_score), 3),
            "risk_factors": risk_factors,
            "time_span_hours": round(time_span, 2),
            "first_event": min(e.occurred_at for e in events).isoformat(),
            "last_event": max(e.occurred_at for e in events).isoformat(),
        }

    # =========================================================================
    # AI Insight Generation
    # =========================================================================

    async def generate_insights(
        self,
        project_id: str,
        days: int = 7,
        max_insights: int = 5,
    ) -> list[GeneratedInsight]:
        """Generate AI-powered insights from correlation analysis.

        Analyzes the SDLC timeline and generates actionable insights.

        Args:
            project_id: Project to analyze
            days: Number of days to analyze
            max_insights: Maximum insights to generate

        Returns:
            List of GeneratedInsight objects
        """
        insights: list[GeneratedInsight] = []

        # Detect failure clusters
        clusters = await self.detect_failure_clusters(project_id, days)
        for cluster in clusters[:2]:  # Top 2 clusters
            if cluster["event_count"] >= 3:
                insights.append(GeneratedInsight(
                    insight_type=InsightType.FAILURE_CLUSTER,
                    severity=Severity.HIGH if cluster["event_count"] >= 5 else Severity.MEDIUM,
                    title=f"Failure cluster detected: {cluster['key_type']}={cluster['key_value']}",
                    description=(
                        f"Found {cluster['event_count']} related failures linked by "
                        f"{cluster['key_type']}. This pattern occurred over "
                        f"{cluster['time_span_hours']:.1f} hours."
                    ),
                    recommendations=[
                        {
                            "action": "investigate_root_cause",
                            "description": f"Investigate the {cluster['key_type']} to find the root cause",
                        },
                        {
                            "action": "add_monitoring",
                            "description": "Add specific monitoring for this failure pattern",
                        },
                    ],
                    event_ids=cluster["event_ids"][:10],
                    confidence=0.8,
                ))

        # Detect deployment risks
        risks = await self.detect_deployment_risks(project_id, days)
        for risk in risks[:2]:  # Top 2 risky deployments
            if risk["error_count"] >= 2:
                insights.append(GeneratedInsight(
                    insight_type=InsightType.DEPLOYMENT_RISK,
                    severity=Severity.HIGH if risk["error_count"] >= 3 else Severity.MEDIUM,
                    title=f"Risky deployment identified: {risk['commit_sha'][:8] if risk['commit_sha'] else 'unknown'}",
                    description=(
                        f"Deployment '{risk['deploy_title'] or 'Unknown'}' led to "
                        f"{risk['error_count']} errors within "
                        f"{risk['time_to_first_error_hours']:.1f} hours."
                    ),
                    recommendations=[
                        {
                            "action": "review_changes",
                            "description": "Review the code changes in this deployment",
                        },
                        {
                            "action": "consider_rollback",
                            "description": "Consider rolling back if errors persist",
                        },
                    ],
                    event_ids=[risk["deploy_id"]] + risk["error_ids"][:9],
                    confidence=0.75,
                ))

        # Use Claude for deeper analysis if we have events
        if len(insights) < max_insights and self.settings.anthropic_api_key:
            ai_insights = await self._generate_ai_insights(project_id, days, max_insights - len(insights))
            insights.extend(ai_insights)

        return insights[:max_insights]

    async def _generate_ai_insights(
        self,
        project_id: str,
        days: int,
        max_insights: int,
    ) -> list[GeneratedInsight]:
        """Use Claude to generate deeper insights.

        Args:
            project_id: Project to analyze
            days: Days to analyze
            max_insights: Max insights to generate

        Returns:
            List of AI-generated insights
        """
        if not self.settings.anthropic_api_key:
            return []

        try:
            import json

            import anthropic

            # Get recent events for analysis
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=days)
            events = await self.get_events_in_window(project_id, start_time, end_time, limit=100)

            if len(events) < 5:
                return []

            # Prepare summary for Claude
            event_summary = self._summarize_events_for_ai(events)

            client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key.get_secret_value()
            )

            prompt = f"""You are an expert at analyzing software development lifecycle (SDLC) data to find actionable insights.

Analyze the following SDLC event summary from the past {days} days:

{event_summary}

Based on this data, generate up to {max_insights} actionable insights. Focus on:
1. Patterns that indicate risk or potential issues
2. Opportunities to improve development velocity
3. Correlations between events that suggest root causes
4. Recommendations for improving quality

Respond with a JSON array of insights:
[
  {{
    "insight_type": "risk_pattern|performance_trend|failure_cluster|deployment_risk|coverage_gap|regression|recommendation",
    "severity": "critical|high|medium|low|info",
    "title": "Brief title",
    "description": "Detailed description of the insight",
    "recommendations": [
      {{"action": "action_name", "description": "what to do"}}
    ],
    "confidence": 0.0-1.0
  }}
]

Return ONLY valid JSON, no markdown or explanations."""

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            insights_data = json.loads(response_text)

            insights = []
            for data in insights_data:
                try:
                    insight = GeneratedInsight(
                        insight_type=InsightType(data.get("insight_type", "recommendation")),
                        severity=Severity(data.get("severity", "info")),
                        title=data["title"],
                        description=data["description"],
                        recommendations=data.get("recommendations", []),
                        event_ids=[],  # AI insights don't have specific event IDs
                        confidence=float(data.get("confidence", 0.6)),
                    )
                    insights.append(insight)
                except (KeyError, ValueError) as e:
                    logger.warning("Failed to parse AI insight", error=str(e))
                    continue

            return insights

        except Exception as e:
            logger.exception("Failed to generate AI insights", error=str(e))
            return []

    def _summarize_events_for_ai(self, events: list[SDLCEventData]) -> str:
        """Create a summary of events for AI analysis."""
        # Count by type
        by_type: dict[str, int] = {}
        by_platform: dict[str, int] = {}
        errors_by_day: dict[str, int] = {}

        for event in events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            by_platform[event.source_platform] = by_platform.get(event.source_platform, 0) + 1

            if event.event_type in ("error", "incident"):
                day = event.occurred_at.strftime("%Y-%m-%d")
                errors_by_day[day] = errors_by_day.get(day, 0) + 1

        summary = f"""## Event Summary
Total Events: {len(events)}

### Events by Type:
{chr(10).join(f'- {t}: {c}' for t, c in sorted(by_type.items(), key=lambda x: -x[1]))}

### Events by Platform:
{chr(10).join(f'- {p}: {c}' for p, c in sorted(by_platform.items(), key=lambda x: -x[1]))}

### Errors/Incidents by Day:
{chr(10).join(f'- {d}: {c}' for d, c in sorted(errors_by_day.items()))}

### Recent Notable Events:
"""

        # Add recent errors/incidents
        notable = [e for e in events if e.event_type in ("error", "incident")][-5:]
        for event in notable:
            summary += f"- [{event.event_type}] {event.title or 'Untitled'} ({event.occurred_at.strftime('%Y-%m-%d %H:%M')})\n"

        return summary

    # =========================================================================
    # Insight Persistence
    # =========================================================================

    async def save_insight(
        self,
        project_id: str,
        insight: GeneratedInsight,
    ) -> str | None:
        """Save a generated insight to the database.

        Args:
            project_id: Project ID
            insight: Insight to save

        Returns:
            Insight ID if saved successfully, None otherwise
        """
        record = {
            "project_id": project_id,
            "insight_type": insight.insight_type.value,
            "severity": insight.severity.value,
            "title": insight.title,
            "description": insight.description,
            "recommendations": insight.recommendations,
            "event_ids": insight.event_ids,
            "status": "active",
        }

        result = await self.supabase.insert("correlation_insights", record)

        if result.get("error"):
            logger.error("Failed to save insight", error=result["error"])
            return None

        saved = result.get("data", [{}])[0]
        return saved.get("id")

    async def save_correlation(
        self,
        correlation: CorrelationResult,
    ) -> str | None:
        """Save a correlation to the database.

        Args:
            correlation: Correlation to save

        Returns:
            Correlation ID if saved successfully, None otherwise
        """
        record = {
            "source_event_id": correlation.source_event_id,
            "target_event_id": correlation.target_event_id,
            "correlation_type": correlation.correlation_type,
            "confidence": correlation.confidence,
            "correlation_method": "automatic",
        }

        result = await self.supabase.request(
            "/event_correlations",
            method="POST",
            body=record,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )

        if result.get("error"):
            # Ignore duplicate errors
            if "duplicate" not in str(result["error"]).lower():
                logger.error("Failed to save correlation", error=result["error"])
            return None

        saved = result.get("data", [{}])[0]
        return saved.get("id")


# =============================================================================
# Module Functions
# =============================================================================


_engine: CorrelationEngine | None = None


def get_correlation_engine() -> CorrelationEngine:
    """Get or create the global correlation engine instance."""
    global _engine
    if _engine is None:
        _engine = CorrelationEngine()
    return _engine
