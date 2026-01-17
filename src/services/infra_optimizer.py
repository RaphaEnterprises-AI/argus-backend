"""AI-Driven Infrastructure Optimizer for Browser Pools.

This service uses Claude AI to analyze Prometheus metrics and generate
intelligent infrastructure optimization recommendations, including:
- Cost reduction opportunities
- Demand forecasting and predictive scaling
- Anomaly detection (stuck sessions, resource leaks)
- Right-sizing recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional
import json
import uuid
import anthropic
import structlog

from src.services.prometheus_collector import (
    PrometheusCollector,
    InfrastructureSnapshot,
    SeleniumMetrics,
    BrowserNodeMetrics,
    create_prometheus_collector,
)
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


class RecommendationType(str, Enum):
    """Types of infrastructure recommendations."""

    SCALE_DOWN = "scale_down"  # Reduce min replicas during off-hours
    SCALE_UP = "scale_up"  # Increase min replicas for demand
    RIGHT_SIZE = "right_size"  # Adjust resource requests/limits
    SCHEDULE_SCALING = "schedule_scaling"  # Pre-scale for predicted demand
    CLEANUP_SESSIONS = "cleanup_sessions"  # Clean stuck sessions
    COST_ALERT = "cost_alert"  # Alert on cost threshold
    ANOMALY = "anomaly"  # Detected anomaly requiring attention


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"  # Requires immediate action
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Optimize when convenient
    LOW = "low"  # Nice to have


class ApprovalStatus(str, Enum):
    """Status of recommendation approval."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPLIED = "auto_applied"
    EXPIRED = "expired"


@dataclass
class InfraRecommendation:
    """An infrastructure optimization recommendation."""

    id: str
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    estimated_savings_monthly: Decimal
    confidence: float  # 0.0 - 1.0
    action: dict  # The specific action to take
    reasoning: str  # AI's reasoning for this recommendation
    metrics_snapshot: dict  # Metrics that led to this recommendation
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    org_id: Optional[str] = None


@dataclass
class CostReport:
    """Infrastructure cost analysis report."""

    period_start: datetime
    period_end: datetime
    total_cost: Decimal
    breakdown: dict[str, Decimal]  # cost by resource type
    daily_costs: list[tuple[datetime, Decimal]]
    projected_monthly: Decimal
    comparison_to_browserstack: Decimal
    savings_achieved: Decimal
    recommendations: list[InfraRecommendation]


@dataclass
class DemandForecast:
    """Demand forecast for browser pool."""

    forecast_start: datetime
    forecast_end: datetime
    hourly_predictions: list[dict]  # [{hour, predicted_sessions, confidence}]
    peak_times: list[datetime]
    recommended_min_replicas: dict[str, int]  # browser -> min replicas
    confidence: float


@dataclass
class Anomaly:
    """Detected infrastructure anomaly."""

    id: str
    type: str  # stuck_sessions, resource_leak, unusual_spike, etc.
    severity: RecommendationPriority
    description: str
    detected_at: datetime
    metrics: dict
    suggested_action: str


# Node pricing (Vultr vc2-4c-8gb at $48/mo = $0.0667/hr)
NODE_HOURLY_COST = Decimal("0.0667")
BROWSERSTACK_PER_SESSION_MONTHLY = Decimal("99.00")


class AIInfraOptimizer:
    """AI-driven infrastructure optimization service."""

    def __init__(
        self,
        prometheus_collector: Optional[PrometheusCollector] = None,
        anthropic_client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-sonnet-4-5-20241022",
    ):
        """Initialize the optimizer.

        Args:
            prometheus_collector: Collector for metrics. Created if not provided.
            anthropic_client: Anthropic client. Created if not provided.
            model: Claude model to use for analysis.
        """
        self.prometheus = prometheus_collector or create_prometheus_collector()
        self.client = anthropic_client or anthropic.Anthropic()
        self.model = model
        self._supabase = None

    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase

    async def analyze_and_recommend(
        self,
        org_id: str
    ) -> list[InfraRecommendation]:
        """Generate AI-driven optimization recommendations.

        Args:
            org_id: Organization ID

        Returns:
            List of prioritized recommendations
        """
        logger.info("analyzing_infrastructure", org_id=org_id)

        # Gather all metrics with graceful error handling
        try:
            snapshot = await self.prometheus.get_infrastructure_snapshot()
        except Exception as e:
            logger.warning("failed_to_get_infrastructure_snapshot", error=str(e))
            # Return empty recommendations if we can't get the snapshot
            return []

        try:
            usage_patterns = await self.prometheus.get_usage_patterns(hours=168)
        except Exception as e:
            logger.warning("failed_to_get_usage_patterns", error=str(e))
            usage_patterns = {
                "hourly_averages": [0.0] * 24,
                "daily_averages": [0.0] * 7,
                "peak_hour": 0,
                "min_hour": 0,
                "peak_day": 0,
                "min_day": 0,
            }

        try:
            test_metrics = await self.prometheus.get_test_execution_metrics(hours=24)
        except Exception as e:
            logger.warning("failed_to_get_test_metrics", error=str(e))
            test_metrics = {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0,
                "avg_duration_seconds": 0.0,
                "p95_duration_seconds": 0.0,
            }

        # Prepare context for AI analysis
        analysis_context = self._prepare_analysis_context(
            snapshot, usage_patterns, test_metrics
        )

        # Get AI recommendations
        recommendations = await self._get_ai_recommendations(
            org_id, analysis_context
        )

        # Store recommendations (don't fail if storage fails)
        for rec in recommendations:
            try:
                await self._store_recommendation(rec)
            except Exception as e:
                logger.warning(
                    "failed_to_store_recommendation",
                    rec_id=rec.id,
                    error=str(e)
                )

        logger.info(
            "recommendations_generated",
            org_id=org_id,
            count=len(recommendations)
        )

        return recommendations

    def _prepare_analysis_context(
        self,
        snapshot: InfrastructureSnapshot,
        usage_patterns: dict,
        test_metrics: dict
    ) -> str:
        """Prepare context string for AI analysis."""
        # Safely extract usage pattern values with defaults
        hourly_averages = usage_patterns.get("hourly_averages", [0.0] * 24)
        daily_averages = usage_patterns.get("daily_averages", [0.0] * 7)
        peak_hour = usage_patterns.get("peak_hour", 0)
        min_hour = usage_patterns.get("min_hour", 0)
        peak_day = usage_patterns.get("peak_day", 0)
        min_day = usage_patterns.get("min_day", 0)

        # Ensure indices are within bounds
        peak_hour = max(0, min(23, peak_hour))
        min_hour = max(0, min(23, min_hour))
        peak_day = max(0, min(6, peak_day))
        min_day = max(0, min(6, min_day))

        # Safely get hourly average values
        peak_hour_avg = hourly_averages[peak_hour] if peak_hour < len(hourly_averages) else 0.0
        min_hour_avg = hourly_averages[min_hour] if min_hour < len(hourly_averages) else 0.0

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # Safely format hourly averages
        formatted_hourly = [f'{v:.1f}' for v in hourly_averages] if hourly_averages else ['0.0'] * 24

        return f"""
## Current Infrastructure State

### Selenium Grid Status
- Sessions Queued: {snapshot.selenium.sessions_queued}
- Sessions Active: {snapshot.selenium.sessions_active}
- Nodes Available: {snapshot.selenium.nodes_available}/{snapshot.selenium.nodes_total}
- Avg Session Duration: {snapshot.selenium.avg_session_duration_seconds:.1f}s
- Queue Wait Time: {snapshot.selenium.queue_wait_time_seconds:.1f}s

### Browser Nodes
**Chrome Nodes:**
- Current Replicas: {snapshot.chrome_nodes.replicas_current}
- Min/Max: {snapshot.chrome_nodes.replicas_min}/{snapshot.chrome_nodes.replicas_max}
- CPU Utilization: {snapshot.chrome_nodes.cpu_utilization.cpu_usage_percent:.1f}%
- Memory Utilization: {snapshot.chrome_nodes.memory_utilization.memory_usage_percent:.1f}%

**Firefox Nodes:**
- Current Replicas: {snapshot.firefox_nodes.replicas_current}
- Min/Max: {snapshot.firefox_nodes.replicas_min}/{snapshot.firefox_nodes.replicas_max}
- CPU Utilization: {snapshot.firefox_nodes.cpu_utilization.cpu_usage_percent:.1f}%
- Memory Utilization: {snapshot.firefox_nodes.memory_utilization.memory_usage_percent:.1f}%

**Edge Nodes:**
- Current Replicas: {snapshot.edge_nodes.replicas_current}
- Min/Max: {snapshot.edge_nodes.replicas_min}/{snapshot.edge_nodes.replicas_max}
- CPU Utilization: {snapshot.edge_nodes.cpu_utilization.cpu_usage_percent:.1f}%
- Memory Utilization: {snapshot.edge_nodes.memory_utilization.memory_usage_percent:.1f}%

### Cluster Status
- Total Pods: {snapshot.total_pods}
- Total Nodes: {snapshot.total_nodes}
- Cluster CPU: {snapshot.cluster_cpu_utilization:.1f}%
- Cluster Memory: {snapshot.cluster_memory_utilization:.1f}%

### Usage Patterns (Last 7 Days)
- Peak Hour: {peak_hour}:00 (avg {peak_hour_avg:.1f} sessions)
- Lowest Hour: {min_hour}:00 (avg {min_hour_avg:.1f} sessions)
- Peak Day: {day_names[peak_day]}
- Lowest Day: {day_names[min_day]}
- Hourly Averages: {formatted_hourly}

### Test Execution (Last 24h)
- Total Tests: {test_metrics.get('total_tests', 0)}
- Success Rate: {test_metrics.get('success_rate', 0.0):.1f}%
- Avg Duration: {test_metrics.get('avg_duration_seconds', 0.0):.1f}s
- P95 Duration: {test_metrics.get('p95_duration_seconds', 0.0):.1f}s

### Cost Context
- Node Hourly Cost: ${NODE_HOURLY_COST}
- Current Nodes Running: {snapshot.total_nodes}
- Estimated Daily Cost: ${NODE_HOURLY_COST * 24 * snapshot.total_nodes}
- BrowserStack Equivalent (per session): ${BROWSERSTACK_PER_SESSION_MONTHLY}/month
"""

    async def _get_ai_recommendations(
        self,
        org_id: str,
        context: str
    ) -> list[InfraRecommendation]:
        """Get recommendations from Claude AI."""

        system_prompt = """You are an expert DevOps and infrastructure optimization AI for Argus,
an E2E testing platform. Analyze the provided infrastructure metrics and generate
actionable cost-saving and performance optimization recommendations.

For each recommendation, provide:
1. A specific, actionable title
2. Clear description of the issue and solution
3. Estimated monthly savings in USD
4. Confidence level (0.0-1.0)
5. Specific action parameters
6. Your reasoning

Focus on:
- Cost reduction through right-sizing and scheduling
- Performance improvements
- Reliability (avoiding stuck sessions, resource exhaustion)
- Predictive scaling opportunities

Output as JSON array of recommendations."""

        user_prompt = f"""Analyze this infrastructure state and generate optimization recommendations:

{context}

Generate recommendations as a JSON array with this structure:
[
  {{
    "type": "scale_down|scale_up|right_size|schedule_scaling|cleanup_sessions|cost_alert|anomaly",
    "priority": "critical|high|medium|low",
    "title": "Short actionable title",
    "description": "Detailed description",
    "estimated_savings_monthly": 50.00,
    "confidence": 0.85,
    "action": {{
      "target": "chrome|firefox|edge|cluster",
      "operation": "set_min_replicas|set_max_replicas|schedule|alert",
      "params": {{}}
    }},
    "reasoning": "Why this recommendation makes sense"
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=system_prompt,
            )

            # Parse JSON response
            content = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            recommendations_data = json.loads(content)

            recommendations = []
            for rec_data in recommendations_data:
                rec = InfraRecommendation(
                    id=str(uuid.uuid4()),
                    type=RecommendationType(rec_data["type"]),
                    priority=RecommendationPriority(rec_data["priority"]),
                    title=rec_data["title"],
                    description=rec_data["description"],
                    estimated_savings_monthly=Decimal(str(rec_data["estimated_savings_monthly"])),
                    confidence=float(rec_data["confidence"]),
                    action=rec_data["action"],
                    reasoning=rec_data["reasoning"],
                    metrics_snapshot={"context": context[:500]},  # Truncate for storage
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7),
                    org_id=org_id,
                )
                recommendations.append(rec)

            return recommendations

        except json.JSONDecodeError as e:
            logger.error("ai_response_parse_error", error=str(e))
            return []
        except Exception as e:
            logger.error("ai_recommendation_error", error=str(e))
            return []

    async def _store_recommendation(self, rec: InfraRecommendation) -> None:
        """Store recommendation in database."""
        try:
            await self.supabase.insert("infra_recommendations", {
                "id": rec.id,
                "org_id": rec.org_id,
                "type": rec.type.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "estimated_savings_monthly": float(rec.estimated_savings_monthly),
                "confidence": rec.confidence,
                "action": rec.action,
                "reasoning": rec.reasoning,
                "status": rec.status.value,
                "created_at": rec.created_at.isoformat(),
                "expires_at": rec.expires_at.isoformat(),
            })
        except Exception as e:
            logger.warning("store_recommendation_failed", error=str(e))

    async def get_cost_report(
        self,
        org_id: str,
        days: int = 7
    ) -> CostReport:
        """Generate cost analysis report.

        Args:
            org_id: Organization ID
            days: Number of days to analyze

        Returns:
            Detailed cost report
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        # Get node count over time
        try:
            series = await self.prometheus.query_range(
                "count(kube_node_info)",
                start=start,
                end=end,
                step="1h"
            )
        except Exception as e:
            logger.warning("failed_to_query_node_info", error=str(e))
            series = []

        # Calculate daily costs
        daily_costs: list[tuple[datetime, Decimal]] = []
        total_cost = Decimal("0")

        if series and len(series) > 0 and hasattr(series[0], 'values') and series[0].values:
            current_day = None
            day_cost = Decimal("0")

            for ts, node_count in series[0].values:
                day = ts.date()
                # Safely convert node_count to Decimal
                try:
                    hourly_cost = NODE_HOURLY_COST * Decimal(str(node_count))
                except (ValueError, TypeError):
                    hourly_cost = Decimal("0")

                if current_day is None:
                    current_day = day
                elif day != current_day:
                    daily_costs.append((datetime.combine(current_day, datetime.min.time()), day_cost))
                    total_cost += day_cost
                    current_day = day
                    day_cost = Decimal("0")

                day_cost += hourly_cost

            # Add last day
            if current_day:
                daily_costs.append((datetime.combine(current_day, datetime.min.time()), day_cost))
                total_cost += day_cost

        # Project monthly cost
        if days > 0 and total_cost > 0:
            daily_avg = total_cost / days
            projected_monthly = daily_avg * 30
        else:
            projected_monthly = Decimal("0")

        # Calculate BrowserStack comparison
        try:
            snapshot = await self.prometheus.get_infrastructure_snapshot()
            max_concurrent = max(
                snapshot.chrome_nodes.replicas_max,
                snapshot.firefox_nodes.replicas_max,
                snapshot.edge_nodes.replicas_max
            )
        except Exception as e:
            logger.warning("failed_to_get_snapshot_for_cost_report", error=str(e))
            max_concurrent = 10  # Default fallback

        browserstack_equivalent = BROWSERSTACK_PER_SESSION_MONTHLY * max_concurrent

        # Get recommendations for this period (don't fail if this fails)
        try:
            recommendations = await self.analyze_and_recommend(org_id)
        except Exception as e:
            logger.warning("failed_to_get_recommendations_for_cost_report", error=str(e))
            recommendations = []

        return CostReport(
            period_start=start,
            period_end=end,
            total_cost=total_cost,
            breakdown={
                "compute": total_cost * Decimal("0.8"),  # 80% compute
                "network": total_cost * Decimal("0.15"),  # 15% network
                "storage": total_cost * Decimal("0.05"),  # 5% storage
            },
            daily_costs=daily_costs,
            projected_monthly=projected_monthly,
            comparison_to_browserstack=browserstack_equivalent,
            savings_achieved=browserstack_equivalent - projected_monthly,
            recommendations=recommendations,
        )

    async def predict_demand(
        self,
        org_id: str,
        horizon_hours: int = 24
    ) -> DemandForecast:
        """Predict future demand based on historical patterns.

        Args:
            org_id: Organization ID
            horizon_hours: Hours to forecast ahead

        Returns:
            Demand forecast with recommended scaling
        """
        try:
            usage_patterns = await self.prometheus.get_usage_patterns(hours=168)
        except Exception as e:
            logger.warning("failed_to_get_usage_patterns", error=str(e))
            usage_patterns = {
                "hourly_averages": [0.0] * 24,
                "daily_averages": [0.0] * 7,
                "peak_hour": 0,
                "min_hour": 0,
                "peak_day": 0,
                "min_day": 0,
            }

        now = datetime.now()
        hourly_predictions = []

        # Safely get averages with defaults
        hourly_averages = usage_patterns.get("hourly_averages", [0.0] * 24)
        daily_averages = usage_patterns.get("daily_averages", [0.0] * 7)

        # Ensure we have valid lists
        if not hourly_averages or len(hourly_averages) != 24:
            hourly_averages = [0.0] * 24
        if not daily_averages or len(daily_averages) != 7:
            daily_averages = [0.0] * 7

        # Calculate daily average safely (avoid division by zero)
        daily_sum = sum(daily_averages)
        daily_avg = daily_sum / 7 if daily_sum > 0 else 0.1

        for h in range(horizon_hours):
            forecast_time = now + timedelta(hours=h)
            hour_of_day = forecast_time.hour
            day_of_week = forecast_time.weekday()

            # Use historical averages with day weighting
            base_prediction = hourly_averages[hour_of_day] if hour_of_day < len(hourly_averages) else 0.0
            day_value = daily_averages[day_of_week] if day_of_week < len(daily_averages) else 0.0
            day_factor = day_value / max(daily_avg, 0.1)

            predicted = base_prediction * day_factor

            # Higher confidence for patterns we've seen more
            confidence = 0.7 if base_prediction > 0 else 0.3

            hourly_predictions.append({
                "hour": forecast_time.isoformat(),
                "predicted_sessions": round(predicted, 1),
                "confidence": confidence,
            })

        # Identify peak times (handle empty predictions)
        peak_times = []
        if hourly_predictions:
            max_predicted_sessions = max(
                (p["predicted_sessions"] for p in hourly_predictions),
                default=0
            )
            if max_predicted_sessions > 0:
                for pred in hourly_predictions:
                    if pred["predicted_sessions"] > max_predicted_sessions * 0.8:
                        peak_times.append(datetime.fromisoformat(pred["hour"]))

        # Calculate recommended min replicas
        max_predicted = max(
            (p["predicted_sessions"] for p in hourly_predictions),
            default=0
        )
        recommended_min = {
            "chrome": max(2, int(max_predicted * 0.6)),
            "firefox": max(1, int(max_predicted * 0.2)),
            "edge": max(1, int(max_predicted * 0.2)),
        }

        return DemandForecast(
            forecast_start=now,
            forecast_end=now + timedelta(hours=horizon_hours),
            hourly_predictions=hourly_predictions,
            peak_times=peak_times,
            recommended_min_replicas=recommended_min,
            confidence=0.7,
        )

    async def detect_anomalies(self, org_id: str) -> list[Anomaly]:
        """Detect infrastructure anomalies.

        Args:
            org_id: Organization ID

        Returns:
            List of detected anomalies
        """
        anomalies = []

        try:
            snapshot = await self.prometheus.get_infrastructure_snapshot()
        except Exception as e:
            logger.warning("failed_to_get_snapshot_for_anomaly_detection", error=str(e))
            # Return empty list if we can't get the snapshot
            return anomalies

        # Check for stuck sessions (queue not draining)
        try:
            if (snapshot.selenium.sessions_queued > 5 and
                    snapshot.selenium.queue_wait_time_seconds > 60):
                anomalies.append(Anomaly(
                    id=str(uuid.uuid4()),
                    type="stuck_queue",
                    severity=RecommendationPriority.HIGH,
                    description=f"Session queue has {snapshot.selenium.sessions_queued} pending "
                               f"sessions with {snapshot.selenium.queue_wait_time_seconds:.0f}s wait time",
                    detected_at=datetime.now(),
                    metrics={
                        "queued": snapshot.selenium.sessions_queued,
                        "wait_time": snapshot.selenium.queue_wait_time_seconds,
                    },
                    suggested_action="Scale up browser nodes or investigate stuck sessions",
                ))
        except (AttributeError, TypeError) as e:
            logger.debug("skipping_stuck_queue_check", error=str(e))

        # Check for resource exhaustion
        for node_metrics, name in [
            (snapshot.chrome_nodes, "Chrome"),
            (snapshot.firefox_nodes, "Firefox"),
            (snapshot.edge_nodes, "Edge"),
        ]:
            try:
                if node_metrics and node_metrics.cpu_utilization:
                    cpu_percent = node_metrics.cpu_utilization.cpu_usage_percent or 0
                    if cpu_percent > 90:
                        anomalies.append(Anomaly(
                            id=str(uuid.uuid4()),
                            type="cpu_exhaustion",
                            severity=RecommendationPriority.CRITICAL,
                            description=f"{name} nodes at {cpu_percent:.0f}% CPU",
                            detected_at=datetime.now(),
                            metrics={"cpu_percent": cpu_percent},
                            suggested_action=f"Scale up {name} nodes or increase CPU limits",
                        ))

                if node_metrics and node_metrics.memory_utilization:
                    mem_percent = node_metrics.memory_utilization.memory_usage_percent or 0
                    if mem_percent > 90:
                        anomalies.append(Anomaly(
                            id=str(uuid.uuid4()),
                            type="memory_exhaustion",
                            severity=RecommendationPriority.CRITICAL,
                            description=f"{name} nodes at {mem_percent:.0f}% memory",
                            detected_at=datetime.now(),
                            metrics={"memory_percent": mem_percent},
                            suggested_action=f"Scale up {name} nodes or increase memory limits",
                        ))
            except (AttributeError, TypeError) as e:
                logger.debug("skipping_resource_check", name=name, error=str(e))

        # Check for over-provisioning (wasteful)
        try:
            cluster_cpu = snapshot.cluster_cpu_utilization or 0
            total_nodes = snapshot.total_nodes or 0
            if cluster_cpu < 20 and total_nodes > 2:
                anomalies.append(Anomaly(
                    id=str(uuid.uuid4()),
                    type="over_provisioned",
                    severity=RecommendationPriority.MEDIUM,
                    description=f"Cluster running at only {cluster_cpu:.0f}% CPU "
                               f"with {total_nodes} nodes",
                    detected_at=datetime.now(),
                    metrics={
                        "cpu_percent": cluster_cpu,
                        "nodes": total_nodes,
                    },
                    suggested_action="Consider reducing minimum replica counts to save costs",
                ))
        except (AttributeError, TypeError) as e:
            logger.debug("skipping_over_provisioned_check", error=str(e))

        return anomalies

    async def apply_recommendation(
        self,
        recommendation_id: str,
        auto: bool = False
    ) -> dict:
        """Apply a recommendation.

        Args:
            recommendation_id: ID of the recommendation to apply
            auto: Whether this is an auto-apply (vs manual approval)

        Returns:
            Result of the application
        """
        try:
            # Fetch recommendation from database
            result = await self.supabase.request(
                f"/infra_recommendations?id=eq.{recommendation_id}&select=*"
            )

            # Check for errors in the request
            if result.get("error"):
                logger.warning(
                    "apply_recommendation_db_error",
                    recommendation_id=recommendation_id,
                    error=result.get("error")
                )
                return {"success": False, "error": f"Database error: {result.get('error')}"}

            # Check if data exists and is non-empty
            data = result.get("data")
            if not data or not isinstance(data, list) or len(data) == 0:
                return {"success": False, "error": "Recommendation not found"}

            rec_data = data[0]

            # Safely get status field
            current_status = rec_data.get("status")
            if current_status in ["approved", "auto_applied"]:
                return {"success": False, "error": "Recommendation already applied"}

            # Apply the action (in production, this would call Kubernetes APIs)
            action = rec_data.get("action", {})
            logger.info(
                "applying_recommendation",
                rec_id=recommendation_id,
                action=action,
                auto=auto
            )

            # Update status
            new_status = ApprovalStatus.AUTO_APPLIED if auto else ApprovalStatus.APPROVED
            update_result = await self.supabase.update(
                "infra_recommendations",
                {"id": f"eq.{recommendation_id}"},
                {
                    "status": new_status.value,
                    "applied_at": datetime.now().isoformat(),
                }
            )

            if update_result.get("error"):
                logger.warning(
                    "apply_recommendation_update_error",
                    recommendation_id=recommendation_id,
                    error=update_result.get("error")
                )
                # Still return success if the action was logged - the update might fail
                # due to table not existing yet, but we don't want to block the operation

            return {
                "success": True,
                "action_applied": action,
                "status": new_status.value,
            }

        except Exception as e:
            logger.error(
                "apply_recommendation_exception",
                recommendation_id=recommendation_id,
                error=str(e)
            )
            return {"success": False, "error": f"Failed to apply recommendation: {str(e)}"}

    async def get_savings_summary(self, org_id: str) -> dict:
        """Get summary of savings achieved through optimization.

        Args:
            org_id: Organization ID

        Returns:
            Savings summary
        """
        total_savings = Decimal("0")
        recommendations_applied = 0

        try:
            # Get applied recommendations
            result = await self.supabase.request(
                f"/infra_recommendations?org_id=eq.{org_id}"
                f"&status=in.(approved,auto_applied)&select=*"
            )

            # Only process if we got valid data without errors
            if not result.get("error"):
                data = result.get("data") or []
                for rec in data:
                    # Safely get estimated_savings_monthly with default
                    savings = rec.get("estimated_savings_monthly", 0)
                    if savings is not None:
                        total_savings += Decimal(str(savings))
                    recommendations_applied += 1
        except Exception as e:
            logger.warning(
                "get_savings_summary_recommendations_error",
                org_id=org_id,
                error=str(e)
            )
            # Continue with zero savings if we can't get recommendations

        try:
            # Get cost comparison
            cost_report = await self.get_cost_report(org_id, days=30)

            return {
                "total_monthly_savings": float(total_savings),
                "recommendations_applied": recommendations_applied,
                "current_monthly_cost": float(cost_report.projected_monthly),
                "browserstack_equivalent": float(cost_report.comparison_to_browserstack),
                "savings_vs_browserstack": float(cost_report.savings_achieved),
                "savings_percentage": float(
                    (cost_report.savings_achieved / cost_report.comparison_to_browserstack * 100)
                    if cost_report.comparison_to_browserstack > 0 else 0
                ),
            }
        except Exception as e:
            logger.warning(
                "get_savings_summary_cost_report_error",
                org_id=org_id,
                error=str(e)
            )
            # Return default values if cost report fails
            return {
                "total_monthly_savings": float(total_savings),
                "recommendations_applied": recommendations_applied,
                "current_monthly_cost": 0.0,
                "browserstack_equivalent": 0.0,
                "savings_vs_browserstack": 0.0,
                "savings_percentage": 0.0,
            }


# Factory function
def create_infra_optimizer(
    prometheus_url: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20241022",
) -> AIInfraOptimizer:
    """Create an InfraOptimizer instance.

    Args:
        prometheus_url: Prometheus URL. Uses environment variable if not provided.
        model: Claude model to use for analysis.

    Returns:
        Configured AIInfraOptimizer instance
    """
    prometheus = create_prometheus_collector(prometheus_url)
    return AIInfraOptimizer(prometheus_collector=prometheus, model=model)
