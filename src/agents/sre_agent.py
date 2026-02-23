"""
Unified SRE (Site Reliability Engineering) Agent.

Combines incident correlation, automated remediation, and infrastructure
optimization into a single cohesive agent following 2026 best practices.

RAP-253: Unified SRE Agent Implementation

Key Capabilities:
- Incident detection and correlation across services
- AI-powered root cause analysis
- Automated runbook execution with approval workflow
- MTTR tracking and optimization
- Integration with PagerDuty, Sentry, and monitoring systems

References:
- Azure SRE Agent (2025): Autonomous incident response
- Datadog Bits AI (2025): Intelligent alerting and remediation
- Google SRE Book: Principles of site reliability
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .base import (
    AgentCapability,
    AgentResult,
    AICapability,
    AIResponse,
    BaseAgent,
)
from ..core.model_router import TaskType

logger = structlog.get_logger()


class IncidentSeverity(str, Enum):
    """Incident severity levels following industry standards."""
    SEV1 = "sev1"  # Critical - immediate response required
    SEV2 = "sev2"  # High - response within 15 minutes
    SEV3 = "sev3"  # Medium - response within 1 hour
    SEV4 = "sev4"  # Low - response within 24 hours
    SEV5 = "sev5"  # Informational - no immediate action


class RemediationStatus(str, Enum):
    """Status of automated remediation."""
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CorrelatedIncident:
    """An incident with correlated signals and context."""
    id: str
    title: str
    severity: IncidentSeverity
    detected_at: datetime

    # Correlated signals
    primary_signal: dict[str, Any]  # Main alert/error
    related_signals: list[dict[str, Any]] = field(default_factory=list)

    # Analysis results
    root_cause_hypothesis: str | None = None
    root_cause_confidence: float = 0.0
    affected_services: list[str] = field(default_factory=list)
    blast_radius: str = "unknown"  # "isolated", "partial", "widespread"

    # Context
    recent_changes: list[dict[str, Any]] = field(default_factory=list)
    similar_past_incidents: list[str] = field(default_factory=list)

    # Resolution
    suggested_actions: list[dict[str, Any]] = field(default_factory=list)
    remediation_status: RemediationStatus = RemediationStatus.PENDING
    resolution_time: datetime | None = None
    resolution_notes: str | None = None


@dataclass
class Runbook:
    """An automated remediation runbook."""
    id: str
    name: str
    description: str
    trigger_conditions: list[str]

    # Steps
    steps: list[dict[str, Any]]  # {"action": "...", "params": {...}, "rollback": {...}}
    requires_approval: bool = True
    max_auto_execute_severity: IncidentSeverity = IncidentSeverity.SEV4

    # Metadata
    success_rate: float = 0.0
    avg_execution_time_seconds: int = 0
    last_executed: datetime | None = None
    execution_count: int = 0


@dataclass
class SREMetrics:
    """SRE performance metrics."""
    # MTTR (Mean Time To Resolve)
    mttr_minutes: float = 0.0
    mttr_trend: str = "stable"  # "improving", "stable", "degrading"

    # Incident metrics
    incidents_last_24h: int = 0
    incidents_last_7d: int = 0
    auto_resolved_percentage: float = 0.0

    # Availability
    uptime_percentage: float = 99.9
    error_budget_remaining: float = 100.0


class SREAgent(BaseAgent):
    """
    Unified Site Reliability Engineering Agent.

    Provides autonomous incident detection, correlation, analysis, and
    remediation with human-in-the-loop approval for critical actions.

    Usage:
        ```python
        sre = SREAgent(
            org_id="org_123",
            pagerduty_api_key="...",
            enable_auto_remediation=True,
        )

        # Analyze an incident
        analysis = await sre.analyze_incident(
            alert_data={
                "service": "api-gateway",
                "error": "Connection timeout to database",
                "count": 150,
                "first_seen": "2026-01-31T10:00:00Z",
            }
        )

        # Execute remediation with approval
        result = await sre.execute_remediation(
            incident_id=analysis.incident.id,
            runbook_id="rb_restart_db_connection_pool",
            approved_by="oncall@example.com",
        )
        ```
    """

    # Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.CODE_ANALYSIS,  # For analyzing error traces
        "incident_correlation",
        "root_cause_analysis",
        "automated_remediation",
        "mttr_optimization",
        "runbook_execution",
    ]

    DEFAULT_TASK_TYPE = TaskType.ROOT_CAUSE_ANALYSIS

    def __init__(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        pagerduty_api_key: str | None = None,
        sentry_dsn: str | None = None,
        enable_auto_remediation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.org_id = org_id
        self.project_id = project_id
        self.pagerduty_api_key = pagerduty_api_key
        self.sentry_dsn = sentry_dsn
        self.enable_auto_remediation = enable_auto_remediation

        self.log = logger.bind(
            agent="SREAgent",
            org_id=org_id,
        )

        # Caches
        self._runbooks: dict[str, Runbook] = {}
        self._active_incidents: dict[str, CorrelatedIncident] = {}
        self._metrics = SREMetrics()

        # Initialize integrations lazily
        self._pagerduty_client = None
        self._incident_correlator = None
        self._infra_optimizer = None

    def _get_system_prompt(self) -> str:
        return """You are an expert Site Reliability Engineer (SRE) agent with deep knowledge of:
- Distributed systems and failure modes
- Observability (metrics, logs, traces)
- Incident response best practices
- Root cause analysis techniques
- Automated remediation patterns

Your responsibilities:
1. Correlate signals across services to identify incidents
2. Perform AI-powered root cause analysis
3. Suggest and execute remediation actions
4. Track MTTR and optimize incident response
5. Learn from past incidents to prevent recurrence

Always prioritize:
- Service availability and user impact
- Safe, reversible remediation actions
- Clear communication with oncall engineers
- Thorough documentation of findings and actions

When analyzing incidents:
- Consider recent deployments and changes
- Look for correlated failures across services
- Identify the blast radius
- Estimate time to resolve based on similar past incidents"""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the SRE agent's main task based on provided context."""
        task_type = kwargs.get("task_type", "analyze")

        if task_type == "analyze":
            alert_data = kwargs.get("alert_data", {})
            result = await self.analyze_incident(alert_data)
            return AgentResult(
                success=True,
                data=result,
            )
        elif task_type == "remediate":
            incident_id = kwargs.get("incident_id")
            runbook_id = kwargs.get("runbook_id")
            approved_by = kwargs.get("approved_by")
            result = await self.execute_remediation(
                incident_id=incident_id,
                runbook_id=runbook_id,
                approved_by=approved_by,
            )
            return AgentResult(success=result.get("success", False), data=result)
        elif task_type == "correlate":
            signals = kwargs.get("signals", [])
            result = await self.correlate_signals(signals)
            return AgentResult(success=True, data=result)
        else:
            return AgentResult(
                success=False,
                error=f"Unknown task type: {task_type}",
            )

    # =========================================================================
    # Incident Detection and Correlation
    # =========================================================================

    async def analyze_incident(
        self,
        alert_data: dict[str, Any],
        include_similar: bool = True,
        include_changes: bool = True,
    ) -> dict[str, Any]:
        """Analyze an alert and create a correlated incident.

        Args:
            alert_data: Raw alert data from monitoring system
            include_similar: Include similar past incidents
            include_changes: Include recent code/infra changes

        Returns:
            Analysis result with incident, root cause, and suggested actions
        """
        self.log.info("Analyzing incident", alert_data=alert_data)

        # Step 1: Correlate with other signals
        related_signals = await self._fetch_related_signals(alert_data)

        # Step 2: Determine severity
        severity = self._assess_severity(alert_data, related_signals)

        # Step 3: Create incident
        incident = CorrelatedIncident(
            id=f"inc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{alert_data.get('service', 'unknown')[:10]}",
            title=self._generate_incident_title(alert_data),
            severity=severity,
            detected_at=datetime.utcnow(),
            primary_signal=alert_data,
            related_signals=related_signals,
        )

        # Step 4: Fetch context
        if include_changes:
            incident.recent_changes = await self._fetch_recent_changes(
                services=self._extract_affected_services(alert_data, related_signals)
            )

        if include_similar:
            incident.similar_past_incidents = await self._find_similar_incidents(alert_data)

        # Step 5: AI-powered root cause analysis
        rca_result = await self._perform_root_cause_analysis(incident)
        incident.root_cause_hypothesis = rca_result.get("hypothesis")
        incident.root_cause_confidence = rca_result.get("confidence", 0.0)
        incident.affected_services = rca_result.get("affected_services", [])
        incident.blast_radius = rca_result.get("blast_radius", "unknown")

        # Step 6: Generate suggested actions
        incident.suggested_actions = await self._generate_remediation_suggestions(incident)

        # Store incident
        self._active_incidents[incident.id] = incident

        # Step 7: Auto-remediation for low severity if enabled
        if (
            self.enable_auto_remediation
            and severity in [IncidentSeverity.SEV4, IncidentSeverity.SEV5]
            and incident.suggested_actions
        ):
            auto_action = incident.suggested_actions[0]
            if auto_action.get("auto_executable", False):
                await self._attempt_auto_remediation(incident, auto_action)

        return {
            "incident": incident,
            "analysis": {
                "root_cause": incident.root_cause_hypothesis,
                "confidence": incident.root_cause_confidence,
                "affected_services": incident.affected_services,
                "blast_radius": incident.blast_radius,
                "similar_incidents": incident.similar_past_incidents,
            },
            "suggested_actions": incident.suggested_actions,
        }

    async def correlate_signals(
        self,
        signals: list[dict[str, Any]],
        time_window_minutes: int = 15,
    ) -> list[CorrelatedIncident]:
        """Correlate multiple signals into incidents.

        Groups related alerts and errors into cohesive incidents.

        Args:
            signals: List of alerts/errors from various sources
            time_window_minutes: Time window for correlation

        Returns:
            List of correlated incidents
        """
        # Use AI to group signals
        correlation_prompt = f"""Analyze these signals and group them into incidents:

Signals:
{self._format_signals_for_prompt(signals)}

Time window: {time_window_minutes} minutes

For each incident group, provide:
1. A title describing the incident
2. The primary signal (most indicative)
3. Related signals
4. Severity assessment (sev1-sev5)
5. Affected services

Return as JSON:
{{
    "incidents": [
        {{
            "title": "...",
            "primary_signal_index": 0,
            "related_signal_indices": [1, 2],
            "severity": "sev2",
            "affected_services": ["api", "database"]
        }}
    ]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": correlation_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
        )

        result = self._parse_json_response(response.content, {"incidents": []})
        incidents = []

        for inc_data in result.get("incidents", []):
            primary_idx = inc_data.get("primary_signal_index", 0)
            primary_signal = signals[primary_idx] if primary_idx < len(signals) else signals[0]

            related = []
            for idx in inc_data.get("related_signal_indices", []):
                if idx < len(signals):
                    related.append(signals[idx])

            incident = CorrelatedIncident(
                id=f"inc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{len(incidents)}",
                title=inc_data.get("title", "Unknown Incident"),
                severity=IncidentSeverity(inc_data.get("severity", "sev3")),
                detected_at=datetime.utcnow(),
                primary_signal=primary_signal,
                related_signals=related,
                affected_services=inc_data.get("affected_services", []),
            )
            incidents.append(incident)
            self._active_incidents[incident.id] = incident

        return incidents

    # =========================================================================
    # Root Cause Analysis
    # =========================================================================

    async def _perform_root_cause_analysis(
        self,
        incident: CorrelatedIncident,
    ) -> dict[str, Any]:
        """Perform AI-powered root cause analysis.

        Uses the Reflexion pattern for iterative improvement of the analysis.
        """
        # Build context for analysis
        context = {
            "primary_signal": incident.primary_signal,
            "related_signals": incident.related_signals[:5],  # Limit for context
            "recent_changes": incident.recent_changes[:5],
            "similar_incidents": incident.similar_past_incidents[:3],
        }

        analysis_prompt = f"""Perform root cause analysis for this incident:

Title: {incident.title}
Severity: {incident.severity.value}

Primary Signal:
{self._format_dict_for_prompt(incident.primary_signal)}

Related Signals ({len(incident.related_signals)} total):
{self._format_signals_for_prompt(incident.related_signals[:5])}

Recent Changes:
{self._format_signals_for_prompt(incident.recent_changes[:5])}

Similar Past Incidents: {incident.similar_past_incidents[:3]}

Analyze and provide:
1. Root cause hypothesis (be specific)
2. Confidence level (0.0-1.0)
3. All affected services
4. Blast radius (isolated/partial/widespread)
5. Evidence supporting the hypothesis

Return as JSON:
{{
    "hypothesis": "Detailed root cause explanation",
    "confidence": 0.85,
    "affected_services": ["service1", "service2"],
    "blast_radius": "partial",
    "evidence": ["evidence1", "evidence2"],
    "alternative_hypotheses": ["alt1", "alt2"]
}}"""

        # Use Reflexion for higher quality analysis
        result = await self._call_ai_with_reflexion(
            messages=[{"role": "user", "content": analysis_prompt}],
            task_type=TaskType.ROOT_CAUSE_ANALYSIS,
            required_capabilities=[AICapability.REASONING],
            max_reflexion_rounds=2,
            quality_threshold=0.75,
        )

        return self._parse_json_response(result.response.content, {
            "hypothesis": "Unable to determine root cause",
            "confidence": 0.0,
            "affected_services": [],
            "blast_radius": "unknown",
        })

    # =========================================================================
    # Automated Remediation
    # =========================================================================

    async def execute_remediation(
        self,
        incident_id: str,
        runbook_id: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """Execute a remediation runbook for an incident.

        Args:
            incident_id: ID of the incident to remediate
            runbook_id: ID of the runbook to execute
            approved_by: Email of approver (required for non-auto runbooks)

        Returns:
            Execution result with status and details
        """
        incident = self._active_incidents.get(incident_id)
        if not incident:
            return {"success": False, "error": f"Incident not found: {incident_id}"}

        runbook = self._runbooks.get(runbook_id)
        if not runbook:
            # Try to load from database
            runbook = await self._load_runbook(runbook_id)
            if not runbook:
                return {"success": False, "error": f"Runbook not found: {runbook_id}"}

        # Check approval requirement
        if runbook.requires_approval and not approved_by:
            incident.remediation_status = RemediationStatus.AWAITING_APPROVAL
            return {
                "success": False,
                "status": "awaiting_approval",
                "message": "This runbook requires approval before execution",
                "incident_id": incident_id,
                "runbook_id": runbook_id,
            }

        # Execute runbook
        self.log.info(
            "Executing remediation runbook",
            incident_id=incident_id,
            runbook_id=runbook_id,
            approved_by=approved_by,
        )

        incident.remediation_status = RemediationStatus.IN_PROGRESS
        execution_start = datetime.utcnow()

        try:
            results = []
            for i, step in enumerate(runbook.steps):
                step_result = await self._execute_runbook_step(
                    step=step,
                    incident=incident,
                    step_number=i + 1,
                )
                results.append(step_result)

                if not step_result.get("success", False):
                    # Attempt rollback
                    await self._rollback_runbook(runbook, results)
                    incident.remediation_status = RemediationStatus.ROLLED_BACK
                    return {
                        "success": False,
                        "status": "rolled_back",
                        "failed_step": i + 1,
                        "error": step_result.get("error"),
                        "results": results,
                    }

            # Success
            incident.remediation_status = RemediationStatus.COMPLETED
            incident.resolution_time = datetime.utcnow()
            execution_time = (incident.resolution_time - execution_start).total_seconds()

            # Update runbook stats
            runbook.execution_count += 1
            runbook.last_executed = datetime.utcnow()
            runbook.success_rate = (
                (runbook.success_rate * (runbook.execution_count - 1) + 1.0)
                / runbook.execution_count
            )

            # Store in memory for learning
            await self._store_remediation_outcome(incident, runbook, True)

            # Persist SRE resolution metrics to agent_evaluations
            await self._persist_sre_metrics(
                incident=incident,
                success=True,
                execution_time_seconds=execution_time,
                runbook_id=runbook_id,
            )

            return {
                "success": True,
                "status": "completed",
                "execution_time_seconds": execution_time,
                "results": results,
            }

        except Exception as e:
            incident.remediation_status = RemediationStatus.FAILED
            self.log.error(
                "Remediation failed",
                incident_id=incident_id,
                error=str(e),
            )

            # Persist failure metrics
            execution_time = (datetime.utcnow() - execution_start).total_seconds()
            await self._persist_sre_metrics(
                incident=incident,
                success=False,
                execution_time_seconds=execution_time,
                runbook_id=runbook_id,
                error=str(e),
            )

            return {"success": False, "status": "failed", "error": str(e)}

    async def _execute_runbook_step(
        self,
        step: dict[str, Any],
        incident: CorrelatedIncident,
        step_number: int,
    ) -> dict[str, Any]:
        """Execute a single runbook step."""
        action = step.get("action")
        params = step.get("params", {})

        self.log.debug(
            "Executing runbook step",
            step=step_number,
            action=action,
        )

        # Supported actions
        if action == "restart_service":
            return await self._action_restart_service(params)
        elif action == "scale_up":
            return await self._action_scale_up(params)
        elif action == "clear_cache":
            return await self._action_clear_cache(params)
        elif action == "rollback_deployment":
            return await self._action_rollback_deployment(params)
        elif action == "run_health_check":
            return await self._action_health_check(params)
        elif action == "notify_team":
            return await self._action_notify_team(params, incident)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    async def _rollback_runbook(
        self,
        runbook: Runbook,
        executed_results: list[dict[str, Any]],
    ) -> None:
        """Rollback executed runbook steps in reverse order."""
        for i in range(len(executed_results) - 1, -1, -1):
            step = runbook.steps[i]
            rollback = step.get("rollback")
            if rollback:
                try:
                    await self._execute_runbook_step(
                        step=rollback,
                        incident=CorrelatedIncident(
                            id="rollback",
                            title="Rollback",
                            severity=IncidentSeverity.SEV2,
                            detected_at=datetime.utcnow(),
                            primary_signal={},
                        ),
                        step_number=-1,
                    )
                except Exception as e:
                    self.log.error("Rollback step failed", step=i, error=str(e))

    # =========================================================================
    # Runbook Action Implementations
    # =========================================================================

    async def _action_restart_service(self, params: dict[str, Any]) -> dict[str, Any]:
        """Restart a service (placeholder - would integrate with K8s/ECS/etc)."""
        service = params.get("service")
        self.log.info("Restarting service", service=service)
        # In production: kubectl rollout restart, aws ecs update-service, etc.
        return {"success": True, "action": "restart_service", "service": service}

    async def _action_scale_up(self, params: dict[str, Any]) -> dict[str, Any]:
        """Scale up a service."""
        service = params.get("service")
        replicas = params.get("replicas", 2)
        self.log.info("Scaling up service", service=service, replicas=replicas)
        return {"success": True, "action": "scale_up", "service": service, "replicas": replicas}

    async def _action_clear_cache(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clear cache for a service."""
        cache_type = params.get("cache_type", "all")
        self.log.info("Clearing cache", cache_type=cache_type)
        return {"success": True, "action": "clear_cache", "cache_type": cache_type}

    async def _action_rollback_deployment(self, params: dict[str, Any]) -> dict[str, Any]:
        """Rollback to a previous deployment."""
        service = params.get("service")
        version = params.get("version", "previous")
        self.log.info("Rolling back deployment", service=service, version=version)
        return {"success": True, "action": "rollback_deployment", "service": service, "version": version}

    async def _action_health_check(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run health check on a service."""
        service = params.get("service")
        endpoint = params.get("endpoint", "/health")
        self.log.info("Running health check", service=service, endpoint=endpoint)
        return {"success": True, "action": "health_check", "service": service, "healthy": True}

    async def _action_notify_team(
        self,
        params: dict[str, Any],
        incident: CorrelatedIncident,
    ) -> dict[str, Any]:
        """Notify the team about an incident."""
        channel = params.get("channel", "oncall")
        message = params.get("message", f"Incident {incident.id}: {incident.title}")
        self.log.info("Notifying team", channel=channel)
        # In production: integrate with Slack, PagerDuty, etc.
        return {"success": True, "action": "notify_team", "channel": channel}

    # =========================================================================
    # Suggestion Generation
    # =========================================================================

    async def _generate_remediation_suggestions(
        self,
        incident: CorrelatedIncident,
    ) -> list[dict[str, Any]]:
        """Generate remediation suggestions for an incident."""
        # First, check if we have matching runbooks
        matching_runbooks = await self._find_matching_runbooks(incident)

        if matching_runbooks:
            return [
                {
                    "type": "runbook",
                    "runbook_id": rb.id,
                    "name": rb.name,
                    "description": rb.description,
                    "success_rate": rb.success_rate,
                    "auto_executable": not rb.requires_approval and incident.severity.value >= "sev4",
                    "confidence": 0.8,
                }
                for rb in matching_runbooks[:3]
            ]

        # Otherwise, use AI to suggest actions
        suggestion_prompt = f"""Suggest remediation actions for this incident:

Incident: {incident.title}
Severity: {incident.severity.value}
Root Cause: {incident.root_cause_hypothesis}
Affected Services: {incident.affected_services}

Provide 2-3 actionable remediation steps as JSON:
{{
    "suggestions": [
        {{
            "type": "action",
            "action": "restart_service|scale_up|clear_cache|rollback_deployment|other",
            "description": "What this does",
            "params": {{}},
            "risk_level": "low|medium|high",
            "estimated_time_minutes": 5,
            "confidence": 0.75
        }}
    ]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": suggestion_prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            json_mode=True,
        )

        result = self._parse_json_response(response.content, {"suggestions": []})
        return result.get("suggestions", [])

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _assess_severity(
        self,
        alert_data: dict[str, Any],
        related_signals: list[dict[str, Any]],
    ) -> IncidentSeverity:
        """Assess incident severity based on signals."""
        # Check for explicit severity
        if "severity" in alert_data:
            sev = alert_data["severity"].lower()
            for s in IncidentSeverity:
                if s.value == sev:
                    return s

        # Heuristic assessment
        error_count = alert_data.get("count", 1)
        services_affected = len(set(
            s.get("service") for s in related_signals if s.get("service")
        ))

        if error_count > 1000 or services_affected > 5:
            return IncidentSeverity.SEV1
        elif error_count > 100 or services_affected > 2:
            return IncidentSeverity.SEV2
        elif error_count > 10:
            return IncidentSeverity.SEV3
        else:
            return IncidentSeverity.SEV4

    def _generate_incident_title(self, alert_data: dict[str, Any]) -> str:
        """Generate a descriptive incident title."""
        service = alert_data.get("service", "Unknown Service")
        error = alert_data.get("error", alert_data.get("message", "Unknown Error"))

        # Truncate long error messages
        if len(error) > 100:
            error = error[:100] + "..."

        return f"[{service}] {error}"

    def _extract_affected_services(
        self,
        alert_data: dict[str, Any],
        related_signals: list[dict[str, Any]],
    ) -> list[str]:
        """Extract list of affected services."""
        services = set()
        if alert_data.get("service"):
            services.add(alert_data["service"])
        for signal in related_signals:
            if signal.get("service"):
                services.add(signal["service"])
        return list(services)

    async def _fetch_related_signals(
        self,
        alert_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch signals related to the primary alert."""
        # In production, query Sentry, Prometheus, etc.
        # For now, return empty list
        return []

    async def _fetch_recent_changes(
        self,
        services: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch recent code/infra changes for affected services."""
        # In production, query GitHub, deployment system, etc.
        return []

    async def _find_similar_incidents(
        self,
        alert_data: dict[str, Any],
    ) -> list[str]:
        """Find similar past incidents from memory."""
        # Query memory manager for similar incidents
        try:
            from ..orchestrator.memory_manager import get_memory_manager, MemoryType
            memory = get_memory_manager(org_id=self.org_id, project_id=self.project_id)

            query = f"{alert_data.get('service', '')} {alert_data.get('error', '')}"
            results = await memory.search(
                query=query,
                memory_types=[MemoryType.EPISODIC],
                limit=5,
            )
            return [r.memory.id for r in results]
        except Exception:
            return []

    async def _find_matching_runbooks(
        self,
        incident: CorrelatedIncident,
    ) -> list[Runbook]:
        """Find runbooks matching the incident."""
        matching = []
        for runbook in self._runbooks.values():
            for condition in runbook.trigger_conditions:
                if (
                    condition.lower() in incident.title.lower()
                    or condition.lower() in (incident.root_cause_hypothesis or "").lower()
                ):
                    matching.append(runbook)
                    break
        return sorted(matching, key=lambda r: r.success_rate, reverse=True)

    async def _load_runbook(self, runbook_id: str) -> Runbook | None:
        """Load a runbook from the database."""
        # In production, load from Supabase
        return None

    async def _attempt_auto_remediation(
        self,
        incident: CorrelatedIncident,
        action: dict[str, Any],
    ) -> None:
        """Attempt automatic remediation for low-severity incidents."""
        self.log.info(
            "Attempting auto-remediation",
            incident_id=incident.id,
            action=action.get("name", action.get("action")),
        )

        if action.get("type") == "runbook":
            await self.execute_remediation(
                incident_id=incident.id,
                runbook_id=action["runbook_id"],
                approved_by="auto",
            )

    async def _store_remediation_outcome(
        self,
        incident: CorrelatedIncident,
        runbook: Runbook,
        success: bool,
    ) -> None:
        """Store remediation outcome in memory for learning."""
        try:
            from ..orchestrator.memory_manager import get_memory_manager, MemoryImportance
            memory = get_memory_manager(org_id=self.org_id, project_id=self.project_id)

            await memory.store_episode(
                event_type="remediation_executed",
                summary=f"Executed runbook '{runbook.name}' for incident '{incident.title}'",
                details={
                    "incident_id": incident.id,
                    "runbook_id": runbook.id,
                    "success": success,
                    "root_cause": incident.root_cause_hypothesis,
                    "severity": incident.severity.value,
                },
                outcome="success" if success else "failure",
                importance=MemoryImportance.HIGH if success else MemoryImportance.MEDIUM,
            )
        except Exception as e:
            self.log.warning("Failed to store remediation outcome", error=str(e))

    def _format_signals_for_prompt(self, signals: list[dict[str, Any]]) -> str:
        """Format signals for inclusion in prompts."""
        if not signals:
            return "None"
        return "\n".join([
            f"- {s.get('service', 'unknown')}: {s.get('error', s.get('message', str(s)))}"
            for s in signals[:10]
        ])

    def _format_dict_for_prompt(self, d: dict[str, Any]) -> str:
        """Format a dict for inclusion in prompts."""
        import json
        return json.dumps(d, indent=2, default=str)[:2000]

    # =========================================================================
    # Metrics Persistence
    # =========================================================================

    async def _persist_sre_metrics(
        self,
        incident: CorrelatedIncident,
        success: bool,
        execution_time_seconds: float,
        runbook_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist SRE incident resolution to agent_evaluations. Fire-and-forget."""
        try:
            mttr_seconds = None
            if incident.resolution_time and incident.detected_at:
                mttr_seconds = (incident.resolution_time - incident.detected_at).total_seconds()

            await self._record_evaluation(
                task_completed=success,
                efficacy_score=incident.root_cause_confidence if success else 0.0,
                latency_ms=int(execution_time_seconds * 1000),
                cost_usd=self._usage.total_cost,
                assurance_score=incident.root_cause_confidence,
                input_tokens=self._usage.total_input_tokens,
                output_tokens=self._usage.total_output_tokens,
                error_type=None if success else "remediation_failed",
                error_message=error,
                trigger_source="sre_remediation",
                organization_id=self.org_id,
                project_id=self.project_id,
                metadata={
                    "incident_id": incident.id,
                    "severity": incident.severity.value,
                    "runbook_id": runbook_id,
                    "mttr_seconds": mttr_seconds,
                    "blast_radius": incident.blast_radius,
                    "affected_services": incident.affected_services,
                },
            )
        except Exception as e:
            self.log.warning("Failed to persist SRE metrics", error=str(e))

    # =========================================================================
    # Metrics and Reporting
    # =========================================================================

    def get_metrics(self) -> SREMetrics:
        """Get current SRE metrics."""
        return self._metrics

    async def calculate_mttr(self, time_range_days: int = 7) -> float:
        """Calculate Mean Time To Resolve for recent incidents."""
        resolved = [
            inc for inc in self._active_incidents.values()
            if inc.resolution_time is not None
        ]

        if not resolved:
            return 0.0

        total_minutes = sum(
            (inc.resolution_time - inc.detected_at).total_seconds() / 60
            for inc in resolved
        )

        return total_minutes / len(resolved)

    # =========================================================================
    # Runbook Management
    # =========================================================================

    def register_runbook(self, runbook: Runbook) -> None:
        """Register a runbook for use in remediation."""
        self._runbooks[runbook.id] = runbook
        self.log.info(
            "Registered runbook",
            runbook_id=runbook.id,
            name=runbook.name,
        )

    def create_runbook(
        self,
        name: str,
        description: str,
        trigger_conditions: list[str],
        steps: list[dict[str, Any]],
        requires_approval: bool = True,
    ) -> Runbook:
        """Create and register a new runbook."""
        from uuid import uuid4

        runbook = Runbook(
            id=f"rb_{uuid4().hex[:12]}",
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            steps=steps,
            requires_approval=requires_approval,
        )

        self.register_runbook(runbook)
        return runbook
