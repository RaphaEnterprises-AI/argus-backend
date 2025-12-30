"""
AI Synthesis Layer for Observability Data

This is where the MAGIC happens. We take raw data from observability platforms
and use AI to synthesize actionable testing intelligence:

1. Convert real user sessions into automated tests
2. Prioritize errors by actual user impact
3. Detect patterns that indicate incoming failures
4. Auto-generate test coverage based on real usage
5. Predict which tests to run based on production data

The key insight: We're not just monitoring - we're LEARNING.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import anthropic

from src.integrations.observability_hub import (
    ObservabilityHub,
    RealUserSession,
    ProductionError,
    PerformanceAnomaly,
    UserJourneyPattern,
    Platform,
)
from src.config import get_settings


class InsightType(str, Enum):
    """Types of AI-generated insights."""
    TEST_SUGGESTION = "test_suggestion"
    ERROR_PRIORITY = "error_priority"
    FAILURE_PREDICTION = "failure_prediction"
    COVERAGE_GAP = "coverage_gap"
    USER_PATTERN = "user_pattern"
    PERFORMANCE_ALERT = "performance_alert"
    FLAKY_DETECTION = "flaky_detection"


class ActionPriority(str, Enum):
    """Priority levels for suggested actions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestSuggestion:
    """An AI-generated test suggestion based on production data."""
    id: str
    name: str
    description: str
    source: str  # "session_replay", "error_pattern", "user_journey"
    source_platform: Platform
    source_id: str  # session_id, error_id, etc.
    priority: ActionPriority
    confidence: float
    steps: list[dict]
    expected_outcomes: list[str]
    tags: list[str]
    estimated_coverage: float  # How much new coverage this adds


@dataclass
class ErrorInsight:
    """AI analysis of a production error."""
    error: ProductionError
    priority: ActionPriority
    impact_score: float  # 0-100, based on users affected, frequency
    root_cause_hypothesis: str
    suggested_test: Optional[TestSuggestion]
    related_errors: list[str]  # IDs of similar errors
    affected_user_journeys: list[str]
    recommended_actions: list[str]


@dataclass
class FailurePrediction:
    """AI prediction of an incoming failure."""
    id: str
    prediction_type: str
    confidence: float
    affected_area: str
    description: str
    evidence: list[dict]
    recommended_actions: list[str]
    predicted_timeframe: str
    prevention_tests: list[TestSuggestion]


@dataclass
class CoverageGap:
    """An identified gap in test coverage based on production usage."""
    id: str
    area: str
    description: str
    user_traffic_percent: float  # How much traffic hits this area
    current_coverage_percent: float
    priority: ActionPriority
    suggested_tests: list[TestSuggestion]


@dataclass
class SynthesisReport:
    """Complete synthesis report from all observability data."""
    generated_at: datetime
    platforms_analyzed: list[Platform]
    sessions_analyzed: int
    errors_analyzed: int

    # Key insights
    test_suggestions: list[TestSuggestion]
    error_insights: list[ErrorInsight]
    failure_predictions: list[FailurePrediction]
    coverage_gaps: list[CoverageGap]

    # Metrics
    overall_health_score: float
    test_coverage_score: float
    error_trend: str  # "improving", "stable", "degrading"

    # Executive summary
    summary: str
    top_actions: list[dict]


class AISynthesizer:
    """
    The AI brain that synthesizes testing intelligence from observability data.

    This is what makes our platform TRULY intelligent:
    - It doesn't just collect data, it UNDERSTANDS it
    - It doesn't just report errors, it PREDICTS failures
    - It doesn't just show coverage, it GENERATES tests
    """

    def __init__(self, observability_hub: ObservabilityHub):
        self.hub = observability_hub
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    async def synthesize(
        self,
        lookback_hours: int = 24,
        session_limit: int = 100,
        error_limit: int = 100
    ) -> SynthesisReport:
        """
        Main synthesis function - analyzes all observability data and generates insights.

        This is the CORE VALUE PROP. We take raw production data and turn it into:
        1. Prioritized test suggestions
        2. Error analysis with root cause hypotheses
        3. Failure predictions before they happen
        4. Coverage gaps based on real usage
        """
        since = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Gather data from all platforms in parallel
        sessions, errors, anomalies, journeys = await asyncio.gather(
            self.hub.get_all_sessions(limit_per_platform=session_limit, since=since),
            self.hub.get_all_errors(limit_per_platform=error_limit, since=since),
            self.hub.get_all_anomalies(since=since),
            self.hub.get_all_user_journeys(limit_per_platform=20),
            return_exceptions=True
        )

        # Handle any exceptions
        sessions = sessions if isinstance(sessions, list) else []
        errors = errors if isinstance(errors, list) else []
        anomalies = anomalies if isinstance(anomalies, list) else []
        journeys = journeys if isinstance(journeys, list) else []

        # Run AI analysis in parallel
        test_suggestions, error_insights, predictions, coverage_gaps = await asyncio.gather(
            self._generate_test_suggestions(sessions, errors, journeys),
            self._analyze_errors(errors, sessions),
            self._predict_failures(sessions, errors, anomalies),
            self._identify_coverage_gaps(sessions, journeys),
        )

        # Calculate metrics
        overall_health = self._calculate_health_score(errors, anomalies)
        coverage_score = self._calculate_coverage_score(coverage_gaps)
        error_trend = self._calculate_error_trend(errors)

        # Generate executive summary
        summary = await self._generate_summary(
            test_suggestions, error_insights, predictions, coverage_gaps,
            overall_health, len(sessions), len(errors)
        )

        # Prioritize top actions
        top_actions = self._prioritize_actions(
            test_suggestions, error_insights, predictions, coverage_gaps
        )

        return SynthesisReport(
            generated_at=datetime.utcnow(),
            platforms_analyzed=list(self.hub.providers.keys()),
            sessions_analyzed=len(sessions),
            errors_analyzed=len(errors),
            test_suggestions=test_suggestions,
            error_insights=error_insights,
            failure_predictions=predictions,
            coverage_gaps=coverage_gaps,
            overall_health_score=overall_health,
            test_coverage_score=coverage_score,
            error_trend=error_trend,
            summary=summary,
            top_actions=top_actions,
        )

    async def _generate_test_suggestions(
        self,
        sessions: list[RealUserSession],
        errors: list[ProductionError],
        journeys: list[UserJourneyPattern]
    ) -> list[TestSuggestion]:
        """Generate test suggestions from real user behavior."""
        suggestions = []

        # 1. Convert high-value sessions to tests
        high_value_sessions = self._identify_high_value_sessions(sessions)
        for session in high_value_sessions[:10]:  # Top 10
            suggestion = await self._session_to_test(session)
            if suggestion:
                suggestions.append(suggestion)

        # 2. Create tests from error patterns
        for error in errors[:10]:  # Top 10 most impactful errors
            suggestion = await self._error_to_test(error)
            if suggestion:
                suggestions.append(suggestion)

        # 3. Convert user journeys to tests
        for journey in journeys[:5]:  # Top 5 journeys
            suggestion = await self._journey_to_test(journey)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _identify_high_value_sessions(
        self,
        sessions: list[RealUserSession]
    ) -> list[RealUserSession]:
        """Identify sessions that are most valuable for test generation."""
        scored_sessions = []

        for session in sessions:
            score = 0

            # Sessions with conversions are valuable
            if session.conversion_events:
                score += 50

            # Sessions with errors teach us what to test
            if session.errors:
                score += 30

            # Long sessions with many actions show complex flows
            if len(session.actions) > 10:
                score += 20

            # Frustration signals indicate pain points to test
            if session.frustration_signals:
                score += 25

            scored_sessions.append((session, score))

        # Sort by score and return top sessions
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sessions]

    async def _session_to_test(
        self,
        session: RealUserSession
    ) -> Optional[TestSuggestion]:
        """Convert a real user session into a test suggestion using AI."""
        if not session.actions:
            return None

        # Use Claude to analyze the session and generate a test
        prompt = f"""Analyze this real user session and generate a test specification.

Session Data:
- Session ID: {session.session_id}
- Duration: {session.duration_ms}ms
- Pages Visited: {json.dumps(session.page_views[:10])}
- Actions: {json.dumps(session.actions[:20])}
- Errors: {json.dumps(session.errors[:5])}
- Frustration Signals: {json.dumps(session.frustration_signals[:5])}
- Conversions: {json.dumps(session.conversion_events[:5])}

Generate a test specification in JSON format:
{{
    "name": "Descriptive test name",
    "description": "What this test validates",
    "priority": "critical|high|medium|low",
    "confidence": 0.0-1.0,
    "steps": [
        {{"action": "navigate|click|type|assert", "target": "selector or url", "value": "optional value"}}
    ],
    "expected_outcomes": ["Expected result 1", "Expected result 2"],
    "tags": ["tag1", "tag2"],
    "estimated_coverage": 0.0-1.0
}}

Focus on the most critical user flow in this session."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response
            content = response.content[0].text
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                spec = json.loads(json_match.group())
                return TestSuggestion(
                    id=f"ts_{session.session_id[:8]}",
                    name=spec.get("name", "Generated Test"),
                    description=spec.get("description", ""),
                    source="session_replay",
                    source_platform=session.platform,
                    source_id=session.session_id,
                    priority=ActionPriority(spec.get("priority", "medium")),
                    confidence=spec.get("confidence", 0.7),
                    steps=spec.get("steps", []),
                    expected_outcomes=spec.get("expected_outcomes", []),
                    tags=spec.get("tags", []),
                    estimated_coverage=spec.get("estimated_coverage", 0.1),
                )
        except Exception as e:
            pass

        return None

    async def _error_to_test(
        self,
        error: ProductionError
    ) -> Optional[TestSuggestion]:
        """Convert an error into a regression test."""
        prompt = f"""Analyze this production error and generate a regression test.

Error Data:
- Message: {error.message}
- Stack Trace: {error.stack_trace[:500] if error.stack_trace else 'N/A'}
- Occurrence Count: {error.occurrence_count}
- Affected Users: {error.affected_users}
- Environment: {error.environment}
- Context: {json.dumps(error.context)}

Generate a regression test specification in JSON format:
{{
    "name": "Regression test name",
    "description": "What this test validates",
    "priority": "critical|high|medium|low",
    "confidence": 0.0-1.0,
    "steps": [
        {{"action": "navigate|click|type|assert", "target": "selector or url", "value": "optional value"}}
    ],
    "expected_outcomes": ["The error should not occur", "Other validations"],
    "tags": ["regression", "error-prevention"],
    "estimated_coverage": 0.0-1.0
}}

This test should PREVENT this error from happening again."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                spec = json.loads(json_match.group())
                return TestSuggestion(
                    id=f"te_{error.error_id[:8]}",
                    name=spec.get("name", "Regression Test"),
                    description=spec.get("description", ""),
                    source="error_pattern",
                    source_platform=error.platform,
                    source_id=error.error_id,
                    priority=ActionPriority(spec.get("priority", "high")),
                    confidence=spec.get("confidence", 0.8),
                    steps=spec.get("steps", []),
                    expected_outcomes=spec.get("expected_outcomes", []),
                    tags=spec.get("tags", ["regression"]),
                    estimated_coverage=spec.get("estimated_coverage", 0.05),
                )
        except Exception:
            pass

        return None

    async def _journey_to_test(
        self,
        journey: UserJourneyPattern
    ) -> Optional[TestSuggestion]:
        """Convert a user journey pattern into an E2E test."""
        if not journey.steps:
            return None

        prompt = f"""Analyze this user journey pattern and generate an E2E test.

Journey Data:
- Name: {journey.name}
- Steps: {json.dumps(journey.steps)}
- Frequency: {journey.frequency} users follow this path
- Conversion Rate: {journey.conversion_rate * 100:.1f}%
- Drop-off Points: {json.dumps(journey.drop_off_points)}
- Is Critical: {journey.is_critical}

Generate an E2E test specification in JSON format:
{{
    "name": "E2E test name",
    "description": "What this test validates",
    "priority": "critical|high|medium|low",
    "confidence": 0.0-1.0,
    "steps": [
        {{"action": "navigate|click|type|assert", "target": "selector or url", "value": "optional value"}}
    ],
    "expected_outcomes": ["User journey completes successfully", "Other validations"],
    "tags": ["e2e", "user-journey"],
    "estimated_coverage": 0.0-1.0
}}

Focus on validating the complete user journey."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                spec = json.loads(json_match.group())
                priority = "critical" if journey.is_critical else spec.get("priority", "high")
                return TestSuggestion(
                    id=f"tj_{journey.pattern_id[:8]}",
                    name=spec.get("name", "User Journey Test"),
                    description=spec.get("description", ""),
                    source="user_journey",
                    source_platform=Platform.POSTHOG,  # Default
                    source_id=journey.pattern_id,
                    priority=ActionPriority(priority),
                    confidence=spec.get("confidence", 0.85),
                    steps=spec.get("steps", []),
                    expected_outcomes=spec.get("expected_outcomes", []),
                    tags=spec.get("tags", ["e2e", "user-journey"]),
                    estimated_coverage=spec.get("estimated_coverage", 0.2),
                )
        except Exception:
            pass

        return None

    async def _analyze_errors(
        self,
        errors: list[ProductionError],
        sessions: list[RealUserSession]
    ) -> list[ErrorInsight]:
        """Analyze errors and generate insights."""
        insights = []

        # Group errors by similarity
        error_groups = self._group_similar_errors(errors)

        for group in error_groups[:10]:  # Top 10 groups
            main_error = group[0]

            # Calculate impact score
            total_occurrences = sum(e.occurrence_count for e in group)
            total_users = sum(e.affected_users for e in group)
            impact_score = min(100, (total_users * 5) + (total_occurrences * 0.1))

            # Determine priority
            if impact_score >= 80 or main_error.severity == "critical":
                priority = ActionPriority.CRITICAL
            elif impact_score >= 50:
                priority = ActionPriority.HIGH
            elif impact_score >= 20:
                priority = ActionPriority.MEDIUM
            else:
                priority = ActionPriority.LOW

            # Generate test suggestion for the error
            test_suggestion = await self._error_to_test(main_error)

            insights.append(ErrorInsight(
                error=main_error,
                priority=priority,
                impact_score=impact_score,
                root_cause_hypothesis=self._generate_root_cause_hypothesis(main_error),
                suggested_test=test_suggestion,
                related_errors=[e.error_id for e in group[1:]],
                affected_user_journeys=[],  # Would need journey correlation
                recommended_actions=self._generate_error_actions(main_error, impact_score),
            ))

        return insights

    def _group_similar_errors(
        self,
        errors: list[ProductionError]
    ) -> list[list[ProductionError]]:
        """Group errors by similarity (message, stack trace)."""
        groups = []
        used = set()

        for error in errors:
            if error.error_id in used:
                continue

            group = [error]
            used.add(error.error_id)

            for other in errors:
                if other.error_id in used:
                    continue

                # Simple similarity check - could use embeddings for better matching
                if (error.message == other.message or
                    (error.stack_trace and other.stack_trace and
                     error.stack_trace[:200] == other.stack_trace[:200])):
                    group.append(other)
                    used.add(other.error_id)

            groups.append(group)

        # Sort by total impact
        groups.sort(key=lambda g: sum(e.occurrence_count for e in g), reverse=True)
        return groups

    def _generate_root_cause_hypothesis(self, error: ProductionError) -> str:
        """Generate a hypothesis for the root cause of an error."""
        message = error.message.lower()

        if "network" in message or "fetch" in message or "xhr" in message:
            return "Network connectivity issue or API failure"
        elif "undefined" in message or "null" in message or "cannot read" in message:
            return "Null reference error - missing data or incorrect state"
        elif "timeout" in message:
            return "Operation timeout - slow response or deadlock"
        elif "permission" in message or "403" in message or "401" in message:
            return "Authentication or authorization issue"
        elif "syntax" in message or "parse" in message:
            return "Data parsing error - malformed input"
        else:
            return "Requires investigation - analyze stack trace and context"

    def _generate_error_actions(
        self,
        error: ProductionError,
        impact_score: float
    ) -> list[str]:
        """Generate recommended actions for an error."""
        actions = []

        if impact_score >= 80:
            actions.append("IMMEDIATE: Investigate and fix this error")
            actions.append("Notify on-call team")

        actions.append("Create regression test to prevent recurrence")

        if error.stack_trace:
            actions.append("Review stack trace for root cause")

        if error.affected_sessions:
            actions.append("Analyze affected user sessions for context")

        return actions

    async def _predict_failures(
        self,
        sessions: list[RealUserSession],
        errors: list[ProductionError],
        anomalies: list[PerformanceAnomaly]
    ) -> list[FailurePrediction]:
        """Predict incoming failures based on patterns."""
        predictions = []

        # 1. Error rate trend prediction
        error_trend = self._analyze_error_trend(errors)
        if error_trend.get("is_increasing"):
            predictions.append(FailurePrediction(
                id=f"fp_error_trend_{datetime.utcnow().timestamp()}",
                prediction_type="error_rate_increase",
                confidence=error_trend.get("confidence", 0.7),
                affected_area="Global",
                description=f"Error rate is increasing by {error_trend.get('increase_percent', 0):.1f}% per hour",
                evidence=[{"type": "trend", "data": error_trend}],
                recommended_actions=[
                    "Review recent deployments",
                    "Check infrastructure health",
                    "Monitor error patterns"
                ],
                predicted_timeframe="Next 1-4 hours",
                prevention_tests=[],
            ))

        # 2. Performance degradation prediction
        for anomaly in anomalies:
            if anomaly.deviation_percent > 50:
                predictions.append(FailurePrediction(
                    id=f"fp_perf_{anomaly.anomaly_id}",
                    prediction_type="performance_degradation",
                    confidence=min(1.0, anomaly.deviation_percent / 100),
                    affected_area=", ".join(anomaly.affected_pages[:3]),
                    description=f"{anomaly.metric} degraded by {anomaly.deviation_percent:.1f}%",
                    evidence=[{
                        "type": "anomaly",
                        "metric": anomaly.metric,
                        "baseline": anomaly.baseline_value,
                        "current": anomaly.current_value
                    }],
                    recommended_actions=[
                        "Investigate cause of performance degradation",
                        "Check for resource contention",
                        "Review recent changes to affected pages"
                    ],
                    predicted_timeframe="Ongoing",
                    prevention_tests=[],
                ))

        # 3. Frustration signal analysis
        frustration_count = sum(len(s.frustration_signals) for s in sessions)
        if frustration_count > 50:  # Threshold
            predictions.append(FailurePrediction(
                id=f"fp_frustration_{datetime.utcnow().timestamp()}",
                prediction_type="user_experience_degradation",
                confidence=min(1.0, frustration_count / 100),
                affected_area="User Experience",
                description=f"High frustration signals detected ({frustration_count} in analyzed sessions)",
                evidence=[{"type": "frustration_count", "count": frustration_count}],
                recommended_actions=[
                    "Review rage click hotspots",
                    "Check for UI responsiveness issues",
                    "Analyze dead click patterns"
                ],
                predicted_timeframe="Immediate attention required",
                prevention_tests=[],
            ))

        return predictions

    def _analyze_error_trend(self, errors: list[ProductionError]) -> dict:
        """Analyze error trend over time."""
        if not errors:
            return {"is_increasing": False}

        # Group errors by hour
        now = datetime.utcnow()
        hourly_counts = {}

        for error in errors:
            hour = error.first_seen.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour] = hourly_counts.get(hour, 0) + error.occurrence_count

        if len(hourly_counts) < 2:
            return {"is_increasing": False}

        # Simple trend analysis
        hours = sorted(hourly_counts.keys())
        first_half = sum(hourly_counts.get(h, 0) for h in hours[:len(hours)//2])
        second_half = sum(hourly_counts.get(h, 0) for h in hours[len(hours)//2:])

        if first_half > 0:
            increase_percent = ((second_half - first_half) / first_half) * 100
            is_increasing = increase_percent > 20  # 20% threshold

            return {
                "is_increasing": is_increasing,
                "increase_percent": increase_percent,
                "confidence": min(1.0, abs(increase_percent) / 100),
                "first_half_count": first_half,
                "second_half_count": second_half,
            }

        return {"is_increasing": False}

    async def _identify_coverage_gaps(
        self,
        sessions: list[RealUserSession],
        journeys: list[UserJourneyPattern]
    ) -> list[CoverageGap]:
        """Identify gaps in test coverage based on real usage."""
        gaps = []

        # Analyze page traffic
        page_traffic = {}
        for session in sessions:
            for page in session.page_views:
                page_url = page if isinstance(page, str) else page.get("url", "")
                page_traffic[page_url] = page_traffic.get(page_url, 0) + 1

        # Sort by traffic
        sorted_pages = sorted(page_traffic.items(), key=lambda x: x[1], reverse=True)

        # Top pages without tests (we'd need to check actual test coverage)
        for page, traffic in sorted_pages[:10]:
            if page and traffic > 5:  # Minimum traffic threshold
                traffic_percent = (traffic / len(sessions)) * 100 if sessions else 0

                gaps.append(CoverageGap(
                    id=f"cg_{hash(page) % 10000}",
                    area=page,
                    description=f"High-traffic page needs test coverage",
                    user_traffic_percent=traffic_percent,
                    current_coverage_percent=0,  # Would need actual coverage data
                    priority=ActionPriority.HIGH if traffic_percent > 30 else ActionPriority.MEDIUM,
                    suggested_tests=[],  # Would generate suggestions
                ))

        return gaps

    def _calculate_health_score(
        self,
        errors: list[ProductionError],
        anomalies: list[PerformanceAnomaly]
    ) -> float:
        """Calculate overall application health score (0-100)."""
        score = 100.0

        # Deduct for errors
        critical_errors = sum(1 for e in errors if e.severity == "critical")
        high_errors = sum(1 for e in errors if e.severity in ["error", "high"])

        score -= critical_errors * 10
        score -= high_errors * 3
        score -= len(errors) * 0.5

        # Deduct for performance anomalies
        for anomaly in anomalies:
            score -= min(20, anomaly.deviation_percent / 5)

        return max(0, min(100, score))

    def _calculate_coverage_score(self, gaps: list[CoverageGap]) -> float:
        """Calculate test coverage score based on gaps."""
        if not gaps:
            return 100.0

        # Simple calculation based on gap severity
        score = 100.0
        for gap in gaps:
            if gap.priority == ActionPriority.CRITICAL:
                score -= 15
            elif gap.priority == ActionPriority.HIGH:
                score -= 10
            elif gap.priority == ActionPriority.MEDIUM:
                score -= 5
            else:
                score -= 2

        return max(0, min(100, score))

    def _calculate_error_trend(self, errors: list[ProductionError]) -> str:
        """Determine if error trend is improving, stable, or degrading."""
        trend = self._analyze_error_trend(errors)

        if trend.get("is_increasing") and trend.get("increase_percent", 0) > 30:
            return "degrading"
        elif trend.get("is_increasing"):
            return "slightly_degrading"
        elif trend.get("increase_percent", 0) < -20:
            return "improving"
        else:
            return "stable"

    async def _generate_summary(
        self,
        test_suggestions: list[TestSuggestion],
        error_insights: list[ErrorInsight],
        predictions: list[FailurePrediction],
        coverage_gaps: list[CoverageGap],
        health_score: float,
        sessions_analyzed: int,
        errors_analyzed: int
    ) -> str:
        """Generate an executive summary of the synthesis."""
        critical_errors = sum(1 for e in error_insights if e.priority == ActionPriority.CRITICAL)
        high_predictions = sum(1 for p in predictions if p.confidence > 0.7)

        summary_parts = [
            f"Analyzed {sessions_analyzed} sessions and {errors_analyzed} errors.",
            f"Overall health score: {health_score:.0f}/100.",
        ]

        if critical_errors > 0:
            summary_parts.append(f"ALERT: {critical_errors} critical errors require immediate attention.")

        if high_predictions > 0:
            summary_parts.append(f"WARNING: {high_predictions} high-confidence failure predictions detected.")

        if test_suggestions:
            summary_parts.append(f"Generated {len(test_suggestions)} test suggestions from production data.")

        if coverage_gaps:
            summary_parts.append(f"Identified {len(coverage_gaps)} coverage gaps based on real user traffic.")

        return " ".join(summary_parts)

    def _prioritize_actions(
        self,
        test_suggestions: list[TestSuggestion],
        error_insights: list[ErrorInsight],
        predictions: list[FailurePrediction],
        coverage_gaps: list[CoverageGap]
    ) -> list[dict]:
        """Generate a prioritized list of recommended actions."""
        actions = []

        # Critical errors first
        for insight in error_insights:
            if insight.priority == ActionPriority.CRITICAL:
                actions.append({
                    "priority": "critical",
                    "type": "error_fix",
                    "title": f"Fix critical error: {insight.error.message[:50]}",
                    "description": insight.root_cause_hypothesis,
                    "source_id": insight.error.error_id,
                    "source_url": insight.error.issue_url,
                })

        # High-confidence predictions
        for prediction in predictions:
            if prediction.confidence > 0.7:
                actions.append({
                    "priority": "high",
                    "type": "prediction",
                    "title": prediction.description[:50],
                    "description": prediction.predicted_timeframe,
                    "actions": prediction.recommended_actions,
                })

        # High-priority test suggestions
        for suggestion in test_suggestions[:5]:
            if suggestion.priority in [ActionPriority.CRITICAL, ActionPriority.HIGH]:
                actions.append({
                    "priority": str(suggestion.priority.value),
                    "type": "test_creation",
                    "title": f"Create test: {suggestion.name}",
                    "description": suggestion.description,
                    "source": suggestion.source,
                    "confidence": suggestion.confidence,
                })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 4))

        return actions[:10]  # Top 10 actions


async def create_ai_synthesizer(hub: Optional[ObservabilityHub] = None) -> AISynthesizer:
    """Create an AI Synthesizer with the observability hub."""
    if hub is None:
        hub = ObservabilityHub()
    return AISynthesizer(hub)
