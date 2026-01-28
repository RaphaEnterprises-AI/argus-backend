"""AI-Powered Insights Generation API.

Provides Claude-powered analysis for:
- Semantic failure clustering (groups failures by root cause)
- Coverage gap detection (identifies high-risk untested areas)
- Resolution suggestions (AI-generated fix recommendations)

This replaces the simple SQL-based insights with Claude-powered semantic analysis.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/insights", tags=["AI Insights"])


# =============================================================================
# Models
# =============================================================================


class InsightType(str, Enum):
    """Types of AI-generated insights."""
    PREDICTION = "prediction"
    ANOMALY = "anomaly"
    SUGGESTION = "suggestion"
    UNDERSTANDING = "understanding"


class InsightSeverity(str, Enum):
    """Severity levels for insights."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FailureClusterRequest(BaseModel):
    """Request for AI failure clustering."""
    project_id: str = Field(..., description="Project ID to analyze")
    days: int = Field(7, ge=1, le=90, description="Days of history to analyze")
    min_cluster_size: int = Field(2, ge=1, le=10, description="Minimum failures to form a cluster")


class FailureCluster(BaseModel):
    """A semantically grouped cluster of failures."""
    id: str
    name: str
    description: str
    count: int
    percentage: float
    error_type: str
    root_cause_analysis: str
    affected_tests: list[str]
    affected_test_count: int
    suggested_fix: str
    severity: str
    trend: str  # 'up', 'down', 'stable'
    sample_errors: list[dict]


class FailureClusterResponse(BaseModel):
    """Response from failure clustering endpoint."""
    clusters: list[FailureCluster]
    total_failures: int
    analysis_summary: str
    generated_at: datetime


class CoverageGapRequest(BaseModel):
    """Request for coverage gap analysis."""
    project_id: str = Field(..., description="Project ID to analyze")
    include_api_gaps: bool = Field(True, description="Include API endpoint gaps")
    include_ui_gaps: bool = Field(True, description="Include UI/page gaps")
    include_flow_gaps: bool = Field(True, description="Include user flow gaps")


class CoverageGap(BaseModel):
    """An identified gap in test coverage."""
    id: str
    area: str
    area_type: str  # 'page', 'flow', 'api', 'component'
    current_coverage: float
    risk_level: str  # 'critical', 'high', 'medium', 'low'
    risk_analysis: str
    impact_description: str
    suggested_tests: list[dict]
    suggested_test_count: int
    priority_score: float
    related_failures: list[str]


class CoverageGapResponse(BaseModel):
    """Response from coverage gap analysis."""
    gaps: list[CoverageGap]
    overall_coverage: float
    critical_gaps: int
    high_gaps: int
    total_suggested_tests: int
    analysis_summary: str
    generated_at: datetime


class ResolveInsightRequest(BaseModel):
    """Request for AI resolution suggestion."""
    context: str | None = Field(None, description="Additional context for resolution")


class ResolutionSuggestion(BaseModel):
    """AI-generated resolution suggestion."""
    summary: str
    root_cause: str
    steps: list[dict]
    code_changes: list[dict] | None
    test_improvements: list[str]
    prevention_measures: list[str]
    estimated_effort: str
    confidence: float


class InsightResolutionResponse(BaseModel):
    """Response from insight resolution endpoint."""
    insight_id: str
    resolution: ResolutionSuggestion
    generated_at: datetime


class GenerateInsightsRequest(BaseModel):
    """Request to generate new AI insights."""
    project_id: str = Field(..., description="Project ID to analyze")
    insight_types: list[str] = Field(
        default=["failure_pattern", "coverage_gap", "risk_alert", "optimization"],
        description="Types of insights to generate"
    )
    force_refresh: bool = Field(False, description="Force regeneration even if recent insights exist")


class AIInsight(BaseModel):
    """An AI-generated insight."""
    id: str
    project_id: str
    insight_type: str
    severity: str
    title: str
    description: str
    confidence: float
    affected_area: str | None
    suggested_action: str | None
    action_url: str | None
    related_test_ids: list[str] | None
    is_resolved: bool
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class GenerateInsightsResponse(BaseModel):
    """Response from insight generation."""
    insights: list[AIInsight]
    total_generated: int
    analysis_duration_ms: int
    generated_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_failed_test_results(
    project_id: str,
    days: int,
    limit: int = 500
) -> list[dict]:
    """Get failed test results for a project within the specified time window."""
    supabase = get_supabase_client()

    # Get test runs for this project
    runs_result = await supabase.request(
        f"/test_runs?project_id=eq.{project_id}"
        f"&order=created_at.desc&limit=100"
    )

    if runs_result.get("error") or not runs_result.get("data"):
        return []

    run_ids = [r["id"] for r in runs_result["data"]]
    if not run_ids:
        return []

    # Get failed test results from these runs
    run_ids_str = ",".join(run_ids)
    results_result = await supabase.request(
        f"/test_results?test_run_id=in.({run_ids_str})"
        f"&status=eq.failed"
        f"&order=created_at.desc"
        f"&limit={limit}"
    )

    if results_result.get("error"):
        return []

    return results_result.get("data") or []


async def _get_tests_for_project(project_id: str) -> list[dict]:
    """Get all tests for a project."""
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/tests?project_id=eq.{project_id}&select=id,name,target_url,test_type,tags"
    )

    if result.get("error"):
        return []

    return result.get("data") or []


async def _get_discovered_pages(project_id: str) -> list[dict]:
    """Get discovered pages from discovery sessions."""
    supabase = get_supabase_client()

    # Get discovery sessions
    sessions_result = await supabase.request(
        f"/discovery_sessions?project_id=eq.{project_id}&select=id"
    )

    if sessions_result.get("error") or not sessions_result.get("data"):
        return []

    session_ids = [s["id"] for s in sessions_result["data"]]
    if not session_ids:
        return []

    session_ids_str = ",".join(session_ids)
    pages_result = await supabase.request(
        f"/discovered_pages?session_id=in.({session_ids_str})"
        "&select=id,url,title,page_type,has_forms,has_interactions"
    )

    if pages_result.get("error"):
        return []

    return pages_result.get("data") or []


async def _get_discovered_flows(project_id: str) -> list[dict]:
    """Get discovered user flows from discovery sessions."""
    supabase = get_supabase_client()

    # Get discovery sessions
    sessions_result = await supabase.request(
        f"/discovery_sessions?project_id=eq.{project_id}&select=id"
    )

    if sessions_result.get("error") or not sessions_result.get("data"):
        return []

    session_ids = [s["id"] for s in sessions_result["data"]]
    if not session_ids:
        return []

    session_ids_str = ",".join(session_ids)
    flows_result = await supabase.request(
        f"/discovered_flows?session_id=in.({session_ids_str})"
        "&select=id,name,description,steps,importance"
    )

    if flows_result.get("error"):
        return []

    return flows_result.get("data") or []


async def _get_insight_by_id(insight_id: str) -> dict | None:
    """Get an insight by ID."""
    supabase = get_supabase_client()

    result = await supabase.request(f"/ai_insights?id=eq.{insight_id}")

    if result.get("error") or not result.get("data"):
        return None

    return result["data"][0] if result["data"] else None


async def _save_ai_insight(
    project_id: str,
    insight_type: str,
    severity: str,
    title: str,
    description: str,
    confidence: float,
    affected_area: str | None = None,
    suggested_action: str | None = None,
    metadata: dict | None = None,
) -> str | None:
    """Save an AI insight to the database."""
    supabase = get_supabase_client()

    record = {
        "project_id": project_id,
        "insight_type": insight_type,
        "severity": severity,
        "title": title,
        "description": description,
        "confidence": confidence,
        "affected_area": affected_area,
        "suggested_action": suggested_action,
        "metadata": metadata or {},
        "is_resolved": False,
    }

    result = await supabase.insert("ai_insights", record)

    if result.get("error"):
        logger.error("Failed to save AI insight", error=result["error"])
        return None

    saved = result.get("data", [{}])[0]
    return saved.get("id")


def _call_claude(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 4000) -> str | None:
    """Call Claude API with the given prompt."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not configured")
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    except Exception as e:
        logger.exception("Failed to call Claude", error=str(e))
        return None


def _parse_json_response(response: str) -> Any:
    """Parse JSON from Claude's response, handling markdown code blocks."""
    if not response:
        return None

    # Remove markdown code blocks if present
    text = response.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse Claude JSON response", error=str(e))
        return None


# =============================================================================
# AI Analysis Functions
# =============================================================================


async def _cluster_failures_with_ai(
    failed_results: list[dict],
    min_cluster_size: int = 2,
) -> tuple[list[FailureCluster], str]:
    """Use Claude to semantically cluster failures by root cause."""

    if not failed_results:
        return [], "No failures to analyze."

    # Prepare failure data for Claude
    failures_summary = []
    for i, result in enumerate(failed_results[:100]):  # Limit to 100 for token efficiency
        failures_summary.append({
            "index": i,
            "test_id": result.get("test_id"),
            "test_name": result.get("test_name", "Unknown"),
            "error_message": (result.get("error_message") or "")[:500],  # Truncate long errors
            "error_type": result.get("error_type", "unknown"),
            "stack_trace": (result.get("stack_trace") or "")[:300],
            "created_at": result.get("created_at"),
        })

    prompt = f"""You are an expert test failure analyst. Analyze these test failures and group them into semantic clusters based on their root causes.

## Failures to Analyze (Total: {len(failures_summary)})

```json
{json.dumps(failures_summary, indent=2, default=str)}
```

## Instructions

1. Group failures that share the same root cause (not just similar error messages, but actual underlying cause)
2. For each cluster, provide:
   - A descriptive name for the failure pattern
   - The underlying root cause
   - A suggested fix
   - Severity assessment (critical/high/medium/low)

3. Return ONLY valid JSON in this exact format:

```json
{{
  "clusters": [
    {{
      "name": "Authentication Token Expiry",
      "description": "Failures caused by expired or invalid authentication tokens",
      "error_type": "authentication",
      "root_cause_analysis": "The authentication tokens are not being refreshed properly before API calls, causing 401 errors",
      "suggested_fix": "Implement token refresh logic before API calls or increase token validity period",
      "severity": "high",
      "failure_indices": [0, 3, 7],
      "sample_error": "Request failed with status 401: Token expired"
    }}
  ],
  "analysis_summary": "Brief summary of the overall failure patterns"
}}
```

Important:
- Only create clusters with at least {min_cluster_size} failures
- Use failure_indices to reference which failures belong to each cluster
- Be specific about root causes, not just symptoms
- Provide actionable suggested fixes"""

    response = _call_claude(prompt)

    if not response:
        return _fallback_cluster_failures(failed_results), "AI analysis unavailable, using rule-based clustering."

    parsed = _parse_json_response(response)

    if not parsed or "clusters" not in parsed:
        return _fallback_cluster_failures(failed_results), "AI response parsing failed, using rule-based clustering."

    total_failures = len(failed_results)
    clusters = []

    for i, cluster_data in enumerate(parsed["clusters"]):
        indices = cluster_data.get("failure_indices", [])
        count = len(indices)

        if count < min_cluster_size:
            continue

        # Get affected test IDs
        affected_tests = list(set(
            failed_results[idx].get("test_id")
            for idx in indices
            if idx < len(failed_results) and failed_results[idx].get("test_id")
        ))

        # Get sample errors
        sample_errors = []
        for idx in indices[:3]:
            if idx < len(failed_results):
                sample_errors.append({
                    "test_name": failed_results[idx].get("test_name", "Unknown"),
                    "error_message": (failed_results[idx].get("error_message") or "")[:200],
                })

        clusters.append(FailureCluster(
            id=str(uuid.uuid4()),
            name=cluster_data.get("name", f"Cluster {i+1}"),
            description=cluster_data.get("description", ""),
            count=count,
            percentage=round((count / total_failures) * 100, 1) if total_failures > 0 else 0,
            error_type=cluster_data.get("error_type", "unknown"),
            root_cause_analysis=cluster_data.get("root_cause_analysis", ""),
            affected_tests=affected_tests,
            affected_test_count=len(affected_tests),
            suggested_fix=cluster_data.get("suggested_fix", ""),
            severity=cluster_data.get("severity", "medium"),
            trend="stable",  # Would need historical data to calculate
            sample_errors=sample_errors,
        ))

    # Sort by count descending
    clusters.sort(key=lambda c: c.count, reverse=True)

    return clusters, parsed.get("analysis_summary", "AI-powered failure clustering completed.")


def _fallback_cluster_failures(failed_results: list[dict]) -> list[FailureCluster]:
    """Fallback rule-based clustering when AI is unavailable."""
    # Simple keyword-based clustering
    categories = {
        "timeout": {"keywords": ["timeout", "timed out"], "name": "Timeout Errors"},
        "element": {"keywords": ["element", "selector", "not found", "locator"], "name": "Element Not Found"},
        "network": {"keywords": ["network", "fetch", "connection", "ECONNREFUSED"], "name": "Network Failures"},
        "assertion": {"keywords": ["assert", "expect", "should", "toBe"], "name": "Assertion Failures"},
        "auth": {"keywords": ["auth", "login", "permission", "unauthorized", "401"], "name": "Authentication Issues"},
    }

    categorized: dict[str, list[dict]] = {k: [] for k in categories}
    categorized["other"] = []

    for result in failed_results:
        error = (result.get("error_message") or "").lower()
        matched = False

        for cat_key, cat_info in categories.items():
            if any(kw in error for kw in cat_info["keywords"]):
                categorized[cat_key].append(result)
                matched = True
                break

        if not matched:
            categorized["other"].append(result)

    total_failures = len(failed_results)
    clusters = []

    for cat_key, results in categorized.items():
        if not results:
            continue

        name = categories.get(cat_key, {}).get("name", "Other Errors")
        affected_tests = list(set(r.get("test_id") for r in results if r.get("test_id")))

        clusters.append(FailureCluster(
            id=str(uuid.uuid4()),
            name=name,
            description=f"Failures matching {cat_key} patterns",
            count=len(results),
            percentage=round((len(results) / total_failures) * 100, 1) if total_failures > 0 else 0,
            error_type=cat_key,
            root_cause_analysis="Rule-based clustering - no AI analysis available",
            affected_tests=affected_tests,
            affected_test_count=len(affected_tests),
            suggested_fix="Review error patterns and test implementation",
            severity="medium",
            trend="stable",
            sample_errors=[{"test_name": r.get("test_name", "Unknown"), "error_message": (r.get("error_message") or "")[:200]} for r in results[:3]],
        ))

    clusters.sort(key=lambda c: c.count, reverse=True)
    return [c for c in clusters if c.count > 0]


async def _find_coverage_gaps_with_ai(
    project_id: str,
    tests: list[dict],
    pages: list[dict],
    flows: list[dict],
    failed_results: list[dict],
) -> tuple[list[CoverageGap], float, str]:
    """Use Claude to identify high-risk areas lacking test coverage."""

    # Prepare data for Claude
    data_summary = {
        "existing_tests": [
            {
                "name": t.get("name"),
                "target_url": t.get("target_url"),
                "test_type": t.get("test_type"),
                "tags": t.get("tags", []),
            }
            for t in tests[:50]
        ],
        "discovered_pages": [
            {
                "url": p.get("url"),
                "title": p.get("title"),
                "page_type": p.get("page_type"),
                "has_forms": p.get("has_forms", False),
                "has_interactions": p.get("has_interactions", False),
            }
            for p in pages[:30]
        ],
        "discovered_flows": [
            {
                "name": f.get("name"),
                "description": f.get("description"),
                "importance": f.get("importance", "medium"),
                "step_count": len(f.get("steps", [])),
            }
            for f in flows[:20]
        ],
        "recent_failure_areas": list(set(
            r.get("error_message", "")[:100]
            for r in failed_results[:20]
            if r.get("error_message")
        )),
    }

    prompt = f"""You are a test coverage expert. Analyze this application's test coverage and identify critical gaps.

## Current State

```json
{json.dumps(data_summary, indent=2, default=str)}
```

## Instructions

1. Identify areas that should have test coverage but don't (or have insufficient coverage)
2. Consider:
   - Critical business flows (checkout, payment, authentication)
   - Areas where recent failures occurred
   - Pages with forms or complex interactions
   - User journeys that span multiple pages
   - API endpoints without corresponding tests

3. For each gap, assess:
   - Risk level (critical/high/medium/low)
   - Business impact
   - Specific tests that should be added

4. Return ONLY valid JSON in this exact format:

```json
{{
  "gaps": [
    {{
      "area": "/checkout",
      "area_type": "flow",
      "current_coverage": 0,
      "risk_level": "critical",
      "risk_analysis": "Checkout flow has no automated tests. Any regression here directly impacts revenue.",
      "impact_description": "High revenue impact - checkout failures cause lost sales",
      "suggested_tests": [
        {{"name": "Complete checkout with valid card", "priority": "critical"}},
        {{"name": "Handle payment failure gracefully", "priority": "high"}}
      ],
      "related_failure_patterns": ["Payment timeout errors seen in production"]
    }}
  ],
  "overall_coverage_estimate": 65,
  "analysis_summary": "Brief summary of coverage status and priorities"
}}
```

Important:
- Focus on high-risk, high-impact areas first
- Be specific about what tests are needed
- Consider the recent failure patterns when prioritizing
- Provide actionable test suggestions"""

    response = _call_claude(prompt)

    if not response:
        return _fallback_coverage_gaps(tests, pages, flows), 0.0, "AI analysis unavailable, using rule-based analysis."

    parsed = _parse_json_response(response)

    if not parsed or "gaps" not in parsed:
        return _fallback_coverage_gaps(tests, pages, flows), 0.0, "AI response parsing failed, using rule-based analysis."

    gaps = []
    for i, gap_data in enumerate(parsed["gaps"]):
        suggested_tests = gap_data.get("suggested_tests", [])

        gaps.append(CoverageGap(
            id=str(uuid.uuid4()),
            area=gap_data.get("area", f"Area {i+1}"),
            area_type=gap_data.get("area_type", "page"),
            current_coverage=gap_data.get("current_coverage", 0),
            risk_level=gap_data.get("risk_level", "medium"),
            risk_analysis=gap_data.get("risk_analysis", ""),
            impact_description=gap_data.get("impact_description", ""),
            suggested_tests=suggested_tests,
            suggested_test_count=len(suggested_tests),
            priority_score=_calculate_priority_score(gap_data.get("risk_level", "medium")),
            related_failures=gap_data.get("related_failure_patterns", []),
        ))

    # Sort by priority
    gaps.sort(key=lambda g: g.priority_score, reverse=True)

    overall_coverage = parsed.get("overall_coverage_estimate", 0)
    summary = parsed.get("analysis_summary", "AI-powered coverage analysis completed.")

    return gaps, overall_coverage, summary


def _fallback_coverage_gaps(
    tests: list[dict],
    pages: list[dict],
    flows: list[dict],
) -> list[CoverageGap]:
    """Fallback rule-based coverage gap analysis."""
    test_urls = set(t.get("target_url", "").lower() for t in tests)
    test_names = set(t.get("name", "").lower() for t in tests)

    gaps = []

    # Check pages
    for page in pages:
        url = page.get("url", "").lower()

        # Check if this page has any tests
        is_tested = any(url in test_url or test_url in url for test_url in test_urls)

        if not is_tested:
            # Determine priority based on page type
            priority = "medium"
            if any(kw in url for kw in ["checkout", "payment", "cart", "order"]):
                priority = "critical"
            elif any(kw in url for kw in ["auth", "login", "signup", "register", "profile", "settings"]):
                priority = "high"

            gaps.append(CoverageGap(
                id=str(uuid.uuid4()),
                area=page.get("url", "Unknown"),
                area_type="page",
                current_coverage=0,
                risk_level=priority,
                risk_analysis="No automated tests cover this page",
                impact_description=f"Page type: {page.get('page_type', 'unknown')}",
                suggested_tests=[{"name": f"Test {page.get('title', 'page')}", "priority": priority}],
                suggested_test_count=1,
                priority_score=_calculate_priority_score(priority),
                related_failures=[],
            ))

    # Check flows
    for flow in flows:
        flow_name = flow.get("name", "").lower()

        is_tested = any(flow_name in name or name in flow_name for name in test_names)

        if not is_tested:
            priority = "high" if flow.get("importance") == "high" else "medium"

            gaps.append(CoverageGap(
                id=str(uuid.uuid4()),
                area=flow.get("name", "Unknown flow"),
                area_type="flow",
                current_coverage=0,
                risk_level=priority,
                risk_analysis="User flow has no automated tests",
                impact_description=flow.get("description", ""),
                suggested_tests=[{"name": f"Test {flow.get('name', 'flow')}", "priority": priority}],
                suggested_test_count=1,
                priority_score=_calculate_priority_score(priority),
                related_failures=[],
            ))

    gaps.sort(key=lambda g: g.priority_score, reverse=True)
    return gaps


def _calculate_priority_score(risk_level: str) -> float:
    """Calculate a numeric priority score from risk level."""
    scores = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.5,
        "low": 0.25,
    }
    return scores.get(risk_level, 0.5)


async def _generate_resolution_with_ai(
    insight: dict,
    context: str | None = None,
) -> ResolutionSuggestion | None:
    """Use Claude to generate a resolution suggestion for an insight."""

    prompt = f"""You are a senior test engineer and software quality expert. Generate a detailed resolution plan for this insight.

## Insight to Resolve

- **Title**: {insight.get('title', 'Unknown')}
- **Type**: {insight.get('insight_type', 'unknown')}
- **Severity**: {insight.get('severity', 'medium')}
- **Description**: {insight.get('description', '')}
- **Suggested Action**: {insight.get('suggested_action', 'None provided')}
- **Affected Area**: {insight.get('affected_area', 'Unknown')}
- **Metadata**: {json.dumps(insight.get('metadata', {}), default=str)}

{f'## Additional Context: {context}' if context else ''}

## Instructions

Provide a comprehensive resolution plan including:
1. Root cause analysis
2. Step-by-step resolution steps
3. Any code changes needed
4. Test improvements to prevent recurrence
5. Prevention measures

Return ONLY valid JSON:

```json
{{
  "summary": "Brief one-sentence summary of the resolution",
  "root_cause": "Detailed analysis of the root cause",
  "steps": [
    {{"step": 1, "action": "Review failing tests", "details": "Look at the error patterns..."}},
    {{"step": 2, "action": "Fix selector", "details": "Update the selector to be more specific..."}}
  ],
  "code_changes": [
    {{"file": "tests/checkout.spec.ts", "change": "Update selector from .btn to [data-testid='checkout-btn']", "reason": "More stable selector"}}
  ],
  "test_improvements": [
    "Add wait for network idle before assertions",
    "Add retry logic for flaky API calls"
  ],
  "prevention_measures": [
    "Implement data-testid attributes for all interactive elements",
    "Add pre-commit hook to validate selectors"
  ],
  "estimated_effort": "2-4 hours",
  "confidence": 0.85
}}
```"""

    response = _call_claude(prompt, model="claude-sonnet-4-5-20241022", max_tokens=3000)

    if not response:
        return None

    parsed = _parse_json_response(response)

    if not parsed:
        return None

    return ResolutionSuggestion(
        summary=parsed.get("summary", "Resolution generated"),
        root_cause=parsed.get("root_cause", "See detailed steps"),
        steps=parsed.get("steps", []),
        code_changes=parsed.get("code_changes"),
        test_improvements=parsed.get("test_improvements", []),
        prevention_measures=parsed.get("prevention_measures", []),
        estimated_effort=parsed.get("estimated_effort", "Unknown"),
        confidence=float(parsed.get("confidence", 0.7)),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/cluster", response_model=FailureClusterResponse)
async def cluster_failures(
    request: Request,
    body: FailureClusterRequest,
    user: UserContext = Depends(get_current_user),
):
    """AI-powered semantic failure clustering.

    Groups test failures by their underlying root cause using Claude's
    semantic understanding, rather than simple keyword matching.
    """
    start_time = datetime.now(UTC)

    try:
        # Get failed test results
        failed_results = await _get_failed_test_results(
            body.project_id,
            body.days,
        )

        if not failed_results:
            return FailureClusterResponse(
                clusters=[],
                total_failures=0,
                analysis_summary="No test failures found in the specified time window.",
                generated_at=datetime.now(UTC),
            )

        # Cluster failures using AI
        clusters, summary = await _cluster_failures_with_ai(
            failed_results,
            body.min_cluster_size,
        )

        logger.info(
            "Failure clustering completed",
            project_id=body.project_id,
            total_failures=len(failed_results),
            clusters_found=len(clusters),
            user_id=user.user_id,
        )

        return FailureClusterResponse(
            clusters=clusters,
            total_failures=len(failed_results),
            analysis_summary=summary,
            generated_at=datetime.now(UTC),
        )

    except Exception as e:
        logger.exception("Failed to cluster failures", error=str(e))
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@router.post("/coverage-gaps", response_model=CoverageGapResponse)
async def find_coverage_gaps(
    request: Request,
    body: CoverageGapRequest,
    user: UserContext = Depends(get_current_user),
):
    """AI-powered coverage gap detection.

    Identifies high-risk areas of the application that lack adequate
    test coverage, using Claude to assess risk and suggest tests.
    """
    try:
        # Gather data for analysis
        tests = await _get_tests_for_project(body.project_id)
        pages = await _get_discovered_pages(body.project_id) if body.include_ui_gaps else []
        flows = await _get_discovered_flows(body.project_id) if body.include_flow_gaps else []
        failed_results = await _get_failed_test_results(body.project_id, 30)

        # Find gaps using AI
        gaps, overall_coverage, summary = await _find_coverage_gaps_with_ai(
            body.project_id,
            tests,
            pages,
            flows,
            failed_results,
        )

        # Calculate stats
        critical_count = sum(1 for g in gaps if g.risk_level == "critical")
        high_count = sum(1 for g in gaps if g.risk_level == "high")
        total_suggested = sum(g.suggested_test_count for g in gaps)

        logger.info(
            "Coverage gap analysis completed",
            project_id=body.project_id,
            gaps_found=len(gaps),
            critical_gaps=critical_count,
            user_id=user.user_id,
        )

        return CoverageGapResponse(
            gaps=gaps,
            overall_coverage=overall_coverage,
            critical_gaps=critical_count,
            high_gaps=high_count,
            total_suggested_tests=total_suggested,
            analysis_summary=summary,
            generated_at=datetime.now(UTC),
        )

    except Exception as e:
        logger.exception("Failed to find coverage gaps", error=str(e))
        raise HTTPException(status_code=500, detail=f"Coverage analysis failed: {str(e)}")


@router.post("/{insight_id}/resolve", response_model=InsightResolutionResponse)
async def resolve_insight(
    insight_id: str,
    request: Request,
    body: ResolveInsightRequest | None = None,
    user: UserContext = Depends(get_current_user),
):
    """AI-powered resolution suggestion for an insight.

    Uses Claude to analyze the insight and generate a detailed
    resolution plan including steps, code changes, and prevention measures.
    """
    try:
        # Get the insight
        insight = await _get_insight_by_id(insight_id)

        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")

        # Generate resolution
        context = body.context if body else None
        resolution = await _generate_resolution_with_ai(insight, context)

        if not resolution:
            raise HTTPException(
                status_code=503,
                detail="AI resolution generation unavailable. Check API key configuration."
            )

        logger.info(
            "Resolution generated for insight",
            insight_id=insight_id,
            confidence=resolution.confidence,
            user_id=user.user_id,
        )

        return InsightResolutionResponse(
            insight_id=insight_id,
            resolution=resolution,
            generated_at=datetime.now(UTC),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to generate resolution", error=str(e))
        raise HTTPException(status_code=500, detail=f"Resolution generation failed: {str(e)}")


@router.post("/generate", response_model=GenerateInsightsResponse)
async def generate_insights(
    request: Request,
    body: GenerateInsightsRequest,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(get_current_user),
):
    """Generate new AI-powered insights for a project.

    Analyzes test results, coverage, and patterns to generate
    actionable insights using Claude.
    """
    start_time = datetime.now(UTC)

    try:
        insights: list[AIInsight] = []

        # Get data for analysis
        failed_results = await _get_failed_test_results(body.project_id, 14)
        tests = await _get_tests_for_project(body.project_id)
        pages = await _get_discovered_pages(body.project_id)
        flows = await _get_discovered_flows(body.project_id)

        # Generate failure pattern insights
        if "failure_pattern" in body.insight_types and failed_results:
            clusters, _ = await _cluster_failures_with_ai(failed_results, 2)

            for cluster in clusters[:3]:  # Top 3 clusters
                if cluster.severity in ["critical", "high"]:
                    insight_id = await _save_ai_insight(
                        project_id=body.project_id,
                        insight_type="anomaly",
                        severity=cluster.severity,
                        title=f"Failure Pattern: {cluster.name}",
                        description=f"{cluster.root_cause_analysis}. {cluster.count} failures affecting {cluster.affected_test_count} tests.",
                        confidence=0.8,
                        affected_area=cluster.error_type,
                        suggested_action=cluster.suggested_fix,
                        metadata={"cluster_id": cluster.id, "failure_count": cluster.count},
                    )

                    if insight_id:
                        insights.append(AIInsight(
                            id=insight_id,
                            project_id=body.project_id,
                            insight_type="anomaly",
                            severity=cluster.severity,
                            title=f"Failure Pattern: {cluster.name}",
                            description=f"{cluster.root_cause_analysis}. {cluster.count} failures affecting {cluster.affected_test_count} tests.",
                            confidence=0.8,
                            affected_area=cluster.error_type,
                            suggested_action=cluster.suggested_fix,
                            related_test_ids=cluster.affected_tests[:10],
                            is_resolved=False,
                            created_at=datetime.now(UTC),
                            metadata={"cluster_id": cluster.id},
                        ))

        # Generate coverage gap insights
        if "coverage_gap" in body.insight_types:
            gaps, _, _ = await _find_coverage_gaps_with_ai(
                body.project_id, tests, pages, flows, failed_results
            )

            for gap in gaps[:3]:  # Top 3 critical gaps
                if gap.risk_level in ["critical", "high"]:
                    insight_id = await _save_ai_insight(
                        project_id=body.project_id,
                        insight_type="suggestion",
                        severity=gap.risk_level,
                        title=f"Coverage Gap: {gap.area}",
                        description=f"{gap.risk_analysis}. {gap.impact_description}",
                        confidence=0.75,
                        affected_area=gap.area,
                        suggested_action=f"Add {gap.suggested_test_count} tests to cover this area",
                        metadata={"gap_id": gap.id, "area_type": gap.area_type},
                    )

                    if insight_id:
                        insights.append(AIInsight(
                            id=insight_id,
                            project_id=body.project_id,
                            insight_type="suggestion",
                            severity=gap.risk_level,
                            title=f"Coverage Gap: {gap.area}",
                            description=f"{gap.risk_analysis}. {gap.impact_description}",
                            confidence=0.75,
                            affected_area=gap.area,
                            suggested_action=f"Add {gap.suggested_test_count} tests to cover this area",
                            related_test_ids=None,
                            is_resolved=False,
                            created_at=datetime.now(UTC),
                            metadata={"gap_id": gap.id},
                        ))

        duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

        logger.info(
            "AI insights generated",
            project_id=body.project_id,
            insights_count=len(insights),
            duration_ms=duration_ms,
            user_id=user.user_id,
        )

        return GenerateInsightsResponse(
            insights=insights,
            total_generated=len(insights),
            analysis_duration_ms=duration_ms,
            generated_at=datetime.now(UTC),
        )

    except Exception as e:
        logger.exception("Failed to generate insights", error=str(e))
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")


@router.get("", response_model=list[AIInsight])
async def list_insights(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    insight_type: str | None = Query(None, description="Filter by insight type"),
    severity: str | None = Query(None, description="Filter by severity"),
    is_resolved: bool | None = Query(None, description="Filter by resolved status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    user: UserContext = Depends(get_current_user),
):
    """List AI insights for a project."""
    supabase = get_supabase_client()

    try:
        query = f"/ai_insights?project_id=eq.{project_id}"

        if insight_type:
            query += f"&insight_type=eq.{insight_type}"
        if severity:
            query += f"&severity=eq.{severity}"
        if is_resolved is not None:
            query += f"&is_resolved=eq.{str(is_resolved).lower()}"

        query += f"&order=created_at.desc&limit={limit}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return []
            raise HTTPException(status_code=500, detail="Failed to list insights")

        insights = []
        for row in result.get("data") or []:
            insights.append(AIInsight(
                id=row["id"],
                project_id=row["project_id"],
                insight_type=row["insight_type"],
                severity=row["severity"],
                title=row["title"],
                description=row["description"],
                confidence=float(row.get("confidence", 0.5)),
                affected_area=row.get("affected_area"),
                suggested_action=row.get("suggested_action"),
                action_url=row.get("action_url"),
                related_test_ids=row.get("related_test_ids"),
                is_resolved=row.get("is_resolved", False),
                created_at=datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ),
                metadata=row.get("metadata") or {},
            ))

        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list insights", error=str(e))
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.patch("/{insight_id}/resolve")
async def mark_insight_resolved(
    insight_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Mark an insight as resolved."""
    supabase = get_supabase_client()

    try:
        result = await supabase.update(
            "ai_insights",
            {"id": f"eq.{insight_id}"},
            {
                "is_resolved": True,
                "resolved_at": datetime.now(UTC).isoformat(),
                "resolved_by": user.user_id,
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to update insight")

        logger.info(
            "Insight marked as resolved",
            insight_id=insight_id,
            user_id=user.user_id,
        )

        return {"success": True, "insight_id": insight_id, "is_resolved": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to resolve insight", error=str(e))
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
