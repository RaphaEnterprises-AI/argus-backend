"""Quality Intelligence API for test generation and risk scoring.

Provides endpoints for:
- AI-powered test generation from production errors
- Batch test generation for multiple events
- Risk score calculation and tracking
- Quality metrics and statistics
- Production event management
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from src.config import get_settings
from src.services.supabase_client import get_supabase_client
from src.services.cache import cache_quality_score

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/quality", tags=["Quality Intelligence"])


# =============================================================================
# Pydantic Models
# =============================================================================

class TestGenerationRequest(BaseModel):
    """Request to generate a test from a production event."""
    production_event_id: str = Field(..., description="ID of the production event")
    project_id: str = Field(..., description="Project ID")
    framework: Literal["playwright", "cypress", "jest"] = Field("playwright", description="Test framework")
    auto_create_pr: bool = Field(False, description="Automatically create GitHub PR")
    github_config: Optional[dict] = Field(None, description="GitHub configuration for PR creation")


class BatchGenerationRequest(BaseModel):
    """Request for batch test generation."""
    project_id: str = Field(..., description="Project ID")
    status: str = Field("new", description="Status of events to process")
    limit: int = Field(10, le=50, description="Max events to process")
    framework: Literal["playwright", "cypress", "jest"] = Field("playwright", description="Test framework")


class TestUpdateRequest(BaseModel):
    """Request to update a generated test."""
    test_id: str = Field(..., description="Generated test ID")
    action: Literal["approve", "reject", "modify"] = Field(..., description="Review action")
    review_notes: Optional[str] = Field(None, description="Review notes")
    modified_code: Optional[str] = Field(None, description="Modified test code (for modify action)")


class RiskScoreRequest(BaseModel):
    """Request to calculate risk scores."""
    project_id: str = Field(..., description="Project ID")
    entity_types: list[str] = Field(
        default=["page", "component", "flow", "endpoint"],
        description="Entity types to calculate scores for",
    )


class GeneratedTestResponse(BaseModel):
    """Response for a generated test."""
    id: str
    name: str
    file_path: str
    confidence_score: float
    framework: str


class TestGenerationResponse(BaseModel):
    """Response after test generation."""
    success: bool
    message: str
    generated_test: Optional[GeneratedTestResponse] = None
    test_code: Optional[str] = None
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None


# =============================================================================
# Test Generation Prompt
# =============================================================================

TEST_GENERATION_PROMPT = """You are an expert QA engineer who converts production errors into automated tests.

Given a production error, generate a Playwright E2E test that would:
1. Navigate to the page where the error occurred
2. Perform the actions that led to the error
3. Assert that the error does NOT occur (the fix is in place)
4. Include proper error handling and assertions

ERROR DETAILS:
Title: {title}
Message: {message}
URL: {url}
Component: {component}
Stack Trace:
{stack_trace}

User Action Context: {user_action}

REQUIREMENTS:
- Use Playwright's best practices
- Include descriptive test names and comments
- Handle async operations properly
- Use robust selectors (data-testid preferred)
- Include both positive and negative assertions
- Add retry logic for flaky elements

Generate a complete, runnable Playwright test file. Return ONLY the test code, no explanations."""


# =============================================================================
# Helper Functions
# =============================================================================

async def generate_test_from_error(event: dict) -> dict:
    """Generate a test using Claude AI from an error event."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    prompt = TEST_GENERATION_PROMPT.format(
        title=event.get("title", "Unknown Error"),
        message=event.get("message") or "No message provided",
        url=event.get("url") or "Unknown URL",
        component=event.get("component") or "Unknown component",
        stack_trace=event.get("stack_trace") or "No stack trace",
        user_action=event.get("user_action") or "Unknown action",
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    test_code = response.content[0].text

    # Extract test name from generated code
    import re
    test_name_match = re.search(r"test\(['\"](.+?)['\"]", test_code)
    test_name = test_name_match.group(1) if test_name_match else f"test-{event.get('component', 'error')}-{int(datetime.now().timestamp())}"

    # Calculate confidence based on available context
    confidence = 0.5
    if event.get("url"):
        confidence += 0.15
    if event.get("stack_trace"):
        confidence += 0.15
    if event.get("component"):
        confidence += 0.1
    if event.get("user_action"):
        confidence += 0.1

    return {
        "test_code": test_code,
        "test_name": test_name,
        "confidence": min(confidence, 0.95),
    }


def get_file_path(component: Optional[str], framework: str, timestamp: int) -> str:
    """Generate file path for the test."""
    import re
    component_slug = re.sub(r"[^a-z0-9]", "-", (component or "unknown").lower())

    if framework == "playwright":
        return f"tests/e2e/generated/{component_slug}-{timestamp}.spec.ts"
    elif framework == "cypress":
        return f"cypress/e2e/generated/{component_slug}-{timestamp}.cy.ts"
    else:
        return f"__tests__/generated/{component_slug}-{timestamp}.test.ts"


# =============================================================================
# Test Generation Endpoints
# =============================================================================

@router.post("/generate-test", response_model=TestGenerationResponse)
async def generate_test(request: TestGenerationRequest):
    """
    Generate a test from a production error event.

    Uses Claude AI to analyze the error and generate a comprehensive E2E test
    that validates the fix is in place.
    """
    supabase = get_supabase_client()

    # Get the production event
    result = await supabase.select(
        "production_events",
        filters={
            "id": f"eq.{request.production_event_id}",
            "project_id": f"eq.{request.project_id}",
        },
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Production event not found")

    event = result["data"][0]

    # Create job record
    job_id = str(uuid.uuid4())
    job_start = datetime.utcnow()

    await supabase.insert("test_generation_jobs", {
        "id": job_id,
        "project_id": request.project_id,
        "production_event_id": request.production_event_id,
        "status": "running",
        "job_type": "single_error",
        "started_at": job_start.isoformat(),
    })

    # Update event status
    await supabase.update(
        "production_events",
        {"id": f"eq.{request.production_event_id}"},
        {"status": "analyzing"},
    )

    try:
        # Generate the test
        generated = await generate_test_from_error(event)
        test_code = generated["test_code"]
        test_name = generated["test_name"]
        confidence = generated["confidence"]

        # Determine file path
        timestamp = int(datetime.now().timestamp())
        file_path = get_file_path(event.get("component"), request.framework, timestamp)

        # Create generated test record
        test_result = await supabase.insert("generated_tests", {
            "project_id": request.project_id,
            "production_event_id": request.production_event_id,
            "name": test_name,
            "description": f"Auto-generated test to prevent: {event.get('title')}",
            "test_type": "e2e",
            "framework": request.framework,
            "test_code": test_code,
            "test_file_path": file_path,
            "confidence_score": confidence,
            "status": "pending",
            "steps": [],
            "assertions": [],
            "metadata": {
                "generated_from_error": event.get("title"),
                "original_url": event.get("url"),
                "component": event.get("component"),
            },
        })

        if test_result.get("error"):
            raise Exception(f"Failed to save test: {test_result['error']}")

        generated_test_id = test_result["data"][0]["id"] if test_result["data"] else None

        # Update event status
        await supabase.update(
            "production_events",
            {"id": f"eq.{request.production_event_id}"},
            {
                "status": "test_pending_review",
                "ai_analysis": {
                    "generated_test_id": generated_test_id,
                    "confidence_score": confidence,
                    "generated_at": datetime.utcnow().isoformat(),
                },
            },
        )

        # Update job as completed
        duration_ms = int((datetime.utcnow() - job_start).total_seconds() * 1000)
        await supabase.update(
            "test_generation_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "completed",
                "tests_generated": 1,
                "completed_at": datetime.utcnow().isoformat(),
                "duration_ms": duration_ms,
            },
        )

        logger.info(
            "Test generated successfully",
            test_id=generated_test_id,
            confidence=confidence,
        )

        return TestGenerationResponse(
            success=True,
            message="Test generated successfully",
            generated_test=GeneratedTestResponse(
                id=generated_test_id,
                name=test_name,
                file_path=file_path,
                confidence_score=confidence,
                framework=request.framework,
            ),
            test_code=test_code,
        )

    except Exception as e:
        # Update job as failed
        await supabase.update(
            "test_generation_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.utcnow().isoformat(),
            },
        )
        logger.exception("Test generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-generate")
async def batch_generate_tests(request: BatchGenerationRequest):
    """
    Generate tests for multiple production events in batch.

    Processes events sequentially to avoid API rate limits.
    """
    supabase = get_supabase_client()

    # Get production events that need tests
    result = await supabase.request(
        f"/production_events?project_id=eq.{request.project_id}&status=eq.{request.status}"
        f"&order=severity.desc,occurrence_count.desc&limit={request.limit}"
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch events")

    events = result.get("data", [])

    if not events:
        return {"success": True, "message": "No events to process", "generated": 0}

    # Create batch job
    job_id = str(uuid.uuid4())
    job_start = datetime.utcnow()

    await supabase.insert("test_generation_jobs", {
        "id": job_id,
        "project_id": request.project_id,
        "status": "running",
        "job_type": "pattern_batch",
        "started_at": job_start.isoformat(),
        "metadata": {"event_count": len(events)},
    })

    results = []

    for event in events:
        try:
            generated = await generate_test_from_error(event)
            timestamp = int(datetime.now().timestamp())
            file_path = get_file_path(event.get("component"), request.framework, timestamp)

            test_result = await supabase.insert("generated_tests", {
                "project_id": request.project_id,
                "production_event_id": event["id"],
                "name": generated["test_name"],
                "description": f"Auto-generated test to prevent: {event.get('title')}",
                "test_type": "e2e",
                "framework": request.framework,
                "test_code": generated["test_code"],
                "test_file_path": file_path,
                "confidence_score": generated["confidence"],
                "status": "pending",
            })

            test_id = test_result["data"][0]["id"] if test_result.get("data") else None

            await supabase.update(
                "production_events",
                {"id": f"eq.{event['id']}"},
                {
                    "status": "test_pending_review",
                    "ai_analysis": {
                        "generated_test_id": test_id,
                        "confidence_score": generated["confidence"],
                    },
                },
            )

            results.append({"event_id": event["id"], "success": True, "test_id": test_id})

        except Exception as e:
            results.append({"event_id": event["id"], "success": False, "error": str(e)})

    success_count = sum(1 for r in results if r["success"])

    # Update job
    await supabase.update(
        "test_generation_jobs",
        {"id": f"eq.{job_id}"},
        {
            "status": "completed",
            "tests_generated": success_count,
            "completed_at": datetime.utcnow().isoformat(),
            "metadata": {"results": results},
        },
    )

    return {
        "success": True,
        "message": f"Generated {success_count}/{len(events)} tests",
        "job_id": job_id,
        "results": results,
    }


@router.post("/update-test")
async def update_generated_test(request: TestUpdateRequest):
    """
    Approve, reject, or modify a generated test.
    """
    supabase = get_supabase_client()

    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "modify": "modified",
    }

    update_data = {
        "status": status_map[request.action],
        "reviewed_at": datetime.utcnow().isoformat(),
        "review_notes": request.review_notes,
    }

    if request.modified_code and request.action == "modify":
        update_data["test_code"] = request.modified_code

    result = await supabase.update(
        "generated_tests",
        {"id": f"eq.{request.test_id}"},
        update_data,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to update test")

    # If approved, update the production event
    if request.action == "approve":
        test_result = await supabase.select(
            "generated_tests",
            filters={"id": f"eq.{request.test_id}"},
        )
        if test_result.get("data") and test_result["data"][0].get("production_event_id"):
            await supabase.update(
                "production_events",
                {"id": f"eq.{test_result['data'][0]['production_event_id']}"},
                {"status": "test_generated"},
            )

    return {"success": True, "message": f"Test {request.action}d successfully"}


# =============================================================================
# Risk Scoring Endpoints
# =============================================================================

RISK_WEIGHTS = {
    "error_frequency": 0.25,
    "error_severity": 0.30,
    "test_coverage": 0.20,
    "user_impact": 0.15,
    "recency": 0.10,
}


@router.post("/calculate-risk")
async def calculate_risk_scores(request: RiskScoreRequest):
    """
    Calculate risk scores for all entities in a project.

    Risk is calculated based on error frequency, severity, test coverage,
    user impact, and recency of errors.
    """
    supabase = get_supabase_client()

    # Get aggregated error data by component/page/flow
    events_result = await supabase.request(
        f"/production_events?project_id=eq.{request.project_id}&select=*"
    )

    if events_result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch events")

    events = events_result.get("data", [])

    if not events:
        return {"success": True, "risk_scores": [], "message": "No events found"}

    # Aggregate by component/URL
    entity_data: dict[str, dict] = {}

    for event in events:
        # Use component or URL as entity identifier
        entity_key = event.get("component") or event.get("url") or "unknown"

        if entity_key not in entity_data:
            entity_data[entity_key] = {
                "entity_type": "component" if event.get("component") else "page",
                "entity_identifier": entity_key,
                "error_count": 0,
                "fatal_count": 0,
                "error_count_severity": 0,
                "warning_count": 0,
                "affected_users": 0,
                "last_error_at": event.get("last_seen_at") or event.get("created_at"),
                "first_error_at": event.get("first_seen_at") or event.get("created_at"),
            }

        data = entity_data[entity_key]
        data["error_count"] += event.get("occurrence_count", 1)
        data["affected_users"] += event.get("affected_users", 1)

        severity = event.get("severity", "error")
        if severity == "fatal":
            data["fatal_count"] += 1
        elif severity == "error":
            data["error_count_severity"] += 1
        else:
            data["warning_count"] += 1

        # Update timestamps
        if event.get("last_seen_at") and event["last_seen_at"] > data["last_error_at"]:
            data["last_error_at"] = event["last_seen_at"]

    # Calculate risk scores
    max_error_count = max((d["error_count"] for d in entity_data.values()), default=1)
    max_affected_users = max((d["affected_users"] for d in entity_data.values()), default=1)

    risk_scores = []
    now = datetime.utcnow()

    for entity_key, data in entity_data.items():
        # Calculate individual factors
        error_frequency = min(100, round((data["error_count"] / max_error_count) * 100)) if max_error_count > 0 else 0

        # Severity weighted score
        total_errors = max(1, data["error_count"])
        error_severity = min(100, round(
            ((data["fatal_count"] * 100) + (data["error_count_severity"] * 70) + (data["warning_count"] * 30)) / total_errors
        ))

        # Test coverage (default to 0, meaning 100 risk from no coverage)
        test_coverage = 100  # No tests = high risk

        # User impact
        user_impact = min(100, round((data["affected_users"] / max_affected_users) * 100)) if max_affected_users > 0 else 0

        # Recency
        try:
            last_error = datetime.fromisoformat(data["last_error_at"].replace("Z", "+00:00"))
            days_since = (now - last_error.replace(tzinfo=None)).days
            recency = max(0, round(100 - (days_since / 30) * 100))
        except Exception:
            recency = 50

        factors = {
            "error_frequency": error_frequency,
            "error_severity": error_severity,
            "test_coverage": test_coverage,
            "user_impact": user_impact,
            "recency": recency,
        }

        # Calculate overall weighted score
        overall = round(
            (factors["error_frequency"] * RISK_WEIGHTS["error_frequency"]) +
            (factors["error_severity"] * RISK_WEIGHTS["error_severity"]) +
            (factors["test_coverage"] * RISK_WEIGHTS["test_coverage"]) +
            (factors["user_impact"] * RISK_WEIGHTS["user_impact"]) +
            (factors["recency"] * RISK_WEIGHTS["recency"])
        )

        risk_scores.append({
            "entity_type": data["entity_type"],
            "entity_identifier": entity_key,
            "overall_risk_score": min(100, overall),
            "factors": factors,
            "error_count": data["error_count"],
            "affected_users": data["affected_users"],
            "trend": "stable",  # Would need historical data to calculate
        })

    # Sort by overall score (highest risk first)
    risk_scores.sort(key=lambda x: x["overall_risk_score"], reverse=True)

    # Store risk scores in database
    for score in risk_scores[:50]:  # Limit to top 50
        await supabase.request(
            "/risk_scores",
            method="POST",
            body={
                "project_id": request.project_id,
                "entity_type": score["entity_type"],
                "entity_identifier": score["entity_identifier"],
                "overall_risk_score": score["overall_risk_score"],
                "factors": score["factors"],
                "error_count": score["error_count"],
                "affected_users": score["affected_users"],
                "trend": score["trend"],
                "calculated_at": datetime.utcnow().isoformat(),
            },
            headers={"Prefer": "resolution=merge-duplicates"},
        )

    return {
        "success": True,
        "risk_scores": risk_scores[:20],  # Return top 20
        "total_entities": len(risk_scores),
    }


# =============================================================================
# Query Endpoints
# =============================================================================

@router.get("/events")
async def get_production_events(
    project_id: str = Query(..., description="Project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(50, le=100, description="Max results"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """Get production events with optional filtering."""
    supabase = get_supabase_client()

    query_path = f"/production_events?project_id=eq.{project_id}"
    if status:
        query_path += f"&status=eq.{status}"
    if severity:
        query_path += f"&severity=eq.{severity}"
    if source:
        query_path += f"&source=eq.{source}"
    query_path += f"&order=created_at.desc&limit={limit}&offset={offset}"

    result = await supabase.request(query_path)

    if result.get("error"):
        raise HTTPException(status_code=500, detail="Failed to fetch events")

    return {"events": result.get("data", [])}


@router.get("/stats")
async def get_quality_stats(
    project_id: str = Query(..., description="Project ID"),
):
    """Get quality intelligence statistics for a project."""
    supabase = get_supabase_client()

    # Get event counts by status
    events_result = await supabase.request(
        f"/production_events?project_id=eq.{project_id}&select=id,status,severity"
    )

    events = events_result.get("data", [])

    # Get generated tests count
    tests_result = await supabase.request(
        f"/generated_tests?project_id=eq.{project_id}&select=id,status"
    )

    tests = tests_result.get("data", [])

    # Calculate statistics
    stats = {
        "total_events": len(events),
        "events_by_status": {},
        "events_by_severity": {},
        "total_generated_tests": len(tests),
        "tests_by_status": {},
        "coverage_rate": 0,
    }

    for event in events:
        status = event.get("status", "unknown")
        severity = event.get("severity", "unknown")
        stats["events_by_status"][status] = stats["events_by_status"].get(status, 0) + 1
        stats["events_by_severity"][severity] = stats["events_by_severity"].get(severity, 0) + 1

    for test in tests:
        status = test.get("status", "unknown")
        stats["tests_by_status"][status] = stats["tests_by_status"].get(status, 0) + 1

    # Calculate coverage (events with tests / total events)
    events_with_tests = sum(1 for e in events if e.get("status") in ("test_pending_review", "test_generated"))
    stats["coverage_rate"] = round((events_with_tests / len(events) * 100) if events else 0, 1)

    return {"stats": stats}


@router.get("/risk-scores")
async def get_risk_scores(
    project_id: str = Query(..., description="Project ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(20, le=100, description="Max results"),
):
    """Get risk scores for a project."""
    supabase = get_supabase_client()

    query_path = f"/risk_scores?project_id=eq.{project_id}"
    if entity_type:
        query_path += f"&entity_type=eq.{entity_type}"
    query_path += f"&order=overall_risk_score.desc&limit={limit}"

    result = await supabase.request(query_path)

    # Handle missing table gracefully
    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
            logger.warning("risk_scores table not found - returning empty list")
            return {"risk_scores": [], "message": "Run migrations to enable risk scoring"}
        raise HTTPException(status_code=500, detail="Failed to fetch risk scores")

    return {"risk_scores": result.get("data", [])}


@router.get("/generated-tests")
async def get_generated_tests(
    project_id: str = Query(..., description="Project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, le=100, description="Max results"),
):
    """Get generated tests for a project."""
    supabase = get_supabase_client()

    query_path = f"/generated_tests?project_id=eq.{project_id}"
    if status:
        query_path += f"&status=eq.{status}"
    query_path += f"&order=created_at.desc&limit={limit}"

    result = await supabase.request(query_path)

    # Handle missing table gracefully
    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42703" in error_msg or "42P01" in error_msg:
            logger.warning("generated_tests table not found - returning empty list")
            return {"tests": [], "message": "Run migrations to enable test storage"}
        raise HTTPException(status_code=500, detail="Failed to fetch tests")

    return {"tests": result.get("data", [])}


@cache_quality_score(key_prefix="project_score")
async def _calculate_quality_score(project_id: str) -> dict:
    """Calculate quality score for a project (cached)."""
    supabase = get_supabase_client()

    # Get events
    events_result = await supabase.request(
        f"/production_events?project_id=eq.{project_id}&select=id,status,severity"
    )
    events = events_result.get("data", [])

    # Get generated tests (handle missing table)
    tests_result = await supabase.request(
        f"/generated_tests?project_id=eq.{project_id}&select=id,status"
    )
    tests = tests_result.get("data", []) if not tests_result.get("error") else []

    # Get risk scores (handle missing table)
    risk_result = await supabase.request(
        f"/risk_scores?project_id=eq.{project_id}&select=overall_risk_score"
    )
    risk_scores = risk_result.get("data", []) if not risk_result.get("error") else []

    # Calculate quality score (inverse of risk)
    avg_risk = sum(r.get("overall_risk_score", 50) for r in risk_scores) / len(risk_scores) if risk_scores else 50
    quality_score = max(0, 100 - avg_risk)

    # Calculate test coverage
    approved_tests = sum(1 for t in tests if t.get("status") == "approved")
    coverage = round((approved_tests / len(events) * 100) if events else 0, 1)

    return {
        "quality_score": round(quality_score, 1),
        "risk_level": "high" if avg_risk > 70 else "medium" if avg_risk > 40 else "low",
        "test_coverage": coverage,
        "total_events": len(events),
        "total_tests": len(tests),
        "approved_tests": approved_tests,
    }


@router.get("/score")
async def get_quality_score(
    project_id: str = Query(..., description="Project ID"),
):
    """Get overall quality score for a project (cached for 5 minutes)."""
    return await _calculate_quality_score(project_id)
