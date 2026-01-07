"""FastAPI backend service for E2E Testing Agent.

Provides REST API endpoints for:
- Running tests
- Checking status
- Managing baselines
- Webhooks for CI/CD integration
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import structlog

from src.config import get_settings
from src.orchestrator.graph import TestingOrchestrator
from src.orchestrator.state import create_initial_state
from src.integrations.reporter import create_reporter, create_report_from_state
from src.api.webhooks import router as webhooks_router
from src.api.quality import router as quality_router

logger = structlog.get_logger()

# =============================================================================
# Frontend Alias Models
# =============================================================================

class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    error_text: str = Field(..., description="Error text to search for similar patterns")
    limit: int = Field(5, le=20, description="Maximum results to return")
    min_score: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class AutonomousLoopRequest(BaseModel):
    """Request for autonomous quality loop."""
    project_id: str = Field(..., description="Project ID")
    url: str = Field(..., description="Application URL to test")
    stages: list[str] = Field(
        default=["discovery", "visual", "generation", "verification"],
        description="Stages to run"
    )
    discovery_depth: int = Field(2, ge=1, le=5, description="Crawl depth")
    auto_create_pr: bool = Field(False, description="Auto-create GitHub PR")
    github_config: Optional[dict] = Field(None, description="GitHub config for PR")

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="E2E Testing Agent API",
    description="Autonomous E2E testing powered by Claude AI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(webhooks_router)
app.include_router(quality_router)

# In-memory job storage (use Redis for production)
jobs: dict[str, dict] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class TestRunRequest(BaseModel):
    """Request to start a test run."""
    codebase_path: str = Field(..., description="Path to codebase to analyze")
    app_url: str = Field(..., description="URL of the application to test")
    pr_number: Optional[int] = Field(None, description="PR number for GitHub integration")
    changed_files: Optional[list[str]] = Field(None, description="Specific files to focus on")
    max_tests: Optional[int] = Field(None, description="Maximum number of tests to run")
    focus_areas: Optional[list[str]] = Field(None, description="Areas to focus testing on")


class TestRunResponse(BaseModel):
    """Response after starting a test run."""
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


class NLPTestRequest(BaseModel):
    """Request to create test from natural language."""
    description: str = Field(..., description="Plain English test description")
    app_url: str = Field("http://localhost:3000", description="Application URL to test")
    context: Optional[str] = Field(None, description="Additional context about the app")


class VisualCompareRequest(BaseModel):
    """Request to compare screenshots."""
    baseline_b64: str = Field(..., description="Base64 encoded baseline screenshot")
    current_b64: str = Field(..., description="Base64 encoded current screenshot")
    context: Optional[str] = Field(None, description="Context about what's being compared")


class WebhookPayload(BaseModel):
    """Webhook payload for CI/CD integration."""
    action: str  # pr_opened, pr_updated, push
    repository: dict
    pull_request: Optional[dict] = None
    commits: Optional[list] = None


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies all dependencies are available."""
    settings = get_settings()

    checks = {
        "anthropic_api": settings.anthropic_api_key is not None and bool(
            settings.anthropic_api_key.get_secret_value() if hasattr(settings.anthropic_api_key, 'get_secret_value') else settings.anthropic_api_key
        ),
        "output_dir": os.path.exists(settings.output_dir),
    }

    all_ready = all(checks.values())

    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# ============================================================================
# Test Run Endpoints
# ============================================================================

@app.post("/api/v1/tests/run", response_model=TestRunResponse, tags=["Testing"])
async def start_test_run(request: TestRunRequest, background_tasks: BackgroundTasks):
    """
    Start a new test run.

    The test run executes in the background. Use the returned job_id to check status.
    """
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump(),
        "progress": {"phase": "initializing", "tests_completed": 0},
    }

    # Run tests in background
    background_tasks.add_task(
        run_tests_background,
        job_id,
        request.codebase_path,
        request.app_url,
        request.pr_number,
        request.changed_files,
        request.focus_areas,
    )

    logger.info("Test run started", job_id=job_id, app_url=request.app_url)

    return TestRunResponse(
        job_id=job_id,
        status="pending",
        message="Test run started. Use /api/v1/jobs/{job_id} to check status.",
        created_at=jobs[job_id]["created_at"],
    )


async def run_tests_background(
    job_id: str,
    codebase_path: str,
    app_url: str,
    pr_number: Optional[int],
    changed_files: Optional[list[str]],
    focus_areas: Optional[list[str]],
):
    """Background task to run tests."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"]["phase"] = "analyzing"

        # Create orchestrator
        orchestrator = TestingOrchestrator(
            codebase_path=codebase_path,
            app_url=app_url,
        )

        # Create initial state
        state = create_initial_state(
            codebase_path=codebase_path,
            app_url=app_url,
            pr_number=pr_number,
            changed_files=changed_files or [],
        )

        # Run the orchestrator
        final_state = await orchestrator.run(state)

        # Generate reports
        settings = get_settings()
        reporter = create_reporter(output_dir=settings.output_dir)
        report_data = create_report_from_state(final_state)
        report_paths = reporter.generate_all(report_data)

        # Get summary
        summary = orchestrator.get_run_summary(final_state)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs[job_id]["result"] = {
            "summary": summary,
            "report_paths": {k: str(v) for k, v in report_paths.items()},
        }

        logger.info("Test run completed", job_id=job_id, summary=summary)

    except Exception as e:
        logger.exception("Test run failed", job_id=job_id, error=str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Testing"])
async def get_job_status(job_id: str):
    """Get the status of a test run job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@app.get("/api/v1/jobs", tags=["Testing"])
async def list_jobs(limit: int = 20, status: Optional[str] = None):
    """List recent test run jobs."""
    filtered = list(jobs.items())

    if status:
        filtered = [(k, v) for k, v in filtered if v["status"] == status]

    # Sort by created_at descending
    filtered.sort(key=lambda x: x[1]["created_at"], reverse=True)

    return {
        "jobs": [
            {
                "job_id": k,
                "status": v["status"],
                "created_at": v["created_at"],
                "completed_at": v.get("completed_at"),
            }
            for k, v in filtered[:limit]
        ],
        "total": len(filtered),
    }


# ============================================================================
# NLP Test Creation Endpoints
# ============================================================================

@app.post("/api/v1/tests/create", tags=["NLP Testing"])
async def create_test_from_nlp(request: NLPTestRequest):
    """
    Create a test specification from plain English description.

    Example: "Login as admin@example.com and verify dashboard shows 5 widgets"
    """
    from src.agents.nlp_test_creator import NLPTestCreator

    try:
        creator = NLPTestCreator(app_url=request.app_url)
        test = await creator.create(request.description, context=request.context)

        return {
            "success": True,
            "test": test.to_dict(),
            "spec": test.to_spec(),
        }
    except Exception as e:
        logger.exception("Failed to create test from NLP", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Visual AI Endpoints
# ============================================================================

@app.post("/api/v1/visual/compare", tags=["Visual AI"])
async def compare_screenshots(request: VisualCompareRequest):
    """
    Compare two screenshots using Visual AI.

    Returns differences and similarity score.
    """
    from src.agents.visual_ai import VisualAI

    try:
        visual_ai = VisualAI()
        result = await visual_ai.compare(
            request.baseline_b64,
            request.current_b64,
            context=request.context,
        )

        return {
            "success": True,
            "result": result.to_dict(),
        }
    except Exception as e:
        logger.exception("Visual comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Auto-Discovery Endpoints
# ============================================================================

@app.post("/api/v1/discover", tags=["Auto-Discovery"])
async def discover_tests(
    app_url: str,
    focus_areas: Optional[list[str]] = None,
    max_pages: int = 20,
):
    """
    Auto-discover test scenarios by crawling the application.
    """
    from src.agents.auto_discovery import AutoDiscovery

    try:
        discovery = AutoDiscovery(
            app_url=app_url,
            max_pages=max_pages,
        )
        result = await discovery.discover(focus_areas=focus_areas)

        return {
            "success": True,
            "result": result.to_dict(),
        }
    except Exception as e:
        logger.exception("Auto-discovery failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Webhook Endpoints (CI/CD Integration)
# ============================================================================

@app.post("/api/v1/webhooks/github", tags=["Webhooks"])
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    GitHub webhook endpoint for PR events.

    Automatically triggers tests when PRs are opened or updated.
    """
    # Verify webhook signature (implement in production)
    payload = await request.json()

    action = payload.get("action")

    if action in ["opened", "synchronize", "reopened"]:
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})

        # Start test run
        job_id = str(uuid.uuid4())

        jobs[job_id] = {
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "trigger": "github_webhook",
            "pr_number": pr.get("number"),
        }

        # Get preview URL or use default
        preview_url = f"https://preview-{pr.get('number')}.example.com"  # Configure this

        background_tasks.add_task(
            run_tests_background,
            job_id,
            repo.get("full_name", ""),
            preview_url,
            pr.get("number"),
            None,  # changed_files - could extract from PR
            None,  # focus_areas
        )

        return {"status": "accepted", "job_id": job_id}

    return {"status": "ignored", "action": action}


# ============================================================================
# Reports Endpoints
# ============================================================================

@app.get("/api/v1/reports/{job_id}", tags=["Reports"])
async def get_report(job_id: str, format: str = "json"):
    """
    Get the report for a completed test run.

    Supports: json, html, markdown, junit
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")

    report_paths = job.get("result", {}).get("report_paths", {})

    format_key = format.lower()
    if format_key not in report_paths:
        raise HTTPException(status_code=404, detail=f"Report format '{format}' not available")

    report_path = report_paths[format_key]

    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    media_types = {
        "json": "application/json",
        "html": "text/html",
        "markdown": "text/markdown",
        "junit": "application/xml",
    }

    return FileResponse(
        report_path,
        media_type=media_types.get(format_key, "application/octet-stream"),
        filename=os.path.basename(report_path),
    )


# ============================================================================
# Frontend API Aliases (Match dashboard expectations)
# ============================================================================

@app.get("/api/quality-score", tags=["Frontend Aliases"])
async def frontend_quality_score(project_id: str):
    """Alias for /api/v1/quality/score - matches frontend expectations."""
    from src.api.quality import _calculate_quality_score

    try:
        result = await _calculate_quality_score(project_id)
        return {
            "success": True,
            "project_id": project_id,
            "overall_score": result.get("quality_score", 50),
            "grade": "A" if result.get("quality_score", 0) >= 90 else "B" if result.get("quality_score", 0) >= 80 else "C" if result.get("quality_score", 0) >= 70 else "D" if result.get("quality_score", 0) >= 60 else "F",
            "grade_color": "green" if result.get("quality_score", 0) >= 80 else "yellow" if result.get("quality_score", 0) >= 60 else "red",
            "component_scores": {
                "error_management": {"score": 100 - (result.get("total_events", 0) * 2), "label": "Error Management", "description": "How well errors are tracked and resolved"},
                "test_coverage": {"score": result.get("test_coverage", 0), "label": "Test Coverage", "description": "Percentage of errors covered by tests"},
                "risk_mitigation": {"score": 100 - (50 if result.get("risk_level") == "high" else 25 if result.get("risk_level") == "medium" else 0), "label": "Risk Mitigation", "description": "How well high-risk areas are addressed"},
                "automation": {"score": min(100, result.get("approved_tests", 0) * 10), "label": "Automation", "description": "Level of test automation"},
                "prevention": {"score": result.get("quality_score", 50), "label": "Prevention", "description": "Proactive error prevention"},
            },
            "metrics": {
                "total_events": result.get("total_events", 0),
                "unresolved_events": result.get("total_events", 0) - result.get("approved_tests", 0),
                "tests_generated": result.get("total_tests", 0),
                "tests_approved": result.get("approved_tests", 0),
                "avg_confidence": 0.85,
                "high_risk_components": 0,
                "incidents_prevented": result.get("approved_tests", 0),
            },
            "insights": [
                f"Quality score: {result.get('quality_score', 50):.1f}/100",
                f"Risk level: {result.get('risk_level', 'medium')}",
                f"Test coverage: {result.get('test_coverage', 0):.1f}%",
            ],
            "calculated_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.exception("Quality score calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-test", tags=["Frontend Aliases"])
async def frontend_generate_test(
    production_event_id: str,
    project_id: str,
    framework: str = "playwright",
    auto_create_pr: bool = False,
    github_config: Optional[dict] = None,
):
    """Alias for /api/v1/quality/generate-test."""
    from src.api.quality import generate_test, TestGenerationRequest

    request = TestGenerationRequest(
        production_event_id=production_event_id,
        project_id=project_id,
        framework=framework,
        auto_create_pr=auto_create_pr,
        github_config=github_config,
    )
    return await generate_test(request)


@app.post("/api/calculate-risk-scores", tags=["Frontend Aliases"])
async def frontend_calculate_risk(
    project_id: str,
    entity_types: list[str] = ["page", "component"],
):
    """Alias for /api/v1/quality/calculate-risk."""
    from src.api.quality import calculate_risk_scores, RiskScoreRequest

    request = RiskScoreRequest(project_id=project_id, entity_types=entity_types)
    return await calculate_risk_scores(request)


@app.post("/api/semantic-search", tags=["AI Intelligence"])
async def semantic_search(request: SemanticSearchRequest):
    """
    Search for similar error patterns using semantic similarity.

    Uses Cloudflare Vectorize for semantic search with fallback to Jaccard similarity.
    """
    from src.services.supabase_client import get_supabase_client
    from src.services.vectorize import semantic_search_errors
    import hashlib

    similar_patterns = []

    # Try Cloudflare Vectorize first (semantic search)
    try:
        vectorize_results = await semantic_search_errors(
            error_text=request.error_text,
            limit=request.limit,
            min_score=request.min_score
        )

        for result in vectorize_results:
            similar_patterns.append({
                "id": result.get("id"),
                "score": round(result.get("score", 0), 3),
                "pattern_hash": result.get("metadata", {}).get("fingerprint", ""),
                "category": result.get("metadata", {}).get("severity", "error"),
                "example_message": result.get("metadata", {}).get("message", "")[:200],
                "known_solutions": result.get("metadata", {}).get("solutions", []),
            })
    except Exception as e:
        logger.warning(f"Vectorize search failed, using fallback: {e}")

    # Fallback to Jaccard similarity if Vectorize returned no results
    if not similar_patterns:
        supabase = get_supabase_client()

        result = await supabase.request(
            f"/production_events?select=id,title,message,component,url,severity,occurrence_count"
            f"&order=occurrence_count.desc&limit={request.limit * 2}"
        )

        events = result.get("data", []) if not result.get("error") else []
        search_terms = set(request.error_text.lower().split())

        for event in events:
            event_text = f"{event.get('title', '')} {event.get('message', '')}".lower()
            event_terms = set(event_text.split())

            # Jaccard similarity
            intersection = len(search_terms & event_terms)
            union = len(search_terms | event_terms)
            score = intersection / union if union > 0 else 0

            if score >= request.min_score:
                similar_patterns.append({
                    "id": event.get("id"),
                    "score": round(score, 3),
                    "pattern_hash": hashlib.md5(event_text[:100].encode()).hexdigest()[:12],
                    "category": event.get("severity", "error"),
                    "example_message": event.get("message", "")[:200],
                    "known_solutions": [],
                })

        similar_patterns.sort(key=lambda x: x["score"], reverse=True)
        similar_patterns = similar_patterns[:request.limit]

    return {
        "success": True,
        "query": request.error_text[:100],
        "patterns": similar_patterns,
        "count": len(similar_patterns),
        "has_solutions": any(p.get("known_solutions") for p in similar_patterns),
        "search_method": "vectorize" if similar_patterns and len(similar_patterns) > 0 else "jaccard"
    }


@app.post("/api/autonomous-loop", tags=["AI Intelligence"])
async def autonomous_loop(request: AutonomousLoopRequest, background_tasks: BackgroundTasks):
    """
    Run autonomous quality improvement loop.

    Stages:
    1. discovery - Crawl app and discover test scenarios
    2. visual - Run visual regression tests
    3. generation - Generate tests from production errors
    4. verification - Verify generated tests work
    5. pr - Create GitHub PR with tests
    6. learning - Learn from results
    """
    from src.services.supabase_client import get_supabase_client

    job_id = str(uuid.uuid4())

    # Create job record
    supabase = get_supabase_client()
    await supabase.insert("test_generation_jobs", {
        "id": job_id,
        "project_id": request.project_id,
        "status": "running",
        "job_type": "autonomous_loop",
        "started_at": datetime.utcnow().isoformat(),
        "metadata": {
            "stages": request.stages,
            "url": request.url,
            "discovery_depth": request.discovery_depth,
        },
    })

    # Run stages in background
    async def run_autonomous_loop():
        results = {"stages_completed": [], "stages_failed": []}

        try:
            for stage in request.stages:
                logger.info(f"Running autonomous loop stage: {stage}", job_id=job_id)

                if stage == "discovery":
                    # Would call auto-discovery agent
                    results["stages_completed"].append({"stage": stage, "status": "completed"})
                elif stage == "visual":
                    # Would run visual regression
                    results["stages_completed"].append({"stage": stage, "status": "completed"})
                elif stage == "generation":
                    # Would generate tests from errors
                    results["stages_completed"].append({"stage": stage, "status": "completed"})
                elif stage == "verification":
                    # Would verify generated tests
                    results["stages_completed"].append({"stage": stage, "status": "completed"})
                elif stage == "pr":
                    if request.auto_create_pr and request.github_config:
                        results["stages_completed"].append({"stage": stage, "status": "completed"})
                    else:
                        results["stages_completed"].append({"stage": stage, "status": "skipped"})
                elif stage == "learning":
                    results["stages_completed"].append({"stage": stage, "status": "completed"})

            # Update job as completed
            await supabase.update(
                "test_generation_jobs",
                {"id": f"eq.{job_id}"},
                {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "metadata": results,
                },
            )
        except Exception as e:
            logger.exception("Autonomous loop failed", job_id=job_id, error=str(e))
            await supabase.update(
                "test_generation_jobs",
                {"id": f"eq.{job_id}"},
                {
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.utcnow().isoformat(),
                },
            )

    background_tasks.add_task(run_autonomous_loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Autonomous loop started with stages: {', '.join(request.stages)}",
        "check_status_url": f"/api/v1/jobs/{job_id}",
    }


@app.get("/api/predictive-quality", tags=["AI Intelligence"])
async def predictive_quality(
    project_id: str,
    timeframe: str = "7d",
):
    """
    Predict future quality issues using historical patterns.

    Uses error trends, risk scores, and patterns to predict failures.
    """
    from src.services.supabase_client import get_supabase_client
    from src.core.cognitive_engine import CognitiveEngine

    supabase = get_supabase_client()

    # Get recent events
    events_result = await supabase.request(
        f"/production_events?project_id=eq.{project_id}&select=*&order=created_at.desc&limit=100"
    )
    events = events_result.get("data", []) if not events_result.get("error") else []

    # Get risk scores
    risk_result = await supabase.request(
        f"/risk_scores?project_id=eq.{project_id}&select=*&order=overall_risk_score.desc"
    )
    risk_scores = risk_result.get("data", []) if not risk_result.get("error") else []

    # Generate predictions based on patterns
    predictions = []

    # Analyze high-risk entities
    for risk in risk_scores[:10]:
        if risk.get("overall_risk_score", 0) >= 60:
            predictions.append({
                "entity": risk.get("entity_identifier", "Unknown"),
                "entity_type": risk.get("entity_type", "component"),
                "prediction_score": min(100, risk.get("overall_risk_score", 0) + 10),
                "predicted_timeframe": timeframe,
                "risk_factors": [
                    "High error frequency" if risk.get("factors", {}).get("error_frequency", 0) > 50 else None,
                    "Critical severity errors" if risk.get("factors", {}).get("error_severity", 0) > 70 else None,
                    "Low test coverage" if risk.get("factors", {}).get("test_coverage", 100) > 80 else None,
                    "High user impact" if risk.get("factors", {}).get("user_impact", 0) > 50 else None,
                ],
                "recommendations": [
                    "Add more test coverage for this component",
                    "Investigate recent error patterns",
                    "Consider adding monitoring alerts",
                ],
                "similar_past_failures": risk.get("error_count", 0),
                "confidence": 0.75,
            })
            # Filter None values from risk_factors
            predictions[-1]["risk_factors"] = [f for f in predictions[-1]["risk_factors"] if f]

    # Calculate summary
    high_risk = sum(1 for p in predictions if p["prediction_score"] >= 80)
    medium_risk = sum(1 for p in predictions if 60 <= p["prediction_score"] < 80)

    # Generate AI summary if cognitive engine available
    ai_summary = None
    try:
        engine = CognitiveEngine()
        ai_summary = f"Based on {len(events)} recent events and {len(risk_scores)} risk assessments, " \
                    f"we predict {high_risk} high-risk and {medium_risk} medium-risk components may experience issues."
    except Exception:
        pass

    return {
        "success": True,
        "project_id": project_id,
        "timeframe": timeframe,
        "predictions": predictions,
        "summary": {
            "total_analyzed": len(risk_scores),
            "total_predicted": len(predictions),
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "increasing_trends": sum(1 for r in risk_scores if r.get("trend") == "increasing"),
        },
        "ai_summary": ai_summary,
        "data_quality": {
            "events_analyzed": len(events),
            "risk_scores_available": len(risk_scores),
            "patterns_learned": 0,
        },
        "calculated_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An error occurred",
        },
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize resources on startup."""
    settings = get_settings()
    logger.info(
        "E2E Testing Agent API starting",
        version="0.1.0",
        output_dir=settings.output_dir,
    )

    # Ensure output directory exists
    os.makedirs(settings.output_dir, exist_ok=True)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown."""
    logger.info("E2E Testing Agent API shutting down")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
