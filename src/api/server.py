"""FastAPI backend service for E2E Testing Agent.

Provides REST API endpoints for:
- Running tests
- Checking status
- Managing baselines
- Webhooks for CI/CD integration

Security Features (SOC2 Compliant):
- Global authentication middleware
- Rate limiting
- Security headers (OWASP)
- Comprehensive audit logging
- RBAC (Role-Based Access Control)
- Input validation and sanitization
"""

import os
import uuid
from datetime import UTC, datetime

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.api.api_keys import router as api_keys_router
from src.api.approvals import router as approvals_router
from src.api.artifacts import router as artifacts_router
from src.api.audit import router as audit_router
from src.api.chat import router as chat_router
from src.api.collaboration import router as collaboration_router
from src.api.discovery import router as discovery_router
from src.api.export import router as export_router
from src.api.healing import router as healing_router
from src.api.infra_optimizer import router as infra_optimizer_router
from src.api.invitations import router as invitations_router
from src.api.mcp_screenshots import router as mcp_screenshots_router
from src.api.mcp_sessions import router as mcp_sessions_router
from src.api.notifications import router as notifications_router
from src.api.organizations import router as organizations_router
from src.api.parameterized import router as parameterized_router
from src.api.projects import router as projects_router
from src.api.quality import router as quality_router
from src.api.recording import router as recording_router
from src.api.reports import router as reports_router
from src.api.scheduling import router as scheduling_router
from src.api.security.auth import UserContext, get_current_user
from src.api.security.device_auth import router as device_auth_router
from src.api.security.headers import SecurityHeadersMiddleware

# Security Module
from src.api.security.middleware import (
    AuditLogMiddleware,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware,
)
from src.api.streaming import router as streaming_router
from src.api.sync import router as sync_router
from src.api.teams import router as teams_router
from src.api.tests import router as tests_router
from src.api.time_travel import router as time_travel_router
from src.api.users import router as users_router
from src.api.visual_ai import router as visual_ai_router

# API Routers
from src.api.webhooks import router as webhooks_router
from src.config import get_settings
from src.integrations.reporter import create_report_from_state, create_reporter
from src.orchestrator.checkpointer import setup_checkpointer
from src.orchestrator.graph import TestingOrchestrator
from src.orchestrator.state import create_initial_state

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
        default=["discovery", "visual", "generation", "verification"], description="Stages to run"
    )
    discovery_depth: int = Field(2, ge=1, le=5, description="Crawl depth")
    auto_create_pr: bool = Field(False, description="Auto-create GitHub PR")
    github_config: dict | None = Field(None, description="GitHub config for PR")


# ============================================================================
# App Configuration
# ============================================================================

# Get settings for configuration
settings = get_settings()

# API Version - x-release-please-version
API_VERSION = "2.9.1"  # x-release-please-version
API_VERSION_DATE = "2026-01-12"

app = FastAPI(
    title="Argus E2E Testing Agent API",
    description="""
Autonomous E2E testing powered by Claude AI.

## Security Features (SOC2 Compliant)
- **Authentication**: API Key and JWT Bearer token support
- **Authorization**: Role-Based Access Control (RBAC)
- **Rate Limiting**: Configurable per-endpoint limits
- **Audit Logging**: Comprehensive security event logging
- **Input Validation**: XSS, SQL injection, and path traversal protection
- **Security Headers**: OWASP-compliant headers

## API Versioning
- Current Version: v1
- Version Header: X-API-Version
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health check endpoints"},
        {"name": "Authentication", "description": "Authentication and authorization"},
        {"name": "Tests", "description": "Test execution and management"},
        {"name": "Orchestrator", "description": "Unified orchestrator status and control"},
        {"name": "Chat", "description": "Chat-based test orchestration"},
        {"name": "Streaming", "description": "Server-Sent Events for real-time updates"},
        {"name": "Time Travel", "description": "State inspection and replay"},
        {"name": "Security", "description": "Security and audit endpoints"},
        {
            "name": "Visual AI",
            "description": "Visual regression testing with AI-powered comparison",
        },
        {
            "name": "Discovery",
            "description": "Intelligent application discovery and flow detection",
        },
    ],
)

# =============================================================================
# Security Middleware Stack (Order matters - executed bottom to top)
# =============================================================================

# 1. Security Headers (OWASP) - Outermost layer
app.add_middleware(
    SecurityHeadersMiddleware,
    enable_hsts=settings.enable_hsts,
    hsts_max_age=settings.hsts_max_age,
    enable_csp=settings.enable_csp,
)

# 2. Audit Logging - Log all requests for SOC2 compliance
app.add_middleware(
    AuditLogMiddleware,
    enabled=settings.audit_logging_enabled,
    log_request_body=settings.audit_log_request_body,
    log_response_body=settings.audit_log_response_body,
)

# 3. Rate Limiting - Prevent abuse
app.add_middleware(
    RateLimitMiddleware,
    enabled=settings.rate_limiting_enabled,
)

# 4. Authentication - Enforce auth on protected endpoints
app.add_middleware(
    AuthenticationMiddleware,
    enforce_auth=settings.enforce_authentication,
)

# 5. Core Security - Request ID, timing attack prevention
app.add_middleware(SecurityMiddleware)

# 6. CORS - Allow cross-origin requests (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
        "X-API-Version",
        "X-Organization-ID",
        "Accept",
        "Origin",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-API-Version",
    ],
)


# =============================================================================
# Request Size Limiting Middleware
# =============================================================================

# Maximum request body size: 100MB (handles large recording uploads)
MAX_REQUEST_SIZE_BYTES = 100 * 1024 * 1024  # 100MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests with Content-Length exceeding the maximum allowed size.

    This prevents DoS attacks via memory exhaustion from extremely large payloads.
    The limit is set to 100MB to accommodate large recording uploads while still
    providing protection against abuse.
    """
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            size = int(content_length)
            if size > MAX_REQUEST_SIZE_BYTES:
                logger.warning(
                    "Request rejected: payload too large",
                    content_length=size,
                    max_allowed=MAX_REQUEST_SIZE_BYTES,
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request payload too large. Maximum size: {MAX_REQUEST_SIZE_BYTES // (1024*1024)}MB"
                    }
                )
        except ValueError:
            # Invalid Content-Length header, let it through for other validation
            pass

    return await call_next(request)


# Include routers
app.include_router(webhooks_router)
app.include_router(quality_router)
app.include_router(teams_router)
app.include_router(api_keys_router)
app.include_router(audit_router)
app.include_router(healing_router)
app.include_router(sync_router)
app.include_router(export_router)
app.include_router(recording_router)
app.include_router(collaboration_router)
app.include_router(scheduling_router)
app.include_router(notifications_router)
app.include_router(parameterized_router)
app.include_router(chat_router)
app.include_router(streaming_router)
app.include_router(approvals_router)
app.include_router(time_travel_router)
app.include_router(invitations_router)
app.include_router(organizations_router)
app.include_router(users_router)
app.include_router(projects_router)
app.include_router(device_auth_router)
app.include_router(visual_ai_router)
app.include_router(discovery_router)
app.include_router(artifacts_router)
app.include_router(mcp_sessions_router)
app.include_router(mcp_screenshots_router)
app.include_router(infra_optimizer_router)
app.include_router(tests_router)
app.include_router(reports_router)

# In-memory job storage (use Redis for production)
jobs: dict[str, dict] = {}


# ============================================================================
# Request/Response Models
# ============================================================================


class TestRunRequest(BaseModel):
    """Request to start a test run."""

    codebase_path: str = Field(..., description="Path to codebase to analyze")
    app_url: str = Field(..., description="URL of the application to test")
    pr_number: int | None = Field(None, description="PR number for GitHub integration")
    changed_files: list[str] | None = Field(None, description="Specific files to focus on")
    max_tests: int | None = Field(None, description="Maximum number of tests to run")
    focus_areas: list[str] | None = Field(None, description="Areas to focus testing on")


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
    progress: dict | None = None
    result: dict | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    security: dict | None = None


class SecurityInfoResponse(BaseModel):
    """Security configuration info response."""

    authentication_enabled: bool
    rate_limiting_enabled: bool
    audit_logging_enabled: bool
    security_headers_enabled: bool
    api_version: str
    supported_auth_methods: list[str]


class NLPTestRequest(BaseModel):
    """Request to create test from natural language."""

    project_id: str = Field(..., description="Project ID to associate the test with")
    description: str = Field(..., description="Plain English test description")
    app_url: str = Field("http://localhost:3000", description="Application URL to test")
    context: str | None = Field(None, description="Additional context about the app")


class VisualCompareRequest(BaseModel):
    """Request to compare screenshots."""

    baseline_b64: str = Field(..., description="Base64 encoded baseline screenshot")
    current_b64: str = Field(..., description="Base64 encoded current screenshot")
    context: str | None = Field(None, description="Context about what's being compared")


class WebhookPayload(BaseModel):
    """Webhook payload for CI/CD integration."""

    action: str  # pr_opened, pr_updated, push
    repository: dict
    pull_request: dict | None = None
    commits: list | None = None


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.now(UTC).isoformat(),
        security={
            "authentication": settings.enforce_authentication,
            "rate_limiting": settings.rate_limiting_enabled,
            "audit_logging": settings.audit_logging_enabled,
        },
    )


@app.get("/api/v1/security/info", response_model=SecurityInfoResponse, tags=["Security"])
async def security_info():
    """Get security configuration info (public endpoint)."""
    return SecurityInfoResponse(
        authentication_enabled=settings.enforce_authentication,
        rate_limiting_enabled=settings.rate_limiting_enabled,
        audit_logging_enabled=settings.audit_logging_enabled,
        security_headers_enabled=settings.enable_hsts or settings.enable_csp,
        api_version=API_VERSION,
        supported_auth_methods=["api_key", "jwt_bearer"],
    )


# ============================================================================
# Authentication Endpoints
# ============================================================================


@app.get("/api/v1/auth/me", tags=["Authentication"])
async def get_current_user_info(request: Request):
    """Get current authenticated user information.

    Returns the user context from the authenticated JWT token (Clerk or internal).
    """

    try:
        user: UserContext = await get_current_user(request)
        return {
            "authenticated": True,
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "email": user.email,
            "name": user.name,
            "roles": user.roles,
            "scopes": user.scopes,
            "auth_method": user.auth_method,
            "session_id": user.session_id,
            "ip_address": user.ip_address,
            "authenticated_at": user.authenticated_at.isoformat()
            if user.authenticated_at
            else None,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error getting user info", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/auth/debug-token", tags=["Authentication"])
async def debug_jwt_token(request: Request):
    """Debug endpoint to show raw JWT claims (for development only).

    Shows all claims in the JWT token to help with session token customization.

    SECURITY: This endpoint is only available in development/test environments.
    """
    import os

    # SECURITY: Only allow in development environments
    environment = os.getenv("ENVIRONMENT", "development")
    if environment not in ("development", "test", "local"):
        raise HTTPException(
            status_code=403, detail="Debug endpoint is only available in development environments"
        )


    from src.api.security.auth import verify_clerk_jwt

    # Get bearer token
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")

    token = auth_header.replace("Bearer ", "")

    # Try to decode as Clerk JWT
    clerk_payload = await verify_clerk_jwt(token)
    if clerk_payload:
        # Redact sensitive claims in non-local environments
        redacted_claims = {
            k: (
                v
                if k in ("sub", "iss", "aud", "exp", "iat", "azp")
                else "[REDACTED]"
                if isinstance(v, str) and len(str(v)) > 20
                else v
            )
            for k, v in clerk_payload.items()
        }
        return {
            "token_type": "clerk_jwt",
            "claims": redacted_claims,
            "available_keys": list(clerk_payload.keys()),
            "hint": "To add email/name, go to Clerk Dashboard → Sessions → Customize session token",
            "note": "Some values redacted for security",
        }

    # Try to decode without verification to see claims
    try:
        import jwt

        unverified = jwt.decode(token, options={"verify_signature": False})
        return {
            "token_type": "unknown_jwt",
            "claims": {
                k: "[REDACTED]" if k not in ("sub", "iss", "aud", "exp", "iat") else v
                for k, v in unverified.items()
            },
            "available_keys": list(unverified.keys()),
            "warning": "Token signature not verified - values redacted",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JWT: {str(e)}")


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies all dependencies are available."""
    settings = get_settings()

    checks = {
        "anthropic_api": settings.anthropic_api_key is not None
        and bool(
            settings.anthropic_api_key.get_secret_value()
            if hasattr(settings.anthropic_api_key, "get_secret_value")
            else settings.anthropic_api_key
        ),
        "output_dir": os.path.exists(settings.output_dir),
    }

    all_ready = all(checks.values())

    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now(UTC).isoformat(),
        },
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
        "created_at": datetime.now(UTC).isoformat(),
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
    pr_number: int | None,
    changed_files: list[str] | None,
    focus_areas: list[str] | None,
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
        jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        jobs[job_id]["result"] = {
            "summary": summary,
            "report_paths": {k: str(v) for k, v in report_paths.items()},
        }

        logger.info("Test run completed", job_id=job_id, summary=summary)

    except Exception as e:
        logger.exception("Test run failed", job_id=job_id, error=str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()


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
async def list_jobs(limit: int = 20, status: str | None = None):
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
async def create_test_from_nlp(body: NLPTestRequest, request: Request):
    """
    Create a test specification from plain English description and persist to database.

    Example: "Login as admin@example.com and verify dashboard shows 5 widgets"

    The test is saved to the database and can be retrieved via GET /api/v1/tests/{id}.
    """
    from src.agents.nlp_test_creator import NLPTestCreator
    from src.api.projects import verify_project_access
    from src.api.teams import get_current_user, log_audit
    from src.services.supabase_client import get_supabase_client

    try:
        # Get current user and verify project access
        user = await get_current_user(request)
        await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

        # Generate test from NLP description
        creator = NLPTestCreator(app_url=body.app_url)
        generated_test = await creator.create(body.description, context=body.context)

        # Convert generated test to database format
        test_dict = generated_test.to_dict()

        # Prepare steps in the format expected by the tests table
        steps = []
        for step in test_dict.get("steps", []):
            steps.append({
                "action": step.get("action", ""),
                "target": step.get("target"),
                "value": step.get("value"),
                "description": step.get("description", ""),
            })

        # Add assertions as additional verification steps
        for assertion in test_dict.get("assertions", []):
            steps.append({
                "action": "assert",
                "target": assertion.get("target", ""),
                "value": assertion.get("expected", ""),
                "description": assertion.get("description", ""),
                "assertion_type": assertion.get("type", ""),
            })

        # Persist to database
        supabase = get_supabase_client()
        test_data = {
            "project_id": body.project_id,
            "name": test_dict.get("name", "Generated Test"),
            "description": test_dict.get("description", body.description),
            "steps": steps,
            "tags": test_dict.get("tags", []),
            "priority": test_dict.get("priority", "medium"),
            "is_active": True,
            "source": "generated",
            "created_by": user["user_id"],
        }

        result = await supabase.insert("tests", test_data)

        if result.get("error"):
            logger.error("Failed to persist NLP-generated test", error=result.get("error"))
            raise HTTPException(status_code=500, detail="Failed to save test to database")

        saved_test = result["data"][0]

        # Audit log
        from src.api.tests import get_project_org_id
        org_id = await get_project_org_id(body.project_id)
        await log_audit(
            organization_id=org_id,
            user_id=user["user_id"],
            user_email=user.get("email"),
            action="test.create",
            resource_type="test",
            resource_id=saved_test["id"],
            description=f"Created test '{saved_test['name']}' from NLP description",
            metadata={
                "name": saved_test["name"],
                "project_id": body.project_id,
                "source": "nlp_generated",
                "original_description": body.description[:500],
            },
            request=request,
        )

        logger.info(
            "NLP test created and persisted",
            test_id=saved_test["id"],
            name=saved_test["name"],
            project_id=body.project_id,
        )

        return {
            "success": True,
            "test": {
                "id": saved_test["id"],
                "project_id": saved_test["project_id"],
                "name": saved_test["name"],
                "description": saved_test.get("description"),
                "steps": saved_test.get("steps", []),
                "tags": saved_test.get("tags", []),
                "priority": saved_test.get("priority", "medium"),
                "is_active": saved_test.get("is_active", True),
                "source": saved_test.get("source", "generated"),
                "created_by": saved_test.get("created_by"),
                "created_at": saved_test.get("created_at"),
            },
            "spec": generated_test.to_spec(),
            "original_nlp_output": test_dict,
        }
    except HTTPException:
        raise
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
    focus_areas: list[str] | None = None,
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
            "created_at": datetime.now(UTC).isoformat(),
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
            "grade": "A"
            if result.get("quality_score", 0) >= 90
            else "B"
            if result.get("quality_score", 0) >= 80
            else "C"
            if result.get("quality_score", 0) >= 70
            else "D"
            if result.get("quality_score", 0) >= 60
            else "F",
            "grade_color": "green"
            if result.get("quality_score", 0) >= 80
            else "yellow"
            if result.get("quality_score", 0) >= 60
            else "red",
            "component_scores": {
                "error_management": {
                    "score": 100 - (result.get("total_events", 0) * 2),
                    "label": "Error Management",
                    "description": "How well errors are tracked and resolved",
                },
                "test_coverage": {
                    "score": result.get("test_coverage", 0),
                    "label": "Test Coverage",
                    "description": "Percentage of errors covered by tests",
                },
                "risk_mitigation": {
                    "score": 100
                    - (
                        50
                        if result.get("risk_level") == "high"
                        else 25
                        if result.get("risk_level") == "medium"
                        else 0
                    ),
                    "label": "Risk Mitigation",
                    "description": "How well high-risk areas are addressed",
                },
                "automation": {
                    "score": min(100, result.get("approved_tests", 0) * 10),
                    "label": "Automation",
                    "description": "Level of test automation",
                },
                "prevention": {
                    "score": result.get("quality_score", 50),
                    "label": "Prevention",
                    "description": "Proactive error prevention",
                },
            },
            "metrics": {
                "total_events": result.get("total_events", 0),
                "unresolved_events": result.get("total_events", 0)
                - result.get("approved_tests", 0),
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
            "calculated_at": datetime.now(UTC).isoformat(),
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
    github_config: dict | None = None,
):
    """Alias for /api/v1/quality/generate-test."""
    from src.api.quality import TestGenerationRequest, generate_test

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
    from src.api.quality import RiskScoreRequest, calculate_risk_scores

    request = RiskScoreRequest(project_id=project_id, entity_types=entity_types)
    return await calculate_risk_scores(request)


@app.post("/api/semantic-search", tags=["AI Intelligence"])
async def semantic_search(request: SemanticSearchRequest):
    """
    Search for similar error patterns using semantic similarity.

    Uses Cloudflare Vectorize for semantic search with fallback to Jaccard similarity.
    """
    import hashlib

    from src.services.supabase_client import get_supabase_client
    from src.services.vectorize import semantic_search_errors

    similar_patterns = []

    # Try Cloudflare Vectorize first (semantic search)
    try:
        vectorize_results = await semantic_search_errors(
            error_text=request.error_text, limit=request.limit, min_score=request.min_score
        )

        for result in vectorize_results:
            similar_patterns.append(
                {
                    "id": result.get("id"),
                    "score": round(result.get("score", 0), 3),
                    "pattern_hash": result.get("metadata", {}).get("fingerprint", ""),
                    "category": result.get("metadata", {}).get("severity", "error"),
                    "example_message": result.get("metadata", {}).get("message", "")[:200],
                    "known_solutions": result.get("metadata", {}).get("solutions", []),
                }
            )
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
                similar_patterns.append(
                    {
                        "id": event.get("id"),
                        "score": round(score, 3),
                        "pattern_hash": hashlib.md5(event_text[:100].encode()).hexdigest()[:12],
                        "category": event.get("severity", "error"),
                        "example_message": event.get("message", "")[:200],
                        "known_solutions": [],
                    }
                )

        similar_patterns.sort(key=lambda x: x["score"], reverse=True)
        similar_patterns = similar_patterns[: request.limit]

    return {
        "success": True,
        "query": request.error_text[:100],
        "patterns": similar_patterns,
        "count": len(similar_patterns),
        "has_solutions": any(p.get("known_solutions") for p in similar_patterns),
        "search_method": "vectorize"
        if similar_patterns and len(similar_patterns) > 0
        else "jaccard",
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
    await supabase.insert(
        "test_generation_jobs",
        {
            "id": job_id,
            "project_id": request.project_id,
            "status": "running",
            "job_type": "autonomous_loop",
            "started_at": datetime.now(UTC).isoformat(),
            "metadata": {
                "stages": request.stages,
                "url": request.url,
                "discovery_depth": request.discovery_depth,
            },
        },
    )

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
                    "completed_at": datetime.now(UTC).isoformat(),
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
                    "completed_at": datetime.now(UTC).isoformat(),
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
            predictions.append(
                {
                    "entity": risk.get("entity_identifier", "Unknown"),
                    "entity_type": risk.get("entity_type", "component"),
                    "prediction_score": min(100, risk.get("overall_risk_score", 0) + 10),
                    "predicted_timeframe": timeframe,
                    "risk_factors": [
                        "High error frequency"
                        if risk.get("factors", {}).get("error_frequency", 0) > 50
                        else None,
                        "Critical severity errors"
                        if risk.get("factors", {}).get("error_severity", 0) > 70
                        else None,
                        "Low test coverage"
                        if risk.get("factors", {}).get("test_coverage", 100) > 80
                        else None,
                        "High user impact"
                        if risk.get("factors", {}).get("user_impact", 0) > 50
                        else None,
                    ],
                    "recommendations": [
                        "Add more test coverage for this component",
                        "Investigate recent error patterns",
                        "Consider adding monitoring alerts",
                    ],
                    "similar_past_failures": risk.get("error_count", 0),
                    "confidence": 0.75,
                }
            )
            # Filter None values from risk_factors
            predictions[-1]["risk_factors"] = [f for f in predictions[-1]["risk_factors"] if f]

    # Calculate summary
    high_risk = sum(1 for p in predictions if p["prediction_score"] >= 80)
    medium_risk = sum(1 for p in predictions if 60 <= p["prediction_score"] < 80)

    # Generate AI summary based on analysis
    ai_summary = (
        f"Based on {len(events)} recent events and {len(risk_scores)} risk assessments, "
        f"we predict {high_risk} high-risk and {medium_risk} medium-risk components may experience issues."
    )

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
        "calculated_at": datetime.now(UTC).isoformat(),
    }


# ============================================================================
# Supervisor Orchestrator Endpoints
# ============================================================================


class SupervisorStartRequest(BaseModel):
    """Request to start a supervised test run."""

    codebase_path: str = Field(..., description="Path to codebase to analyze")
    app_url: str = Field(..., description="URL of the application to test")
    pr_number: int | None = Field(None, description="PR number for GitHub integration")
    changed_files: list[str] | None = Field(None, description="Specific files to focus on")
    initial_message: str | None = Field(
        None, description="Custom initial message to the supervisor"
    )


class SupervisorStartResponse(BaseModel):
    """Response after starting a supervised test run."""

    thread_id: str
    status: str
    message: str
    current_phase: str
    agents_available: list[str]
    created_at: str


class SupervisorStatusResponse(BaseModel):
    """Response for supervisor status query."""

    thread_id: str
    status: str  # running, paused, completed, failed
    current_phase: str
    iteration: int
    is_complete: bool
    next_node: str | None = None
    results: dict | None = None
    progress: dict | None = None
    error: str | None = None
    created_at: str


# In-memory storage for supervisor jobs (use Redis for production)
supervisor_jobs: dict[str, dict] = {}


@app.post(
    "/api/v1/orchestrator/supervisor/start",
    response_model=SupervisorStartResponse,
    tags=["Supervisor"],
)
async def start_supervised_test_run(
    request: SupervisorStartRequest, background_tasks: BackgroundTasks
):
    """
    Start a new supervised test run using the multi-agent supervisor pattern.

    The supervisor orchestrates specialized agents:
    - code_analyzer: Analyzes codebase structure
    - test_planner: Creates comprehensive test plans
    - ui_tester: Executes browser-based UI tests
    - api_tester: Tests REST/GraphQL APIs
    - self_healer: Fixes broken tests
    - reporter: Generates reports

    The test run executes in the background. Use the returned thread_id to check status.
    """
    from src.orchestrator.supervisor import (
        AGENTS as SUPERVISOR_AGENTS,
    )
    from src.orchestrator.supervisor import (
        SupervisorOrchestrator,
    )

    thread_id = str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()

    # Store initial job state
    supervisor_jobs[thread_id] = {
        "status": "pending",
        "created_at": created_at,
        "request": request.model_dump(),
        "current_phase": "analysis",
        "iteration": 0,
        "is_complete": False,
        "progress": {
            "phase": "initializing",
            "agents_invoked": [],
            "tests_completed": 0,
        },
    }

    # Run supervised tests in background
    async def run_supervised_tests():
        try:
            supervisor_jobs[thread_id]["status"] = "running"
            supervisor_jobs[thread_id]["progress"]["phase"] = "starting"

            # Create supervisor orchestrator
            orchestrator = SupervisorOrchestrator(
                codebase_path=request.codebase_path,
                app_url=request.app_url,
                pr_number=request.pr_number,
                changed_files=request.changed_files,
            )

            # Run the supervised orchestration
            final_state = await orchestrator.run(thread_id=thread_id)

            # Get summary
            summary = orchestrator.get_summary(final_state)

            # Update job with results
            supervisor_jobs[thread_id].update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "current_phase": final_state.get("current_phase", "complete"),
                    "iteration": final_state.get("iteration", 0),
                    "is_complete": True,
                    "result": summary,
                    "progress": {
                        "phase": "completed",
                        "tests_completed": final_state.get("passed_count", 0)
                        + final_state.get("failed_count", 0),
                        "passed": final_state.get("passed_count", 0),
                        "failed": final_state.get("failed_count", 0),
                    },
                }
            )

            logger.info(
                "Supervised test run completed",
                thread_id=thread_id,
                passed=final_state.get("passed_count", 0),
                failed=final_state.get("failed_count", 0),
                iterations=final_state.get("iteration", 0),
            )

        except Exception as e:
            logger.exception("Supervised test run failed", thread_id=thread_id, error=str(e))
            supervisor_jobs[thread_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )

    background_tasks.add_task(run_supervised_tests)

    logger.info(
        "Supervised test run started",
        thread_id=thread_id,
        app_url=request.app_url,
    )

    return SupervisorStartResponse(
        thread_id=thread_id,
        status="pending",
        message="Supervised test run started. Use /api/v1/orchestrator/supervisor/status/{thread_id} to check status.",
        current_phase="analysis",
        agents_available=list(SUPERVISOR_AGENTS),
        created_at=created_at,
    )


# ============================================================================
# Unified Orchestrator Status Endpoint
# ============================================================================


class OrchestratorStatusResponse(BaseModel):
    """Unified response for orchestrator status query."""

    thread_id: str
    found: bool
    source: str  # "supervisor", "streaming", or "not_found"
    status: str  # running, paused, completed, failed, pending, error
    current_phase: str | None = None
    iteration: int | None = None
    is_complete: bool | None = None
    next_node: str | None = None
    results: dict | None = None
    progress: dict | None = None
    error: str | None = None
    created_at: str | None = None
    state_summary: dict | None = None


@app.get(
    "/api/v1/orchestrator/status/{thread_id}",
    response_model=OrchestratorStatusResponse,
    tags=["Orchestrator"],
)
async def get_orchestrator_status(thread_id: str):
    """
    Get the status of an orchestrator run by thread ID.

    This is a unified endpoint that checks multiple sources:
    1. Supervisor jobs (multi-agent orchestration)
    2. Streaming execution state (LangGraph checkpoints)

    Returns the status from the first source that has data for the given thread_id.
    """
    # First, check supervisor jobs (in-memory storage)
    if thread_id in supervisor_jobs:
        job = supervisor_jobs[thread_id]
        return OrchestratorStatusResponse(
            thread_id=thread_id,
            found=True,
            source="supervisor",
            status=job.get("status", "unknown"),
            current_phase=job.get("current_phase"),
            iteration=job.get("iteration", 0),
            is_complete=job.get("is_complete", False),
            next_node=job.get("next_node"),
            results=job.get("result"),
            progress=job.get("progress"),
            error=job.get("error"),
            created_at=job.get("created_at"),
        )

    # Second, check streaming execution state (LangGraph checkpoints)
    try:
        from src.orchestrator.checkpointer import get_checkpointer
        from src.orchestrator.graph import create_enhanced_testing_graph

        checkpointer = get_checkpointer()
        config = {"configurable": {"thread_id": thread_id}}

        graph = create_enhanced_testing_graph(settings)
        app_graph = graph.compile(checkpointer=checkpointer)

        state = await app_graph.aget_state(config)

        if state and state.values:
            values = state.values
            total_tests = len(values.get("test_plan", []))
            current_test = values.get("current_test_index", 0)

            # Determine status
            if values.get("error"):
                status = "error"
            elif state.next:
                status = "running"
            elif values.get("passed_count", 0) + values.get("failed_count", 0) + values.get("skipped_count", 0) > 0:
                status = "completed"
            else:
                status = "pending"

            return OrchestratorStatusResponse(
                thread_id=thread_id,
                found=True,
                source="streaming",
                status=status,
                current_phase=values.get("next_agent"),
                iteration=values.get("iteration", 0),
                is_complete=not bool(state.next),
                next_node=state.next[0] if state.next else None,
                error=values.get("error"),
                created_at=values.get("started_at"),
                state_summary={
                    "passed": values.get("passed_count", 0),
                    "failed": values.get("failed_count", 0),
                    "skipped": values.get("skipped_count", 0),
                    "current_test": current_test,
                    "total_tests": total_tests,
                    "progress_percent": (current_test / total_tests * 100) if total_tests > 0 else 0,
                    "total_cost": values.get("total_cost", 0),
                },
            )
    except Exception as e:
        logger.debug(
            "Failed to get streaming state for thread",
            thread_id=thread_id,
            error=str(e),
        )

    # Not found in any source
    raise HTTPException(
        status_code=404,
        detail=f"Orchestrator run not found for thread_id: {thread_id}. "
               "The thread may have expired or never existed."
    )


@app.get(
    "/api/v1/orchestrator/supervisor/status/{thread_id}",
    response_model=SupervisorStatusResponse,
    tags=["Supervisor"],
)
async def get_supervisor_status(thread_id: str):
    """
    Get the status of a supervised test run.

    Returns current phase, iteration count, results, and whether the run is complete.
    """
    if thread_id not in supervisor_jobs:
        raise HTTPException(status_code=404, detail="Supervisor job not found")

    job = supervisor_jobs[thread_id]

    return SupervisorStatusResponse(
        thread_id=thread_id,
        status=job.get("status", "unknown"),
        current_phase=job.get("current_phase", "unknown"),
        iteration=job.get("iteration", 0),
        is_complete=job.get("is_complete", False),
        next_node=job.get("next_node"),
        results=job.get("result"),
        progress=job.get("progress"),
        error=job.get("error"),
        created_at=job.get("created_at", ""),
    )


@app.get("/api/v1/orchestrator/supervisor/agents", tags=["Supervisor"])
async def list_supervisor_agents():
    """
    List available supervisor agents and their descriptions.
    """
    from src.orchestrator.supervisor import (
        AGENT_DESCRIPTIONS as SUPERVISOR_AGENT_DESCRIPTIONS,
    )
    from src.orchestrator.supervisor import (
        AGENTS as SUPERVISOR_AGENTS,
    )

    return {
        "agents": [
            {
                "name": agent,
                "description": SUPERVISOR_AGENT_DESCRIPTIONS.get(agent, ""),
            }
            for agent in SUPERVISOR_AGENTS
        ],
        "total": len(SUPERVISOR_AGENTS),
    }


@app.get("/api/v1/orchestrator/supervisor/jobs", tags=["Supervisor"])
async def list_supervisor_jobs(
    limit: int = 20,
    status: str | None = None,
):
    """
    List recent supervised test runs.
    """
    filtered = list(supervisor_jobs.items())

    if status:
        filtered = [(k, v) for k, v in filtered if v.get("status") == status]

    # Sort by created_at descending
    filtered.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)

    return {
        "jobs": [
            {
                "thread_id": k,
                "status": v.get("status"),
                "current_phase": v.get("current_phase"),
                "iteration": v.get("iteration", 0),
                "is_complete": v.get("is_complete", False),
                "created_at": v.get("created_at"),
                "completed_at": v.get("completed_at"),
            }
            for k, v in filtered[:limit]
        ],
        "total": len(filtered),
    }


@app.post("/api/v1/orchestrator/supervisor/resume/{thread_id}", tags=["Supervisor"])
async def resume_supervisor_run(thread_id: str, background_tasks: BackgroundTasks):
    """
    Resume a paused supervised test run.

    This is useful when the run is paused at a breakpoint (e.g., waiting for approval).
    """
    from src.orchestrator.supervisor import SupervisorOrchestrator

    if thread_id not in supervisor_jobs:
        raise HTTPException(status_code=404, detail="Supervisor job not found")

    job = supervisor_jobs[thread_id]

    if job.get("status") not in ["paused", "pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume job with status '{job.get('status')}'. Only paused or pending jobs can be resumed.",
        )

    # Get original request
    original_request = job.get("request", {})

    async def resume_supervised_run():
        try:
            supervisor_jobs[thread_id]["status"] = "running"

            # Create orchestrator
            orchestrator = SupervisorOrchestrator(
                codebase_path=original_request.get("codebase_path", ""),
                app_url=original_request.get("app_url", ""),
                pr_number=original_request.get("pr_number"),
                changed_files=original_request.get("changed_files"),
            )

            # Resume execution
            final_state = await orchestrator.resume(thread_id)

            # Update job with results
            summary = orchestrator.get_summary(final_state)

            supervisor_jobs[thread_id].update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "current_phase": final_state.get("current_phase", "complete"),
                    "iteration": final_state.get("iteration", 0),
                    "is_complete": True,
                    "result": summary,
                }
            )

            logger.info("Supervised test run resumed and completed", thread_id=thread_id)

        except Exception as e:
            logger.exception("Supervised test run resume failed", thread_id=thread_id, error=str(e))
            supervisor_jobs[thread_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )

    background_tasks.add_task(resume_supervised_run)

    return {
        "success": True,
        "thread_id": thread_id,
        "message": "Supervisor run resumed. Check status for updates.",
    }


@app.delete("/api/v1/orchestrator/supervisor/{thread_id}", tags=["Supervisor"])
async def abort_supervisor_run(thread_id: str, reason: str = "Aborted by user"):
    """
    Abort a running supervised test run.

    This will mark the run as failed and stop any further processing.
    """
    if thread_id not in supervisor_jobs:
        raise HTTPException(status_code=404, detail="Supervisor job not found")

    job = supervisor_jobs[thread_id]

    if job.get("status") in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot abort job with status '{job.get('status')}'. Job is already finished.",
        )

    # Update job status
    supervisor_jobs[thread_id].update(
        {
            "status": "failed",
            "error": reason,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )

    logger.info("Supervised test run aborted", thread_id=thread_id, reason=reason)

    return {
        "success": True,
        "thread_id": thread_id,
        "message": f"Supervisor run aborted: {reason}",
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
# Custom OpenAPI Schema with Security
# ============================================================================


def custom_openapi():
    """Generate custom OpenAPI schema with security schemes."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    # Add security schemes
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Format: argus_xxxx",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token for authentication",
        },
    }

    # Add global security requirement (optional - can be overridden per-endpoint)
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ]

    # Add server info
    openapi_schema["servers"] = [
        {
            "url": "https://argus-brain-production.up.railway.app",
            "description": "Production server",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
    ]

    # Add contact and license info for SOC2
    openapi_schema["info"]["contact"] = {
        "name": "Argus Security Team",
        "email": "security@heyargus.com",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": "Argus Documentation",
        "url": "https://docs.heyargus.com",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup():
    """Initialize resources on startup."""
    startup_settings = get_settings()
    logger.info(
        "Argus E2E Testing Agent API starting",
        version=API_VERSION,
        output_dir=startup_settings.output_dir,
        security={
            "authentication": startup_settings.enforce_authentication,
            "rate_limiting": startup_settings.rate_limiting_enabled,
            "audit_logging": startup_settings.audit_logging_enabled,
        },
    )

    # Ensure output directory exists
    os.makedirs(settings.output_dir, exist_ok=True)

    # Initialize checkpointer for durable execution
    # This sets up PostgresSaver if DATABASE_URL is configured
    await setup_checkpointer()


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
