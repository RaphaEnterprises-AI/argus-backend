"""Accessibility API for WCAG compliance testing and auditing.

Provides endpoints for:
- Running accessibility audits on URLs
- Fetching audit results and history
- Managing accessibility issues
- WCAG compliance reporting
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.middleware.tenant import validate_uuid, validate_uuid_optional
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/accessibility", tags=["Accessibility"])


# =============================================================================
# Pydantic Models
# =============================================================================


class AccessibilityIssueResponse(BaseModel):
    """Response model for accessibility issue."""

    id: str
    audit_id: str
    rule: str
    severity: Literal["critical", "serious", "moderate", "minor"]
    element_selector: str | None = None
    description: str
    suggested_fix: str | None = None
    wcag_criteria: list[str] | None = None
    created_at: str


class QualityAuditResponse(BaseModel):
    """Response model for quality audit."""

    id: str
    project_id: str
    url: str
    status: Literal["pending", "running", "completed", "failed"]
    accessibility_score: float | None = None
    performance_score: float | None = None
    best_practices_score: float | None = None
    seo_score: float | None = None
    lcp_ms: float | None = None
    fid_ms: float | None = None
    cls: float | None = None
    ttfb_ms: float | None = None
    fcp_ms: float | None = None
    tti_ms: float | None = None
    started_at: str | None = None
    completed_at: str | None = None
    triggered_by: str | None = None
    created_at: str


class AccessibilityAuditResponse(BaseModel):
    """Response model for accessibility audit with issues."""

    audit: QualityAuditResponse
    issues: list[AccessibilityIssueResponse]


class RunAuditRequest(BaseModel):
    """Request to run an accessibility audit."""

    project_id: str = Field(..., description="Project ID")
    url: str = Field(..., description="URL to audit")
    wcag_level: Literal["A", "AA", "AAA"] = Field("AA", description="WCAG conformance level")
    include_best_practices: bool = Field(True, description="Include best practices checks")
    triggered_by: str | None = Field(None, description="User ID who triggered the audit")


class RunAuditResponse(BaseModel):
    """Response after starting an accessibility audit."""

    success: bool
    message: str
    audit_id: str | None = None
    status: str


# =============================================================================
# Helper Functions
# =============================================================================


def map_audit_response(audit: dict) -> QualityAuditResponse:
    """Map database audit row to response model."""
    return QualityAuditResponse(
        id=audit["id"],
        project_id=audit["project_id"],
        url=audit["url"],
        status=audit["status"],
        accessibility_score=audit.get("accessibility_score"),
        performance_score=audit.get("performance_score"),
        best_practices_score=audit.get("best_practices_score"),
        seo_score=audit.get("seo_score"),
        lcp_ms=audit.get("lcp_ms"),
        fid_ms=audit.get("fid_ms"),
        cls=audit.get("cls"),
        ttfb_ms=audit.get("ttfb_ms"),
        fcp_ms=audit.get("fcp_ms"),
        tti_ms=audit.get("tti_ms"),
        started_at=audit.get("started_at"),
        completed_at=audit.get("completed_at"),
        triggered_by=audit.get("triggered_by"),
        created_at=audit["created_at"],
    )


def map_issue_response(issue: dict) -> AccessibilityIssueResponse:
    """Map database issue row to response model."""
    return AccessibilityIssueResponse(
        id=issue["id"],
        audit_id=issue["audit_id"],
        rule=issue["rule"],
        severity=issue["severity"],
        element_selector=issue.get("element_selector"),
        description=issue["description"],
        suggested_fix=issue.get("suggested_fix"),
        wcag_criteria=issue.get("wcag_criteria"),
        created_at=issue["created_at"],
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/audit/latest")
async def get_latest_audit(
    project_id: str = Query(..., description="Project ID"),
) -> AccessibilityAuditResponse | None:
    """
    Get the latest completed accessibility audit for a project.

    Returns the audit along with all associated accessibility issues.
    """
    validate_uuid(project_id, "project_id")
    supabase = get_supabase_client()

    try:
        # Get latest completed audit
        audit_result = await supabase.request(
            f"/quality_audits?project_id=eq.{project_id}&status=eq.completed"
            f"&order=created_at.desc&limit=1"
        )

        if audit_result.get("error"):
            error_msg = str(audit_result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return None
            raise HTTPException(status_code=500, detail="Failed to fetch audit")

        audits = audit_result.get("data") or []
        if not audits:
            return None

        audit = audits[0]

        # Get accessibility issues for this audit
        issues_result = await supabase.request(
            f"/accessibility_issues?audit_id=eq.{audit['id']}&order=severity.asc"
        )

        if issues_result.get("error"):
            error_msg = str(issues_result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                issues = []
            else:
                raise HTTPException(status_code=500, detail="Failed to fetch issues")
        else:
            issues = issues_result.get("data") or []

        return AccessibilityAuditResponse(
            audit=map_audit_response(audit),
            issues=[map_issue_response(i) for i in issues],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get latest audit", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get audit: {str(e)}")


@router.get("/audit/{audit_id}")
async def get_audit(
    audit_id: str,
) -> AccessibilityAuditResponse:
    """
    Get a specific accessibility audit by ID.

    Returns the audit along with all associated accessibility issues.
    """
    validate_uuid(audit_id, "audit_id")
    supabase = get_supabase_client()

    try:
        # Get audit
        audit_result = await supabase.request(f"/quality_audits?id=eq.{audit_id}")

        if audit_result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch audit")

        audits = audit_result.get("data") or []
        if not audits:
            raise HTTPException(status_code=404, detail="Audit not found")

        audit = audits[0]

        # Get accessibility issues for this audit
        issues_result = await supabase.request(
            f"/accessibility_issues?audit_id=eq.{audit_id}&order=severity.asc"
        )

        issues = issues_result.get("data") or [] if not issues_result.get("error") else []

        return AccessibilityAuditResponse(
            audit=map_audit_response(audit),
            issues=[map_issue_response(i) for i in issues],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get audit", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get audit: {str(e)}")


@router.get("/audits/history")
async def get_audit_history(
    project_id: str = Query(..., description="Project ID"),
    limit: int = Query(10, le=100, description="Max results"),
    offset: int = Query(0, description="Offset for pagination"),
) -> dict:
    """
    Get accessibility audit history for a project.

    Returns a list of past audits ordered by creation date (newest first).
    """
    validate_uuid(project_id, "project_id")
    supabase = get_supabase_client()

    try:
        result = await supabase.request(
            f"/quality_audits?project_id=eq.{project_id}"
            f"&order=created_at.desc&limit={limit}&offset={offset}"
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"audits": [], "total": 0}
            raise HTTPException(status_code=500, detail="Failed to fetch audits")

        audits = result.get("data") or []

        return {
            "audits": [map_audit_response(a) for a in audits],
            "total": len(audits),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get audit history", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/issues")
async def get_issues(
    audit_id: str = Query(..., description="Audit ID"),
    severity: str | None = Query(None, description="Filter by severity"),
) -> dict:
    """
    Get accessibility issues for a specific audit.
    """
    validate_uuid(audit_id, "audit_id")
    supabase = get_supabase_client()

    try:
        query = f"/accessibility_issues?audit_id=eq.{audit_id}&order=severity.asc"
        if severity:
            query += f"&severity=eq.{severity}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"issues": []}
            raise HTTPException(status_code=500, detail="Failed to fetch issues")

        issues = result.get("data") or []

        return {"issues": [map_issue_response(i) for i in issues]}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get issues", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get issues: {str(e)}")


@router.post("/audit")
async def run_audit(request: RunAuditRequest) -> RunAuditResponse:
    """
    Run an accessibility audit on a URL.

    Creates an audit record and triggers the accessibility analysis.
    The audit runs asynchronously and results can be fetched later.
    """
    validate_uuid(request.project_id, "project_id")
    validate_uuid_optional(request.triggered_by, "triggered_by")
    supabase = get_supabase_client()

    try:
        # Create audit record with 'running' status
        audit_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        audit_data = {
            "id": audit_id,
            "project_id": request.project_id,
            "url": request.url,
            "status": "running",
            "started_at": now,
            "triggered_by": request.triggered_by,
            "created_at": now,
        }

        result = await supabase.insert("quality_audits", audit_data)

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create audit")

        # TODO: Trigger actual accessibility analysis via browser pool
        # For now, we'll rely on the frontend fallback to browser/observe
        # In production, this would trigger an async job using Kafka

        logger.info(
            "Accessibility audit started",
            audit_id=audit_id,
            project_id=request.project_id,
            url=request.url,
        )

        return RunAuditResponse(
            success=True,
            message="Accessibility audit started",
            audit_id=audit_id,
            status="running",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start audit", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start audit: {str(e)}")


@router.patch("/audit/{audit_id}")
async def update_audit(
    audit_id: str,
    status: str | None = Query(None, description="New status"),
    accessibility_score: float | None = Query(None, description="Accessibility score"),
) -> dict:
    """
    Update an audit's status and/or score.

    Used by the audit runner to update results.
    """
    validate_uuid(audit_id, "audit_id")
    supabase = get_supabase_client()

    try:
        update_data: dict = {}
        if status:
            update_data["status"] = status
            if status in ("completed", "failed"):
                update_data["completed_at"] = datetime.now(UTC).isoformat()
        if accessibility_score is not None:
            update_data["accessibility_score"] = accessibility_score

        if not update_data:
            raise HTTPException(status_code=400, detail="No update data provided")

        result = await supabase.update(
            "quality_audits",
            {"id": f"eq.{audit_id}"},
            update_data,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to update audit")

        return {"success": True, "message": "Audit updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update audit", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update: {str(e)}")


@router.post("/issues/batch")
async def create_issues_batch(
    audit_id: str = Query(..., description="Audit ID"),
    issues: list[dict] = [],
) -> dict:
    """
    Create multiple accessibility issues for an audit.

    Used by the audit runner to save discovered issues.
    """
    validate_uuid(audit_id, "audit_id")
    supabase = get_supabase_client()

    if not issues:
        return {"success": True, "created": 0}

    try:
        # Add audit_id and generate IDs for each issue
        now = datetime.now(UTC).isoformat()
        issues_data = [
            {
                "id": str(uuid.uuid4()),
                "audit_id": audit_id,
                "rule": issue.get("rule", "unknown"),
                "severity": issue.get("severity", "moderate"),
                "element_selector": issue.get("element_selector"),
                "description": issue.get("description", ""),
                "suggested_fix": issue.get("suggested_fix"),
                "wcag_criteria": issue.get("wcag_criteria"),
                "created_at": now,
            }
            for issue in issues
        ]

        result = await supabase.insert("accessibility_issues", issues_data)

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create issues")

        return {"success": True, "created": len(issues_data)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create issues", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create issues: {str(e)}")
