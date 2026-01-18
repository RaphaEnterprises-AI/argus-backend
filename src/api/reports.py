"""Reports API endpoints.

Provides CRUD endpoints for test reports:
- Listing reports (with filtering by project_id, date range)
- Generating new reports from test runs
- Getting specific report details
- Deleting reports
- Downloading reports as PDF/HTML

Reports include:
- Test execution results
- Coverage metrics
- Trend analysis
- Quality scores
"""

import io
import json
from datetime import UTC, datetime, timedelta
from enum import Enum

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.context import get_current_organization_id
from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit, verify_org_access
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["Reports"])


# ============================================================================
# Enums and Types
# ============================================================================

class ReportType(str, Enum):
    """Report type options."""
    TEST_EXECUTION = "test_execution"
    COVERAGE = "coverage"
    TREND = "trend"
    QUALITY = "quality"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report format options."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JUNIT = "junit"


class ReportStatus(str, Enum):
    """Report status options."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Request/Response Models
# ============================================================================

class ReportSummary(BaseModel):
    """Summary of report content."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    pass_rate: float = 0.0
    duration_ms: int | None = None
    coverage_percentage: float | None = None


class ReportMetrics(BaseModel):
    """Report metrics."""
    avg_duration_ms: int | None = None
    flaky_tests: int = 0
    new_failures: int = 0
    regressions: int = 0
    improvements: int = 0


class CreateReportRequest(BaseModel):
    """Request to create/generate a new report."""
    project_id: str = Field(..., description="Project ID for the report")
    name: str = Field(..., min_length=1, max_length=255, description="Report name")
    description: str | None = Field(None, max_length=2000, description="Report description")
    report_type: ReportType = Field(default=ReportType.TEST_EXECUTION, description="Type of report")
    format: ReportFormat = Field(default=ReportFormat.JSON, description="Output format")
    test_run_id: str | None = Field(None, description="Specific test run ID to report on")
    date_from: datetime | None = Field(None, description="Start date for trend reports")
    date_to: datetime | None = Field(None, description="End date for trend reports")
    include_details: bool = Field(default=True, description="Include detailed test results")


class UpdateReportRequest(BaseModel):
    """Request to update a report."""
    name: str | None = Field(None, min_length=1, max_length=255, description="Report name")
    description: str | None = Field(None, max_length=2000, description="Report description")


class ReportResponse(BaseModel):
    """Report details response."""
    id: str
    organization_id: str
    project_id: str
    test_run_id: str | None
    name: str
    description: str | None
    report_type: str
    status: str
    format: str
    summary: dict
    content: dict
    metrics: dict
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_ms: int | None
    coverage_percentage: float | None
    date_from: str | None
    date_to: str | None
    file_url: str | None
    file_size_bytes: int | None
    created_by: str | None
    created_at: str
    updated_at: str | None
    generated_at: str | None
    expires_at: str | None


class ReportListResponse(BaseModel):
    """Report list item response."""
    id: str
    organization_id: str
    project_id: str
    name: str
    report_type: str
    status: str
    format: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage_percentage: float | None
    created_at: str
    generated_at: str | None


class ReportListPaginatedResponse(BaseModel):
    """Paginated report list response."""
    reports: list[ReportListResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# Helper Functions
# ============================================================================

async def verify_report_access(report_id: str, user_id: str, user_email: str = None, request: Request = None) -> dict:
    """Verify user has access to the report via organization membership.

    Returns the report data if access is granted.
    """
    supabase = get_supabase_client()

    # Get report
    report_result = await supabase.request(
        f"/reports?id=eq.{report_id}&select=*"
    )

    if not report_result.get("data"):
        raise HTTPException(status_code=404, detail="Report not found")

    report = report_result["data"][0]

    # Verify user has access to the organization
    _, _ = await verify_org_access(report["organization_id"], user_id, user_email=user_email, request=request)

    return report


async def get_project_org_id(project_id: str) -> str:
    """Get the organization ID for a project."""
    supabase = get_supabase_client()

    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )

    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    return project_result["data"][0]["organization_id"]


async def generate_report_content(
    project_id: str,
    report_type: ReportType,
    test_run_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    include_details: bool = True,
) -> dict:
    """Generate report content based on report type."""
    supabase = get_supabase_client()
    content = {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": report_type,
    }

    if report_type == ReportType.TEST_EXECUTION:
        # Get test run results
        if test_run_id:
            run_result = await supabase.request(
                f"/test_runs?id=eq.{test_run_id}&select=*"
            )
            if run_result.get("data"):
                run = run_result["data"][0]
                content["test_run"] = run

                if include_details:
                    results_result = await supabase.request(
                        f"/test_results?test_run_id=eq.{test_run_id}&select=*&order=created_at.asc"
                    )
                    content["test_results"] = results_result.get("data", [])
        else:
            # Get latest test runs
            runs_result = await supabase.request(
                f"/test_runs?project_id=eq.{project_id}&select=*&order=created_at.desc&limit=10"
            )
            content["test_runs"] = runs_result.get("data", [])

    elif report_type == ReportType.COVERAGE:
        # Get coverage data from quality audits
        audits_result = await supabase.request(
            f"/quality_audits?project_id=eq.{project_id}&select=*&order=created_at.desc&limit=1"
        )
        if audits_result.get("data"):
            content["latest_audit"] = audits_result["data"][0]

        # Get test coverage from risk scores
        risk_result = await supabase.request(
            f"/risk_scores?project_id=eq.{project_id}&select=*&order=calculated_at.desc"
        )
        content["risk_scores"] = risk_result.get("data", [])

    elif report_type == ReportType.TREND:
        # Get historical data
        if not date_from:
            date_from = datetime.now(UTC) - timedelta(days=30)
        if not date_to:
            date_to = datetime.now(UTC)

        content["date_range"] = {
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        }

        # Get daily stats
        stats_result = await supabase.request(
            f"/daily_stats?project_id=eq.{project_id}"
            f"&date=gte.{date_from.date().isoformat()}"
            f"&date=lte.{date_to.date().isoformat()}"
            f"&select=*&order=date.asc"
        )
        content["daily_stats"] = stats_result.get("data", [])

        # Get test runs in range
        runs_result = await supabase.request(
            f"/test_runs?project_id=eq.{project_id}"
            f"&created_at=gte.{date_from.isoformat()}"
            f"&created_at=lte.{date_to.isoformat()}"
            f"&select=*&order=created_at.desc"
        )
        content["test_runs"] = runs_result.get("data", [])

    elif report_type == ReportType.QUALITY:
        # Get quality scores
        from src.api.quality import _calculate_quality_score

        try:
            quality_data = await _calculate_quality_score(project_id)
            content["quality_score"] = quality_data
        except Exception as e:
            logger.warning("Failed to calculate quality score", error=str(e))
            content["quality_score"] = {"error": str(e)}

        # Get AI insights
        insights_result = await supabase.request(
            f"/ai_insights?project_id=eq.{project_id}&is_resolved=eq.false"
            f"&select=*&order=created_at.desc&limit=20"
        )
        content["ai_insights"] = insights_result.get("data", [])

    return content


def calculate_report_summary(content: dict, report_type: ReportType) -> dict:
    """Calculate summary statistics from report content."""
    summary = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "pass_rate": 0.0,
    }

    if report_type == ReportType.TEST_EXECUTION:
        if "test_run" in content:
            run = content["test_run"]
            summary["total_tests"] = run.get("total_tests", 0)
            summary["passed_tests"] = run.get("passed_tests", 0)
            summary["failed_tests"] = run.get("failed_tests", 0)
            summary["skipped_tests"] = run.get("skipped_tests", 0)
            summary["duration_ms"] = run.get("duration_ms")
        elif "test_runs" in content:
            for run in content["test_runs"]:
                summary["total_tests"] += run.get("total_tests", 0)
                summary["passed_tests"] += run.get("passed_tests", 0)
                summary["failed_tests"] += run.get("failed_tests", 0)
                summary["skipped_tests"] += run.get("skipped_tests", 0)

    elif report_type == ReportType.TREND:
        if "daily_stats" in content:
            for stat in content["daily_stats"]:
                summary["total_tests"] += stat.get("runs", 0)
                summary["passed_tests"] += stat.get("passed", 0)
                summary["failed_tests"] += stat.get("failed", 0)

    # Calculate pass rate
    if summary["total_tests"] > 0:
        summary["pass_rate"] = round(
            (summary["passed_tests"] / summary["total_tests"]) * 100, 2
        )

    return summary


def generate_html_report(report: dict) -> str:
    """Generate HTML report content."""
    summary = report.get("summary", {})
    content = report.get("content", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.get('name', 'Test Report')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .passed {{ color: #4CAF50; }}
        .failed {{ color: #f44336; }}
        .skipped {{ color: #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        .status-passed {{ background: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; }}
        .status-failed {{ background: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 4px; }}
        .status-skipped {{ background: #fff3e0; color: #ef6c00; padding: 4px 8px; border-radius: 4px; }}
        .meta {{ color: #888; font-size: 0.9em; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report.get('name', 'Test Report')}</h1>
        {f"<p>{report.get('description')}</p>" if report.get('description') else ""}

        <h2>Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{summary.get('total_tests', 0)}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value passed">{summary.get('passed_tests', 0)}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">{summary.get('failed_tests', 0)}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value skipped">{summary.get('skipped_tests', 0)}</div>
                <div class="stat-label">Skipped</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.get('pass_rate', 0)}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
        </div>
"""

    # Add test results if available
    test_results = content.get("test_results", [])
    if test_results:
        html += """
        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>
"""
        for result in test_results:
            status = result.get("status", "unknown")
            status_class = f"status-{status}" if status in ["passed", "failed", "skipped"] else ""
            duration = f"{result.get('duration_ms', 0)}ms" if result.get("duration_ms") else "-"
            error = result.get("error_message", "-")[:100] if result.get("error_message") else "-"

            html += f"""
                <tr>
                    <td>{result.get('name', 'Unknown')}</td>
                    <td><span class="{status_class}">{status.upper()}</span></td>
                    <td>{duration}</td>
                    <td>{error}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""

    html += f"""
        <div class="meta">
            <p>Report Type: {report.get('report_type', 'test_execution')}</p>
            <p>Generated: {report.get('generated_at', report.get('created_at', 'N/A'))}</p>
            <p>Report ID: {report.get('id', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_markdown_report(report: dict) -> str:
    """Generate Markdown report content."""
    summary = report.get("summary", {})
    content = report.get("content", {})

    md = f"""# {report.get('name', 'Test Report')}

{report.get('description', '') if report.get('description') else ''}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary.get('total_tests', 0)} |
| Passed | {summary.get('passed_tests', 0)} |
| Failed | {summary.get('failed_tests', 0)} |
| Skipped | {summary.get('skipped_tests', 0)} |
| Pass Rate | {summary.get('pass_rate', 0)}% |
"""

    # Add test results if available
    test_results = content.get("test_results", [])
    if test_results:
        md += """
## Test Results

| Test Name | Status | Duration | Error |
|-----------|--------|----------|-------|
"""
        for result in test_results:
            status = result.get("status", "unknown").upper()
            duration = f"{result.get('duration_ms', 0)}ms" if result.get("duration_ms") else "-"
            error = result.get("error_message", "-")[:50] if result.get("error_message") else "-"
            md += f"| {result.get('name', 'Unknown')} | {status} | {duration} | {error} |\n"

    md += f"""
---

**Report Type:** {report.get('report_type', 'test_execution')}
**Generated:** {report.get('generated_at', report.get('created_at', 'N/A'))}
**Report ID:** {report.get('id', 'N/A')}
"""
    return md


def generate_junit_report(report: dict) -> str:
    """Generate JUnit XML report content."""
    summary = report.get("summary", {})
    content = report.get("content", {})
    test_results = content.get("test_results", [])

    total = summary.get("total_tests", len(test_results))
    failures = summary.get("failed_tests", 0)
    skipped = summary.get("skipped_tests", 0)
    duration = summary.get("duration_ms", 0) / 1000 if summary.get("duration_ms") else 0

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="{report.get('name', 'Test Report')}" tests="{total}" failures="{failures}" skipped="{skipped}" time="{duration:.3f}">
    <testsuite name="{report.get('name', 'Test Suite')}" tests="{total}" failures="{failures}" skipped="{skipped}" time="{duration:.3f}">
'''

    for result in test_results:
        test_name = result.get("name", "Unknown").replace('"', '&quot;')
        test_time = (result.get("duration_ms", 0) / 1000) if result.get("duration_ms") else 0
        status = result.get("status", "unknown")

        xml += f'        <testcase name="{test_name}" time="{test_time:.3f}"'

        if status == "failed":
            error_msg = (result.get("error_message", "Test failed") or "Test failed").replace("<", "&lt;").replace(">", "&gt;")
            xml += f'>\n            <failure message="{error_msg[:100]}">{error_msg}</failure>\n        </testcase>\n'
        elif status == "skipped":
            xml += '>\n            <skipped/>\n        </testcase>\n'
        else:
            xml += '/>\n'

    xml += '''    </testsuite>
</testsuites>
'''
    return xml


# ============================================================================
# Report Endpoints
# ============================================================================

@router.get("/reports", response_model=ReportListPaginatedResponse)
async def list_reports(
    request: Request,
    project_id: str | None = None,
    report_type: ReportType | None = None,
    status: ReportStatus | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List reports with optional filters.

    Args:
        project_id: Filter by project ID
        report_type: Filter by report type
        status: Filter by report status
        date_from: Filter reports created after this date
        date_to: Filter reports created before this date
        limit: Maximum number of results (default 50, max 100)
        offset: Offset for pagination

    Returns:
        Paginated list of reports
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Limit max results
    limit = min(limit, 100)

    if project_id:
        # Verify access to the specific project
        await verify_project_access(project_id, user["user_id"], user.get("email"), request)
        org_id = await get_project_org_id(project_id)

        # Build query for specific project
        query = f"/reports?project_id=eq.{project_id}&select=*&order=created_at.desc"
    else:
        # Get reports from all organizations the user has access to
        org_id = await get_current_organization_id(request)

        if org_id:
            # Verify access to the organization
            _, supabase_org_id = await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)
            query = f"/reports?organization_id=eq.{supabase_org_id}&select=*&order=created_at.desc"
        else:
            # Get all organizations user has access to
            memberships = await supabase.request(
                f"/organization_members?user_id=eq.{user['user_id']}&status=eq.active&select=organization_id"
            )

            if not memberships.get("data"):
                return ReportListPaginatedResponse(reports=[], total=0, limit=limit, offset=offset)

            org_ids = [m["organization_id"] for m in memberships["data"]]
            query = f"/reports?organization_id=in.({','.join(org_ids)})&select=*&order=created_at.desc"

    # Apply filters
    if report_type:
        query += f"&report_type=eq.{report_type.value}"

    if status:
        query += f"&status=eq.{status.value}"

    if date_from:
        query += f"&created_at=gte.{date_from.isoformat()}"

    if date_to:
        query += f"&created_at=lte.{date_to.isoformat()}"

    # Get total count first
    count_query = query.replace("&select=*", "&select=id")
    count_result = await supabase.request(count_query)
    total = len(count_result.get("data", []))

    # Apply pagination
    query += f"&limit={limit}&offset={offset}"

    reports_result = await supabase.request(query)

    if reports_result.get("error"):
        logger.error("Failed to fetch reports", error=reports_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

    reports = reports_result.get("data", [])

    result = [
        ReportListResponse(
            id=report["id"],
            organization_id=report["organization_id"],
            project_id=report["project_id"],
            name=report["name"],
            report_type=report.get("report_type", "test_execution"),
            status=report.get("status", "pending"),
            format=report.get("format", "json"),
            total_tests=report.get("total_tests", 0),
            passed_tests=report.get("passed_tests", 0),
            failed_tests=report.get("failed_tests", 0),
            coverage_percentage=report.get("coverage_percentage"),
            created_at=report["created_at"],
            generated_at=report.get("generated_at"),
        )
        for report in reports
    ]

    return ReportListPaginatedResponse(
        reports=result,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/reports", response_model=ReportResponse)
async def create_report(
    body: CreateReportRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Generate a new report.

    Creates a report record and generates the content asynchronously.
    The report will be in 'generating' status initially.

    Requires membership in the project's organization.
    """
    user = await get_current_user(request)

    # Verify access to the project
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()
    org_id = await get_project_org_id(body.project_id)

    # Generate report content synchronously for now
    # For larger reports, this could be moved to background task
    content = await generate_report_content(
        project_id=body.project_id,
        report_type=body.report_type,
        test_run_id=body.test_run_id,
        date_from=body.date_from,
        date_to=body.date_to,
        include_details=body.include_details,
    )

    summary = calculate_report_summary(content, body.report_type)

    # Create report record
    report_data = {
        "organization_id": org_id,
        "project_id": body.project_id,
        "test_run_id": body.test_run_id,
        "name": body.name,
        "description": body.description,
        "report_type": body.report_type.value,
        "status": "completed",
        "format": body.format.value,
        "summary": summary,
        "content": content,
        "metrics": {},
        "total_tests": summary.get("total_tests", 0),
        "passed_tests": summary.get("passed_tests", 0),
        "failed_tests": summary.get("failed_tests", 0),
        "skipped_tests": summary.get("skipped_tests", 0),
        "duration_ms": summary.get("duration_ms"),
        "date_from": body.date_from.isoformat() if body.date_from else None,
        "date_to": body.date_to.isoformat() if body.date_to else None,
        "created_by": user["user_id"],
        "generated_at": datetime.now(UTC).isoformat(),
    }

    result = await supabase.insert("reports", report_data)

    if result.get("error"):
        logger.error("Failed to create report", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create report")

    report = result["data"][0]

    # Audit log
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="report.create",
        resource_type="report",
        resource_id=report["id"],
        description=f"Generated report '{body.name}'",
        metadata={"name": body.name, "type": body.report_type.value, "format": body.format.value},
        request=request,
    )

    logger.info("Report created", report_id=report["id"], name=body.name, type=body.report_type.value)

    return ReportResponse(
        id=report["id"],
        organization_id=report["organization_id"],
        project_id=report["project_id"],
        test_run_id=report.get("test_run_id"),
        name=report["name"],
        description=report.get("description"),
        report_type=report.get("report_type", "test_execution"),
        status=report.get("status", "completed"),
        format=report.get("format", "json"),
        summary=report.get("summary", {}),
        content=report.get("content", {}),
        metrics=report.get("metrics", {}),
        total_tests=report.get("total_tests", 0),
        passed_tests=report.get("passed_tests", 0),
        failed_tests=report.get("failed_tests", 0),
        skipped_tests=report.get("skipped_tests", 0),
        duration_ms=report.get("duration_ms"),
        coverage_percentage=report.get("coverage_percentage"),
        date_from=report.get("date_from"),
        date_to=report.get("date_to"),
        file_url=report.get("file_url"),
        file_size_bytes=report.get("file_size_bytes"),
        created_by=report.get("created_by"),
        created_at=report["created_at"],
        updated_at=report.get("updated_at"),
        generated_at=report.get("generated_at"),
        expires_at=report.get("expires_at"),
    )


@router.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str, request: Request):
    """Get a specific report by ID.

    Requires membership in the report's organization.
    """
    user = await get_current_user(request)
    report = await verify_report_access(report_id, user["user_id"], user.get("email"), request)

    return ReportResponse(
        id=report["id"],
        organization_id=report["organization_id"],
        project_id=report["project_id"],
        test_run_id=report.get("test_run_id"),
        name=report["name"],
        description=report.get("description"),
        report_type=report.get("report_type", "test_execution"),
        status=report.get("status", "completed"),
        format=report.get("format", "json"),
        summary=report.get("summary", {}),
        content=report.get("content", {}),
        metrics=report.get("metrics", {}),
        total_tests=report.get("total_tests", 0),
        passed_tests=report.get("passed_tests", 0),
        failed_tests=report.get("failed_tests", 0),
        skipped_tests=report.get("skipped_tests", 0),
        duration_ms=report.get("duration_ms"),
        coverage_percentage=report.get("coverage_percentage"),
        date_from=report.get("date_from"),
        date_to=report.get("date_to"),
        file_url=report.get("file_url"),
        file_size_bytes=report.get("file_size_bytes"),
        created_by=report.get("created_by"),
        created_at=report["created_at"],
        updated_at=report.get("updated_at"),
        generated_at=report.get("generated_at"),
        expires_at=report.get("expires_at"),
    )


@router.put("/reports/{report_id}", response_model=ReportResponse)
@router.patch("/reports/{report_id}", response_model=ReportResponse)
async def update_report(report_id: str, body: UpdateReportRequest, request: Request):
    """Update a report's name or description.

    Requires membership in the report's organization.
    """
    user = await get_current_user(request)
    report = await verify_report_access(report_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Build update data
    update_data = {"updated_at": datetime.now(UTC).isoformat()}

    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description

    result = await supabase.update("reports", {"id": f"eq.{report_id}"}, update_data)

    if result.get("error"):
        logger.error("Failed to update report", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update report")

    # Audit log
    await log_audit(
        organization_id=report["organization_id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="report.update",
        resource_type="report",
        resource_id=report_id,
        description=f"Updated report '{report['name']}'",
        metadata={"changes": {k: v for k, v in update_data.items() if k != "updated_at"}},
        request=request,
    )

    logger.info("Report updated", report_id=report_id)

    return await get_report(report_id, request)


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str, request: Request):
    """Delete a report.

    Requires membership in the report's organization.
    """
    user = await get_current_user(request)
    report = await verify_report_access(report_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Delete report
    delete_result = await supabase.request(
        f"/reports?id=eq.{report_id}",
        method="DELETE"
    )

    if delete_result.get("error"):
        logger.error("Failed to delete report", error=delete_result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete report")

    # Audit log
    await log_audit(
        organization_id=report["organization_id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="report.delete",
        resource_type="report",
        resource_id=report_id,
        description=f"Deleted report '{report['name']}'",
        metadata={"name": report["name"], "project_id": report["project_id"]},
        request=request,
    )

    logger.info("Report deleted", report_id=report_id, name=report["name"])

    return {"success": True, "message": f"Report '{report['name']}' has been deleted"}


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    request: Request,
    format: ReportFormat | None = None,
):
    """Download a report in the specified format.

    Args:
        report_id: Report ID to download
        format: Output format (json, html, pdf, markdown, junit). Defaults to report's original format.

    Returns:
        Report file as streaming response
    """
    user = await get_current_user(request)
    report = await verify_report_access(report_id, user["user_id"], user.get("email"), request)

    # Use specified format or report's default format
    output_format = format.value if format else report.get("format", "json")

    # Generate content based on format
    if output_format == "json":
        content = json.dumps({
            "id": report["id"],
            "name": report["name"],
            "description": report.get("description"),
            "report_type": report.get("report_type"),
            "summary": report.get("summary", {}),
            "content": report.get("content", {}),
            "metrics": report.get("metrics", {}),
            "generated_at": report.get("generated_at"),
        }, indent=2)
        media_type = "application/json"
        filename = f"report-{report_id}.json"

    elif output_format == "html":
        content = generate_html_report(report)
        media_type = "text/html"
        filename = f"report-{report_id}.html"

    elif output_format == "markdown":
        content = generate_markdown_report(report)
        media_type = "text/markdown"
        filename = f"report-{report_id}.md"

    elif output_format == "junit":
        content = generate_junit_report(report)
        media_type = "application/xml"
        filename = f"report-{report_id}.xml"

    elif output_format == "pdf":
        # PDF generation would require additional libraries like weasyprint or reportlab
        # For now, return HTML with a note
        content = generate_html_report(report)
        media_type = "text/html"
        filename = f"report-{report_id}.html"
        # Note: PDF generation can be added with weasyprint:
        # from weasyprint import HTML
        # pdf_content = HTML(string=html_content).write_pdf()

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {output_format}")

    # Audit log
    await log_audit(
        organization_id=report["organization_id"],
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="report.download",
        resource_type="report",
        resource_id=report_id,
        description=f"Downloaded report '{report['name']}' as {output_format}",
        metadata={"name": report["name"], "format": output_format},
        request=request,
    )

    logger.info("Report downloaded", report_id=report_id, format=output_format)

    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        },
    )


# ============================================================================
# Convenience Endpoints
# ============================================================================

@router.get("/projects/{project_id}/reports", response_model=ReportListPaginatedResponse)
async def list_project_reports(
    project_id: str,
    request: Request,
    report_type: ReportType | None = None,
    status: ReportStatus | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List reports for a specific project.

    Convenience endpoint that's equivalent to GET /reports?project_id={project_id}
    """
    return await list_reports(
        request=request,
        project_id=project_id,
        report_type=report_type,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.post("/test-runs/{test_run_id}/report", response_model=ReportResponse)
async def generate_test_run_report(
    test_run_id: str,
    request: Request,
    name: str | None = None,
    format: ReportFormat = ReportFormat.JSON,
):
    """Generate a report from a specific test run.

    Convenience endpoint to quickly generate a report from a test run.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Get test run to find project_id
    run_result = await supabase.request(
        f"/test_runs?id=eq.{test_run_id}&select=*"
    )

    if not run_result.get("data"):
        raise HTTPException(status_code=404, detail="Test run not found")

    test_run = run_result["data"][0]

    # Verify access to the project
    await verify_project_access(test_run["project_id"], user["user_id"], user.get("email"), request)

    # Generate report name if not provided
    report_name = name or f"Test Run Report - {test_run.get('name', test_run_id)}"

    # Create report
    body = CreateReportRequest(
        project_id=test_run["project_id"],
        name=report_name,
        description=f"Report generated from test run {test_run_id}",
        report_type=ReportType.TEST_EXECUTION,
        format=format,
        test_run_id=test_run_id,
        include_details=True,
    )

    return await create_report(body, request, BackgroundTasks())
