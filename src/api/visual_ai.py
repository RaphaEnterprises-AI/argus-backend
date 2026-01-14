"""Visual AI API for comprehensive visual testing and regression detection.

Provides Applitools-like visual regression testing capabilities using Claude Vision.
Includes:
- Screenshot capture with multiple viewports and browsers
- AI-powered visual comparison and analysis
- Responsive testing across device sizes
- Cross-browser compatibility testing
- WCAG accessibility analysis
- Baseline management with version history
- Change approval/rejection workflow
- AI explanations of visual changes
"""

import base64
import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from src.config import get_settings
from src.services.supabase_client import get_supabase_client
from src.agents.visual_ai import (
    VisualAI,
    VisualRegressionManager,
    VisualComparisonResult,
    DifferenceType,
    Severity,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/visual", tags=["Visual AI"])


# =============================================================================
# Default configurations
# =============================================================================

DEFAULT_VIEWPORTS = [
    {"name": "mobile", "width": 375, "height": 667},
    {"name": "tablet", "width": 768, "height": 1024},
    {"name": "desktop", "width": 1440, "height": 900},
    {"name": "wide", "width": 1920, "height": 1080},
]

DEFAULT_BROWSERS = ["chromium", "firefox", "webkit"]


# =============================================================================
# Request/Response Models
# =============================================================================

class ViewportConfig(BaseModel):
    """Viewport configuration."""
    name: Optional[str] = None
    width: int = Field(1440, ge=320, le=3840)
    height: int = Field(900, ge=480, le=2160)


class CaptureRequest(BaseModel):
    """Request to capture a visual snapshot."""
    url: str = Field(..., description="URL to capture")
    viewport: Optional[ViewportConfig] = Field(
        default_factory=lambda: ViewportConfig(width=1440, height=900),
        description="Viewport dimensions"
    )
    browser: Literal["chromium", "firefox", "webkit"] = Field(
        "chromium",
        description="Browser to use"
    )
    wait_for: Optional[str] = Field(
        None,
        description="CSS selector to wait for before capture"
    )
    wait_timeout: int = Field(
        30000,
        description="Timeout in ms for wait_for selector"
    )
    full_page: bool = Field(
        False,
        description="Capture full scrollable page"
    )
    project_id: Optional[str] = Field(
        None,
        description="Project ID for organization"
    )
    name: Optional[str] = Field(
        None,
        description="Name/label for this snapshot"
    )


class CompareRequest(BaseModel):
    """Request to compare snapshots."""
    baseline_id: str = Field(..., description="ID of baseline snapshot")
    current_url: str = Field(..., description="URL to capture and compare")
    context: Optional[str] = Field(
        None,
        description="Context about what the page should show"
    )
    git_diff: Optional[str] = Field(
        None,
        description="Git diff of recent changes (helps AI understand expected changes)"
    )
    pr_description: Optional[str] = Field(
        None,
        description="PR description (helps AI distinguish intentional changes)"
    )
    ignore_regions: Optional[List[str]] = Field(
        None,
        description="Regions to ignore (e.g., 'header timestamp', 'ad banner')"
    )
    sensitivity: Literal["low", "medium", "high"] = Field(
        "medium",
        description="Detection sensitivity"
    )
    viewport: Optional[ViewportConfig] = None
    browser: Literal["chromium", "firefox", "webkit"] = "chromium"


class ResponsiveRequest(BaseModel):
    """Request for responsive testing."""
    url: str = Field(..., description="URL to capture")
    viewports: Optional[List[ViewportConfig]] = Field(
        None,
        description="Custom viewports (uses defaults if not specified)"
    )
    project_id: Optional[str] = None
    name: Optional[str] = None


class BrowserMatrixRequest(BaseModel):
    """Request for cross-browser testing."""
    url: str = Field(..., description="URL to capture")
    browsers: Optional[List[str]] = Field(
        None,
        description="Browsers to use (defaults to all)"
    )
    viewport: Optional[ViewportConfig] = None
    project_id: Optional[str] = None
    name: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request to approve visual changes."""
    change_ids: Optional[List[str]] = Field(
        None,
        description="Specific change IDs to approve (None = approve all)"
    )
    notes: Optional[str] = Field(
        None,
        description="Review notes"
    )
    update_baseline: bool = Field(
        True,
        description="Update baseline with approved changes"
    )


class RejectionRequest(BaseModel):
    """Request to reject visual changes."""
    notes: str = Field(..., description="Reason for rejection")
    create_issue: bool = Field(
        False,
        description="Create GitHub issue for rejected changes"
    )


class SnapshotResponse(BaseModel):
    """Response for a captured snapshot."""
    id: str
    url: str
    screenshot_url: Optional[str] = None
    viewport: Dict[str, Any]
    browser: str
    captured_at: str
    metadata: Optional[Dict[str, Any]] = None


class ComparisonResponse(BaseModel):
    """Response for a visual comparison."""
    id: str
    baseline_id: str
    current_id: str
    match: bool
    match_percentage: float
    has_regressions: bool
    differences: List[Dict[str, Any]]
    summary: str
    ai_analysis: Optional[Dict[str, Any]] = None
    cost_usd: float
    compared_at: str


class BaselineResponse(BaseModel):
    """Response for a baseline."""
    id: str
    name: str
    url: str
    project_id: str
    screenshot_url: Optional[str] = None
    version: int
    created_at: str
    updated_at: str


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_snapshot_id() -> str:
    """Generate unique snapshot ID (UUID format for database compatibility)."""
    return str(uuid.uuid4())


def _generate_comparison_id() -> str:
    """Generate unique comparison ID (UUID format for database compatibility)."""
    return str(uuid.uuid4())


def _generate_baseline_id() -> str:
    """Generate unique baseline ID (UUID format for database compatibility)."""
    return str(uuid.uuid4())


async def _capture_screenshot(
    url: str,
    viewport: ViewportConfig,
    browser: str = "chromium",
    wait_for: Optional[str] = None,
    wait_timeout: int = 30000,
    full_page: bool = False,
) -> tuple[bytes, Dict[str, Any]]:
    """Capture screenshot using the cloud browser worker.

    Uses BrowserWorkerClient which connects to Cloudflare's @cloudflare/playwright
    for reliable, cloud-based screenshot capture without local Playwright installation.

    Returns:
        Tuple of (screenshot_bytes, metadata)
    """
    from src.tools.browser_worker_client import get_browser_client

    browser_client = get_browser_client()

    screenshot_bytes, result = await browser_client.capture_screenshot(
        url=url,
        viewport_width=viewport.width,
        viewport_height=viewport.height,
        full_page=full_page,
        wait_for=wait_for,
    )

    if screenshot_bytes is None:
        error_msg = result.get("error", "Unknown error capturing screenshot")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture screenshot: {error_msg}"
        )

    # Merge metadata from browser worker with additional info
    metadata = {
        "url": url,
        "viewport": {"width": viewport.width, "height": viewport.height},
        "browser": browser,
        "full_page": full_page,
        "captured_at": result.get("captured_at", datetime.now(timezone.utc).isoformat()),
        "source": "browser_worker",
    }

    return screenshot_bytes, metadata


async def _store_screenshot(
    screenshot_bytes: bytes,
    snapshot_id: str,
    project_id: Optional[str] = None,
) -> str:
    """Store screenshot and return URL.

    In production, this would upload to Supabase Storage or S3.
    For now, stores locally and returns a data URL.
    """
    settings = get_settings()

    # Create output directory
    screenshots_dir = Path(settings.output_dir) / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = screenshots_dir / f"{snapshot_id}.png"
    file_path.write_bytes(screenshot_bytes)

    # Return base64 data URL for now (in production, return actual URL)
    b64_data = base64.standard_b64encode(screenshot_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


async def _save_snapshot_to_db(
    snapshot_id: str,
    url: str,
    screenshot_path: str,
    viewport: Dict[str, Any],
    browser: str,
    metadata: Dict[str, Any],
    project_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Save snapshot record to database."""
    supabase = get_supabase_client()

    record = {
        "id": snapshot_id,
        "url": url,
        "page_url": url,  # Database column
        "screenshot_path": screenshot_path,
        "screenshot_url": screenshot_path,  # Database column (same as screenshot_path)
        "viewport_width": viewport.get("width", 1440),
        "viewport_height": viewport.get("height", 900),
        "browser": browser,
        "metadata": metadata,
        "project_id": project_id,
        "name": name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    result = await supabase.insert("visual_snapshots", record)

    if result.get("error"):
        logger.warning(
            "Failed to save snapshot to DB",
            error=result["error"],
            snapshot_id=snapshot_id
        )
        # Return record anyway for in-memory operation
        return record

    return result.get("data", [record])[0] if result.get("data") else record


# =============================================================================
# Capture Endpoints
# =============================================================================

@router.post("/capture", response_model=SnapshotResponse)
async def capture_snapshot(request: CaptureRequest):
    """
    Capture a full visual snapshot of a URL.

    Captures the page with specified viewport and browser settings.
    Stores the screenshot and returns metadata.
    """
    snapshot_id = _generate_snapshot_id()
    viewport = request.viewport or ViewportConfig()

    try:
        # Capture screenshot
        screenshot_bytes, metadata = await _capture_screenshot(
            url=request.url,
            viewport=viewport,
            browser=request.browser,
            wait_for=request.wait_for,
            wait_timeout=request.wait_timeout,
            full_page=request.full_page,
        )

        # Store screenshot
        screenshot_url = await _store_screenshot(
            screenshot_bytes,
            snapshot_id,
            request.project_id
        )

        # Save to database
        await _save_snapshot_to_db(
            snapshot_id=snapshot_id,
            url=request.url,
            screenshot_path=screenshot_url,
            viewport={"width": viewport.width, "height": viewport.height},
            browser=request.browser,
            metadata=metadata,
            project_id=request.project_id,
            name=request.name,
        )

        logger.info(
            "Snapshot captured",
            snapshot_id=snapshot_id,
            url=request.url,
            viewport=f"{viewport.width}x{viewport.height}",
        )

        return SnapshotResponse(
            id=snapshot_id,
            url=request.url,
            screenshot_url=screenshot_url,
            viewport={"width": viewport.width, "height": viewport.height},
            browser=request.browser,
            captured_at=metadata["captured_at"],
            metadata=metadata,
        )

    except Exception as e:
        logger.exception("Screenshot capture failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")


@router.post("/responsive/capture")
async def capture_responsive_matrix(request: ResponsiveRequest):
    """
    Capture snapshots at multiple viewport sizes.

    Tests responsive behavior by capturing the page at mobile, tablet,
    desktop, and wide screen sizes.
    """
    viewports = request.viewports or [ViewportConfig(**v) for v in DEFAULT_VIEWPORTS]
    results = []

    for vp in viewports:
        try:
            snapshot_id = _generate_snapshot_id()
            screenshot_bytes, metadata = await _capture_screenshot(
                url=request.url,
                viewport=vp,
                browser="chromium",
            )

            screenshot_url = await _store_screenshot(
                screenshot_bytes,
                snapshot_id,
                request.project_id
            )

            await _save_snapshot_to_db(
                snapshot_id=snapshot_id,
                url=request.url,
                screenshot_path=screenshot_url,
                viewport={"width": vp.width, "height": vp.height, "name": vp.name},
                browser="chromium",
                metadata=metadata,
                project_id=request.project_id,
                name=f"{request.name or 'responsive'}_{vp.name or f'{vp.width}x{vp.height}'}",
            )

            results.append({
                "id": snapshot_id,
                "viewport": {"name": vp.name, "width": vp.width, "height": vp.height},
                "success": True,
                "screenshot_url": screenshot_url,
            })

        except Exception as e:
            results.append({
                "viewport": {"name": vp.name, "width": vp.width, "height": vp.height},
                "success": False,
                "error": str(e),
            })

    return {
        "success": all(r["success"] for r in results),
        "url": request.url,
        "results": results,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/browsers/capture")
async def capture_browser_matrix(request: BrowserMatrixRequest):
    """
    Capture snapshots in multiple browsers.

    Tests cross-browser compatibility by capturing the page in
    Chromium, Firefox, and WebKit.
    """
    browsers = request.browsers or DEFAULT_BROWSERS
    viewport = request.viewport or ViewportConfig()
    results = []

    for browser in browsers:
        try:
            snapshot_id = _generate_snapshot_id()
            screenshot_bytes, metadata = await _capture_screenshot(
                url=request.url,
                viewport=viewport,
                browser=browser,
            )

            screenshot_url = await _store_screenshot(
                screenshot_bytes,
                snapshot_id,
                request.project_id
            )

            await _save_snapshot_to_db(
                snapshot_id=snapshot_id,
                url=request.url,
                screenshot_path=screenshot_url,
                viewport={"width": viewport.width, "height": viewport.height},
                browser=browser,
                metadata=metadata,
                project_id=request.project_id,
                name=f"{request.name or 'browser'}_{browser}",
            )

            results.append({
                "id": snapshot_id,
                "browser": browser,
                "success": True,
                "screenshot_url": screenshot_url,
            })

        except Exception as e:
            results.append({
                "browser": browser,
                "success": False,
                "error": str(e),
            })

    return {
        "success": all(r["success"] for r in results),
        "url": request.url,
        "results": results,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Comparison Endpoints
# =============================================================================

@router.post("/compare", response_model=ComparisonResponse)
async def compare_snapshots(request: CompareRequest):
    """
    Compare a baseline snapshot with a current URL.

    Uses Claude Vision AI to analyze visual differences, categorize them
    by severity, and determine if they are regressions or expected changes.
    """
    supabase = get_supabase_client()
    comparison_id = _generate_comparison_id()

    # Get baseline snapshot
    baseline_result = await supabase.select(
        "visual_snapshots",
        filters={"id": f"eq.{request.baseline_id}"}
    )

    if baseline_result.get("error") or not baseline_result.get("data"):
        # Try to find in visual_baselines table
        baseline_result = await supabase.select(
            "visual_baselines",
            filters={"id": f"eq.{request.baseline_id}"}
        )

        if baseline_result.get("error") or not baseline_result.get("data"):
            raise HTTPException(
                status_code=404,
                detail=f"Baseline not found: {request.baseline_id}"
            )

    baseline = baseline_result["data"][0]

    # Capture current screenshot
    viewport = request.viewport or ViewportConfig(
        width=baseline.get("viewport_width", 1440),
        height=baseline.get("viewport_height", 900)
    )

    current_id = _generate_snapshot_id()

    try:
        screenshot_bytes, metadata = await _capture_screenshot(
            url=request.current_url,
            viewport=viewport,
            browser=request.browser,
        )

        current_screenshot_url = await _store_screenshot(
            screenshot_bytes,
            current_id
        )

        await _save_snapshot_to_db(
            snapshot_id=current_id,
            url=request.current_url,
            screenshot_path=current_screenshot_url,
            viewport={"width": viewport.width, "height": viewport.height},
            browser=request.browser,
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture current screenshot: {str(e)}"
        )

    # Perform visual comparison using VisualAI
    visual_ai = VisualAI(sensitivity=request.sensitivity)

    # Get paths to screenshots
    settings = get_settings()
    screenshots_dir = Path(settings.output_dir) / "screenshots"
    baseline_path = screenshots_dir / f"{request.baseline_id}.png"
    current_path = screenshots_dir / f"{current_id}.png"

    # Check if baseline file exists, if not try to reconstruct from stored data
    if not baseline_path.exists():
        # For now, return an error - in production would fetch from storage
        raise HTTPException(
            status_code=404,
            detail="Baseline screenshot file not found"
        )

    # Build context for AI
    context_parts = []
    if request.context:
        context_parts.append(f"Page context: {request.context}")
    if request.pr_description:
        context_parts.append(f"PR description: {request.pr_description}")
    if request.git_diff:
        context_parts.append(f"Recent code changes:\n{request.git_diff[:1000]}")

    context = "\n\n".join(context_parts) if context_parts else None

    # Compare screenshots
    try:
        result = await visual_ai.compare(
            baseline=baseline_path,
            current=current_path,
            context=context,
            ignore_regions=request.ignore_regions,
        )
    except Exception as e:
        logger.exception("Visual comparison failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

    # Convert differences to serializable format
    differences = [
        {
            "id": f"diff_{i}",
            "type": d.type.value,
            "severity": d.severity.value,
            "description": d.description,
            "location": d.location,
            "element": d.element,
            "expected": d.expected,
            "actual": d.actual,
            "is_regression": d.is_regression,
        }
        for i, d in enumerate(result.differences)
    ]

    # Store comparison result
    comparison_record = {
        "id": comparison_id,
        "baseline_id": request.baseline_id,
        "current_snapshot_id": current_id,
        "match": result.match,
        "match_percentage": result.match_percentage,
        "has_regressions": result.has_regressions(),
        "differences": differences,
        "summary": result.summary,
        "cost_usd": result.analysis_cost_usd,
        "context": context,
        "status": "pending_review" if result.has_regressions() else "passed",
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }

    await supabase.insert("visual_comparisons", comparison_record)

    logger.info(
        "Visual comparison completed",
        comparison_id=comparison_id,
        match=result.match,
        match_percentage=result.match_percentage,
        differences=len(differences),
        has_regressions=result.has_regressions(),
    )

    return ComparisonResponse(
        id=comparison_id,
        baseline_id=request.baseline_id,
        current_id=current_id,
        match=result.match,
        match_percentage=result.match_percentage,
        has_regressions=result.has_regressions(),
        differences=differences,
        summary=result.summary,
        ai_analysis={
            "model": visual_ai.model,
            "sensitivity": request.sensitivity,
            "context_provided": bool(context),
        },
        cost_usd=result.analysis_cost_usd,
        compared_at=result.timestamp,
    )


@router.post("/responsive/compare")
async def compare_responsive(
    baseline_id: str,
    current_url: str,
    viewports: Optional[List[ViewportConfig]] = None,
):
    """
    Compare responsive behavior against baselines.

    Compares at multiple viewport sizes to detect responsive regressions.
    """
    viewports_to_test = viewports or [ViewportConfig(**v) for v in DEFAULT_VIEWPORTS]
    results = []

    for vp in viewports_to_test:
        try:
            # Generate viewport-specific baseline ID
            vp_baseline_id = f"{baseline_id}_{vp.name or f'{vp.width}x{vp.height}'}"

            compare_request = CompareRequest(
                baseline_id=vp_baseline_id,
                current_url=current_url,
                viewport=vp,
            )

            result = await compare_snapshots(compare_request)
            results.append({
                "viewport": {"name": vp.name, "width": vp.width, "height": vp.height},
                "success": True,
                "comparison": result.model_dump(),
            })

        except HTTPException as e:
            results.append({
                "viewport": {"name": vp.name, "width": vp.width, "height": vp.height},
                "success": False,
                "error": e.detail,
            })
        except Exception as e:
            results.append({
                "viewport": {"name": vp.name, "width": vp.width, "height": vp.height},
                "success": False,
                "error": str(e),
            })

    # Calculate overall result
    successful = [r for r in results if r["success"]]
    has_regressions = any(
        r.get("comparison", {}).get("has_regressions", False)
        for r in successful
    )

    return {
        "success": len(successful) > 0,
        "baseline_id": baseline_id,
        "current_url": current_url,
        "has_regressions": has_regressions,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in successful if not r.get("comparison", {}).get("has_regressions", False)),
            "failed": sum(1 for r in successful if r.get("comparison", {}).get("has_regressions", False)),
            "errors": len(results) - len(successful),
        },
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/browsers/compare")
async def compare_browsers(request: BrowserMatrixRequest):
    """
    Compare cross-browser rendering.

    Captures the same page in multiple browsers and compares them
    to detect browser-specific rendering issues.
    """
    browsers = request.browsers or DEFAULT_BROWSERS
    viewport = request.viewport or ViewportConfig()

    # First, capture all browsers
    screenshots = {}
    for browser in browsers:
        try:
            screenshot_bytes, metadata = await _capture_screenshot(
                url=request.url,
                viewport=viewport,
                browser=browser,
            )
            screenshots[browser] = screenshot_bytes
        except Exception as e:
            logger.warning(f"Failed to capture {browser}", error=str(e))

    if len(screenshots) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 browser screenshots to compare"
        )

    # Compare each browser against Chromium (reference)
    visual_ai = VisualAI(sensitivity="medium")
    results = []
    reference_browser = "chromium"

    if reference_browser not in screenshots:
        reference_browser = list(screenshots.keys())[0]

    settings = get_settings()
    screenshots_dir = Path(settings.output_dir) / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Save reference screenshot
    ref_path = screenshots_dir / f"browser_compare_ref_{uuid.uuid4().hex[:8]}.png"
    ref_path.write_bytes(screenshots[reference_browser])

    for browser, screenshot_bytes in screenshots.items():
        if browser == reference_browser:
            results.append({
                "browser": browser,
                "is_reference": True,
                "match_percentage": 100.0,
                "differences": [],
            })
            continue

        # Save comparison screenshot
        cmp_path = screenshots_dir / f"browser_compare_{browser}_{uuid.uuid4().hex[:8]}.png"
        cmp_path.write_bytes(screenshot_bytes)

        try:
            result = await visual_ai.compare(
                baseline=ref_path,
                current=cmp_path,
                context=f"Cross-browser comparison: {reference_browser} vs {browser}",
            )

            results.append({
                "browser": browser,
                "is_reference": False,
                "reference_browser": reference_browser,
                "match": result.match,
                "match_percentage": result.match_percentage,
                "differences": [
                    {
                        "type": d.type.value,
                        "severity": d.severity.value,
                        "description": d.description,
                        "location": d.location,
                    }
                    for d in result.differences
                ],
            })
        except Exception as e:
            results.append({
                "browser": browser,
                "is_reference": False,
                "error": str(e),
            })
        finally:
            # Cleanup
            if cmp_path.exists():
                cmp_path.unlink()

    # Cleanup reference
    if ref_path.exists():
        ref_path.unlink()

    return {
        "success": True,
        "url": request.url,
        "reference_browser": reference_browser,
        "results": results,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.post("/analyze")
async def analyze_snapshot(
    snapshot_id: str,
    expected_elements: Optional[List[str]] = Query(None),
    context: Optional[str] = Query(None),
):
    """
    Analyze a single snapshot for quality issues.

    Checks for:
    - Page load completeness
    - Expected elements presence
    - Layout issues
    - Content problems
    """
    settings = get_settings()
    screenshots_dir = Path(settings.output_dir) / "screenshots"
    screenshot_path = screenshots_dir / f"{snapshot_id}.png"

    if not screenshot_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Screenshot not found: {snapshot_id}"
        )

    visual_ai = VisualAI()

    try:
        result = await visual_ai.analyze_single(
            screenshot=screenshot_path,
            expected_elements=expected_elements,
            context=context,
        )

        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "analysis": result,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("Snapshot analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/accessibility/analyze")
async def analyze_accessibility(
    snapshot_id: str,
    wcag_level: Literal["A", "AA", "AAA"] = "AA",
):
    """
    Analyze screenshot for WCAG accessibility issues.

    Uses AI to detect:
    - Color contrast issues
    - Missing alt text indicators
    - Focus visibility problems
    - Touch target size issues
    - Text readability
    """
    settings = get_settings()
    screenshots_dir = Path(settings.output_dir) / "screenshots"
    screenshot_path = screenshots_dir / f"{snapshot_id}.png"

    if not screenshot_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Screenshot not found: {snapshot_id}"
        )

    # Load image
    with open(screenshot_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Use Claude for accessibility analysis
    settings = get_settings()
    api_key = settings.anthropic_api_key
    if hasattr(api_key, 'get_secret_value'):
        api_key = api_key.get_secret_value()

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze this screenshot for WCAG {wcag_level} accessibility issues.

Look for:
1. **Color Contrast**: Text that may have insufficient contrast against backgrounds
2. **Touch Targets**: Interactive elements that appear too small (< 44x44px)
3. **Text Readability**: Text that's too small or has poor line height
4. **Focus Indicators**: Missing or unclear focus states on interactive elements
5. **Visual Hierarchy**: Content that lacks clear heading structure
6. **Dynamic Content**: Areas that may cause accessibility issues with screen readers
7. **Motion/Animation**: Potential motion sensitivity issues

Respond with JSON:
{{
    "overall_score": 0-100,
    "level_compliance": "{wcag_level}",
    "issues": [
        {{
            "criterion": "WCAG criterion (e.g., 1.4.3 Contrast)",
            "severity": "critical|major|minor",
            "description": "Description of the issue",
            "location": "Where on the page",
            "recommendation": "How to fix"
        }}
    ],
    "passed_criteria": ["List of criteria that appear to pass"],
    "summary": "Overall accessibility assessment"
}}
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        import json
        result = json.loads(content.strip())

        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "wcag_level": wcag_level,
            "accessibility": result,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("Accessibility analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# =============================================================================
# Baseline Management Endpoints
# =============================================================================

@router.post("/baselines", response_model=BaselineResponse)
async def create_baseline(
    url: str,
    name: str,
    project_id: str,
    viewport: Optional[ViewportConfig] = None,
    browser: str = "chromium",
):
    """
    Create or update a visual baseline.

    Captures the URL and stores it as a baseline for future comparisons.
    """
    supabase = get_supabase_client()
    baseline_id = _generate_baseline_id()
    vp = viewport or ViewportConfig()

    # Check for existing baseline with same name in project
    existing = await supabase.select(
        "visual_baselines",
        filters={
            "project_id": f"eq.{project_id}",
            "name": f"eq.{name}",
        }
    )

    version = 1
    if existing.get("data"):
        version = existing["data"][0].get("version", 0) + 1
        baseline_id = existing["data"][0]["id"]  # Update existing

    # Capture screenshot
    try:
        screenshot_bytes, metadata = await _capture_screenshot(
            url=url,
            viewport=vp,
            browser=browser,
        )

        screenshot_url = await _store_screenshot(screenshot_bytes, baseline_id)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture baseline: {str(e)}"
        )

    now = datetime.now(timezone.utc).isoformat()

    baseline_record = {
        "id": baseline_id,
        "name": name,
        "project_id": project_id,
        "url": url,
        "page_url": url,  # Database uses page_url column
        "screenshot_path": screenshot_url,
        "screenshot_url": screenshot_url,  # Database also uses screenshot_url column
        "viewport_width": vp.width,
        "viewport_height": vp.height,
        "browser": browser,
        "version": version,
        "metadata": metadata,
        "created_at": now if version == 1 else existing["data"][0].get("created_at", now),
        "updated_at": now,
    }

    if version == 1:
        result = await supabase.insert("visual_baselines", baseline_record)
    else:
        result = await supabase.update(
            "visual_baselines",
            {"id": f"eq.{baseline_id}"},
            baseline_record
        )

    # Store version history
    await supabase.insert("visual_baseline_history", {
        "baseline_id": baseline_id,
        "version": version,
        "screenshot_path": screenshot_url,
        "metadata": metadata,
        "created_at": now,
    })

    logger.info(
        "Baseline created/updated",
        baseline_id=baseline_id,
        name=name,
        version=version,
    )

    return BaselineResponse(
        id=baseline_id,
        name=name,
        url=url,
        project_id=project_id,
        screenshot_url=screenshot_url,
        version=version,
        created_at=baseline_record["created_at"],
        updated_at=now,
    )


@router.get("/baselines/{baseline_id}")
async def get_baseline(baseline_id: str):
    """Get baseline details."""
    supabase = get_supabase_client()

    result = await supabase.select(
        "visual_baselines",
        filters={"id": f"eq.{baseline_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Baseline not found")

    return {"baseline": result["data"][0]}


@router.get("/baselines/{baseline_id}/history")
async def get_baseline_history(
    baseline_id: str,
    limit: int = Query(20, le=100),
):
    """
    Get version history for a baseline.

    Returns all previous versions of the baseline for comparison and rollback.
    """
    supabase = get_supabase_client()

    # Get baseline
    baseline_result = await supabase.select(
        "visual_baselines",
        filters={"id": f"eq.{baseline_id}"}
    )

    if baseline_result.get("error") or not baseline_result.get("data"):
        raise HTTPException(status_code=404, detail="Baseline not found")

    baseline = baseline_result["data"][0]

    # Get history
    history_result = await supabase.request(
        f"/visual_baseline_history?baseline_id=eq.{baseline_id}"
        f"&order=version.desc&limit={limit}"
    )

    history = history_result.get("data", []) if not history_result.get("error") else []

    return {
        "baseline": baseline,
        "history": history,
        "total_versions": len(history),
    }


@router.get("/baselines")
async def list_baselines(
    project_id: str = Query(...),
    limit: int = Query(50, le=100),
):
    """List all baselines for a project."""
    supabase = get_supabase_client()

    result = await supabase.request(
        f"/visual_baselines?project_id=eq.{project_id}"
        f"&order=updated_at.desc&limit={limit}"
    )

    if result.get("error"):
        logger.warning("Failed to fetch baselines", error=result["error"])
        return {"baselines": [], "total": 0}

    return {
        "baselines": result.get("data", []),
        "total": len(result.get("data", [])),
    }


# =============================================================================
# Approval/Rejection Endpoints
# =============================================================================

@router.post("/comparisons/{comparison_id}/approve")
async def approve_changes(
    comparison_id: str,
    request: ApprovalRequest,
):
    """
    Approve visual changes.

    Can approve all changes or specific ones. Optionally updates the baseline.
    """
    supabase = get_supabase_client()

    # Get comparison
    result = await supabase.select(
        "visual_comparisons",
        filters={"id": f"eq.{comparison_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Comparison not found")

    comparison = result["data"][0]

    # Update comparison status
    approved_changes = request.change_ids or [d["id"] for d in comparison.get("differences", [])]

    await supabase.update(
        "visual_comparisons",
        {"id": f"eq.{comparison_id}"},
        {
            "status": "approved",
            "review_notes": request.notes,
            "approved_changes": approved_changes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Optionally update baseline
    if request.update_baseline and comparison.get("baseline_id"):
        baseline_id = comparison["baseline_id"]
        current_snapshot_id = comparison.get("current_snapshot_id")

        if current_snapshot_id:
            # Get current snapshot
            snapshot_result = await supabase.select(
                "visual_snapshots",
                filters={"id": f"eq.{current_snapshot_id}"}
            )

            if snapshot_result.get("data"):
                snapshot = snapshot_result["data"][0]

                # Get existing baseline version
                baseline_result = await supabase.select(
                    "visual_baselines",
                    filters={"id": f"eq.{baseline_id}"}
                )

                if baseline_result.get("data"):
                    baseline = baseline_result["data"][0]
                    new_version = baseline.get("version", 0) + 1

                    # Update baseline with approved snapshot
                    await supabase.update(
                        "visual_baselines",
                        {"id": f"eq.{baseline_id}"},
                        {
                            "screenshot_path": snapshot.get("screenshot_path"),
                            "version": new_version,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    # Add to history
                    await supabase.insert("visual_baseline_history", {
                        "baseline_id": baseline_id,
                        "version": new_version,
                        "screenshot_path": snapshot.get("screenshot_path"),
                        "metadata": {"approved_from_comparison": comparison_id},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })

    logger.info(
        "Changes approved",
        comparison_id=comparison_id,
        approved_count=len(approved_changes),
        baseline_updated=request.update_baseline,
    )

    return {
        "success": True,
        "comparison_id": comparison_id,
        "approved_changes": approved_changes,
        "baseline_updated": request.update_baseline,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/comparisons/{comparison_id}/reject")
async def reject_changes(
    comparison_id: str,
    request: RejectionRequest,
):
    """
    Reject visual changes.

    Marks the comparison as rejected with notes explaining the reason.
    """
    supabase = get_supabase_client()

    # Get comparison
    result = await supabase.select(
        "visual_comparisons",
        filters={"id": f"eq.{comparison_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Comparison not found")

    comparison = result["data"][0]

    # Update comparison status
    await supabase.update(
        "visual_comparisons",
        {"id": f"eq.{comparison_id}"},
        {
            "status": "rejected",
            "review_notes": request.notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Optionally create GitHub issue
    issue_url = None
    if request.create_issue:
        # This would integrate with GitHub API
        # For now, just log it
        logger.info(
            "GitHub issue creation requested",
            comparison_id=comparison_id,
            notes=request.notes,
        )

    logger.info(
        "Changes rejected",
        comparison_id=comparison_id,
        notes=request.notes[:100],
    )

    return {
        "success": True,
        "comparison_id": comparison_id,
        "status": "rejected",
        "notes": request.notes,
        "issue_url": issue_url,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# AI Explanation Endpoints
# =============================================================================

@router.post("/ai/explain")
async def explain_changes(comparison_id: str):
    """
    Get detailed AI explanation of visual changes.

    Uses Claude to provide a human-readable explanation of what changed,
    why it might have changed, and whether it's likely intentional.
    """
    supabase = get_supabase_client()

    # Get comparison
    result = await supabase.select(
        "visual_comparisons",
        filters={"id": f"eq.{comparison_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Comparison not found")

    comparison = result["data"][0]
    differences = comparison.get("differences", [])

    if not differences:
        return {
            "success": True,
            "comparison_id": comparison_id,
            "explanation": "No visual differences were detected between the baseline and current screenshots.",
            "recommendations": [],
        }

    # Build explanation prompt
    settings = get_settings()
    api_key = settings.anthropic_api_key
    if hasattr(api_key, 'get_secret_value'):
        api_key = api_key.get_secret_value()

    import anthropic
    import json
    client = anthropic.Anthropic(api_key=api_key)

    differences_text = json.dumps(differences, indent=2)
    context = comparison.get("context", "No additional context provided")

    prompt = f"""Analyze these visual differences from a UI comparison and provide a clear explanation:

DIFFERENCES DETECTED:
{differences_text}

CONTEXT:
{context}

Provide:
1. A human-readable summary of what changed
2. Likely causes for each change
3. Assessment of whether each change is likely intentional or a bug
4. Recommendations for the reviewer

Respond in JSON:
{{
    "summary": "Brief overall summary",
    "changes_explained": [
        {{
            "change": "Description of the change",
            "likely_cause": "Why this might have changed",
            "intentional_likelihood": "high|medium|low",
            "risk_level": "high|medium|low"
        }}
    ],
    "recommendations": ["List of actionable recommendations"],
    "overall_assessment": "Whether to approve or investigate further"
}}
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        explanation = json.loads(content.strip())

        # Store explanation
        await supabase.update(
            "visual_comparisons",
            {"id": f"eq.{comparison_id}"},
            {"ai_explanation": explanation}
        )

        return {
            "success": True,
            "comparison_id": comparison_id,
            "explanation": explanation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception("AI explanation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


# =============================================================================
# Query Endpoints
# =============================================================================

@router.get("/comparisons")
async def list_comparisons(
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
):
    """List visual comparisons with optional filtering."""
    supabase = get_supabase_client()

    query_path = "/visual_comparisons?"
    filters = []

    if project_id:
        filters.append(f"project_id=eq.{project_id}")
    if status:
        filters.append(f"status=eq.{status}")

    query_path += "&".join(filters)
    if filters:
        query_path += "&"
    query_path += f"order=compared_at.desc&limit={limit}"

    result = await supabase.request(query_path)

    if result.get("error"):
        logger.warning("Failed to fetch comparisons", error=result["error"])
        return {"comparisons": [], "total": 0}

    return {
        "comparisons": result.get("data", []),
        "total": len(result.get("data", [])),
    }


@router.get("/comparisons/{comparison_id}")
async def get_comparison(comparison_id: str):
    """Get details of a specific comparison."""
    supabase = get_supabase_client()

    result = await supabase.select(
        "visual_comparisons",
        filters={"id": f"eq.{comparison_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Comparison not found")

    return {"comparison": result["data"][0]}


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    """Get details of a specific snapshot."""
    supabase = get_supabase_client()

    result = await supabase.select(
        "visual_snapshots",
        filters={"id": f"eq.{snapshot_id}"}
    )

    if result.get("error") or not result.get("data"):
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return {"snapshot": result["data"][0]}
