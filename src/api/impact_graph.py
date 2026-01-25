"""Impact Graph API for Commit Impact Prediction.

The Test Impact Graph maps source files to the tests that exercise them,
enabling intelligent test selection when files change.

Data sources:
1. Code coverage - lcov, istanbul, pytest-cov
2. Import analysis - AST parsing of test files
3. Historical failures - Past test runs correlated with changes
4. AI inference - Semantic analysis of code relationships
"""

import hashlib
import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/impact-graph", tags=["Impact Graph"])


# =============================================================================
# Models
# =============================================================================


class ImpactMapping(BaseModel):
    """A single file-to-test mapping."""

    id: str | None = None
    file_path: str
    function_name: str | None = None
    class_name: str | None = None
    test_file_path: str
    test_name: str
    test_function: str | None = None
    impact_score: float = Field(ge=0, le=1, default=1.0)
    relationship_type: str = Field(
        default="direct",
        pattern="^(direct|transitive|coverage|historical|semantic)$"
    )
    confidence: float = Field(ge=0, le=1, default=1.0)


class AffectedTest(BaseModel):
    """A test affected by file changes."""

    test_id: str | None = None
    test_file_path: str
    test_name: str
    test_function: str | None = None
    total_impact_score: float
    relationship_types: list[str]
    affected_files: list[str]


class AffectedTestsResponse(BaseModel):
    """Response for affected tests query."""

    files: list[str]
    tests: list[AffectedTest]
    total_tests: int
    min_score_used: float


class ImpactGraphStats(BaseModel):
    """Statistics for a project's impact graph."""

    project_id: str
    total_mappings: int
    unique_source_files: int
    unique_tests: int
    by_relationship_type: dict[str, int]
    avg_impact_score: float
    last_updated: datetime | None = None
    coverage_imports_count: int
    pending_jobs: int


class CoverageImport(BaseModel):
    """Coverage import record."""

    id: str
    file_name: str
    format: str
    status: str
    files_processed: int
    mappings_created: int
    error_message: str | None = None
    created_at: datetime


class JobStatus(BaseModel):
    """Impact graph job status."""

    id: str
    job_type: str
    status: str
    progress: float
    mappings_added: int
    mappings_updated: int
    mappings_removed: int
    files_analyzed: int
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class CreateMappingRequest(BaseModel):
    """Request to create a manual mapping."""

    file_path: str
    test_file_path: str
    test_name: str
    test_function: str | None = None
    function_name: str | None = None
    class_name: str | None = None
    impact_score: float = Field(ge=0, le=1, default=1.0)
    relationship_type: str = "direct"


class BulkMappingRequest(BaseModel):
    """Request to create multiple mappings."""

    mappings: list[CreateMappingRequest]


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_lcov(content: str) -> list[dict]:
    """Parse LCOV format coverage data.

    LCOV format:
    SF:<source file path>
    DA:<line number>,<execution count>
    ...
    end_of_record
    """
    mappings = []
    current_file = None
    current_lines = []

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("SF:"):
            current_file = line[3:]
            current_lines = []
        elif line.startswith("DA:"):
            parts = line[3:].split(",")
            if len(parts) >= 2:
                line_num, count = int(parts[0]), int(parts[1])
                if count > 0:
                    current_lines.append((line_num, count))
        elif line == "end_of_record" and current_file:
            if current_lines:
                total_hits = sum(c for _, c in current_lines)
                mappings.append({
                    "file_path": current_file,
                    "hit_count": total_hits,
                    "lines_covered": len(current_lines),
                })
            current_file = None
            current_lines = []

    return mappings


def _parse_istanbul(content: str) -> list[dict]:
    """Parse Istanbul/NYC JSON coverage format."""
    import json

    mappings = []
    try:
        data = json.loads(content)
        for file_path, coverage in data.items():
            if isinstance(coverage, dict):
                # Count executed statements
                stmt_hits = sum(1 for v in coverage.get("s", {}).values() if v > 0)
                fn_hits = sum(1 for v in coverage.get("f", {}).values() if v > 0)
                total_hits = stmt_hits + fn_hits
                if total_hits > 0:
                    mappings.append({
                        "file_path": file_path,
                        "hit_count": total_hits,
                        "lines_covered": stmt_hits,
                    })
    except json.JSONDecodeError:
        logger.warning("Failed to parse Istanbul JSON")

    return mappings


def _parse_cobertura(content: str) -> list[dict]:
    """Parse Cobertura XML coverage format."""
    import xml.etree.ElementTree as ET

    mappings = []
    try:
        root = ET.fromstring(content)
        for package in root.findall(".//package"):
            for cls in package.findall(".//class"):
                filename = cls.get("filename", "")
                lines_covered = 0
                total_hits = 0
                for line in cls.findall(".//line"):
                    hits = int(line.get("hits", "0"))
                    if hits > 0:
                        lines_covered += 1
                        total_hits += hits
                if lines_covered > 0:
                    mappings.append({
                        "file_path": filename,
                        "hit_count": total_hits,
                        "lines_covered": lines_covered,
                    })
    except ET.ParseError:
        logger.warning("Failed to parse Cobertura XML")

    return mappings


def _detect_coverage_format(content: str) -> str:
    """Detect the format of coverage data."""
    content_start = content[:1000].strip()

    if content_start.startswith("TN:") or content_start.startswith("SF:"):
        return "lcov"
    elif content_start.startswith("{"):
        return "istanbul"
    elif content_start.startswith("<?xml") or "<coverage" in content_start:
        return "cobertura"
    else:
        return "unknown"


async def _process_coverage_import(
    import_id: str,
    project_id: str,
    content: str,
    format_type: str,
    test_file_pattern: str = r"test[s]?[/_]|_test\.py$|\.test\.(ts|js)$|spec\.(ts|js)$",
):
    """Background task to process coverage data import."""
    supabase = get_supabase_client()

    try:
        # Update status to processing
        await supabase.update(
            "coverage_imports",
            {"id": f"eq.{import_id}"},
            {"status": "processing", "started_at": datetime.now(UTC).isoformat()},
        )

        # Parse coverage data
        if format_type == "lcov":
            parsed = _parse_lcov(content)
        elif format_type == "istanbul":
            parsed = _parse_istanbul(content)
        elif format_type == "cobertura":
            parsed = _parse_cobertura(content)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Separate source files and test files
        test_pattern = re.compile(test_file_pattern, re.IGNORECASE)
        test_files = [p for p in parsed if test_pattern.search(p["file_path"])]
        source_files = [p for p in parsed if not test_pattern.search(p["file_path"])]

        mappings_created = 0
        files_processed = len(parsed)

        # Create mappings: for each source file, find which tests covered it
        # This is a simplified approach - in reality, we'd need per-test coverage
        for source in source_files:
            for test in test_files:
                # Call the upsert function
                result = await supabase.rpc(
                    "upsert_coverage_mapping",
                    {
                        "p_project_id": project_id,
                        "p_file_path": source["file_path"],
                        "p_test_file_path": test["file_path"],
                        "p_test_name": test["file_path"].split("/")[-1],
                        "p_coverage_run_id": import_id,
                        "p_hit_count": min(source["hit_count"], test["hit_count"]),
                    },
                )
                if result.get("data"):
                    mappings_created += 1

        # Update import record
        await supabase.update(
            "coverage_imports",
            {"id": f"eq.{import_id}"},
            {
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "files_processed": files_processed,
                "mappings_created": mappings_created,
            },
        )

        logger.info(
            "Coverage import completed",
            import_id=import_id,
            files_processed=files_processed,
            mappings_created=mappings_created,
        )

    except Exception as e:
        logger.exception("Coverage import failed", import_id=import_id, error=str(e))
        await supabase.update(
            "coverage_imports",
            {"id": f"eq.{import_id}"},
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )


# =============================================================================
# Endpoints - Query
# =============================================================================


@router.get("/tests", response_model=AffectedTestsResponse)
async def get_affected_tests(
    request: Request,
    files: Annotated[list[str], Query(description="File paths to check")],
    project_id: str = Query(..., description="Project ID"),
    min_score: float = Query(0.3, ge=0, le=1, description="Minimum impact score threshold"),
    limit: int = Query(100, ge=1, le=500, description="Maximum tests to return"),
    user: UserContext = Depends(get_current_user),
):
    """Get tests affected by the given files.

    This is the primary endpoint for commit impact prediction.
    Given a list of changed files, returns the tests most likely
    to be affected, sorted by impact score.
    """
    supabase = get_supabase_client()

    try:
        # Call the database function
        result = await supabase.rpc(
            "get_affected_tests",
            {
                "p_project_id": project_id,
                "p_file_paths": files,
                "p_min_score": min_score,
                "p_limit": limit,
            },
        )

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return AffectedTestsResponse(
                    files=files,
                    tests=[],
                    total_tests=0,
                    min_score_used=min_score,
                )
            raise HTTPException(status_code=500, detail="Failed to query impact graph")

        tests = [
            AffectedTest(
                test_id=row.get("test_id"),
                test_file_path=row["test_file_path"],
                test_name=row["test_name"],
                test_function=row.get("test_function"),
                total_impact_score=float(row["total_impact_score"]),
                relationship_types=row.get("relationship_types", []),
                affected_files=row.get("affected_files", []),
            )
            for row in (result.get("data") or [])
        ]

        logger.info(
            "Affected tests query",
            project_id=project_id,
            files_count=len(files),
            tests_found=len(tests),
            user_id=user.user_id,
        )

        return AffectedTestsResponse(
            files=files,
            tests=tests,
            total_tests=len(tests),
            min_score_used=min_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to query affected tests", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/stats", response_model=ImpactGraphStats)
async def get_impact_graph_stats(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Get statistics for a project's impact graph."""
    supabase = get_supabase_client()

    try:
        # Get mapping counts
        mappings_result = await supabase.request(
            f"/test_impact_graph?project_id=eq.{project_id}&select=id,relationship_type,impact_score"
        )

        mappings = mappings_result.get("data") or []
        total_mappings = len(mappings)

        # Count by relationship type
        by_type: dict[str, int] = {}
        total_score = 0.0
        for m in mappings:
            rel_type = m.get("relationship_type", "unknown")
            by_type[rel_type] = by_type.get(rel_type, 0) + 1
            total_score += float(m.get("impact_score", 0))

        avg_score = total_score / total_mappings if total_mappings > 0 else 0

        # Get unique file count
        files_result = await supabase.request(
            f"/test_impact_graph?project_id=eq.{project_id}&select=file_path"
        )
        unique_files = len(set(r.get("file_path") for r in (files_result.get("data") or [])))

        # Get unique test count
        tests_result = await supabase.request(
            f"/test_impact_graph?project_id=eq.{project_id}&select=test_file_path,test_name"
        )
        unique_tests = len(set(
            (r.get("test_file_path"), r.get("test_name"))
            for r in (tests_result.get("data") or [])
        ))

        # Get coverage imports count
        imports_result = await supabase.request(
            f"/coverage_imports?project_id=eq.{project_id}&select=id"
        )
        imports_count = len(imports_result.get("data") or [])

        # Get pending jobs count
        jobs_result = await supabase.request(
            f"/impact_graph_jobs?project_id=eq.{project_id}"
            "&status=in.(pending,running)&select=id"
        )
        pending_jobs = len(jobs_result.get("data") or [])

        # Get last update time
        last_updated_result = await supabase.request(
            f"/test_impact_graph?project_id=eq.{project_id}"
            "&order=updated_at.desc&limit=1&select=updated_at"
        )
        last_updated = None
        if last_updated_result.get("data"):
            last_updated = datetime.fromisoformat(
                last_updated_result["data"][0]["updated_at"].replace("Z", "+00:00")
            )

        return ImpactGraphStats(
            project_id=project_id,
            total_mappings=total_mappings,
            unique_source_files=unique_files,
            unique_tests=unique_tests,
            by_relationship_type=by_type,
            avg_impact_score=round(avg_score, 3),
            last_updated=last_updated,
            coverage_imports_count=imports_count,
            pending_jobs=pending_jobs,
        )

    except Exception as e:
        logger.exception("Failed to get impact graph stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/mappings")
async def list_mappings(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    file_path: str | None = Query(None, description="Filter by source file path"),
    test_file_path: str | None = Query(None, description="Filter by test file path"),
    relationship_type: str | None = Query(None, description="Filter by relationship type"),
    min_score: float = Query(0, ge=0, le=1, description="Minimum impact score"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user: UserContext = Depends(get_current_user),
):
    """List impact graph mappings with optional filters."""
    supabase = get_supabase_client()

    try:
        query = f"/test_impact_graph?project_id=eq.{project_id}"

        if file_path:
            query += f"&file_path=eq.{file_path}"
        if test_file_path:
            query += f"&test_file_path=eq.{test_file_path}"
        if relationship_type:
            query += f"&relationship_type=eq.{relationship_type}"
        if min_score > 0:
            query += f"&impact_score=gte.{min_score}"

        query += f"&order=impact_score.desc&limit={limit}&offset={offset}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"mappings": [], "total": 0}
            raise HTTPException(status_code=500, detail="Failed to list mappings")

        mappings = result.get("data") or []

        return {
            "mappings": mappings,
            "total": len(mappings),
            "limit": limit,
            "offset": offset,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list mappings", error=str(e))
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


# =============================================================================
# Endpoints - Mutations
# =============================================================================


@router.post("/mappings")
async def create_mapping(
    request: Request,
    mapping: CreateMappingRequest,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Create a manual impact mapping."""
    supabase = get_supabase_client()

    try:
        result = await supabase.insert(
            "test_impact_graph",
            {
                "project_id": project_id,
                "file_path": mapping.file_path,
                "function_name": mapping.function_name,
                "class_name": mapping.class_name,
                "test_file_path": mapping.test_file_path,
                "test_name": mapping.test_name,
                "test_function": mapping.test_function,
                "impact_score": mapping.impact_score,
                "relationship_type": mapping.relationship_type,
                "confidence": 1.0,
                "verification_method": "manual",
                "last_verified_at": datetime.now(UTC).isoformat(),
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create mapping")

        logger.info(
            "Manual mapping created",
            project_id=project_id,
            file_path=mapping.file_path,
            test_name=mapping.test_name,
            user_id=user.user_id,
        )

        return {"success": True, "mapping": result.get("data", [{}])[0]}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create mapping", error=str(e))
        raise HTTPException(status_code=500, detail=f"Create failed: {str(e)}")


@router.post("/mappings/bulk")
async def create_bulk_mappings(
    request: Request,
    bulk: BulkMappingRequest,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Create multiple impact mappings at once."""
    supabase = get_supabase_client()

    try:
        rows = [
            {
                "project_id": project_id,
                "file_path": m.file_path,
                "function_name": m.function_name,
                "class_name": m.class_name,
                "test_file_path": m.test_file_path,
                "test_name": m.test_name,
                "test_function": m.test_function,
                "impact_score": m.impact_score,
                "relationship_type": m.relationship_type,
                "confidence": 1.0,
                "verification_method": "manual",
                "last_verified_at": datetime.now(UTC).isoformat(),
            }
            for m in bulk.mappings
        ]

        result = await supabase.insert("test_impact_graph", rows)

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create mappings")

        created = result.get("data") or []

        logger.info(
            "Bulk mappings created",
            project_id=project_id,
            count=len(created),
            user_id=user.user_id,
        )

        return {"success": True, "created": len(created)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create bulk mappings", error=str(e))
        raise HTTPException(status_code=500, detail=f"Bulk create failed: {str(e)}")


@router.delete("/mappings/{mapping_id}")
async def delete_mapping(
    mapping_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Delete an impact mapping."""
    supabase = get_supabase_client()

    try:
        result = await supabase.delete(
            "test_impact_graph",
            {"id": f"eq.{mapping_id}"},
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to delete mapping")

        logger.info(
            "Mapping deleted",
            mapping_id=mapping_id,
            user_id=user.user_id,
        )

        return {"success": True, "deleted_id": mapping_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete mapping", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# =============================================================================
# Endpoints - Coverage Import
# =============================================================================


@router.post("/coverage/upload")
async def upload_coverage(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Query(..., description="Project ID"),
    format: str | None = Query(None, description="Coverage format (auto-detected if not provided)"),
    commit_sha: str | None = Query(None, description="Git commit SHA"),
    branch: str | None = Query(None, description="Git branch name"),
    user: UserContext = Depends(get_current_user),
):
    """Upload coverage data file for import.

    Supported formats: lcov, istanbul, cobertura
    """
    supabase = get_supabase_client()

    try:
        # Read file content
        content = await file.read()
        content_str = content.decode("utf-8")

        # Detect format if not specified
        format_type = format or _detect_coverage_format(content_str)
        if format_type == "unknown":
            raise HTTPException(
                status_code=400,
                detail="Could not detect coverage format. Specify format parameter."
            )

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(content).hexdigest()

        # Check for duplicate import
        existing = await supabase.request(
            f"/coverage_imports?project_id=eq.{project_id}&file_hash=eq.{file_hash}"
        )
        if existing.get("data"):
            return {
                "success": False,
                "message": "This coverage file has already been imported",
                "existing_import_id": existing["data"][0]["id"],
            }

        # Create import record
        result = await supabase.insert(
            "coverage_imports",
            {
                "project_id": project_id,
                "file_name": file.filename,
                "file_hash": file_hash,
                "format": format_type,
                "status": "pending",
                "commit_sha": commit_sha,
                "branch": branch,
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create import record")

        import_record = result["data"][0]
        import_id = import_record["id"]

        # Schedule background processing
        background_tasks.add_task(
            _process_coverage_import,
            import_id,
            project_id,
            content_str,
            format_type,
        )

        logger.info(
            "Coverage import started",
            import_id=import_id,
            project_id=project_id,
            format=format_type,
            file_name=file.filename,
            user_id=user.user_id,
        )

        return {
            "success": True,
            "import_id": import_id,
            "status": "pending",
            "format": format_type,
            "message": "Coverage import started. Check status with GET /coverage/{import_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to upload coverage", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/coverage/{import_id}", response_model=CoverageImport)
async def get_coverage_import(
    import_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get status of a coverage import."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/coverage_imports?id=eq.{import_id}")

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch import")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Import not found")

        record = result["data"][0]
        return CoverageImport(
            id=record["id"],
            file_name=record["file_name"],
            format=record["format"],
            status=record["status"],
            files_processed=record.get("files_processed", 0),
            mappings_created=record.get("mappings_created", 0),
            error_message=record.get("error_message"),
            created_at=datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get coverage import", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@router.get("/coverage")
async def list_coverage_imports(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    user: UserContext = Depends(get_current_user),
):
    """List coverage imports for a project."""
    supabase = get_supabase_client()

    try:
        query = f"/coverage_imports?project_id=eq.{project_id}"
        if status:
            query += f"&status=eq.{status}"
        query += f"&order=created_at.desc&limit={limit}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"imports": [], "total": 0}
            raise HTTPException(status_code=500, detail="Failed to list imports")

        imports = result.get("data") or []

        return {
            "imports": imports,
            "total": len(imports),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list coverage imports", error=str(e))
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


# =============================================================================
# Endpoints - Jobs
# =============================================================================


@router.post("/refresh")
async def trigger_refresh(
    request: Request,
    background_tasks: BackgroundTasks,
    project_id: str = Query(..., description="Project ID"),
    job_type: str = Query("incremental", description="Job type"),
    user: UserContext = Depends(get_current_user),
):
    """Trigger an impact graph refresh job."""
    supabase = get_supabase_client()

    valid_types = ["full_rebuild", "incremental", "ast_analysis", "historical", "semantic"]
    if job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job type. Must be one of: {', '.join(valid_types)}"
        )

    try:
        # Create job record
        result = await supabase.insert(
            "impact_graph_jobs",
            {
                "project_id": project_id,
                "job_type": job_type,
                "status": "pending",
                "triggered_by": "user",
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create job")

        job = result["data"][0]

        # TODO: Add background task to actually run the job
        # For now, just create the record

        logger.info(
            "Impact graph refresh triggered",
            job_id=job["id"],
            project_id=project_id,
            job_type=job_type,
            user_id=user.user_id,
        )

        return {
            "success": True,
            "job_id": job["id"],
            "status": "pending",
            "job_type": job_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to trigger refresh", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trigger failed: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get status of an impact graph job."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/impact_graph_jobs?id=eq.{job_id}")

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch job")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Job not found")

        record = result["data"][0]
        return JobStatus(
            id=record["id"],
            job_type=record["job_type"],
            status=record["status"],
            progress=float(record.get("progress", 0)),
            mappings_added=record.get("mappings_added", 0),
            mappings_updated=record.get("mappings_updated", 0),
            mappings_removed=record.get("mappings_removed", 0),
            files_analyzed=record.get("files_analyzed", 0),
            error_message=record.get("error_message"),
            created_at=datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            ),
            completed_at=datetime.fromisoformat(
                record["completed_at"].replace("Z", "+00:00")
            ) if record.get("completed_at") else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get job status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@router.get("/jobs")
async def list_jobs(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    user: UserContext = Depends(get_current_user),
):
    """List impact graph jobs for a project."""
    supabase = get_supabase_client()

    try:
        query = f"/impact_graph_jobs?project_id=eq.{project_id}"
        if status:
            query += f"&status=eq.{status}"
        query += f"&order=created_at.desc&limit={limit}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return {"jobs": [], "total": 0}
            raise HTTPException(status_code=500, detail="Failed to list jobs")

        jobs = result.get("data") or []

        return {
            "jobs": jobs,
            "total": len(jobs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list jobs", error=str(e))
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")
