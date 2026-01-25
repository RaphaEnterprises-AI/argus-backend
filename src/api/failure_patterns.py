"""Failure Pattern Learning System API.

Learns from historical test failures to predict future failures.
This is the machine learning core of the Commit Impact Predictor.

Pattern Types:
- file_change: Specific files cause failures
- author: Author-specific patterns
- time_of_day: Time-based patterns (Friday deploys)
- dependency: Dependency updates cause issues
- size: Large commits fail more
- component: Fragile components
- combination: File combinations that break together
- flaky: Inherently flaky tests
- environment: Environment-specific failures
"""

import fnmatch
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/patterns", tags=["Failure Patterns"])


# =============================================================================
# Models
# =============================================================================


class PatternDefinition(BaseModel):
    """Flexible pattern definition."""

    # For file_change patterns
    file_patterns: list[str] | None = None
    test_patterns: list[str] | None = None

    # For author patterns
    author_email: str | None = None
    author_name: str | None = None

    # For time_of_day patterns
    hours: list[int] | None = None
    days: list[int] | None = None  # 0=Monday, 6=Sunday

    # For size patterns
    min_lines: int | None = None
    min_files: int | None = None

    # For dependency patterns
    package: str | None = None
    version_pattern: str | None = None

    # For combination patterns
    files: list[str] | None = None
    together: bool = True

    # Generic
    extra: dict[str, Any] | None = None


class FailurePattern(BaseModel):
    """A learned failure pattern."""

    id: str
    project_id: str
    pattern_type: str
    pattern_definition: dict
    description: str | None = None
    target_tests: list[str] | None = None
    occurrences: int
    predictions_made: int
    predictions_correct: int
    accuracy: float
    confidence: float
    is_active: bool
    discovered_by: str
    created_at: datetime
    updated_at: datetime


class CreatePatternRequest(BaseModel):
    """Request to create a new pattern."""

    pattern_type: str = Field(
        ...,
        pattern="^(file_change|author|time_of_day|dependency|size|component|combination|flaky|environment)$"
    )
    pattern_definition: PatternDefinition
    description: str | None = None
    target_tests: list[str] | None = None
    target_test_patterns: list[str] | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)


class PredictionResult(BaseModel):
    """Result of failure prediction."""

    commit_sha: str
    failure_probability: float
    confidence_level: str  # 'high', 'medium', 'low'
    contributing_patterns: list[str]
    reasons: list[str]
    recommended_tests: list[str]


class RecordOutcomeRequest(BaseModel):
    """Request to record actual test outcome."""

    commit_sha: str
    test_name: str
    outcome: str = Field(..., pattern="^(passed|failed|skipped|error)$")
    test_file_path: str | None = None


class CommitFeatures(BaseModel):
    """Features extracted from a commit."""

    commit_sha: str
    author_email: str | None = None
    author_name: str | None = None
    commit_timestamp: datetime | None = None
    day_of_week: int | None = None
    hour_of_day: int | None = None
    files_changed: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    changed_files: list[str] = Field(default_factory=list)
    changed_directories: list[str] = Field(default_factory=list)
    file_extensions: list[str] = Field(default_factory=list)
    components_affected: list[str] = Field(default_factory=list)
    is_dependency_update: bool = False
    updated_dependencies: list[str] = Field(default_factory=list)
    touches_tests: bool = False
    touches_config: bool = False
    touches_migrations: bool = False
    is_merge_commit: bool = False
    is_revert: bool = False


class TrainingJobStatus(BaseModel):
    """Status of a training job."""

    id: str
    job_type: str
    status: str
    progress: float
    commits_analyzed: int
    test_runs_analyzed: int
    patterns_discovered: int
    patterns_updated: int
    patterns_deactivated: int
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AccuracyStats(BaseModel):
    """Overall prediction accuracy statistics."""

    project_id: str
    total_patterns: int
    active_patterns: int
    total_predictions: int
    correct_predictions: int
    overall_accuracy: float
    accuracy_by_type: dict[str, float]
    precision: float
    recall: float
    f1_score: float


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_commit_features(commit_data: dict) -> dict:
    """Extract features from commit data for pattern matching."""
    changed_files = commit_data.get("files", [])
    timestamp = commit_data.get("timestamp")

    # Parse timestamp
    dt = None
    if timestamp:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif isinstance(timestamp, datetime):
            dt = timestamp

    # Extract directories and extensions
    directories = set()
    extensions = set()
    for f in changed_files:
        if "/" in f:
            directories.add("/".join(f.split("/")[:-1]))
        if "." in f:
            extensions.add(f.split(".")[-1])

    # Detect dependency updates
    is_dep_update = any(
        f in changed_files or fnmatch.fnmatch(f, pattern)
        for f in changed_files
        for pattern in ["package.json", "package-lock.json", "requirements.txt",
                        "Pipfile*", "poetry.lock", "go.mod", "Cargo.toml"]
    )

    # Detect test/config/migration changes
    touches_tests = any(
        "test" in f.lower() or "spec" in f.lower()
        for f in changed_files
    )
    touches_config = any(
        f.endswith((".yml", ".yaml", ".toml", ".ini", ".env"))
        or "config" in f.lower()
        for f in changed_files
    )
    touches_migrations = any(
        "migration" in f.lower() or "migrate" in f.lower()
        for f in changed_files
    )

    return {
        "commit_sha": commit_data.get("sha", ""),
        "author_email": commit_data.get("author_email"),
        "author_name": commit_data.get("author_name"),
        "commit_timestamp": dt.isoformat() if dt else None,
        "day_of_week": dt.weekday() if dt else None,
        "hour_of_day": dt.hour if dt else None,
        "files_changed": len(changed_files),
        "lines_added": commit_data.get("lines_added", 0),
        "lines_deleted": commit_data.get("lines_deleted", 0),
        "total_lines_changed": commit_data.get("lines_added", 0) + commit_data.get("lines_deleted", 0),
        "changed_files": changed_files,
        "changed_directories": list(directories),
        "file_extensions": list(extensions),
        "is_dependency_update": is_dep_update,
        "touches_tests": touches_tests,
        "touches_config": touches_config,
        "touches_migrations": touches_migrations,
        "is_merge_commit": commit_data.get("is_merge", False),
        "is_revert": "revert" in commit_data.get("message", "").lower(),
    }


def _match_pattern(pattern: dict, features: dict) -> tuple[bool, float, str]:
    """Check if a pattern matches the commit features.

    Returns: (matches, score, reason)
    """
    pattern_type = pattern.get("pattern_type")
    definition = pattern.get("pattern_definition", {})

    if pattern_type == "file_change":
        file_patterns = definition.get("file_patterns", [])
        changed_files = features.get("changed_files", [])

        for file_pattern in file_patterns:
            for changed_file in changed_files:
                if fnmatch.fnmatch(changed_file, file_pattern):
                    return True, 1.0, f"File {changed_file} matches pattern {file_pattern}"
        return False, 0.0, ""

    elif pattern_type == "author":
        author_email = definition.get("author_email")
        if author_email and features.get("author_email") == author_email:
            return True, 1.0, f"Author {author_email} has historical failure pattern"
        return False, 0.0, ""

    elif pattern_type == "time_of_day":
        target_days = definition.get("days", [])
        target_hours = definition.get("hours", [])
        day = features.get("day_of_week")
        hour = features.get("hour_of_day")

        day_match = not target_days or day in target_days
        hour_match = not target_hours or hour in target_hours

        if day_match and hour_match:
            reasons = []
            if target_days:
                reasons.append(f"Day {day} in risky days")
            if target_hours:
                reasons.append(f"Hour {hour} in risky hours")
            return True, 0.8, "; ".join(reasons)
        return False, 0.0, ""

    elif pattern_type == "size":
        min_lines = definition.get("min_lines", 0)
        min_files = definition.get("min_files", 0)
        total_lines = features.get("total_lines_changed", 0)
        files_changed = features.get("files_changed", 0)

        if total_lines >= min_lines or files_changed >= min_files:
            return True, 0.9, f"Large commit: {total_lines} lines, {files_changed} files"
        return False, 0.0, ""

    elif pattern_type == "dependency":
        if features.get("is_dependency_update"):
            package = definition.get("package")
            if package:
                deps = features.get("updated_dependencies", [])
                if package in deps:
                    return True, 1.0, f"Dependency {package} updated"
            else:
                return True, 0.7, "Dependency update detected"
        return False, 0.0, ""

    elif pattern_type == "combination":
        required_files = definition.get("files", [])
        changed_files = features.get("changed_files", [])
        together = definition.get("together", True)

        matches = sum(1 for f in required_files if f in changed_files)

        if together and matches == len(required_files):
            return True, 1.0, f"Files {required_files} changed together"
        elif not together and matches > 0:
            return True, 0.8, f"{matches}/{len(required_files)} risky files changed"
        return False, 0.0, ""

    elif pattern_type == "flaky":
        # Flaky tests always "match" - they predict random failures
        return True, 0.5, "Test has flaky history"

    elif pattern_type == "component":
        components = definition.get("components", [])
        changed_dirs = features.get("changed_directories", [])

        for comp in components:
            for dir_ in changed_dirs:
                if comp in dir_:
                    return True, 0.9, f"Fragile component {comp} affected"
        return False, 0.0, ""

    return False, 0.0, ""


async def _discover_patterns_from_failure(
    project_id: str,
    commit_features: dict,
    failed_tests: list[str],
) -> list[dict]:
    """Discover new patterns from a failure occurrence."""
    new_patterns = []

    # File change pattern: which files were changed when this test failed
    changed_files = commit_features.get("changed_files", [])
    if changed_files and failed_tests:
        # Group by directory
        dir_patterns = {}
        for f in changed_files:
            if "/" in f:
                dir_pattern = "/".join(f.split("/")[:-1]) + "/*"
                dir_patterns.setdefault(dir_pattern, []).append(f)

        for dir_pattern, files in dir_patterns.items():
            if len(files) >= 2:  # Only if multiple files in same dir
                new_patterns.append({
                    "pattern_type": "file_change",
                    "pattern_definition": {
                        "file_patterns": [dir_pattern],
                    },
                    "target_tests": failed_tests,
                    "description": f"Changes to {dir_pattern} may cause test failures",
                })

    # Size pattern: if this was a large commit
    total_lines = commit_features.get("total_lines_changed", 0)
    if total_lines > 200:
        new_patterns.append({
            "pattern_type": "size",
            "pattern_definition": {
                "min_lines": total_lines - 50,  # Slightly lower threshold
            },
            "target_tests": failed_tests,
            "description": f"Large commits (>{total_lines - 50} lines) may cause failures",
        })

    # Time pattern: Friday afternoon
    day = commit_features.get("day_of_week")
    hour = commit_features.get("hour_of_day")
    if day == 4 and hour and hour >= 14:  # Friday 2pm+
        new_patterns.append({
            "pattern_type": "time_of_day",
            "pattern_definition": {
                "days": [4],  # Friday
                "hours": list(range(14, 18)),  # 2pm-6pm
            },
            "description": "Friday afternoon commits have higher failure rate",
        })

    return new_patterns


async def _run_training_job(job_id: str, project_id: str, job_type: str, days: int = 30):
    """Background task to run pattern training."""
    supabase = get_supabase_client()

    try:
        # Update status to running
        await supabase.update(
            "pattern_training_jobs",
            {"id": f"eq.{job_id}"},
            {"status": "running", "started_at": datetime.now(UTC).isoformat()},
        )

        # Get historical test runs with failures
        since_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        runs_result = await supabase.request(
            f"/schedule_runs?project_id=eq.{project_id}"
            f"&created_at=gte.{since_date}"
            "&status=eq.failed"
            "&order=created_at.desc"
            "&limit=500"
        )

        runs = runs_result.get("data") or []
        commits_analyzed = 0
        test_runs_analyzed = len(runs)
        patterns_discovered = 0

        # Analyze each failed run
        for run in runs:
            # Get commit features
            commit_sha = run.get("commit_sha")
            if not commit_sha:
                continue

            features_result = await supabase.request(
                f"/commit_features?project_id=eq.{project_id}&commit_sha=eq.{commit_sha}"
            )

            if not features_result.get("data"):
                continue

            features = features_result["data"][0]
            commits_analyzed += 1

            # Get failed tests from this run
            failed_tests = run.get("failed_tests", [])
            if not failed_tests:
                continue

            # Discover patterns
            new_patterns = await _discover_patterns_from_failure(
                project_id, features, failed_tests
            )

            for pattern in new_patterns:
                # Check if similar pattern exists
                existing = await supabase.request(
                    f"/failure_patterns?project_id=eq.{project_id}"
                    f"&pattern_type=eq.{pattern['pattern_type']}"
                    "&is_active=eq.true"
                )

                if not existing.get("data"):
                    # Create new pattern
                    await supabase.insert("failure_patterns", {
                        "project_id": project_id,
                        "pattern_type": pattern["pattern_type"],
                        "pattern_definition": pattern["pattern_definition"],
                        "description": pattern.get("description"),
                        "target_tests": pattern.get("target_tests"),
                        "discovered_by": "algorithm",
                        "confidence": 0.5,
                    })
                    patterns_discovered += 1
                else:
                    # Update occurrence count
                    existing_pattern = existing["data"][0]
                    await supabase.update(
                        "failure_patterns",
                        {"id": f"eq.{existing_pattern['id']}"},
                        {
                            "occurrences": existing_pattern.get("occurrences", 0) + 1,
                            "last_occurrence": datetime.now(UTC).isoformat(),
                        },
                    )

            # Update progress
            progress = (commits_analyzed / max(1, len(runs))) * 100
            await supabase.update(
                "pattern_training_jobs",
                {"id": f"eq.{job_id}"},
                {"progress": progress, "commits_analyzed": commits_analyzed},
            )

        # Complete job
        await supabase.update(
            "pattern_training_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "progress": 100,
                "commits_analyzed": commits_analyzed,
                "test_runs_analyzed": test_runs_analyzed,
                "patterns_discovered": patterns_discovered,
            },
        )

        logger.info(
            "Training job completed",
            job_id=job_id,
            commits_analyzed=commits_analyzed,
            patterns_discovered=patterns_discovered,
        )

    except Exception as e:
        logger.exception("Training job failed", job_id=job_id, error=str(e))
        await supabase.update(
            "pattern_training_jobs",
            {"id": f"eq.{job_id}"},
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )


# =============================================================================
# Endpoints - Query
# =============================================================================


@router.get("", response_model=list[FailurePattern])
async def list_patterns(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    pattern_type: str | None = Query(None, description="Filter by pattern type"),
    is_active: bool | None = Query(None, description="Filter by active status"),
    min_confidence: float = Query(0, ge=0, le=1, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    user: UserContext = Depends(get_current_user),
):
    """List failure patterns for a project."""
    supabase = get_supabase_client()

    try:
        query = f"/failure_patterns?project_id=eq.{project_id}"

        if pattern_type:
            query += f"&pattern_type=eq.{pattern_type}"
        if is_active is not None:
            query += f"&is_active=eq.{str(is_active).lower()}"
        if min_confidence > 0:
            query += f"&confidence=gte.{min_confidence}"

        query += f"&order=confidence.desc,occurrences.desc&limit={limit}"

        result = await supabase.request(query)

        if result.get("error"):
            error_msg = str(result.get("error", ""))
            if "does not exist" in error_msg or "42P01" in error_msg:
                return []
            raise HTTPException(status_code=500, detail="Failed to list patterns")

        patterns = []
        for row in result.get("data") or []:
            # Calculate accuracy
            made = row.get("predictions_made", 0)
            correct = row.get("predictions_correct", 0)
            accuracy = correct / made if made > 0 else 0.5

            patterns.append(FailurePattern(
                id=row["id"],
                project_id=row["project_id"],
                pattern_type=row["pattern_type"],
                pattern_definition=row["pattern_definition"],
                description=row.get("description"),
                target_tests=row.get("target_tests"),
                occurrences=row.get("occurrences", 1),
                predictions_made=made,
                predictions_correct=correct,
                accuracy=round(accuracy, 2),
                confidence=float(row.get("confidence", 0.5)),
                is_active=row.get("is_active", True),
                discovered_by=row.get("discovered_by", "algorithm"),
                created_at=datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ),
                updated_at=datetime.fromisoformat(
                    row["updated_at"].replace("Z", "+00:00")
                ),
            ))

        return patterns

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list patterns", error=str(e))
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


@router.get("/{pattern_id}", response_model=FailurePattern)
async def get_pattern(
    pattern_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get a specific pattern by ID."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/failure_patterns?id=eq.{pattern_id}")

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch pattern")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Pattern not found")

        row = result["data"][0]
        made = row.get("predictions_made", 0)
        correct = row.get("predictions_correct", 0)
        accuracy = correct / made if made > 0 else 0.5

        return FailurePattern(
            id=row["id"],
            project_id=row["project_id"],
            pattern_type=row["pattern_type"],
            pattern_definition=row["pattern_definition"],
            description=row.get("description"),
            target_tests=row.get("target_tests"),
            occurrences=row.get("occurrences", 1),
            predictions_made=made,
            predictions_correct=correct,
            accuracy=round(accuracy, 2),
            confidence=float(row.get("confidence", 0.5)),
            is_active=row.get("is_active", True),
            discovered_by=row.get("discovered_by", "algorithm"),
            created_at=datetime.fromisoformat(
                row["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                row["updated_at"].replace("Z", "+00:00")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get pattern", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@router.get("/accuracy/stats", response_model=AccuracyStats)
async def get_accuracy_stats(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Get overall prediction accuracy statistics."""
    supabase = get_supabase_client()

    try:
        # Get all patterns
        patterns_result = await supabase.request(
            f"/failure_patterns?project_id=eq.{project_id}"
        )

        patterns = patterns_result.get("data") or []
        total_patterns = len(patterns)
        active_patterns = sum(1 for p in patterns if p.get("is_active"))

        # Aggregate statistics
        total_predictions = sum(p.get("predictions_made", 0) for p in patterns)
        correct_predictions = sum(p.get("predictions_correct", 0) for p in patterns)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # True positives, etc. for precision/recall
        true_positives = sum(p.get("true_positives", 0) for p in patterns)
        false_positives = sum(p.get("false_positives", 0) for p in patterns)
        false_negatives = sum(p.get("false_negatives", 0) for p in patterns)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Accuracy by type
        accuracy_by_type: dict[str, float] = {}
        type_stats: dict[str, dict] = {}

        for p in patterns:
            ptype = p.get("pattern_type", "unknown")
            if ptype not in type_stats:
                type_stats[ptype] = {"made": 0, "correct": 0}
            type_stats[ptype]["made"] += p.get("predictions_made", 0)
            type_stats[ptype]["correct"] += p.get("predictions_correct", 0)

        for ptype, stats in type_stats.items():
            if stats["made"] > 0:
                accuracy_by_type[ptype] = round(stats["correct"] / stats["made"], 2)

        return AccuracyStats(
            project_id=project_id,
            total_patterns=total_patterns,
            active_patterns=active_patterns,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            overall_accuracy=round(overall_accuracy, 3),
            accuracy_by_type=accuracy_by_type,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1_score=round(f1_score, 3),
        )

    except Exception as e:
        logger.exception("Failed to get accuracy stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


# =============================================================================
# Endpoints - Prediction
# =============================================================================


@router.post("/predict", response_model=PredictionResult)
async def predict_failure(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    commit_sha: str = Query(..., description="Commit SHA to predict"),
    user: UserContext = Depends(get_current_user),
):
    """Predict failure probability for a commit.

    Uses learned patterns to estimate which tests are likely to fail.
    """
    supabase = get_supabase_client()

    try:
        # Get commit features
        features_result = await supabase.request(
            f"/commit_features?project_id=eq.{project_id}&commit_sha=eq.{commit_sha}"
        )

        if not features_result.get("data"):
            raise HTTPException(
                status_code=404,
                detail="Commit features not found. Extract features first with POST /features"
            )

        features = features_result["data"][0]

        # Get active patterns
        patterns_result = await supabase.request(
            f"/failure_patterns?project_id=eq.{project_id}&is_active=eq.true&confidence=gte.0.3"
        )

        patterns = patterns_result.get("data") or []

        # Match patterns
        contributing_patterns = []
        reasons = []
        recommended_tests = set()
        total_score = 0.0
        total_weight = 0.0

        for pattern in patterns:
            matches, score, reason = _match_pattern(pattern, features)
            if matches and score > 0:
                contributing_patterns.append(pattern["id"])
                reasons.append(reason)
                total_score += pattern.get("confidence", 0.5) * score
                total_weight += score

                # Add target tests
                if pattern.get("target_tests"):
                    recommended_tests.update(pattern["target_tests"])

        # Calculate probability
        if total_weight > 0:
            failure_prob = min(1.0, total_score / total_weight)
        else:
            failure_prob = 0.1  # Base probability

        # Determine confidence level
        if failure_prob >= 0.7:
            confidence_level = "high"
        elif failure_prob >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        logger.info(
            "Failure prediction made",
            project_id=project_id,
            commit_sha=commit_sha,
            probability=failure_prob,
            patterns_matched=len(contributing_patterns),
            user_id=user.user_id,
        )

        return PredictionResult(
            commit_sha=commit_sha,
            failure_probability=round(failure_prob, 2),
            confidence_level=confidence_level,
            contributing_patterns=contributing_patterns,
            reasons=reasons,
            recommended_tests=list(recommended_tests),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to predict failure", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/outcome")
async def record_outcome(
    request: Request,
    outcome: RecordOutcomeRequest,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Record actual test outcome for learning.

    This updates pattern accuracy based on whether predictions were correct.
    """
    supabase = get_supabase_client()

    try:
        # Find predictions for this commit/test
        predictions_result = await supabase.request(
            f"/pattern_predictions?project_id=eq.{project_id}"
            f"&commit_sha=eq.{outcome.commit_sha}"
            f"&test_name=eq.{outcome.test_name}"
            "&actual_outcome=is.null"  # Not yet recorded
        )

        predictions = predictions_result.get("data") or []
        updated_count = 0

        for pred in predictions:
            # Call the database function to update pattern accuracy
            result = await supabase.rpc(
                "update_pattern_from_outcome",
                {
                    "p_prediction_id": pred["id"],
                    "p_actual_outcome": outcome.outcome,
                },
            )
            if result.get("data"):
                updated_count += 1

        logger.info(
            "Outcome recorded",
            project_id=project_id,
            commit_sha=outcome.commit_sha,
            test_name=outcome.test_name,
            outcome=outcome.outcome,
            predictions_updated=updated_count,
            user_id=user.user_id,
        )

        return {
            "success": True,
            "commit_sha": outcome.commit_sha,
            "test_name": outcome.test_name,
            "outcome": outcome.outcome,
            "predictions_updated": updated_count,
        }

    except Exception as e:
        logger.exception("Failed to record outcome", error=str(e))
        raise HTTPException(status_code=500, detail=f"Record failed: {str(e)}")


# =============================================================================
# Endpoints - Features
# =============================================================================


@router.post("/features")
async def extract_features(
    request: Request,
    features: CommitFeatures,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Extract and store features for a commit.

    Call this when a new commit is received to prepare for prediction.
    """
    supabase = get_supabase_client()

    try:
        # Build feature record
        record = {
            "project_id": project_id,
            "commit_sha": features.commit_sha,
            "author_email": features.author_email,
            "author_name": features.author_name,
            "commit_timestamp": features.commit_timestamp.isoformat() if features.commit_timestamp else None,
            "day_of_week": features.day_of_week,
            "hour_of_day": features.hour_of_day,
            "files_changed": features.files_changed,
            "lines_added": features.lines_added,
            "lines_deleted": features.lines_deleted,
            "total_lines_changed": features.lines_added + features.lines_deleted,
            "changed_files": features.changed_files,
            "changed_directories": features.changed_directories,
            "file_extensions": features.file_extensions,
            "components_affected": features.components_affected,
            "is_dependency_update": features.is_dependency_update,
            "updated_dependencies": features.updated_dependencies,
            "touches_tests": features.touches_tests,
            "touches_config": features.touches_config,
            "touches_migrations": features.touches_migrations,
            "is_merge_commit": features.is_merge_commit,
            "is_revert": features.is_revert,
        }

        # Upsert (update or insert)
        result = await supabase.request(
            "/commit_features",
            method="POST",
            body=record,
            headers={"Prefer": "resolution=merge-duplicates"},
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to store features")

        logger.info(
            "Features extracted",
            project_id=project_id,
            commit_sha=features.commit_sha,
            user_id=user.user_id,
        )

        return {"success": True, "commit_sha": features.commit_sha}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to extract features", error=str(e))
        raise HTTPException(status_code=500, detail=f"Extract failed: {str(e)}")


# =============================================================================
# Endpoints - Mutations
# =============================================================================


@router.post("", response_model=FailurePattern)
async def create_pattern(
    request: Request,
    pattern: CreatePatternRequest,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Create a new failure pattern manually."""
    supabase = get_supabase_client()

    try:
        result = await supabase.insert(
            "failure_patterns",
            {
                "project_id": project_id,
                "pattern_type": pattern.pattern_type,
                "pattern_definition": pattern.pattern_definition.model_dump(exclude_none=True),
                "description": pattern.description,
                "target_tests": pattern.target_tests,
                "target_test_patterns": pattern.target_test_patterns,
                "confidence": pattern.confidence,
                "discovered_by": "user",
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create pattern")

        row = result["data"][0]

        logger.info(
            "Pattern created",
            project_id=project_id,
            pattern_type=pattern.pattern_type,
            user_id=user.user_id,
        )

        return FailurePattern(
            id=row["id"],
            project_id=row["project_id"],
            pattern_type=row["pattern_type"],
            pattern_definition=row["pattern_definition"],
            description=row.get("description"),
            target_tests=row.get("target_tests"),
            occurrences=1,
            predictions_made=0,
            predictions_correct=0,
            accuracy=0.5,
            confidence=float(row.get("confidence", 0.5)),
            is_active=True,
            discovered_by="user",
            created_at=datetime.fromisoformat(
                row["created_at"].replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(
                row["updated_at"].replace("Z", "+00:00")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create pattern", error=str(e))
        raise HTTPException(status_code=500, detail=f"Create failed: {str(e)}")


@router.delete("/{pattern_id}")
async def deactivate_pattern(
    pattern_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Deactivate a pattern (soft delete)."""
    supabase = get_supabase_client()

    try:
        result = await supabase.update(
            "failure_patterns",
            {"id": f"eq.{pattern_id}"},
            {"is_active": False},
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to deactivate pattern")

        logger.info(
            "Pattern deactivated",
            pattern_id=pattern_id,
            user_id=user.user_id,
        )

        return {"success": True, "pattern_id": pattern_id, "is_active": False}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to deactivate pattern", error=str(e))
        raise HTTPException(status_code=500, detail=f"Deactivate failed: {str(e)}")


@router.post("/{pattern_id}/activate")
async def activate_pattern(
    pattern_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Reactivate a deactivated pattern."""
    supabase = get_supabase_client()

    try:
        result = await supabase.update(
            "failure_patterns",
            {"id": f"eq.{pattern_id}"},
            {"is_active": True},
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to activate pattern")

        logger.info(
            "Pattern activated",
            pattern_id=pattern_id,
            user_id=user.user_id,
        )

        return {"success": True, "pattern_id": pattern_id, "is_active": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to activate pattern", error=str(e))
        raise HTTPException(status_code=500, detail=f"Activate failed: {str(e)}")


# =============================================================================
# Endpoints - Training
# =============================================================================


@router.post("/train", response_model=TrainingJobStatus)
async def train_patterns(
    request: Request,
    background_tasks: BackgroundTasks,
    project_id: str = Query(..., description="Project ID"),
    job_type: str = Query("incremental", description="Training job type"),
    days: int = Query(30, ge=1, le=365, description="Days of history to analyze"),
    user: UserContext = Depends(get_current_user),
):
    """Trigger pattern training from historical data."""
    supabase = get_supabase_client()

    valid_types = ["full_history", "incremental", "pattern_discovery", "accuracy_update", "cleanup"]
    if job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job type. Must be one of: {', '.join(valid_types)}"
        )

    try:
        # Create job record
        result = await supabase.insert(
            "pattern_training_jobs",
            {
                "project_id": project_id,
                "job_type": job_type,
                "status": "pending",
                "start_date": (datetime.now(UTC) - timedelta(days=days)).isoformat(),
                "end_date": datetime.now(UTC).isoformat(),
            },
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to create training job")

        job = result["data"][0]

        # Schedule background training
        background_tasks.add_task(_run_training_job, job["id"], project_id, job_type, days)

        logger.info(
            "Training job started",
            job_id=job["id"],
            project_id=project_id,
            job_type=job_type,
            days=days,
            user_id=user.user_id,
        )

        return TrainingJobStatus(
            id=job["id"],
            job_type=job_type,
            status="pending",
            progress=0,
            commits_analyzed=0,
            test_runs_analyzed=0,
            patterns_discovered=0,
            patterns_updated=0,
            patterns_deactivated=0,
            created_at=datetime.fromisoformat(
                job["created_at"].replace("Z", "+00:00")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start training", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/train/{job_id}", response_model=TrainingJobStatus)
async def get_training_status(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
):
    """Get status of a training job."""
    supabase = get_supabase_client()

    try:
        result = await supabase.request(f"/pattern_training_jobs?id=eq.{job_id}")

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to fetch job")

        if not result.get("data"):
            raise HTTPException(status_code=404, detail="Job not found")

        job = result["data"][0]

        return TrainingJobStatus(
            id=job["id"],
            job_type=job["job_type"],
            status=job["status"],
            progress=float(job.get("progress", 0)),
            commits_analyzed=job.get("commits_analyzed", 0),
            test_runs_analyzed=job.get("test_runs_analyzed", 0),
            patterns_discovered=job.get("patterns_discovered", 0),
            patterns_updated=job.get("patterns_updated", 0),
            patterns_deactivated=job.get("patterns_deactivated", 0),
            error_message=job.get("error_message"),
            created_at=datetime.fromisoformat(
                job["created_at"].replace("Z", "+00:00")
            ),
            completed_at=datetime.fromisoformat(
                job["completed_at"].replace("Z", "+00:00")
            ) if job.get("completed_at") else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get training status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")
