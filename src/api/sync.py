"""Sync API endpoints for two-way IDE synchronization."""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

from src.sync import (
    SyncManager,
    SyncConfig,
    SyncSource,
    SyncEventType,
    SyncStatus,
    ConflictResolutionStrategy,
    create_sync_manager,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/sync", tags=["Sync"])

# Global sync manager instance (would be per-user in production)
_sync_managers: dict[str, SyncManager] = {}


def get_sync_manager(project_id: str) -> SyncManager:
    """Get or create a sync manager for a project."""
    if project_id not in _sync_managers:
        _sync_managers[project_id] = create_sync_manager(
            auto_sync=False,  # Manual sync via API
            resolution_strategy=ConflictResolutionStrategy.MERGE,
        )
    return _sync_managers[project_id]


# =============================================================================
# Request/Response Models
# =============================================================================


class TestStepModel(BaseModel):
    """Test step model."""
    action: str
    target: Optional[str] = None
    value: Optional[str] = None


class TestAssertionModel(BaseModel):
    """Test assertion model."""
    type: str
    target: Optional[str] = None
    expected: Optional[str] = None


class TestContentModel(BaseModel):
    """Test content model for sync."""
    id: str
    name: str
    description: Optional[str] = None
    steps: list[TestStepModel]
    assertions: Optional[list[TestAssertionModel]] = None
    metadata: Optional[dict] = None


class SyncPushRequest(BaseModel):
    """Request to push local changes."""
    project_id: str = Field(..., description="Project UUID")
    test_id: str = Field(..., description="Test UUID")
    content: TestContentModel = Field(..., description="Test specification")
    local_version: int = Field(..., description="Local version number")
    source: str = Field("mcp", description="Source of the push (ide, mcp)")


class SyncPushResponse(BaseModel):
    """Response from push operation."""
    success: bool
    events_pushed: int = 0
    new_version: Optional[int] = None
    conflicts: Optional[list[dict]] = None
    error: Optional[str] = None


class SyncPullResponse(BaseModel):
    """Response from pull operation."""
    success: bool
    events: list[dict] = []
    new_version: Optional[int] = None
    error: Optional[str] = None


class SyncStatusResponse(BaseModel):
    """Response for sync status."""
    success: bool
    project_id: str
    status: str
    tests: dict = {}
    total_pending: int = 0
    total_conflicts: int = 0


class SyncResolveRequest(BaseModel):
    """Request to resolve a conflict."""
    project_id: str
    conflict_id: str
    strategy: str = Field(..., description="keep_local, keep_remote, merge, manual")
    manual_value: Optional[dict] = None


class SyncResolveResponse(BaseModel):
    """Response from conflict resolution."""
    success: bool
    resolved: bool = False
    conflict_id: str
    resolved_value: Optional[dict] = None
    error: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/push", response_model=SyncPushResponse)
async def push_changes(request: SyncPushRequest):
    """
    Push local test changes to Argus cloud.

    Syncs test specifications from IDE to the platform.
    Detects conflicts if remote has diverged.
    """
    try:
        manager = get_sync_manager(request.project_id)

        # Convert content to dict
        spec = request.content.model_dump()

        # Track test if not already tracked
        existing = manager.get_local_spec(request.test_id)
        if existing is None:
            manager.track_test(request.project_id, request.test_id, spec)

        # Update local and get diff
        diff = manager.update_local(request.project_id, request.test_id, spec)

        if not diff.has_changes:
            return SyncPushResponse(
                success=True,
                events_pushed=0,
                new_version=request.local_version,
            )

        # In production, this would push to database
        # For now, simulate successful push
        project_state = manager.get_project_state(request.project_id)
        test_state = project_state.get_test_state(request.test_id)

        # Clear pending changes (simulating successful push)
        events_count = len(test_state.pending_changes)
        test_state.pending_changes.clear()
        test_state.status = SyncStatus.SYNCED
        test_state.remote_version = test_state.local_version

        logger.info(
            "Sync push completed",
            project_id=request.project_id,
            test_id=request.test_id,
            events_pushed=events_count,
        )

        return SyncPushResponse(
            success=True,
            events_pushed=events_count,
            new_version=test_state.local_version,
        )

    except Exception as e:
        logger.exception("Sync push failed", error=str(e))
        return SyncPushResponse(
            success=False,
            error=str(e),
        )


@router.get("/pull", response_model=SyncPullResponse)
async def pull_changes(
    project_id: str,
    since_version: int = 0,
    test_id: Optional[str] = None,
):
    """
    Pull test changes from Argus cloud.

    Fetches the latest test specifications and updates from team members.
    """
    try:
        manager = get_sync_manager(project_id)

        # In production, this would fetch from database
        # For now, return empty (no remote changes)
        events = []

        logger.info(
            "Sync pull completed",
            project_id=project_id,
            since_version=since_version,
            events_count=len(events),
        )

        return SyncPullResponse(
            success=True,
            events=events,
            new_version=since_version,
        )

    except Exception as e:
        logger.exception("Sync pull failed", error=str(e))
        return SyncPullResponse(
            success=False,
            error=str(e),
        )


@router.get("/status/{project_id}", response_model=SyncStatusResponse)
async def get_sync_status(project_id: str):
    """
    Get synchronization status for a project.

    Shows pending changes, conflicts, and sync state for all tests.
    """
    try:
        manager = get_sync_manager(project_id)
        status = manager.get_sync_status(project_id)

        # Format tests for response
        tests = {}
        for test_id, test_state in status.get("tests", {}).items():
            tests[test_id] = {
                "test_id": test_id,
                "status": test_state.get("status", "synced"),
                "local_version": test_state.get("local_version", 0),
                "remote_version": test_state.get("remote_version", 0),
                "pending_changes": len(test_state.get("pending_changes", [])),
                "conflicts": len([c for c in test_state.get("conflicts", []) if not c.get("resolved")]),
            }

        return SyncStatusResponse(
            success=True,
            project_id=project_id,
            status=status.get("status", "synced"),
            tests=tests,
            total_pending=status.get("total_pending", 0),
            total_conflicts=status.get("total_conflicts", 0),
        )

    except Exception as e:
        logger.exception("Get sync status failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve", response_model=SyncResolveResponse)
async def resolve_conflict(request: SyncResolveRequest):
    """
    Resolve a synchronization conflict.

    Strategies:
    - keep_local: Keep local version
    - keep_remote: Keep remote version
    - merge: Attempt automatic merge
    - manual: Use provided manual_value
    """
    try:
        manager = get_sync_manager(request.project_id)

        # Map string strategy to enum
        strategy_map = {
            "keep_local": ConflictResolutionStrategy.KEEP_LOCAL,
            "keep_remote": ConflictResolutionStrategy.KEEP_REMOTE,
            "merge": ConflictResolutionStrategy.MERGE,
            "manual": ConflictResolutionStrategy.MANUAL,
        }

        strategy = strategy_map.get(request.strategy)
        if not strategy:
            return SyncResolveResponse(
                success=False,
                resolved=False,
                conflict_id=request.conflict_id,
                error=f"Invalid strategy: {request.strategy}",
            )

        # Resolve the conflict
        resolved = await manager.resolve_conflict(
            request.project_id,
            request.conflict_id,
            strategy,
            request.manual_value,
        )

        if resolved:
            return SyncResolveResponse(
                success=True,
                resolved=True,
                conflict_id=request.conflict_id,
                resolved_value=request.manual_value,
            )
        else:
            return SyncResolveResponse(
                success=False,
                resolved=False,
                conflict_id=request.conflict_id,
                error="Could not resolve conflict",
            )

    except Exception as e:
        logger.exception("Conflict resolution failed", error=str(e))
        return SyncResolveResponse(
            success=False,
            resolved=False,
            conflict_id=request.conflict_id,
            error=str(e),
        )


@router.post("/track")
async def track_test(
    project_id: str,
    test_id: str,
    content: TestContentModel,
):
    """
    Start tracking a test for synchronization.

    Call this when opening a test file in the IDE.
    """
    try:
        manager = get_sync_manager(project_id)
        spec = content.model_dump()

        manager.track_test(project_id, test_id, spec)

        return {
            "success": True,
            "message": f"Test {test_id} is now tracked for sync",
        }

    except Exception as e:
        logger.exception("Track test failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
