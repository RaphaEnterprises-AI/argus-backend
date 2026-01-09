"""API endpoints for human-in-the-loop approvals.

Provides endpoints to:
- List pending approvals across all threads
- Get details of a specific pending approval
- Approve or reject pending actions
- Resume paused executions
- Modify state before approval

This integrates with LangGraph's interrupt_before breakpoints to enable
human approval workflows for self-healing and test plan validation.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

from src.orchestrator.checkpointer import get_checkpointer, CheckpointManager
from src.orchestrator.graph import create_testing_graph, get_interrupt_nodes
from src.config import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/approvals", tags=["Approvals"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ApprovalRequest(BaseModel):
    """Request to approve or reject a pending action."""

    thread_id: str = Field(..., description="Thread ID of the paused execution")
    approved: bool = Field(..., description="Whether to approve the action")
    modifications: Optional[Dict[str, Any]] = Field(
        None, description="Optional state modifications to apply before resuming"
    )
    reason: Optional[str] = Field(None, description="Reason for approval/rejection")


class PendingApproval(BaseModel):
    """A pending approval."""

    thread_id: str
    paused_at: str
    paused_before: str
    state_summary: Dict[str, Any]
    action_description: str
    created_at: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Response from an approval action."""

    status: str  # approved, rejected, error
    thread_id: str
    message: str
    result: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    """Request to resume a paused execution."""

    modifications: Optional[Dict[str, Any]] = Field(
        None, description="Optional state modifications to apply before resuming"
    )


class StateModification(BaseModel):
    """State modification request."""

    healing_queue: Optional[List[str]] = Field(
        None, description="Modified list of test IDs to heal"
    )
    should_continue: Optional[bool] = Field(
        None, description="Whether to continue execution"
    )
    test_plan: Optional[List[Dict[str, Any]]] = Field(
        None, description="Modified test plan"
    )
    custom: Optional[Dict[str, Any]] = Field(
        None, description="Custom state modifications"
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _build_action_description(next_node: str, values: dict) -> str:
    """Build a human-readable description of the pending action."""
    if next_node == "self_heal":
        failures = values.get("failures", [])
        healing_queue = values.get("healing_queue", [])
        failure_ids = [f.get("test_id", "unknown") for f in failures[:3]]

        if len(healing_queue) > 1:
            return f"Self-heal {len(healing_queue)} failed tests. Failed tests include: {', '.join(failure_ids)}"
        elif len(healing_queue) == 1:
            return f"Self-heal 1 failed test: {healing_queue[0]}"
        else:
            return "Self-healing action (no tests in queue)"

    elif next_node == "execute_test":
        test_plan = values.get("test_plan", [])
        return f"Execute test plan with {len(test_plan)} tests"

    elif next_node == "report":
        return "Generate final test report"

    else:
        return f"Execute node: {next_node}"


def _build_state_summary(values: dict) -> dict:
    """Build a summary of relevant state for display."""
    return {
        "iteration": values.get("iteration", 0),
        "passed_count": values.get("passed_count", 0),
        "failed_count": values.get("failed_count", 0),
        "skipped_count": values.get("skipped_count", 0),
        "healing_queue": values.get("healing_queue", []),
        "healing_queue_count": len(values.get("healing_queue", [])),
        "total_cost": values.get("total_cost", 0.0),
        "error": values.get("error"),
    }


async def _get_compiled_graph():
    """Get a compiled graph for state operations."""
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    interrupt_nodes = get_interrupt_nodes(settings)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes if interrupt_nodes else None,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/pending", response_model=List[PendingApproval])
async def list_pending_approvals():
    """
    List all pending approvals across all threads.

    Returns threads that are currently paused at an approval breakpoint,
    waiting for human intervention.
    """
    checkpoint_manager = CheckpointManager()

    try:
        pending = await checkpoint_manager.get_pending_approvals()

        result = []
        for p in pending:
            thread_id = p.get("thread_id", "unknown")

            # Get more details for each pending approval
            details = await checkpoint_manager.get_approval_details(thread_id)

            if details:
                context = details.get("context", {})
                result.append(
                    PendingApproval(
                        thread_id=thread_id,
                        paused_at=datetime.now(timezone.utc).isoformat(),
                        paused_before=p.get("paused_at", "unknown"),
                        state_summary=_build_state_summary(context),
                        action_description=details.get(
                            "description", "Pending action"
                        ),
                        created_at=p.get("created_at"),
                    )
                )

        return result

    except Exception as e:
        logger.exception("Failed to list pending approvals", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/{thread_id}", response_model=PendingApproval)
async def get_pending_approval(thread_id: str):
    """
    Get details of a pending approval for a specific thread.

    Returns detailed information about what action is pending
    and the current state of the execution.
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await app.aget_state(config)

        if not state or not state.next:
            raise HTTPException(
                status_code=404, detail="No pending approval for this thread"
            )

        next_node = state.next[0] if state.next else None
        values = state.values or {}

        # Check if paused at an approval point
        settings = get_settings()
        interrupt_nodes = get_interrupt_nodes(settings)

        if next_node not in interrupt_nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Thread is not paused at an approval point. Current node: {next_node}",
            )

        return PendingApproval(
            thread_id=thread_id,
            paused_at=datetime.now(timezone.utc).isoformat(),
            paused_before=next_node,
            state_summary=_build_state_summary(values),
            action_description=_build_action_description(next_node, values),
            created_at=state.metadata.get("created_at") if state.metadata else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get pending approval", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve", response_model=ApprovalResponse)
async def approve_action(request: ApprovalRequest):
    """
    Approve or reject a pending action and resume execution.

    If approved, the execution continues from where it was paused.
    If rejected, the execution is aborted with the provided reason.

    State modifications can be applied before resuming (e.g., modifying
    the healing queue to only heal specific tests).
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # Get current state
        state = await app.aget_state(config)

        if not state or not state.next:
            raise HTTPException(
                status_code=404, detail="No pending action for this thread"
            )

        if not request.approved:
            # Rejection: Update state to skip the action
            await app.aupdate_state(
                config,
                {
                    "healing_queue": [],  # Clear healing queue to skip healing
                    "error": f"Action rejected by user: {request.reason or 'No reason provided'}",
                    "should_continue": False,
                },
            )

            logger.info(
                "Action rejected",
                thread_id=request.thread_id,
                reason=request.reason,
            )

            # Resume to let it complete to report
            final_state = await app.ainvoke(None, config)

            return ApprovalResponse(
                status="rejected",
                thread_id=request.thread_id,
                message="Action was rejected. Execution skipped to reporting.",
                result={
                    "passed": final_state.get("passed_count", 0),
                    "failed": final_state.get("failed_count", 0),
                    "error": final_state.get("error"),
                },
            )

        # Approval: Apply any modifications and resume
        if request.modifications:
            logger.info(
                "Applying state modifications before resume",
                thread_id=request.thread_id,
                modifications=list(request.modifications.keys()),
            )
            await app.aupdate_state(config, request.modifications)

        logger.info("Action approved", thread_id=request.thread_id)

        # Resume execution
        final_state = await app.ainvoke(None, config)

        # Check if we hit another breakpoint
        new_state = await app.aget_state(config)
        is_paused = bool(new_state and new_state.next)

        return ApprovalResponse(
            status="approved",
            thread_id=request.thread_id,
            message="Action approved and execution resumed."
            + (" Paused at next breakpoint." if is_paused else " Completed."),
            result={
                "passed": final_state.get("passed_count", 0),
                "failed": final_state.get("failed_count", 0),
                "completed": not is_paused,
                "paused_at": new_state.next[0] if is_paused and new_state.next else None,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Approval error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/{thread_id}", response_model=ApprovalResponse)
async def resume_execution(thread_id: str, request: Optional[ResumeRequest] = None):
    """
    Resume a paused execution without explicit approval.

    Use this for non-approval breakpoints or when you've already
    modified the state and just want to continue.
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Check if there's a paused state
        state = await app.aget_state(config)

        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not state.next:
            raise HTTPException(
                status_code=400, detail="Thread is not paused - nothing to resume"
            )

        # Apply modifications if provided
        if request and request.modifications:
            await app.aupdate_state(config, request.modifications)

        # Resume execution
        final_state = await app.ainvoke(None, config)

        # Check if we hit another breakpoint
        new_state = await app.aget_state(config)
        is_paused = bool(new_state and new_state.next)

        return ApprovalResponse(
            status="resumed",
            thread_id=thread_id,
            message="Execution resumed."
            + (" Paused at next breakpoint." if is_paused else " Completed."),
            result={
                "passed": final_state.get("passed_count", 0),
                "failed": final_state.get("failed_count", 0),
                "completed": not is_paused,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Resume error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{thread_id}")
async def get_thread_state(thread_id: str):
    """
    Get the full state of a thread.

    Returns the current execution state, useful for debugging
    or building approval UIs.
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await app.aget_state(config)

        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        values = state.values or {}

        return {
            "thread_id": thread_id,
            "is_paused": bool(state.next),
            "next_node": state.next[0] if state.next else None,
            "all_next": state.next,
            "values": {
                # Return safe subset of values
                "run_id": values.get("run_id"),
                "started_at": values.get("started_at"),
                "iteration": values.get("iteration"),
                "passed_count": values.get("passed_count"),
                "failed_count": values.get("failed_count"),
                "skipped_count": values.get("skipped_count"),
                "total_cost": values.get("total_cost"),
                "current_test_index": values.get("current_test_index"),
                "test_plan_count": len(values.get("test_plan", [])),
                "healing_queue": values.get("healing_queue", []),
                "failures": [
                    {
                        "test_id": f.get("test_id"),
                        "failure_type": f.get("failure_type"),
                        "root_cause": f.get("root_cause", "")[:200],
                    }
                    for f in values.get("failures", [])[:10]
                ],
                "error": values.get("error"),
                "should_continue": values.get("should_continue"),
            },
            "metadata": state.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get thread state", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/state/{thread_id}")
async def update_thread_state(thread_id: str, modifications: StateModification):
    """
    Update the state of a paused thread.

    Allows modifying state before approving, such as:
    - Filtering the healing queue to specific tests
    - Modifying the test plan
    - Setting custom values
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await app.aget_state(config)

        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not state.next:
            raise HTTPException(
                status_code=400, detail="Cannot modify state of completed thread"
            )

        # Build updates from modifications
        updates = {}

        if modifications.healing_queue is not None:
            updates["healing_queue"] = modifications.healing_queue

        if modifications.should_continue is not None:
            updates["should_continue"] = modifications.should_continue

        if modifications.test_plan is not None:
            updates["test_plan"] = modifications.test_plan

        if modifications.custom:
            updates.update(modifications.custom)

        if not updates:
            raise HTTPException(status_code=400, detail="No modifications provided")

        # Apply updates
        await app.aupdate_state(config, updates)

        logger.info(
            "State updated",
            thread_id=thread_id,
            modified_keys=list(updates.keys()),
        )

        return {
            "success": True,
            "thread_id": thread_id,
            "modified_keys": list(updates.keys()),
            "message": "State updated successfully. Use /approve or /resume to continue.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update state", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{thread_id}")
async def abort_execution(thread_id: str, reason: Optional[str] = None):
    """
    Abort a paused execution.

    Sets the error state and completes the run without executing
    the pending action.
    """
    app = await _get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await app.aget_state(config)

        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not state.next:
            raise HTTPException(
                status_code=400, detail="Thread is not paused - nothing to abort"
            )

        abort_reason = reason or "Execution aborted by user"

        # Update state to abort
        await app.aupdate_state(
            config,
            {
                "error": abort_reason,
                "should_continue": False,
                "healing_queue": [],
            },
        )

        # Resume to complete reporting
        final_state = await app.ainvoke(None, config)

        logger.info("Execution aborted", thread_id=thread_id, reason=abort_reason)

        return {
            "success": True,
            "thread_id": thread_id,
            "message": "Execution aborted",
            "reason": abort_reason,
            "final_state": {
                "passed": final_state.get("passed_count", 0),
                "failed": final_state.get("failed_count", 0),
                "error": final_state.get("error"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Abort failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_approval_config():
    """
    Get the current approval configuration.

    Shows which breakpoints are enabled and their settings.
    """
    settings = get_settings()
    interrupt_nodes = get_interrupt_nodes(settings)

    return {
        "breakpoints_enabled": len(interrupt_nodes) > 0,
        "interrupt_before": interrupt_nodes,
        "settings": {
            "require_healing_approval": settings.require_healing_approval,
            "require_test_plan_approval": settings.require_test_plan_approval,
            "require_human_approval_for_healing": settings.require_human_approval_for_healing,
            "approval_timeout_seconds": settings.approval_timeout_seconds,
        },
    }
