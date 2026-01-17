"""Time Travel API for debugging and state replay.

Enables browsing historical states, replaying from checkpoints,
and forking test runs from any point in history.
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from src.orchestrator.checkpointer import get_checkpointer
from src.orchestrator.graph import create_testing_graph
from src.config import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/time-travel", tags=["Time Travel"])


class CheckpointInfo(BaseModel):
    """Basic checkpoint information."""
    checkpoint_id: str
    parent_checkpoint_id: Optional[str] = None
    created_at: str
    next_node: Optional[str] = None


class CheckpointsResponse(BaseModel):
    """Response containing checkpoints for a thread."""
    thread_id: str
    checkpoints: List[CheckpointInfo]
    total_count: int


class StateSnapshot(BaseModel):
    """A snapshot of graph state at a point in time."""
    checkpoint_id: str
    parent_checkpoint_id: Optional[str] = None
    thread_id: str
    created_at: str
    next_node: Optional[str] = None
    state_summary: dict


class StateHistoryResponse(BaseModel):
    """Response containing state history."""
    thread_id: str
    snapshots: List[StateSnapshot]
    total_count: int


class ReplayRequest(BaseModel):
    """Request to replay from a checkpoint."""
    thread_id: str
    checkpoint_id: str
    new_thread_id: Optional[str] = None  # Fork to new thread


class ReplayResponse(BaseModel):
    """Response from replay operation."""
    success: bool
    source_checkpoint: str
    target_thread_id: str
    final_state_summary: dict


class ForkRequest(BaseModel):
    """Request to fork from a checkpoint with modifications."""
    thread_id: str
    checkpoint_id: str
    new_thread_id: str
    state_modifications: Optional[dict] = None


class ForkResponse(BaseModel):
    """Response from fork operation."""
    success: bool
    source_thread: str
    source_checkpoint: str
    new_thread_id: str
    modifications_applied: List[str]


class StateAtCheckpointResponse(BaseModel):
    """Full state at a specific checkpoint."""
    checkpoint_id: str
    thread_id: str
    next_node: Optional[str] = None
    state: dict
    metadata: dict


class CompareStatesResponse(BaseModel):
    """Response from state comparison."""
    thread_1: str
    thread_2: str
    checkpoint_1: Optional[str] = None
    checkpoint_2: Optional[str] = None
    differences: dict
    difference_count: int


@router.get("/checkpoints", response_model=CheckpointsResponse)
async def get_checkpoints(
    thread_id: str = Query(..., description="Thread ID to get checkpoints for"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of checkpoints to return"),
):
    """Get checkpoints for a thread.

    Returns a list of checkpoints from newest to oldest.
    This is a simplified endpoint that returns just checkpoint metadata
    without the full state summary.
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        checkpoints = []
        count = 0

        # Get state history using LangGraph's built-in method
        async for state in app.aget_state_history(config):
            count += 1

            current_checkpoint_id = state.config["configurable"].get("checkpoint_id", "")

            # Extract parent checkpoint ID if available
            parent_checkpoint_id = None
            if state.parent_config:
                parent_checkpoint_id = state.parent_config.get("configurable", {}).get("checkpoint_id")

            # Get created_at timestamp
            created_at = datetime.now().isoformat()
            if hasattr(state, 'created_at') and state.created_at:
                created_at = state.created_at.isoformat() if hasattr(state.created_at, 'isoformat') else str(state.created_at)

            checkpoint = CheckpointInfo(
                checkpoint_id=current_checkpoint_id,
                parent_checkpoint_id=parent_checkpoint_id,
                created_at=created_at,
                next_node=state.next[0] if state.next else None,
            )
            checkpoints.append(checkpoint)

            if len(checkpoints) >= limit:
                break

        logger.info(
            "Retrieved checkpoints",
            thread_id=thread_id,
            checkpoint_count=len(checkpoints),
            total_count=count,
        )

        return CheckpointsResponse(
            thread_id=thread_id,
            checkpoints=checkpoints,
            total_count=count,
        )

    except Exception as e:
        logger.exception("Failed to get checkpoints", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get checkpoints: {str(e)}")


@router.get("/history/{thread_id}", response_model=StateHistoryResponse)
async def get_state_history(
    thread_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of snapshots to return"),
    before_checkpoint: Optional[str] = Query(None, description="Checkpoint ID for pagination (return snapshots before this one)"),
):
    """Get the state history for a thread.

    Returns a list of checkpoints from newest to oldest.
    Use before_checkpoint for pagination.

    This endpoint enables time-travel debugging by showing all the states
    a test run has passed through, allowing developers to understand
    exactly what happened at each step.
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshots = []
        count = 0
        skip_until_found = before_checkpoint is not None

        # Get state history using LangGraph's built-in method
        async for state in app.aget_state_history(config):
            count += 1

            current_checkpoint_id = state.config["configurable"].get("checkpoint_id", "")

            # If we're paginating, skip until we find the before_checkpoint
            if skip_until_found:
                if current_checkpoint_id == before_checkpoint:
                    skip_until_found = False
                continue

            # Extract parent checkpoint ID if available
            parent_checkpoint_id = None
            if state.parent_config:
                parent_checkpoint_id = state.parent_config.get("configurable", {}).get("checkpoint_id")

            # Get created_at timestamp
            created_at = datetime.now().isoformat()
            if hasattr(state, 'created_at') and state.created_at:
                created_at = state.created_at.isoformat() if hasattr(state.created_at, 'isoformat') else str(state.created_at)

            # Build state summary with safe access
            values = state.values or {}
            state_summary = {
                "iteration": values.get("iteration", 0),
                "passed_count": values.get("passed_count", 0),
                "failed_count": values.get("failed_count", 0),
                "current_test": values.get("current_test_index", 0),
                "error": values.get("error"),
                "should_continue": values.get("should_continue", True),
                "healing_attempts": values.get("healing_attempts", 0),
            }

            snapshot = StateSnapshot(
                checkpoint_id=current_checkpoint_id,
                parent_checkpoint_id=parent_checkpoint_id,
                thread_id=thread_id,
                created_at=created_at,
                next_node=state.next[0] if state.next else None,
                state_summary=state_summary,
            )
            snapshots.append(snapshot)

            if len(snapshots) >= limit:
                break

        logger.info(
            "Retrieved state history",
            thread_id=thread_id,
            snapshot_count=len(snapshots),
            total_count=count,
        )

        return StateHistoryResponse(
            thread_id=thread_id,
            snapshots=snapshots,
            total_count=count,
        )

    except Exception as e:
        logger.exception("Failed to get state history", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get state history: {str(e)}")


@router.get("/state/{thread_id}/{checkpoint_id}", response_model=StateAtCheckpointResponse)
async def get_state_at_checkpoint(thread_id: str, checkpoint_id: str):
    """Get the full state at a specific checkpoint.

    Returns the complete state values at a given checkpoint,
    useful for debugging exactly what data was present at that moment.
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    try:
        state = await app.aget_state(config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Extract metadata safely
        metadata = {}
        if hasattr(state, 'metadata') and state.metadata:
            metadata = dict(state.metadata)

        logger.info(
            "Retrieved state at checkpoint",
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )

        return StateAtCheckpointResponse(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            next_node=state.next[0] if state.next else None,
            state=dict(state.values),
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get state", thread_id=thread_id, checkpoint_id=checkpoint_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@router.post("/replay", response_model=ReplayResponse)
async def replay_from_checkpoint(request: ReplayRequest):
    """Replay execution from a specific checkpoint.

    Continues execution from the given checkpoint state.
    Optionally fork to a new thread to preserve original history.

    This is useful for:
    - Re-running a test from a specific point after fixing an issue
    - Testing different paths from the same starting point
    - Debugging by replaying execution step by step
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    # Get the state at checkpoint
    source_config = {
        "configurable": {
            "thread_id": request.thread_id,
            "checkpoint_id": request.checkpoint_id,
        }
    }

    try:
        state = await app.aget_state(source_config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Determine target thread
        target_thread_id = request.new_thread_id or request.thread_id
        target_config = {"configurable": {"thread_id": target_thread_id}}

        # If forking to a new thread, copy the state first
        if request.new_thread_id:
            # Copy state to new thread
            await app.aupdate_state(target_config, dict(state.values))
            logger.info(
                "Forked state to new thread",
                source_thread=request.thread_id,
                target_thread=target_thread_id,
                checkpoint_id=request.checkpoint_id,
            )

        # Resume execution from checkpoint
        final_state = await app.ainvoke(None, target_config)

        # Build summary
        final_state_summary = {
            "iteration": final_state.get("iteration", 0),
            "passed_count": final_state.get("passed_count", 0),
            "failed_count": final_state.get("failed_count", 0),
            "skipped_count": final_state.get("skipped_count", 0),
            "error": final_state.get("error"),
        }

        logger.info(
            "Replay completed",
            source_checkpoint=request.checkpoint_id,
            target_thread=target_thread_id,
            final_summary=final_state_summary,
        )

        return ReplayResponse(
            success=True,
            source_checkpoint=request.checkpoint_id,
            target_thread_id=target_thread_id,
            final_state_summary=final_state_summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to replay", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to replay: {str(e)}")


@router.post("/fork", response_model=ForkResponse)
async def fork_from_checkpoint(request: ForkRequest):
    """Fork a test run from a checkpoint with optional modifications.

    Creates a new thread starting from the checkpoint state,
    optionally applying state modifications before continuing.

    This is useful for:
    - Testing "what if" scenarios by modifying state before continuing
    - Creating variations of test runs with different parameters
    - Experimenting with different healing strategies
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    source_config = {
        "configurable": {
            "thread_id": request.thread_id,
            "checkpoint_id": request.checkpoint_id,
        }
    }

    try:
        state = await app.aget_state(source_config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        # Apply modifications to state if provided
        new_state = dict(state.values)
        modifications_applied = []

        if request.state_modifications:
            for key, value in request.state_modifications.items():
                if key in new_state:
                    new_state[key] = value
                    modifications_applied.append(key)
                else:
                    logger.warning(
                        "Modification key not in state",
                        key=key,
                        available_keys=list(new_state.keys()),
                    )

        # Create new thread with modified state
        target_config = {"configurable": {"thread_id": request.new_thread_id}}

        # Update state in new thread
        await app.aupdate_state(target_config, new_state)

        logger.info(
            "Forked from checkpoint",
            source_thread=request.thread_id,
            source_checkpoint=request.checkpoint_id,
            new_thread=request.new_thread_id,
            modifications=modifications_applied,
        )

        return ForkResponse(
            success=True,
            source_thread=request.thread_id,
            source_checkpoint=request.checkpoint_id,
            new_thread_id=request.new_thread_id,
            modifications_applied=modifications_applied,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fork", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fork: {str(e)}")


@router.get("/compare/{thread_id_1}/{thread_id_2}", response_model=CompareStatesResponse)
async def compare_states(
    thread_id_1: str,
    thread_id_2: str,
    checkpoint_id_1: Optional[str] = Query(None, description="Checkpoint ID for thread 1 (latest if not specified)"),
    checkpoint_id_2: Optional[str] = Query(None, description="Checkpoint ID for thread 2 (latest if not specified)"),
):
    """Compare states between two threads or checkpoints.

    Useful for debugging why test runs diverged.
    Shows which state values differ between two execution points.
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    try:
        # Get state 1
        config_1 = {"configurable": {"thread_id": thread_id_1}}
        if checkpoint_id_1:
            config_1["configurable"]["checkpoint_id"] = checkpoint_id_1
        state_1 = await app.aget_state(config_1)

        # Get state 2
        config_2 = {"configurable": {"thread_id": thread_id_2}}
        if checkpoint_id_2:
            config_2["configurable"]["checkpoint_id"] = checkpoint_id_2
        state_2 = await app.aget_state(config_2)

        if not state_1 or not state_1.values:
            raise HTTPException(status_code=404, detail=f"State not found for thread {thread_id_1}")
        if not state_2 or not state_2.values:
            raise HTTPException(status_code=404, detail=f"State not found for thread {thread_id_2}")

        # Compare states
        differences = {}
        all_keys = set(state_1.values.keys()) | set(state_2.values.keys())

        for key in all_keys:
            val_1 = state_1.values.get(key)
            val_2 = state_2.values.get(key)

            if val_1 != val_2:
                # Truncate large values for readability
                str_val_1 = str(val_1)[:500] if val_1 is not None else None
                str_val_2 = str(val_2)[:500] if val_2 is not None else None

                differences[key] = {
                    "thread_1": str_val_1,
                    "thread_2": str_val_2,
                }

        logger.info(
            "Compared states",
            thread_1=thread_id_1,
            thread_2=thread_id_2,
            difference_count=len(differences),
        )

        return CompareStatesResponse(
            thread_1=thread_id_1,
            thread_2=thread_id_2,
            checkpoint_1=checkpoint_id_1,
            checkpoint_2=checkpoint_id_2,
            differences=differences,
            difference_count=len(differences),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compare states", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to compare states: {str(e)}")


@router.get("/threads")
async def list_threads(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of threads to return"),
    status: Optional[str] = Query(None, description="Filter by status: running, paused, completed"),
):
    """List all known thread IDs with their current status.

    Returns threads that have been checkpointed, along with their
    current state (running, paused at a breakpoint, or completed).
    """
    checkpointer = get_checkpointer()
    settings = get_settings()

    graph = create_testing_graph(settings)
    app = graph.compile(checkpointer=checkpointer)

    threads = []

    try:
        # For MemorySaver, we can iterate over stored checkpoints
        from langgraph.checkpoint.memory import MemorySaver

        if isinstance(checkpointer, MemorySaver):
            storage = getattr(checkpointer, 'storage', {})

            for thread_id in list(storage.keys())[:limit]:
                # Get the latest state for this thread
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    state = await app.aget_state(config)

                    if state:
                        # Determine status
                        thread_status = "completed"
                        if state.next:
                            thread_status = "paused"
                        elif state.values.get("should_continue", False):
                            thread_status = "running"

                        # Apply status filter if provided
                        if status and thread_status != status:
                            continue

                        threads.append({
                            "thread_id": thread_id,
                            "status": thread_status,
                            "next_node": state.next[0] if state.next else None,
                            "iteration": state.values.get("iteration", 0),
                            "passed_count": state.values.get("passed_count", 0),
                            "failed_count": state.values.get("failed_count", 0),
                        })
                except Exception:
                    continue
        else:
            # For PostgresSaver or other backends, we'd need different iteration
            # For now, return empty list with a note
            logger.info("Thread listing not fully supported for this checkpointer type")

        logger.info("Listed threads", count=len(threads))

        return {
            "threads": threads,
            "total": len(threads),
            "limit": limit,
        }

    except Exception as e:
        logger.exception("Failed to list threads", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {str(e)}")


@router.delete("/thread/{thread_id}")
async def delete_thread_history(thread_id: str):
    """Delete all checkpoints for a thread.

    This permanently removes all history for a test run.
    Use with caution - this cannot be undone.
    """
    checkpointer = get_checkpointer()

    try:
        # For MemorySaver, we can directly remove from storage
        from langgraph.checkpoint.memory import MemorySaver

        if isinstance(checkpointer, MemorySaver):
            storage = getattr(checkpointer, 'storage', {})

            if thread_id in storage:
                del storage[thread_id]
                logger.info("Deleted thread history", thread_id=thread_id)
                return {
                    "success": True,
                    "thread_id": thread_id,
                    "message": "Thread history deleted",
                }
            else:
                raise HTTPException(status_code=404, detail="Thread not found")
        else:
            # For PostgresSaver, we'd need to run SQL delete
            # This would require database access
            raise HTTPException(
                status_code=501,
                detail="Thread deletion not implemented for this checkpointer type"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete thread", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {str(e)}")
