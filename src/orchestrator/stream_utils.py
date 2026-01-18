"""Utilities for streaming custom events from nodes.

These utilities allow LangGraph nodes to emit custom events during execution
that are streamed to clients in real-time via SSE.
"""

from typing import Any

import structlog

logger = structlog.get_logger()


def get_stream_writer():
    """
    Get the stream writer from the current LangGraph context.

    Returns None if not in a streaming context.
    """
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except ImportError:
        logger.debug("LangGraph stream writer not available")
        return None
    except Exception as e:
        logger.debug("Not in streaming context", error=str(e))
        return None


def emit_progress(
    step: int,
    total: int,
    message: str,
    phase: str | None = None,
    **kwargs
) -> bool:
    """
    Emit a progress event to the stream.

    Args:
        step: Current step number
        total: Total number of steps
        message: Human-readable progress message
        phase: Optional phase name (e.g., "analyzing", "executing")
        **kwargs: Additional data to include

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "progress",
            "step": step,
            "total": total,
            "message": message,
            "phase": phase,
            "percentage": (step / total * 100) if total > 0 else 0,
            **kwargs
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit progress event", error=str(e))
        return False


def emit_screenshot(
    screenshot_base64: str,
    step_index: int,
    step_name: str = "",
    element_info: dict[str, Any] | None = None
) -> bool:
    """
    Emit a screenshot event to the stream.

    Args:
        screenshot_base64: Base64-encoded screenshot data
        step_index: Index of the current step
        step_name: Name or description of the step
        element_info: Optional info about the element being captured

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "screenshot",
            "screenshot": screenshot_base64,
            "step_index": step_index,
            "step_name": step_name,
            "element_info": element_info,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit screenshot event", error=str(e))
        return False


def emit_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    status: str = "calling",
    tool_output: Any | None = None,
    duration_ms: float | None = None
) -> bool:
    """
    Emit a tool call event to the stream.

    Args:
        tool_name: Name of the tool being called
        tool_input: Input parameters for the tool
        status: Status of the call ("calling", "success", "error")
        tool_output: Optional output from the tool
        duration_ms: Optional execution time in milliseconds

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        event_data = {
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "status": status,
        }

        if tool_output is not None:
            # Truncate large outputs
            output_str = str(tool_output)
            if len(output_str) > 1000:
                output_str = output_str[:1000] + "... (truncated)"
            event_data["tool_output"] = output_str

        if duration_ms is not None:
            event_data["duration_ms"] = duration_ms

        writer(event_data)
        return True
    except Exception as e:
        logger.debug("Failed to emit tool call event", error=str(e))
        return False


def emit_agent_transition(
    from_agent: str,
    to_agent: str,
    reason: str = "",
    state_summary: dict[str, Any] | None = None
) -> bool:
    """
    Emit an agent transition event to the stream.

    Args:
        from_agent: Name of the agent transitioning from
        to_agent: Name of the agent transitioning to
        reason: Reason for the transition
        state_summary: Optional summary of current state

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "agent_transition",
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "state_summary": state_summary,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit agent transition event", error=str(e))
        return False


def emit_test_started(
    test_id: str,
    test_name: str,
    test_type: str,
    test_index: int,
    total_tests: int
) -> bool:
    """
    Emit a test started event to the stream.

    Args:
        test_id: Unique identifier for the test
        test_name: Human-readable test name
        test_type: Type of test (ui, api, db)
        test_index: Index of the test in the plan
        total_tests: Total number of tests

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "test_started",
            "test_id": test_id,
            "test_name": test_name,
            "test_type": test_type,
            "test_index": test_index,
            "total_tests": total_tests,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit test started event", error=str(e))
        return False


def emit_test_completed(
    test_id: str,
    status: str,
    duration_seconds: float,
    error_message: str | None = None,
    assertions_passed: int = 0,
    assertions_failed: int = 0
) -> bool:
    """
    Emit a test completed event to the stream.

    Args:
        test_id: Unique identifier for the test
        status: Final status (passed, failed, skipped)
        duration_seconds: How long the test took
        error_message: Error message if failed
        assertions_passed: Number of passed assertions
        assertions_failed: Number of failed assertions

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "test_completed",
            "test_id": test_id,
            "status": status,
            "duration_seconds": duration_seconds,
            "error_message": error_message,
            "assertions_passed": assertions_passed,
            "assertions_failed": assertions_failed,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit test completed event", error=str(e))
        return False


def emit_healing_attempt(
    test_id: str,
    failure_type: str,
    healing_strategy: str,
    attempt_number: int = 1
) -> bool:
    """
    Emit a healing attempt event to the stream.

    Args:
        test_id: ID of the test being healed
        failure_type: Type of failure being addressed
        healing_strategy: Strategy being used to heal
        attempt_number: Which attempt this is

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "healing_attempt",
            "test_id": test_id,
            "failure_type": failure_type,
            "healing_strategy": healing_strategy,
            "attempt_number": attempt_number,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit healing attempt event", error=str(e))
        return False


def emit_healing_result(
    test_id: str,
    success: bool,
    changes_made: dict[str, Any] | None = None,
    error: str | None = None
) -> bool:
    """
    Emit a healing result event to the stream.

    Args:
        test_id: ID of the test being healed
        success: Whether healing was successful
        changes_made: Description of changes made
        error: Error message if healing failed

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "healing_result",
            "test_id": test_id,
            "success": success,
            "changes_made": changes_made,
            "error": error,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit healing result event", error=str(e))
        return False


def emit_cost_update(
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
    model: str = "unknown"
) -> bool:
    """
    Emit a cost update event to the stream.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        total_cost: Total cost so far in USD
        model: Model being used

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "cost_update",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
            "model": model,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit cost update event", error=str(e))
        return False


def emit_llm_thinking(
    thought: str,
    model: str = "unknown",
    node: str = ""
) -> bool:
    """
    Emit an LLM thinking event to show reasoning in real-time.

    Args:
        thought: The LLM's current thinking/reasoning
        model: Model being used
        node: Current graph node

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": "llm_thinking",
            "thought": thought,
            "model": model,
            "node": node,
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit LLM thinking event", error=str(e))
        return False


def emit_custom_event(event_type: str, data: dict[str, Any]) -> bool:
    """
    Emit a custom event to the stream.

    Args:
        event_type: Type of the custom event
        data: Event data

    Returns:
        True if event was emitted, False otherwise
    """
    try:
        writer = get_stream_writer()
        if writer is None:
            return False

        writer({
            "type": event_type,
            **data
        })
        return True
    except Exception as e:
        logger.debug("Failed to emit custom event", error=str(e), event_type=event_type)
        return False
