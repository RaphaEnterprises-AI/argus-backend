"""
Langfuse integration for LangGraph orchestration.

This module provides the CallbackHandler setup for tracing LangGraph
executions with Langfuse. It automatically captures:
- All graph node executions
- LLM calls with token usage and costs
- Tool invocations
- Span timings and metadata

Usage:
    from src.orchestrator.langfuse_integration import get_langfuse_handler, get_langfuse_config

    # Get the callback handler
    langfuse_handler = get_langfuse_handler(
        user_id="user-123",
        session_id="session-456",
        tags=["test-run", "pr-123"],
    )

    # Add to graph config
    config = get_langfuse_config(
        thread_id="thread-789",
        langfuse_handler=langfuse_handler,
    )

    # Invoke graph with tracing
    result = await graph.ainvoke(initial_state, config)

Environment variables:
    LANGFUSE_PUBLIC_KEY: Public key from Langfuse project settings
    LANGFUSE_SECRET_KEY: Secret key from Langfuse project settings
    LANGFUSE_HOST: Langfuse server URL (default: https://cloud.langfuse.com)
    LANGFUSE_ENABLED: Set to "false" to disable tracing (default: true)

References:
    - https://langfuse.com/guides/cookbook/integration_langgraph
    - https://langfuse.com/docs/integrations/langchain
"""

import os
import structlog
from typing import Any

logger = structlog.get_logger(__name__)

# Lazy import to avoid import errors if langfuse is not installed
_langfuse_available = None
_CallbackHandler = None


def _check_langfuse_available() -> bool:
    """Check if Langfuse is available and properly configured."""
    global _langfuse_available, _CallbackHandler

    if _langfuse_available is not None:
        return _langfuse_available

    # Check if explicitly disabled
    if os.environ.get("LANGFUSE_ENABLED", "true").lower() == "false":
        logger.info("Langfuse tracing disabled via LANGFUSE_ENABLED=false")
        _langfuse_available = False
        return False

    # Check for required environment variables
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse tracing disabled - missing credentials. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
        )
        _langfuse_available = False
        return False

    # Try to import langfuse (v2.x uses langfuse.callback)
    try:
        from langfuse.callback import CallbackHandler
        _CallbackHandler = CallbackHandler
        _langfuse_available = True

        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        logger.info(
            "Langfuse tracing enabled",
            host=host,
            public_key=f"{public_key[:15]}...",
        )
        return True

    except ImportError as e:
        logger.warning(f"Langfuse package not installed: {e}")
        _langfuse_available = False
        return False


def get_langfuse_handler(
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    release: str | None = None,
):
    """
    Get a Langfuse CallbackHandler for LangGraph tracing.

    This handler should be passed to graph.ainvoke() via the config parameter
    to enable automatic tracing of all LLM calls, tool invocations, and node
    executions.

    Args:
        user_id: User identifier for filtering traces in Langfuse UI
        session_id: Session identifier for grouping related traces
        trace_name: Custom name for the trace (defaults to graph name)
        tags: List of tags for filtering and organizing traces
        metadata: Additional metadata to attach to the trace
        release: Release/version identifier

    Returns:
        CallbackHandler instance if Langfuse is available, None otherwise

    Example:
        handler = get_langfuse_handler(
            user_id="org-123",
            session_id="run-456",
            tags=["e2e-test", "pr-789"],
            metadata={"pr_number": 789, "branch": "feature/x"},
        )

        config = {"callbacks": [handler]} if handler else {}
        result = await graph.ainvoke(state, config)
    """
    if not _check_langfuse_available():
        return None

    try:
        handler = _CallbackHandler(
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
            tags=tags or [],
            metadata=metadata or {},
            release=release,
        )

        logger.debug(
            "Created Langfuse callback handler",
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
        )

        return handler

    except Exception as e:
        logger.warning(f"Failed to create Langfuse handler: {e}")
        return None


def get_langfuse_config(
    thread_id: str,
    langfuse_handler=None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """
    Get a complete config dict for LangGraph invocation with Langfuse tracing.

    This is a convenience function that combines thread configuration with
    Langfuse callback handling.

    Args:
        thread_id: LangGraph thread ID for checkpointing
        langfuse_handler: Pre-created CallbackHandler (if None, creates one)
        user_id: User ID for trace (used if creating new handler)
        session_id: Session ID for trace (defaults to thread_id)
        tags: Tags for the trace
        metadata: Additional metadata

    Returns:
        Config dict ready for graph.ainvoke()

    Example:
        config = get_langfuse_config(
            thread_id="run-123",
            user_id="org-456",
            tags=["test-suite"],
        )
        result = await graph.ainvoke(state, config)
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # Get or create handler
    handler = langfuse_handler
    if handler is None and _check_langfuse_available():
        handler = get_langfuse_handler(
            user_id=user_id,
            session_id=session_id or thread_id,
            tags=tags,
            metadata=metadata,
        )

    # Add callbacks if handler available
    if handler is not None:
        config["callbacks"] = [handler]

    return config


def flush_langfuse():
    """
    Flush any pending Langfuse events.

    Should be called before application shutdown to ensure all traces
    are sent to the Langfuse server.
    """
    if not _check_langfuse_available():
        return

    try:
        from langfuse import get_client
        client = get_client()
        client.flush()
        logger.debug("Langfuse events flushed")
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")


def score_trace(
    trace_id: str,
    name: str,
    value: float | str,
    comment: str | None = None,
    data_type: str = "NUMERIC",
):
    """
    Add a score to an existing trace.

    Scores are used for evaluation and quality tracking in Langfuse.
    Common use cases:
    - Test pass/fail rates
    - Healing success rates
    - User satisfaction ratings

    Args:
        trace_id: The trace ID to score
        name: Score name (e.g., "accuracy", "relevance", "success")
        value: Score value (float for NUMERIC, string for CATEGORICAL)
        comment: Optional explanation for the score
        data_type: "NUMERIC" or "CATEGORICAL"

    Example:
        # Score a test run
        score_trace(
            trace_id=run_id,
            name="test_pass_rate",
            value=0.95,
            comment="95% of tests passed",
        )
    """
    if not _check_langfuse_available():
        return

    try:
        from langfuse import get_client
        client = get_client()

        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
        )

        logger.debug(
            "Added Langfuse score",
            trace_id=trace_id,
            name=name,
            value=value,
        )
    except Exception as e:
        logger.warning(f"Failed to add Langfuse score: {e}")


def get_trace_url(trace_id: str) -> str | None:
    """
    Get the Langfuse UI URL for a trace.

    Args:
        trace_id: The trace ID

    Returns:
        URL to view the trace in Langfuse UI, or None if not available
    """
    if not _check_langfuse_available():
        return None

    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return f"{host}/trace/{trace_id}"
