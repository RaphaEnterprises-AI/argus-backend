"""Orchestrator package for LangGraph-based test execution.

This package provides:
- TestingOrchestrator: Main orchestrator for E2E testing (backward compatible)
- EnhancedTestingOrchestrator: Advanced orchestrator with LangGraph 1.0 patterns
  - Parallel test execution via Send API (map-reduce pattern)
  - Human-in-the-loop via interrupt() function
  - Resume with human input via Command(resume=...)
  - Multi-mode streaming (values, updates, messages)
  - Time-travel debugging via checkpoint history
- ChatGraph: Chat-enabled graph for conversational testing
- SupervisorGraph: Multi-agent supervisor pattern
- PostgresSaver: Durable execution with PostgreSQL
- MemoryStore: Long-term learning with semantic search
- Human-in-the-loop approval workflows with breakpoints
"""

from .graph import (
    # Original orchestrator (backward compatible)
    TestingOrchestrator,
    create_testing_graph,
    create_testing_graph_with_interrupts,
    get_interrupt_nodes,
    # Enhanced orchestrator (LangGraph 1.0 patterns)
    EnhancedTestingOrchestrator,
    create_enhanced_testing_graph,
    create_quality_subgraph,
    # Data classes for parallel execution
    ParallelTestBatch,
    HumanApprovalRequest,
)
from .state import TestingState, create_initial_state
from .checkpointer import (
    get_checkpointer,
    setup_checkpointer,
    reset_checkpointer,
    list_pending_threads,
    get_thread_state,
    CheckpointManager,
)

# Multi-agent Supervisor pattern
from .supervisor import (
    SupervisorState,
    SupervisorOrchestrator,
    create_supervisor_graph,
    create_initial_supervisor_state,
    AGENTS as SUPERVISOR_AGENTS,
    AGENT_DESCRIPTIONS as SUPERVISOR_AGENT_DESCRIPTIONS,
)


# Lazy imports for optional components
def get_supervisor_graph_factory():
    """Get the supervisor graph factory function."""
    from .supervisor import create_supervisor_graph
    return create_supervisor_graph


def get_chat_graph():
    from .chat_graph import create_chat_graph
    return create_chat_graph


def get_memory_store():
    from .memory_store import get_memory_store as _get_memory_store
    return _get_memory_store()


__all__ = [
    # Core orchestrator (backward compatible)
    "TestingOrchestrator",
    "create_testing_graph",
    "create_testing_graph_with_interrupts",
    "get_interrupt_nodes",
    # Enhanced orchestrator (LangGraph 1.0 patterns)
    "EnhancedTestingOrchestrator",
    "create_enhanced_testing_graph",
    "create_quality_subgraph",
    "ParallelTestBatch",
    "HumanApprovalRequest",
    # State
    "TestingState",
    "create_initial_state",
    # Checkpointing and approvals
    "get_checkpointer",
    "setup_checkpointer",
    "reset_checkpointer",
    "list_pending_threads",
    "get_thread_state",
    "CheckpointManager",
    # Supervisor Pattern
    "SupervisorState",
    "SupervisorOrchestrator",
    "create_supervisor_graph",
    "create_initial_supervisor_state",
    "SUPERVISOR_AGENTS",
    "SUPERVISOR_AGENT_DESCRIPTIONS",
    # Lazy-loaded components
    "get_supervisor_graph_factory",
    "get_chat_graph",
    "get_memory_store",
]
