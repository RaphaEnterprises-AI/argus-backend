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
- Human-in-the-loop approval workflows with breakpoints
- AgentRegistry: Agent discovery and health monitoring for A2A communication (RAP-228)
- A2AProtocol: Agent-to-agent communication protocol over Kafka (RAP-229)
  - Request/response messaging with correlation and timeout
  - Broadcast messaging to all subscribers
  - Circuit breaker pattern for fault tolerance
  - Heartbeat monitoring for agent health
- MARP: Multi-Agent Reasoning Protocol for consensus decisions (RAP-235)
  - Proposal submission and voting mechanisms
  - Multiple consensus strategies (majority, confidence-weighted, expertise-weighted)
  - Tie-breaking and human escalation
  - Full audit trail for compliance and debugging
- WorkflowComposer: Dynamic workflow composition at runtime (RAP-232)
  - Task decomposition using LLM analysis
  - Capability-based agent discovery and assignment
  - Dependency graph building with parallel execution
  - LangGraph state machine compilation
- TaskDecomposer: Break complex tasks into executable subtasks
  - LLM-powered task analysis
  - Capability identification
  - Dependency inference
- ParallelExecutor: Execute workflow tasks concurrently
  - Dependency-aware scheduling
  - Configurable parallelism limits
  - Streaming execution updates
  - Resource management and rate limiting

.. note::
    For knowledge/memory operations, use `src.knowledge.CogneeKnowledgeClient`
    instead of the deprecated `get_memory_store()`. See RAP-132.
"""

import warnings

from .checkpointer import (
    CheckpointManager,
    get_checkpointer,
    get_thread_state,
    list_pending_threads,
    reset_checkpointer,
    setup_checkpointer,
)
from .graph import (
    # Enhanced orchestrator (LangGraph 1.0 patterns)
    EnhancedTestingOrchestrator,
    HumanApprovalRequest,
    # Data classes for parallel execution
    ParallelTestBatch,
    # Original orchestrator (backward compatible)
    TestingOrchestrator,
    create_enhanced_testing_graph,
    create_quality_subgraph,
    create_testing_graph,
    create_testing_graph_with_interrupts,
    get_interrupt_nodes,
)
from .state import TestingState, create_initial_state
from .supervisor import (
    AGENT_DESCRIPTIONS as SUPERVISOR_AGENT_DESCRIPTIONS,
)
from .supervisor import (
    AGENTS as SUPERVISOR_AGENTS,
)

# Multi-agent Supervisor pattern
from .supervisor import (
    SupervisorOrchestrator,
    SupervisorState,
    create_initial_supervisor_state,
    create_supervisor_graph,
)

# Agent Registry for A2A communication (RAP-228)
from .agent_registry import (
    AgentInfo,
    AgentRegistry,
    Capability,
    DEFAULT_AGENT_CAPABILITIES,
    discover_agents,
    get_agent_registry,
    get_default_capabilities,
    heartbeat,
    init_agent_registry,
    register_agent,
    reset_agent_registry,
    shutdown_agent_registry,
    unregister_agent,
)

# A2A Protocol for agent-to-agent communication (RAP-229)
from .a2a_protocol import (
    A2AProtocol,
    AgentBroadcast,
    AgentHeartbeat,
    AgentRequestEvent,
    AgentResponse,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    MessageType,
    create_a2a_protocol_from_settings,
)

# MARP - Multi-Agent Reasoning Protocol for consensus (RAP-235)
from .marp import (
    MARP,
    Proposal,
    Resolution,
    ResolutionStatus,
    TieError,
    Vote,
    VoteType,
)
from .consensus import (
    BordaCount,
    ConfidenceWeighted,
    ExpertiseWeighted,
    MajorityVoting,
    QuadraticVoting,
    SuperMajority,
    create_strategy,
)
from .resolver import (
    AuditEntry,
    ConflictResolver,
    DashboardEscalationHandler,
    EscalationLevel,
    EscalationRequest,
    SlackEscalationHandler,
    TieBreakStrategy,
)

# Dynamic Workflow Composer (RAP-232)
from .workflow_composer import (
    CompiledWorkflow,
    ExecutionMode,
    TaskDefinition,
    TaskPriority,
    WorkflowComposer,
    WorkflowConstraints,
    WorkflowPlan,
    WorkflowResult,
    WorkflowStage,
    WorkflowState,
)
from .task_decomposer import (
    DecompositionConfig,
    SmartDecomposer,
    TaskDecomposer,
)
from .parallel_executor import (
    ExecutionProgress,
    ExecutionStrategy,
    ParallelExecutor,
    ParallelStrategy,
    PipelineStrategy,
    ResourceManager,
    SequentialStrategy,
    StageExecutionUpdate,
    StageStatus,
    WorkflowResult as ParallelWorkflowResult,
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
    """Get memory store instance.

    .. deprecated:: 2026.01
        Use `src.knowledge.get_cognee_client()` instead.
    """
    warnings.warn(
        "get_memory_store is deprecated. Use src.knowledge.get_cognee_client() instead. "
        "See RAP-132 for migration details.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    # Agent Registry (RAP-228)
    "AgentRegistry",
    "AgentInfo",
    "Capability",
    "get_agent_registry",
    "reset_agent_registry",
    "init_agent_registry",
    "shutdown_agent_registry",
    "register_agent",
    "unregister_agent",
    "discover_agents",
    "heartbeat",
    "get_default_capabilities",
    "DEFAULT_AGENT_CAPABILITIES",
    # A2A Protocol (RAP-229)
    "A2AProtocol",
    "AgentRequestEvent",
    "AgentResponse",
    "AgentBroadcast",
    "AgentHeartbeat",
    "MessageType",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "create_a2a_protocol_from_settings",
    # MARP - Multi-Agent Reasoning Protocol (RAP-235)
    "MARP",
    "Proposal",
    "Vote",
    "VoteType",
    "Resolution",
    "ResolutionStatus",
    "TieError",
    # Consensus Strategies
    "MajorityVoting",
    "ConfidenceWeighted",
    "ExpertiseWeighted",
    "SuperMajority",
    "BordaCount",
    "QuadraticVoting",
    "create_strategy",
    # Conflict Resolution
    "ConflictResolver",
    "TieBreakStrategy",
    "EscalationLevel",
    "EscalationRequest",
    "AuditEntry",
    "SlackEscalationHandler",
    "DashboardEscalationHandler",
    # Lazy-loaded components
    "get_supervisor_graph_factory",
    "get_chat_graph",
    "get_memory_store",
    # Dynamic Workflow Composer (RAP-232)
    "WorkflowComposer",
    "TaskDefinition",
    "TaskPriority",
    "WorkflowConstraints",
    "WorkflowStage",
    "WorkflowPlan",
    "WorkflowState",
    "CompiledWorkflow",
    "WorkflowResult",
    "ExecutionMode",
    # Task Decomposer
    "TaskDecomposer",
    "SmartDecomposer",
    "DecompositionConfig",
    # Parallel Executor
    "ParallelExecutor",
    "StageStatus",
    "StageExecutionUpdate",
    "ExecutionProgress",
    "ExecutionStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "PipelineStrategy",
    "ResourceManager",
    "ParallelWorkflowResult",
]
