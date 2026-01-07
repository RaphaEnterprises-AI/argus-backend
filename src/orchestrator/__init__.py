"""LangGraph Orchestrator for E2E Testing.

This module provides the test orchestration layer using LangGraph.

Features:
- State machine for test workflow management
- Conditional routing between agents
- Parallel test execution (v2)
- Human-in-the-loop approval (v2)
- Streaming support (v2)
- Checkpointing and time-travel (v2)
"""

from .state import (
    TestingState,
    TestType,
    TestStatus,
    Priority,
    TestSpec,
    TestResult,
    FailureAnalysis,
    create_initial_state,
)

from .graph import (
    TestingOrchestrator,
    create_testing_graph,
)

# Enhanced orchestrator with advanced features
from .graph_v2 import (
    EnhancedTestingOrchestrator,
    create_enhanced_testing_graph,
    create_parallel_test_batches,
    aggregate_parallel_results,
    request_human_approval,
    create_quality_subgraph,
)

__all__ = [
    # State
    "TestingState",
    "TestType",
    "TestStatus",
    "Priority",
    "TestSpec",
    "TestResult",
    "FailureAnalysis",
    "create_initial_state",
    # Original Orchestrator
    "TestingOrchestrator",
    "create_testing_graph",
    # Enhanced Orchestrator (v2)
    "EnhancedTestingOrchestrator",
    "create_enhanced_testing_graph",
    "create_parallel_test_batches",
    "aggregate_parallel_results",
    "request_human_approval",
    "create_quality_subgraph",
]
