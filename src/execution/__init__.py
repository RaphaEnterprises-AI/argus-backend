"""Hybrid Execution Engine for E2E Testing.

This module provides a sophisticated execution engine that combines
DOM-based (fast) and Vision-based (reliable) test execution strategies.

Key Features:
- DOM-first execution for speed (50-200ms per action)
- Automatic escalation to Vision on failures
- Multiple retry levels before Vision fallback
- Per-step execution configuration
- Fallback event recording for analytics
- Cost tracking for Vision API calls

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     HybridExecutor                          │
    ├─────────────────────────────────────────────────────────────┤
    │  Step: Click "#login-button"                                │
    │                                                             │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
    │  │    DOM      │──►│   Retry     │──►│     Vision      │   │
    │  │  (100ms)    │   │  (500ms)    │   │    (2-5s)       │   │
    │  └─────────────┘   └─────────────┘   └─────────────────┘   │
    │                                                             │
    │  Escalation: DOM → DOM+wait → DOM+refresh → Vision          │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from src.execution import HybridExecutor, ExecutionStrategy

    # Create executor with custom strategy
    executor = HybridExecutor(
        dom_executor=playwright_automation,
        vision_executor=computer_use_automation,
        strategy=ExecutionStrategy(
            dom_retries=3,
            vision_fallback_enabled=True,
        )
    )

    # Execute a test step
    result = await executor.execute_step(step, step_index=0)

    # Check if fallback was used
    if result.fallback_triggered:
        print(f"Vision fallback used: {result.fallback_event}")
"""

from .fallback_manager import FallbackManager
from .hybrid_executor import HybridExecutor
from .models import (
    ExecutionMode,
    ExecutionStrategy,
    FallbackEvent,
    FallbackLevel,
    HybridStepResult,
    StepExecutionConfig,
)

__all__ = [
    # Models
    "ExecutionMode",
    "FallbackLevel",
    "ExecutionStrategy",
    "FallbackEvent",
    "HybridStepResult",
    "StepExecutionConfig",
    # Executor
    "HybridExecutor",
    "FallbackManager",
]
