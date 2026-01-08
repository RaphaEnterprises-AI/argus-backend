"""Data models for the Hybrid Execution Engine.

This module defines the core data structures used by the HybridExecutor:
- ExecutionMode: DOM-only, Vision-only, or Hybrid
- FallbackLevel: Escalation levels for retry logic
- ExecutionStrategy: Configuration for hybrid execution
- FallbackEvent: Records when Vision fallback was triggered
- HybridStepResult: Result from hybrid step execution
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional


class ExecutionMode(str, Enum):
    """Execution mode for test steps."""

    DOM_ONLY = "dom_only"  # Never use vision - fastest but may fail
    VISION_ONLY = "vision_only"  # Always use vision - slowest but most reliable
    HYBRID = "hybrid"  # DOM first, vision fallback - best balance
    HYBRID_VERIFY = "hybrid_verify"  # DOM execute, vision verify - highest confidence


class FallbackLevel(str, Enum):
    """Escalation levels for retry logic.

    The HybridExecutor escalates through these levels:
    1. NONE - First attempt with DOM
    2. RETRY - Simple retry (maybe transient failure)
    3. WAIT_AND_RETRY - Wait for element to stabilize
    4. REFRESH_AND_RETRY - Refresh page and retry (element might be stale)
    5. VISION_FALLBACK - Use Vision AI to locate and interact
    """

    NONE = "none"  # First attempt, no fallback yet
    RETRY = "retry"  # Simple retry
    WAIT_AND_RETRY = "wait_and_retry"  # Wait 1s then retry
    REFRESH_AND_RETRY = "refresh_and_retry"  # Refresh page then retry
    VISION_FALLBACK = "vision_fallback"  # Use Vision AI


@dataclass
class ExecutionStrategy:
    """Configuration for hybrid execution.

    This controls how the HybridExecutor behaves:
    - How many DOM retries before Vision fallback
    - Timeout and delay settings
    - Which steps should always use Vision
    - Cost tracking settings

    Example:
        strategy = ExecutionStrategy(
            mode=ExecutionMode.HYBRID,
            dom_retries=3,
            vision_fallback_enabled=True,
            always_use_vision_for=["captcha", "canvas"],
        )
    """

    # Execution mode
    mode: ExecutionMode = ExecutionMode.HYBRID

    # DOM execution settings
    dom_timeout_ms: int = 5000  # Timeout for each DOM attempt
    dom_retries: int = 2  # Number of DOM retries before escalation
    retry_delay_ms: int = 500  # Delay between retries
    wait_before_retry_ms: int = 1000  # Wait time for WAIT_AND_RETRY level
    refresh_on_failure: bool = False  # Enable REFRESH_AND_RETRY level

    # Vision fallback settings
    vision_fallback_enabled: bool = True
    vision_timeout_ms: int = 30000  # Timeout for Vision execution
    vision_model: str = "claude-sonnet-4-5"  # Model for Computer Use

    # Cost tracking
    vision_cost_per_call: float = 0.003  # Estimated cost per Vision API call
    track_costs: bool = True  # Enable cost tracking

    # Screenshots
    auto_screenshot_on_failure: bool = True  # Capture screenshot on failure
    auto_screenshot_on_success: bool = False  # Capture screenshot on success

    # Logging
    log_fallback_events: bool = True  # Log when fallback is triggered
    log_all_attempts: bool = False  # Log every attempt (verbose)

    # Per-action overrides - these actions ALWAYS use Vision
    always_use_vision_for: list[str] = field(
        default_factory=lambda: [
            "visual_assertion",  # Always verify visually
            "captcha",  # Captchas need vision
            "canvas",  # Canvas elements
            "drag_and_drop",  # Complex interactions
            "slider",  # Slider interactions
        ]
    )

    # Per-action overrides - these actions NEVER use Vision (even on failure)
    never_use_vision_for: list[str] = field(
        default_factory=lambda: [
            "goto",  # Navigation doesn't benefit from Vision
            "evaluate",  # JavaScript execution
        ]
    )


@dataclass
class StepExecutionConfig:
    """Per-step execution configuration.

    This can override the global ExecutionStrategy for specific steps.
    Useful for steps that need special handling.

    Example:
        step = TestStep(
            action="click",
            target="#submit",
            execution=StepExecutionConfig(use_vision=True)
        )
    """

    use_vision: bool = False  # Force Vision for this step
    always_vision: bool = False  # ONLY use Vision (skip DOM entirely)
    skip_vision: bool = False  # Never use Vision (even on failure)
    timeout_ms: Optional[int] = None  # Override timeout
    retries: Optional[int] = None  # Override retry count
    description: Optional[str] = None  # Human-readable description for Vision


@dataclass
class FallbackEvent:
    """Records when Vision fallback was triggered.

    This is used for analytics and debugging:
    - Which selectors are causing fallbacks
    - Success rate of Vision vs DOM
    - Cost tracking

    Example:
        event = FallbackEvent(
            step_index=3,
            original_selector="#dynamic-button",
            fallback_level=FallbackLevel.VISION_FALLBACK,
            dom_error="Element not found",
            success=True,
        )
    """

    # Step information
    step_index: int
    step_action: str = ""
    original_selector: str = ""

    # Fallback details
    fallback_level: FallbackLevel = FallbackLevel.NONE
    dom_attempts: int = 0
    dom_error: Optional[str] = None

    # Vision details
    vision_description: Optional[str] = None
    vision_coordinates: Optional[tuple[int, int]] = None

    # Result
    success: bool = False
    final_error: Optional[str] = None

    # Timing
    dom_duration_ms: int = 0
    vision_duration_ms: int = 0
    total_duration_ms: int = 0

    # Cost
    estimated_cost: float = 0.0

    # Evidence
    screenshot: Optional[bytes] = None
    screenshot_base64: Optional[str] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "step_index": self.step_index,
            "step_action": self.step_action,
            "original_selector": self.original_selector,
            "fallback_level": self.fallback_level.value,
            "dom_attempts": self.dom_attempts,
            "dom_error": self.dom_error,
            "vision_description": self.vision_description,
            "success": self.success,
            "final_error": self.final_error,
            "dom_duration_ms": self.dom_duration_ms,
            "vision_duration_ms": self.vision_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "estimated_cost": self.estimated_cost,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HybridStepResult:
    """Result from hybrid step execution.

    This provides detailed information about how a step was executed:
    - Which mode was used (DOM or Vision)
    - Whether fallback was triggered
    - Timing and cost information

    Example:
        result = await executor.execute_step(step, 0)
        if result.fallback_triggered:
            print(f"Vision fallback was needed: {result.fallback_event}")
        print(f"Total time: {result.total_duration_ms}ms")
    """

    # Success/failure
    success: bool
    error: Optional[str] = None

    # Execution mode
    mode_used: Literal["dom", "vision", "hybrid"] = "dom"
    fallback_triggered: bool = False

    # Attempts
    dom_attempts: int = 0
    vision_attempts: int = 0

    # Timing
    total_duration_ms: int = 0
    dom_duration_ms: int = 0
    vision_duration_ms: int = 0

    # Cost
    estimated_cost: float = 0.0

    # Fallback details
    fallback_event: Optional[FallbackEvent] = None
    fallback_level: FallbackLevel = FallbackLevel.NONE

    # Evidence
    screenshot: Optional[bytes] = None
    action_data: Any = None  # Action-specific return data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "success": self.success,
            "error": self.error,
            "mode_used": self.mode_used,
            "fallback_triggered": self.fallback_triggered,
            "dom_attempts": self.dom_attempts,
            "vision_attempts": self.vision_attempts,
            "total_duration_ms": self.total_duration_ms,
            "dom_duration_ms": self.dom_duration_ms,
            "vision_duration_ms": self.vision_duration_ms,
            "estimated_cost": self.estimated_cost,
            "fallback_level": self.fallback_level.value,
            "fallback_event": self.fallback_event.to_dict() if self.fallback_event else None,
        }


@dataclass
class ExecutionStats:
    """Aggregate statistics for execution.

    This tracks overall execution performance:
    - Total steps executed
    - Fallback rate
    - Total cost

    Example:
        stats = executor.get_stats()
        print(f"Fallback rate: {stats.fallback_rate:.1%}")
        print(f"Total cost: ${stats.total_cost:.3f}")
    """

    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0

    dom_only_steps: int = 0
    vision_fallback_steps: int = 0
    vision_only_steps: int = 0

    total_dom_attempts: int = 0
    total_vision_attempts: int = 0

    total_dom_duration_ms: int = 0
    total_vision_duration_ms: int = 0
    total_duration_ms: int = 0

    total_cost: float = 0.0

    fallback_events: list[FallbackEvent] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate (how often Vision was needed)."""
        if self.total_steps == 0:
            return 0.0
        return self.vision_fallback_steps / self.total_steps

    @property
    def average_dom_duration_ms(self) -> float:
        """Calculate average DOM execution time."""
        if self.dom_only_steps == 0:
            return 0.0
        return self.total_dom_duration_ms / self.dom_only_steps

    @property
    def average_vision_duration_ms(self) -> float:
        """Calculate average Vision execution time."""
        if self.vision_fallback_steps + self.vision_only_steps == 0:
            return 0.0
        return self.total_vision_duration_ms / (self.vision_fallback_steps + self.vision_only_steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "dom_only_steps": self.dom_only_steps,
            "vision_fallback_steps": self.vision_fallback_steps,
            "vision_only_steps": self.vision_only_steps,
            "success_rate": self.success_rate,
            "fallback_rate": self.fallback_rate,
            "total_dom_attempts": self.total_dom_attempts,
            "total_vision_attempts": self.total_vision_attempts,
            "total_dom_duration_ms": self.total_dom_duration_ms,
            "total_vision_duration_ms": self.total_vision_duration_ms,
            "total_duration_ms": self.total_duration_ms,
            "average_dom_duration_ms": self.average_dom_duration_ms,
            "average_vision_duration_ms": self.average_vision_duration_ms,
            "total_cost": self.total_cost,
            "fallback_events_count": len(self.fallback_events),
        }
