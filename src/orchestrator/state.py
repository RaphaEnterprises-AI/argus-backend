"""State definitions for the testing orchestrator."""

from typing import TypedDict, Annotated, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TestType(str, Enum):
    """Types of tests."""
    UI = "ui"
    API = "api"
    DATABASE = "db"
    INTEGRATION = "integration"


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    HEALED = "healed"


class Priority(str, Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestSpec:
    """Specification for a single test."""
    id: str
    name: str
    type: TestType
    priority: Priority
    steps: list[dict]
    assertions: list[dict]
    preconditions: list[str] = field(default_factory=list)
    cleanup: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "priority": self.priority.value,
            "steps": self.steps,
            "assertions": self.assertions,
            "preconditions": self.preconditions,
            "cleanup": self.cleanup,
            "tags": self.tags,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: str
    status: TestStatus
    duration_seconds: float
    error_message: Optional[str] = None
    screenshots: list[str] = field(default_factory=list)  # Base64 encoded
    actions_taken: list[dict] = field(default_factory=list)
    assertions_passed: int = 0
    assertions_failed: int = 0
    healing_applied: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "screenshots": self.screenshots,
            "actions_taken": self.actions_taken,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "healing_applied": self.healing_applied,
        }


@dataclass
class FailureAnalysis:
    """Analysis of a test failure."""
    test_id: str
    failure_type: Literal["selector_changed", "timing_issue", "ui_change", "real_bug", "unknown"]
    root_cause: str
    suggested_fix: Optional[dict] = None
    confidence: float = 0.0
    screenshot_at_failure: Optional[str] = None  # Base64
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
        }


class TestingState(TypedDict):
    """
    Shared state for the testing orchestrator.
    
    This state is passed between all agents in the LangGraph.
    """
    # Conversation history
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Codebase context
    codebase_path: str
    app_url: str
    codebase_summary: str
    testable_surfaces: list[dict]
    changed_files: list[str]  # For PR-triggered runs
    
    # Test planning
    test_plan: list[dict]  # List of TestSpec dicts
    test_priorities: dict[str, str]  # test_id -> priority
    
    # Execution tracking
    current_test_index: int
    current_test: Optional[dict]  # Current TestSpec dict
    
    # Results
    test_results: list[dict]  # List of TestResult dicts
    passed_count: int
    failed_count: int
    skipped_count: int
    
    # Failures for healing
    failures: list[dict]  # List of FailureAnalysis dicts
    healing_queue: list[str]  # test_ids to heal
    
    # Screenshots and evidence
    screenshots: list[str]  # Base64 encoded
    
    # Cost tracking
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    
    # Iteration control
    iteration: int
    max_iterations: int
    
    # Control flow
    next_agent: str
    should_continue: bool
    error: Optional[str]
    
    # Metadata
    run_id: str
    started_at: str
    pr_number: Optional[int]  # If triggered by PR

    # Security & compliance
    user_id: Optional[str]  # For audit trail
    session_id: Optional[str]  # For tracking across requests
    security_summary: Optional[dict]  # Files analyzed, secrets redacted, etc.


def create_initial_state(
    codebase_path: str,
    app_url: str,
    pr_number: Optional[int] = None,
    changed_files: Optional[list[str]] = None,
    user_id: Optional[str] = None,
) -> TestingState:
    """Create initial state for a test run."""
    import uuid

    session_id = str(uuid.uuid4())

    return TestingState(
        messages=[],
        codebase_path=codebase_path,
        app_url=app_url,
        codebase_summary="",
        testable_surfaces=[],
        changed_files=changed_files or [],
        test_plan=[],
        test_priorities={},
        current_test_index=0,
        current_test=None,
        test_results=[],
        passed_count=0,
        failed_count=0,
        skipped_count=0,
        failures=[],
        healing_queue=[],
        screenshots=[],
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost=0.0,
        iteration=0,
        max_iterations=100,
        next_agent="analyze_code",
        should_continue=True,
        error=None,
        run_id=session_id,
        started_at=datetime.utcnow().isoformat(),
        pr_number=pr_number,
        user_id=user_id or "anonymous",
        session_id=session_id,
        security_summary=None,
    )
