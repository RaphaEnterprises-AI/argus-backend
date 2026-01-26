"""
Multi-tenant Kafka Event Schemas

All events include tenant context (org_id, project_id) for data isolation.
Event versioning follows semantic versioning (major.minor).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types for routing and processing."""

    CODEBASE_INGESTED = "codebase.ingested"
    CODEBASE_ANALYZED = "codebase.analyzed"
    TEST_CREATED = "test.created"
    TEST_EXECUTED = "test.executed"
    TEST_FAILED = "test.failed"
    HEALING_REQUESTED = "healing.requested"
    HEALING_COMPLETED = "healing.completed"
    DLQ = "dlq"


class TenantInfo(BaseModel):
    """Tenant context for multi-tenant isolation.

    Every event MUST include org_id. project_id and user_id are optional
    depending on the event scope.
    """

    org_id: str = Field(..., description="Organization ID (required for all events)")
    project_id: Optional[str] = Field(None, description="Project ID if event is project-scoped")
    user_id: Optional[str] = Field(None, description="User ID who triggered the event")

    def to_cognee_dataset_prefix(self) -> str:
        """Generate Cognee dataset prefix for this tenant.

        Returns:
            Dataset prefix like 'org_abc123_project_xyz789'
        """
        if self.project_id:
            return f"org_{self.org_id}_project_{self.project_id}"
        return f"org_{self.org_id}"


class EventMetadata(BaseModel):
    """Standard metadata for all events."""

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Correlation ID for request tracing"
    )
    source: str = Field(..., description="Service that generated the event")
    triggered_by: Optional[str] = Field(
        None,
        description="What triggered this event (e.g., 'api', 'scheduler', 'webhook')"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the event was created"
    )
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    span_id: Optional[str] = Field(None, description="Distributed tracing span")


class BaseEvent(BaseModel):
    """Base class for all Argus events.

    All events inherit from this and add their specific payload fields.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier"
    )
    event_type: EventType = Field(..., description="Type of event")
    event_version: str = Field("1.0", description="Schema version (semver)")
    tenant: TenantInfo = Field(..., description="Tenant context")
    metadata: EventMetadata = Field(..., description="Event metadata")

    def to_kafka_key(self) -> str:
        """Generate Kafka message key for partitioning.

        Uses org_id + project_id to ensure all events for a project
        go to the same partition (ordering guarantee).
        """
        if self.tenant.project_id:
            return f"{self.tenant.org_id}:{self.tenant.project_id}"
        return self.tenant.org_id

    def to_dict(self) -> dict:
        """Convert to dict for Kafka serialization."""
        return self.model_dump(mode="json")


# =============================================================================
# Codebase Events
# =============================================================================

class CodebaseIngestedEvent(BaseEvent):
    """Emitted when a codebase is uploaded/synced for analysis."""

    event_type: EventType = Field(default=EventType.CODEBASE_INGESTED, frozen=True)

    # Payload
    repository_id: str = Field(..., description="Repository ID")
    repository_url: str = Field(..., description="Git repository URL")
    branch: str = Field(..., description="Branch being analyzed")
    commit_sha: str = Field(..., description="Commit SHA")
    file_count: int = Field(..., description="Number of files in codebase")
    total_size_bytes: int = Field(..., description="Total size in bytes")
    languages: list[str] = Field(default_factory=list, description="Detected languages")
    ingestion_method: str = Field(
        default="github_webhook",
        description="How codebase was ingested (github_webhook, upload, sync)"
    )


class CodebaseAnalyzedEvent(BaseEvent):
    """Emitted when codebase analysis is complete."""

    event_type: EventType = Field(default=EventType.CODEBASE_ANALYZED, frozen=True)

    # Payload
    repository_id: str = Field(..., description="Repository ID")
    commit_sha: str = Field(..., description="Analyzed commit SHA")
    analysis_duration_ms: int = Field(..., description="Analysis duration in milliseconds")

    # Analysis results summary
    functions_found: int = Field(default=0, description="Number of functions discovered")
    classes_found: int = Field(default=0, description="Number of classes discovered")
    test_files_found: int = Field(default=0, description="Number of test files")
    endpoints_found: int = Field(default=0, description="Number of API endpoints")

    # Cognee processing
    cognee_dataset_name: str = Field(..., description="Cognee dataset where results stored")
    cognee_documents_created: int = Field(default=0, description="Documents added to Cognee")

    # Graph stats
    nodes_created: int = Field(default=0, description="Neo4j nodes created")
    relationships_created: int = Field(default=0, description="Neo4j relationships created")


# =============================================================================
# Test Events
# =============================================================================

class TestCreatedEvent(BaseEvent):
    """Emitted when a new test case is created."""

    event_type: EventType = Field(default=EventType.TEST_CREATED, frozen=True)

    # Payload
    test_id: str = Field(..., description="Test case ID")
    test_name: str = Field(..., description="Test case name")
    test_type: str = Field(..., description="Type: ui, api, unit, integration")
    framework: str = Field(..., description="Test framework: playwright, pytest, jest")
    priority: str = Field(default="medium", description="Priority: critical, high, medium, low")

    # Optional context
    suite_id: Optional[str] = Field(None, description="Parent test suite ID")
    repository_id: Optional[str] = Field(None, description="Source repository")
    target_url: Optional[str] = Field(None, description="Target URL for UI tests")
    target_endpoint: Optional[str] = Field(None, description="Target endpoint for API tests")

    # Test definition
    selectors: list[str] = Field(default_factory=list, description="UI selectors used")
    steps_count: int = Field(default=0, description="Number of test steps")


class TestExecutedEvent(BaseEvent):
    """Emitted when a test execution completes (pass or fail)."""

    event_type: EventType = Field(default=EventType.TEST_EXECUTED, frozen=True)

    # Payload
    test_id: str = Field(..., description="Test case ID")
    run_id: str = Field(..., description="Test run ID")
    status: str = Field(..., description="Result: passed, failed, skipped, error")
    duration_ms: int = Field(..., description="Execution duration in milliseconds")

    # Execution context
    trigger: str = Field(default="manual", description="Trigger: manual, ci, scheduled")
    commit_sha: Optional[str] = Field(None, description="Commit being tested")
    branch: Optional[str] = Field(None, description="Branch being tested")
    environment: str = Field(default="staging", description="Environment: staging, production")

    # Results
    assertions_passed: int = Field(default=0, description="Number of assertions passed")
    assertions_failed: int = Field(default=0, description="Number of assertions failed")
    screenshots_captured: int = Field(default=0, description="Number of screenshots taken")

    # Artifacts
    screenshot_urls: list[str] = Field(default_factory=list, description="Screenshot URLs")
    video_url: Optional[str] = Field(None, description="Video recording URL")
    trace_url: Optional[str] = Field(None, description="Playwright trace URL")


class TestFailedEvent(BaseEvent):
    """Emitted when a test fails - triggers self-healing pipeline."""

    event_type: EventType = Field(default=EventType.TEST_FAILED, frozen=True)

    # Payload
    test_id: str = Field(..., description="Test case ID")
    run_id: str = Field(..., description="Test run ID")
    failure_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique failure ID for tracking"
    )

    # Failure details
    error_type: str = Field(
        ...,
        description="Error category: assertion, timeout, selector, network, script"
    )
    error_message: str = Field(..., description="Full error message")
    stack_trace: Optional[str] = Field(None, description="Full stack trace")
    failed_step: Optional[int] = Field(None, description="Step number that failed")
    failed_selector: Optional[str] = Field(None, description="Selector that failed (UI tests)")
    failed_assertion: Optional[str] = Field(None, description="Assertion that failed")

    # Context for healing
    page_url: Optional[str] = Field(None, description="Page URL at time of failure")
    page_html_snapshot: Optional[str] = Field(None, description="DOM snapshot (truncated)")
    screenshot_url: Optional[str] = Field(None, description="Failure screenshot URL")

    # Healing hints
    is_flaky: bool = Field(default=False, description="Whether this test is known to be flaky")
    previous_failure_count: int = Field(default=0, description="Times this test has failed recently")
    similar_failure_ids: list[str] = Field(
        default_factory=list,
        description="IDs of similar past failures"
    )


# =============================================================================
# Self-Healing Events
# =============================================================================

class HealingRequestedEvent(BaseEvent):
    """Emitted when self-healing is triggered for a failed test."""

    event_type: EventType = Field(default=EventType.HEALING_REQUESTED, frozen=True)

    # Payload
    failure_id: str = Field(..., description="ID of the failure being healed")
    test_id: str = Field(..., description="Test case ID")

    # Healing strategy
    strategy: str = Field(
        default="auto",
        description="Strategy: auto, selector_fallback, semantic_match, llm_fix"
    )
    priority: str = Field(default="normal", description="Priority: urgent, high, normal, low")

    # Context
    error_type: str = Field(..., description="Type of error being healed")
    failed_selector: Optional[str] = Field(None, description="Original failed selector")
    page_context: Optional[dict[str, Any]] = Field(
        None,
        description="Page context for healing (DOM, screenshot, etc.)"
    )


class HealingCompletedEvent(BaseEvent):
    """Emitted when self-healing completes (success or failure)."""

    event_type: EventType = Field(default=EventType.HEALING_COMPLETED, frozen=True)

    # Payload
    failure_id: str = Field(..., description="ID of the failure that was healed")
    test_id: str = Field(..., description="Test case ID")
    healing_request_id: str = Field(..., description="ID of the healing request event")

    # Result
    success: bool = Field(..., description="Whether healing was successful")
    strategy_used: str = Field(..., description="Strategy that was applied")
    duration_ms: int = Field(..., description="Healing duration in milliseconds")

    # Fix details (if successful)
    original_selector: Optional[str] = Field(None, description="Original failed selector")
    healed_selector: Optional[str] = Field(None, description="New working selector")
    healing_pattern_id: Optional[str] = Field(
        None,
        description="ID of the healing pattern used/created"
    )

    # Verification
    verification_passed: bool = Field(
        default=False,
        description="Whether healed test passed verification"
    )
    verification_run_id: Optional[str] = Field(
        None,
        description="Run ID of verification test"
    )

    # Failure details (if unsuccessful)
    failure_reason: Optional[str] = Field(
        None,
        description="Why healing failed (if success=false)"
    )


# =============================================================================
# Dead Letter Queue Event
# =============================================================================

class DLQEvent(BaseEvent):
    """Event sent to dead letter queue when processing fails."""

    event_type: EventType = Field(default=EventType.DLQ, frozen=True)

    # Original event info
    original_event_id: str = Field(..., description="ID of the failed event")
    original_event_type: str = Field(..., description="Type of the failed event")
    original_topic: str = Field(..., description="Topic the event came from")
    original_payload: dict[str, Any] = Field(..., description="Original event payload")

    # Failure info
    error_message: str = Field(..., description="Processing error message")
    error_type: str = Field(..., description="Error type/class")
    stack_trace: Optional[str] = Field(None, description="Full stack trace")

    # Retry info
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retries configured")
    first_failed_at: datetime = Field(..., description="When first failure occurred")
    last_failed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When most recent failure occurred"
    )

    # Resolution
    is_resolved: bool = Field(default=False, description="Whether issue was resolved")
    resolution_notes: Optional[str] = Field(None, description="Notes on resolution")
