"""Data models for two-way IDE synchronization."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class SyncSource(str, Enum):
    """Source of a sync event."""

    IDE = "ide"  # From VS Code/Cursor/Windsurf
    CLOUD = "cloud"  # From Argus cloud
    MCP = "mcp"  # From MCP tool call


class SyncEventType(str, Enum):
    """Types of sync events."""

    # Test operations
    TEST_CREATED = "test_created"
    TEST_UPDATED = "test_updated"
    TEST_DELETED = "test_deleted"

    # Step operations
    STEP_ADDED = "step_added"
    STEP_UPDATED = "step_updated"
    STEP_REMOVED = "step_removed"
    STEPS_REORDERED = "steps_reordered"

    # Metadata operations
    METADATA_UPDATED = "metadata_updated"
    ASSERTIONS_UPDATED = "assertions_updated"

    # Sync control
    FULL_SYNC = "full_sync"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


class ConflictResolutionStrategy(str, Enum):
    """How to resolve sync conflicts."""

    KEEP_LOCAL = "keep_local"  # Local changes win
    KEEP_REMOTE = "keep_remote"  # Remote changes win
    MANUAL = "manual"  # Require user decision
    MERGE = "merge"  # Attempt automatic merge
    LATEST_WINS = "latest_wins"  # Most recent timestamp wins


class SyncStatus(str, Enum):
    """Sync status for a test or project."""

    SYNCED = "synced"  # Fully synchronized
    PENDING = "pending"  # Changes pending upload
    CONFLICT = "conflict"  # Has unresolved conflicts
    SYNCING = "syncing"  # Currently syncing
    ERROR = "error"  # Sync error occurred


@dataclass
class SyncEvent:
    """A single synchronization event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    type: SyncEventType = SyncEventType.TEST_UPDATED
    source: SyncSource = SyncSource.IDE
    project_id: str = ""
    test_id: Optional[str] = None
    path: list[str] = field(default_factory=list)  # JSON path for partial updates
    content: Optional[dict] = None  # Full or partial content
    previous_content: Optional[dict] = None  # For undo/conflict detection
    local_version: int = 0
    remote_version: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    checksum: Optional[str] = None  # Content hash for validation

    def to_dict(self) -> dict:
        """Convert to dictionary for transmission."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source.value,
            "project_id": self.project_id,
            "test_id": self.test_id,
            "path": self.path,
            "content": self.content,
            "previous_content": self.previous_content,
            "local_version": self.local_version,
            "remote_version": self.remote_version,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyncEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            type=SyncEventType(data.get("type", "test_updated")),
            source=SyncSource(data.get("source", "ide")),
            project_id=data.get("project_id", ""),
            test_id=data.get("test_id"),
            path=data.get("path", []),
            content=data.get("content"),
            previous_content=data.get("previous_content"),
            local_version=data.get("local_version", 0),
            remote_version=data.get("remote_version", 0),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            user_id=data.get("user_id"),
            checksum=data.get("checksum"),
        )


@dataclass
class SyncConflict:
    """A sync conflict that needs resolution."""

    id: str = field(default_factory=lambda: str(uuid4()))
    test_id: str = ""
    path: list[str] = field(default_factory=list)
    local_value: Any = None
    remote_value: Any = None
    local_version: int = 0
    remote_version: int = 0
    local_timestamp: Optional[datetime] = None
    remote_timestamp: Optional[datetime] = None
    resolved: bool = False
    resolution: Optional[ConflictResolutionStrategy] = None
    resolved_value: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "test_id": self.test_id,
            "path": self.path,
            "local_value": self.local_value,
            "remote_value": self.remote_value,
            "local_version": self.local_version,
            "remote_version": self.remote_version,
            "local_timestamp": self.local_timestamp.isoformat()
            if self.local_timestamp
            else None,
            "remote_timestamp": self.remote_timestamp.isoformat()
            if self.remote_timestamp
            else None,
            "resolved": self.resolved,
            "resolution": self.resolution.value if self.resolution else None,
            "resolved_value": self.resolved_value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TestSyncState:
    """Sync state for a single test."""

    test_id: str
    local_version: int = 0
    remote_version: int = 0
    status: SyncStatus = SyncStatus.SYNCED
    last_synced: Optional[datetime] = None
    last_local_change: Optional[datetime] = None
    last_remote_change: Optional[datetime] = None
    pending_changes: list[SyncEvent] = field(default_factory=list)
    conflicts: list[SyncConflict] = field(default_factory=list)
    checksum: Optional[str] = None

    @property
    def has_conflicts(self) -> bool:
        """Check if there are unresolved conflicts."""
        return any(not c.resolved for c in self.conflicts)

    @property
    def has_pending_changes(self) -> bool:
        """Check if there are pending changes to sync."""
        return len(self.pending_changes) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "local_version": self.local_version,
            "remote_version": self.remote_version,
            "status": self.status.value,
            "last_synced": self.last_synced.isoformat() if self.last_synced else None,
            "last_local_change": self.last_local_change.isoformat()
            if self.last_local_change
            else None,
            "last_remote_change": self.last_remote_change.isoformat()
            if self.last_remote_change
            else None,
            "pending_changes_count": len(self.pending_changes),
            "conflicts_count": len(self.conflicts),
            "has_conflicts": self.has_conflicts,
            "checksum": self.checksum,
        }


@dataclass
class ProjectSyncState:
    """Sync state for an entire project."""

    project_id: str
    tests: dict[str, TestSyncState] = field(default_factory=dict)
    last_full_sync: Optional[datetime] = None
    sync_enabled: bool = True

    @property
    def status(self) -> SyncStatus:
        """Get overall project sync status."""
        if not self.tests:
            return SyncStatus.SYNCED

        statuses = [t.status for t in self.tests.values()]

        if SyncStatus.ERROR in statuses:
            return SyncStatus.ERROR
        if SyncStatus.CONFLICT in statuses:
            return SyncStatus.CONFLICT
        if SyncStatus.SYNCING in statuses:
            return SyncStatus.SYNCING
        if SyncStatus.PENDING in statuses:
            return SyncStatus.PENDING

        return SyncStatus.SYNCED

    @property
    def total_pending(self) -> int:
        """Get total pending changes across all tests."""
        return sum(len(t.pending_changes) for t in self.tests.values())

    @property
    def total_conflicts(self) -> int:
        """Get total conflicts across all tests."""
        return sum(len(t.conflicts) for t in self.tests.values())

    def get_test_state(self, test_id: str) -> TestSyncState:
        """Get or create sync state for a test."""
        if test_id not in self.tests:
            self.tests[test_id] = TestSyncState(test_id=test_id)
        return self.tests[test_id]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "status": self.status.value,
            "tests_count": len(self.tests),
            "total_pending": self.total_pending,
            "total_conflicts": self.total_conflicts,
            "last_full_sync": self.last_full_sync.isoformat()
            if self.last_full_sync
            else None,
            "sync_enabled": self.sync_enabled,
            "tests": {tid: t.to_dict() for tid, t in self.tests.items()},
        }


@dataclass
class SyncPushResult:
    """Result of a push operation."""

    success: bool
    events_pushed: int = 0
    new_version: int = 0
    conflicts: list[SyncConflict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "events_pushed": self.events_pushed,
            "new_version": self.new_version,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "error": self.error,
        }


@dataclass
class SyncPullResult:
    """Result of a pull operation."""

    success: bool
    events: list[SyncEvent] = field(default_factory=list)
    new_version: int = 0
    has_more: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "events": [e.to_dict() for e in self.events],
            "new_version": self.new_version,
            "has_more": self.has_more,
            "error": self.error,
        }


@dataclass
class TestSpec:
    """A test specification for syncing."""

    id: str
    name: str = ""
    description: str = ""
    steps: list[dict] = field(default_factory=list)
    assertions: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    version: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "assertions": self.assertions,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestSpec":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=data.get("steps", []),
            assertions=data.get("assertions", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", 0),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
        )
