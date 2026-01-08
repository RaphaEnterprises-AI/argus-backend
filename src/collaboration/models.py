"""Models for real-time collaboration features."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class PresenceStatus(str, Enum):
    """User presence status."""

    ONLINE = "online"
    IDLE = "idle"
    OFFLINE = "offline"
    BUSY = "busy"


class CollaborationEventType(str, Enum):
    """Types of collaboration events."""

    # Presence events
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_IDLE = "user_idle"

    # Cursor events
    CURSOR_MOVE = "cursor_move"
    CURSOR_SELECT = "cursor_select"

    # Edit events
    EDIT_START = "edit_start"
    EDIT_END = "edit_end"
    EDIT_CHANGE = "edit_change"

    # Comment events
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"
    COMMENT_RESOLVED = "comment_resolved"

    # Test events
    TEST_UPDATED = "test_updated"
    TEST_RUNNING = "test_running"
    TEST_COMPLETED = "test_completed"


# Predefined cursor colors for different users
CURSOR_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
]


@dataclass
class CursorPosition:
    """A user's cursor position in the editor."""

    x: float = 0
    y: float = 0
    element_id: Optional[str] = None  # ID of element cursor is over
    step_index: Optional[int] = None  # Index if in test step editor
    field_name: Optional[str] = None  # Field being edited


@dataclass
class SelectionRange:
    """A text selection range."""

    start: int = 0
    end: int = 0
    element_id: Optional[str] = None


@dataclass
class UserPresence:
    """Real-time presence information for a user."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    avatar_url: Optional[str] = None
    workspace_id: str = ""
    test_id: Optional[str] = None
    status: PresenceStatus = PresenceStatus.ONLINE
    color: str = ""
    cursor: Optional[CursorPosition] = None
    selection: Optional[SelectionRange] = None
    last_active: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    connected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        """Assign color if not set."""
        if not self.color:
            # Deterministic color based on user_id
            color_index = hash(self.user_id) % len(CURSOR_COLORS)
            self.color = CURSOR_COLORS[color_index]

    def to_dict(self) -> dict:
        """Convert to dictionary for broadcast."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "avatar_url": self.avatar_url,
            "workspace_id": self.workspace_id,
            "test_id": self.test_id,
            "status": self.status.value,
            "color": self.color,
            "cursor": {
                "x": self.cursor.x,
                "y": self.cursor.y,
                "element_id": self.cursor.element_id,
                "step_index": self.cursor.step_index,
                "field_name": self.cursor.field_name,
            } if self.cursor else None,
            "selection": {
                "start": self.selection.start,
                "end": self.selection.end,
                "element_id": self.selection.element_id,
            } if self.selection else None,
            "last_active": self.last_active.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserPresence":
        """Create from dictionary."""
        cursor = None
        if data.get("cursor"):
            cursor = CursorPosition(**data["cursor"])

        selection = None
        if data.get("selection"):
            selection = SelectionRange(**data["selection"])

        return cls(
            id=data.get("id", str(uuid4())),
            user_id=data.get("user_id", ""),
            user_name=data.get("user_name", ""),
            user_email=data.get("user_email", ""),
            avatar_url=data.get("avatar_url"),
            workspace_id=data.get("workspace_id", ""),
            test_id=data.get("test_id"),
            status=PresenceStatus(data.get("status", "online")),
            color=data.get("color", ""),
            cursor=cursor,
            selection=selection,
        )


@dataclass
class CollaborationEvent:
    """An event in the collaboration system."""

    id: str = field(default_factory=lambda: str(uuid4()))
    type: CollaborationEventType = CollaborationEventType.USER_JOINED
    user_id: str = ""
    workspace_id: str = ""
    test_id: Optional[str] = None
    payload: dict = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for broadcast."""
        return {
            "id": self.id,
            "type": self.type.value,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "test_id": self.test_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CollaborativeComment:
    """A comment in the collaborative editing system."""

    id: str = field(default_factory=lambda: str(uuid4()))
    test_id: str = ""
    step_index: Optional[int] = None  # Which step (null = test level)
    author_id: str = ""
    author_name: str = ""
    author_avatar: Optional[str] = None
    content: str = ""
    mentions: list[str] = field(default_factory=list)  # User IDs
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    parent_id: Optional[str] = None  # For threaded comments
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "test_id": self.test_id,
            "step_index": self.step_index,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_avatar": self.author_avatar,
            "content": self.content,
            "mentions": self.mentions,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class EditOperation:
    """A single edit operation for CRDT sync."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    test_id: str = ""
    operation: str = "update"  # "insert", "update", "delete"
    path: list[str] = field(default_factory=list)  # JSON path to edited field
    value: Any = None
    previous_value: Any = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    vector_clock: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "test_id": self.test_id,
            "operation": self.operation,
            "path": self.path,
            "value": self.value,
            "previous_value": self.previous_value,
            "timestamp": self.timestamp.isoformat(),
            "vector_clock": self.vector_clock,
        }


@dataclass
class BroadcastMessage:
    """A message to broadcast to collaboration channel."""

    channel: str  # e.g., "workspace:uuid", "test:uuid"
    event: str  # Event name
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "event": self.event,
            "payload": self.payload,
        }
