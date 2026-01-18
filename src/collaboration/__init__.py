"""Team collaboration features."""

from .crdt import (
    CRDTDocument,
    CRDTOperation,
    LWWRegister,
    TestSpecCRDT,
    VectorClock,
)
from .cursors import CursorState, CursorTracker, interpolate_cursor_position
from .models import (
    CURSOR_COLORS,
    BroadcastMessage,
    CollaborationEvent,
    CollaborationEventType,
    CollaborativeComment,
    CursorPosition,
    EditOperation,
    PresenceStatus,
    SelectionRange,
    UserPresence,
)
from .presence import PresenceManager
from .realtime import (
    RealtimeConfig,
    RealtimeManager,
    RealtimeSession,
    create_realtime_manager,
)
from .team import (
    ApprovalRequest,
    AuditLogEntry,
    Comment,
    Permission,
    Role,
    Team,
    TeamManager,
    User,
    Workspace,
)

__all__ = [
    # Team management
    "TeamManager",
    "User",
    "Team",
    "Workspace",
    "Role",
    "Permission",
    "Comment",
    "ApprovalRequest",
    "AuditLogEntry",
    # Real-time models
    "PresenceStatus",
    "CollaborationEventType",
    "CursorPosition",
    "SelectionRange",
    "UserPresence",
    "CollaborationEvent",
    "CollaborativeComment",
    "EditOperation",
    "BroadcastMessage",
    "CURSOR_COLORS",
    # Presence
    "PresenceManager",
    # Cursors
    "CursorTracker",
    "CursorState",
    "interpolate_cursor_position",
    # CRDT
    "VectorClock",
    "CRDTOperation",
    "LWWRegister",
    "CRDTDocument",
    "TestSpecCRDT",
    # Realtime
    "RealtimeConfig",
    "RealtimeSession",
    "RealtimeManager",
    "create_realtime_manager",
]
