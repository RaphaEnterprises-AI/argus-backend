"""Team collaboration features."""

from .team import (
    TeamManager,
    User,
    Team,
    Workspace,
    Role,
    Permission,
    Comment,
    ApprovalRequest,
    AuditLogEntry,
)
from .models import (
    PresenceStatus,
    CollaborationEventType,
    CursorPosition,
    SelectionRange,
    UserPresence,
    CollaborationEvent,
    CollaborativeComment,
    EditOperation,
    BroadcastMessage,
    CURSOR_COLORS,
)
from .presence import PresenceManager
from .cursors import CursorTracker, CursorState, interpolate_cursor_position
from .crdt import (
    VectorClock,
    CRDTOperation,
    LWWRegister,
    CRDTDocument,
    TestSpecCRDT,
)
from .realtime import (
    RealtimeConfig,
    RealtimeSession,
    RealtimeManager,
    create_realtime_manager,
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
