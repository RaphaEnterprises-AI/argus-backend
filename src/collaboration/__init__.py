"""Team collaboration features.

This module provides real-time collaboration capabilities including:
- User presence tracking (who is online and what they're viewing)
- CRDT-based collaborative editing for test specifications
- Cursor position tracking for live editing
- Comments and @mentions on tests

Persistence Strategy:
- Sessions and CRDT documents: Persisted to database (critical state)
- Comments: Persisted to database (user content)
- Presence: In-memory with optional database persistence (ephemeral but useful for analytics)
- Cursor positions: In-memory only (highly ephemeral, recreated on reconnect)
"""

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
from .persistence import CollaborationPersistence, get_collaboration_persistence
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
    # Persistence
    "CollaborationPersistence",
    "get_collaboration_persistence",
]
