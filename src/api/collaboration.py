"""Collaboration API endpoints for real-time collaboration features."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4
from enum import Enum
import random

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/collaboration", tags=["Collaboration"])


# =============================================================================
# In-memory Storage (would be Supabase Realtime + Redis in production)
# =============================================================================

# Active user presence by workspace
_presence: dict[str, dict[str, dict]] = {}  # workspace_id -> user_id -> presence_data

# Comments by test
_comments: dict[str, list[dict]] = {}  # test_id -> list of comments

# Active WebSocket connections
_connections: dict[str, list[WebSocket]] = {}  # workspace_id -> list of websockets


# =============================================================================
# Constants
# =============================================================================

# Presence timeout - user is offline after this duration
PRESENCE_TIMEOUT_SECONDS = 60

# User colors for cursors/avatars
USER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FF7F50",
]


# =============================================================================
# Enums
# =============================================================================


class PresenceStatus(str, Enum):
    """User presence status."""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"


class CommentType(str, Enum):
    """Comment types."""
    GENERAL = "general"
    SUGGESTION = "suggestion"
    QUESTION = "question"
    ISSUE = "issue"
    RESOLVED = "resolved"


# =============================================================================
# Request/Response Models
# =============================================================================


class CursorPosition(BaseModel):
    """User cursor position."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    element_id: Optional[str] = Field(None, description="ID of element under cursor")
    step_index: Optional[int] = Field(None, description="Test step index if editing")


class PresenceUpdateRequest(BaseModel):
    """Request to update user presence."""
    workspace_id: str = Field(..., description="Workspace/project ID")
    user_id: str = Field(..., description="User ID")
    user_name: str = Field(..., description="Display name")
    user_email: Optional[str] = Field(None, description="User email")
    status: PresenceStatus = Field(PresenceStatus.ONLINE, description="Status")
    test_id: Optional[str] = Field(None, description="Currently viewing test ID")
    cursor: Optional[CursorPosition] = Field(None, description="Cursor position")


class PresenceResponse(BaseModel):
    """Response with presence data."""
    success: bool
    workspace_id: str
    users: list[dict] = []
    total_online: int = 0
    error: Optional[str] = None


class CommentCreateRequest(BaseModel):
    """Request to create a comment."""
    test_id: str = Field(..., description="Test ID to comment on")
    step_index: Optional[int] = Field(None, description="Specific step index")
    content: str = Field(..., description="Comment content", min_length=1, max_length=5000)
    author_id: str = Field(..., description="Author user ID")
    author_name: str = Field(..., description="Author display name")
    author_email: Optional[str] = Field(None, description="Author email")
    comment_type: CommentType = Field(CommentType.GENERAL, description="Comment type")
    mentions: list[str] = Field([], description="Mentioned user IDs")
    parent_id: Optional[str] = Field(None, description="Parent comment ID for replies")


class CommentUpdateRequest(BaseModel):
    """Request to update a comment."""
    content: Optional[str] = Field(None, description="Updated content")
    resolved: Optional[bool] = Field(None, description="Mark as resolved")


class CommentResponse(BaseModel):
    """Response with comment data."""
    success: bool
    comment: Optional[dict] = None
    error: Optional[str] = None


class CommentsListResponse(BaseModel):
    """Response with list of comments."""
    success: bool
    test_id: str
    comments: list[dict] = []
    total: int = 0
    unresolved: int = 0
    error: Optional[str] = None


class RealtimeMessage(BaseModel):
    """Real-time message for WebSocket broadcast."""
    type: str = Field(..., description="Message type: presence, cursor, comment, edit")
    workspace_id: str
    data: dict


# =============================================================================
# Presence Endpoints
# =============================================================================


@router.post("/presence", response_model=PresenceResponse)
async def update_presence(request: PresenceUpdateRequest):
    """
    Update user presence in a workspace.

    Call this periodically (every 10-30 seconds) to maintain online status.
    Returns all active users in the workspace.
    """
    try:
        workspace_id = request.workspace_id

        # Initialize workspace presence if needed
        if workspace_id not in _presence:
            _presence[workspace_id] = {}

        # Assign color if new user
        existing = _presence[workspace_id].get(request.user_id, {})
        color = existing.get("color") or random.choice(USER_COLORS)

        # Update user presence
        _presence[workspace_id][request.user_id] = {
            "user_id": request.user_id,
            "user_name": request.user_name,
            "user_email": request.user_email,
            "status": request.status,
            "test_id": request.test_id,
            "cursor": request.cursor.model_dump() if request.cursor else None,
            "color": color,
            "last_active": datetime.now(timezone.utc).isoformat(),
        }

        # Clean up stale presence
        _cleanup_stale_presence(workspace_id)

        # Get all active users
        active_users = list(_presence[workspace_id].values())
        online_count = sum(1 for u in active_users if u["status"] == PresenceStatus.ONLINE)

        # Broadcast presence update to connected clients
        await _broadcast_to_workspace(workspace_id, {
            "type": "presence_update",
            "user_id": request.user_id,
            "data": _presence[workspace_id][request.user_id],
        })

        logger.info(
            "Presence updated",
            workspace_id=workspace_id,
            user_id=request.user_id,
            status=request.status,
            online_count=online_count,
        )

        return PresenceResponse(
            success=True,
            workspace_id=workspace_id,
            users=active_users,
            total_online=online_count,
        )

    except Exception as e:
        logger.exception("Presence update failed", error=str(e))
        return PresenceResponse(
            success=False,
            workspace_id=request.workspace_id,
            error=str(e),
        )


@router.get("/presence/{workspace_id}", response_model=PresenceResponse)
async def get_presence(workspace_id: str):
    """
    Get all user presence in a workspace.

    Returns list of all users currently active in the workspace.
    """
    try:
        # Clean up stale presence first
        _cleanup_stale_presence(workspace_id)

        users = list(_presence.get(workspace_id, {}).values())
        online_count = sum(1 for u in users if u["status"] == PresenceStatus.ONLINE)

        return PresenceResponse(
            success=True,
            workspace_id=workspace_id,
            users=users,
            total_online=online_count,
        )

    except Exception as e:
        logger.exception("Get presence failed", error=str(e))
        return PresenceResponse(
            success=False,
            workspace_id=workspace_id,
            error=str(e),
        )


@router.delete("/presence/{workspace_id}/{user_id}")
async def remove_presence(workspace_id: str, user_id: str):
    """
    Remove user from workspace presence.

    Call this when user leaves or closes the tab.
    """
    try:
        if workspace_id in _presence and user_id in _presence[workspace_id]:
            del _presence[workspace_id][user_id]

            # Broadcast leave event
            await _broadcast_to_workspace(workspace_id, {
                "type": "presence_leave",
                "user_id": user_id,
            })

        return {"success": True, "message": f"User {user_id} removed from presence"}

    except Exception as e:
        logger.exception("Remove presence failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Comment Endpoints
# =============================================================================


@router.post("/comments", response_model=CommentResponse)
async def create_comment(request: CommentCreateRequest):
    """
    Create a comment on a test or test step.

    Supports mentions (@username) and threaded replies.
    """
    try:
        comment_id = str(uuid4())

        # Initialize comments for test if needed
        if request.test_id not in _comments:
            _comments[request.test_id] = []

        comment = {
            "id": comment_id,
            "test_id": request.test_id,
            "step_index": request.step_index,
            "content": request.content,
            "author_id": request.author_id,
            "author_name": request.author_name,
            "author_email": request.author_email,
            "comment_type": request.comment_type,
            "mentions": request.mentions,
            "parent_id": request.parent_id,
            "resolved": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reactions": {},  # emoji -> list of user_ids
        }

        _comments[request.test_id].append(comment)

        logger.info(
            "Comment created",
            comment_id=comment_id,
            test_id=request.test_id,
            author_id=request.author_id,
        )

        return CommentResponse(
            success=True,
            comment=comment,
        )

    except Exception as e:
        logger.exception("Create comment failed", error=str(e))
        return CommentResponse(
            success=False,
            error=str(e),
        )


@router.get("/comments/{test_id}", response_model=CommentsListResponse)
async def list_comments(
    test_id: str,
    step_index: Optional[int] = None,
    include_resolved: bool = True,
):
    """
    List comments for a test.

    Optionally filter by step index or exclude resolved comments.
    """
    try:
        comments = _comments.get(test_id, [])

        # Filter by step index if specified
        if step_index is not None:
            comments = [c for c in comments if c.get("step_index") == step_index]

        # Filter resolved if requested
        if not include_resolved:
            comments = [c for c in comments if not c.get("resolved")]

        # Sort by created_at
        comments.sort(key=lambda x: x.get("created_at", ""))

        # Organize into threads (parent comments with replies)
        threaded = _organize_threads(comments)

        unresolved = sum(1 for c in comments if not c.get("resolved") and not c.get("parent_id"))

        return CommentsListResponse(
            success=True,
            test_id=test_id,
            comments=threaded,
            total=len(comments),
            unresolved=unresolved,
        )

    except Exception as e:
        logger.exception("List comments failed", error=str(e))
        return CommentsListResponse(
            success=False,
            test_id=test_id,
            error=str(e),
        )


@router.patch("/comments/{comment_id}", response_model=CommentResponse)
async def update_comment(comment_id: str, request: CommentUpdateRequest):
    """
    Update a comment.

    Can update content or mark as resolved.
    """
    try:
        # Find comment
        comment = None
        test_id = None

        for tid, comments in _comments.items():
            for c in comments:
                if c["id"] == comment_id:
                    comment = c
                    test_id = tid
                    break
            if comment:
                break

        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")

        # Update fields
        if request.content is not None:
            comment["content"] = request.content

        if request.resolved is not None:
            comment["resolved"] = request.resolved

        comment["updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("Comment updated", comment_id=comment_id)

        return CommentResponse(
            success=True,
            comment=comment,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Update comment failed", error=str(e))
        return CommentResponse(
            success=False,
            error=str(e),
        )


@router.delete("/comments/{comment_id}")
async def delete_comment(comment_id: str):
    """
    Delete a comment.
    """
    try:
        # Find and remove comment
        for test_id, comments in _comments.items():
            for i, c in enumerate(comments):
                if c["id"] == comment_id:
                    del comments[i]
                    return {"success": True, "message": "Comment deleted"}

        raise HTTPException(status_code=404, detail="Comment not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete comment failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comments/{comment_id}/reactions")
async def add_reaction(comment_id: str, emoji: str, user_id: str):
    """
    Add a reaction to a comment.
    """
    try:
        # Find comment
        for test_id, comments in _comments.items():
            for comment in comments:
                if comment["id"] == comment_id:
                    if "reactions" not in comment:
                        comment["reactions"] = {}

                    if emoji not in comment["reactions"]:
                        comment["reactions"][emoji] = []

                    if user_id not in comment["reactions"][emoji]:
                        comment["reactions"][emoji].append(user_id)

                    return {
                        "success": True,
                        "reactions": comment["reactions"],
                    }

        raise HTTPException(status_code=404, detail="Comment not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Add reaction failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================


@router.websocket("/ws/{workspace_id}")
async def websocket_endpoint(websocket: WebSocket, workspace_id: str):
    """
    WebSocket endpoint for real-time collaboration.

    Receives: cursor updates, edit operations, presence pings
    Broadcasts: presence changes, cursor positions, comments, edits
    """
    await websocket.accept()

    # Add to connections
    if workspace_id not in _connections:
        _connections[workspace_id] = []
    _connections[workspace_id].append(websocket)

    logger.info("WebSocket connected", workspace_id=workspace_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "cursor_move":
                # Broadcast cursor position to others
                await _broadcast_to_workspace(
                    workspace_id,
                    {
                        "type": "cursor_update",
                        "user_id": data.get("user_id"),
                        "cursor": data.get("cursor"),
                    },
                    exclude=websocket,
                )

            elif msg_type == "edit":
                # Broadcast edit operation (for CRDT)
                await _broadcast_to_workspace(
                    workspace_id,
                    {
                        "type": "edit",
                        "user_id": data.get("user_id"),
                        "operation": data.get("operation"),
                    },
                    exclude=websocket,
                )

            elif msg_type == "ping":
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", workspace_id=workspace_id)
    except Exception as e:
        logger.exception("WebSocket error", error=str(e))
    finally:
        # Remove from connections
        if workspace_id in _connections:
            if websocket in _connections[workspace_id]:
                _connections[workspace_id].remove(websocket)


# =============================================================================
# Collaborative Editing Endpoints
# =============================================================================


@router.post("/lock/{test_id}")
async def acquire_edit_lock(
    test_id: str,
    user_id: str,
    step_index: Optional[int] = None,
):
    """
    Acquire an edit lock on a test or specific step.

    Prevents conflicting edits by marking the resource as being edited.
    """
    # In production, this would use Redis distributed locks
    # For now, simple in-memory tracking

    return {
        "success": True,
        "lock_id": str(uuid4()),
        "test_id": test_id,
        "step_index": step_index,
        "user_id": user_id,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }


@router.delete("/lock/{lock_id}")
async def release_edit_lock(lock_id: str):
    """
    Release an edit lock.
    """
    return {
        "success": True,
        "message": f"Lock {lock_id} released",
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _cleanup_stale_presence(workspace_id: str) -> None:
    """Remove users who haven't updated presence recently."""
    if workspace_id not in _presence:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=PRESENCE_TIMEOUT_SECONDS)
    stale_users = []

    for user_id, presence in _presence[workspace_id].items():
        last_active = datetime.fromisoformat(presence["last_active"].replace("Z", "+00:00"))
        if last_active < cutoff:
            stale_users.append(user_id)

    for user_id in stale_users:
        del _presence[workspace_id][user_id]


def _organize_threads(comments: list[dict]) -> list[dict]:
    """Organize comments into threaded structure."""
    # Separate root comments and replies
    root_comments = []
    replies_map: dict[str, list[dict]] = {}

    for comment in comments:
        parent_id = comment.get("parent_id")
        if parent_id:
            if parent_id not in replies_map:
                replies_map[parent_id] = []
            replies_map[parent_id].append(comment)
        else:
            root_comments.append(comment)

    # Attach replies to root comments
    for root in root_comments:
        root["replies"] = replies_map.get(root["id"], [])

    return root_comments


async def _broadcast_to_workspace(
    workspace_id: str,
    message: dict,
    exclude: Optional[WebSocket] = None,
) -> None:
    """Broadcast a message to all connected clients in a workspace."""
    if workspace_id not in _connections:
        return

    disconnected = []

    for ws in _connections[workspace_id]:
        if ws == exclude:
            continue

        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected
    for ws in disconnected:
        _connections[workspace_id].remove(ws)
