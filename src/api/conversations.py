"""Chat Conversations API endpoints.

Provides endpoints for managing chat conversations:
- Listing conversations for a user
- Creating new conversations
- Getting conversation details
- Updating conversations (title, etc.)
- Deleting conversations
- Getting messages for a conversation
- Adding messages to a conversation
"""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.tenant import validate_uuid
from src.api.security.auth import UserContext, get_current_user
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    project_id: str | None = Field(None, description="Optional project to associate with")
    title: str | None = Field(None, max_length=200, description="Conversation title")

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str | None) -> str | None:
        if v is not None:
            validate_uuid(v, "project_id")
        return v


class UpdateConversationRequest(BaseModel):
    """Request to update a conversation."""
    title: str | None = Field(None, max_length=200)
    preview: str | None = Field(None, max_length=500)


class ConversationResponse(BaseModel):
    """Conversation details response."""
    id: str
    project_id: str | None
    user_id: str
    title: str | None
    preview: str | None
    message_count: int
    created_at: str
    updated_at: str


class CreateMessageRequest(BaseModel):
    """Request to add a message to a conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=100000)
    tool_invocations: dict | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be empty or whitespace only")
        return v


class MessageResponse(BaseModel):
    """Message details response."""
    id: str
    conversation_id: str
    role: str
    content: str
    tool_invocations: dict | None
    created_at: str


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def list_conversations(
    project_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    user: UserContext = Depends(get_current_user),
) -> dict:
    """List conversations for the current user.

    Args:
        project_id: Optional filter by project
        limit: Maximum number of conversations to return (default 50, max 100)
        offset: Number of conversations to skip
        user: Current authenticated user

    Returns:
        List of conversations with metadata
    """
    if project_id:
        validate_uuid(project_id, "project_id")

    limit = min(limit, 100)

    supabase = get_supabase_client()

    # Build query
    query_parts = [
        f"user_id=eq.{user.user_id}",
        "order=updated_at.desc",
        f"limit={limit}",
        f"offset={offset}",
    ]

    if project_id:
        query_parts.insert(1, f"project_id=eq.{project_id}")

    query = "&".join(query_parts)

    result = await supabase.request(f"/chat_conversations?{query}")

    if result.get("error"):
        logger.error("Failed to list conversations", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to list conversations")

    conversations = result.get("data", [])

    # Transform to response format
    return {
        "conversations": [
            ConversationResponse(
                id=c["id"],
                project_id=c.get("project_id"),
                user_id=c["user_id"],
                title=c.get("title"),
                preview=c.get("preview"),
                message_count=c.get("message_count", 0),
                created_at=c["created_at"],
                updated_at=c["updated_at"],
            ).model_dump()
            for c in conversations
        ],
        "total": len(conversations),  # Note: For accurate total, would need count query
        "limit": limit,
        "offset": offset,
    }


@router.post("")
async def create_conversation(
    request: CreateConversationRequest,
    user: UserContext = Depends(get_current_user),
) -> ConversationResponse:
    """Create a new conversation.

    Args:
        request: Conversation creation request
        user: Current authenticated user

    Returns:
        Created conversation details
    """
    supabase = get_supabase_client()

    now = datetime.now(UTC).isoformat()

    insert_data = {
        "user_id": user.user_id,
        "project_id": request.project_id,
        "title": request.title or "New Conversation",
        "message_count": 0,
        "created_at": now,
        "updated_at": now,
    }

    result = await supabase.request(
        "/chat_conversations",
        method="POST",
        body=insert_data,
    )

    if result.get("error"):
        logger.error("Failed to create conversation", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to create conversation")

    data = result.get("data", [{}])[0]

    logger.info(
        "Conversation created",
        conversation_id=data.get("id"),
        user_id=user.user_id,
        project_id=request.project_id,
    )

    return ConversationResponse(
        id=data["id"],
        project_id=data.get("project_id"),
        user_id=data["user_id"],
        title=data.get("title"),
        preview=data.get("preview"),
        message_count=data.get("message_count", 0),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: UserContext = Depends(get_current_user),
) -> ConversationResponse:
    """Get a specific conversation.

    Args:
        conversation_id: The conversation ID
        user: Current authenticated user

    Returns:
        Conversation details
    """
    validate_uuid(conversation_id, "conversation_id")

    supabase = get_supabase_client()

    result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}"
    )

    if result.get("error"):
        logger.error("Failed to get conversation", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to get conversation")

    data = result.get("data", [])
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    c = data[0]
    return ConversationResponse(
        id=c["id"],
        project_id=c.get("project_id"),
        user_id=c["user_id"],
        title=c.get("title"),
        preview=c.get("preview"),
        message_count=c.get("message_count", 0),
        created_at=c["created_at"],
        updated_at=c["updated_at"],
    )


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    user: UserContext = Depends(get_current_user),
) -> ConversationResponse:
    """Update a conversation.

    Args:
        conversation_id: The conversation ID
        request: Update request
        user: Current authenticated user

    Returns:
        Updated conversation details
    """
    validate_uuid(conversation_id, "conversation_id")

    supabase = get_supabase_client()

    # Build update data, only including non-None fields
    update_data = {"updated_at": datetime.now(UTC).isoformat()}
    if request.title is not None:
        update_data["title"] = request.title
    if request.preview is not None:
        update_data["preview"] = request.preview

    result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}",
        method="PATCH",
        body=update_data,
    )

    if result.get("error"):
        logger.error("Failed to update conversation", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to update conversation")

    data = result.get("data", [])
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    c = data[0]

    logger.info("Conversation updated", conversation_id=conversation_id)

    return ConversationResponse(
        id=c["id"],
        project_id=c.get("project_id"),
        user_id=c["user_id"],
        title=c.get("title"),
        preview=c.get("preview"),
        message_count=c.get("message_count", 0),
        created_at=c["created_at"],
        updated_at=c["updated_at"],
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: UserContext = Depends(get_current_user),
) -> dict:
    """Delete a conversation and all its messages.

    Args:
        conversation_id: The conversation ID
        user: Current authenticated user

    Returns:
        Success message
    """
    validate_uuid(conversation_id, "conversation_id")

    supabase = get_supabase_client()

    # First verify the conversation belongs to the user
    check_result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}&select=id"
    )

    if not check_result.get("data"):
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete messages first (due to foreign key constraint)
    await supabase.request(
        f"/chat_messages?conversation_id=eq.{conversation_id}",
        method="DELETE",
    )

    # Then delete the conversation
    result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}",
        method="DELETE",
    )

    if result.get("error"):
        logger.error("Failed to delete conversation", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

    logger.info("Conversation deleted", conversation_id=conversation_id)

    return {"success": True, "message": "Conversation deleted"}


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 100,
    offset: int = 0,
    user: UserContext = Depends(get_current_user),
) -> dict:
    """Get messages for a conversation.

    Args:
        conversation_id: The conversation ID
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        user: Current authenticated user

    Returns:
        List of messages
    """
    validate_uuid(conversation_id, "conversation_id")

    limit = min(limit, 500)

    supabase = get_supabase_client()

    # Verify conversation belongs to user
    check_result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}&select=id"
    )

    if not check_result.get("data"):
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get messages
    result = await supabase.request(
        f"/chat_messages?conversation_id=eq.{conversation_id}"
        f"&order=created_at.asc&limit={limit}&offset={offset}"
    )

    if result.get("error"):
        logger.error("Failed to get messages", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to get messages")

    messages = result.get("data", [])

    return {
        "messages": [
            MessageResponse(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                tool_invocations=m.get("tool_invocations"),
                created_at=m["created_at"],
            ).model_dump()
            for m in messages
        ],
        "total": len(messages),
        "limit": limit,
        "offset": offset,
    }


@router.post("/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    request: CreateMessageRequest,
    user: UserContext = Depends(get_current_user),
) -> MessageResponse:
    """Add a message to a conversation.

    Args:
        conversation_id: The conversation ID
        request: Message creation request
        user: Current authenticated user

    Returns:
        Created message details
    """
    validate_uuid(conversation_id, "conversation_id")

    supabase = get_supabase_client()

    # Verify conversation belongs to user
    check_result = await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}&user_id=eq.{user.user_id}&select=id"
    )

    if not check_result.get("data"):
        raise HTTPException(status_code=404, detail="Conversation not found")

    now = datetime.now(UTC).isoformat()

    # Create the message
    message_data = {
        "conversation_id": conversation_id,
        "role": request.role,
        "content": request.content,
        "tool_invocations": request.tool_invocations,
        "created_at": now,
    }

    result = await supabase.request(
        "/chat_messages",
        method="POST",
        body=message_data,
    )

    if result.get("error"):
        logger.error("Failed to add message", error=result.get("error"))
        raise HTTPException(status_code=500, detail="Failed to add message")

    data = result.get("data", [{}])[0]

    # Update conversation's message_count and updated_at
    # Also update preview with truncated content
    preview = request.content[:200] + "..." if len(request.content) > 200 else request.content
    await supabase.request(
        f"/chat_conversations?id=eq.{conversation_id}",
        method="PATCH",
        body={
            "message_count": check_result["data"][0].get("message_count", 0) + 1,
            "updated_at": now,
            "preview": preview if request.role == "user" else None,
        },
    )

    logger.info(
        "Message added",
        conversation_id=conversation_id,
        message_id=data.get("id"),
        role=request.role,
    )

    return MessageResponse(
        id=data["id"],
        conversation_id=data["conversation_id"],
        role=data["role"],
        content=data["content"],
        tool_invocations=data.get("tool_invocations"),
        created_at=data["created_at"],
    )
