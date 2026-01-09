"""Chat API endpoint that routes through LangGraph orchestrator."""

from typing import Optional, List
from datetime import datetime, timezone
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
import structlog

from src.orchestrator.chat_graph import create_chat_graph, ChatState
from src.orchestrator.checkpointer import get_checkpointer

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """A chat message."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    messages: List[ChatMessage]
    thread_id: Optional[str] = None
    app_url: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat."""
    message: str
    thread_id: str
    tool_calls: Optional[List[dict]] = None


@router.post("/message")
async def send_message(request: ChatRequest):
    """Send a message and get a response (non-streaming)."""

    thread_id = request.thread_id or str(uuid.uuid4())

    # Create graph with checkpointer
    checkpointer = get_checkpointer()
    graph = create_chat_graph()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    # Convert messages to LangChain format
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))

    # Create initial state
    initial_state: ChatState = {
        "messages": lc_messages,
        "app_url": request.app_url or "",
        "current_tool": None,
        "tool_results": [],
        "session_id": thread_id,
    }

    logger.info("Processing chat message", thread_id=thread_id, message_count=len(lc_messages))

    # Run the graph
    final_state = await app.ainvoke(initial_state, config)

    # Get the last AI message
    messages = final_state.get("messages", [])
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if last_ai_message:
        tool_calls_list = None
        if hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
            tool_calls_list = [tc for tc in last_ai_message.tool_calls]

        return ChatResponse(
            message=last_ai_message.content,
            thread_id=thread_id,
            tool_calls=tool_calls_list,
        )
    else:
        return ChatResponse(
            message="No response generated",
            thread_id=thread_id,
        )


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """Send a message and stream the response using Server-Sent Events."""

    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        try:
            # Create graph with checkpointer
            checkpointer = get_checkpointer()
            graph = create_chat_graph()
            app = graph.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": thread_id}}

            # Convert messages
            lc_messages = []
            for msg in request.messages:
                if msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content))

            initial_state: ChatState = {
                "messages": lc_messages,
                "app_url": request.app_url or "",
                "current_tool": None,
                "tool_results": [],
                "session_id": thread_id,
            }

            logger.info("Starting chat stream", thread_id=thread_id)

            # Stream the graph execution
            async for event in app.astream(
                initial_state,
                config,
                stream_mode=["messages", "values"]
            ):
                event_type = event[0] if isinstance(event, tuple) else "unknown"
                event_data = event[1] if isinstance(event, tuple) else event

                if event_type == "messages":
                    # Handle message streaming
                    if isinstance(event_data, tuple):
                        chunk, metadata = event_data
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            yield {
                                "event": "token",
                                "data": json.dumps({
                                    "content": chunk.content,
                                    "thread_id": thread_id,
                                })
                            }
                elif event_type == "values":
                    # Check for tool calls in the state
                    state = event_data
                    if isinstance(state, dict):
                        messages = state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    yield {
                                        "event": "tool_call",
                                        "data": json.dumps({
                                            "name": tc["name"],
                                            "args": tc["args"],
                                            "thread_id": thread_id,
                                        })
                                    }

            yield {
                "event": "complete",
                "data": json.dumps({"thread_id": thread_id})
            }

        except Exception as e:
            logger.exception("Chat stream error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e), "thread_id": thread_id})
            }

    return EventSourceResponse(event_generator())


@router.get("/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """Get chat history for a thread."""
    checkpointer = get_checkpointer()
    graph = create_chat_graph()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = await app.aget_state(config)

        if state and state.values:
            messages = state.values.get("messages", [])
            return {
                "thread_id": thread_id,
                "messages": [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content if hasattr(msg, "content") else str(msg),
                    }
                    for msg in messages
                    if isinstance(msg, (HumanMessage, AIMessage))
                ],
            }
        else:
            return {"thread_id": thread_id, "messages": []}

    except Exception as e:
        logger.exception("Failed to get chat history", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=404, detail=f"Thread not found: {e}")


@router.get("/threads")
async def list_threads(limit: int = 20):
    """List recent chat threads.

    Note: This is a simplified implementation. In production,
    you would query the checkpointer's underlying storage.
    """
    # For MemorySaver, we don't have easy access to list all threads
    # This would require PostgresSaver with direct DB queries
    return {
        "threads": [],
        "message": "Thread listing requires PostgresSaver. Use /history/{thread_id} with known thread IDs.",
    }


@router.delete("/history/{thread_id}")
async def delete_chat_history(thread_id: str):
    """Delete chat history for a thread.

    Note: MemorySaver doesn't support deletion. Use PostgresSaver for full support.
    """
    # For production, implement with PostgresSaver
    logger.info("Delete chat history requested", thread_id=thread_id)
    return {
        "thread_id": thread_id,
        "deleted": False,
        "message": "Deletion not supported with MemorySaver. Use PostgresSaver for full support.",
    }
