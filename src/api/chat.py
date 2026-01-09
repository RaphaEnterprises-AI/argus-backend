"""Chat API endpoint that routes through LangGraph orchestrator."""

from typing import Optional, List
from datetime import datetime, timezone
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
    """Send a message and stream the response using Vercel AI SDK compatible format."""

    thread_id = request.thread_id or str(uuid.uuid4())

    async def generate_ai_sdk_stream():
        """Generate stream in Vercel AI SDK format (text streaming protocol)."""
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

            # Track sent items to avoid duplicates (values stream sends full state each time)
            sent_tool_calls = set()
            sent_tool_results = set()

            # Stream the graph execution
            async for event in app.astream(
                initial_state,
                config,
                stream_mode=["messages", "values"]
            ):
                event_type = event[0] if isinstance(event, tuple) else "unknown"
                event_data = event[1] if isinstance(event, tuple) else event

                if event_type == "messages":
                    # Handle message streaming - output in AI SDK format
                    if isinstance(event_data, tuple):
                        chunk, metadata = event_data
                        # Skip ToolMessage content - it's handled as 'a:' in values stream
                        # Only emit AIMessage text content as '0:' text
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            # Check if this is a ToolMessage (skip - handled separately)
                            if isinstance(chunk, ToolMessage):
                                continue

                            content = chunk.content
                            # Handle different content types
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        # AI SDK text format: 0:"text"\n
                                        yield f'0:{json.dumps(item["text"])}\n'
                            elif isinstance(content, str):
                                yield f'0:{json.dumps(content)}\n'

                elif event_type == "values":
                    # Check for tool calls and tool results in the state
                    state = event_data
                    if isinstance(state, dict):
                        messages = state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]

                            # Check for tool calls (AI requesting tool execution)
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    tc_id = tc.get("id", str(uuid.uuid4()))
                                    # Skip if already sent
                                    if tc_id in sent_tool_calls:
                                        continue
                                    sent_tool_calls.add(tc_id)

                                    # AI SDK tool call format: 9:{"toolCallId":...}\n
                                    tool_call_data = {
                                        "toolCallId": tc_id,
                                        "toolName": tc["name"],
                                        "args": tc["args"],
                                    }
                                    yield f'9:{json.dumps(tool_call_data)}\n'

                            # Check for tool results (results from tool execution)
                            if isinstance(last_msg, ToolMessage):
                                # Skip if already sent
                                if last_msg.tool_call_id in sent_tool_results:
                                    continue
                                sent_tool_results.add(last_msg.tool_call_id)

                                # Parse content if it's a JSON string, otherwise use as-is
                                result_content = last_msg.content
                                try:
                                    # Try to parse as JSON to avoid double-encoding
                                    result_content = json.loads(last_msg.content)
                                except (json.JSONDecodeError, TypeError):
                                    pass  # Keep as string if not valid JSON

                                # AI SDK tool result format: a:{"toolCallId":...,"result":...}\n
                                tool_result_data = {
                                    "toolCallId": last_msg.tool_call_id,
                                    "result": result_content,
                                }
                                yield f'a:{json.dumps(tool_result_data)}\n'

            # AI SDK finish message: d:{"finishReason":"stop"}\n
            yield f'd:{json.dumps({"finishReason": "stop", "threadId": thread_id})}\n'

        except Exception as e:
            logger.exception("Chat stream error", error=str(e))
            # AI SDK error format: 3:"error message"\n (must be a string, not object)
            yield f'3:{json.dumps(str(e))}\n'

    return StreamingResponse(
        generate_ai_sdk_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-Id": thread_id,
        }
    )


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
