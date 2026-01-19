"""Chat API endpoint that routes through LangGraph orchestrator."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from src.orchestrator.chat_graph import ChatState, create_chat_graph
from src.orchestrator.checkpointer import get_checkpointer
from src.services.audit_logger import (
    AuditAction,
    AuditStatus,
    ResourceType,
    get_audit_logger,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """A chat message."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    messages: list[ChatMessage]
    thread_id: str | None = None
    app_url: str | None = None


class ChatResponse(BaseModel):
    """Response from chat."""
    message: str
    thread_id: str
    tool_calls: list[dict] | None = None


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

            # Log chat start to audit trail
            audit = get_audit_logger()
            start_time = datetime.now(UTC)
            await audit.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                resource_id=thread_id,
                description=f"Chat stream started with {len(request.messages)} messages",
                metadata={
                    "thread_id": thread_id,
                    "message_count": len(request.messages),
                    "app_url": request.app_url,
                    "last_user_message": request.messages[-1].content[:200] if request.messages else None,
                },
            )

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

                                    # Log tool call to audit trail
                                    await audit.log_tool_execution(
                                        tool_name=tc["name"],
                                        tool_args=tc["args"],
                                        result=None,  # Result comes later
                                        success=True,  # Call initiated
                                        duration_ms=0,  # Unknown at this point
                                        thread_id=thread_id,
                                    )

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

                                # Log tool result to audit trail
                                # Determine success based on result content
                                is_success = True
                                error_msg = None
                                if isinstance(result_content, dict):
                                    is_success = result_content.get("success", True)
                                    error_msg = result_content.get("error")
                                await audit.log_tool_execution(
                                    tool_name=getattr(last_msg, "name", "unknown"),
                                    tool_args={},  # Args were logged with tool call
                                    result=result_content,
                                    success=is_success,
                                    duration_ms=0,  # Not tracked per-tool in stream
                                    thread_id=thread_id,
                                    error=error_msg,
                                )

            # Log successful completion
            duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
            await audit.log(
                action=AuditAction.TEST_RUN,
                resource_type=ResourceType.TEST,
                resource_id=thread_id,
                description="Chat stream completed successfully",
                metadata={
                    "thread_id": thread_id,
                    "tool_calls_count": len(sent_tool_calls),
                    "tool_results_count": len(sent_tool_results),
                },
                duration_ms=duration_ms,
            )

            # AI SDK finish message: d:{"finishReason":"stop"}\n
            yield f'd:{json.dumps({"finishReason": "stop", "threadId": thread_id})}\n'

        except Exception as e:
            logger.exception("Chat stream error", error=str(e))
            # Log error to audit trail
            audit = get_audit_logger()
            await audit.log_error(
                error=e,
                context="chat_streaming",
                resource_id=thread_id,
                metadata={
                    "thread_id": thread_id,
                    "message_count": len(request.messages),
                    "app_url": request.app_url,
                },
            )
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


def _serialize_message_content(content) -> Any:
    """Serialize message content to a JSON-compatible format.

    Handles:
    - String content (simple text)
    - List content (AI messages with text blocks and tool_use blocks)
    - Dict content (tool results)
    """
    if isinstance(content, str):
        # Try to parse as JSON for tool results
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
    elif isinstance(content, list):
        # AI messages can have list content with text blocks and tool_use blocks
        return content
    elif isinstance(content, dict):
        return content
    else:
        return str(content)


def _serialize_message(msg: BaseMessage) -> dict:
    """Serialize a LangChain message to a dict for the API response.

    Includes all message types:
    - HumanMessage -> role: "user"
    - AIMessage -> role: "assistant" (includes tool_calls if present)
    - ToolMessage -> role: "tool" (includes tool_call_id and tool result)
    """
    if isinstance(msg, HumanMessage):
        return {
            "role": "user",
            "content": _serialize_message_content(msg.content),
        }
    elif isinstance(msg, AIMessage):
        result = {
            "role": "assistant",
            "content": _serialize_message_content(msg.content),
        }
        # Include tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                }
                for tc in msg.tool_calls
            ]
        return result
    elif isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "name": getattr(msg, "name", None),
            "content": _serialize_message_content(msg.content),
        }
    else:
        # Fallback for any other message type
        return {
            "role": "unknown",
            "content": str(msg.content) if hasattr(msg, "content") else str(msg),
        }


@router.get("/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """Get chat history for a thread.

    Returns all messages including:
    - User messages (HumanMessage)
    - Assistant messages (AIMessage) with tool_calls if present
    - Tool results (ToolMessage) with full content including _artifact_refs
    """
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
                    _serialize_message(msg)
                    for msg in messages
                    if isinstance(msg, (HumanMessage, AIMessage, ToolMessage))
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


@router.delete("/cancel/{thread_id}")
async def cancel_chat(thread_id: str):
    """Cancel an ongoing chat execution.

    This updates the state to stop execution by setting should_continue to False.
    The chat graph will check this flag and stop processing.
    """
    checkpointer = get_checkpointer()
    graph = create_chat_graph()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Get current state
        state = await app.aget_state(config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Update state to stop execution
        updated_values = dict(state.values)
        updated_values["should_continue"] = False

        # Update the state
        await app.aupdate_state(config, updated_values)

        logger.info("Chat execution cancelled", thread_id=thread_id)

        return {
            "success": True,
            "thread_id": thread_id,
            "message": "Chat execution cancelled",
            "cancelled_at": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error cancelling chat", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
