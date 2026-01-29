"""Chat API endpoint that routes through LangGraph orchestrator.

This module provides the chat API with support for:
- User-configurable AI model selection (BYOK)
- Multi-provider routing (Anthropic, OpenAI, Google, Groq, Together)
- Token usage tracking for billing
- Langfuse integration for LLM observability and cost tracking

Langfuse Cost Tracking:
- CallbackHandler automatically captures model name and token usage
- Model name from user's AI Hub selection is passed to Langfuse
- Langfuse matches model name to pricing definitions for cost calculation
- Enable with LANGFUSE_ENABLED=true and provide credentials
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from src.api.security.auth import UserContext, get_current_user
from src.orchestrator.chat_graph import AIConfig, ChatState, create_chat_graph
from src.orchestrator.checkpointer import get_checkpointer
from src.orchestrator.langfuse_integration import (
    flush_langfuse,
    get_langfuse_handler,
)
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


class AIConfigRequest(BaseModel):
    """AI configuration from frontend (user's model selection)."""
    model: str | None = None  # Model ID (e.g., "claude-sonnet-4-5", "gpt-4o")
    provider: str | None = None  # Provider name (e.g., "anthropic", "openai")
    use_byok: bool = True  # Whether to use user's BYOK key if available


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    messages: list[ChatMessage]
    thread_id: str | None = None
    app_url: str | None = None
    ai_config: AIConfigRequest | None = None  # User's AI model selection
    user_id: str | None = None  # User ID for BYOK key lookup (from auth)


class ChatResponse(BaseModel):
    """Response from chat."""
    message: str
    thread_id: str
    tool_calls: list[dict] | None = None


async def _build_ai_config(
    ai_config_request: AIConfigRequest | None,
    user_id: str | None,
) -> AIConfig | None:
    """Build AI configuration for chat graph from user request.

    This function:
    1. Gets user's AI preferences if no model specified
    2. Looks up and decrypts user's BYOK key if available
    3. Returns config for chat graph to use

    Args:
        ai_config_request: AI config from frontend request
        user_id: User ID for BYOK key lookup

    Returns:
        AIConfig dict for chat graph, or None to use defaults
    """
    # Use provided config or create default one
    if ai_config_request:
        model = ai_config_request.model
        provider = ai_config_request.provider
        use_byok = ai_config_request.use_byok
    else:
        model = None
        provider = None
        use_byok = True  # Default to using BYOK if available

    # If no model specified, use defaults
    if not model:
        model = "claude-sonnet-4-20250514"
        provider = "anthropic"

    # Build base config
    config: AIConfig = {
        "model": model,
        "provider": provider or "anthropic",
        "api_key": None,
        "user_id": user_id,
        "track_usage": True,
    }

    # Try to get user's API key (BYOK or platform fallback)
    if user_id:
        try:
            from src.services.provider_router import get_provider_router

            router_instance = get_provider_router()
            ai_config_result = await router_instance.get_ai_config(
                user_id=user_id,
                model=model,
                allow_platform_fallback=True,  # Always allow platform fallback
            )

            # Use the key (could be BYOK or platform)
            if ai_config_result.api_key:
                config["api_key"] = ai_config_result.api_key
                logger.info(
                    "Got API key for chat",
                    user_id=user_id,
                    provider=config["provider"],
                    model=model,
                    key_source=ai_config_result.key_source,
                )
        except Exception as e:
            # Log the error - will try platform fallback below
            logger.warning(
                "Failed to get API key from provider router",
                user_id=user_id,
                provider=config["provider"],
                error=str(e),
            )

    # If still no API key, try direct platform key fallback
    if not config["api_key"]:
        import os
        platform_key_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
        key_var = platform_key_vars.get(provider)
        if key_var:
            platform_key = os.environ.get(key_var)
            if platform_key:
                config["api_key"] = platform_key
                logger.info(
                    "Using platform API key directly",
                    provider=config["provider"],
                    model=model,
                )

    return config


@router.post("/message")
async def send_message(
    request: ChatRequest,
    user: UserContext = Depends(get_current_user),
):
    """Send a message and get a response (non-streaming)."""

    thread_id = request.thread_id or str(uuid.uuid4())

    logger.info(
        "Chat message request",
        thread_id=thread_id,
        user_id=user.user_id,
        auth_method=user.auth_method,
    )

    # Build AI config from user preferences - use authenticated user_id
    ai_config = await _build_ai_config(request.ai_config, user.user_id)

    # Get model info for Langfuse tracking
    model_id = ai_config.get("model", "claude-sonnet-4-20250514") if ai_config else "claude-sonnet-4-20250514"
    provider = ai_config.get("provider", "anthropic") if ai_config else "anthropic"

    # Create graph with checkpointer
    checkpointer = get_checkpointer()
    graph = create_chat_graph()
    app = graph.compile(checkpointer=checkpointer)

    # Configure Langfuse tracing for cost tracking
    config = {"configurable": {"thread_id": thread_id}}
    langfuse_handler = get_langfuse_handler(
        user_id=user.user_id,
        session_id=thread_id,
        trace_name="chat_message",
        tags=["chat", f"provider:{provider}", f"model:{model_id}"],
        metadata={
            "model": model_id,
            "provider": provider,
            "message_count": len(request.messages),
            "app_url": request.app_url,
        },
    )
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    # Convert messages to LangChain format
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))

    # Create initial state with AI config
    initial_state: ChatState = {
        "messages": lc_messages,
        "app_url": request.app_url or "",
        "current_tool": None,
        "tool_results": [],
        "session_id": thread_id,
        "ai_config": ai_config,  # Pass AI config for model selection
    }

    logger.info("Processing chat message", thread_id=thread_id, message_count=len(lc_messages))

    # Run the graph
    try:
        final_state = await app.ainvoke(initial_state, config)
    finally:
        # Flush Langfuse to ensure cost tracking data is sent
        flush_langfuse()

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
async def stream_message(
    request: ChatRequest,
    user: UserContext = Depends(get_current_user),
):
    """Send a message and stream the response using Vercel AI SDK compatible format.

    Supports user-configurable AI models via the ai_config field:
    - model: Model ID (e.g., "claude-sonnet-4-5", "gpt-4o")
    - provider: Provider name (e.g., "anthropic", "openai")
    - use_byok: Whether to use user's BYOK key if available
    """

    thread_id = request.thread_id or str(uuid.uuid4())

    # Build AI config from user preferences - use authenticated user_id
    try:
        ai_config = await _build_ai_config(request.ai_config, user.user_id)
    except Exception as config_error:
        logger.exception("Failed to build AI config", user_id=user.user_id, error=str(config_error))
        # Capture error message before it goes out of scope
        error_message = str(config_error) if str(config_error) else "Failed to configure AI model"
        # Return error in AI SDK stream format
        async def error_stream():
            yield f'3:{json.dumps(error_message)}\n'
            yield f'd:{json.dumps({"finishReason": "error"})}\n'
        return StreamingResponse(
            error_stream(),
            media_type="text/plain; charset=utf-8",
        )

    async def generate_ai_sdk_stream():
        """Generate stream in Vercel AI SDK format (text streaming protocol)."""
        langfuse_handler = None
        try:
            # Get model info for logging and Langfuse
            model_id = ai_config.get("model", "claude-sonnet-4-20250514") if ai_config else "claude-sonnet-4-20250514"
            provider = ai_config.get("provider", "anthropic") if ai_config else "anthropic"
            using_byok = bool(ai_config.get("api_key")) if ai_config else False

            # Create graph with checkpointer
            checkpointer = get_checkpointer()
            graph = create_chat_graph()
            app = graph.compile(checkpointer=checkpointer)

            # Configure Langfuse tracing for cost tracking
            config = {"configurable": {"thread_id": thread_id}}
            langfuse_handler = get_langfuse_handler(
                user_id=user.user_id,
                session_id=thread_id,
                trace_name="chat_stream",
                tags=["chat", "stream", f"provider:{provider}", f"model:{model_id}"],
                metadata={
                    "model": model_id,
                    "provider": provider,
                    "using_byok": using_byok,
                    "message_count": len(request.messages),
                    "app_url": request.app_url,
                },
            )
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

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
                "ai_config": ai_config,  # Pass AI config for model selection
            }

            logger.info(
                "Starting chat stream",
                thread_id=thread_id,
                model=model_id,
                provider=provider,
                using_byok=using_byok,
                langfuse_enabled=langfuse_handler is not None,
            )

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
                    "model": model_id,
                    "provider": provider,
                    "using_byok": using_byok,
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

            # AI SDK finish message with model info: d:{"finishReason":"stop",...}\n
            yield f'd:{json.dumps({"finishReason": "stop", "threadId": thread_id, "model": model_id, "provider": provider})}\n'

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

        finally:
            # Flush Langfuse to ensure cost tracking data is sent
            flush_langfuse()

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
